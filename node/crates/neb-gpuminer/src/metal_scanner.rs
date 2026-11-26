//! M1 Max KEP1 scanner. Metal evaluates the batched gradient reduction. The host
//! uses the same ChaCha8 sample selection and BLAKE3 ticket code as the CPU miner.
use metal::foreign_types::ForeignType;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions,
    MTLSize,
};
use neb_block::Target;
use neb_engines::table::Table;
use neb_pow::shares::{Arch, Model, SharedDataset, share_indices};
use objc::{msg_send, rc::autoreleasepool, sel, sel_impl};

use crate::Scanner;

const BATCH: usize = 4096;
const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;
const ROW: usize = 3634;
const STRIDE: usize = ROW.next_multiple_of(4);
const SHADER: &str = include_str!("../../../../pow/gpu/kep1.metal");

// The Objective-C allocation API can return nil. metal-rs 0.33 models its
// return as a non-null Buffer, so check the raw return before taking ownership.
#[allow(unexpected_cfgs)] // objc 0.2's selector macro tests the legacy cargo-clippy cfg
fn shared_buffer(device: &Device, len: usize) -> Result<Buffer, String> {
    // SAFETY: MTLDevice takes two NSUInteger arguments and returns a retained
    // MTLBuffer or nil. The Objective-C "new" ownership transfers to Buffer.
    let raw: *mut metal::MTLBuffer = unsafe { msg_send![&**device, newBufferWithLength:len options:SHARED.bits()] };
    if raw.is_null() {
        return Err(format!("Metal could not allocate {len} shared bytes"));
    }
    // SAFETY: raw is the checked, retained MTLBuffer from the allocation above.
    let buffer = unsafe { Buffer::from_ptr(raw) };
    if buffer.contents().is_null() {
        return Err("Metal shared buffer has no CPU mapping".into());
    }
    Ok(buffer)
}

pub struct MetalScanner {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    dataset: SharedDataset,
    rows: Option<Buffer>,
    indices: Buffer,
    digests: Buffer,
    keys: Vec<[u8; 32]>,
}

impl MetalScanner {
    pub fn open(dataset: SharedDataset) -> Result<Self, String> {
        if dataset.arch() != Arch::Kep1 || dataset.n() == 0 || u64::from(dataset.n()) * STRIDE as u64 > u32::MAX as u64
        {
            return Err("Metal requires a nonempty KEP1 dataset addressable with 32-bit row offsets".into());
        }
        let device = Device::system_default().ok_or("macOS did not expose a Metal device")?;
        let library = device.new_library_with_source(SHADER, &CompileOptions::new())?;
        let function = library.get_function("kep1_digest", None)?;
        let pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        if pipeline.thread_execution_width() != 32 || pipeline.max_total_threads_per_threadgroup() < 128 {
            return Err("the KEP1 shader requires 32-lane SIMD and 128 threads per group".into());
        }
        let indices = shared_buffer(&device, BATCH * 64)?;
        let digests = shared_buffer(&device, BATCH * 32)?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue, pipeline, dataset, rows: None, indices, digests, keys: vec![[0; 32]; BATCH] })
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    /// Returns every ticket in nonce order, including misses, for parity tests
    /// and target filtering. Host and device access the shared buffers in turns.
    pub fn tickets(&mut self, header: &[u8; 80], start: u32, count: u32) -> Result<&[[u8; 32]], String> {
        if count as usize > BATCH || u64::from(start) + u64::from(count) > 1u64 << 32 {
            return Err("nonce range exceeds the batch or nonce space".into());
        }
        if count == 0 {
            return Ok(&[]);
        }
        let rows = self.rows.as_ref().ok_or("upload a model before scanning")?;
        let keys = &mut self.keys[..count as usize];
        // SAFETY: indices holds BATCH 64-byte records in page-aligned shared storage.
        // The previous dispatch finished before tickets returned; &mut self excludes other writers.
        let indices =
            unsafe { std::slice::from_raw_parts_mut(self.indices.contents().cast::<[u32; 16]>(), keys.len()) };
        let mut h = *header;
        for (i, (key, idx)) in keys.iter_mut().zip(indices).enumerate() {
            h[76..].copy_from_slice(&(start + i as u32).to_le_bytes());
            *key = *blake3::hash(&h).as_bytes();
            *idx = share_indices(key, self.dataset.n());
        }
        autoreleasepool(|| {
            let command = self.queue.new_command_buffer();
            let enc = command.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline);
            for (i, buf) in [rows, &self.indices, &self.digests].iter().enumerate() {
                enc.set_buffer(i as u64, Some(buf), 0);
            }
            enc.dispatch_thread_groups(MTLSize::new(count.into(), 1, 1), MTLSize::new(128, 1, 1));
            enc.end_encoding();
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(format!("Metal dispatch failed: {:?}", command.status()));
            }
            Ok(())
        })?;
        // SAFETY: the completed dispatch wrote exactly one 32-byte digest per nonce.
        let digests = unsafe { std::slice::from_raw_parts(self.digests.contents().cast::<[u8; 32]>(), keys.len()) };
        for (digest, key) in digests.iter().zip(keys.iter_mut()) {
            let mut input = [0; 64];
            input[..32].copy_from_slice(digest);
            input[32..].copy_from_slice(key);
            *key = *blake3::hash(&input).as_bytes();
        }
        Ok(keys)
    }
}

impl Scanner for MetalScanner {
    fn capacity(&self) -> u32 {
        BATCH as u32
    }
    fn set_model(&mut self, _: &[i8], _: &[i8]) -> Result<(), String> {
        Err("Metal supports KEP1 models".into())
    }
    fn set_weights(&mut self, model: &Model) -> Result<(), String> {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let table = Table::build(model, &*self.dataset, threads, vec![])?.into_spare();
        if table.len() != ROW * self.dataset.n() as usize {
            return Err("unexpected KEP1 row layout".into());
        }
        let len = STRIDE * self.dataset.n() as usize;
        if self.rows.is_none() {
            self.rows = Some(shared_buffer(&self.device, len)?);
        }
        let rows = self.rows.as_ref().unwrap();
        // SAFETY: the slot's dataset fixes len for the lifetime of this shared
        // buffer. All dispatches complete synchronously under exclusive access.
        let out = unsafe { std::slice::from_raw_parts_mut(rows.contents().cast::<u8>(), len) };
        for (dst, src) in out.as_chunks_mut::<STRIDE>().0.iter_mut().zip(table.as_chunks::<ROW>().0) {
            dst[..ROW].copy_from_slice(src);
        }
        Ok(())
    }
    fn scan(&mut self, header: &[u8; 80], start: u32, count: u32, bound: &[u8; 32]) -> Result<Vec<u32>, String> {
        let bound = Target(*bound);
        Ok(self
            .tickets(header, start, count)?
            .iter()
            .enumerate()
            .filter_map(|(i, t)| bound.admits(t).then_some(start + i as u32))
            .collect())
    }
}
