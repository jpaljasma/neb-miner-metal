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
#[path = "metal_ahead.rs"]
mod metal_ahead;
#[path = "metal_indices.rs"]
mod metal_indices;
pub use metal_ahead::TessAhead;
#[path = "metal_screen.rs"]
mod metal_screen;
#[cfg(feature = "metal-bench")]
#[path = "metal_turnover_bench.rs"]
mod metal_turnover_bench;
pub use metal_screen::{MINING_SCREEN_CAP, PROBE_CAP, SCREEN_CAP, ScreenResult};
#[cfg(feature = "metal-bench")]
pub struct PairScreenVariant {
    pipeline: ComputePipelineState,
    mode: u8,
    tuning: metal_screen::PairTuning,
}
const SCREEN_BATCH_SCALE: u64 = (MINING_SCREEN_CAP / SCREEN_CAP) as u64;

const BATCH: usize = 16384;
const DISPATCH: usize = 2048;
const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;
const ROW: usize = 3634;
const STRIDE: usize = ROW.next_multiple_of(4);
// Support words by layer: 1,360 convolution weights, then 32×768 and 2×32 dense weights.
const SUPPORT_WORDS: usize = 1360usize.div_ceil(32) + 32 * 24 + 2;
const SHADER: &str = include_str!("../../../../pow/gpu/kep1.metal");
// Exact short-vector products, closed-form zero salt and 256 threads per nonce.
const DEFAULT_VARIANT: u8 = 4 | 8 | 16;

fn pipeline(device: &Device, mask: u8) -> Result<(ComputePipelineState, u64), String> {
    table_pipeline(device, mask, 0, 0)
}

fn table_pipeline(device: &Device, mask: u8, build: u8, read: u8) -> Result<(ComputePipelineState, u64), String> {
    table_pipeline_options(device, mask, build, read, PairBuilder::General)
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum PairBuilder {
    General,
    TwoRows,
    #[cfg(feature = "metal-bench")]
    MaskedTwoRows,
    DirectMaskedTwoRows,
}
impl PairBuilder {
    fn uses_output_mask(self) -> bool {
        match self {
            Self::General | Self::TwoRows => false,
            #[cfg(feature = "metal-bench")]
            Self::MaskedTwoRows => true,
            Self::DirectMaskedTwoRows => true,
        }
    }
}

fn table_pipeline_options(
    device: &Device,
    mask: u8,
    build: u8,
    read: u8,
    pair_style: PairBuilder,
) -> Result<(ComputePipelineState, u64), String> {
    let width = match pair_style {
        PairBuilder::TwoRows | PairBuilder::DirectMaskedTwoRows => 32,
        _ if mask & 16 == 0 => 128,
        _ => 256,
    };
    let direct = u8::from(pair_style == PairBuilder::DirectMaskedTwoRows);
    table_pipeline_geometry(device, mask, build, read, pair_style, width, direct)
}
fn table_pipeline_geometry(
    device: &Device,
    mask: u8,
    build: u8,
    read: u8,
    pair_style: PairBuilder,
    width: u64,
    access: u8,
) -> Result<(ComputePipelineState, u64), String> {
    let source = format!(
        "#define ROW_CACHE {}\n#define PLANAR {}\n#define SHORT_MAC {}\n#define SALT_BASE {}\n#define SINGLE_CACHE {}\n#define COMPACT_ROWS {}\n#define SHORT_DOT {}\n#define COMBO_BUILD {build}\n#define COMBO_READ {read}\n#define GROUP_WIDTH {width}\n{SHADER}",
        mask & 1,
        (mask >> 1) & 1,
        (mask >> 2) & 1,
        (mask >> 3) & 1,
        (mask >> 5) & 1,
        (mask >> 6) & 1,
        (mask >> 7) & 1
    );
    // Build code 4 is the incremental variant of the exact pair builder.
    let source = if build == 4 {
        format!("#define REUSE_PAIRS 1\n{}", source.replace("#define COMBO_BUILD 4", "#define COMBO_BUILD 2"))
    } else {
        source
    };
    let source = format!(
        "#define TWO_ROW {}\n#define PAIR_OUTPUT_MASK {}\n#define DIRECT_FACTORS {access}\n{}",
        u8::from(pair_style != PairBuilder::General),
        u8::from(pair_style.uses_output_mask()),
        source
    );
    let library = device.new_library_with_source(&source, &CompileOptions::new())?;
    let function = library.get_function("kep1_digest", None)?;
    let state = device.new_compute_pipeline_state_with_function(&function)?;
    if state.thread_execution_width() != 32 || state.max_total_threads_per_threadgroup() < width {
        return Err(format!("KEP1 requires 32-lane SIMD and {width} threads per group"));
    }
    Ok((state, width))
}

// The Objective-C allocation API can return nil. metal-rs 0.33 models its
// return as a non-null Buffer, so check the raw return before taking ownership.
#[allow(unexpected_cfgs)] // objc 0.2's selector macro tests the legacy cargo-clippy cfg
fn shared_buffer(device: &metal::DeviceRef, len: usize) -> Result<Buffer, String> {
    // SAFETY: MTLDevice takes two NSUInteger arguments and returns a retained
    // MTLBuffer or nil. The Objective-C "new" ownership transfers to Buffer.
    let raw: *mut metal::MTLBuffer = unsafe { msg_send![device, newBufferWithLength:len options:SHARED.bits()] };
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

fn mapped_single_lookup(device: &Device, table: &Table, threads: usize) -> Result<Buffer, String> {
    let count = table.live_rows().len() + 1;
    let storage = shared_buffer(device, count * size_of::<[u32; 9]>())?;
    // SAFETY: newBufferWithLength zero-initializes this aligned, count-record mapping.
    // The CPU workers join before any GPU command can read it.
    let entries = unsafe { std::slice::from_raw_parts_mut(storage.contents().cast::<[u32; 9]>(), count) };
    table.fill_zeroed_single_row_lookup(threads, entries);
    Ok(storage)
}

struct ModelRows {
    buffer: Buffer,
    weights: Vec<i8>,
    singles: Option<Buffer>,
    live: Vec<bool>,
    combos: Option<Combos>,
}

// Lay out each layer on a word boundary. The two dense layers have 768 and 32 inputs.
fn turnover_tables(
    device: &Device,
    rows: &ModelRows,
    ids: &[u32],
    masks: bool,
    triples: bool,
) -> Result<(Option<Buffer>, Option<Buffer>), String> {
    if !masks && !triples {
        return Ok((None, None));
    }
    let active = masks.then(|| shared_buffer(device, rows.live.len() * 4)).transpose()?;
    let support = triples.then(|| shared_buffer(device, ids.len().max(1) * SUPPORT_WORDS * 4)).transpose()?;
    // SAFETY: each buffer is exclusive here, zeroed by Metal and sized by its respective index domain.
    let mut outputs =
        active.as_ref().map(|b| unsafe { std::slice::from_raw_parts_mut(b.contents().cast::<u32>(), rows.live.len()) });
    let mut bits = support.as_ref().map(|b| {
        // SAFETY: this new mapping has SUPPORT_WORDS aligned words per live rank and has not been submitted.
        unsafe { std::slice::from_raw_parts_mut(b.contents().cast::<u32>(), ids.len() * SUPPORT_WORDS) }
    });
    // SAFETY: the padded row buffer has this fixed stride and all previous commands completed.
    let packed = unsafe { std::slice::from_raw_parts(rows.buffer.contents().cast::<u8>(), rows.live.len() * STRIDE) };
    for (rank, &sample) in ids.iter().enumerate() {
        let row = &packed[sample as usize * STRIDE..][..ROW];
        let factors = &row[2800..];
        if let Some(out) = outputs.as_mut() {
            out[sample as usize] = factors[..32].iter().enumerate().fold(0, |m, (i, &x)| m | (u32::from(x != 0) << i));
        }
        if let Some(bits) = bits.as_mut() {
            let dest = &mut bits[SUPPORT_WORDS * rank..][..SUPPORT_WORDS];
            let wide = row[..160].as_chunks::<4>().0.iter().map(|g| *g != [0; 4]);
            let narrow = row[160..2800].as_chunks::<2>().0.iter().map(|g| *g != [0; 2]);
            for (i, nonzero) in wide.chain(narrow).enumerate() {
                if nonzero {
                    dest[i / 32] |= 1 << (i % 32);
                }
            }
            for (base, errors, inputs) in
                [(43, &factors[..32], &factors[32..800]), (811, &factors[800..802], &factors[802..834])]
            {
                for (w, chunk) in inputs.as_chunks::<32>().0.iter().enumerate() {
                    let mask = chunk.iter().enumerate().fold(0, |m, (i, &v)| m | (u32::from(v != 0) << i));
                    for (o, &error) in errors.iter().enumerate() {
                        if error != 0 {
                            dest[base + o * (inputs.len() / 32) + w] = mask;
                        }
                    }
                }
            }
        }
    }
    Ok((active, support))
}

struct Combos {
    rank_buffers: [Option<Buffer>; 3],
    values: Buffer,
    args: [u32; 3], // ready pairs, ready triple prefix, allocated pairs
    ids: Buffer,
    pair_builder: (ComputePipelineState, u64),
    active_outputs: Option<Buffer>,
    support: Option<Buffer>,
    host_ranks: Vec<u32>,
    last_slice: std::time::Instant,
    slice_delay: std::time::Duration,
    reuse: Option<PairReuse>,
}

// Reassess a marginal model after it has accumulated real mining time.
// An inactive slot cannot spend the other slot's exploratory build budget.
struct RetryScreen {
    evaluated: u64,
    active_s: f64,
    build_s: f64,
    next_probe: u32,
}

struct PairReuse {
    ranks: Buffer,
    old_values: Buffer,
    ready: u32,
}

impl Combos {
    #[allow(clippy::too_many_arguments)]
    fn slice(
        &self,
        queue: &CommandQueue,
        rows: &Buffer,
        pipeline: &(ComputePipelineState, u64),
        offset: usize,
        first: usize,
        len: usize,
        limit: usize,
    ) -> Result<(), String> {
        autoreleasepool(|| {
            let args = [first as u32, limit as u32, self.args[2]];
            let cmd = queue.new_command_buffer();
            let e = cmd.new_compute_command_encoder();
            e.set_compute_pipeline_state(&pipeline.0);
            if let Some(active) = &self.active_outputs {
                e.set_buffer(10, Some(active), 0);
            }
            e.set_buffer(0, Some(rows), 0);
            e.set_buffer(1, Some(&self.ids), 0);
            e.set_buffer(2, Some(&self.values), ((offset + first) * 32) as u64);
            e.set_bytes(6, 12, args.as_ptr().cast());
            if let Some(hint) = self.reuse.as_ref().filter(|_| offset == 0) {
                e.set_buffer(7, Some(&hint.ranks), 0);
                e.set_buffer(8, Some(&hint.old_values), 0);
                e.set_bytes(9, 4, (&hint.ready as *const u32).cast());
            }
            e.dispatch_thread_groups(MTLSize::new(len as u64, 1, 1), MTLSize::new(pipeline.1, 1, 1));
            e.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            if cmd.status() != MTLCommandBufferStatus::Completed {
                return Err("digest table slice failed".to_string());
            }
            Ok(())
        })
    }
    fn advance(&mut self, queue: &CommandQueue, rows: &Buffer) -> Result<(), String> {
        let [ready, _, total] = self.args;
        if ready == total || self.last_slice.elapsed() < self.slice_delay {
            return Ok(());
        }
        let len = (total - ready).min(32768);
        let t = std::time::Instant::now();
        self.slice(queue, rows, &self.pair_builder, 0, ready as usize, len as usize, 0)?;
        self.args[0] += len;
        self.slice_delay = t.elapsed() * 2;
        self.last_slice = std::time::Instant::now();
        if self.args[0] == total {
            self.reuse = None;
            eprintln!("Metal pair table ready: {total} entries");
        }
        Ok(())
    }
}

const HASH_IV: [u32; 8] =
    [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
const CHUNK_START: u8 = 1;
const CHUNK_END: u8 = 2;
const ROOT: u8 = 8;

fn finish_tickets(platform: blake3::platform::Platform, keys: &mut [[u8; 32]], digests: &[[u8; 32]], simd: bool) {
    // Each digest || key is a complete, independent 64-byte BLAKE3 root.
    // hash_many can use the M1's four-lane NEON implementation for these inputs.
    for (keys, ds) in keys.chunks_mut(16).zip(digests.chunks(16)) {
        if simd && keys.len() == 16 {
            let mut blocks = [[0u8; 64]; 16];
            for ((block, key), digest) in blocks.iter_mut().zip(keys.iter()).zip(ds) {
                block[..32].copy_from_slice(digest);
                block[32..].copy_from_slice(key);
            }
            platform.hash_many(
                &blocks.each_ref(),
                &HASH_IV,
                0,
                blake3::IncrementCounter::No,
                ROOT,
                CHUNK_START,
                CHUNK_END,
                keys.as_flattened_mut(),
            );
        } else {
            for (key, digest) in keys.iter_mut().zip(ds) {
                let mut input = [0; 64];
                input[..32].copy_from_slice(digest);
                input[32..].copy_from_slice(key);
                *key = *blake3::hash(&input).as_bytes();
            }
        }
    }
}

pub struct MetalScanner {
    recover_triples: bool,
    skip_unchanged: bool,
    pair_style: PairBuilder,
    #[cfg(feature = "metal-bench")]
    skipped_bytes: usize,
    #[cfg(feature = "metal-bench")]
    copy_seconds: f64,
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    width: u64,
    dispatch_size: usize,
    host_mask: u8,
    allow_screen: bool,
    compact_tickets: bool,
    ahead: Option<TessAhead>,
    build_threads: Option<usize>,
    allow_reuse: bool,
    reuse_hint: Option<PairReuse>,
    use_screen: bool,
    retry_screen: Option<RetryScreen>,
    screen_batches: u64,
    #[cfg(feature = "metal-bench")]
    gpu_seconds: f64,
    screen: Option<metal_screen::Screen>,
    dataset: SharedDataset,
    rows: Option<ModelRows>,
    indices: Buffer,
    digests: Buffer,
    keys: Vec<[u8; 32]>,
}

impl MetalScanner {
    /// Both slots group rank loads. Packing eligible lanes together also paid off on TESS.
    pub fn for_slot(dataset: SharedDataset, slot: usize) -> Result<Self, String> {
        let mut scanner = Self::open(dataset)?;
        scanner.compact_tickets = slot == 1;
        if slot == 1 {
            scanner.pair_style = PairBuilder::TwoRows;
        }
        Ok(scanner)
    }
    /// Compare lookup construction and upload, including the destination allocation.
    #[cfg(feature = "metal-bench")]
    pub fn benchmark_single_lookup(&self, table: &Table, threads: usize, mapped: bool) -> Result<Buffer, String> {
        if mapped {
            return mapped_single_lookup(&self.device, table, threads);
        }
        let entries = table.single_row_lookup(threads);
        let storage = shared_buffer(&self.device, std::mem::size_of_val(entries.as_slice()))?;
        // SAFETY: entries and the new destination have the same byte length and never overlap.
        unsafe { std::ptr::copy_nonoverlapping(entries.as_ptr(), storage.contents().cast(), entries.len()) };
        Ok(storage)
    }
    pub fn open(dataset: SharedDataset) -> Result<Self, String> {
        if dataset.arch() != Arch::Kep1 || dataset.n() == 0 || u64::from(dataset.n()) * STRIDE as u64 > u32::MAX as u64
        {
            return Err("Metal requires a nonempty KEP1 dataset addressable with 32-bit row offsets".into());
        }
        let device = Device::system_default().ok_or("macOS did not expose a Metal device")?;
        let (pipeline, width) = pipeline(&device, DEFAULT_VARIANT)?;
        let indices = shared_buffer(&device, BATCH * 64)?;
        let digests = shared_buffer(&device, BATCH * 32)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            recover_triples: false,
            skip_unchanged: false,
            pair_style: PairBuilder::DirectMaskedTwoRows,
            #[cfg(feature = "metal-bench")]
            skipped_bytes: 0,
            #[cfg(feature = "metal-bench")]
            copy_seconds: 0.,
            queue,
            pipeline,
            width,
            dispatch_size: DISPATCH,
            host_mask: 0,
            allow_screen: false,
            compact_tickets: false,
            ahead: None,
            build_threads: None,
            allow_reuse: true,
            reuse_hint: None,
            use_screen: false,
            retry_screen: None,
            screen_batches: 0,
            dataset,
            rows: None,
            indices,
            digests,
            keys: vec![[0; 32]; BATCH],
            #[cfg(feature = "metal-bench")]
            gpu_seconds: 0.0,
            screen: None,
        })
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    /// Returns every ticket in nonce order, including misses, for parity tests
    /// and target filtering. Host and device access the shared buffers in turns.
    pub fn tickets(&mut self, header: &[u8; 80], start: u32, count: u32) -> Result<&[[u8; 32]], String> {
        if count as usize > self.keys.len() || u64::from(start) + u64::from(count) > 1u64 << 32 {
            return Err("nonce range exceeds the batch or nonce space".into());
        }
        if count == 0 {
            return Ok(&[]);
        }
        let rows = self.rows.as_ref().ok_or("upload a model before scanning")?;
        let keys = &mut self.keys[..count as usize];
        let mut h = *header;
        let platform = blake3::platform::Platform::detect();
        let midstate = self.host_mask & 1 != 0;
        let mut prefix = HASH_IV;
        let mut tail = [0; 64];
        if midstate {
            platform.compress_in_place(&mut prefix, header[..64].try_into().unwrap(), 64, 0, CHUNK_START);
            tail[..12].copy_from_slice(&header[64..76]);
        }
        let n = self.dataset.n();
        autoreleasepool(|| {
            let mut pending = Vec::with_capacity(keys.len().div_ceil(self.dispatch_size));
            for (i, chunk) in keys.chunks_mut(self.dispatch_size).enumerate() {
                let offset = i * self.dispatch_size;
                // SAFETY: this chunk is within the shared buffer sized with keys.
                // Earlier commands read disjoint byte ranges, and the last call drained its queue.
                let indices = unsafe {
                    std::slice::from_raw_parts_mut(self.indices.contents().cast::<[u32; 16]>().add(offset), chunk.len())
                };
                for (j, key) in chunk.iter_mut().enumerate() {
                    let nonce = (start + (offset + j) as u32).to_le_bytes();
                    *key = if midstate {
                        tail[12..16].copy_from_slice(&nonce);
                        let mut cv = prefix;
                        platform.compress_in_place(&mut cv, &tail, 16, 0, CHUNK_END | ROOT);
                        blake3::platform::le_bytes_from_words_32(&cv)
                    } else {
                        h[76..].copy_from_slice(&nonce);
                        *blake3::hash(&h).as_bytes()
                    };
                }
                let mut vectorized = 0;
                if self.host_mask & 32 != 0 {
                    for (keys, out) in chunk.as_chunks::<4>().0.iter().zip(indices.as_chunks_mut::<4>().0) {
                        *out = metal_indices::four_keys(keys, n);
                        vectorized += 4;
                    }
                }
                for (key, idx) in chunk[vectorized..].iter().zip(&mut indices[vectorized..]) {
                    *idx = share_indices(key, n);
                }
                if self.host_mask & 8 != 0 {
                    for idx in indices {
                        // Preserve repeated live samples. The upper byte holds the
                        // count; the constructor's row-offset bound leaves it free.
                        let mut live = 0;
                        for sample in *idx {
                            idx[live] = sample;
                            live += usize::from(rows.live[sample as usize]);
                        }
                        idx[0] |= (live as u32) << 24;
                    }
                }
                let command = self.queue.new_command_buffer();
                let enc = command.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.pipeline);
                enc.set_buffer(0, Some(&rows.buffer), 0);
                enc.set_buffer(1, Some(&self.indices), (offset * 64) as u64);
                enc.set_buffer(2, Some(&self.digests), (offset * 32) as u64);
                if self.host_mask & 4 != 0 {
                    enc.set_buffer(3, rows.singles.as_deref(), 0);
                }
                #[cfg(feature = "metal-bench")]
                if self.host_mask & 64 != 0 {
                    let c = rows.combos.as_ref().expect("combination replay prepared");
                    enc.set_buffer(4, c.rank_buffers[metal_screen::RankLayout::Plain as usize].as_deref(), 0);
                    enc.set_buffer(5, Some(&c.values), 0);
                    enc.set_bytes(6, 12, c.args.as_ptr().cast());
                }
                enc.dispatch_thread_groups(MTLSize::new(chunk.len() as u64, 1, 1), MTLSize::new(self.width, 1, 1));
                enc.end_encoding();
                command.commit();
                pending.push(command);
            }
            let mut failed = None;
            #[cfg(feature = "metal-bench")]
            let mut gpu_end = 0.0_f64;
            for (i, (command, chunk)) in pending.into_iter().zip(keys.chunks_mut(self.dispatch_size)).enumerate() {
                let offset = i * self.dispatch_size;
                command.wait_until_completed();
                if command.status() != MTLCommandBufferStatus::Completed {
                    failed = Some(command.status());
                    continue; // always drain every submitted command before buffers can be reused
                }
                #[cfg(feature = "metal-bench")]
                {
                    // SAFETY: completed MTLCommandBuffers return CFTimeInterval (f64) for both timestamps.
                    #[allow(unexpected_cfgs)]
                    let (begin, end): (f64, f64) =
                        unsafe { (msg_send![command, GPUStartTime], msg_send![command, GPUEndTime]) };
                    self.gpu_seconds += (end - begin.max(gpu_end)).max(0.0);
                    gpu_end = gpu_end.max(end);
                }
                // SAFETY: this command finished writing its disjoint digest range.
                let digests = unsafe {
                    std::slice::from_raw_parts(self.digests.contents().cast::<[u8; 32]>().add(offset), chunk.len())
                };
                finish_tickets(platform, chunk, digests, self.host_mask & 2 != 0);
            }
            if let Some(status) = failed {
                return Err(format!("Metal dispatch failed: {status:?}"));
            }
            Ok(())
        })?;
        Ok(keys)
    }
}

impl MetalScanner {
    pub fn enable_screen(&mut self) {
        self.allow_screen = true;
    }
    pub fn enable_tess_ahead(&mut self) -> TessAhead {
        let ahead = TessAhead::new(self.dataset.clone());
        self.ahead = Some(ahead.clone());
        ahead
    }

    fn audit_screen(&mut self, model: &Model, header: &[u8; 80], start: u32, count: u32) -> Result<(), String> {
        let mut bound = [0xff; 32];
        bound[31] = 0x3f;
        let got = self.screen_range(header, start, count, &bound, 2, true)?;
        let fast = self.screen_range(header, start, count, &bound, 2, false)?;
        let combos = self.rows.as_ref().unwrap().combos.as_ref().unwrap();
        let mut scratch = neb_pow::shares::ShareBuffers::new();
        let mut eligible = 0;
        let mut want = vec![];
        let mut h = *header;
        for (i, record) in got.records.iter().enumerate() {
            let nonce = start + i as u32;
            h[76..].copy_from_slice(&nonce.to_le_bytes());
            let key = blake3::hash(&h);
            let mut live = share_indices(key.as_bytes(), self.dataset.n())
                .into_iter()
                .map(|id| combos.host_ranks[id as usize])
                .filter(|r| *r != u32::MAX);
            let selected = match (live.next(), live.next(), live.next()) {
                (_, _, Some(_)) => false,
                (Some(a), Some(b), None) => {
                    let hi = a.max(b);
                    hi * (hi + 1) / 2 + a.min(b) < combos.args[0]
                }
                _ => true,
            };
            if record[0] != u32::from(selected) {
                return Err("GPU nonce eligibility mismatch".into());
            }
            if selected {
                eligible += 1;
                let ticket = scratch.ticket(model, &*self.dataset, key.as_bytes());
                if record[1..] != blake3::platform::words_from_le_bytes_32(&ticket) {
                    return Err("GPU lookup ticket mismatch".into());
                }
                if Target(bound).admits(&ticket) {
                    want.push(nonce);
                }
            }
        }
        let mut hits = fast.hits;
        hits.sort_unstable();
        if hits != want || fast.evaluated != eligible || got.evaluated != eligible {
            return Err("GPU nonce-screen accounting mismatch".into());
        }
        Ok(())
    }

    fn choose_screen(&mut self, model: &Model) -> Result<bool, String> {
        let mut header = [0; 80];
        header[..32].copy_from_slice(&model.weights_hash());
        let start = std::time::Instant::now();
        {
            let r = self.rows.as_mut().unwrap();
            r.combos.as_mut().unwrap().advance(&self.queue, &r.buffer)?;
        }
        let slice_s = start.elapsed().as_secs_f64();
        self.audit_screen(model, &header, 0, 128)?;
        let t = std::time::Instant::now();
        self.tickets(&header, 0, 8192)?;
        let full = 8192. / t.elapsed().as_secs_f64();
        let mut rates = [0.; 3];
        for (i, rate) in rates.iter_mut().enumerate() {
            let t = std::time::Instant::now();
            self.screen_range(&header, (i as u32 + 1) * MINING_SCREEN_CAP, MINING_SCREEN_CAP, &[0; 32], 2, false)?;
            *rate = f64::from(MINING_SCREEN_CAP) / t.elapsed().as_secs_f64();
        }
        rates.sort_by(f64::total_cmp);
        let rows = self.rows.as_ref().unwrap();
        let c = rows.combos.as_ref().unwrap();
        let n = rows.live.iter().filter(|&&v| v).count() as f64;
        let p = n / f64::from(self.dataset.n());
        let q = 1. - p;
        let pair_coverage = (f64::from(c.args[2]) / (n * (n + 1.) / 2.).max(1.)).min(1.);
        let expected = rates[1] * (q.powi(16) + 16. * p * q.powi(15) + 120. * p * p * q.powi(14) * pair_coverage);
        let build_s = slice_s * f64::from(c.args[2]) / f64::from(c.args[0].max(1));
        // Compare payoff over a 30-second job, charging the entire predicted build.
        // The live builder interleaves individual slices with ticket evaluation.
        let keep = expected * (30. - build_s).max(0.) > full * 30. * 1.05;
        if !keep && expected > 1.25 * full {
            self.retry_screen = Some(RetryScreen { evaluated: 0, active_s: 0., build_s: 0., next_probe: c.args[0] });
        }
        eprintln!(
            "model trial full {:.0} attempts/s, predicted pair {:.0} attempts/s, build {:.2}s, selected {}",
            full, expected, build_s, keep
        );
        self.screen_batches = 0;
        Ok(keep)
    }

    fn retry_after_full_batch(
        &mut self,
        model: &Model,
        header: &[u8; 80],
        start: u32,
        count: u32,
        seconds: f64,
    ) -> Result<(), String> {
        let Some(mut retry) = self.retry_screen.take() else {
            return Ok(());
        };
        retry.active_s += seconds;
        retry.evaluated += u64::from(count);
        // After the grace period, allow one build second per eight seconds
        // of full scanning. The budget can overshoot by at most one slice.
        if retry.active_s < 15. || retry.build_s * 8. > retry.active_s - 15. {
            self.retry_screen = Some(retry);
            return Ok(());
        }
        let r = self.rows.as_mut().unwrap();
        let c = r.combos.as_mut().unwrap();
        let t = std::time::Instant::now();
        c.advance(&self.queue, &r.buffer)?;
        retry.build_s += t.elapsed().as_secs_f64();
        let [ready, _, total] = c.args;
        if ready >= retry.next_probe || ready == total {
            let mut rates = [0.; 3];
            for (i, rate) in rates.iter_mut().enumerate() {
                let nonce = start.wrapping_add((i as u32) * MINING_SCREEN_CAP).min(u32::MAX - MINING_SCREEN_CAP + 1);
                let t = std::time::Instant::now();
                let probe = self.screen_range(header, nonce, MINING_SCREEN_CAP, &[0; 32], 2, false)?;
                *rate = f64::from(probe.evaluated) / t.elapsed().as_secs_f64();
            }
            rates.sort_by(f64::total_cmp);
            // While building, the scanner can spend up to a third of its
            // time on slices. Charge that cost before switching.
            let full_rate = retry.evaluated as f64 / retry.active_s;
            let required = full_rate * if ready == total { 1.05 } else { 1.60 };
            if rates[1] > required {
                self.audit_screen(model, header, start.min(u32::MAX - 15), 16)?;
                self.use_screen = true;
                eprintln!(
                    "long-job retry selected GPU screen at {ready}/{total} pairs: measured {:.0} attempts/s (full {:.0})",
                    rates[1], full_rate
                );
                return Ok(());
            }
            if ready == total {
                return Ok(());
            }
            retry.next_probe = ready.saturating_add((total / 16).max(32768));
        }
        self.retry_screen = Some(retry);
        Ok(())
    }
    pub fn screen_range(
        &mut self,
        header: &[u8; 80],
        start: u32,
        count: u32,
        bound: &[u8; 32],
        arity: u8,
        check: bool,
    ) -> Result<ScreenResult, String> {
        self.run_screen(header, start, count, bound, arity, check, false)
    }

    fn productive_range(
        &mut self,
        header: &[u8; 80],
        start: u32,
        count: u32,
        bound: &[u8; 32],
    ) -> Result<ScreenResult, String> {
        self.run_screen(header, start, count, bound, 2, false, self.recover_triples)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_screen(
        &mut self,
        header: &[u8; 80],
        start: u32,
        count: u32,
        bound: &[u8; 32],
        arity: u8,
        check: bool,
        recover: bool,
    ) -> Result<ScreenResult, String> {
        let screen = match &mut self.screen {
            Some(screen) => screen,
            empty @ None => {
                empty.insert(metal_screen::Screen::new(&self.device, self.compact_tickets, self.dataset.n())?)
            }
        };
        screen.scan(
            &self.queue,
            self.rows.as_mut().ok_or("missing model")?,
            self.dataset.n(),
            header,
            start,
            count,
            bound,
            arity,
            check,
            recover,
        )
    }
    pub fn combo_pipeline(&self, arity: u8) -> Result<(ComputePipelineState, u64), String> {
        table_pipeline(&self.device, DEFAULT_VARIANT | 32 | 64 | 128, 0, arity)
    }
    /// Build exact pair and partial triple digests in bounded commands. Ranks
    /// include repeated rows, so (i,i), (i,i,j) and (i,i,i) are valid entries.
    pub fn build_combos(&mut self, triple_mib: usize) -> Result<(), String> {
        self.prepare_combos(triple_mib, false)
    }
    fn prepare_combos(&mut self, triple_mib: usize, progressive: bool) -> Result<(), String> {
        let rows = self.rows.as_mut().ok_or("missing model rows")?;
        let n = rows.live.iter().filter(|&&live| live).count();
        let mut pairs = n * (n + 1) / 2;
        if progressive {
            pairs = pairs.min((1024 * 1024 * 1024) / 32);
        }
        let tet = |v: usize| v * (v + 1) * (v + 2) / 6;
        let limit = (0..=n.min(1024)).take_while(|&v| tet(v) * 32 <= triple_mib * 1024 * 1024).last().unwrap();
        let triples = tet(limit);
        let bytes = (pairs + triples) * 32;
        if (!progressive && n == 0) || bytes > 4usize << 30 || bytes as u64 > self.device.max_buffer_length() {
            return Err("combination table exceeds the replay budget or is empty".into());
        }
        let live_ids = shared_buffer(&self.device, n.max(1) * size_of::<u32>())?;
        // SAFETY: this unpublished mapping has space for every live row, including the empty case.
        let ids = unsafe { std::slice::from_raw_parts_mut(live_ids.contents().cast::<u32>(), n) };
        let mut rank = vec![u32::MAX; rows.live.len()];
        let live_samples = rows.live.iter().enumerate().filter_map(|(sample, &live)| live.then_some(sample));
        for (r, sample) in live_samples.enumerate() {
            ids[r] = sample as u32;
            rank[sample] = r as u32;
        }
        let layout = metal_screen::PairTuning::production(self.compact_tickets, self.dataset.n()).ranks;
        let mut rank_buffers = [None, None, None];
        rank_buffers[layout as usize] = Some(layout.build(&self.device, &rank)?);
        // The exhaustive combo-read benchmark uses the upstream 32-bit sentinel representation.
        #[cfg(feature = "metal-bench")]
        {
            rank_buffers[0] = Some(metal_screen::RankLayout::Plain.build(&self.device, &rank)?);
        }
        let values = shared_buffer(&self.device, bytes.max(32))?;
        let reuse = self.reuse_hint.take();
        let build_kind = if reuse.is_some() { 4 } else { 2 };
        let pair_builder =
            table_pipeline_options(&self.device, DEFAULT_VARIANT | 64 | 128, build_kind, 0, self.pair_style)?;
        let (active_outputs, support) =
            turnover_tables(&self.device, rows, ids, self.pair_style.uses_output_mask(), self.recover_triples)?;
        rows.combos = Some(Combos {
            rank_buffers,
            values,
            ids: live_ids,
            pair_builder,
            active_outputs,
            support,
            host_ranks: rank,
            args: [0, 0, pairs as u32],
            last_slice: std::time::Instant::now(),
            slice_delay: std::time::Duration::ZERO,
            reuse,
        });
        if progressive {
            return Ok(());
        }
        let combos = rows.combos.as_mut().unwrap();
        for (arity, count, offset) in [(2, pairs, 0), (3, triples, pairs)] {
            if count == 0 {
                continue;
            }
            let p = table_pipeline_options(
                &self.device,
                DEFAULT_VARIANT | 64 | 128,
                if arity == 2 { build_kind } else { arity },
                0,
                if arity == 2 { self.pair_style } else { PairBuilder::General },
            )?;
            let start = std::time::Instant::now();
            for first in (0..count).step_by(32768) {
                combos.slice(&self.queue, &rows.buffer, &p, offset, first, (count - first).min(32768), limit)?;
            }
            if arity == 2 {
                combos.args[0] = pairs as u32;
            } else {
                combos.args[1] = limit as u32;
            }
            eprintln!(
                "live_rows={n} arity={arity} entries={count} bytes={} build_ms={:.2} triple_prefix={limit}",
                count * 32,
                start.elapsed().as_secs_f64() * 1000.
            );
        }
        combos.reuse = None;
        Ok(())
    }
}

#[cfg(feature = "metal-bench")]
impl MetalScanner {
    pub fn set_pair_reuse(&mut self, on: bool) {
        self.allow_reuse = on;
    }
    pub fn set_host_mask(&mut self, flags: u8) {
        self.host_mask = flags;
    }
    pub fn compile_pair_screen(&self, mask: u8) -> Result<ComputePipelineState, String> {
        metal_screen::Screen::variant(&self.device, false, mask & 3)
    }
    pub fn use_pair_screen(&mut self, p: &ComputePipelineState) -> Result<(), String> {
        self.use_screen_variant(&PairScreenVariant { pipeline: p.clone(), mode: 0, tuning: Default::default() })
    }
    pub fn compile_screen_variant(&self, mode: u8) -> Result<PairScreenVariant, String> {
        Ok(PairScreenVariant {
            pipeline: metal_screen::Screen::variant(&self.device, false, mode)?,
            mode,
            tuning: Default::default(),
        })
    }
    pub fn compile_selection_replay(&self, deferred_diagonals: bool) -> Result<PairScreenVariant, String> {
        let mode = metal_screen::pair_mode(self.compact_tickets);
        let mut tuning = metal_screen::PairTuning::production(self.compact_tickets, self.dataset.n());
        tuning.deferred_diagonals = deferred_diagonals;
        let pipeline = metal_screen::Screen::tuned_variant(&self.device, false, mode, tuning)?;
        Ok(PairScreenVariant { pipeline, mode, tuning })
    }
    pub fn compile_rank_tuning(&self, id: u8) -> Result<PairScreenVariant, String> {
        use metal_screen::{PairTuning, RankLayout, Screen};
        let threads = *[128, 64, 256].get((id / 3) as usize).ok_or("invalid tuning id")?;
        let ranks = [RankLayout::Plain, RankLayout::Tagged32, RankLayout::Tagged16][(id % 3) as usize];
        let tuning = PairTuning { ranks, threads, deferred_diagonals: true };
        let mode = metal_screen::pair_mode(self.compact_tickets);
        let pipeline = Screen::tuned_variant(&self.device, false, mode, tuning)?;
        Ok(PairScreenVariant { pipeline, mode, tuning })
    }
    pub fn use_screen_variant(&mut self, variant: &PairScreenVariant) -> Result<(), String> {
        if self.screen.is_none() {
            self.screen = Some(metal_screen::Screen::new(&self.device, self.compact_tickets, self.dataset.n())?);
        }
        self.screen.as_mut().unwrap().set_pair_pipeline(variant.pipeline.clone(), variant.mode, variant.tuning);
        Ok(())
    }
    pub fn compile_variant(&self, mask: u8) -> Result<(ComputePipelineState, u64), String> {
        pipeline(&self.device, mask)
    }
    pub fn use_variant(&mut self, variant: &(ComputePipelineState, u64)) {
        self.pipeline = variant.0.clone();
        self.width = variant.1;
    }
    pub fn gpu_seconds(&self) -> f64 {
        self.gpu_seconds
    }
    pub fn allocated_bytes(&self) -> u64 {
        self.device.current_allocated_size()
    }
    pub fn schedule(&mut self, capacity: usize, dispatch: usize) -> Result<(), String> {
        if dispatch == 0 || capacity == 0 || dispatch > capacity || capacity > 65536 {
            return Err("invalid Metal replay schedule".into());
        }
        if self.keys.len() < capacity {
            let indices = shared_buffer(&self.device, capacity * 64)?;
            let digests = shared_buffer(&self.device, capacity * 32)?;
            self.indices = indices;
            self.digests = digests;
            self.keys.resize(capacity, [0; 32]);
        }
        self.dispatch_size = dispatch;
        Ok(())
    }
}

impl Scanner for MetalScanner {
    fn mining_mode(&self) -> Option<&'static str> {
        Some(if self.use_screen { "selective" } else { "exhaustive" })
    }
    fn mining_capacity(&self) -> u32 {
        if self.use_screen { MINING_SCREEN_CAP } else { self.capacity() }
    }
    fn request_roll(&self) -> bool {
        self.allow_screen
    }
    fn mine_range(
        &mut self,
        model: &Model,
        header: &[u8; 80],
        start: u32,
        count: u32,
        bound: &[u8; 32],
    ) -> Result<(Vec<u32>, u32), String> {
        if !self.use_screen {
            if self.retry_screen.is_none() {
                return Ok((self.scan(header, start, count, bound)?, count));
            }
            let t = std::time::Instant::now();
            let hits = self.scan(header, start, count, bound)?;
            self.retry_after_full_batch(model, header, start, count, t.elapsed().as_secs_f64())?;
            return Ok((hits, count));
        }
        let rows = self.rows.as_mut().unwrap();
        rows.combos.as_mut().unwrap().advance(&self.queue, &rows.buffer)?;
        self.screen_batches += 1;
        if self.screen_batches.is_multiple_of(512 / SCREEN_BATCH_SCALE) {
            self.audit_screen(model, header, start, count.min(16))?;
        }
        let result = self.productive_range(header, start, count, bound)?;
        Ok((result.hits, result.evaluated))
    }
    fn capacity(&self) -> u32 {
        self.keys.len() as u32
    }
    fn audit_batches(&self) -> u64 {
        if self.use_screen {
            crate::AUDIT_BATCHES / SCREEN_BATCH_SCALE
        } else {
            (crate::AUDIT_BATCHES * 4096 / u64::from(self.capacity())).max(1)
        }
    }
    fn set_model(&mut self, _: &[i8], _: &[i8]) -> Result<(), String> {
        Err("Metal supports KEP1 models".into())
    }
    fn set_weights(&mut self, model: &Model) -> Result<(), String> {
        // Each scanner owns one immutable dataset. A refreshed job can resend
        // identical weights in a new Model; residuals do not affect its tickets.
        let prepared = self.ahead.as_ref().and_then(|ahead| ahead.take(model));
        if self.rows.as_ref().is_some_and(|rows| rows.weights == model.weights()) {
            return Ok(());
        }
        if let Some(mut prepared) = prepared {
            prepared.ahead = self.ahead.take();
            *self = *prepared;
            eprintln!("using verified TESS table prepared ahead");
            return Ok(());
        }
        let threads = self.build_threads.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        let table = Table::build(model, &*self.dataset, threads, vec![])?;
        if !cfg!(feature = "metal-bench") || self.allow_screen {
            // The trained fixtures were 65–77% idle. Use the compact scanner
            // only on sparse models; dense jobs keep the full-width kernel.
            let compact = table.idle_rows() * 2 >= self.dataset.n() as usize;
            if compact != (self.host_mask & 8 != 0) {
                let mode = DEFAULT_VARIANT | if compact { 64 | 128 } else { 0 };
                (self.pipeline, self.width) = pipeline(&self.device, mode)?;
                self.host_mask = if compact { 8 | 32 } else { 0 };
            }
        }
        let try_screen = self.allow_screen && table.idle_rows() * 2 >= self.dataset.n() as usize;
        let singles = if cfg!(feature = "metal-bench") || try_screen || self.host_mask & 4 != 0 {
            Some(mapped_single_lookup(&self.device, &table, threads)?)
        } else {
            None
        };
        let live =
            if cfg!(feature = "metal-bench") || self.host_mask & 8 != 0 { table.live_rows().to_vec() } else { vec![] };
        let table = table.into_spare();
        if table.len() != ROW * self.dataset.n() as usize {
            return Err("unexpected KEP1 row layout".into());
        }
        let len = STRIDE * self.dataset.n() as usize;
        let rows = match &mut self.rows {
            Some(rows) => rows,
            slot @ None => slot.insert(ModelRows {
                buffer: shared_buffer(&self.device, len)?,
                weights: Vec::new(),
                singles: None,
                live: vec![],
                combos: None,
            }),
        };
        // SAFETY: the slot's dataset fixes len for the lifetime of this shared
        // buffer. All dispatches complete synchronously under exclusive access.
        let out = unsafe { std::slice::from_raw_parts_mut(rows.buffer.contents().cast::<u8>(), len) };
        self.reuse_hint = None;
        let mut remap = Vec::new();
        if self.allow_reuse
            && let Some(old) = &rows.combos
        {
            let mut same = 0;
            for (i, (dst, src)) in out.as_chunks::<STRIDE>().0.iter().zip(table.as_chunks::<ROW>().0).enumerate() {
                if live.get(i) == Some(&true) {
                    let rank = old.host_ranks[i];
                    let unchanged = rank != u32::MAX && dst[..ROW] == src[..];
                    same += usize::from(unchanged);
                    remap.push(if unchanged { rank } else { u32::MAX });
                }
            }
            // If fewer than 10% of pairs can survive, a second random GPU read
            // and transient old table are unlikely to repay their cost.
            if old.args[0] > 0 && same * (same + 1) * 10 > remap.len() * (remap.len() + 1) {
                let ranks = shared_buffer(&self.device, remap.len() * 4)?;
                // SAFETY: new aligned shared storage with one u32 for every rank.
                unsafe {
                    std::ptr::copy_nonoverlapping(remap.as_ptr(), ranks.contents().cast(), remap.len());
                }
                self.reuse_hint = Some(PairReuse { ranks, old_values: old.values.clone(), ready: old.args[0] });
            }
        }
        let mut prior = remap.into_iter();
        #[cfg(feature = "metal-bench")]
        let copy_start = std::time::Instant::now();
        #[cfg(feature = "metal-bench")]
        {
            self.skipped_bytes = 0;
        }
        for (i, (dst, src)) in out.as_chunks_mut::<STRIDE>().0.iter_mut().zip(table.as_chunks::<ROW>().0).enumerate() {
            if self.skip_unchanged && live.get(i) == Some(&true) && prior.next().is_some_and(|r| r != u32::MAX) {
                #[cfg(feature = "metal-bench")]
                {
                    self.skipped_bytes += ROW;
                }
                continue;
            }
            dst[..ROW].copy_from_slice(src);
        }
        rows.weights.clear();
        #[cfg(feature = "metal-bench")]
        {
            self.copy_seconds = copy_start.elapsed().as_secs_f64();
        }
        rows.weights.extend_from_slice(model.weights());
        rows.singles = singles;
        rows.live = live;
        rows.combos = None;
        self.use_screen = false;
        self.retry_screen = None;
        if try_screen {
            match self.prepare_combos(0, true).and_then(|()| self.choose_screen(model)) {
                Ok(keep) => self.use_screen = keep,
                Err(e) => {
                    eprintln!("Metal sparse engine unavailable: {e}; using exhaustive scanner");
                }
            }
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
