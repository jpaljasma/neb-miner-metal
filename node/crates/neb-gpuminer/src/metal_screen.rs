//! FreeForGem-style bounded lookup screening. Throughput counts finished
//! tickets; nonces outside the prepared pair/triple domain earn no work.
use super::*;

/// Size of the fixed calibration/replay range.
pub const SCREEN_CAP: u32 = 262144;
/// Amortize one synchronous GPU submission across sixteen calibration ranges.
pub const MINING_SCREEN_CAP: u32 = SCREEN_CAP * 16;
pub const PROBE_CAP: u32 = 4096;
const SCALAR_CHACHA: u8 = 1;
const PAIRS_ONLY: u8 = 2;
const MAPPED_ARGS: u8 = 4;
const HEADER_COLUMNS: u8 = 8;
const HIGH_WORD: u8 = 16;
const GROUP_ARGS: u8 = 32;
const RANK_QUADS: u8 = 64;
const COMPACT_TICKETS: u8 = 128;
const DEFAULT_SCREEN: u8 = PAIRS_ONLY | SCALAR_CHACHA | HEADER_COLUMNS | RANK_QUADS;
const SCREEN_GROUP_THREADS: u64 = 128;
type ScreenArgs = [u32; 40];

pub(super) fn pair_mode(compact_tickets: bool) -> u8 {
    DEFAULT_SCREEN | if compact_tickets { COMPACT_TICKETS } else { 0 }
}

#[derive(Clone, Copy, Default)]
#[repr(u8)]
pub(super) enum RankLayout {
    #[default]
    Plain,
    Tagged32,
    Tagged16,
}

#[derive(Clone, Copy)]
pub(super) struct PairTuning {
    pub ranks: RankLayout,
    pub threads: u64,
    pub deferred_diagonals: bool,
}
impl Default for PairTuning {
    fn default() -> Self {
        Self { ranks: RankLayout::Plain, threads: SCREEN_GROUP_THREADS, deferred_diagonals: false }
    }
}

impl PairTuning {
    pub(super) fn production(compact_tickets: bool, rows: u32) -> Self {
        // With at most 16,384 rows, even two maximum ranks cannot carry into bit 15.
        let ranks = if compact_tickets && rows <= 16_384 { RankLayout::Tagged16 } else { RankLayout::Tagged32 };
        Self { ranks, threads: SCREEN_GROUP_THREADS, deferred_diagonals: true }
    }
}
impl RankLayout {
    pub(super) fn build(self, device: &metal::DeviceRef, ranks: &[u32]) -> Result<Buffer, String> {
        fn fill<T>(device: &metal::DeviceRef, ranks: &[u32], tag: u32) -> Result<Buffer, String>
        where
            T: TryFrom<u32> + Copy,
        {
            let storage = shared_buffer(device, size_of::<T>() * ranks.len())?;
            // SAFETY: this fresh Metal allocation holds ranks.len() aligned u16 or u32 entries.
            // Only live rows need writes. Metal zeroes idle rows and no command has the buffer yet.
            let out = unsafe {
                std::slice::from_raw_parts_mut(storage.contents().cast::<std::mem::MaybeUninit<T>>(), ranks.len())
            };
            for (&rank, entry) in ranks.iter().zip(out) {
                if rank != u32::MAX {
                    // The sum of two live ranks must fit below one count bit.
                    if rank >= tag / 2 {
                        return Err("rank sum exceeds the packed lookup domain".into());
                    }
                    entry.write(T::try_from(tag | rank).map_err(|_| "tagged rank overflow")?);
                }
            }
            Ok(storage)
        }
        match self {
            Self::Tagged32 => fill::<u32>(device, ranks, 1 << 27),
            Self::Tagged16 => fill::<u16>(device, ranks, 1 << 15),
            Self::Plain => {
                let storage = shared_buffer(device, std::mem::size_of_val(ranks))?;
                // SAFETY: fresh, aligned storage exactly fits the source ranks.
                unsafe { std::ptr::copy_nonoverlapping(ranks.as_ptr(), storage.contents().cast(), ranks.len()) };
                Ok(storage)
            }
        }
    }
}

// Finish three independent BLAKE3 columns on the host. Column 1 stops just
// before message word 3 (the nonce) enters the second half of its G function.
fn prepare_header(args: &mut ScreenArgs, header: &[u8; 76], columns: bool) {
    let mut cv = HASH_IV;
    blake3::platform::Platform::detect().compress_in_place(
        &mut cv,
        header[..64].try_into().unwrap(),
        64,
        0,
        CHUNK_START,
    );
    args[..8].copy_from_slice(&cv);
    for (word, bytes) in args[8..11].iter_mut().zip(header[64..].as_chunks::<4>().0) {
        *word = u32::from_le_bytes(*bytes);
    }
    if !columns {
        return;
    }
    let words = [args[8], args[9], args[10], 0, 0, 0, 0, 0];
    for i in 0..4 {
        let mut a = cv[i].wrapping_add(cv[i + 4]).wrapping_add(words[2 * i]);
        let mut d = (a ^ [0, 0, 16, u32::from(CHUNK_END | ROOT)][i]).rotate_right(16);
        let mut c = HASH_IV[i].wrapping_add(d);
        let mut b = (cv[i + 4] ^ c).rotate_right(12);
        if i != 1 {
            a = a.wrapping_add(b).wrapping_add(words[2 * i + 1]);
            d = (d ^ a).rotate_right(8);
            c = c.wrapping_add(d);
            b = (b ^ c).rotate_right(7);
        }
        args[24 + i] = a;
        args[28 + i] = b;
        args[32 + i] = c;
        args[36 + i] = d;
    }
}
pub struct Screen {
    pair: ComputePipelineState,
    recovered: Option<ComputePipelineState>,
    compact_tickets: bool,
    tuning: PairTuning,
    reference_tuning: PairTuning,
    general: Option<ComputePipelineState>,
    check: ComputePipelineState,
    parameters: Buffer,
    #[cfg(feature = "metal-bench")]
    mode: u8,
    prefix: Option<[u8; 76]>,
    counts: Buffer,
    hits: Buffer,
    records: Buffer,
}
pub struct ScreenResult {
    #[cfg(feature = "metal-bench")]
    pub gpu_s: f64,
    pub evaluated: u32,
    pub hits: Vec<u32>,
    pub records: Vec<[u32; 9]>,
}
impl Screen {
    #[cfg(feature = "metal-bench")]
    pub(super) fn variant(device: &metal::DeviceRef, check: bool, mask: u8) -> Result<ComputePipelineState, String> {
        Self::tuned_variant(device, check, mask, PairTuning::default())
    }
    pub(super) fn tuned_variant(
        device: &metal::DeviceRef,
        check: bool,
        mask: u8,
        tuning: PairTuning,
    ) -> Result<ComputePipelineState, String> {
        Self::compile(device, check, mask, tuning, false)
    }
    fn compile(
        device: &metal::DeviceRef,
        check: bool,
        mask: u8,
        tuning: PairTuning,
        recover: bool,
    ) -> Result<ComputePipelineState, String> {
        let src = format!(
            concat!(
                "#define DEFER_DIAGONALS {}\n#define CHECK_RECORDS {}\n#define SCALAR_CHACHA {}\n#define PAIRS_ONLY {}\n",
                "#define HEADER_COLUMNS {}\n#define HIGH_WORD {}\n#define GROUP_ARGS {}\n",
                "#define RANK_QUADS {}\n#define COMPACT_TICKETS {}\n#define SCREEN_GROUP_THREADS {}\n",
                "#define RANK_LAYOUT {}\n#define RECOVER_THREE {}\n#define SUPPORT_WORDS {}\n{}",
            ),
            u8::from(tuning.deferred_diagonals),
            u8::from(check),
            u8::from(mask & SCALAR_CHACHA != 0),
            u8::from(mask & PAIRS_ONLY != 0),
            u8::from(mask & HEADER_COLUMNS != 0),
            u8::from(mask & HIGH_WORD != 0),
            u8::from(mask & GROUP_ARGS != 0),
            u8::from(mask & RANK_QUADS != 0),
            u8::from(mask & COMPACT_TICKETS != 0),
            tuning.threads,
            tuning.ranks as u8,
            u8::from(recover),
            SUPPORT_WORDS,
            include_str!("../../../../pow/gpu/kep1_screen.metal")
        );
        let lib = device.new_library_with_source(&src, &CompileOptions::new())?;
        let f = lib.get_function("lookup_screen", None)?;
        let pipeline = device.new_compute_pipeline_state_with_function(&f)?;
        if pipeline.thread_execution_width() != 32 || pipeline.max_total_threads_per_threadgroup() < tuning.threads {
            return Err(format!("screen requires 32-lane SIMD and {} threads per group", tuning.threads));
        }
        Ok(pipeline)
    }
    #[cfg(feature = "metal-bench")]
    pub(super) fn set_pair_pipeline(&mut self, p: ComputePipelineState, mode: u8, tuning: PairTuning) {
        self.pair = p;
        self.recovered = None;
        self.mode = mode;
        self.tuning = tuning;
    }
    pub fn new(device: &Device, compact_tickets: bool, rows: u32) -> Result<Self, String> {
        let mode = pair_mode(compact_tickets);
        let tuning = PairTuning::production(compact_tickets, rows);
        let reference_tuning = PairTuning { ranks: tuning.ranks, ..Default::default() };
        Ok(Self {
            pair: Self::tuned_variant(device, false, mode, tuning)?,
            recovered: None,
            compact_tickets,
            tuning,
            reference_tuning,
            general: None,
            check: Self::tuned_variant(device, true, 0, reference_tuning)?,
            parameters: shared_buffer(device, size_of::<ScreenArgs>())?,
            #[cfg(feature = "metal-bench")]
            mode,
            prefix: None,
            counts: shared_buffer(device, 8)?,
            hits: shared_buffer(device, MINING_SCREEN_CAP as usize * 4)?,
            records: shared_buffer(device, PROBE_CAP as usize * 36)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn scan(
        &mut self,
        queue: &CommandQueue,
        rows: &mut ModelRows,
        n: u32,
        header: &[u8; 80],
        start: u32,
        count: u32,
        bound: &[u8; 32],
        arity: u8,
        check: bool,
        recover: bool,
    ) -> Result<ScreenResult, String> {
        if count == 0
            || count > MINING_SCREEN_CAP
            || (check && count > PROBE_CAP)
            || start.checked_add(count - 1).is_none()
        {
            return Err("invalid screen range".into());
        }
        let tables = rows.combos.as_mut().ok_or("combination tables unavailable")?;
        let single = rows.singles.as_ref().ok_or("single table unavailable")?;
        if recover && (check || arity != 2 || tables.support.is_none()) {
            return Err("missing exact three-row support".into());
        }
        let pipeline = if recover {
            match &mut self.recovered {
                Some(p) => p,
                empty @ None => empty.insert(Self::compile(
                    queue.device(),
                    false,
                    pair_mode(self.compact_tickets),
                    self.tuning,
                    true,
                )?),
            }
        } else if check {
            &self.check
        } else if arity <= 2 {
            &self.pair
        } else {
            match &mut self.general {
                Some(pipeline) => pipeline,
                empty @ None => empty.insert(Self::tuned_variant(queue.device(), false, 0, self.reference_tuning)?),
            }
        };
        let tuning = if check || arity > 2 { self.reference_tuning } else { self.tuning };
        let ranks = match &mut tables.rank_buffers[tuning.ranks as usize] {
            Some(table) => &*table,
            empty @ None => &*empty.insert(tuning.ranks.build(queue.device(), &tables.host_ranks)?),
        };
        let mut inline_args = [0u32; 40];
        #[cfg(feature = "metal-bench")]
        let mode = self.mode;
        #[cfg(not(feature = "metal-bench"))]
        let mode = DEFAULT_SCREEN;
        let columns = mode & HEADER_COLUMNS != 0;
        let mapped = (columns || mode & MAPPED_ARGS != 0).then_some(&self.parameters);
        let args = if let Some(buffer) = mapped {
            // SAFETY: this zero-initialized mapping holds one aligned ScreenArgs. The last
            // dispatch completed and exclusive access to Screen covers the next submission.
            unsafe { &mut *buffer.contents().cast::<ScreenArgs>() }
        } else {
            &mut inline_args
        };
        let prefix: &[u8; 76] = header[..76].try_into().unwrap();
        if !columns || self.prefix.as_ref() != Some(prefix) {
            prepare_header(args, prefix, columns);
            if mapped.is_some() {
                self.prefix = columns.then_some(*prefix);
            }
        }
        args[11..16].copy_from_slice(&[
            start,
            n,
            if arity >= 2 { tables.args[0] } else { 0 },
            if arity >= 3 { tables.args[1] } else { 0 },
            tables.args[2],
        ]);
        args[16..24].copy_from_slice(&blake3::platform::words_from_le_bytes_32(bound));
        // SAFETY: exclusive access; the owned shared buffer holds two aligned u32 counters.
        unsafe {
            self.counts.contents().cast::<[u32; 2]>().write([0, 0]);
        }
        #[cfg(feature = "metal-bench")]
        let mut gpu_s = 0.0;
        autoreleasepool(|| {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            if recover {
                enc.set_buffer(8, tables.support.as_deref(), 0);
            }
            if let Some(buffer) = mapped {
                enc.set_buffer(0, Some(buffer), 0);
            } else {
                enc.set_bytes(0, 96, inline_args.as_ptr().cast());
            }
            for (i, buf) in [ranks, single, &tables.values, &self.counts, &self.hits, &self.records].iter().enumerate()
            {
                enc.set_buffer((i + 1) as u64, Some(buf), 0);
            }
            if !check && arity <= 2 && !matches!(tuning.ranks, RankLayout::Plain) {
                enc.set_buffer(7, Some(&tables.ids), 0);
            }
            enc.dispatch_threads(MTLSize::new(count as u64, 1, 1), MTLSize::new(tuning.threads, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            if cmd.status() != MTLCommandBufferStatus::Completed {
                return Err("Metal nonce screen failed".to_string());
            }
            #[cfg(feature = "metal-bench")]
            {
                // SAFETY: a completed MTLCommandBuffer exposes f64 GPU timestamps.
                #[allow(unexpected_cfgs)]
                let (begin, end): (f64, f64) = unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
                gpu_s = end - begin;
            }
            Ok(())
        })?;
        // SAFETY: the completed command wrote these two counters. Their values
        // are bounded below before they can be used as output lengths.
        let [evaluated, found] = unsafe { self.counts.contents().cast::<[u32; 2]>().read() };
        if evaluated > count || found > evaluated {
            return Err("invalid GPU screen count".into());
        }
        // SAFETY: found <= count <= MINING_SCREEN_CAP. No writer is in flight.
        let hits = unsafe { std::slice::from_raw_parts(self.hits.contents().cast::<u32>(), found as usize) }.to_vec();
        let records = if check {
            // SAFETY: count <= PROBE_CAP and the validation pipeline wrote one record per input.
            unsafe { std::slice::from_raw_parts(self.records.contents().cast::<[u32; 9]>(), count as usize) }.to_vec()
        } else {
            Vec::new()
        };
        Ok(ScreenResult {
            evaluated,
            hits,
            records,
            #[cfg(feature = "metal-bench")]
            gpu_s,
        })
    }
}
