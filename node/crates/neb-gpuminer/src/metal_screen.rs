//! FreeForGem-style bounded lookup screening. Throughput counts finished
//! tickets; nonces outside the prepared pair/triple domain earn no work.
use super::*;

pub const SCREEN_CAP: u32 = 262144;
pub const PROBE_CAP: u32 = 4096;
pub struct Screen {
    fast: ComputePipelineState,
    check: ComputePipelineState,
    counts: Buffer,
    hits: Buffer,
    records: Buffer,
}
pub struct ScreenResult {
    pub evaluated: u32,
    pub hits: Vec<u32>,
    pub records: Vec<[u32; 9]>,
}
impl Screen {
    pub fn new(device: &Device) -> Result<Self, String> {
        let compile = |check: u8| {
            let src =
                format!("#define CHECK_RECORDS {check}\n{}", include_str!("../../../../pow/gpu/kep1_screen.metal"));
            let lib = device.new_library_with_source(&src, &CompileOptions::new())?;
            let f = lib.get_function("lookup_screen", None)?;
            device.new_compute_pipeline_state_with_function(&f)
        };
        Ok(Self {
            fast: compile(0)?,
            check: compile(1)?,
            counts: shared_buffer(device, 8)?,
            hits: shared_buffer(device, SCREEN_CAP as usize * 4)?,
            records: shared_buffer(device, PROBE_CAP as usize * 36)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn scan(
        &mut self,
        queue: &CommandQueue,
        rows: &ModelRows,
        n: u32,
        header: &[u8; 80],
        start: u32,
        count: u32,
        bound: &[u8; 32],
        arity: u8,
        check: bool,
    ) -> Result<ScreenResult, String> {
        if count == 0 || count > SCREEN_CAP || (check && count > PROBE_CAP) || start.checked_add(count - 1).is_none() {
            return Err("invalid screen range".into());
        }
        let tables = rows.combos.as_ref().ok_or("combination tables unavailable")?;
        let single = rows.singles.as_ref().ok_or("single table unavailable")?;
        let mut args = [0u32; 24];
        let mut cv = HASH_IV;
        blake3::platform::Platform::detect().compress_in_place(
            &mut cv,
            header[..64].try_into().unwrap(),
            64,
            0,
            CHUNK_START,
        );
        args[..8].copy_from_slice(&cv);
        for (word, bytes) in args[8..11].iter_mut().zip(header[64..76].as_chunks::<4>().0) {
            *word = u32::from_le_bytes(*bytes);
        }
        args[11..16].copy_from_slice(&[
            start,
            n,
            if arity >= 2 { tables.args[0] } else { 0 },
            if arity >= 3 { tables.args[1] } else { 0 },
            tables.args[2],
        ]);
        args[16..].copy_from_slice(&blake3::platform::words_from_le_bytes_32(bound));
        // SAFETY: exclusive access; the owned shared buffer holds two aligned u32 counters.
        unsafe {
            self.counts.contents().cast::<[u32; 2]>().write([0, 0]);
        }
        autoreleasepool(|| {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(if check { &self.check } else { &self.fast });
            enc.set_bytes(0, 96, args.as_ptr().cast());
            for (i, buf) in
                [&tables.ranks, single, &tables.values, &self.counts, &self.hits, &self.records].iter().enumerate()
            {
                enc.set_buffer((i + 1) as u64, Some(buf), 0);
            }
            enc.dispatch_threads(MTLSize::new(count as u64, 1, 1), MTLSize::new(128, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            if cmd.status() != MTLCommandBufferStatus::Completed {
                return Err("Metal nonce screen failed".to_string());
            }
            Ok(())
        })?;
        // SAFETY: the completed command wrote these two counters. Their values
        // are bounded below before they can be used as output lengths.
        let [evaluated, found] = unsafe { self.counts.contents().cast::<[u32; 2]>().read() };
        if evaluated > count || found > evaluated {
            return Err("invalid GPU screen count".into());
        }
        // SAFETY: found <= count <= SCREEN_CAP. No writer is in flight.
        let hits = unsafe { std::slice::from_raw_parts(self.hits.contents().cast::<u32>(), found as usize) }.to_vec();
        let records = if check {
            // SAFETY: count <= PROBE_CAP and the validation pipeline wrote one record per input.
            unsafe { std::slice::from_raw_parts(self.records.contents().cast::<[u32; 9]>(), count as usize) }.to_vec()
        } else {
            Vec::new()
        };
        Ok(ScreenResult { evaluated, hits, records })
    }
}
