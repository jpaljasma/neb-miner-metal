//! Model-turnover replay controls and table parity measurements.
use super::*;

impl MetalScanner {
    pub fn use_direct_pair_builder(&mut self) {
        self.pair_style = PairBuilder::DirectMaskedTwoRows;
    }
    pub fn turnover_experiment(&mut self, mask: u8, masked_builder: bool) {
        self.recover_triples = mask & 1 != 0;
        self.skip_unchanged = mask & 4 != 0;
        self.pair_style = match (mask & 2 != 0, masked_builder) {
            (false, _) => PairBuilder::General,
            (true, false) => PairBuilder::TwoRows,
            (true, true) => PairBuilder::MaskedTwoRows,
        };
    }
    pub fn turnover_evaluate(
        &mut self,
        h: &[u8; 80],
        start: u32,
        n: u32,
        bound: &[u8; 32],
    ) -> Result<ScreenResult, String> {
        self.productive_range(h, start, n, bound)
    }
    pub fn pair_prefix_hash(&self) -> blake3::Hash {
        let table = self.rows.as_ref().unwrap().combos.as_ref().unwrap();
        // SAFETY: synchronous commands have finished writing the published prefix.
        blake3::hash(unsafe { std::slice::from_raw_parts(table.values.contents().cast(), table.args[0] as usize * 32) })
    }
    pub fn turnover_progress(&self) -> (u32, usize) {
        (self.rows.as_ref().unwrap().combos.as_ref().unwrap().args[0], self.skipped_bytes)
    }
    pub fn turnover_limit_pairs(&mut self, maximum: u32) {
        let c = self.rows.as_mut().unwrap().combos.as_mut().unwrap();
        c.args[0] = c.args[0].min(maximum);
    }
    pub fn turnover_prepare_prefix(&mut self) -> Result<(), String> {
        self.prepare_combos(0, true)?;
        let r = self.rows.as_mut().unwrap();
        r.combos.as_mut().unwrap().advance(&self.queue, &r.buffer)
    }

    pub fn turnover_upload_check(&self) -> (f64, usize, blake3::Hash) {
        let r = self.rows.as_ref().unwrap();
        // SAFETY: all commands finish before the exclusive model refresh returns.
        let bytes = unsafe { std::slice::from_raw_parts(r.buffer.contents().cast(), r.buffer.length() as usize) };
        (self.copy_seconds, self.skipped_bytes, blake3::hash(bytes))
    }
    pub fn benchmark_builders(&mut self, slot: usize, geometry: bool, seed: u8) -> Result<(), String> {
        let r = self.rows.as_ref().ok_or("missing rows")?;
        let c = r.combos.as_ref().ok_or("missing pairs")?;
        let count = c.args[2].min(65536) as usize;
        if count == 0 || c.active_outputs.is_none() {
            return Err("builder replay requires a populated masked model".into());
        }
        let kind = if c.reuse.is_some() { 4 } else { 2 };
        let pipelines = if geometry {
            (0..9)
                .map(|id| {
                    table_pipeline_geometry(
                        &self.device,
                        DEFAULT_VARIANT | 64 | 128,
                        kind,
                        0,
                        PairBuilder::MaskedTwoRows,
                        [256, 128, 64][id / 3],
                        (id % 3) as u8,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            [PairBuilder::General, PairBuilder::TwoRows, PairBuilder::MaskedTwoRows]
                .into_iter()
                .map(|style| table_pipeline_options(&self.device, DEFAULT_VARIANT | 64 | 128, kind, 0, style))
                .collect::<Result<Vec<_>, _>>()?
        };
        for block in 0..=48usize {
            let offset = if geometry { u64::from(seed) } else { 0 };
            let first = ((block as u64 + offset) * 2654435761 % u64::from(c.args[2] - count as u32 + 1)) as usize;
            let arms = pipelines.len();
            let mut order: Vec<_> = (0..arms).collect();
            order.rotate_left((block + slot) % arms);
            if block / arms % 2 == 1 {
                order.reverse();
            }
            let mut expected = None;
            for style in order {
                let start = std::time::Instant::now();
                c.slice(&self.queue, &r.buffer, &pipelines[style], 0, first, count, 0)?;
                let ms = start.elapsed().as_secs_f64() * 1000.;
                // SAFETY: the completed command wrote this bounded slice of the allocated pair table.
                let digest = blake3::hash(unsafe {
                    std::slice::from_raw_parts(c.values.contents().cast::<u8>().add(first * 32), count * 32)
                });
                if expected.replace(digest).is_some_and(|other| other != digest) {
                    return Err("pair slice changed".into());
                }
                if block != 0 {
                    println!(
                        "builder,{slot},{block},{style},{count},{ms:.6},{}",
                        pipelines[style].0.static_threadgroup_memory_length()
                    );
                }
            }
        }
        Ok(())
    }
}
