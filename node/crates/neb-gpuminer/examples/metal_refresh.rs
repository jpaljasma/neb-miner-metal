//! Replay identical weights arriving in fresh Model frames. Run with a pinned mainnet dataset.
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the Metal replay requires macOS");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    replay::run()
}

#[cfg(target_os = "macos")]
mod replay {
    // The same source also builds as a standalone example in the pinned PR #2
    // checkout. Only the dispatch adapter differs; all inputs and timers are shared.
    #[cfg(not(feature = "metal"))]
    use super::pr2::MetalScanner;
    #[cfg(feature = "metal")]
    use neb_gpuminer::{Scanner, metal_scanner::MetalScanner};
    use neb_pow::shares::{Arch, Model, ShareBuffers, load_datasets};
    use std::{path::Path, time::Instant};

    const COUNT: u32 = 8192;
    const BATCH: u32 = 4096;
    const START: u32 = u32::MAX - (COUNT - 1);
    // Independent PR #2 replay on the same ordered nonce range.
    const STREAM_HASHES: [&str; 2] = [
        "4920248cc99fed55ebe13ee797846228b3f08ab0a96d805e166f4aefe06bde14",
        "2df194c5886d16874c2d719afa0b38656376b084f7e7c45b8795a674579cb818",
    ];

    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    }

    pub fn run() -> Result<(), String> {
        let path = std::env::args().nth(1).ok_or("expected the mainnet dataset path")?;
        let pin = blake3::Hash::from_hex("790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72").unwrap();
        let sets = load_datasets(Path::new(&path), Some(pin.as_bytes()))?;
        let genesis =
            blake3::Hash::from_hex("da2f5af7bb4c5e2bcc8d558b2f4f01d4c88557a943e6cff9cd035638fe9439ae").unwrap();
        for (slot, dataset) in sets.iter().enumerate() {
            let mut seed = blake3::Hasher::new();
            seed.update(b"neb-model-slot").update(genesis.as_bytes()).update(&(slot as u32).to_le_bytes());
            let model = Model::from_seed_arch(Arch::Kep1, seed.finalize().as_bytes());
            let packed = model.packed_weights();
            let mut header = std::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(slot as u8));
            header[76..].fill(0);
            let mut engine = MetalScanner::open(dataset.clone())?;
            let t = Instant::now();
            engine.set_weights(&model)?;
            let cold_ms = t.elapsed().as_secs_f64() * 1e3;
            let mut refresh_us = vec![];
            let mut rates = vec![];
            let mut whole_ms = vec![];
            let expected = blake3::Hash::from_hex(STREAM_HASHES[slot]).unwrap();
            for rep in 0..6 {
                // A new work-service Model frame decodes into a distinct allocation.
                let repeated = Model::from_packed(Arch::Kep1, &packed)?;
                let mut tickets = Vec::<[u8; 32]>::with_capacity(COUNT as usize);
                let begin = Instant::now();
                engine.set_weights(&repeated)?;
                let ready = Instant::now();
                for offset in (0..COUNT).step_by(BATCH as usize) {
                    tickets.extend_from_slice(engine.tickets(&header, START + offset, BATCH)?);
                }
                let end = Instant::now();
                // Compare the complete ordered ticket stream for the two implementations.
                let sum = blake3::hash(tickets.as_flattened());
                if sum != expected {
                    return Err(format!("slot {slot} ticket stream differs from the independent replay"));
                }
                if rep > 0 {
                    refresh_us.push((ready - begin).as_secs_f64() * 1e6);
                    rates.push(COUNT as f64 / (end - ready).as_secs_f64());
                    whole_ms.push((end - begin).as_secs_f64() * 1e3);
                }
            }
            // Both transitions must invalidate any previously prepared table.
            let changed = Model::from_seed_arch(Arch::Kep1, &[42; 32]);
            for fresh in [&changed, &model] {
                engine.set_weights(fresh)?;
                let mut scratch = ShareBuffers::new();
                let tickets = engine.tickets(&header, START, 16)?;
                for (i, ticket) in tickets.iter().enumerate() {
                    let mut h = header;
                    h[76..].copy_from_slice(&(START + i as u32).to_le_bytes());
                    if scratch.ticket(fresh, &**dataset, blake3::hash(&h).as_bytes()) != *ticket {
                        return Err(format!("changed model parity failed at slot {slot} offset {i}"));
                    }
                }
            }
            println!(
                "slot={slot} cold_ms={cold_ms:.3} median_refresh_us={:.3} median_scan_per_s={:.0} median_refresh_and_scan_ms={:.3} tickets_hash={}",
                median(&mut refresh_us),
                median(&mut rates),
                median(&mut whole_ms),
                expected
            );
            // Compare the upstream guide's per-core KEP1 estimate to the same
            // portable C path on one thread here. Exclude the Metal dispatch.
            let tickets = engine.tickets(&header, START, 256)?;
            let mut scratch = ShareBuffers::new();
            let mut per_ticket_ms = [0.0; 3];
            for sample in &mut per_ticket_ms {
                let t = Instant::now();
                for (i, ticket) in tickets.iter().enumerate() {
                    let mut h = header;
                    h[76..].copy_from_slice(&(START + i as u32).to_le_bytes());
                    if scratch.ticket(&model, &**dataset, blake3::hash(&h).as_bytes()) != *ticket {
                        return Err("single-thread CPU parity failed".into());
                    }
                }
                *sample = 1e3 * t.elapsed().as_secs_f64() / tickets.len() as f64;
            }
            println!("slot={slot} cpu_ref_single_thread_ms_per_ticket={:.3}", median(&mut per_ticket_ms));
        }
        Ok(())
    }
}

#[cfg(all(target_os = "macos", not(feature = "metal")))]
mod pr2 {
    use neb_metalminer::MetalEngine;
    use neb_pow::shares::{Model, SharedDataset};

    pub struct MetalScanner {
        engine: MetalEngine,
        tickets: Vec<[u8; 32]>,
    }

    impl MetalScanner {
        pub fn open(dataset: SharedDataset) -> Result<Self, String> {
            Ok(Self { engine: MetalEngine::new(&*dataset, 4096)?, tickets: Vec::with_capacity(4096) })
        }

        pub fn set_weights(&mut self, model: &Model) -> Result<(), String> {
            self.engine.set_model(model)
        }

        pub fn tickets(&mut self, header: &[u8; 80], start: u32, count: u32) -> Result<&[[u8; 32]], String> {
            self.tickets.clear();
            self.tickets.extend(self.engine.scan(header, start, count)?.iter().map(|record| record.ticket_bytes()));
            Ok(&self.tickets)
        }
    }
}
