//! Matched two-second Kepler lead and five-second TESS block-start replay.
#[cfg(not(target_os = "macos"))]
fn main() {
    panic!("requires Metal")
}
#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    replay::run()
}

#[cfg(target_os = "macos")]
mod replay {
    use neb_block::Target;
    use neb_gpuminer::{Scanner, metal_scanner::MetalScanner};
    use neb_pow::shares::{Arch, Dataset, Model, ShareBuffers, load_datasets};
    use std::{
        fs,
        path::Path,
        thread,
        time::{Duration, Instant},
    };

    fn load(dir: &Path, slot: usize) -> Result<Model, String> {
        let bytes = fs::read(dir.join(format!("slot-{slot}.bin"))).map_err(|e| e.to_string())?;
        let model = Model::from_packed(Arch::Kep1, &bytes)?;
        if fs::read_to_string(dir.join(format!("slot-{slot}.hash"))).map_err(|e| e.to_string())?.trim()
            != blake3::Hash::from(model.weights_hash()).to_hex().as_str()
        {
            return Err("replay model hash mismatch".into());
        }
        Ok(model)
    }

    fn work(s: &mut MetalScanner, m: &Model, ds: &dyn Dataset, mut h: [u8; 80], end: Instant) -> Result<u64, String> {
        let mut total = 0u64;
        let mut raw = 0u64;
        let mut cpu = ShareBuffers::new();
        let mut bound = [0xff; 32];
        bound[29..].fill(0);
        bound[28] = 0x3f;
        while Instant::now() < end {
            let start = raw as u32;
            let n = (s.mining_capacity() as u64).min((1u64 << 32) - u64::from(start)) as u32;
            h[40..44].copy_from_slice(&((raw >> 32) as u32).to_le_bytes());
            let (hits, evaluated) = s.mine_range(m, &h, start, n, &bound)?;
            for nonce in hits {
                h[76..].copy_from_slice(&nonce.to_le_bytes());
                if !Target(bound).admits(&cpu.ticket(m, ds, blake3::hash(&h).as_bytes())) {
                    return Err("CPU rejected a block-start replay hit".into());
                }
            }
            raw += u64::from(n);
            total += u64::from(evaluated);
        }
        Ok(total)
    }

    pub fn run() -> Result<(), String> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        if args.len() == 3 && args[2] == "selector" {
            let pin =
                blake3::Hash::from_hex("790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72").unwrap();
            let sets = load_datasets(Path::new(&args[0]), Some(pin.as_bytes()))?;
            let model = load(Path::new(&args[1]), 0)?;
            let mut scanner = MetalScanner::open(sets[0].clone())?;
            scanner.enable_screen();
            let clock = Instant::now();
            scanner.set_weights(&model)?;
            let mut total = 0;
            for i in 1..=12 {
                let mut header = [0u8; 80];
                header[8] = i as u8;
                let n = work(&mut scanner, &model, &*sets[0], header, clock + Duration::from_secs(i * 5))?;
                total += n;
                println!("{i},{n},{total}");
            }
            return Ok(());
        }
        if args.len() != 2 {
            return Err("DATASET ADJACENT_MODELS_DIRECTORY".into());
        }
        let pin = blake3::Hash::from_hex("790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72").unwrap();
        let sets = load_datasets(Path::new(&args[0]), Some(pin.as_bytes()))?;
        let dir = Path::new(&args[1]);
        let k = load(&dir.join("0"), 0)?;
        let models = [load(&dir.join("0"), 1)?, load(&dir.join("1"), 1)?];
        let mut kepler = MetalScanner::open(sets[0].clone())?;
        kepler.enable_screen();
        kepler.set_weights(&k)?;
        let mut tess = [MetalScanner::open(sets[1].clone())?, MetalScanner::open(sets[1].clone())?];
        for s in &mut tess {
            s.enable_screen();
        }
        let ahead = tess[1].enable_tess_ahead();
        // Nonmatching hints must be harmless, and an invalid declared hash must be refused.
        if ahead.prepare(&[0; 32], &models[0].packed_weights()).is_ok() {
            return Err("invalid hint was accepted".into());
        }
        ahead.prepare(&models[0].weights_hash(), &models[0].packed_weights())?;
        tess[1].set_weights(&models[1])?;
        let mut scratch = ShareBuffers::new();
        for (i, t) in tess[1].tickets(&[0; 80], 0, 16)?.iter().enumerate() {
            let mut h = [0; 80];
            h[76..].copy_from_slice(&(i as u32).to_le_bytes());
            if *t != scratch.ticket(&models[1], &*sets[1], blake3::hash(&h).as_bytes()) {
                return Err("stale hint changed ticket".into());
            }
        }
        work(&mut kepler, &k, &*sets[0], [1; 80], Instant::now() + Duration::from_secs(12))?;
        println!("block,ahead,kepler_tickets,tess_tickets,setup_ms");
        for block in 0..=10usize {
            let model = &models[block % 2];
            let mut order = [0, 1];
            if (block / 2) % 2 == 1 {
                order.reverse();
            }
            for id in order {
                let mut h = [17; 80];
                h[..4].copy_from_slice(&(block as u32).to_le_bytes());
                let (packed, hash) = (model.packed_weights(), model.weights_hash());
                let ahead = ahead.clone();
                let prep = (id == 1).then(|| thread::spawn(move || ahead.prepare(&hash, &packed)));
                let kcount = work(&mut kepler, &k, &*sets[0], h, Instant::now() + Duration::from_secs(2))?;
                let t = Instant::now();
                tess[id].set_weights(model)?;
                let setup = t.elapsed().as_secs_f64() * 1000.;
                h[4] = 1;
                let tcount = work(&mut tess[id], model, &*sets[1], h, t + Duration::from_secs(5))?;
                if let Some(p) = prep {
                    p.join().map_err(|_| "preparation thread failed")??;
                }
                if block > 0 {
                    println!("{block},{id},{kcount},{tcount},{setup:.3}");
                }
            }
        }
        Ok(())
    }
}
