//! Compare effective computed tickets/s, including the cost of screening skipped nonces.
#[cfg(not(target_os = "macos"))]
fn main() {
    panic!("requires Metal");
}
#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    replay::run()
}

#[cfg(target_os = "macos")]
mod replay {
    use neb_block::Target;
    use neb_engines::table::Table;
    use neb_gpuminer::{
        Scanner,
        metal_scanner::{MetalScanner, SCREEN_CAP},
    };
    use neb_pow::shares::{Arch, Model, ShareBuffers, load_datasets, share_indices};
    use std::{fs, path::Path, time::Instant};

    pub fn run() -> Result<(), String> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        if args.len() != 2 {
            return Err("DATASET SNAPSHOT_DIR".into());
        }
        let pin = blake3::Hash::from_hex("790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72").unwrap();
        let sets = load_datasets(Path::new(&args[0]), Some(pin.as_bytes()))?;
        println!("block,slot,arity,screened,evaluated,wall_ms");
        for (slot, ds) in sets.into_iter().enumerate() {
            let dir = Path::new(&args[1]);
            let packed = fs::read(dir.join(format!("slot-{slot}.bin"))).map_err(|e| e.to_string())?;
            let model = Model::from_packed(Arch::Kep1, &packed)?;
            let hash = fs::read_to_string(dir.join(format!("slot-{slot}.hash"))).map_err(|e| e.to_string())?;
            if hash != blake3::Hash::from(model.weights_hash()).to_hex().as_str() {
                return Err("snapshot mismatch".into());
            }
            let mut s = MetalScanner::open(ds.clone())?;
            s.set_weights(&model)?;
            s.build_combos(128)?;
            let pipe = s.compile_variant(28 | 64 | 128)?;
            s.use_variant(&pipe);
            s.set_host_mask(8 | 32);
            let table = Table::build(&model, &*ds, 10, vec![])?;
            let mut prefix = vec![false; ds.n() as usize];
            let live: Vec<_> = table.live_rows().iter().enumerate().filter(|x| *x.1).map(|x| x.0).collect();
            let limit = (0..=live.len().min(1024))
                .take_while(|&v| v * (v + 1) * (v + 2) / 6 * 32 <= 128 * 1024 * 1024)
                .last()
                .unwrap();
            for &id in &live[..limit] {
                prefix[id] = true;
            }
            let mut scratch = ShareBuffers::new();
            let mut acc = Vec::new();
            for start in [0, 1403109, u32::MAX - 4095] {
                let mut header = [0x56 + slot as u8; 80];
                let mut bound = [0xff; 32];
                bound[31] = 0x3f;
                for arity in [1, 2, 3] {
                    let got = s.screen_range(&header, start, 4096, &bound, arity, true)?;
                    let mut want_hits = vec![];
                    let mut eligible = 0;
                    for (i, rec) in got.records.iter().enumerate() {
                        let nonce = start + i as u32;
                        header[76..].copy_from_slice(&nonce.to_le_bytes());
                        let key = blake3::hash(&header);
                        let idx = share_indices(key.as_bytes(), ds.n());
                        let mut n = 0;
                        let mut in_prefix = true;
                        for j in idx {
                            if table.live_rows()[j as usize] {
                                n += 1;
                                in_prefix &= prefix[j as usize];
                            }
                        }
                        let selected = n <= arity && (n < 3 || in_prefix);
                        if rec[0] != u32::from(selected) {
                            return Err(format!("selection differs at {slot}/{nonce}"));
                        }
                        if selected {
                            eligible += 1;
                            let ticket = table.eval(key.as_bytes(), &mut acc).ticket;
                            let words = blake3::platform::words_from_le_bytes_32(&ticket);
                            if rec[1..] != words
                                || (i % 31 == 0 && ticket != scratch.ticket(&model, &*ds, key.as_bytes()))
                            {
                                return Err(format!(
                                    "selected ticket mismatch at slot {slot}, arity {arity}, nonce {nonce}"
                                ));
                            }
                            if Target(bound).admits(&ticket) {
                                want_hits.push(nonce);
                            }
                        }
                    }
                    let mut fast = s.screen_range(&header, start, 4096, &bound, arity, false)?;
                    fast.hits.sort_unstable();
                    if fast.hits != want_hits || got.evaluated != eligible || fast.evaluated != eligible {
                        return Err("fast screen hit/count mismatch".into());
                    }
                }
            }
            let bound = [0; 32];
            for block in 0..=24u32 {
                let mut h = [0x84 + slot as u8; 80];
                h[40..44].copy_from_slice(&block.to_le_bytes());
                let start = SCREEN_CAP * block;
                let mut order = [0, 1, 2, 3];
                order.rotate_left((block as usize + slot) % 4);
                if block / 4 % 2 == 1 {
                    order.reverse();
                }
                for arity in order {
                    let t = Instant::now();
                    let evaluated = if arity == 0 {
                        for off in (0..SCREEN_CAP).step_by(16384) {
                            std::hint::black_box(s.tickets(&h, start + off, 16384)?);
                        }
                        SCREEN_CAP
                    } else {
                        s.screen_range(&h, start, SCREEN_CAP, &bound, arity, false)?.evaluated
                    };
                    let ms = t.elapsed().as_secs_f64() * 1000.;
                    if block > 0 {
                        println!("{block},{slot},{arity},{SCREEN_CAP},{evaluated},{ms:.5}");
                    }
                }
            }
        }
        Ok(())
    }
}
