//! Full-factorial screen and independent paired confirmation on two captured KEP1 models.
#[cfg(not(target_os = "macos"))]
fn main() {
    panic!("Metal benchmarks require macOS");
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    bench::run()
}

#[cfg(target_os = "macos")]
mod bench {
    use neb_engines::table::Table;
    use neb_gpuminer::{Scanner, metal_scanner::MetalScanner};
    use neb_pow::shares::{Arch, Model, ShareBuffers, SharedDataset, load_datasets};
    use std::{fs, path::Path, time::Instant};

    const COUNT: usize = 32768;
    const BATCH: usize = 4096;
    // (nonces per worker batch, nonces per command buffer). ID 0 is the original schedule.
    const SCHEDULES: [(usize, usize); 9] = [
        (4096, 4096),
        (2048, 2048),
        (8192, 8192),
        (16384, 16384),
        (4096, 1024),
        (8192, 4096),
        (16384, 4096),
        (16384, 2048),
        (8192, 2048),
    ];
    struct Job {
        scanner: MetalScanner,
        table: Table,
        model: Model,
        dataset: SharedDataset,
    }

    fn shuffled(values: &mut [u8], mut state: u64) {
        for i in (1..values.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values.swap(i, (state % (i as u64 + 1)) as usize);
        }
    }

    pub fn run() -> Result<(), String> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        if args.len() == 3 && args[2] == "row-delta" {
            return row_delta(&args[0], Path::new(&args[1]));
        }
        if args.len() == 3 && args[2] == "reuse" {
            return pair_reuse(&args[0], Path::new(&args[1]));
        }
        let usage = "DATASET SNAPSHOT_DIR (screen|queue|host|compact|exhaustive) or (confirm|queue-confirm|host-confirm|compact-confirm|exhaustive-confirm) ID";
        if args.len() < 3 {
            return Err(usage.into());
        }
        let phase = args[2].as_str();
        let confirm = phase.ends_with("confirm");
        let family = if phase == "confirm" { "screen" } else { phase.trim_end_matches("-confirm") };
        let (levels, screen_seed, holdout_seed): (u8, u64, u64) = match family {
            "screen" => (32, 0x450912, 0x76443a),
            "queue" => (SCHEDULES.len() as u8, 0xe5f103, 0xb18d21),
            "host" => (8, 0x3175b1, 0x964e96),
            "exhaustive" => (2, 0x491012, 0x154023),
            "compact" => (6, 0xa18a44, 0xa9c045),
            "tables" => (4, 0x658de3, 0x1382ab),
            _ => return Err(usage.into()),
        };
        let (choices, blocks, seed) = match (confirm, args.len()) {
            (false, 3) => ((0..levels).collect::<Vec<_>>(), 12, screen_seed),
            (true, 4) => {
                let id = args[3].parse::<u8>().map_err(|_| usage)?;
                if !(1..levels).contains(&id) {
                    return Err(usage.into());
                }
                (vec![0, id], 24, holdout_seed)
            }
            _ => return Err(usage.into()),
        };
        let pin = blake3::Hash::from_hex("790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72").unwrap();
        let datasets = load_datasets(Path::new(&args[0]), Some(pin.as_bytes()))?;
        let dir = Path::new(&args[1]);
        let mut jobs = vec![];
        for (slot, dataset) in datasets.into_iter().enumerate() {
            let read = |suffix: &str| fs::read(dir.join(format!("slot-{slot}.{suffix}"))).map_err(|e| e.to_string());
            let model = Model::from_packed(Arch::Kep1, &read("bin")?)?;
            let digest = blake3::Hash::from(model.weights_hash()).to_hex();
            if digest.as_bytes() != read("hash")? {
                return Err("model hash mismatch".into());
            }
            let mut scanner = MetalScanner::open(dataset.clone())?;
            scanner.set_weights(&model)?;
            if family == "tables" {
                scanner.build_combos(128)?;
            }
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            let table = Table::build(&model, &*dataset, threads, vec![])?;
            eprintln!("slot={slot} rows={} idle_rows={}", dataset.n(), table.idle_rows());
            jobs.push(Job { scanner, table, model, dataset });
        }
        let mut pipes = std::collections::BTreeMap::new();
        for &mask in &choices {
            let kernel = match family {
                "host" => 28 | ((mask & 4) << 3),
                "compact" => {
                    28 | match mask % 3 {
                        0 => 0,
                        1 => 64,
                        _ => 64 | 128,
                    }
                }
                "queue" => 28,
                _ => mask,
            };
            let pipe = if family == "exhaustive" {
                if mask == 0 {
                    jobs[0].scanner.compile_variant(28)?
                } else {
                    jobs[0].scanner.compile_compact_exhaustive()?
                }
            } else if family == "tables" && mask >= 2 {
                jobs[0].scanner.combo_pipeline(mask)?
            } else if family == "tables" {
                jobs[0].scanner.compile_variant(28 | 64 | 128 | if mask == 1 { 32 } else { 0 })?
            } else {
                jobs[0].scanner.compile_variant(kernel)?
            };
            pipes.insert(mask, pipe);
        }
        println!("phase,block,slot,mask,count,wall_ms,gpu_ms,threadgroup_bytes,metal_allocated_bytes");
        for block in 0..=blocks {
            for pos in 0..2 {
                let slot = (block + pos) % 2;
                let job = &mut jobs[slot];
                let mut header: [u8; 80] = std::array::from_fn(|i| (37 * i + slot) as u8);
                header[40..48].copy_from_slice(&(seed + block as u64).to_le_bytes());
                let start = block as u32 * COUNT as u32;
                // Fresh work per block; all 32 configurations receive exactly the same keys.
                let mut acc = Vec::new();
                let mut scratch = ShareBuffers::new();
                let mut expected = Vec::with_capacity(COUNT);
                for i in 0..COUNT {
                    header[76..].copy_from_slice(&(start + i as u32).to_le_bytes());
                    let key = blake3::hash(&header);
                    let t = job.table.eval(key.as_bytes(), &mut acc).ticket;
                    if i % 1021 == 0 && t != scratch.ticket(&job.model, &*job.dataset, key.as_bytes()) {
                        return Err("CPU table differs from C reference".into());
                    }
                    expected.push(t);
                }
                let mut order = choices.clone();
                if args[2].ends_with("confirm") {
                    if (block + slot) % 2 == 0 {
                        order.reverse();
                    }
                } else {
                    shuffled(&mut order, seed ^ ((1 + block * 2 + slot) as u64 * 0x9e37_79b9));
                }
                for mask in order {
                    let pipe = &pipes[&mask];
                    job.scanner.use_variant(pipe);
                    job.scanner.set_host_mask(match family {
                        "host" => mask,
                        "exhaustive" => {
                            if mask == 0 {
                                0
                            } else {
                                8 | 32
                            }
                        }
                        "compact" => 8 * u8::from(mask % 3 != 0) + 32 * u8::from(mask >= 3),
                        "tables" => 8 | 32 | if mask > 0 { 4 } else { 0 } | if mask >= 2 { 64 } else { 0 },
                        _ => 0,
                    });
                    let (batch, dispatch) = match family {
                        "queue" => SCHEDULES[mask as usize],
                        "host" | "compact" | "tables" | "exhaustive" => SCHEDULES[7],
                        _ => (BATCH, BATCH),
                    };
                    job.scanner.schedule(batch, dispatch)?;
                    let gpu_start = job.scanner.gpu_seconds();
                    let mut seconds = 0.0;
                    for offset in (0..COUNT).step_by(batch) {
                        let t = Instant::now();
                        let tickets = job.scanner.tickets(&header, start + offset as u32, batch as u32)?;
                        seconds += t.elapsed().as_secs_f64();
                        if tickets != &expected[offset..offset + batch] {
                            return Err(format!(
                                "wrong ticket: phase {}, block {block}, slot {slot}, mask {mask}",
                                args[2]
                            ));
                        }
                    }
                    if block > 0 {
                        println!(
                            "{},{block},{slot},{mask},{COUNT},{:.5},{:.5},{},{}",
                            args[2],
                            1000.0 * seconds,
                            1000.0 * (job.scanner.gpu_seconds() - gpu_start),
                            pipe.0.static_threadgroup_memory_length(),
                            job.scanner.allocated_bytes()
                        );
                    }
                }
            }
        }
        Ok(())
    }

    // Estimate exactly which existing pair entries could survive a change of
    // weights, before investing in a rank remapping or incremental update.
    fn row_delta(path: &str, dir: &Path) -> Result<(), String> {
        let pin = blake3::Hash::from_hex("790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72").unwrap();
        for (slot, ds) in load_datasets(Path::new(path), Some(pin.as_bytes()))?.into_iter().enumerate() {
            let table = |i| -> Result<Table, String> {
                let p = dir.join(format!("{i}/slot-{slot}.bin"));
                let bytes = fs::read(p).map_err(|e| e.to_string())?;
                let m = Model::from_packed(Arch::Kep1, &bytes)?;
                if blake3::Hash::from(m.weights_hash()).to_hex().as_str()
                    != fs::read_to_string(dir.join(format!("{i}/slot-{slot}.hash"))).map_err(|e| e.to_string())?
                {
                    return Err("adjacent snapshot hash differs".into());
                }
                Table::build(&m, &*ds, 2, vec![])
            };
            let a = table(0)?;
            let b = table(1)?;
            let (al, bl) = (a.live_rows().to_vec(), b.live_rows().to_vec());
            let (old, new) = (a.into_spare(), b.into_spare());
            let row = old.len() / al.len();
            let live = bl.iter().filter(|v| **v).count();
            let equal = al
                .iter()
                .zip(&bl)
                .zip(old.chunks_exact(row).zip(new.chunks_exact(row)))
                .filter(|((x, y), (a, b))| **x && **y && a == b)
                .count();
            let membership = al.iter().zip(bl).filter(|(x, y)| **x != *y).count();
            println!(
                "slot={slot} new_live={live} changed_membership={membership} identical_live_rows={equal} reusable_pair_pct={:.3}",
                100. * (equal * (equal + 1)) as f64 / (live * (live + 1)).max(1) as f64
            );
        }
        Ok(())
    }

    fn pair_reuse(path: &str, dir: &Path) -> Result<(), String> {
        let pin = blake3::Hash::from_hex("790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72").unwrap();
        println!("slot,block,reuse,setup_s,stream_hash");
        for (slot, ds) in load_datasets(Path::new(path), Some(pin.as_bytes()))?.into_iter().enumerate() {
            let mut models = Vec::new();
            let mut oracles = Vec::new();
            for i in 0..2 {
                let dir = dir.join(i.to_string());
                let bytes = fs::read(dir.join(format!("slot-{slot}.bin"))).map_err(|e| e.to_string())?;
                let model = Model::from_packed(Arch::Kep1, &bytes)?;
                if fs::read_to_string(dir.join(format!("slot-{slot}.hash"))).map_err(|e| e.to_string())?
                    != blake3::Hash::from(model.weights_hash()).to_hex().as_str()
                {
                    return Err("weight hash mismatch".into());
                }
                oracles.push(Table::build(&model, &*ds, 10, vec![])?);
                models.push(model);
            }
            let mut lanes = [MetalScanner::open(ds.clone())?, MetalScanner::open(ds.clone())?];
            for (id, lane) in lanes.iter_mut().enumerate() {
                lane.set_pair_reuse(id == 1);
                lane.set_weights(&models[0])?;
                lane.build_combos(0)?;
            }
            for block in 1..=12 {
                let state = block % 2;
                let header = [block as u8; 80];
                let mut want = None;
                // Crossover order balanced independently of the direction of the model change.
                let order = if (block / 2 + slot) % 2 == 0 { [0, 1] } else { [1, 0] };
                for id in order {
                    let lane = &mut lanes[id];
                    let t = Instant::now();
                    lane.set_weights(&models[state])?;
                    lane.build_combos(0)?;
                    let setup = t.elapsed().as_secs_f64();
                    let records = lane.screen_range(&header, 0, 4096, &[0xff; 32], 2, true)?.records;
                    let mut sum = blake3::Hasher::new();
                    let mut acc = Vec::new();
                    for (i, r) in records.iter().enumerate() {
                        let mut h = header;
                        h[76..].copy_from_slice(&(i as u32).to_le_bytes());
                        let key = blake3::hash(&h);
                        let indices = neb_pow::shares::share_indices(key.as_bytes(), ds.n());
                        let live = indices.iter().filter(|&&j| oracles[state].live_rows()[j as usize]).count();
                        if r[0] != u32::from(live <= 2) {
                            return Err("incorrect remapped screen".into());
                        }
                        if live <= 2 {
                            let ticket = oracles[state].eval(key.as_bytes(), &mut acc).ticket;
                            if r[1..] != blake3::platform::words_from_le_bytes_32(&ticket) {
                                return Err("incorrect remapped pair digest".into());
                            }
                            sum.update(&ticket);
                        }
                    }
                    let hash = sum.finalize();
                    if *want.get_or_insert(hash) != hash {
                        return Err("pair reuse changed stream".into());
                    }
                    println!("{slot},{block},{id},{setup:.6},{hash}");
                }
            }
        }
        Ok(())
    }
}
