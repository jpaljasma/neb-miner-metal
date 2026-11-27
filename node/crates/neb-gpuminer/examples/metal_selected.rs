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
        metal_scanner::{MINING_SCREEN_CAP, MetalScanner, SCREEN_CAP},
    };
    use neb_pow::shares::{Arch, Model, ShareBuffers, load_datasets, share_indices};
    use std::{fs, path::Path, time::Instant};

    fn screen_variants(
        s: &mut MetalScanner,
        model: &Model,
        ds: &dyn neb_pow::shares::Dataset,
        slot: usize,
        phase: &str,
        hold: Option<u8>,
    ) -> Result<(), String> {
        let hot = matches!(phase, "hot" | "hot-confirm");
        let (masks, blocks, seed): (Vec<u8>, u32, u8) = match (phase, hold) {
            ("hot", _) => ((0..8).map(|factors| 3 | (factors << 3)).collect(), 64, 73),
            ("hot-confirm", _) => (vec![3, 11, 19, 27], 128, 149),
            ("variants", None) => ((0..8).collect(), 32, 39),
            ("variants", Some(mask)) => {
                let mut controls = vec![0, mask & 2, mask & 3, mask & 6, mask];
                controls.sort_unstable();
                controls.dedup();
                (controls, 32, 171)
            }
            _ => return Err("unknown screen replay".into()),
        };
        let pipes: Vec<_> = masks.iter().map(|&m| s.compile_screen_variant(m)).collect::<Result<_, _>>()?;
        let header = [0xb4; 80];
        let mut easy = [0xff; 32];
        easy[31] = 63;
        let mut reference = ShareBuffers::new();
        let mut baseline = None;
        for pipe in &pipes {
            s.use_screen_variant(pipe)?;
            let mut r = s.screen_range(&header, u32::MAX - 4095, 4096, &easy, 2, false)?;
            r.hits.sort_unstable();
            for &nonce in &r.hits {
                let mut h = header;
                h[76..].copy_from_slice(&nonce.to_le_bytes());
                let ticket = reference.ticket(model, ds, blake3::hash(&h).as_bytes());
                if !Target(easy).admits(&ticket) {
                    return Err("CPU rejected screen hit".into());
                }
                // A bound equal to the oracle ticket must hit; one less must miss.
                if s.screen_range(&h, nonce, 1, &ticket, 2, false)?.hits != [nonce] {
                    return Err("exact-ticket probe missed".into());
                }
                let mut below = ticket;
                for b in &mut below {
                    let (v, borrow) = b.overflowing_sub(1);
                    *b = v;
                    if !borrow {
                        break;
                    }
                }
                if ticket != [0; 32] && !s.screen_range(&h, nonce, 1, &below, 2, false)?.hits.is_empty() {
                    return Err("below-ticket probe hit".into());
                }
            }
            let result = (r.evaluated, r.hits);
            if baseline.as_ref().is_some_and(|b| *b != result) {
                return Err("screen validation differs".into());
            }
            baseline = Some(result);
        }
        let count = if hot { 8 } else { 4 } * MINING_SCREEN_CAP;
        let mut bound = [0xff; 32];
        bound[29..].fill(0);
        for block in 0..=blocks {
            let mut h = [seed; 80];
            let prefix_id = if hot { block / 2 } else { block };
            h[..4].copy_from_slice(&prefix_id.to_le_bytes());
            h[4] = slot as u8;
            if hot {
                // Adjacent blocks can share all first 64 bytes but have a different tail.
                h[72..76].copy_from_slice(&block.rotate_left(13).to_le_bytes());
            }
            let mut order: Vec<_> = (0..masks.len()).collect();
            let n = order.len();
            order.rotate_left((block as usize + slot) % n);
            if (block as usize / n) % 2 == 1 {
                order.reverse();
            }
            let mut expected = None;
            for i in order {
                s.use_screen_variant(&pipes[i])?;
                let mut evaluated = 0;
                let mut hits = vec![];
                let mut gpu = 0.;
                let t = Instant::now();
                for off in (0..count).step_by(MINING_SCREEN_CAP as usize) {
                    let r = s.screen_range(&h, block * count + off, MINING_SCREEN_CAP, &bound, 2, false)?;
                    evaluated += r.evaluated;
                    hits.extend(r.hits);
                    gpu += r.gpu_s;
                }
                let wall = t.elapsed().as_secs_f64();
                hits.sort_unstable();
                let result = (evaluated, hits);
                if expected.as_ref().is_some_and(|prev| prev != &result) {
                    return Err("timed hit set or count differs".into());
                }
                expected = Some(result);
                if block > 0 {
                    println!("{block},{slot},{},{count},{evaluated},{:.6},{:.6}", masks[i], wall * 1000., gpu * 1000.);
                }
            }
        }
        Ok(())
    }

    fn lookup_upload(
        scanner: &MetalScanner,
        model: &Model,
        ds: &dyn neb_pow::shares::Dataset,
        slot: usize,
    ) -> Result<(), String> {
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        let table = Table::build(model, ds, threads, Vec::new())?;
        let bytes = (ds.n() as usize + 1) * size_of::<[u32; 9]>();
        for block in 0..=32 {
            let mut order = [false, true];
            if (block + slot) % 2 == 1 {
                order.reverse();
            }
            let mut previous = None;
            for mapped in order {
                let start = Instant::now();
                let buffer = scanner.benchmark_single_lookup(&table, threads, mapped)?;
                let ms = start.elapsed().as_secs_f64() * 1000.;
                if buffer.length() as usize != bytes {
                    return Err("lookup buffer length differs".into());
                }
                // SAFETY: both builders finished all CPU writes before returning this shared buffer.
                let entries = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u8>(), bytes) };
                let digest = blake3::hash(entries);
                if previous.replace(digest).is_some_and(|other| other != digest) {
                    return Err("lookup table differs".into());
                }
                if block > 0 {
                    println!("{block},{slot},{},{bytes},{ms:.6}", u8::from(mapped));
                }
            }
        }
        Ok(())
    }

    pub fn run() -> Result<(), String> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        let dispatch = args.len() == 3 && args[2] == "dispatch";
        let upload = args.len() == 3 && args[2] == "lookup-upload";
        let variants = args.get(2).is_some_and(|m| m == "variants") && (3..=4).contains(&args.len());
        let hot = args.get(2).is_some_and(|m| m == "hot" || m == "hot-confirm") && args.len() == 3;
        let hold = if variants && args.len() == 4 {
            let id = args[3].parse::<u8>().map_err(|_| "bad mask")?;
            if !(1..=7).contains(&id) {
                return Err("holdout mask must be between 1 and 7".into());
            }
            Some(id)
        } else {
            None
        };
        if args.len() != 2 && !dispatch && !variants && !upload && !hot {
            return Err(
                "DATASET SNAPSHOT_DIR [dispatch | variants [HOLDOUT_MASK] | lookup-upload | hot | hot-confirm]".into(),
            );
        }
        let pin = blake3::Hash::from_hex("790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72").unwrap();
        let sets = load_datasets(Path::new(&args[0]), Some(pin.as_bytes()))?;
        if upload {
            println!("block,slot,mapped,table_bytes,wall_ms");
        } else if variants || hot {
            println!("block,slot,mask,screened,evaluated,wall_ms,gpu_ms");
        } else if dispatch {
            println!("block,slot,batch,screened,evaluated,wall_ms,gpu_ms");
        } else {
            println!("block,slot,arity,screened,evaluated,wall_ms");
        }
        for (slot, ds) in sets.into_iter().enumerate() {
            let dir = Path::new(&args[1]);
            let packed = fs::read(dir.join(format!("slot-{slot}.bin"))).map_err(|e| e.to_string())?;
            let model = Model::from_packed(Arch::Kep1, &packed)?;
            let hash = fs::read_to_string(dir.join(format!("slot-{slot}.hash"))).map_err(|e| e.to_string())?;
            if hash != blake3::Hash::from(model.weights_hash()).to_hex().as_str() {
                return Err("snapshot mismatch".into());
            }
            let mut s = MetalScanner::open(ds.clone())?;
            if upload {
                lookup_upload(&s, &model, &*ds, slot)?;
                continue;
            }
            s.set_weights(&model)?;
            s.build_combos(if dispatch || variants || hot { 0 } else { 128 })?;
            if variants || hot {
                screen_variants(&mut s, &model, &*ds, slot, &args[2], hold)?;
                continue;
            }
            if dispatch {
                let total = 4_194_304;
                for block in 0..=32u32 {
                    let mut h = [slot as u8 + 37; 80];
                    h[0..4].copy_from_slice(&block.to_le_bytes());
                    let start = if block == 32 { u32::MAX - (total - 1) } else { block * total };
                    let mut bound = [0xff; 32];
                    bound[30..].fill(0);
                    let mut order = [SCREEN_CAP, MINING_SCREEN_CAP];
                    if (block + slot as u32).is_multiple_of(2) {
                        order.reverse();
                    }
                    let mut expected = None;
                    for batch in order {
                        let mut evaluated = 0;
                        let mut hits = vec![];
                        let mut gpu_s = 0.;
                        let t = Instant::now();
                        for offset in (0..total).step_by(batch as usize) {
                            let r = s.screen_range(&h, start + offset, batch, &bound, 2, false)?;
                            evaluated += r.evaluated;
                            gpu_s += r.gpu_s;
                            hits.extend(r.hits);
                        }
                        let wall = t.elapsed().as_secs_f64();
                        hits.sort_unstable();
                        if let Some((count, want)) = &expected {
                            if *count != evaluated || *want != hits {
                                return Err("dispatch changed evaluated count or hit set".into());
                            }
                        } else {
                            let mut scratch = ShareBuffers::new();
                            for &nonce in &hits {
                                h[76..].copy_from_slice(&nonce.to_le_bytes());
                                if !Target(bound).admits(&scratch.ticket(&model, &*ds, blake3::hash(&h).as_bytes())) {
                                    return Err("reference rejected dispatch hit".into());
                                }
                            }
                            expected = Some((evaluated, hits));
                        }
                        if block > 0 {
                            println!(
                                "{block},{slot},{batch},{total},{evaluated},{:.6},{:.6}",
                                wall * 1000.,
                                gpu_s * 1000.
                            );
                        }
                    }
                }
                continue;
            }
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
