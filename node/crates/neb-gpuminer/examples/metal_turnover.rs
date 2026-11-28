//! Measure exact three-row coverage and model-turnover optimizations on local snapshots.
use neb_pow::shares::{Arch, Dataset, Model, load_datasets, sample_grad};
use std::{fs, path::Path};

fn model(dir: &Path, slot: usize) -> Result<Model, String> {
    let bytes = fs::read(dir.join(format!("slot-{slot}.bin"))).map_err(|e| e.to_string())?;
    let m = Model::from_packed(Arch::Kep1, &bytes)?;
    let hash = fs::read_to_string(dir.join(format!("slot-{slot}.hash"))).map_err(|e| e.to_string())?;
    if blake3::Hash::from(m.weights_hash()).to_hex().as_str() != hash.trim() {
        return Err("snapshot hash mismatch".into());
    }
    Ok(m)
}

fn supports(ds: &dyn Dataset, model: &Model) -> (Vec<Vec<u64>>, Vec<usize>, u32) {
    let mut ranks = vec![usize::MAX; ds.n() as usize];
    let mut rows = Vec::new();
    let mut gradient = vec![0; model.arch().weights()];
    let mut pixels = vec![0; model.arch().nin()];
    let mut common = vec![u64::MAX; gradient.len().div_ceil(64)];
    for i in 0..ds.n() {
        let label = ds.sample(i, &mut pixels);
        sample_grad(model, &pixels, label, &mut gradient, None);
        if gradient.iter().all(|&g| g == 0) {
            continue;
        }
        let bits: Vec<u64> = gradient
            .chunks(64)
            .map(|chunk| chunk.iter().enumerate().fold(0, |word, (j, &g)| word | (u64::from(g != 0) << j)))
            .collect();
        for (all, &this) in common.iter_mut().zip(&bits) {
            *all &= this;
        }
        ranks[i as usize] = rows.len();
        rows.push(bits);
    }
    let common_bits = if rows.is_empty() { 0 } else { common.iter().map(|v| v.count_ones()).sum() };
    (rows, ranks, common_bits)
}

fn support(ds: &dyn Dataset, model: &Model, slot: usize) {
    let (rows, _, common_bits) = supports(ds, model);
    let trials = if rows.is_empty() { 0 } else { 1_000_000 };
    let mut recoverable = 0u32;
    let mut random = blake3::Hasher::new();
    random.update(b"turnover triple support v1").update(&model.weights_hash());
    let mut random = random.finalize_xof();
    if common_bits == 0 {
        for _ in 0..trials {
            let mut words = [0; 12];
            random.fill(&mut words);
            let mut indices = words
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| ((u64::from(u32::from_le_bytes(*x)) * rows.len() as u64) >> 32) as usize);
            let a = &rows[indices.next().unwrap()];
            let b = &rows[indices.next().unwrap()];
            let c = &rows[indices.next().unwrap()];
            recoverable += u32::from(a.iter().zip(b).zip(c).all(|((&x, &y), &z)| x & y & z == 0));
        }
    }
    println!(
        "slot={slot} live={} common_nonzero_weights={common_bits} sampled={} recoverable={recoverable}",
        rows.len(),
        if common_bits > 0 { 0 } else { trials }
    );
}

const PROBES: u32 = 131072;
const PROBE_START: u32 = u32::MAX - PROBES + 1;
fn oracle(ds: &dyn Dataset, m: &Model, path: &Path) -> Result<(), String> {
    use neb_engines::table::Table;
    use neb_pow::shares::{ShareBuffers, share_indices};
    let (bits, ranks, _) = supports(ds, m);
    let t = Table::build(m, ds, 8, Vec::new())?;
    let mut h = [41; 80];
    let mut out = Vec::new();
    let mut acc = Vec::new();
    let mut cpu = ShareBuffers::new();
    let mut recovered = 0;
    for off in 0..PROBES {
        let nonce = PROBE_START + off;
        h[76..].copy_from_slice(&nonce.to_le_bytes());
        let key = blake3::hash(&h);
        let mut live = Vec::with_capacity(3);
        for row in share_indices(key.as_bytes(), ds.n()) {
            let rank = ranks[row as usize];
            if rank != usize::MAX {
                live.push(rank);
            }
            if live.len() > 3 {
                break;
            }
        }
        if live.len() > 3 {
            continue;
        }
        if let [a, b, c] = live[..] {
            if bits[a].iter().zip(&bits[b]).zip(&bits[c]).any(|((&x, &y), &z)| x & y & z != 0) {
                continue;
            }
            recovered += 1;
        }
        live.sort_unstable();
        let required = if live.len() < 2 {
            0
        } else {
            let b = live[live.len() - 1] as u32;
            b * (b + 1) / 2 + live[live.len() - 2] as u32 + 1
        };
        let ticket = t.eval(key.as_bytes(), &mut acc).ticket;
        if (live.len() == 3 || off % 997 == 0) && ticket != cpu.ticket(m, ds, key.as_bytes()) {
            return Err("CPU table differs from reference".into());
        }
        out.extend_from_slice(&nonce.to_le_bytes());
        out.push(live.len() as u8);
        out.extend_from_slice(&required.to_le_bytes());
        out.extend_from_slice(&ticket);
    }
    fs::write(path, &out).map_err(|e| e.to_string())?;
    println!("oracle_rows={} recoverable_triplets={recovered}", out.len() / 41);
    Ok(())
}

#[cfg(target_os = "macos")]
mod apple {
    use super::*;
    use neb_block::Target;
    use neb_gpuminer::{
        Scanner,
        metal_scanner::{MINING_SCREEN_CAP, MetalScanner},
    };
    use neb_pow::shares::{ShareBuffers, SharedDataset};
    use std::time::Instant;
    fn verify(s: &mut MetalScanner, path: &Path, recover: bool) -> Result<(), String> {
        let data = fs::read(path).map_err(|e| e.to_string())?;
        if !data.len().is_multiple_of(41) {
            return Err("incomplete oracle".into());
        }
        let ready = s.turnover_progress().0;
        let h = [41; 80];
        for bound in [[0xff; 32], [0x28; 32], [0; 32]] {
            let mut want = vec![];
            let mut count = 0;
            for rec in data.as_chunks::<41>().0 {
                let arity = rec[4];
                if arity == 3 && !recover || u32::from_le_bytes(rec[5..9].try_into().unwrap()) > ready {
                    continue;
                }
                let nonce = u32::from_le_bytes(rec[..4].try_into().unwrap());
                let ticket: [u8; 32] = rec[9..].try_into().unwrap();
                count += 1;
                if Target(bound).admits(&ticket) {
                    want.push(nonce);
                }
                if arity == 3 && bound == [0xff; 32] {
                    if s.turnover_evaluate(&h, nonce, 1, &ticket)?.hits != [nonce] {
                        return Err("exact triple ticket missed".into());
                    }
                    let mut below = ticket;
                    for v in &mut below {
                        let (n, borrow) = v.overflowing_sub(1);
                        *v = n;
                        if !borrow {
                            break;
                        }
                    }
                    if ticket != [0; 32] && !s.turnover_evaluate(&h, nonce, 1, &below)?.hits.is_empty() {
                        return Err("below triple ticket hit".into());
                    }
                }
            }
            let mut got = s.turnover_evaluate(&h, PROBE_START, PROBES, &bound)?;
            got.hits.sort_unstable();
            if got.evaluated != count || got.hits != want {
                return Err(format!("oracle differs at readiness {ready}, GPU {} CPU {count}", got.evaluated));
            }
        }
        Ok(())
    }
    /// Return the completion timestamp before the host verifies the GPU's candidates.
    fn window_batch(
        scanner: &mut MetalScanner,
        model: &Model,
        dataset: &dyn Dataset,
        header: &mut [u8; 80],
        raw: &mut u64,
        bound: &[u8; 32],
        cpu: &mut ShareBuffers,
    ) -> Result<(u32, Instant), String> {
        let start = *raw as u32;
        header[40..44].copy_from_slice(&((*raw >> 32) as u32).to_le_bytes());
        let count = u64::from(scanner.mining_capacity()).min((1u64 << 32) - u64::from(start)) as u32;
        let (hits, evaluated) = scanner.mine_range(model, header, start, count, bound)?;
        let finished = Instant::now();
        for nonce in hits {
            header[76..].copy_from_slice(&nonce.to_le_bytes());
            if !Target(*bound).admits(&cpu.ticket(model, dataset, blake3::hash(header).as_bytes())) {
                return Err("CPU rejected candidate".into());
            }
        }
        *raw += u64::from(count);
        Ok((evaluated, finished))
    }

    pub fn cpu_feed(args: &[String], sets: &[SharedDataset]) -> Result<(), String> {
        let [_, _, old, new, slot, threads, seed, oracle] = args else {
            return Err("cpu-feed DATASET OLD NEW SLOT THREADS SEED ORACLE".into());
        };
        let slot: usize = slot.parse().map_err(|_| "invalid slot")?;
        let ds = sets.get(slot).ok_or("invalid slot")?;
        let threads: std::num::NonZeroUsize = threads.parse().map_err(|_| "expected a positive CPU thread count")?;
        let available = std::thread::available_parallelism().map_err(|e| e.to_string())?;
        if threads > available {
            return Err(format!("requested {threads} CPU threads, only {available} available"));
        }
        let seed: u8 = seed.parse().map_err(|_| "invalid header seed")?;
        let old = model(Path::new(old), slot)?;
        let new = model(Path::new(new), slot)?;
        if old.weights_hash() == new.weights_hash() {
            return Err("expected model turnover".into());
        }
        let mut s = MetalScanner::for_slot(ds.clone(), slot)?;
        s.set_setup_threads(threads);
        s.set_weights(&old)?;
        s.build_combos(0)?;
        s.turnover_evaluate(&[19; 80], 0, MINING_SCREEN_CAP, &[0; 32])?;
        s.enable_screen();

        let clock = Instant::now();
        let deadline = clock + std::time::Duration::from_secs(5);
        s.set_weights(&new)?;
        let setup = clock.elapsed().as_secs_f64();
        if s.mining_mode() != Some("selective") {
            return Err("cpu-feed requires a destination model that selects the pair scanner".into());
        }
        let mut cpu = ShareBuffers::new();
        let mut header = [seed; 80];
        let mut raw = 0;
        let mut tickets = 0u64;
        let mut bound = [0xff; 32];
        bound[29..].fill(0);
        while Instant::now() < deadline {
            let (evaluated, finished) = window_batch(&mut s, &new, &**ds, &mut header, &mut raw, &bound, &mut cpu)?;
            if finished <= deadline {
                tickets += u64::from(evaluated);
            }
        }
        verify(&mut s, Path::new(oracle), false)?;
        println!("cpu-feed,{slot},{threads},{seed},{setup:.6},{tickets}");
        Ok(())
    }

    pub fn run(args: &[String], sets: &[SharedDataset]) -> Result<(), String> {
        // MODE DATASET OLD NEW SLOT MASK STYLE SEED ORACLE
        if args.len() != 9 {
            return Err("fixed|window DATASET OLD NEW SLOT MASK MASKED_BUILDER SEED ORACLE".into());
        }
        let parse = |i: usize| args[i].parse::<u8>().map_err(|e| e.to_string());
        let slot = parse(4)? as usize;
        if slot >= sets.len() {
            return Err("invalid slot".into());
        }
        let mask = parse(5)?;
        if mask > 7 {
            return Err("invalid mask".into());
        }
        let seed = parse(7)?;
        if args[0] == "paired-direct" {
            let ordered = if seed.is_multiple_of(2) { ["short", "direct-short"] } else { ["direct-short", "short"] };
            let mut job = args.to_vec();
            for name in ordered {
                job[0] = name.into();
                run(&job, sets)?;
            }
            return Ok(());
        }
        if args[0] == "paired-short" {
            let mut order = [0, 2, 6];
            order.rotate_left((seed as usize + slot) % 3);
            if seed / 3 % 2 == 1 {
                order.reverse();
            }
            let mut job = args.to_vec();
            job[0] = "short".into();
            for mask in order {
                job[5] = mask.to_string();
                run(&job, sets)?;
            }
            return Ok(());
        }
        let ds = &sets[slot];
        let old = model(Path::new(&args[2]), slot)?;
        let new = model(Path::new(&args[3]), slot)?;
        if old.weights_hash() == new.weights_hash() {
            return Err("expected model turnover".into());
        }
        if args[0] == "copy" {
            let mut lanes = [MetalScanner::for_slot(ds.clone(), slot)?, MetalScanner::for_slot(ds.clone(), slot)?];
            for (i, lane) in lanes.iter_mut().enumerate() {
                lane.turnover_experiment(2 | ((i as u8) << 2), true);
                lane.set_weights(&old)?;
                lane.turnover_prepare_prefix()?;
            }
            for block in 0..=20 {
                let next = if block % 2 == 0 { &new } else { &old };
                let order = if (block / 2 + slot).is_multiple_of(2) { [0, 1] } else { [1, 0] };
                let mut expected = None;
                for i in order {
                    let lane = &mut lanes[i];
                    let begin = Instant::now();
                    lane.set_weights(next)?;
                    let setup = begin.elapsed().as_secs_f64();
                    let (upload, skipped, hash) = lane.turnover_upload_check();
                    if expected.replace(hash).is_some_and(|other| other != hash) {
                        return Err("mapped row bytes differ".into());
                    }
                    if block > 0 {
                        println!("copy,{slot},{block},{i},{setup:.8},{upload:.8},{skipped}");
                    }
                    lane.turnover_prepare_prefix()?;
                }
            }
            return Ok(());
        }
        let mut s = MetalScanner::for_slot(ds.clone(), slot)?;
        s.turnover_experiment(mask, parse(6)? != 0);
        let direct = args[0].starts_with("direct-");
        if direct {
            s.use_direct_pair_builder();
        }
        s.set_weights(&old)?;
        s.build_combos(0)?;
        s.turnover_evaluate(&[19; 80], 0, MINING_SCREEN_CAP, &[0; 32])?;
        if args[0] == "builders" || args[0] == "geometry" {
            s.set_weights(&new)?;
            s.turnover_prepare_prefix()?;
            return s.benchmark_builders(slot, args[0] == "geometry", seed);
        }
        if args[0] == "extra" {
            s.set_weights(&new)?;
            s.build_combos(0)?;
            let mut confirmed = 0;
            let mut cpu = ShareBuffers::new();
            for b in 1..=64u32 {
                let mut h = [seed; 80];
                h[..4].copy_from_slice(&b.to_le_bytes());
                for offset in [0, MINING_SCREEN_CAP] {
                    let start = b * (2 * MINING_SCREEN_CAP) + offset;
                    let n = MINING_SCREEN_CAP;
                    let normal = s.screen_range(&h, start, n, &[0; 32], 2, false)?.evaluated;
                    let added = s.turnover_evaluate(&h, start, n, &[0; 32])?.evaluated - normal;
                    if added == 0 {
                        continue;
                    }
                    let base = s.screen_range(&h, start, n, &[0xff; 32], 2, false)?;
                    let all = s.turnover_evaluate(&h, start, n, &[0xff; 32])?;
                    let mut existing = vec![false; n as usize];
                    for nonce in base.hits {
                        existing[(nonce - start) as usize] = true;
                    }
                    let mut checked = 0;
                    for nonce in all.hits.into_iter().filter(|v| !existing[(*v - start) as usize]) {
                        h[76..].copy_from_slice(&nonce.to_le_bytes());
                        let ticket = cpu.ticket(&new, &**ds, blake3::hash(&h).as_bytes());
                        if s.turnover_evaluate(&h, nonce, 1, &ticket)?.hits != [nonce] {
                            return Err("reference ticket missed".into());
                        }
                        let mut lower = ticket;
                        for x in &mut lower {
                            let (v, borrow) = x.overflowing_sub(1);
                            *x = v;
                            if !borrow {
                                break;
                            }
                        }
                        if ticket != [0; 32] && !s.turnover_evaluate(&h, nonce, 1, &lower)?.hits.is_empty() {
                            return Err("invalid recovered ticket".into());
                        }
                        checked += 1;
                    }
                    if checked != added {
                        return Err("extra ticket count differs".into());
                    }
                    confirmed += checked;
                }
                if confirmed >= 3 {
                    break;
                }
            }
            println!("extra,{slot},{confirmed}");
            return Ok(());
        }
        let window = matches!(args[0].as_str(), "window" | "short" | "direct-short");
        if window {
            s.enable_screen();
        }
        let clock = Instant::now();
        s.set_weights(&new)?;
        let setup = clock.elapsed().as_secs_f64();
        let mut cpu = ShareBuffers::new();
        let mut h = [seed; 80];
        let mut raw = 0u64;
        let mut bound = [0xff; 32];
        bound[29..].fill(0);
        if window {
            let mut total = 0u64;
            for mark in [5., 15., 30.] {
                if args[0].ends_with("short") && mark > 5. {
                    break;
                }
                while clock.elapsed().as_secs_f64() < mark {
                    let (evaluated, _) = window_batch(&mut s, &new, &**ds, &mut h, &mut raw, &bound, &mut cpu)?;
                    total += u64::from(evaluated);
                }
                println!(
                    "window,{slot},{mask},{seed},{mark},{:.6},{total},{setup:.6},{:?},{direct}",
                    clock.elapsed().as_secs_f64(),
                    s.mining_mode()
                );
            }
            verify(&mut s, Path::new(&args[8]), mask & 1 != 0)?;
        } else {
            let build = Instant::now();
            s.build_combos(0)?;
            let build_s = build.elapsed().as_secs_f64();
            println!(
                "setup,{slot},{mask},{seed},{setup:.6},{build_s:.6},{},{}",
                s.pair_prefix_hash(),
                s.turnover_progress().1
            );
            for block in 0..=32u32 {
                h[..4].copy_from_slice(&block.to_le_bytes());
                let start = block * 2 * MINING_SCREEN_CAP;
                let mut total = 0;
                let mut hits = Vec::new();
                let begin = Instant::now();
                for offset in [0, MINING_SCREEN_CAP] {
                    let r = s.turnover_evaluate(&h, start + offset, MINING_SCREEN_CAP, &bound)?;
                    total += r.evaluated;
                    hits.extend(r.hits);
                }
                let ms = 1000. * begin.elapsed().as_secs_f64();
                for nonce in hits {
                    h[76..].copy_from_slice(&nonce.to_le_bytes());
                    if !Target(bound).admits(&cpu.ticket(&new, &**ds, blake3::hash(&h).as_bytes())) {
                        return Err("CPU rejected candidate".into());
                    }
                }
                if block > 0 {
                    println!("scan,{slot},{mask},{seed},{block},{total},{ms:.6}");
                }
            }
            verify(&mut s, Path::new(&args[8]), mask & 1 != 0)?;
            s.turnover_limit_pairs(65536);
            verify(&mut s, Path::new(&args[8]), mask & 1 != 0)?;
        }
        Ok(())
    }
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("expected mode, dataset and snapshots".into());
    }
    let pin = blake3::Hash::from_hex("790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72").unwrap();
    let sets = load_datasets(Path::new(&args[1]), Some(pin.as_bytes()))?;
    match args[0].as_str() {
        #[cfg(target_os = "macos")]
        "cpu-feed" => apple::cpu_feed(&args, &sets),
        "support" => {
            for dir in &args[2..] {
                for (slot, ds) in sets.iter().enumerate() {
                    support(ds.as_ref(), &model(Path::new(dir), slot)?, slot);
                }
            }
            Ok(())
        }
        "oracle" if args.len() == 5 => {
            let slot: usize = args[3].parse::<usize>().map_err(|e| e.to_string())?;
            let ds = sets.get(slot).ok_or("invalid slot")?;
            oracle(ds.as_ref(), &model(Path::new(&args[2]), slot)?, Path::new(&args[4]))
        }
        #[cfg(target_os = "macos")]
        "fixed" | "direct-fixed" | "window" | "short" | "direct-short" | "paired-direct" | "paired-short" | "copy"
        | "builders" | "geometry" | "extra" => apple::run(&args, &sets),
        _ => Err("unknown experiment".into()),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn inclusion_exclusion_handles_repeats_clipping_and_wrapping_salts() {
        let values = [-513, -257, -256, -255, -1, 0, 1, 255, 256, 257, 513];
        for salt in [7919u32, 6 * 7919 + 63, 0xffff_ffff] {
            let fold = |x: i32| (x.clamp(-256, 256) as u32) ^ salt;
            for &a in &values {
                for &b in &values {
                    for &c in &values {
                        if a != 0 && b != 0 && c != 0 {
                            continue;
                        }
                        let reconstructed = fold(a + b)
                            .wrapping_add(fold(a + c))
                            .wrapping_add(fold(b + c))
                            .wrapping_sub(fold(a))
                            .wrapping_sub(fold(b))
                            .wrapping_sub(fold(c))
                            .wrapping_add(fold(0));
                        assert_eq!(reconstructed, fold(a + b + c));
                    }
                }
            }
        }
    }
}
