#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("neb-miner-metal requires macOS");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
mod apple {
    use neb_block::Target;
    use neb_engines::{Engine, EngineKind, Prepared};
    use neb_gpuminer::{Lane, Scanner, metal_scanner::MetalScanner, spawn_worker_with_fallback};
    use neb_pow::shares::{Arch, Model, ShareBuffers, SharedDataset, load_datasets};
    use std::{
        path::Path,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
        thread,
        time::{Duration, Instant},
    };

    const DATASET_HASH: &str = "790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72";
    const USAGE: &str =
        "neb-miner-metal --dataset FILE (--node HOST:PORT [--stratum-adapter [--tess-ahead]] | --benchmark)";

    // A failing model probe or audit makes the existing GPU worker rescan the
    // range with this scanner. A small capacity keeps subsequent CPU batches short.
    struct ReferenceScanner {
        dataset: SharedDataset,
        model: Option<Model>,
        scratch: ShareBuffers,
    }
    impl Scanner for ReferenceScanner {
        fn capacity(&self) -> u32 {
            16
        }
        fn set_model(&mut self, _: &[i8], _: &[i8]) -> Result<(), String> {
            Err("expected KEP1 weights".into())
        }
        fn set_weights(&mut self, model: &Model) -> Result<(), String> {
            self.model = Some(model.clone());
            Ok(())
        }
        fn scan(&mut self, header: &[u8; 80], start: u32, count: u32, bound: &[u8; 32]) -> Result<Vec<u32>, String> {
            let mut h = *header;
            let model = self.model.as_ref().ok_or("missing reference model")?;
            Ok((0..count)
                .map(|i| start + i)
                .filter(|&nonce| {
                    h[76..].copy_from_slice(&nonce.to_le_bytes());
                    Target(*bound).admits(&self.scratch.ticket(model, &*self.dataset, blake3::hash(&h).as_bytes()))
                })
                .collect())
        }
    }

    fn log_slots(stats: &neb_cpuminer::Stats, kind: &str) {
        for (slot, s) in stats.gpu_slots.iter().enumerate() {
            eprintln!(
                "slot {kind} {slot} evaluated {} screened {} work_ns {}",
                s.evaluated.load(Ordering::Relaxed),
                s.screened.load(Ordering::Relaxed),
                s.work_ns.load(Ordering::Relaxed)
            );
        }
    }

    fn mine(addr: &str, datasets: Vec<SharedDataset>, selective: bool, tess_ahead: bool) -> Result<(), String> {
        let mut lanes = Vec::new();
        for (slot, dataset) in datasets.into_iter().enumerate() {
            let mut scanner = MetalScanner::for_slot(dataset.clone(), slot)?;
            if slot == 1 && tess_ahead {
                scanner.enable_tess_ahead().listen();
            }
            if selective {
                scanner.enable_screen();
            }
            let cpu_data = dataset.clone();
            let fallback = Box::new(move || {
                Ok(Box::new(ReferenceScanner { dataset: cpu_data, model: None, scratch: ShareBuffers::new() })
                    as Box<dyn Scanner>)
            });
            lanes.push(Lane { dataset, scanner: Box::new(scanner), fallback: Some(fallback) });
        }
        let handle = neb_cpuminer::start_with(addr, "neb-miner-metal", |_| vec![]).map_err(|e| e.to_string())?;
        let mismatches = Arc::new(AtomicU64::default());
        let fallbacks = Arc::new(AtomicU64::default());
        let worker =
            spawn_worker_with_fallback(handle.session(), lanes, Arc::default(), mismatches.clone(), fallbacks.clone());
        let mut last = (Instant::now(), 0u64);
        while !handle.stopped() && !worker.is_finished() {
            thread::sleep(Duration::from_millis(100));
            if last.0.elapsed() >= Duration::from_secs(5) {
                let s = handle.stats();
                let n = s.attempts.load(Ordering::Relaxed);
                eprintln!(
                    "{:.0} attempts/s, shares {}, accepted blocks {}, mismatches {}, fallbacks {}, attempts {}",
                    (n - last.1) as f64 / last.0.elapsed().as_secs_f64(),
                    s.shares.load(Ordering::Relaxed),
                    s.accepted.load(Ordering::Relaxed),
                    mismatches.load(Ordering::Relaxed),
                    fallbacks.load(Ordering::Relaxed),
                    n
                );
                last = (Instant::now(), n);
                log_slots(s, "sample");
            }
        }
        let reason = if handle.stopped() { handle.stats().disconnect_message() } else { "scanner stopped" };
        let stats = handle.stats().clone();
        handle.stop();
        worker.join().map_err(|_| "scanner panicked")?;
        log_slots(&stats, "final");
        eprintln!(
            "final attempts {}, mismatches {}, fallbacks {}",
            stats.attempts.load(Ordering::Relaxed),
            mismatches.load(Ordering::Relaxed),
            fallbacks.load(Ordering::Relaxed)
        );
        Err(reason.into())
    }

    // Drive the upstream CPU miner's Worker::attempt directly so every engine
    // receives the same finite nonce interval. Timing includes nonce hashing.
    fn cpu_tickets(
        engine: &Arc<Engine>,
        model: &Prepared,
        header: &[u8; 80],
        start: u32,
        count: usize,
        threads: usize,
    ) -> (Vec<[u8; 32]>, Duration) {
        let mut tickets = vec![[0; 32]; count];
        let chunk_size = count.div_ceil(threads);
        let barrier = &std::sync::Barrier::new(count.div_ceil(chunk_size) + 1);
        let elapsed = thread::scope(|s| {
            for (i, chunk) in tickets.chunks_mut(chunk_size).enumerate() {
                s.spawn(move || {
                    let mut w = engine.worker();
                    let mut h = *header;
                    let first = start + (i * chunk_size) as u32;
                    // Allocate worker scratch before timing, just as Metal retains its scan buffers.
                    w.attempt(model, blake3::hash(&h).as_bytes(), &Target([0; 32]), &Target([0; 32]));
                    barrier.wait();
                    barrier.wait();
                    for (j, t) in chunk.iter_mut().enumerate() {
                        h[76..].copy_from_slice(&(first + j as u32).to_le_bytes());
                        *t = w.attempt(model, blake3::hash(&h).as_bytes(), &Target([0; 32]), &Target([0; 32])).ticket;
                    }
                    barrier.wait();
                });
            }
            barrier.wait();
            let start = Instant::now();
            barrier.wait();
            barrier.wait();
            start.elapsed()
        });
        (tickets, elapsed)
    }

    fn benchmark(datasets: Vec<SharedDataset>) -> Result<(), String> {
        let threads = thread::available_parallelism().map_or(1, |n| n.get());
        let genesis =
            blake3::Hash::from_hex("da2f5af7bb4c5e2bcc8d558b2f4f01d4c88557a943e6cff9cd035638fe9439ae").unwrap();
        const COUNT: usize = 8192;
        const START: u32 = u32::MAX - (COUNT as u32 - 1);
        for (slot, dataset) in datasets.into_iter().enumerate() {
            let mut seed = blake3::Hasher::new();
            seed.update(b"neb-model-slot").update(genesis.as_bytes()).update(&(slot as u32).to_le_bytes());
            let model = Arc::new(Model::from_seed_arch(Arch::Kep1, seed.finalize().as_bytes()));
            let header = std::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(slot as u8));
            let reference = Engine::reference(dataset.clone());
            let ref_model = reference.prepare(&model);
            let table = Engine::new(EngineKind::Table, dataset.clone());
            let timer = Instant::now();
            let cpu_model = table.prepare(&model);
            let cpu_setup = timer.elapsed();
            let mut metal = MetalScanner::open(dataset.clone())?;
            let timer = Instant::now();
            metal.set_weights(&model)?;
            println!(
                "slot={slot} device={} threads={threads} samples={} weights={} nonce_start={START} count={COUNT} cpu_table_setup_ms={} metal_table_setup_ms={}",
                metal.device_name(),
                dataset.n(),
                blake3::Hash::from(model.weights_hash()),
                cpu_setup.as_millis(),
                timer.elapsed().as_millis()
            );
            neb_gpuminer::check_scanner(&mut metal, &model, &*dataset, &header, 64, 8)?;
            let mut expected = Vec::new();
            let mut rates = [Vec::new(), Vec::new(), Vec::new()];
            for rep in 0..4 {
                for step in 0..3 {
                    let backend = (step + rep) % 3;
                    let (result, elapsed) = match backend {
                        0 => cpu_tickets(&reference, &ref_model, &header, START, COUNT, threads),
                        1 => cpu_tickets(&table, &cpu_model, &header, START, COUNT, threads),
                        _ => {
                            let mut tickets = Vec::with_capacity(COUNT);
                            let timer = Instant::now();
                            for i in (0..COUNT).step_by(metal.capacity() as usize) {
                                tickets.extend_from_slice(metal.tickets(
                                    &header,
                                    START + i as u32,
                                    (COUNT - i).min(metal.capacity() as usize) as u32,
                                )?);
                            }
                            (tickets, timer.elapsed())
                        }
                    };
                    let rate = COUNT as f64 / elapsed.as_secs_f64();
                    if expected.is_empty() {
                        expected = result;
                    } else if result != expected {
                        return Err(format!("ticket parity failed: slot {slot} backend {backend}"));
                    }
                    if rep > 0 {
                        rates[backend].push(rate);
                    }
                }
            }
            if table.active() != EngineKind::Table || table.stats().mismatches != 0 {
                return Err("CPU table fallback".into());
            }
            for (name, mut r) in ["cpu_ref", "cpu_table", "metal"].into_iter().zip(rates) {
                r.sort_by(f64::total_cmp);
                println!(
                    "slot={slot} backend={name} median_attempts_s={:.0} min={:.0} max={:.0} identical_tickets={COUNT}",
                    r[1], r[0], r[2]
                );
            }
        }
        Ok(())
    }

    pub fn run() -> Result<(), String> {
        let mut args = std::env::args().skip(1);
        let (mut dataset, mut node, mut bench, mut selective, mut tess_ahead) = (None, None, false, false, false);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--help" | "-h" => {
                    println!("{USAGE}");
                    return Ok(());
                }
                "--dataset" => dataset = Some(args.next().ok_or(USAGE)?),
                "--node" => node = Some(args.next().ok_or(USAGE)?),
                "--benchmark" => bench = true,
                "--stratum-adapter" => selective = true,
                "--tess-ahead" => tess_ahead = true,
                _ => return Err(USAGE.into()),
            }
        }
        let pin = blake3::Hash::from_hex(DATASET_HASH).unwrap();
        let datasets = load_datasets(Path::new(&dataset.ok_or(USAGE)?), Some(pin.as_bytes()))?;
        match (node, bench) {
            (Some(addr), false) if !tess_ahead || selective => mine(&addr, datasets, selective, tess_ahead),
            (None, true) => benchmark(datasets),
            _ => Err(USAGE.into()),
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    if let Err(e) = apple::run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
