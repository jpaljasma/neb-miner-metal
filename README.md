# Mining NEB on Apple silicon

If you have an Apple silicon Mac, its GPU can contribute to NEB’s Kepler and TESS models. `neb-miner-metal` takes jobs from NEB’s existing work service, computes tickets with Metal, and checks each candidate against the upstream CPU reference before submitting a share. [Upstream NEB](https://github.com/SilasEigen/neb)

We’d like to contribute this backend to NEB. A starting point for upstream review is the exhaustive scanner behind the optional `metal` Cargo feature. Run the offline benchmark below to compare every byte of 8,192 GPU tickets per model against the CPU on the same jobs. The selective pool scanner and terminal dashboard can be evaluated separately. [Metal scanner](node/crates/neb-gpuminer/src/metal_scanner.rs) · [CPU parity benchmark](docs/metal-benchmark.md)

## M1 Max scan speed

On a 32-core M1 Max with 64 GB of unified memory, neb-miner-metal scanned **146.7× as fast on Kepler** and **135.6× as fast on TESS** as `pramonoutomo/neb-metalminer` once each model was loaded. Both miners produced the same 8,192 tickets per model on identical jobs. The table shows median exhaustive scan rates over five passes.

| Model | neb-miner-metal | pramonoutomo/neb-metalminer |
| --- | ---: | ---: |
| Kepler | 695,146 tickets/s | 4,737 tickets/s |
| TESS | 676,496 tickets/s | 4,989 tickets/s |

The two miners make different tradeoffs. This backend spent 306 ms on Kepler and 164 ms on TESS building lookup tables before scanning. Pramonoutomo’s implementation loaded each model in 0.003 ms and computed gradients during the scan. Its public Metal proposal made this same-Mac comparison possible. [Benchmark and replay commands](docs/metal-benchmark.md#matched-metal-scan-on-november-27-2026) · [Upstream Metal proposal](https://github.com/SilasEigen/neb/pull/2)

The trained-model screen can stop the last ChaCha8 round as soon as a nonce has too many live rows for the pair table. In ten-second settled Kepler replays, the M1 Max completed **2.1–4.2% more tickets** across three recorded models. [GPU comparisons and methods](docs/metal-pool-tuning.md#suprminer-1963-and-intel-arc)

On the 64 GB M1 Max, the miner preloads the Metal pipelines and prepares gradient rows directly in shared memory. Larger dispatches after table warm-up raised **audited throughput by 5.9–12.2% on Kepler and 10.0–10.9% on TESS** across three recorded model transitions. On a moderately dense Kepler model the exhaustive engine also improved from **0.91 to 1.05–1.06 MH/s**. [GPU utilization, latency and exhaustive replays](docs/metal-pool-tuning.md#feeding-the-m1-max-gpu)

## Pool mining on M1 Max

In a 30-minute Suprnova pool run on the same Mac, Kepler averaged **23.22 MH/s** and TESS averaged **93.30 MH/s**. Each rate counts completed tickets over the time the pool assigned that model, including setup. Suprnova accepted 59 of 59 Kepler shares and 19 of 20 TESS shares. One TESS share went stale at a job change. [Pool tests and tuning](docs/metal-pool-tuning.md)

When a job changes model weights, the GPU builds a lookup table of pair digests. On held-out M1 Max jobs, using one 32-lane SIMD group per pair and simplifying the dense arithmetic cut full table build time from **1.653 to 1.541 s on Kepler** and **270 to 239 ms on TESS**. Across six matched job starts per model, Metal completed **2.10% more Kepler tickets** and **1.06% more TESS tickets** in the first five seconds. [Model-turnover benchmarks](docs/metal-turnover.md#one-simd-group-per-pair)

The miner uses the available CPU cores to prepare new model weights for the GPU. Across two M1 Max replays per model, ten setup threads produced **2.04–2.06× as many Kepler tickets** and **1.29–1.31× as many TESS tickets** as one thread during the first five seconds of a job. [CPU thread benchmark](docs/metal-turnover.md#cpu-threads-during-metal-job-setup)

![NEB dashboard mining over TCP, with Kepler active, the last TESS job and recent pool replies](docs/images/metal-miner-dashboard.png)

## Try it on your Mac

Install the Xcode Command Line Tools and a stable arm64 Rust toolchain. From the repository root, build the miner, download the pinned NEB dataset, and run the offline CPU and GPU benchmark.

```sh
cargo build --manifest-path node/Cargo.toml --release --locked -p neb-gpuminer --features metal --bin neb-miner-metal
mkdir -p build
curl --fail --location --output build/neb-datasets.bin https://github.com/SilasEigen/neb/releases/download/v1.0.9/neb-datasets.bin
node/target/release/neb-miner-metal --dataset build/neb-datasets.bin --benchmark
```

**Have an M2 Max, M3 Max, M4 Max or M5 Max?** Try the benchmark above and share your results. Include your chip, GPU core count, unified memory, macOS version, commit (`git rev-parse --short HEAD`) and the benchmark output for both slots. Slot 0 is Kepler and slot 1 is TESS. [Report a benchmark](https://github.com/jpaljasma/neb-miner-metal/issues/new)

## Mine with Suprnova

With Python 3 installed, save your NEB payout address in `.neb-payout-address` and your Suprnova pool URL in `.neb-pool-url`, each on one line in the repository root. Use `stratum+tcp://` for plain TCP or `stratum+ssl://` for TLS, with the corresponding port from the pool guide. Git ignores both files. [Suprnova pool setup](https://neb.suprnova.cc/StartMining.html)

```sh
./start-neb.command
```

The launcher opens a 24-hour pool worker and its dashboard in separate Terminal windows. Press **q** to close the dashboard. Press **Control-C** in the worker window to stop mining.

The dashboard shows device-wide GPU load and miner-process CPU load below it (100% CPU is one core). Pool difficulty uses K, M, G and T, with G denoting one billion. Each model’s work time shows days, hours, minutes and seconds as needed.

## Optional TESS sky search

The Metal miner includes the CPU sky-search client from NEB v1.0.11. On a fresh pool launch, pass the upstream server URL to enable it alongside mining:

```sh
./start-neb.command --sky http://sky.neuralenginebyte.org --sky-threads 1
```

The same flags work on `neb-miner-metal --node HOST:PORT` and the Stratum adapter. Mining alone is the default. Sky search starts with one CPU thread unless you set `--sky-threads N`. The dashboard shows whether search is enabled, its thread count and batch progress. The server receives the Metal device name by default. Use `--sky-device TEXT` to change it and `--sky-address ADDRESS` to send an optional address. The sky address defaults to empty independently of the pool payout address. [Upstream sky-search release](https://github.com/SilasEigen/neb/releases/tag/v1.0.11) · [Batch and result formats](spec/sky-search.md)

The mining base is [SilasEigen/neb v1.0.9](https://github.com/SilasEigen/neb/tree/v1.0.9). [Apache-2.0 license](LICENSE)
