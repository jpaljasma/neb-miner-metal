# M1 Max pool mining and kernel measurements

The ordinary node client evaluates every nonce in its assigned range. The optional Suprnova adapter can use selective mining. It screens a larger range and evaluates only combinations for which the GPU has an exact digest. The displayed rate counts completed tickets. Every submitted share still passes the independent upstream CPU verifier.

```sh
cargo build --manifest-path node/Cargo.toml --release --locked -p neb-gpuminer --features metal --bin neb-miner-metal
mkdir -p build
python3 tools/test_metal_stratum.py --pool "$POOL_URL" --worker "$WORKER" \
  --duration 1800 --selective --log build/pool-session.jsonl
```

Use a new log filename per session. The adapter reads the payout address from the ignored `.neb-payout-address` file. TLS validates the certificate and hostname. A fresh subscription, authorization, target, model and job are required after a disconnect. The log distinguishes submissions, acknowledgements, rejections and unanswered submissions. A pool share acceptance is distinct from an on-chain payout.

For an intentional reconnect test add `--test-reconnect`. For exclusive benchmarks add `--pause-file build/PAUSE`; create this file to drain pending shares, disconnect and reap the worker, then remove it to resume. The duration begins at the first job and excludes these pauses. The deadline stops mining and gives outstanding replies a short grace period. Local nonce exhaustion rolls extranonce2 under the original pool job, retaining the connection and using a distinct header.

## Controlled measurements

Hardware: Apple M1 Max, 32 GPU cores, 64 GB unified memory. Two trained mainnet KEP1 models were captured from pool jobs and verified against their advertised weight hashes. The models had 4,725 live rows out of 13,404 for Kepler and 1,638 out of 7,089 for TESS. These are measurements on those snapshots. Pool hashrate, wall power and return on electricity are separate quantities.

Mining was paused during GPU timings. A 2⁵ factorial crossed cached row addresses, planar shared factors, 16-bit four-lane multiply-accumulates, a zero-gradient salt sum, and 128 versus 256 threads per threadgroup. These experiments draw on the upstream [CUDA table kernel](https://github.com/SilasEigen/neb/blob/v1.0.9/pow/gpu/nebhash2_table_cuda.cu) and [sparse tensor-core kernel](https://github.com/SilasEigen/neb/blob/v1.0.9/pow/gpu/shares_tc.cu). With four-bit factors all 16 products sum to at most 1,024 in magnitude, so the narrower integer arithmetic is exact.

Each screen used 12 randomized complete blocks per model after warm-up. Each nominated candidate then ran on fresh headers in 24 order-balanced paired blocks per model. The interval is a 90% paired bootstrap of equal-slot log speed ratios. The promotion threshold was a lower bound of +1.5%. All tickets in the full-coverage benchmarks matched the CPU table, with additional direct C reference checks. The stages have separate baselines.

| Change tested | Kepler | TESS | Combined gain (90% interval) | Decision |
| --- | ---: | ---: | ---: | --- |
| Short-vector arithmetic, zero-salt precomputation and 256 threads | +34.6% | +33.7% | +34.1% (+32.5 to +35.8%) | Use on M1 Max |
| Rust staging, 16,384 nonces across eight commands of 2,048 | +24.4% | +23.7% | +24.0% (+22.5 to +25.6%) | Use; preserve full-path audit frequency |
| CPU BLAKE3 header-prefix caching alone | +1.6% | +0.5% | +1.0% (+0.1 to +2.1%) | Keep as replay experiment |
| Four-key ChaCha8 in Rust, zero-row compaction and shortened Metal dot products | +45.0% | +66.8% | +55.5% (+53.5 to +57.6%) | Use on sparse models |
| Pair lookup in an otherwise exhaustive kernel | −5.1% | +9.1% | +1.8% (+0.1 to +3.5%) | Use the device-side screen for lookup mining |

Metal command execution covered 76.6% of wall time with the blocking queue and 94.5% with eight commands. Each command uses disjoint shared-buffer regions. Every command finishes before its region is reused, including when another command fails. Staging increased by 1.5 MiB per slot. Device-wide utilization from IORegistry also includes macOS and other applications.

## Exact result tables and nonce selection

The Rust host constructs the upstream per-sample gradient table. A sample with zero gradient contributes nothing, including when drawn repeatedly. The Metal screen computes the header key and ChaCha8 sample indices. Zero and single live rows give a direct digest lookup. For two live row ranks `a ≤ b`, the pair digest is at `b*(b+1)/2+a`. The GPU hashes the digest and key and compares all 256 ticket bits with the target. Skipped nonces do not count towards throughput.

| Strategy | Kepler completed tickets/s | TESS completed tickets/s | Fraction of screened nonces evaluated, Kepler / TESS |
| --- | ---: | ---: | ---: |
| Exhaustive, compacted rows | 1.37 M | 1.44 M | 100% / 100% |
| GPU screen, zero or one live row | 5.87 M | 49.23 M | 0.92% / 8.69% |
| GPU screen through exact pairs | 24.75 M | 138.62 M | 4.31% / 24.88% |
| Pairs plus partial triples (128 MiB per slot) | 24.97 M | 137.67 M | 4.32% / 25.01% |

Values are per-slot medians from 24 timed blocks. The independent checker verifies eligibility, complete ticket bytes, CPU-checked samples, hit lists and evaluated counts at three nonce intervals, including the upper boundary. The counter and result list are both bounded by the number of screened nonces.

Complete pair tables occupied 341 MiB on Kepler and 41 MiB on TESS and took roughly 3.6 s and 0.6 s to build in the offline replay. The triple slice took another 1.6–2.1 s per slot and did not meet the incremental gain threshold. A full triple table on these models would take approximately 525 GiB on Kepler and 21.9 GiB on TESS. Memory capacity alone does not recover table-build time.

In selective pool mode the miner verifies the screen on every new model and compares its predicted productive rate with a measured exhaustive scan, charging the estimated build cost against a 30-second job. It can return to exhaustive scanning. Pair tables grow in bounded slices interleaved with mining; the GPU reads only the completed prefix. The production limit is 1 GiB of pair digests per slot. Per-model screening audits and an independent recheck of all outgoing hits remain active. The triple builder is available in the benchmark.

### Incremental model changes

The startup trial uses the production selective dispatch size. When exhaustive scanning wins the 30-second budget but the fully built screen is predicted to beat it by at least 25%, the worker can revisit that decision on a long-lived model. It first spends 15 seconds doing exhaustive work. It then budgets one second of incremental building per eight additional seconds of full scanning. At successive table coverage levels it probes the actual completed-ticket rate. A partial table must beat measured exhaustive throughput by 60% to cover continuing construction, or a complete table by 5%. A successful switch requires a fresh CPU audit. Missing, invalid or unprofitable tables keep the exhaustive path. Probe attempts are excluded from the miner's reported rate.

On a later pair of same-slot transitions, 23.6% of the Kepler pair entries and 4.8% of TESS entries were eligible for reuse. The host compares each new live row's complete packed gradient/factor bytes with that dataset row at the previous model. The GPU can copy an old digest only if both rows are identical, both previous ranks exist, and the old pair index falls inside the completed prefix of the previous table. Changed pairs are computed from the new model. Old storage is released when the new pair table finishes. A transition with less than 10% potential reuse takes the fresh-build route.

In 12 crossover trials on the Kepler transition, median total model setup fell from 3.790 to 3.051 seconds (19.5% less elapsed time). The two-sided 90% paired interval for setup **speedup** was +23.9% to +25.4%. CPU checks compared both eligibility and full selected ticket bytes. On TESS the gate stayed off: 0.885 versus 0.876 seconds, inside the 1.5% selection threshold. Incremental reuse improves job turnover; it does not change the per-nonce scan kernel.

### Scope of the port

The implementation preserves the high-output miner techniques that reproduced locally: exact per-sample caching, GPU-side nonce screening, one-/two-row digests, bounded partial triple replay, incremental pair generation, safe reuse across model versions, per-model selection, overlapping Rust/Metal work, and independent share verification. Pool mode preserves eligible in-flight same-parent jobs and rolls fresh extranonces. The source does not claim equivalent throughput to proprietary miners. The exact triple builder uses direct gradient evaluation; the vendor's private inclusion-exclusion implementation and hardware-specific Ada/Ampere kernels were not available for inspection.

## Selective dispatch size on M1 Max

On the same two captured mainnet models, a separate replay changed only the number of raw nonces per synchronous selective dispatch. Each arm screened the same 4,194,304 nonces per block. After warm-up, order alternated across 32 paired blocks per slot. Each pair had identical evaluated counts and hit sets, and the CPU reference checked every returned hit. The last interval included nonce `u32::MAX`.

| Slot | 262,144 nonces/dispatch | 1,048,576 nonces/dispatch | GPU command time / wall time |
| --- | ---: | ---: | ---: |
| Kepler | 25.91 M tickets/s | 43.20 M tickets/s | 49.7% → 75.1% |
| TESS | 102.34 M tickets/s | 162.93 M tickets/s | 59.8% → 82.2% |

Rates are medians of completed tickets per second with the pair tables ready. The equal-slot paired geometric gain was **56.5%** (90% bootstrap interval **47.0% to 66.8%**). GPU command coverage uses Metal timestamps. Activity Monitor reports a separate process utilization metric. The larger output buffer costs 3 MiB per slot, and the production audit intervals scale with the batch size to preserve checks per nonce. Repeat with `metal_selected DATASET SNAPSHOT_DIR dispatch`. Live throughput depends on the model and job turnover.

## Two additional Metal screen experiments

The first variant gives each ChaCha8 quarter-round fixed scalar indices. The second compiles a pair-only screen that exits at the third live row. The ideas came from the scalar quarter-rounds in an independent Metal miner and compile-time sparse specialization in the reference CUDA miner. [Metal source](https://github.com/pramonoutomo/neb/blob/52af7b84d897659359f4426e49f0c0efcfbdbb17/node/crates/neb-metalminer/src/neb_common.metal) · [CUDA source](https://github.com/SilasEigen/neb/blob/v1.0.9/pow/gpu/shares_tc.cu)

Each of 32 order-balanced blocks per model screened the same 4,194,304 nonces in all four configurations. Across both models the scalar variant changed throughput by −0.01%, the pair-only variant by +0.22%, and the combination by −0.05%. The best candidate, pair-only, ran again on fresh Kepler and TESS snapshots and headers. Each timed pair returned identical evaluated counts and hit sets. Separate CPU checks exercised exact-target and just-below-target comparisons.

| Pair-only holdout | Throughput gain | 90% paired bootstrap interval |
| --- | ---: | ---: |
| Kepler | +1.80% | +1.19% to +2.44% |
| TESS | +0.39% | −0.35% to +1.16% |
| Both, equally weighted | +1.09% | +0.57% to +1.61% |

Production selective mining now combines the pair-only specialization and scalar ChaCha8. It uses the general shader for triple lookups and an independent vector implementation for its audit records. The earlier pair-only result above motivated inclusion despite the uncertainty. A follow-up 32-block paired replay measured the combined kernel at **+0.83% on Kepler** (90% interval −1.26% to +2.89%) and **−0.45% on TESS** (−1.27% to +0.42%) against the original vector, general-purpose screen. The equal-weight change was +0.19% (−0.84% to +1.22%).

The screen still reserves three local row indices. Reducing that array to two and changing its write guard slowed the vector pair kernel by 6.8% relative to the original general shader in an exploratory run. Moving the original 96-byte dispatch parameters to persistent shared memory alone measured −0.66% across the two models (−1.85% to +0.49%). The production buffer also caches header calculations, as measured below.

Run `metal_selected DATASET SNAPSHOT_DIR variants` for all eight combinations. Mask bits 1, 2 and 4 select scalar ChaCha8, pair-only specialization and mapped parameters respectively. Use `variants 7` on fresh snapshots to compare the combined kernel with its component controls.

### Zero-copy single-row lookup table

Model setup writes digest records directly into the shared Metal buffer. Each model rebuild eliminates a 482,580-byte allocation and copy for Kepler or a 255,240-byte allocation and copy for TESS on the pinned dataset. The table builder writes only live records and the empty digest into the newly zeroed destination. [Metal buffer initialization](https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28length%3Aoptions%3A%29)

In 32 paired rebuilds per model on each of two snapshot pairs, both paths produced byte-identical tables every time. On the holdout, median build-and-upload time went from **18.67 to 18.05 ms on Kepler** and **4.76 to 4.75 ms on TESS**. All setup timing intervals included zero. The allocation and copy savings apply on each rebuild. Run `metal_selected DATASET SNAPSHOT_DIR lookup-upload` to reproduce the comparison.

### Reusing BLAKE3 work across nonces on M1 Max

The pair kernel uses a persistent 160-byte shared argument buffer per scanner. For a new 76-byte header prefix, the CPU computes the BLAKE3 chaining value and the nonce-independent work in the next block's first round. Three columns are complete. The fourth stops where the nonce enters. The GPU reads those states directly for each nonce. Changes to the target, start nonce or pair-table coverage update their own fields each dispatch. A change anywhere in the header prefix refreshes the cached state.

Decred's open-source BLAKE3 miner folds initialization into round one and filters on the high output word. The NEB CUDA table kernel shares common inputs within a block and checks an arbitrary 256-bit target. We adapted these ideas to NEB's shorter header and exact tie handling, then tested the cache, an early high-word comparison, and explicit Metal threadgroup staging. [Decred OpenCL kernel](https://github.com/decred/gominer/blob/68791b06465250db861e8d7ce9c6b5b2ccdb074b/blake3.cl) · [NEB CUDA table kernel](https://github.com/SilasEigen/neb/blob/v1.0.9/pow/gpu/nebhash2_table_cuda.cu)

The eight-way screen used 64 order-balanced blocks per model. Explicit threadgroup staging alone cost 4.79% across the models (90% interval −5.71% to −3.85%). Four configurations that used no staging then ran on different model snapshots and headers for 128 paired blocks per model. Each block covered 8,388,608 raw nonces per configuration. Each pair had identical evaluated counts and hit sets. Parity probes exercised the last nonce, exact ticket targets and just-below targets.

| Holdout gain over the scalar pair kernel | Kepler | TESS | Both, equally weighted |
| --- | ---: | ---: | ---: |
| Cache header columns in unified memory | **+2.29%** (+0.79% to +3.80%) | **+1.33%** (+0.34% to +2.34%) | **+1.81%** (+0.95% to +2.64%) |
| Filter on the high word first | +1.29% | −0.15% | +0.57% (−0.38% to +1.53%) |
| Combine the cache and high-word filter | +0.78% | +0.32% | +0.55% (−0.30% to +1.45%) |

The cache alone gave the highest confirmed throughput on both models and exceeded the current 0.5% point-estimate cutoff. Run `metal_selected DATASET SNAPSHOT_DIR hot` for the factorial and `metal_selected DATASET FRESH_SNAPSHOT_DIR hot-confirm` for confirmation. Save the CSV output and run `python3 tools/analyze_metal_factorial.py FILE.csv` for the paired estimates and 90% intervals. The masks add bits 8, 16 and 32 for cached columns, high-word filtering and threadgroup staging to the scalar pair baseline of 3.

## Ahead-of-job TESS preparation

[Suprminer v1.9.62](https://github.com/ocminer/suprminer/releases/tag/v1.9.62) prepares announced TESS models during Kepler on selected Linux RTX 50 cards. The Metal adapter supports this scheduling idea with `--selective --tess-ahead`. Its input is a standard `mining.set_model` for TESS received while a Kepler job remains active. The adapter waits 100 ms to let an ordinary immediate job notification take precedence. A separate scanner validates the weights, builds the CPU table with up to two threads, and prepares the GPU pair table in slices. Each slot retains its own active scanner. The authoritative TESS job adopts the prepared scanner only on an exact weight match. An incorrect hint, missing hint or unfinished preparation leaves the normal job path available.

In ten paired M1 Max transition replays, a simulated two-second announcement led a five-second TESS job. Trained weights alternated between two captured TESS models. Relative to on-demand setup, Kepler completed 15.5% fewer tickets during the lead, TESS completed 24.9% more during the first five seconds, and combined work increased **19.8%** (90% paired bootstrap interval **18.3–21.3%**). Median on-job setup fell from **205.8 ms to 1.1 ms**. The replay CPU-checks hits, exercises a stale hint, and rejects a hint with an incorrect hash. In an earlier live log the median lead of 15 ordinary TESS notifications was zero and the maximum was 95 ms. The replay result applies when an actual advance announcement is available. Any additional pool-specific prediction protocol remains unverified.

Run `metal_starts DATASET ADJACENT_MODELS_DIRECTORY` (Cargo example, feature `metal`) with no live miner sharing the GPU. The launcher enables ahead preparation for new sessions. The underlying adapter flag is optional.

## What other miners disclose

| Source | Reported evidence | Metal takeaway |
| --- | --- | --- |
| [FreeForGem NEB](https://github.com/pepsykolya/FreeForGemNeb) | The author reports +17–18% on TESS with partial triples on two 12 GB NVIDIA cards. The release notes describe a device nonce screen, exact pair digests, gradual triple construction, per-model trials and stale-share handling. | Implemented the GPU screen, exact pair and bounded triple builders, model trials, progressive pair construction, CPU verification and stale-job retirement. The measured M1 Max triple slice stays experimental. |
| [PI-neb-miner v1.3.3](https://github.com/PItheWanderer/PI-neb-miner/releases/tag/v1.3.3) | On RX 9070 XT, the author measured 207 MH/s Kepler and 1,373 MH/s TESS at a reduced power limit. Its manual documents pair mode and optional larger triple tables. | Measure digest-table size and build cost separately for each slot. |
| [credesbois comparison](https://github.com/crecre1589/credesbois/blob/main/COMPARATIF.md) | The author reports about 148 MH/s pool-side on one RTX 5070 Ti, with pool-credit methodology and uncertainty. Its miner exposes an adjustable live-row threshold. | Separate evaluated tickets, raw screened nonces and pool acknowledgements. |
| [Aria Energy NEB](https://pool.ariabrain.com/neb.html) | The operator documents retaining valid shares across same-parent jobs and keeping GPUs ready on disconnect. | The adapter keeps same-parent job mappings, retires stale jobs and starts a fresh TLS session after loss. The current supervisor restarts its Metal child on reconnect. |
| [Pro7as/nebminer](https://github.com/Pro7as/nebminer) | Author-published RTX 5090 results of 202–203 MH/s Kepler and 223–225 MH/s TESS on November 19, with automatic kernel selection. | Select using measurements on the local GPU and model. |

The explorer ranks mining addresses by blocks. It does not identify the binary behind each address. Vendor rates above use different models, hardware, clocks and counting methods. They do not establish a cross-miner ranking. The implementations here are independent. NVIDIA clock locks, undervolting, fan control, reboot policy, proprietary binaries, developer-fee routing and a second live pool connection for duplicate work are outside the M1 Max compute path.

## Replay

```sh
mkdir -p build/models
python3 tools/capture_metal_models.py --pool "$POOL_URL" --worker "$WORKER" build/models
cargo build --manifest-path node/Cargo.toml --release --locked -p neb-gpuminer --features metal-bench --examples
node/target/release/examples/metal_factorial build/neb-datasets.bin build/models screen > build/screen.csv
python3 tools/analyze_metal_factorial.py build/screen.csv
node/target/release/examples/metal_factorial build/neb-datasets.bin build/models confirm 28
node/target/release/examples/metal_selected build/neb-datasets.bin build/models
```

The factorial example also accepts `queue`, `host`, `compact`, `tables` and the corresponding `*-confirm ID`. For model turnover, capture distinct weights with `--per-slot 2` and pass the capture directory to `metal_factorial DATASET DIR row-delta` or `metal_factorial DATASET DIR reuse`. Capturing models is a separate read-only pool subscription. Pause live mining before timing a replay.

For status during a timed session, run `python3 tools/report_metal_session.py --live LOG_DIRECTORY`. Omit `--live` for the final report. The per-slot report separates raw nonces screened from full tickets evaluated, unique job exposure from worker compute time, and acknowledged share difficulty from displayed hashrate. The final miner counters include the last partial sampling interval.
