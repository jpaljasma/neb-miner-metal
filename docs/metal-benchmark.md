# Apple M1 Max KEP1 benchmark

Measured on November 26, 2026 with a 32-core Apple M1 Max GPU, 10 CPU threads, macOS 26.6.2 and a release build with arm64 Rust 1.100.0. Source baseline: [NEB v1.0.9](https://github.com/SilasEigen/neb/tree/v1.0.9).

| Fixed job | CPU reference | CPU table | Metal | Metal / CPU table |
| --- | ---: | ---: | ---: | ---: |
| Kepler | 2,724 | 151,197 | 307,990 | 2.04× |
| TESS | 2,774 | 166,692 | 335,433 | 2.01× |

Rates are median **attempts per second** across three timed passes after one warm-up. On Kepler the Metal passes ranged from 224,599 to 330,056 attempts/s. On TESS they ranged from 275,295 to 336,688. Each backend evaluated the last 8,192 nonces of the same deterministic 80-byte header for each slot. The harness compares the full 32-byte ticket for every nonce and fails on any difference. The order of the three backends rotates between passes.

The CPU paths call the same reference and optimized table workers as `neb-miner-cpu`. CPU workers initialize scratch before timing. The Metal path includes host sample selection, dispatch, synchronization, digest readback and final host hashing. The harness times local evaluation. Pool latency, job changes and candidate submission have separate costs during mining.

| Mainnet genesis model | Samples | CPU table setup | Metal table setup |
| --- | ---: | ---: | ---: |
| Kepler | 13,404 | 348 ms | 339 ms |
| TESS | 7,089 | 205 ms | 167 ms |

Setup is measured separately per model. The weights come from the mainnet genesis ID and slot number. Their hashes are `fafb313bcc36ff69b6ff51079979aab71be91b3596be6d930cf97d4167f90935` and `a13b29341195ed90717e45e69b5bf2a7fbc0e11451f4e3ac7a02dbf4ec9cce59`. Rates can change as training changes the model. Cross-miner comparisons require the same weights, header, dataset and nonce range.

To repeat the measurement, build and download the pinned release dataset as described in the [README](../README.md), then run:

```sh
node/target/release/neb-miner-metal --dataset build/neb-datasets.bin --benchmark
```

## Same-Mac comparison with upstream PR #2 and table reuse

The first run of this experiment used the unchanged miner at `b433ada`. The regular CPU-parity benchmark measured 255,593 attempts/s on Kepler and 278,007 on TESS. All 8,192 tickets per slot matched the C reference.

The separately published M4 Pro KEP1 testnet measurement in [PR #2](https://github.com/SilasEigen/neb/pull/2#issuecomment-7024496254) is 6,162 attempts/s. Its 58,435 attempts/s figure uses the older v1 network. SilasEigen's own [per-core estimate](https://github.com/SilasEigen/neb/blob/v1.0.9/docs/testnet.md) is about 2 ms per KEP1 reference attempt. On this M1 Max the single-thread C reference took about 2.6 ms per ticket on the two mainnet genesis jobs.

For a same-hardware comparison, the replay exercised the unchanged [PR #2 scan implementation at 52af7b8](https://github.com/pramonoutomo/neb/tree/52af7b84d897659359f4426e49f0c0efcfbdbb17/node/crates/neb-metalminer) and our unchanged table scanner sequentially. Both evaluated the same genesis weights, header, dataset and last 8,192 nonces in two requests of 4,096. Each figure below is the median of five passes after a warm-up. Scan timing covers completed tickets on the host. Model preparation is separate.

| Matched job | PR #2 scan, attempts/s | Local table scan before change, attempts/s | Local / PR #2 |
| --- | ---: | ---: | ---: |
| Kepler | 4,742 | 284,880 | 60.1× |
| TESS | 4,792 | 240,003 | 50.1× |

PR #2 computes forward and backward passes on the GPU. The local scanner first builds per-sample gradients on the CPU and reuses them across nonces. On the first matched run, local table preparation cost 326 ms on Kepler and 182 ms on TESS, while the PR's model upload took less than 0.01 ms in each slot. These scan ratios describe prepared jobs.

### Repeated model frames

[Suprminer v1.9.50](https://github.com/ocminer/suprminer/releases/tag/v1.9.50) keeps result tables resident across jobs and updates them where the model changes. The Metal adaptation keeps one copy of the 26,000 weight bytes per scanner and compares them when the node resends a model. Identical bytes reuse the slot's GPU buffer. Changed weights rebuild the complete table. The work-service worker still probes newly received models and CPU-verifies hits.

The sequence was baseline, modified, baseline, modified. Each pass supplied a freshly decoded model with identical weights, then scanned 8,192 nonces. Ranges below span the two run medians for each build.

| Slot | Baseline reload | With weight comparison | Reload + scan before | Reload + scan after |
| --- | ---: | ---: | ---: | ---: |
| Kepler | 314–330 ms | 1.63–1.67 µs | 345–357 ms | 19.5–19.7 ms |
| TESS | 171–174 ms | 1.54 µs | 203–208 ms | 19.3–20.1 ms |

This saves time on repeated model frames. The node can resend them on a same-tip refresh after 30 seconds. The Stratum test adapter already suppresses consecutive identical model notifications. The shader is unchanged, and this experiment establishes no steady-state scan speedup. Every engine produced the same ordered ticket-stream hash per slot. Switching to a different model and back also passed exact CPU ticket checks.

Run the local replay from the repository root:

```sh
cargo run --manifest-path node/Cargo.toml --release --locked -p neb-gpuminer --features metal \
  --example metal_refresh -- build/neb-datasets.bin
```

To repeat with PR #2, check out the pinned commit in a separate directory and set `PR2_DIR` to that checkout. The replay source has an adapter for the PR's native scan API. Only the example is added to that checkout.

```sh
mkdir -p "$PR2_DIR/node/crates/neb-metalminer/examples"
cp node/crates/neb-gpuminer/examples/metal_refresh.rs "$PR2_DIR/node/crates/neb-metalminer/examples/"
cargo run --manifest-path "$PR2_DIR/node/Cargo.toml" --release --locked -p neb-metalminer \
  --example metal_refresh -- "$PWD/build/neb-datasets.bin"
```
