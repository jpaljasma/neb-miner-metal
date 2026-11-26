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
