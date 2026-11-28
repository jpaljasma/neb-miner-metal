# M1 Max model-turnover replay

The baseline is the merged miner with tagged rank lookups and 4,194,304 nonces per selective dispatch. All measurements below use completed tickets. Each GPU replay ran alone on the 32-core M1 Max with 64 GB of unified memory. [Baseline revision](https://github.com/jpaljasma/neb-miner-metal/commit/73b45007291ede3d5f9680cb791a6eab522da2e8)

## What changed in each arm

| Factor | Enabled behavior |
| --- | --- |
| A · exact triples | Consider a nonce with exactly three live samples. Check that all three pair entries are ready and that no weight has a nonzero gradient in all three samples. Reconstruct its digest from the pair, single and empty digests. |
| B · two-row builder | Build a pair using two signed byte factors per element in threadgroup memory. Use the union of the rows' precomputed dense-output masks to visit the outputs that can contribute. A separate crossover also tests the two-row builder without the mask. |
| C · skip unchanged writes | On a model change, reuse the byte comparison already performed for pair remapping. A live row with identical packed bytes stays in its existing shared buffer. Changed rows get the full copy. |

The eight arms toggle A, B and C independently. Mask values are 1, 2 and 4. The control has mask 0. All arms retain the baseline's pair reuse and CPU share verification.

## Which triples can be recovered?

Let D(X) be the eight 32-bit digest lanes of the summed gradients in X. When at least one of a, b or c contributes zero at every weight, the following identity holds lane by lane modulo 2³², including after clipping and XOR with the weight salt.

```text
D(abc) = D(ab) + D(ac) + D(bc) - D(a) - D(b) - D(c) + D(empty)
```

The support mask comes from the actual packed convolution gradients and dense-layer factors. The CPU coverage test independently obtains its masks from the reference sample-gradient function. In one million uniformly drawn live triples per model, sampled with replacement, Kepler yielded 937, 468, 982 and 6,430 recoverable cases on four snapshots. TESS yielded zero in the four samples. These rates are conditional on drawing exactly three live rows. A zero sample count does not prove the model has no recoverable triples.

## All eight combinations on the fixed holdout

Each run loaded a predecessor and warmed its pair table before transitioning to different weights. Setup includes the new model and complete pair table. Scan rates are medians over 32 fixed ranges of 8,388,608 nonces after setup. Within each slot, all eight full pair tables had the same byte hash. Candidates in the timing runs were CPU-checked.

| Enabled | Kepler setup (s) | Kepler scan (MH/s) | TESS setup (s) | TESS scan (MH/s) |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 5.396 | 13.60 | 0.652 | 480.86 |
| A | 5.370 | 6.34 | 0.656 | 298.71 |
| B | 2.789 | 13.46 | 0.440 | 471.26 |
| A+B | 2.799 | 6.50 | 0.421 | 288.34 |
| C | 5.435 | 13.56 | 0.657 | 484.36 |
| A+C | 5.452 | 6.47 | 0.636 | 303.80 |
| B+C | 2.760 | 12.98 | 0.412 | 464.93 |
| A+B+C | 2.844 | 6.35 | 0.408 | 307.34 |

The triple path added 39,265 exact Kepler tickets across the 32 holdout ranges. Its extra classification and support reads took more time than those tickets repaid. B and C affect setup. Their scan-only differences include variation between runs of the same pair screen.

## First tickets after a model change

Each arm also started from a prepared predecessor on an adjacent-weight transition. The clock started before loading the new weights. It included the adaptive selector, building pairs between scans, scanner audits and CPU verification of returned hits. At each deadline the outstanding batch finished and the rate used its observed elapsed time. Each cell below is one timed transition; the small differences between nearby arms need separate paired confirmation.

| Enabled | Kepler first 5 s | First 15 s | First 30 s | TESS first 5 s | First 15 s | First 30 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 20.33 | 43.82 | 55.40 | 186.07 | 236.64 | 250.19 |
| A | 10.80 | 19.20 | 23.34 | 135.20 | 165.17 | 172.08 |
| B | 29.08 | 53.20 | 59.42 | 214.78 | 246.18 | 253.35 |
| A+B | 14.02 | 23.22 | 25.72 | 148.55 | 167.01 | 173.97 |
| C | 19.50 | 41.37 | 54.18 | 191.01 | 239.57 | 248.05 |
| A+C | 11.22 | 19.21 | 23.63 | 127.67 | 163.22 | 170.58 |
| B+C | 25.09 | 49.16 | 55.53 | 203.73 | 239.29 | 244.42 |
| A+B+C | 14.20 | 22.07 | 25.15 | 153.33 | 170.14 | 175.42 |

Units are millions of completed tickets per second. Both slots selected the lookup screen in these runs. Pool acknowledgements and the outer work-service audit are covered separately in a real pooled run.

## Which parts of the builder helped?

We compared the general builder, the two-row builder and the two-row builder with a precomputed dense-output mask on 48 order-balanced slices per snapshot and slot. Each arm wrote the same 65,536 pair entries per slice. The entire output slice was byte-hashed outside the timer. Metal reported 13,600 bytes of static threadgroup storage for the general builder and 1,936 bytes for both specialized builders.

| Paired gain in builder throughput | Adjacent Kepler | Holdout Kepler | Adjacent TESS | Holdout TESS |
| --- | ---: | ---: | ---: | ---: |
| Masked two-row builder over general | +96.64% | +106.03% | +89.29% | +88.82% |
| Adding the mask to the two-row builder | +2.47% | +4.19% | −1.57% | −1.91% |

The 90% paired bootstrap intervals for the incremental mask effect were +1.21% to +3.83% and +3.70% to +4.67% on Kepler. On TESS they were −1.86% to −1.27% and −2.15% to −1.67%. The selected Kepler builder uses the mask. The selected TESS builder checks its two error values directly.

Skipping confirmed identical rows eliminated 7.84 MiB of writes per adjacent Kepler transition and 1.38 MiB on TESS. The more distant transition had no reusable rows on either slot. In 20 paired adjacent refreshes with byte-hashed mapped output, the estimated change in complete model-preparation speed with C was −0.63% on Kepler (90% interval −3.11% to +1.89%) and −2.76% on TESS (−6.68% to +1.16%). Skipped bytes alone did not establish a throughput gain.

On the reverse transition, six order-balanced five-second trials per slot compared the baseline, the selected builder and the builder plus skipped writes. The selected builder improved completed tickets per second by **39.06% on Kepler** (90% paired interval +35.86% to +42.22%) and **15.75% on TESS** (+14.33% to +17.25%). Adding skipped writes changed those rates by −0.82% (−3.00% to +1.06%) and −0.34% (−1.35% to +0.58%). Production enables B alone. A and C remain benchmark options.

The CPU oracle covers 131,072 nonces through the end of the nonce range on each checked model. It verifies full and restricted pair-table prefixes, eligible counts, hit sets and exact-target comparisons. A targeted search also located and checked 335 newly admitted Kepler tickets and three rare TESS tickets against the reference, including targets one below each reference ticket. A separate folding test covers repeated values, zero contributions, signed clipping boundaries and modulo-32-bit arithmetic.

## CPU threads during Metal job setup

When a job brings new weights, CPU workers build gradient rows and write the single-row lookup table into shared Metal memory. The foreground scanner uses the available CPU parallelism. The separate TESS look-ahead path uses two threads while Kepler mines.

The `cpu-feed` replay measures how much work Metal completes in the first five seconds after a weight change. Each run warms the old model, starts the clock, installs the new weights and scans with the slot's production pair kernel. Only batches that finish by the deadline contribute tickets. The CPU checks returned candidates and compares the new model's results with an independent oracle over 131,072 nonces before the replay prints a result.

On November 27, 2026, the 32-core M1 Max GPU with 64 GB of unified memory ran two recorded weight changes per model. Each cell gives setup time in milliseconds and millions of Metal tickets completed in the five-second window. Each thread count ran once per job. The second job used different weights and a different header seed. [Scanner revision](https://github.com/jpaljasma/neb-miner-metal/commit/884ed4a)

| Job | 1 thread (ms / M tickets) | 4 threads | 8 threads | 10 threads | Tickets, 10 vs. 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kepler A | 2,005 / 73.90 | 554 / 129.42 | 306 / 141.54 | 298 / 150.53 | 2.04× |
| Kepler B | 2,036 / 8.96 | 568 / 16.90 | 351 / 18.30 | 312 / 18.43 | 2.06× |
| TESS A | 1,081 / 839.56 | 377 / 1,018.68 | 194 / 1,084.68 | 220 / 1,103.76 | 1.31× |
| TESS B | 1,075 / 1,386.46 | 311 / 1,769.55 | 206 / 1,785.25 | 187 / 1,789.64 | 1.29× |

An earlier run on the same jobs with the direct-read Kepler builder also favored ten threads over one. The ticket ratios were 2.09–2.23× on Kepler and 1.25× on TESS. Eight and ten threads traded places on that revision. Both sets of measurements cover job startup. [Direct-read builder revision](https://github.com/jpaljasma/neb-miner-metal/commit/ee84724)

To repeat the comparison, use the release build, dataset and snapshot directories described below. Slot 0 is Kepler and slot 1 is TESS. Use the same seed and destination oracle for every thread count on a given job. Reverse or rotate the thread order on the next job.

```sh
bench=node/target/release/examples/metal_turnover
dataset=build/neb-datasets.bin
for slot in 0 1; do
    oracle="build/cpu-feed-oracle-$slot.bin"
    "$bench" oracle "$dataset" "$new" "$slot" "$oracle" || exit 1
    for threads in 1 4 8 10; do
        "$bench" cpu-feed "$dataset" "$old" "$new" "$slot" "$threads" 37 "$oracle" || exit 1
    done
done
```

Each successful run prints `cpu-feed,slot,threads,seed,setup_seconds,completed_tickets`. Divide the last field by 5,000,000 for MH/s across the five-second window.

## Reproduce the replay

Build the `metal_turnover` Cargo example in release mode with the `metal-bench` feature. Use the pinned mainnet dataset, two distinct same-slot model snapshots in the format produced by `tools/capture_metal_models.py`, and an otherwise idle GPU. Generate the independent CPU oracle for the destination model before timing any GPU arm.

```sh
cargo build --release --locked --manifest-path node/Cargo.toml -p neb-gpuminer --features metal-bench --example metal_turnover
bench=node/target/release/examples/metal_turnover
dataset=build/neb-datasets.bin
"$bench" support "$dataset" "$old" "$new"
for slot in 0 1; do
    oracle="build/turnover-oracle-$slot.bin"
    "$bench" oracle "$dataset" "$new" "$slot" "$oracle"
    for mask in 0 1 2 3 4 5 6 7; do
        "$bench" fixed "$dataset" "$old" "$new" "$slot" "$mask" 1 37 "$oracle"
        "$bench" window "$dataset" "$old" "$new" "$slot" "$mask" 1 71 "$oracle"
    done
done
```

Set `old` and `new` to your snapshot directories. The argument after the mask toggles the precomputed dense-output mask for B. The following integer seeds the replay headers. Repeat with different weights for holdout testing and rotate arm order when recording timings. The `builders`, `copy`, `extra` and `paired-short` modes use the same argument order for component and five-second confirmation runs.

## Reading Kepler factors directly

A follow-up crossed 256, 128 and 64 threads per pair with three ways to access the dense inputs: staging both rows in threadgroup memory, reading them directly from the existing Metal row buffer, and direct reads with the input loop outside the active-output loop. Each of 48 order-balanced blocks computed and byte-hashed the same 65,536 pair digests for every configuration. The holdout used different weights and different pair ranges.

Kepler's selected configuration uses direct reads and 64 threads per pair. It eliminates the factor-staging copy and its barrier. The only threadgroup allocation is 64 bytes for combining the two SIMD groups' partial digests. Against the previous 256-thread masked builder, builder throughput rose 58.20% on the initial snapshot and **55.54%** on the holdout (90% paired interval +53.32% to +58.21%). Direct reads at the original width gave +24.04% on the holdout. The narrower group with the original staging gave +17.75%.

The complete pair tables matched byte for byte. Kepler construction went from 1.421 s to 0.946 s on the adjacent transition and from **2.485 s to 1.640 s** on the holdout. Six paired five-second reverse-transition runs measured **39.47%** more completed tickets (+36.00% to +42.76%). These gains describe model setup and the first five seconds after a change. The prepared-table nonce scan uses the same kernel as before. TESS retains its previous builder.

We also tried combining rank checks in groups of eight or sixteen and using the already-ordered ranks to simplify triangular indexing. The best scan-only candidate gained 0.62% in the initial screen and lost 1.14% on the holdout. The production scan retains the four-sample early exit.

Use the `geometry` replay mode with slot 0, mask 2 and the masked builder to compare the nine pair kernels. IDs 0–2 use 256 threads, 3–5 use 128, and 6–8 use 64. Within each width, the order is staged, direct, then input-first. Use `direct-fixed` to verify the full table and `paired-direct` to compare first-five-second work in alternating order.
