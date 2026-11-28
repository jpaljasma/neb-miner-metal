//! The table engine of nebhash v2 (`spec/nebhash-v2.md` section 4).
//!
//! The gradient of an attempt is the sum of the gradients of its 16 samples, clipped afterwards, and the gradient of
//! a sample depends on the model and on that sample only. So for one model state the engine evaluates every sample of
//! the dataset once ([`Table::build`], with `neb2_sample_grad` of the C reference) and keeps the result; an attempt
//! then reads 16 rows, adds them, clips, and computes the digest and the ticket ([`Table::eval`]). Nothing of the
//! forward and backward passes is left in an attempt.
//!
//! A row holds, per layer in table order: for a convolution, every gradient entry (16 bits when the bound of one
//! sample fits, 32 bits otherwise); for a linear layer, its `cout` errors and its `cin` inputs, whose outer product
//! is the gradient of the layer. A row of KEP1 takes 3 714 bytes and a row of RET2 12 386 bytes, so the tables of the
//! two datasets of the network take about 50 MB and 640 MB.
//!
//! The table follows the model: it is built again for every model state, which costs one sample evaluation per
//! sample of the dataset, spread over the threads of the machine.

use neb_pow::shares::{Arch, CLIP, Dataset, Layer, Model, NB, sample_grad, share_indices};

/// How the entries of one layer are kept in a row.
#[derive(Clone, Copy, Debug)]
enum Kind {
    Wide,
    Narrow,
    Linear,
}

#[derive(Clone, Copy, Debug)]
struct Part {
    kind: Kind,
    layer: Layer,
    /// Offset of the layer in a row.
    at: usize,
}

/// Bytes of the layer in a row.
fn part_bytes(kind: Kind, l: &Layer) -> usize {
    match kind {
        Kind::Wide => 4 * l.weights,
        Kind::Narrow => 2 * l.weights,
        Kind::Linear => l.cout + l.cin,
    }
}

fn parts(arch: Arch) -> (Vec<Part>, usize) {
    let mut at = 0;
    let parts = arch
        .layers()
        .into_iter()
        .map(|layer| {
            // one sample adds at most `positions` products of an error (at most 8) and an input (at most 8)
            let kind = match (layer.linear, layer.positions * 64 <= i16::MAX as usize) {
                (true, _) => Kind::Linear,
                (false, true) => Kind::Narrow,
                (false, false) => Kind::Wide,
            };
            let part = Part { kind, layer, at };
            at += part_bytes(kind, &layer);
            part
        })
        .collect();
    (parts, at)
}

/// The per-sample gradients of one model state on one dataset.
pub struct Table<S = Vec<u8>> {
    arch: Arch,
    n: u32,
    parts: Vec<Part>,
    row: usize,
    rows: S,
    /// Whether the gradient of the sample has a non-zero entry: a sample that the model already answers exactly adds
    /// nothing to an attempt.
    live: Vec<bool>,
}

/// The result of an attempt read from a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableEval {
    pub digest: [u8; 32],
    pub ticket: [u8; 32],
    pub indices: [u32; NB],
}

impl Table {
    /// Bytes of the rows of a table of `arch` over `n` samples.
    pub fn bytes(arch: Arch, n: u32) -> usize {
        parts(arch).1 * n as usize
    }

    /// Evaluates every sample of `dataset` on `model` with `threads` threads. `spare` is a buffer to reuse (the rows
    /// of a table that is no longer needed), or an empty vector. Fails on a v1 model, on a dataset of another
    /// architecture and when a value does not fit its field, which the bounds of the arithmetic exclude.
    pub fn build(model: &Model, dataset: &dyn Dataset, threads: usize, mut spare: Vec<u8>) -> Result<Table, String> {
        let arch = model.arch();
        if arch == Arch::V1 || dataset.arch() != arch {
            return Err("the table engine needs a v2 model and a dataset of its architecture".into());
        }
        let (parts, row) = parts(arch);
        let n = dataset.n();
        spare.clear();
        spare.resize(row * n as usize, 0);
        let live = fill_rows(model, dataset, &parts, row, threads, &mut spare, None)?;
        Ok(Table { arch, n, parts, row, rows: spare, live })
    }

    /// Build directly into caller-owned storage. Bytes following each logical row are padding.
    /// The returned view borrows the storage. When supplied, the n + 1 single-row
    /// records are filled by the same workers, before their gradient scratch is reused.
    pub fn build_into<'a>(
        model: &Model,
        dataset: &dyn Dataset,
        threads: usize,
        stride: usize,
        storage: &'a mut [u8],
        singles: Option<&mut [[u32; 9]]>,
    ) -> Result<Table<&'a [u8]>, String> {
        let arch = model.arch();
        if arch == Arch::V1 || dataset.arch() != arch {
            return Err("the table engine needs a v2 model and a dataset of its architecture".into());
        }
        let (parts, logical) = parts(arch);
        let n = dataset.n();
        if stride < logical || stride.checked_mul(n as usize) != Some(storage.len()) {
            return Err("invalid gradient row stride or destination length".into());
        }
        let live = fill_rows(model, dataset, &parts, stride, threads, storage, singles)?;
        Ok(Table { arch, n, parts, row: stride, rows: storage, live })
    }

    /// The rows, to build the next table in the same memory.
    pub fn into_spare(self) -> Vec<u8> {
        self.rows
    }
}

impl<S: AsRef<[u8]> + Sync> Table<S> {
    /// Transfer the live-row flags without copying them.
    pub fn into_live_rows(self) -> Vec<bool> {
        self.live
    }

    pub fn arch(&self) -> Arch {
        self.arch
    }

    /// Samples whose gradient is zero on this model.
    pub fn idle_rows(&self) -> usize {
        self.live.iter().filter(|&&l| !l).count()
    }

    /// True for each dataset row with a nonzero gradient at the prepared weights.
    pub fn live_rows(&self) -> &[bool] {
        &self.live
    }

    /// The attempt `key`: its digest, its ticket and its sample indices. `acc` is the scratch of the caller; after
    /// the call its first `arch.weights()` entries are the clipped gradient of the attempt.
    pub fn eval(&self, key: &[u8; 32], acc: &mut Vec<i32>) -> TableEval {
        let indices = share_indices(key, self.n);
        let digest = self.digest_for_rows(&indices, acc);
        let mut input = [0u8; 64];
        input[..32].copy_from_slice(&digest);
        input[32..].copy_from_slice(key);
        TableEval { digest, ticket: *blake3::hash(&input).as_bytes(), indices }
    }

    /// Exact digest lookup for attempts with zero or one live sample. Entry
    /// zero is the empty digest; entry i+1 holds a live flag and sample i's digest.
    pub fn single_row_lookup(&self, threads: usize) -> Vec<[u32; 9]> {
        let mut out = vec![[0u32; 9]; self.n as usize + 1];
        self.fill_zeroed_single_row_lookup(threads, &mut out);
        out
    }

    /// Fill a zero-initialized destination of n + 1 records. Idle rows keep their zero records.
    pub fn fill_zeroed_single_row_lookup(&self, threads: usize, out: &mut [[u32; 9]]) {
        assert_eq!(out.len(), self.n as usize + 1);
        let words = |d: [u8; 32]| blake3::platform::words_from_le_bytes_32(&d);
        out[0][1..].copy_from_slice(&words(self.digest_for_rows(&[], &mut Vec::new())));
        let chunk = (self.n as usize).div_ceil(threads.max(1)).max(1);
        std::thread::scope(|scope| {
            for (part, entries) in out[1..].chunks_mut(chunk).enumerate() {
                let words = &words;
                scope.spawn(move || {
                    let mut acc = Vec::new();
                    for (j, entry) in entries.iter_mut().enumerate() {
                        let i = part * chunk + j;
                        if self.live[i] {
                            entry[0] = 1;
                            entry[1..].copy_from_slice(&words(self.digest_for_rows(&[i as u32], &mut acc)));
                        }
                    }
                });
            }
        });
    }

    fn digest_for_rows(&self, indices: &[u32], acc: &mut Vec<i32>) -> [u8; 32] {
        let weights = self.arch.weights();
        if acc.len() < weights {
            acc.resize(weights, 0);
        }
        let acc = &mut acc[..weights];
        acc.fill(0);
        for &i in indices {
            if !self.live[i as usize] {
                continue;
            }
            let row = &self.rows.as_ref()[i as usize * self.row..][..self.row];
            for p in &self.parts {
                let out = &mut acc[p.layer.offset..][..p.layer.weights];
                let bytes = &row[p.at..][..part_bytes(p.kind, &p.layer)];
                match p.kind {
                    Kind::Wide => {
                        for (o, b) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                            *o += i32::from_le_bytes(*b);
                        }
                    }
                    Kind::Narrow => {
                        for (o, b) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
                            *o += i32::from(i16::from_le_bytes(*b));
                        }
                    }
                    Kind::Linear => {
                        let (d, input) = bytes.split_at(p.layer.cout);
                        for (out, &d) in out.chunks_exact_mut(p.layer.cin).zip(d) {
                            let d = i32::from(d as i8);
                            if d == 0 {
                                continue;
                            }
                            for (o, &x) in out.iter_mut().zip(input) {
                                *o += d * i32::from(x as i8);
                            }
                        }
                    }
                }
            }
        }
        let lanes = clip_digest(&self.parts, acc);
        let mut digest = [0u8; 32];
        for (k, lane) in lanes.iter().enumerate() {
            digest[4 * k..4 * k + 4].copy_from_slice(&lane.to_le_bytes());
        }
        digest
    }
}

fn clip_digest(parts: &[Part], acc: &mut [i32]) -> [u32; 8] {
    // clip, then the digest of spec/shares.md section 5.3 over the layers l = 1 .. L
    let mut lanes = [0u32; 8];
    for (l, p) in parts.iter().enumerate() {
        let base = 7919 * (l as u32 + 1);
        let layer = &mut acc[p.layer.offset..][..p.layer.weights];
        for (j, chunk) in layer.chunks_mut(8).enumerate() {
            for (k, v) in chunk.iter_mut().enumerate() {
                *v = (*v).clamp(-CLIP, CLIP);
                lanes[k] = lanes[k].wrapping_add((*v as u32) ^ (base + (8 * j + k) as u32));
            }
        }
    }
    lanes
}

fn fill_rows(
    model: &Model,
    dataset: &dyn Dataset,
    parts: &[Part],
    row: usize,
    threads: usize,
    storage: &mut [u8],
    singles: Option<&mut [[u32; 9]]>,
) -> Result<Vec<bool>, String> {
    let n = dataset.n();
    let mut live = vec![false; n as usize];
    let chunk = (n as usize).div_ceil(threads.max(1)).max(1);
    let mut output = if let Some(entries) = singles {
        if entries.len() != n as usize + 1 {
            return Err("invalid single-row lookup length".into());
        }
        entries[0][0] = 0;
        // With a zero gradient, each lane sums its arithmetic progression of salts.
        for (k, lane) in entries[0][1..].iter_mut().enumerate() {
            *lane = parts
                .iter()
                .enumerate()
                .map(|(l, p)| {
                    let terms = p.layer.weights.saturating_sub(k).div_ceil(8) as u64;
                    (terms * (7919 * (l + 1) + k) as u64 + 4 * terms * terms.saturating_sub(1)) as u32
                })
                .fold(0u32, u32::wrapping_add);
        }
        Some(entries[1..].chunks_mut(chunk))
    } else {
        None
    };
    let result: Result<(), String> = std::thread::scope(|scope| {
        let handles: Vec<_> = storage
            .chunks_mut(row * chunk)
            .zip(live.chunks_mut(chunk))
            .enumerate()
            .map(|(c, (rows, live))| {
                let singles = output.as_mut().map(|chunks| chunks.next().unwrap());
                scope.spawn(move || fill(model, dataset, parts, row, (c * chunk) as u32, rows, (live, singles)))
            })
            .collect();
        handles.into_iter().try_for_each(|h| h.join().map_err(|_| "a table thread panicked".to_string())?)
    });
    result?;
    Ok(live)
}

/// Fills the rows of the samples `first ..`, one per entry of `live`.
fn fill(
    model: &Model,
    dataset: &dyn Dataset,
    parts: &[Part],
    row: usize,
    first: u32,
    rows: &mut [u8],
    (live, mut singles): (&mut [bool], Option<&mut [[u32; 9]]>),
) -> Result<(), String> {
    let arch = model.arch();
    let mut x = vec![0u8; arch.nin()];
    let mut g = vec![0i32; arch.weights()];
    let mut fac = vec![0i8; arch.factor_bytes()];
    for (i, (out, live)) in rows.chunks_exact_mut(row).zip(live.iter_mut()).enumerate() {
        let label = dataset.sample(first + i as u32, &mut x);
        sample_grad(model, &x, label, &mut g, Some(&mut fac));
        *live = g.iter().any(|&v| v != 0);
        let mut f = 0;
        for p in parts {
            let values = &g[p.layer.offset..][..p.layer.weights];
            let bytes = &mut out[p.at..][..part_bytes(p.kind, &p.layer)];
            match p.kind {
                Kind::Wide => {
                    for (b, v) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                        *b = v.to_le_bytes();
                    }
                }
                Kind::Narrow => {
                    for (b, &v) in bytes.as_chunks_mut::<2>().0.iter_mut().zip(values) {
                        let v = i16::try_from(v).map_err(|_| "a gradient entry does not fit 16 bits".to_string())?;
                        *b = v.to_le_bytes();
                    }
                }
                Kind::Linear => {
                    let len = p.layer.cout + p.layer.cin;
                    for (b, &v) in bytes.iter_mut().zip(&fac[f..f + len]) {
                        *b = v as u8;
                    }
                    f += len;
                }
            }
        }
        if let Some(entries) = singles.as_deref_mut() {
            entries[i] = [0; 9];
            if *live {
                entries[i][0] = 1;
                entries[i][1..].copy_from_slice(&clip_digest(parts, &mut g));
            }
        }
    }
    Ok(())
}
