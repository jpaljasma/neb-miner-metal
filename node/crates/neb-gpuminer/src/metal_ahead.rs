//! Speculative work never becomes active until the actual job's weights match.
use super::MetalScanner;
use crate::Scanner;
use neb_pow::shares::{Arch, Model, SharedDataset};
use std::{
    io::Read,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

#[derive(Clone)]
pub struct TessAhead {
    dataset: SharedDataset,
    ready: Arc<Mutex<Option<Box<MetalScanner>>>>,
    epoch: Arc<AtomicU64>,
}

impl TessAhead {
    pub fn new(dataset: SharedDataset) -> Self {
        Self { dataset, ready: Arc::default(), epoch: Arc::default() }
    }

    pub fn prepare(&self, digest: &[u8; 32], packed: &[u8]) -> Result<(), String> {
        let epoch = self.epoch.load(Ordering::Acquire);
        let model = Model::from_packed(Arch::Kep1, packed)?;
        if model.weights_hash() != *digest {
            return Err("TESS hint weights hash mismatch".into());
        }
        {
            let ready = self.ready.lock().unwrap();
            if ready.as_ref().and_then(|s| s.rows.as_ref()).is_some_and(|r| r.weights == model.weights()) {
                return Ok(());
            }
        }
        let mut fresh = Box::new(MetalScanner::open(self.dataset.clone())?);
        fresh.build_threads = Some(2);
        fresh.enable_screen();
        fresh.set_weights(&model)?;
        fresh.build_threads = None;
        {
            let mut ready = self.ready.lock().unwrap();
            if self.epoch.load(Ordering::Acquire) != epoch {
                return Ok(());
            }
            *ready = Some(fresh);
        }
        // Leave the verified scanner available between small GPU slices.
        loop {
            let delay = {
                let mut ready = self.ready.lock().unwrap();
                if self.epoch.load(Ordering::Acquire) != epoch {
                    return Ok(());
                }
                let Some(s) = ready.as_mut() else { return Ok(()) };
                if !s.use_screen {
                    return Ok(());
                }
                let r = s.rows.as_mut().unwrap();
                let c = r.combos.as_mut().unwrap();
                if c.args[0] == c.args[2] {
                    return Ok(());
                }
                c.advance(&s.queue, &r.buffer)?;
                // Give active Kepler work most of the GPU time.
                c.slice_delay.max(Duration::from_millis(4))
            };
            std::thread::sleep(delay);
        }
    }

    pub(super) fn take(&self, model: &Model) -> Option<Box<MetalScanner>> {
        let mut ready = self.ready.lock().unwrap();
        self.epoch.fetch_add(1, Ordering::AcqRel);
        ready.take().filter(|s| s.rows.as_ref().is_some_and(|r| r.weights == model.weights()))
    }

    /// Private adapter pipe: 32-byte expected hash then 13,000 packed KEP1 bytes.
    pub fn listen(self) {
        std::thread::spawn(move || {
            let mut pipe = std::io::stdin().lock();
            let mut record = [0u8; 32 + 13000];
            while pipe.read_exact(&mut record).is_ok() {
                if let Err(e) = self.prepare(record[..32].try_into().unwrap(), &record[32..]) {
                    eprintln!("ignored early TESS model: {e}");
                }
            }
        });
    }
}
