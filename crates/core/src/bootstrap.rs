//! Small deterministic bootstrap sampler used by ensemble trainers.
//!
//! The sampler lives in its own module because the same sampling primitive is
//! reused by random forests and gradient boosting.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Full-size bootstrap sampler.
///
/// The sample size always matches the training row count. That keeps the
/// semantics close to standard random forests and also makes OOB accounting
/// straightforward.
#[derive(Debug, Clone)]
pub struct BootstrapSampler {
    sample_size: usize,
}

impl BootstrapSampler {
    pub(crate) fn new(sample_size: usize) -> Self {
        Self { sample_size }
    }

    pub(crate) fn sample(&self, seed: u64) -> Vec<usize> {
        self.sample_with_oob(seed).0
    }

    pub(crate) fn sample_with_oob(&self, seed: u64) -> (Vec<usize>, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut seen = vec![false; self.sample_size];
        let sample = (0..self.sample_size)
            .map(|_| {
                let row = rng.gen_range(0..self.sample_size);
                seen[row] = true;
                row
            })
            .collect();
        let oob_rows = seen
            .into_iter()
            .enumerate()
            .filter_map(|(row, was_seen)| (!was_seen).then_some(row))
            .collect();
        (sample, oob_rows)
    }
}

#[cfg(test)]
mod tests {
    use super::BootstrapSampler;

    #[test]
    fn bootstrap_sampler_is_deterministic_per_seed() {
        let sampler = BootstrapSampler::new(8);
        assert_eq!(sampler.sample_with_oob(7), sampler.sample_with_oob(7));
    }

    #[test]
    fn bootstrap_sampler_samples_full_size_with_replacement() {
        let sampler = BootstrapSampler::new(32);
        let (sample, _oob_rows) = sampler.sample_with_oob(3);

        assert_eq!(sample.len(), 32);
        assert!(sample.iter().all(|row| *row < 32));
        assert!(
            sample.windows(2).any(|pair| pair[0] == pair[1]) || {
                let unique = sample
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>();
                unique.len() < sample.len()
            }
        );
    }

    #[test]
    fn bootstrap_sampler_returns_oob_rows() {
        let sampler = BootstrapSampler::new(32);
        let (sample, oob_rows) = sampler.sample_with_oob(3);

        assert_eq!(sample.len(), 32);
        assert!(oob_rows.iter().all(|row| *row < 32));
        assert!(oob_rows.iter().all(|row| !sample.contains(row)));
    }
}
