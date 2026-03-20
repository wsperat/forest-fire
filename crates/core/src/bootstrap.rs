use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct BootstrapSampler {
    sample_size: usize,
}

impl BootstrapSampler {
    pub(crate) fn new(sample_size: usize) -> Self {
        Self { sample_size }
    }

    pub(crate) fn sample(&self, seed: u64) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..self.sample_size)
            .map(|_| rng.gen_range(0..self.sample_size))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::BootstrapSampler;

    #[test]
    fn bootstrap_sampler_is_deterministic_per_seed() {
        let sampler = BootstrapSampler::new(8);
        assert_eq!(sampler.sample(7), sampler.sample(7));
    }

    #[test]
    fn bootstrap_sampler_samples_full_size_with_replacement() {
        let sampler = BootstrapSampler::new(32);
        let sample = sampler.sample(3);

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
}
