//! Small sampling helpers shared by multiple trainers.
//!
//! The functions here are intentionally tiny and deterministic so higher-level
//! trainers can compose them without dragging sampling policy into the tree
//! implementations themselves.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

pub(crate) fn sample_feature_subset(
    feature_count: usize,
    sample_size: usize,
    seed: u64,
) -> Vec<usize> {
    if sample_size >= feature_count {
        return (0..feature_count).collect();
    }

    let mut features: Vec<usize> = (0..feature_count).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    features.shuffle(&mut rng);
    features.truncate(sample_size);
    features
}

#[cfg(test)]
mod tests {
    use super::sample_feature_subset;

    #[test]
    fn feature_subset_sampling_is_deterministic() {
        assert_eq!(
            sample_feature_subset(8, 3, 11),
            sample_feature_subset(8, 3, 11)
        );
    }

    #[test]
    fn feature_subset_sampling_respects_requested_size_without_duplicates() {
        let sample = sample_feature_subset(10, 4, 5);
        let unique = sample
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();

        assert_eq!(sample.len(), 4);
        assert_eq!(unique.len(), 4);
        assert!(sample.iter().all(|feature| *feature < 10));
    }
}
