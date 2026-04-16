use crate::{CanaryFilter, sampling::sample_feature_subset};
use forestfire_data::{BINARY_MISSING_BIN, TableAccess};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_MULTIPLIER_A: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_MULTIPLIER_B: u64 = 0x94D0_49BB_1331_11EB;

#[derive(Debug, Clone)]
pub(crate) enum FeatureHistogram<T> {
    Binary {
        false_bin: T,
        true_bin: T,
        missing_bin: T,
    },
    Numeric {
        bins: Vec<T>,
        observed_bins: Vec<usize>,
    },
}

pub(crate) trait HistogramBin: Clone {
    fn subtract(parent: &Self, child: &Self) -> Self;
    fn is_observed(&self) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MissingBranchDirection {
    Left,
    Right,
    Node,
}

#[inline]
pub(crate) fn numeric_missing_bin(table: &dyn TableAccess) -> u16 {
    table.numeric_bin_cap() as u16
}

#[allow(dead_code)]
#[inline]
pub(crate) fn missing_bin_for_feature(table: &dyn TableAccess, feature_index: usize) -> u16 {
    if table.is_binary_binned_feature(feature_index) {
        BINARY_MISSING_BIN
    } else {
        numeric_missing_bin(table)
    }
}

fn build_feature_histogram_for_feature<T, MakeBin, AddRow>(
    table: &dyn TableAccess,
    rows: &[usize],
    feature_index: usize,
    make_bin: &MakeBin,
    add_row: &AddRow,
) -> FeatureHistogram<T>
where
    T: HistogramBin,
    MakeBin: Fn(usize) -> T,
    AddRow: Fn(usize, &mut T, usize),
{
    if table.is_binary_binned_feature(feature_index) {
        let mut false_bin = make_bin(feature_index);
        let mut true_bin = make_bin(feature_index);
        let mut missing_bin = make_bin(feature_index);
        for &row_idx in rows {
            match table.binned_boolean_value(feature_index, row_idx) {
                Some(false) => add_row(feature_index, &mut false_bin, row_idx),
                Some(true) => add_row(feature_index, &mut true_bin, row_idx),
                None => add_row(feature_index, &mut missing_bin, row_idx),
            }
        }
        FeatureHistogram::Binary {
            false_bin,
            true_bin,
            missing_bin,
        }
    } else {
        let bin_cap = table.numeric_bin_cap() + 1;
        let mut bins = vec![make_bin(feature_index); bin_cap];
        for &row_idx in rows {
            let bin = table.binned_value(feature_index, row_idx) as usize;
            add_row(feature_index, &mut bins[bin], row_idx);
        }
        let observed_bins = bins
            .iter()
            .enumerate()
            .filter_map(|(bin, payload)| payload.is_observed().then_some(bin))
            .collect();
        FeatureHistogram::Numeric {
            bins,
            observed_bins,
        }
    }
}

pub(crate) fn build_feature_histograms<T, MakeBin, AddRow>(
    table: &dyn TableAccess,
    rows: &[usize],
    make_bin: MakeBin,
    add_row: AddRow,
) -> Vec<FeatureHistogram<T>>
where
    T: HistogramBin,
    MakeBin: Fn(usize) -> T,
    AddRow: Fn(usize, &mut T, usize),
{
    (0..table.binned_feature_count())
        .map(|feature_index| {
            build_feature_histogram_for_feature(table, rows, feature_index, &make_bin, &add_row)
        })
        .collect()
}

pub(crate) fn build_feature_histograms_parallel<T, MakeBin, AddRow>(
    table: &dyn TableAccess,
    rows: &[usize],
    make_bin: MakeBin,
    add_row: AddRow,
) -> Vec<FeatureHistogram<T>>
where
    T: HistogramBin + Send,
    MakeBin: Fn(usize) -> T + Sync,
    AddRow: Fn(usize, &mut T, usize) + Sync,
{
    (0..table.binned_feature_count())
        .into_par_iter()
        .map(|feature_index| {
            build_feature_histogram_for_feature(table, rows, feature_index, &make_bin, &add_row)
        })
        .collect()
}

pub(crate) fn subtract_feature_histograms<T: HistogramBin>(
    parent: &[FeatureHistogram<T>],
    child: &[FeatureHistogram<T>],
) -> Vec<FeatureHistogram<T>> {
    parent
        .iter()
        .zip(child.iter())
        .map(
            |(parent_hist, child_hist)| match (parent_hist, child_hist) {
                (
                    FeatureHistogram::Binary {
                        false_bin: parent_false,
                        true_bin: parent_true,
                        missing_bin: parent_missing,
                    },
                    FeatureHistogram::Binary {
                        false_bin: child_false,
                        true_bin: child_true,
                        missing_bin: child_missing,
                    },
                ) => FeatureHistogram::Binary {
                    false_bin: T::subtract(parent_false, child_false),
                    true_bin: T::subtract(parent_true, child_true),
                    missing_bin: T::subtract(parent_missing, child_missing),
                },
                (
                    FeatureHistogram::Numeric {
                        bins: parent_bins, ..
                    },
                    FeatureHistogram::Numeric {
                        bins: child_bins, ..
                    },
                ) => {
                    let bins = parent_bins
                        .iter()
                        .zip(child_bins.iter())
                        .map(|(parent_bin, child_bin)| T::subtract(parent_bin, child_bin))
                        .collect::<Vec<_>>();
                    let observed_bins = bins
                        .iter()
                        .enumerate()
                        .filter_map(|(bin, payload)| payload.is_observed().then_some(bin))
                        .collect::<Vec<_>>();
                    FeatureHistogram::Numeric {
                        bins,
                        observed_bins,
                    }
                }
                _ => unreachable!("histogram shapes must match"),
            },
        )
        .collect()
}

pub(crate) fn choose_random_threshold(
    candidate_thresholds: &[u16],
    feature_index: usize,
    rows: &[usize],
    salt: u64,
) -> Option<u16> {
    if candidate_thresholds.is_empty() {
        return None;
    }

    let seed = avalanche64(
        salt ^ mix_seed(feature_index as u64, candidate_thresholds.len() as u64)
            ^ fingerprint_rows(rows)
            ^ fingerprint_thresholds(candidate_thresholds),
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let selected = rng.gen_range(0..candidate_thresholds.len());
    candidate_thresholds.get(selected).copied()
}

pub(crate) fn partition_rows_for_binary_split(
    table: &dyn TableAccess,
    feature_index: usize,
    threshold_bin: u16,
    missing_direction: MissingBranchDirection,
    rows: &mut [usize],
) -> usize {
    let mut left = 0usize;
    for index in 0..rows.len() {
        let go_left = if table.is_missing(feature_index, rows[index]) {
            matches!(missing_direction, MissingBranchDirection::Left)
        } else if table.is_binary_binned_feature(feature_index) {
            !table
                .binned_boolean_value(feature_index, rows[index])
                .expect("observed binary feature must expose boolean values")
        } else {
            table.binned_value(feature_index, rows[index]) <= threshold_bin
        };
        if go_left {
            rows.swap(left, index);
            left += 1;
        }
    }
    left
}

pub(crate) fn candidate_feature_indices(
    table: &dyn TableAccess,
    max_features: Option<usize>,
    seed: u64,
) -> Vec<usize> {
    let real_feature_count = table.n_features();
    let sampled_real_features = match max_features {
        Some(count) => sample_feature_subset(real_feature_count, count, seed),
        None => (0..real_feature_count).collect(),
    };
    if table.canaries() == 0 {
        return sampled_real_features;
    }

    let sampled_real_features = sampled_real_features
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    (0..table.binned_feature_count())
        .filter(
            |&feature_index| match table.binned_column_kind(feature_index) {
                forestfire_data::BinnedColumnKind::Real { source_index }
                | forestfire_data::BinnedColumnKind::Canary { source_index, .. } => {
                    sampled_real_features.contains(&source_index)
                }
            },
        )
        .collect()
}

pub(crate) fn oblivious_max_depth_limit(
    requested_max_depth: usize,
    row_count: usize,
    min_samples_leaf: usize,
) -> usize {
    if requested_max_depth == 0 || row_count == 0 {
        return 0;
    }
    let leaf_capacity = (row_count / min_samples_leaf.max(1)).max(1);
    let structural_limit = usize::BITS as usize - 1 - leaf_capacity.leading_zeros() as usize;
    requested_max_depth.min(structural_limit.max(1))
}

pub(crate) struct CanarySelection<T> {
    pub(crate) selected: Option<T>,
    pub(crate) blocked_by_canary: bool,
}

pub(crate) fn select_best_non_canary_candidate<T, Score, FeatureIndex>(
    table: &dyn TableAccess,
    candidates: Vec<T>,
    canary_filter: CanaryFilter,
    score: Score,
    feature_index: FeatureIndex,
) -> CanarySelection<T>
where
    Score: Fn(&T) -> f64,
    FeatureIndex: Fn(&T) -> usize,
{
    let mut ranked = candidates;
    ranked.sort_by(|left, right| score(right).total_cmp(&score(left)));
    let selection_size = canary_filter.selection_size(ranked.len());
    let mut blocked_by_canary = false;
    for candidate in ranked.into_iter().take(selection_size) {
        if table.is_canary_binned_feature(feature_index(&candidate)) {
            blocked_by_canary = true;
            continue;
        }
        return CanarySelection {
            selected: Some(candidate),
            blocked_by_canary: false,
        };
    }
    CanarySelection {
        selected: None,
        blocked_by_canary,
    }
}

pub(crate) fn mix_seed(base_seed: u64, value: u64) -> u64 {
    base_seed ^ value.wrapping_mul(GOLDEN_GAMMA).rotate_left(17)
}

pub(crate) fn node_seed(base_seed: u64, depth: usize, rows: &[usize], salt: u64) -> u64 {
    avalanche64(mix_seed(base_seed ^ salt, depth as u64) ^ fingerprint_rows(rows))
}

fn avalanche64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(MIX_MULTIPLIER_A);
    value ^= value >> 27;
    value = value.wrapping_mul(MIX_MULTIPLIER_B);
    value ^ (value >> 31)
}

fn fingerprint_rows(rows: &[usize]) -> u64 {
    let mut xor = 0u64;
    let mut sum = 0u64;
    let mut rotated_sum = 0u64;

    for &row in rows {
        let mixed = avalanche64((row as u64).wrapping_add(GOLDEN_GAMMA));
        xor ^= mixed;
        sum = sum.wrapping_add(mixed);
        rotated_sum = rotated_sum.wrapping_add(mixed.rotate_left((row as u32) & 63));
    }

    avalanche64(
        xor ^ sum.rotate_left(7)
            ^ rotated_sum.rotate_left(19)
            ^ (rows.len() as u64).wrapping_mul(MIX_MULTIPLIER_A),
    )
}

fn fingerprint_thresholds(candidate_thresholds: &[u16]) -> u64 {
    let mut xor = 0u64;
    let mut sum = 0u64;

    for (index, threshold) in candidate_thresholds.iter().copied().enumerate() {
        let mixed = avalanche64((threshold as u64).wrapping_add((index as u64) << 16));
        xor ^= mixed;
        sum = sum.wrapping_add(mixed.rotate_left((index as u32) & 63));
    }

    avalanche64(
        xor ^ sum.rotate_left(13)
            ^ (candidate_thresholds.len() as u64).wrapping_mul(MIX_MULTIPLIER_B),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use forestfire_data::{BinnedColumnKind, TableAccess};
    use std::collections::BTreeSet;

    struct SamplingTestTable {
        n_features: usize,
        canaries: usize,
    }

    impl TableAccess for SamplingTestTable {
        fn n_rows(&self) -> usize {
            0
        }

        fn n_features(&self) -> usize {
            self.n_features
        }

        fn canaries(&self) -> usize {
            self.canaries
        }

        fn numeric_bin_cap(&self) -> usize {
            0
        }

        fn binned_feature_count(&self) -> usize {
            self.n_features * (1 + self.canaries)
        }

        fn feature_value(&self, _feature_index: usize, _row_index: usize) -> f64 {
            unreachable!()
        }

        fn is_missing(&self, _feature_index: usize, _row_index: usize) -> bool {
            unreachable!()
        }

        fn is_binary_feature(&self, _index: usize) -> bool {
            unreachable!()
        }

        fn binned_value(&self, _feature_index: usize, _row_index: usize) -> u16 {
            unreachable!()
        }

        fn binned_boolean_value(&self, _feature_index: usize, _row_index: usize) -> Option<bool> {
            unreachable!()
        }

        fn binned_column_kind(&self, index: usize) -> BinnedColumnKind {
            if index < self.n_features {
                BinnedColumnKind::Real {
                    source_index: index,
                }
            } else {
                let canary_index = index - self.n_features;
                BinnedColumnKind::Canary {
                    source_index: canary_index % self.n_features,
                    copy_index: canary_index / self.n_features,
                }
            }
        }

        fn is_binary_binned_feature(&self, _index: usize) -> bool {
            unreachable!()
        }

        fn target_value(&self, _row_index: usize) -> f64 {
            unreachable!()
        }
    }

    #[test]
    fn mix_seed_is_deterministic_and_unique_across_many_values() {
        let mixed = (0..4096u64)
            .map(|value| mix_seed(17, value))
            .collect::<Vec<_>>();
        let unique = mixed.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(mixed.len(), unique.len());
        assert_eq!(mixed[123], mix_seed(17, 123));
    }

    #[test]
    fn node_seed_is_order_invariant_for_same_row_set() {
        let rows = vec![9usize, 3, 1, 7, 11, 5];
        let mut reversed = rows.clone();
        reversed.reverse();

        assert_eq!(node_seed(41, 2, &rows, 99), node_seed(41, 2, &reversed, 99));
    }

    #[test]
    fn node_seed_changes_when_depth_salt_or_row_membership_changes() {
        let rows = vec![1usize, 2, 3, 4];
        let with_extra = vec![1usize, 2, 3, 4, 5];
        let base = node_seed(41, 2, &rows, 99);

        assert_ne!(base, node_seed(41, 3, &rows, 99));
        assert_ne!(base, node_seed(41, 2, &rows, 100));
        assert_ne!(base, node_seed(41, 2, &with_extra, 99));
    }

    #[test]
    fn choose_random_threshold_is_order_invariant_for_same_row_set() {
        let thresholds = vec![1u16, 3, 7, 9, 11];
        let rows = vec![8usize, 2, 6, 4, 10];
        let mut permuted = rows.clone();
        permuted.rotate_left(2);

        assert_eq!(
            choose_random_threshold(&thresholds, 5, &rows, 1234),
            choose_random_threshold(&thresholds, 5, &permuted, 1234)
        );
    }

    #[test]
    fn feature_subset_sampling_stays_well_formed_under_many_seeds() {
        let feature_count = 32usize;
        let sample_size = 6usize;
        let mut counts = vec![0usize; feature_count];
        let table = SamplingTestTable {
            n_features: feature_count,
            canaries: 0,
        };

        for seed in 0..4096u64 {
            let sample = candidate_feature_indices(&table, Some(sample_size), seed);
            let unique = sample.iter().copied().collect::<BTreeSet<_>>();

            assert_eq!(sample.len(), sample_size);
            assert_eq!(unique.len(), sample_size);
            assert!(sample.iter().all(|feature| *feature < feature_count));

            for feature in sample {
                counts[feature] += 1;
            }
        }

        let expected = (4096 * sample_size / feature_count) as isize;
        let min = *counts.iter().min().unwrap() as isize;
        let max = *counts.iter().max().unwrap() as isize;

        assert!(
            min >= expected - 280,
            "min count {min} too far below {expected}"
        );
        assert!(
            max <= expected + 280,
            "max count {max} too far above {expected}"
        );
    }

    #[test]
    fn feature_subset_sampling_expands_sampled_reals_to_their_canaries() {
        let table = SamplingTestTable {
            n_features: 5,
            canaries: 2,
        };

        let sample = candidate_feature_indices(&table, Some(2), 7);
        let sampled_sources = sample
            .iter()
            .map(
                |&feature_index| match table.binned_column_kind(feature_index) {
                    BinnedColumnKind::Real { source_index }
                    | BinnedColumnKind::Canary { source_index, .. } => source_index,
                },
            )
            .collect::<Vec<_>>();
        let unique_sources = sampled_sources.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(unique_sources.len(), 2);
        assert_eq!(sample.len(), 2 * (1 + table.canaries));
        for source_index in unique_sources {
            assert!(sample.contains(&source_index));
            assert!(sample.contains(&(table.n_features + source_index)));
            assert!(sample.contains(&(table.n_features * 2 + source_index)));
        }
    }

    #[test]
    fn randomized_threshold_selection_covers_candidates_without_extreme_bias() {
        let thresholds = (0u16..8).collect::<Vec<_>>();
        let mut counts = vec![0usize; thresholds.len()];

        for context in 0..4096usize {
            let rows = (0..17usize)
                .map(|offset| (context * 37 + offset * 13) % 257)
                .collect::<Vec<_>>();
            let selected =
                choose_random_threshold(&thresholds, context % 11, &rows, 0xC1A5_5EED).unwrap();
            counts[selected as usize] += 1;
        }

        assert!(counts.iter().all(|count| *count > 300));
        assert!(counts.iter().all(|count| *count < 800));
    }

    #[test]
    fn top_fraction_selection_size_rounds_up() {
        assert_eq!(CanaryFilter::TopFraction(0.5).selection_size(3), 2);
        assert_eq!(CanaryFilter::TopFraction(0.05).selection_size(20), 1);
    }
}
