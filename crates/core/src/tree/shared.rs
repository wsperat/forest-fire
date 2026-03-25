use crate::sampling::sample_feature_subset;
use forestfire_data::TableAccess;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub(crate) enum FeatureHistogram<T> {
    Binary {
        false_bin: T,
        true_bin: T,
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

pub(crate) fn build_feature_histograms<T, MakeBin, AddRow>(
    table: &dyn TableAccess,
    rows: &[usize],
    mut make_bin: MakeBin,
    mut add_row: AddRow,
) -> Vec<FeatureHistogram<T>>
where
    T: HistogramBin,
    MakeBin: FnMut(usize) -> T,
    AddRow: FnMut(usize, &mut T, usize),
{
    (0..table.binned_feature_count())
        .map(|feature_index| {
            if table.is_binary_binned_feature(feature_index) {
                let mut false_bin = make_bin(feature_index);
                let mut true_bin = make_bin(feature_index);
                for &row_idx in rows {
                    if !table
                        .binned_boolean_value(feature_index, row_idx)
                        .expect("binary feature must expose boolean values")
                    {
                        add_row(feature_index, &mut false_bin, row_idx);
                    } else {
                        add_row(feature_index, &mut true_bin, row_idx);
                    }
                }
                FeatureHistogram::Binary {
                    false_bin,
                    true_bin,
                }
            } else {
                let bin_cap = table.numeric_bin_cap();
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
                    },
                    FeatureHistogram::Binary {
                        false_bin: child_false,
                        true_bin: child_true,
                    },
                ) => FeatureHistogram::Binary {
                    false_bin: T::subtract(parent_false, child_false),
                    true_bin: T::subtract(parent_true, child_true),
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

    let mut seed = salt ^ ((feature_index as u64) << 32) ^ (rows.len() as u64);
    for row_idx in rows {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add((*row_idx as u64) + 1);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let selected = rng.gen_range(0..candidate_thresholds.len());
    candidate_thresholds.get(selected).copied()
}

pub(crate) fn partition_rows_for_binary_split(
    table: &dyn TableAccess,
    feature_index: usize,
    threshold_bin: u16,
    rows: &mut [usize],
) -> usize {
    let mut left = 0usize;
    for index in 0..rows.len() {
        let go_left = if table.is_binary_binned_feature(feature_index) {
            !table
                .binned_boolean_value(feature_index, rows[index])
                .expect("binary feature must expose boolean values")
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
    feature_count: usize,
    max_features: Option<usize>,
    seed: u64,
) -> Vec<usize> {
    match max_features {
        Some(count) => sample_feature_subset(feature_count, count, seed),
        None => (0..feature_count).collect(),
    }
}

pub(crate) fn node_seed(base_seed: u64, depth: usize, rows: &[usize], salt: u64) -> u64 {
    rows.iter().fold(
        base_seed
            ^ salt
            ^ (depth as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .rotate_left(11),
        |seed, row_index| {
            seed.wrapping_mul(0xA076_1D64_78BD_642F)
                ^ (*row_index as u64).wrapping_add(0xE703_7ED1_A0B4_28DB)
        },
    )
}
