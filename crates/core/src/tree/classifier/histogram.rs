use super::*;

#[derive(Debug, Clone)]
pub(super) enum ClassificationFeatureHistogram {
    Binary {
        false_counts: Vec<usize>,
        true_counts: Vec<usize>,
        false_size: usize,
        true_size: usize,
    },
    Numeric {
        bin_class_counts: Vec<Vec<usize>>,
        observed_bins: Vec<usize>,
    },
}

pub(super) fn build_classification_node_histograms(
    table: &dyn TableAccess,
    class_indices: &[usize],
    rows: &[usize],
    num_classes: usize,
) -> Vec<ClassificationFeatureHistogram> {
    (0..table.binned_feature_count())
        .map(|feature_index| {
            if table.is_binary_binned_feature(feature_index) {
                let mut false_counts = vec![0usize; num_classes];
                let mut true_counts = vec![0usize; num_classes];
                let mut false_size = 0usize;
                let mut true_size = 0usize;
                for row_idx in rows {
                    let class_index = class_indices[*row_idx];
                    if !table
                        .binned_boolean_value(feature_index, *row_idx)
                        .expect("binary feature must expose boolean values")
                    {
                        false_counts[class_index] += 1;
                        false_size += 1;
                    } else {
                        true_counts[class_index] += 1;
                        true_size += 1;
                    }
                }
                ClassificationFeatureHistogram::Binary {
                    false_counts,
                    true_counts,
                    false_size,
                    true_size,
                }
            } else {
                let bin_cap = table.numeric_bin_cap();
                let mut bin_class_counts = vec![vec![0usize; num_classes]; bin_cap];
                let mut observed_bins = vec![false; bin_cap];
                for row_idx in rows {
                    let bin = table.binned_value(feature_index, *row_idx) as usize;
                    bin_class_counts[bin][class_indices[*row_idx]] += 1;
                    observed_bins[bin] = true;
                }
                ClassificationFeatureHistogram::Numeric {
                    bin_class_counts,
                    observed_bins: observed_bins
                        .into_iter()
                        .enumerate()
                        .filter_map(|(bin, seen)| seen.then_some(bin))
                        .collect(),
                }
            }
        })
        .collect()
}

pub(super) fn subtract_classification_node_histograms(
    parent: &[ClassificationFeatureHistogram],
    child: &[ClassificationFeatureHistogram],
) -> Vec<ClassificationFeatureHistogram> {
    parent
        .iter()
        .zip(child.iter())
        .map(
            |(parent_hist, child_hist)| match (parent_hist, child_hist) {
                (
                    ClassificationFeatureHistogram::Binary {
                        false_counts: parent_false_counts,
                        true_counts: parent_true_counts,
                        false_size: parent_false_size,
                        true_size: parent_true_size,
                    },
                    ClassificationFeatureHistogram::Binary {
                        false_counts: child_false_counts,
                        true_counts: child_true_counts,
                        false_size: child_false_size,
                        true_size: child_true_size,
                    },
                ) => ClassificationFeatureHistogram::Binary {
                    false_counts: parent_false_counts
                        .iter()
                        .zip(child_false_counts.iter())
                        .map(|(parent, child)| parent - child)
                        .collect(),
                    true_counts: parent_true_counts
                        .iter()
                        .zip(child_true_counts.iter())
                        .map(|(parent, child)| parent - child)
                        .collect(),
                    false_size: parent_false_size - child_false_size,
                    true_size: parent_true_size - child_true_size,
                },
                (
                    ClassificationFeatureHistogram::Numeric {
                        bin_class_counts: parent_bin_class_counts,
                        ..
                    },
                    ClassificationFeatureHistogram::Numeric {
                        bin_class_counts: child_bin_class_counts,
                        ..
                    },
                ) => {
                    let bin_class_counts = parent_bin_class_counts
                        .iter()
                        .zip(child_bin_class_counts.iter())
                        .map(|(parent_counts, child_counts)| {
                            parent_counts
                                .iter()
                                .zip(child_counts.iter())
                                .map(|(parent, child)| parent - child)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let observed_bins = bin_class_counts
                        .iter()
                        .enumerate()
                        .filter_map(|(bin, counts)| {
                            counts.iter().any(|count| *count > 0).then_some(bin)
                        })
                        .collect::<Vec<_>>();
                    ClassificationFeatureHistogram::Numeric {
                        bin_class_counts,
                        observed_bins,
                    }
                }
                _ => unreachable!("histogram shapes must match"),
            },
        )
        .collect()
}
