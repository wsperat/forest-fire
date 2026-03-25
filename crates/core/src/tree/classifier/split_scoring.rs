use super::histogram::ClassificationHistogramBin;
use super::*;

pub(super) struct SplitScoringContext<'a> {
    pub(super) table: &'a dyn TableAccess,
    pub(super) class_indices: &'a [usize],
    pub(super) num_classes: usize,
    pub(super) criterion: Criterion,
    pub(super) min_samples_leaf: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum MultiwayMetric {
    InformationGain,
    GainRatio,
}

pub(super) fn score_multiway_split_choice(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
    metric: MultiwayMetric,
) -> Option<MultiwaySplitChoice> {
    let grouped_counts = if context.table.is_binary_binned_feature(feature_index) {
        let mut false_counts = vec![0usize; context.num_classes];
        let mut true_counts = vec![0usize; context.num_classes];
        let mut false_size = 0usize;
        let mut true_size = 0usize;
        for row_idx in rows {
            let class_index = context.class_indices[*row_idx];
            if !context
                .table
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
        [
            (0u16, (false_size, false_counts)),
            (1u16, (true_size, true_counts)),
        ]
        .into_iter()
        .filter(|(_, (size, _))| *size > 0)
        .collect::<Vec<_>>()
    } else {
        let mut grouped = BTreeMap::<u16, (usize, Vec<usize>)>::new();
        for row_idx in rows {
            let bin = context.table.binned_value(feature_index, *row_idx);
            let entry = grouped
                .entry(bin)
                .or_insert_with(|| (0usize, vec![0usize; context.num_classes]));
            entry.0 += 1;
            entry.1[context.class_indices[*row_idx]] += 1;
        }
        grouped.into_iter().collect::<Vec<_>>()
    };

    if grouped_counts.len() <= 1
        || grouped_counts
            .iter()
            .any(|(_, (group_size, _))| *group_size < context.min_samples_leaf)
    {
        return None;
    }

    let parent_counts = class_counts(rows, context.class_indices, context.num_classes);
    let parent_impurity = classification_impurity(&parent_counts, rows.len(), context.criterion);
    let weighted_child_impurity = grouped_counts
        .iter()
        .map(|(_, (group_size, counts))| {
            (*group_size as f64 / rows.len() as f64)
                * classification_impurity(counts, *group_size, context.criterion)
        })
        .sum::<f64>();
    let information_gain = parent_impurity - weighted_child_impurity;

    let score = match metric {
        MultiwayMetric::InformationGain => information_gain,
        MultiwayMetric::GainRatio => {
            let split_info = grouped_counts
                .iter()
                .map(|(_, (group_size, _))| {
                    let probability = *group_size as f64 / rows.len() as f64;
                    -probability * probability.log2()
                })
                .sum::<f64>();
            if split_info == 0.0 {
                return None;
            }
            information_gain / split_info
        }
    };

    Some(MultiwaySplitChoice {
        feature_index,
        score,
        branch_bins: grouped_counts.into_iter().map(|(bin, _)| bin).collect(),
    })
}

pub(super) fn score_binary_split_choice_from_hist(
    context: &SplitScoringContext<'_>,
    histogram: &ClassificationFeatureHistogram,
    feature_index: usize,
    rows: &[usize],
    parent_counts: &[usize],
    algorithm: DecisionTreeAlgorithm,
) -> Option<BinarySplitChoice> {
    match (algorithm, histogram) {
        (
            DecisionTreeAlgorithm::Cart,
            ClassificationFeatureHistogram::Binary {
                false_bin,
                true_bin,
            },
        ) => score_binary_cart_split_choice_from_counts(
            context,
            feature_index,
            parent_counts,
            &false_bin.counts,
            false_bin.size(),
            &true_bin.counts,
            true_bin.size(),
        ),
        (
            DecisionTreeAlgorithm::Cart,
            ClassificationFeatureHistogram::Numeric {
                bins,
                observed_bins,
            },
        ) => score_numeric_cart_split_choice_from_hist(
            context,
            feature_index,
            parent_counts,
            rows.len(),
            bins,
            observed_bins,
        ),
        (
            DecisionTreeAlgorithm::Randomized,
            ClassificationFeatureHistogram::Binary {
                false_bin,
                true_bin,
            },
        ) => score_binary_cart_split_choice_from_counts(
            context,
            feature_index,
            parent_counts,
            &false_bin.counts,
            false_bin.size(),
            &true_bin.counts,
            true_bin.size(),
        ),
        (
            DecisionTreeAlgorithm::Randomized,
            ClassificationFeatureHistogram::Numeric { observed_bins, .. },
        ) => score_numeric_randomized_split_choice_from_hist(
            context,
            feature_index,
            rows,
            parent_counts,
            observed_bins,
            histogram,
        ),
        _ => None,
    }
}

fn score_binary_cart_split_choice_from_counts(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    parent_counts: &[usize],
    left_counts: &[usize],
    left_size: usize,
    right_counts: &[usize],
    right_size: usize,
) -> Option<BinarySplitChoice> {
    if left_size < context.min_samples_leaf || right_size < context.min_samples_leaf {
        return None;
    }
    let parent_impurity =
        classification_impurity(parent_counts, left_size + right_size, context.criterion);
    let weighted_impurity = (left_size as f64 / (left_size + right_size) as f64)
        * classification_impurity(left_counts, left_size, context.criterion)
        + (right_size as f64 / (left_size + right_size) as f64)
            * classification_impurity(right_counts, right_size, context.criterion);
    Some(BinarySplitChoice {
        feature_index,
        score: parent_impurity - weighted_impurity,
        threshold_bin: 0,
    })
}

fn score_numeric_cart_split_choice_from_hist(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    parent_counts: &[usize],
    row_count: usize,
    bin_class_counts: &[ClassificationHistogramBin],
    observed_bins: &[usize],
) -> Option<BinarySplitChoice> {
    if observed_bins.len() <= 1 {
        return None;
    }
    let parent_impurity = classification_impurity(parent_counts, row_count, context.criterion);
    let mut left_counts = vec![0usize; context.num_classes];
    let mut left_size = 0usize;
    let mut best_threshold = None;
    let mut best_score = f64::NEG_INFINITY;

    for &bin in observed_bins {
        for class_index in 0..context.num_classes {
            left_counts[class_index] += bin_class_counts[bin].counts[class_index];
        }
        left_size += bin_class_counts[bin].size();
        let right_size = row_count - left_size;
        if left_size < context.min_samples_leaf || right_size < context.min_samples_leaf {
            continue;
        }
        let right_counts = parent_counts
            .iter()
            .zip(left_counts.iter())
            .map(|(parent, left)| parent - left)
            .collect::<Vec<_>>();
        let weighted_impurity = (left_size as f64 / row_count as f64)
            * classification_impurity(&left_counts, left_size, context.criterion)
            + (right_size as f64 / row_count as f64)
                * classification_impurity(&right_counts, right_size, context.criterion);
        let score = parent_impurity - weighted_impurity;
        if score > best_score {
            best_score = score;
            best_threshold = Some(bin as u16);
        }
    }

    best_threshold.map(|threshold_bin| BinarySplitChoice {
        feature_index,
        score: best_score,
        threshold_bin,
    })
}

fn score_numeric_randomized_split_choice_from_hist(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
    parent_counts: &[usize],
    observed_bins: &[usize],
    histogram: &ClassificationFeatureHistogram,
) -> Option<BinarySplitChoice> {
    let candidate_thresholds = observed_bins
        .iter()
        .copied()
        .map(|bin| bin as u16)
        .collect::<Vec<_>>();
    let threshold_bin =
        choose_random_threshold(&candidate_thresholds, feature_index, rows, 0xC1A551F1u64)?;
    let ClassificationFeatureHistogram::Numeric { bins, .. } = histogram else {
        unreachable!("randomized numeric histogram must be numeric");
    };
    let mut left_counts = vec![0usize; context.num_classes];
    let mut left_size = 0usize;
    for bin in 0..=threshold_bin as usize {
        if bin >= bins.len() {
            break;
        }
        for class_index in 0..context.num_classes {
            left_counts[class_index] += bins[bin].counts[class_index];
        }
        left_size += bins[bin].size();
    }
    let row_count = rows.len();
    let right_size = row_count - left_size;
    if left_size < context.min_samples_leaf || right_size < context.min_samples_leaf {
        return None;
    }
    let right_counts = parent_counts
        .iter()
        .zip(left_counts.iter())
        .map(|(parent, left)| parent - left)
        .collect::<Vec<_>>();
    let parent_impurity = classification_impurity(parent_counts, row_count, context.criterion);
    let weighted_impurity = (left_size as f64 / row_count as f64)
        * classification_impurity(&left_counts, left_size, context.criterion)
        + (right_size as f64 / row_count as f64)
            * classification_impurity(&right_counts, right_size, context.criterion);
    Some(BinarySplitChoice {
        feature_index,
        score: parent_impurity - weighted_impurity,
        threshold_bin,
    })
}
