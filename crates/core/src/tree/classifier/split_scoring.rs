use super::histogram::ClassificationHistogramBin;
use super::*;
use crate::MissingValueStrategy;
use crate::tree::shared::{MissingBranchDirection, numeric_missing_bin};

pub(super) struct SplitScoringContext<'a> {
    pub(super) table: &'a dyn TableAccess,
    pub(super) class_indices: &'a [usize],
    pub(super) num_classes: usize,
    pub(super) criterion: Criterion,
    pub(super) min_samples_leaf: usize,
    pub(super) missing_value_strategies: &'a [MissingValueStrategy],
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
    let (grouped_counts, missing_counts, missing_size) =
        if context.table.is_binary_binned_feature(feature_index) {
            let mut false_counts = vec![0usize; context.num_classes];
            let mut true_counts = vec![0usize; context.num_classes];
            let mut missing_counts = vec![0usize; context.num_classes];
            let mut false_size = 0usize;
            let mut true_size = 0usize;
            let mut missing_size = 0usize;
            for row_idx in rows {
                let class_index = context.class_indices[*row_idx];
                match context.table.binned_boolean_value(feature_index, *row_idx) {
                    Some(false) => {
                        false_counts[class_index] += 1;
                        false_size += 1;
                    }
                    Some(true) => {
                        true_counts[class_index] += 1;
                        true_size += 1;
                    }
                    None => {
                        missing_counts[class_index] += 1;
                        missing_size += 1;
                    }
                }
            }
            (
                [
                    (0u16, (false_size, false_counts)),
                    (1u16, (true_size, true_counts)),
                ]
                .into_iter()
                .filter(|(_, (size, _))| *size > 0)
                .collect::<Vec<_>>(),
                missing_counts,
                missing_size,
            )
        } else {
            let mut grouped = BTreeMap::<u16, (usize, Vec<usize>)>::new();
            let mut missing_counts = vec![0usize; context.num_classes];
            let mut missing_size = 0usize;
            let missing_bin = numeric_missing_bin(context.table);
            for row_idx in rows {
                let bin = context.table.binned_value(feature_index, *row_idx);
                if bin == missing_bin {
                    missing_counts[context.class_indices[*row_idx]] += 1;
                    missing_size += 1;
                    continue;
                }
                let entry = grouped
                    .entry(bin)
                    .or_insert_with(|| (0usize, vec![0usize; context.num_classes]));
                entry.0 += 1;
                entry.1[context.class_indices[*row_idx]] += 1;
            }
            (
                grouped.into_iter().collect::<Vec<_>>(),
                missing_counts,
                missing_size,
            )
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
    let (score, missing_branch_bin) = if missing_size == 0 {
        (
            score_multiway_grouping(
                context,
                &grouped_counts,
                rows.len(),
                parent_impurity,
                metric,
            )?,
            None,
        )
    } else {
        grouped_counts
            .iter()
            .enumerate()
            .filter_map(|(missing_index, (bin, _))| {
                let augmented = grouped_counts
                    .iter()
                    .enumerate()
                    .map(|(index, (candidate_bin, (group_size, counts)))| {
                        if index == missing_index {
                            let merged_counts = counts
                                .iter()
                                .zip(missing_counts.iter())
                                .map(|(observed, missing)| observed + missing)
                                .collect::<Vec<_>>();
                            (*candidate_bin, (group_size + missing_size, merged_counts))
                        } else {
                            (*candidate_bin, (*group_size, counts.clone()))
                        }
                    })
                    .collect::<Vec<_>>();
                score_multiway_grouping(context, &augmented, rows.len(), parent_impurity, metric)
                    .map(|score| (score, Some(*bin)))
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))?
    };

    Some(MultiwaySplitChoice {
        feature_index,
        score,
        branch_bins: grouped_counts.into_iter().map(|(bin, _)| bin).collect(),
        missing_branch_bin,
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
                missing_bin,
            },
        ) => score_binary_cart_split_choice_from_counts(
            context,
            feature_index,
            parent_counts,
            &false_bin.counts,
            false_bin.size(),
            &true_bin.counts,
            true_bin.size(),
            &missing_bin.counts,
            missing_bin.size(),
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
                missing_bin,
            },
        ) => score_binary_cart_split_choice_from_counts(
            context,
            feature_index,
            parent_counts,
            &false_bin.counts,
            false_bin.size(),
            &true_bin.counts,
            true_bin.size(),
            &missing_bin.counts,
            missing_bin.size(),
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

#[allow(clippy::too_many_arguments)]
fn score_binary_cart_split_choice_from_counts(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    parent_counts: &[usize],
    left_counts: &[usize],
    left_size: usize,
    right_counts: &[usize],
    right_size: usize,
    missing_counts: &[usize],
    missing_size: usize,
) -> Option<BinarySplitChoice> {
    if matches!(
        context
            .missing_value_strategies
            .get(feature_index)
            .copied()
            .unwrap_or(MissingValueStrategy::Heuristic),
        MissingValueStrategy::Heuristic
    ) {
        return score_binary_observed_split_then_assign_missing(
            context,
            feature_index,
            0,
            left_counts.to_vec(),
            left_size,
            right_counts.to_vec(),
            right_size,
            missing_counts,
            missing_size,
        );
    }
    score_binary_missing_assignment(
        context,
        feature_index,
        0,
        parent_counts,
        left_counts.to_vec(),
        left_size,
        right_counts.to_vec(),
        right_size,
        missing_counts,
        missing_size,
    )
}

fn score_numeric_cart_split_choice_from_hist(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    parent_counts: &[usize],
    row_count: usize,
    bin_class_counts: &[ClassificationHistogramBin],
    observed_bins: &[usize],
) -> Option<BinarySplitChoice> {
    if matches!(
        context
            .missing_value_strategies
            .get(feature_index)
            .copied()
            .unwrap_or(MissingValueStrategy::Heuristic),
        MissingValueStrategy::Heuristic
    ) {
        return score_numeric_cart_split_choice_from_hist_heuristic(
            context,
            feature_index,
            parent_counts,
            row_count,
            bin_class_counts,
            observed_bins,
        );
    }
    let missing_bin = usize::from(numeric_missing_bin(context.table));
    let observed_bins = observed_bins
        .iter()
        .copied()
        .filter(|bin| *bin != missing_bin)
        .collect::<Vec<_>>();
    if observed_bins.len() <= 1 {
        return None;
    }
    let missing_counts = bin_class_counts
        .get(missing_bin)
        .map(|bin| bin.counts.clone())
        .unwrap_or_else(|| vec![0usize; context.num_classes]);
    let missing_size = bin_class_counts
        .get(missing_bin)
        .map_or(0usize, ClassificationHistogramBin::size);
    let mut left_counts = vec![0usize; context.num_classes];
    let mut left_size = 0usize;
    let mut best = None;

    for bin in observed_bins {
        for (left_count, bin_count) in left_counts
            .iter_mut()
            .zip(bin_class_counts[bin].counts.iter())
        {
            *left_count += *bin_count;
        }
        left_size += bin_class_counts[bin].size();
        let right_size = row_count - left_size;
        let right_counts = parent_counts
            .iter()
            .zip(left_counts.iter())
            .zip(missing_counts.iter())
            .map(|((parent, left), missing)| parent - left - missing)
            .collect::<Vec<_>>();
        if let Some(candidate) = score_binary_missing_assignment(
            context,
            feature_index,
            bin as u16,
            parent_counts,
            left_counts.clone(),
            left_size,
            right_counts,
            right_size.saturating_sub(missing_size),
            &missing_counts,
            missing_size,
        ) && best
            .as_ref()
            .is_none_or(|current: &BinarySplitChoice| candidate.score > current.score)
        {
            best = Some(candidate);
        }
    }

    best
}

fn score_numeric_cart_split_choice_from_hist_heuristic(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    parent_counts: &[usize],
    row_count: usize,
    bin_class_counts: &[ClassificationHistogramBin],
    observed_bins: &[usize],
) -> Option<BinarySplitChoice> {
    let missing_bin = usize::from(numeric_missing_bin(context.table));
    let observed_bins = observed_bins
        .iter()
        .copied()
        .filter(|bin| *bin != missing_bin)
        .collect::<Vec<_>>();
    if observed_bins.len() <= 1 {
        return None;
    }
    let missing_counts = bin_class_counts
        .get(missing_bin)
        .map(|bin| bin.counts.clone())
        .unwrap_or_else(|| vec![0usize; context.num_classes]);
    let missing_size = bin_class_counts
        .get(missing_bin)
        .map_or(0usize, ClassificationHistogramBin::size);
    let observed_parent_counts = parent_counts
        .iter()
        .zip(missing_counts.iter())
        .map(|(parent, missing)| parent - missing)
        .collect::<Vec<_>>();
    let observed_total = row_count.saturating_sub(missing_size);
    let parent_impurity =
        classification_impurity(&observed_parent_counts, observed_total, context.criterion);
    let mut left_counts = vec![0usize; context.num_classes];
    let mut left_size = 0usize;
    let mut best_threshold = None;
    let mut best_observed_score = f64::NEG_INFINITY;

    for bin in observed_bins {
        for (left_count, bin_count) in left_counts
            .iter_mut()
            .zip(bin_class_counts[bin].counts.iter())
        {
            *left_count += *bin_count;
        }
        left_size += bin_class_counts[bin].size();
        let right_size = observed_total.saturating_sub(left_size);
        if left_size < context.min_samples_leaf || right_size < context.min_samples_leaf {
            continue;
        }
        let right_counts = observed_parent_counts
            .iter()
            .zip(left_counts.iter())
            .map(|(parent, left)| parent - left)
            .collect::<Vec<_>>();
        let weighted_impurity = (left_size as f64 / observed_total as f64)
            * classification_impurity(&left_counts, left_size, context.criterion)
            + (right_size as f64 / observed_total as f64)
                * classification_impurity(&right_counts, right_size, context.criterion);
        let observed_score = parent_impurity - weighted_impurity;
        if observed_score > best_observed_score {
            best_observed_score = observed_score;
            best_threshold = Some((bin as u16, left_counts.clone(), left_size, right_counts));
        }
    }

    let (threshold_bin, left_counts, left_size, right_counts) = best_threshold?;
    let right_size = observed_total.saturating_sub(left_size);
    score_binary_missing_assignment(
        context,
        feature_index,
        threshold_bin,
        parent_counts,
        left_counts,
        left_size,
        right_counts,
        right_size,
        &missing_counts,
        missing_size,
    )
}

fn score_numeric_randomized_split_choice_from_hist(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
    parent_counts: &[usize],
    observed_bins: &[usize],
    histogram: &ClassificationFeatureHistogram,
) -> Option<BinarySplitChoice> {
    let missing_bin = usize::from(numeric_missing_bin(context.table));
    let observed_bins = observed_bins
        .iter()
        .copied()
        .filter(|bin| *bin != missing_bin)
        .collect::<Vec<_>>();
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
    let missing_counts = bins
        .get(missing_bin)
        .map(|bin| bin.counts.clone())
        .unwrap_or_else(|| vec![0usize; context.num_classes]);
    let missing_size = bins
        .get(missing_bin)
        .map_or(0usize, ClassificationHistogramBin::size);
    let mut left_counts = vec![0usize; context.num_classes];
    let mut left_size = 0usize;
    for bin in 0..=threshold_bin as usize {
        if bin >= bins.len() {
            break;
        }
        for (left_count, bin_count) in left_counts.iter_mut().zip(bins[bin].counts.iter()) {
            *left_count += *bin_count;
        }
        left_size += bins[bin].size();
    }
    let row_count = rows.len();
    let right_size = row_count - left_size;
    let right_counts = parent_counts
        .iter()
        .zip(left_counts.iter())
        .zip(missing_counts.iter())
        .map(|((parent, left), missing)| parent - left - missing)
        .collect::<Vec<_>>();
    score_binary_missing_assignment(
        context,
        feature_index,
        threshold_bin,
        parent_counts,
        left_counts,
        left_size,
        right_counts,
        right_size.saturating_sub(missing_size),
        &missing_counts,
        missing_size,
    )
}

fn score_multiway_grouping(
    context: &SplitScoringContext<'_>,
    grouped_counts: &[(u16, (usize, Vec<usize>))],
    row_count: usize,
    parent_impurity: f64,
    metric: MultiwayMetric,
) -> Option<f64> {
    if grouped_counts
        .iter()
        .any(|(_, (group_size, _))| *group_size < context.min_samples_leaf)
    {
        return None;
    }

    let weighted_child_impurity = grouped_counts
        .iter()
        .map(|(_, (group_size, counts))| {
            (*group_size as f64 / row_count as f64)
                * classification_impurity(counts, *group_size, context.criterion)
        })
        .sum::<f64>();
    let information_gain = parent_impurity - weighted_child_impurity;

    match metric {
        MultiwayMetric::InformationGain => Some(information_gain),
        MultiwayMetric::GainRatio => {
            let split_info = grouped_counts
                .iter()
                .map(|(_, (group_size, _))| {
                    let probability = *group_size as f64 / row_count as f64;
                    -probability * probability.log2()
                })
                .sum::<f64>();
            (split_info != 0.0).then_some(information_gain / split_info)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn score_binary_missing_assignment(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    threshold_bin: u16,
    parent_counts: &[usize],
    left_counts: Vec<usize>,
    left_size: usize,
    right_counts: Vec<usize>,
    right_size: usize,
    missing_counts: &[usize],
    missing_size: usize,
) -> Option<BinarySplitChoice> {
    let parent_impurity = classification_impurity(
        parent_counts,
        left_size + right_size + missing_size,
        context.criterion,
    );

    let evaluate = |direction: MissingBranchDirection| {
        let mut candidate_left_counts = left_counts.clone();
        let mut candidate_right_counts = right_counts.clone();
        let mut candidate_left_size = left_size;
        let mut candidate_right_size = right_size;
        match direction {
            MissingBranchDirection::Left => {
                candidate_left_size += missing_size;
                for (left, missing) in candidate_left_counts.iter_mut().zip(missing_counts.iter()) {
                    *left += *missing;
                }
            }
            MissingBranchDirection::Right => {
                candidate_right_size += missing_size;
                for (right, missing) in candidate_right_counts.iter_mut().zip(missing_counts.iter())
                {
                    *right += *missing;
                }
            }
            MissingBranchDirection::Node => {}
        }
        if candidate_left_size < context.min_samples_leaf
            || candidate_right_size < context.min_samples_leaf
        {
            return None;
        }
        let total_count = candidate_left_size + candidate_right_size;
        let weighted_impurity = (candidate_left_size as f64 / total_count as f64)
            * classification_impurity(
                &candidate_left_counts,
                candidate_left_size,
                context.criterion,
            )
            + (candidate_right_size as f64 / total_count as f64)
                * classification_impurity(
                    &candidate_right_counts,
                    candidate_right_size,
                    context.criterion,
                );
        Some(BinarySplitChoice {
            feature_index,
            score: parent_impurity - weighted_impurity,
            threshold_bin,
            missing_direction: direction,
        })
    };

    if missing_size == 0 {
        evaluate(MissingBranchDirection::Node)
    } else {
        [MissingBranchDirection::Left, MissingBranchDirection::Right]
            .into_iter()
            .filter_map(evaluate)
            .max_by(|left, right| left.score.total_cmp(&right.score))
    }
}

#[allow(clippy::too_many_arguments)]
fn score_binary_observed_split_then_assign_missing(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    threshold_bin: u16,
    left_counts: Vec<usize>,
    left_size: usize,
    right_counts: Vec<usize>,
    right_size: usize,
    missing_counts: &[usize],
    missing_size: usize,
) -> Option<BinarySplitChoice> {
    if left_size < context.min_samples_leaf || right_size < context.min_samples_leaf {
        return None;
    }
    let parent_counts = left_counts
        .iter()
        .zip(right_counts.iter())
        .map(|(left, right)| left + right)
        .collect::<Vec<_>>();
    let parent_impurity =
        classification_impurity(&parent_counts, left_size + right_size, context.criterion);
    let weighted_impurity = (left_size as f64 / (left_size + right_size) as f64)
        * classification_impurity(&left_counts, left_size, context.criterion)
        + (right_size as f64 / (left_size + right_size) as f64)
            * classification_impurity(&right_counts, right_size, context.criterion);
    let observed_score = parent_impurity - weighted_impurity;
    if !observed_score.is_finite() {
        return None;
    }

    let parent_counts_with_missing = parent_counts
        .iter()
        .zip(missing_counts.iter())
        .map(|(observed, missing)| observed + missing)
        .collect::<Vec<_>>();

    score_binary_missing_assignment(
        context,
        feature_index,
        threshold_bin,
        &parent_counts_with_missing,
        left_counts,
        left_size,
        right_counts,
        right_size,
        missing_counts,
        missing_size,
    )
}
