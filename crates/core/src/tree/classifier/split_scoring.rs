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

#[allow(dead_code)]
pub(super) fn score_split(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
    algorithm: DecisionTreeAlgorithm,
) -> Option<SplitCandidate> {
    match algorithm {
        DecisionTreeAlgorithm::Id3 => score_multiway_split(
            context,
            feature_index,
            rows,
            MultiwayMetric::InformationGain,
        ),
        DecisionTreeAlgorithm::C45 => {
            score_multiway_split(context, feature_index, rows, MultiwayMetric::GainRatio)
        }
        DecisionTreeAlgorithm::Cart => score_cart_split(context, feature_index, rows),
        DecisionTreeAlgorithm::Randomized => score_randomized_split(context, feature_index, rows),
        DecisionTreeAlgorithm::Oblivious => None,
    }
}

#[allow(dead_code)]
fn score_multiway_split(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
    metric: MultiwayMetric,
) -> Option<SplitCandidate> {
    let grouped_rows = if context.table.is_binary_binned_feature(feature_index) {
        let (false_rows, true_rows): (Vec<usize>, Vec<usize>) =
            rows.iter().copied().partition(|row_idx| {
                !context
                    .table
                    .binned_boolean_value(feature_index, *row_idx)
                    .expect("binary feature must expose boolean values")
            });
        [(0u16, false_rows), (1u16, true_rows)]
            .into_iter()
            .filter(|(_bin, group_rows)| !group_rows.is_empty())
            .collect::<BTreeMap<_, _>>()
    } else {
        rows.iter()
            .fold(BTreeMap::<u16, Vec<usize>>::new(), |mut groups, row_idx| {
                groups
                    .entry(context.table.binned_value(feature_index, *row_idx))
                    .or_default()
                    .push(*row_idx);
                groups
            })
    };

    if grouped_rows.len() <= 1
        || grouped_rows
            .values()
            .any(|group| group.len() < context.min_samples_leaf)
    {
        return None;
    }

    let parent_counts = class_counts(rows, context.class_indices, context.num_classes);
    let parent_impurity = classification_impurity(&parent_counts, rows.len(), context.criterion);
    let weighted_child_impurity = grouped_rows
        .values()
        .map(|group_rows| {
            let counts = class_counts(group_rows, context.class_indices, context.num_classes);
            (group_rows.len() as f64 / rows.len() as f64)
                * classification_impurity(&counts, group_rows.len(), context.criterion)
        })
        .sum::<f64>();
    let information_gain = parent_impurity - weighted_child_impurity;

    let score = match metric {
        MultiwayMetric::InformationGain => information_gain,
        MultiwayMetric::GainRatio => {
            let split_info = grouped_rows
                .values()
                .map(|group_rows| {
                    let probability = group_rows.len() as f64 / rows.len() as f64;
                    -probability * probability.log2()
                })
                .sum::<f64>();

            if split_info == 0.0 {
                return None;
            }

            information_gain / split_info
        }
    };

    Some(SplitCandidate::Multiway {
        feature_index,
        score,
        branches: grouped_rows.into_iter().collect(),
    })
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

#[allow(dead_code)]
fn score_cart_split(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<SplitCandidate> {
    if context.table.is_binary_binned_feature(feature_index) {
        return score_binary_cart_split(context, feature_index, rows);
    }
    if let Some(candidate) = score_numeric_cart_split_fast(context, feature_index, rows) {
        return Some(candidate);
    }
    let parent_counts = class_counts(rows, context.class_indices, context.num_classes);
    let parent_impurity = classification_impurity(&parent_counts, rows.len(), context.criterion);

    rows.iter()
        .map(|row_idx| context.table.binned_value(feature_index, *row_idx))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|threshold_bin| {
            let (left_rows, right_rows): (Vec<usize>, Vec<usize>) =
                rows.iter().copied().partition(|row_idx| {
                    context.table.binned_value(feature_index, *row_idx) <= threshold_bin
                });

            if left_rows.len() < context.min_samples_leaf
                || right_rows.len() < context.min_samples_leaf
            {
                return None;
            }

            let left_counts = class_counts(&left_rows, context.class_indices, context.num_classes);
            let right_counts =
                class_counts(&right_rows, context.class_indices, context.num_classes);
            let weighted_impurity = (left_rows.len() as f64 / rows.len() as f64)
                * classification_impurity(&left_counts, left_rows.len(), context.criterion)
                + (right_rows.len() as f64 / rows.len() as f64)
                    * classification_impurity(&right_counts, right_rows.len(), context.criterion);

            Some(SplitCandidate::Binary {
                feature_index,
                score: parent_impurity - weighted_impurity,
                threshold_bin,
                left_rows,
                right_rows,
            })
        })
        .max_by(|left, right| split_score(left).total_cmp(&split_score(right)))
}

#[allow(dead_code)]
fn score_randomized_split(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<SplitCandidate> {
    if context.table.is_binary_binned_feature(feature_index) {
        return score_binary_cart_split(context, feature_index, rows);
    }
    if let Some(candidate) = score_numeric_randomized_split_fast(context, feature_index, rows) {
        return Some(candidate);
    }

    let candidate_thresholds = rows
        .iter()
        .map(|row_idx| context.table.binned_value(feature_index, *row_idx))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let threshold_bin =
        choose_random_threshold(&candidate_thresholds, feature_index, rows, 0xC1A551F1u64)?;

    let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = rows
        .iter()
        .copied()
        .partition(|row_idx| context.table.binned_value(feature_index, *row_idx) <= threshold_bin);

    if left_rows.len() < context.min_samples_leaf || right_rows.len() < context.min_samples_leaf {
        return None;
    }

    let parent_counts = class_counts(rows, context.class_indices, context.num_classes);
    let parent_impurity = classification_impurity(&parent_counts, rows.len(), context.criterion);
    let left_counts = class_counts(&left_rows, context.class_indices, context.num_classes);
    let right_counts = class_counts(&right_rows, context.class_indices, context.num_classes);
    let weighted_impurity = (left_rows.len() as f64 / rows.len() as f64)
        * classification_impurity(&left_counts, left_rows.len(), context.criterion)
        + (right_rows.len() as f64 / rows.len() as f64)
            * classification_impurity(&right_counts, right_rows.len(), context.criterion);

    Some(SplitCandidate::Binary {
        feature_index,
        score: parent_impurity - weighted_impurity,
        threshold_bin,
        left_rows,
        right_rows,
    })
}

#[allow(dead_code)]
fn score_binary_cart_split(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<SplitCandidate> {
    let (left_rows, right_rows): (Vec<usize>, Vec<usize>) =
        rows.iter().copied().partition(|row_idx| {
            !context
                .table
                .binned_boolean_value(feature_index, *row_idx)
                .expect("binary feature must expose boolean values")
        });

    if left_rows.len() < context.min_samples_leaf || right_rows.len() < context.min_samples_leaf {
        return None;
    }

    let parent_counts = class_counts(rows, context.class_indices, context.num_classes);
    let parent_impurity = classification_impurity(&parent_counts, rows.len(), context.criterion);
    let left_counts = class_counts(&left_rows, context.class_indices, context.num_classes);
    let right_counts = class_counts(&right_rows, context.class_indices, context.num_classes);
    let weighted_impurity = (left_rows.len() as f64 / rows.len() as f64)
        * classification_impurity(&left_counts, left_rows.len(), context.criterion)
        + (right_rows.len() as f64 / rows.len() as f64)
            * classification_impurity(&right_counts, right_rows.len(), context.criterion);

    Some(SplitCandidate::Binary {
        feature_index,
        score: parent_impurity - weighted_impurity,
        threshold_bin: 0,
        left_rows,
        right_rows,
    })
}

#[allow(dead_code)]
fn score_numeric_cart_split_fast(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<SplitCandidate> {
    let parent_counts = class_counts(rows, context.class_indices, context.num_classes);
    let parent_impurity = classification_impurity(&parent_counts, rows.len(), context.criterion);
    let bin_cap = context.table.numeric_bin_cap();
    if bin_cap == 0 {
        return None;
    }

    let mut bin_class_counts = vec![vec![0usize; context.num_classes]; bin_cap];
    let mut observed_bins = vec![false; bin_cap];
    for row_idx in rows {
        let bin = context.table.binned_value(feature_index, *row_idx) as usize;
        if bin >= bin_cap {
            return None;
        }
        bin_class_counts[bin][context.class_indices[*row_idx]] += 1;
        observed_bins[bin] = true;
    }

    let observed_bins: Vec<usize> = observed_bins
        .into_iter()
        .enumerate()
        .filter_map(|(bin, seen)| seen.then_some(bin))
        .collect();
    if observed_bins.len() <= 1 {
        return None;
    }

    let mut left_counts = vec![0usize; context.num_classes];
    let mut left_size = 0usize;
    let mut best_threshold = None;
    let mut best_score = f64::NEG_INFINITY;

    for &bin in &observed_bins {
        for class_index in 0..context.num_classes {
            left_counts[class_index] += bin_class_counts[bin][class_index];
        }
        left_size += bin_class_counts[bin].iter().sum::<usize>();
        let right_size = rows.len() - left_size;

        if left_size < context.min_samples_leaf || right_size < context.min_samples_leaf {
            continue;
        }

        let right_counts = parent_counts
            .iter()
            .zip(left_counts.iter())
            .map(|(parent, left)| parent - left)
            .collect::<Vec<_>>();
        let weighted_impurity = (left_size as f64 / rows.len() as f64)
            * classification_impurity(&left_counts, left_size, context.criterion)
            + (right_size as f64 / rows.len() as f64)
                * classification_impurity(&right_counts, right_size, context.criterion);
        let score = parent_impurity - weighted_impurity;
        if score > best_score {
            best_score = score;
            best_threshold = Some(bin as u16);
        }
    }

    let threshold_bin = best_threshold?;
    let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = rows
        .iter()
        .copied()
        .partition(|row_idx| context.table.binned_value(feature_index, *row_idx) <= threshold_bin);

    Some(SplitCandidate::Binary {
        feature_index,
        score: best_score,
        threshold_bin,
        left_rows,
        right_rows,
    })
}

#[allow(dead_code)]
fn score_numeric_randomized_split_fast(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<SplitCandidate> {
    let bin_cap = context.table.numeric_bin_cap();
    if bin_cap == 0 {
        return None;
    }
    let mut observed_bins = vec![false; bin_cap];
    for row_idx in rows {
        let bin = context.table.binned_value(feature_index, *row_idx) as usize;
        if bin >= bin_cap {
            return None;
        }
        observed_bins[bin] = true;
    }
    let candidate_thresholds = observed_bins
        .into_iter()
        .enumerate()
        .filter_map(|(bin, seen)| seen.then_some(bin as u16))
        .collect::<Vec<_>>();
    let threshold_bin =
        choose_random_threshold(&candidate_thresholds, feature_index, rows, 0xC1A551F1u64)?;

    let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = rows
        .iter()
        .copied()
        .partition(|row_idx| context.table.binned_value(feature_index, *row_idx) <= threshold_bin);

    if left_rows.len() < context.min_samples_leaf || right_rows.len() < context.min_samples_leaf {
        return None;
    }

    let parent_counts = class_counts(rows, context.class_indices, context.num_classes);
    let parent_impurity = classification_impurity(&parent_counts, rows.len(), context.criterion);
    let left_counts = class_counts(&left_rows, context.class_indices, context.num_classes);
    let right_counts = class_counts(&right_rows, context.class_indices, context.num_classes);
    let weighted_impurity = (left_rows.len() as f64 / rows.len() as f64)
        * classification_impurity(&left_counts, left_rows.len(), context.criterion)
        + (right_rows.len() as f64 / rows.len() as f64)
            * classification_impurity(&right_counts, right_rows.len(), context.criterion);

    Some(SplitCandidate::Binary {
        feature_index,
        score: parent_impurity - weighted_impurity,
        threshold_bin,
        left_rows,
        right_rows,
    })
}

#[allow(dead_code)]
pub(super) fn split_score(candidate: &SplitCandidate) -> f64 {
    match candidate {
        SplitCandidate::Multiway { score, .. } | SplitCandidate::Binary { score, .. } => *score,
    }
}

#[allow(dead_code)]
pub(super) fn score_binary_split_choice(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
    algorithm: DecisionTreeAlgorithm,
) -> Option<BinarySplitChoice> {
    match algorithm {
        DecisionTreeAlgorithm::Cart => {
            if context.table.is_binary_binned_feature(feature_index) {
                score_binary_cart_split_choice(context, feature_index, rows)
            } else {
                score_numeric_cart_split_choice_fast(context, feature_index, rows)
            }
        }
        DecisionTreeAlgorithm::Randomized => {
            if context.table.is_binary_binned_feature(feature_index) {
                score_binary_cart_split_choice(context, feature_index, rows)
            } else {
                score_numeric_randomized_split_choice_fast(context, feature_index, rows)
            }
        }
        _ => None,
    }
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
                false_counts,
                true_counts,
                false_size,
                true_size,
            },
        ) => score_binary_cart_split_choice_from_counts(
            context,
            feature_index,
            parent_counts,
            false_counts,
            *false_size,
            true_counts,
            *true_size,
        ),
        (
            DecisionTreeAlgorithm::Cart,
            ClassificationFeatureHistogram::Numeric {
                bin_class_counts,
                observed_bins,
            },
        ) => score_numeric_cart_split_choice_from_hist(
            context,
            feature_index,
            parent_counts,
            rows.len(),
            bin_class_counts,
            observed_bins,
        ),
        (
            DecisionTreeAlgorithm::Randomized,
            ClassificationFeatureHistogram::Binary {
                false_counts,
                true_counts,
                false_size,
                true_size,
            },
        ) => score_binary_cart_split_choice_from_counts(
            context,
            feature_index,
            parent_counts,
            false_counts,
            *false_size,
            true_counts,
            *true_size,
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
    bin_class_counts: &[Vec<usize>],
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
            left_counts[class_index] += bin_class_counts[bin][class_index];
        }
        left_size += bin_class_counts[bin].iter().sum::<usize>();
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
    let ClassificationFeatureHistogram::Numeric {
        bin_class_counts, ..
    } = histogram
    else {
        unreachable!("randomized numeric histogram must be numeric");
    };
    let mut left_counts = vec![0usize; context.num_classes];
    let mut left_size = 0usize;
    for bin in 0..=threshold_bin as usize {
        if bin >= bin_class_counts.len() {
            break;
        }
        for class_index in 0..context.num_classes {
            left_counts[class_index] += bin_class_counts[bin][class_index];
        }
        left_size += bin_class_counts[bin].iter().sum::<usize>();
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

#[allow(dead_code)]
fn score_binary_cart_split_choice(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<BinarySplitChoice> {
    let parent_counts = class_counts(rows, context.class_indices, context.num_classes);
    let parent_impurity = classification_impurity(&parent_counts, rows.len(), context.criterion);
    let mut left_counts = vec![0usize; context.num_classes];
    let mut left_size = 0usize;

    for row_idx in rows {
        if !context
            .table
            .binned_boolean_value(feature_index, *row_idx)
            .expect("binary feature must expose boolean values")
        {
            left_counts[context.class_indices[*row_idx]] += 1;
            left_size += 1;
        }
    }

    let right_size = rows.len() - left_size;
    if left_size < context.min_samples_leaf || right_size < context.min_samples_leaf {
        return None;
    }

    let right_counts = parent_counts
        .iter()
        .zip(left_counts.iter())
        .map(|(parent, left)| parent - left)
        .collect::<Vec<_>>();
    let weighted_impurity = (left_size as f64 / rows.len() as f64)
        * classification_impurity(&left_counts, left_size, context.criterion)
        + (right_size as f64 / rows.len() as f64)
            * classification_impurity(&right_counts, right_size, context.criterion);

    Some(BinarySplitChoice {
        feature_index,
        score: parent_impurity - weighted_impurity,
        threshold_bin: 0,
    })
}

#[allow(dead_code)]
fn score_numeric_cart_split_choice_fast(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<BinarySplitChoice> {
    let parent_counts = class_counts(rows, context.class_indices, context.num_classes);
    let parent_impurity = classification_impurity(&parent_counts, rows.len(), context.criterion);
    let bin_cap = context.table.numeric_bin_cap();
    if bin_cap == 0 {
        return None;
    }

    let mut bin_class_counts = vec![vec![0usize; context.num_classes]; bin_cap];
    let mut observed_bins = vec![false; bin_cap];
    for row_idx in rows {
        let bin = context.table.binned_value(feature_index, *row_idx) as usize;
        if bin >= bin_cap {
            return None;
        }
        bin_class_counts[bin][context.class_indices[*row_idx]] += 1;
        observed_bins[bin] = true;
    }

    let observed_bins: Vec<usize> = observed_bins
        .into_iter()
        .enumerate()
        .filter_map(|(bin, seen)| seen.then_some(bin))
        .collect();
    if observed_bins.len() <= 1 {
        return None;
    }

    let mut left_counts = vec![0usize; context.num_classes];
    let mut left_size = 0usize;
    let mut best_threshold = None;
    let mut best_score = f64::NEG_INFINITY;

    for &bin in &observed_bins {
        for class_index in 0..context.num_classes {
            left_counts[class_index] += bin_class_counts[bin][class_index];
        }
        left_size += bin_class_counts[bin].iter().sum::<usize>();
        let right_size = rows.len() - left_size;

        if left_size < context.min_samples_leaf || right_size < context.min_samples_leaf {
            continue;
        }

        let right_counts = parent_counts
            .iter()
            .zip(left_counts.iter())
            .map(|(parent, left)| parent - left)
            .collect::<Vec<_>>();
        let weighted_impurity = (left_size as f64 / rows.len() as f64)
            * classification_impurity(&left_counts, left_size, context.criterion)
            + (right_size as f64 / rows.len() as f64)
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

#[allow(dead_code)]
fn score_numeric_randomized_split_choice_fast(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<BinarySplitChoice> {
    let bin_cap = context.table.numeric_bin_cap();
    if bin_cap == 0 {
        return None;
    }
    let mut observed_bins = vec![false; bin_cap];
    for row_idx in rows {
        let bin = context.table.binned_value(feature_index, *row_idx) as usize;
        if bin >= bin_cap {
            return None;
        }
        observed_bins[bin] = true;
    }
    let candidate_thresholds = observed_bins
        .into_iter()
        .enumerate()
        .filter_map(|(bin, seen)| seen.then_some(bin as u16))
        .collect::<Vec<_>>();
    let threshold_bin =
        choose_random_threshold(&candidate_thresholds, feature_index, rows, 0xC1A551F1u64)?;

    let parent_counts = class_counts(rows, context.class_indices, context.num_classes);
    let parent_impurity = classification_impurity(&parent_counts, rows.len(), context.criterion);
    let mut left_counts = vec![0usize; context.num_classes];
    let mut left_size = 0usize;
    for row_idx in rows {
        if context.table.binned_value(feature_index, *row_idx) <= threshold_bin {
            left_counts[context.class_indices[*row_idx]] += 1;
            left_size += 1;
        }
    }
    let right_size = rows.len() - left_size;
    if left_size < context.min_samples_leaf || right_size < context.min_samples_leaf {
        return None;
    }
    let right_counts = parent_counts
        .iter()
        .zip(left_counts.iter())
        .map(|(parent, left)| parent - left)
        .collect::<Vec<_>>();
    let weighted_impurity = (left_size as f64 / rows.len() as f64)
        * classification_impurity(&left_counts, left_size, context.criterion)
        + (right_size as f64 / rows.len() as f64)
            * classification_impurity(&right_counts, right_size, context.criterion);

    Some(BinarySplitChoice {
        feature_index,
        score: parent_impurity - weighted_impurity,
        threshold_bin,
    })
}
