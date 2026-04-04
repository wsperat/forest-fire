use super::*;
use crate::tree::shared::MissingBranchDirection;
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
struct ObliviousLeafState {
    start: usize,
    end: usize,
    class_index: usize,
    class_counts: Vec<usize>,
}

impl ObliviousLeafState {
    fn len(&self) -> usize {
        self.end - self.start
    }
}

#[derive(Debug, Clone, Copy)]
struct ObliviousSplitCandidate {
    feature_index: usize,
    threshold_bin: u16,
    score: f64,
}

pub(super) fn train_oblivious_structure(
    table: &dyn TableAccess,
    class_indices: &[usize],
    class_labels: &[f64],
    criterion: Criterion,
    parallelism: Parallelism,
    options: DecisionTreeOptions,
) -> TreeStructure {
    let mut row_indices: Vec<usize> = (0..table.n_rows()).collect();
    let total_class_counts = class_counts(&row_indices, class_indices, class_labels.len());
    let total_impurity = classification_impurity(&total_class_counts, row_indices.len(), criterion);
    let mut leaves = vec![ObliviousLeafState {
        start: 0,
        end: row_indices.len(),
        class_index: majority_class(&row_indices, class_indices, class_labels.len()),
        class_counts: total_class_counts.clone(),
    }];
    let mut splits = Vec::new();

    for depth in 0..options.max_depth {
        if leaves
            .iter()
            .all(|leaf| leaf.len() < options.min_samples_split)
        {
            break;
        }
        let feature_indices = candidate_feature_indices(
            table.binned_feature_count(),
            options.max_features,
            node_seed(options.random_seed, depth, &[], 0x0B11_A10Cu64),
        );
        let best_split = if parallelism.enabled() {
            feature_indices
                .into_par_iter()
                .filter_map(|feature_index| {
                    score_oblivious_split(
                        table,
                        &row_indices,
                        class_indices,
                        feature_index,
                        &leaves,
                        class_labels.len(),
                        criterion,
                        options.min_samples_leaf,
                    )
                })
                .max_by(|left, right| left.score.total_cmp(&right.score))
        } else {
            feature_indices
                .into_iter()
                .filter_map(|feature_index| {
                    score_oblivious_split(
                        table,
                        &row_indices,
                        class_indices,
                        feature_index,
                        &leaves,
                        class_labels.len(),
                        criterion,
                        options.min_samples_leaf,
                    )
                })
                .max_by(|left, right| left.score.total_cmp(&right.score))
        };

        let Some(best_split) = best_split.filter(|candidate| candidate.score > 0.0) else {
            break;
        };
        if table.is_canary_binned_feature(best_split.feature_index) {
            break;
        }

        leaves = split_oblivious_leaves_in_place(
            table,
            &mut row_indices,
            class_indices,
            class_labels.len(),
            leaves,
            best_split.feature_index,
            best_split.threshold_bin,
        );
        splits.push(ObliviousSplit {
            feature_index: best_split.feature_index,
            threshold_bin: best_split.threshold_bin,
            missing_directions: Vec::new(),
            sample_count: table.n_rows(),
            impurity: total_impurity,
            gain: best_split.score,
        });
    }

    TreeStructure::Oblivious {
        splits,
        leaf_class_indices: leaves.iter().map(|leaf| leaf.class_index).collect(),
        leaf_sample_counts: leaves.iter().map(ObliviousLeafState::len).collect(),
        leaf_class_counts: leaves
            .iter()
            .map(|leaf| leaf.class_counts.clone())
            .collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn score_oblivious_split(
    table: &dyn TableAccess,
    row_indices: &[usize],
    class_indices: &[usize],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
    num_classes: usize,
    criterion: Criterion,
    min_samples_leaf: usize,
) -> Option<ObliviousSplitCandidate> {
    if table.is_binary_binned_feature(feature_index) {
        return score_binary_oblivious_split(
            table,
            row_indices,
            class_indices,
            feature_index,
            leaves,
            num_classes,
            criterion,
            min_samples_leaf,
        );
    }
    if let Some(candidate) = score_numeric_oblivious_split_fast(
        table,
        row_indices,
        class_indices,
        feature_index,
        leaves,
        num_classes,
        criterion,
        min_samples_leaf,
    ) {
        return Some(candidate);
    }
    let candidate_thresholds = leaves
        .iter()
        .flat_map(|leaf| {
            row_indices[leaf.start..leaf.end]
                .iter()
                .map(|row_idx| table.binned_value(feature_index, *row_idx))
        })
        .collect::<BTreeSet<_>>();

    candidate_thresholds
        .into_iter()
        .filter_map(|threshold_bin| {
            let score = leaves.iter().fold(0.0, |score, leaf| {
                let leaf_rows = &row_indices[leaf.start..leaf.end];
                let (left_rows, right_rows): (Vec<usize>, Vec<usize>) =
                    leaf_rows.iter().copied().partition(|row_idx| {
                        table.binned_value(feature_index, *row_idx) <= threshold_bin
                    });

                if left_rows.len() < min_samples_leaf || right_rows.len() < min_samples_leaf {
                    return score;
                }

                let parent_counts = leaf.class_counts.clone();
                let left_counts = class_counts(&left_rows, class_indices, num_classes);
                let right_counts = class_counts(&right_rows, class_indices, num_classes);

                let weighted_parent_impurity = leaf.len() as f64
                    * classification_impurity(&parent_counts, leaf.len(), criterion);
                let weighted_children_impurity = left_rows.len() as f64
                    * classification_impurity(&left_counts, left_rows.len(), criterion)
                    + right_rows.len() as f64
                        * classification_impurity(&right_counts, right_rows.len(), criterion);

                score + (weighted_parent_impurity - weighted_children_impurity)
            });

            (score > 0.0).then_some(ObliviousSplitCandidate {
                feature_index,
                threshold_bin,
                score,
            })
        })
        .max_by(|left, right| left.score.total_cmp(&right.score))
}

fn split_oblivious_leaves_in_place(
    table: &dyn TableAccess,
    row_indices: &mut [usize],
    class_indices: &[usize],
    num_classes: usize,
    leaves: Vec<ObliviousLeafState>,
    feature_index: usize,
    threshold_bin: u16,
) -> Vec<ObliviousLeafState> {
    let mut next_leaves = Vec::with_capacity(leaves.len() * 2);
    for leaf in leaves {
        let left_count = partition_rows_for_binary_split(
            table,
            feature_index,
            threshold_bin,
            MissingBranchDirection::Right,
            &mut row_indices[leaf.start..leaf.end],
        );
        let mid = leaf.start + left_count;
        let mut left_class_counts = vec![0usize; num_classes];
        let mut right_class_counts = vec![0usize; num_classes];
        for row_idx in &row_indices[leaf.start..mid] {
            left_class_counts[class_indices[*row_idx]] += 1;
        }
        for row_idx in &row_indices[mid..leaf.end] {
            right_class_counts[class_indices[*row_idx]] += 1;
        }
        let left_class_index = if left_count == 0 {
            leaf.class_index
        } else {
            majority_class_from_counts(&left_class_counts)
        };
        let right_class_index = if mid == leaf.end {
            leaf.class_index
        } else {
            majority_class_from_counts(&right_class_counts)
        };
        next_leaves.push(ObliviousLeafState {
            start: leaf.start,
            end: mid,
            class_index: left_class_index,
            class_counts: left_class_counts,
        });
        next_leaves.push(ObliviousLeafState {
            start: mid,
            end: leaf.end,
            class_index: right_class_index,
            class_counts: right_class_counts,
        });
    }
    next_leaves
}

#[allow(clippy::too_many_arguments)]
fn score_binary_oblivious_split(
    table: &dyn TableAccess,
    row_indices: &[usize],
    class_indices: &[usize],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
    num_classes: usize,
    criterion: Criterion,
    min_samples_leaf: usize,
) -> Option<ObliviousSplitCandidate> {
    let mut score = 0.0;
    let mut found_valid = false;

    for leaf in leaves {
        let mut left_counts = vec![0usize; num_classes];
        let mut left_size = 0usize;
        for row_idx in &row_indices[leaf.start..leaf.end] {
            if !table
                .binned_boolean_value(feature_index, *row_idx)
                .expect("binary feature must expose boolean values")
            {
                left_counts[class_indices[*row_idx]] += 1;
                left_size += 1;
            }
        }
        let right_size = leaf.len() - left_size;
        if left_size < min_samples_leaf || right_size < min_samples_leaf {
            continue;
        }
        found_valid = true;
        let right_counts = leaf
            .class_counts
            .iter()
            .zip(left_counts.iter())
            .map(|(parent, left)| parent - left)
            .collect::<Vec<_>>();
        let weighted_parent_impurity =
            leaf.len() as f64 * classification_impurity(&leaf.class_counts, leaf.len(), criterion);
        let weighted_children_impurity = left_size as f64
            * classification_impurity(&left_counts, left_size, criterion)
            + right_size as f64 * classification_impurity(&right_counts, right_size, criterion);
        score += weighted_parent_impurity - weighted_children_impurity;
    }

    (found_valid && score > 0.0).then_some(ObliviousSplitCandidate {
        feature_index,
        threshold_bin: 0,
        score,
    })
}

#[allow(clippy::too_many_arguments)]
fn score_numeric_oblivious_split_fast(
    table: &dyn TableAccess,
    row_indices: &[usize],
    class_indices: &[usize],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
    num_classes: usize,
    criterion: Criterion,
    min_samples_leaf: usize,
) -> Option<ObliviousSplitCandidate> {
    let bin_cap = table.numeric_bin_cap();
    if bin_cap == 0 {
        return None;
    }

    let mut threshold_scores = vec![0.0; bin_cap];
    let mut observed_any = false;

    for leaf in leaves {
        let mut bin_class_counts = vec![vec![0usize; num_classes]; bin_cap];
        let mut observed_bins = vec![false; bin_cap];
        for row_idx in &row_indices[leaf.start..leaf.end] {
            let bin = table.binned_value(feature_index, *row_idx) as usize;
            if bin >= bin_cap {
                return None;
            }
            bin_class_counts[bin][class_indices[*row_idx]] += 1;
            observed_bins[bin] = true;
        }

        let observed_bins: Vec<usize> = observed_bins
            .into_iter()
            .enumerate()
            .filter_map(|(bin, seen)| seen.then_some(bin))
            .collect();
        if observed_bins.len() <= 1 {
            continue;
        }
        observed_any = true;

        let parent_weighted_impurity =
            leaf.len() as f64 * classification_impurity(&leaf.class_counts, leaf.len(), criterion);
        let mut left_counts = vec![0usize; num_classes];
        let mut left_size = 0usize;

        for &bin in &observed_bins {
            for class_index in 0..num_classes {
                left_counts[class_index] += bin_class_counts[bin][class_index];
            }
            left_size += bin_class_counts[bin].iter().sum::<usize>();
            let right_size = leaf.len() - left_size;

            if left_size < min_samples_leaf || right_size < min_samples_leaf {
                continue;
            }

            let right_counts = leaf
                .class_counts
                .iter()
                .zip(left_counts.iter())
                .map(|(parent, left)| parent - left)
                .collect::<Vec<_>>();
            let weighted_children_impurity = left_size as f64
                * classification_impurity(&left_counts, left_size, criterion)
                + right_size as f64 * classification_impurity(&right_counts, right_size, criterion);
            threshold_scores[bin] += parent_weighted_impurity - weighted_children_impurity;
        }
    }

    if !observed_any {
        return None;
    }

    threshold_scores
        .into_iter()
        .enumerate()
        .filter(|(_, score)| *score > 0.0)
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(threshold_bin, score)| ObliviousSplitCandidate {
            feature_index,
            threshold_bin: threshold_bin as u16,
            score,
        })
}
