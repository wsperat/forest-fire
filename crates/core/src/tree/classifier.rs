use crate::{Criterion, Parallelism};
use forestfire_data::TableAccess;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionTreeAlgorithm {
    Id3,
    C45,
    Cart,
    Oblivious,
}

#[derive(Debug, Clone, Copy)]
pub struct DecisionTreeOptions {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
}

impl Default for DecisionTreeOptions {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }
}

#[derive(Debug)]
pub enum DecisionTreeError {
    EmptyTarget,
    InvalidTargetValue { row: usize, value: f64 },
}

impl Display for DecisionTreeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DecisionTreeError::EmptyTarget => write!(f, "Cannot train on an empty target vector."),
            DecisionTreeError::InvalidTargetValue { row, value } => write!(
                f,
                "Classification targets must be finite values. Found {} at row {}.",
                value, row
            ),
        }
    }
}

impl Error for DecisionTreeError {}

#[derive(Debug, Clone)]
pub struct DecisionTreeClassifier {
    algorithm: DecisionTreeAlgorithm,
    criterion: Criterion,
    class_labels: Vec<f64>,
    structure: TreeStructure,
}

#[derive(Debug, Clone)]
enum TreeStructure {
    Standard {
        nodes: Vec<TreeNode>,
        root: usize,
    },
    Oblivious {
        splits: Vec<ObliviousSplit>,
        leaf_class_indices: Vec<usize>,
    },
}

#[derive(Debug, Clone, Copy)]
struct ObliviousSplit {
    feature_index: usize,
    threshold_bin: u16,
}

#[derive(Debug, Clone)]
enum TreeNode {
    Leaf {
        class_index: usize,
    },
    MultiwaySplit {
        feature_index: usize,
        fallback_class_index: usize,
        branches: Vec<(u16, usize)>,
    },
    BinarySplit {
        feature_index: usize,
        threshold_bin: u16,
        left_child: usize,
        right_child: usize,
    },
}

#[derive(Debug, Clone)]
enum SplitCandidate {
    Multiway {
        feature_index: usize,
        score: f64,
        branches: Vec<(u16, Vec<usize>)>,
    },
    Binary {
        feature_index: usize,
        score: f64,
        threshold_bin: u16,
        left_rows: Vec<usize>,
        right_rows: Vec<usize>,
    },
}

pub fn train_id3(train_set: &dyn TableAccess) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_id3_with_criterion(train_set, Criterion::Entropy)
}

pub fn train_id3_with_criterion(
    train_set: &dyn TableAccess,
    criterion: Criterion,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_id3_with_criterion_and_parallelism(train_set, criterion, Parallelism::sequential())
}

pub(crate) fn train_id3_with_criterion_and_parallelism(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_classifier(
        train_set,
        DecisionTreeAlgorithm::Id3,
        criterion,
        parallelism,
        DecisionTreeOptions::default(),
    )
}

pub fn train_c45(train_set: &dyn TableAccess) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_c45_with_criterion(train_set, Criterion::Entropy)
}

pub fn train_c45_with_criterion(
    train_set: &dyn TableAccess,
    criterion: Criterion,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_c45_with_criterion_and_parallelism(train_set, criterion, Parallelism::sequential())
}

pub(crate) fn train_c45_with_criterion_and_parallelism(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_classifier(
        train_set,
        DecisionTreeAlgorithm::C45,
        criterion,
        parallelism,
        DecisionTreeOptions::default(),
    )
}

pub fn train_cart(
    train_set: &dyn TableAccess,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_cart_with_criterion(train_set, Criterion::Gini)
}

pub fn train_cart_with_criterion(
    train_set: &dyn TableAccess,
    criterion: Criterion,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_cart_with_criterion_and_parallelism(train_set, criterion, Parallelism::sequential())
}

pub(crate) fn train_cart_with_criterion_and_parallelism(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_classifier(
        train_set,
        DecisionTreeAlgorithm::Cart,
        criterion,
        parallelism,
        DecisionTreeOptions::default(),
    )
}

pub fn train_oblivious(
    train_set: &dyn TableAccess,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_oblivious_with_criterion(train_set, Criterion::Gini)
}

pub fn train_oblivious_with_criterion(
    train_set: &dyn TableAccess,
    criterion: Criterion,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_oblivious_with_criterion_and_parallelism(train_set, criterion, Parallelism::sequential())
}

pub(crate) fn train_oblivious_with_criterion_and_parallelism(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_classifier(
        train_set,
        DecisionTreeAlgorithm::Oblivious,
        criterion,
        parallelism,
        DecisionTreeOptions::default(),
    )
}

fn train_classifier(
    train_set: &dyn TableAccess,
    algorithm: DecisionTreeAlgorithm,
    criterion: Criterion,
    parallelism: Parallelism,
    options: DecisionTreeOptions,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    if train_set.n_rows() == 0 {
        return Err(DecisionTreeError::EmptyTarget);
    }

    let (class_labels, class_indices) = encode_class_labels(train_set)?;
    let structure = match algorithm {
        DecisionTreeAlgorithm::Oblivious => train_oblivious_structure(
            train_set,
            &class_indices,
            &class_labels,
            criterion,
            parallelism,
            options,
        ),
        _ => {
            let mut nodes = Vec::new();
            let all_rows: Vec<usize> = (0..train_set.n_rows()).collect();
            let context = BuildContext {
                table: train_set,
                class_indices: &class_indices,
                class_labels: &class_labels,
                algorithm,
                criterion,
                parallelism,
                options,
            };
            let root = build_node(&context, &mut nodes, &all_rows, 0);
            TreeStructure::Standard { nodes, root }
        }
    };

    Ok(DecisionTreeClassifier {
        algorithm,
        criterion,
        class_labels,
        structure,
    })
}

impl DecisionTreeClassifier {
    pub fn algorithm(&self) -> DecisionTreeAlgorithm {
        self.algorithm
    }

    pub fn criterion(&self) -> Criterion {
        self.criterion
    }

    pub fn predict_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        (0..table.n_rows())
            .map(|row_idx| self.predict_row(table, row_idx))
            .collect()
    }

    fn predict_row(&self, table: &dyn TableAccess, row_idx: usize) -> f64 {
        match &self.structure {
            TreeStructure::Standard { nodes, root } => {
                let mut node_index = *root;

                loop {
                    match &nodes[node_index] {
                        TreeNode::Leaf { class_index } => return self.class_labels[*class_index],
                        TreeNode::MultiwaySplit {
                            feature_index,
                            fallback_class_index,
                            branches,
                        } => {
                            let bin = table.binned_value(*feature_index, row_idx);
                            if let Some((_, child_index)) =
                                branches.iter().find(|(branch_bin, _)| *branch_bin == bin)
                            {
                                node_index = *child_index;
                            } else {
                                return self.class_labels[*fallback_class_index];
                            }
                        }
                        TreeNode::BinarySplit {
                            feature_index,
                            threshold_bin,
                            left_child,
                            right_child,
                        } => {
                            let bin = table.binned_value(*feature_index, row_idx);
                            node_index = if bin <= *threshold_bin {
                                *left_child
                            } else {
                                *right_child
                            };
                        }
                    }
                }
            }
            TreeStructure::Oblivious {
                splits,
                leaf_class_indices,
            } => {
                let leaf_index = splits.iter().fold(0usize, |leaf_index, split| {
                    let go_right =
                        table.binned_value(split.feature_index, row_idx) > split.threshold_bin;
                    (leaf_index << 1) | usize::from(go_right)
                });

                self.class_labels[leaf_class_indices[leaf_index]]
            }
        }
    }
}

fn encode_class_labels(
    train_set: &dyn TableAccess,
) -> Result<(Vec<f64>, Vec<usize>), DecisionTreeError> {
    let targets: Vec<f64> = (0..train_set.n_rows())
        .map(|row_idx| {
            let value = train_set.target_value(row_idx);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(DecisionTreeError::InvalidTargetValue {
                    row: row_idx,
                    value,
                })
            }
        })
        .collect::<Result<_, _>>()?;

    let class_labels = targets
        .iter()
        .copied()
        .fold(Vec::<f64>::new(), |mut labels, value| {
            if labels
                .binary_search_by(|candidate| candidate.total_cmp(&value))
                .is_err()
            {
                labels.push(value);
                labels.sort_by(|left, right| left.total_cmp(right));
            }
            labels
        });

    let class_indices = targets
        .iter()
        .map(|value| {
            class_labels
                .binary_search_by(|candidate| candidate.total_cmp(value))
                .expect("target value must exist in class label vocabulary")
        })
        .collect();

    Ok((class_labels, class_indices))
}

fn build_node(
    context: &BuildContext<'_>,
    nodes: &mut Vec<TreeNode>,
    rows: &[usize],
    depth: usize,
) -> usize {
    let majority_class_index =
        majority_class(rows, context.class_indices, context.class_labels.len());

    if rows.is_empty()
        || depth >= context.options.max_depth
        || rows.len() < context.options.min_samples_split
        || is_pure(rows, context.class_indices)
    {
        return push_leaf(nodes, majority_class_index);
    }

    let scoring = SplitScoringContext {
        table: context.table,
        class_indices: context.class_indices,
        num_classes: context.class_labels.len(),
        criterion: context.criterion,
        min_samples_leaf: context.options.min_samples_leaf,
    };
    let best_split = if context.parallelism.enabled() {
        (0..context.table.binned_feature_count())
            .into_par_iter()
            .filter_map(|feature_index| {
                score_split(&scoring, feature_index, rows, context.algorithm)
            })
            .max_by(|left, right| split_score(left).total_cmp(&split_score(right)))
    } else {
        (0..context.table.binned_feature_count())
            .filter_map(|feature_index| {
                score_split(&scoring, feature_index, rows, context.algorithm)
            })
            .max_by(|left, right| split_score(left).total_cmp(&split_score(right)))
    };

    match best_split {
        Some(best_split)
            if context
                .table
                .is_canary_binned_feature(split_feature_index(&best_split)) =>
        {
            push_leaf(nodes, majority_class_index)
        }
        Some(SplitCandidate::Multiway {
            feature_index,
            score,
            branches,
        }) if score > 0.0 => {
            let branch_nodes = branches
                .into_iter()
                .map(|(bin, branch_rows)| {
                    (bin, build_node(context, nodes, &branch_rows, depth + 1))
                })
                .collect();

            push_node(
                nodes,
                TreeNode::MultiwaySplit {
                    feature_index,
                    fallback_class_index: majority_class_index,
                    branches: branch_nodes,
                },
            )
        }
        Some(SplitCandidate::Binary {
            feature_index,
            score,
            threshold_bin,
            left_rows,
            right_rows,
        }) if score > 0.0 => {
            let left_child = build_node(context, nodes, &left_rows, depth + 1);
            let right_child = build_node(context, nodes, &right_rows, depth + 1);

            push_node(
                nodes,
                TreeNode::BinarySplit {
                    feature_index,
                    threshold_bin,
                    left_child,
                    right_child,
                },
            )
        }
        _ => push_leaf(nodes, majority_class_index),
    }
}

struct BuildContext<'a> {
    table: &'a dyn TableAccess,
    class_indices: &'a [usize],
    class_labels: &'a [f64],
    algorithm: DecisionTreeAlgorithm,
    criterion: Criterion,
    parallelism: Parallelism,
    options: DecisionTreeOptions,
}

struct SplitScoringContext<'a> {
    table: &'a dyn TableAccess,
    class_indices: &'a [usize],
    num_classes: usize,
    criterion: Criterion,
    min_samples_leaf: usize,
}

#[derive(Debug, Clone)]
struct ObliviousLeafState {
    rows: Vec<usize>,
    class_index: usize,
}

fn train_oblivious_structure(
    table: &dyn TableAccess,
    class_indices: &[usize],
    class_labels: &[f64],
    criterion: Criterion,
    parallelism: Parallelism,
    options: DecisionTreeOptions,
) -> TreeStructure {
    let all_rows: Vec<usize> = (0..table.n_rows()).collect();
    let mut leaves = vec![ObliviousLeafState {
        class_index: majority_class(&all_rows, class_indices, class_labels.len()),
        rows: all_rows,
    }];
    let mut splits = Vec::new();

    for _depth in 0..options.max_depth {
        let best_split = if parallelism.enabled() {
            (0..table.binned_feature_count())
                .into_par_iter()
                .filter_map(|feature_index| {
                    score_oblivious_split(
                        table,
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
            (0..table.binned_feature_count())
                .filter_map(|feature_index| {
                    score_oblivious_split(
                        table,
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

        let next_leaves = leaves
            .into_iter()
            .flat_map(|leaf| {
                split_oblivious_leaf(
                    table,
                    class_indices,
                    class_labels.len(),
                    leaf,
                    best_split.feature_index,
                    best_split.threshold_bin,
                )
            })
            .collect();

        leaves = next_leaves;
        splits.push(ObliviousSplit {
            feature_index: best_split.feature_index,
            threshold_bin: best_split.threshold_bin,
        });
    }

    TreeStructure::Oblivious {
        splits,
        leaf_class_indices: leaves.into_iter().map(|leaf| leaf.class_index).collect(),
    }
}

#[derive(Debug, Clone, Copy)]
struct ObliviousSplitCandidate {
    feature_index: usize,
    threshold_bin: u16,
    score: f64,
}

fn score_oblivious_split(
    table: &dyn TableAccess,
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
            class_indices,
            feature_index,
            leaves,
            num_classes,
            criterion,
            min_samples_leaf,
        );
    }
    let feature_value = |row_idx: usize| table.binned_value(feature_index, row_idx);
    let candidate_thresholds = leaves
        .iter()
        .flat_map(|leaf| leaf.rows.iter().map(|row_idx| feature_value(*row_idx)))
        .collect::<BTreeSet<_>>();

    candidate_thresholds
        .into_iter()
        .filter_map(|threshold_bin| {
            let score = leaves.iter().fold(0.0, |score, leaf| {
                let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = leaf
                    .rows
                    .iter()
                    .copied()
                    .partition(|row_idx| feature_value(*row_idx) <= threshold_bin);

                if left_rows.len() < min_samples_leaf || right_rows.len() < min_samples_leaf {
                    return score;
                }

                let parent_counts = class_counts(&leaf.rows, class_indices, num_classes);
                let left_counts = class_counts(&left_rows, class_indices, num_classes);
                let right_counts = class_counts(&right_rows, class_indices, num_classes);

                let weighted_parent_impurity = leaf.rows.len() as f64
                    * classification_impurity(&parent_counts, leaf.rows.len(), criterion);
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

fn split_oblivious_leaf(
    table: &dyn TableAccess,
    class_indices: &[usize],
    num_classes: usize,
    leaf: ObliviousLeafState,
    feature_index: usize,
    threshold_bin: u16,
) -> [ObliviousLeafState; 2] {
    let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = leaf
        .rows
        .into_iter()
        .partition(|row_idx| table.binned_value(feature_index, *row_idx) <= threshold_bin);

    let left_class_index = if left_rows.is_empty() {
        leaf.class_index
    } else {
        majority_class(&left_rows, class_indices, num_classes)
    };
    let right_class_index = if right_rows.is_empty() {
        leaf.class_index
    } else {
        majority_class(&right_rows, class_indices, num_classes)
    };

    [
        ObliviousLeafState {
            rows: left_rows,
            class_index: left_class_index,
        },
        ObliviousLeafState {
            rows: right_rows,
            class_index: right_class_index,
        },
    ]
}

fn score_split(
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
        DecisionTreeAlgorithm::Oblivious => None,
    }
}

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

fn score_cart_split(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<SplitCandidate> {
    if context.table.is_binary_binned_feature(feature_index) {
        return score_binary_cart_split(context, feature_index, rows);
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

fn score_binary_oblivious_split(
    table: &dyn TableAccess,
    class_indices: &[usize],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
    num_classes: usize,
    criterion: Criterion,
    min_samples_leaf: usize,
) -> Option<ObliviousSplitCandidate> {
    let score = leaves.iter().fold(0.0, |score, leaf| {
        let (left_rows, right_rows): (Vec<usize>, Vec<usize>) =
            leaf.rows.iter().copied().partition(|row_idx| {
                !table
                    .binned_boolean_value(feature_index, *row_idx)
                    .expect("binary feature must expose boolean values")
            });

        if left_rows.len() < min_samples_leaf || right_rows.len() < min_samples_leaf {
            return score;
        }

        let parent_counts = class_counts(&leaf.rows, class_indices, num_classes);
        let left_counts = class_counts(&left_rows, class_indices, num_classes);
        let right_counts = class_counts(&right_rows, class_indices, num_classes);
        let weighted_parent_impurity = leaf.rows.len() as f64
            * classification_impurity(&parent_counts, leaf.rows.len(), criterion);
        let weighted_children_impurity = left_rows.len() as f64
            * classification_impurity(&left_counts, left_rows.len(), criterion)
            + right_rows.len() as f64
                * classification_impurity(&right_counts, right_rows.len(), criterion);

        score + (weighted_parent_impurity - weighted_children_impurity)
    });

    (score > 0.0).then_some(ObliviousSplitCandidate {
        feature_index,
        threshold_bin: 0,
        score,
    })
}

fn class_counts(rows: &[usize], class_indices: &[usize], num_classes: usize) -> Vec<usize> {
    rows.iter()
        .fold(vec![0usize; num_classes], |mut counts, row_idx| {
            counts[class_indices[*row_idx]] += 1;
            counts
        })
}

fn majority_class(rows: &[usize], class_indices: &[usize], num_classes: usize) -> usize {
    class_counts(rows, class_indices, num_classes)
        .into_iter()
        .enumerate()
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(class_index, _count)| class_index)
        .unwrap_or(0)
}

fn is_pure(rows: &[usize], class_indices: &[usize]) -> bool {
    rows.first().is_none_or(|first_row| {
        rows.iter()
            .all(|row_idx| class_indices[*row_idx] == class_indices[*first_row])
    })
}

fn entropy(counts: &[usize], total: usize) -> f64 {
    counts
        .iter()
        .copied()
        .filter(|count| *count > 0)
        .map(|count| {
            let probability = count as f64 / total as f64;
            -probability * probability.log2()
        })
        .sum()
}

fn gini(counts: &[usize], total: usize) -> f64 {
    1.0 - counts
        .iter()
        .copied()
        .map(|count| {
            let probability = count as f64 / total as f64;
            probability * probability
        })
        .sum::<f64>()
}

fn classification_impurity(counts: &[usize], total: usize, criterion: Criterion) -> f64 {
    match criterion {
        Criterion::Entropy => entropy(counts, total),
        Criterion::Gini => gini(counts, total),
        _ => unreachable!("classification impurity only supports gini or entropy"),
    }
}

fn split_score(candidate: &SplitCandidate) -> f64 {
    match candidate {
        SplitCandidate::Multiway { score, .. } | SplitCandidate::Binary { score, .. } => *score,
    }
}

fn split_feature_index(candidate: &SplitCandidate) -> usize {
    match candidate {
        SplitCandidate::Multiway { feature_index, .. }
        | SplitCandidate::Binary { feature_index, .. } => *feature_index,
    }
}

fn push_leaf(nodes: &mut Vec<TreeNode>, class_index: usize) -> usize {
    push_node(nodes, TreeNode::Leaf { class_index })
}

fn push_node(nodes: &mut Vec<TreeNode>, node: TreeNode) -> usize {
    nodes.push(node);
    nodes.len() - 1
}

#[derive(Debug, Clone, Copy)]
enum MultiwayMetric {
    InformationGain,
    GainRatio,
}

#[cfg(test)]
mod tests {
    use super::*;
    use forestfire_data::DenseTable;

    fn and_table() -> DenseTable {
        DenseTable::new(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap()
    }

    fn criterion_choice_table() -> DenseTable {
        DenseTable::with_canaries(
            vec![
                vec![0.0, 1.0],
                vec![4.0, 1.0],
                vec![4.0, 0.0],
                vec![0.0, 1.0],
                vec![5.0, 2.0],
                vec![2.0, 4.0],
                vec![1.0, 2.0],
            ],
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            0,
        )
        .unwrap()
    }

    fn canary_target_table() -> DenseTable {
        let x: Vec<Vec<f64>> = (0..8).map(|value| vec![value as f64]).collect();
        let probe = DenseTable::with_canaries(x.clone(), vec![0.0; 8], 1).unwrap();
        let canary_index = probe.n_features();
        let y = (0..probe.n_rows())
            .map(|row_idx| {
                if probe.binned_value(canary_index, row_idx) > 255 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        DenseTable::with_canaries(x, y, 1).unwrap()
    }

    #[test]
    fn id3_fits_basic_boolean_pattern() {
        let table = and_table();
        let model = train_id3(&table).unwrap();

        assert_eq!(model.algorithm(), DecisionTreeAlgorithm::Id3);
        assert_eq!(model.criterion(), Criterion::Entropy);
        assert_eq!(model.predict_table(&table), table_targets(&table));
    }

    #[test]
    fn c45_fits_basic_boolean_pattern() {
        let table = and_table();
        let model = train_c45(&table).unwrap();

        assert_eq!(model.algorithm(), DecisionTreeAlgorithm::C45);
        assert_eq!(model.criterion(), Criterion::Entropy);
        assert_eq!(model.predict_table(&table), table_targets(&table));
    }

    #[test]
    fn cart_fits_basic_boolean_pattern() {
        let table = and_table();
        let model = train_cart(&table).unwrap();

        assert_eq!(model.algorithm(), DecisionTreeAlgorithm::Cart);
        assert_eq!(model.criterion(), Criterion::Gini);
        assert_eq!(model.predict_table(&table), table_targets(&table));
    }

    #[test]
    fn oblivious_fits_basic_boolean_pattern() {
        let table = and_table();
        let model = train_oblivious(&table).unwrap();

        assert_eq!(model.algorithm(), DecisionTreeAlgorithm::Oblivious);
        assert_eq!(model.criterion(), Criterion::Gini);
        assert_eq!(model.predict_table(&table), table_targets(&table));
    }

    #[test]
    fn cart_can_choose_between_gini_and_entropy() {
        let table = criterion_choice_table();
        let options = DecisionTreeOptions {
            max_depth: 1,
            ..DecisionTreeOptions::default()
        };
        let gini_model = train_classifier(
            &table,
            DecisionTreeAlgorithm::Cart,
            Criterion::Gini,
            Parallelism::sequential(),
            options,
        )
        .unwrap();
        let entropy_model = train_classifier(
            &table,
            DecisionTreeAlgorithm::Cart,
            Criterion::Entropy,
            Parallelism::sequential(),
            options,
        )
        .unwrap();

        let root_feature = |model: &DecisionTreeClassifier| match &model.structure {
            TreeStructure::Standard { nodes, root } => match &nodes[*root] {
                TreeNode::BinarySplit { feature_index, .. } => *feature_index,
                node => panic!("expected binary root split, found {node:?}"),
            },
            TreeStructure::Oblivious { .. } => panic!("expected standard tree"),
        };

        assert_eq!(gini_model.criterion(), Criterion::Gini);
        assert_eq!(entropy_model.criterion(), Criterion::Entropy);
        assert_eq!(root_feature(&gini_model), 0);
        assert_eq!(root_feature(&entropy_model), 1);
    }

    #[test]
    fn rejects_non_finite_class_labels() {
        let table = DenseTable::new(vec![vec![0.0], vec![1.0]], vec![0.0, f64::NAN]).unwrap();

        let err = train_id3(&table).unwrap_err();
        assert!(matches!(
            err,
            DecisionTreeError::InvalidTargetValue { row: 1, value } if value.is_nan()
        ));
    }

    #[test]
    fn stops_standard_tree_growth_when_a_canary_wins() {
        let table = canary_target_table();
        for trainer in [train_id3, train_c45, train_cart] {
            let model = trainer(&table).unwrap();
            let preds = model.predict_table(&table);

            assert!(preds.iter().all(|pred| *pred == preds[0]));
            assert_ne!(preds, table_targets(&table));
        }
    }

    #[test]
    fn stops_oblivious_tree_growth_when_a_canary_wins() {
        let table = canary_target_table();
        let model = train_oblivious(&table).unwrap();
        let preds = model.predict_table(&table);

        assert!(preds.iter().all(|pred| *pred == preds[0]));
        assert_ne!(preds, table_targets(&table));
    }

    fn table_targets(table: &dyn TableAccess) -> Vec<f64> {
        (0..table.n_rows())
            .map(|row_idx| table.target_value(row_idx))
            .collect()
    }
}
