use forestfire_data::DenseTable;
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

pub fn train_id3(train_set: &DenseTable) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_classifier(
        train_set,
        DecisionTreeAlgorithm::Id3,
        DecisionTreeOptions::default(),
    )
}

pub fn train_c45(train_set: &DenseTable) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_classifier(
        train_set,
        DecisionTreeAlgorithm::C45,
        DecisionTreeOptions::default(),
    )
}

pub fn train_cart(train_set: &DenseTable) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_classifier(
        train_set,
        DecisionTreeAlgorithm::Cart,
        DecisionTreeOptions::default(),
    )
}

pub fn train_oblivious(
    train_set: &DenseTable,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_classifier(
        train_set,
        DecisionTreeAlgorithm::Oblivious,
        DecisionTreeOptions::default(),
    )
}

fn train_classifier(
    train_set: &DenseTable,
    algorithm: DecisionTreeAlgorithm,
    options: DecisionTreeOptions,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    if train_set.n_rows() == 0 {
        return Err(DecisionTreeError::EmptyTarget);
    }

    let (class_labels, class_indices) = encode_class_labels(train_set)?;
    let structure = match algorithm {
        DecisionTreeAlgorithm::Oblivious => {
            train_oblivious_structure(train_set, &class_indices, &class_labels, options)
        }
        _ => {
            let mut nodes = Vec::new();
            let all_rows: Vec<usize> = (0..train_set.n_rows()).collect();
            let context = BuildContext {
                table: train_set,
                class_indices: &class_indices,
                class_labels: &class_labels,
                algorithm,
                options,
            };
            let root = build_node(&context, &mut nodes, &all_rows, 0);
            TreeStructure::Standard { nodes, root }
        }
    };

    Ok(DecisionTreeClassifier {
        algorithm,
        class_labels,
        structure,
    })
}

impl DecisionTreeClassifier {
    pub fn algorithm(&self) -> DecisionTreeAlgorithm {
        self.algorithm
    }

    pub fn predict_table(&self, table: &DenseTable) -> Vec<f64> {
        (0..table.n_rows())
            .map(|row_idx| self.predict_row(table, row_idx))
            .collect()
    }

    fn predict_row(&self, table: &DenseTable, row_idx: usize) -> f64 {
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
                            let bin = table.binned_feature_column(*feature_index).value(row_idx);
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
                            let bin = table.binned_feature_column(*feature_index).value(row_idx);
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
                    let go_right = table
                        .binned_feature_column(split.feature_index)
                        .value(row_idx)
                        > split.threshold_bin;
                    (leaf_index << 1) | usize::from(go_right)
                });

                self.class_labels[leaf_class_indices[leaf_index]]
            }
        }
    }
}

fn encode_class_labels(
    train_set: &DenseTable,
) -> Result<(Vec<f64>, Vec<usize>), DecisionTreeError> {
    let targets: Vec<f64> = (0..train_set.n_rows())
        .map(|row_idx| {
            let value = train_set.target().value(row_idx);
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

    let best_split = (0..context.table.binned_feature_count())
        .filter_map(|feature_index| {
            score_split(
                context.table,
                context.class_indices,
                feature_index,
                rows,
                context.class_labels.len(),
                context.algorithm,
                context.options.min_samples_leaf,
            )
        })
        .max_by(|left, right| split_score(left).total_cmp(&split_score(right)));

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
    table: &'a DenseTable,
    class_indices: &'a [usize],
    class_labels: &'a [f64],
    algorithm: DecisionTreeAlgorithm,
    options: DecisionTreeOptions,
}

#[derive(Debug, Clone)]
struct ObliviousLeafState {
    rows: Vec<usize>,
    class_index: usize,
}

fn train_oblivious_structure(
    table: &DenseTable,
    class_indices: &[usize],
    class_labels: &[f64],
    options: DecisionTreeOptions,
) -> TreeStructure {
    let all_rows: Vec<usize> = (0..table.n_rows()).collect();
    let mut leaves = vec![ObliviousLeafState {
        class_index: majority_class(&all_rows, class_indices, class_labels.len()),
        rows: all_rows,
    }];
    let mut splits = Vec::new();

    for _depth in 0..options.max_depth {
        let best_split = (0..table.binned_feature_count())
            .filter_map(|feature_index| {
                score_oblivious_split(
                    table,
                    class_indices,
                    feature_index,
                    &leaves,
                    class_labels.len(),
                    options.min_samples_leaf,
                )
            })
            .max_by(|left, right| left.score.total_cmp(&right.score));

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
    table: &DenseTable,
    class_indices: &[usize],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
    num_classes: usize,
    min_samples_leaf: usize,
) -> Option<ObliviousSplitCandidate> {
    let feature = table.binned_feature_column(feature_index);
    let candidate_thresholds = leaves
        .iter()
        .flat_map(|leaf| leaf.rows.iter().map(|row_idx| feature.value(*row_idx)))
        .collect::<BTreeSet<_>>();

    candidate_thresholds
        .into_iter()
        .filter_map(|threshold_bin| {
            let score = leaves.iter().fold(0.0, |score, leaf| {
                let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = leaf
                    .rows
                    .iter()
                    .copied()
                    .partition(|row_idx| feature.value(*row_idx) <= threshold_bin);

                if left_rows.len() < min_samples_leaf || right_rows.len() < min_samples_leaf {
                    return score;
                }

                let parent_counts = class_counts(&leaf.rows, class_indices, num_classes);
                let left_counts = class_counts(&left_rows, class_indices, num_classes);
                let right_counts = class_counts(&right_rows, class_indices, num_classes);

                let weighted_parent_gini =
                    leaf.rows.len() as f64 * gini(&parent_counts, leaf.rows.len());
                let weighted_children_gini = left_rows.len() as f64
                    * gini(&left_counts, left_rows.len())
                    + right_rows.len() as f64 * gini(&right_counts, right_rows.len());

                score + (weighted_parent_gini - weighted_children_gini)
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
    table: &DenseTable,
    class_indices: &[usize],
    num_classes: usize,
    leaf: ObliviousLeafState,
    feature_index: usize,
    threshold_bin: u16,
) -> [ObliviousLeafState; 2] {
    let feature = table.binned_feature_column(feature_index);
    let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = leaf
        .rows
        .into_iter()
        .partition(|row_idx| feature.value(*row_idx) <= threshold_bin);

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
    table: &DenseTable,
    class_indices: &[usize],
    feature_index: usize,
    rows: &[usize],
    num_classes: usize,
    algorithm: DecisionTreeAlgorithm,
    min_samples_leaf: usize,
) -> Option<SplitCandidate> {
    match algorithm {
        DecisionTreeAlgorithm::Id3 => score_multiway_split(
            table,
            class_indices,
            feature_index,
            rows,
            num_classes,
            MultiwayMetric::InformationGain,
            min_samples_leaf,
        ),
        DecisionTreeAlgorithm::C45 => score_multiway_split(
            table,
            class_indices,
            feature_index,
            rows,
            num_classes,
            MultiwayMetric::GainRatio,
            min_samples_leaf,
        ),
        DecisionTreeAlgorithm::Cart => score_cart_split(
            table,
            class_indices,
            feature_index,
            rows,
            num_classes,
            min_samples_leaf,
        ),
        DecisionTreeAlgorithm::Oblivious => None,
    }
}

fn score_multiway_split(
    table: &DenseTable,
    class_indices: &[usize],
    feature_index: usize,
    rows: &[usize],
    num_classes: usize,
    metric: MultiwayMetric,
    min_samples_leaf: usize,
) -> Option<SplitCandidate> {
    let feature = table.binned_feature_column(feature_index);
    let grouped_rows =
        rows.iter()
            .fold(BTreeMap::<u16, Vec<usize>>::new(), |mut groups, row_idx| {
                groups
                    .entry(feature.value(*row_idx))
                    .or_default()
                    .push(*row_idx);
                groups
            });

    if grouped_rows.len() <= 1
        || grouped_rows
            .values()
            .any(|group| group.len() < min_samples_leaf)
    {
        return None;
    }

    let parent_counts = class_counts(rows, class_indices, num_classes);
    let parent_entropy = entropy(&parent_counts, rows.len());
    let weighted_child_entropy = grouped_rows
        .values()
        .map(|group_rows| {
            let counts = class_counts(group_rows, class_indices, num_classes);
            (group_rows.len() as f64 / rows.len() as f64) * entropy(&counts, group_rows.len())
        })
        .sum::<f64>();
    let information_gain = parent_entropy - weighted_child_entropy;

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
    table: &DenseTable,
    class_indices: &[usize],
    feature_index: usize,
    rows: &[usize],
    num_classes: usize,
    min_samples_leaf: usize,
) -> Option<SplitCandidate> {
    let feature = table.binned_feature_column(feature_index);
    let parent_counts = class_counts(rows, class_indices, num_classes);
    let parent_gini = gini(&parent_counts, rows.len());

    rows.iter()
        .map(|row_idx| feature.value(*row_idx))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|threshold_bin| {
            let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = rows
                .iter()
                .copied()
                .partition(|row_idx| feature.value(*row_idx) <= threshold_bin);

            if left_rows.len() < min_samples_leaf || right_rows.len() < min_samples_leaf {
                return None;
            }

            let left_counts = class_counts(&left_rows, class_indices, num_classes);
            let right_counts = class_counts(&right_rows, class_indices, num_classes);
            let weighted_gini = (left_rows.len() as f64 / rows.len() as f64)
                * gini(&left_counts, left_rows.len())
                + (right_rows.len() as f64 / rows.len() as f64)
                    * gini(&right_counts, right_rows.len());

            Some(SplitCandidate::Binary {
                feature_index,
                score: parent_gini - weighted_gini,
                threshold_bin,
                left_rows,
                right_rows,
            })
        })
        .max_by(|left, right| split_score(left).total_cmp(&split_score(right)))
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

    fn canary_target_table() -> DenseTable {
        let x: Vec<Vec<f64>> = (0..8).map(|value| vec![value as f64]).collect();
        let probe = DenseTable::with_canaries(x.clone(), vec![0.0; 8], 1).unwrap();
        let canary_index = probe.n_features();
        let y = (0..probe.n_rows())
            .map(|row_idx| {
                if probe.binned_feature_column(canary_index).value(row_idx) > 255 {
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
        assert_eq!(model.predict_table(&table), table_targets(&table));
    }

    #[test]
    fn c45_fits_basic_boolean_pattern() {
        let table = and_table();
        let model = train_c45(&table).unwrap();

        assert_eq!(model.algorithm(), DecisionTreeAlgorithm::C45);
        assert_eq!(model.predict_table(&table), table_targets(&table));
    }

    #[test]
    fn cart_fits_basic_boolean_pattern() {
        let table = and_table();
        let model = train_cart(&table).unwrap();

        assert_eq!(model.algorithm(), DecisionTreeAlgorithm::Cart);
        assert_eq!(model.predict_table(&table), table_targets(&table));
    }

    #[test]
    fn oblivious_fits_basic_boolean_pattern() {
        let table = and_table();
        let model = train_oblivious(&table).unwrap();

        assert_eq!(model.algorithm(), DecisionTreeAlgorithm::Oblivious);
        assert_eq!(model.predict_table(&table), table_targets(&table));
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
        let model = train_id3(&table).unwrap();
        let preds = model.predict_table(&table);

        assert!(preds.iter().all(|pred| *pred == preds[0]));
        assert_ne!(preds, table_targets(&table));
    }

    #[test]
    fn stops_oblivious_tree_growth_when_a_canary_wins() {
        let table = canary_target_table();
        let model = train_oblivious(&table).unwrap();
        let preds = model.predict_table(&table);

        assert!(preds.iter().all(|pred| *pred == preds[0]));
        assert_ne!(preds, table_targets(&table));
    }

    fn table_targets(table: &DenseTable) -> Vec<f64> {
        (0..table.n_rows())
            .map(|row_idx| table.target().value(row_idx))
            .collect()
    }
}
