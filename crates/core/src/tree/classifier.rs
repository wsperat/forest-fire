use forestfire_data::DenseTable;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionTreeAlgorithm {
    Id3,
    C45,
    Cart,
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
    nodes: Vec<TreeNode>,
    root: usize,
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

fn train_classifier(
    train_set: &DenseTable,
    algorithm: DecisionTreeAlgorithm,
    options: DecisionTreeOptions,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    if train_set.n_rows() == 0 {
        return Err(DecisionTreeError::EmptyTarget);
    }

    let (class_labels, class_indices) = encode_class_labels(train_set)?;
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

    Ok(DecisionTreeClassifier {
        algorithm,
        class_labels,
        nodes,
        root,
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
        let mut node_index = self.root;

        loop {
            match &self.nodes[node_index] {
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

    let best_split = (0..context.table.n_features())
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
    fn rejects_non_finite_class_labels() {
        let table = DenseTable::new(vec![vec![0.0], vec![1.0]], vec![0.0, f64::NAN]).unwrap();

        let err = train_id3(&table).unwrap_err();
        assert!(matches!(
            err,
            DecisionTreeError::InvalidTargetValue { row: 1, value } if value.is_nan()
        ));
    }

    fn table_targets(table: &DenseTable) -> Vec<f64> {
        (0..table.n_rows())
            .map(|row_idx| table.target().value(row_idx))
            .collect()
    }
}
