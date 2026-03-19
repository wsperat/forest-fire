use crate::{Criterion, Parallelism};
use forestfire_data::DenseTable;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionTreeAlgorithm {
    Cart,
    Oblivious,
}

#[derive(Debug, Clone, Copy)]
pub struct RegressionTreeOptions {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
}

impl Default for RegressionTreeOptions {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }
}

#[derive(Debug)]
pub enum RegressionTreeError {
    EmptyTarget,
    InvalidTargetValue { row: usize, value: f64 },
}

impl Display for RegressionTreeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            RegressionTreeError::EmptyTarget => {
                write!(f, "Cannot train on an empty target vector.")
            }
            RegressionTreeError::InvalidTargetValue { row, value } => write!(
                f,
                "Regression targets must be finite values. Found {} at row {}.",
                value, row
            ),
        }
    }
}

impl Error for RegressionTreeError {}

#[derive(Debug, Clone)]
pub struct DecisionTreeRegressor {
    algorithm: RegressionTreeAlgorithm,
    criterion: Criterion,
    structure: RegressionTreeStructure,
}

#[derive(Debug, Clone)]
enum RegressionTreeStructure {
    Standard {
        nodes: Vec<RegressionNode>,
        root: usize,
    },
    Oblivious {
        splits: Vec<ObliviousSplit>,
        leaf_values: Vec<f64>,
    },
}

#[derive(Debug, Clone)]
enum RegressionNode {
    Leaf {
        value: f64,
    },
    BinarySplit {
        feature_index: usize,
        threshold_bin: u16,
        left_child: usize,
        right_child: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct ObliviousSplit {
    feature_index: usize,
    threshold_bin: u16,
}

#[derive(Debug, Clone)]
struct RegressionSplitCandidate {
    feature_index: usize,
    threshold_bin: u16,
    score: f64,
    left_rows: Vec<usize>,
    right_rows: Vec<usize>,
}

#[derive(Debug, Clone)]
struct ObliviousLeafState {
    rows: Vec<usize>,
    value: f64,
}

#[derive(Debug, Clone, Copy)]
struct ObliviousSplitCandidate {
    feature_index: usize,
    threshold_bin: u16,
    score: f64,
}

pub fn train_cart_regressor(
    train_set: &DenseTable,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_cart_regressor_with_criterion(train_set, Criterion::Mean)
}

pub fn train_cart_regressor_with_criterion(
    train_set: &DenseTable,
    criterion: Criterion,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_cart_regressor_with_criterion_and_parallelism(
        train_set,
        criterion,
        Parallelism::sequential(),
    )
}

pub(crate) fn train_cart_regressor_with_criterion_and_parallelism(
    train_set: &DenseTable,
    criterion: Criterion,
    parallelism: Parallelism,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_regressor(
        train_set,
        RegressionTreeAlgorithm::Cart,
        criterion,
        parallelism,
        RegressionTreeOptions::default(),
    )
}

pub fn train_oblivious_regressor(
    train_set: &DenseTable,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_oblivious_regressor_with_criterion(train_set, Criterion::Mean)
}

pub fn train_oblivious_regressor_with_criterion(
    train_set: &DenseTable,
    criterion: Criterion,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_oblivious_regressor_with_criterion_and_parallelism(
        train_set,
        criterion,
        Parallelism::sequential(),
    )
}

pub(crate) fn train_oblivious_regressor_with_criterion_and_parallelism(
    train_set: &DenseTable,
    criterion: Criterion,
    parallelism: Parallelism,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_regressor(
        train_set,
        RegressionTreeAlgorithm::Oblivious,
        criterion,
        parallelism,
        RegressionTreeOptions::default(),
    )
}

fn train_regressor(
    train_set: &DenseTable,
    algorithm: RegressionTreeAlgorithm,
    criterion: Criterion,
    parallelism: Parallelism,
    options: RegressionTreeOptions,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    if train_set.n_rows() == 0 {
        return Err(RegressionTreeError::EmptyTarget);
    }

    let targets = finite_targets(train_set)?;
    let structure = match algorithm {
        RegressionTreeAlgorithm::Cart => {
            let mut nodes = Vec::new();
            let all_rows: Vec<usize> = (0..train_set.n_rows()).collect();
            let context = BuildContext {
                table: train_set,
                targets: &targets,
                criterion,
                parallelism,
                options,
            };
            let root = build_node(&context, &mut nodes, &all_rows, 0);
            RegressionTreeStructure::Standard { nodes, root }
        }
        RegressionTreeAlgorithm::Oblivious => {
            train_oblivious_structure(train_set, &targets, criterion, parallelism, options)
        }
    };

    Ok(DecisionTreeRegressor {
        algorithm,
        criterion,
        structure,
    })
}

impl DecisionTreeRegressor {
    pub fn algorithm(&self) -> RegressionTreeAlgorithm {
        self.algorithm
    }

    pub fn criterion(&self) -> Criterion {
        self.criterion
    }

    pub fn predict_table(&self, table: &DenseTable) -> Vec<f64> {
        (0..table.n_rows())
            .map(|row_idx| self.predict_row(table, row_idx))
            .collect()
    }

    fn predict_row(&self, table: &DenseTable, row_idx: usize) -> f64 {
        match &self.structure {
            RegressionTreeStructure::Standard { nodes, root } => {
                let mut node_index = *root;

                loop {
                    match &nodes[node_index] {
                        RegressionNode::Leaf { value } => return *value,
                        RegressionNode::BinarySplit {
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
            RegressionTreeStructure::Oblivious {
                splits,
                leaf_values,
            } => {
                let leaf_index = splits.iter().fold(0usize, |leaf_index, split| {
                    let go_right =
                        table.binned_value(split.feature_index, row_idx) > split.threshold_bin;
                    (leaf_index << 1) | usize::from(go_right)
                });

                leaf_values[leaf_index]
            }
        }
    }
}

struct BuildContext<'a> {
    table: &'a DenseTable,
    targets: &'a [f64],
    criterion: Criterion,
    parallelism: Parallelism,
    options: RegressionTreeOptions,
}

fn finite_targets(train_set: &DenseTable) -> Result<Vec<f64>, RegressionTreeError> {
    (0..train_set.n_rows())
        .map(|row_idx| {
            let value = train_set.target().value(row_idx);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(RegressionTreeError::InvalidTargetValue {
                    row: row_idx,
                    value,
                })
            }
        })
        .collect()
}

fn build_node(
    context: &BuildContext<'_>,
    nodes: &mut Vec<RegressionNode>,
    rows: &[usize],
    depth: usize,
) -> usize {
    let leaf_value = regression_value(rows, context.targets, context.criterion);

    if rows.is_empty()
        || depth >= context.options.max_depth
        || rows.len() < context.options.min_samples_split
        || has_constant_target(rows, context.targets)
    {
        return push_leaf(nodes, leaf_value);
    }

    let best_split = if context.parallelism.enabled() {
        (0..context.table.binned_feature_count())
            .into_par_iter()
            .filter_map(|feature_index| {
                score_split(
                    context.table,
                    context.targets,
                    feature_index,
                    rows,
                    context.criterion,
                    context.options.min_samples_leaf,
                )
            })
            .max_by(|left, right| left.score.total_cmp(&right.score))
    } else {
        (0..context.table.binned_feature_count())
            .filter_map(|feature_index| {
                score_split(
                    context.table,
                    context.targets,
                    feature_index,
                    rows,
                    context.criterion,
                    context.options.min_samples_leaf,
                )
            })
            .max_by(|left, right| left.score.total_cmp(&right.score))
    };

    match best_split {
        Some(best_split)
            if context
                .table
                .is_canary_binned_feature(best_split.feature_index) =>
        {
            push_leaf(nodes, leaf_value)
        }
        Some(RegressionSplitCandidate {
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
                RegressionNode::BinarySplit {
                    feature_index,
                    threshold_bin,
                    left_child,
                    right_child,
                },
            )
        }
        _ => push_leaf(nodes, leaf_value),
    }
}

fn train_oblivious_structure(
    table: &DenseTable,
    targets: &[f64],
    criterion: Criterion,
    parallelism: Parallelism,
    options: RegressionTreeOptions,
) -> RegressionTreeStructure {
    let all_rows: Vec<usize> = (0..table.n_rows()).collect();
    let mut leaves = vec![ObliviousLeafState {
        value: regression_value(&all_rows, targets, criterion),
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
                        targets,
                        feature_index,
                        &leaves,
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
                        targets,
                        feature_index,
                        &leaves,
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

        leaves = leaves
            .into_iter()
            .flat_map(|leaf| {
                split_oblivious_leaf(
                    table,
                    targets,
                    leaf,
                    best_split.feature_index,
                    best_split.threshold_bin,
                    criterion,
                )
            })
            .collect();
        splits.push(ObliviousSplit {
            feature_index: best_split.feature_index,
            threshold_bin: best_split.threshold_bin,
        });
    }

    RegressionTreeStructure::Oblivious {
        splits,
        leaf_values: leaves.into_iter().map(|leaf| leaf.value).collect(),
    }
}

fn score_split(
    table: &DenseTable,
    targets: &[f64],
    feature_index: usize,
    rows: &[usize],
    criterion: Criterion,
    min_samples_leaf: usize,
) -> Option<RegressionSplitCandidate> {
    if table.is_binary_binned_feature(feature_index) {
        return score_binary_split(
            table,
            targets,
            feature_index,
            rows,
            criterion,
            min_samples_leaf,
        );
    }
    let parent_loss = regression_loss(rows, targets, criterion);

    rows.iter()
        .map(|row_idx| table.binned_value(feature_index, *row_idx))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|threshold_bin| {
            let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = rows
                .iter()
                .copied()
                .partition(|row_idx| table.binned_value(feature_index, *row_idx) <= threshold_bin);

            if left_rows.len() < min_samples_leaf || right_rows.len() < min_samples_leaf {
                return None;
            }

            let score = parent_loss
                - (regression_loss(&left_rows, targets, criterion)
                    + regression_loss(&right_rows, targets, criterion));

            Some(RegressionSplitCandidate {
                feature_index,
                threshold_bin,
                score,
                left_rows,
                right_rows,
            })
        })
        .max_by(|left, right| left.score.total_cmp(&right.score))
}

fn score_oblivious_split(
    table: &DenseTable,
    targets: &[f64],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
    criterion: Criterion,
    min_samples_leaf: usize,
) -> Option<ObliviousSplitCandidate> {
    if table.is_binary_binned_feature(feature_index) {
        return score_binary_oblivious_split(
            table,
            targets,
            feature_index,
            leaves,
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

                score + regression_loss(&leaf.rows, targets, criterion)
                    - (regression_loss(&left_rows, targets, criterion)
                        + regression_loss(&right_rows, targets, criterion))
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
    targets: &[f64],
    leaf: ObliviousLeafState,
    feature_index: usize,
    threshold_bin: u16,
    criterion: Criterion,
) -> [ObliviousLeafState; 2] {
    let fallback_value = leaf.value;
    let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = leaf
        .rows
        .into_iter()
        .partition(|row_idx| table.binned_value(feature_index, *row_idx) <= threshold_bin);

    [
        ObliviousLeafState {
            value: if left_rows.is_empty() {
                fallback_value
            } else {
                regression_value(&left_rows, targets, criterion)
            },
            rows: left_rows,
        },
        ObliviousLeafState {
            value: if right_rows.is_empty() {
                fallback_value
            } else {
                regression_value(&right_rows, targets, criterion)
            },
            rows: right_rows,
        },
    ]
}

fn mean(rows: &[usize], targets: &[f64]) -> f64 {
    if rows.is_empty() {
        0.0
    } else {
        rows.iter().map(|row_idx| targets[*row_idx]).sum::<f64>() / rows.len() as f64
    }
}

fn median(rows: &[usize], targets: &[f64]) -> f64 {
    let mut values: Vec<f64> = rows.iter().map(|row_idx| targets[*row_idx]).collect();
    values.sort_by(|left, right| left.total_cmp(right));

    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn sum_squared_error(rows: &[usize], targets: &[f64]) -> f64 {
    let mean = mean(rows, targets);
    rows.iter()
        .map(|row_idx| {
            let diff = targets[*row_idx] - mean;
            diff * diff
        })
        .sum()
}

fn sum_absolute_error(rows: &[usize], targets: &[f64]) -> f64 {
    let median = median(rows, targets);
    rows.iter()
        .map(|row_idx| (targets[*row_idx] - median).abs())
        .sum()
}

fn regression_value(rows: &[usize], targets: &[f64], criterion: Criterion) -> f64 {
    match criterion {
        Criterion::Mean => mean(rows, targets),
        Criterion::Median => median(rows, targets),
        _ => unreachable!("regression criterion only supports mean or median"),
    }
}

fn regression_loss(rows: &[usize], targets: &[f64], criterion: Criterion) -> f64 {
    match criterion {
        Criterion::Mean => sum_squared_error(rows, targets),
        Criterion::Median => sum_absolute_error(rows, targets),
        _ => unreachable!("regression criterion only supports mean or median"),
    }
}

fn score_binary_split(
    table: &DenseTable,
    targets: &[f64],
    feature_index: usize,
    rows: &[usize],
    criterion: Criterion,
    min_samples_leaf: usize,
) -> Option<RegressionSplitCandidate> {
    let (left_rows, right_rows): (Vec<usize>, Vec<usize>) =
        rows.iter().copied().partition(|row_idx| {
            !table
                .binned_boolean_value(feature_index, *row_idx)
                .expect("binary feature must expose boolean values")
        });

    if left_rows.len() < min_samples_leaf || right_rows.len() < min_samples_leaf {
        return None;
    }

    let parent_loss = regression_loss(rows, targets, criterion);
    let score = parent_loss
        - (regression_loss(&left_rows, targets, criterion)
            + regression_loss(&right_rows, targets, criterion));

    Some(RegressionSplitCandidate {
        feature_index,
        threshold_bin: 0,
        score,
        left_rows,
        right_rows,
    })
}

fn score_binary_oblivious_split(
    table: &DenseTable,
    targets: &[f64],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
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

        score + regression_loss(&leaf.rows, targets, criterion)
            - (regression_loss(&left_rows, targets, criterion)
                + regression_loss(&right_rows, targets, criterion))
    });

    (score > 0.0).then_some(ObliviousSplitCandidate {
        feature_index,
        threshold_bin: 0,
        score,
    })
}

fn has_constant_target(rows: &[usize], targets: &[f64]) -> bool {
    rows.first().is_none_or(|first_row| {
        rows.iter()
            .all(|row_idx| targets[*row_idx] == targets[*first_row])
    })
}

fn push_leaf(nodes: &mut Vec<RegressionNode>, value: f64) -> usize {
    push_node(nodes, RegressionNode::Leaf { value })
}

fn push_node(nodes: &mut Vec<RegressionNode>, node: RegressionNode) -> usize {
    nodes.push(node);
    nodes.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic_table() -> DenseTable {
        DenseTable::with_canaries(
            vec![
                vec![0.0],
                vec![1.0],
                vec![2.0],
                vec![3.0],
                vec![4.0],
                vec![5.0],
            ],
            vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
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
                    100.0
                } else {
                    -100.0
                }
            })
            .collect();

        DenseTable::with_canaries(x, y, 1).unwrap()
    }

    #[test]
    fn cart_regressor_fits_basic_numeric_pattern() {
        let table = quadratic_table();
        let model = train_cart_regressor(&table).unwrap();
        let preds = model.predict_table(&table);

        assert_eq!(model.algorithm(), RegressionTreeAlgorithm::Cart);
        assert_eq!(model.criterion(), Criterion::Mean);
        assert_eq!(preds, table_targets(&table));
    }

    #[test]
    fn oblivious_regressor_fits_basic_numeric_pattern() {
        let table = quadratic_table();
        let model = train_oblivious_regressor(&table).unwrap();
        let preds = model.predict_table(&table);

        assert_eq!(model.algorithm(), RegressionTreeAlgorithm::Oblivious);
        assert_eq!(model.criterion(), Criterion::Mean);
        assert_eq!(preds, table_targets(&table));
    }

    #[test]
    fn regressors_can_choose_between_mean_and_median() {
        let table = DenseTable::with_canaries(
            vec![vec![0.0], vec![0.0], vec![0.0]],
            vec![0.0, 0.0, 100.0],
            0,
        )
        .unwrap();

        let mean_model = train_cart_regressor_with_criterion(&table, Criterion::Mean).unwrap();
        let median_model = train_cart_regressor_with_criterion(&table, Criterion::Median).unwrap();

        assert_eq!(mean_model.criterion(), Criterion::Mean);
        assert_eq!(median_model.criterion(), Criterion::Median);
        assert_eq!(
            mean_model.predict_table(&table),
            vec![100.0 / 3.0, 100.0 / 3.0, 100.0 / 3.0]
        );
        assert_eq!(median_model.predict_table(&table), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn rejects_non_finite_targets() {
        let table = DenseTable::new(vec![vec![0.0], vec![1.0]], vec![0.0, f64::NAN]).unwrap();

        let err = train_cart_regressor(&table).unwrap_err();
        assert!(matches!(
            err,
            RegressionTreeError::InvalidTargetValue { row: 1, value } if value.is_nan()
        ));
    }

    #[test]
    fn stops_cart_regressor_growth_when_a_canary_wins() {
        let table = canary_target_table();
        let model = train_cart_regressor(&table).unwrap();
        let preds = model.predict_table(&table);

        assert!(preds.iter().all(|pred| *pred == preds[0]));
        assert_ne!(preds, table_targets(&table));
    }

    #[test]
    fn stops_oblivious_regressor_growth_when_a_canary_wins() {
        let table = canary_target_table();
        let model = train_oblivious_regressor(&table).unwrap();
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
