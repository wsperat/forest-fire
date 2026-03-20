#![allow(dead_code)]

use crate::sampling::sample_feature_subset;
use crate::tree::regressor::{
    DecisionTreeRegressor, ObliviousSplit, RegressionNode, RegressionTreeAlgorithm,
    RegressionTreeOptions, RegressionTreeStructure,
};
use crate::{Criterion, Parallelism, capture_feature_preprocessing};
use forestfire_data::TableAccess;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy)]
pub struct SecondOrderRegressionTreeOptions {
    pub tree_options: RegressionTreeOptions,
    pub l2_regularization: f64,
    pub min_sum_hessian_in_leaf: f64,
    pub min_gain_to_split: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct SecondOrderRegressionTreeTrainingResult {
    pub(crate) model: DecisionTreeRegressor,
    pub(crate) root_canary_selected: bool,
}

impl Default for SecondOrderRegressionTreeOptions {
    fn default() -> Self {
        Self {
            tree_options: RegressionTreeOptions::default(),
            l2_regularization: 1.0,
            min_sum_hessian_in_leaf: 1e-3,
            min_gain_to_split: 0.0,
        }
    }
}

#[derive(Debug)]
pub enum SecondOrderRegressionTreeError {
    EmptyRows,
    GradientLengthMismatch { expected: usize, actual: usize },
    HessianLengthMismatch { expected: usize, actual: usize },
    InvalidGradientValue { row: usize, value: f64 },
    InvalidHessianValue { row: usize, value: f64 },
    NegativeHessian { row: usize, value: f64 },
    InvalidL2Regularization(f64),
    InvalidMinSumHessianInLeaf(f64),
    InvalidMinGainToSplit(f64),
}

impl Display for SecondOrderRegressionTreeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRows => write!(f, "Cannot train on an empty table."),
            Self::GradientLengthMismatch { expected, actual } => write!(
                f,
                "Gradient length mismatch: expected {} values, found {}.",
                expected, actual
            ),
            Self::HessianLengthMismatch { expected, actual } => write!(
                f,
                "Hessian length mismatch: expected {} values, found {}.",
                expected, actual
            ),
            Self::InvalidGradientValue { row, value } => write!(
                f,
                "Gradients must be finite values. Found {} at row {}.",
                value, row
            ),
            Self::InvalidHessianValue { row, value } => write!(
                f,
                "Hessians must be finite values. Found {} at row {}.",
                value, row
            ),
            Self::NegativeHessian { row, value } => write!(
                f,
                "Hessians must be non-negative. Found {} at row {}.",
                value, row
            ),
            Self::InvalidL2Regularization(value) => {
                write!(
                    f,
                    "l2_regularization must be finite and non-negative. Found {}.",
                    value
                )
            }
            Self::InvalidMinSumHessianInLeaf(value) => write!(
                f,
                "min_sum_hessian_in_leaf must be finite and non-negative. Found {}.",
                value
            ),
            Self::InvalidMinGainToSplit(value) => write!(
                f,
                "min_gain_to_split must be finite and non-negative. Found {}.",
                value
            ),
        }
    }
}

impl Error for SecondOrderRegressionTreeError {}

#[derive(Clone, Copy)]
struct BuildContext<'a> {
    table: &'a dyn TableAccess,
    gradients: &'a [f64],
    hessians: &'a [f64],
    parallelism: Parallelism,
    algorithm: RegressionTreeAlgorithm,
    options: SecondOrderRegressionTreeOptions,
}

#[derive(Debug, Clone, Copy)]
struct NodeStats {
    gradient_sum: f64,
    hessian_sum: f64,
}

#[derive(Debug, Clone, Copy)]
struct SplitChoice {
    feature_index: usize,
    threshold_bin: u16,
    gain: f64,
}

#[derive(Debug, Clone)]
enum FeatureHistogram {
    Binary {
        false_count: usize,
        false_gradient_sum: f64,
        false_hessian_sum: f64,
        true_count: usize,
        true_gradient_sum: f64,
        true_hessian_sum: f64,
    },
    Numeric {
        bin_count: Vec<usize>,
        bin_gradient_sum: Vec<f64>,
        bin_hessian_sum: Vec<f64>,
        observed_bins: Vec<usize>,
    },
}

#[derive(Debug, Clone, Copy)]
struct ObliviousLeafState {
    start: usize,
    end: usize,
    gradient_sum: f64,
    hessian_sum: f64,
}

impl ObliviousLeafState {
    fn len(&self) -> usize {
        self.end - self.start
    }

    fn prediction(&self, options: SecondOrderRegressionTreeOptions) -> f64 {
        leaf_value(
            self.gradient_sum,
            self.hessian_sum,
            options.l2_regularization,
        )
    }
}

pub(crate) fn train_cart_regressor_from_gradients_and_hessians(
    train_set: &dyn TableAccess,
    gradients: &[f64],
    hessians: &[f64],
    parallelism: Parallelism,
    options: SecondOrderRegressionTreeOptions,
) -> Result<DecisionTreeRegressor, SecondOrderRegressionTreeError> {
    train_cart_regressor_from_gradients_and_hessians_with_status(
        train_set,
        gradients,
        hessians,
        parallelism,
        options,
    )
    .map(|result| result.model)
}

pub(crate) fn train_cart_regressor_from_gradients_and_hessians_with_status(
    train_set: &dyn TableAccess,
    gradients: &[f64],
    hessians: &[f64],
    parallelism: Parallelism,
    options: SecondOrderRegressionTreeOptions,
) -> Result<SecondOrderRegressionTreeTrainingResult, SecondOrderRegressionTreeError> {
    train_second_order_regressor(
        train_set,
        gradients,
        hessians,
        RegressionTreeAlgorithm::Cart,
        parallelism,
        options,
    )
}

pub(crate) fn train_randomized_regressor_from_gradients_and_hessians(
    train_set: &dyn TableAccess,
    gradients: &[f64],
    hessians: &[f64],
    parallelism: Parallelism,
    options: SecondOrderRegressionTreeOptions,
) -> Result<DecisionTreeRegressor, SecondOrderRegressionTreeError> {
    train_randomized_regressor_from_gradients_and_hessians_with_status(
        train_set,
        gradients,
        hessians,
        parallelism,
        options,
    )
    .map(|result| result.model)
}

pub(crate) fn train_randomized_regressor_from_gradients_and_hessians_with_status(
    train_set: &dyn TableAccess,
    gradients: &[f64],
    hessians: &[f64],
    parallelism: Parallelism,
    options: SecondOrderRegressionTreeOptions,
) -> Result<SecondOrderRegressionTreeTrainingResult, SecondOrderRegressionTreeError> {
    train_second_order_regressor(
        train_set,
        gradients,
        hessians,
        RegressionTreeAlgorithm::Randomized,
        parallelism,
        options,
    )
}

pub(crate) fn train_oblivious_regressor_from_gradients_and_hessians(
    train_set: &dyn TableAccess,
    gradients: &[f64],
    hessians: &[f64],
    parallelism: Parallelism,
    options: SecondOrderRegressionTreeOptions,
) -> Result<DecisionTreeRegressor, SecondOrderRegressionTreeError> {
    train_oblivious_regressor_from_gradients_and_hessians_with_status(
        train_set,
        gradients,
        hessians,
        parallelism,
        options,
    )
    .map(|result| result.model)
}

pub(crate) fn train_oblivious_regressor_from_gradients_and_hessians_with_status(
    train_set: &dyn TableAccess,
    gradients: &[f64],
    hessians: &[f64],
    parallelism: Parallelism,
    options: SecondOrderRegressionTreeOptions,
) -> Result<SecondOrderRegressionTreeTrainingResult, SecondOrderRegressionTreeError> {
    train_second_order_regressor(
        train_set,
        gradients,
        hessians,
        RegressionTreeAlgorithm::Oblivious,
        parallelism,
        options,
    )
}

fn train_second_order_regressor(
    train_set: &dyn TableAccess,
    gradients: &[f64],
    hessians: &[f64],
    algorithm: RegressionTreeAlgorithm,
    parallelism: Parallelism,
    options: SecondOrderRegressionTreeOptions,
) -> Result<SecondOrderRegressionTreeTrainingResult, SecondOrderRegressionTreeError> {
    validate_inputs(train_set, gradients, hessians, options)?;

    let (structure, root_canary_selected) = match algorithm {
        RegressionTreeAlgorithm::Cart | RegressionTreeAlgorithm::Randomized => {
            let mut nodes = Vec::new();
            let mut rows: Vec<usize> = (0..train_set.n_rows()).collect();
            let context = BuildContext {
                table: train_set,
                gradients,
                hessians,
                parallelism,
                algorithm,
                options,
            };
            let mut root_canary_selected = false;
            let root = build_standard_node(
                &context,
                &mut nodes,
                &mut rows,
                0,
                None,
                &mut root_canary_selected,
            );
            (
                RegressionTreeStructure::Standard { nodes, root },
                root_canary_selected,
            )
        }
        RegressionTreeAlgorithm::Oblivious => {
            train_oblivious_structure(train_set, gradients, hessians, parallelism, options)
        }
    };

    Ok(SecondOrderRegressionTreeTrainingResult {
        model: DecisionTreeRegressor::from_ir_parts(
            algorithm,
            Criterion::Mean,
            structure,
            options.tree_options,
            train_set.n_features(),
            capture_feature_preprocessing(train_set),
            train_set.canaries(),
        ),
        root_canary_selected,
    })
}

fn validate_inputs(
    train_set: &dyn TableAccess,
    gradients: &[f64],
    hessians: &[f64],
    options: SecondOrderRegressionTreeOptions,
) -> Result<(), SecondOrderRegressionTreeError> {
    if train_set.n_rows() == 0 {
        return Err(SecondOrderRegressionTreeError::EmptyRows);
    }
    if gradients.len() != train_set.n_rows() {
        return Err(SecondOrderRegressionTreeError::GradientLengthMismatch {
            expected: train_set.n_rows(),
            actual: gradients.len(),
        });
    }
    if hessians.len() != train_set.n_rows() {
        return Err(SecondOrderRegressionTreeError::HessianLengthMismatch {
            expected: train_set.n_rows(),
            actual: hessians.len(),
        });
    }
    if !options.l2_regularization.is_finite() || options.l2_regularization < 0.0 {
        return Err(SecondOrderRegressionTreeError::InvalidL2Regularization(
            options.l2_regularization,
        ));
    }
    if !options.min_sum_hessian_in_leaf.is_finite() || options.min_sum_hessian_in_leaf < 0.0 {
        return Err(SecondOrderRegressionTreeError::InvalidMinSumHessianInLeaf(
            options.min_sum_hessian_in_leaf,
        ));
    }
    if !options.min_gain_to_split.is_finite() || options.min_gain_to_split < 0.0 {
        return Err(SecondOrderRegressionTreeError::InvalidMinGainToSplit(
            options.min_gain_to_split,
        ));
    }

    for (row, value) in gradients.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(SecondOrderRegressionTreeError::InvalidGradientValue { row, value });
        }
    }
    for (row, value) in hessians.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(SecondOrderRegressionTreeError::InvalidHessianValue { row, value });
        }
        if value < 0.0 {
            return Err(SecondOrderRegressionTreeError::NegativeHessian { row, value });
        }
    }

    Ok(())
}

fn train_oblivious_structure(
    table: &dyn TableAccess,
    gradients: &[f64],
    hessians: &[f64],
    parallelism: Parallelism,
    options: SecondOrderRegressionTreeOptions,
) -> (RegressionTreeStructure, bool) {
    let mut row_indices: Vec<usize> = (0..table.n_rows()).collect();
    let (root_gradient_sum, root_hessian_sum) =
        sum_gradient_hessian_stats(&row_indices, gradients, hessians);
    let mut leaves = vec![ObliviousLeafState {
        start: 0,
        end: row_indices.len(),
        gradient_sum: root_gradient_sum,
        hessian_sum: root_hessian_sum,
    }];
    let mut splits = Vec::new();

    let mut root_canary_selected = false;
    for depth in 0..options.tree_options.max_depth {
        if leaves
            .iter()
            .all(|leaf| leaf.len() < options.tree_options.min_samples_split)
        {
            break;
        }

        let feature_indices = candidate_feature_indices(
            table.binned_feature_count(),
            options.tree_options.max_features,
            node_seed(options.tree_options.random_seed, depth, &[], 0x0B11_A10Cu64),
        );
        let best_split = if parallelism.enabled() {
            feature_indices
                .into_par_iter()
                .filter_map(|feature_index| {
                    score_oblivious_split(
                        table,
                        &row_indices,
                        gradients,
                        hessians,
                        feature_index,
                        &leaves,
                        options,
                    )
                })
                .max_by(|left, right| left.gain.total_cmp(&right.gain))
        } else {
            feature_indices
                .into_iter()
                .filter_map(|feature_index| {
                    score_oblivious_split(
                        table,
                        &row_indices,
                        gradients,
                        hessians,
                        feature_index,
                        &leaves,
                        options,
                    )
                })
                .max_by(|left, right| left.gain.total_cmp(&right.gain))
        };

        let Some(best_split) = best_split.filter(|split| split.gain > options.min_gain_to_split)
        else {
            break;
        };
        if table.is_canary_binned_feature(best_split.feature_index) {
            if depth == 0 {
                root_canary_selected = true;
            }
            break;
        }

        leaves = split_oblivious_leaves_in_place(
            table,
            &mut row_indices,
            gradients,
            hessians,
            leaves,
            best_split.feature_index,
            best_split.threshold_bin,
        );
        let impurity = leaves
            .iter()
            .map(|leaf| {
                node_objective_strength(
                    leaf.gradient_sum,
                    leaf.hessian_sum,
                    options.l2_regularization,
                )
            })
            .sum();
        splits.push(ObliviousSplit {
            feature_index: best_split.feature_index,
            threshold_bin: best_split.threshold_bin,
            sample_count: table.n_rows(),
            impurity,
            gain: best_split.gain,
        });
    }

    (
        RegressionTreeStructure::Oblivious {
            splits,
            leaf_values: leaves.iter().map(|leaf| leaf.prediction(options)).collect(),
            leaf_sample_counts: leaves.iter().map(ObliviousLeafState::len).collect(),
            leaf_variances: vec![None; leaves.len()],
        },
        root_canary_selected,
    )
}

fn build_standard_node(
    context: &BuildContext<'_>,
    nodes: &mut Vec<RegressionNode>,
    rows: &mut [usize],
    depth: usize,
    histograms: Option<Vec<FeatureHistogram>>,
    root_canary_selected: &mut bool,
) -> usize {
    let stats = node_stats(rows, context.gradients, context.hessians);
    let leaf_prediction = leaf_value(
        stats.gradient_sum,
        stats.hessian_sum,
        context.options.l2_regularization,
    );

    if rows.is_empty()
        || depth >= context.options.tree_options.max_depth
        || rows.len() < context.options.tree_options.min_samples_split
        || stats.hessian_sum <= 0.0
    {
        return push_leaf(nodes, leaf_prediction, rows.len());
    }

    let histograms = histograms.unwrap_or_else(|| {
        build_feature_histograms(context.table, context.gradients, context.hessians, rows)
    });
    let feature_indices = candidate_feature_indices(
        context.table.binned_feature_count(),
        context.options.tree_options.max_features,
        node_seed(
            context.options.tree_options.random_seed,
            depth,
            rows,
            0xA11C_E5E1u64,
        ),
    );

    let best_split = if context.parallelism.enabled() {
        feature_indices
            .into_par_iter()
            .filter_map(|feature_index| {
                score_feature_from_hist(context, &histograms[feature_index], feature_index, rows)
            })
            .max_by(|left, right| left.gain.total_cmp(&right.gain))
    } else {
        feature_indices
            .into_iter()
            .filter_map(|feature_index| {
                score_feature_from_hist(context, &histograms[feature_index], feature_index, rows)
            })
            .max_by(|left, right| left.gain.total_cmp(&right.gain))
    };

    match best_split {
        Some(split)
            if split.gain > context.options.min_gain_to_split
                && !context.table.is_canary_binned_feature(split.feature_index) =>
        {
            let left_count = partition_rows_for_binary_split(
                context.table,
                split.feature_index,
                split.threshold_bin,
                rows,
            );
            let (left_rows, right_rows) = rows.split_at_mut(left_count);
            let (left_child, right_child) = if left_rows.len() <= right_rows.len() {
                let left_histograms = build_feature_histograms(
                    context.table,
                    context.gradients,
                    context.hessians,
                    left_rows,
                );
                let right_histograms = subtract_feature_histograms(&histograms, &left_histograms);
                (
                    build_standard_node(
                        context,
                        nodes,
                        left_rows,
                        depth + 1,
                        Some(left_histograms),
                        root_canary_selected,
                    ),
                    build_standard_node(
                        context,
                        nodes,
                        right_rows,
                        depth + 1,
                        Some(right_histograms),
                        root_canary_selected,
                    ),
                )
            } else {
                let right_histograms = build_feature_histograms(
                    context.table,
                    context.gradients,
                    context.hessians,
                    right_rows,
                );
                let left_histograms = subtract_feature_histograms(&histograms, &right_histograms);
                (
                    build_standard_node(
                        context,
                        nodes,
                        left_rows,
                        depth + 1,
                        Some(left_histograms),
                        root_canary_selected,
                    ),
                    build_standard_node(
                        context,
                        nodes,
                        right_rows,
                        depth + 1,
                        Some(right_histograms),
                        root_canary_selected,
                    ),
                )
            };

            push_node(
                nodes,
                RegressionNode::BinarySplit {
                    feature_index: split.feature_index,
                    threshold_bin: split.threshold_bin,
                    left_child,
                    right_child,
                    sample_count: rows.len(),
                    impurity: node_objective_strength(
                        stats.gradient_sum,
                        stats.hessian_sum,
                        context.options.l2_regularization,
                    ),
                    gain: split.gain,
                    variance: None,
                },
            )
        }
        Some(split)
            if split.gain > context.options.min_gain_to_split
                && context.table.is_canary_binned_feature(split.feature_index) =>
        {
            if depth == 0 {
                *root_canary_selected = true;
            }
            push_leaf(nodes, leaf_prediction, rows.len())
        }
        _ => push_leaf(nodes, leaf_prediction, rows.len()),
    }
}

fn node_stats(rows: &[usize], gradients: &[f64], hessians: &[f64]) -> NodeStats {
    let (gradient_sum, hessian_sum) = sum_gradient_hessian_stats(rows, gradients, hessians);
    NodeStats {
        gradient_sum,
        hessian_sum,
    }
}

fn sum_gradient_hessian_stats(rows: &[usize], gradients: &[f64], hessians: &[f64]) -> (f64, f64) {
    rows.iter()
        .fold((0.0, 0.0), |(gradient_sum, hessian_sum), row_idx| {
            (
                gradient_sum + gradients[*row_idx],
                hessian_sum + hessians[*row_idx],
            )
        })
}

fn leaf_value(gradient_sum: f64, hessian_sum: f64, l2_regularization: f64) -> f64 {
    let denominator = hessian_sum + l2_regularization;
    if denominator <= 0.0 {
        0.0
    } else {
        -gradient_sum / denominator
    }
}

fn node_objective_strength(gradient_sum: f64, hessian_sum: f64, l2_regularization: f64) -> f64 {
    let denominator = hessian_sum + l2_regularization;
    if denominator <= 0.0 {
        0.0
    } else {
        0.5 * gradient_sum * gradient_sum / denominator
    }
}

fn split_gain(
    left_gradient_sum: f64,
    left_hessian_sum: f64,
    right_gradient_sum: f64,
    right_hessian_sum: f64,
    parent_gradient_sum: f64,
    parent_hessian_sum: f64,
    l2_regularization: f64,
) -> f64 {
    node_objective_strength(left_gradient_sum, left_hessian_sum, l2_regularization)
        + node_objective_strength(right_gradient_sum, right_hessian_sum, l2_regularization)
        - node_objective_strength(parent_gradient_sum, parent_hessian_sum, l2_regularization)
}

fn build_feature_histograms(
    table: &dyn TableAccess,
    gradients: &[f64],
    hessians: &[f64],
    rows: &[usize],
) -> Vec<FeatureHistogram> {
    (0..table.binned_feature_count())
        .map(|feature_index| {
            if table.is_binary_binned_feature(feature_index) {
                let mut false_count = 0usize;
                let mut false_gradient_sum = 0.0;
                let mut false_hessian_sum = 0.0;
                let mut true_count = 0usize;
                let mut true_gradient_sum = 0.0;
                let mut true_hessian_sum = 0.0;
                for row_idx in rows {
                    if !table
                        .binned_boolean_value(feature_index, *row_idx)
                        .expect("binary feature must expose boolean values")
                    {
                        false_count += 1;
                        false_gradient_sum += gradients[*row_idx];
                        false_hessian_sum += hessians[*row_idx];
                    } else {
                        true_count += 1;
                        true_gradient_sum += gradients[*row_idx];
                        true_hessian_sum += hessians[*row_idx];
                    }
                }
                FeatureHistogram::Binary {
                    false_count,
                    false_gradient_sum,
                    false_hessian_sum,
                    true_count,
                    true_gradient_sum,
                    true_hessian_sum,
                }
            } else {
                let bin_cap = table.numeric_bin_cap();
                let mut bin_count = vec![0usize; bin_cap];
                let mut bin_gradient_sum = vec![0.0; bin_cap];
                let mut bin_hessian_sum = vec![0.0; bin_cap];
                let mut observed_bins = vec![false; bin_cap];
                for row_idx in rows {
                    let bin = table.binned_value(feature_index, *row_idx) as usize;
                    bin_count[bin] += 1;
                    bin_gradient_sum[bin] += gradients[*row_idx];
                    bin_hessian_sum[bin] += hessians[*row_idx];
                    observed_bins[bin] = true;
                }
                FeatureHistogram::Numeric {
                    bin_count,
                    bin_gradient_sum,
                    bin_hessian_sum,
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

fn subtract_feature_histograms(
    parent: &[FeatureHistogram],
    child: &[FeatureHistogram],
) -> Vec<FeatureHistogram> {
    parent
        .iter()
        .zip(child.iter())
        .map(
            |(parent_hist, child_hist)| match (parent_hist, child_hist) {
                (
                    FeatureHistogram::Binary {
                        false_count: parent_false_count,
                        false_gradient_sum: parent_false_gradient_sum,
                        false_hessian_sum: parent_false_hessian_sum,
                        true_count: parent_true_count,
                        true_gradient_sum: parent_true_gradient_sum,
                        true_hessian_sum: parent_true_hessian_sum,
                    },
                    FeatureHistogram::Binary {
                        false_count: child_false_count,
                        false_gradient_sum: child_false_gradient_sum,
                        false_hessian_sum: child_false_hessian_sum,
                        true_count: child_true_count,
                        true_gradient_sum: child_true_gradient_sum,
                        true_hessian_sum: child_true_hessian_sum,
                    },
                ) => FeatureHistogram::Binary {
                    false_count: parent_false_count - child_false_count,
                    false_gradient_sum: parent_false_gradient_sum - child_false_gradient_sum,
                    false_hessian_sum: parent_false_hessian_sum - child_false_hessian_sum,
                    true_count: parent_true_count - child_true_count,
                    true_gradient_sum: parent_true_gradient_sum - child_true_gradient_sum,
                    true_hessian_sum: parent_true_hessian_sum - child_true_hessian_sum,
                },
                (
                    FeatureHistogram::Numeric {
                        bin_count: parent_bin_count,
                        bin_gradient_sum: parent_bin_gradient_sum,
                        bin_hessian_sum: parent_bin_hessian_sum,
                        ..
                    },
                    FeatureHistogram::Numeric {
                        bin_count: child_bin_count,
                        bin_gradient_sum: child_bin_gradient_sum,
                        bin_hessian_sum: child_bin_hessian_sum,
                        ..
                    },
                ) => {
                    let bin_count = parent_bin_count
                        .iter()
                        .zip(child_bin_count.iter())
                        .map(|(parent, child)| parent - child)
                        .collect::<Vec<_>>();
                    let bin_gradient_sum = parent_bin_gradient_sum
                        .iter()
                        .zip(child_bin_gradient_sum.iter())
                        .map(|(parent, child)| parent - child)
                        .collect::<Vec<_>>();
                    let bin_hessian_sum = parent_bin_hessian_sum
                        .iter()
                        .zip(child_bin_hessian_sum.iter())
                        .map(|(parent, child)| parent - child)
                        .collect::<Vec<_>>();
                    let observed_bins = bin_count
                        .iter()
                        .enumerate()
                        .filter_map(|(bin, count)| (*count > 0).then_some(bin))
                        .collect::<Vec<_>>();
                    FeatureHistogram::Numeric {
                        bin_count,
                        bin_gradient_sum,
                        bin_hessian_sum,
                        observed_bins,
                    }
                }
                _ => unreachable!("histogram shapes must match"),
            },
        )
        .collect()
}

fn score_feature_from_hist(
    context: &BuildContext<'_>,
    histogram: &FeatureHistogram,
    feature_index: usize,
    rows: &[usize],
) -> Option<SplitChoice> {
    match histogram {
        FeatureHistogram::Binary {
            false_count,
            false_gradient_sum,
            false_hessian_sum,
            true_count,
            true_gradient_sum,
            true_hessian_sum,
        } => score_binary_split_from_stats(
            context,
            feature_index,
            *false_count,
            *false_gradient_sum,
            *false_hessian_sum,
            *true_count,
            *true_gradient_sum,
            *true_hessian_sum,
        ),
        FeatureHistogram::Numeric {
            bin_count,
            bin_gradient_sum,
            bin_hessian_sum,
            observed_bins,
        } => match context.algorithm {
            RegressionTreeAlgorithm::Cart => score_numeric_split_from_hist(
                context,
                feature_index,
                rows.len(),
                bin_count,
                bin_gradient_sum,
                bin_hessian_sum,
                observed_bins,
            ),
            RegressionTreeAlgorithm::Randomized => score_randomized_split_from_hist(
                context,
                feature_index,
                rows,
                bin_count,
                bin_gradient_sum,
                bin_hessian_sum,
                observed_bins,
            ),
            RegressionTreeAlgorithm::Oblivious => None,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn score_binary_split_from_stats(
    context: &BuildContext<'_>,
    feature_index: usize,
    left_count: usize,
    left_gradient_sum: f64,
    left_hessian_sum: f64,
    right_count: usize,
    right_gradient_sum: f64,
    right_hessian_sum: f64,
) -> Option<SplitChoice> {
    if !children_are_splittable(
        context.options,
        left_count,
        left_hessian_sum,
        right_count,
        right_hessian_sum,
    ) {
        return None;
    }
    let gain = split_gain(
        left_gradient_sum,
        left_hessian_sum,
        right_gradient_sum,
        right_hessian_sum,
        left_gradient_sum + right_gradient_sum,
        left_hessian_sum + right_hessian_sum,
        context.options.l2_regularization,
    );
    Some(SplitChoice {
        feature_index,
        threshold_bin: 0,
        gain,
    })
}

fn score_numeric_split_from_hist(
    context: &BuildContext<'_>,
    feature_index: usize,
    row_count: usize,
    bin_count: &[usize],
    bin_gradient_sum: &[f64],
    bin_hessian_sum: &[f64],
    observed_bins: &[usize],
) -> Option<SplitChoice> {
    if observed_bins.len() <= 1 {
        return None;
    }

    let total_gradient_sum = bin_gradient_sum.iter().sum::<f64>();
    let total_hessian_sum = bin_hessian_sum.iter().sum::<f64>();
    let mut left_count = 0usize;
    let mut left_gradient_sum = 0.0;
    let mut left_hessian_sum = 0.0;
    let mut best_threshold = None;
    let mut best_gain = f64::NEG_INFINITY;

    for &bin in observed_bins {
        left_count += bin_count[bin];
        left_gradient_sum += bin_gradient_sum[bin];
        left_hessian_sum += bin_hessian_sum[bin];

        let right_count = row_count - left_count;
        let right_gradient_sum = total_gradient_sum - left_gradient_sum;
        let right_hessian_sum = total_hessian_sum - left_hessian_sum;

        if !children_are_splittable(
            context.options,
            left_count,
            left_hessian_sum,
            right_count,
            right_hessian_sum,
        ) {
            continue;
        }

        let gain = split_gain(
            left_gradient_sum,
            left_hessian_sum,
            right_gradient_sum,
            right_hessian_sum,
            total_gradient_sum,
            total_hessian_sum,
            context.options.l2_regularization,
        );
        if gain > best_gain {
            best_gain = gain;
            best_threshold = Some(bin as u16);
        }
    }

    best_threshold.map(|threshold_bin| SplitChoice {
        feature_index,
        threshold_bin,
        gain: best_gain,
    })
}

fn score_randomized_split_from_hist(
    context: &BuildContext<'_>,
    feature_index: usize,
    rows: &[usize],
    bin_count: &[usize],
    bin_gradient_sum: &[f64],
    bin_hessian_sum: &[f64],
    observed_bins: &[usize],
) -> Option<SplitChoice> {
    let candidate_thresholds = observed_bins
        .iter()
        .copied()
        .map(|bin| bin as u16)
        .collect::<Vec<_>>();
    let threshold_bin =
        choose_random_threshold(&candidate_thresholds, feature_index, rows, 0xA11C_E551u64)?;

    let total_gradient_sum = bin_gradient_sum.iter().sum::<f64>();
    let total_hessian_sum = bin_hessian_sum.iter().sum::<f64>();
    let mut left_count = 0usize;
    let mut left_gradient_sum = 0.0;
    let mut left_hessian_sum = 0.0;
    for bin in 0..=threshold_bin as usize {
        if bin >= bin_count.len() {
            break;
        }
        left_count += bin_count[bin];
        left_gradient_sum += bin_gradient_sum[bin];
        left_hessian_sum += bin_hessian_sum[bin];
    }
    let right_count = rows.len() - left_count;
    let right_gradient_sum = total_gradient_sum - left_gradient_sum;
    let right_hessian_sum = total_hessian_sum - left_hessian_sum;
    if !children_are_splittable(
        context.options,
        left_count,
        left_hessian_sum,
        right_count,
        right_hessian_sum,
    ) {
        return None;
    }

    Some(SplitChoice {
        feature_index,
        threshold_bin,
        gain: split_gain(
            left_gradient_sum,
            left_hessian_sum,
            right_gradient_sum,
            right_hessian_sum,
            total_gradient_sum,
            total_hessian_sum,
            context.options.l2_regularization,
        ),
    })
}

fn score_oblivious_split(
    table: &dyn TableAccess,
    row_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
    options: SecondOrderRegressionTreeOptions,
) -> Option<SplitChoice> {
    if table.is_binary_binned_feature(feature_index) {
        return score_binary_oblivious_split(
            table,
            row_indices,
            gradients,
            hessians,
            feature_index,
            leaves,
            options,
        );
    }

    let bin_cap = table.numeric_bin_cap();
    if bin_cap == 0 {
        return None;
    }
    let mut threshold_gains = vec![0.0; bin_cap];
    let mut found_valid = vec![false; bin_cap];
    let mut observed_any = false;

    for leaf in leaves {
        let mut bin_count = vec![0usize; bin_cap];
        let mut bin_gradient_sum = vec![0.0; bin_cap];
        let mut bin_hessian_sum = vec![0.0; bin_cap];
        let mut observed_bins = vec![false; bin_cap];

        for row_idx in &row_indices[leaf.start..leaf.end] {
            let bin = table.binned_value(feature_index, *row_idx) as usize;
            bin_count[bin] += 1;
            bin_gradient_sum[bin] += gradients[*row_idx];
            bin_hessian_sum[bin] += hessians[*row_idx];
            observed_bins[bin] = true;
        }

        let observed_bins = observed_bins
            .into_iter()
            .enumerate()
            .filter_map(|(bin, seen)| seen.then_some(bin))
            .collect::<Vec<_>>();
        if observed_bins.len() <= 1 {
            continue;
        }
        observed_any = true;

        let mut left_count = 0usize;
        let mut left_gradient_sum = 0.0;
        let mut left_hessian_sum = 0.0;
        for &bin in &observed_bins {
            left_count += bin_count[bin];
            left_gradient_sum += bin_gradient_sum[bin];
            left_hessian_sum += bin_hessian_sum[bin];
            let right_count = leaf.len() - left_count;
            let right_gradient_sum = leaf.gradient_sum - left_gradient_sum;
            let right_hessian_sum = leaf.hessian_sum - left_hessian_sum;
            if !children_are_splittable(
                options,
                left_count,
                left_hessian_sum,
                right_count,
                right_hessian_sum,
            ) {
                continue;
            }
            found_valid[bin] = true;
            threshold_gains[bin] += split_gain(
                left_gradient_sum,
                left_hessian_sum,
                right_gradient_sum,
                right_hessian_sum,
                leaf.gradient_sum,
                leaf.hessian_sum,
                options.l2_regularization,
            );
        }
    }

    if !observed_any {
        return None;
    }

    threshold_gains
        .into_iter()
        .enumerate()
        .filter(|(bin, gain)| found_valid[*bin] && *gain > options.min_gain_to_split)
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(threshold_bin, gain)| SplitChoice {
            feature_index,
            threshold_bin: threshold_bin as u16,
            gain,
        })
}

fn score_binary_oblivious_split(
    table: &dyn TableAccess,
    row_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
    options: SecondOrderRegressionTreeOptions,
) -> Option<SplitChoice> {
    let mut gain = 0.0;
    let mut found_valid = false;

    for leaf in leaves {
        let mut left_count = 0usize;
        let mut left_gradient_sum = 0.0;
        let mut left_hessian_sum = 0.0;
        for row_idx in &row_indices[leaf.start..leaf.end] {
            if !table
                .binned_boolean_value(feature_index, *row_idx)
                .expect("binary feature must expose boolean values")
            {
                left_count += 1;
                left_gradient_sum += gradients[*row_idx];
                left_hessian_sum += hessians[*row_idx];
            }
        }
        let right_count = leaf.len() - left_count;
        let right_gradient_sum = leaf.gradient_sum - left_gradient_sum;
        let right_hessian_sum = leaf.hessian_sum - left_hessian_sum;
        if !children_are_splittable(
            options,
            left_count,
            left_hessian_sum,
            right_count,
            right_hessian_sum,
        ) {
            continue;
        }
        found_valid = true;
        gain += split_gain(
            left_gradient_sum,
            left_hessian_sum,
            right_gradient_sum,
            right_hessian_sum,
            leaf.gradient_sum,
            leaf.hessian_sum,
            options.l2_regularization,
        );
    }

    (found_valid && gain > options.min_gain_to_split).then_some(SplitChoice {
        feature_index,
        threshold_bin: 0,
        gain,
    })
}

fn split_oblivious_leaves_in_place(
    table: &dyn TableAccess,
    row_indices: &mut [usize],
    gradients: &[f64],
    hessians: &[f64],
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
            &mut row_indices[leaf.start..leaf.end],
        );
        let mid = leaf.start + left_count;
        let (left_gradient_sum, left_hessian_sum) =
            sum_gradient_hessian_stats(&row_indices[leaf.start..mid], gradients, hessians);
        let (right_gradient_sum, right_hessian_sum) =
            sum_gradient_hessian_stats(&row_indices[mid..leaf.end], gradients, hessians);
        next_leaves.push(ObliviousLeafState {
            start: leaf.start,
            end: mid,
            gradient_sum: left_gradient_sum,
            hessian_sum: left_hessian_sum,
        });
        next_leaves.push(ObliviousLeafState {
            start: mid,
            end: leaf.end,
            gradient_sum: right_gradient_sum,
            hessian_sum: right_hessian_sum,
        });
    }
    next_leaves
}

fn children_are_splittable(
    options: SecondOrderRegressionTreeOptions,
    left_count: usize,
    left_hessian_sum: f64,
    right_count: usize,
    right_hessian_sum: f64,
) -> bool {
    left_count >= options.tree_options.min_samples_leaf
        && right_count >= options.tree_options.min_samples_leaf
        && left_hessian_sum >= options.min_sum_hessian_in_leaf
        && right_hessian_sum >= options.min_sum_hessian_in_leaf
}

fn push_leaf(nodes: &mut Vec<RegressionNode>, value: f64, sample_count: usize) -> usize {
    push_node(
        nodes,
        RegressionNode::Leaf {
            value,
            sample_count,
            variance: None,
        },
    )
}

fn push_node(nodes: &mut Vec<RegressionNode>, node: RegressionNode) -> usize {
    nodes.push(node);
    nodes.len() - 1
}

fn partition_rows_for_binary_split(
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

fn candidate_feature_indices(
    feature_count: usize,
    max_features: Option<usize>,
    seed: u64,
) -> Vec<usize> {
    match max_features {
        Some(count) => sample_feature_subset(feature_count, count, seed),
        None => (0..feature_count).collect(),
    }
}

fn node_seed(base_seed: u64, depth: usize, rows: &[usize], salt: u64) -> u64 {
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

fn choose_random_threshold(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::regressor::RegressionTreeStructure;
    use forestfire_data::{DenseTable, NumericBins};

    fn simple_table() -> DenseTable {
        DenseTable::with_options(
            vec![vec![0.0], vec![0.0], vec![1.0], vec![1.0]],
            vec![0.0, 0.0, 0.0, 0.0],
            0,
            NumericBins::fixed(8).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn cart_second_order_tree_learns_newton_leaf_values() {
        let table = simple_table();
        let model = train_cart_regressor_from_gradients_and_hessians(
            &table,
            &[2.0, 1.0, -2.0, -1.0],
            &[1.0, 1.0, 1.0, 1.0],
            Parallelism::sequential(),
            SecondOrderRegressionTreeOptions {
                tree_options: RegressionTreeOptions {
                    max_depth: 1,
                    ..Default::default()
                },
                l2_regularization: 0.0,
                min_sum_hessian_in_leaf: 0.1,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(model.predict_table(&table), vec![-1.5, -1.5, 1.5, 1.5]);
        match model.structure() {
            RegressionTreeStructure::Standard { nodes, .. } => {
                assert!(nodes.len() >= 3);
            }
            RegressionTreeStructure::Oblivious { .. } => panic!("expected standard tree"),
        }
    }

    #[test]
    fn randomized_second_order_tree_uses_same_leaf_formula() {
        let table = simple_table();
        let model = train_randomized_regressor_from_gradients_and_hessians(
            &table,
            &[2.0, 1.0, -2.0, -1.0],
            &[1.0, 1.0, 1.0, 1.0],
            Parallelism::sequential(),
            SecondOrderRegressionTreeOptions {
                tree_options: RegressionTreeOptions {
                    max_depth: 1,
                    ..Default::default()
                },
                l2_regularization: 0.0,
                min_sum_hessian_in_leaf: 0.1,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(model.predict_table(&table), vec![-1.5, -1.5, 1.5, 1.5]);
    }

    #[test]
    fn oblivious_second_order_tree_trains_from_gradient_hessian_pairs() {
        let table = simple_table();
        let model = train_oblivious_regressor_from_gradients_and_hessians(
            &table,
            &[2.0, 1.0, -2.0, -1.0],
            &[1.0, 1.0, 1.0, 1.0],
            Parallelism::sequential(),
            SecondOrderRegressionTreeOptions {
                tree_options: RegressionTreeOptions {
                    max_depth: 1,
                    ..Default::default()
                },
                l2_regularization: 0.0,
                min_sum_hessian_in_leaf: 0.1,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(model.predict_table(&table), vec![-1.5, -1.5, 1.5, 1.5]);
        match model.structure() {
            RegressionTreeStructure::Oblivious { splits, .. } => {
                assert_eq!(splits.len(), 1);
            }
            RegressionTreeStructure::Standard { .. } => panic!("expected oblivious tree"),
        }
    }

    #[test]
    fn second_order_tree_respects_min_sum_hessian_in_leaf() {
        let table = simple_table();
        let model = train_cart_regressor_from_gradients_and_hessians(
            &table,
            &[2.0, 2.0, -1.0, -1.0],
            &[0.1, 0.1, 0.1, 0.1],
            Parallelism::sequential(),
            SecondOrderRegressionTreeOptions {
                l2_regularization: 0.0,
                min_sum_hessian_in_leaf: 0.25,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(model.predict_table(&table), vec![-5.0; 4]);
        match model.structure() {
            RegressionTreeStructure::Standard { nodes, .. } => assert_eq!(nodes.len(), 1),
            RegressionTreeStructure::Oblivious { .. } => panic!("expected standard tree"),
        }
    }

    #[test]
    fn second_order_tree_rejects_negative_hessian() {
        let table = simple_table();
        let error = train_cart_regressor_from_gradients_and_hessians(
            &table,
            &[1.0, 1.0, -1.0, -1.0],
            &[1.0, -1.0, 1.0, 1.0],
            Parallelism::sequential(),
            SecondOrderRegressionTreeOptions::default(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            SecondOrderRegressionTreeError::NegativeHessian {
                row: 1,
                value: -1.0
            }
        ));
    }

    #[test]
    fn second_order_tree_uses_l2_regularization_in_leaf_values() {
        let table = simple_table();
        let model = train_cart_regressor_from_gradients_and_hessians(
            &table,
            &[2.0, 2.0, 2.0, 2.0],
            &[1.0, 1.0, 1.0, 1.0],
            Parallelism::sequential(),
            SecondOrderRegressionTreeOptions {
                tree_options: RegressionTreeOptions {
                    max_depth: 1,
                    ..Default::default()
                },
                l2_regularization: 4.0,
                min_sum_hessian_in_leaf: 10.0,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(model.predict_table(&table), vec![-1.0; 4]);
    }
}
