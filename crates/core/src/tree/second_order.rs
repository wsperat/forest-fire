#![allow(dead_code)]

//! Second-order regression tree learner used by gradient boosting.
//!
//! The tree structure is intentionally reused from the ordinary regression tree
//! path. The difference is not in how predictions are represented, but in how
//! split gain and leaf values are computed: boosting provides per-row gradients
//! and Hessians, and this learner turns them into Newton-style trees.

use crate::tree::oblique::{
    all_feature_pairs, matched_canary_feature_pairs, missing_mask_for_pair, normalize_weights,
    oblique_feature_value, partition_rows_for_oblique_split, projected_rows_for_pair,
    resolve_oblique_missing_direction,
};
use crate::tree::regressor::{
    DecisionTreeRegressor, ObliviousSplit, RegressionNode, RegressionTreeAlgorithm,
    RegressionTreeOptions, RegressionTreeStructure,
};
use crate::tree::shared::{
    FeatureHistogram, HistogramBin, MissingBranchDirection, aggregate_beam_non_canary_score,
    build_feature_histograms, build_feature_histograms_parallel, candidate_feature_indices,
    choose_random_threshold, node_seed, partition_rows_for_binary_split,
    select_best_non_canary_candidate, subtract_feature_histograms,
};
use crate::{
    BuilderStrategy, Criterion, Parallelism, SplitStrategy, capture_feature_preprocessing,
};
use forestfire_data::TableAccess;
use rayon::prelude::*;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Extra controls required by second-order tree training.
#[derive(Debug, Clone)]
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

#[derive(Clone)]
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
struct ObliqueSplitChoice {
    feature_indices: Vec<usize>,
    weights: Vec<f64>,
    missing_directions: Vec<MissingBranchDirection>,
    threshold: f64,
    gain: f64,
}

#[derive(Debug, Clone)]
enum StandardSplitChoice {
    Axis(SplitChoice),
    Oblique(ObliqueSplitChoice),
}

impl StandardSplitChoice {
    fn gain(&self) -> f64 {
        match self {
            Self::Axis(choice) => choice.gain,
            Self::Oblique(choice) => choice.gain,
        }
    }

    fn ranking_feature_index(&self) -> usize {
        match self {
            Self::Axis(choice) => choice.feature_index,
            Self::Oblique(choice) => choice.feature_indices[0],
        }
    }
}

#[derive(Debug, Clone)]
struct RankedStandardSplitChoice {
    choice: StandardSplitChoice,
    ranking_score: f64,
}

#[derive(Debug, Clone, Copy)]
struct RankedSplitChoice {
    choice: SplitChoice,
    ranking_score: f64,
}

/// Result of evaluating one standard node before mutating the shared row buffer.
///
/// Keeping node evaluation separate from child recursion is the first step
/// toward level-wise GBM parallelism: a later implementation can evaluate a
/// whole batch of active nodes in parallel, then perform row partitioning only
/// for the nodes that actually split.
struct StandardNodeEvaluation {
    sample_count: usize,
    leaf_prediction: f64,
    parent_strength: f64,
    histograms: Option<Vec<SecondOrderFeatureHistogram>>,
    best_split: Option<StandardSplitChoice>,
    blocked_by_canary: bool,
}

struct ActiveNode {
    node_index: usize,
    depth: usize,
    start: usize,
    end: usize,
    histograms: Option<Vec<SecondOrderFeatureHistogram>>,
}

struct FrontierNodeEvaluation {
    node_index: usize,
    depth: usize,
    start: usize,
    end: usize,
    evaluation: StandardNodeEvaluation,
}

struct PendingSplit {
    node_index: usize,
    depth: usize,
    start: usize,
    mid: usize,
    end: usize,
    evaluation: StandardNodeEvaluation,
}

#[derive(Debug, Clone)]
struct SecondOrderHistogramBin {
    count: usize,
    gradient_sum: f64,
    hessian_sum: f64,
}

impl HistogramBin for SecondOrderHistogramBin {
    fn subtract(parent: &Self, child: &Self) -> Self {
        Self {
            count: parent.count - child.count,
            gradient_sum: parent.gradient_sum - child.gradient_sum,
            hessian_sum: parent.hessian_sum - child.hessian_sum,
        }
    }

    fn is_observed(&self) -> bool {
        self.count > 0
    }
}

type SecondOrderFeatureHistogram = FeatureHistogram<SecondOrderHistogramBin>;

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

    fn prediction(&self, options: &SecondOrderRegressionTreeOptions) -> f64 {
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
    validate_inputs(train_set, gradients, hessians, &options)?;

    let (structure, root_canary_selected) = match algorithm {
        RegressionTreeAlgorithm::Cart | RegressionTreeAlgorithm::Randomized => {
            let mut nodes = vec![RegressionNode::Leaf {
                value: 0.0,
                sample_count: 0,
                variance: None,
            }];
            let mut rows: Vec<usize> = (0..train_set.n_rows()).collect();
            let context = BuildContext {
                table: train_set,
                gradients,
                hessians,
                parallelism,
                algorithm,
                options: options.clone(),
            };
            let mut root_canary_selected = false;
            let root = train_standard_structure(
                &context,
                &mut nodes,
                &mut rows,
                &mut root_canary_selected,
            );
            (
                RegressionTreeStructure::Standard { nodes, root },
                root_canary_selected,
            )
        }
        RegressionTreeAlgorithm::Oblivious => {
            // Oblivious trees are grown a level at a time because every node at
            // the same depth shares the same feature/threshold pair.
            train_oblivious_structure(train_set, gradients, hessians, parallelism, options.clone())
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
    options: &SecondOrderRegressionTreeOptions,
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

fn train_standard_structure(
    context: &BuildContext<'_>,
    nodes: &mut Vec<RegressionNode>,
    rows: &mut [usize],
    root_canary_selected: &mut bool,
) -> usize {
    let root = 0usize;
    let mut frontier = vec![ActiveNode {
        node_index: root,
        depth: 0,
        start: 0,
        end: rows.len(),
        histograms: None,
    }];

    while !frontier.is_empty() {
        let evaluations = if context.parallelism.enabled() {
            frontier
                .into_par_iter()
                .map(|active| FrontierNodeEvaluation {
                    node_index: active.node_index,
                    depth: active.depth,
                    start: active.start,
                    end: active.end,
                    evaluation: evaluate_standard_node(
                        context,
                        &rows[active.start..active.end],
                        active.depth,
                        active.histograms,
                    ),
                })
                .collect::<Vec<_>>()
        } else {
            frontier
                .into_iter()
                .map(|active| FrontierNodeEvaluation {
                    node_index: active.node_index,
                    depth: active.depth,
                    start: active.start,
                    end: active.end,
                    evaluation: evaluate_standard_node(
                        context,
                        &rows[active.start..active.end],
                        active.depth,
                        active.histograms,
                    ),
                })
                .collect::<Vec<_>>()
        };
        let mut pending_splits = Vec::new();

        for FrontierNodeEvaluation {
            node_index,
            depth,
            start,
            end,
            evaluation,
        } in evaluations
        {
            match evaluation.best_split.clone() {
                Some(split) if split.gain() > context.options.min_gain_to_split => {
                    let left_count = match &split {
                        StandardSplitChoice::Axis(split) => partition_rows_for_binary_split(
                            context.table,
                            split.feature_index,
                            split.threshold_bin,
                            MissingBranchDirection::Right,
                            &mut rows[start..end],
                        ),
                        StandardSplitChoice::Oblique(split) => partition_rows_for_oblique_split(
                            context.table,
                            [split.feature_indices[0], split.feature_indices[1]],
                            [split.weights[0], split.weights[1]],
                            split.threshold,
                            [split.missing_directions[0], split.missing_directions[1]],
                            &mut rows[start..end],
                        ),
                    };
                    let mid = start + left_count;

                    let left_child = push_leaf(nodes, evaluation.leaf_prediction, mid - start);
                    let right_child = push_leaf(nodes, evaluation.leaf_prediction, end - mid);
                    nodes[node_index] = match split {
                        StandardSplitChoice::Axis(split) => RegressionNode::BinarySplit {
                            feature_index: split.feature_index,
                            threshold_bin: split.threshold_bin,
                            missing_direction: MissingBranchDirection::Node,
                            missing_values: vec![evaluation.leaf_prediction],
                            left_child,
                            right_child,
                            sample_count: evaluation.sample_count,
                            impurity: evaluation.parent_strength,
                            gain: split.gain,
                            variance: None,
                        },
                        StandardSplitChoice::Oblique(split) => RegressionNode::ObliqueSplit {
                            feature_indices: split.feature_indices,
                            weights: split.weights,
                            missing_directions: split.missing_directions,
                            threshold: split.threshold,
                            missing_values: vec![evaluation.leaf_prediction],
                            left_child,
                            right_child,
                            sample_count: evaluation.sample_count,
                            impurity: evaluation.parent_strength,
                            gain: split.gain,
                            variance: None,
                        },
                    };
                    pending_splits.push(PendingSplit {
                        node_index,
                        depth,
                        start,
                        mid: start,
                        end,
                        evaluation,
                    });
                }
                _ => {
                    if depth == 0 && evaluation.blocked_by_canary {
                        *root_canary_selected = true;
                    }
                    nodes[node_index] = RegressionNode::Leaf {
                        value: evaluation.leaf_prediction,
                        sample_count: evaluation.sample_count,
                        variance: None,
                    };
                }
            }
        }

        pending_splits.sort_unstable_by_key(|pending| pending.start);
        partition_pending_splits_in_place(
            context.table,
            rows,
            &mut pending_splits,
            context.parallelism,
        );

        frontier = if context.parallelism.enabled() {
            pending_splits
                .into_par_iter()
                .flat_map_iter(|pending| {
                    let parent_histograms = pending
                        .evaluation
                        .histograms
                        .expect("splittable second-order node must retain histograms");
                    let left_len = pending.mid - pending.start;
                    let right_len = pending.end - pending.mid;
                    let (left_histograms, right_histograms) = if left_len <= right_len {
                        let left_histograms = build_second_order_feature_histograms(
                            context.table,
                            context.gradients,
                            context.hessians,
                            &rows[pending.start..pending.mid],
                            Parallelism::sequential(),
                        );
                        let right_histograms =
                            subtract_feature_histograms(&parent_histograms, &left_histograms);
                        (left_histograms, right_histograms)
                    } else {
                        let right_histograms = build_second_order_feature_histograms(
                            context.table,
                            context.gradients,
                            context.hessians,
                            &rows[pending.mid..pending.end],
                            Parallelism::sequential(),
                        );
                        let left_histograms =
                            subtract_feature_histograms(&parent_histograms, &right_histograms);
                        (left_histograms, right_histograms)
                    };
                    let left_child = match &nodes[pending.node_index] {
                        RegressionNode::BinarySplit { left_child, .. }
                        | RegressionNode::ObliqueSplit { left_child, .. } => *left_child,
                        RegressionNode::Leaf { .. } | RegressionNode::MultiTargetLeaf { .. } => {
                            unreachable!("split node must exist")
                        }
                    };
                    let right_child = match &nodes[pending.node_index] {
                        RegressionNode::BinarySplit { right_child, .. }
                        | RegressionNode::ObliqueSplit { right_child, .. } => *right_child,
                        RegressionNode::Leaf { .. } | RegressionNode::MultiTargetLeaf { .. } => {
                            unreachable!("split node must exist")
                        }
                    };
                    [
                        ActiveNode {
                            node_index: left_child,
                            depth: pending.depth + 1,
                            start: pending.start,
                            end: pending.mid,
                            histograms: Some(left_histograms),
                        },
                        ActiveNode {
                            node_index: right_child,
                            depth: pending.depth + 1,
                            start: pending.mid,
                            end: pending.end,
                            histograms: Some(right_histograms),
                        },
                    ]
                })
                .collect::<Vec<_>>()
        } else {
            pending_splits
                .into_iter()
                .flat_map(|pending| {
                    let parent_histograms = pending
                        .evaluation
                        .histograms
                        .expect("splittable second-order node must retain histograms");
                    let left_len = pending.mid - pending.start;
                    let right_len = pending.end - pending.mid;
                    let (left_histograms, right_histograms) = if left_len <= right_len {
                        let left_histograms = build_second_order_feature_histograms(
                            context.table,
                            context.gradients,
                            context.hessians,
                            &rows[pending.start..pending.mid],
                            Parallelism::sequential(),
                        );
                        let right_histograms =
                            subtract_feature_histograms(&parent_histograms, &left_histograms);
                        (left_histograms, right_histograms)
                    } else {
                        let right_histograms = build_second_order_feature_histograms(
                            context.table,
                            context.gradients,
                            context.hessians,
                            &rows[pending.mid..pending.end],
                            Parallelism::sequential(),
                        );
                        let left_histograms =
                            subtract_feature_histograms(&parent_histograms, &right_histograms);
                        (left_histograms, right_histograms)
                    };
                    let left_child = match &nodes[pending.node_index] {
                        RegressionNode::BinarySplit { left_child, .. }
                        | RegressionNode::ObliqueSplit { left_child, .. } => *left_child,
                        RegressionNode::Leaf { .. } | RegressionNode::MultiTargetLeaf { .. } => {
                            unreachable!("split node must exist")
                        }
                    };
                    let right_child = match &nodes[pending.node_index] {
                        RegressionNode::BinarySplit { right_child, .. }
                        | RegressionNode::ObliqueSplit { right_child, .. } => *right_child,
                        RegressionNode::Leaf { .. } | RegressionNode::MultiTargetLeaf { .. } => {
                            unreachable!("split node must exist")
                        }
                    };
                    [
                        ActiveNode {
                            node_index: left_child,
                            depth: pending.depth + 1,
                            start: pending.start,
                            end: pending.mid,
                            histograms: Some(left_histograms),
                        },
                        ActiveNode {
                            node_index: right_child,
                            depth: pending.depth + 1,
                            start: pending.mid,
                            end: pending.end,
                            histograms: Some(right_histograms),
                        },
                    ]
                })
                .collect::<Vec<_>>()
        };
    }

    root
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
    let max_depth = options.tree_options.max_depth;
    for depth in 0..max_depth {
        if leaves
            .iter()
            .all(|leaf| leaf.len() < options.tree_options.min_samples_split)
        {
            break;
        }

        // Feature subsampling is performed per level. That keeps oblivious trees
        // faithful to their "one split per depth" structure while still allowing
        // forest/boosting-style stochasticity.
        let feature_indices = candidate_feature_indices(
            table,
            options.tree_options.max_features,
            node_seed(options.tree_options.random_seed, depth, &[], 0x0B11_A10Cu64),
        );
        let split_candidates = if parallelism.enabled() {
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
                        &options,
                    )
                })
                .collect::<Vec<_>>()
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
                        &options,
                    )
                })
                .collect::<Vec<_>>()
        };
        let (search_depth, top_k, future_weight) =
            if matches!(options.tree_options.builder, BuilderStrategy::Optimal) {
                (None, split_candidates.len(), 1.0)
            } else {
                (
                    Some(options.tree_options.effective_lookahead_depth()),
                    options.tree_options.lookahead_top_k,
                    options.tree_options.lookahead_weight,
                )
            };
        let selection = select_best_non_canary_candidate(
            table,
            rank_oblivious_split_choices_with_limits(
                table,
                &row_indices,
                gradients,
                hessians,
                &leaves,
                &options,
                depth,
                split_candidates,
                search_depth,
                top_k,
                future_weight,
            ),
            options.tree_options.canary_filter,
            |candidate| candidate.ranking_score,
            |candidate| candidate.choice.feature_index,
        );
        let Some(best_split) = selection
            .selected
            .map(|candidate| candidate.choice)
            .filter(|split| split.gain > options.min_gain_to_split)
        else {
            if depth == 0 && selection.blocked_by_canary {
                root_canary_selected = true;
            }
            break;
        };

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
            leaf_values: leaves
                .iter()
                .map(|leaf| leaf.prediction(&options))
                .collect(),
            leaf_sample_counts: leaves.iter().map(ObliviousLeafState::len).collect(),
            leaf_variances: vec![None; leaves.len()],
        },
        root_canary_selected,
    )
}

fn partition_pending_splits_in_place(
    table: &dyn TableAccess,
    rows: &mut [usize],
    pending_splits: &mut [PendingSplit],
    parallelism: Parallelism,
) {
    if pending_splits.is_empty() {
        return;
    }
    partition_pending_splits_recursive(table, rows, pending_splits, 0, parallelism);
}

fn partition_pending_splits_recursive(
    table: &dyn TableAccess,
    rows: &mut [usize],
    pending_splits: &mut [PendingSplit],
    base_offset: usize,
    parallelism: Parallelism,
) {
    if pending_splits.is_empty() {
        return;
    }
    if pending_splits.len() == 1 {
        let pending = &mut pending_splits[0];
        let local_start = pending.start - base_offset;
        let local_end = pending.end - base_offset;
        let split = pending
            .evaluation
            .best_split
            .clone()
            .expect("pending split must retain split choice");
        let left_count = match split {
            StandardSplitChoice::Axis(split) => partition_rows_for_binary_split(
                table,
                split.feature_index,
                split.threshold_bin,
                MissingBranchDirection::Right,
                &mut rows[local_start..local_end],
            ),
            StandardSplitChoice::Oblique(split) => partition_rows_for_oblique_split(
                table,
                [split.feature_indices[0], split.feature_indices[1]],
                [split.weights[0], split.weights[1]],
                split.threshold,
                [split.missing_directions[0], split.missing_directions[1]],
                &mut rows[local_start..local_end],
            ),
        };
        pending.mid = pending.start + left_count;
        return;
    }

    let pending_len = pending_splits.len();
    let mid_index = pending_len / 2;
    let split_start = pending_splits[mid_index].start;
    let split_offset = split_start - base_offset;
    let (left_rows, right_rows) = rows.split_at_mut(split_offset);
    let (left_pending, right_pending) = pending_splits.split_at_mut(mid_index);

    if parallelism.enabled() && pending_len >= 4 {
        rayon::join(
            || {
                partition_pending_splits_recursive(
                    table,
                    left_rows,
                    left_pending,
                    base_offset,
                    parallelism,
                )
            },
            || {
                partition_pending_splits_recursive(
                    table,
                    right_rows,
                    right_pending,
                    split_start,
                    parallelism,
                )
            },
        );
    } else {
        partition_pending_splits_recursive(
            table,
            left_rows,
            left_pending,
            base_offset,
            parallelism,
        );
        partition_pending_splits_recursive(
            table,
            right_rows,
            right_pending,
            split_start,
            parallelism,
        );
    }
}

fn evaluate_standard_node(
    context: &BuildContext<'_>,
    rows: &[usize],
    depth: usize,
    histograms: Option<Vec<SecondOrderFeatureHistogram>>,
) -> StandardNodeEvaluation {
    let stats = node_stats(rows, context.gradients, context.hessians);
    let leaf_prediction = leaf_value(
        stats.gradient_sum,
        stats.hessian_sum,
        context.options.l2_regularization,
    );
    let sample_count = rows.len();
    let parent_strength = node_objective_strength(
        stats.gradient_sum,
        stats.hessian_sum,
        context.options.l2_regularization,
    );

    if rows.is_empty()
        || depth >= context.options.tree_options.max_depth
        || rows.len() < context.options.tree_options.min_samples_split
        || stats.hessian_sum <= 0.0
    {
        return StandardNodeEvaluation {
            sample_count,
            leaf_prediction,
            parent_strength,
            histograms: None,
            best_split: None,
            blocked_by_canary: false,
        };
    }

    let histograms = histograms.unwrap_or_else(|| {
        build_second_order_feature_histograms(
            context.table,
            context.gradients,
            context.hessians,
            rows,
            context.parallelism,
        )
    });
    let feature_indices = candidate_feature_indices(
        context.table,
        context.options.tree_options.max_features,
        node_seed(
            context.options.tree_options.random_seed,
            depth,
            rows,
            0xA11C_E5E1u64,
        ),
    );
    let split_candidates = if context.parallelism.enabled() {
        feature_indices
            .par_iter()
            .filter_map(|feature_index| {
                score_feature_from_hist(context, &histograms[*feature_index], *feature_index, rows)
            })
            .collect::<Vec<_>>()
    } else {
        feature_indices
            .iter()
            .filter_map(|feature_index| {
                score_feature_from_hist(context, &histograms[*feature_index], *feature_index, rows)
            })
            .collect::<Vec<_>>()
    };
    let selection = select_best_non_canary_candidate(
        context.table,
        rank_standard_split_choices(
            context,
            rows,
            depth,
            &split_candidates,
            &feature_indices,
            context.options.tree_options.effective_lookahead_depth(),
        ),
        context.options.tree_options.canary_filter,
        |candidate| candidate.ranking_score,
        |candidate| candidate.choice.ranking_feature_index(),
    );

    StandardNodeEvaluation {
        sample_count,
        leaf_prediction,
        parent_strength,
        histograms: Some(histograms),
        best_split: selection.selected.map(|candidate| candidate.choice),
        blocked_by_canary: selection.blocked_by_canary,
    }
}

fn score_standard_split_choices(
    context: &BuildContext<'_>,
    rows: &[usize],
    axis_candidates: &[SplitChoice],
    candidate_features: &[usize],
) -> Vec<StandardSplitChoice> {
    let real_features = candidate_features
        .iter()
        .copied()
        .filter(|feature_index| *feature_index < context.table.n_features())
        .collect::<Vec<_>>();
    let mut ranked = axis_candidates
        .iter()
        .cloned()
        .map(StandardSplitChoice::Axis)
        .collect::<Vec<_>>();
    if real_features.len() < 2 {
        return ranked;
    }

    let real_pairs = all_feature_pairs(&real_features);
    ranked.extend(
        collect_oblique_split_choices(context, rows, &real_pairs)
            .into_iter()
            .map(StandardSplitChoice::Oblique),
    );
    let canary_pairs = matched_canary_feature_pairs(context.table, &real_features);
    ranked.extend(
        collect_oblique_split_choices(context, rows, &canary_pairs)
            .into_iter()
            .map(StandardSplitChoice::Oblique),
    );
    ranked
}

fn rank_standard_split_choices(
    context: &BuildContext<'_>,
    rows: &[usize],
    depth: usize,
    axis_candidates: &[SplitChoice],
    candidate_features: &[usize],
    lookahead_depth: usize,
) -> Vec<RankedStandardSplitChoice> {
    let candidates = if matches!(
        context.options.tree_options.split_strategy,
        SplitStrategy::Oblique
    ) && matches!(
        context.algorithm,
        RegressionTreeAlgorithm::Cart | RegressionTreeAlgorithm::Randomized
    ) {
        score_standard_split_choices(context, rows, axis_candidates, candidate_features)
    } else {
        axis_candidates
            .iter()
            .cloned()
            .map(StandardSplitChoice::Axis)
            .collect()
    };
    let (search_depth, top_k, future_weight) = if matches!(
        context.options.tree_options.builder,
        BuilderStrategy::Optimal
    ) {
        (None, candidates.len(), 1.0)
    } else {
        (
            Some(lookahead_depth),
            context.options.tree_options.lookahead_top_k,
            context.options.tree_options.lookahead_weight,
        )
    };
    rank_standard_split_choices_with_limits(
        context,
        rows,
        depth,
        candidates,
        search_depth,
        top_k,
        future_weight,
    )
}

fn rank_standard_split_choices_with_limits(
    context: &BuildContext<'_>,
    rows: &[usize],
    depth: usize,
    candidates: Vec<StandardSplitChoice>,
    search_depth: Option<usize>,
    top_k: usize,
    future_weight: f64,
) -> Vec<RankedStandardSplitChoice> {
    let mut shortlist = candidates
        .iter()
        .enumerate()
        .map(|(index, choice)| (index, choice.gain()))
        .collect::<Vec<_>>();
    shortlist.sort_by(|left, right| right.1.total_cmp(&left.1));
    let shortlisted = shortlist
        .into_iter()
        .take(top_k)
        .map(|(index, _)| index)
        .collect::<std::collections::BTreeSet<_>>();
    candidates
        .into_iter()
        .enumerate()
        .map(|(index, choice)| RankedStandardSplitChoice {
            ranking_score: if shortlisted.contains(&index) {
                standard_split_recursive_ranking_score(
                    context,
                    rows,
                    depth,
                    &choice,
                    search_depth,
                    future_weight,
                )
            } else {
                choice.gain()
            },
            choice,
        })
        .collect()
}

fn standard_split_recursive_ranking_score(
    context: &BuildContext<'_>,
    rows: &[usize],
    depth: usize,
    choice: &StandardSplitChoice,
    search_depth: Option<usize>,
    future_weight: f64,
) -> f64 {
    let immediate = choice.gain();
    if immediate <= context.options.min_gain_to_split
        || depth + 1 >= context.options.tree_options.max_depth
        || search_depth.is_some_and(|remaining| remaining <= 1)
    {
        return immediate;
    }

    let mut partitioned_rows = rows.to_vec();
    let left_count = match choice {
        StandardSplitChoice::Axis(split) => partition_rows_for_binary_split(
            context.table,
            split.feature_index,
            split.threshold_bin,
            MissingBranchDirection::Right,
            &mut partitioned_rows,
        ),
        StandardSplitChoice::Oblique(split) => partition_rows_for_oblique_split(
            context.table,
            [split.feature_indices[0], split.feature_indices[1]],
            [split.weights[0], split.weights[1]],
            split.threshold,
            [split.missing_directions[0], split.missing_directions[1]],
            &mut partitioned_rows,
        ),
    };
    let (left_rows, right_rows) = partitioned_rows.split_at_mut(left_count);
    let future = best_standard_split_recursive_score(
        context,
        left_rows,
        depth + 1,
        search_depth.map(|remaining| remaining - 1),
        context.options.tree_options.effective_beam_width(),
        future_weight,
    ) + best_standard_split_recursive_score(
        context,
        right_rows,
        depth + 1,
        search_depth.map(|remaining| remaining - 1),
        context.options.tree_options.effective_beam_width(),
        future_weight,
    );
    immediate + future_weight * future
}

fn best_standard_split_recursive_score(
    context: &BuildContext<'_>,
    rows: &mut [usize],
    depth: usize,
    search_depth: Option<usize>,
    beam_width: usize,
    future_weight: f64,
) -> f64 {
    if rows.is_empty()
        || search_depth == Some(0)
        || depth >= context.options.tree_options.max_depth
        || rows.len() < context.options.tree_options.min_samples_split
    {
        return 0.0;
    }
    let stats = sum_gradient_hessian_stats(rows, context.gradients, context.hessians);
    if stats.1 <= 0.0 {
        return 0.0;
    }

    let histograms = build_second_order_feature_histograms(
        context.table,
        context.gradients,
        context.hessians,
        rows,
        Parallelism::sequential(),
    );
    let feature_indices = candidate_feature_indices(
        context.table,
        context.options.tree_options.max_features,
        node_seed(
            context.options.tree_options.random_seed,
            depth,
            rows,
            0xA11C_E5E1u64,
        ),
    );
    let split_candidates = feature_indices
        .iter()
        .filter_map(|feature_index| {
            score_feature_from_hist(context, &histograms[*feature_index], *feature_index, rows)
        })
        .collect::<Vec<_>>();
    let candidates = if matches!(
        context.options.tree_options.split_strategy,
        SplitStrategy::Oblique
    ) && matches!(
        context.algorithm,
        RegressionTreeAlgorithm::Cart | RegressionTreeAlgorithm::Randomized
    ) {
        score_standard_split_choices(context, rows, &split_candidates, &feature_indices)
    } else {
        split_candidates
            .into_iter()
            .map(StandardSplitChoice::Axis)
            .collect()
    };
    let top_k = if search_depth.is_none() {
        candidates.len()
    } else {
        context.options.tree_options.lookahead_top_k
    };
    let ranked = rank_standard_split_choices_with_limits(
        context,
        rows,
        depth,
        candidates,
        search_depth,
        top_k,
        future_weight,
    );
    aggregate_beam_non_canary_score(
        context.table,
        ranked,
        context.options.tree_options.canary_filter,
        beam_width,
        |candidate| candidate.ranking_score,
        |candidate| candidate.choice.ranking_feature_index(),
    )
}

fn rank_optimal_standard_split_choices(
    context: &BuildContext<'_>,
    rows: &[usize],
    depth: usize,
    candidates: Vec<StandardSplitChoice>,
) -> Vec<RankedStandardSplitChoice> {
    candidates
        .into_iter()
        .map(|choice| RankedStandardSplitChoice {
            ranking_score: optimal_standard_split_ranking_score(context, rows, depth, &choice),
            choice,
        })
        .collect()
}

fn optimal_standard_split_ranking_score(
    context: &BuildContext<'_>,
    rows: &[usize],
    depth: usize,
    choice: &StandardSplitChoice,
) -> f64 {
    let immediate = choice.gain();
    if immediate <= context.options.min_gain_to_split
        || depth + 1 >= context.options.tree_options.max_depth
    {
        return immediate;
    }

    let mut partitioned_rows = rows.to_vec();
    let left_count = match choice {
        StandardSplitChoice::Axis(split) => partition_rows_for_binary_split(
            context.table,
            split.feature_index,
            split.threshold_bin,
            MissingBranchDirection::Right,
            &mut partitioned_rows,
        ),
        StandardSplitChoice::Oblique(split) => partition_rows_for_oblique_split(
            context.table,
            [split.feature_indices[0], split.feature_indices[1]],
            [split.weights[0], split.weights[1]],
            split.threshold,
            [split.missing_directions[0], split.missing_directions[1]],
            &mut partitioned_rows,
        ),
    };
    let (left_rows, right_rows) = partitioned_rows.split_at_mut(left_count);
    immediate
        + best_standard_split_optimal_score(context, left_rows, depth + 1)
        + best_standard_split_optimal_score(context, right_rows, depth + 1)
}

fn best_standard_split_optimal_score(
    context: &BuildContext<'_>,
    rows: &mut [usize],
    depth: usize,
) -> f64 {
    if rows.is_empty()
        || depth >= context.options.tree_options.max_depth
        || rows.len() < context.options.tree_options.min_samples_split
    {
        return 0.0;
    }
    let stats = sum_gradient_hessian_stats(rows, context.gradients, context.hessians);
    if stats.1 <= 0.0 {
        return 0.0;
    }

    let histograms = build_second_order_feature_histograms(
        context.table,
        context.gradients,
        context.hessians,
        rows,
        Parallelism::sequential(),
    );
    let feature_indices = candidate_feature_indices(
        context.table,
        context.options.tree_options.max_features,
        node_seed(
            context.options.tree_options.random_seed,
            depth,
            rows,
            0xA11C_E5E1u64,
        ),
    );
    let split_candidates = feature_indices
        .iter()
        .filter_map(|feature_index| {
            score_feature_from_hist(context, &histograms[*feature_index], *feature_index, rows)
        })
        .collect::<Vec<_>>();
    let candidates = if matches!(
        context.options.tree_options.split_strategy,
        SplitStrategy::Oblique
    ) && matches!(
        context.algorithm,
        RegressionTreeAlgorithm::Cart | RegressionTreeAlgorithm::Randomized
    ) {
        score_standard_split_choices(context, rows, &split_candidates, &feature_indices)
    } else {
        split_candidates
            .into_iter()
            .map(StandardSplitChoice::Axis)
            .collect()
    };
    select_best_non_canary_candidate(
        context.table,
        rank_optimal_standard_split_choices(context, rows, depth, candidates),
        context.options.tree_options.canary_filter,
        |candidate| candidate.ranking_score,
        |candidate| candidate.choice.ranking_feature_index(),
    )
    .selected
    .map(|candidate| candidate.ranking_score.max(0.0))
    .unwrap_or(0.0)
}

fn best_standard_split_lookahead_score(
    context: &BuildContext<'_>,
    rows: &mut [usize],
    depth: usize,
    lookahead_depth: usize,
    beam_width: usize,
) -> f64 {
    if rows.is_empty()
        || lookahead_depth == 0
        || depth >= context.options.tree_options.max_depth
        || rows.len() < context.options.tree_options.min_samples_split
    {
        return 0.0;
    }
    let stats = sum_gradient_hessian_stats(rows, context.gradients, context.hessians);
    if stats.1 <= 0.0 {
        return 0.0;
    }

    let histograms = build_second_order_feature_histograms(
        context.table,
        context.gradients,
        context.hessians,
        rows,
        Parallelism::sequential(),
    );
    let feature_indices = candidate_feature_indices(
        context.table,
        context.options.tree_options.max_features,
        node_seed(
            context.options.tree_options.random_seed,
            depth,
            rows,
            0xA11C_E5E1u64,
        ),
    );
    let split_candidates = feature_indices
        .iter()
        .filter_map(|feature_index| {
            score_feature_from_hist(context, &histograms[*feature_index], *feature_index, rows)
        })
        .collect::<Vec<_>>();
    let ranked = rank_standard_split_choices(
        context,
        rows,
        depth,
        &split_candidates,
        &feature_indices,
        lookahead_depth,
    );
    aggregate_beam_non_canary_score(
        context.table,
        ranked,
        context.options.tree_options.canary_filter,
        beam_width,
        |candidate| candidate.ranking_score,
        |candidate| match &candidate.choice {
            StandardSplitChoice::Axis(split) => split.feature_index,
            StandardSplitChoice::Oblique(split) => split.feature_indices[0],
        },
    )
}

fn rank_shortlisted_candidates(
    candidates: Vec<StandardSplitChoice>,
    top_k: usize,
    immediate_score: fn(&StandardSplitChoice) -> f64,
    rescore: impl Fn(&StandardSplitChoice) -> f64,
) -> Vec<RankedStandardSplitChoice> {
    let mut shortlist = candidates
        .iter()
        .enumerate()
        .map(|(index, choice)| (index, immediate_score(choice)))
        .collect::<Vec<_>>();
    shortlist.sort_by(|left, right| right.1.total_cmp(&left.1));
    let shortlisted = shortlist
        .into_iter()
        .take(top_k)
        .map(|(index, _)| index)
        .collect::<std::collections::BTreeSet<_>>();
    candidates
        .into_iter()
        .enumerate()
        .map(|(index, choice)| RankedStandardSplitChoice {
            ranking_score: if shortlisted.contains(&index) {
                rescore(&choice)
            } else {
                immediate_score(&choice)
            },
            choice,
        })
        .collect()
}

fn collect_oblique_split_choices(
    context: &BuildContext<'_>,
    rows: &[usize],
    feature_pairs: &[[usize; 2]],
) -> Vec<ObliqueSplitChoice> {
    let mut candidates = Vec::new();
    for &feature_pair in feature_pairs {
        let observed_rows = rows
            .iter()
            .copied()
            .filter(|row_index| missing_mask_for_pair(context.table, feature_pair, *row_index) == 0)
            .collect::<Vec<_>>();
        if observed_rows.len() < 2 {
            continue;
        }
        let Some(weights) = oblique_second_order_weights(
            context.table,
            &observed_rows,
            context.gradients,
            feature_pair,
        ) else {
            continue;
        };
        let Some(projected) =
            projected_rows_for_pair(context.table, &observed_rows, feature_pair, weights)
        else {
            continue;
        };
        let mut missing_rows_by_mask = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for &row_index in rows {
            let mask = missing_mask_for_pair(context.table, feature_pair, row_index) as usize;
            if mask != 0 {
                missing_rows_by_mask[mask].push(row_index);
            }
        }
        let direction_choices_0 =
            if missing_rows_by_mask[1].is_empty() && missing_rows_by_mask[3].is_empty() {
                vec![MissingBranchDirection::Node]
            } else {
                vec![MissingBranchDirection::Left, MissingBranchDirection::Right]
            };
        let direction_choices_1 =
            if missing_rows_by_mask[2].is_empty() && missing_rows_by_mask[3].is_empty() {
                vec![MissingBranchDirection::Node]
            } else {
                vec![MissingBranchDirection::Left, MissingBranchDirection::Right]
            };
        let total_gradient_sum = projected
            .iter()
            .map(|projected_row| context.gradients[projected_row.row_index])
            .sum::<f64>();
        let total_hessian_sum = projected
            .iter()
            .map(|projected_row| context.hessians[projected_row.row_index])
            .sum::<f64>();
        let mut left_gradient_sum = 0.0;
        let mut left_hessian_sum = 0.0;
        for split_index in 0..projected.len().saturating_sub(1) {
            let row_index = projected[split_index].row_index;
            left_gradient_sum += context.gradients[row_index];
            left_hessian_sum += context.hessians[row_index];
            if projected[split_index].value == projected[split_index + 1].value {
                continue;
            }
            let threshold = (projected[split_index].value + projected[split_index + 1].value) / 2.0;
            for &direction_0 in &direction_choices_0 {
                for &direction_1 in &direction_choices_1 {
                    let missing_directions = [direction_0, direction_1];
                    let mut candidate_left_count = split_index + 1;
                    let mut candidate_left_gradient_sum = left_gradient_sum;
                    let mut candidate_left_hessian_sum = left_hessian_sum;
                    let mut candidate_right_count = projected.len() - candidate_left_count;
                    let mut candidate_right_gradient_sum = total_gradient_sum - left_gradient_sum;
                    let mut candidate_right_hessian_sum = total_hessian_sum - left_hessian_sum;
                    for mask in [1u8, 2, 3] {
                        let target_rows = &missing_rows_by_mask[mask as usize];
                        if target_rows.is_empty() {
                            continue;
                        }
                        let Some(go_left) =
                            resolve_oblique_missing_direction(mask, weights, missing_directions)
                        else {
                            continue;
                        };
                        for &missing_row in target_rows {
                            let gradient = context.gradients[missing_row];
                            let hessian = context.hessians[missing_row];
                            if go_left {
                                candidate_left_count += 1;
                                candidate_left_gradient_sum += gradient;
                                candidate_left_hessian_sum += hessian;
                                candidate_right_count -= 1;
                                candidate_right_gradient_sum -= gradient;
                                candidate_right_hessian_sum -= hessian;
                            }
                        }
                    }
                    if !children_are_splittable(
                        &context.options,
                        candidate_left_count,
                        candidate_left_hessian_sum,
                        candidate_right_count,
                        candidate_right_hessian_sum,
                    ) {
                        continue;
                    }
                    let gain = split_gain(
                        candidate_left_gradient_sum,
                        candidate_left_hessian_sum,
                        candidate_right_gradient_sum,
                        candidate_right_hessian_sum,
                        candidate_left_gradient_sum + candidate_right_gradient_sum,
                        candidate_left_hessian_sum + candidate_right_hessian_sum,
                        context.options.l2_regularization,
                    );
                    if !gain.is_finite() {
                        continue;
                    }
                    candidates.push(ObliqueSplitChoice {
                        feature_indices: vec![feature_pair[0], feature_pair[1]],
                        weights: vec![weights[0], weights[1]],
                        missing_directions: vec![direction_0, direction_1],
                        threshold,
                        gain,
                    });
                }
            }
        }
    }
    candidates
}

fn oblique_second_order_weights(
    table: &dyn TableAccess,
    rows: &[usize],
    gradients: &[f64],
    feature_pair: [usize; 2],
) -> Option<[f64; 2]> {
    let feature_values = rows
        .iter()
        .map(|&row_index| {
            Some([
                oblique_feature_value(table, feature_pair[0], row_index)?,
                oblique_feature_value(table, feature_pair[1], row_index)?,
            ])
        })
        .collect::<Option<Vec<_>>>()?;
    let mean_gradient = rows
        .iter()
        .map(|row_index| gradients[*row_index])
        .sum::<f64>()
        / rows.len() as f64;
    let mean_x0 = feature_values.iter().map(|values| values[0]).sum::<f64>() / rows.len() as f64;
    let mean_x1 = feature_values.iter().map(|values| values[1]).sum::<f64>() / rows.len() as f64;
    let mut weights = [0.0; 2];
    for (&row_index, values) in rows.iter().zip(feature_values.iter()) {
        let centered_gradient = gradients[row_index] - mean_gradient;
        weights[0] += (values[0] - mean_x0) * centered_gradient;
        weights[1] += (values[1] - mean_x1) * centered_gradient;
    }
    normalize_weights(weights)
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

fn build_second_order_feature_histograms(
    table: &dyn TableAccess,
    gradients: &[f64],
    hessians: &[f64],
    rows: &[usize],
    parallelism: Parallelism,
) -> Vec<SecondOrderFeatureHistogram> {
    let make_bin = |_| SecondOrderHistogramBin {
        count: 0,
        gradient_sum: 0.0,
        hessian_sum: 0.0,
    };
    let add_row = |_feature_index, payload: &mut SecondOrderHistogramBin, row_idx| {
        payload.count += 1;
        payload.gradient_sum += gradients[row_idx];
        payload.hessian_sum += hessians[row_idx];
    };

    if parallelism.enabled() {
        build_feature_histograms_parallel(table, rows, make_bin, add_row)
    } else {
        build_feature_histograms(table, rows, make_bin, add_row)
    }
}

fn score_feature_from_hist(
    context: &BuildContext<'_>,
    histogram: &SecondOrderFeatureHistogram,
    feature_index: usize,
    rows: &[usize],
) -> Option<SplitChoice> {
    match histogram {
        FeatureHistogram::Binary {
            false_bin,
            true_bin,
            ..
        } => score_binary_split_from_stats(
            context,
            feature_index,
            false_bin.count,
            false_bin.gradient_sum,
            false_bin.hessian_sum,
            true_bin.count,
            true_bin.gradient_sum,
            true_bin.hessian_sum,
        ),
        FeatureHistogram::Numeric {
            bins,
            observed_bins,
        } => match context.algorithm {
            RegressionTreeAlgorithm::Cart => score_numeric_split_from_hist(
                context,
                feature_index,
                rows.len(),
                bins,
                observed_bins,
            ),
            RegressionTreeAlgorithm::Randomized => {
                score_randomized_split_from_hist(context, feature_index, rows, bins, observed_bins)
            }
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
        &context.options,
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
    bins: &[SecondOrderHistogramBin],
    observed_bins: &[usize],
) -> Option<SplitChoice> {
    if observed_bins.len() <= 1 {
        return None;
    }

    let total_gradient_sum = bins.iter().map(|bin| bin.gradient_sum).sum::<f64>();
    let total_hessian_sum = bins.iter().map(|bin| bin.hessian_sum).sum::<f64>();
    let mut left_count = 0usize;
    let mut left_gradient_sum = 0.0;
    let mut left_hessian_sum = 0.0;
    let mut best_threshold = None;
    let mut best_gain = f64::NEG_INFINITY;

    for &bin in observed_bins {
        left_count += bins[bin].count;
        left_gradient_sum += bins[bin].gradient_sum;
        left_hessian_sum += bins[bin].hessian_sum;

        let right_count = row_count - left_count;
        let right_gradient_sum = total_gradient_sum - left_gradient_sum;
        let right_hessian_sum = total_hessian_sum - left_hessian_sum;

        if !children_are_splittable(
            &context.options,
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
    bins: &[SecondOrderHistogramBin],
    observed_bins: &[usize],
) -> Option<SplitChoice> {
    let candidate_thresholds = observed_bins
        .iter()
        .copied()
        .map(|bin| bin as u16)
        .collect::<Vec<_>>();
    let threshold_bin =
        choose_random_threshold(&candidate_thresholds, feature_index, rows, 0xA11C_E551u64)?;

    let total_gradient_sum = bins.iter().map(|bin| bin.gradient_sum).sum::<f64>();
    let total_hessian_sum = bins.iter().map(|bin| bin.hessian_sum).sum::<f64>();
    let mut left_count = 0usize;
    let mut left_gradient_sum = 0.0;
    let mut left_hessian_sum = 0.0;
    for bin in 0..=threshold_bin as usize {
        if bin >= bins.len() {
            break;
        }
        left_count += bins[bin].count;
        left_gradient_sum += bins[bin].gradient_sum;
        left_hessian_sum += bins[bin].hessian_sum;
    }
    let right_count = rows.len() - left_count;
    let right_gradient_sum = total_gradient_sum - left_gradient_sum;
    let right_hessian_sum = total_hessian_sum - left_hessian_sum;
    if !children_are_splittable(
        &context.options,
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
    options: &SecondOrderRegressionTreeOptions,
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
    let mut threshold_valid = vec![true; bin_cap];

    let mut bin_count = vec![0usize; bin_cap];
    let mut bin_gradient_sum = vec![0.0; bin_cap];
    let mut bin_hessian_sum = vec![0.0; bin_cap];
    let mut observed_bins = vec![false; bin_cap];

    for leaf in leaves {
        bin_count.fill(0);
        bin_gradient_sum.fill(0.0);
        bin_hessian_sum.fill(0.0);
        observed_bins.fill(false);

        for row_idx in &row_indices[leaf.start..leaf.end] {
            let bin = table.binned_value(feature_index, *row_idx) as usize;
            bin_count[bin] += 1;
            bin_gradient_sum[bin] += gradients[*row_idx];
            bin_hessian_sum[bin] += hessians[*row_idx];
            observed_bins[bin] = true;
        }

        let observed_bins = observed_bins
            .iter()
            .enumerate()
            .filter_map(|(bin, seen)| (*seen).then_some(bin))
            .collect::<Vec<_>>();
        if observed_bins.len() <= 1 {
            return None;
        }

        let mut left_count = 0usize;
        let mut left_gradient_sum = 0.0;
        let mut left_hessian_sum = 0.0;
        let mut leaf_valid = vec![false; bin_cap];
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
            leaf_valid[bin] = true;
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
        for threshold_bin in 0..bin_cap {
            threshold_valid[threshold_bin] &= leaf_valid[threshold_bin];
        }
    }

    threshold_gains
        .into_iter()
        .enumerate()
        .filter(|(bin, gain)| threshold_valid[*bin] && *gain > options.min_gain_to_split)
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
    options: &SecondOrderRegressionTreeOptions,
) -> Option<SplitChoice> {
    let mut gain = 0.0;

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
            return None;
        }
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

    (gain > options.min_gain_to_split).then_some(SplitChoice {
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
            MissingBranchDirection::Right,
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

#[allow(clippy::too_many_arguments)]
fn oblivious_split_ranking_score(
    table: &dyn TableAccess,
    row_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    leaves: &[ObliviousLeafState],
    options: &SecondOrderRegressionTreeOptions,
    depth: usize,
    candidate: &SplitChoice,
    lookahead_depth: usize,
) -> f64 {
    oblivious_split_recursive_ranking_score(
        table,
        row_indices,
        gradients,
        hessians,
        leaves,
        options,
        depth,
        candidate,
        Some(lookahead_depth),
        options.tree_options.lookahead_weight,
    )
}

#[allow(clippy::too_many_arguments)]
fn rank_oblivious_split_choices_with_limits(
    table: &dyn TableAccess,
    row_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    leaves: &[ObliviousLeafState],
    options: &SecondOrderRegressionTreeOptions,
    depth: usize,
    candidates: Vec<SplitChoice>,
    search_depth: Option<usize>,
    top_k: usize,
    future_weight: f64,
) -> Vec<RankedSplitChoice> {
    let mut shortlist = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (index, candidate.gain))
        .collect::<Vec<_>>();
    shortlist.sort_by(|left, right| right.1.total_cmp(&left.1));
    let shortlisted = shortlist
        .into_iter()
        .take(top_k)
        .map(|(index, _)| index)
        .collect::<std::collections::BTreeSet<_>>();
    candidates
        .into_iter()
        .enumerate()
        .map(|(index, choice)| RankedSplitChoice {
            ranking_score: if shortlisted.contains(&index) {
                oblivious_split_recursive_ranking_score(
                    table,
                    row_indices,
                    gradients,
                    hessians,
                    leaves,
                    options,
                    depth,
                    &choice,
                    search_depth,
                    future_weight,
                )
            } else {
                choice.gain
            },
            choice,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn oblivious_split_recursive_ranking_score(
    table: &dyn TableAccess,
    row_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    leaves: &[ObliviousLeafState],
    options: &SecondOrderRegressionTreeOptions,
    depth: usize,
    candidate: &SplitChoice,
    search_depth: Option<usize>,
    future_weight: f64,
) -> f64 {
    let immediate = candidate.gain;
    if immediate <= options.min_gain_to_split
        || depth + 1 >= options.tree_options.max_depth
        || search_depth.is_some_and(|remaining| remaining <= 1)
    {
        return immediate;
    }

    let mut next_row_indices = row_indices.to_vec();
    let next_leaves = split_oblivious_leaves_in_place(
        table,
        &mut next_row_indices,
        gradients,
        hessians,
        leaves.to_vec(),
        candidate.feature_index,
        candidate.threshold_bin,
    );
    let future = best_oblivious_split_recursive_score(
        table,
        &mut next_row_indices,
        gradients,
        hessians,
        next_leaves,
        options,
        depth + 1,
        search_depth.map(|remaining| remaining - 1),
        options.tree_options.effective_beam_width(),
        future_weight,
    );
    immediate + future_weight * future
}

#[allow(clippy::too_many_arguments)]
fn rank_optimal_oblivious_split_choices(
    table: &dyn TableAccess,
    row_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    leaves: &[ObliviousLeafState],
    options: &SecondOrderRegressionTreeOptions,
    depth: usize,
    candidates: Vec<SplitChoice>,
) -> Vec<RankedSplitChoice> {
    candidates
        .into_iter()
        .map(|choice| RankedSplitChoice {
            ranking_score: optimal_oblivious_split_ranking_score(
                table,
                row_indices,
                gradients,
                hessians,
                leaves,
                options,
                depth,
                &choice,
            ),
            choice,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn optimal_oblivious_split_ranking_score(
    table: &dyn TableAccess,
    row_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    leaves: &[ObliviousLeafState],
    options: &SecondOrderRegressionTreeOptions,
    depth: usize,
    candidate: &SplitChoice,
) -> f64 {
    let immediate = candidate.gain;
    if immediate <= options.min_gain_to_split || depth + 1 >= options.tree_options.max_depth {
        return immediate;
    }

    let mut next_row_indices = row_indices.to_vec();
    let next_leaves = split_oblivious_leaves_in_place(
        table,
        &mut next_row_indices,
        gradients,
        hessians,
        leaves.to_vec(),
        candidate.feature_index,
        candidate.threshold_bin,
    );
    immediate
        + best_oblivious_split_optimal_score(
            table,
            &mut next_row_indices,
            gradients,
            hessians,
            next_leaves,
            options,
            depth + 1,
        )
}

#[allow(clippy::too_many_arguments)]
fn best_oblivious_split_lookahead_score(
    table: &dyn TableAccess,
    row_indices: &mut [usize],
    gradients: &[f64],
    hessians: &[f64],
    leaves: Vec<ObliviousLeafState>,
    options: &SecondOrderRegressionTreeOptions,
    depth: usize,
    lookahead_depth: usize,
    beam_width: usize,
) -> f64 {
    best_oblivious_split_recursive_score(
        table,
        row_indices,
        gradients,
        hessians,
        leaves,
        options,
        depth,
        Some(lookahead_depth),
        beam_width,
        options.tree_options.lookahead_weight,
    )
}

#[allow(clippy::too_many_arguments)]
fn best_oblivious_split_recursive_score(
    table: &dyn TableAccess,
    row_indices: &mut [usize],
    gradients: &[f64],
    hessians: &[f64],
    leaves: Vec<ObliviousLeafState>,
    options: &SecondOrderRegressionTreeOptions,
    depth: usize,
    search_depth: Option<usize>,
    beam_width: usize,
    future_weight: f64,
) -> f64 {
    if leaves
        .iter()
        .all(|leaf| leaf.len() < options.tree_options.min_samples_split)
        || search_depth == Some(0)
        || depth >= options.tree_options.max_depth
    {
        return 0.0;
    }

    let feature_indices = candidate_feature_indices(
        table,
        options.tree_options.max_features,
        node_seed(options.tree_options.random_seed, depth, &[], 0x0B11_A10Cu64),
    );
    let split_candidates = feature_indices
        .into_iter()
        .filter_map(|feature_index| {
            score_oblivious_split(
                table,
                row_indices,
                gradients,
                hessians,
                feature_index,
                &leaves,
                options,
            )
        })
        .collect::<Vec<_>>();
    let top_k = if search_depth.is_none() {
        split_candidates.len()
    } else {
        options.tree_options.lookahead_top_k
    };
    let ranked = rank_oblivious_split_choices_with_limits(
        table,
        row_indices,
        gradients,
        hessians,
        &leaves,
        options,
        depth,
        split_candidates,
        search_depth,
        top_k,
        future_weight,
    );
    aggregate_beam_non_canary_score(
        table,
        ranked,
        options.tree_options.canary_filter,
        beam_width,
        |candidate| candidate.ranking_score,
        |candidate| candidate.choice.feature_index,
    )
}

#[allow(clippy::too_many_arguments)]
fn best_oblivious_split_optimal_score(
    table: &dyn TableAccess,
    row_indices: &mut [usize],
    gradients: &[f64],
    hessians: &[f64],
    leaves: Vec<ObliviousLeafState>,
    options: &SecondOrderRegressionTreeOptions,
    depth: usize,
) -> f64 {
    if leaves
        .iter()
        .all(|leaf| leaf.len() < options.tree_options.min_samples_split)
        || depth >= options.tree_options.max_depth
    {
        return 0.0;
    }

    let feature_indices = candidate_feature_indices(
        table,
        options.tree_options.max_features,
        node_seed(options.tree_options.random_seed, depth, &[], 0x0B11_A10Cu64),
    );
    let split_candidates = feature_indices
        .into_iter()
        .filter_map(|feature_index| {
            score_oblivious_split(
                table,
                row_indices,
                gradients,
                hessians,
                feature_index,
                &leaves,
                options,
            )
        })
        .collect::<Vec<_>>();
    select_best_non_canary_candidate(
        table,
        rank_optimal_oblivious_split_choices(
            table,
            row_indices,
            gradients,
            hessians,
            &leaves,
            options,
            depth,
            split_candidates,
        ),
        options.tree_options.canary_filter,
        |candidate| candidate.ranking_score,
        |candidate| candidate.choice.feature_index,
    )
    .selected
    .map(|candidate| candidate.ranking_score.max(0.0))
    .unwrap_or(0.0)
}

fn rank_shortlisted_oblivious_candidates(
    candidates: Vec<SplitChoice>,
    top_k: usize,
    rescore: impl Fn(&SplitChoice) -> f64,
) -> Vec<RankedSplitChoice> {
    let mut shortlist = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (index, candidate.gain))
        .collect::<Vec<_>>();
    shortlist.sort_by(|left, right| right.1.total_cmp(&left.1));
    let shortlisted = shortlist
        .into_iter()
        .take(top_k)
        .map(|(index, _)| index)
        .collect::<std::collections::BTreeSet<_>>();
    candidates
        .into_iter()
        .enumerate()
        .map(|(index, choice)| RankedSplitChoice {
            ranking_score: if shortlisted.contains(&index) {
                rescore(&choice)
            } else {
                choice.gain
            },
            choice,
        })
        .collect()
}

fn children_are_splittable(
    options: &SecondOrderRegressionTreeOptions,
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

    fn randomized_permutation_table() -> DenseTable {
        DenseTable::with_options(
            vec![
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 1.0, 1.0],
                vec![1.0, 0.0, 0.0],
                vec![1.0, 0.0, 1.0],
                vec![1.0, 1.0, 0.0],
                vec![1.0, 1.0, 1.0],
                vec![0.0, 0.0, 2.0],
                vec![0.0, 1.0, 2.0],
                vec![1.0, 0.0, 2.0],
                vec![1.0, 1.0, 2.0],
            ],
            vec![0.0; 12],
            0,
            NumericBins::Fixed(8),
        )
        .unwrap()
    }

    fn oblique_second_order_table() -> DenseTable {
        DenseTable::with_options(
            vec![
                vec![-2.0, 1.0],
                vec![1.0, -2.0],
                vec![-1.0, 2.0],
                vec![2.0, -1.0],
                vec![-3.0, 1.0],
                vec![1.0, -3.0],
                vec![-1.0, 3.0],
                vec![3.0, -1.0],
            ],
            vec![0.0; 8],
            0,
            NumericBins::Fixed(64),
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
    fn cart_second_order_tree_supports_oblique_split_strategy() {
        let table = oblique_second_order_table();
        let gradients = [2.0, 2.0, -2.0, -2.0, 3.0, 3.0, -3.0, -3.0];
        let hessians = [1.0; 8];
        let model = train_cart_regressor_from_gradients_and_hessians(
            &table,
            &gradients,
            &hessians,
            Parallelism::sequential(),
            SecondOrderRegressionTreeOptions {
                tree_options: RegressionTreeOptions {
                    max_depth: 1,
                    max_features: Some(2),
                    split_strategy: SplitStrategy::Oblique,
                    ..Default::default()
                },
                l2_regularization: 0.0,
                min_sum_hessian_in_leaf: 0.1,
                min_gain_to_split: 0.0,
            },
        )
        .unwrap();

        match model.structure() {
            RegressionTreeStructure::Standard { nodes, root } => {
                assert!(matches!(nodes[*root], RegressionNode::ObliqueSplit { .. }));
            }
            RegressionTreeStructure::Oblivious { .. } => panic!("expected standard tree"),
        }
        assert_eq!(
            model.predict_table(&table),
            vec![-2.5, -2.5, 2.5, 2.5, -2.5, -2.5, 2.5, 2.5]
        );
    }

    #[test]
    fn randomized_second_order_tree_is_repeatable_for_fixed_seed_and_varies_across_seeds() {
        let table = randomized_permutation_table();
        let gradients = vec![
            3.0, 2.0, 1.5, 0.5, -0.5, -1.0, -2.0, -3.0, 2.5, 1.0, -1.5, -2.5,
        ];
        let hessians = vec![1.0; 12];
        let make_options = |random_seed| SecondOrderRegressionTreeOptions {
            tree_options: RegressionTreeOptions {
                max_depth: 4,
                max_features: Some(2),
                random_seed,
                ..RegressionTreeOptions::default()
            },
            l2_regularization: 1.0,
            min_sum_hessian_in_leaf: 0.1,
            min_gain_to_split: 0.0,
        };

        let base_model = train_randomized_regressor_from_gradients_and_hessians(
            &table,
            &gradients,
            &hessians,
            Parallelism::sequential(),
            make_options(123),
        )
        .unwrap();
        let repeated_model = train_randomized_regressor_from_gradients_and_hessians(
            &table,
            &gradients,
            &hessians,
            Parallelism::sequential(),
            make_options(123),
        )
        .unwrap();
        let unique_prediction_signatures = (0..32u64)
            .map(|seed| {
                format!(
                    "{:?}",
                    train_randomized_regressor_from_gradients_and_hessians(
                        &table,
                        &gradients,
                        &hessians,
                        Parallelism::sequential(),
                        make_options(seed),
                    )
                    .unwrap()
                    .predict_table(&table)
                )
            })
            .collect::<std::collections::BTreeSet<_>>();

        assert_eq!(
            base_model.predict_table(&table),
            repeated_model.predict_table(&table)
        );
        assert!(unique_prediction_signatures.len() >= 4);
    }

    #[test]
    fn cart_second_order_tree_parallel_training_matches_sequential() {
        let table = randomized_permutation_table();
        let gradients = vec![
            3.0, 2.0, 1.5, 0.5, -0.5, -1.0, -2.0, -3.0, 2.5, 1.0, -1.5, -2.5,
        ];
        let hessians = vec![1.0; 12];
        let options = SecondOrderRegressionTreeOptions {
            tree_options: RegressionTreeOptions {
                max_depth: 4,
                max_features: Some(2),
                random_seed: 19,
                ..RegressionTreeOptions::default()
            },
            l2_regularization: 1.0,
            min_sum_hessian_in_leaf: 0.1,
            min_gain_to_split: 0.0,
        };

        let sequential = train_cart_regressor_from_gradients_and_hessians(
            &table,
            &gradients,
            &hessians,
            Parallelism::sequential(),
            options.clone(),
        )
        .unwrap();
        let parallel = train_cart_regressor_from_gradients_and_hessians(
            &table,
            &gradients,
            &hessians,
            Parallelism::with_threads(2),
            options.clone(),
        )
        .unwrap();

        assert_eq!(
            sequential.predict_table(&table),
            parallel.predict_table(&table)
        );
    }

    #[test]
    fn randomized_second_order_tree_parallel_training_matches_sequential() {
        let table = randomized_permutation_table();
        let gradients = vec![
            3.0, 2.0, 1.5, 0.5, -0.5, -1.0, -2.0, -3.0, 2.5, 1.0, -1.5, -2.5,
        ];
        let hessians = vec![1.0; 12];
        let options = SecondOrderRegressionTreeOptions {
            tree_options: RegressionTreeOptions {
                max_depth: 4,
                max_features: Some(2),
                random_seed: 23,
                ..RegressionTreeOptions::default()
            },
            l2_regularization: 1.0,
            min_sum_hessian_in_leaf: 0.1,
            min_gain_to_split: 0.0,
        };

        let sequential = train_randomized_regressor_from_gradients_and_hessians(
            &table,
            &gradients,
            &hessians,
            Parallelism::sequential(),
            options.clone(),
        )
        .unwrap();
        let parallel = train_randomized_regressor_from_gradients_and_hessians(
            &table,
            &gradients,
            &hessians,
            Parallelism::with_threads(2),
            options,
        )
        .unwrap();

        assert_eq!(
            sequential.predict_table(&table),
            parallel.predict_table(&table)
        );
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
