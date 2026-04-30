//! First-order regression tree learners.
//!
//! This module is the regression analogue of `classifier`, but the split logic
//! differs in one important way: regression quality depends on leaf value
//! statistics rather than class counts. The implementation therefore leans on
//! cached count/sum/sum-of-squares histograms in the mean-criterion hot path.

use crate::ir::{
    BinaryChildren, BinarySplit, IndexedLeaf, LeafIndexing, LeafPayload, NodeStats, NodeTreeNode,
    ObliviousLevel, ObliviousSplit as IrObliviousSplit, TrainingMetadata, TreeDefinition,
    criterion_name, feature_name, threshold_upper_bound,
};
use crate::tree::oblique::{
    all_feature_pairs, matched_canary_feature_pairs, missing_mask_for_pair, normalize_weights,
    oblique_feature_value, partition_rows_for_oblique_split, projected_rows_for_pair,
    resolve_oblique_missing_direction,
};
use crate::tree::shared::{
    FeatureHistogram, HistogramBin, MissingBranchDirection, aggregate_beam_non_canary_score,
    build_feature_histograms, build_feature_histograms_parallel, candidate_feature_indices,
    choose_random_threshold, node_seed, partition_rows_for_binary_split,
    select_best_non_canary_candidate, subtract_feature_histograms,
};
use crate::{
    BuilderStrategy, CanaryFilter, Criterion, FeaturePreprocessing, MissingValueStrategy,
    Parallelism, SplitStrategy, capture_feature_preprocessing,
};
use forestfire_data::TableAccess;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionTreeAlgorithm {
    Cart,
    Randomized,
    Oblivious,
}

/// Shared training controls for regression tree learners.
#[derive(Debug, Clone)]
pub struct RegressionTreeOptions {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
    pub random_seed: u64,
    pub missing_value_strategies: Vec<MissingValueStrategy>,
    pub canary_filter: CanaryFilter,
    pub split_strategy: SplitStrategy,
    pub builder: BuilderStrategy,
    pub lookahead_depth: usize,
    pub lookahead_top_k: usize,
    pub lookahead_weight: f64,
    pub beam_width: usize,
}

impl Default for RegressionTreeOptions {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            random_seed: 0,
            missing_value_strategies: Vec::new(),
            canary_filter: CanaryFilter::default(),
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            beam_width: 4,
        }
    }
}

impl RegressionTreeOptions {
    pub(crate) fn effective_lookahead_depth(&self) -> usize {
        match self.builder {
            BuilderStrategy::Greedy => 1,
            BuilderStrategy::Lookahead | BuilderStrategy::Beam => self.lookahead_depth,
            BuilderStrategy::Optimal => self.max_depth,
        }
    }

    pub(crate) fn effective_beam_width(&self) -> usize {
        match self.builder {
            BuilderStrategy::Greedy | BuilderStrategy::Lookahead | BuilderStrategy::Optimal => 1,
            BuilderStrategy::Beam => self.beam_width,
        }
    }

    fn missing_value_strategy(&self, feature_index: usize) -> MissingValueStrategy {
        self.missing_value_strategies
            .get(feature_index)
            .copied()
            .unwrap_or(MissingValueStrategy::Heuristic)
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

/// Concrete trained regression tree.
#[derive(Debug, Clone)]
pub struct DecisionTreeRegressor {
    algorithm: RegressionTreeAlgorithm,
    criterion: Criterion,
    structure: RegressionTreeStructure,
    options: RegressionTreeOptions,
    num_features: usize,
    feature_preprocessing: Vec<FeaturePreprocessing>,
    training_canaries: usize,
}

#[derive(Debug, Clone)]
pub(crate) enum RegressionTreeStructure {
    Standard {
        nodes: Vec<RegressionNode>,
        root: usize,
    },
    Oblivious {
        splits: Vec<ObliviousSplit>,
        leaf_values: Vec<f64>,
        leaf_sample_counts: Vec<usize>,
        leaf_variances: Vec<Option<f64>>,
    },
}

#[derive(Debug, Clone)]
pub(crate) enum RegressionNode {
    Leaf {
        value: f64,
        sample_count: usize,
        variance: Option<f64>,
    },
    MultiTargetLeaf {
        values: Vec<f64>,
        sample_count: usize,
    },
    BinarySplit {
        feature_index: usize,
        threshold_bin: u16,
        missing_direction: MissingBranchDirection,
        missing_values: Vec<f64>,
        left_child: usize,
        right_child: usize,
        sample_count: usize,
        impurity: f64,
        gain: f64,
        variance: Option<f64>,
    },
    ObliqueSplit {
        feature_indices: Vec<usize>,
        weights: Vec<f64>,
        missing_directions: Vec<MissingBranchDirection>,
        threshold: f64,
        missing_values: Vec<f64>,
        left_child: usize,
        right_child: usize,
        sample_count: usize,
        impurity: f64,
        gain: f64,
        variance: Option<f64>,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ObliviousSplit {
    pub(crate) feature_index: usize,
    pub(crate) threshold_bin: u16,
    pub(crate) sample_count: usize,
    pub(crate) impurity: f64,
    pub(crate) gain: f64,
}

#[derive(Debug, Clone)]
struct RegressionSplitCandidate {
    feature_index: usize,
    threshold_bin: u16,
    score: f64,
    missing_direction: MissingBranchDirection,
}

#[derive(Debug, Clone)]
struct ObliviousLeafState {
    start: usize,
    end: usize,
    value: f64,
    variance: Option<f64>,
    sum: f64,
    sum_sq: f64,
    total_weight: f64,
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

#[derive(Debug, Clone, Copy)]
struct RankedObliviousSplitCandidate {
    candidate: ObliviousSplitCandidate,
    ranking_score: f64,
}

#[derive(Debug, Clone, Copy)]
struct BinarySplitChoice {
    feature_index: usize,
    threshold_bin: u16,
    score: f64,
    missing_direction: MissingBranchDirection,
}

#[derive(Debug, Clone)]
struct ObliqueSplitChoice {
    feature_indices: Vec<usize>,
    weights: Vec<f64>,
    missing_directions: Vec<MissingBranchDirection>,
    threshold: f64,
    score: f64,
}

#[derive(Debug, Clone)]
enum StandardSplitChoice {
    Axis(BinarySplitChoice),
    Oblique(ObliqueSplitChoice),
}

impl StandardSplitChoice {
    fn score(&self) -> f64 {
        match self {
            Self::Axis(choice) => choice.score,
            Self::Oblique(choice) => choice.score,
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

#[derive(Debug, Clone)]
struct RegressionHistogramBin {
    count: f64,
    sum: f64,
    sum_sq: f64,
}

impl HistogramBin for RegressionHistogramBin {
    fn subtract(parent: &Self, child: &Self) -> Self {
        Self {
            count: parent.count - child.count,
            sum: parent.sum - child.sum,
            sum_sq: parent.sum_sq - child.sum_sq,
        }
    }

    fn is_observed(&self) -> bool {
        self.count > 0.0
    }
}

type RegressionFeatureHistogram = FeatureHistogram<RegressionHistogramBin>;

pub fn train_cart_regressor(
    train_set: &dyn TableAccess,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_cart_regressor_with_criterion(train_set, Criterion::Mean)
}

pub fn train_cart_regressor_with_criterion(
    train_set: &dyn TableAccess,
    criterion: Criterion,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_cart_regressor_with_criterion_and_parallelism(
        train_set,
        criterion,
        Parallelism::sequential(),
    )
}

pub(crate) fn train_cart_regressor_with_criterion_and_parallelism(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_cart_regressor_with_criterion_parallelism_and_options(
        train_set,
        criterion,
        parallelism,
        RegressionTreeOptions::default(),
    )
}

pub(crate) fn train_cart_regressor_with_criterion_parallelism_and_options(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
    options: RegressionTreeOptions,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_regressor(
        train_set,
        RegressionTreeAlgorithm::Cart,
        criterion,
        parallelism,
        options,
    )
}

pub fn train_oblivious_regressor(
    train_set: &dyn TableAccess,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_oblivious_regressor_with_criterion(train_set, Criterion::Mean)
}

pub fn train_oblivious_regressor_with_criterion(
    train_set: &dyn TableAccess,
    criterion: Criterion,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_oblivious_regressor_with_criterion_and_parallelism(
        train_set,
        criterion,
        Parallelism::sequential(),
    )
}

pub(crate) fn train_oblivious_regressor_with_criterion_and_parallelism(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_oblivious_regressor_with_criterion_parallelism_and_options(
        train_set,
        criterion,
        parallelism,
        RegressionTreeOptions::default(),
    )
}

pub(crate) fn train_oblivious_regressor_with_criterion_parallelism_and_options(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
    options: RegressionTreeOptions,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_regressor(
        train_set,
        RegressionTreeAlgorithm::Oblivious,
        criterion,
        parallelism,
        options,
    )
}

pub fn train_randomized_regressor(
    train_set: &dyn TableAccess,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_randomized_regressor_with_criterion(train_set, Criterion::Mean)
}

pub fn train_randomized_regressor_with_criterion(
    train_set: &dyn TableAccess,
    criterion: Criterion,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_randomized_regressor_with_criterion_and_parallelism(
        train_set,
        criterion,
        Parallelism::sequential(),
    )
}

pub(crate) fn train_randomized_regressor_with_criterion_and_parallelism(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_randomized_regressor_with_criterion_parallelism_and_options(
        train_set,
        criterion,
        parallelism,
        RegressionTreeOptions::default(),
    )
}

pub(crate) fn train_randomized_regressor_with_criterion_parallelism_and_options(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
    options: RegressionTreeOptions,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    train_regressor(
        train_set,
        RegressionTreeAlgorithm::Randomized,
        criterion,
        parallelism,
        options,
    )
}

fn train_regressor(
    train_set: &dyn TableAccess,
    algorithm: RegressionTreeAlgorithm,
    criterion: Criterion,
    parallelism: Parallelism,
    options: RegressionTreeOptions,
) -> Result<DecisionTreeRegressor, RegressionTreeError> {
    if train_set.n_rows() == 0 {
        return Err(RegressionTreeError::EmptyTarget);
    }

    let targets = finite_targets(train_set)?;

    // Multi-target path: only supported for Mean criterion with Cart/Randomized.
    if train_set.n_targets() > 1
        && matches!(criterion, Criterion::Mean)
        && matches!(
            algorithm,
            RegressionTreeAlgorithm::Cart | RegressionTreeAlgorithm::Randomized
        )
    {
        let structure = train_multi_target_regressor_structure(
            train_set, &targets, criterion, &options, algorithm,
        );
        return Ok(DecisionTreeRegressor {
            algorithm,
            criterion,
            structure,
            options,
            num_features: train_set.n_features(),
            feature_preprocessing: capture_feature_preprocessing(train_set),
            training_canaries: train_set.canaries(),
        });
    }

    let structure = match algorithm {
        RegressionTreeAlgorithm::Cart | RegressionTreeAlgorithm::Randomized => {
            let mut nodes = vec![RegressionNode::Leaf {
                value: 0.0,
                sample_count: 0,
                variance: None,
            }];
            let mut all_rows: Vec<usize> = (0..train_set.n_rows()).collect();
            let context = BuildContext {
                table: train_set,
                targets: &targets,
                criterion,
                parallelism,
                options: options.clone(),
                algorithm,
            };
            let root = train_binary_structure(&context, &mut nodes, &mut all_rows);
            RegressionTreeStructure::Standard { nodes, root }
        }
        RegressionTreeAlgorithm::Oblivious => {
            // Oblivious trees are built level by level because every node at a
            // given depth must share the same split.
            train_oblivious_structure(train_set, &targets, criterion, parallelism, options.clone())
        }
    };

    Ok(DecisionTreeRegressor {
        algorithm,
        criterion,
        structure,
        options,
        num_features: train_set.n_features(),
        feature_preprocessing: capture_feature_preprocessing(train_set),
        training_canaries: train_set.canaries(),
    })
}

impl DecisionTreeRegressor {
    /// Which learner family produced this tree.
    pub fn algorithm(&self) -> RegressionTreeAlgorithm {
        self.algorithm
    }

    /// Split criterion used during training.
    pub fn criterion(&self) -> Criterion {
        self.criterion
    }

    /// Predict one numeric value per row from a preprocessed table.
    pub fn predict_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        (0..table.n_rows())
            .map(|row_idx| self.predict_row(table, row_idx))
            .collect()
    }

    fn predict_row(&self, table: &dyn TableAccess, row_idx: usize) -> f64 {
        match &self.structure {
            RegressionTreeStructure::Standard { nodes, root } => {
                let mut node_index = *root;

                loop {
                    match &nodes[node_index] {
                        RegressionNode::Leaf { value, .. } => return *value,
                        RegressionNode::MultiTargetLeaf { values, .. } => return values[0],
                        RegressionNode::BinarySplit {
                            feature_index,
                            threshold_bin,
                            missing_direction,
                            missing_values,
                            left_child,
                            right_child,
                            ..
                        } => {
                            if table.is_missing(*feature_index, row_idx) {
                                match missing_direction {
                                    MissingBranchDirection::Left => {
                                        node_index = *left_child;
                                    }
                                    MissingBranchDirection::Right => {
                                        node_index = *right_child;
                                    }
                                    MissingBranchDirection::Node => return missing_values[0],
                                }
                                continue;
                            }
                            let bin = table.binned_value(*feature_index, row_idx);
                            node_index = if bin <= *threshold_bin {
                                *left_child
                            } else {
                                *right_child
                            };
                        }
                        RegressionNode::ObliqueSplit {
                            feature_indices,
                            weights,
                            missing_directions,
                            threshold,
                            missing_values,
                            left_child,
                            right_child,
                            ..
                        } => {
                            let missing_mask = missing_mask_for_pair(
                                table,
                                [feature_indices[0], feature_indices[1]],
                                row_idx,
                            );
                            if let Some(go_left) = resolve_oblique_missing_direction(
                                missing_mask,
                                [weights[0], weights[1]],
                                [missing_directions[0], missing_directions[1]],
                            ) {
                                node_index = if go_left { *left_child } else { *right_child };
                                continue;
                            }
                            if missing_mask != 0 {
                                return missing_values[0];
                            }
                            let projection = weights
                                .iter()
                                .zip(feature_indices.iter())
                                .map(|(weight, feature_index)| {
                                    *weight * table.feature_value(*feature_index, row_idx)
                                })
                                .sum::<f64>();
                            node_index = if projection <= *threshold {
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
                ..
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

    /// Predict all target values for a single row (multi-target support).
    pub fn predict_all_row(&self, table: &dyn TableAccess, row_idx: usize) -> Vec<f64> {
        match &self.structure {
            RegressionTreeStructure::Standard { nodes, root } => {
                let mut node_index = *root;
                loop {
                    match &nodes[node_index] {
                        RegressionNode::Leaf { value, .. } => return vec![*value],
                        RegressionNode::MultiTargetLeaf { values, .. } => {
                            return values.clone();
                        }
                        RegressionNode::BinarySplit {
                            feature_index,
                            threshold_bin,
                            missing_direction,
                            missing_values,
                            left_child,
                            right_child,
                            ..
                        } => {
                            if table.is_missing(*feature_index, row_idx) {
                                match missing_direction {
                                    MissingBranchDirection::Left => {
                                        node_index = *left_child;
                                    }
                                    MissingBranchDirection::Right => {
                                        node_index = *right_child;
                                    }
                                    MissingBranchDirection::Node => return missing_values.clone(),
                                }
                                continue;
                            }
                            let bin = table.binned_value(*feature_index, row_idx);
                            node_index = if bin <= *threshold_bin {
                                *left_child
                            } else {
                                *right_child
                            };
                        }
                        RegressionNode::ObliqueSplit {
                            feature_indices,
                            weights,
                            missing_directions,
                            threshold,
                            missing_values,
                            left_child,
                            right_child,
                            ..
                        } => {
                            let missing_mask = missing_mask_for_pair(
                                table,
                                [feature_indices[0], feature_indices[1]],
                                row_idx,
                            );
                            if let Some(go_left) = resolve_oblique_missing_direction(
                                missing_mask,
                                [weights[0], weights[1]],
                                [missing_directions[0], missing_directions[1]],
                            ) {
                                node_index = if go_left { *left_child } else { *right_child };
                                continue;
                            }
                            if missing_mask != 0 {
                                return missing_values.clone();
                            }
                            let projection = weights
                                .iter()
                                .zip(feature_indices.iter())
                                .map(|(weight, feature_index)| {
                                    *weight * table.feature_value(*feature_index, row_idx)
                                })
                                .sum::<f64>();
                            node_index = if projection <= *threshold {
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
                ..
            } => {
                let leaf_index = splits.iter().fold(0usize, |leaf_index, split| {
                    let go_right =
                        table.binned_value(split.feature_index, row_idx) > split.threshold_bin;
                    (leaf_index << 1) | usize::from(go_right)
                });
                vec![leaf_values[leaf_index]]
            }
        }
    }

    /// Returns true if any node in the tree is a MultiTargetLeaf.
    pub fn is_multi_target(&self) -> bool {
        match &self.structure {
            RegressionTreeStructure::Standard { nodes, .. } => nodes
                .iter()
                .any(|n| matches!(n, RegressionNode::MultiTargetLeaf { .. })),
            RegressionTreeStructure::Oblivious { .. } => false,
        }
    }

    pub fn predict_all_table(&self, table: &dyn TableAccess) -> Vec<Vec<f64>> {
        (0..table.n_rows())
            .map(|row_idx| self.predict_all_row(table, row_idx))
            .collect()
    }

    pub(crate) fn num_features(&self) -> usize {
        self.num_features
    }

    pub(crate) fn structure(&self) -> &RegressionTreeStructure {
        &self.structure
    }

    pub(crate) fn feature_preprocessing(&self) -> &[FeaturePreprocessing] {
        &self.feature_preprocessing
    }

    pub(crate) fn training_metadata(&self) -> TrainingMetadata {
        TrainingMetadata {
            algorithm: "dt".to_string(),
            task: "regression".to_string(),
            tree_type: match self.algorithm {
                RegressionTreeAlgorithm::Cart => "cart".to_string(),
                RegressionTreeAlgorithm::Randomized => "randomized".to_string(),
                RegressionTreeAlgorithm::Oblivious => "oblivious".to_string(),
            },
            criterion: criterion_name(self.criterion).to_string(),
            canaries: self.training_canaries,
            compute_oob: false,
            max_depth: Some(self.options.max_depth),
            min_samples_split: Some(self.options.min_samples_split),
            min_samples_leaf: Some(self.options.min_samples_leaf),
            n_trees: None,
            max_features: self.options.max_features,
            seed: None,
            oob_score: None,
            class_labels: None,
            learning_rate: None,
            bootstrap: None,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
        }
    }

    pub(crate) fn to_ir_tree(&self) -> TreeDefinition {
        match &self.structure {
            RegressionTreeStructure::Standard { nodes, root } => {
                let depths = standard_node_depths(nodes, *root);
                TreeDefinition::NodeTree {
                    tree_id: 0,
                    weight: 1.0,
                    root_node_id: *root,
                    nodes: nodes
                        .iter()
                        .enumerate()
                        .map(|(node_id, node)| match node {
                            RegressionNode::Leaf {
                                value,
                                sample_count,
                                variance,
                            } => NodeTreeNode::Leaf {
                                node_id,
                                depth: depths[node_id],
                                leaf: LeafPayload::RegressionValue { value: *value },
                                stats: NodeStats {
                                    sample_count: *sample_count,
                                    impurity: None,
                                    gain: None,
                                    class_counts: None,
                                    variance: *variance,
                                },
                            },
                            RegressionNode::MultiTargetLeaf {
                                values,
                                sample_count,
                            } => NodeTreeNode::Leaf {
                                node_id,
                                depth: depths[node_id],
                                leaf: LeafPayload::MultiTargetRegressionValue {
                                    values: values.clone(),
                                },
                                stats: NodeStats {
                                    sample_count: *sample_count,
                                    impurity: None,
                                    gain: None,
                                    class_counts: None,
                                    variance: None,
                                },
                            },
                            RegressionNode::BinarySplit {
                                feature_index,
                                threshold_bin,
                                missing_direction,
                                missing_values: _,
                                left_child,
                                right_child,
                                sample_count,
                                impurity,
                                gain,
                                variance,
                            } => NodeTreeNode::BinaryBranch {
                                node_id,
                                depth: depths[node_id],
                                split: binary_split_ir(
                                    *feature_index,
                                    *threshold_bin,
                                    *missing_direction,
                                    &self.feature_preprocessing,
                                ),
                                children: BinaryChildren {
                                    left: *left_child,
                                    right: *right_child,
                                },
                                stats: NodeStats {
                                    sample_count: *sample_count,
                                    impurity: Some(*impurity),
                                    gain: Some(*gain),
                                    class_counts: None,
                                    variance: *variance,
                                },
                            },
                            RegressionNode::ObliqueSplit {
                                feature_indices,
                                weights,
                                missing_directions,
                                threshold,
                                left_child,
                                right_child,
                                sample_count,
                                impurity,
                                gain,
                                variance,
                                ..
                            } => NodeTreeNode::BinaryBranch {
                                node_id,
                                depth: depths[node_id],
                                split: BinarySplit::ObliqueLinearCombination {
                                    feature_indices: feature_indices.clone(),
                                    feature_names: feature_indices
                                        .iter()
                                        .map(|feature_index| feature_name(*feature_index))
                                        .collect(),
                                    weights: weights.clone(),
                                    missing_directions: missing_directions
                                        .iter()
                                        .map(|direction| match direction {
                                            MissingBranchDirection::Left => "left".to_string(),
                                            MissingBranchDirection::Right => "right".to_string(),
                                            MissingBranchDirection::Node => "node".to_string(),
                                        })
                                        .collect(),
                                    operator: "<=".to_string(),
                                    threshold: *threshold,
                                },
                                children: BinaryChildren {
                                    left: *left_child,
                                    right: *right_child,
                                },
                                stats: NodeStats {
                                    sample_count: *sample_count,
                                    impurity: Some(*impurity),
                                    gain: Some(*gain),
                                    class_counts: None,
                                    variance: *variance,
                                },
                            },
                        })
                        .collect(),
                }
            }
            RegressionTreeStructure::Oblivious {
                splits,
                leaf_values,
                leaf_sample_counts,
                leaf_variances,
            } => TreeDefinition::ObliviousLevels {
                tree_id: 0,
                weight: 1.0,
                depth: splits.len(),
                levels: splits
                    .iter()
                    .enumerate()
                    .map(|(level, split)| ObliviousLevel {
                        level,
                        split: oblivious_split_ir(
                            split.feature_index,
                            split.threshold_bin,
                            &self.feature_preprocessing,
                        ),
                        stats: NodeStats {
                            sample_count: split.sample_count,
                            impurity: Some(split.impurity),
                            gain: Some(split.gain),
                            class_counts: None,
                            variance: None,
                        },
                    })
                    .collect(),
                leaf_indexing: LeafIndexing {
                    bit_order: "msb_first".to_string(),
                    index_formula: "sum(bit[level] << (depth - 1 - level))".to_string(),
                },
                leaves: leaf_values
                    .iter()
                    .enumerate()
                    .map(|(leaf_index, value)| IndexedLeaf {
                        leaf_index,
                        leaf: LeafPayload::RegressionValue { value: *value },
                        stats: NodeStats {
                            sample_count: leaf_sample_counts[leaf_index],
                            impurity: None,
                            gain: None,
                            class_counts: None,
                            variance: leaf_variances[leaf_index],
                        },
                    })
                    .collect(),
            },
        }
    }

    pub(crate) fn from_ir_parts(
        algorithm: RegressionTreeAlgorithm,
        criterion: Criterion,
        structure: RegressionTreeStructure,
        options: RegressionTreeOptions,
        num_features: usize,
        feature_preprocessing: Vec<FeaturePreprocessing>,
        training_canaries: usize,
    ) -> Self {
        Self {
            algorithm,
            criterion,
            structure,
            options: options.clone(),
            num_features,
            feature_preprocessing,
            training_canaries,
        }
    }
}

fn standard_node_depths(nodes: &[RegressionNode], root: usize) -> Vec<usize> {
    let mut depths = vec![0; nodes.len()];
    populate_depths(nodes, root, 0, &mut depths);
    depths
}

fn populate_depths(nodes: &[RegressionNode], node_id: usize, depth: usize, depths: &mut [usize]) {
    depths[node_id] = depth;
    match &nodes[node_id] {
        RegressionNode::Leaf { .. } | RegressionNode::MultiTargetLeaf { .. } => {}
        RegressionNode::BinarySplit {
            left_child,
            right_child,
            ..
        }
        | RegressionNode::ObliqueSplit {
            left_child,
            right_child,
            ..
        } => {
            populate_depths(nodes, *left_child, depth + 1, depths);
            populate_depths(nodes, *right_child, depth + 1, depths);
        }
    }
}

fn binary_split_ir(
    feature_index: usize,
    threshold_bin: u16,
    _missing_direction: MissingBranchDirection,
    preprocessing: &[FeaturePreprocessing],
) -> BinarySplit {
    match preprocessing.get(feature_index) {
        Some(FeaturePreprocessing::Binary) => BinarySplit::BooleanTest {
            feature_index,
            feature_name: feature_name(feature_index),
            false_child_semantics: "left".to_string(),
            true_child_semantics: "right".to_string(),
        },
        Some(FeaturePreprocessing::Numeric { .. }) | None => BinarySplit::NumericBinThreshold {
            feature_index,
            feature_name: feature_name(feature_index),
            operator: "<=".to_string(),
            threshold_bin,
            threshold_upper_bound: threshold_upper_bound(
                preprocessing,
                feature_index,
                threshold_bin,
            ),
            comparison_dtype: "uint16".to_string(),
        },
    }
}

fn oblivious_split_ir(
    feature_index: usize,
    threshold_bin: u16,
    preprocessing: &[FeaturePreprocessing],
) -> IrObliviousSplit {
    match preprocessing.get(feature_index) {
        Some(FeaturePreprocessing::Binary) => IrObliviousSplit::BooleanTest {
            feature_index,
            feature_name: feature_name(feature_index),
            bit_when_false: 0,
            bit_when_true: 1,
        },
        Some(FeaturePreprocessing::Numeric { .. }) | None => {
            IrObliviousSplit::NumericBinThreshold {
                feature_index,
                feature_name: feature_name(feature_index),
                operator: "<=".to_string(),
                threshold_bin,
                threshold_upper_bound: threshold_upper_bound(
                    preprocessing,
                    feature_index,
                    threshold_bin,
                ),
                comparison_dtype: "uint16".to_string(),
                bit_when_true: 0,
                bit_when_false: 1,
            }
        }
    }
}

struct BuildContext<'a> {
    table: &'a dyn TableAccess,
    targets: &'a [f64],
    criterion: Criterion,
    parallelism: Parallelism,
    options: RegressionTreeOptions,
    algorithm: RegressionTreeAlgorithm,
}

fn build_regression_node_histograms(
    table: &dyn TableAccess,
    targets: &[f64],
    rows: &[usize],
) -> Vec<RegressionFeatureHistogram> {
    build_feature_histograms(
        table,
        rows,
        |_| RegressionHistogramBin {
            count: 0.0,
            sum: 0.0,
            sum_sq: 0.0,
        },
        |_feature_index, payload, row_idx| {
            let w = table.sample_weight(row_idx);
            let value = targets[row_idx];
            payload.count += w;
            payload.sum += w * value;
            payload.sum_sq += w * value * value;
        },
    )
}

fn subtract_regression_node_histograms(
    parent: &[RegressionFeatureHistogram],
    child: &[RegressionFeatureHistogram],
) -> Vec<RegressionFeatureHistogram> {
    subtract_feature_histograms(parent, child)
}

fn finite_targets(train_set: &dyn TableAccess) -> Result<Vec<f64>, RegressionTreeError> {
    (0..train_set.n_rows())
        .map(|row_idx| {
            let value = train_set.target_value(row_idx);
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

// ── Frontier-based binary structure builder ─────────────────────────────────
//
// Replaces the recursive `build_binary_node_in_place` with a level-wise loop
// so nodes at the same depth can be evaluated and have their histograms built
// in parallel, using the same two-level (node-parallel × feature-parallel)
// rayon work-stealing strategy as the second-order tree.

struct ActiveRegressionNode {
    node_index: usize,
    depth: usize,
    start: usize,
    end: usize,
    histograms: Option<Vec<RegressionFeatureHistogram>>,
}

struct RegressionNodeEval {
    leaf_value: f64,
    leaf_variance: Option<f64>,
    impurity: f64,
    histograms: Option<Vec<RegressionFeatureHistogram>>,
    best_split: Option<StandardSplitChoice>,
}

struct RegressionSplitWork {
    depth: usize,
    start: usize,
    mid: usize,
    end: usize,
    left_child: usize,
    right_child: usize,
    // None when Criterion::Variance (no histograms); subtraction is skipped and
    // children build fresh histograms on demand.
    histograms: Option<Vec<RegressionFeatureHistogram>>,
}

fn build_regression_node_histograms_par(
    context: &BuildContext<'_>,
    rows: &[usize],
    parallelism: Parallelism,
) -> Vec<RegressionFeatureHistogram> {
    let make_bin = |_| RegressionHistogramBin {
        count: 0.0,
        sum: 0.0,
        sum_sq: 0.0,
    };
    let add_row = |_feature_index, payload: &mut RegressionHistogramBin, row_idx| {
        let w = context.table.sample_weight(row_idx);
        let value = context.targets[row_idx];
        payload.count += w;
        payload.sum += w * value;
        payload.sum_sq += w * value * value;
    };
    if parallelism.enabled() {
        build_feature_histograms_parallel(context.table, rows, make_bin, add_row)
    } else {
        build_feature_histograms(context.table, rows, make_bin, add_row)
    }
}

fn evaluate_regression_node(
    context: &BuildContext<'_>,
    rows: &[usize],
    depth: usize,
    histograms: Option<Vec<RegressionFeatureHistogram>>,
    scoring_par: Parallelism,
) -> RegressionNodeEval {
    let leaf_value = regression_value(context.table, rows, context.targets, context.criterion);
    let leaf_variance = variance(context.table, rows, context.targets);
    let impurity = regression_loss(context.table, rows, context.targets, context.criterion);

    if rows.is_empty()
        || depth >= context.options.max_depth
        || rows.len() < context.options.min_samples_split
        || has_constant_target(rows, context.targets)
    {
        return RegressionNodeEval {
            leaf_value,
            leaf_variance,
            impurity,
            histograms: None,
            best_split: None,
        };
    }

    let histograms =
        if matches!(context.criterion, Criterion::Mean) {
            Some(histograms.unwrap_or_else(|| {
                build_regression_node_histograms_par(context, rows, scoring_par)
            }))
        } else {
            None
        };

    let feature_indices = candidate_feature_indices(
        context.table,
        context.options.max_features,
        node_seed(context.options.random_seed, depth, rows, 0xA11C_E5E1u64),
    );
    let split_candidates = if scoring_par.enabled() {
        feature_indices
            .par_iter()
            .filter_map(|feature_index| {
                if let Some(histograms) = histograms.as_ref() {
                    score_binary_split_choice_from_hist(
                        context,
                        &histograms[*feature_index],
                        *feature_index,
                        rows,
                    )
                } else {
                    score_binary_split_choice(context, *feature_index, rows)
                }
            })
            .collect::<Vec<_>>()
    } else {
        feature_indices
            .iter()
            .filter_map(|feature_index| {
                if let Some(histograms) = histograms.as_ref() {
                    score_binary_split_choice_from_hist(
                        context,
                        &histograms[*feature_index],
                        *feature_index,
                        rows,
                    )
                } else {
                    score_binary_split_choice(context, *feature_index, rows)
                }
            })
            .collect::<Vec<_>>()
    };
    let ranked = rank_standard_split_choices(
        context,
        rows,
        depth,
        &split_candidates,
        &feature_indices,
        context.options.effective_lookahead_depth(),
    );
    let best_split = select_best_non_canary_candidate(
        context.table,
        ranked,
        context.options.canary_filter,
        |c| c.ranking_score,
        |c| c.choice.ranking_feature_index(),
    )
    .selected
    .map(|c| c.choice);

    RegressionNodeEval {
        leaf_value,
        leaf_variance,
        impurity,
        histograms,
        best_split,
    }
}

fn regression_child_active_nodes(
    work: RegressionSplitWork,
    context: &BuildContext<'_>,
    rows: &[usize],
    hist_par: Parallelism,
) -> [ActiveRegressionNode; 2] {
    let left_len = work.mid - work.start;
    let right_len = work.end - work.mid;
    let (left_histograms, right_histograms) = match work.histograms {
        Some(parent_histograms) => {
            if left_len <= right_len {
                let left = build_regression_node_histograms_par(
                    context,
                    &rows[work.start..work.mid],
                    hist_par,
                );
                let right = subtract_regression_node_histograms(&parent_histograms, &left);
                (Some(left), Some(right))
            } else {
                let right = build_regression_node_histograms_par(
                    context,
                    &rows[work.mid..work.end],
                    hist_par,
                );
                let left = subtract_regression_node_histograms(&parent_histograms, &right);
                (Some(left), Some(right))
            }
        }
        None => (None, None),
    };
    [
        ActiveRegressionNode {
            node_index: work.left_child,
            depth: work.depth + 1,
            start: work.start,
            end: work.mid,
            histograms: left_histograms,
        },
        ActiveRegressionNode {
            node_index: work.right_child,
            depth: work.depth + 1,
            start: work.mid,
            end: work.end,
            histograms: right_histograms,
        },
    ]
}

fn train_binary_structure(
    context: &BuildContext<'_>,
    nodes: &mut Vec<RegressionNode>,
    rows: &mut [usize],
) -> usize {
    let root = 0usize;
    let mut frontier = vec![ActiveRegressionNode {
        node_index: root,
        depth: 0,
        start: 0,
        end: rows.len(),
        histograms: None,
    }];

    while !frontier.is_empty() {
        let use_par = context.parallelism.enabled() && frontier.len() >= 2;
        let evals: Vec<(usize, usize, usize, usize, RegressionNodeEval)> = if use_par {
            frontier
                .into_par_iter()
                .map(|active| {
                    let eval = evaluate_regression_node(
                        context,
                        &rows[active.start..active.end],
                        active.depth,
                        active.histograms,
                        context.parallelism,
                    );
                    (
                        active.node_index,
                        active.depth,
                        active.start,
                        active.end,
                        eval,
                    )
                })
                .collect()
        } else {
            frontier
                .into_iter()
                .map(|active| {
                    let eval = evaluate_regression_node(
                        context,
                        &rows[active.start..active.end],
                        active.depth,
                        active.histograms,
                        context.parallelism,
                    );
                    (
                        active.node_index,
                        active.depth,
                        active.start,
                        active.end,
                        eval,
                    )
                })
                .collect()
        };

        let mut split_work: Vec<RegressionSplitWork> = Vec::new();

        for (node_index, depth, start, end, eval) in evals {
            match eval.best_split {
                Some(split) if split.score() > 0.0 => {
                    let left_count = match &split {
                        StandardSplitChoice::Axis(s) => partition_rows_for_binary_split(
                            context.table,
                            s.feature_index,
                            s.threshold_bin,
                            s.missing_direction,
                            &mut rows[start..end],
                        ),
                        StandardSplitChoice::Oblique(s) => partition_rows_for_oblique_split(
                            context.table,
                            [s.feature_indices[0], s.feature_indices[1]],
                            [s.weights[0], s.weights[1]],
                            s.threshold,
                            [s.missing_directions[0], s.missing_directions[1]],
                            &mut rows[start..end],
                        ),
                    };
                    let mid = start + left_count;
                    let left_child =
                        push_leaf(nodes, eval.leaf_value, mid - start, eval.leaf_variance);
                    let right_child =
                        push_leaf(nodes, eval.leaf_value, end - mid, eval.leaf_variance);
                    nodes[node_index] = match split {
                        StandardSplitChoice::Axis(s) => RegressionNode::BinarySplit {
                            feature_index: s.feature_index,
                            threshold_bin: s.threshold_bin,
                            missing_direction: s.missing_direction,
                            missing_values: vec![eval.leaf_value],
                            left_child,
                            right_child,
                            sample_count: end - start,
                            impurity: eval.impurity,
                            gain: s.score,
                            variance: eval.leaf_variance,
                        },
                        StandardSplitChoice::Oblique(s) => RegressionNode::ObliqueSplit {
                            feature_indices: s.feature_indices,
                            weights: s.weights,
                            missing_directions: s.missing_directions,
                            threshold: s.threshold,
                            missing_values: vec![eval.leaf_value],
                            left_child,
                            right_child,
                            sample_count: end - start,
                            impurity: eval.impurity,
                            gain: s.score,
                            variance: eval.leaf_variance,
                        },
                    };
                    split_work.push(RegressionSplitWork {
                        depth,
                        start,
                        mid,
                        end,
                        left_child,
                        right_child,
                        histograms: eval.histograms,
                    });
                }
                _ => {
                    nodes[node_index] = RegressionNode::Leaf {
                        value: eval.leaf_value,
                        sample_count: end - start,
                        variance: eval.leaf_variance,
                    };
                }
            }
        }

        let use_child_par = context.parallelism.enabled() && split_work.len() >= 2;
        frontier = if use_child_par {
            split_work
                .into_par_iter()
                .flat_map_iter(|work| {
                    regression_child_active_nodes(work, context, rows, context.parallelism)
                })
                .collect()
        } else {
            split_work
                .into_iter()
                .flat_map(|work| {
                    regression_child_active_nodes(work, context, rows, context.parallelism)
                })
                .collect()
        };
    }

    root
}

fn score_oblique_split_choices(
    context: &BuildContext<'_>,
    rows: &[usize],
    axis_candidates: &[BinarySplitChoice],
    candidate_features: &[usize],
) -> Vec<StandardSplitChoice> {
    if !matches!(context.options.split_strategy, SplitStrategy::Oblique)
        || rows.len() < context.options.min_samples_leaf * 2
    {
        return axis_candidates
            .iter()
            .cloned()
            .map(StandardSplitChoice::Axis)
            .collect();
    }

    let real_features = candidate_features
        .iter()
        .copied()
        .filter(|feature_index| *feature_index < context.table.n_features())
        .collect::<Vec<_>>();
    if real_features.len() < 2 {
        return axis_candidates
            .iter()
            .cloned()
            .map(StandardSplitChoice::Axis)
            .collect();
    }

    let mut ranked = axis_candidates
        .iter()
        .cloned()
        .map(StandardSplitChoice::Axis)
        .collect::<Vec<_>>();
    let real_pairs = all_feature_pairs(&real_features);
    let mut candidates = collect_oblique_regression_candidates(context, rows, &real_pairs);
    let canary_pairs = matched_canary_feature_pairs(context.table, &real_features);
    candidates.extend(collect_oblique_regression_candidates(
        context,
        rows,
        &canary_pairs,
    ));
    ranked.extend(candidates.into_iter().map(StandardSplitChoice::Oblique));
    ranked
}

fn rank_standard_split_choices(
    context: &BuildContext<'_>,
    rows: &[usize],
    depth: usize,
    axis_candidates: &[BinarySplitChoice],
    candidate_features: &[usize],
    _lookahead_depth: usize,
) -> Vec<RankedStandardSplitChoice> {
    let candidates = if matches!(context.options.split_strategy, SplitStrategy::Oblique) {
        score_oblique_split_choices(context, rows, axis_candidates, candidate_features)
    } else {
        axis_candidates
            .iter()
            .cloned()
            .map(StandardSplitChoice::Axis)
            .collect()
    };
    let (search_depth, top_k, future_weight) =
        if matches!(context.options.builder, BuilderStrategy::Optimal) {
            (None, candidates.len(), 1.0)
        } else {
            (
                Some(context.options.effective_lookahead_depth()),
                context.options.lookahead_top_k,
                context.options.lookahead_weight,
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
        .map(|(index, choice)| (index, choice.score()))
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
                choice.score()
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
    let immediate = choice.score();
    if immediate <= 0.0
        || depth + 1 >= context.options.max_depth
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
            split.missing_direction,
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
        context.options.effective_beam_width(),
        future_weight,
    ) + best_standard_split_recursive_score(
        context,
        right_rows,
        depth + 1,
        search_depth.map(|remaining| remaining - 1),
        context.options.effective_beam_width(),
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
        || depth >= context.options.max_depth
        || rows.len() < context.options.min_samples_split
        || has_constant_target(rows, context.targets)
    {
        return 0.0;
    }

    let histograms = if matches!(context.criterion, Criterion::Mean) {
        Some(build_regression_node_histograms(
            context.table,
            context.targets,
            rows,
        ))
    } else {
        None
    };
    let feature_indices = candidate_feature_indices(
        context.table,
        context.options.max_features,
        node_seed(context.options.random_seed, depth, rows, 0xA11C_E5E1u64),
    );
    let split_candidates = feature_indices
        .iter()
        .filter_map(|feature_index| {
            if let Some(histograms) = histograms.as_ref() {
                score_binary_split_choice_from_hist(
                    context,
                    &histograms[*feature_index],
                    *feature_index,
                    rows,
                )
            } else {
                score_binary_split_choice(context, *feature_index, rows)
            }
        })
        .collect::<Vec<_>>();
    let candidates = if matches!(context.options.split_strategy, SplitStrategy::Oblique) {
        score_oblique_split_choices(context, rows, &split_candidates, &feature_indices)
    } else {
        split_candidates
            .into_iter()
            .map(StandardSplitChoice::Axis)
            .collect()
    };
    let top_k = if search_depth.is_none() {
        candidates.len()
    } else {
        context.options.lookahead_top_k
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
        context.options.canary_filter,
        beam_width,
        |candidate| candidate.ranking_score,
        |candidate| candidate.choice.ranking_feature_index(),
    )
}

#[allow(dead_code)]
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

#[allow(dead_code)]
fn optimal_standard_split_ranking_score(
    context: &BuildContext<'_>,
    rows: &[usize],
    depth: usize,
    choice: &StandardSplitChoice,
) -> f64 {
    let immediate = choice.score();
    if immediate <= 0.0 || depth + 1 >= context.options.max_depth {
        return immediate;
    }

    let mut partitioned_rows = rows.to_vec();
    let left_count = match choice {
        StandardSplitChoice::Axis(split) => partition_rows_for_binary_split(
            context.table,
            split.feature_index,
            split.threshold_bin,
            split.missing_direction,
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

#[allow(dead_code)]
fn best_standard_split_optimal_score(
    context: &BuildContext<'_>,
    rows: &mut [usize],
    depth: usize,
) -> f64 {
    if rows.is_empty()
        || depth >= context.options.max_depth
        || rows.len() < context.options.min_samples_split
        || has_constant_target(rows, context.targets)
    {
        return 0.0;
    }

    let histograms = if matches!(context.criterion, Criterion::Mean) {
        Some(build_regression_node_histograms(
            context.table,
            context.targets,
            rows,
        ))
    } else {
        None
    };
    let feature_indices = candidate_feature_indices(
        context.table,
        context.options.max_features,
        node_seed(context.options.random_seed, depth, rows, 0xA11C_E5E1u64),
    );
    let split_candidates = feature_indices
        .iter()
        .filter_map(|feature_index| {
            if let Some(histograms) = histograms.as_ref() {
                score_binary_split_choice_from_hist(
                    context,
                    &histograms[*feature_index],
                    *feature_index,
                    rows,
                )
            } else {
                score_binary_split_choice(context, *feature_index, rows)
            }
        })
        .collect::<Vec<_>>();
    let candidates = if matches!(context.options.split_strategy, SplitStrategy::Oblique) {
        score_oblique_split_choices(context, rows, &split_candidates, &feature_indices)
    } else {
        split_candidates
            .into_iter()
            .map(StandardSplitChoice::Axis)
            .collect()
    };
    select_best_non_canary_candidate(
        context.table,
        rank_optimal_standard_split_choices(context, rows, depth, candidates),
        context.options.canary_filter,
        |candidate| candidate.ranking_score,
        |candidate| candidate.choice.ranking_feature_index(),
    )
    .selected
    .map(|candidate| candidate.ranking_score.max(0.0))
    .unwrap_or(0.0)
}

#[allow(dead_code)]
fn best_standard_split_lookahead_score(
    context: &BuildContext<'_>,
    rows: &mut [usize],
    depth: usize,
    lookahead_depth: usize,
    beam_width: usize,
) -> f64 {
    if rows.is_empty()
        || lookahead_depth == 0
        || depth >= context.options.max_depth
        || rows.len() < context.options.min_samples_split
        || has_constant_target(rows, context.targets)
    {
        return 0.0;
    }

    let histograms = if matches!(context.criterion, Criterion::Mean) {
        Some(build_regression_node_histograms(
            context.table,
            context.targets,
            rows,
        ))
    } else {
        None
    };
    let feature_indices = candidate_feature_indices(
        context.table,
        context.options.max_features,
        node_seed(context.options.random_seed, depth, rows, 0xA11C_E5E1u64),
    );
    let split_candidates = feature_indices
        .iter()
        .filter_map(|feature_index| {
            if let Some(histograms) = histograms.as_ref() {
                score_binary_split_choice_from_hist(
                    context,
                    &histograms[*feature_index],
                    *feature_index,
                    rows,
                )
            } else {
                score_binary_split_choice(context, *feature_index, rows)
            }
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
        context.options.canary_filter,
        beam_width,
        |candidate| candidate.ranking_score,
        |candidate| match &candidate.choice {
            StandardSplitChoice::Axis(split) => split.feature_index,
            StandardSplitChoice::Oblique(split) => split.feature_indices[0],
        },
    )
}

#[allow(dead_code)]
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
        .map(|(index, choice)| {
            let ranking_score = if shortlisted.contains(&index) {
                rescore(&choice)
            } else {
                immediate_score(&choice)
            };
            RankedStandardSplitChoice {
                choice,
                ranking_score,
            }
        })
        .collect()
}

fn collect_oblique_regression_candidates(
    context: &BuildContext<'_>,
    rows: &[usize],
    feature_pairs: &[[usize; 2]],
) -> Vec<ObliqueSplitChoice> {
    let parent_loss = regression_loss(context.table, rows, context.targets, context.criterion);
    let mut candidates = Vec::new();
    for &feature_pair in feature_pairs {
        let observed_rows = rows
            .iter()
            .copied()
            .filter(|row_index| missing_mask_for_pair(context.table, feature_pair, *row_index) == 0)
            .collect::<Vec<_>>();
        if observed_rows.len() < context.options.min_samples_leaf * 2 {
            continue;
        }
        let Some(weights) = oblique_regression_weights(
            context.table,
            &observed_rows,
            context.targets,
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

        for split_index in 0..projected.len().saturating_sub(1) {
            if projected[split_index].value == projected[split_index + 1].value {
                continue;
            }
            let threshold = (projected[split_index].value + projected[split_index + 1].value) / 2.0;
            for &direction_0 in &direction_choices_0 {
                for &direction_1 in &direction_choices_1 {
                    let missing_directions = [direction_0, direction_1];
                    let mut left_rows = projected[..split_index + 1]
                        .iter()
                        .map(|projected_row| projected_row.row_index)
                        .collect::<Vec<_>>();
                    let mut right_rows = projected[split_index + 1..]
                        .iter()
                        .map(|projected_row| projected_row.row_index)
                        .collect::<Vec<_>>();
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
                        if go_left {
                            left_rows.extend(target_rows.iter().copied());
                        } else {
                            right_rows.extend(target_rows.iter().copied());
                        }
                    }
                    if left_rows.len() < context.options.min_samples_leaf
                        || right_rows.len() < context.options.min_samples_leaf
                    {
                        continue;
                    }
                    let score = match context.criterion {
                        Criterion::Mean | Criterion::Median => {
                            parent_loss
                                - (regression_loss(
                                    context.table,
                                    &left_rows,
                                    context.targets,
                                    context.criterion,
                                ) + regression_loss(
                                    context.table,
                                    &right_rows,
                                    context.targets,
                                    context.criterion,
                                ))
                        }
                        _ => unreachable!("regression criterion only supports mean or median"),
                    };
                    if !score.is_finite() {
                        continue;
                    }
                    candidates.push(ObliqueSplitChoice {
                        feature_indices: vec![feature_pair[0], feature_pair[1]],
                        weights: vec![weights[0], weights[1]],
                        missing_directions: vec![direction_0, direction_1],
                        threshold,
                        score,
                    });
                }
            }
        }
    }
    candidates
}

fn oblique_regression_weights(
    table: &dyn TableAccess,
    rows: &[usize],
    targets: &[f64],
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
    let mean_target = rows
        .iter()
        .map(|row_index| targets[*row_index])
        .sum::<f64>()
        / rows.len() as f64;
    let mean_x0 = feature_values.iter().map(|values| values[0]).sum::<f64>() / rows.len() as f64;
    let mean_x1 = feature_values.iter().map(|values| values[1]).sum::<f64>() / rows.len() as f64;
    let mut weights = [0.0; 2];
    for (&row_index, values) in rows.iter().zip(feature_values.iter()) {
        let centered_target = targets[row_index] - mean_target;
        weights[0] += (values[0] - mean_x0) * centered_target;
        weights[1] += (values[1] - mean_x1) * centered_target;
    }
    normalize_weights(weights)
}

fn multi_target_leaf_values(
    table: &dyn TableAccess,
    rows: &[usize],
    targets_per_col: &[Vec<f64>],
) -> Vec<f64> {
    targets_per_col
        .iter()
        .map(|targets| {
            let (weighted_sum, _, total_w) = sum_stats(table, rows, targets);
            if total_w == 0.0 {
                0.0
            } else {
                weighted_sum / total_w
            }
        })
        .collect()
}

fn train_multi_target_regressor_structure(
    table: &dyn TableAccess,
    targets_col0: &[f64],
    criterion: Criterion,
    options: &RegressionTreeOptions,
    algorithm: RegressionTreeAlgorithm,
) -> RegressionTreeStructure {
    let n_targets = table.n_targets();
    // Build per-target column vectors.
    let targets_per_col: Vec<Vec<f64>> = (0..n_targets)
        .map(|t| {
            (0..table.n_rows())
                .map(|row| table.target_value_at(row, t))
                .collect()
        })
        .collect();

    let parallelism = Parallelism::sequential();
    // Use the first target column for the BuildContext (split scoring sums gains across targets).
    let context = BuildContext {
        table,
        targets: targets_col0,
        criterion,
        parallelism,
        options: options.clone(),
        algorithm,
    };

    let mut nodes = Vec::new();
    let mut all_rows: Vec<usize> = (0..table.n_rows()).collect();
    let root =
        build_multi_target_node_in_place(&context, &targets_per_col, &mut nodes, &mut all_rows, 0);
    RegressionTreeStructure::Standard { nodes, root }
}

fn build_multi_target_node_in_place(
    context: &BuildContext<'_>,
    targets_per_col: &[Vec<f64>],
    nodes: &mut Vec<RegressionNode>,
    rows: &mut [usize],
    depth: usize,
) -> usize {
    let leaf_values = multi_target_leaf_values(context.table, rows, targets_per_col);
    let sample_count = rows.len();

    if rows.is_empty()
        || depth >= context.options.max_depth
        || rows.len() < context.options.min_samples_split
        || targets_per_col
            .iter()
            .all(|targets| has_constant_target(rows, targets))
    {
        return push_node(
            nodes,
            RegressionNode::MultiTargetLeaf {
                values: leaf_values,
                sample_count,
            },
        );
    }

    // Build histograms for each target column.
    let all_hists: Vec<Vec<RegressionFeatureHistogram>> = targets_per_col
        .iter()
        .map(|targets| build_regression_node_histograms(context.table, targets, rows))
        .collect();

    let feature_indices = candidate_feature_indices(
        context.table,
        context.options.max_features,
        node_seed(context.options.random_seed, depth, rows, 0xA11C_E5E1u64),
    );

    // Score each feature by evaluating the combined gain across all targets at
    // each candidate threshold, so the score and applied threshold are always
    // for the same split point.
    let best_split = feature_indices
        .iter()
        .filter_map(|&feature_index| {
            let first_hist = &all_hists[0][feature_index];
            match first_hist {
                RegressionFeatureHistogram::Binary {
                    false_bin,
                    true_bin,
                    missing_bin,
                } if missing_bin.count == 0.0 => {
                    // Binary features have a single threshold, so summing each
                    // target's gain at that threshold is already consistent.
                    let mut total_score = 0.0;
                    let mut any_valid = false;
                    for hists in &all_hists {
                        if let RegressionFeatureHistogram::Binary {
                            false_bin,
                            true_bin,
                            missing_bin,
                        } = &hists[feature_index]
                        {
                            if missing_bin.count != 0.0 {
                                return None;
                            }
                            if let Some(c) = score_binary_split_choice_mean_from_stats(
                                context,
                                feature_index,
                                false_bin.count,
                                false_bin.sum,
                                false_bin.sum_sq,
                                true_bin.count,
                                true_bin.sum,
                                true_bin.sum_sq,
                            ) {
                                total_score += c.score;
                                any_valid = true;
                            }
                        }
                    }
                    if any_valid && total_score > 0.0 {
                        Some(BinarySplitChoice {
                            feature_index,
                            threshold_bin: 0,
                            score: total_score,
                            missing_direction: MissingBranchDirection::Node,
                        })
                    } else {
                        None
                    }
                }
                RegressionFeatureHistogram::Numeric {
                    bins,
                    observed_bins,
                } if bins
                    .get(context.table.numeric_bin_cap())
                    .is_none_or(|mb| mb.count == 0.0) =>
                {
                    // Collect the bins slice for every target, then find the
                    // threshold that maximises the combined gain in one pass.
                    let target_bins: Vec<&[RegressionHistogramBin]> = all_hists
                        .iter()
                        .filter_map(|hists| {
                            if let RegressionFeatureHistogram::Numeric { bins, .. } =
                                &hists[feature_index]
                            {
                                Some(bins.as_slice())
                            } else {
                                None
                            }
                        })
                        .collect();
                    if target_bins.len() != all_hists.len() {
                        return None;
                    }
                    score_multi_target_numeric_split_combined(
                        context,
                        feature_index,
                        &target_bins,
                        observed_bins,
                    )
                }
                _ => None,
            }
        })
        .filter(|c| c.score > 0.0)
        .max_by(|a, b| a.score.total_cmp(&b.score));

    match best_split {
        Some(split) => {
            let left_count = partition_rows_for_binary_split(
                context.table,
                split.feature_index,
                split.threshold_bin,
                split.missing_direction,
                rows,
            );
            let (left_rows, right_rows) = rows.split_at_mut(left_count);
            let left_child = build_multi_target_node_in_place(
                context,
                targets_per_col,
                nodes,
                left_rows,
                depth + 1,
            );
            let right_child = build_multi_target_node_in_place(
                context,
                targets_per_col,
                nodes,
                right_rows,
                depth + 1,
            );
            // Compute impurity using first target column.
            let impurity = regression_loss(context.table, rows, context.targets, context.criterion);
            push_node(
                nodes,
                RegressionNode::BinarySplit {
                    feature_index: split.feature_index,
                    threshold_bin: split.threshold_bin,
                    missing_direction: split.missing_direction,
                    missing_values: leaf_values.clone(),
                    left_child,
                    right_child,
                    sample_count,
                    impurity,
                    gain: split.score,
                    variance: None,
                },
            )
        }
        None => push_node(
            nodes,
            RegressionNode::MultiTargetLeaf {
                values: leaf_values,
                sample_count,
            },
        ),
    }
}

fn train_oblivious_structure(
    table: &dyn TableAccess,
    targets: &[f64],
    criterion: Criterion,
    parallelism: Parallelism,
    options: RegressionTreeOptions,
) -> RegressionTreeStructure {
    let mut row_indices: Vec<usize> = (0..table.n_rows()).collect();
    let (root_sum, root_sum_sq, root_total_weight) = sum_stats(table, &row_indices, targets);
    let mut leaves = vec![ObliviousLeafState {
        start: 0,
        end: row_indices.len(),
        value: regression_value_from_stats(
            &row_indices,
            targets,
            criterion,
            root_sum,
            root_total_weight,
        ),
        variance: variance_from_stats(root_total_weight, root_sum, root_sum_sq),
        sum: root_sum,
        sum_sq: root_sum_sq,
        total_weight: root_total_weight,
    }];
    let mut splits = Vec::new();

    let max_depth = options.max_depth;
    for depth in 0..max_depth {
        if leaves
            .iter()
            .all(|leaf| leaf.len() < options.min_samples_split)
        {
            break;
        }
        let feature_indices = candidate_feature_indices(
            table,
            options.max_features,
            node_seed(options.random_seed, depth, &[], 0x0B11_A10Cu64),
        );
        let split_candidates = if parallelism.enabled() {
            feature_indices
                .into_par_iter()
                .filter_map(|feature_index| {
                    score_oblivious_split(
                        table,
                        &row_indices,
                        targets,
                        feature_index,
                        &leaves,
                        criterion,
                        options.min_samples_leaf,
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
                        targets,
                        feature_index,
                        &leaves,
                        criterion,
                        options.min_samples_leaf,
                    )
                })
                .collect::<Vec<_>>()
        };

        let (search_depth, top_k, future_weight) =
            if matches!(options.builder, BuilderStrategy::Optimal) {
                (None, split_candidates.len(), 1.0)
            } else {
                (
                    Some(options.effective_lookahead_depth()),
                    options.lookahead_top_k,
                    options.lookahead_weight,
                )
            };
        let ranked_candidates = rank_oblivious_split_choices_with_limits(
            table,
            &row_indices,
            targets,
            &leaves,
            criterion,
            &options,
            depth,
            split_candidates,
            search_depth,
            top_k,
            future_weight,
        );
        let Some(best_split) = select_best_non_canary_candidate(
            table,
            ranked_candidates,
            options.canary_filter,
            |candidate| candidate.ranking_score,
            |candidate| candidate.candidate.feature_index,
        )
        .selected
        .map(|candidate| candidate.candidate)
        .filter(|candidate| candidate.score > 0.0) else {
            break;
        };

        leaves = split_oblivious_leaves_in_place(
            table,
            &mut row_indices,
            targets,
            leaves,
            best_split.feature_index,
            best_split.threshold_bin,
            criterion,
        );
        splits.push(ObliviousSplit {
            feature_index: best_split.feature_index,
            threshold_bin: best_split.threshold_bin,
            sample_count: table.n_rows(),
            impurity: leaves
                .iter()
                .map(|leaf| leaf_regression_loss(leaf, &row_indices, targets, criterion))
                .sum(),
            gain: best_split.score,
        });
    }

    RegressionTreeStructure::Oblivious {
        splits,
        leaf_values: leaves.iter().map(|leaf| leaf.value).collect(),
        leaf_sample_counts: leaves.iter().map(ObliviousLeafState::len).collect(),
        leaf_variances: leaves.iter().map(|leaf| leaf.variance).collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn score_split(
    table: &dyn TableAccess,
    targets: &[f64],
    feature_index: usize,
    rows: &[usize],
    criterion: Criterion,
    min_samples_leaf: usize,
    algorithm: RegressionTreeAlgorithm,
    strategy: MissingValueStrategy,
) -> Option<RegressionSplitCandidate> {
    if table.is_binary_binned_feature(feature_index) {
        return score_binary_split(
            table,
            targets,
            feature_index,
            rows,
            criterion,
            min_samples_leaf,
            strategy,
        );
    }
    let has_missing = feature_has_missing(table, feature_index, rows);
    if matches!(criterion, Criterion::Mean) && !has_missing {
        if matches!(algorithm, RegressionTreeAlgorithm::Randomized) {
            if let Some(candidate) = score_randomized_split_mean_fast(
                table,
                targets,
                feature_index,
                rows,
                min_samples_leaf,
            ) {
                return Some(candidate);
            }
        } else if let Some(candidate) =
            score_numeric_split_mean_fast(table, targets, feature_index, rows, min_samples_leaf)
        {
            return Some(candidate);
        }
    }
    if matches!(algorithm, RegressionTreeAlgorithm::Randomized) {
        return score_randomized_split(
            table,
            targets,
            feature_index,
            rows,
            criterion,
            min_samples_leaf,
            strategy,
        );
    }
    if has_missing && matches!(strategy, MissingValueStrategy::Heuristic) {
        return score_split_heuristic_missing_assignment(
            table,
            targets,
            feature_index,
            rows,
            criterion,
            min_samples_leaf,
        );
    }
    let parent_loss = regression_loss(table, rows, targets, criterion);

    rows.iter()
        .copied()
        .filter(|row_idx| !table.is_missing(feature_index, *row_idx))
        .map(|row_idx| table.binned_value(feature_index, row_idx))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|threshold_bin| {
            evaluate_regression_missing_assignment(
                table,
                targets,
                feature_index,
                rows,
                criterion,
                min_samples_leaf,
                threshold_bin,
                parent_loss,
            )
        })
        .max_by(|left, right| left.score.total_cmp(&right.score))
}

fn score_randomized_split(
    table: &dyn TableAccess,
    targets: &[f64],
    feature_index: usize,
    rows: &[usize],
    criterion: Criterion,
    min_samples_leaf: usize,
    _strategy: MissingValueStrategy,
) -> Option<RegressionSplitCandidate> {
    let candidate_thresholds = rows
        .iter()
        .copied()
        .filter(|row_idx| !table.is_missing(feature_index, *row_idx))
        .map(|row_idx| table.binned_value(feature_index, row_idx))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let threshold_bin =
        choose_random_threshold(&candidate_thresholds, feature_index, rows, 0xA11CE551u64)?;

    let parent_loss = regression_loss(table, rows, targets, criterion);
    evaluate_regression_missing_assignment(
        table,
        targets,
        feature_index,
        rows,
        criterion,
        min_samples_leaf,
        threshold_bin,
        parent_loss,
    )
}

fn score_oblivious_split(
    table: &dyn TableAccess,
    row_indices: &[usize],
    targets: &[f64],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
    criterion: Criterion,
    min_samples_leaf: usize,
) -> Option<ObliviousSplitCandidate> {
    if table.is_binary_binned_feature(feature_index) {
        if matches!(criterion, Criterion::Mean)
            && let Some(candidate) = score_binary_oblivious_split_mean_fast(
                table,
                row_indices,
                targets,
                feature_index,
                leaves,
                min_samples_leaf,
            )
        {
            return Some(candidate);
        }
        return score_binary_oblivious_split(
            table,
            row_indices,
            targets,
            feature_index,
            leaves,
            criterion,
            min_samples_leaf,
        );
    }
    if matches!(criterion, Criterion::Mean)
        && let Some(candidate) = score_numeric_oblivious_split_mean_fast(
            table,
            row_indices,
            targets,
            feature_index,
            leaves,
            min_samples_leaf,
        )
    {
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

                score + regression_loss(table, leaf_rows, targets, criterion)
                    - (regression_loss(table, &left_rows, targets, criterion)
                        + regression_loss(table, &right_rows, targets, criterion))
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
    targets: &[f64],
    leaves: Vec<ObliviousLeafState>,
    feature_index: usize,
    threshold_bin: u16,
    criterion: Criterion,
) -> Vec<ObliviousLeafState> {
    let mut next_leaves = Vec::with_capacity(leaves.len() * 2);
    for leaf in leaves {
        let fallback_value = leaf.value;
        let left_count = partition_rows_for_binary_split(
            table,
            feature_index,
            threshold_bin,
            MissingBranchDirection::Right,
            &mut row_indices[leaf.start..leaf.end],
        );
        let mid = leaf.start + left_count;
        let left_rows = &row_indices[leaf.start..mid];
        let right_rows = &row_indices[mid..leaf.end];
        let (left_sum, left_sum_sq, left_total_weight) = sum_stats(table, left_rows, targets);
        let (right_sum, right_sum_sq, right_total_weight) = sum_stats(table, right_rows, targets);
        next_leaves.push(ObliviousLeafState {
            start: leaf.start,
            end: mid,
            value: if left_rows.is_empty() {
                fallback_value
            } else {
                regression_value_from_stats(
                    left_rows,
                    targets,
                    criterion,
                    left_sum,
                    left_total_weight,
                )
            },
            variance: variance_from_stats(left_total_weight, left_sum, left_sum_sq),
            sum: left_sum,
            sum_sq: left_sum_sq,
            total_weight: left_total_weight,
        });
        next_leaves.push(ObliviousLeafState {
            start: mid,
            end: leaf.end,
            value: if right_rows.is_empty() {
                fallback_value
            } else {
                regression_value_from_stats(
                    right_rows,
                    targets,
                    criterion,
                    right_sum,
                    right_total_weight,
                )
            },
            variance: variance_from_stats(right_total_weight, right_sum, right_sum_sq),
            sum: right_sum,
            sum_sq: right_sum_sq,
            total_weight: right_total_weight,
        });
    }
    next_leaves
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn oblivious_split_ranking_score(
    table: &dyn TableAccess,
    row_indices: &[usize],
    targets: &[f64],
    leaves: &[ObliviousLeafState],
    criterion: Criterion,
    options: &RegressionTreeOptions,
    depth: usize,
    candidate: &ObliviousSplitCandidate,
    lookahead_depth: usize,
) -> f64 {
    oblivious_split_recursive_ranking_score(
        table,
        row_indices,
        targets,
        leaves,
        criterion,
        options,
        depth,
        candidate,
        Some(lookahead_depth),
        options.lookahead_weight,
    )
}

#[allow(clippy::too_many_arguments)]
fn rank_oblivious_split_choices_with_limits(
    table: &dyn TableAccess,
    row_indices: &[usize],
    targets: &[f64],
    leaves: &[ObliviousLeafState],
    criterion: Criterion,
    options: &RegressionTreeOptions,
    depth: usize,
    candidates: Vec<ObliviousSplitCandidate>,
    search_depth: Option<usize>,
    top_k: usize,
    future_weight: f64,
) -> Vec<RankedObliviousSplitCandidate> {
    let mut shortlist = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (index, candidate.score))
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
        .map(|(index, candidate)| RankedObliviousSplitCandidate {
            ranking_score: if shortlisted.contains(&index) {
                oblivious_split_recursive_ranking_score(
                    table,
                    row_indices,
                    targets,
                    leaves,
                    criterion,
                    options,
                    depth,
                    &candidate,
                    search_depth,
                    future_weight,
                )
            } else {
                candidate.score
            },
            candidate,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn oblivious_split_recursive_ranking_score(
    table: &dyn TableAccess,
    row_indices: &[usize],
    targets: &[f64],
    leaves: &[ObliviousLeafState],
    criterion: Criterion,
    options: &RegressionTreeOptions,
    depth: usize,
    candidate: &ObliviousSplitCandidate,
    search_depth: Option<usize>,
    future_weight: f64,
) -> f64 {
    let immediate = candidate.score;
    if immediate <= 0.0
        || depth + 1 >= options.max_depth
        || search_depth.is_some_and(|remaining| remaining <= 1)
    {
        return immediate;
    }

    let mut next_row_indices = row_indices.to_vec();
    let next_leaves = split_oblivious_leaves_in_place(
        table,
        &mut next_row_indices,
        targets,
        leaves.to_vec(),
        candidate.feature_index,
        candidate.threshold_bin,
        criterion,
    );
    let future = best_oblivious_split_recursive_score(
        table,
        &mut next_row_indices,
        targets,
        next_leaves,
        criterion,
        options,
        depth + 1,
        search_depth.map(|remaining| remaining - 1),
        options.effective_beam_width(),
        future_weight,
    );
    immediate + future_weight * future
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn rank_optimal_oblivious_split_choices(
    table: &dyn TableAccess,
    row_indices: &[usize],
    targets: &[f64],
    leaves: &[ObliviousLeafState],
    criterion: Criterion,
    options: &RegressionTreeOptions,
    depth: usize,
    candidates: Vec<ObliviousSplitCandidate>,
) -> Vec<RankedObliviousSplitCandidate> {
    candidates
        .into_iter()
        .map(|candidate| RankedObliviousSplitCandidate {
            ranking_score: optimal_oblivious_split_ranking_score(
                table,
                row_indices,
                targets,
                leaves,
                criterion,
                options,
                depth,
                &candidate,
            ),
            candidate,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn optimal_oblivious_split_ranking_score(
    table: &dyn TableAccess,
    row_indices: &[usize],
    targets: &[f64],
    leaves: &[ObliviousLeafState],
    criterion: Criterion,
    options: &RegressionTreeOptions,
    depth: usize,
    candidate: &ObliviousSplitCandidate,
) -> f64 {
    let immediate = candidate.score;
    if immediate <= 0.0 || depth + 1 >= options.max_depth {
        return immediate;
    }

    let mut next_row_indices = row_indices.to_vec();
    let next_leaves = split_oblivious_leaves_in_place(
        table,
        &mut next_row_indices,
        targets,
        leaves.to_vec(),
        candidate.feature_index,
        candidate.threshold_bin,
        criterion,
    );
    immediate
        + best_oblivious_split_optimal_score(
            table,
            &mut next_row_indices,
            targets,
            next_leaves,
            criterion,
            options,
            depth + 1,
        )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn best_oblivious_split_lookahead_score(
    table: &dyn TableAccess,
    row_indices: &mut [usize],
    targets: &[f64],
    leaves: Vec<ObliviousLeafState>,
    criterion: Criterion,
    options: &RegressionTreeOptions,
    depth: usize,
    lookahead_depth: usize,
    beam_width: usize,
) -> f64 {
    best_oblivious_split_recursive_score(
        table,
        row_indices,
        targets,
        leaves,
        criterion,
        options,
        depth,
        Some(lookahead_depth),
        beam_width,
        options.lookahead_weight,
    )
}

#[allow(clippy::too_many_arguments)]
fn best_oblivious_split_recursive_score(
    table: &dyn TableAccess,
    row_indices: &mut [usize],
    targets: &[f64],
    leaves: Vec<ObliviousLeafState>,
    criterion: Criterion,
    options: &RegressionTreeOptions,
    depth: usize,
    search_depth: Option<usize>,
    beam_width: usize,
    future_weight: f64,
) -> f64 {
    if leaves
        .iter()
        .all(|leaf| leaf.len() < options.min_samples_split)
        || search_depth == Some(0)
        || depth >= options.max_depth
    {
        return 0.0;
    }

    let feature_indices = candidate_feature_indices(
        table,
        options.max_features,
        node_seed(options.random_seed, depth, &[], 0x0B11_A10Cu64),
    );
    let split_candidates = feature_indices
        .into_iter()
        .filter_map(|feature_index| {
            score_oblivious_split(
                table,
                row_indices,
                targets,
                feature_index,
                &leaves,
                criterion,
                options.min_samples_leaf,
            )
        })
        .collect::<Vec<_>>();
    let top_k = if search_depth.is_none() {
        split_candidates.len()
    } else {
        options.lookahead_top_k
    };
    let ranked = rank_oblivious_split_choices_with_limits(
        table,
        row_indices,
        targets,
        &leaves,
        criterion,
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
        options.canary_filter,
        beam_width,
        |candidate| candidate.ranking_score,
        |candidate| candidate.candidate.feature_index,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn best_oblivious_split_optimal_score(
    table: &dyn TableAccess,
    row_indices: &mut [usize],
    targets: &[f64],
    leaves: Vec<ObliviousLeafState>,
    criterion: Criterion,
    options: &RegressionTreeOptions,
    depth: usize,
) -> f64 {
    if leaves
        .iter()
        .all(|leaf| leaf.len() < options.min_samples_split)
        || depth >= options.max_depth
    {
        return 0.0;
    }

    let feature_indices = candidate_feature_indices(
        table,
        options.max_features,
        node_seed(options.random_seed, depth, &[], 0x0B11_A10Cu64),
    );
    let split_candidates = feature_indices
        .into_iter()
        .filter_map(|feature_index| {
            score_oblivious_split(
                table,
                row_indices,
                targets,
                feature_index,
                &leaves,
                criterion,
                options.min_samples_leaf,
            )
        })
        .collect::<Vec<_>>();
    select_best_non_canary_candidate(
        table,
        rank_optimal_oblivious_split_choices(
            table,
            row_indices,
            targets,
            &leaves,
            criterion,
            options,
            depth,
            split_candidates,
        ),
        options.canary_filter,
        |candidate| candidate.ranking_score,
        |candidate| candidate.candidate.feature_index,
    )
    .selected
    .map(|candidate| candidate.ranking_score.max(0.0))
    .unwrap_or(0.0)
}

#[allow(dead_code)]
fn rank_shortlisted_oblivious_candidates(
    candidates: Vec<ObliviousSplitCandidate>,
    top_k: usize,
    rescore: impl Fn(&ObliviousSplitCandidate) -> f64,
) -> Vec<RankedObliviousSplitCandidate> {
    let mut shortlist = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (index, candidate.score))
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
        .map(|(index, candidate)| RankedObliviousSplitCandidate {
            ranking_score: if shortlisted.contains(&index) {
                rescore(&candidate)
            } else {
                candidate.score
            },
            candidate,
        })
        .collect()
}

fn variance(table: &dyn TableAccess, rows: &[usize], targets: &[f64]) -> Option<f64> {
    let (sum, sum_sq, total_w) = sum_stats(table, rows, targets);
    variance_from_stats(total_w, sum, sum_sq)
}

fn median(rows: &[usize], targets: &[f64]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    let mut values: Vec<f64> = rows.iter().map(|row_idx| targets[*row_idx]).collect();
    values.sort_by(|left, right| left.total_cmp(right));

    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn sum_absolute_error(rows: &[usize], targets: &[f64]) -> f64 {
    let median = median(rows, targets);
    rows.iter()
        .map(|row_idx| (targets[*row_idx] - median).abs())
        .sum()
}

fn regression_value(
    table: &dyn TableAccess,
    rows: &[usize],
    targets: &[f64],
    criterion: Criterion,
) -> f64 {
    let (sum, _sum_sq, total_w) = sum_stats(table, rows, targets);
    regression_value_from_stats(rows, targets, criterion, sum, total_w)
}

fn regression_value_from_stats(
    rows: &[usize],
    targets: &[f64],
    criterion: Criterion,
    weighted_sum: f64,
    total_weight: f64,
) -> f64 {
    match criterion {
        Criterion::Mean => {
            if total_weight == 0.0 {
                0.0
            } else {
                weighted_sum / total_weight
            }
        }
        Criterion::Median => median(rows, targets),
        _ => unreachable!("regression criterion only supports mean or median"),
    }
}

fn regression_loss(
    table: &dyn TableAccess,
    rows: &[usize],
    targets: &[f64],
    criterion: Criterion,
) -> f64 {
    match criterion {
        Criterion::Mean => {
            let (sum, sum_sq, total_w) = sum_stats(table, rows, targets);
            if total_w == 0.0 {
                0.0
            } else {
                sum_sq - (sum * sum) / total_w
            }
        }
        Criterion::Median => sum_absolute_error(rows, targets),
        _ => unreachable!("regression criterion only supports mean or median"),
    }
}

fn score_binary_split(
    table: &dyn TableAccess,
    targets: &[f64],
    feature_index: usize,
    rows: &[usize],
    criterion: Criterion,
    min_samples_leaf: usize,
    strategy: MissingValueStrategy,
) -> Option<RegressionSplitCandidate> {
    if matches!(strategy, MissingValueStrategy::Heuristic) {
        return score_binary_split_heuristic(
            table,
            targets,
            feature_index,
            rows,
            criterion,
            min_samples_leaf,
        );
    }
    let parent_loss = regression_loss(table, rows, targets, criterion);
    evaluate_regression_missing_assignment(
        table,
        targets,
        feature_index,
        rows,
        criterion,
        min_samples_leaf,
        0,
        parent_loss,
    )
}

fn score_binary_split_heuristic(
    table: &dyn TableAccess,
    targets: &[f64],
    feature_index: usize,
    rows: &[usize],
    criterion: Criterion,
    min_samples_leaf: usize,
) -> Option<RegressionSplitCandidate> {
    let observed_rows = rows
        .iter()
        .copied()
        .filter(|row_idx| !table.is_missing(feature_index, *row_idx))
        .collect::<Vec<_>>();
    if observed_rows.is_empty() {
        return None;
    }
    let parent_loss = regression_loss(table, &observed_rows, targets, criterion);
    let mut left_rows = Vec::new();
    let mut right_rows = Vec::new();
    for row_idx in observed_rows.iter().copied() {
        if !table
            .binned_boolean_value(feature_index, row_idx)
            .expect("observed binary feature must expose boolean values")
        {
            left_rows.push(row_idx);
        } else {
            right_rows.push(row_idx);
        }
    }
    if left_rows.len() < min_samples_leaf || right_rows.len() < min_samples_leaf {
        return None;
    }
    evaluate_regression_missing_assignment(
        table,
        targets,
        feature_index,
        rows,
        criterion,
        min_samples_leaf,
        0,
        parent_loss,
    )
}

fn score_split_heuristic_missing_assignment(
    table: &dyn TableAccess,
    targets: &[f64],
    feature_index: usize,
    rows: &[usize],
    criterion: Criterion,
    min_samples_leaf: usize,
) -> Option<RegressionSplitCandidate> {
    let observed_rows = rows
        .iter()
        .copied()
        .filter(|row_idx| !table.is_missing(feature_index, *row_idx))
        .collect::<Vec<_>>();
    if observed_rows.is_empty() {
        return None;
    }
    let parent_loss = regression_loss(table, &observed_rows, targets, criterion);
    let threshold_bin = observed_rows
        .iter()
        .copied()
        .map(|row_idx| table.binned_value(feature_index, row_idx))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|threshold_bin| {
            evaluate_regression_observed_split(
                table,
                targets,
                feature_index,
                &observed_rows,
                criterion,
                min_samples_leaf,
                threshold_bin,
                parent_loss,
            )
            .map(|score| (threshold_bin, score))
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(threshold_bin, _)| threshold_bin)?;
    evaluate_regression_missing_assignment(
        table,
        targets,
        feature_index,
        rows,
        criterion,
        min_samples_leaf,
        threshold_bin,
        parent_loss,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_regression_observed_split(
    table: &dyn TableAccess,
    targets: &[f64],
    feature_index: usize,
    observed_rows: &[usize],
    criterion: Criterion,
    min_samples_leaf: usize,
    threshold_bin: u16,
    parent_loss: f64,
) -> Option<f64> {
    let mut left_rows = Vec::new();
    let mut right_rows = Vec::new();
    for row_idx in observed_rows.iter().copied() {
        if table.binned_value(feature_index, row_idx) <= threshold_bin {
            left_rows.push(row_idx);
        } else {
            right_rows.push(row_idx);
        }
    }
    if left_rows.len() < min_samples_leaf || right_rows.len() < min_samples_leaf {
        return None;
    }
    Some(
        parent_loss
            - (regression_loss(table, &left_rows, targets, criterion)
                + regression_loss(table, &right_rows, targets, criterion)),
    )
}

fn score_binary_split_choice(
    context: &BuildContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<BinarySplitChoice> {
    if matches!(context.criterion, Criterion::Mean) {
        if context.table.is_binary_binned_feature(feature_index) {
            if feature_has_missing(context.table, feature_index, rows) {
                return score_split(
                    context.table,
                    context.targets,
                    feature_index,
                    rows,
                    context.criterion,
                    context.options.min_samples_leaf,
                    context.algorithm,
                    context.options.missing_value_strategy(feature_index),
                )
                .map(|candidate| BinarySplitChoice {
                    feature_index: candidate.feature_index,
                    threshold_bin: candidate.threshold_bin,
                    score: candidate.score,
                    missing_direction: candidate.missing_direction,
                });
            }
            return score_binary_split_choice_mean(context, feature_index, rows);
        }
        if feature_has_missing(context.table, feature_index, rows) {
            return score_split(
                context.table,
                context.targets,
                feature_index,
                rows,
                context.criterion,
                context.options.min_samples_leaf,
                context.algorithm,
                context.options.missing_value_strategy(feature_index),
            )
            .map(|candidate| BinarySplitChoice {
                feature_index: candidate.feature_index,
                threshold_bin: candidate.threshold_bin,
                score: candidate.score,
                missing_direction: candidate.missing_direction,
            });
        }
        return match context.algorithm {
            RegressionTreeAlgorithm::Cart => {
                score_numeric_split_choice_mean_fast(context, feature_index, rows)
            }
            RegressionTreeAlgorithm::Randomized => {
                score_randomized_split_choice_mean_fast(context, feature_index, rows)
            }
            RegressionTreeAlgorithm::Oblivious => None,
        };
    }

    score_split(
        context.table,
        context.targets,
        feature_index,
        rows,
        context.criterion,
        context.options.min_samples_leaf,
        context.algorithm,
        context.options.missing_value_strategy(feature_index),
    )
    .map(|candidate| BinarySplitChoice {
        feature_index: candidate.feature_index,
        threshold_bin: candidate.threshold_bin,
        score: candidate.score,
        missing_direction: candidate.missing_direction,
    })
}

fn score_binary_split_choice_from_hist(
    context: &BuildContext<'_>,
    histogram: &RegressionFeatureHistogram,
    feature_index: usize,
    rows: &[usize],
) -> Option<BinarySplitChoice> {
    if !matches!(context.criterion, Criterion::Mean) {
        return score_binary_split_choice(context, feature_index, rows);
    }

    match histogram {
        RegressionFeatureHistogram::Binary {
            false_bin,
            true_bin,
            missing_bin,
        } if missing_bin.count == 0.0 => score_binary_split_choice_mean_from_stats(
            context,
            feature_index,
            false_bin.count,
            false_bin.sum,
            false_bin.sum_sq,
            true_bin.count,
            true_bin.sum,
            true_bin.sum_sq,
        ),
        RegressionFeatureHistogram::Binary { .. } => {
            score_binary_split_choice(context, feature_index, rows)
        }
        RegressionFeatureHistogram::Numeric {
            bins,
            observed_bins,
        } if bins
            .get(context.table.numeric_bin_cap())
            .is_none_or(|missing_bin| missing_bin.count == 0.0) =>
        {
            match context.algorithm {
                RegressionTreeAlgorithm::Cart => score_numeric_split_choice_mean_from_hist(
                    context,
                    feature_index,
                    bins.iter().map(|b| b.count).sum::<f64>(),
                    bins,
                    observed_bins,
                ),
                RegressionTreeAlgorithm::Randomized => {
                    score_randomized_split_choice_mean_from_hist(
                        context,
                        feature_index,
                        rows,
                        bins,
                        observed_bins,
                    )
                }
                RegressionTreeAlgorithm::Oblivious => None,
            }
        }
        RegressionFeatureHistogram::Numeric { .. } => {
            score_binary_split_choice(context, feature_index, rows)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn score_binary_split_choice_mean_from_stats(
    context: &BuildContext<'_>,
    feature_index: usize,
    left_count: f64,
    left_sum: f64,
    left_sum_sq: f64,
    right_count: f64,
    right_sum: f64,
    right_sum_sq: f64,
) -> Option<BinarySplitChoice> {
    if left_count < context.options.min_samples_leaf as f64
        || right_count < context.options.min_samples_leaf as f64
    {
        return None;
    }
    let total_count = left_count + right_count;
    let total_sum = left_sum + right_sum;
    let total_sum_sq = left_sum_sq + right_sum_sq;
    let parent_loss = total_sum_sq - (total_sum * total_sum) / total_count;
    let left_loss = left_sum_sq - (left_sum * left_sum) / left_count;
    let right_loss = right_sum_sq - (right_sum * right_sum) / right_count;
    Some(BinarySplitChoice {
        feature_index,
        threshold_bin: 0,
        score: parent_loss - (left_loss + right_loss),
        missing_direction: MissingBranchDirection::Node,
    })
}

/// Score a numeric feature by evaluating each candidate threshold against ALL
/// target histograms simultaneously, so the chosen threshold and combined gain
/// are always consistent — the split that is applied is the same one that was
/// scored.
fn score_multi_target_numeric_split_combined(
    context: &BuildContext<'_>,
    feature_index: usize,
    target_bins: &[&[RegressionHistogramBin]],
    observed_bins: &[usize],
) -> Option<BinarySplitChoice> {
    if observed_bins.len() <= 1 {
        return None;
    }
    // All targets share the same row weights, so counts are identical across
    // targets. We use target 0's counts for the min_samples_leaf check.
    let totals: Vec<(f64, f64, f64)> = target_bins
        .iter()
        .map(|bins| {
            let w = bins.iter().map(|b| b.count).sum::<f64>();
            let s = bins.iter().map(|b| b.sum).sum::<f64>();
            let sq = bins.iter().map(|b| b.sum_sq).sum::<f64>();
            (w, s, sq)
        })
        .collect();

    let mut left_counts = vec![0.0_f64; target_bins.len()];
    let mut left_sums = vec![0.0_f64; target_bins.len()];
    let mut left_sums_sq = vec![0.0_f64; target_bins.len()];
    let mut best_threshold = None;
    let mut best_score = f64::NEG_INFINITY;

    for &bin in observed_bins {
        for (t, bins) in target_bins.iter().enumerate() {
            left_counts[t] += bins[bin].count;
            left_sums[t] += bins[bin].sum;
            left_sums_sq[t] += bins[bin].sum_sq;
        }
        let right_count = totals[0].0 - left_counts[0];
        if left_counts[0] < context.options.min_samples_leaf as f64
            || right_count < context.options.min_samples_leaf as f64
        {
            continue;
        }
        let mut combined_gain = 0.0;
        for t in 0..target_bins.len() {
            let (total_w, total_s, total_sq) = totals[t];
            let lc = left_counts[t];
            let ls = left_sums[t];
            let lsq = left_sums_sq[t];
            let rc = total_w - lc;
            let rs = total_s - ls;
            let rsq = total_sq - lsq;
            let parent_loss = total_sq - (total_s * total_s) / total_w;
            let left_loss = lsq - (ls * ls) / lc;
            let right_loss = rsq - (rs * rs) / rc;
            combined_gain += parent_loss - (left_loss + right_loss);
        }
        if combined_gain > best_score {
            best_score = combined_gain;
            best_threshold = Some(bin as u16);
        }
    }

    best_threshold
        .filter(|_| best_score > 0.0)
        .map(|threshold_bin| BinarySplitChoice {
            feature_index,
            threshold_bin,
            score: best_score,
            missing_direction: MissingBranchDirection::Node,
        })
}

fn score_numeric_split_choice_mean_from_hist(
    context: &BuildContext<'_>,
    feature_index: usize,
    total_weight: f64,
    bins: &[RegressionHistogramBin],
    observed_bins: &[usize],
) -> Option<BinarySplitChoice> {
    if observed_bins.len() <= 1 {
        return None;
    }
    let total_sum = bins.iter().map(|bin| bin.sum).sum::<f64>();
    let total_sum_sq = bins.iter().map(|bin| bin.sum_sq).sum::<f64>();
    let parent_loss = total_sum_sq - (total_sum * total_sum) / total_weight;
    let mut left_count = 0.0_f64;
    let mut left_sum = 0.0;
    let mut left_sum_sq = 0.0;
    let mut best_threshold = None;
    let mut best_score = f64::NEG_INFINITY;

    for &bin in observed_bins {
        left_count += bins[bin].count;
        left_sum += bins[bin].sum;
        left_sum_sq += bins[bin].sum_sq;
        let right_count = total_weight - left_count;
        if left_count < context.options.min_samples_leaf as f64
            || right_count < context.options.min_samples_leaf as f64
        {
            continue;
        }
        let right_sum = total_sum - left_sum;
        let right_sum_sq = total_sum_sq - left_sum_sq;
        let left_loss = left_sum_sq - (left_sum * left_sum) / left_count;
        let right_loss = right_sum_sq - (right_sum * right_sum) / right_count;
        let score = parent_loss - (left_loss + right_loss);
        if score > best_score {
            best_score = score;
            best_threshold = Some(bin as u16);
        }
    }

    best_threshold.map(|threshold_bin| BinarySplitChoice {
        feature_index,
        threshold_bin,
        score: best_score,
        missing_direction: MissingBranchDirection::Node,
    })
}

fn score_randomized_split_choice_mean_from_hist(
    context: &BuildContext<'_>,
    feature_index: usize,
    rows: &[usize],
    bins: &[RegressionHistogramBin],
    observed_bins: &[usize],
) -> Option<BinarySplitChoice> {
    let candidate_thresholds = observed_bins
        .iter()
        .copied()
        .map(|bin| bin as u16)
        .collect::<Vec<_>>();
    let threshold_bin =
        choose_random_threshold(&candidate_thresholds, feature_index, rows, 0xA11CE551u64)?;
    let total_weight = bins.iter().map(|bin| bin.count).sum::<f64>();
    let total_sum = bins.iter().map(|bin| bin.sum).sum::<f64>();
    let total_sum_sq = bins.iter().map(|bin| bin.sum_sq).sum::<f64>();
    let mut left_count = 0.0_f64;
    let mut left_sum = 0.0;
    let mut left_sum_sq = 0.0;
    for bin in 0..=threshold_bin as usize {
        if bin >= bins.len() {
            break;
        }
        left_count += bins[bin].count;
        left_sum += bins[bin].sum;
        left_sum_sq += bins[bin].sum_sq;
    }
    let right_count = total_weight - left_count;
    if left_count < context.options.min_samples_leaf as f64
        || right_count < context.options.min_samples_leaf as f64
    {
        return None;
    }
    let parent_loss = total_sum_sq - (total_sum * total_sum) / total_weight;
    let right_sum = total_sum - left_sum;
    let right_sum_sq = total_sum_sq - left_sum_sq;
    let left_loss = left_sum_sq - (left_sum * left_sum) / left_count;
    let right_loss = right_sum_sq - (right_sum * right_sum) / right_count;
    Some(BinarySplitChoice {
        feature_index,
        threshold_bin,
        score: parent_loss - (left_loss + right_loss),
        missing_direction: MissingBranchDirection::Node,
    })
}

fn score_binary_split_choice_mean(
    context: &BuildContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<BinarySplitChoice> {
    let mut left_count = 0.0_f64;
    let mut left_sum = 0.0;
    let mut left_sum_sq = 0.0;
    let mut total_weight = 0.0_f64;
    let mut total_sum = 0.0;
    let mut total_sum_sq = 0.0;

    for row_idx in rows {
        let w = context.table.sample_weight(*row_idx);
        let target = context.targets[*row_idx];
        total_weight += w;
        total_sum += w * target;
        total_sum_sq += w * target * target;
        if !context
            .table
            .binned_boolean_value(feature_index, *row_idx)
            .expect("binary feature must expose boolean values")
        {
            left_count += w;
            left_sum += w * target;
            left_sum_sq += w * target * target;
        }
    }

    let right_count = total_weight - left_count;
    if left_count < context.options.min_samples_leaf as f64
        || right_count < context.options.min_samples_leaf as f64
    {
        return None;
    }

    let parent_loss = total_sum_sq - (total_sum * total_sum) / total_weight;
    let right_sum = total_sum - left_sum;
    let right_sum_sq = total_sum_sq - left_sum_sq;
    let left_loss = left_sum_sq - (left_sum * left_sum) / left_count;
    let right_loss = right_sum_sq - (right_sum * right_sum) / right_count;

    Some(BinarySplitChoice {
        feature_index,
        threshold_bin: 0,
        score: parent_loss - (left_loss + right_loss),
        missing_direction: MissingBranchDirection::Node,
    })
}

fn score_numeric_split_mean_fast(
    table: &dyn TableAccess,
    targets: &[f64],
    feature_index: usize,
    rows: &[usize],
    min_samples_leaf: usize,
) -> Option<RegressionSplitCandidate> {
    let bin_cap = table.numeric_bin_cap();
    if bin_cap == 0 {
        return None;
    }

    let mut bin_count = vec![0.0_f64; bin_cap];
    let mut bin_sum = vec![0.0; bin_cap];
    let mut bin_sum_sq = vec![0.0; bin_cap];
    let mut observed_bins = vec![false; bin_cap];
    let mut total_sum = 0.0;
    let mut total_sum_sq = 0.0;
    let mut total_weight = 0.0_f64;

    for row_idx in rows {
        let bin = table.binned_value(feature_index, *row_idx) as usize;
        if bin >= bin_cap {
            return None;
        }
        let w = table.sample_weight(*row_idx);
        let target = targets[*row_idx];
        bin_count[bin] += w;
        bin_sum[bin] += w * target;
        bin_sum_sq[bin] += w * target * target;
        observed_bins[bin] = true;
        total_sum += w * target;
        total_sum_sq += w * target * target;
        total_weight += w;
    }

    let observed_bins: Vec<usize> = observed_bins
        .into_iter()
        .enumerate()
        .filter_map(|(bin, seen)| seen.then_some(bin))
        .collect();
    if observed_bins.len() <= 1 {
        return None;
    }

    let parent_loss = total_sum_sq - (total_sum * total_sum) / total_weight;
    let mut left_count = 0.0_f64;
    let mut left_sum = 0.0;
    let mut left_sum_sq = 0.0;
    let mut best_threshold = None;
    let mut best_score = f64::NEG_INFINITY;

    for &bin in &observed_bins {
        left_count += bin_count[bin];
        left_sum += bin_sum[bin];
        left_sum_sq += bin_sum_sq[bin];
        let right_count = total_weight - left_count;

        if left_count < min_samples_leaf as f64 || right_count < min_samples_leaf as f64 {
            continue;
        }

        let right_sum = total_sum - left_sum;
        let right_sum_sq = total_sum_sq - left_sum_sq;
        let left_loss = left_sum_sq - (left_sum * left_sum) / left_count;
        let right_loss = right_sum_sq - (right_sum * right_sum) / right_count;
        let score = parent_loss - (left_loss + right_loss);
        if score > best_score {
            best_score = score;
            best_threshold = Some(bin as u16);
        }
    }

    let threshold_bin = best_threshold?;
    Some(RegressionSplitCandidate {
        feature_index,
        threshold_bin,
        score: best_score,
        missing_direction: MissingBranchDirection::Node,
    })
}

fn score_numeric_split_choice_mean_fast(
    context: &BuildContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<BinarySplitChoice> {
    score_numeric_split_mean_fast(
        context.table,
        context.targets,
        feature_index,
        rows,
        context.options.min_samples_leaf,
    )
    .map(|candidate| BinarySplitChoice {
        feature_index: candidate.feature_index,
        threshold_bin: candidate.threshold_bin,
        score: candidate.score,
        missing_direction: MissingBranchDirection::Node,
    })
}

fn score_randomized_split_mean_fast(
    table: &dyn TableAccess,
    targets: &[f64],
    feature_index: usize,
    rows: &[usize],
    min_samples_leaf: usize,
) -> Option<RegressionSplitCandidate> {
    let bin_cap = table.numeric_bin_cap();
    if bin_cap == 0 {
        return None;
    }
    let mut observed_bins = vec![false; bin_cap];
    for row_idx in rows {
        let bin = table.binned_value(feature_index, *row_idx) as usize;
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
        choose_random_threshold(&candidate_thresholds, feature_index, rows, 0xA11CE551u64)?;

    let mut left_count = 0.0_f64;
    let mut left_sum = 0.0;
    let mut left_sum_sq = 0.0;
    let mut total_weight = 0.0_f64;
    let mut total_sum = 0.0;
    let mut total_sum_sq = 0.0;
    for row_idx in rows {
        let w = table.sample_weight(*row_idx);
        let target = targets[*row_idx];
        total_weight += w;
        total_sum += w * target;
        total_sum_sq += w * target * target;
        if table.binned_value(feature_index, *row_idx) <= threshold_bin {
            left_count += w;
            left_sum += w * target;
            left_sum_sq += w * target * target;
        }
    }
    let right_count = total_weight - left_count;
    if left_count < min_samples_leaf as f64 || right_count < min_samples_leaf as f64 {
        return None;
    }
    let parent_loss = total_sum_sq - (total_sum * total_sum) / total_weight;
    let right_sum = total_sum - left_sum;
    let right_sum_sq = total_sum_sq - left_sum_sq;
    let left_loss = left_sum_sq - (left_sum * left_sum) / left_count;
    let right_loss = right_sum_sq - (right_sum * right_sum) / right_count;
    let score = parent_loss - (left_loss + right_loss);

    Some(RegressionSplitCandidate {
        feature_index,
        threshold_bin,
        score,
        missing_direction: MissingBranchDirection::Node,
    })
}

fn score_randomized_split_choice_mean_fast(
    context: &BuildContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<BinarySplitChoice> {
    score_randomized_split_mean_fast(
        context.table,
        context.targets,
        feature_index,
        rows,
        context.options.min_samples_leaf,
    )
    .map(|candidate| BinarySplitChoice {
        feature_index: candidate.feature_index,
        threshold_bin: candidate.threshold_bin,
        score: candidate.score,
        missing_direction: MissingBranchDirection::Node,
    })
}

fn feature_has_missing(table: &dyn TableAccess, feature_index: usize, rows: &[usize]) -> bool {
    rows.iter()
        .any(|row_idx| table.is_missing(feature_index, *row_idx))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_regression_missing_assignment(
    table: &dyn TableAccess,
    targets: &[f64],
    feature_index: usize,
    rows: &[usize],
    criterion: Criterion,
    min_samples_leaf: usize,
    threshold_bin: u16,
    parent_loss: f64,
) -> Option<RegressionSplitCandidate> {
    let mut left_rows = Vec::new();
    let mut right_rows = Vec::new();
    let mut missing_rows = Vec::new();

    for row_idx in rows.iter().copied() {
        if table.is_missing(feature_index, row_idx) {
            missing_rows.push(row_idx);
        } else if table.is_binary_binned_feature(feature_index) {
            if !table
                .binned_boolean_value(feature_index, row_idx)
                .expect("observed binary feature must expose boolean values")
            {
                left_rows.push(row_idx);
            } else {
                right_rows.push(row_idx);
            }
        } else if table.binned_value(feature_index, row_idx) <= threshold_bin {
            left_rows.push(row_idx);
        } else {
            right_rows.push(row_idx);
        }
    }

    let evaluate = |direction: MissingBranchDirection| {
        let mut candidate_left = left_rows.clone();
        let mut candidate_right = right_rows.clone();
        match direction {
            MissingBranchDirection::Left => candidate_left.extend(missing_rows.iter().copied()),
            MissingBranchDirection::Right => candidate_right.extend(missing_rows.iter().copied()),
            MissingBranchDirection::Node => {}
        }
        if candidate_left.len() < min_samples_leaf || candidate_right.len() < min_samples_leaf {
            return None;
        }

        let score = parent_loss
            - (regression_loss(table, &candidate_left, targets, criterion)
                + regression_loss(table, &candidate_right, targets, criterion));
        Some(RegressionSplitCandidate {
            feature_index,
            threshold_bin,
            score,
            missing_direction: direction,
        })
    };

    if missing_rows.is_empty() {
        evaluate(MissingBranchDirection::Node)
    } else {
        [MissingBranchDirection::Left, MissingBranchDirection::Right]
            .into_iter()
            .filter_map(evaluate)
            .max_by(|left, right| left.score.total_cmp(&right.score))
    }
}

fn score_numeric_oblivious_split_mean_fast(
    table: &dyn TableAccess,
    row_indices: &[usize],
    targets: &[f64],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
    min_samples_leaf: usize,
) -> Option<ObliviousSplitCandidate> {
    let bin_cap = table.numeric_bin_cap();
    if bin_cap == 0 {
        return None;
    }
    let mut threshold_scores = vec![0.0; bin_cap];
    let mut observed_any = false;

    let mut bin_count = vec![0.0_f64; bin_cap];
    let mut bin_sum = vec![0.0; bin_cap];
    let mut bin_sum_sq = vec![0.0; bin_cap];
    let mut observed_bins = vec![false; bin_cap];

    for leaf in leaves {
        bin_count.fill(0.0);
        bin_sum.fill(0.0);
        bin_sum_sq.fill(0.0);
        observed_bins.fill(false);

        for row_idx in &row_indices[leaf.start..leaf.end] {
            let bin = table.binned_value(feature_index, *row_idx) as usize;
            if bin >= bin_cap {
                return None;
            }
            let w = table.sample_weight(*row_idx);
            let target = targets[*row_idx];
            bin_count[bin] += w;
            bin_sum[bin] += w * target;
            bin_sum_sq[bin] += w * target * target;
            observed_bins[bin] = true;
        }

        let observed_bins: Vec<usize> = observed_bins
            .iter()
            .enumerate()
            .filter_map(|(bin, seen)| (*seen).then_some(bin))
            .collect();
        if observed_bins.len() <= 1 {
            continue;
        }
        observed_any = true;

        let parent_loss = if leaf.total_weight == 0.0 {
            0.0
        } else {
            leaf.sum_sq - (leaf.sum * leaf.sum) / leaf.total_weight
        };
        let mut left_count = 0.0_f64;
        let mut left_sum = 0.0;
        let mut left_sum_sq = 0.0;

        for &bin in &observed_bins {
            left_count += bin_count[bin];
            left_sum += bin_sum[bin];
            left_sum_sq += bin_sum_sq[bin];
            let right_count = leaf.total_weight - left_count;

            if left_count < min_samples_leaf as f64 || right_count < min_samples_leaf as f64 {
                continue;
            }

            let right_sum = leaf.sum - left_sum;
            let right_sum_sq = leaf.sum_sq - left_sum_sq;
            let left_loss = left_sum_sq - (left_sum * left_sum) / left_count;
            let right_loss = right_sum_sq - (right_sum * right_sum) / right_count;
            threshold_scores[bin] += parent_loss - (left_loss + right_loss);
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

fn score_binary_oblivious_split(
    table: &dyn TableAccess,
    row_indices: &[usize],
    targets: &[f64],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
    criterion: Criterion,
    min_samples_leaf: usize,
) -> Option<ObliviousSplitCandidate> {
    let mut score = 0.0;
    for leaf in leaves {
        let leaf_rows = &row_indices[leaf.start..leaf.end];
        let (left_rows, right_rows): (Vec<usize>, Vec<usize>) =
            leaf_rows.iter().copied().partition(|row_idx| {
                !table
                    .binned_boolean_value(feature_index, *row_idx)
                    .expect("binary feature must expose boolean values")
            });

        if left_rows.len() < min_samples_leaf || right_rows.len() < min_samples_leaf {
            continue;
        }

        score += regression_loss(table, leaf_rows, targets, criterion)
            - (regression_loss(table, &left_rows, targets, criterion)
                + regression_loss(table, &right_rows, targets, criterion));
    }

    (score > 0.0).then_some(ObliviousSplitCandidate {
        feature_index,
        threshold_bin: 0,
        score,
    })
}

fn score_binary_oblivious_split_mean_fast(
    table: &dyn TableAccess,
    row_indices: &[usize],
    targets: &[f64],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
    min_samples_leaf: usize,
) -> Option<ObliviousSplitCandidate> {
    let mut score = 0.0;
    let mut found_valid = false;

    for leaf in leaves {
        let mut left_count = 0.0_f64;
        let mut left_sum = 0.0;
        let mut left_sum_sq = 0.0;

        for row_idx in &row_indices[leaf.start..leaf.end] {
            if !table
                .binned_boolean_value(feature_index, *row_idx)
                .expect("binary feature must expose boolean values")
            {
                let w = table.sample_weight(*row_idx);
                let target = targets[*row_idx];
                left_count += w;
                left_sum += w * target;
                left_sum_sq += w * target * target;
            }
        }

        let right_count = leaf.total_weight - left_count;
        if left_count < min_samples_leaf as f64 || right_count < min_samples_leaf as f64 {
            continue;
        }

        found_valid = true;
        let parent_loss = if leaf.total_weight == 0.0 {
            0.0
        } else {
            leaf.sum_sq - (leaf.sum * leaf.sum) / leaf.total_weight
        };
        let right_sum = leaf.sum - left_sum;
        let right_sum_sq = leaf.sum_sq - left_sum_sq;
        let left_loss = left_sum_sq - (left_sum * left_sum) / left_count;
        let right_loss = right_sum_sq - (right_sum * right_sum) / right_count;
        score += parent_loss - (left_loss + right_loss);
    }

    (found_valid && score > 0.0).then_some(ObliviousSplitCandidate {
        feature_index,
        threshold_bin: 0,
        score,
    })
}

fn sum_stats(table: &dyn TableAccess, rows: &[usize], targets: &[f64]) -> (f64, f64, f64) {
    rows.iter()
        .fold((0.0, 0.0, 0.0), |(sum, sum_sq, total_w), row_idx| {
            let w = table.sample_weight(*row_idx);
            let v = targets[*row_idx];
            (sum + w * v, sum_sq + w * v * v, total_w + w)
        })
}

fn variance_from_stats(total_weight: f64, sum: f64, sum_sq: f64) -> Option<f64> {
    if total_weight == 0.0 {
        None
    } else {
        Some((sum_sq / total_weight) - (sum / total_weight).powi(2))
    }
}

fn leaf_regression_loss(
    leaf: &ObliviousLeafState,
    row_indices: &[usize],
    targets: &[f64],
    criterion: Criterion,
) -> f64 {
    match criterion {
        Criterion::Mean => {
            if leaf.total_weight == 0.0 {
                0.0
            } else {
                leaf.sum_sq - (leaf.sum * leaf.sum) / leaf.total_weight
            }
        }
        Criterion::Median => sum_absolute_error(&row_indices[leaf.start..leaf.end], targets),
        _ => unreachable!("regression criterion only supports mean or median"),
    }
}

fn has_constant_target(rows: &[usize], targets: &[f64]) -> bool {
    rows.first().is_none_or(|first_row| {
        rows.iter()
            .all(|row_idx| targets[*row_idx] == targets[*first_row])
    })
}

fn push_leaf(
    nodes: &mut Vec<RegressionNode>,
    value: f64,
    sample_count: usize,
    variance: Option<f64>,
) -> usize {
    push_node(
        nodes,
        RegressionNode::Leaf {
            value,
            sample_count,
            variance,
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
    use crate::{CanaryFilter, FeaturePreprocessing, Model, NumericBinBoundary, SplitStrategy};
    use forestfire_data::{DenseTable, NumericBins};

    fn quadratic_table() -> DenseTable {
        DenseTable::with_options(
            vec![
                vec![0.0],
                vec![1.0],
                vec![2.0],
                vec![3.0],
                vec![4.0],
                vec![5.0],
                vec![6.0],
                vec![7.0],
            ],
            vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0],
            0,
            NumericBins::Fixed(128),
        )
        .unwrap()
    }

    fn canary_target_table() -> DenseTable {
        let x: Vec<Vec<f64>> = (0..8).map(|value| vec![value as f64]).collect();
        let probe =
            DenseTable::with_options(x.clone(), vec![0.0; 8], 1, NumericBins::Auto).unwrap();
        let canary_index = probe.n_features();
        let y = (0..probe.n_rows())
            .map(|row_idx| probe.binned_value(canary_index, row_idx) as f64)
            .collect();

        DenseTable::with_options(x, y, 1, NumericBins::Auto).unwrap()
    }

    fn canary_target_table_with_noise_feature() -> DenseTable {
        let x: Vec<Vec<f64>> = (0..8)
            .map(|value| vec![value as f64, (value % 2) as f64])
            .collect();
        let probe =
            DenseTable::with_options(x.clone(), vec![0.0; 8], 1, NumericBins::Auto).unwrap();
        let canary_index = probe.n_features();
        let y = (0..probe.n_rows())
            .map(|row_idx| probe.binned_value(canary_index, row_idx) as f64)
            .collect();

        DenseTable::with_options(x, y, 1, NumericBins::Auto).unwrap()
    }

    fn oblique_canary_target_table() -> DenseTable {
        let x = vec![
            vec![0.0, 9.0],
            vec![1.0, 4.0],
            vec![2.0, 7.0],
            vec![3.0, 2.0],
            vec![4.0, 8.0],
            vec![5.0, 1.0],
            vec![6.0, 6.0],
            vec![7.0, 3.0],
            vec![8.0, 5.0],
            vec![9.0, 0.0],
        ];
        let probe =
            DenseTable::with_options(x.clone(), vec![0.0; x.len()], 1, NumericBins::Fixed(32))
                .unwrap();
        let left_canary = probe.n_features();
        let right_canary = left_canary + 1;
        let y = (0..probe.n_rows())
            .map(|row_idx| {
                probe.binned_value(left_canary, row_idx) as f64
                    - probe.binned_value(right_canary, row_idx) as f64
            })
            .collect::<Vec<_>>();

        DenseTable::with_options(x, y, 1, NumericBins::Fixed(32)).unwrap()
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
            vec![0.0, 1.0, 2.5, 3.5, 4.0, 5.0, 6.5, 7.5, 2.0, 4.5, 6.0, 8.5],
            0,
            NumericBins::Fixed(8),
        )
        .unwrap()
    }

    fn oblique_regression_table() -> DenseTable {
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
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            0,
            NumericBins::Fixed(64),
        )
        .unwrap()
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
    fn randomized_regressor_fits_basic_numeric_pattern() {
        let table = quadratic_table();
        let model = train_randomized_regressor(&table).unwrap();
        let preds = model.predict_table(&table);
        let targets = table_targets(&table);
        let baseline_mean = targets.iter().sum::<f64>() / targets.len() as f64;
        let baseline_sse = targets
            .iter()
            .map(|target| {
                let diff = target - baseline_mean;
                diff * diff
            })
            .sum::<f64>();
        let model_sse = preds
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| {
                let diff = pred - target;
                diff * diff
            })
            .sum::<f64>();

        assert_eq!(model.algorithm(), RegressionTreeAlgorithm::Randomized);
        assert_eq!(model.criterion(), Criterion::Mean);
        assert!(model_sse < baseline_sse);
    }

    #[test]
    fn randomized_regressor_is_repeatable_for_fixed_seed_and_varies_across_seeds() {
        let table = randomized_permutation_table();
        let make_options = |random_seed| RegressionTreeOptions {
            max_depth: 4,
            max_features: Some(2),
            random_seed,
            ..RegressionTreeOptions::default()
        };

        let base_model = train_randomized_regressor_with_criterion_parallelism_and_options(
            &table,
            Criterion::Mean,
            Parallelism::sequential(),
            make_options(91),
        )
        .unwrap();
        let repeated_model = train_randomized_regressor_with_criterion_parallelism_and_options(
            &table,
            Criterion::Mean,
            Parallelism::sequential(),
            make_options(91),
        )
        .unwrap();
        let unique_serializations = (0..32u64)
            .map(|seed| {
                Model::DecisionTreeRegressor(
                    train_randomized_regressor_with_criterion_parallelism_and_options(
                        &table,
                        Criterion::Mean,
                        Parallelism::sequential(),
                        make_options(seed),
                    )
                    .unwrap(),
                )
                .serialize()
                .unwrap()
            })
            .collect::<std::collections::BTreeSet<_>>();

        assert_eq!(
            Model::DecisionTreeRegressor(base_model.clone())
                .serialize()
                .unwrap(),
            Model::DecisionTreeRegressor(repeated_model)
                .serialize()
                .unwrap()
        );
        assert!(unique_serializations.len() >= 4);
    }

    #[test]
    fn cart_regressor_supports_oblique_split_strategy() {
        let table = oblique_regression_table();
        let model = train_cart_regressor_with_criterion_parallelism_and_options(
            &table,
            Criterion::Mean,
            Parallelism::sequential(),
            RegressionTreeOptions {
                max_depth: 1,
                max_features: Some(2),
                split_strategy: SplitStrategy::Oblique,
                ..RegressionTreeOptions::default()
            },
        )
        .unwrap();

        assert_eq!(model.predict_table(&table), table_targets(&table));
        match model.structure() {
            RegressionTreeStructure::Standard { nodes, root } => {
                assert!(matches!(nodes[*root], RegressionNode::ObliqueSplit { .. }));
            }
            RegressionTreeStructure::Oblivious { .. } => panic!("expected standard tree"),
        }
    }

    #[test]
    fn randomized_regressor_supports_oblique_split_strategy() {
        let table = oblique_regression_table();
        let model = train_randomized_regressor_with_criterion_parallelism_and_options(
            &table,
            Criterion::Mean,
            Parallelism::sequential(),
            RegressionTreeOptions {
                max_depth: 1,
                max_features: Some(2),
                random_seed: 11,
                split_strategy: SplitStrategy::Oblique,
                ..RegressionTreeOptions::default()
            },
        )
        .unwrap();

        assert_eq!(model.predict_table(&table), table_targets(&table));
        match model.structure() {
            RegressionTreeStructure::Standard { nodes, root } => {
                assert!(matches!(nodes[*root], RegressionNode::ObliqueSplit { .. }));
            }
            RegressionTreeStructure::Oblivious { .. } => panic!("expected standard tree"),
        }
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

    #[test]
    fn top_n_canary_filter_can_choose_real_regression_split() {
        let table = canary_target_table();
        let model = train_cart_regressor_with_criterion_parallelism_and_options(
            &table,
            Criterion::Mean,
            Parallelism::sequential(),
            RegressionTreeOptions {
                canary_filter: CanaryFilter::TopN(2),
                ..RegressionTreeOptions::default()
            },
        )
        .unwrap();
        let preds = model.predict_table(&table);

        assert!(preds.iter().any(|pred| *pred != preds[0]));
    }

    #[test]
    fn top_fraction_canary_filter_can_choose_real_regression_split() {
        let table = canary_target_table_with_noise_feature();
        let model = train_cart_regressor_with_criterion_parallelism_and_options(
            &table,
            Criterion::Mean,
            Parallelism::sequential(),
            RegressionTreeOptions {
                canary_filter: CanaryFilter::TopFraction(0.5),
                ..RegressionTreeOptions::default()
            },
        )
        .unwrap();
        let preds = model.predict_table(&table);

        assert!(preds.iter().any(|pred| *pred != preds[0]));
    }

    #[test]
    fn oblique_canary_filter_blocks_oblique_regression_growth_when_canary_pair_wins() {
        let table = oblique_canary_target_table();
        let model = train_cart_regressor_with_criterion_parallelism_and_options(
            &table,
            Criterion::Mean,
            Parallelism::sequential(),
            RegressionTreeOptions {
                canary_filter: CanaryFilter::TopN(1),
                split_strategy: SplitStrategy::Oblique,
                max_depth: 1,
                ..RegressionTreeOptions::default()
            },
        )
        .unwrap();
        let preds = model.predict_table(&table);

        assert!(preds.iter().all(|pred| *pred == preds[0]));
        assert_ne!(preds, table_targets(&table));
    }

    #[test]
    fn oblique_regressor_routes_missing_features_independently() {
        let table = DenseTable::with_options(
            vec![
                vec![f64::NAN, 1.0],
                vec![1.0, f64::NAN],
                vec![f64::NAN, f64::NAN],
                vec![2.0, 2.0],
            ],
            vec![0.0; 4],
            0,
            NumericBins::Fixed(16),
        )
        .unwrap();
        let model = DecisionTreeRegressor {
            algorithm: RegressionTreeAlgorithm::Cart,
            criterion: Criterion::Mean,
            structure: RegressionTreeStructure::Standard {
                nodes: vec![
                    RegressionNode::Leaf {
                        value: -1.0,
                        sample_count: 2,
                        variance: None,
                    },
                    RegressionNode::Leaf {
                        value: 1.0,
                        sample_count: 2,
                        variance: None,
                    },
                    RegressionNode::ObliqueSplit {
                        feature_indices: vec![0, 1],
                        weights: vec![1.0, 0.5],
                        missing_directions: vec![
                            crate::tree::shared::MissingBranchDirection::Left,
                            crate::tree::shared::MissingBranchDirection::Right,
                        ],
                        threshold: 1.5,
                        missing_values: vec![0.0],
                        left_child: 0,
                        right_child: 1,
                        sample_count: 4,
                        impurity: 0.0,
                        gain: 1.0,
                        variance: None,
                    },
                ],
                root: 2,
            },
            options: RegressionTreeOptions::default(),
            num_features: 2,
            feature_preprocessing: vec![
                FeaturePreprocessing::Numeric {
                    bin_boundaries: vec![],
                    missing_bin: 16,
                },
                FeaturePreprocessing::Numeric {
                    bin_boundaries: vec![],
                    missing_bin: 16,
                },
            ],
            training_canaries: 0,
        };

        assert_eq!(model.predict_table(&table), vec![-1.0, 1.0, -1.0, 1.0]);
    }

    #[test]
    fn manually_built_regressor_models_serialize_for_each_tree_type() {
        let preprocessing = vec![
            FeaturePreprocessing::Binary,
            FeaturePreprocessing::Numeric {
                bin_boundaries: vec![
                    NumericBinBoundary {
                        bin: 0,
                        upper_bound: 1.0,
                    },
                    NumericBinBoundary {
                        bin: 127,
                        upper_bound: 10.0,
                    },
                ],
                missing_bin: 128,
            },
        ];
        let options = RegressionTreeOptions::default();

        let cart = Model::DecisionTreeRegressor(DecisionTreeRegressor {
            algorithm: RegressionTreeAlgorithm::Cart,
            criterion: Criterion::Mean,
            structure: RegressionTreeStructure::Standard {
                nodes: vec![
                    RegressionNode::Leaf {
                        value: -1.0,
                        sample_count: 2,
                        variance: Some(0.25),
                    },
                    RegressionNode::Leaf {
                        value: 2.5,
                        sample_count: 3,
                        variance: Some(1.0),
                    },
                    RegressionNode::BinarySplit {
                        feature_index: 0,
                        threshold_bin: 0,
                        missing_direction: crate::tree::shared::MissingBranchDirection::Node,
                        missing_values: vec![-1.0],
                        left_child: 0,
                        right_child: 1,
                        sample_count: 5,
                        impurity: 3.5,
                        gain: 1.25,
                        variance: Some(0.7),
                    },
                ],
                root: 2,
            },
            options: options.clone(),
            num_features: 2,
            feature_preprocessing: preprocessing.clone(),
            training_canaries: 0,
        });
        let randomized = Model::DecisionTreeRegressor(DecisionTreeRegressor {
            algorithm: RegressionTreeAlgorithm::Randomized,
            criterion: Criterion::Median,
            structure: RegressionTreeStructure::Standard {
                nodes: vec![
                    RegressionNode::Leaf {
                        value: -1.0,
                        sample_count: 2,
                        variance: Some(0.25),
                    },
                    RegressionNode::Leaf {
                        value: 2.5,
                        sample_count: 3,
                        variance: Some(1.0),
                    },
                    RegressionNode::BinarySplit {
                        feature_index: 0,
                        threshold_bin: 0,
                        missing_direction: crate::tree::shared::MissingBranchDirection::Node,
                        missing_values: vec![-1.0],
                        left_child: 0,
                        right_child: 1,
                        sample_count: 5,
                        impurity: 3.5,
                        gain: 0.8,
                        variance: Some(0.7),
                    },
                ],
                root: 2,
            },
            options: options.clone(),
            num_features: 2,
            feature_preprocessing: preprocessing.clone(),
            training_canaries: 0,
        });
        let oblivious = Model::DecisionTreeRegressor(DecisionTreeRegressor {
            algorithm: RegressionTreeAlgorithm::Oblivious,
            criterion: Criterion::Median,
            structure: RegressionTreeStructure::Oblivious {
                splits: vec![ObliviousSplit {
                    feature_index: 1,
                    threshold_bin: 127,
                    sample_count: 4,
                    impurity: 2.0,
                    gain: 0.5,
                }],
                leaf_values: vec![0.0, 10.0],
                leaf_sample_counts: vec![2, 2],
                leaf_variances: vec![Some(0.0), Some(1.0)],
            },
            options,
            num_features: 2,
            feature_preprocessing: preprocessing,
            training_canaries: 0,
        });

        for (tree_type, model) in [
            ("cart", cart),
            ("randomized", randomized),
            ("oblivious", oblivious),
        ] {
            let json = model.serialize().unwrap();
            assert!(json.contains(&format!("\"tree_type\":\"{tree_type}\"")));
            assert!(json.contains("\"task\":\"regression\""));
        }
    }

    #[test]
    fn cart_regressor_assigns_training_missing_values_to_best_child() {
        let table = DenseTable::with_canaries(
            vec![
                vec![0.0],
                vec![0.0],
                vec![1.0],
                vec![1.0],
                vec![f64::NAN],
                vec![f64::NAN],
            ],
            vec![0.0, 0.0, 10.0, 10.0, 0.0, 0.0],
            0,
        )
        .unwrap();

        let model = train_cart_regressor(&table).unwrap();

        let wrapped = Model::DecisionTreeRegressor(model.clone());
        assert_eq!(
            wrapped.predict_rows(vec![vec![f64::NAN]]).unwrap(),
            vec![0.0]
        );
    }

    #[test]
    fn cart_regressor_defaults_unseen_missing_to_node_mean() {
        let table = DenseTable::with_canaries(
            vec![vec![0.0], vec![0.0], vec![1.0]],
            vec![0.0, 0.0, 9.0],
            0,
        )
        .unwrap();

        let model = train_cart_regressor(&table).unwrap();
        let wrapped = Model::DecisionTreeRegressor(model.clone());
        let prediction = wrapped.predict_rows(vec![vec![f64::NAN]]).unwrap()[0];

        assert!((prediction - 3.0).abs() < 1e-9);
    }

    fn table_targets(table: &dyn TableAccess) -> Vec<f64> {
        (0..table.n_rows())
            .map(|row_idx| table.target_value(row_idx))
            .collect()
    }
}
