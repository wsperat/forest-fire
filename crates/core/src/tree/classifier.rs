//! Classification tree learners.
//!
//! The module intentionally supports multiple tree families because they express
//! different tradeoffs:
//!
//! - `id3` / `c45` keep multiway splits for categorical-like binned features.
//! - `cart` is the standard binary threshold learner.
//! - `randomized` keeps the CART structure but cheapens split search.
//! - `oblivious` uses one split per depth, which is attractive for some runtime
//!   layouts and boosting-style ensembles.
//!
//! The hot numeric paths are written around binned histograms and in-place row
//! partitioning. That is why many helpers operate on row-index buffers instead of
//! allocating fresh row vectors at every recursive step.

use crate::ir::{
    BinaryChildren, BinarySplit, IndexedLeaf, LeafIndexing, LeafPayload, MultiwayBranch,
    MultiwaySplit, NodeStats, NodeTreeNode, ObliviousLevel, ObliviousSplit as IrObliviousSplit,
    TrainingMetadata, TreeDefinition, criterion_name, feature_name, threshold_upper_bound,
    tree_type_name,
};
use crate::tree::shared::{
    MissingBranchDirection, candidate_feature_indices, choose_random_threshold, node_seed,
    partition_rows_for_binary_split,
};
use crate::{Criterion, FeaturePreprocessing, Parallelism, capture_feature_preprocessing};
use forestfire_data::TableAccess;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

mod histogram;
mod ir_support;
mod oblivious;
mod partitioning;
mod split_scoring;

use histogram::{
    ClassificationFeatureHistogram, build_classification_node_histograms,
    subtract_classification_node_histograms,
};
use ir_support::{
    binary_split_ir, normalized_class_probabilities, oblivious_split_ir, standard_node_depths,
};
use oblivious::train_oblivious_structure;
use partitioning::partition_rows_for_multiway_split;
use split_scoring::{
    MultiwayMetric, SplitScoringContext, score_binary_split_choice_from_hist,
    score_multiway_split_choice,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionTreeAlgorithm {
    Id3,
    C45,
    Cart,
    Randomized,
    Oblivious,
}

/// Shared training controls for classification tree learners.
///
/// The defaults are intentionally modest rather than "grow until pure", because
/// ForestFire wants trees to be a stable building block for ensembles and
/// interpretable standalone models.
#[derive(Debug, Clone, Copy)]
pub struct DecisionTreeOptions {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
    pub random_seed: u64,
}

impl Default for DecisionTreeOptions {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            random_seed: 0,
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

/// Concrete trained classification tree.
#[derive(Debug, Clone)]
pub struct DecisionTreeClassifier {
    algorithm: DecisionTreeAlgorithm,
    criterion: Criterion,
    class_labels: Vec<f64>,
    structure: TreeStructure,
    options: DecisionTreeOptions,
    num_features: usize,
    feature_preprocessing: Vec<FeaturePreprocessing>,
    training_canaries: usize,
}

#[derive(Debug, Clone)]
pub(crate) enum TreeStructure {
    Standard {
        nodes: Vec<TreeNode>,
        root: usize,
    },
    Oblivious {
        splits: Vec<ObliviousSplit>,
        leaf_class_indices: Vec<usize>,
        leaf_sample_counts: Vec<usize>,
        leaf_class_counts: Vec<Vec<usize>>,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct ObliviousSplit {
    pub(crate) feature_index: usize,
    pub(crate) threshold_bin: u16,
    pub(crate) missing_directions: Vec<MissingBranchDirection>,
    pub(crate) sample_count: usize,
    pub(crate) impurity: f64,
    pub(crate) gain: f64,
}

#[derive(Debug, Clone)]
pub(crate) enum TreeNode {
    Leaf {
        class_index: usize,
        sample_count: usize,
        class_counts: Vec<usize>,
    },
    MultiwaySplit {
        feature_index: usize,
        fallback_class_index: usize,
        branches: Vec<(u16, usize)>,
        missing_child: Option<usize>,
        sample_count: usize,
        impurity: f64,
        gain: f64,
        class_counts: Vec<usize>,
    },
    BinarySplit {
        feature_index: usize,
        threshold_bin: u16,
        missing_direction: MissingBranchDirection,
        left_child: usize,
        right_child: usize,
        sample_count: usize,
        impurity: f64,
        gain: f64,
        class_counts: Vec<usize>,
    },
}

#[derive(Debug, Clone, Copy)]
struct BinarySplitChoice {
    feature_index: usize,
    score: f64,
    threshold_bin: u16,
    missing_direction: MissingBranchDirection,
}

#[derive(Debug, Clone)]
struct MultiwaySplitChoice {
    feature_index: usize,
    score: f64,
    branch_bins: Vec<u16>,
    missing_branch_bin: Option<u16>,
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
    train_id3_with_criterion_parallelism_and_options(
        train_set,
        criterion,
        parallelism,
        DecisionTreeOptions::default(),
    )
}

pub(crate) fn train_id3_with_criterion_parallelism_and_options(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
    options: DecisionTreeOptions,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_classifier(
        train_set,
        DecisionTreeAlgorithm::Id3,
        criterion,
        parallelism,
        options,
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
    train_c45_with_criterion_parallelism_and_options(
        train_set,
        criterion,
        parallelism,
        DecisionTreeOptions::default(),
    )
}

pub(crate) fn train_c45_with_criterion_parallelism_and_options(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
    options: DecisionTreeOptions,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_classifier(
        train_set,
        DecisionTreeAlgorithm::C45,
        criterion,
        parallelism,
        options,
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
    train_cart_with_criterion_parallelism_and_options(
        train_set,
        criterion,
        parallelism,
        DecisionTreeOptions::default(),
    )
}

pub(crate) fn train_cart_with_criterion_parallelism_and_options(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
    options: DecisionTreeOptions,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_classifier(
        train_set,
        DecisionTreeAlgorithm::Cart,
        criterion,
        parallelism,
        options,
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
    train_oblivious_with_criterion_parallelism_and_options(
        train_set,
        criterion,
        parallelism,
        DecisionTreeOptions::default(),
    )
}

pub(crate) fn train_oblivious_with_criterion_parallelism_and_options(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
    options: DecisionTreeOptions,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_classifier(
        train_set,
        DecisionTreeAlgorithm::Oblivious,
        criterion,
        parallelism,
        options,
    )
}

pub fn train_randomized(
    train_set: &dyn TableAccess,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_randomized_with_criterion(train_set, Criterion::Gini)
}

pub fn train_randomized_with_criterion(
    train_set: &dyn TableAccess,
    criterion: Criterion,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_randomized_with_criterion_and_parallelism(train_set, criterion, Parallelism::sequential())
}

pub(crate) fn train_randomized_with_criterion_and_parallelism(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_randomized_with_criterion_parallelism_and_options(
        train_set,
        criterion,
        parallelism,
        DecisionTreeOptions::default(),
    )
}

pub(crate) fn train_randomized_with_criterion_parallelism_and_options(
    train_set: &dyn TableAccess,
    criterion: Criterion,
    parallelism: Parallelism,
    options: DecisionTreeOptions,
) -> Result<DecisionTreeClassifier, DecisionTreeError> {
    train_classifier(
        train_set,
        DecisionTreeAlgorithm::Randomized,
        criterion,
        parallelism,
        options,
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
        DecisionTreeAlgorithm::Cart | DecisionTreeAlgorithm::Randomized => {
            let mut nodes = Vec::new();
            let mut all_rows: Vec<usize> = (0..train_set.n_rows()).collect();
            let context = BuildContext {
                table: train_set,
                class_indices: &class_indices,
                class_labels: &class_labels,
                algorithm,
                criterion,
                parallelism,
                options,
            };
            let root = build_binary_node_in_place(&context, &mut nodes, &mut all_rows, 0);
            TreeStructure::Standard { nodes, root }
        }
        DecisionTreeAlgorithm::Id3 | DecisionTreeAlgorithm::C45 => {
            let mut nodes = Vec::new();
            let mut all_rows: Vec<usize> = (0..train_set.n_rows()).collect();
            let context = BuildContext {
                table: train_set,
                class_indices: &class_indices,
                class_labels: &class_labels,
                algorithm,
                criterion,
                parallelism,
                options,
            };
            let root = build_multiway_node_in_place(&context, &mut nodes, &mut all_rows, 0);
            TreeStructure::Standard { nodes, root }
        }
    };

    Ok(DecisionTreeClassifier {
        algorithm,
        criterion,
        class_labels,
        structure,
        options,
        num_features: train_set.n_features(),
        feature_preprocessing: capture_feature_preprocessing(train_set),
        training_canaries: train_set.canaries(),
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

    pub fn predict_proba_table(&self, table: &dyn TableAccess) -> Vec<Vec<f64>> {
        (0..table.n_rows())
            .map(|row_idx| self.predict_proba_row(table, row_idx))
            .collect()
    }

    fn predict_row(&self, table: &dyn TableAccess, row_idx: usize) -> f64 {
        match &self.structure {
            TreeStructure::Standard { nodes, root } => {
                let mut node_index = *root;
                loop {
                    match &nodes[node_index] {
                        TreeNode::Leaf { class_index, .. } => {
                            return self.class_labels[*class_index];
                        }
                        TreeNode::MultiwaySplit {
                            feature_index,
                            fallback_class_index,
                            branches,
                            missing_child,
                            ..
                        } => {
                            if table.is_missing(*feature_index, row_idx) {
                                if let Some(child_index) = missing_child {
                                    node_index = *child_index;
                                } else {
                                    return self.class_labels[*fallback_class_index];
                                }
                                continue;
                            }
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
                            missing_direction,
                            left_child,
                            right_child,
                            class_counts,
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
                                    MissingBranchDirection::Node => {
                                        return self.class_labels
                                            [majority_class_from_counts(class_counts)];
                                    }
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
                    }
                }
            }
            TreeStructure::Oblivious {
                splits,
                leaf_class_indices,
                ..
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

    fn predict_proba_row(&self, table: &dyn TableAccess, row_idx: usize) -> Vec<f64> {
        match &self.structure {
            TreeStructure::Standard { nodes, root } => {
                let mut node_index = *root;
                loop {
                    match &nodes[node_index] {
                        TreeNode::Leaf { class_counts, .. } => {
                            return normalized_class_probabilities(class_counts);
                        }
                        TreeNode::MultiwaySplit {
                            feature_index,
                            branches,
                            missing_child,
                            class_counts,
                            ..
                        } => {
                            if table.is_missing(*feature_index, row_idx) {
                                if let Some(child_index) = missing_child {
                                    node_index = *child_index;
                                } else {
                                    return normalized_class_probabilities(class_counts);
                                }
                                continue;
                            }
                            let bin = table.binned_value(*feature_index, row_idx);
                            if let Some((_, child_index)) =
                                branches.iter().find(|(branch_bin, _)| *branch_bin == bin)
                            {
                                node_index = *child_index;
                            } else {
                                return normalized_class_probabilities(class_counts);
                            }
                        }
                        TreeNode::BinarySplit {
                            feature_index,
                            threshold_bin,
                            missing_direction,
                            left_child,
                            right_child,
                            class_counts,
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
                                    MissingBranchDirection::Node => {
                                        return normalized_class_probabilities(class_counts);
                                    }
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
                    }
                }
            }
            TreeStructure::Oblivious {
                splits,
                leaf_class_counts,
                ..
            } => {
                let leaf_index = splits.iter().fold(0usize, |leaf_index, split| {
                    let go_right =
                        table.binned_value(split.feature_index, row_idx) > split.threshold_bin;
                    (leaf_index << 1) | usize::from(go_right)
                });

                normalized_class_probabilities(&leaf_class_counts[leaf_index])
            }
        }
    }

    pub(crate) fn class_labels(&self) -> &[f64] {
        &self.class_labels
    }

    pub(crate) fn structure(&self) -> &TreeStructure {
        &self.structure
    }

    pub(crate) fn num_features(&self) -> usize {
        self.num_features
    }

    pub(crate) fn feature_preprocessing(&self) -> &[FeaturePreprocessing] {
        &self.feature_preprocessing
    }

    pub(crate) fn training_metadata(&self) -> TrainingMetadata {
        TrainingMetadata {
            algorithm: "dt".to_string(),
            task: "classification".to_string(),
            tree_type: tree_type_name(match self.algorithm {
                DecisionTreeAlgorithm::Id3 => crate::TreeType::Id3,
                DecisionTreeAlgorithm::C45 => crate::TreeType::C45,
                DecisionTreeAlgorithm::Cart => crate::TreeType::Cart,
                DecisionTreeAlgorithm::Randomized => crate::TreeType::Randomized,
                DecisionTreeAlgorithm::Oblivious => crate::TreeType::Oblivious,
            })
            .to_string(),
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
            class_labels: Some(self.class_labels.clone()),
            learning_rate: None,
            bootstrap: None,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
        }
    }

    pub(crate) fn to_ir_tree(&self) -> TreeDefinition {
        match &self.structure {
            TreeStructure::Standard { nodes, root } => {
                let depths = standard_node_depths(nodes, *root);
                TreeDefinition::NodeTree {
                    tree_id: 0,
                    weight: 1.0,
                    root_node_id: *root,
                    nodes: nodes
                        .iter()
                        .enumerate()
                        .map(|(node_id, node)| match node {
                            TreeNode::Leaf {
                                class_index,
                                sample_count,
                                class_counts,
                            } => NodeTreeNode::Leaf {
                                node_id,
                                depth: depths[node_id],
                                leaf: self.class_leaf(*class_index),
                                stats: NodeStats {
                                    sample_count: *sample_count,
                                    impurity: None,
                                    gain: None,
                                    class_counts: Some(class_counts.clone()),
                                    variance: None,
                                },
                            },
                            TreeNode::BinarySplit {
                                feature_index,
                                threshold_bin,
                                missing_direction,
                                left_child,
                                right_child,
                                sample_count,
                                impurity,
                                gain,
                                class_counts,
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
                                    class_counts: Some(class_counts.clone()),
                                    variance: None,
                                },
                            },
                            TreeNode::MultiwaySplit {
                                feature_index,
                                fallback_class_index,
                                branches,
                                missing_child: _,
                                sample_count,
                                impurity,
                                gain,
                                class_counts,
                            } => NodeTreeNode::MultiwayBranch {
                                node_id,
                                depth: depths[node_id],
                                split: MultiwaySplit {
                                    split_type: "binned_value_multiway".to_string(),
                                    feature_index: *feature_index,
                                    feature_name: feature_name(*feature_index),
                                    comparison_dtype: "uint16".to_string(),
                                },
                                branches: branches
                                    .iter()
                                    .map(|(bin, child)| MultiwayBranch {
                                        bin: *bin,
                                        child: *child,
                                    })
                                    .collect(),
                                unmatched_leaf: self.class_leaf(*fallback_class_index),
                                stats: NodeStats {
                                    sample_count: *sample_count,
                                    impurity: Some(*impurity),
                                    gain: Some(*gain),
                                    class_counts: Some(class_counts.clone()),
                                    variance: None,
                                },
                            },
                        })
                        .collect(),
                }
            }
            TreeStructure::Oblivious {
                splits,
                leaf_class_indices,
                leaf_sample_counts,
                leaf_class_counts,
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
                leaves: leaf_class_indices
                    .iter()
                    .enumerate()
                    .map(|(leaf_index, class_index)| IndexedLeaf {
                        leaf_index,
                        leaf: self.class_leaf(*class_index),
                        stats: NodeStats {
                            sample_count: leaf_sample_counts[leaf_index],
                            impurity: None,
                            gain: None,
                            class_counts: Some(leaf_class_counts[leaf_index].clone()),
                            variance: None,
                        },
                    })
                    .collect(),
            },
        }
    }

    fn class_leaf(&self, class_index: usize) -> LeafPayload {
        LeafPayload::ClassIndex {
            class_index,
            class_value: self.class_labels[class_index],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_ir_parts(
        algorithm: DecisionTreeAlgorithm,
        criterion: Criterion,
        class_labels: Vec<f64>,
        structure: TreeStructure,
        options: DecisionTreeOptions,
        num_features: usize,
        feature_preprocessing: Vec<FeaturePreprocessing>,
        training_canaries: usize,
    ) -> Self {
        Self {
            algorithm,
            criterion,
            class_labels,
            structure,
            options,
            num_features,
            feature_preprocessing,
            training_canaries,
        }
    }
}

fn build_binary_node_in_place(
    context: &BuildContext<'_>,
    nodes: &mut Vec<TreeNode>,
    rows: &mut [usize],
    depth: usize,
) -> usize {
    build_binary_node_in_place_with_hist(context, nodes, rows, depth, None)
}

fn build_binary_node_in_place_with_hist(
    context: &BuildContext<'_>,
    nodes: &mut Vec<TreeNode>,
    rows: &mut [usize],
    depth: usize,
    histograms: Option<Vec<ClassificationFeatureHistogram>>,
) -> usize {
    let majority_class_index =
        majority_class(rows, context.class_indices, context.class_labels.len());
    let current_class_counts =
        class_counts(rows, context.class_indices, context.class_labels.len());

    if rows.is_empty()
        || depth >= context.options.max_depth
        || rows.len() < context.options.min_samples_split
        || is_pure(rows, context.class_indices)
    {
        return push_leaf(
            nodes,
            majority_class_index,
            rows.len(),
            current_class_counts,
        );
    }

    let scoring = SplitScoringContext {
        table: context.table,
        class_indices: context.class_indices,
        num_classes: context.class_labels.len(),
        criterion: context.criterion,
        min_samples_leaf: context.options.min_samples_leaf,
    };
    let histograms = histograms.unwrap_or_else(|| {
        build_classification_node_histograms(
            context.table,
            context.class_indices,
            rows,
            context.class_labels.len(),
        )
    });
    let feature_indices = candidate_feature_indices(
        context.table.binned_feature_count(),
        context.options.max_features,
        node_seed(context.options.random_seed, depth, rows, 0xC1A5_5EEDu64),
    );
    let best_split = if context.parallelism.enabled() {
        feature_indices
            .into_par_iter()
            .filter_map(|feature_index| {
                score_binary_split_choice_from_hist(
                    &scoring,
                    &histograms[feature_index],
                    feature_index,
                    rows,
                    &current_class_counts,
                    context.algorithm,
                )
            })
            .max_by(|left, right| left.score.total_cmp(&right.score))
    } else {
        feature_indices
            .into_iter()
            .filter_map(|feature_index| {
                score_binary_split_choice_from_hist(
                    &scoring,
                    &histograms[feature_index],
                    feature_index,
                    rows,
                    &current_class_counts,
                    context.algorithm,
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
            push_leaf(
                nodes,
                majority_class_index,
                rows.len(),
                current_class_counts,
            )
        }
        Some(best_split) if best_split.score > 0.0 => {
            let impurity =
                classification_impurity(&current_class_counts, rows.len(), context.criterion);
            let left_count = partition_rows_for_binary_split(
                context.table,
                best_split.feature_index,
                best_split.threshold_bin,
                best_split.missing_direction,
                rows,
            );
            let (left_rows, right_rows) = rows.split_at_mut(left_count);
            let (left_histograms, right_histograms) = if left_rows.len() <= right_rows.len() {
                let left_histograms = build_classification_node_histograms(
                    context.table,
                    context.class_indices,
                    left_rows,
                    context.class_labels.len(),
                );
                let right_histograms =
                    subtract_classification_node_histograms(&histograms, &left_histograms);
                (left_histograms, right_histograms)
            } else {
                let right_histograms = build_classification_node_histograms(
                    context.table,
                    context.class_indices,
                    right_rows,
                    context.class_labels.len(),
                );
                let left_histograms =
                    subtract_classification_node_histograms(&histograms, &right_histograms);
                (left_histograms, right_histograms)
            };
            let left_child = build_binary_node_in_place_with_hist(
                context,
                nodes,
                left_rows,
                depth + 1,
                Some(left_histograms),
            );
            let right_child = build_binary_node_in_place_with_hist(
                context,
                nodes,
                right_rows,
                depth + 1,
                Some(right_histograms),
            );

            push_node(
                nodes,
                TreeNode::BinarySplit {
                    feature_index: best_split.feature_index,
                    threshold_bin: best_split.threshold_bin,
                    missing_direction: best_split.missing_direction,
                    left_child,
                    right_child,
                    sample_count: rows.len(),
                    impurity,
                    gain: best_split.score,
                    class_counts: current_class_counts,
                },
            )
        }
        _ => push_leaf(
            nodes,
            majority_class_index,
            rows.len(),
            current_class_counts,
        ),
    }
}

fn build_multiway_node_in_place(
    context: &BuildContext<'_>,
    nodes: &mut Vec<TreeNode>,
    rows: &mut [usize],
    depth: usize,
) -> usize {
    let majority_class_index =
        majority_class(rows, context.class_indices, context.class_labels.len());
    let current_class_counts =
        class_counts(rows, context.class_indices, context.class_labels.len());

    if rows.is_empty()
        || depth >= context.options.max_depth
        || rows.len() < context.options.min_samples_split
        || is_pure(rows, context.class_indices)
    {
        return push_leaf(
            nodes,
            majority_class_index,
            rows.len(),
            current_class_counts,
        );
    }

    let metric = match context.algorithm {
        DecisionTreeAlgorithm::Id3 => MultiwayMetric::InformationGain,
        DecisionTreeAlgorithm::C45 => MultiwayMetric::GainRatio,
        _ => unreachable!("multiway builder only supports id3/c45"),
    };
    let scoring = SplitScoringContext {
        table: context.table,
        class_indices: context.class_indices,
        num_classes: context.class_labels.len(),
        criterion: context.criterion,
        min_samples_leaf: context.options.min_samples_leaf,
    };
    let feature_indices = candidate_feature_indices(
        context.table.binned_feature_count(),
        context.options.max_features,
        node_seed(context.options.random_seed, depth, rows, 0xC1A5_5EEDu64),
    );
    let best_split = if context.parallelism.enabled() {
        feature_indices
            .into_par_iter()
            .filter_map(|feature_index| {
                score_multiway_split_choice(&scoring, feature_index, rows, metric)
            })
            .max_by(|left, right| left.score.total_cmp(&right.score))
    } else {
        feature_indices
            .into_iter()
            .filter_map(|feature_index| {
                score_multiway_split_choice(&scoring, feature_index, rows, metric)
            })
            .max_by(|left, right| left.score.total_cmp(&right.score))
    };

    match best_split {
        Some(best_split)
            if context
                .table
                .is_canary_binned_feature(best_split.feature_index) =>
        {
            push_leaf(
                nodes,
                majority_class_index,
                rows.len(),
                current_class_counts,
            )
        }
        Some(best_split) if best_split.score > 0.0 => {
            let impurity =
                classification_impurity(&current_class_counts, rows.len(), context.criterion);
            let branch_ranges = partition_rows_for_multiway_split(
                context.table,
                best_split.feature_index,
                &best_split.branch_bins,
                best_split.missing_branch_bin,
                rows,
            );
            let mut branch_nodes = Vec::with_capacity(branch_ranges.len());
            let mut missing_child = None;
            for (bin, start, end) in branch_ranges {
                let child =
                    build_multiway_node_in_place(context, nodes, &mut rows[start..end], depth + 1);
                if best_split.missing_branch_bin == Some(bin) {
                    missing_child = Some(child);
                }
                branch_nodes.push((bin, child));
            }

            push_node(
                nodes,
                TreeNode::MultiwaySplit {
                    feature_index: best_split.feature_index,
                    fallback_class_index: majority_class_index,
                    branches: branch_nodes,
                    missing_child,
                    sample_count: rows.len(),
                    impurity,
                    gain: best_split.score,
                    class_counts: current_class_counts,
                },
            )
        }
        _ => push_leaf(
            nodes,
            majority_class_index,
            rows.len(),
            current_class_counts,
        ),
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

fn class_counts(rows: &[usize], class_indices: &[usize], num_classes: usize) -> Vec<usize> {
    rows.iter()
        .fold(vec![0usize; num_classes], |mut counts, row_idx| {
            counts[class_indices[*row_idx]] += 1;
            counts
        })
}

fn majority_class(rows: &[usize], class_indices: &[usize], num_classes: usize) -> usize {
    majority_class_from_counts(&class_counts(rows, class_indices, num_classes))
}

fn majority_class_from_counts(counts: &[usize]) -> usize {
    counts
        .iter()
        .copied()
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

fn push_leaf(
    nodes: &mut Vec<TreeNode>,
    class_index: usize,
    sample_count: usize,
    class_counts: Vec<usize>,
) -> usize {
    push_node(
        nodes,
        TreeNode::Leaf {
            class_index,
            sample_count,
            class_counts,
        },
    )
}

fn push_node(nodes: &mut Vec<TreeNode>, node: TreeNode) -> usize {
    nodes.push(node);
    nodes.len() - 1
}

#[cfg(test)]
mod tests;
