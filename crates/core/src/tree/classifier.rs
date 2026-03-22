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
use crate::sampling::sample_feature_subset;
use crate::{Criterion, FeaturePreprocessing, Parallelism, capture_feature_preprocessing};
use forestfire_data::TableAccess;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

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

#[derive(Debug, Clone, Copy)]
pub(crate) struct ObliviousSplit {
    pub(crate) feature_index: usize,
    pub(crate) threshold_bin: u16,
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
        sample_count: usize,
        impurity: f64,
        gain: f64,
        class_counts: Vec<usize>,
    },
    BinarySplit {
        feature_index: usize,
        threshold_bin: u16,
        left_child: usize,
        right_child: usize,
        sample_count: usize,
        impurity: f64,
        gain: f64,
        class_counts: Vec<usize>,
    },
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
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

#[derive(Debug, Clone, Copy)]
struct BinarySplitChoice {
    feature_index: usize,
    score: f64,
    threshold_bin: u16,
}

#[derive(Debug, Clone)]
struct MultiwaySplitChoice {
    feature_index: usize,
    score: f64,
    branch_bins: Vec<u16>,
}

#[derive(Debug, Clone)]
enum ClassificationFeatureHistogram {
    Binary {
        false_counts: Vec<usize>,
        true_counts: Vec<usize>,
        false_size: usize,
        true_size: usize,
    },
    Numeric {
        bin_class_counts: Vec<Vec<usize>>,
        observed_bins: Vec<usize>,
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
            // The standard binary builders partition one shared row-index buffer
            // in place. That avoids repeatedly allocating left/right row vectors
            // during recursive growth.
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
            // The multiway path now uses the same in-place row-buffer idea even
            // though its branching factor is larger than the binary trainers.
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
    /// Which learner family produced this tree.
    pub fn algorithm(&self) -> DecisionTreeAlgorithm {
        self.algorithm
    }

    /// Split criterion used during training.
    pub fn criterion(&self) -> Criterion {
        self.criterion
    }

    /// Predict one label per row from a preprocessed table.
    pub fn predict_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        (0..table.n_rows())
            .map(|row_idx| self.predict_row(table, row_idx))
            .collect()
    }

    /// Return one class-probability vector per row.
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
                            ..
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
                            ..
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
                            class_counts,
                            ..
                        } => {
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
                            left_child,
                            right_child,
                            ..
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
                rows,
            );
            let mut branch_nodes = Vec::with_capacity(branch_ranges.len());
            for (bin, start, end) in branch_ranges {
                let child =
                    build_multiway_node_in_place(context, nodes, &mut rows[start..end], depth + 1);
                branch_nodes.push((bin, child));
            }

            push_node(
                nodes,
                TreeNode::MultiwaySplit {
                    feature_index: best_split.feature_index,
                    fallback_class_index: majority_class_index,
                    branches: branch_nodes,
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

fn normalized_class_probabilities(class_counts: &[usize]) -> Vec<f64> {
    let total = class_counts.iter().sum::<usize>();
    if total == 0 {
        return vec![0.0; class_counts.len()];
    }

    class_counts
        .iter()
        .map(|count| *count as f64 / total as f64)
        .collect()
}

fn standard_node_depths(nodes: &[TreeNode], root: usize) -> Vec<usize> {
    let mut depths = vec![0; nodes.len()];
    populate_depths(nodes, root, 0, &mut depths);
    depths
}

fn populate_depths(nodes: &[TreeNode], node_id: usize, depth: usize, depths: &mut [usize]) {
    depths[node_id] = depth;
    match &nodes[node_id] {
        TreeNode::Leaf { .. } => {}
        TreeNode::BinarySplit {
            left_child,
            right_child,
            ..
        } => {
            populate_depths(nodes, *left_child, depth + 1, depths);
            populate_depths(nodes, *right_child, depth + 1, depths);
        }
        TreeNode::MultiwaySplit { branches, .. } => {
            for (_, child) in branches {
                populate_depths(nodes, *child, depth + 1, depths);
            }
        }
    }
}

fn binary_split_ir(
    feature_index: usize,
    threshold_bin: u16,
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

#[allow(dead_code)]
fn build_node(
    context: &BuildContext<'_>,
    nodes: &mut Vec<TreeNode>,
    rows: &[usize],
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
                score_split(&scoring, feature_index, rows, context.algorithm)
            })
            .max_by(|left, right| split_score(left).total_cmp(&split_score(right)))
    } else {
        feature_indices
            .into_iter()
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
            push_leaf(
                nodes,
                majority_class_index,
                rows.len(),
                current_class_counts,
            )
        }
        Some(SplitCandidate::Multiway {
            feature_index,
            score,
            branches,
        }) if score > 0.0 => {
            let impurity =
                classification_impurity(&current_class_counts, rows.len(), context.criterion);
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
                    sample_count: rows.len(),
                    impurity,
                    gain: score,
                    class_counts: current_class_counts,
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
            let impurity =
                classification_impurity(&current_class_counts, rows.len(), context.criterion);
            let left_child = build_node(context, nodes, &left_rows, depth + 1);
            let right_child = build_node(context, nodes, &right_rows, depth + 1);

            push_node(
                nodes,
                TreeNode::BinarySplit {
                    feature_index,
                    threshold_bin,
                    left_child,
                    right_child,
                    sample_count: rows.len(),
                    impurity,
                    gain: score,
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

struct SplitScoringContext<'a> {
    table: &'a dyn TableAccess,
    class_indices: &'a [usize],
    num_classes: usize,
    criterion: Criterion,
    min_samples_leaf: usize,
}

fn build_classification_node_histograms(
    table: &dyn TableAccess,
    class_indices: &[usize],
    rows: &[usize],
    num_classes: usize,
) -> Vec<ClassificationFeatureHistogram> {
    (0..table.binned_feature_count())
        .map(|feature_index| {
            if table.is_binary_binned_feature(feature_index) {
                let mut false_counts = vec![0usize; num_classes];
                let mut true_counts = vec![0usize; num_classes];
                let mut false_size = 0usize;
                let mut true_size = 0usize;
                for row_idx in rows {
                    let class_index = class_indices[*row_idx];
                    if !table
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
                ClassificationFeatureHistogram::Binary {
                    false_counts,
                    true_counts,
                    false_size,
                    true_size,
                }
            } else {
                let bin_cap = table.numeric_bin_cap();
                let mut bin_class_counts = vec![vec![0usize; num_classes]; bin_cap];
                let mut observed_bins = vec![false; bin_cap];
                for row_idx in rows {
                    let bin = table.binned_value(feature_index, *row_idx) as usize;
                    bin_class_counts[bin][class_indices[*row_idx]] += 1;
                    observed_bins[bin] = true;
                }
                ClassificationFeatureHistogram::Numeric {
                    bin_class_counts,
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

fn subtract_classification_node_histograms(
    parent: &[ClassificationFeatureHistogram],
    child: &[ClassificationFeatureHistogram],
) -> Vec<ClassificationFeatureHistogram> {
    parent
        .iter()
        .zip(child.iter())
        .map(
            |(parent_hist, child_hist)| match (parent_hist, child_hist) {
                (
                    ClassificationFeatureHistogram::Binary {
                        false_counts: parent_false_counts,
                        true_counts: parent_true_counts,
                        false_size: parent_false_size,
                        true_size: parent_true_size,
                    },
                    ClassificationFeatureHistogram::Binary {
                        false_counts: child_false_counts,
                        true_counts: child_true_counts,
                        false_size: child_false_size,
                        true_size: child_true_size,
                    },
                ) => ClassificationFeatureHistogram::Binary {
                    false_counts: parent_false_counts
                        .iter()
                        .zip(child_false_counts.iter())
                        .map(|(parent, child)| parent - child)
                        .collect(),
                    true_counts: parent_true_counts
                        .iter()
                        .zip(child_true_counts.iter())
                        .map(|(parent, child)| parent - child)
                        .collect(),
                    false_size: parent_false_size - child_false_size,
                    true_size: parent_true_size - child_true_size,
                },
                (
                    ClassificationFeatureHistogram::Numeric {
                        bin_class_counts: parent_bin_class_counts,
                        ..
                    },
                    ClassificationFeatureHistogram::Numeric {
                        bin_class_counts: child_bin_class_counts,
                        ..
                    },
                ) => {
                    let bin_class_counts = parent_bin_class_counts
                        .iter()
                        .zip(child_bin_class_counts.iter())
                        .map(|(parent_counts, child_counts)| {
                            parent_counts
                                .iter()
                                .zip(child_counts.iter())
                                .map(|(parent, child)| parent - child)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let observed_bins = bin_class_counts
                        .iter()
                        .enumerate()
                        .filter_map(|(bin, counts)| {
                            counts.iter().any(|count| *count > 0).then_some(bin)
                        })
                        .collect::<Vec<_>>();
                    ClassificationFeatureHistogram::Numeric {
                        bin_class_counts,
                        observed_bins,
                    }
                }
                _ => unreachable!("histogram shapes must match"),
            },
        )
        .collect()
}

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

fn train_oblivious_structure(
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

#[derive(Debug, Clone, Copy)]
struct ObliviousSplitCandidate {
    feature_index: usize,
    threshold_bin: u16,
    score: f64,
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

#[allow(dead_code)]
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

fn score_multiway_split_choice(
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

#[allow(dead_code)]
fn split_score(candidate: &SplitCandidate) -> f64 {
    match candidate {
        SplitCandidate::Multiway { score, .. } | SplitCandidate::Binary { score, .. } => *score,
    }
}

#[allow(dead_code)]
fn score_binary_split_choice(
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

fn score_binary_split_choice_from_hist(
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

fn partition_rows_for_multiway_split(
    table: &dyn TableAccess,
    feature_index: usize,
    branch_bins: &[u16],
    rows: &mut [usize],
) -> Vec<(u16, usize, usize)> {
    let mut scratch = vec![0usize; rows.len()];
    let mut counts = vec![0usize; branch_bins.len()];

    for row_idx in rows.iter().copied() {
        let bin = if table.is_binary_binned_feature(feature_index) {
            if table
                .binned_boolean_value(feature_index, row_idx)
                .expect("binary feature must expose boolean values")
            {
                1
            } else {
                0
            }
        } else {
            table.binned_value(feature_index, row_idx)
        };
        let branch_index = branch_bins
            .binary_search(&bin)
            .expect("branch bins must cover all observed bins");
        counts[branch_index] += 1;
    }

    let mut offsets = Vec::with_capacity(branch_bins.len());
    let mut next = 0usize;
    for count in &counts {
        offsets.push(next);
        next += *count;
    }
    let mut write_positions = offsets.clone();
    for row_idx in rows.iter().copied() {
        let bin = if table.is_binary_binned_feature(feature_index) {
            if table
                .binned_boolean_value(feature_index, row_idx)
                .expect("binary feature must expose boolean values")
            {
                1
            } else {
                0
            }
        } else {
            table.binned_value(feature_index, row_idx)
        };
        let branch_index = branch_bins
            .binary_search(&bin)
            .expect("branch bins must cover all observed bins");
        let write_index = write_positions[branch_index];
        scratch[write_index] = row_idx;
        write_positions[branch_index] += 1;
    }
    rows.copy_from_slice(&scratch);

    branch_bins
        .iter()
        .copied()
        .zip(offsets)
        .zip(counts)
        .map(|((bin, start), count)| (bin, start, start + count))
        .collect()
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

#[allow(dead_code)]
fn split_feature_index(candidate: &SplitCandidate) -> usize {
    match candidate {
        SplitCandidate::Multiway { feature_index, .. }
        | SplitCandidate::Binary { feature_index, .. } => *feature_index,
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

#[derive(Debug, Clone, Copy)]
enum MultiwayMetric {
    InformationGain,
    GainRatio,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeaturePreprocessing, Model, NumericBinBoundary};
    use forestfire_data::{DenseTable, NumericBins};

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
        DenseTable::with_options(
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
            NumericBins::Fixed(8),
        )
        .unwrap()
    }

    fn canary_target_table() -> DenseTable {
        let x: Vec<Vec<f64>> = (0..8).map(|value| vec![value as f64]).collect();
        let probe =
            DenseTable::with_options(x.clone(), vec![0.0; 8], 1, NumericBins::Auto).unwrap();
        let canary_index = probe.n_features();
        let mut observed_bins = (0..probe.n_rows())
            .map(|row_idx| probe.binned_value(canary_index, row_idx))
            .collect::<Vec<_>>();
        observed_bins.sort_unstable();
        observed_bins.dedup();
        let threshold = observed_bins[observed_bins.len() / 2];
        let y = (0..probe.n_rows())
            .map(|row_idx| {
                if probe.binned_value(canary_index, row_idx) >= threshold {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        DenseTable::with_options(x, y, 1, NumericBins::Auto).unwrap()
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
    fn randomized_fits_basic_boolean_pattern() {
        let table = and_table();
        let model = train_randomized(&table).unwrap();

        assert_eq!(model.algorithm(), DecisionTreeAlgorithm::Randomized);
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

    #[test]
    fn manually_built_classifier_models_serialize_for_each_tree_type() {
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
            },
        ];
        let options = DecisionTreeOptions::default();
        let class_labels = vec![10.0, 20.0];

        let id3 = Model::DecisionTreeClassifier(DecisionTreeClassifier {
            algorithm: DecisionTreeAlgorithm::Id3,
            criterion: Criterion::Entropy,
            class_labels: class_labels.clone(),
            structure: TreeStructure::Standard {
                nodes: vec![
                    TreeNode::Leaf {
                        class_index: 0,
                        sample_count: 3,
                        class_counts: vec![3, 0],
                    },
                    TreeNode::Leaf {
                        class_index: 1,
                        sample_count: 2,
                        class_counts: vec![0, 2],
                    },
                    TreeNode::MultiwaySplit {
                        feature_index: 1,
                        fallback_class_index: 0,
                        branches: vec![(0, 0), (127, 1)],
                        sample_count: 5,
                        impurity: 0.48,
                        gain: 0.24,
                        class_counts: vec![3, 2],
                    },
                ],
                root: 2,
            },
            options,
            num_features: 2,
            feature_preprocessing: preprocessing.clone(),
            training_canaries: 0,
        });
        let c45 = Model::DecisionTreeClassifier(DecisionTreeClassifier {
            algorithm: DecisionTreeAlgorithm::C45,
            criterion: Criterion::Entropy,
            class_labels: class_labels.clone(),
            structure: TreeStructure::Standard {
                nodes: vec![
                    TreeNode::Leaf {
                        class_index: 0,
                        sample_count: 3,
                        class_counts: vec![3, 0],
                    },
                    TreeNode::Leaf {
                        class_index: 1,
                        sample_count: 2,
                        class_counts: vec![0, 2],
                    },
                    TreeNode::MultiwaySplit {
                        feature_index: 1,
                        fallback_class_index: 0,
                        branches: vec![(0, 0), (127, 1)],
                        sample_count: 5,
                        impurity: 0.48,
                        gain: 0.24,
                        class_counts: vec![3, 2],
                    },
                ],
                root: 2,
            },
            options,
            num_features: 2,
            feature_preprocessing: preprocessing.clone(),
            training_canaries: 0,
        });
        let cart = Model::DecisionTreeClassifier(DecisionTreeClassifier {
            algorithm: DecisionTreeAlgorithm::Cart,
            criterion: Criterion::Gini,
            class_labels: class_labels.clone(),
            structure: TreeStructure::Standard {
                nodes: vec![
                    TreeNode::Leaf {
                        class_index: 0,
                        sample_count: 3,
                        class_counts: vec![3, 0],
                    },
                    TreeNode::Leaf {
                        class_index: 1,
                        sample_count: 2,
                        class_counts: vec![0, 2],
                    },
                    TreeNode::BinarySplit {
                        feature_index: 0,
                        threshold_bin: 0,
                        left_child: 0,
                        right_child: 1,
                        sample_count: 5,
                        impurity: 0.48,
                        gain: 0.24,
                        class_counts: vec![3, 2],
                    },
                ],
                root: 2,
            },
            options,
            num_features: 2,
            feature_preprocessing: preprocessing.clone(),
            training_canaries: 0,
        });
        let randomized = Model::DecisionTreeClassifier(DecisionTreeClassifier {
            algorithm: DecisionTreeAlgorithm::Randomized,
            criterion: Criterion::Entropy,
            class_labels: class_labels.clone(),
            structure: TreeStructure::Standard {
                nodes: vec![
                    TreeNode::Leaf {
                        class_index: 0,
                        sample_count: 3,
                        class_counts: vec![3, 0],
                    },
                    TreeNode::Leaf {
                        class_index: 1,
                        sample_count: 2,
                        class_counts: vec![0, 2],
                    },
                    TreeNode::BinarySplit {
                        feature_index: 0,
                        threshold_bin: 0,
                        left_child: 0,
                        right_child: 1,
                        sample_count: 5,
                        impurity: 0.48,
                        gain: 0.2,
                        class_counts: vec![3, 2],
                    },
                ],
                root: 2,
            },
            options,
            num_features: 2,
            feature_preprocessing: preprocessing.clone(),
            training_canaries: 0,
        });
        let oblivious = Model::DecisionTreeClassifier(DecisionTreeClassifier {
            algorithm: DecisionTreeAlgorithm::Oblivious,
            criterion: Criterion::Gini,
            class_labels,
            structure: TreeStructure::Oblivious {
                splits: vec![ObliviousSplit {
                    feature_index: 0,
                    threshold_bin: 0,
                    sample_count: 4,
                    impurity: 0.5,
                    gain: 0.25,
                }],
                leaf_class_indices: vec![0, 1],
                leaf_sample_counts: vec![2, 2],
                leaf_class_counts: vec![vec![2, 0], vec![0, 2]],
            },
            options,
            num_features: 2,
            feature_preprocessing: preprocessing,
            training_canaries: 0,
        });

        for (tree_type, model) in [
            ("id3", id3),
            ("c45", c45),
            ("cart", cart),
            ("randomized", randomized),
            ("oblivious", oblivious),
        ] {
            let json = model.serialize().unwrap();
            assert!(json.contains(&format!("\"tree_type\":\"{tree_type}\"")));
            assert!(json.contains("\"task\":\"classification\""));
        }
    }

    fn table_targets(table: &dyn TableAccess) -> Vec<f64> {
        (0..table.n_rows())
            .map(|row_idx| table.target_value(row_idx))
            .collect()
    }
}
