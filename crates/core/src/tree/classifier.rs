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
            max_depth: Some(self.options.max_depth),
            min_samples_split: Some(self.options.min_samples_split),
            min_samples_leaf: Some(self.options.min_samples_leaf),
            n_trees: None,
            max_features: self.options.max_features,
            seed: None,
            class_labels: Some(self.class_labels.clone()),
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
    let total_class_counts = class_counts(&all_rows, class_indices, class_labels.len());
    let total_impurity = classification_impurity(&total_class_counts, all_rows.len(), criterion);
    let mut leaves = vec![ObliviousLeafState {
        class_index: majority_class(&all_rows, class_indices, class_labels.len()),
        rows: all_rows,
    }];
    let mut splits = Vec::new();

    for depth in 0..options.max_depth {
        if leaves
            .iter()
            .all(|leaf| leaf.rows.len() < options.min_samples_split)
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
            sample_count: table.n_rows(),
            impurity: total_impurity,
            gain: best_split.score,
        });
    }

    TreeStructure::Oblivious {
        splits,
        leaf_class_indices: leaves.iter().map(|leaf| leaf.class_index).collect(),
        leaf_sample_counts: leaves.iter().map(|leaf| leaf.rows.len()).collect(),
        leaf_class_counts: leaves
            .iter()
            .map(|leaf| class_counts(&leaf.rows, class_indices, class_labels.len()))
            .collect(),
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
        DecisionTreeAlgorithm::Randomized => score_randomized_split(context, feature_index, rows),
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

fn score_randomized_split(
    context: &SplitScoringContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<SplitCandidate> {
    if context.table.is_binary_binned_feature(feature_index) {
        return score_binary_cart_split(context, feature_index, rows);
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
                        bin: 511,
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
                        branches: vec![(0, 0), (511, 1)],
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
                        branches: vec![(0, 0), (511, 1)],
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
