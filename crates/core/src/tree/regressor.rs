use crate::ir::{
    BinaryChildren, BinarySplit, IndexedLeaf, LeafIndexing, LeafPayload, NodeStats, NodeTreeNode,
    ObliviousLevel, ObliviousSplit as IrObliviousSplit, TrainingMetadata, TreeDefinition,
    criterion_name, feature_name, threshold_upper_bound,
};
use crate::sampling::sample_feature_subset;
use crate::{Criterion, FeaturePreprocessing, Parallelism, capture_feature_preprocessing};
use forestfire_data::TableAccess;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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

#[derive(Debug, Clone, Copy)]
pub struct RegressionTreeOptions {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
    pub random_seed: u64,
}

impl Default for RegressionTreeOptions {
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
    BinarySplit {
        feature_index: usize,
        threshold_bin: u16,
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
}

#[derive(Debug, Clone)]
struct ObliviousLeafState {
    rows: Vec<usize>,
    value: f64,
    variance: Option<f64>,
    sum: f64,
    sum_sq: f64,
}

#[derive(Debug, Clone, Copy)]
struct ObliviousSplitCandidate {
    feature_index: usize,
    threshold_bin: u16,
    score: f64,
}

#[derive(Debug, Clone, Copy)]
struct BinarySplitChoice {
    feature_index: usize,
    threshold_bin: u16,
    score: f64,
}

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
    let structure = match algorithm {
        RegressionTreeAlgorithm::Cart => {
            let mut nodes = Vec::new();
            let mut all_rows: Vec<usize> = (0..train_set.n_rows()).collect();
            let context = BuildContext {
                table: train_set,
                targets: &targets,
                criterion,
                parallelism,
                options,
                algorithm,
            };
            let root = build_binary_node_in_place(&context, &mut nodes, &mut all_rows, 0);
            RegressionTreeStructure::Standard { nodes, root }
        }
        RegressionTreeAlgorithm::Randomized => {
            let mut nodes = Vec::new();
            let mut all_rows: Vec<usize> = (0..train_set.n_rows()).collect();
            let context = BuildContext {
                table: train_set,
                targets: &targets,
                criterion,
                parallelism,
                options,
                algorithm,
            };
            let root = build_binary_node_in_place(&context, &mut nodes, &mut all_rows, 0);
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
        options,
        num_features: train_set.n_features(),
        feature_preprocessing: capture_feature_preprocessing(train_set),
        training_canaries: train_set.canaries(),
    })
}

impl DecisionTreeRegressor {
    pub fn algorithm(&self) -> RegressionTreeAlgorithm {
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
            RegressionTreeStructure::Standard { nodes, root } => {
                let mut node_index = *root;

                loop {
                    match &nodes[node_index] {
                        RegressionNode::Leaf { value, .. } => return *value,
                        RegressionNode::BinarySplit {
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
                            RegressionNode::BinarySplit {
                                feature_index,
                                threshold_bin,
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
            options,
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
        RegressionNode::Leaf { .. } => {}
        RegressionNode::BinarySplit {
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

fn build_binary_node_in_place(
    context: &BuildContext<'_>,
    nodes: &mut Vec<RegressionNode>,
    rows: &mut [usize],
    depth: usize,
) -> usize {
    let leaf_value = regression_value(rows, context.targets, context.criterion);
    let leaf_variance = variance(rows, context.targets);

    if rows.is_empty()
        || depth >= context.options.max_depth
        || rows.len() < context.options.min_samples_split
        || has_constant_target(rows, context.targets)
    {
        return push_leaf(nodes, leaf_value, rows.len(), leaf_variance);
    }

    let feature_indices = candidate_feature_indices(
        context.table.binned_feature_count(),
        context.options.max_features,
        node_seed(context.options.random_seed, depth, rows, 0xA11C_E5E1u64),
    );
    let best_split = if context.parallelism.enabled() {
        feature_indices
            .into_par_iter()
            .filter_map(|feature_index| score_binary_split_choice(context, feature_index, rows))
            .max_by(|left, right| left.score.total_cmp(&right.score))
    } else {
        feature_indices
            .into_iter()
            .filter_map(|feature_index| score_binary_split_choice(context, feature_index, rows))
            .max_by(|left, right| left.score.total_cmp(&right.score))
    };

    match best_split {
        Some(best_split)
            if context
                .table
                .is_canary_binned_feature(best_split.feature_index) =>
        {
            push_leaf(nodes, leaf_value, rows.len(), leaf_variance)
        }
        Some(best_split) if best_split.score > 0.0 => {
            let impurity = regression_loss(rows, context.targets, context.criterion);
            let left_count = partition_rows_for_binary_split(
                context.table,
                best_split.feature_index,
                best_split.threshold_bin,
                rows,
            );
            let (left_rows, right_rows) = rows.split_at_mut(left_count);
            let left_child = build_binary_node_in_place(context, nodes, left_rows, depth + 1);
            let right_child = build_binary_node_in_place(context, nodes, right_rows, depth + 1);

            push_node(
                nodes,
                RegressionNode::BinarySplit {
                    feature_index: best_split.feature_index,
                    threshold_bin: best_split.threshold_bin,
                    left_child,
                    right_child,
                    sample_count: rows.len(),
                    impurity,
                    gain: best_split.score,
                    variance: leaf_variance,
                },
            )
        }
        _ => push_leaf(nodes, leaf_value, rows.len(), leaf_variance),
    }
}

fn train_oblivious_structure(
    table: &dyn TableAccess,
    targets: &[f64],
    criterion: Criterion,
    parallelism: Parallelism,
    options: RegressionTreeOptions,
) -> RegressionTreeStructure {
    let all_rows: Vec<usize> = (0..table.n_rows()).collect();
    let (root_sum, root_sum_sq) = sum_stats(&all_rows, targets);
    let mut leaves = vec![ObliviousLeafState {
        value: regression_value_from_stats(&all_rows, targets, criterion, root_sum),
        variance: variance_from_stats(all_rows.len(), root_sum, root_sum_sq),
        rows: all_rows,
        sum: root_sum,
        sum_sq: root_sum_sq,
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
                        targets,
                        feature_index,
                        &leaves,
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
            sample_count: table.n_rows(),
            impurity: leaves
                .iter()
                .map(|leaf| leaf_regression_loss(leaf, targets, criterion))
                .sum(),
            gain: best_split.score,
        });
    }

    RegressionTreeStructure::Oblivious {
        splits,
        leaf_values: leaves.iter().map(|leaf| leaf.value).collect(),
        leaf_sample_counts: leaves.iter().map(|leaf| leaf.rows.len()).collect(),
        leaf_variances: leaves.iter().map(|leaf| leaf.variance).collect(),
    }
}

fn score_split(
    table: &dyn TableAccess,
    targets: &[f64],
    feature_index: usize,
    rows: &[usize],
    criterion: Criterion,
    min_samples_leaf: usize,
    algorithm: RegressionTreeAlgorithm,
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
    if matches!(criterion, Criterion::Mean) {
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
            })
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
) -> Option<RegressionSplitCandidate> {
    let candidate_thresholds = rows
        .iter()
        .map(|row_idx| table.binned_value(feature_index, *row_idx))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let threshold_bin =
        choose_random_threshold(&candidate_thresholds, feature_index, rows, 0xA11CE551u64)?;

    let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = rows
        .iter()
        .copied()
        .partition(|row_idx| table.binned_value(feature_index, *row_idx) <= threshold_bin);

    if left_rows.len() < min_samples_leaf || right_rows.len() < min_samples_leaf {
        return None;
    }

    let parent_loss = regression_loss(rows, targets, criterion);
    let score = parent_loss
        - (regression_loss(&left_rows, targets, criterion)
            + regression_loss(&right_rows, targets, criterion));

    Some(RegressionSplitCandidate {
        feature_index,
        threshold_bin,
        score,
    })
}

fn score_oblivious_split(
    table: &dyn TableAccess,
    targets: &[f64],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
    criterion: Criterion,
    min_samples_leaf: usize,
) -> Option<ObliviousSplitCandidate> {
    if table.is_binary_binned_feature(feature_index) {
        if matches!(criterion, Criterion::Mean) {
            if let Some(candidate) = score_binary_oblivious_split_mean_fast(
                table,
                targets,
                feature_index,
                leaves,
                min_samples_leaf,
            ) {
                return Some(candidate);
            }
        }
        return score_binary_oblivious_split(
            table,
            targets,
            feature_index,
            leaves,
            criterion,
            min_samples_leaf,
        );
    }
    if matches!(criterion, Criterion::Mean) {
        if let Some(candidate) = score_numeric_oblivious_split_mean_fast(
            table,
            targets,
            feature_index,
            leaves,
            min_samples_leaf,
        ) {
            return Some(candidate);
        }
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
    table: &dyn TableAccess,
    targets: &[f64],
    leaf: ObliviousLeafState,
    feature_index: usize,
    threshold_bin: u16,
    criterion: Criterion,
) -> [ObliviousLeafState; 2] {
    let fallback_value = leaf.value;
    let mut left_rows = Vec::new();
    let mut right_rows = Vec::new();
    let mut left_sum = 0.0;
    let mut left_sum_sq = 0.0;
    let mut right_sum = 0.0;
    let mut right_sum_sq = 0.0;

    for row_idx in leaf.rows {
        let target = targets[row_idx];
        if table.binned_value(feature_index, row_idx) <= threshold_bin {
            left_rows.push(row_idx);
            left_sum += target;
            left_sum_sq += target * target;
        } else {
            right_rows.push(row_idx);
            right_sum += target;
            right_sum_sq += target * target;
        }
    }

    [
        ObliviousLeafState {
            value: if left_rows.is_empty() {
                fallback_value
            } else {
                regression_value_from_stats(&left_rows, targets, criterion, left_sum)
            },
            variance: variance_from_stats(left_rows.len(), left_sum, left_sum_sq),
            rows: left_rows,
            sum: left_sum,
            sum_sq: left_sum_sq,
        },
        ObliviousLeafState {
            value: if right_rows.is_empty() {
                fallback_value
            } else {
                regression_value_from_stats(&right_rows, targets, criterion, right_sum)
            },
            variance: variance_from_stats(right_rows.len(), right_sum, right_sum_sq),
            rows: right_rows,
            sum: right_sum,
            sum_sq: right_sum_sq,
        },
    ]
}

fn variance(rows: &[usize], targets: &[f64]) -> Option<f64> {
    let (sum, sum_sq) = sum_stats(rows, targets);
    variance_from_stats(rows.len(), sum, sum_sq)
}

fn mean(rows: &[usize], targets: &[f64]) -> f64 {
    if rows.is_empty() {
        0.0
    } else {
        rows.iter().map(|row_idx| targets[*row_idx]).sum::<f64>() / rows.len() as f64
    }
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
    let (sum, _sum_sq) = sum_stats(rows, targets);
    regression_value_from_stats(rows, targets, criterion, sum)
}

fn regression_value_from_stats(
    rows: &[usize],
    targets: &[f64],
    criterion: Criterion,
    sum: f64,
) -> f64 {
    match criterion {
        Criterion::Mean => {
            if rows.is_empty() {
                0.0
            } else {
                sum / rows.len() as f64
            }
        }
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
    table: &dyn TableAccess,
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
    })
}

fn score_binary_split_choice(
    context: &BuildContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<BinarySplitChoice> {
    if matches!(context.criterion, Criterion::Mean) {
        if context.table.is_binary_binned_feature(feature_index) {
            return score_binary_split_choice_mean(context, feature_index, rows);
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
    )
    .map(|candidate| BinarySplitChoice {
        feature_index: candidate.feature_index,
        threshold_bin: candidate.threshold_bin,
        score: candidate.score,
    })
}

fn score_binary_split_choice_mean(
    context: &BuildContext<'_>,
    feature_index: usize,
    rows: &[usize],
) -> Option<BinarySplitChoice> {
    let mut left_count = 0usize;
    let mut left_sum = 0.0;
    let mut left_sum_sq = 0.0;
    let mut total_sum = 0.0;
    let mut total_sum_sq = 0.0;

    for row_idx in rows {
        let target = context.targets[*row_idx];
        total_sum += target;
        total_sum_sq += target * target;
        if !context
            .table
            .binned_boolean_value(feature_index, *row_idx)
            .expect("binary feature must expose boolean values")
        {
            left_count += 1;
            left_sum += target;
            left_sum_sq += target * target;
        }
    }

    let right_count = rows.len() - left_count;
    if left_count < context.options.min_samples_leaf
        || right_count < context.options.min_samples_leaf
    {
        return None;
    }

    let parent_loss = total_sum_sq - (total_sum * total_sum) / rows.len() as f64;
    let right_sum = total_sum - left_sum;
    let right_sum_sq = total_sum_sq - left_sum_sq;
    let left_loss = left_sum_sq - (left_sum * left_sum) / left_count as f64;
    let right_loss = right_sum_sq - (right_sum * right_sum) / right_count as f64;

    Some(BinarySplitChoice {
        feature_index,
        threshold_bin: 0,
        score: parent_loss - (left_loss + right_loss),
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

    let mut bin_count = vec![0usize; bin_cap];
    let mut bin_sum = vec![0.0; bin_cap];
    let mut bin_sum_sq = vec![0.0; bin_cap];
    let mut observed_bins = vec![false; bin_cap];
    let mut total_sum = 0.0;
    let mut total_sum_sq = 0.0;

    for row_idx in rows {
        let bin = table.binned_value(feature_index, *row_idx) as usize;
        if bin >= bin_cap {
            return None;
        }
        let target = targets[*row_idx];
        bin_count[bin] += 1;
        bin_sum[bin] += target;
        bin_sum_sq[bin] += target * target;
        observed_bins[bin] = true;
        total_sum += target;
        total_sum_sq += target * target;
    }

    let observed_bins: Vec<usize> = observed_bins
        .into_iter()
        .enumerate()
        .filter_map(|(bin, seen)| seen.then_some(bin))
        .collect();
    if observed_bins.len() <= 1 {
        return None;
    }

    let parent_loss = total_sum_sq - (total_sum * total_sum) / rows.len() as f64;
    let mut left_count = 0usize;
    let mut left_sum = 0.0;
    let mut left_sum_sq = 0.0;
    let mut best_threshold = None;
    let mut best_score = f64::NEG_INFINITY;

    for &bin in &observed_bins {
        left_count += bin_count[bin];
        left_sum += bin_sum[bin];
        left_sum_sq += bin_sum_sq[bin];
        let right_count = rows.len() - left_count;

        if left_count < min_samples_leaf || right_count < min_samples_leaf {
            continue;
        }

        let right_sum = total_sum - left_sum;
        let right_sum_sq = total_sum_sq - left_sum_sq;
        let left_loss = left_sum_sq - (left_sum * left_sum) / left_count as f64;
        let right_loss = right_sum_sq - (right_sum * right_sum) / right_count as f64;
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

    let mut left_count = 0usize;
    let mut left_sum = 0.0;
    let mut left_sum_sq = 0.0;
    let mut total_sum = 0.0;
    let mut total_sum_sq = 0.0;
    for row_idx in rows {
        let target = targets[*row_idx];
        total_sum += target;
        total_sum_sq += target * target;
        if table.binned_value(feature_index, *row_idx) <= threshold_bin {
            left_count += 1;
            left_sum += target;
            left_sum_sq += target * target;
        }
    }
    let right_count = rows.len() - left_count;
    if left_count < min_samples_leaf || right_count < min_samples_leaf {
        return None;
    }
    let parent_loss = total_sum_sq - (total_sum * total_sum) / rows.len() as f64;
    let right_sum = total_sum - left_sum;
    let right_sum_sq = total_sum_sq - left_sum_sq;
    let left_loss = left_sum_sq - (left_sum * left_sum) / left_count as f64;
    let right_loss = right_sum_sq - (right_sum * right_sum) / right_count as f64;
    let score = parent_loss - (left_loss + right_loss);

    Some(RegressionSplitCandidate {
        feature_index,
        threshold_bin,
        score,
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
    })
}

fn score_numeric_oblivious_split_mean_fast(
    table: &dyn TableAccess,
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

    for leaf in leaves {
        let mut bin_count = vec![0usize; bin_cap];
        let mut bin_sum = vec![0.0; bin_cap];
        let mut bin_sum_sq = vec![0.0; bin_cap];
        let mut observed_bins = vec![false; bin_cap];

        for row_idx in &leaf.rows {
            let bin = table.binned_value(feature_index, *row_idx) as usize;
            if bin >= bin_cap {
                return None;
            }
            let target = targets[*row_idx];
            bin_count[bin] += 1;
            bin_sum[bin] += target;
            bin_sum_sq[bin] += target * target;
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

        let parent_loss = leaf.sum_sq - (leaf.sum * leaf.sum) / leaf.rows.len() as f64;
        let mut left_count = 0usize;
        let mut left_sum = 0.0;
        let mut left_sum_sq = 0.0;

        for &bin in &observed_bins {
            left_count += bin_count[bin];
            left_sum += bin_sum[bin];
            left_sum_sq += bin_sum_sq[bin];
            let right_count = leaf.rows.len() - left_count;

            if left_count < min_samples_leaf || right_count < min_samples_leaf {
                continue;
            }

            let right_sum = leaf.sum - left_sum;
            let right_sum_sq = leaf.sum_sq - left_sum_sq;
            let left_loss = left_sum_sq - (left_sum * left_sum) / left_count as f64;
            let right_loss = right_sum_sq - (right_sum * right_sum) / right_count as f64;
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

fn score_binary_oblivious_split(
    table: &dyn TableAccess,
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

fn score_binary_oblivious_split_mean_fast(
    table: &dyn TableAccess,
    targets: &[f64],
    feature_index: usize,
    leaves: &[ObliviousLeafState],
    min_samples_leaf: usize,
) -> Option<ObliviousSplitCandidate> {
    let mut score = 0.0;
    let mut found_valid = false;

    for leaf in leaves {
        let mut left_count = 0usize;
        let mut left_sum = 0.0;
        let mut left_sum_sq = 0.0;

        for row_idx in &leaf.rows {
            if !table
                .binned_boolean_value(feature_index, *row_idx)
                .expect("binary feature must expose boolean values")
            {
                let target = targets[*row_idx];
                left_count += 1;
                left_sum += target;
                left_sum_sq += target * target;
            }
        }

        let right_count = leaf.rows.len() - left_count;
        if left_count < min_samples_leaf || right_count < min_samples_leaf {
            continue;
        }

        found_valid = true;
        let parent_loss = leaf.sum_sq - (leaf.sum * leaf.sum) / leaf.rows.len() as f64;
        let right_sum = leaf.sum - left_sum;
        let right_sum_sq = leaf.sum_sq - left_sum_sq;
        let left_loss = left_sum_sq - (left_sum * left_sum) / left_count as f64;
        let right_loss = right_sum_sq - (right_sum * right_sum) / right_count as f64;
        score += parent_loss - (left_loss + right_loss);
    }

    (found_valid && score > 0.0).then_some(ObliviousSplitCandidate {
        feature_index,
        threshold_bin: 0,
        score,
    })
}

fn sum_stats(rows: &[usize], targets: &[f64]) -> (f64, f64) {
    rows.iter().fold((0.0, 0.0), |(sum, sum_sq), row_idx| {
        let value = targets[*row_idx];
        (sum + value, sum_sq + value * value)
    })
}

fn variance_from_stats(count: usize, sum: f64, sum_sq: f64) -> Option<f64> {
    if count == 0 {
        None
    } else {
        Some((sum_sq / count as f64) - (sum / count as f64).powi(2))
    }
}

fn leaf_regression_loss(leaf: &ObliviousLeafState, targets: &[f64], criterion: Criterion) -> f64 {
    match criterion {
        Criterion::Mean => leaf.sum_sq - (leaf.sum * leaf.sum) / leaf.rows.len() as f64,
        Criterion::Median => regression_loss(&leaf.rows, targets, criterion),
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
    use crate::{FeaturePreprocessing, Model, NumericBinBoundary};
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
            options,
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
            options,
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

    fn table_targets(table: &dyn TableAccess) -> Vec<f64> {
        (0..table.n_rows())
            .map(|row_idx| table.target_value(row_idx))
            .collect()
    }
}
