//! Stable intermediate representation for ForestFire models.
//!
//! The IR sits between two worlds:
//!
//! - the semantic in-memory model types used for training and introspection
//! - the lowered runtime structures used by optimized inference
//!
//! It exists so models can be serialized, schema-checked, inspected from other
//! languages, and reconstructed without depending on the exact Rust memory
//! layout of the training structs.

use crate::tree::classifier::{
    DecisionTreeAlgorithm, DecisionTreeClassifier, DecisionTreeOptions,
    ObliviousSplit as ClassifierObliviousSplit, TreeNode as ClassifierTreeNode,
    TreeStructure as ClassifierTreeStructure,
};
use crate::tree::regressor::{
    DecisionTreeRegressor, ObliviousSplit as RegressorObliviousSplit, RegressionNode,
    RegressionTreeAlgorithm, RegressionTreeOptions, RegressionTreeStructure,
};
use crate::{
    Criterion, FeaturePreprocessing, GradientBoostedTrees, InputFeatureKind, Model,
    NumericBinBoundary, RandomForest, Task, TrainAlgorithm, TreeType,
};
use schemars::schema::RootSchema;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

const IR_VERSION: &str = "1.0.0";
const FORMAT_NAME: &str = "forestfire-ir";

/// Top-level model package serialized by the library.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModelPackageIr {
    pub ir_version: String,
    pub format_name: String,
    pub producer: ProducerMetadata,
    pub model: ModelSection,
    pub input_schema: InputSchema,
    pub output_schema: OutputSchema,
    pub inference_options: InferenceOptions,
    pub preprocessing: PreprocessingSection,
    pub postprocessing: PostprocessingSection,
    pub training_metadata: TrainingMetadata,
    pub integrity: IntegritySection,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProducerMetadata {
    pub library: String,
    pub library_version: String,
    pub language: String,
    pub platform: String,
}

/// Structural model description independent of any concrete runtime layout.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModelSection {
    pub algorithm: String,
    pub task: String,
    pub tree_type: String,
    pub representation: String,
    pub num_features: usize,
    pub num_outputs: usize,
    pub supports_missing: bool,
    pub supports_categorical: bool,
    pub is_ensemble: bool,
    pub trees: Vec<TreeDefinition>,
    pub aggregation: Aggregation,
}

/// Concrete tree payload stored in the IR.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "representation", rename_all = "snake_case")]
pub enum TreeDefinition {
    NodeTree {
        tree_id: usize,
        weight: f64,
        root_node_id: usize,
        nodes: Vec<NodeTreeNode>,
    },
    ObliviousLevels {
        tree_id: usize,
        weight: f64,
        depth: usize,
        levels: Vec<ObliviousLevel>,
        leaf_indexing: LeafIndexing,
        leaves: Vec<IndexedLeaf>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NodeTreeNode {
    Leaf {
        node_id: usize,
        depth: usize,
        leaf: LeafPayload,
        stats: NodeStats,
    },
    BinaryBranch {
        node_id: usize,
        depth: usize,
        split: BinarySplit,
        children: BinaryChildren,
        stats: NodeStats,
    },
    MultiwayBranch {
        node_id: usize,
        depth: usize,
        split: MultiwaySplit,
        branches: Vec<MultiwayBranch>,
        unmatched_leaf: LeafPayload,
        stats: NodeStats,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BinaryChildren {
    pub left: usize,
    pub right: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MultiwayBranch {
    pub bin: u16,
    pub child: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "split_type", rename_all = "snake_case")]
pub enum BinarySplit {
    NumericBinThreshold {
        feature_index: usize,
        feature_name: String,
        operator: String,
        threshold_bin: u16,
        threshold_upper_bound: Option<f64>,
        comparison_dtype: String,
    },
    BooleanTest {
        feature_index: usize,
        feature_name: String,
        false_child_semantics: String,
        true_child_semantics: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MultiwaySplit {
    pub split_type: String,
    pub feature_index: usize,
    pub feature_name: String,
    pub comparison_dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ObliviousLevel {
    pub level: usize,
    pub split: ObliviousSplit,
    pub stats: NodeStats,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "split_type", rename_all = "snake_case")]
pub enum ObliviousSplit {
    NumericBinThreshold {
        feature_index: usize,
        feature_name: String,
        operator: String,
        threshold_bin: u16,
        threshold_upper_bound: Option<f64>,
        comparison_dtype: String,
        bit_when_true: u8,
        bit_when_false: u8,
    },
    BooleanTest {
        feature_index: usize,
        feature_name: String,
        bit_when_false: u8,
        bit_when_true: u8,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LeafIndexing {
    pub bit_order: String,
    pub index_formula: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IndexedLeaf {
    pub leaf_index: usize,
    pub leaf: LeafPayload,
    pub stats: NodeStats,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "prediction_kind", rename_all = "snake_case")]
pub enum LeafPayload {
    RegressionValue {
        value: f64,
    },
    ClassIndex {
        class_index: usize,
        class_value: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Aggregation {
    pub kind: String,
    pub tree_weights: Vec<f64>,
    pub normalize_by_weight_sum: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct InputSchema {
    pub feature_count: usize,
    pub features: Vec<InputFeature>,
    pub ordering: String,
    pub input_tensor_layout: String,
    pub accepts_feature_names: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct InputFeature {
    pub index: usize,
    pub name: String,
    pub dtype: String,
    pub logical_type: String,
    pub nullable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OutputSchema {
    pub raw_outputs: Vec<OutputField>,
    pub final_outputs: Vec<OutputField>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class_order: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OutputField {
    pub name: String,
    pub kind: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct InferenceOptions {
    pub numeric_precision: String,
    pub threshold_comparison: String,
    pub nan_policy: String,
    pub bool_encoding: BoolEncoding,
    pub tie_breaking: TieBreaking,
    pub determinism: Determinism,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BoolEncoding {
    pub false_values: Vec<String>,
    pub true_values: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TieBreaking {
    pub classification: String,
    pub argmax: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Determinism {
    pub guaranteed: bool,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PreprocessingSection {
    pub included_in_model: bool,
    pub numeric_binning: NumericBinning,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct NumericBinning {
    pub kind: String,
    pub features: Vec<FeatureBinning>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FeatureBinning {
    Numeric {
        feature_index: usize,
        boundaries: Vec<NumericBinBoundary>,
    },
    Binary {
        feature_index: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PostprocessingSection {
    pub raw_output_kind: String,
    pub steps: Vec<PostprocessingStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum PostprocessingStep {
    Identity,
    MapClassIndexToLabel { labels: Vec<f64> },
}

/// Serialized training metadata reflected back to bindings and docs.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TrainingMetadata {
    pub algorithm: String,
    pub task: String,
    pub tree_type: String,
    pub criterion: String,
    pub canaries: usize,
    pub compute_oob: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_depth: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_samples_split: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_samples_leaf: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_trees: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_features: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oob_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class_labels: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_gradient_fraction: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub other_gradient_fraction: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IntegritySection {
    pub serialization: String,
    pub canonical_json: bool,
    pub compatibility: Compatibility,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Compatibility {
    pub minimum_runtime_version: String,
    pub required_capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct NodeStats {
    pub sample_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub impurity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gain: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class_counts: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variance: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrError {
    UnsupportedIrVersion(String),
    UnsupportedFormatName(String),
    UnsupportedAlgorithm(String),
    UnsupportedTask(String),
    UnsupportedTreeType(String),
    InvalidTreeCount(usize),
    UnsupportedRepresentation(String),
    InvalidFeatureCount { schema: usize, preprocessing: usize },
    MissingClassLabels,
    InvalidLeaf(String),
    InvalidNode(String),
    InvalidPreprocessing(String),
    InvalidInferenceOption(String),
    Json(String),
}

impl Display for IrError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            IrError::UnsupportedIrVersion(version) => {
                write!(f, "Unsupported IR version: {}.", version)
            }
            IrError::UnsupportedFormatName(name) => {
                write!(f, "Unsupported IR format: {}.", name)
            }
            IrError::UnsupportedAlgorithm(algorithm) => {
                write!(f, "Unsupported algorithm: {}.", algorithm)
            }
            IrError::UnsupportedTask(task) => write!(f, "Unsupported task: {}.", task),
            IrError::UnsupportedTreeType(tree_type) => {
                write!(f, "Unsupported tree type: {}.", tree_type)
            }
            IrError::InvalidTreeCount(count) => {
                write!(f, "Expected exactly one tree in the IR, found {}.", count)
            }
            IrError::UnsupportedRepresentation(representation) => {
                write!(f, "Unsupported tree representation: {}.", representation)
            }
            IrError::InvalidFeatureCount {
                schema,
                preprocessing,
            } => write!(
                f,
                "Input schema declares {} features, but preprocessing declares {}.",
                schema, preprocessing
            ),
            IrError::MissingClassLabels => {
                write!(f, "Classification IR requires explicit class labels.")
            }
            IrError::InvalidLeaf(message) => write!(f, "Invalid leaf payload: {}.", message),
            IrError::InvalidNode(message) => write!(f, "Invalid tree node: {}.", message),
            IrError::InvalidPreprocessing(message) => {
                write!(f, "Invalid preprocessing section: {}.", message)
            }
            IrError::InvalidInferenceOption(message) => {
                write!(f, "Invalid inference options: {}.", message)
            }
            IrError::Json(message) => write!(f, "Invalid JSON: {}.", message),
        }
    }
}

impl std::error::Error for IrError {}

impl ModelPackageIr {
    pub fn json_schema() -> RootSchema {
        schema_for!(ModelPackageIr)
    }

    pub fn json_schema_json() -> Result<String, IrError> {
        serde_json::to_string(&Self::json_schema()).map_err(|err| IrError::Json(err.to_string()))
    }

    pub fn json_schema_json_pretty() -> Result<String, IrError> {
        serde_json::to_string_pretty(&Self::json_schema())
            .map_err(|err| IrError::Json(err.to_string()))
    }
}

pub(crate) fn model_to_ir(model: &Model) -> ModelPackageIr {
    let trees = match model {
        Model::RandomForest(forest) => forest
            .trees()
            .iter()
            .map(model_tree_definition)
            .collect::<Vec<_>>(),
        Model::GradientBoostedTrees(boosted) => boosted
            .trees()
            .iter()
            .map(model_tree_definition)
            .collect::<Vec<_>>(),
        _ => vec![model_tree_definition(model)],
    };
    let representation = if let Some(first_tree) = trees.first() {
        match first_tree {
            TreeDefinition::NodeTree { .. } => "node_tree",
            TreeDefinition::ObliviousLevels { .. } => "oblivious_levels",
        }
    } else {
        match model.tree_type() {
            TreeType::Oblivious => "oblivious_levels",
            TreeType::Id3 | TreeType::C45 | TreeType::Cart | TreeType::Randomized => "node_tree",
        }
    };
    let class_labels = model.class_labels();
    let is_ensemble = matches!(
        model,
        Model::RandomForest(_) | Model::GradientBoostedTrees(_)
    );
    let tree_count = trees.len();
    let (aggregation_kind, tree_weights, normalize_by_weight_sum, base_score) = match model {
        Model::RandomForest(_) => (
            match model.task() {
                Task::Regression => "average",
                Task::Classification => "average_class_probabilities",
            },
            vec![1.0; tree_count],
            true,
            None,
        ),
        Model::GradientBoostedTrees(boosted) => (
            match boosted.task() {
                Task::Regression => "sum_tree_outputs",
                Task::Classification => "sum_tree_outputs_then_sigmoid",
            },
            boosted.tree_weights().to_vec(),
            false,
            Some(boosted.base_score()),
        ),
        _ => ("identity_single_tree", vec![1.0; tree_count], true, None),
    };

    ModelPackageIr {
        ir_version: IR_VERSION.to_string(),
        format_name: FORMAT_NAME.to_string(),
        producer: ProducerMetadata {
            library: "forestfire-core".to_string(),
            library_version: env!("CARGO_PKG_VERSION").to_string(),
            language: "rust".to_string(),
            platform: std::env::consts::ARCH.to_string(),
        },
        model: ModelSection {
            algorithm: algorithm_name(model.algorithm()).to_string(),
            task: task_name(model.task()).to_string(),
            tree_type: tree_type_name(model.tree_type()).to_string(),
            representation: representation.to_string(),
            num_features: model.num_features(),
            num_outputs: 1,
            supports_missing: false,
            supports_categorical: false,
            is_ensemble,
            trees,
            aggregation: Aggregation {
                kind: aggregation_kind.to_string(),
                tree_weights,
                normalize_by_weight_sum,
                base_score,
            },
        },
        input_schema: input_schema(model),
        output_schema: output_schema(model, class_labels.clone()),
        inference_options: InferenceOptions {
            numeric_precision: "float64".to_string(),
            threshold_comparison: "leq_left_gt_right".to_string(),
            nan_policy: "not_supported".to_string(),
            bool_encoding: BoolEncoding {
                false_values: vec!["0".to_string(), "false".to_string()],
                true_values: vec!["1".to_string(), "true".to_string()],
            },
            tie_breaking: TieBreaking {
                classification: "lowest_class_index".to_string(),
                argmax: "first_max_index".to_string(),
            },
            determinism: Determinism {
                guaranteed: true,
                notes: "Inference is deterministic when the serialized preprocessing artifacts are applied before split evaluation."
                    .to_string(),
            },
        },
        preprocessing: preprocessing(model),
        postprocessing: postprocessing(model, class_labels),
        training_metadata: model.training_metadata(),
        integrity: IntegritySection {
            serialization: "json".to_string(),
            canonical_json: true,
            compatibility: Compatibility {
                minimum_runtime_version: IR_VERSION.to_string(),
                required_capabilities: required_capabilities(model, representation),
            },
        },
    }
}

pub(crate) fn model_from_ir(ir: ModelPackageIr) -> Result<Model, IrError> {
    validate_ir_header(&ir)?;
    validate_inference_options(&ir.inference_options)?;

    let algorithm = parse_algorithm(&ir.model.algorithm)?;
    let task = parse_task(&ir.model.task)?;
    let tree_type = parse_tree_type(&ir.model.tree_type)?;
    let criterion = parse_criterion(&ir.training_metadata.criterion)?;
    let feature_preprocessing = feature_preprocessing_from_ir(&ir)?;
    let num_features = ir.input_schema.feature_count;
    let options = tree_options(&ir.training_metadata);
    let training_canaries = ir.training_metadata.canaries;
    let deserialized_class_labels = classification_labels(&ir).ok();

    if algorithm == TrainAlgorithm::Dt && ir.model.trees.len() != 1 {
        return Err(IrError::InvalidTreeCount(ir.model.trees.len()));
    }

    if algorithm == TrainAlgorithm::Rf {
        let trees = ir
            .model
            .trees
            .into_iter()
            .map(|tree| {
                single_model_from_ir_parts(
                    task,
                    tree_type,
                    criterion,
                    feature_preprocessing.clone(),
                    num_features,
                    options,
                    training_canaries,
                    deserialized_class_labels.clone(),
                    tree,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Model::RandomForest(RandomForest::new(
            task,
            criterion,
            tree_type,
            trees,
            ir.training_metadata.compute_oob,
            ir.training_metadata.oob_score,
            ir.training_metadata
                .max_features
                .unwrap_or(num_features.max(1)),
            ir.training_metadata.seed,
            num_features,
            feature_preprocessing,
        )));
    }

    if algorithm == TrainAlgorithm::Gbm {
        let tree_weights = ir.model.aggregation.tree_weights.clone();
        let base_score = ir.model.aggregation.base_score.unwrap_or(0.0);
        let trees = ir
            .model
            .trees
            .into_iter()
            .map(|tree| {
                single_model_from_ir_parts(
                    task,
                    tree_type,
                    criterion,
                    feature_preprocessing.clone(),
                    num_features,
                    options,
                    training_canaries,
                    deserialized_class_labels.clone(),
                    tree,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Model::GradientBoostedTrees(GradientBoostedTrees::new(
            task,
            tree_type,
            trees,
            tree_weights,
            base_score,
            ir.training_metadata.learning_rate.unwrap_or(0.1),
            ir.training_metadata.bootstrap.unwrap_or(false),
            ir.training_metadata.top_gradient_fraction.unwrap_or(0.2),
            ir.training_metadata.other_gradient_fraction.unwrap_or(0.1),
            ir.training_metadata
                .max_features
                .unwrap_or(num_features.max(1)),
            ir.training_metadata.seed,
            num_features,
            feature_preprocessing,
            deserialized_class_labels,
            training_canaries,
        )));
    }

    let tree = ir
        .model
        .trees
        .into_iter()
        .next()
        .expect("validated single tree");

    single_model_from_ir_parts(
        task,
        tree_type,
        criterion,
        feature_preprocessing,
        num_features,
        options,
        training_canaries,
        deserialized_class_labels,
        tree,
    )
}

#[allow(clippy::too_many_arguments)]
fn single_model_from_ir_parts(
    task: Task,
    tree_type: TreeType,
    criterion: Criterion,
    feature_preprocessing: Vec<FeaturePreprocessing>,
    num_features: usize,
    options: DecisionTreeOptions,
    training_canaries: usize,
    deserialized_class_labels: Option<Vec<f64>>,
    tree: TreeDefinition,
) -> Result<Model, IrError> {
    match (task, tree_type, tree) {
        (
            Task::Classification,
            TreeType::Id3 | TreeType::C45 | TreeType::Cart | TreeType::Randomized,
            TreeDefinition::NodeTree {
                nodes,
                root_node_id,
                ..
            },
        ) => {
            let class_labels = deserialized_class_labels.ok_or(IrError::MissingClassLabels)?;
            let structure = ClassifierTreeStructure::Standard {
                nodes: rebuild_classifier_nodes(nodes, &class_labels)?,
                root: root_node_id,
            };
            Ok(Model::DecisionTreeClassifier(
                DecisionTreeClassifier::from_ir_parts(
                    match tree_type {
                        TreeType::Id3 => DecisionTreeAlgorithm::Id3,
                        TreeType::C45 => DecisionTreeAlgorithm::C45,
                        TreeType::Cart => DecisionTreeAlgorithm::Cart,
                        TreeType::Randomized => DecisionTreeAlgorithm::Randomized,
                        TreeType::Oblivious => unreachable!(),
                    },
                    criterion,
                    class_labels,
                    structure,
                    options,
                    num_features,
                    feature_preprocessing,
                    training_canaries,
                ),
            ))
        }
        (
            Task::Classification,
            TreeType::Oblivious,
            TreeDefinition::ObliviousLevels { levels, leaves, .. },
        ) => {
            let class_labels = deserialized_class_labels.ok_or(IrError::MissingClassLabels)?;
            let leaf_sample_counts = rebuild_leaf_sample_counts(&leaves)?;
            let leaf_class_counts =
                rebuild_classifier_leaf_class_counts(&leaves, class_labels.len())?;
            let structure = ClassifierTreeStructure::Oblivious {
                splits: rebuild_classifier_oblivious_splits(levels)?,
                leaf_class_indices: rebuild_classifier_leaf_indices(leaves, &class_labels)?,
                leaf_sample_counts,
                leaf_class_counts,
            };
            Ok(Model::DecisionTreeClassifier(
                DecisionTreeClassifier::from_ir_parts(
                    DecisionTreeAlgorithm::Oblivious,
                    criterion,
                    class_labels,
                    structure,
                    options,
                    num_features,
                    feature_preprocessing,
                    training_canaries,
                ),
            ))
        }
        (
            Task::Regression,
            TreeType::Cart | TreeType::Randomized,
            TreeDefinition::NodeTree {
                nodes,
                root_node_id,
                ..
            },
        ) => Ok(Model::DecisionTreeRegressor(
            DecisionTreeRegressor::from_ir_parts(
                match tree_type {
                    TreeType::Cart => RegressionTreeAlgorithm::Cart,
                    TreeType::Randomized => RegressionTreeAlgorithm::Randomized,
                    _ => unreachable!(),
                },
                criterion,
                RegressionTreeStructure::Standard {
                    nodes: rebuild_regressor_nodes(nodes)?,
                    root: root_node_id,
                },
                RegressionTreeOptions {
                    max_depth: options.max_depth,
                    min_samples_split: options.min_samples_split,
                    min_samples_leaf: options.min_samples_leaf,
                    max_features: None,
                    random_seed: 0,
                },
                num_features,
                feature_preprocessing,
                training_canaries,
            ),
        )),
        (
            Task::Regression,
            TreeType::Oblivious,
            TreeDefinition::ObliviousLevels { levels, leaves, .. },
        ) => {
            let leaf_sample_counts = rebuild_leaf_sample_counts(&leaves)?;
            let leaf_variances = rebuild_leaf_variances(&leaves)?;
            Ok(Model::DecisionTreeRegressor(
                DecisionTreeRegressor::from_ir_parts(
                    RegressionTreeAlgorithm::Oblivious,
                    criterion,
                    RegressionTreeStructure::Oblivious {
                        splits: rebuild_regressor_oblivious_splits(levels)?,
                        leaf_values: rebuild_regressor_leaf_values(leaves)?,
                        leaf_sample_counts,
                        leaf_variances,
                    },
                    RegressionTreeOptions {
                        max_depth: options.max_depth,
                        min_samples_split: options.min_samples_split,
                        min_samples_leaf: options.min_samples_leaf,
                        max_features: None,
                        random_seed: 0,
                    },
                    num_features,
                    feature_preprocessing,
                    training_canaries,
                ),
            ))
        }
        (_, _, tree) => Err(IrError::UnsupportedRepresentation(match tree {
            TreeDefinition::NodeTree { .. } => "node_tree".to_string(),
            TreeDefinition::ObliviousLevels { .. } => "oblivious_levels".to_string(),
        })),
    }
}

fn validate_ir_header(ir: &ModelPackageIr) -> Result<(), IrError> {
    if ir.ir_version != IR_VERSION {
        return Err(IrError::UnsupportedIrVersion(ir.ir_version.clone()));
    }
    if ir.format_name != FORMAT_NAME {
        return Err(IrError::UnsupportedFormatName(ir.format_name.clone()));
    }
    if ir.model.supports_missing {
        return Err(IrError::InvalidInferenceOption(
            "missing values are not supported in IR v1".to_string(),
        ));
    }
    if ir.model.supports_categorical {
        return Err(IrError::InvalidInferenceOption(
            "categorical features are not supported in IR v1".to_string(),
        ));
    }
    Ok(())
}

fn validate_inference_options(options: &InferenceOptions) -> Result<(), IrError> {
    if options.threshold_comparison != "leq_left_gt_right" {
        return Err(IrError::InvalidInferenceOption(format!(
            "unsupported threshold comparison '{}'",
            options.threshold_comparison
        )));
    }
    if options.nan_policy != "not_supported" {
        return Err(IrError::InvalidInferenceOption(format!(
            "unsupported nan policy '{}'",
            options.nan_policy
        )));
    }
    Ok(())
}

fn parse_algorithm(value: &str) -> Result<TrainAlgorithm, IrError> {
    match value {
        "dt" => Ok(TrainAlgorithm::Dt),
        "rf" => Ok(TrainAlgorithm::Rf),
        "gbm" => Ok(TrainAlgorithm::Gbm),
        _ => Err(IrError::UnsupportedAlgorithm(value.to_string())),
    }
}

fn parse_task(value: &str) -> Result<Task, IrError> {
    match value {
        "regression" => Ok(Task::Regression),
        "classification" => Ok(Task::Classification),
        _ => Err(IrError::UnsupportedTask(value.to_string())),
    }
}

fn parse_tree_type(value: &str) -> Result<TreeType, IrError> {
    match value {
        "id3" => Ok(TreeType::Id3),
        "c45" => Ok(TreeType::C45),
        "cart" => Ok(TreeType::Cart),
        "randomized" => Ok(TreeType::Randomized),
        "oblivious" => Ok(TreeType::Oblivious),
        _ => Err(IrError::UnsupportedTreeType(value.to_string())),
    }
}

fn parse_criterion(value: &str) -> Result<crate::Criterion, IrError> {
    match value {
        "gini" => Ok(crate::Criterion::Gini),
        "entropy" => Ok(crate::Criterion::Entropy),
        "mean" => Ok(crate::Criterion::Mean),
        "median" => Ok(crate::Criterion::Median),
        "second_order" => Ok(crate::Criterion::SecondOrder),
        "auto" => Ok(crate::Criterion::Auto),
        _ => Err(IrError::InvalidInferenceOption(format!(
            "unsupported criterion '{}'",
            value
        ))),
    }
}

fn tree_options(training: &TrainingMetadata) -> DecisionTreeOptions {
    DecisionTreeOptions {
        max_depth: training.max_depth.unwrap_or(8),
        min_samples_split: training.min_samples_split.unwrap_or(2),
        min_samples_leaf: training.min_samples_leaf.unwrap_or(1),
        max_features: None,
        random_seed: 0,
    }
}

fn feature_preprocessing_from_ir(
    ir: &ModelPackageIr,
) -> Result<Vec<FeaturePreprocessing>, IrError> {
    let mut features: Vec<Option<FeaturePreprocessing>> = vec![None; ir.input_schema.feature_count];

    for feature in &ir.preprocessing.numeric_binning.features {
        match feature {
            FeatureBinning::Numeric {
                feature_index,
                boundaries,
            } => {
                let slot = features.get_mut(*feature_index).ok_or_else(|| {
                    IrError::InvalidFeatureCount {
                        schema: ir.input_schema.feature_count,
                        preprocessing: feature_index + 1,
                    }
                })?;
                *slot = Some(FeaturePreprocessing::Numeric {
                    bin_boundaries: boundaries.clone(),
                });
            }
            FeatureBinning::Binary { feature_index } => {
                let slot = features.get_mut(*feature_index).ok_or_else(|| {
                    IrError::InvalidFeatureCount {
                        schema: ir.input_schema.feature_count,
                        preprocessing: feature_index + 1,
                    }
                })?;
                *slot = Some(FeaturePreprocessing::Binary);
            }
        }
    }

    if features.len() != ir.input_schema.feature_count {
        return Err(IrError::InvalidFeatureCount {
            schema: ir.input_schema.feature_count,
            preprocessing: features.len(),
        });
    }

    features
        .into_iter()
        .map(|feature| {
            feature.ok_or_else(|| {
                IrError::InvalidPreprocessing(
                    "every feature must have a preprocessing entry".to_string(),
                )
            })
        })
        .collect()
}

fn classification_labels(ir: &ModelPackageIr) -> Result<Vec<f64>, IrError> {
    ir.output_schema
        .class_order
        .clone()
        .or_else(|| ir.training_metadata.class_labels.clone())
        .ok_or(IrError::MissingClassLabels)
}

fn classifier_class_index(leaf: &LeafPayload, class_labels: &[f64]) -> Result<usize, IrError> {
    match leaf {
        LeafPayload::ClassIndex {
            class_index,
            class_value,
        } => {
            let Some(expected) = class_labels.get(*class_index) else {
                return Err(IrError::InvalidLeaf(format!(
                    "class index {} out of bounds",
                    class_index
                )));
            };
            if expected.total_cmp(class_value).is_ne() {
                return Err(IrError::InvalidLeaf(format!(
                    "class value {} does not match class order entry {}",
                    class_value, expected
                )));
            }
            Ok(*class_index)
        }
        LeafPayload::RegressionValue { .. } => Err(IrError::InvalidLeaf(
            "expected class_index leaf".to_string(),
        )),
    }
}

fn rebuild_classifier_nodes(
    nodes: Vec<NodeTreeNode>,
    class_labels: &[f64],
) -> Result<Vec<ClassifierTreeNode>, IrError> {
    let mut rebuilt = vec![None; nodes.len()];
    for node in nodes {
        match node {
            NodeTreeNode::Leaf {
                node_id,
                leaf,
                stats,
                ..
            } => {
                let class_index = classifier_class_index(&leaf, class_labels)?;
                assign_node(
                    &mut rebuilt,
                    node_id,
                    ClassifierTreeNode::Leaf {
                        class_index,
                        sample_count: stats.sample_count,
                        class_counts: stats
                            .class_counts
                            .unwrap_or_else(|| vec![0; class_labels.len()]),
                    },
                )?;
            }
            NodeTreeNode::BinaryBranch {
                node_id,
                split,
                children,
                stats,
                ..
            } => {
                let (feature_index, threshold_bin) = classifier_binary_split(split)?;
                assign_node(
                    &mut rebuilt,
                    node_id,
                    ClassifierTreeNode::BinarySplit {
                        feature_index,
                        threshold_bin,
                        left_child: children.left,
                        right_child: children.right,
                        sample_count: stats.sample_count,
                        impurity: stats.impurity.unwrap_or(0.0),
                        gain: stats.gain.unwrap_or(0.0),
                        class_counts: stats
                            .class_counts
                            .unwrap_or_else(|| vec![0; class_labels.len()]),
                    },
                )?;
            }
            NodeTreeNode::MultiwayBranch {
                node_id,
                split,
                branches,
                unmatched_leaf,
                stats,
                ..
            } => {
                let fallback_class_index = classifier_class_index(&unmatched_leaf, class_labels)?;
                assign_node(
                    &mut rebuilt,
                    node_id,
                    ClassifierTreeNode::MultiwaySplit {
                        feature_index: split.feature_index,
                        fallback_class_index,
                        branches: branches
                            .into_iter()
                            .map(|branch| (branch.bin, branch.child))
                            .collect(),
                        sample_count: stats.sample_count,
                        impurity: stats.impurity.unwrap_or(0.0),
                        gain: stats.gain.unwrap_or(0.0),
                        class_counts: stats
                            .class_counts
                            .unwrap_or_else(|| vec![0; class_labels.len()]),
                    },
                )?;
            }
        }
    }
    collect_nodes(rebuilt)
}

fn rebuild_regressor_nodes(nodes: Vec<NodeTreeNode>) -> Result<Vec<RegressionNode>, IrError> {
    let mut rebuilt = vec![None; nodes.len()];
    for node in nodes {
        match node {
            NodeTreeNode::Leaf {
                node_id,
                leaf: LeafPayload::RegressionValue { value },
                stats,
                ..
            } => {
                assign_node(
                    &mut rebuilt,
                    node_id,
                    RegressionNode::Leaf {
                        value,
                        sample_count: stats.sample_count,
                        variance: stats.variance,
                    },
                )?;
            }
            NodeTreeNode::Leaf { .. } => {
                return Err(IrError::InvalidLeaf(
                    "regression trees require regression_value leaves".to_string(),
                ));
            }
            NodeTreeNode::BinaryBranch {
                node_id,
                split,
                children,
                stats,
                ..
            } => {
                let (feature_index, threshold_bin) = regressor_binary_split(split)?;
                assign_node(
                    &mut rebuilt,
                    node_id,
                    RegressionNode::BinarySplit {
                        feature_index,
                        threshold_bin,
                        left_child: children.left,
                        right_child: children.right,
                        sample_count: stats.sample_count,
                        impurity: stats.impurity.unwrap_or(0.0),
                        gain: stats.gain.unwrap_or(0.0),
                        variance: stats.variance,
                    },
                )?;
            }
            NodeTreeNode::MultiwayBranch { .. } => {
                return Err(IrError::InvalidNode(
                    "regression trees do not support multiway branches".to_string(),
                ));
            }
        }
    }
    collect_nodes(rebuilt)
}

fn rebuild_classifier_oblivious_splits(
    levels: Vec<ObliviousLevel>,
) -> Result<Vec<ClassifierObliviousSplit>, IrError> {
    let mut rebuilt = Vec::with_capacity(levels.len());
    for level in levels {
        rebuilt.push(match level.split {
            ObliviousSplit::NumericBinThreshold {
                feature_index,
                threshold_bin,
                ..
            } => ClassifierObliviousSplit {
                feature_index,
                threshold_bin,
                sample_count: level.stats.sample_count,
                impurity: level.stats.impurity.unwrap_or(0.0),
                gain: level.stats.gain.unwrap_or(0.0),
            },
            ObliviousSplit::BooleanTest { feature_index, .. } => ClassifierObliviousSplit {
                feature_index,
                threshold_bin: 0,
                sample_count: level.stats.sample_count,
                impurity: level.stats.impurity.unwrap_or(0.0),
                gain: level.stats.gain.unwrap_or(0.0),
            },
        });
    }
    Ok(rebuilt)
}

fn rebuild_regressor_oblivious_splits(
    levels: Vec<ObliviousLevel>,
) -> Result<Vec<RegressorObliviousSplit>, IrError> {
    let mut rebuilt = Vec::with_capacity(levels.len());
    for level in levels {
        rebuilt.push(match level.split {
            ObliviousSplit::NumericBinThreshold {
                feature_index,
                threshold_bin,
                ..
            } => RegressorObliviousSplit {
                feature_index,
                threshold_bin,
                sample_count: level.stats.sample_count,
                impurity: level.stats.impurity.unwrap_or(0.0),
                gain: level.stats.gain.unwrap_or(0.0),
            },
            ObliviousSplit::BooleanTest { feature_index, .. } => RegressorObliviousSplit {
                feature_index,
                threshold_bin: 0,
                sample_count: level.stats.sample_count,
                impurity: level.stats.impurity.unwrap_or(0.0),
                gain: level.stats.gain.unwrap_or(0.0),
            },
        });
    }
    Ok(rebuilt)
}

fn rebuild_classifier_leaf_indices(
    leaves: Vec<IndexedLeaf>,
    class_labels: &[f64],
) -> Result<Vec<usize>, IrError> {
    let mut rebuilt = vec![None; leaves.len()];
    for indexed_leaf in leaves {
        let class_index = classifier_class_index(&indexed_leaf.leaf, class_labels)?;
        assign_node(&mut rebuilt, indexed_leaf.leaf_index, class_index)?;
    }
    collect_nodes(rebuilt)
}

fn rebuild_regressor_leaf_values(leaves: Vec<IndexedLeaf>) -> Result<Vec<f64>, IrError> {
    let mut rebuilt = vec![None; leaves.len()];
    for indexed_leaf in leaves {
        let value = match indexed_leaf.leaf {
            LeafPayload::RegressionValue { value } => value,
            LeafPayload::ClassIndex { .. } => {
                return Err(IrError::InvalidLeaf(
                    "regression oblivious leaves require regression_value".to_string(),
                ));
            }
        };
        assign_node(&mut rebuilt, indexed_leaf.leaf_index, value)?;
    }
    collect_nodes(rebuilt)
}

fn rebuild_leaf_sample_counts(leaves: &[IndexedLeaf]) -> Result<Vec<usize>, IrError> {
    let mut rebuilt = vec![None; leaves.len()];
    for indexed_leaf in leaves {
        assign_node(
            &mut rebuilt,
            indexed_leaf.leaf_index,
            indexed_leaf.stats.sample_count,
        )?;
    }
    collect_nodes(rebuilt)
}

fn rebuild_leaf_variances(leaves: &[IndexedLeaf]) -> Result<Vec<Option<f64>>, IrError> {
    let mut rebuilt = vec![None; leaves.len()];
    for indexed_leaf in leaves {
        assign_node(
            &mut rebuilt,
            indexed_leaf.leaf_index,
            indexed_leaf.stats.variance,
        )?;
    }
    collect_nodes(rebuilt)
}

fn rebuild_classifier_leaf_class_counts(
    leaves: &[IndexedLeaf],
    num_classes: usize,
) -> Result<Vec<Vec<usize>>, IrError> {
    let mut rebuilt = vec![None; leaves.len()];
    for indexed_leaf in leaves {
        assign_node(
            &mut rebuilt,
            indexed_leaf.leaf_index,
            indexed_leaf
                .stats
                .class_counts
                .clone()
                .unwrap_or_else(|| vec![0; num_classes]),
        )?;
    }
    collect_nodes(rebuilt)
}

fn classifier_binary_split(split: BinarySplit) -> Result<(usize, u16), IrError> {
    match split {
        BinarySplit::NumericBinThreshold {
            feature_index,
            threshold_bin,
            ..
        } => Ok((feature_index, threshold_bin)),
        BinarySplit::BooleanTest { feature_index, .. } => Ok((feature_index, 0)),
    }
}

fn regressor_binary_split(split: BinarySplit) -> Result<(usize, u16), IrError> {
    classifier_binary_split(split)
}

fn assign_node<T>(slots: &mut [Option<T>], index: usize, value: T) -> Result<(), IrError> {
    let Some(slot) = slots.get_mut(index) else {
        return Err(IrError::InvalidNode(format!(
            "node index {} is out of bounds",
            index
        )));
    };
    if slot.is_some() {
        return Err(IrError::InvalidNode(format!(
            "duplicate node index {}",
            index
        )));
    }
    *slot = Some(value);
    Ok(())
}

fn collect_nodes<T>(slots: Vec<Option<T>>) -> Result<Vec<T>, IrError> {
    slots
        .into_iter()
        .enumerate()
        .map(|(index, slot)| {
            slot.ok_or_else(|| IrError::InvalidNode(format!("missing node index {}", index)))
        })
        .collect()
}

fn input_schema(model: &Model) -> InputSchema {
    let features = model
        .feature_preprocessing()
        .iter()
        .enumerate()
        .map(|(feature_index, preprocessing)| {
            let kind = match preprocessing {
                FeaturePreprocessing::Numeric { .. } => InputFeatureKind::Numeric,
                FeaturePreprocessing::Binary => InputFeatureKind::Binary,
            };

            InputFeature {
                index: feature_index,
                name: feature_name(feature_index),
                dtype: match kind {
                    InputFeatureKind::Numeric => "float64".to_string(),
                    InputFeatureKind::Binary => "bool".to_string(),
                },
                logical_type: match kind {
                    InputFeatureKind::Numeric => "numeric".to_string(),
                    InputFeatureKind::Binary => "boolean".to_string(),
                },
                nullable: false,
            }
        })
        .collect();

    InputSchema {
        feature_count: model.num_features(),
        features,
        ordering: "strict_index_order".to_string(),
        input_tensor_layout: "row_major".to_string(),
        accepts_feature_names: false,
    }
}

fn output_schema(model: &Model, class_labels: Option<Vec<f64>>) -> OutputSchema {
    match model.task() {
        Task::Regression => OutputSchema {
            raw_outputs: vec![OutputField {
                name: "value".to_string(),
                kind: "regression_value".to_string(),
                shape: Vec::new(),
                dtype: "float64".to_string(),
            }],
            final_outputs: vec![OutputField {
                name: "prediction".to_string(),
                kind: "value".to_string(),
                shape: Vec::new(),
                dtype: "float64".to_string(),
            }],
            class_order: None,
        },
        Task::Classification => OutputSchema {
            raw_outputs: vec![OutputField {
                name: "class_index".to_string(),
                kind: "class_index".to_string(),
                shape: Vec::new(),
                dtype: "uint64".to_string(),
            }],
            final_outputs: vec![OutputField {
                name: "predicted_class".to_string(),
                kind: "class_label".to_string(),
                shape: Vec::new(),
                dtype: "float64".to_string(),
            }],
            class_order: class_labels,
        },
    }
}

fn preprocessing(model: &Model) -> PreprocessingSection {
    let features = model
        .feature_preprocessing()
        .iter()
        .enumerate()
        .map(|(feature_index, preprocessing)| match preprocessing {
            FeaturePreprocessing::Numeric { bin_boundaries } => FeatureBinning::Numeric {
                feature_index,
                boundaries: bin_boundaries.clone(),
            },
            FeaturePreprocessing::Binary => FeatureBinning::Binary { feature_index },
        })
        .collect();

    PreprocessingSection {
        included_in_model: true,
        numeric_binning: NumericBinning {
            kind: "rank_bin_128".to_string(),
            features,
        },
        notes: "Numeric features use serialized training-time rank bins. Binary features are serialized as booleans. Missing values and categorical encodings are not implemented in IR v1."
            .to_string(),
    }
}

fn postprocessing(model: &Model, class_labels: Option<Vec<f64>>) -> PostprocessingSection {
    match model.task() {
        Task::Regression => PostprocessingSection {
            raw_output_kind: "regression_value".to_string(),
            steps: vec![PostprocessingStep::Identity],
        },
        Task::Classification => PostprocessingSection {
            raw_output_kind: "class_index".to_string(),
            steps: vec![PostprocessingStep::MapClassIndexToLabel {
                labels: class_labels.expect("classification IR requires class labels"),
            }],
        },
    }
}

fn required_capabilities(model: &Model, representation: &str) -> Vec<String> {
    let mut capabilities = vec![
        representation.to_string(),
        "training_rank_bin_128".to_string(),
    ];
    match model.tree_type() {
        TreeType::Id3 | TreeType::C45 => {
            capabilities.push("binned_multiway_splits".to_string());
        }
        TreeType::Cart | TreeType::Randomized | TreeType::Oblivious => {
            capabilities.push("numeric_bin_threshold_splits".to_string());
        }
    }
    if model
        .feature_preprocessing()
        .iter()
        .any(|feature| matches!(feature, FeaturePreprocessing::Binary))
    {
        capabilities.push("boolean_features".to_string());
    }
    match model.task() {
        Task::Regression => capabilities.push("regression_value_leaves".to_string()),
        Task::Classification => capabilities.push("class_index_leaves".to_string()),
    }
    capabilities
}

pub(crate) fn algorithm_name(algorithm: TrainAlgorithm) -> &'static str {
    match algorithm {
        TrainAlgorithm::Dt => "dt",
        TrainAlgorithm::Rf => "rf",
        TrainAlgorithm::Gbm => "gbm",
    }
}

fn model_tree_definition(model: &Model) -> TreeDefinition {
    match model {
        Model::DecisionTreeClassifier(classifier) => classifier.to_ir_tree(),
        Model::DecisionTreeRegressor(regressor) => regressor.to_ir_tree(),
        Model::RandomForest(_) | Model::GradientBoostedTrees(_) => {
            unreachable!("ensemble IR expands into member trees")
        }
    }
}

pub(crate) fn criterion_name(criterion: crate::Criterion) -> &'static str {
    match criterion {
        crate::Criterion::Auto => "auto",
        crate::Criterion::Gini => "gini",
        crate::Criterion::Entropy => "entropy",
        crate::Criterion::Mean => "mean",
        crate::Criterion::Median => "median",
        crate::Criterion::SecondOrder => "second_order",
    }
}

pub(crate) fn task_name(task: Task) -> &'static str {
    match task {
        Task::Regression => "regression",
        Task::Classification => "classification",
    }
}

pub(crate) fn tree_type_name(tree_type: TreeType) -> &'static str {
    match tree_type {
        TreeType::Id3 => "id3",
        TreeType::C45 => "c45",
        TreeType::Cart => "cart",
        TreeType::Randomized => "randomized",
        TreeType::Oblivious => "oblivious",
    }
}

pub(crate) fn feature_name(feature_index: usize) -> String {
    format!("f{}", feature_index)
}

pub(crate) fn threshold_upper_bound(
    preprocessing: &[FeaturePreprocessing],
    feature_index: usize,
    threshold_bin: u16,
) -> Option<f64> {
    match preprocessing.get(feature_index)? {
        FeaturePreprocessing::Numeric { bin_boundaries } => bin_boundaries
            .iter()
            .find(|boundary| boundary.bin == threshold_bin)
            .map(|boundary| boundary.upper_bound),
        FeaturePreprocessing::Binary => None,
    }
}
