use crate::{
    FeaturePreprocessing, InputFeatureKind, Model, NumericBinBoundary, Task, TrainAlgorithm,
    TreeType,
};
use serde::{Deserialize, Serialize};

const IR_VERSION: &str = "1.0.0";
const FORMAT_NAME: &str = "forestfire-ir";

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProducerMetadata {
    pub library: String,
    pub library_version: String,
    pub language: String,
    pub platform: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NodeTreeNode {
    Leaf {
        node_id: usize,
        depth: usize,
        leaf: LeafPayload,
    },
    BinaryBranch {
        node_id: usize,
        depth: usize,
        split: BinarySplit,
        children: BinaryChildren,
    },
    MultiwayBranch {
        node_id: usize,
        depth: usize,
        split: MultiwaySplit,
        branches: Vec<MultiwayBranch>,
        unmatched_leaf: LeafPayload,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryChildren {
    pub left: usize,
    pub right: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiwayBranch {
    pub bin: u16,
    pub child: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiwaySplit {
    pub split_type: String,
    pub feature_index: usize,
    pub feature_name: String,
    pub comparison_dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObliviousLevel {
    pub level: usize,
    pub split: ObliviousSplit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeafIndexing {
    pub bit_order: String,
    pub index_formula: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedLeaf {
    pub leaf_index: usize,
    pub leaf: LeafPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aggregation {
    pub kind: String,
    pub tree_weights: Vec<f64>,
    pub normalize_by_weight_sum: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSchema {
    pub feature_count: usize,
    pub features: Vec<InputFeature>,
    pub ordering: String,
    pub input_tensor_layout: String,
    pub accepts_feature_names: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputFeature {
    pub index: usize,
    pub name: String,
    pub dtype: String,
    pub logical_type: String,
    pub nullable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSchema {
    pub raw_outputs: Vec<OutputField>,
    pub final_outputs: Vec<OutputField>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class_order: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputField {
    pub name: String,
    pub kind: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOptions {
    pub numeric_precision: String,
    pub threshold_comparison: String,
    pub nan_policy: String,
    pub bool_encoding: BoolEncoding,
    pub tie_breaking: TieBreaking,
    pub determinism: Determinism,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoolEncoding {
    pub false_values: Vec<String>,
    pub true_values: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieBreaking {
    pub classification: String,
    pub argmax: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Determinism {
    pub guaranteed: bool,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingSection {
    pub included_in_model: bool,
    pub numeric_binning: NumericBinning,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericBinning {
    pub kind: String,
    pub features: Vec<FeatureBinning>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostprocessingSection {
    pub raw_output_kind: String,
    pub steps: Vec<PostprocessingStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum PostprocessingStep {
    Identity,
    MapClassIndexToLabel { labels: Vec<f64> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetadata {
    pub algorithm: String,
    pub task: String,
    pub tree_type: String,
    pub criterion: String,
    pub canaries: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_depth: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_samples_split: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_samples_leaf: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class_labels: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegritySection {
    pub serialization: String,
    pub canonical_json: bool,
    pub compatibility: Compatibility,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compatibility {
    pub minimum_runtime_version: String,
    pub required_capabilities: Vec<String>,
}

pub(crate) fn model_to_ir(model: &Model) -> ModelPackageIr {
    let tree = match model {
        Model::TargetMean(target_mean) => target_mean.to_ir_tree(),
        Model::DecisionTreeClassifier(classifier) => classifier.to_ir_tree(),
        Model::DecisionTreeRegressor(regressor) => regressor.to_ir_tree(),
    };
    let representation = match &tree {
        TreeDefinition::NodeTree { .. } => "node_tree",
        TreeDefinition::ObliviousLevels { .. } => "oblivious_levels",
    };
    let class_labels = match model {
        Model::DecisionTreeClassifier(classifier) => Some(classifier.class_labels().to_vec()),
        Model::TargetMean(_) | Model::DecisionTreeRegressor(_) => None,
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
            is_ensemble: false,
            trees: vec![tree],
            aggregation: Aggregation {
                kind: "identity_single_tree".to_string(),
                tree_weights: vec![1.0],
                normalize_by_weight_sum: true,
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
        training_metadata: match model {
            Model::TargetMean(target_mean) => target_mean.training_metadata(),
            Model::DecisionTreeClassifier(classifier) => classifier.training_metadata(),
            Model::DecisionTreeRegressor(regressor) => regressor.training_metadata(),
        },
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
            kind: "rank_bin_512".to_string(),
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
        "training_rank_bin_512".to_string(),
    ];
    match model.tree_type() {
        TreeType::Id3 | TreeType::C45 => {
            capabilities.push("binned_multiway_splits".to_string());
        }
        TreeType::TargetMean => {
            capabilities.push("constant_leaf_prediction".to_string());
        }
        TreeType::Cart | TreeType::Oblivious => {
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
    }
}

pub(crate) fn criterion_name(criterion: crate::Criterion) -> &'static str {
    match criterion {
        crate::Criterion::Auto => "auto",
        crate::Criterion::Gini => "gini",
        crate::Criterion::Entropy => "entropy",
        crate::Criterion::Mean => "mean",
        crate::Criterion::Median => "median",
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
        TreeType::TargetMean => "target_mean",
        TreeType::Id3 => "id3",
        TreeType::C45 => "c45",
        TreeType::Cart => "cart",
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
