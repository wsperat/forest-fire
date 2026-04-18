use crate::compiled_artifact::{
    COMPILED_ARTIFACT_BACKEND_CPU, COMPILED_ARTIFACT_HEADER_LEN, COMPILED_ARTIFACT_MAGIC,
    COMPILED_ARTIFACT_VERSION,
};
use crate::ir::{
    CategoricalDummyFeature, CategoricalEncoder as CategoricalEncoderIr, CategoricalFisherFeature,
    CategoricalMappingEntry, CategoricalRankEntry, CategoricalTargetFeature,
    CategoricalTransformSection, InputFeature, InputSchema, ModelPackageIr,
};
use crate::{
    CompiledArtifactError, IrError, Model, OptimizedModel, PredictError, TrainConfig, TrainError,
};
use forestfire_data::categorical::{
    DummyFeatureSpec, TableCategoricalConfig, TableCategoricalEncoder, TableCategoricalError,
    TableCategoricalStrategy, TargetFeatureSpec, category_key, category_signal_strength,
    resolve_categorical_features, shuffled_feature_rows, validate_rows,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

pub use forestfire_data::categorical::CategoricalValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CategoricalStrategy {
    Dummy,
    Target,
    Fisher,
}

#[derive(Debug, Clone)]
pub struct CategoricalConfig {
    pub strategy: CategoricalStrategy,
    pub categorical_features: Option<Vec<usize>>,
    pub target_smoothing: f64,
}

impl Default for CategoricalConfig {
    fn default() -> Self {
        Self {
            strategy: CategoricalStrategy::Dummy,
            categorical_features: None,
            target_smoothing: 20.0,
        }
    }
}

#[derive(Debug)]
pub enum CategoricalError {
    Table(TableCategoricalError),
    TableBuild(String),
    Train(TrainError),
    Predict(PredictError),
}

impl Display for CategoricalError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Table(err) => err.fmt(f),
            Self::TableBuild(message) => write!(f, "{message}"),
            Self::Train(err) => err.fmt(f),
            Self::Predict(err) => err.fmt(f),
        }
    }
}

impl Error for CategoricalError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FisherFeatureSpec {
    feature_index: usize,
    mapping: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum CoreEncoder {
    Table(TableCategoricalEncoder),
    Fisher {
        column_names: Vec<String>,
        passthrough_features: Vec<usize>,
        fisher_specs: Vec<FisherFeatureSpec>,
    },
}

#[derive(Debug, Clone)]
pub struct CategoricalModel {
    inner: Model,
    encoder: CoreEncoder,
    config: CategoricalConfig,
    raw_input_schema: InputSchema,
}

#[derive(Debug, Clone)]
pub struct CategoricalOptimizedModel {
    inner: OptimizedModel,
    encoder: CoreEncoder,
    config: CategoricalConfig,
    raw_input_schema: InputSchema,
}

impl CategoricalModel {
    pub fn inner(&self) -> &Model {
        &self.inner
    }

    pub fn predict_rows(
        &self,
        rows: Vec<Vec<CategoricalValue>>,
    ) -> Result<Vec<f64>, CategoricalError> {
        self.inner
            .predict_rows(self.encoder.transform_rows(&rows)?)
            .map_err(CategoricalError::Predict)
    }

    pub fn predict_proba_rows(
        &self,
        rows: Vec<Vec<CategoricalValue>>,
    ) -> Result<Vec<Vec<f64>>, CategoricalError> {
        self.inner
            .predict_proba_rows(self.encoder.transform_rows(&rows)?)
            .map_err(CategoricalError::Predict)
    }

    pub fn optimize_inference(
        &self,
        physical_cores: Option<usize>,
        missing_features: Option<&[usize]>,
    ) -> Result<CategoricalOptimizedModel, crate::OptimizeError> {
        let encoded_missing_features =
            missing_features.map(|features| self.encoder.encoded_missing_features(features));
        Ok(CategoricalOptimizedModel {
            inner: self.inner.optimize_inference_with_missing_features(
                physical_cores,
                encoded_missing_features,
            )?,
            encoder: self.encoder.clone(),
            config: self.config.clone(),
            raw_input_schema: self.raw_input_schema.clone(),
        })
    }

    pub fn to_ir(&self) -> ModelPackageIr {
        let mut ir = self.inner.to_ir();
        ir.model.supports_categorical = true;
        ir.input_schema = self.raw_input_schema.clone();
        ir.preprocessing.categorical = Some(categorical_transform_section(
            &self.encoder,
            &self.config,
            &self.raw_input_schema,
        ));
        ir.preprocessing.notes = "Numeric features use serialized training-time rank bins. Categorical transforms are serialized alongside the model and applied before numeric/binary inference."
            .to_string();
        ir
    }

    pub fn to_ir_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.to_ir())
    }

    pub fn to_ir_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.to_ir())
    }

    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.to_ir_json()
    }

    pub fn serialize_pretty(&self) -> Result<String, serde_json::Error> {
        self.to_ir_json_pretty()
    }

    pub fn deserialize(serialized: &str) -> Result<Self, IrError> {
        let ir: ModelPackageIr =
            serde_json::from_str(serialized).map_err(|err| IrError::Json(err.to_string()))?;
        categorical_model_from_ir(ir)
    }
}

impl CategoricalOptimizedModel {
    pub fn inner(&self) -> &OptimizedModel {
        &self.inner
    }

    pub fn predict_rows(
        &self,
        rows: Vec<Vec<CategoricalValue>>,
    ) -> Result<Vec<f64>, CategoricalError> {
        self.inner
            .predict_rows(self.encoder.transform_rows(&rows)?)
            .map_err(CategoricalError::Predict)
    }

    pub fn predict_proba_rows(
        &self,
        rows: Vec<Vec<CategoricalValue>>,
    ) -> Result<Vec<Vec<f64>>, CategoricalError> {
        self.inner
            .predict_proba_rows(self.encoder.transform_rows(&rows)?)
            .map_err(CategoricalError::Predict)
    }

    pub fn to_ir(&self) -> ModelPackageIr {
        let mut ir = self.inner.to_ir();
        ir.model.supports_categorical = true;
        ir.input_schema = self.raw_input_schema.clone();
        ir.preprocessing.categorical = Some(categorical_transform_section(
            &self.encoder,
            &self.config,
            &self.raw_input_schema,
        ));
        ir.preprocessing.notes = "Numeric features use serialized training-time rank bins. Categorical transforms are serialized alongside the model and applied before optimized inference."
            .to_string();
        ir
    }

    pub fn to_ir_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.to_ir())
    }

    pub fn to_ir_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.to_ir())
    }

    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.to_ir_json()
    }

    pub fn serialize_pretty(&self) -> Result<String, serde_json::Error> {
        self.to_ir_json_pretty()
    }

    pub fn serialize_compiled(&self) -> Result<Vec<u8>, CompiledArtifactError> {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct Payload {
            semantic_ir: ModelPackageIr,
            runtime: crate::OptimizedRuntime,
            feature_projection: Option<Vec<usize>>,
        }

        let payload = Payload {
            semantic_ir: self.to_ir(),
            runtime: self.inner.runtime.clone(),
            feature_projection: Some(self.inner.feature_projection.clone()),
        };
        let mut payload_bytes = Vec::new();
        ciborium::into_writer(&payload, &mut payload_bytes)
            .map_err(|err| CompiledArtifactError::Encode(err.to_string()))?;
        let mut bytes = Vec::with_capacity(COMPILED_ARTIFACT_HEADER_LEN + payload_bytes.len());
        bytes.extend_from_slice(&COMPILED_ARTIFACT_MAGIC);
        bytes.extend_from_slice(&COMPILED_ARTIFACT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&COMPILED_ARTIFACT_BACKEND_CPU.to_le_bytes());
        bytes.extend_from_slice(&payload_bytes);
        Ok(bytes)
    }

    pub fn deserialize_compiled(
        serialized: &[u8],
        physical_cores: Option<usize>,
    ) -> Result<Self, CompiledArtifactError> {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct Payload {
            semantic_ir: ModelPackageIr,
            runtime: crate::OptimizedRuntime,
            feature_projection: Option<Vec<usize>>,
        }
        if serialized.len() < COMPILED_ARTIFACT_HEADER_LEN {
            return Err(CompiledArtifactError::ArtifactTooShort {
                actual: serialized.len(),
                minimum: COMPILED_ARTIFACT_HEADER_LEN,
            });
        }
        let magic = [serialized[0], serialized[1], serialized[2], serialized[3]];
        if magic != COMPILED_ARTIFACT_MAGIC {
            return Err(CompiledArtifactError::InvalidMagic(magic));
        }
        let version = u16::from_le_bytes([serialized[4], serialized[5]]);
        if version != COMPILED_ARTIFACT_VERSION {
            return Err(CompiledArtifactError::UnsupportedVersion(version));
        }
        let backend = u16::from_le_bytes([serialized[6], serialized[7]]);
        if backend != COMPILED_ARTIFACT_BACKEND_CPU {
            return Err(CompiledArtifactError::UnsupportedBackend(backend));
        }
        let _payload: Payload = ciborium::from_reader(std::io::Cursor::new(
            &serialized[COMPILED_ARTIFACT_HEADER_LEN..],
        ))
        .map_err(|err| CompiledArtifactError::Decode(err.to_string()))?;
        let restored = categorical_model_from_ir(_payload.semantic_ir)
            .map_err(CompiledArtifactError::InvalidSemanticModel)?;
        restored
            .optimize_inference(physical_cores, None)
            .map_err(CompiledArtifactError::InvalidRuntime)
    }
}

pub fn train(
    rows: Vec<Vec<CategoricalValue>>,
    targets: Vec<f64>,
    column_names: Option<Vec<String>>,
    canaries: usize,
    numeric_bins: forestfire_data::NumericBins,
    config: TrainConfig,
    categorical: CategoricalConfig,
) -> Result<CategoricalModel, CategoricalError> {
    validate_rows(&rows).map_err(CategoricalError::Table)?;
    let column_names = column_names.unwrap_or_else(|| {
        (0..rows.first().map_or(0, |row| row.len()))
            .map(|index| format!("f{index}"))
            .collect()
    });
    let raw_input_schema = raw_input_schema(&column_names, &rows);

    let encoder = match categorical.strategy {
        CategoricalStrategy::Dummy => CoreEncoder::Table(
            TableCategoricalEncoder::fit(
                &rows,
                &targets,
                column_names.clone(),
                &TableCategoricalConfig {
                    strategy: TableCategoricalStrategy::Dummy,
                    categorical_features: categorical.categorical_features.clone(),
                    target_smoothing: categorical.target_smoothing,
                    target_smoothing_overrides: BTreeMap::new(),
                },
            )
            .map_err(CategoricalError::Table)?,
        ),
        CategoricalStrategy::Target => {
            let smoothing_overrides = canary_informed_target_smoothing(
                &rows,
                &targets,
                categorical.categorical_features.as_deref(),
                categorical.target_smoothing,
            )
            .map_err(CategoricalError::Table)?;
            CoreEncoder::Table(
                TableCategoricalEncoder::fit(
                    &rows,
                    &targets,
                    column_names.clone(),
                    &TableCategoricalConfig {
                        strategy: TableCategoricalStrategy::Target,
                        categorical_features: categorical.categorical_features.clone(),
                        target_smoothing: categorical.target_smoothing,
                        target_smoothing_overrides: smoothing_overrides,
                    },
                )
                .map_err(CategoricalError::Table)?,
            )
        }
        CategoricalStrategy::Fisher => {
            let smoothing_overrides = canary_informed_target_smoothing(
                &rows,
                &targets,
                categorical.categorical_features.as_deref(),
                categorical.target_smoothing,
            )
            .map_err(CategoricalError::Table)?;
            let fisher_specs = fit_fisher_specs(
                &rows,
                &targets,
                categorical.categorical_features.as_deref(),
                &smoothing_overrides,
                categorical.target_smoothing,
            )
            .map_err(CategoricalError::Table)?;
            CoreEncoder::Fisher {
                passthrough_features: collect_passthrough_features(&rows, &fisher_specs),
                column_names,
                fisher_specs,
            }
        }
    };

    let encoded = encoder.transform_rows(&rows)?;
    let table = forestfire_data::Table::with_options(encoded, targets, canaries, numeric_bins)
        .map_err(|err| CategoricalError::TableBuild(err.to_string()))?;
    let model = crate::train(&table, config).map_err(CategoricalError::Train)?;
    Ok(CategoricalModel {
        inner: model,
        encoder,
        config: categorical,
        raw_input_schema,
    })
}

impl CoreEncoder {
    fn transform_rows(
        &self,
        rows: &[Vec<CategoricalValue>],
    ) -> Result<Vec<Vec<f64>>, CategoricalError> {
        match self {
            Self::Table(encoder) => encoder
                .transform_rows(rows, Some(&encoder.column_names))
                .map_err(CategoricalError::Table),
            Self::Fisher {
                column_names,
                passthrough_features,
                fisher_specs,
            } => transform_fisher_rows(rows, column_names, passthrough_features, fisher_specs)
                .map_err(CategoricalError::Table),
        }
    }

    fn encoded_missing_features(&self, raw_missing_features: &[usize]) -> Vec<usize> {
        let mut encoded_missing = Vec::new();
        match self {
            Self::Table(encoder) => {
                let mut offset = 0usize;
                for &feature_index in &encoder.numeric_features {
                    if raw_missing_features.contains(&feature_index) {
                        encoded_missing.push(offset);
                    }
                    offset += 1;
                }
                for spec in &encoder.dummy_specs {
                    let width = spec.categories.len() + 1;
                    if raw_missing_features.contains(&spec.feature_index) {
                        encoded_missing.extend(offset..offset + width);
                    }
                    offset += width;
                }
                for spec in &encoder.target_specs {
                    let width = spec.priors.len();
                    if raw_missing_features.contains(&spec.feature_index) {
                        encoded_missing.extend(offset..offset + width);
                    }
                    offset += width;
                }
            }
            Self::Fisher {
                passthrough_features,
                fisher_specs,
                ..
            } => {
                let mut offset = 0usize;
                for &feature_index in passthrough_features {
                    if raw_missing_features.contains(&feature_index) {
                        encoded_missing.push(offset);
                    }
                    offset += 1;
                }
                for spec in fisher_specs {
                    if raw_missing_features.contains(&spec.feature_index) {
                        encoded_missing.push(offset);
                    }
                    offset += 1;
                }
            }
        }
        encoded_missing.sort_unstable();
        encoded_missing.dedup();
        encoded_missing
    }
}

fn canary_informed_target_smoothing(
    rows: &[Vec<CategoricalValue>],
    targets: &[f64],
    explicit: Option<&[usize]>,
    base_smoothing: f64,
) -> Result<BTreeMap<usize, f64>, TableCategoricalError> {
    let categorical_features = resolve_categorical_features(rows, explicit)?;
    let mut overrides = BTreeMap::new();
    for feature_index in categorical_features {
        let real_strength = category_signal_strength(rows, targets, feature_index, base_smoothing);
        let shuffled_rows = shuffled_feature_rows(rows, feature_index, 0);
        let canary_strength =
            category_signal_strength(&shuffled_rows, targets, feature_index, base_smoothing);
        let baseline = base_smoothing.max(1.0);
        let ratio = if real_strength <= f64::EPSILON {
            4.0
        } else {
            (canary_strength / real_strength).clamp(0.0, 4.0)
        };
        overrides.insert(feature_index, base_smoothing + baseline * ratio);
    }
    Ok(overrides)
}

fn fit_fisher_specs(
    rows: &[Vec<CategoricalValue>],
    targets: &[f64],
    explicit: Option<&[usize]>,
    smoothing_overrides: &BTreeMap<usize, f64>,
    smoothing: f64,
) -> Result<Vec<FisherFeatureSpec>, TableCategoricalError> {
    let categorical_features = resolve_categorical_features(rows, explicit)?;
    let mut specs = Vec::new();
    for feature_index in categorical_features {
        let mut buckets = BTreeMap::<String, Vec<f64>>::new();
        for (row, target) in rows.iter().zip(targets.iter().copied()) {
            if let Some(category) = category_key(&row[feature_index]) {
                buckets.entry(category).or_default().push(target);
            }
        }
        let global_mean = if targets.is_empty() {
            0.0
        } else {
            targets.iter().sum::<f64>() / targets.len() as f64
        };
        let mut ordered = buckets
            .into_iter()
            .map(|(category, values)| {
                let effective_smoothing = smoothing_overrides
                    .get(&feature_index)
                    .copied()
                    .unwrap_or(smoothing);
                let score = (values.iter().sum::<f64>() + effective_smoothing * global_mean)
                    / (values.len() as f64 + effective_smoothing);
                (category, score)
            })
            .collect::<Vec<_>>();
        ordered.sort_by(|left, right| left.1.total_cmp(&right.1));
        specs.push(FisherFeatureSpec {
            feature_index,
            mapping: ordered
                .into_iter()
                .enumerate()
                .map(|(rank, (category, _))| (category, rank as f64))
                .collect(),
        });
    }
    Ok(specs)
}

fn collect_passthrough_features(
    rows: &[Vec<CategoricalValue>],
    fisher_specs: &[FisherFeatureSpec],
) -> Vec<usize> {
    let fisher_features = fisher_specs
        .iter()
        .map(|spec| spec.feature_index)
        .collect::<std::collections::BTreeSet<_>>();
    (0..rows.first().map_or(0, Vec::len))
        .filter(|feature_index| !fisher_features.contains(feature_index))
        .collect()
}

fn transform_fisher_rows(
    rows: &[Vec<CategoricalValue>],
    column_names: &[String],
    passthrough_features: &[usize],
    fisher_specs: &[FisherFeatureSpec],
) -> Result<Vec<Vec<f64>>, TableCategoricalError> {
    validate_rows(rows)?;
    if rows.first().map_or(0, Vec::len) != column_names.len() {
        return Err(TableCategoricalError::ColumnNameMismatch);
    }
    let mut encoded_rows = Vec::with_capacity(rows.len());
    for row in rows {
        let mut encoded = Vec::new();
        for &feature_index in passthrough_features {
            encoded.push(match &row[feature_index] {
                CategoricalValue::Numeric(value) => *value,
                CategoricalValue::Missing | CategoricalValue::String(_) => f64::NAN,
            });
        }
        for spec in fisher_specs {
            encoded.push(
                category_key(&row[spec.feature_index])
                    .and_then(|category| spec.mapping.get(&category).copied())
                    .unwrap_or(f64::NAN),
            );
        }
        encoded_rows.push(encoded);
    }
    Ok(encoded_rows)
}

fn raw_input_schema(column_names: &[String], rows: &[Vec<CategoricalValue>]) -> InputSchema {
    let features = column_names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let logical_type = if rows
                .iter()
                .any(|row| matches!(row[index], CategoricalValue::String(_)))
            {
                "categorical"
            } else {
                "numeric"
            };
            InputFeature {
                index,
                name: name.clone(),
                dtype: if logical_type == "categorical" {
                    "string".to_string()
                } else {
                    "float64".to_string()
                },
                logical_type: logical_type.to_string(),
                nullable: rows
                    .iter()
                    .any(|row| matches!(row[index], CategoricalValue::Missing)),
            }
        })
        .collect::<Vec<_>>();
    InputSchema {
        feature_count: column_names.len(),
        features,
        ordering: "strict_index_order".to_string(),
        input_tensor_layout: "row_major".to_string(),
        accepts_feature_names: false,
    }
}

fn categorical_transform_section(
    encoder: &CoreEncoder,
    config: &CategoricalConfig,
    raw_input_schema: &InputSchema,
) -> CategoricalTransformSection {
    CategoricalTransformSection {
        raw_feature_count: raw_input_schema.feature_count,
        raw_features: raw_input_schema.features.clone(),
        strategy: match config.strategy {
            CategoricalStrategy::Dummy => "dummy",
            CategoricalStrategy::Target => "target",
            CategoricalStrategy::Fisher => "fisher",
        }
        .to_string(),
        categorical_features: config.categorical_features.clone(),
        target_smoothing: config.target_smoothing,
        encoder: match encoder {
            CoreEncoder::Table(table) => CategoricalEncoderIr::Table {
                column_names: table.column_names.clone(),
                numeric_features: table.numeric_features.clone(),
                dummy_specs: table
                    .dummy_specs
                    .iter()
                    .map(|spec| CategoricalDummyFeature {
                        feature_index: spec.feature_index,
                        categories: spec.categories.clone(),
                    })
                    .collect(),
                target_specs: table
                    .target_specs
                    .iter()
                    .map(|spec| CategoricalTargetFeature {
                        feature_index: spec.feature_index,
                        priors: spec.priors.clone(),
                        mapping: spec
                            .mapping
                            .iter()
                            .map(|(category, encoded)| CategoricalMappingEntry {
                                category: category.clone(),
                                encoded: encoded.clone(),
                            })
                            .collect(),
                    })
                    .collect(),
            },
            CoreEncoder::Fisher {
                column_names,
                passthrough_features,
                fisher_specs,
            } => CategoricalEncoderIr::Fisher {
                column_names: column_names.clone(),
                passthrough_features: passthrough_features.clone(),
                fisher_specs: fisher_specs
                    .iter()
                    .map(|spec| CategoricalFisherFeature {
                        feature_index: spec.feature_index,
                        mapping: spec
                            .mapping
                            .iter()
                            .map(|(category, rank)| CategoricalRankEntry {
                                category: category.clone(),
                                rank: *rank,
                            })
                            .collect(),
                    })
                    .collect(),
            },
        },
    }
}

fn categorical_model_from_ir(ir: ModelPackageIr) -> Result<CategoricalModel, IrError> {
    if !ir.model.supports_categorical {
        return Err(IrError::InvalidInferenceOption(
            "categorical transform metadata is missing".to_string(),
        ));
    }
    let categorical = ir.preprocessing.categorical.clone().ok_or_else(|| {
        IrError::InvalidPreprocessing("categorical preprocessing block is missing".to_string())
    })?;
    let mut inner_ir = ir.clone();
    inner_ir.model.supports_categorical = false;
    inner_ir.input_schema = encoded_input_schema(&inner_ir);
    inner_ir.preprocessing.categorical = None;
    let inner = crate::ir::model_from_ir(inner_ir)?;
    let encoder = match categorical.encoder {
        CategoricalEncoderIr::Table {
            column_names,
            numeric_features,
            dummy_specs,
            target_specs,
        } => CoreEncoder::Table(TableCategoricalEncoder {
            column_names,
            numeric_features,
            dummy_specs: dummy_specs
                .into_iter()
                .map(|spec| DummyFeatureSpec {
                    feature_index: spec.feature_index,
                    categories: spec.categories,
                })
                .collect(),
            target_specs: target_specs
                .into_iter()
                .map(|spec| TargetFeatureSpec {
                    feature_index: spec.feature_index,
                    priors: spec.priors,
                    mapping: spec
                        .mapping
                        .into_iter()
                        .map(|entry| (entry.category, entry.encoded))
                        .collect(),
                })
                .collect(),
        }),
        CategoricalEncoderIr::Fisher {
            column_names,
            passthrough_features,
            fisher_specs,
        } => CoreEncoder::Fisher {
            column_names,
            passthrough_features,
            fisher_specs: fisher_specs
                .into_iter()
                .map(|spec| FisherFeatureSpec {
                    feature_index: spec.feature_index,
                    mapping: spec
                        .mapping
                        .into_iter()
                        .map(|entry| (entry.category, entry.rank))
                        .collect(),
                })
                .collect(),
        },
    };
    Ok(CategoricalModel {
        inner,
        encoder,
        config: CategoricalConfig {
            strategy: match categorical.strategy.as_str() {
                "dummy" => CategoricalStrategy::Dummy,
                "target" => CategoricalStrategy::Target,
                "fisher" => CategoricalStrategy::Fisher,
                other => {
                    return Err(IrError::InvalidPreprocessing(format!(
                        "unsupported categorical strategy '{}'",
                        other
                    )));
                }
            },
            categorical_features: categorical.categorical_features,
            target_smoothing: categorical.target_smoothing,
        },
        raw_input_schema: ir.input_schema,
    })
}

fn encoded_input_schema(ir: &ModelPackageIr) -> InputSchema {
    let features = ir
        .preprocessing
        .numeric_binning
        .features
        .iter()
        .map(|feature| match feature {
            crate::ir::FeatureBinning::Numeric { feature_index, .. } => InputFeature {
                index: *feature_index,
                name: format!("f{}", feature_index),
                dtype: "float64".to_string(),
                logical_type: "numeric".to_string(),
                nullable: false,
            },
            crate::ir::FeatureBinning::Binary { feature_index } => InputFeature {
                index: *feature_index,
                name: format!("f{}", feature_index),
                dtype: "bool".to_string(),
                logical_type: "boolean".to_string(),
                nullable: false,
            },
        })
        .collect();
    InputSchema {
        feature_count: ir.model.num_features,
        features,
        ordering: "strict_index_order".to_string(),
        input_tensor_layout: "row_major".to_string(),
        accepts_feature_names: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Criterion, Task, TrainAlgorithm, TrainConfig, TreeType};
    use forestfire_data::NumericBins;

    fn rows() -> Vec<Vec<CategoricalValue>> {
        vec![
            vec![CategoricalValue::String("a".into())],
            vec![CategoricalValue::String("a".into())],
            vec![CategoricalValue::String("b".into())],
            vec![CategoricalValue::String("b".into())],
            vec![CategoricalValue::String("c".into())],
            vec![CategoricalValue::String("c".into())],
        ]
    }

    #[test]
    fn canary_informed_smoothing_never_undershoots_base() {
        let overrides =
            canary_informed_target_smoothing(&rows(), &[0.0, 0.0, 1.0, 1.0, 0.0, 1.0], None, 2.0)
                .unwrap();
        assert!(overrides.get(&0).copied().unwrap() >= 2.0);
    }

    #[test]
    fn fisher_transform_emits_ranked_numeric_feature() {
        let specs = fit_fisher_specs(
            &rows(),
            &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            None,
            &BTreeMap::from([(0usize, 1.0)]),
            1.0,
        )
        .unwrap();
        let transformed =
            transform_fisher_rows(&rows(), &[String::from("f0")], &[], &specs).unwrap();
        assert!(
            transformed
                .iter()
                .all(|row| row.len() == 1 && row[0].is_finite())
        );
    }

    #[test]
    fn optimize_inference_maps_raw_missing_features_for_dummy_strategy() {
        assert_optimized_missing_feature_matches_semantic(CategoricalStrategy::Dummy);
    }

    #[test]
    fn optimize_inference_maps_raw_missing_features_for_target_strategy() {
        assert_optimized_missing_feature_matches_semantic(CategoricalStrategy::Target);
    }

    #[test]
    fn optimize_inference_maps_raw_missing_features_for_fisher_strategy() {
        assert_optimized_missing_feature_matches_semantic(CategoricalStrategy::Fisher);
    }

    fn assert_optimized_missing_feature_matches_semantic(strategy: CategoricalStrategy) {
        let model = train(
            regression_rows_with_numeric_signal(),
            regression_targets_with_numeric_signal(),
            Some(vec!["cat".into(), "num".into()]),
            0,
            NumericBins::Auto,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Regression,
                tree_type: TreeType::Cart,
                criterion: Criterion::Mean,
                max_depth: Some(2),
                min_samples_split: Some(2),
                min_samples_leaf: Some(1),
                physical_cores: Some(1),
                seed: Some(7),
                ..TrainConfig::default()
            },
            CategoricalConfig {
                strategy,
                categorical_features: Some(vec![0]),
                target_smoothing: 20.0,
            },
        )
        .unwrap();
        let prediction_rows = vec![
            vec![CategoricalValue::Missing, CategoricalValue::Numeric(0.0)],
            vec![CategoricalValue::Missing, CategoricalValue::Numeric(3.0)],
        ];
        let expected = model.predict_rows(prediction_rows.clone()).unwrap();
        let optimized = model.optimize_inference(Some(1), Some(&[0])).unwrap();
        let actual = optimized.predict_rows(prediction_rows).unwrap();
        assert_eq!(actual, expected);
    }

    fn regression_rows_with_numeric_signal() -> Vec<Vec<CategoricalValue>> {
        vec![
            vec![
                CategoricalValue::String("x".into()),
                CategoricalValue::Numeric(0.0),
            ],
            vec![
                CategoricalValue::String("y".into()),
                CategoricalValue::Numeric(0.0),
            ],
            vec![
                CategoricalValue::String("x".into()),
                CategoricalValue::Numeric(1.0),
            ],
            vec![
                CategoricalValue::String("y".into()),
                CategoricalValue::Numeric(1.0),
            ],
            vec![
                CategoricalValue::String("x".into()),
                CategoricalValue::Numeric(2.0),
            ],
            vec![
                CategoricalValue::String("y".into()),
                CategoricalValue::Numeric(2.0),
            ],
            vec![
                CategoricalValue::String("x".into()),
                CategoricalValue::Numeric(3.0),
            ],
            vec![
                CategoricalValue::String("y".into()),
                CategoricalValue::Numeric(3.0),
            ],
        ]
    }

    fn regression_targets_with_numeric_signal() -> Vec<f64> {
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    }
}
