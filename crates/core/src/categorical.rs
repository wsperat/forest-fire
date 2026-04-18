use crate::{Model, OptimizedModel, PredictError, TrainConfig, TrainError};
use forestfire_data::categorical::{
    TableCategoricalConfig, TableCategoricalEncoder, TableCategoricalError,
    TableCategoricalStrategy, category_key, validate_rows,
};
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

#[derive(Debug, Clone)]
struct FisherFeatureSpec {
    feature_index: usize,
    mapping: BTreeMap<String, f64>,
}

#[derive(Debug, Clone)]
enum CoreEncoder {
    Table(TableCategoricalEncoder),
    Fisher {
        table: TableCategoricalEncoder,
        fisher_specs: Vec<FisherFeatureSpec>,
    },
}

#[derive(Debug, Clone)]
pub struct CategoricalModel {
    inner: Model,
    encoder: CoreEncoder,
}

#[derive(Debug, Clone)]
pub struct CategoricalOptimizedModel {
    inner: OptimizedModel,
    encoder: CoreEncoder,
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
        Ok(CategoricalOptimizedModel {
            inner: self.inner.optimize_inference_with_missing_features(
                physical_cores,
                missing_features.map(|features| features.to_vec()),
            )?,
            encoder: self.encoder.clone(),
        })
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

    let encoder = match categorical.strategy {
        CategoricalStrategy::Dummy => CoreEncoder::Table(
            TableCategoricalEncoder::fit(
                &rows,
                &targets,
                column_names,
                &TableCategoricalConfig {
                    strategy: TableCategoricalStrategy::Dummy,
                    categorical_features: categorical.categorical_features.clone(),
                    target_smoothing: categorical.target_smoothing,
                },
            )
            .map_err(CategoricalError::Table)?,
        ),
        CategoricalStrategy::Target => CoreEncoder::Table(
            TableCategoricalEncoder::fit(
                &rows,
                &targets,
                column_names,
                &TableCategoricalConfig {
                    strategy: TableCategoricalStrategy::Target,
                    categorical_features: categorical.categorical_features.clone(),
                    target_smoothing: categorical.target_smoothing,
                },
            )
            .map_err(CategoricalError::Table)?,
        ),
        CategoricalStrategy::Fisher => {
            let fisher_specs = fit_fisher_specs(
                &rows,
                &targets,
                categorical.categorical_features.as_deref(),
                categorical.target_smoothing,
            )
            .map_err(CategoricalError::Table)?;
            let table_encoder = TableCategoricalEncoder::fit(
                &apply_fisher_mapping(&rows, &fisher_specs),
                &targets,
                column_names,
                &TableCategoricalConfig {
                    strategy: TableCategoricalStrategy::Target,
                    categorical_features: Some(
                        fisher_specs.iter().map(|spec| spec.feature_index).collect(),
                    ),
                    target_smoothing: 0.0,
                },
            )
            .map_err(CategoricalError::Table)?;
            CoreEncoder::Fisher {
                table: table_encoder,
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
                table,
                fisher_specs,
            } => table
                .transform_rows(
                    &apply_fisher_mapping(rows, fisher_specs),
                    Some(&table.column_names),
                )
                .map_err(CategoricalError::Table),
        }
    }
}

fn fit_fisher_specs(
    rows: &[Vec<CategoricalValue>],
    targets: &[f64],
    explicit: Option<&[usize]>,
    smoothing: f64,
) -> Result<Vec<FisherFeatureSpec>, TableCategoricalError> {
    let categorical_features =
        forestfire_data::categorical::resolve_categorical_features(rows, explicit)?;
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
                let score = (values.iter().sum::<f64>() + smoothing * global_mean)
                    / (values.len() as f64 + smoothing);
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

fn apply_fisher_mapping(
    rows: &[Vec<CategoricalValue>],
    fisher_specs: &[FisherFeatureSpec],
) -> Vec<Vec<CategoricalValue>> {
    let mut encoded = rows.to_vec();
    for row in &mut encoded {
        for spec in fisher_specs {
            row[spec.feature_index] = category_key(&row[spec.feature_index])
                .and_then(|category| spec.mapping.get(&category).copied())
                .map(CategoricalValue::Numeric)
                .unwrap_or(CategoricalValue::Missing);
        }
    }
    encoded
}
