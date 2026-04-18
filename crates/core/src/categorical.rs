use crate::{Model, OptimizedModel, PredictError, TrainConfig, TrainError};
use forestfire_data::categorical::{
    TableCategoricalConfig, TableCategoricalEncoder, TableCategoricalError,
    TableCategoricalStrategy, category_key, category_signal_strength, resolve_categorical_features,
    shuffled_feature_rows, validate_rows,
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
        column_names: Vec<String>,
        passthrough_features: Vec<usize>,
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
                    column_names,
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
