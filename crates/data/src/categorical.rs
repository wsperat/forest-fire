use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CategoricalValue {
    Numeric(f64),
    String(String),
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TableCategoricalStrategy {
    Dummy,
    Target,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableCategoricalConfig {
    pub strategy: TableCategoricalStrategy,
    pub categorical_features: Option<Vec<usize>>,
    pub target_smoothing: f64,
    pub target_smoothing_overrides: BTreeMap<usize, f64>,
}

impl Default for TableCategoricalConfig {
    fn default() -> Self {
        Self {
            strategy: TableCategoricalStrategy::Dummy,
            categorical_features: None,
            target_smoothing: 20.0,
            target_smoothing_overrides: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DummyFeatureSpec {
    pub feature_index: usize,
    pub categories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetFeatureSpec {
    pub feature_index: usize,
    pub priors: Vec<f64>,
    pub mapping: BTreeMap<String, Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableCategoricalEncoder {
    pub column_names: Vec<String>,
    pub numeric_features: Vec<usize>,
    pub dummy_specs: Vec<DummyFeatureSpec>,
    pub target_specs: Vec<TargetFeatureSpec>,
}

#[derive(Debug)]
pub enum TableCategoricalError {
    EmptyRows,
    RaggedRows {
        row: usize,
        expected: usize,
        actual: usize,
    },
    TargetLengthMismatch {
        rows: usize,
        targets: usize,
    },
    InvalidCategoricalFeature {
        feature_index: usize,
        feature_count: usize,
    },
    ColumnNameMismatch,
    InvalidTargetSmoothing(f64),
}

impl std::fmt::Display for TableCategoricalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRows => write!(f, "Categorical inputs cannot be empty."),
            Self::RaggedRows {
                row,
                expected,
                actual,
            } => write!(
                f,
                "Ragged row at index {}: expected {} columns, found {}.",
                row, expected, actual
            ),
            Self::TargetLengthMismatch { rows, targets } => write!(
                f,
                "Categorical training rows/targets length mismatch: {} rows vs {} targets.",
                rows, targets
            ),
            Self::InvalidCategoricalFeature {
                feature_index,
                feature_count,
            } => write!(
                f,
                "Invalid categorical feature index {} for feature count {}.",
                feature_index, feature_count
            ),
            Self::ColumnNameMismatch => {
                write!(
                    f,
                    "Prediction columns do not match the categorical training columns."
                )
            }
            Self::InvalidTargetSmoothing(value) => write!(
                f,
                "target_smoothing must be finite and non-negative. Found {}.",
                value
            ),
        }
    }
}

impl std::error::Error for TableCategoricalError {}

impl TableCategoricalEncoder {
    pub fn fit(
        rows: &[Vec<CategoricalValue>],
        targets: &[f64],
        column_names: Vec<String>,
        config: &TableCategoricalConfig,
    ) -> Result<Self, TableCategoricalError> {
        validate_rows(rows)?;
        if rows.len() != targets.len() {
            return Err(TableCategoricalError::TargetLengthMismatch {
                rows: rows.len(),
                targets: targets.len(),
            });
        }
        if !config.target_smoothing.is_finite() || config.target_smoothing < 0.0 {
            return Err(TableCategoricalError::InvalidTargetSmoothing(
                config.target_smoothing,
            ));
        }
        let categorical_features =
            resolve_categorical_features(rows, config.categorical_features.as_deref())?;
        let feature_count = rows.first().map_or(0, Vec::len);
        let numeric_features = (0..feature_count)
            .filter(|feature_index| !categorical_features.contains(feature_index))
            .collect::<Vec<_>>();
        let mut dummy_specs = Vec::new();
        let mut target_specs = Vec::new();
        for feature_index in 0..feature_count {
            if !categorical_features.contains(&feature_index) {
                continue;
            }
            match config.strategy {
                TableCategoricalStrategy::Dummy => dummy_specs.push(DummyFeatureSpec {
                    feature_index,
                    categories: distinct_categories(rows, feature_index),
                }),
                TableCategoricalStrategy::Target => target_specs.push(fit_target_spec(
                    rows,
                    targets,
                    feature_index,
                    config
                        .target_smoothing_overrides
                        .get(&feature_index)
                        .copied()
                        .unwrap_or(config.target_smoothing),
                )),
            }
        }
        Ok(Self {
            column_names,
            numeric_features,
            dummy_specs,
            target_specs,
        })
    }

    pub fn transform_rows(
        &self,
        rows: &[Vec<CategoricalValue>],
        column_names: Option<&[String]>,
    ) -> Result<Vec<Vec<f64>>, TableCategoricalError> {
        validate_rows(rows)?;
        if let Some(names) = column_names
            && names != self.column_names
        {
            return Err(TableCategoricalError::ColumnNameMismatch);
        }
        let mut encoded_rows = Vec::with_capacity(rows.len());
        for row in rows {
            let mut encoded = Vec::new();
            for &feature_index in &self.numeric_features {
                encoded.push(numeric_value(&row[feature_index]));
            }
            for spec in &self.dummy_specs {
                let category = category_key(&row[spec.feature_index]);
                for known in &spec.categories {
                    encoded.push(f64::from(category.as_ref() == Some(known)));
                }
                encoded.push(f64::from(
                    category.is_none()
                        || !spec
                            .categories
                            .iter()
                            .any(|known| Some(known) == category.as_ref()),
                ));
            }
            for spec in &self.target_specs {
                let category = category_key(&row[spec.feature_index]);
                encoded.extend(
                    spec.mapping
                        .get(category.as_deref().unwrap_or_default())
                        .cloned()
                        .unwrap_or_else(|| spec.priors.clone()),
                );
            }
            encoded_rows.push(encoded);
        }
        Ok(encoded_rows)
    }
}

pub fn validate_rows(rows: &[Vec<CategoricalValue>]) -> Result<(), TableCategoricalError> {
    let Some(first) = rows.first() else {
        return Err(TableCategoricalError::EmptyRows);
    };
    let expected = first.len();
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != expected {
            return Err(TableCategoricalError::RaggedRows {
                row: row_index,
                expected,
                actual: row.len(),
            });
        }
    }
    Ok(())
}

pub fn resolve_categorical_features(
    rows: &[Vec<CategoricalValue>],
    explicit: Option<&[usize]>,
) -> Result<BTreeSet<usize>, TableCategoricalError> {
    let feature_count = rows.first().map_or(0, Vec::len);
    if let Some(indices) = explicit {
        let mut set = BTreeSet::new();
        for &feature_index in indices {
            if feature_index >= feature_count {
                return Err(TableCategoricalError::InvalidCategoricalFeature {
                    feature_index,
                    feature_count,
                });
            }
            set.insert(feature_index);
        }
        return Ok(set);
    }
    Ok((0..feature_count)
        .filter(|&feature_index| {
            rows.iter()
                .any(|row| matches!(row[feature_index], CategoricalValue::String(_)))
        })
        .collect())
}

pub fn category_key(value: &CategoricalValue) -> Option<String> {
    match value {
        CategoricalValue::String(value) => Some(value.clone()),
        CategoricalValue::Missing => None,
        CategoricalValue::Numeric(_) => None,
    }
}

pub fn numeric_value(value: &CategoricalValue) -> f64 {
    match value {
        CategoricalValue::Numeric(value) => *value,
        CategoricalValue::Missing => f64::NAN,
        CategoricalValue::String(_) => f64::NAN,
    }
}

pub fn distinct_categories(rows: &[Vec<CategoricalValue>], feature_index: usize) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut categories = Vec::new();
    for row in rows {
        if let Some(category) = category_key(&row[feature_index])
            && seen.insert(category.clone())
        {
            categories.push(category);
        }
    }
    categories
}

fn fit_target_spec(
    rows: &[Vec<CategoricalValue>],
    targets: &[f64],
    feature_index: usize,
    smoothing: f64,
) -> TargetFeatureSpec {
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
    let mapping = buckets
        .into_iter()
        .map(|(category, values)| {
            let encoded = (values.iter().sum::<f64>() + smoothing * global_mean)
                / (values.len() as f64 + smoothing);
            (category, vec![encoded])
        })
        .collect();
    TargetFeatureSpec {
        feature_index,
        priors: vec![global_mean],
        mapping,
    }
}

pub fn category_signal_strength(
    rows: &[Vec<CategoricalValue>],
    targets: &[f64],
    feature_index: usize,
    smoothing: f64,
) -> f64 {
    if rows.is_empty() || targets.is_empty() {
        return 0.0;
    }
    let mut buckets = BTreeMap::<String, (f64, usize)>::new();
    for (row, target) in rows.iter().zip(targets.iter().copied()) {
        if let Some(category) = category_key(&row[feature_index]) {
            let entry = buckets.entry(category).or_insert((0.0, 0));
            entry.0 += target;
            entry.1 += 1;
        }
    }
    if buckets.is_empty() {
        return 0.0;
    }
    let global_mean = targets.iter().sum::<f64>() / targets.len() as f64;
    buckets
        .into_values()
        .map(|(sum, count)| {
            let encoded = (sum + smoothing * global_mean) / (count as f64 + smoothing);
            count as f64 * (encoded - global_mean).powi(2)
        })
        .sum::<f64>()
        / targets.len() as f64
}

pub fn shuffled_feature_rows(
    rows: &[Vec<CategoricalValue>],
    feature_index: usize,
    copy_index: usize,
) -> Vec<Vec<CategoricalValue>> {
    let mut shuffled_rows = rows.to_vec();
    let mut feature_values = rows
        .iter()
        .map(|row| row[feature_index].clone())
        .collect::<Vec<_>>();
    shuffle_values(&mut feature_values, copy_index, feature_index);
    for (row, shuffled) in shuffled_rows.iter_mut().zip(feature_values) {
        row[feature_index] = shuffled;
    }
    shuffled_rows
}

fn shuffle_values<T>(values: &mut [T], copy_index: usize, source_index: usize) {
    let seed = 0xA11CE5EED_u64
        ^ ((copy_index as u64) << 32)
        ^ (source_index as u64)
        ^ ((values.len() as u64) << 16);
    let mut rng = StdRng::seed_from_u64(seed);
    values.shuffle(&mut rng);
}
