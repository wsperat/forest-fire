use crate::ir::{LeafPayload, NodeTreeNode, TrainingMetadata, TreeDefinition, criterion_name};
use crate::{Criterion, FeaturePreprocessing, capture_feature_preprocessing};
use forestfire_data::TableAccess;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// The simplest possible "tree":
/// it stores only one global target statistic and predicts that for every row.
#[derive(Debug, Clone)]
pub struct TargetMeanTree {
    pub mean: f64,
    criterion: Criterion,
    num_features: usize,
    feature_preprocessing: Vec<FeaturePreprocessing>,
    training_canaries: usize,
}

#[derive(Debug)]
pub enum ModelError {
    EmptyTarget,
}

impl Display for ModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::EmptyTarget => write!(f, "Cannot train on an empty target vector."),
        }
    }
}

impl Error for ModelError {}

pub fn train_target_mean(train_set: &dyn TableAccess) -> Result<TargetMeanTree, ModelError> {
    train_target_mean_with_criterion(train_set, Criterion::Mean)
}

pub fn train_target_mean_with_criterion(
    train_set: &dyn TableAccess,
    criterion: Criterion,
) -> Result<TargetMeanTree, ModelError> {
    if train_set.n_rows() == 0 {
        return Err(ModelError::EmptyTarget);
    }

    let targets: Vec<f64> = (0..train_set.n_rows())
        .map(|row_idx| train_set.target_value(row_idx))
        .collect();
    let prediction = match criterion {
        Criterion::Mean => targets.iter().sum::<f64>() / train_set.n_rows() as f64,
        Criterion::Median => median(&targets),
        _ => unreachable!("target statistic only supports mean or median"),
    };

    Ok(TargetMeanTree {
        mean: prediction,
        criterion,
        num_features: train_set.n_features(),
        feature_preprocessing: capture_feature_preprocessing(train_set),
        training_canaries: train_set.canaries(),
    })
}

impl TargetMeanTree {
    pub fn criterion(&self) -> Criterion {
        self.criterion
    }

    /// Predict the constant mean for `n` samples.
    pub fn predict_many(&self, n: usize) -> Vec<f64> {
        vec![self.mean; n]
    }

    /// Convenience: predict for a table (ignores features).
    pub fn predict_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        self.predict_many(table.n_rows())
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
            task: "regression".to_string(),
            tree_type: "target_mean".to_string(),
            criterion: criterion_name(self.criterion).to_string(),
            canaries: self.training_canaries,
            max_depth: None,
            min_samples_split: None,
            min_samples_leaf: None,
            class_labels: None,
        }
    }

    pub(crate) fn to_ir_tree(&self) -> TreeDefinition {
        TreeDefinition::NodeTree {
            tree_id: 0,
            weight: 1.0,
            root_node_id: 0,
            nodes: vec![NodeTreeNode::Leaf {
                node_id: 0,
                depth: 0,
                leaf: LeafPayload::RegressionValue { value: self.mean },
            }],
        }
    }

    pub(crate) fn from_ir_parts(
        mean: f64,
        criterion: Criterion,
        num_features: usize,
        feature_preprocessing: Vec<FeaturePreprocessing>,
        training_canaries: usize,
    ) -> Self {
        Self {
            mean,
            criterion,
            num_features,
            feature_preprocessing,
            training_canaries,
        }
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));

    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Model;
    use forestfire_data::DenseTable;

    #[test]
    fn trains_and_predicts_mean() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![10.0, 12.0, 14.0, 20.0];
        let table = DenseTable::new(x, y).unwrap();

        let model = train_target_mean(&table).unwrap();
        assert!((model.mean - 14.0).abs() < 1e-12);
        assert_eq!(model.criterion(), Criterion::Mean);

        let preds = model.predict_table(&table);
        assert_eq!(preds.len(), 4);
        for pred in preds {
            assert!((pred - 14.0).abs() < 1e-12);
        }
    }

    #[test]
    fn trains_and_predicts_median() {
        let table =
            DenseTable::new(vec![vec![0.0], vec![0.0], vec![0.0]], vec![1.0, 2.0, 100.0]).unwrap();

        let model = train_target_mean_with_criterion(&table, Criterion::Median).unwrap();

        assert_eq!(model.criterion(), Criterion::Median);
        assert_eq!(model.mean, 2.0);
        assert_eq!(model.predict_table(&table), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn rejects_empty_training_table() {
        let table = DenseTable::new(Vec::new(), Vec::new()).unwrap();

        let err = train_target_mean(&table).unwrap_err();
        assert!(matches!(err, ModelError::EmptyTarget));
    }

    #[test]
    fn manually_built_target_mean_model_serializes() {
        let model = Model::TargetMean(TargetMeanTree {
            mean: 3.5,
            criterion: Criterion::Mean,
            num_features: 2,
            feature_preprocessing: vec![
                FeaturePreprocessing::Binary,
                FeaturePreprocessing::Numeric {
                    bin_boundaries: vec![
                        crate::NumericBinBoundary {
                            bin: 0,
                            upper_bound: 1.0,
                        },
                        crate::NumericBinBoundary {
                            bin: 511,
                            upper_bound: 10.0,
                        },
                    ],
                },
            ],
            training_canaries: 1,
        });

        let json = model.serialize().unwrap();

        assert!(json.contains("\"tree_type\":\"target_mean\""));
        assert!(json.contains("\"canaries\":1"));
        assert!(json.contains("\"prediction_kind\":\"regression_value\""));
    }
}
