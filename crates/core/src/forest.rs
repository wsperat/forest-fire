use crate::ir::TrainingMetadata;
use crate::{Criterion, FeaturePreprocessing, Model, PredictError, Task, TreeType};
use forestfire_data::TableAccess;

#[derive(Debug, Clone)]
pub struct RandomForest {
    task: Task,
    criterion: Criterion,
    tree_type: TreeType,
    trees: Vec<Model>,
    num_features: usize,
    feature_preprocessing: Vec<FeaturePreprocessing>,
}

impl RandomForest {
    pub fn new(
        task: Task,
        criterion: Criterion,
        tree_type: TreeType,
        trees: Vec<Model>,
        num_features: usize,
        feature_preprocessing: Vec<FeaturePreprocessing>,
    ) -> Self {
        Self {
            task,
            criterion,
            tree_type,
            trees,
            num_features,
            feature_preprocessing,
        }
    }

    pub fn predict_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        match self.task {
            Task::Regression => self.predict_regression_table(table),
            Task::Classification => self.predict_classification_table(table),
        }
    }

    pub fn predict_proba_table(
        &self,
        table: &dyn TableAccess,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        if self.task != Task::Classification {
            return Err(PredictError::ProbabilityPredictionRequiresClassification);
        }

        let mut totals = self.trees[0].predict_proba_table(table)?;
        for tree in &self.trees[1..] {
            let probs = tree.predict_proba_table(table)?;
            for (row_totals, row_probs) in totals.iter_mut().zip(probs.iter()) {
                for (total, prob) in row_totals.iter_mut().zip(row_probs.iter()) {
                    *total += *prob;
                }
            }
        }

        let tree_count = self.trees.len() as f64;
        for row in &mut totals {
            for value in row {
                *value /= tree_count;
            }
        }

        Ok(totals)
    }

    pub fn task(&self) -> Task {
        self.task
    }

    pub fn criterion(&self) -> Criterion {
        self.criterion
    }

    pub fn tree_type(&self) -> TreeType {
        self.tree_type
    }

    pub fn trees(&self) -> &[Model] {
        &self.trees
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn feature_preprocessing(&self) -> &[FeaturePreprocessing] {
        &self.feature_preprocessing
    }

    pub fn training_metadata(&self) -> TrainingMetadata {
        let mut metadata = self.trees[0].training_metadata();
        metadata.algorithm = "rf".to_string();
        metadata
    }

    pub fn class_labels(&self) -> Option<Vec<f64>> {
        match self.task {
            Task::Classification => self.trees[0].class_labels(),
            Task::Regression => None,
        }
    }

    fn predict_regression_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        let mut totals = self.trees[0].predict_table(table);
        for tree in &self.trees[1..] {
            let preds = tree.predict_table(table);
            for (total, pred) in totals.iter_mut().zip(preds.iter()) {
                *total += *pred;
            }
        }

        let tree_count = self.trees.len() as f64;
        for value in &mut totals {
            *value /= tree_count;
        }

        totals
    }

    fn predict_classification_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        let probabilities = self
            .predict_proba_table(table)
            .expect("classification forest supports probabilities");
        let class_labels = self
            .class_labels()
            .expect("classification forest stores class labels");

        probabilities
            .into_iter()
            .map(|row| {
                let (best_index, _) = row
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|(left_index, left), (right_index, right)| {
                        left.total_cmp(right)
                            .then_with(|| right_index.cmp(left_index))
                    })
                    .expect("classification probability row is non-empty");
                class_labels[best_index]
            })
            .collect()
    }
}
