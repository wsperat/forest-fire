use forestfire_data::DenseTable;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// The simplest possible "tree":
/// it stores only the global mean of y and predicts that for every row.
#[derive(Debug, Clone, Copy)]
pub struct TargetMeanTree {
    pub mean: f64,
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

pub fn train_target_mean(train_set: &DenseTable) -> Result<TargetMeanTree, ModelError> {
    if train_set.n_rows() == 0 {
        return Err(ModelError::EmptyTarget);
    }

    let sum: f64 = (0..train_set.n_rows())
        .map(|row_idx| train_set.target().value(row_idx))
        .sum();

    Ok(TargetMeanTree {
        mean: sum / train_set.n_rows() as f64,
    })
}

impl TargetMeanTree {
    /// Predict the constant mean for `n` samples.
    pub fn predict_many(&self, n: usize) -> Vec<f64> {
        vec![self.mean; n]
    }

    /// Convenience: predict for a table (ignores features).
    pub fn predict_table(&self, table: &DenseTable) -> Vec<f64> {
        self.predict_many(table.n_rows())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forestfire_data::DenseTable;

    #[test]
    fn trains_and_predicts_mean() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![10.0, 12.0, 14.0, 20.0];
        let table = DenseTable::new(x, y).unwrap();

        let model = train_target_mean(&table).unwrap();
        assert!((model.mean - 14.0).abs() < 1e-12);

        let preds = model.predict_table(&table);
        assert_eq!(preds.len(), 4);
        for pred in preds {
            assert!((pred - 14.0).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_empty_training_table() {
        let table = DenseTable::new(Vec::new(), Vec::new()).unwrap();

        let err = train_target_mean(&table).unwrap_err();
        assert!(matches!(err, ModelError::EmptyTarget));
    }
}
