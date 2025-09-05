use forestfire_data::DenseDataset;
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
    MismatchedLengths { x: usize, y: usize },
}

impl Display for ModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::EmptyTarget => write!(f, "Cannot train on an empty target vector."),
            ModelError::MismatchedLengths { x, y } => write!(
                f,
                "Mismatched lengths: X has {} rows while y has {} values.",
                x, y
            ),
        }
    }
}
impl Error for ModelError {}

impl TargetMeanTree {
    /// Train from a DenseDataset (uses y only).
    pub fn train(ds: &DenseDataset) -> Result<Self, ModelError> {
        if ds.n_samples() == 0 {
            return Err(ModelError::EmptyTarget);
        }
        if ds.x.len() != ds.y.len() {
            return Err(ModelError::MismatchedLengths {
                x: ds.x.len(),
                y: ds.y.len(),
            });
        }
        let sum: f64 = ds.y.iter().copied().sum();
        Ok(Self {
            mean: sum / ds.y.len() as f64,
        })
    }

    /// Predict the constant mean for `n` samples.
    pub fn predict_many(&self, n: usize) -> Vec<f64> {
        vec![self.mean; n]
    }

    /// Convenience: predict for a dataset (ignores features).
    pub fn predict_dataset(&self, ds: &DenseDataset) -> Vec<f64> {
        self.predict_many(ds.n_samples())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forestfire_data::DenseDataset;

    #[test]
    fn trains_and_predicts_mean() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![10.0, 12.0, 14.0, 20.0];
        let ds = DenseDataset::new(x, y).unwrap();

        let model = TargetMeanTree::train(&ds).unwrap();
        assert!((model.mean - 14.0).abs() < 1e-12);

        let preds = model.predict_dataset(&ds);
        assert_eq!(preds.len(), 4);
        for p in preds {
            assert!((p - 14.0).abs() < 1e-12);
        }
    }
}
