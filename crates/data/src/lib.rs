/// Minimal dense dataset for tabular regression.
/// Features are unused by the mean model, but we keep them so the API matches future models.
#[derive(Debug, Clone)]
pub struct DenseDataset {
    pub x: Vec<Vec<f64>>, // shape: [n_samples][n_features]
    pub y: Vec<f64>,      // shape: [n_samples]
}

impl DenseDataset {
    pub fn new(x: Vec<Vec<f64>>, y: Vec<f64>) -> Result<Self, String> {
        if x.len() != y.len() {
            return Err(format!(
                "Mismatched lengths: X has {} rows, y has {}",
                x.len(),
                y.len()
            ));
        }
        Ok(Self { x, y })
    }

    #[inline]
    pub fn n_samples(&self) -> usize {
        self.x.len()
    }
}
