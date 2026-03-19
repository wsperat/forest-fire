use forestfire_data::DenseTable;

/// Common interface for regression models.
pub trait Regressor {
    /// Predict for `n_rows` rows (feature-agnostic).
    fn predict_rows(&self, n_rows: usize) -> Vec<f64>;

    /// Predict for a table; default delegates to `predict_rows`.
    fn predict_table(&self, table: &DenseTable) -> Vec<f64> {
        self.predict_rows(table.n_rows())
    }
}

// Hook up the existing mean model.
impl Regressor for forestfire_core::TargetMeanTree {
    fn predict_rows(&self, n_rows: usize) -> Vec<f64> {
        // Call the inherent method explicitly to avoid trait recursion.
        forestfire_core::TargetMeanTree::predict_many(self, n_rows)
    }

    fn predict_table(&self, table: &DenseTable) -> Vec<f64> {
        forestfire_core::TargetMeanTree::predict_table(self, table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forestfire_core::TargetMeanTree;
    use forestfire_core::train;

    #[test]
    fn trait_predictions_match_inherent_methods() {
        let table = forestfire_data::DenseTable::new(
            vec![vec![0.0], vec![1.0], vec![2.0]],
            vec![2.0, 4.0, 6.0],
        )
        .unwrap();

        let m = train(&table).unwrap();
        // trait path
        let via_trait = <TargetMeanTree as Regressor>::predict_table(&m, &table);
        // inherent path
        let via_inherent = m.predict_table(&table);

        assert_eq!(via_trait, via_inherent);
        assert!(via_trait.iter().all(|&p| (p - m.mean).abs() < 1e-12));
    }
}
