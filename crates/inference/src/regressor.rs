use forestfire_data::TableAccess;

/// Common interface for regression models.
///
/// The trait is intentionally tiny: it captures the one operation inference code
/// usually needs while allowing different model backends to implement their own
/// storage and traversal details.
pub trait Regressor {
    /// Predict for `n_rows` rows (feature-agnostic).
    fn predict_rows(&self, n_rows: usize) -> Vec<f64>;

    /// Predict for a table; default delegates to `predict_rows`.
    fn predict_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        self.predict_rows(table.n_rows())
    }
}

impl Regressor for forestfire_core::DecisionTreeRegressor {
    fn predict_rows(&self, _n_rows: usize) -> Vec<f64> {
        unreachable!("regression trees require feature data for prediction")
    }

    fn predict_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        forestfire_core::DecisionTreeRegressor::predict_table(self, table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forestfire_core::{Model, TrainConfig, TreeType, train};
    use forestfire_data::NumericBins;

    #[test]
    fn regression_tree_trait_predictions_match_inherent_methods() {
        let table = forestfire_data::DenseTable::with_options(
            vec![
                vec![0.0],
                vec![1.0],
                vec![2.0],
                vec![3.0],
                vec![4.0],
                vec![5.0],
                vec![6.0],
                vec![7.0],
            ],
            vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0],
            0,
            NumericBins::Fixed(64),
        )
        .unwrap();

        let m = train(
            &table,
            TrainConfig {
                tree_type: TreeType::Cart,
                ..TrainConfig::default()
            },
        )
        .unwrap();
        let Model::DecisionTreeRegressor(m) = m else {
            panic!("expected decision tree regressor");
        };
        let via_trait =
            <forestfire_core::DecisionTreeRegressor as Regressor>::predict_table(&m, &table);
        let via_inherent = m.predict_table(&table);

        assert_eq!(via_trait, via_inherent);
        assert_eq!(via_trait, vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0]);
    }
}
