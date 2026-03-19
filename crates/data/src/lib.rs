use arrow::array::{Float64Array, UInt16Array};
use std::error::Error;
use std::fmt::{Display, Formatter};

const NUMERIC_BINS: usize = 512;

/// Arrow-backed dense table for tabular regression/classification data.
#[derive(Debug, Clone)]
pub struct DenseTable {
    feature_columns: Vec<Float64Array>,
    binned_feature_columns: Vec<UInt16Array>,
    target: Float64Array,
    n_rows: usize,
    n_features: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseTableError {
    MismatchedLengths {
        x: usize,
        y: usize,
    },
    RaggedRows {
        row: usize,
        expected: usize,
        actual: usize,
    },
}

impl Display for DenseTableError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DenseTableError::MismatchedLengths { x, y } => write!(
                f,
                "Mismatched lengths: X has {} rows while y has {} values.",
                x, y
            ),
            DenseTableError::RaggedRows {
                row,
                expected,
                actual,
            } => write!(
                f,
                "Ragged row at index {}: expected {} columns, found {}.",
                row, expected, actual
            ),
        }
    }
}

impl Error for DenseTableError {}

impl DenseTable {
    pub fn new(x: Vec<Vec<f64>>, y: Vec<f64>) -> Result<Self, DenseTableError> {
        if x.len() != y.len() {
            return Err(DenseTableError::MismatchedLengths {
                x: x.len(),
                y: y.len(),
            });
        }

        let n_rows = x.len();
        let n_features = x.first().map_or(0, Vec::len);

        for (row_idx, row) in x.iter().enumerate() {
            if row.len() != n_features {
                return Err(DenseTableError::RaggedRows {
                    row: row_idx,
                    expected: n_features,
                    actual: row.len(),
                });
            }
        }

        let mut columns = vec![Vec::with_capacity(n_rows); n_features];
        for row in &x {
            for (col_idx, value) in row.iter().enumerate() {
                columns[col_idx].push(*value);
            }
        }

        let feature_columns = columns
            .iter()
            .map(|column| Float64Array::from(column.clone()))
            .collect();
        let binned_feature_columns = columns
            .iter()
            .map(|column| UInt16Array::from(bin_numeric_column(column)))
            .collect();

        Ok(Self {
            feature_columns,
            binned_feature_columns,
            target: Float64Array::from(y),
            n_rows,
            n_features,
        })
    }

    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    #[inline]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    #[inline]
    pub fn feature_column(&self, index: usize) -> &Float64Array {
        &self.feature_columns[index]
    }

    #[inline]
    pub fn binned_feature_column(&self, index: usize) -> &UInt16Array {
        &self.binned_feature_columns[index]
    }

    #[inline]
    pub fn target(&self) -> &Float64Array {
        &self.target
    }
}

fn bin_numeric_column(values: &[f64]) -> Vec<u16> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut ranked_values: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    ranked_values.sort_by(|left, right| left.1.total_cmp(&right.1));

    let mut bins = vec![0u16; values.len()];
    let max_bin = (NUMERIC_BINS - 1) as u16;

    for (rank, (row_idx, _value)) in ranked_values.into_iter().enumerate() {
        let bin = if values.len() == 1 {
            0
        } else {
            ((rank * usize::from(max_bin)) / (values.len() - 1)) as u16
        };
        bins[row_idx] = bin;
    }

    bins
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_arrow_backed_dense_table() {
        let table =
            DenseTable::new(vec![vec![0.0, 10.0], vec![1.0, 20.0]], vec![3.0, 5.0]).unwrap();

        assert_eq!(table.n_rows(), 2);
        assert_eq!(table.n_features(), 2);
        assert_eq!(table.feature_column(0).value(0), 0.0);
        assert_eq!(table.feature_column(0).value(1), 1.0);
        assert_eq!(table.target().value(0), 3.0);
        assert_eq!(table.target().value(1), 5.0);
    }

    #[test]
    fn bins_numeric_columns_into_512_rank_bins() {
        use std::collections::BTreeSet;

        let x: Vec<Vec<f64>> = (0..1024).map(|value| vec![value as f64]).collect();
        let y: Vec<f64> = vec![1.0; 1024];

        let table = DenseTable::new(x, y).unwrap();
        let bins = table.binned_feature_column(0);

        assert_eq!(bins.value(0), 0);
        assert_eq!(bins.value(1023), 511);
        assert!((1..1024).all(|idx| bins.value(idx - 1) <= bins.value(idx)));
        assert_eq!(
            (0..1024)
                .map(|idx| bins.value(idx))
                .collect::<BTreeSet<_>>()
                .len(),
            512
        );
    }

    #[test]
    fn rejects_ragged_rows() {
        let err = DenseTable::new(vec![vec![1.0, 2.0], vec![3.0]], vec![1.0, 2.0]).unwrap_err();

        assert_eq!(
            err,
            DenseTableError::RaggedRows {
                row: 1,
                expected: 2,
                actual: 1,
            }
        );
    }
}
