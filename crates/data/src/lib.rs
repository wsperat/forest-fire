use arrow::array::{BooleanArray, Float64Array, UInt16Array};
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};
use std::cmp::Ordering;
use std::error::Error;
use std::fmt::{Display, Formatter};

const NUMERIC_BINS: usize = 512;
const DEFAULT_CANARIES: usize = 2;

type PreprocessedRows = (Vec<Vec<f64>>, Float64Array, usize, usize);

pub trait TableAccess: Sync {
    fn n_rows(&self) -> usize;
    fn n_features(&self) -> usize;
    fn canaries(&self) -> usize;
    fn binned_feature_count(&self) -> usize;
    fn feature_value(&self, feature_index: usize, row_index: usize) -> f64;
    fn is_binary_feature(&self, index: usize) -> bool;
    fn binned_value(&self, feature_index: usize, row_index: usize) -> u16;
    fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool>;
    fn binned_column_kind(&self, index: usize) -> BinnedColumnKind;
    fn is_binary_binned_feature(&self, index: usize) -> bool;
    fn target_value(&self, row_index: usize) -> f64;

    fn is_canary_binned_feature(&self, index: usize) -> bool {
        matches!(
            self.binned_column_kind(index),
            BinnedColumnKind::Canary { .. }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableKind {
    Dense,
    Sparse,
}

/// Arrow-backed dense table for tabular regression/classification data.
#[derive(Debug, Clone)]
pub struct DenseTable {
    feature_columns: Vec<FeatureColumn>,
    binned_feature_columns: Vec<BinnedFeatureColumn>,
    binned_column_kinds: Vec<BinnedColumnKind>,
    target: Float64Array,
    n_rows: usize,
    n_features: usize,
    canaries: usize,
}

/// Arrow-backed sparse table specialized for binary feature matrices.
#[derive(Debug, Clone)]
pub struct SparseTable {
    feature_columns: Vec<BooleanArray>,
    binned_feature_columns: Vec<BooleanArray>,
    binned_column_kinds: Vec<BinnedColumnKind>,
    target: Float64Array,
    n_rows: usize,
    n_features: usize,
    canaries: usize,
}

#[derive(Debug, Clone)]
pub enum Table {
    Dense(DenseTable),
    Sparse(SparseTable),
}

#[derive(Debug, Clone)]
enum FeatureColumn {
    Numeric(Float64Array),
    Binary(BooleanArray),
}

#[derive(Debug, Clone)]
enum BinnedFeatureColumn {
    Numeric(UInt16Array),
    Binary(BooleanArray),
}

#[derive(Debug, Clone, Copy)]
pub enum FeatureColumnRef<'a> {
    Numeric(&'a Float64Array),
    Binary(&'a BooleanArray),
}

#[derive(Debug, Clone, Copy)]
pub enum BinnedFeatureColumnRef<'a> {
    Numeric(&'a UInt16Array),
    Binary(&'a BooleanArray),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinnedColumnKind {
    Real {
        source_index: usize,
    },
    Canary {
        source_index: usize,
        copy_index: usize,
    },
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
    NonBinaryColumn {
        column: usize,
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
            DenseTableError::NonBinaryColumn { column } => write!(
                f,
                "SparseTable requires binary features, but column {} contains non-binary values.",
                column
            ),
        }
    }
}

impl Error for DenseTableError {}

impl DenseTable {
    pub fn new(x: Vec<Vec<f64>>, y: Vec<f64>) -> Result<Self, DenseTableError> {
        Self::with_canaries(x, y, DEFAULT_CANARIES)
    }

    pub fn with_canaries(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        canaries: usize,
    ) -> Result<Self, DenseTableError> {
        let (columns, target, n_rows, n_features) = preprocess_rows(&x, y)?;
        Ok(Self::from_columns(
            &columns, target, n_rows, n_features, canaries,
        ))
    }

    fn from_columns(
        columns: &[Vec<f64>],
        target: Float64Array,
        n_rows: usize,
        n_features: usize,
        canaries: usize,
    ) -> Self {
        let feature_columns = columns
            .iter()
            .map(|column| build_feature_column(column))
            .collect();

        let real_binned_columns: Vec<BinnedFeatureColumn> = columns
            .iter()
            .map(|column| build_binned_feature_column(column))
            .collect();
        let canary_columns: Vec<(BinnedColumnKind, BinnedFeatureColumn)> = (0..canaries)
            .flat_map(|copy_index| {
                real_binned_columns
                    .iter()
                    .enumerate()
                    .map(move |(source_index, column)| {
                        (
                            BinnedColumnKind::Canary {
                                source_index,
                                copy_index,
                            },
                            shuffle_canary_column(column, copy_index, source_index),
                        )
                    })
            })
            .collect();

        let (binned_column_kinds, binned_feature_columns): (Vec<_>, Vec<_>) = (0..n_features)
            .map(|source_index| BinnedColumnKind::Real { source_index })
            .zip(real_binned_columns)
            .chain(canary_columns)
            .unzip();

        Self {
            feature_columns,
            binned_feature_columns,
            binned_column_kinds,
            target,
            n_rows,
            n_features,
            canaries,
        }
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
    pub fn canaries(&self) -> usize {
        self.canaries
    }

    #[inline]
    pub fn binned_feature_count(&self) -> usize {
        self.binned_feature_columns.len()
    }

    #[inline]
    pub fn feature_column(&self, index: usize) -> FeatureColumnRef<'_> {
        match &self.feature_columns[index] {
            FeatureColumn::Numeric(column) => FeatureColumnRef::Numeric(column),
            FeatureColumn::Binary(column) => FeatureColumnRef::Binary(column),
        }
    }

    #[inline]
    pub fn feature_value(&self, feature_index: usize, row_index: usize) -> f64 {
        match &self.feature_columns[feature_index] {
            FeatureColumn::Numeric(column) => column.value(row_index),
            FeatureColumn::Binary(column) => f64::from(u8::from(column.value(row_index))),
        }
    }

    #[inline]
    pub fn is_binary_feature(&self, index: usize) -> bool {
        matches!(self.feature_columns[index], FeatureColumn::Binary(_))
    }

    #[inline]
    pub fn binned_feature_column(&self, index: usize) -> BinnedFeatureColumnRef<'_> {
        match &self.binned_feature_columns[index] {
            BinnedFeatureColumn::Numeric(column) => BinnedFeatureColumnRef::Numeric(column),
            BinnedFeatureColumn::Binary(column) => BinnedFeatureColumnRef::Binary(column),
        }
    }

    #[inline]
    pub fn binned_value(&self, feature_index: usize, row_index: usize) -> u16 {
        match &self.binned_feature_columns[feature_index] {
            BinnedFeatureColumn::Numeric(column) => column.value(row_index),
            BinnedFeatureColumn::Binary(column) => u16::from(u8::from(column.value(row_index))),
        }
    }

    #[inline]
    pub fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool> {
        match &self.binned_feature_columns[feature_index] {
            BinnedFeatureColumn::Binary(column) => Some(column.value(row_index)),
            BinnedFeatureColumn::Numeric(_) => None,
        }
    }

    #[inline]
    pub fn binned_column_kind(&self, index: usize) -> BinnedColumnKind {
        self.binned_column_kinds[index]
    }

    #[inline]
    pub fn is_canary_binned_feature(&self, index: usize) -> bool {
        matches!(
            self.binned_column_kinds[index],
            BinnedColumnKind::Canary { .. }
        )
    }

    #[inline]
    pub fn is_binary_binned_feature(&self, index: usize) -> bool {
        matches!(
            self.binned_feature_columns[index],
            BinnedFeatureColumn::Binary(_)
        )
    }

    #[inline]
    pub fn target(&self) -> &Float64Array {
        &self.target
    }
}

impl SparseTable {
    pub fn new(x: Vec<Vec<f64>>, y: Vec<f64>) -> Result<Self, DenseTableError> {
        Self::with_canaries(x, y, DEFAULT_CANARIES)
    }

    pub fn with_canaries(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        canaries: usize,
    ) -> Result<Self, DenseTableError> {
        let (columns, target, n_rows, n_features) = preprocess_rows(&x, y)?;
        validate_binary_columns(&columns)?;
        Ok(Self::from_columns(
            &columns, target, n_rows, n_features, canaries,
        ))
    }

    fn from_columns(
        columns: &[Vec<f64>],
        target: Float64Array,
        n_rows: usize,
        n_features: usize,
        canaries: usize,
    ) -> Self {
        let feature_columns: Vec<BooleanArray> = columns
            .iter()
            .map(|column| BooleanArray::from(to_binary_values(column)))
            .collect();

        let canary_columns: Vec<(BinnedColumnKind, BooleanArray)> = (0..canaries)
            .flat_map(|copy_index| {
                feature_columns
                    .iter()
                    .enumerate()
                    .map(move |(source_index, column)| {
                        (
                            BinnedColumnKind::Canary {
                                source_index,
                                copy_index,
                            },
                            shuffle_boolean_array(column, copy_index, source_index),
                        )
                    })
            })
            .collect();

        let (binned_column_kinds, binned_feature_columns): (Vec<_>, Vec<_>) = (0..n_features)
            .map(|source_index| BinnedColumnKind::Real { source_index })
            .zip(feature_columns.iter().cloned())
            .chain(canary_columns)
            .unzip();

        Self {
            feature_columns,
            binned_feature_columns,
            binned_column_kinds,
            target,
            n_rows,
            n_features,
            canaries,
        }
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
    pub fn canaries(&self) -> usize {
        self.canaries
    }

    #[inline]
    pub fn binned_feature_count(&self) -> usize {
        self.binned_feature_columns.len()
    }

    #[inline]
    pub fn feature_value(&self, feature_index: usize, row_index: usize) -> f64 {
        f64::from(u8::from(
            self.feature_columns[feature_index].value(row_index),
        ))
    }

    #[inline]
    pub fn is_binary_feature(&self, _index: usize) -> bool {
        true
    }

    #[inline]
    pub fn binned_value(&self, feature_index: usize, row_index: usize) -> u16 {
        u16::from(u8::from(
            self.binned_feature_columns[feature_index].value(row_index),
        ))
    }

    #[inline]
    pub fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool> {
        Some(self.binned_feature_columns[feature_index].value(row_index))
    }

    #[inline]
    pub fn binned_column_kind(&self, index: usize) -> BinnedColumnKind {
        self.binned_column_kinds[index]
    }

    #[inline]
    pub fn is_canary_binned_feature(&self, index: usize) -> bool {
        matches!(
            self.binned_column_kinds[index],
            BinnedColumnKind::Canary { .. }
        )
    }

    #[inline]
    pub fn is_binary_binned_feature(&self, _index: usize) -> bool {
        true
    }

    #[inline]
    pub fn target(&self) -> &Float64Array {
        &self.target
    }
}

impl Table {
    pub fn new(x: Vec<Vec<f64>>, y: Vec<f64>) -> Result<Self, DenseTableError> {
        Self::with_canaries(x, y, DEFAULT_CANARIES)
    }

    pub fn with_canaries(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        canaries: usize,
    ) -> Result<Self, DenseTableError> {
        let (columns, target, n_rows, n_features) = preprocess_rows(&x, y)?;

        if columns.iter().all(|column| is_binary_column(column)) {
            Ok(Self::Sparse(SparseTable::from_columns(
                &columns, target, n_rows, n_features, canaries,
            )))
        } else {
            Ok(Self::Dense(DenseTable::from_columns(
                &columns, target, n_rows, n_features, canaries,
            )))
        }
    }

    pub fn kind(&self) -> TableKind {
        match self {
            Table::Dense(_) => TableKind::Dense,
            Table::Sparse(_) => TableKind::Sparse,
        }
    }

    pub fn as_dense(&self) -> Option<&DenseTable> {
        match self {
            Table::Dense(table) => Some(table),
            Table::Sparse(_) => None,
        }
    }

    pub fn as_sparse(&self) -> Option<&SparseTable> {
        match self {
            Table::Dense(_) => None,
            Table::Sparse(table) => Some(table),
        }
    }
}

impl TableAccess for DenseTable {
    fn n_rows(&self) -> usize {
        self.n_rows()
    }

    fn n_features(&self) -> usize {
        self.n_features()
    }

    fn canaries(&self) -> usize {
        self.canaries()
    }

    fn binned_feature_count(&self) -> usize {
        self.binned_feature_count()
    }

    fn feature_value(&self, feature_index: usize, row_index: usize) -> f64 {
        self.feature_value(feature_index, row_index)
    }

    fn is_binary_feature(&self, index: usize) -> bool {
        self.is_binary_feature(index)
    }

    fn binned_value(&self, feature_index: usize, row_index: usize) -> u16 {
        self.binned_value(feature_index, row_index)
    }

    fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool> {
        self.binned_boolean_value(feature_index, row_index)
    }

    fn binned_column_kind(&self, index: usize) -> BinnedColumnKind {
        self.binned_column_kind(index)
    }

    fn is_binary_binned_feature(&self, index: usize) -> bool {
        self.is_binary_binned_feature(index)
    }

    fn target_value(&self, row_index: usize) -> f64 {
        self.target().value(row_index)
    }
}

impl TableAccess for SparseTable {
    fn n_rows(&self) -> usize {
        self.n_rows()
    }

    fn n_features(&self) -> usize {
        self.n_features()
    }

    fn canaries(&self) -> usize {
        self.canaries()
    }

    fn binned_feature_count(&self) -> usize {
        self.binned_feature_count()
    }

    fn feature_value(&self, feature_index: usize, row_index: usize) -> f64 {
        self.feature_value(feature_index, row_index)
    }

    fn is_binary_feature(&self, index: usize) -> bool {
        self.is_binary_feature(index)
    }

    fn binned_value(&self, feature_index: usize, row_index: usize) -> u16 {
        self.binned_value(feature_index, row_index)
    }

    fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool> {
        self.binned_boolean_value(feature_index, row_index)
    }

    fn binned_column_kind(&self, index: usize) -> BinnedColumnKind {
        self.binned_column_kind(index)
    }

    fn is_binary_binned_feature(&self, index: usize) -> bool {
        self.is_binary_binned_feature(index)
    }

    fn target_value(&self, row_index: usize) -> f64 {
        self.target().value(row_index)
    }
}

impl TableAccess for Table {
    fn n_rows(&self) -> usize {
        match self {
            Table::Dense(table) => table.n_rows(),
            Table::Sparse(table) => table.n_rows(),
        }
    }

    fn n_features(&self) -> usize {
        match self {
            Table::Dense(table) => table.n_features(),
            Table::Sparse(table) => table.n_features(),
        }
    }

    fn canaries(&self) -> usize {
        match self {
            Table::Dense(table) => table.canaries(),
            Table::Sparse(table) => table.canaries(),
        }
    }

    fn binned_feature_count(&self) -> usize {
        match self {
            Table::Dense(table) => table.binned_feature_count(),
            Table::Sparse(table) => table.binned_feature_count(),
        }
    }

    fn feature_value(&self, feature_index: usize, row_index: usize) -> f64 {
        match self {
            Table::Dense(table) => table.feature_value(feature_index, row_index),
            Table::Sparse(table) => table.feature_value(feature_index, row_index),
        }
    }

    fn is_binary_feature(&self, index: usize) -> bool {
        match self {
            Table::Dense(table) => table.is_binary_feature(index),
            Table::Sparse(table) => table.is_binary_feature(index),
        }
    }

    fn binned_value(&self, feature_index: usize, row_index: usize) -> u16 {
        match self {
            Table::Dense(table) => table.binned_value(feature_index, row_index),
            Table::Sparse(table) => table.binned_value(feature_index, row_index),
        }
    }

    fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool> {
        match self {
            Table::Dense(table) => table.binned_boolean_value(feature_index, row_index),
            Table::Sparse(table) => table.binned_boolean_value(feature_index, row_index),
        }
    }

    fn binned_column_kind(&self, index: usize) -> BinnedColumnKind {
        match self {
            Table::Dense(table) => table.binned_column_kind(index),
            Table::Sparse(table) => table.binned_column_kind(index),
        }
    }

    fn is_binary_binned_feature(&self, index: usize) -> bool {
        match self {
            Table::Dense(table) => table.is_binary_binned_feature(index),
            Table::Sparse(table) => table.is_binary_binned_feature(index),
        }
    }

    fn target_value(&self, row_index: usize) -> f64 {
        match self {
            Table::Dense(table) => table.target().value(row_index),
            Table::Sparse(table) => table.target().value(row_index),
        }
    }
}

fn preprocess_rows(x: &[Vec<f64>], y: Vec<f64>) -> Result<PreprocessedRows, DenseTableError> {
    validate_shape(x, &y)?;
    let n_rows = x.len();
    let n_features = x.first().map_or(0, Vec::len);
    let columns = collect_columns(x, n_features);
    Ok((columns, Float64Array::from(y), n_rows, n_features))
}

fn validate_shape(x: &[Vec<f64>], y: &[f64]) -> Result<(), DenseTableError> {
    if x.len() != y.len() {
        return Err(DenseTableError::MismatchedLengths {
            x: x.len(),
            y: y.len(),
        });
    }

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

    Ok(())
}

fn collect_columns(x: &[Vec<f64>], n_features: usize) -> Vec<Vec<f64>> {
    (0..n_features)
        .map(|col_idx| x.iter().map(|row| row[col_idx]).collect())
        .collect()
}

fn validate_binary_columns(columns: &[Vec<f64>]) -> Result<(), DenseTableError> {
    for (column_idx, column) in columns.iter().enumerate() {
        if !is_binary_column(column) {
            return Err(DenseTableError::NonBinaryColumn { column: column_idx });
        }
    }

    Ok(())
}

fn build_feature_column(values: &[f64]) -> FeatureColumn {
    if is_binary_column(values) {
        FeatureColumn::Binary(BooleanArray::from(to_binary_values(values)))
    } else {
        FeatureColumn::Numeric(Float64Array::from(values.to_vec()))
    }
}

fn build_binned_feature_column(values: &[f64]) -> BinnedFeatureColumn {
    if is_binary_column(values) {
        BinnedFeatureColumn::Binary(BooleanArray::from(to_binary_values(values)))
    } else {
        BinnedFeatureColumn::Numeric(UInt16Array::from(bin_numeric_column(values)))
    }
}

fn is_binary_column(values: &[f64]) -> bool {
    values.iter().all(|value| {
        matches!(value.total_cmp(&0.0), Ordering::Equal)
            || matches!(value.total_cmp(&1.0), Ordering::Equal)
    })
}

fn to_binary_values(values: &[f64]) -> Vec<bool> {
    values
        .iter()
        .map(|value| value.total_cmp(&1.0) == Ordering::Equal)
        .collect()
}

fn bin_numeric_column(values: &[f64]) -> Vec<u16> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut ranked_values: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    ranked_values.sort_by(|left, right| left.1.total_cmp(&right.1));

    let unique_value_count = ranked_values
        .iter()
        .map(|(_row_idx, value)| *value)
        .fold(Vec::<f64>::new(), |mut unique_values, value| {
            let is_new_value = unique_values
                .last()
                .is_none_or(|last_value| last_value.total_cmp(&value) != Ordering::Equal);
            if is_new_value {
                unique_values.push(value);
            }
            unique_values
        })
        .len();

    let mut bins = vec![0u16; values.len()];
    let max_bin = (NUMERIC_BINS - 1) as u16;
    let mut unique_rank = 0usize;
    let mut start = 0usize;

    while start < ranked_values.len() {
        let current_value = ranked_values[start].1;
        let end = ranked_values[start..]
            .iter()
            .position(|(_row_idx, value)| value.total_cmp(&current_value) != Ordering::Equal)
            .map_or(ranked_values.len(), |offset| start + offset);

        let bin = if unique_value_count == 1 {
            0
        } else {
            ((unique_rank * usize::from(max_bin)) / (unique_value_count - 1)) as u16
        };

        for (row_idx, _value) in &ranked_values[start..end] {
            bins[*row_idx] = bin;
        }

        unique_rank += 1;
        start = end;
    }

    bins
}

fn shuffle_canary_column(
    values: &BinnedFeatureColumn,
    copy_index: usize,
    source_index: usize,
) -> BinnedFeatureColumn {
    match values {
        BinnedFeatureColumn::Numeric(values) => {
            let mut shuffled = (0..values.len())
                .map(|idx| values.value(idx))
                .collect::<Vec<_>>();
            shuffle_values(&mut shuffled, copy_index, source_index);
            BinnedFeatureColumn::Numeric(UInt16Array::from(shuffled))
        }
        BinnedFeatureColumn::Binary(values) => {
            BinnedFeatureColumn::Binary(shuffle_boolean_array(values, copy_index, source_index))
        }
    }
}

fn shuffle_boolean_array(
    values: &BooleanArray,
    copy_index: usize,
    source_index: usize,
) -> BooleanArray {
    let mut shuffled = (0..values.len())
        .map(|idx| values.value(idx))
        .collect::<Vec<_>>();
    shuffle_values(&mut shuffled, copy_index, source_index);
    BooleanArray::from(shuffled)
}

fn shuffle_values<T>(values: &mut [T], copy_index: usize, source_index: usize) {
    let seed = 0xA11CE5EED_u64
        ^ ((copy_index as u64) << 32)
        ^ (source_index as u64)
        ^ ((values.len() as u64) << 16);
    let mut rng = StdRng::seed_from_u64(seed);
    values.shuffle(&mut rng);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn builds_arrow_backed_dense_table() {
        let table =
            DenseTable::new(vec![vec![0.0, 10.0], vec![1.0, 20.0]], vec![3.0, 5.0]).unwrap();

        assert_eq!(table.n_rows(), 2);
        assert_eq!(table.n_features(), 2);
        assert_eq!(table.canaries(), 2);
        assert_eq!(table.binned_feature_count(), 6);
        assert_eq!(table.feature_value(0, 0), 0.0);
        assert_eq!(table.feature_value(0, 1), 1.0);
        assert_eq!(table.target().value(0), 3.0);
        assert_eq!(table.target().value(1), 5.0);
        assert!(!table.is_canary_binned_feature(0));
        assert!(table.is_canary_binned_feature(2));
    }

    #[test]
    fn builds_sparse_table_for_all_binary_features() {
        let table = Table::with_canaries(
            vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]],
            vec![0.0, 1.0, 1.0],
            1,
        )
        .unwrap();

        assert_eq!(table.kind(), TableKind::Sparse);
        assert!(table.is_binary_feature(0));
        assert!(table.is_binary_feature(1));
        assert!(table.is_binary_binned_feature(0));
        assert_eq!(table.binned_feature_count(), 4);
    }

    #[test]
    fn builds_dense_table_when_any_feature_is_non_binary() {
        let table = Table::with_canaries(
            vec![vec![0.0, 1.5], vec![1.0, 0.0], vec![1.0, 2.0]],
            vec![0.0, 1.0, 1.0],
            1,
        )
        .unwrap();

        assert_eq!(table.kind(), TableKind::Dense);
        assert!(table.is_binary_feature(0));
        assert!(!table.is_binary_feature(1));
    }

    #[test]
    fn sparse_table_rejects_non_binary_columns() {
        let err =
            SparseTable::with_canaries(vec![vec![0.0, 2.0], vec![1.0, 0.0]], vec![0.0, 1.0], 0)
                .unwrap_err();

        assert_eq!(err, DenseTableError::NonBinaryColumn { column: 1 });
    }

    #[test]
    fn bins_numeric_columns_into_512_rank_bins() {
        let x: Vec<Vec<f64>> = (0..1024).map(|value| vec![value as f64]).collect();
        let y: Vec<f64> = vec![1.0; 1024];

        let table = DenseTable::with_canaries(x, y, 0).unwrap();

        assert_eq!(table.binned_value(0, 0), 0);
        assert_eq!(table.binned_value(0, 1023), 511);
        assert!((1..1024).all(|idx| table.binned_value(0, idx - 1) <= table.binned_value(0, idx)));
        assert_eq!(
            (0..1024)
                .map(|idx| table.binned_value(0, idx))
                .collect::<BTreeSet<_>>()
                .len(),
            512
        );
    }

    #[test]
    fn keeps_equal_values_in_the_same_bin() {
        let table = DenseTable::with_canaries(
            vec![vec![0.0], vec![0.0], vec![1.0], vec![1.0], vec![2.0]],
            vec![0.0; 5],
            0,
        )
        .unwrap();

        assert_eq!(table.binned_value(0, 0), table.binned_value(0, 1));
        assert_eq!(table.binned_value(0, 2), table.binned_value(0, 3));
        assert!(table.binned_value(0, 1) < table.binned_value(0, 2));
    }

    #[test]
    fn stores_binary_columns_as_booleans() {
        let table = DenseTable::with_canaries(
            vec![vec![0.0, 2.0], vec![1.0, 3.0], vec![0.0, 4.0]],
            vec![0.0; 3],
            1,
        )
        .unwrap();

        assert!(table.is_binary_feature(0));
        assert!(!table.is_binary_feature(1));
        assert!(table.is_binary_binned_feature(0));
        assert!(!table.is_binary_binned_feature(1));
        assert!(table.is_binary_binned_feature(2));
        assert_eq!(table.feature_value(0, 0), 0.0);
        assert_eq!(table.feature_value(0, 1), 1.0);
        assert_eq!(table.binned_boolean_value(0, 0), Some(false));
        assert_eq!(table.binned_boolean_value(0, 1), Some(true));
    }

    #[test]
    fn creates_canary_columns_as_shuffled_binned_copies() {
        let table = DenseTable::with_canaries(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
            vec![0.0; 5],
            1,
        )
        .unwrap();

        assert!(matches!(
            table.binned_column_kind(1),
            BinnedColumnKind::Canary {
                source_index: 0,
                copy_index: 0
            }
        ));
        assert_eq!(
            (0..table.n_rows())
                .map(|idx| table.binned_value(0, idx))
                .collect::<BTreeSet<_>>(),
            (0..table.n_rows())
                .map(|idx| table.binned_value(1, idx))
                .collect::<BTreeSet<_>>()
        );
        assert_ne!(
            (0..table.n_rows())
                .map(|idx| table.binned_value(0, idx))
                .collect::<Vec<_>>(),
            (0..table.n_rows())
                .map(|idx| table.binned_value(1, idx))
                .collect::<Vec<_>>()
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

    #[test]
    fn rejects_mismatched_lengths() {
        let err = DenseTable::new(vec![vec![1.0], vec![2.0]], vec![1.0]).unwrap_err();

        assert_eq!(err, DenseTableError::MismatchedLengths { x: 2, y: 1 });
    }

    #[test]
    fn canary_generation_is_deterministic_for_identical_inputs() {
        let left = DenseTable::with_canaries(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
            vec![0.0; 5],
            2,
        )
        .unwrap();
        let right = DenseTable::with_canaries(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
            vec![0.0; 5],
            2,
        )
        .unwrap();

        let left_values = binned_snapshot(&left);
        let right_values = binned_snapshot(&right);

        assert_eq!(left_values, right_values);
    }

    #[test]
    fn binary_canaries_remain_boolean_and_preserve_value_counts() {
        let table = DenseTable::with_canaries(
            vec![
                vec![0.0],
                vec![1.0],
                vec![0.0],
                vec![1.0],
                vec![1.0],
                vec![0.0],
            ],
            vec![0.0; 6],
            2,
        )
        .unwrap();

        let real_true_count = (0..table.n_rows())
            .filter(|row_idx| table.binned_boolean_value(0, *row_idx) == Some(true))
            .count();

        for feature_index in 1..table.binned_feature_count() {
            assert!(table.is_binary_binned_feature(feature_index));
            let canary_true_count = (0..table.n_rows())
                .filter(|row_idx| table.binned_boolean_value(feature_index, *row_idx) == Some(true))
                .count();
            assert_eq!(canary_true_count, real_true_count);
        }
    }

    fn binned_snapshot(table: &DenseTable) -> Vec<u16> {
        let mut values = Vec::new();

        for feature_idx in 0..table.binned_feature_count() {
            for row_idx in 0..table.n_rows() {
                values.push(table.binned_value(feature_idx, row_idx));
            }
        }

        values
    }
}
