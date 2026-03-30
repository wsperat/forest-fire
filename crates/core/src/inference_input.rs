use super::*;

#[derive(Debug, Clone)]
enum InferenceFeatureColumn {
    Numeric(Vec<f64>),
    Binary(Vec<bool>),
}

#[derive(Debug, Clone)]
enum InferenceBinnedColumn {
    Numeric(Vec<u16>),
    Binary(Vec<bool>),
}

#[derive(Debug, Clone)]
pub(crate) enum CompactBinnedColumn {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl CompactBinnedColumn {
    #[inline(always)]
    pub(crate) fn value_at(&self, row_index: usize) -> u16 {
        match self {
            CompactBinnedColumn::U8(values) => u16::from(values[row_index]),
            CompactBinnedColumn::U16(values) => values[row_index],
        }
    }

    #[inline(always)]
    pub(crate) fn slice_u8(&self, start: usize, len: usize) -> Option<&[u8]> {
        match self {
            CompactBinnedColumn::U8(values) => Some(&values[start..start + len]),
            CompactBinnedColumn::U16(_) => None,
        }
    }

    #[inline(always)]
    pub(crate) fn slice_u16(&self, start: usize, len: usize) -> Option<&[u16]> {
        match self {
            CompactBinnedColumn::U8(_) => None,
            CompactBinnedColumn::U16(values) => Some(&values[start..start + len]),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct InferenceTable {
    feature_columns: Vec<InferenceFeatureColumn>,
    binned_feature_columns: Vec<InferenceBinnedColumn>,
    n_rows: usize,
}

impl InferenceTable {
    pub(crate) fn from_rows_projected(
        rows: Vec<Vec<f64>>,
        preprocessing: &[FeaturePreprocessing],
        projection: &[usize],
    ) -> Result<Self, PredictError> {
        let expected = preprocessing.len();
        if let Some((row_index, actual)) = rows
            .iter()
            .enumerate()
            .find_map(|(row_index, row)| (row.len() != expected).then_some((row_index, row.len())))
        {
            return Err(PredictError::RaggedRows {
                row: row_index,
                expected,
                actual,
            });
        }

        let columns = projection
            .iter()
            .map(|feature_index| {
                rows.iter()
                    .map(|row| row[*feature_index])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let projected_preprocessing = projection
            .iter()
            .map(|feature_index| preprocessing[*feature_index].clone())
            .collect::<Vec<_>>();

        Self::from_columns(columns, &projected_preprocessing, Some(rows.len()))
    }

    pub(crate) fn from_rows(
        rows: Vec<Vec<f64>>,
        preprocessing: &[FeaturePreprocessing],
    ) -> Result<Self, PredictError> {
        let expected = preprocessing.len();
        if let Some((row_index, actual)) = rows
            .iter()
            .enumerate()
            .find_map(|(row_index, row)| (row.len() != expected).then_some((row_index, row.len())))
        {
            return Err(PredictError::RaggedRows {
                row: row_index,
                expected,
                actual,
            });
        }

        let columns = (0..expected)
            .map(|feature_index| {
                rows.iter()
                    .map(|row| row[feature_index])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self::from_columns(columns, preprocessing, Some(rows.len()))
    }

    pub(crate) fn from_named_columns(
        columns: BTreeMap<String, Vec<f64>>,
        preprocessing: &[FeaturePreprocessing],
    ) -> Result<Self, PredictError> {
        let expected = preprocessing.len();
        if columns.len() != expected {
            for feature_index in 0..expected {
                let name = format!("f{}", feature_index);
                if !columns.contains_key(&name) {
                    return Err(PredictError::MissingFeature(name));
                }
            }
            if let Some(unexpected) = columns.keys().find(|name| {
                name.strip_prefix('f')
                    .and_then(|idx| idx.parse::<usize>().ok())
                    .is_none_or(|idx| idx >= expected)
            }) {
                return Err(PredictError::UnexpectedFeature(unexpected.clone()));
            }
        }

        let n_rows = columns.values().next().map_or(0, Vec::len);
        let ordered = (0..expected)
            .map(|feature_index| {
                let feature_name = format!("f{}", feature_index);
                let values = columns
                    .get(&feature_name)
                    .ok_or_else(|| PredictError::MissingFeature(feature_name.clone()))?;
                if values.len() != n_rows {
                    return Err(PredictError::ColumnLengthMismatch {
                        feature: feature_name,
                        expected: n_rows,
                        actual: values.len(),
                    });
                }
                Ok(values.clone())
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::from_columns(ordered, preprocessing, Some(n_rows))
    }

    pub(crate) fn from_named_columns_projected(
        columns: BTreeMap<String, Vec<f64>>,
        preprocessing: &[FeaturePreprocessing],
        projection: &[usize],
    ) -> Result<Self, PredictError> {
        let expected = preprocessing.len();
        if columns.len() != expected {
            for feature_index in 0..expected {
                let name = format!("f{}", feature_index);
                if !columns.contains_key(&name) {
                    return Err(PredictError::MissingFeature(name));
                }
            }
            if let Some(unexpected) = columns.keys().find(|name| {
                name.strip_prefix('f')
                    .and_then(|idx| idx.parse::<usize>().ok())
                    .is_none_or(|idx| idx >= expected)
            }) {
                return Err(PredictError::UnexpectedFeature(unexpected.clone()));
            }
        }

        let n_rows = columns.values().next().map_or(0, Vec::len);
        let ordered = projection
            .iter()
            .map(|feature_index| {
                let feature_name = format!("f{}", feature_index);
                let values = columns
                    .get(&feature_name)
                    .ok_or_else(|| PredictError::MissingFeature(feature_name.clone()))?;
                if values.len() != n_rows {
                    return Err(PredictError::ColumnLengthMismatch {
                        feature: feature_name,
                        expected: n_rows,
                        actual: values.len(),
                    });
                }
                Ok(values.clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let projected_preprocessing = projection
            .iter()
            .map(|feature_index| preprocessing[*feature_index].clone())
            .collect::<Vec<_>>();

        Self::from_columns(ordered, &projected_preprocessing, Some(n_rows))
    }

    pub(crate) fn from_sparse_binary_columns(
        n_rows: usize,
        n_features: usize,
        columns: Vec<Vec<usize>>,
        preprocessing: &[FeaturePreprocessing],
    ) -> Result<Self, PredictError> {
        if n_features != preprocessing.len() {
            return Err(PredictError::FeatureCountMismatch {
                expected: preprocessing.len(),
                actual: n_features,
            });
        }

        let mut dense_columns = Vec::with_capacity(n_features);
        for (feature_index, row_indices) in columns.into_iter().enumerate() {
            match preprocessing.get(feature_index) {
                Some(FeaturePreprocessing::Binary) => {
                    let mut values = vec![false; n_rows];
                    for row_index in row_indices {
                        if row_index >= n_rows {
                            return Err(PredictError::ColumnLengthMismatch {
                                feature: format!("f{}", feature_index),
                                expected: n_rows,
                                actual: row_index + 1,
                            });
                        }
                        values[row_index] = true;
                    }
                    dense_columns.push(values.into_iter().map(f64::from).collect());
                }
                Some(FeaturePreprocessing::Numeric { .. }) => {
                    return Err(PredictError::InvalidBinaryValue {
                        feature_index,
                        row_index: 0,
                        value: 1.0,
                    });
                }
                None => unreachable!("validated feature count"),
            }
        }

        Self::from_columns(dense_columns, preprocessing, Some(n_rows))
    }

    pub(crate) fn from_sparse_binary_columns_projected(
        n_rows: usize,
        n_features: usize,
        columns: Vec<Vec<usize>>,
        preprocessing: &[FeaturePreprocessing],
        projection: &[usize],
    ) -> Result<Self, PredictError> {
        if n_features != preprocessing.len() {
            return Err(PredictError::FeatureCountMismatch {
                expected: preprocessing.len(),
                actual: n_features,
            });
        }

        let projected_preprocessing = projection
            .iter()
            .map(|feature_index| preprocessing[*feature_index].clone())
            .collect::<Vec<_>>();
        let mut dense_columns = Vec::with_capacity(projection.len());
        for (local_index, feature_index) in projection.iter().copied().enumerate() {
            match preprocessing.get(feature_index) {
                Some(FeaturePreprocessing::Binary) => {
                    let mut values = vec![false; n_rows];
                    for row_index in columns.get(feature_index).cloned().unwrap_or_default() {
                        if row_index >= n_rows {
                            return Err(PredictError::ColumnLengthMismatch {
                                feature: format!("f{}", feature_index),
                                expected: n_rows,
                                actual: row_index + 1,
                            });
                        }
                        values[row_index] = true;
                    }
                    dense_columns.push(values.into_iter().map(f64::from).collect());
                }
                Some(FeaturePreprocessing::Numeric { .. }) => {
                    return Err(PredictError::InvalidBinaryValue {
                        feature_index: local_index,
                        row_index: 0,
                        value: 1.0,
                    });
                }
                None => unreachable!("validated feature count"),
            }
        }

        Self::from_columns(dense_columns, &projected_preprocessing, Some(n_rows))
    }

    fn from_columns(
        columns: Vec<Vec<f64>>,
        preprocessing: &[FeaturePreprocessing],
        n_rows_hint: Option<usize>,
    ) -> Result<Self, PredictError> {
        if columns.len() != preprocessing.len() {
            return Err(PredictError::FeatureCountMismatch {
                expected: preprocessing.len(),
                actual: columns.len(),
            });
        }

        let n_rows = columns
            .first()
            .map_or_else(|| n_rows_hint.unwrap_or(0), Vec::len);
        let mut feature_columns = Vec::with_capacity(columns.len());
        let mut binned_feature_columns = Vec::with_capacity(columns.len());

        for (feature_index, (column, feature_preprocessing)) in
            columns.into_iter().zip(preprocessing.iter()).enumerate()
        {
            if column.len() != n_rows {
                return Err(PredictError::ColumnLengthMismatch {
                    feature: format!("f{}", feature_index),
                    expected: n_rows,
                    actual: column.len(),
                });
            }
            match feature_preprocessing {
                FeaturePreprocessing::Binary => {
                    let values = column
                        .into_iter()
                        .enumerate()
                        .map(|(row_index, value)| match value {
                            v if v.total_cmp(&0.0).is_eq() => Ok(false),
                            v if v.total_cmp(&1.0).is_eq() => Ok(true),
                            v => Err(PredictError::InvalidBinaryValue {
                                feature_index,
                                row_index,
                                value: v,
                            }),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    feature_columns.push(InferenceFeatureColumn::Binary(values.clone()));
                    binned_feature_columns.push(InferenceBinnedColumn::Binary(values));
                }
                FeaturePreprocessing::Numeric { bin_boundaries } => {
                    let bins = column
                        .iter()
                        .map(|value| infer_numeric_bin(*value, bin_boundaries))
                        .collect();
                    feature_columns.push(InferenceFeatureColumn::Numeric(column));
                    binned_feature_columns.push(InferenceBinnedColumn::Numeric(bins));
                }
            }
        }

        Ok(Self {
            feature_columns,
            binned_feature_columns,
            n_rows,
        })
    }

    pub(crate) fn to_column_major_binned_matrix(&self) -> ColumnMajorBinnedMatrix {
        let n_features = self.feature_columns.len();
        let columns = (0..n_features)
            .map(
                |feature_index| match &self.binned_feature_columns[feature_index] {
                    InferenceBinnedColumn::Numeric(values) => compact_binned_column(values),
                    InferenceBinnedColumn::Binary(values) => CompactBinnedColumn::U8(
                        values.iter().map(|value| u8::from(*value)).collect(),
                    ),
                },
            )
            .collect();

        ColumnMajorBinnedMatrix {
            n_rows: self.n_rows,
            columns,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ColumnMajorBinnedMatrix {
    pub(crate) n_rows: usize,
    columns: Vec<CompactBinnedColumn>,
}

impl ColumnMajorBinnedMatrix {
    pub(crate) fn from_table_access_projected(
        table: &dyn TableAccess,
        projection: &[usize],
    ) -> Self {
        let columns = projection
            .iter()
            .map(|feature_index| {
                if table.is_binary_binned_feature(*feature_index) {
                    CompactBinnedColumn::U8(
                        (0..table.n_rows())
                            .map(|row_index| {
                                u8::from(
                                    table
                                        .binned_boolean_value(*feature_index, row_index)
                                        .unwrap_or(false),
                                )
                            })
                            .collect(),
                    )
                } else {
                    compact_binned_column(
                        &(0..table.n_rows())
                            .map(|row_index| table.binned_value(*feature_index, row_index))
                            .collect::<Vec<_>>(),
                    )
                }
            })
            .collect();

        Self {
            n_rows: table.n_rows(),
            columns,
        }
    }

    pub(crate) fn from_table_access(table: &dyn TableAccess) -> Self {
        let columns = (0..table.n_features())
            .map(|feature_index| {
                if table.is_binary_binned_feature(feature_index) {
                    CompactBinnedColumn::U8(
                        (0..table.n_rows())
                            .map(|row_index| {
                                u8::from(
                                    table
                                        .binned_boolean_value(feature_index, row_index)
                                        .unwrap_or(false),
                                )
                            })
                            .collect(),
                    )
                } else {
                    compact_binned_column(
                        &(0..table.n_rows())
                            .map(|row_index| table.binned_value(feature_index, row_index))
                            .collect::<Vec<_>>(),
                    )
                }
            })
            .collect();

        Self {
            n_rows: table.n_rows(),
            columns,
        }
    }

    #[inline(always)]
    pub(crate) fn column(&self, feature_index: usize) -> &CompactBinnedColumn {
        &self.columns[feature_index]
    }
}

pub(crate) struct ProjectedTableView<'a> {
    base: &'a dyn TableAccess,
    projection: &'a [usize],
}

impl<'a> ProjectedTableView<'a> {
    pub(crate) fn new(base: &'a dyn TableAccess, projection: &'a [usize]) -> Self {
        Self { base, projection }
    }
}

impl TableAccess for ProjectedTableView<'_> {
    fn n_rows(&self) -> usize {
        self.base.n_rows()
    }

    fn n_features(&self) -> usize {
        self.projection.len()
    }

    fn canaries(&self) -> usize {
        0
    }

    fn numeric_bin_cap(&self) -> usize {
        self.base.numeric_bin_cap()
    }

    fn binned_feature_count(&self) -> usize {
        self.projection.len()
    }

    fn feature_value(&self, feature_index: usize, row_index: usize) -> f64 {
        self.base
            .feature_value(self.projection[feature_index], row_index)
    }

    fn is_binary_feature(&self, index: usize) -> bool {
        self.base.is_binary_feature(self.projection[index])
    }

    fn binned_value(&self, feature_index: usize, row_index: usize) -> u16 {
        self.base
            .binned_value(self.projection[feature_index], row_index)
    }

    fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool> {
        self.base
            .binned_boolean_value(self.projection[feature_index], row_index)
    }

    fn binned_column_kind(&self, index: usize) -> BinnedColumnKind {
        BinnedColumnKind::Real {
            source_index: self.projection[index],
        }
    }

    fn is_binary_binned_feature(&self, index: usize) -> bool {
        self.base.is_binary_binned_feature(self.projection[index])
    }

    fn target_value(&self, row_index: usize) -> f64 {
        self.base.target_value(row_index)
    }
}

fn infer_numeric_bin(value: f64, boundaries: &[NumericBinBoundary]) -> u16 {
    boundaries
        .iter()
        .find(|boundary| value <= boundary.upper_bound)
        .map_or_else(
            || boundaries.last().map_or(0, |boundary| boundary.bin),
            |boundary| boundary.bin,
        )
}

fn compact_binned_column(values: &[u16]) -> CompactBinnedColumn {
    if values.iter().all(|value| *value <= u16::from(u8::MAX)) {
        CompactBinnedColumn::U8(values.iter().map(|value| *value as u8).collect())
    } else {
        CompactBinnedColumn::U16(values.to_vec())
    }
}

impl TableAccess for InferenceTable {
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_features(&self) -> usize {
        self.feature_columns.len()
    }

    fn canaries(&self) -> usize {
        0
    }

    fn numeric_bin_cap(&self) -> usize {
        MAX_NUMERIC_BINS
    }

    fn binned_feature_count(&self) -> usize {
        self.binned_feature_columns.len()
    }

    fn feature_value(&self, feature_index: usize, row_index: usize) -> f64 {
        match &self.feature_columns[feature_index] {
            InferenceFeatureColumn::Numeric(values) => values[row_index],
            InferenceFeatureColumn::Binary(values) => f64::from(u8::from(values[row_index])),
        }
    }

    fn is_binary_feature(&self, index: usize) -> bool {
        matches!(
            self.feature_columns[index],
            InferenceFeatureColumn::Binary(_)
        )
    }

    fn binned_value(&self, feature_index: usize, row_index: usize) -> u16 {
        match &self.binned_feature_columns[feature_index] {
            InferenceBinnedColumn::Numeric(values) => values[row_index],
            InferenceBinnedColumn::Binary(values) => u16::from(values[row_index]),
        }
    }

    fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool> {
        match &self.binned_feature_columns[feature_index] {
            InferenceBinnedColumn::Numeric(_) => None,
            InferenceBinnedColumn::Binary(values) => Some(values[row_index]),
        }
    }

    fn binned_column_kind(&self, index: usize) -> BinnedColumnKind {
        BinnedColumnKind::Real {
            source_index: index,
        }
    }

    fn is_binary_binned_feature(&self, index: usize) -> bool {
        matches!(
            self.binned_feature_columns[index],
            InferenceBinnedColumn::Binary(_)
        )
    }

    fn target_value(&self, _row_index: usize) -> f64 {
        0.0
    }
}

#[cfg(feature = "polars")]
pub(crate) fn polars_named_columns(
    df: &DataFrame,
) -> Result<BTreeMap<String, Vec<f64>>, PredictError> {
    df.get_columns()
        .iter()
        .map(|column| {
            let name = column.name().to_string();
            Ok((name, polars_column_values(column)?))
        })
        .collect()
}

#[cfg(feature = "polars")]
fn polars_column_values(column: &Column) -> Result<Vec<f64>, PredictError> {
    let name = column.name().to_string();
    let series = column.as_materialized_series();
    match series.dtype() {
        DataType::Boolean => series
            .bool()?
            .into_iter()
            .enumerate()
            .map(|(row_index, value)| {
                value
                    .map(|value| f64::from(u8::from(value)))
                    .ok_or_else(|| PredictError::NullValue {
                        feature: name.clone(),
                        row_index,
                    })
            })
            .collect(),
        DataType::Float64 => series
            .f64()?
            .into_iter()
            .enumerate()
            .map(|(row_index, value)| {
                value.ok_or_else(|| PredictError::NullValue {
                    feature: name.clone(),
                    row_index,
                })
            })
            .collect(),
        DataType::Float32
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => {
            let casted = series.cast(&DataType::Float64)?;
            casted
                .f64()?
                .into_iter()
                .enumerate()
                .map(|(row_index, value)| {
                    value.ok_or_else(|| PredictError::NullValue {
                        feature: name.clone(),
                        row_index,
                    })
                })
                .collect()
        }
        dtype => Err(PredictError::UnsupportedFeatureType {
            feature: name,
            dtype: dtype.to_string(),
        }),
    }
}
