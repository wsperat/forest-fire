use forestfire_data::{BinnedColumnKind, NumericBins, TableAccess, numeric_missing_bin};
use std::collections::BTreeMap;

pub(crate) const OBLIQUE_SHORTLIST_SIZE: usize = 8;

#[derive(Debug, Clone)]
pub(crate) struct ProjectedRow {
    pub(crate) row_index: usize,
    pub(crate) value: f64,
}

pub(crate) fn shortlist_real_features(
    feature_scores: &[(usize, f64)],
    real_feature_count: usize,
    limit: usize,
) -> Vec<usize> {
    let mut ranked = feature_scores
        .iter()
        .copied()
        .filter(|(feature_index, _)| *feature_index < real_feature_count)
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut selected = Vec::new();
    for (feature_index, _) in ranked {
        if selected.contains(&feature_index) {
            continue;
        }
        selected.push(feature_index);
        if selected.len() >= limit {
            break;
        }
    }
    selected
}

pub(crate) fn normalize_weights(weights: [f64; 2]) -> Option<[f64; 2]> {
    let norm = (weights[0] * weights[0] + weights[1] * weights[1]).sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return None;
    }
    Some([weights[0] / norm, weights[1] / norm])
}

pub(crate) fn matched_canary_feature_pairs(
    table: &dyn TableAccess,
    shortlisted_real_features: &[usize],
) -> Vec<[usize; 2]> {
    let mut canary_index_by_copy_and_source = BTreeMap::new();
    for feature_index in table.n_features()..table.binned_feature_count() {
        if let BinnedColumnKind::Canary {
            source_index,
            copy_index,
        } = table.binned_column_kind(feature_index)
        {
            canary_index_by_copy_and_source.insert((copy_index, source_index), feature_index);
        }
    }

    let mut pairs = Vec::new();
    for left_index in 0..shortlisted_real_features.len() {
        for right_index in left_index + 1..shortlisted_real_features.len() {
            let left_source = shortlisted_real_features[left_index];
            let right_source = shortlisted_real_features[right_index];
            for copy_index in 0..table.canaries() {
                let Some(&left_canary) =
                    canary_index_by_copy_and_source.get(&(copy_index, left_source))
                else {
                    continue;
                };
                let Some(&right_canary) =
                    canary_index_by_copy_and_source.get(&(copy_index, right_source))
                else {
                    continue;
                };
                pairs.push([left_canary, right_canary]);
            }
        }
    }
    pairs
}

pub(crate) fn oblique_feature_value(
    table: &dyn TableAccess,
    feature_index: usize,
    row_index: usize,
) -> Option<f64> {
    if feature_index < table.n_features() {
        if table.is_missing(feature_index, row_index) {
            return None;
        }
        return Some(table.feature_value(feature_index, row_index));
    }

    if table.is_binary_binned_feature(feature_index) {
        return table
            .binned_boolean_value(feature_index, row_index)
            .map(|value| f64::from(u8::from(value)));
    }

    let value = table.binned_value(feature_index, row_index);
    let missing_bin = numeric_missing_bin(NumericBins::Fixed(table.numeric_bin_cap()));
    if value == missing_bin {
        None
    } else {
        Some(value as f64)
    }
}

pub(crate) fn projected_rows_for_pair(
    table: &dyn TableAccess,
    rows: &[usize],
    feature_indices: [usize; 2],
    weights: [f64; 2],
) -> Option<Vec<ProjectedRow>> {
    let mut projected = Vec::with_capacity(rows.len());
    for &row_index in rows {
        let value = weights[0] * oblique_feature_value(table, feature_indices[0], row_index)?
            + weights[1] * oblique_feature_value(table, feature_indices[1], row_index)?;
        if !value.is_finite() {
            return None;
        }
        projected.push(ProjectedRow { row_index, value });
    }
    projected.sort_by(|left, right| left.value.total_cmp(&right.value));
    Some(projected)
}

pub(crate) fn partition_rows_for_oblique_split(
    table: &dyn TableAccess,
    feature_indices: [usize; 2],
    weights: [f64; 2],
    threshold: f64,
    rows: &mut [usize],
) -> usize {
    let mut left = 0usize;
    for index in 0..rows.len() {
        let row_index = rows[index];
        let Some(projection) = oblique_feature_value(table, feature_indices[0], row_index)
            .zip(oblique_feature_value(table, feature_indices[1], row_index))
            .map(|(left_value, right_value)| weights[0] * left_value + weights[1] * right_value)
        else {
            continue;
        };
        if projection <= threshold {
            rows.swap(left, index);
            left += 1;
        }
    }
    left
}
