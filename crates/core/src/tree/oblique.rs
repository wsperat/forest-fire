use forestfire_data::{BinnedColumnKind, NumericBins, TableAccess, numeric_missing_bin};
use std::collections::BTreeMap;

use super::shared::MissingBranchDirection;

#[derive(Debug, Clone)]
pub(crate) struct ProjectedRow {
    pub(crate) row_index: usize,
    pub(crate) value: f64,
}

pub(crate) fn all_feature_pairs(features: &[usize]) -> Vec<[usize; 2]> {
    features
        .iter()
        .enumerate()
        .flat_map(|(left_index, &left_feature)| {
            features[left_index + 1..]
                .iter()
                .map(move |&right_feature| [left_feature, right_feature])
        })
        .collect()
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

pub(crate) fn missing_mask_for_pair(
    table: &dyn TableAccess,
    feature_indices: [usize; 2],
    row_index: usize,
) -> u8 {
    let mut mask = 0u8;
    if oblique_feature_value(table, feature_indices[0], row_index).is_none() {
        mask |= 1;
    }
    if oblique_feature_value(table, feature_indices[1], row_index).is_none() {
        mask |= 2;
    }
    mask
}

pub(crate) fn resolve_oblique_missing_direction(
    mask: u8,
    weights: [f64; 2],
    missing_directions: [MissingBranchDirection; 2],
) -> Option<bool> {
    match mask {
        0 => None,
        1 => match missing_directions[0] {
            MissingBranchDirection::Left => Some(true),
            MissingBranchDirection::Right => Some(false),
            MissingBranchDirection::Node => None,
        },
        2 => match missing_directions[1] {
            MissingBranchDirection::Left => Some(true),
            MissingBranchDirection::Right => Some(false),
            MissingBranchDirection::Node => None,
        },
        3 => {
            let left = missing_directions[0];
            let right = missing_directions[1];
            if matches!(left, MissingBranchDirection::Node)
                || matches!(right, MissingBranchDirection::Node)
            {
                return None;
            }
            if left == right {
                return Some(matches!(left, MissingBranchDirection::Left));
            }
            Some(if weights[0].abs() >= weights[1].abs() {
                matches!(left, MissingBranchDirection::Left)
            } else {
                matches!(right, MissingBranchDirection::Left)
            })
        }
        _ => None,
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
    missing_directions: [MissingBranchDirection; 2],
    rows: &mut [usize],
) -> usize {
    let mut left = 0usize;
    for index in 0..rows.len() {
        let row_index = rows[index];
        let mask = missing_mask_for_pair(table, feature_indices, row_index);
        let go_left = if let Some(direction) =
            resolve_oblique_missing_direction(mask, weights, missing_directions)
        {
            direction
        } else if let Some(projection) = oblique_feature_value(table, feature_indices[0], row_index)
            .zip(oblique_feature_value(table, feature_indices[1], row_index))
            .map(|(left_value, right_value)| weights[0] * left_value + weights[1] * right_value)
        {
            projection <= threshold
        } else {
            false
        };
        if go_left {
            rows.swap(left, index);
            left += 1;
        }
    }
    left
}
