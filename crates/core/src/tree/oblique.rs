use forestfire_data::TableAccess;

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

pub(crate) fn projected_rows_for_pair(
    table: &dyn TableAccess,
    rows: &[usize],
    feature_indices: [usize; 2],
    weights: [f64; 2],
) -> Option<Vec<ProjectedRow>> {
    let mut projected = Vec::with_capacity(rows.len());
    for &row_index in rows {
        if table.is_missing(feature_indices[0], row_index)
            || table.is_missing(feature_indices[1], row_index)
        {
            return None;
        }
        let value = weights[0] * table.feature_value(feature_indices[0], row_index)
            + weights[1] * table.feature_value(feature_indices[1], row_index);
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
        let projection = weights[0] * table.feature_value(feature_indices[0], row_index)
            + weights[1] * table.feature_value(feature_indices[1], row_index);
        if projection <= threshold {
            rows.swap(left, index);
            left += 1;
        }
    }
    left
}
