use super::*;

pub(super) fn partition_rows_for_multiway_split(
    table: &dyn TableAccess,
    feature_index: usize,
    branch_bins: &[u16],
    missing_branch_bin: Option<u16>,
    rows: &mut [usize],
) -> Vec<(u16, usize, usize)> {
    let mut scratch = vec![0usize; rows.len()];
    let mut counts = vec![0usize; branch_bins.len()];

    for row_idx in rows.iter().copied() {
        let bin = if table.is_missing(feature_index, row_idx) {
            missing_branch_bin.expect("training-time missing values must map to a branch")
        } else if table.is_binary_binned_feature(feature_index) {
            if table
                .binned_boolean_value(feature_index, row_idx)
                .expect("binary feature must expose boolean values")
            {
                1
            } else {
                0
            }
        } else {
            table.binned_value(feature_index, row_idx)
        };
        let branch_index = branch_bins
            .binary_search(&bin)
            .expect("branch bins must cover all observed bins");
        counts[branch_index] += 1;
    }

    let mut offsets = Vec::with_capacity(branch_bins.len());
    let mut next = 0usize;
    for count in &counts {
        offsets.push(next);
        next += *count;
    }
    let mut write_positions = offsets.clone();
    for row_idx in rows.iter().copied() {
        let bin = if table.is_missing(feature_index, row_idx) {
            missing_branch_bin.expect("training-time missing values must map to a branch")
        } else if table.is_binary_binned_feature(feature_index) {
            if table
                .binned_boolean_value(feature_index, row_idx)
                .expect("binary feature must expose boolean values")
            {
                1
            } else {
                0
            }
        } else {
            table.binned_value(feature_index, row_idx)
        };
        let branch_index = branch_bins
            .binary_search(&bin)
            .expect("branch bins must cover all observed bins");
        let write_index = write_positions[branch_index];
        scratch[write_index] = row_idx;
        write_positions[branch_index] += 1;
    }
    rows.copy_from_slice(&scratch);

    branch_bins
        .iter()
        .copied()
        .zip(offsets)
        .zip(counts)
        .map(|((bin, start), count)| (bin, start, start + count))
        .collect()
}
