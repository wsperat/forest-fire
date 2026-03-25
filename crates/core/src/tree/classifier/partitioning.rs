use super::*;

pub(super) fn partition_rows_for_binary_split(
    table: &dyn TableAccess,
    feature_index: usize,
    threshold_bin: u16,
    rows: &mut [usize],
) -> usize {
    let mut left = 0usize;
    for index in 0..rows.len() {
        let go_left = if table.is_binary_binned_feature(feature_index) {
            !table
                .binned_boolean_value(feature_index, rows[index])
                .expect("binary feature must expose boolean values")
        } else {
            table.binned_value(feature_index, rows[index]) <= threshold_bin
        };
        if go_left {
            rows.swap(left, index);
            left += 1;
        }
    }
    left
}

pub(super) fn partition_rows_for_multiway_split(
    table: &dyn TableAccess,
    feature_index: usize,
    branch_bins: &[u16],
    rows: &mut [usize],
) -> Vec<(u16, usize, usize)> {
    let mut scratch = vec![0usize; rows.len()];
    let mut counts = vec![0usize; branch_bins.len()];

    for row_idx in rows.iter().copied() {
        let bin = if table.is_binary_binned_feature(feature_index) {
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
        let bin = if table.is_binary_binned_feature(feature_index) {
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
