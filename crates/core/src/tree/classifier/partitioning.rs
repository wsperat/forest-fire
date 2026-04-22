use super::*;
use crate::tree::shared::numeric_missing_bin;

pub(super) fn partition_rows_for_multiway_split(
    table: &dyn TableAccess,
    feature_index: usize,
    branch_bins: &[u16],
    missing_branch_bin: Option<u16>,
    rows: &mut [usize],
) -> Vec<(u16, usize, usize)> {
    let mut scratch = vec![0usize; rows.len()];
    let mut counts = vec![0usize; branch_bins.len()];
    let missing_bin = numeric_missing_bin(table);

    for row_idx in rows.iter().copied() {
        let bin = multiway_row_bin(
            table,
            feature_index,
            row_idx,
            missing_bin,
            missing_branch_bin,
        );
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
        let bin = multiway_row_bin(
            table,
            feature_index,
            row_idx,
            missing_bin,
            missing_branch_bin,
        );
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

fn multiway_row_bin(
    table: &dyn TableAccess,
    feature_index: usize,
    row_idx: usize,
    missing_bin: u16,
    missing_branch_bin: Option<u16>,
) -> u16 {
    if table.is_binary_binned_feature(feature_index) {
        table
            .binned_boolean_value(feature_index, row_idx)
            .map_or_else(
                || missing_branch_bin.expect("training-time missing values must map to a branch"),
                u16::from,
            )
    } else {
        let bin = table.binned_value(feature_index, row_idx);
        if bin == missing_bin {
            missing_branch_bin.expect("training-time missing values must map to a branch")
        } else {
            bin
        }
    }
}
