use super::*;

pub(super) fn normalized_class_probabilities(class_counts: &[f64]) -> Vec<f64> {
    let total: f64 = class_counts.iter().sum();
    if total == 0.0 {
        return vec![0.0; class_counts.len()];
    }

    class_counts.iter().map(|count| count / total).collect()
}

pub(super) fn standard_node_depths(nodes: &[TreeNode], root: usize) -> Vec<usize> {
    let mut depths = vec![0; nodes.len()];
    populate_depths(nodes, root, 0, &mut depths);
    depths
}

fn populate_depths(nodes: &[TreeNode], node_id: usize, depth: usize, depths: &mut [usize]) {
    depths[node_id] = depth;
    match &nodes[node_id] {
        TreeNode::Leaf { .. } => {}
        TreeNode::BinarySplit {
            left_child,
            right_child,
            ..
        }
        | TreeNode::ObliqueSplit {
            left_child,
            right_child,
            ..
        } => {
            populate_depths(nodes, *left_child, depth + 1, depths);
            populate_depths(nodes, *right_child, depth + 1, depths);
        }
        TreeNode::MultiwaySplit { branches, .. } => {
            for (_, child) in branches {
                populate_depths(nodes, *child, depth + 1, depths);
            }
        }
    }
}

pub(super) fn binary_split_ir(
    feature_index: usize,
    threshold_bin: u16,
    _missing_direction: MissingBranchDirection,
    preprocessing: &[FeaturePreprocessing],
) -> BinarySplit {
    match preprocessing.get(feature_index) {
        Some(FeaturePreprocessing::Binary) => BinarySplit::BooleanTest {
            feature_index,
            feature_name: feature_name(feature_index),
            false_child_semantics: "left".to_string(),
            true_child_semantics: "right".to_string(),
        },
        Some(FeaturePreprocessing::Numeric { .. }) | None => BinarySplit::NumericBinThreshold {
            feature_index,
            feature_name: feature_name(feature_index),
            operator: "<=".to_string(),
            threshold_bin,
            threshold_upper_bound: threshold_upper_bound(
                preprocessing,
                feature_index,
                threshold_bin,
            ),
            comparison_dtype: "uint16".to_string(),
        },
    }
}

pub(super) fn oblique_split_ir(
    feature_indices: &[usize],
    weights: &[f64],
    missing_directions: &[crate::tree::shared::MissingBranchDirection],
    threshold: f64,
) -> BinarySplit {
    BinarySplit::ObliqueLinearCombination {
        feature_indices: feature_indices.to_vec(),
        feature_names: feature_indices
            .iter()
            .map(|feature_index| feature_name(*feature_index))
            .collect(),
        weights: weights.to_vec(),
        missing_directions: missing_directions
            .iter()
            .map(|direction| match direction {
                crate::tree::shared::MissingBranchDirection::Left => "left".to_string(),
                crate::tree::shared::MissingBranchDirection::Right => "right".to_string(),
                crate::tree::shared::MissingBranchDirection::Node => "node".to_string(),
            })
            .collect(),
        operator: "<=".to_string(),
        threshold,
    }
}

pub(super) fn oblivious_split_ir(
    feature_index: usize,
    threshold_bin: u16,
    preprocessing: &[FeaturePreprocessing],
) -> IrObliviousSplit {
    match preprocessing.get(feature_index) {
        Some(FeaturePreprocessing::Binary) => IrObliviousSplit::BooleanTest {
            feature_index,
            feature_name: feature_name(feature_index),
            bit_when_false: 0,
            bit_when_true: 1,
        },
        Some(FeaturePreprocessing::Numeric { .. }) | None => {
            IrObliviousSplit::NumericBinThreshold {
                feature_index,
                feature_name: feature_name(feature_index),
                operator: "<=".to_string(),
                threshold_bin,
                threshold_upper_bound: threshold_upper_bound(
                    preprocessing,
                    feature_index,
                    threshold_bin,
                ),
                comparison_dtype: "uint16".to_string(),
                bit_when_true: 0,
                bit_when_false: 1,
            }
        }
    }
}
