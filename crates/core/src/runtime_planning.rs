use super::*;
use std::collections::BTreeSet;

pub(crate) fn build_feature_projection(model: &Model) -> Vec<usize> {
    model_used_feature_indices(model)
}

pub(crate) fn build_feature_index_map(num_features: usize, projection: &[usize]) -> Vec<usize> {
    let mut map = vec![usize::MAX; num_features];
    for (local_index, feature_index) in projection.iter().copied().enumerate() {
        map[feature_index] = local_index;
    }
    map
}

pub(crate) fn remap_feature_index(feature_index: usize, feature_index_map: &[usize]) -> usize {
    feature_index_map[feature_index]
}

pub(crate) fn ordered_ensemble_indices(trees: &[Model]) -> Vec<usize> {
    let mut keyed = trees
        .iter()
        .enumerate()
        .map(|(tree_index, tree)| {
            let used = model_used_feature_indices(tree);
            let primary_feature = tree_primary_feature(tree).unwrap_or(usize::MAX);
            (tree_index, primary_feature, used.len(), used)
        })
        .collect::<Vec<_>>();

    keyed.sort_by(|left, right| {
        left.1
            .cmp(&right.1)
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.3.cmp(&right.3))
            .then_with(|| left.0.cmp(&right.0))
    });

    keyed
        .into_iter()
        .map(|(tree_index, _, _, _)| tree_index)
        .collect()
}

pub(crate) fn model_used_feature_indices(model: &Model) -> Vec<usize> {
    let ir = model.to_ir();
    let mut used = BTreeSet::new();
    for tree in &ir.model.trees {
        collect_tree_used_features(tree, &mut used);
    }
    used.into_iter().collect()
}

pub(crate) fn tree_primary_feature(model: &Model) -> Option<usize> {
    let ir = model.to_ir();
    ir.model
        .trees
        .first()
        .and_then(tree_definition_primary_feature)
}

fn collect_tree_used_features(tree: &ir::TreeDefinition, used: &mut BTreeSet<usize>) {
    match tree {
        ir::TreeDefinition::NodeTree { nodes, .. } => {
            for node in nodes {
                match node {
                    ir::NodeTreeNode::Leaf { .. } => {}
                    ir::NodeTreeNode::BinaryBranch { split, .. } => {
                        for feature_index in binary_split_feature_indices(split) {
                            used.insert(feature_index);
                        }
                    }
                    ir::NodeTreeNode::MultiwayBranch { split, .. } => {
                        used.insert(split.feature_index);
                    }
                }
            }
        }
        ir::TreeDefinition::ObliviousLevels { levels, .. } => {
            for level in levels {
                used.insert(oblivious_split_feature_index(&level.split));
            }
        }
    }
}

fn tree_definition_primary_feature(tree: &ir::TreeDefinition) -> Option<usize> {
    match tree {
        ir::TreeDefinition::NodeTree {
            root_node_id,
            nodes,
            ..
        } => nodes.iter().find_map(|node| match node {
            ir::NodeTreeNode::Leaf { node_id, .. } if node_id == root_node_id => None,
            ir::NodeTreeNode::BinaryBranch { node_id, split, .. } if node_id == root_node_id => {
                binary_split_feature_indices(split).into_iter().next()
            }
            ir::NodeTreeNode::MultiwayBranch { node_id, split, .. } if node_id == root_node_id => {
                Some(split.feature_index)
            }
            _ => None,
        }),
        ir::TreeDefinition::ObliviousLevels { levels, .. } => levels
            .first()
            .map(|level| oblivious_split_feature_index(&level.split)),
    }
}

fn binary_split_feature_indices(split: &ir::BinarySplit) -> Vec<usize> {
    match split {
        ir::BinarySplit::NumericBinThreshold { feature_index, .. }
        | ir::BinarySplit::BooleanTest { feature_index, .. } => vec![*feature_index],
        ir::BinarySplit::ObliqueLinearCombination {
            feature_indices, ..
        } => feature_indices.clone(),
    }
}

fn oblivious_split_feature_index(split: &ir::ObliviousSplit) -> usize {
    match split {
        ir::ObliviousSplit::NumericBinThreshold { feature_index, .. }
        | ir::ObliviousSplit::BooleanTest { feature_index, .. } => *feature_index,
    }
}
