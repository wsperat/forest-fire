use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeStructureSummary {
    pub representation: String,
    pub node_count: usize,
    pub internal_node_count: usize,
    pub leaf_count: usize,
    pub actual_depth: usize,
    pub shortest_path: usize,
    pub longest_path: usize,
    pub average_path: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionValueStats {
    pub count: usize,
    pub unique_count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub histogram: Vec<PredictionHistogramEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionHistogramEntry {
    pub prediction: f64,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntrospectionError {
    TreeIndexOutOfBounds { requested: usize, available: usize },
    NodeIndexOutOfBounds { requested: usize, available: usize },
    LevelIndexOutOfBounds { requested: usize, available: usize },
    LeafIndexOutOfBounds { requested: usize, available: usize },
    NotANodeTree,
    NotAnObliviousTree,
}

impl Display for IntrospectionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            IntrospectionError::TreeIndexOutOfBounds {
                requested,
                available,
            } => write!(
                f,
                "Tree index {} is out of bounds for model with {} trees.",
                requested, available
            ),
            IntrospectionError::NodeIndexOutOfBounds {
                requested,
                available,
            } => write!(
                f,
                "Node index {} is out of bounds for tree with {} nodes.",
                requested, available
            ),
            IntrospectionError::LevelIndexOutOfBounds {
                requested,
                available,
            } => write!(
                f,
                "Level index {} is out of bounds for tree with {} levels.",
                requested, available
            ),
            IntrospectionError::LeafIndexOutOfBounds {
                requested,
                available,
            } => write!(
                f,
                "Leaf index {} is out of bounds for tree with {} leaves.",
                requested, available
            ),
            IntrospectionError::NotANodeTree => write!(
                f,
                "This tree uses oblivious-level representation; inspect levels or leaves instead."
            ),
            IntrospectionError::NotAnObliviousTree => write!(
                f,
                "This tree uses node-tree representation; inspect nodes instead."
            ),
        }
    }
}

impl Error for IntrospectionError {}

pub(crate) fn tree_structure_summary(
    tree: ir::TreeDefinition,
) -> Result<TreeStructureSummary, IntrospectionError> {
    match tree {
        ir::TreeDefinition::NodeTree {
            root_node_id,
            nodes,
            ..
        } => {
            let node_map = nodes
                .iter()
                .cloned()
                .map(|node| match &node {
                    ir::NodeTreeNode::Leaf { node_id, .. }
                    | ir::NodeTreeNode::BinaryBranch { node_id, .. }
                    | ir::NodeTreeNode::MultiwayBranch { node_id, .. } => (*node_id, node),
                })
                .collect::<BTreeMap<_, _>>();
            let mut leaf_depths = Vec::new();
            collect_leaf_depths(&node_map, root_node_id, &mut leaf_depths)?;
            let internal_node_count = nodes
                .iter()
                .filter(|node| !matches!(node, ir::NodeTreeNode::Leaf { .. }))
                .count();
            let leaf_count = leaf_depths.len();
            let shortest_path = *leaf_depths.iter().min().unwrap_or(&0);
            let longest_path = *leaf_depths.iter().max().unwrap_or(&0);
            let average_path = if leaf_depths.is_empty() {
                0.0
            } else {
                leaf_depths.iter().sum::<usize>() as f64 / leaf_depths.len() as f64
            };
            Ok(TreeStructureSummary {
                representation: "node_tree".to_string(),
                node_count: internal_node_count + leaf_count,
                internal_node_count,
                leaf_count,
                actual_depth: longest_path,
                shortest_path,
                longest_path,
                average_path,
            })
        }
        ir::TreeDefinition::ObliviousLevels { depth, leaves, .. } => Ok(TreeStructureSummary {
            representation: "oblivious_levels".to_string(),
            node_count: ((1usize << depth) - 1) + leaves.len(),
            internal_node_count: (1usize << depth) - 1,
            leaf_count: leaves.len(),
            actual_depth: depth,
            shortest_path: depth,
            longest_path: depth,
            average_path: depth as f64,
        }),
    }
}

fn collect_leaf_depths(
    nodes: &BTreeMap<usize, ir::NodeTreeNode>,
    node_id: usize,
    output: &mut Vec<usize>,
) -> Result<(), IntrospectionError> {
    match nodes
        .get(&node_id)
        .ok_or(IntrospectionError::NodeIndexOutOfBounds {
            requested: node_id,
            available: nodes.len(),
        })? {
        ir::NodeTreeNode::Leaf { depth, .. } => output.push(*depth),
        ir::NodeTreeNode::BinaryBranch {
            depth: _, children, ..
        } => {
            collect_leaf_depths(nodes, children.left, output)?;
            collect_leaf_depths(nodes, children.right, output)?;
        }
        ir::NodeTreeNode::MultiwayBranch {
            depth,
            branches,
            unmatched_leaf: _,
            ..
        } => {
            output.push(depth + 1);
            for branch in branches {
                collect_leaf_depths(nodes, branch.child, output)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn prediction_value_stats(
    tree: ir::TreeDefinition,
) -> Result<PredictionValueStats, IntrospectionError> {
    let predictions = match tree {
        ir::TreeDefinition::NodeTree { nodes, .. } => nodes
            .into_iter()
            .flat_map(|node| match node {
                ir::NodeTreeNode::Leaf { leaf, .. } => vec![leaf_payload_value(&leaf)],
                ir::NodeTreeNode::MultiwayBranch { unmatched_leaf, .. } => {
                    vec![leaf_payload_value(&unmatched_leaf)]
                }
                ir::NodeTreeNode::BinaryBranch { .. } => Vec::new(),
            })
            .collect::<Vec<_>>(),
        ir::TreeDefinition::ObliviousLevels { leaves, .. } => leaves
            .into_iter()
            .map(|leaf| leaf_payload_value(&leaf.leaf))
            .collect::<Vec<_>>(),
    };

    let count = predictions.len();
    let min = predictions
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let max = predictions
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    let mean = if count == 0 {
        0.0
    } else {
        predictions.iter().sum::<f64>() / count as f64
    };
    let std_dev = if count == 0 {
        0.0
    } else {
        let variance = predictions
            .iter()
            .map(|value| (*value - mean).powi(2))
            .sum::<f64>()
            / count as f64;
        variance.sqrt()
    };
    let mut histogram = BTreeMap::<String, usize>::new();
    for prediction in &predictions {
        *histogram.entry(prediction.to_string()).or_insert(0) += 1;
    }
    let histogram = histogram
        .into_iter()
        .map(|(prediction, count)| PredictionHistogramEntry {
            prediction: prediction
                .parse::<f64>()
                .expect("histogram keys are numeric"),
            count,
        })
        .collect::<Vec<_>>();

    Ok(PredictionValueStats {
        count,
        unique_count: histogram.len(),
        min,
        max,
        mean,
        std_dev,
        histogram,
    })
}

fn leaf_payload_value(leaf: &ir::LeafPayload) -> f64 {
    match leaf {
        ir::LeafPayload::RegressionValue { value } => *value,
        ir::LeafPayload::ClassIndex { class_value, .. } => *class_value,
    }
}
