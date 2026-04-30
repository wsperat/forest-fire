use super::*;
use std::collections::BTreeSet;
use wide::{u16x8, u32x8};

pub(crate) const PARALLEL_INFERENCE_ROW_THRESHOLD: usize = 256;
pub(crate) const PARALLEL_INFERENCE_CHUNK_ROWS: usize = 256;
pub(crate) const STANDARD_BATCH_INFERENCE_CHUNK_ROWS: usize = 4096;
pub(crate) const OBLIVIOUS_SIMD_LANES: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum OptimizedRuntime {
    BinaryClassifier {
        nodes: Vec<OptimizedBinaryClassifierNode>,
        class_labels: Vec<f64>,
    },
    StandardClassifier {
        nodes: Vec<OptimizedClassifierNode>,
        root: usize,
        class_labels: Vec<f64>,
    },
    ObliviousClassifier {
        feature_indices: Vec<usize>,
        threshold_bins: Vec<u16>,
        leaf_values: Vec<Vec<f64>>,
        class_labels: Vec<f64>,
    },
    BinaryRegressor {
        nodes: Vec<OptimizedBinaryRegressorNode>,
    },
    ObliviousRegressor {
        feature_indices: Vec<usize>,
        threshold_bins: Vec<u16>,
        leaf_values: Vec<f64>,
    },
    ForestClassifier {
        trees: Vec<OptimizedRuntime>,
        class_labels: Vec<f64>,
    },
    ForestRegressor {
        trees: Vec<OptimizedRuntime>,
    },
    BoostedBinaryClassifier {
        trees: Vec<OptimizedRuntime>,
        tree_weights: Vec<f64>,
        base_score: f64,
        class_labels: Vec<f64>,
    },
    BoostedRegressor {
        trees: Vec<OptimizedRuntime>,
        tree_weights: Vec<f64>,
        base_score: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum OptimizedClassifierNode {
    Leaf(Vec<f64>),
    Binary {
        feature_index: usize,
        threshold_bin: u16,
        children: [usize; 2],
        missing_bin: Option<u16>,
        missing_child: Option<usize>,
        missing_probabilities: Option<Vec<f64>>,
    },
    Multiway {
        feature_index: usize,
        child_lookup: Vec<usize>,
        max_bin_index: usize,
        missing_bin: Option<u16>,
        missing_child: Option<usize>,
        fallback_probabilities: Vec<f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum OptimizedBinaryClassifierNode {
    Leaf(Vec<f64>),
    Branch {
        feature_index: usize,
        threshold_bin: u16,
        jump_index: usize,
        jump_if_greater: bool,
        missing_bin: Option<u16>,
        missing_jump_index: Option<usize>,
        missing_probabilities: Option<Vec<f64>>,
    },
    ObliqueBranch {
        feature_indices: [usize; 2],
        weights: [f64; 2],
        missing_directions: [super::tree::shared::MissingBranchDirection; 2],
        threshold: f64,
        jump_index: usize,
        jump_if_greater: bool,
        missing_probabilities: Vec<f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum OptimizedBinaryRegressorNode {
    Leaf(f64),
    Branch {
        feature_index: usize,
        threshold_bin: u16,
        jump_index: usize,
        jump_if_greater: bool,
        missing_bin: Option<u16>,
        missing_jump_index: Option<usize>,
        missing_value: Option<f64>,
    },
    ObliqueBranch {
        feature_indices: [usize; 2],
        weights: [f64; 2],
        missing_directions: [super::tree::shared::MissingBranchDirection; 2],
        threshold: f64,
        jump_index: usize,
        jump_if_greater: bool,
        missing_value: f64,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct InferenceExecutor {
    pub(crate) thread_count: usize,
    pub(crate) pool: Option<Arc<rayon::ThreadPool>>,
}

impl InferenceExecutor {
    pub(crate) fn new(thread_count: usize) -> Result<Self, OptimizeError> {
        let pool = if thread_count > 1 {
            Some(Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(thread_count)
                    .build()
                    .map_err(|err| OptimizeError::ThreadPoolBuildFailed(err.to_string()))?,
            ))
        } else {
            None
        };

        Ok(Self { thread_count, pool })
    }

    pub(crate) fn predict_rows<F>(&self, n_rows: usize, predict_row: F) -> Vec<f64>
    where
        F: Fn(usize) -> f64 + Sync + Send,
    {
        if self.thread_count == 1 || n_rows < PARALLEL_INFERENCE_ROW_THRESHOLD {
            return (0..n_rows).map(predict_row).collect();
        }

        self.pool
            .as_ref()
            .expect("thread pool exists when parallel inference is enabled")
            .install(|| (0..n_rows).into_par_iter().map(predict_row).collect())
    }

    pub(crate) fn fill_chunks<F>(&self, outputs: &mut [f64], chunk_rows: usize, fill_chunk: F)
    where
        F: Fn(usize, &mut [f64]) + Sync + Send,
    {
        if self.thread_count == 1 || outputs.len() < PARALLEL_INFERENCE_ROW_THRESHOLD {
            for (chunk_index, chunk) in outputs.chunks_mut(chunk_rows).enumerate() {
                fill_chunk(chunk_index * chunk_rows, chunk);
            }
            return;
        }

        self.pool
            .as_ref()
            .expect("thread pool exists when parallel inference is enabled")
            .install(|| {
                outputs
                    .par_chunks_mut(chunk_rows)
                    .enumerate()
                    .for_each(|(chunk_index, chunk)| fill_chunk(chunk_index * chunk_rows, chunk));
            });
    }
}

pub(crate) fn resolve_inference_thread_count(
    physical_cores: Option<usize>,
) -> Result<usize, OptimizeError> {
    let available = num_cpus::get_physical().max(1);
    let requested = physical_cores.unwrap_or(available);

    if requested == 0 {
        return Err(OptimizeError::InvalidPhysicalCoreCount {
            requested,
            available,
        });
    }

    Ok(requested.min(available))
}

fn missing_feature_enabled(
    feature_index: usize,
    missing_features: Option<&BTreeSet<usize>>,
) -> bool {
    missing_features.is_none_or(|features| features.contains(&feature_index))
}

fn optimized_missing_bin(
    preprocessing: &[FeaturePreprocessing],
    feature_index: usize,
    missing_features: Option<&BTreeSet<usize>>,
) -> Option<u16> {
    if !missing_feature_enabled(feature_index, missing_features) {
        return None;
    }

    match preprocessing.get(feature_index) {
        Some(FeaturePreprocessing::Binary) => Some(forestfire_data::BINARY_MISSING_BIN),
        Some(FeaturePreprocessing::Numeric { missing_bin, .. }) => Some(*missing_bin),
        None => None,
    }
}

impl OptimizedRuntime {
    fn supports_batch_matrix(&self) -> bool {
        match self {
            OptimizedRuntime::BinaryClassifier { nodes, .. } => {
                !binary_classifier_nodes_require_rowwise_raw(nodes)
            }
            OptimizedRuntime::BinaryRegressor { nodes } => {
                !binary_regressor_nodes_require_rowwise_raw(nodes)
            }
            OptimizedRuntime::ObliviousClassifier { .. }
            | OptimizedRuntime::ObliviousRegressor { .. } => true,
            OptimizedRuntime::StandardClassifier { .. } => false,
            OptimizedRuntime::ForestClassifier { trees, .. }
            | OptimizedRuntime::ForestRegressor { trees }
            | OptimizedRuntime::BoostedBinaryClassifier { trees, .. }
            | OptimizedRuntime::BoostedRegressor { trees, .. } => {
                trees.iter().all(OptimizedRuntime::supports_batch_matrix)
            }
        }
    }

    pub(crate) fn should_use_batch_matrix(&self, n_rows: usize) -> bool {
        n_rows > 1 && self.supports_batch_matrix()
    }

    pub(crate) fn from_model(
        model: &Model,
        feature_index_map: &[usize],
        missing_features: Option<&BTreeSet<usize>>,
    ) -> Self {
        match model {
            Model::DecisionTreeClassifier(classifier) => {
                Self::from_classifier(classifier, feature_index_map, missing_features)
            }
            Model::DecisionTreeRegressor(regressor) => {
                Self::from_regressor(regressor, feature_index_map, missing_features)
            }
            Model::RandomForest(forest) => match forest.task() {
                Task::Classification => {
                    let tree_order = ordered_ensemble_indices(forest.trees());
                    Self::ForestClassifier {
                        trees: tree_order
                            .into_iter()
                            .map(|tree_index| {
                                Self::from_model(
                                    &forest.trees()[tree_index],
                                    feature_index_map,
                                    missing_features,
                                )
                            })
                            .collect(),
                        class_labels: forest
                            .class_labels()
                            .expect("classification forest stores class labels"),
                    }
                }
                Task::Regression => {
                    let tree_order = ordered_ensemble_indices(forest.trees());
                    Self::ForestRegressor {
                        trees: tree_order
                            .into_iter()
                            .map(|tree_index| {
                                Self::from_model(
                                    &forest.trees()[tree_index],
                                    feature_index_map,
                                    missing_features,
                                )
                            })
                            .collect(),
                    }
                }
            },
            Model::GradientBoostedTrees(model) => match model.task() {
                Task::Classification => {
                    let tree_order = ordered_ensemble_indices(model.trees());
                    Self::BoostedBinaryClassifier {
                        trees: tree_order
                            .iter()
                            .map(|tree_index| {
                                Self::from_model(
                                    &model.trees()[*tree_index],
                                    feature_index_map,
                                    missing_features,
                                )
                            })
                            .collect(),
                        tree_weights: tree_order
                            .iter()
                            .map(|tree_index| model.tree_weights()[*tree_index])
                            .collect(),
                        base_score: model.base_score(),
                        class_labels: model
                            .class_labels()
                            .expect("classification boosting stores class labels"),
                    }
                }
                Task::Regression => {
                    let tree_order = ordered_ensemble_indices(model.trees());
                    Self::BoostedRegressor {
                        trees: tree_order
                            .iter()
                            .map(|tree_index| {
                                Self::from_model(
                                    &model.trees()[*tree_index],
                                    feature_index_map,
                                    missing_features,
                                )
                            })
                            .collect(),
                        tree_weights: tree_order
                            .iter()
                            .map(|tree_index| model.tree_weights()[*tree_index])
                            .collect(),
                        base_score: model.base_score(),
                    }
                }
            },
        }
    }

    fn from_classifier(
        classifier: &DecisionTreeClassifier,
        feature_index_map: &[usize],
        missing_features: Option<&BTreeSet<usize>>,
    ) -> Self {
        match classifier.structure() {
            tree::classifier::TreeStructure::Standard { nodes, root } => {
                if classifier_nodes_are_binary_only(nodes) {
                    return Self::BinaryClassifier {
                        nodes: build_binary_classifier_layout(
                            nodes,
                            *root,
                            classifier.class_labels(),
                            feature_index_map,
                            classifier.feature_preprocessing(),
                            missing_features,
                        ),
                        class_labels: classifier.class_labels().to_vec(),
                    };
                }

                let optimized_nodes = nodes
                    .iter()
                    .map(|node| match node {
                        tree::classifier::TreeNode::Leaf { class_counts, .. } => {
                            OptimizedClassifierNode::Leaf(normalized_probabilities_from_counts(
                                class_counts,
                            ))
                        }
                        tree::classifier::TreeNode::BinarySplit {
                            feature_index,
                            threshold_bin,
                            missing_direction,
                            left_child,
                            right_child,
                            class_counts,
                            ..
                        } => OptimizedClassifierNode::Binary {
                            feature_index: remap_feature_index(*feature_index, feature_index_map),
                            threshold_bin: *threshold_bin,
                            children: [*left_child, *right_child],
                            missing_bin: optimized_missing_bin(
                                classifier.feature_preprocessing(),
                                *feature_index,
                                missing_features,
                            ),
                            missing_child: if missing_feature_enabled(
                                *feature_index,
                                missing_features,
                            ) {
                                match missing_direction {
                                    tree::shared::MissingBranchDirection::Left => Some(*left_child),
                                    tree::shared::MissingBranchDirection::Right => {
                                        Some(*right_child)
                                    }
                                    tree::shared::MissingBranchDirection::Node => None,
                                }
                            } else {
                                None
                            },
                            missing_probabilities: if missing_feature_enabled(
                                *feature_index,
                                missing_features,
                            ) && matches!(
                                missing_direction,
                                tree::shared::MissingBranchDirection::Node
                            ) {
                                Some(normalized_probabilities_from_counts(class_counts))
                            } else {
                                None
                            },
                        },
                        tree::classifier::TreeNode::MultiwaySplit {
                            feature_index,
                            class_counts,
                            branches,
                            missing_child,
                            ..
                        } => {
                            let max_bin_index = branches
                                .iter()
                                .map(|(bin, _)| usize::from(*bin))
                                .max()
                                .unwrap_or(0);
                            let mut child_lookup = vec![usize::MAX; max_bin_index + 1];
                            for (bin, child_index) in branches {
                                child_lookup[usize::from(*bin)] = *child_index;
                            }
                            OptimizedClassifierNode::Multiway {
                                feature_index: remap_feature_index(
                                    *feature_index,
                                    feature_index_map,
                                ),
                                child_lookup,
                                max_bin_index,
                                missing_bin: optimized_missing_bin(
                                    classifier.feature_preprocessing(),
                                    *feature_index,
                                    missing_features,
                                ),
                                missing_child: if missing_feature_enabled(
                                    *feature_index,
                                    missing_features,
                                ) {
                                    *missing_child
                                } else {
                                    None
                                },
                                fallback_probabilities: normalized_probabilities_from_counts(
                                    class_counts,
                                ),
                            }
                        }
                        tree::classifier::TreeNode::ObliqueSplit { .. } => {
                            unreachable!("oblique nodes are rejected before optimized lowering")
                        }
                    })
                    .collect();

                Self::StandardClassifier {
                    nodes: optimized_nodes,
                    root: *root,
                    class_labels: classifier.class_labels().to_vec(),
                }
            }
            tree::classifier::TreeStructure::Oblivious {
                splits,
                leaf_class_counts,
                ..
            } => Self::ObliviousClassifier {
                feature_indices: splits
                    .iter()
                    .map(|split| remap_feature_index(split.feature_index, feature_index_map))
                    .collect(),
                threshold_bins: splits.iter().map(|split| split.threshold_bin).collect(),
                leaf_values: leaf_class_counts
                    .iter()
                    .map(|class_counts| normalized_probabilities_from_counts(class_counts))
                    .collect(),
                class_labels: classifier.class_labels().to_vec(),
            },
        }
    }

    fn from_regressor(
        regressor: &DecisionTreeRegressor,
        feature_index_map: &[usize],
        missing_features: Option<&BTreeSet<usize>>,
    ) -> Self {
        match regressor.structure() {
            tree::regressor::RegressionTreeStructure::Standard { nodes, root } => {
                Self::BinaryRegressor {
                    nodes: build_binary_regressor_layout(
                        nodes,
                        *root,
                        feature_index_map,
                        regressor.feature_preprocessing(),
                        missing_features,
                    ),
                }
            }
            tree::regressor::RegressionTreeStructure::Oblivious {
                splits,
                leaf_values,
                ..
            } => Self::ObliviousRegressor {
                feature_indices: splits
                    .iter()
                    .map(|split| remap_feature_index(split.feature_index, feature_index_map))
                    .collect(),
                threshold_bins: splits.iter().map(|split| split.threshold_bin).collect(),
                leaf_values: leaf_values.clone(),
            },
        }
    }

    #[inline(always)]
    pub(crate) fn predict_table_row(&self, table: &dyn TableAccess, row_index: usize) -> f64 {
        match self {
            OptimizedRuntime::BinaryClassifier { .. }
            | OptimizedRuntime::StandardClassifier { .. }
            | OptimizedRuntime::ObliviousClassifier { .. }
            | OptimizedRuntime::ForestClassifier { .. }
            | OptimizedRuntime::BoostedBinaryClassifier { .. } => {
                let probabilities = self
                    .predict_proba_table_row(table, row_index)
                    .expect("classifier runtime supports probability prediction");
                class_label_from_probabilities(&probabilities, self.class_labels())
            }
            OptimizedRuntime::BinaryRegressor { nodes } => predict_binary_regressor_row(
                nodes,
                |feature_index| table.binned_value(feature_index, row_index),
                |feature_index| table.feature_value(feature_index, row_index),
            ),
            OptimizedRuntime::ObliviousRegressor {
                feature_indices,
                threshold_bins,
                leaf_values,
            } => predict_oblivious_row(
                feature_indices,
                threshold_bins,
                leaf_values,
                |feature_index| table.binned_value(feature_index, row_index),
            ),
            OptimizedRuntime::ForestRegressor { trees } => {
                trees
                    .iter()
                    .map(|tree| tree.predict_table_row(table, row_index))
                    .sum::<f64>()
                    / trees.len() as f64
            }
            OptimizedRuntime::BoostedRegressor {
                trees,
                tree_weights,
                base_score,
            } => {
                *base_score
                    + trees
                        .iter()
                        .zip(tree_weights.iter().copied())
                        .map(|(tree, weight)| weight * tree.predict_table_row(table, row_index))
                        .sum::<f64>()
            }
        }
    }

    #[inline(always)]
    pub(crate) fn predict_proba_table_row(
        &self,
        table: &dyn TableAccess,
        row_index: usize,
    ) -> Result<Vec<f64>, PredictError> {
        match self {
            OptimizedRuntime::BinaryClassifier { nodes, .. } => {
                Ok(predict_binary_classifier_probabilities_row(
                    nodes,
                    |feature_index| table.binned_value(feature_index, row_index),
                    |feature_index| table.feature_value(feature_index, row_index),
                )
                .to_vec())
            }
            OptimizedRuntime::StandardClassifier { nodes, root, .. } => Ok(
                predict_standard_classifier_probabilities_row(nodes, *root, |feature_index| {
                    table.binned_value(feature_index, row_index)
                })
                .to_vec(),
            ),
            OptimizedRuntime::ObliviousClassifier {
                feature_indices,
                threshold_bins,
                leaf_values,
                ..
            } => Ok(predict_oblivious_probabilities_row(
                feature_indices,
                threshold_bins,
                leaf_values,
                |feature_index| table.binned_value(feature_index, row_index),
            )
            .to_vec()),
            OptimizedRuntime::ForestClassifier { trees, .. } => {
                let mut totals = trees[0].predict_proba_table_row(table, row_index)?;
                for tree in &trees[1..] {
                    let row = tree.predict_proba_table_row(table, row_index)?;
                    for (total, value) in totals.iter_mut().zip(row) {
                        *total += value;
                    }
                }
                let tree_count = trees.len() as f64;
                for value in &mut totals {
                    *value /= tree_count;
                }
                Ok(totals)
            }
            OptimizedRuntime::BoostedBinaryClassifier {
                trees,
                tree_weights,
                base_score,
                ..
            } => {
                let raw_score = *base_score
                    + trees
                        .iter()
                        .zip(tree_weights.iter().copied())
                        .map(|(tree, weight)| weight * tree.predict_table_row(table, row_index))
                        .sum::<f64>();
                let positive = sigmoid(raw_score);
                Ok(vec![1.0 - positive, positive])
            }
            OptimizedRuntime::BinaryRegressor { .. }
            | OptimizedRuntime::ObliviousRegressor { .. }
            | OptimizedRuntime::ForestRegressor { .. }
            | OptimizedRuntime::BoostedRegressor { .. } => {
                Err(PredictError::ProbabilityPredictionRequiresClassification)
            }
        }
    }

    pub(crate) fn predict_proba_table(
        &self,
        table: &dyn TableAccess,
        executor: &InferenceExecutor,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        match self {
            OptimizedRuntime::BinaryClassifier { .. }
            | OptimizedRuntime::StandardClassifier { .. }
            | OptimizedRuntime::ObliviousClassifier { .. }
            | OptimizedRuntime::ForestClassifier { .. }
            | OptimizedRuntime::BoostedBinaryClassifier { .. } => {
                if self.should_use_batch_matrix(table.n_rows()) {
                    let matrix = ColumnMajorBinnedMatrix::from_table_access(table);
                    self.predict_proba_column_major_matrix(&matrix, executor)
                } else {
                    (0..table.n_rows())
                        .map(|row_index| self.predict_proba_table_row(table, row_index))
                        .collect()
                }
            }
            OptimizedRuntime::BinaryRegressor { .. }
            | OptimizedRuntime::ObliviousRegressor { .. }
            | OptimizedRuntime::ForestRegressor { .. }
            | OptimizedRuntime::BoostedRegressor { .. } => {
                Err(PredictError::ProbabilityPredictionRequiresClassification)
            }
        }
    }

    pub(crate) fn predict_column_major_matrix(
        &self,
        matrix: &ColumnMajorBinnedMatrix,
        executor: &InferenceExecutor,
    ) -> Vec<f64> {
        match self {
            OptimizedRuntime::BinaryClassifier { .. }
            | OptimizedRuntime::StandardClassifier { .. }
            | OptimizedRuntime::ObliviousClassifier { .. }
            | OptimizedRuntime::ForestClassifier { .. }
            | OptimizedRuntime::BoostedBinaryClassifier { .. } => self
                .predict_proba_column_major_matrix(matrix, executor)
                .expect("classifier runtime supports probability prediction")
                .into_iter()
                .map(|row| class_label_from_probabilities(&row, self.class_labels()))
                .collect(),
            OptimizedRuntime::BinaryRegressor { nodes } => {
                predict_binary_regressor_column_major_matrix(nodes, matrix, executor)
            }
            OptimizedRuntime::ObliviousRegressor {
                feature_indices,
                threshold_bins,
                leaf_values,
            } => predict_oblivious_column_major_matrix(
                feature_indices,
                threshold_bins,
                leaf_values,
                matrix,
                executor,
            ),
            OptimizedRuntime::ForestRegressor { trees } => {
                let mut totals = trees[0].predict_column_major_matrix(matrix, executor);
                for tree in &trees[1..] {
                    let values = tree.predict_column_major_matrix(matrix, executor);
                    for (total, value) in totals.iter_mut().zip(values) {
                        *total += value;
                    }
                }
                let tree_count = trees.len() as f64;
                for total in &mut totals {
                    *total /= tree_count;
                }
                totals
            }
            OptimizedRuntime::BoostedRegressor {
                trees,
                tree_weights,
                base_score,
            } => {
                let mut totals = vec![*base_score; matrix.n_rows];
                for (tree, weight) in trees.iter().zip(tree_weights.iter().copied()) {
                    let values = tree.predict_column_major_matrix(matrix, executor);
                    for (total, value) in totals.iter_mut().zip(values) {
                        *total += weight * value;
                    }
                }
                totals
            }
        }
    }

    pub(crate) fn predict_proba_column_major_matrix(
        &self,
        matrix: &ColumnMajorBinnedMatrix,
        executor: &InferenceExecutor,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        match self {
            OptimizedRuntime::BinaryClassifier { nodes, .. } => {
                Ok(predict_binary_classifier_probabilities_column_major_matrix(
                    nodes, matrix, executor,
                ))
            }
            OptimizedRuntime::StandardClassifier { .. } => Ok((0..matrix.n_rows)
                .map(|row_index| {
                    self.predict_proba_binned_row_from_columns(matrix, row_index)
                        .expect("classifier runtime supports probability prediction")
                })
                .collect()),
            OptimizedRuntime::ObliviousClassifier {
                feature_indices,
                threshold_bins,
                leaf_values,
                ..
            } => Ok(predict_oblivious_probabilities_column_major_matrix(
                feature_indices,
                threshold_bins,
                leaf_values,
                matrix,
                executor,
            )),
            OptimizedRuntime::ForestClassifier { trees, .. } => {
                let mut totals = trees[0].predict_proba_column_major_matrix(matrix, executor)?;
                for tree in &trees[1..] {
                    let rows = tree.predict_proba_column_major_matrix(matrix, executor)?;
                    for (row_totals, row_values) in totals.iter_mut().zip(rows) {
                        for (total, value) in row_totals.iter_mut().zip(row_values) {
                            *total += value;
                        }
                    }
                }
                let tree_count = trees.len() as f64;
                for row in &mut totals {
                    for value in row {
                        *value /= tree_count;
                    }
                }
                Ok(totals)
            }
            OptimizedRuntime::BoostedBinaryClassifier {
                trees,
                tree_weights,
                base_score,
                ..
            } => {
                let mut raw_scores = vec![*base_score; matrix.n_rows];
                for (tree, weight) in trees.iter().zip(tree_weights.iter().copied()) {
                    let values = tree.predict_column_major_matrix(matrix, executor);
                    for (raw_score, value) in raw_scores.iter_mut().zip(values) {
                        *raw_score += weight * value;
                    }
                }
                Ok(raw_scores
                    .into_iter()
                    .map(|raw_score| {
                        let positive = sigmoid(raw_score);
                        vec![1.0 - positive, positive]
                    })
                    .collect())
            }
            OptimizedRuntime::BinaryRegressor { .. }
            | OptimizedRuntime::ObliviousRegressor { .. }
            | OptimizedRuntime::ForestRegressor { .. }
            | OptimizedRuntime::BoostedRegressor { .. } => {
                Err(PredictError::ProbabilityPredictionRequiresClassification)
            }
        }
    }

    pub(crate) fn class_labels(&self) -> &[f64] {
        match self {
            OptimizedRuntime::BinaryClassifier { class_labels, .. }
            | OptimizedRuntime::StandardClassifier { class_labels, .. }
            | OptimizedRuntime::ObliviousClassifier { class_labels, .. }
            | OptimizedRuntime::ForestClassifier { class_labels, .. }
            | OptimizedRuntime::BoostedBinaryClassifier { class_labels, .. } => class_labels,
            _ => &[],
        }
    }

    #[inline(always)]
    pub(crate) fn predict_binned_row_from_columns(
        &self,
        matrix: &ColumnMajorBinnedMatrix,
        row_index: usize,
    ) -> f64 {
        match self {
            OptimizedRuntime::BinaryRegressor { nodes } => {
                predict_binary_regressor_row(nodes, |_| 0, |_| f64::NAN)
            }
            OptimizedRuntime::ObliviousRegressor {
                feature_indices,
                threshold_bins,
                leaf_values,
            } => predict_oblivious_row(
                feature_indices,
                threshold_bins,
                leaf_values,
                |feature_index| matrix.column(feature_index).value_at(row_index),
            ),
            OptimizedRuntime::BoostedRegressor {
                trees,
                tree_weights,
                base_score,
            } => {
                *base_score
                    + trees
                        .iter()
                        .zip(tree_weights.iter().copied())
                        .map(|(tree, weight)| {
                            weight * tree.predict_binned_row_from_columns(matrix, row_index)
                        })
                        .sum::<f64>()
            }
            _ => self.predict_column_major_matrix(
                matrix,
                &InferenceExecutor::new(1).expect("inference executor"),
            )[row_index],
        }
    }

    #[inline(always)]
    pub(crate) fn predict_proba_binned_row_from_columns(
        &self,
        matrix: &ColumnMajorBinnedMatrix,
        row_index: usize,
    ) -> Result<Vec<f64>, PredictError> {
        match self {
            OptimizedRuntime::BinaryClassifier { nodes, .. } => Ok(
                predict_binary_classifier_probabilities_row(nodes, |_| 0, |_| f64::NAN).to_vec(),
            ),
            OptimizedRuntime::StandardClassifier { nodes, root, .. } => Ok(
                predict_standard_classifier_probabilities_row(nodes, *root, |feature_index| {
                    matrix.column(feature_index).value_at(row_index)
                })
                .to_vec(),
            ),
            OptimizedRuntime::ObliviousClassifier {
                feature_indices,
                threshold_bins,
                leaf_values,
                ..
            } => Ok(predict_oblivious_probabilities_row(
                feature_indices,
                threshold_bins,
                leaf_values,
                |feature_index| matrix.column(feature_index).value_at(row_index),
            )
            .to_vec()),
            OptimizedRuntime::ForestClassifier { trees, .. } => {
                let mut totals =
                    trees[0].predict_proba_binned_row_from_columns(matrix, row_index)?;
                for tree in &trees[1..] {
                    let row = tree.predict_proba_binned_row_from_columns(matrix, row_index)?;
                    for (total, value) in totals.iter_mut().zip(row) {
                        *total += value;
                    }
                }
                let tree_count = trees.len() as f64;
                for value in &mut totals {
                    *value /= tree_count;
                }
                Ok(totals)
            }
            OptimizedRuntime::BoostedBinaryClassifier {
                trees,
                tree_weights,
                base_score,
                ..
            } => {
                let raw_score = *base_score
                    + trees
                        .iter()
                        .zip(tree_weights.iter().copied())
                        .map(|(tree, weight)| {
                            weight * tree.predict_binned_row_from_columns(matrix, row_index)
                        })
                        .sum::<f64>();
                let positive = sigmoid(raw_score);
                Ok(vec![1.0 - positive, positive])
            }
            OptimizedRuntime::BinaryRegressor { .. }
            | OptimizedRuntime::ObliviousRegressor { .. }
            | OptimizedRuntime::ForestRegressor { .. }
            | OptimizedRuntime::BoostedRegressor { .. } => {
                Err(PredictError::ProbabilityPredictionRequiresClassification)
            }
        }
    }
}

#[inline(always)]
fn predict_standard_classifier_probabilities_row<F>(
    nodes: &[OptimizedClassifierNode],
    root: usize,
    bin_at: F,
) -> &[f64]
where
    F: Fn(usize) -> u16,
{
    let mut node_index = root;
    loop {
        match &nodes[node_index] {
            OptimizedClassifierNode::Leaf(value) => return value,
            OptimizedClassifierNode::Binary {
                feature_index,
                threshold_bin,
                children,
                missing_bin,
                missing_child,
                missing_probabilities,
            } => {
                let bin = bin_at(*feature_index);
                if missing_bin.is_some_and(|expected| expected == bin) {
                    if let Some(probabilities) = missing_probabilities {
                        return probabilities;
                    }
                    if let Some(child_index) = missing_child {
                        node_index = *child_index;
                        continue;
                    }
                }
                let go_right = usize::from(bin > *threshold_bin);
                node_index = children[go_right];
            }
            OptimizedClassifierNode::Multiway {
                feature_index,
                child_lookup,
                max_bin_index,
                missing_bin,
                missing_child,
                fallback_probabilities,
            } => {
                let bin_value = bin_at(*feature_index);
                if missing_bin.is_some_and(|expected| expected == bin_value) {
                    if let Some(child_index) = missing_child {
                        node_index = *child_index;
                        continue;
                    }
                    return fallback_probabilities;
                }
                let bin = usize::from(bin_value);
                if bin > *max_bin_index {
                    return fallback_probabilities;
                }
                let child_index = child_lookup[bin];
                if child_index == usize::MAX {
                    return fallback_probabilities;
                }
                node_index = child_index;
            }
        }
    }
}

#[inline(always)]
fn predict_binary_classifier_probabilities_row<F, G>(
    nodes: &[OptimizedBinaryClassifierNode],
    bin_at: F,
    value_at: G,
) -> &[f64]
where
    F: Fn(usize) -> u16,
    G: Fn(usize) -> f64,
{
    let mut node_index = 0usize;
    loop {
        match &nodes[node_index] {
            OptimizedBinaryClassifierNode::Leaf(value) => return value,
            OptimizedBinaryClassifierNode::Branch {
                feature_index,
                threshold_bin,
                jump_index,
                jump_if_greater,
                missing_bin,
                missing_jump_index,
                missing_probabilities,
            } => {
                let bin = bin_at(*feature_index);
                if missing_bin.is_some_and(|expected| expected == bin) {
                    if let Some(probabilities) = missing_probabilities {
                        return probabilities;
                    }
                    if let Some(jump_index) = missing_jump_index {
                        node_index = *jump_index;
                        continue;
                    }
                }
                let go_right = bin > *threshold_bin;
                node_index = if go_right == *jump_if_greater {
                    *jump_index
                } else {
                    node_index + 1
                };
            }
            OptimizedBinaryClassifierNode::ObliqueBranch {
                feature_indices,
                weights,
                missing_directions,
                threshold,
                jump_index,
                jump_if_greater,
                missing_probabilities,
            } => {
                let left_value = value_at(feature_indices[0]);
                let right_value = value_at(feature_indices[1]);
                let missing_mask =
                    u8::from(!left_value.is_finite()) | (u8::from(!right_value.is_finite()) << 1);
                if let Some(go_left) = crate::tree::oblique::resolve_oblique_missing_direction(
                    missing_mask,
                    *weights,
                    *missing_directions,
                ) {
                    node_index = if go_left == *jump_if_greater {
                        node_index + 1
                    } else {
                        *jump_index
                    };
                    continue;
                }
                if missing_mask != 0 {
                    return missing_probabilities;
                }
                let go_right = weights[0] * left_value + weights[1] * right_value > *threshold;
                node_index = if go_right == *jump_if_greater {
                    *jump_index
                } else {
                    node_index + 1
                };
            }
        }
    }
}

#[inline(always)]
fn predict_binary_regressor_row<F, G>(
    nodes: &[OptimizedBinaryRegressorNode],
    bin_at: F,
    value_at: G,
) -> f64
where
    F: Fn(usize) -> u16,
    G: Fn(usize) -> f64,
{
    let mut node_index = 0usize;
    loop {
        match &nodes[node_index] {
            OptimizedBinaryRegressorNode::Leaf(value) => return *value,
            OptimizedBinaryRegressorNode::Branch {
                feature_index,
                threshold_bin,
                jump_index,
                jump_if_greater,
                missing_bin,
                missing_jump_index,
                missing_value,
            } => {
                let bin = bin_at(*feature_index);
                if missing_bin.is_some_and(|expected| expected == bin) {
                    if let Some(value) = missing_value {
                        return *value;
                    }
                    if let Some(jump_index) = missing_jump_index {
                        node_index = *jump_index;
                        continue;
                    }
                }
                let go_right = bin > *threshold_bin;
                node_index = if go_right == *jump_if_greater {
                    *jump_index
                } else {
                    node_index + 1
                };
            }
            OptimizedBinaryRegressorNode::ObliqueBranch {
                feature_indices,
                weights,
                missing_directions,
                threshold,
                jump_index,
                jump_if_greater,
                missing_value,
            } => {
                let left_value = value_at(feature_indices[0]);
                let right_value = value_at(feature_indices[1]);
                let missing_mask =
                    u8::from(!left_value.is_finite()) | (u8::from(!right_value.is_finite()) << 1);
                if let Some(go_left) = crate::tree::oblique::resolve_oblique_missing_direction(
                    missing_mask,
                    *weights,
                    *missing_directions,
                ) {
                    node_index = if go_left == *jump_if_greater {
                        node_index + 1
                    } else {
                        *jump_index
                    };
                    continue;
                }
                if missing_mask != 0 {
                    return *missing_value;
                }
                let go_right = weights[0] * left_value + weights[1] * right_value > *threshold;
                node_index = if go_right == *jump_if_greater {
                    *jump_index
                } else {
                    node_index + 1
                };
            }
        }
    }
}

fn predict_binary_classifier_probabilities_column_major_matrix(
    nodes: &[OptimizedBinaryClassifierNode],
    matrix: &ColumnMajorBinnedMatrix,
    _executor: &InferenceExecutor,
) -> Vec<Vec<f64>> {
    if binary_classifier_nodes_require_rowwise_missing(nodes)
        || binary_classifier_nodes_require_rowwise_raw(nodes)
    {
        return (0..matrix.n_rows)
            .map(|row_index| {
                predict_binary_classifier_probabilities_row(
                    nodes,
                    |feature_index| matrix.column(feature_index).value_at(row_index),
                    |_| f64::NAN,
                )
                .to_vec()
            })
            .collect();
    }
    (0..matrix.n_rows)
        .map(|row_index| {
            predict_binary_classifier_probabilities_row(
                nodes,
                |feature_index| matrix.column(feature_index).value_at(row_index),
                |_| f64::NAN,
            )
            .to_vec()
        })
        .collect()
}

fn predict_binary_regressor_column_major_matrix(
    nodes: &[OptimizedBinaryRegressorNode],
    matrix: &ColumnMajorBinnedMatrix,
    executor: &InferenceExecutor,
) -> Vec<f64> {
    if binary_regressor_nodes_require_rowwise_missing(nodes)
        || binary_regressor_nodes_require_rowwise_raw(nodes)
    {
        return (0..matrix.n_rows)
            .map(|row_index| {
                predict_binary_regressor_row(
                    nodes,
                    |feature_index| matrix.column(feature_index).value_at(row_index),
                    |_| f64::NAN,
                )
            })
            .collect();
    }
    let mut outputs = vec![0.0; matrix.n_rows];
    executor.fill_chunks(
        &mut outputs,
        STANDARD_BATCH_INFERENCE_CHUNK_ROWS,
        |start_row, chunk| predict_binary_regressor_chunk(nodes, matrix, start_row, chunk),
    );
    outputs
}

fn predict_binary_regressor_chunk(
    nodes: &[OptimizedBinaryRegressorNode],
    matrix: &ColumnMajorBinnedMatrix,
    start_row: usize,
    output: &mut [f64],
) {
    let mut row_indices: Vec<usize> = (0..output.len()).collect();
    let mut stack = vec![(0usize, 0usize, output.len())];

    while let Some((node_index, start, end)) = stack.pop() {
        match &nodes[node_index] {
            OptimizedBinaryRegressorNode::Leaf(value) => {
                for position in start..end {
                    output[row_indices[position]] = *value;
                }
            }
            OptimizedBinaryRegressorNode::Branch {
                feature_index,
                threshold_bin,
                jump_index,
                jump_if_greater,
                ..
            } => {
                let fallthrough_index = node_index + 1;
                if *jump_index == fallthrough_index {
                    stack.push((fallthrough_index, start, end));
                    continue;
                }

                let column = matrix.column(*feature_index);
                let mut partition = start;
                let mut jump_start = end;
                match column {
                    CompactBinnedColumn::U8(values) if *threshold_bin <= u16::from(u8::MAX) => {
                        let threshold = *threshold_bin as u8;
                        while partition < jump_start {
                            let row_offset = row_indices[partition];
                            let go_right = values[start_row + row_offset] > threshold;
                            let goes_jump = go_right == *jump_if_greater;
                            if goes_jump {
                                jump_start -= 1;
                                row_indices.swap(partition, jump_start);
                            } else {
                                partition += 1;
                            }
                        }
                    }
                    _ => {
                        while partition < jump_start {
                            let row_offset = row_indices[partition];
                            let go_right = column.value_at(start_row + row_offset) > *threshold_bin;
                            let goes_jump = go_right == *jump_if_greater;
                            if goes_jump {
                                jump_start -= 1;
                                row_indices.swap(partition, jump_start);
                            } else {
                                partition += 1;
                            }
                        }
                    }
                }

                if jump_start < end {
                    stack.push((*jump_index, jump_start, end));
                }
                if start < jump_start {
                    stack.push((fallthrough_index, start, jump_start));
                }
            }
            OptimizedBinaryRegressorNode::ObliqueBranch { .. } => {
                unreachable!("oblique regressor nodes should use rowwise prediction");
            }
        }
    }
}

fn binary_classifier_nodes_require_rowwise_missing(
    nodes: &[OptimizedBinaryClassifierNode],
) -> bool {
    nodes.iter().any(|node| match node {
        OptimizedBinaryClassifierNode::Leaf(_) => false,
        OptimizedBinaryClassifierNode::Branch {
            missing_bin,
            missing_jump_index,
            missing_probabilities,
            ..
        } => {
            missing_bin.is_some() || missing_jump_index.is_some() || missing_probabilities.is_some()
        }
        OptimizedBinaryClassifierNode::ObliqueBranch { .. } => false,
    })
}

fn binary_regressor_nodes_require_rowwise_missing(nodes: &[OptimizedBinaryRegressorNode]) -> bool {
    nodes.iter().any(|node| match node {
        OptimizedBinaryRegressorNode::Leaf(_) => false,
        OptimizedBinaryRegressorNode::Branch {
            missing_bin,
            missing_jump_index,
            missing_value,
            ..
        } => missing_bin.is_some() || missing_jump_index.is_some() || missing_value.is_some(),
        OptimizedBinaryRegressorNode::ObliqueBranch { .. } => false,
    })
}

fn binary_classifier_nodes_require_rowwise_raw(nodes: &[OptimizedBinaryClassifierNode]) -> bool {
    nodes
        .iter()
        .any(|node| matches!(node, OptimizedBinaryClassifierNode::ObliqueBranch { .. }))
}

fn binary_regressor_nodes_require_rowwise_raw(nodes: &[OptimizedBinaryRegressorNode]) -> bool {
    nodes
        .iter()
        .any(|node| matches!(node, OptimizedBinaryRegressorNode::ObliqueBranch { .. }))
}

#[inline(always)]
fn predict_oblivious_row<F>(
    feature_indices: &[usize],
    threshold_bins: &[u16],
    leaf_values: &[f64],
    bin_at: F,
) -> f64
where
    F: Fn(usize) -> u16,
{
    let mut leaf_index = 0usize;
    for (&feature_index, &threshold_bin) in feature_indices.iter().zip(threshold_bins) {
        let go_right = usize::from(bin_at(feature_index) > threshold_bin);
        leaf_index = (leaf_index << 1) | go_right;
    }
    leaf_values[leaf_index]
}

#[inline(always)]
fn predict_oblivious_probabilities_row<'a, F>(
    feature_indices: &[usize],
    threshold_bins: &[u16],
    leaf_values: &'a [Vec<f64>],
    bin_at: F,
) -> &'a [f64]
where
    F: Fn(usize) -> u16,
{
    let mut leaf_index = 0usize;
    for (&feature_index, &threshold_bin) in feature_indices.iter().zip(threshold_bins) {
        let go_right = usize::from(bin_at(feature_index) > threshold_bin);
        leaf_index = (leaf_index << 1) | go_right;
    }
    leaf_values[leaf_index].as_slice()
}

fn normalized_probabilities_from_counts(class_counts: &[f64]) -> Vec<f64> {
    let total: f64 = class_counts.iter().sum();
    if total == 0.0 {
        return vec![0.0; class_counts.len()];
    }

    class_counts.iter().map(|count| count / total).collect()
}

fn class_label_from_probabilities(probabilities: &[f64], class_labels: &[f64]) -> f64 {
    let best_index = probabilities
        .iter()
        .copied()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .expect("classification probability row is non-empty");
    class_labels[best_index]
}

#[inline(always)]
fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn classifier_nodes_are_binary_only(nodes: &[tree::classifier::TreeNode]) -> bool {
    nodes.iter().all(|node| {
        matches!(
            node,
            tree::classifier::TreeNode::Leaf { .. }
                | tree::classifier::TreeNode::BinarySplit { .. }
                | tree::classifier::TreeNode::ObliqueSplit { .. }
        )
    })
}

fn classifier_node_sample_count(nodes: &[tree::classifier::TreeNode], node_index: usize) -> usize {
    match &nodes[node_index] {
        tree::classifier::TreeNode::Leaf { sample_count, .. }
        | tree::classifier::TreeNode::BinarySplit { sample_count, .. }
        | tree::classifier::TreeNode::ObliqueSplit { sample_count, .. }
        | tree::classifier::TreeNode::MultiwaySplit { sample_count, .. } => *sample_count,
    }
}

fn build_binary_classifier_layout(
    nodes: &[tree::classifier::TreeNode],
    root: usize,
    _class_labels: &[f64],
    feature_index_map: &[usize],
    preprocessing: &[FeaturePreprocessing],
    missing_features: Option<&BTreeSet<usize>>,
) -> Vec<OptimizedBinaryClassifierNode> {
    let mut layout = Vec::with_capacity(nodes.len());
    append_binary_classifier_node(
        nodes,
        root,
        &mut layout,
        feature_index_map,
        preprocessing,
        missing_features,
    );
    layout
}

fn append_binary_classifier_node(
    nodes: &[tree::classifier::TreeNode],
    node_index: usize,
    layout: &mut Vec<OptimizedBinaryClassifierNode>,
    feature_index_map: &[usize],
    preprocessing: &[FeaturePreprocessing],
    missing_features: Option<&BTreeSet<usize>>,
) -> usize {
    let current_index = layout.len();
    layout.push(OptimizedBinaryClassifierNode::Leaf(Vec::new()));

    match &nodes[node_index] {
        tree::classifier::TreeNode::Leaf { class_counts, .. } => {
            layout[current_index] = OptimizedBinaryClassifierNode::Leaf(
                normalized_probabilities_from_counts(class_counts),
            );
        }
        tree::classifier::TreeNode::BinarySplit {
            feature_index,
            threshold_bin,
            missing_direction,
            left_child,
            right_child,
            class_counts,
            ..
        } => {
            let (fallthrough_child, jump_child, jump_if_greater) = if left_child == right_child {
                (*left_child, *left_child, true)
            } else {
                let left_count = classifier_node_sample_count(nodes, *left_child);
                let right_count = classifier_node_sample_count(nodes, *right_child);
                if left_count >= right_count {
                    (*left_child, *right_child, true)
                } else {
                    (*right_child, *left_child, false)
                }
            };

            let fallthrough_index = append_binary_classifier_node(
                nodes,
                fallthrough_child,
                layout,
                feature_index_map,
                preprocessing,
                missing_features,
            );
            debug_assert_eq!(fallthrough_index, current_index + 1);
            let jump_index = if jump_child == fallthrough_child {
                fallthrough_index
            } else {
                append_binary_classifier_node(
                    nodes,
                    jump_child,
                    layout,
                    feature_index_map,
                    preprocessing,
                    missing_features,
                )
            };

            let missing_bin =
                optimized_missing_bin(preprocessing, *feature_index, missing_features);
            let (missing_jump_index, missing_probabilities) =
                if missing_feature_enabled(*feature_index, missing_features) {
                    match missing_direction {
                        tree::shared::MissingBranchDirection::Left => (
                            Some(if *left_child == fallthrough_child {
                                fallthrough_index
                            } else {
                                jump_index
                            }),
                            None,
                        ),
                        tree::shared::MissingBranchDirection::Right => (
                            Some(if *right_child == fallthrough_child {
                                fallthrough_index
                            } else {
                                jump_index
                            }),
                            None,
                        ),
                        tree::shared::MissingBranchDirection::Node => (
                            None,
                            Some(normalized_probabilities_from_counts(class_counts)),
                        ),
                    }
                } else {
                    (None, None)
                };

            layout[current_index] = OptimizedBinaryClassifierNode::Branch {
                feature_index: remap_feature_index(*feature_index, feature_index_map),
                threshold_bin: *threshold_bin,
                jump_index,
                jump_if_greater,
                missing_bin,
                missing_jump_index,
                missing_probabilities,
            };
        }
        tree::classifier::TreeNode::MultiwaySplit { .. } => {
            unreachable!("multiway nodes are filtered out before binary layout construction");
        }
        tree::classifier::TreeNode::ObliqueSplit {
            feature_indices,
            weights,
            missing_directions,
            threshold,
            left_child,
            right_child,
            class_counts,
            ..
        } => {
            let (fallthrough_child, jump_child, jump_if_greater) = if left_child == right_child {
                (*left_child, *left_child, true)
            } else {
                let left_count = classifier_node_sample_count(nodes, *left_child);
                let right_count = classifier_node_sample_count(nodes, *right_child);
                if left_count >= right_count {
                    (*left_child, *right_child, true)
                } else {
                    (*right_child, *left_child, false)
                }
            };

            let fallthrough_index = append_binary_classifier_node(
                nodes,
                fallthrough_child,
                layout,
                feature_index_map,
                preprocessing,
                missing_features,
            );
            debug_assert_eq!(fallthrough_index, current_index + 1);
            let jump_index = if jump_child == fallthrough_child {
                fallthrough_index
            } else {
                append_binary_classifier_node(
                    nodes,
                    jump_child,
                    layout,
                    feature_index_map,
                    preprocessing,
                    missing_features,
                )
            };

            layout[current_index] = OptimizedBinaryClassifierNode::ObliqueBranch {
                feature_indices: [
                    remap_feature_index(feature_indices[0], feature_index_map),
                    remap_feature_index(feature_indices[1], feature_index_map),
                ],
                weights: [weights[0], weights[1]],
                missing_directions: [missing_directions[0], missing_directions[1]],
                threshold: *threshold,
                jump_index,
                jump_if_greater,
                missing_probabilities: normalized_probabilities_from_counts(class_counts),
            };
        }
    }

    current_index
}

fn regressor_node_sample_count(
    nodes: &[tree::regressor::RegressionNode],
    node_index: usize,
) -> usize {
    match &nodes[node_index] {
        tree::regressor::RegressionNode::Leaf { sample_count, .. }
        | tree::regressor::RegressionNode::BinarySplit { sample_count, .. }
        | tree::regressor::RegressionNode::ObliqueSplit { sample_count, .. } => *sample_count,
    }
}

fn build_binary_regressor_layout(
    nodes: &[tree::regressor::RegressionNode],
    root: usize,
    feature_index_map: &[usize],
    preprocessing: &[FeaturePreprocessing],
    missing_features: Option<&BTreeSet<usize>>,
) -> Vec<OptimizedBinaryRegressorNode> {
    let mut layout = Vec::with_capacity(nodes.len());
    append_binary_regressor_node(
        nodes,
        root,
        &mut layout,
        feature_index_map,
        preprocessing,
        missing_features,
    );
    layout
}

fn append_binary_regressor_node(
    nodes: &[tree::regressor::RegressionNode],
    node_index: usize,
    layout: &mut Vec<OptimizedBinaryRegressorNode>,
    feature_index_map: &[usize],
    preprocessing: &[FeaturePreprocessing],
    missing_features: Option<&BTreeSet<usize>>,
) -> usize {
    let current_index = layout.len();
    layout.push(OptimizedBinaryRegressorNode::Leaf(0.0));

    match &nodes[node_index] {
        tree::regressor::RegressionNode::Leaf { value, .. } => {
            layout[current_index] = OptimizedBinaryRegressorNode::Leaf(*value);
        }
        tree::regressor::RegressionNode::BinarySplit {
            feature_index,
            threshold_bin,
            missing_direction,
            missing_value,
            left_child,
            right_child,
            ..
        } => {
            let (fallthrough_child, jump_child, jump_if_greater) = if left_child == right_child {
                (*left_child, *left_child, true)
            } else {
                let left_count = regressor_node_sample_count(nodes, *left_child);
                let right_count = regressor_node_sample_count(nodes, *right_child);
                if left_count >= right_count {
                    (*left_child, *right_child, true)
                } else {
                    (*right_child, *left_child, false)
                }
            };

            let fallthrough_index = append_binary_regressor_node(
                nodes,
                fallthrough_child,
                layout,
                feature_index_map,
                preprocessing,
                missing_features,
            );
            debug_assert_eq!(fallthrough_index, current_index + 1);
            let jump_index = if jump_child == fallthrough_child {
                fallthrough_index
            } else {
                append_binary_regressor_node(
                    nodes,
                    jump_child,
                    layout,
                    feature_index_map,
                    preprocessing,
                    missing_features,
                )
            };

            let missing_bin =
                optimized_missing_bin(preprocessing, *feature_index, missing_features);
            let (missing_jump_index, missing_value) =
                if missing_feature_enabled(*feature_index, missing_features) {
                    match missing_direction {
                        tree::shared::MissingBranchDirection::Left => (
                            Some(if *left_child == fallthrough_child {
                                fallthrough_index
                            } else {
                                jump_index
                            }),
                            None,
                        ),
                        tree::shared::MissingBranchDirection::Right => (
                            Some(if *right_child == fallthrough_child {
                                fallthrough_index
                            } else {
                                jump_index
                            }),
                            None,
                        ),
                        tree::shared::MissingBranchDirection::Node => (None, Some(*missing_value)),
                    }
                } else {
                    (None, None)
                };

            layout[current_index] = OptimizedBinaryRegressorNode::Branch {
                feature_index: remap_feature_index(*feature_index, feature_index_map),
                threshold_bin: *threshold_bin,
                jump_index,
                jump_if_greater,
                missing_bin,
                missing_jump_index,
                missing_value,
            };
        }
        tree::regressor::RegressionNode::ObliqueSplit {
            feature_indices,
            weights,
            missing_directions,
            threshold,
            missing_value,
            left_child,
            right_child,
            ..
        } => {
            let (fallthrough_child, jump_child, jump_if_greater) = if left_child == right_child {
                (*left_child, *left_child, true)
            } else {
                let left_count = regressor_node_sample_count(nodes, *left_child);
                let right_count = regressor_node_sample_count(nodes, *right_child);
                if left_count >= right_count {
                    (*left_child, *right_child, true)
                } else {
                    (*right_child, *left_child, false)
                }
            };

            let fallthrough_index = append_binary_regressor_node(
                nodes,
                fallthrough_child,
                layout,
                feature_index_map,
                preprocessing,
                missing_features,
            );
            debug_assert_eq!(fallthrough_index, current_index + 1);
            let jump_index = if jump_child == fallthrough_child {
                fallthrough_index
            } else {
                append_binary_regressor_node(
                    nodes,
                    jump_child,
                    layout,
                    feature_index_map,
                    preprocessing,
                    missing_features,
                )
            };

            layout[current_index] = OptimizedBinaryRegressorNode::ObliqueBranch {
                feature_indices: [
                    remap_feature_index(feature_indices[0], feature_index_map),
                    remap_feature_index(feature_indices[1], feature_index_map),
                ],
                weights: [weights[0], weights[1]],
                missing_directions: [missing_directions[0], missing_directions[1]],
                threshold: *threshold,
                jump_index,
                jump_if_greater,
                missing_value: *missing_value,
            };
        }
    }

    current_index
}

fn predict_oblivious_column_major_matrix(
    feature_indices: &[usize],
    threshold_bins: &[u16],
    leaf_values: &[f64],
    matrix: &ColumnMajorBinnedMatrix,
    executor: &InferenceExecutor,
) -> Vec<f64> {
    let mut outputs = vec![0.0; matrix.n_rows];
    executor.fill_chunks(
        &mut outputs,
        PARALLEL_INFERENCE_CHUNK_ROWS,
        |start_row, chunk| {
            predict_oblivious_chunk(
                feature_indices,
                threshold_bins,
                leaf_values,
                matrix,
                start_row,
                chunk,
            )
        },
    );
    outputs
}

fn predict_oblivious_probabilities_column_major_matrix(
    feature_indices: &[usize],
    threshold_bins: &[u16],
    leaf_values: &[Vec<f64>],
    matrix: &ColumnMajorBinnedMatrix,
    _executor: &InferenceExecutor,
) -> Vec<Vec<f64>> {
    (0..matrix.n_rows)
        .map(|row_index| {
            predict_oblivious_probabilities_row(
                feature_indices,
                threshold_bins,
                leaf_values,
                |feature_index| matrix.column(feature_index).value_at(row_index),
            )
            .to_vec()
        })
        .collect()
}

fn predict_oblivious_chunk(
    feature_indices: &[usize],
    threshold_bins: &[u16],
    leaf_values: &[f64],
    matrix: &ColumnMajorBinnedMatrix,
    start_row: usize,
    output: &mut [f64],
) {
    let processed = simd_predict_oblivious_chunk(
        feature_indices,
        threshold_bins,
        leaf_values,
        matrix,
        start_row,
        output,
    );

    for (offset, out) in output.iter_mut().enumerate().skip(processed) {
        let row_index = start_row + offset;
        *out = predict_oblivious_row(
            feature_indices,
            threshold_bins,
            leaf_values,
            |feature_index| matrix.column(feature_index).value_at(row_index),
        );
    }
}

fn simd_predict_oblivious_chunk(
    feature_indices: &[usize],
    threshold_bins: &[u16],
    leaf_values: &[f64],
    matrix: &ColumnMajorBinnedMatrix,
    start_row: usize,
    output: &mut [f64],
) -> usize {
    let mut processed = 0usize;
    let ones = u32x8::splat(1);

    while processed + OBLIVIOUS_SIMD_LANES <= output.len() {
        let base_row = start_row + processed;
        let mut leaf_indices = u32x8::splat(0);

        for (&feature_index, &threshold_bin) in feature_indices.iter().zip(threshold_bins) {
            let column = matrix.column(feature_index);
            let bins = if let Some(lanes) = column.slice_u8(base_row, OBLIVIOUS_SIMD_LANES) {
                let lanes: [u8; OBLIVIOUS_SIMD_LANES] = lanes
                    .try_into()
                    .expect("lane width matches the fixed SIMD width");
                u32x8::new([
                    u32::from(lanes[0]),
                    u32::from(lanes[1]),
                    u32::from(lanes[2]),
                    u32::from(lanes[3]),
                    u32::from(lanes[4]),
                    u32::from(lanes[5]),
                    u32::from(lanes[6]),
                    u32::from(lanes[7]),
                ])
            } else {
                let lanes: [u16; OBLIVIOUS_SIMD_LANES] = column
                    .slice_u16(base_row, OBLIVIOUS_SIMD_LANES)
                    .expect("column is u16 when not u8")
                    .try_into()
                    .expect("lane width matches the fixed SIMD width");
                u32x8::from(u16x8::new(lanes))
            };
            let threshold = u32x8::splat(u32::from(threshold_bin));
            let bit = bins.cmp_gt(threshold) & ones;
            leaf_indices = (leaf_indices << 1) | bit;
        }

        let lane_indices = leaf_indices.to_array();
        for lane in 0..OBLIVIOUS_SIMD_LANES {
            output[processed + lane] =
                leaf_values[usize::try_from(lane_indices[lane]).expect("leaf index fits usize")];
        }
        processed += OBLIVIOUS_SIMD_LANES;
    }

    processed
}
