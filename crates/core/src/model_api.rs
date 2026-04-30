use super::*;
use std::collections::BTreeSet;

/// Runtime-lowered model used for faster inference.
///
/// The optimized model keeps a copy of the source [`Model`] so it can preserve
/// serialization and introspection behavior even after the runtime has been
/// lowered into lookup-table-friendly structures.
#[derive(Debug, Clone)]
pub struct OptimizedModel {
    pub(crate) source_model: Model,
    pub(crate) runtime: OptimizedRuntime,
    pub(crate) executor: InferenceExecutor,
    pub(crate) feature_projection: Vec<usize>,
}

impl OptimizedModel {
    pub(crate) fn new(
        source_model: Model,
        physical_cores: Option<usize>,
        missing_features: Option<&[usize]>,
    ) -> Result<Self, OptimizeError> {
        let thread_count = resolve_inference_thread_count(physical_cores)?;
        let feature_projection = build_feature_projection(&source_model);
        let feature_index_map =
            build_feature_index_map(source_model.num_features(), &feature_projection);
        let missing_feature_set =
            missing_features.map(|features| features.iter().copied().collect::<BTreeSet<_>>());
        let runtime = OptimizedRuntime::from_model(
            &source_model,
            &feature_index_map,
            missing_feature_set.as_ref(),
        );
        let executor = InferenceExecutor::new(thread_count)?;

        Ok(Self {
            source_model,
            runtime,
            executor,
            feature_projection,
        })
    }

    pub fn predict_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        let projected = ProjectedTableView::new(table, &self.feature_projection);
        if self.runtime.should_use_batch_matrix(table.n_rows()) {
            let matrix = ColumnMajorBinnedMatrix::from_table_access_projected(
                table,
                &self.feature_projection,
            );
            return self.predict_column_major_binned_matrix(&matrix);
        }

        self.executor.predict_rows(projected.n_rows(), |row_index| {
            self.runtime.predict_table_row(&projected, row_index)
        })
    }

    pub fn predict_rows(&self, rows: Vec<Vec<f64>>) -> Result<Vec<f64>, PredictError> {
        let table = InferenceTable::from_rows_projected(
            rows,
            self.source_model.feature_preprocessing(),
            &self.feature_projection,
        )?;
        if self.runtime.should_use_batch_matrix(table.n_rows()) {
            let matrix = table.to_column_major_binned_matrix();
            Ok(self.predict_column_major_binned_matrix(&matrix))
        } else {
            Ok(self.executor.predict_rows(table.n_rows(), |row_index| {
                self.runtime.predict_table_row(&table, row_index)
            }))
        }
    }

    pub fn predict_named_columns(
        &self,
        columns: BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<f64>, PredictError> {
        let table = InferenceTable::from_named_columns_projected(
            columns,
            self.source_model.feature_preprocessing(),
            &self.feature_projection,
        )?;
        if self.runtime.should_use_batch_matrix(table.n_rows()) {
            let matrix = table.to_column_major_binned_matrix();
            Ok(self.predict_column_major_binned_matrix(&matrix))
        } else {
            Ok(self.executor.predict_rows(table.n_rows(), |row_index| {
                self.runtime.predict_table_row(&table, row_index)
            }))
        }
    }

    pub fn predict_proba_table(
        &self,
        table: &dyn TableAccess,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        let projected = ProjectedTableView::new(table, &self.feature_projection);
        self.runtime.predict_proba_table(&projected, &self.executor)
    }

    pub fn predict_proba_rows(&self, rows: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, PredictError> {
        let table = InferenceTable::from_rows_projected(
            rows,
            self.source_model.feature_preprocessing(),
            &self.feature_projection,
        )?;
        self.runtime.predict_proba_table(&table, &self.executor)
    }

    pub fn predict_proba_named_columns(
        &self,
        columns: BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        let table = InferenceTable::from_named_columns_projected(
            columns,
            self.source_model.feature_preprocessing(),
            &self.feature_projection,
        )?;
        self.runtime.predict_proba_table(&table, &self.executor)
    }

    pub fn predict_proba_sparse_binary_columns(
        &self,
        n_rows: usize,
        n_features: usize,
        columns: Vec<Vec<usize>>,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        let table = InferenceTable::from_sparse_binary_columns_projected(
            n_rows,
            n_features,
            columns,
            self.source_model.feature_preprocessing(),
            &self.feature_projection,
        )?;
        self.runtime.predict_proba_table(&table, &self.executor)
    }

    pub fn predict_sparse_binary_columns(
        &self,
        n_rows: usize,
        n_features: usize,
        columns: Vec<Vec<usize>>,
    ) -> Result<Vec<f64>, PredictError> {
        let table = InferenceTable::from_sparse_binary_columns_projected(
            n_rows,
            n_features,
            columns,
            self.source_model.feature_preprocessing(),
            &self.feature_projection,
        )?;
        if self.runtime.should_use_batch_matrix(table.n_rows()) {
            let matrix = table.to_column_major_binned_matrix();
            Ok(self.predict_column_major_binned_matrix(&matrix))
        } else {
            Ok(self.executor.predict_rows(table.n_rows(), |row_index| {
                self.runtime.predict_table_row(&table, row_index)
            }))
        }
    }

    #[cfg(feature = "polars")]
    pub fn predict_polars_dataframe(&self, df: &DataFrame) -> Result<Vec<f64>, PredictError> {
        let columns = polars_named_columns(df)?;
        self.predict_named_columns(columns)
    }

    #[cfg(feature = "polars")]
    pub fn predict_polars_lazyframe(&self, lf: &LazyFrame) -> Result<Vec<f64>, PredictError> {
        let mut predictions = Vec::new();
        let mut offset = 0i64;
        loop {
            let batch = lf
                .clone()
                .slice(offset, LAZYFRAME_PREDICT_BATCH_ROWS as IdxSize)
                .collect()?;
            let height = batch.height();
            if height == 0 {
                break;
            }
            predictions.extend(self.predict_polars_dataframe(&batch)?);
            if height < LAZYFRAME_PREDICT_BATCH_ROWS {
                break;
            }
            offset += height as i64;
        }
        Ok(predictions)
    }

    pub fn algorithm(&self) -> TrainAlgorithm {
        self.source_model.algorithm()
    }

    pub fn task(&self) -> Task {
        self.source_model.task()
    }

    pub fn criterion(&self) -> Criterion {
        self.source_model.criterion()
    }

    pub fn tree_type(&self) -> TreeType {
        self.source_model.tree_type()
    }

    pub fn mean_value(&self) -> Option<f64> {
        self.source_model.mean_value()
    }

    pub fn canaries(&self) -> usize {
        self.source_model.canaries()
    }

    pub fn max_depth(&self) -> Option<usize> {
        self.source_model.max_depth()
    }

    pub fn min_samples_split(&self) -> Option<usize> {
        self.source_model.min_samples_split()
    }

    pub fn min_samples_leaf(&self) -> Option<usize> {
        self.source_model.min_samples_leaf()
    }

    pub fn n_trees(&self) -> Option<usize> {
        self.source_model.n_trees()
    }

    pub fn max_features(&self) -> Option<usize> {
        self.source_model.max_features()
    }

    pub fn seed(&self) -> Option<u64> {
        self.source_model.seed()
    }

    pub fn compute_oob(&self) -> bool {
        self.source_model.compute_oob()
    }

    pub fn oob_score(&self) -> Option<f64> {
        self.source_model.oob_score()
    }

    pub fn learning_rate(&self) -> Option<f64> {
        self.source_model.learning_rate()
    }

    pub fn bootstrap(&self) -> bool {
        self.source_model.bootstrap()
    }

    pub fn top_gradient_fraction(&self) -> Option<f64> {
        self.source_model.top_gradient_fraction()
    }

    pub fn other_gradient_fraction(&self) -> Option<f64> {
        self.source_model.other_gradient_fraction()
    }

    pub fn tree_count(&self) -> usize {
        self.source_model.tree_count()
    }

    pub fn used_feature_indices(&self) -> Vec<usize> {
        self.feature_projection.clone()
    }

    pub fn used_feature_count(&self) -> usize {
        self.feature_projection.len()
    }

    pub fn feature_importances(&self) -> Vec<f64> {
        self.source_model.feature_importances()
    }

    pub fn tree_structure(
        &self,
        tree_index: usize,
    ) -> Result<TreeStructureSummary, IntrospectionError> {
        self.source_model.tree_structure(tree_index)
    }

    pub fn tree_prediction_stats(
        &self,
        tree_index: usize,
    ) -> Result<PredictionValueStats, IntrospectionError> {
        self.source_model.tree_prediction_stats(tree_index)
    }

    pub fn tree_node(
        &self,
        tree_index: usize,
        node_index: usize,
    ) -> Result<ir::NodeTreeNode, IntrospectionError> {
        self.source_model.tree_node(tree_index, node_index)
    }

    pub fn tree_level(
        &self,
        tree_index: usize,
        level_index: usize,
    ) -> Result<ir::ObliviousLevel, IntrospectionError> {
        self.source_model.tree_level(tree_index, level_index)
    }

    pub fn tree_leaf(
        &self,
        tree_index: usize,
        leaf_index: usize,
    ) -> Result<ir::IndexedLeaf, IntrospectionError> {
        self.source_model.tree_leaf(tree_index, leaf_index)
    }

    pub fn to_ir(&self) -> ModelPackageIr {
        self.source_model.to_ir()
    }

    pub fn to_ir_json(&self) -> Result<String, serde_json::Error> {
        self.source_model.to_ir_json()
    }

    pub fn to_ir_json_pretty(&self) -> Result<String, serde_json::Error> {
        self.source_model.to_ir_json_pretty()
    }

    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.source_model.serialize()
    }

    pub fn serialize_pretty(&self) -> Result<String, serde_json::Error> {
        self.source_model.serialize_pretty()
    }

    pub(crate) fn predict_column_major_binned_matrix(
        &self,
        matrix: &ColumnMajorBinnedMatrix,
    ) -> Vec<f64> {
        self.runtime
            .predict_column_major_matrix(matrix, &self.executor)
    }
}

impl Model {
    pub fn predict_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        match self {
            Model::DecisionTreeClassifier(model) => model.predict_table(table),
            Model::DecisionTreeRegressor(model) => model.predict_table(table),
            Model::RandomForest(model) => model.predict_table(table),
            Model::GradientBoostedTrees(model) => model.predict_table(table),
        }
    }

    pub fn predict_rows(&self, rows: Vec<Vec<f64>>) -> Result<Vec<f64>, PredictError> {
        let table = InferenceTable::from_rows(rows, self.feature_preprocessing())?;
        Ok(self.predict_table(&table))
    }

    pub fn predict_all_rows(&self, rows: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, PredictError> {
        let table = InferenceTable::from_rows(rows, self.feature_preprocessing())?;
        Ok(self.predict_all_table(&table))
    }

    pub fn predict_proba_table(
        &self,
        table: &dyn TableAccess,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        match self {
            Model::DecisionTreeClassifier(model) => Ok(model.predict_proba_table(table)),
            Model::RandomForest(model) => model.predict_proba_table(table),
            Model::GradientBoostedTrees(model) => model.predict_proba_table(table),
            Model::DecisionTreeRegressor(_) => {
                Err(PredictError::ProbabilityPredictionRequiresClassification)
            }
        }
    }

    pub fn predict_proba_rows(&self, rows: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, PredictError> {
        let table = InferenceTable::from_rows(rows, self.feature_preprocessing())?;
        self.predict_proba_table(&table)
    }

    pub fn predict_named_columns(
        &self,
        columns: BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<f64>, PredictError> {
        let table = InferenceTable::from_named_columns(columns, self.feature_preprocessing())?;
        Ok(self.predict_table(&table))
    }

    pub fn predict_proba_named_columns(
        &self,
        columns: BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        let table = InferenceTable::from_named_columns(columns, self.feature_preprocessing())?;
        self.predict_proba_table(&table)
    }

    pub fn predict_sparse_binary_columns(
        &self,
        n_rows: usize,
        n_features: usize,
        columns: Vec<Vec<usize>>,
    ) -> Result<Vec<f64>, PredictError> {
        let table = InferenceTable::from_sparse_binary_columns(
            n_rows,
            n_features,
            columns,
            self.feature_preprocessing(),
        )?;
        Ok(self.predict_table(&table))
    }

    pub fn predict_proba_sparse_binary_columns(
        &self,
        n_rows: usize,
        n_features: usize,
        columns: Vec<Vec<usize>>,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        let table = InferenceTable::from_sparse_binary_columns(
            n_rows,
            n_features,
            columns,
            self.feature_preprocessing(),
        )?;
        self.predict_proba_table(&table)
    }

    #[cfg(feature = "polars")]
    pub fn predict_polars_dataframe(&self, df: &DataFrame) -> Result<Vec<f64>, PredictError> {
        let columns = polars_named_columns(df)?;
        self.predict_named_columns(columns)
    }

    #[cfg(feature = "polars")]
    pub fn predict_polars_lazyframe(&self, lf: &LazyFrame) -> Result<Vec<f64>, PredictError> {
        let mut predictions = Vec::new();
        let mut offset = 0i64;
        loop {
            let batch = lf
                .clone()
                .slice(offset, LAZYFRAME_PREDICT_BATCH_ROWS as IdxSize)
                .collect()?;
            let height = batch.height();
            if height == 0 {
                break;
            }
            predictions.extend(self.predict_polars_dataframe(&batch)?);
            if height < LAZYFRAME_PREDICT_BATCH_ROWS {
                break;
            }
            offset += height as i64;
        }
        Ok(predictions)
    }

    pub fn algorithm(&self) -> TrainAlgorithm {
        match self {
            Model::DecisionTreeClassifier(_) | Model::DecisionTreeRegressor(_) => {
                TrainAlgorithm::Dt
            }
            Model::RandomForest(_) => TrainAlgorithm::Rf,
            Model::GradientBoostedTrees(_) => TrainAlgorithm::Gbm,
        }
    }

    pub fn task(&self) -> Task {
        match self {
            Model::DecisionTreeRegressor(_) => Task::Regression,
            Model::DecisionTreeClassifier(_) => Task::Classification,
            Model::RandomForest(model) => model.task(),
            Model::GradientBoostedTrees(model) => model.task(),
        }
    }

    pub fn is_multi_target(&self) -> bool {
        match self {
            Model::DecisionTreeRegressor(model) => model.is_multi_target(),
            _ => false,
        }
    }

    pub fn predict_all_table(&self, table: &dyn TableAccess) -> Vec<Vec<f64>> {
        match self {
            Model::DecisionTreeRegressor(model) if model.is_multi_target() => {
                model.predict_all_table(table)
            }
            _ => self
                .predict_table(table)
                .into_iter()
                .map(|v| vec![v])
                .collect(),
        }
    }

    pub fn criterion(&self) -> Criterion {
        match self {
            Model::DecisionTreeClassifier(model) => model.criterion(),
            Model::DecisionTreeRegressor(model) => model.criterion(),
            Model::RandomForest(model) => model.criterion(),
            Model::GradientBoostedTrees(model) => model.criterion(),
        }
    }

    pub fn tree_type(&self) -> TreeType {
        match self {
            Model::DecisionTreeClassifier(model) => match model.algorithm() {
                DecisionTreeAlgorithm::Id3 => TreeType::Id3,
                DecisionTreeAlgorithm::C45 => TreeType::C45,
                DecisionTreeAlgorithm::Cart => TreeType::Cart,
                DecisionTreeAlgorithm::Randomized => TreeType::Randomized,
                DecisionTreeAlgorithm::Oblivious => TreeType::Oblivious,
            },
            Model::DecisionTreeRegressor(model) => match model.algorithm() {
                RegressionTreeAlgorithm::Cart => TreeType::Cart,
                RegressionTreeAlgorithm::Randomized => TreeType::Randomized,
                RegressionTreeAlgorithm::Oblivious => TreeType::Oblivious,
            },
            Model::RandomForest(model) => model.tree_type(),
            Model::GradientBoostedTrees(model) => model.tree_type(),
        }
    }

    pub fn mean_value(&self) -> Option<f64> {
        match self {
            Model::DecisionTreeClassifier(_)
            | Model::DecisionTreeRegressor(_)
            | Model::RandomForest(_)
            | Model::GradientBoostedTrees(_) => None,
        }
    }

    pub fn canaries(&self) -> usize {
        self.training_metadata().canaries
    }

    pub fn max_depth(&self) -> Option<usize> {
        self.training_metadata().max_depth
    }

    pub fn min_samples_split(&self) -> Option<usize> {
        self.training_metadata().min_samples_split
    }

    pub fn min_samples_leaf(&self) -> Option<usize> {
        self.training_metadata().min_samples_leaf
    }

    pub fn n_trees(&self) -> Option<usize> {
        self.training_metadata().n_trees
    }

    pub fn max_features(&self) -> Option<usize> {
        self.training_metadata().max_features
    }

    pub fn seed(&self) -> Option<u64> {
        self.training_metadata().seed
    }

    pub fn compute_oob(&self) -> bool {
        self.training_metadata().compute_oob
    }

    pub fn oob_score(&self) -> Option<f64> {
        self.training_metadata().oob_score
    }

    pub fn learning_rate(&self) -> Option<f64> {
        self.training_metadata().learning_rate
    }

    pub fn bootstrap(&self) -> bool {
        self.training_metadata().bootstrap.unwrap_or(false)
    }

    pub fn top_gradient_fraction(&self) -> Option<f64> {
        self.training_metadata().top_gradient_fraction
    }

    pub fn other_gradient_fraction(&self) -> Option<f64> {
        self.training_metadata().other_gradient_fraction
    }

    pub fn tree_count(&self) -> usize {
        self.to_ir().model.trees.len()
    }

    pub fn tree_structure(
        &self,
        tree_index: usize,
    ) -> Result<TreeStructureSummary, IntrospectionError> {
        tree_structure_summary(self.tree_definition(tree_index)?)
    }

    pub fn tree_prediction_stats(
        &self,
        tree_index: usize,
    ) -> Result<PredictionValueStats, IntrospectionError> {
        prediction_value_stats(self.tree_definition(tree_index)?)
    }

    pub fn tree_node(
        &self,
        tree_index: usize,
        node_index: usize,
    ) -> Result<ir::NodeTreeNode, IntrospectionError> {
        match self.tree_definition(tree_index)? {
            ir::TreeDefinition::NodeTree { nodes, .. } => {
                let available = nodes.len();
                nodes
                    .into_iter()
                    .nth(node_index)
                    .ok_or(IntrospectionError::NodeIndexOutOfBounds {
                        requested: node_index,
                        available,
                    })
            }
            ir::TreeDefinition::ObliviousLevels { .. } => Err(IntrospectionError::NotANodeTree),
        }
    }

    pub fn tree_level(
        &self,
        tree_index: usize,
        level_index: usize,
    ) -> Result<ir::ObliviousLevel, IntrospectionError> {
        match self.tree_definition(tree_index)? {
            ir::TreeDefinition::ObliviousLevels { levels, .. } => {
                let available = levels.len();
                levels.into_iter().nth(level_index).ok_or(
                    IntrospectionError::LevelIndexOutOfBounds {
                        requested: level_index,
                        available,
                    },
                )
            }
            ir::TreeDefinition::NodeTree { .. } => Err(IntrospectionError::NotAnObliviousTree),
        }
    }

    pub fn tree_leaf(
        &self,
        tree_index: usize,
        leaf_index: usize,
    ) -> Result<ir::IndexedLeaf, IntrospectionError> {
        match self.tree_definition(tree_index)? {
            ir::TreeDefinition::ObliviousLevels { leaves, .. } => {
                let available = leaves.len();
                leaves
                    .into_iter()
                    .nth(leaf_index)
                    .ok_or(IntrospectionError::LeafIndexOutOfBounds {
                        requested: leaf_index,
                        available,
                    })
            }
            ir::TreeDefinition::NodeTree { nodes, .. } => {
                let leaves = nodes
                    .into_iter()
                    .filter_map(|node| match node {
                        ir::NodeTreeNode::Leaf {
                            node_id,
                            leaf,
                            stats,
                            ..
                        } => Some(ir::IndexedLeaf {
                            leaf_index: node_id,
                            leaf,
                            stats: ir::NodeStats {
                                sample_count: stats.sample_count,
                                impurity: stats.impurity,
                                gain: stats.gain,
                                class_counts: stats.class_counts,
                                variance: stats.variance,
                            },
                        }),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let available = leaves.len();
                leaves
                    .into_iter()
                    .nth(leaf_index)
                    .ok_or(IntrospectionError::LeafIndexOutOfBounds {
                        requested: leaf_index,
                        available,
                    })
            }
        }
    }

    pub fn to_ir(&self) -> ModelPackageIr {
        ir::model_to_ir(self)
    }

    pub fn to_ir_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.to_ir())
    }

    pub fn to_ir_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.to_ir())
    }

    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.to_ir_json()
    }

    pub fn serialize_pretty(&self) -> Result<String, serde_json::Error> {
        self.to_ir_json_pretty()
    }

    pub fn optimize_inference(
        &self,
        physical_cores: Option<usize>,
    ) -> Result<OptimizedModel, OptimizeError> {
        OptimizedModel::new(self.clone(), physical_cores, None)
    }

    pub fn optimize_inference_with_missing_features(
        &self,
        physical_cores: Option<usize>,
        missing_features: Option<Vec<usize>>,
    ) -> Result<OptimizedModel, OptimizeError> {
        OptimizedModel::new(self.clone(), physical_cores, missing_features.as_deref())
    }

    pub fn json_schema() -> schemars::schema::RootSchema {
        ModelPackageIr::json_schema()
    }

    pub fn json_schema_json() -> Result<String, IrError> {
        ModelPackageIr::json_schema_json()
    }

    pub fn json_schema_json_pretty() -> Result<String, IrError> {
        ModelPackageIr::json_schema_json_pretty()
    }

    pub fn deserialize(serialized: &str) -> Result<Self, IrError> {
        let ir: ModelPackageIr =
            serde_json::from_str(serialized).map_err(|err| IrError::Json(err.to_string()))?;
        ir::model_from_ir(ir)
    }

    pub(crate) fn num_features(&self) -> usize {
        match self {
            Model::DecisionTreeClassifier(model) => model.num_features(),
            Model::DecisionTreeRegressor(model) => model.num_features(),
            Model::RandomForest(model) => model.num_features(),
            Model::GradientBoostedTrees(model) => model.num_features(),
        }
    }

    pub(crate) fn feature_preprocessing(&self) -> &[FeaturePreprocessing] {
        match self {
            Model::DecisionTreeClassifier(model) => model.feature_preprocessing(),
            Model::DecisionTreeRegressor(model) => model.feature_preprocessing(),
            Model::RandomForest(model) => model.feature_preprocessing(),
            Model::GradientBoostedTrees(model) => model.feature_preprocessing(),
        }
    }

    pub fn used_feature_indices(&self) -> Vec<usize> {
        model_used_feature_indices(self)
    }

    pub fn used_feature_count(&self) -> usize {
        self.used_feature_indices().len()
    }

    /// Mean decrease impurity feature importances, one value per input feature.
    ///
    /// Values are non-negative and sum to 1.0. Features that never appear in
    /// any split have importance 0.0. For ensembles the per-tree normalized
    /// vectors are averaged.
    pub fn feature_importances(&self) -> Vec<f64> {
        let ir = self.to_ir();
        feature_importances(&ir.model.trees, self.num_features())
    }

    pub(crate) fn class_labels(&self) -> Option<Vec<f64>> {
        match self {
            Model::DecisionTreeClassifier(model) => Some(model.class_labels().to_vec()),
            Model::RandomForest(model) => model.class_labels(),
            Model::GradientBoostedTrees(model) => model.class_labels(),
            Model::DecisionTreeRegressor(_) => None,
        }
    }

    pub(crate) fn training_metadata(&self) -> ir::TrainingMetadata {
        match self {
            Model::DecisionTreeClassifier(model) => model.training_metadata(),
            Model::DecisionTreeRegressor(model) => model.training_metadata(),
            Model::RandomForest(model) => model.training_metadata(),
            Model::GradientBoostedTrees(model) => model.training_metadata(),
        }
    }

    fn tree_definition(&self, tree_index: usize) -> Result<ir::TreeDefinition, IntrospectionError> {
        let trees = self.to_ir().model.trees;
        let available = trees.len();
        trees
            .into_iter()
            .nth(tree_index)
            .ok_or(IntrospectionError::TreeIndexOutOfBounds {
                requested: tree_index,
                available,
            })
    }
}
