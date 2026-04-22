use super::*;
use crate::compiled_artifact::COMPILED_ARTIFACT_MAGIC;
use forestfire_data::DenseTable;
#[cfg(feature = "polars")]
use polars::prelude::{DataFrame, IntoLazy, NamedFrom, Series};
use std::collections::BTreeMap;

const PREDICTION_TOLERANCE: f64 = 10e-6;

fn assert_predictions_close(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (idx, (lhs, rhs)) in left.iter().zip(right.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() <= PREDICTION_TOLERANCE,
            "prediction mismatch at index {}: left={} right={} tolerance={}",
            idx,
            lhs,
            rhs,
            PREDICTION_TOLERANCE
        );
    }
}

fn oblique_classification_table() -> DenseTable {
    DenseTable::with_options(
        vec![
            vec![-2.0, 1.0],
            vec![1.0, -2.0],
            vec![-1.0, 2.0],
            vec![2.0, -1.0],
            vec![-3.0, 1.0],
            vec![1.0, -3.0],
            vec![-1.0, 3.0],
            vec![3.0, -1.0],
        ],
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        0,
        NumericBins::Fixed(64),
    )
    .unwrap()
}

#[test]
fn unified_train_dispatches_regression_cart() {
    let table = DenseTable::new(
        vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ],
        vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
    )
    .unwrap();

    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();

    assert!(matches!(model, Model::DecisionTreeRegressor(_)));
    assert_eq!(model.task(), Task::Regression);
    assert_eq!(model.tree_type(), TreeType::Cart);
    assert_eq!(model.criterion(), Criterion::Mean);
}

#[test]
fn unified_train_dispatches_randomized_for_both_tasks() {
    let regression_table = DenseTable::with_options(
        vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
        vec![0.0, 1.0, 4.0, 9.0],
        0,
        NumericBins::Fixed(64),
    )
    .unwrap();
    let classification_table = DenseTable::new(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
    )
    .unwrap();

    let regression_model = train(
        &regression_table,
        TrainConfig {
            task: Task::Regression,
            tree_type: TreeType::Randomized,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            ..TrainConfig::default()
        },
    )
    .unwrap();
    let classification_model = train(
        &classification_table,
        TrainConfig {
            task: Task::Classification,
            tree_type: TreeType::Randomized,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            ..TrainConfig::default()
        },
    )
    .unwrap();

    assert!(matches!(regression_model, Model::DecisionTreeRegressor(_)));
    assert_eq!(regression_model.tree_type(), TreeType::Randomized);
    assert!(matches!(
        classification_model,
        Model::DecisionTreeClassifier(_)
    ));
    assert_eq!(classification_model.tree_type(), TreeType::Randomized);
}

#[test]
fn unified_train_rejects_unsupported_task_tree_pair() {
    let table = DenseTable::new(vec![vec![0.0], vec![1.0]], vec![0.0, 1.0]).unwrap();

    let err = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Id3,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap_err();

    assert!(matches!(
        err,
        TrainError::UnsupportedConfiguration {
            task: Task::Regression,
            tree_type: TreeType::Id3,
            criterion: Criterion::Mean,
        }
    ));
}

#[test]
fn unified_train_accepts_oblique_strategy_for_gbm_cart() {
    let table = oblique_classification_table();

    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Gbm,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::Oblique,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Auto,
            n_trees: Some(4),
            max_depth: Some(1),
            physical_cores: Some(1),
            ..TrainConfig::default()
        },
    )
    .unwrap();

    assert!(matches!(model, Model::GradientBoostedTrees(_)));
}

#[test]
fn oblique_models_round_trip_through_ir_and_optimized_runtime() {
    let table = oblique_classification_table();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::Oblique,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: Some(1),
            max_features: MaxFeatures::Count(2),
            physical_cores: Some(1),
            ..TrainConfig::default()
        },
    )
    .unwrap();

    let serialized = model.serialize().unwrap();
    let restored = Model::deserialize(&serialized).unwrap();

    assert_eq!(model.predict_table(&table), restored.predict_table(&table));
    let optimized = model.optimize_inference(Some(1)).unwrap();
    assert_eq!(model.predict_table(&table), optimized.predict_table(&table));

    let compiled = optimized.serialize_compiled().unwrap();
    let restored_optimized = OptimizedModel::deserialize_compiled(&compiled, Some(1)).unwrap();
    assert_eq!(
        optimized.predict_table(&table),
        restored_optimized.predict_table(&table)
    );
}

#[test]
fn unified_train_resolves_auto_criterion_across_supported_matrix() {
    let classification_table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let regression_table = DenseTable::with_canaries(
        vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
        vec![1.0, 3.0, 5.0, 7.0],
        0,
    )
    .unwrap();

    for (table, config, expected_criterion) in [
        (
            &regression_table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Regression,
                tree_type: TreeType::Cart,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Greedy,
                lookahead_depth: 1,
                lookahead_top_k: 8,
                lookahead_weight: 1.0,
                criterion: Criterion::Auto,
                max_depth: None,

                min_samples_split: None,
                min_samples_leaf: None,

                physical_cores: Some(1),
                n_trees: None,
                max_features: MaxFeatures::Auto,
                seed: None,
                canary_filter: CanaryFilter::default(),
                compute_oob: false,
                learning_rate: None,
                bootstrap: false,
                top_gradient_fraction: None,
                other_gradient_fraction: None,
                missing_value_strategy: MissingValueStrategyConfig::heuristic(),
                histogram_bins: None,
            },
            Criterion::Mean,
        ),
        (
            &regression_table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Regression,
                tree_type: TreeType::Randomized,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Greedy,
                lookahead_depth: 1,
                lookahead_top_k: 8,
                lookahead_weight: 1.0,
                criterion: Criterion::Auto,
                max_depth: None,

                min_samples_split: None,
                min_samples_leaf: None,

                physical_cores: Some(1),
                n_trees: None,
                max_features: MaxFeatures::Auto,
                seed: None,
                canary_filter: CanaryFilter::default(),
                compute_oob: false,
                learning_rate: None,
                bootstrap: false,
                top_gradient_fraction: None,
                other_gradient_fraction: None,
                missing_value_strategy: MissingValueStrategyConfig::heuristic(),
                histogram_bins: None,
            },
            Criterion::Mean,
        ),
        (
            &regression_table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Regression,
                tree_type: TreeType::Oblivious,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Greedy,
                lookahead_depth: 1,
                lookahead_top_k: 8,
                lookahead_weight: 1.0,
                criterion: Criterion::Auto,
                max_depth: None,

                min_samples_split: None,
                min_samples_leaf: None,

                physical_cores: Some(1),
                n_trees: None,
                max_features: MaxFeatures::Auto,
                seed: None,
                canary_filter: CanaryFilter::default(),
                compute_oob: false,
                learning_rate: None,
                bootstrap: false,
                top_gradient_fraction: None,
                other_gradient_fraction: None,
                missing_value_strategy: MissingValueStrategyConfig::heuristic(),
                histogram_bins: None,
            },
            Criterion::Mean,
        ),
        (
            &classification_table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Id3,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Greedy,
                lookahead_depth: 1,
                lookahead_top_k: 8,
                lookahead_weight: 1.0,
                criterion: Criterion::Auto,
                max_depth: None,

                min_samples_split: None,
                min_samples_leaf: None,

                physical_cores: Some(1),
                n_trees: None,
                max_features: MaxFeatures::Auto,
                seed: None,
                canary_filter: CanaryFilter::default(),
                compute_oob: false,
                learning_rate: None,
                bootstrap: false,
                top_gradient_fraction: None,
                other_gradient_fraction: None,
                missing_value_strategy: MissingValueStrategyConfig::heuristic(),
                histogram_bins: None,
            },
            Criterion::Entropy,
        ),
        (
            &classification_table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::C45,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Greedy,
                lookahead_depth: 1,
                lookahead_top_k: 8,
                lookahead_weight: 1.0,
                criterion: Criterion::Auto,
                max_depth: None,

                min_samples_split: None,
                min_samples_leaf: None,

                physical_cores: Some(1),
                n_trees: None,
                max_features: MaxFeatures::Auto,
                seed: None,
                canary_filter: CanaryFilter::default(),
                compute_oob: false,
                learning_rate: None,
                bootstrap: false,
                top_gradient_fraction: None,
                other_gradient_fraction: None,
                missing_value_strategy: MissingValueStrategyConfig::heuristic(),
                histogram_bins: None,
            },
            Criterion::Entropy,
        ),
        (
            &classification_table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Cart,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Greedy,
                lookahead_depth: 1,
                lookahead_top_k: 8,
                lookahead_weight: 1.0,
                criterion: Criterion::Auto,
                max_depth: None,

                min_samples_split: None,
                min_samples_leaf: None,

                physical_cores: Some(1),
                n_trees: None,
                max_features: MaxFeatures::Auto,
                seed: None,
                canary_filter: CanaryFilter::default(),
                compute_oob: false,
                learning_rate: None,
                bootstrap: false,
                top_gradient_fraction: None,
                other_gradient_fraction: None,
                missing_value_strategy: MissingValueStrategyConfig::heuristic(),
                histogram_bins: None,
            },
            Criterion::Gini,
        ),
        (
            &classification_table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Randomized,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Greedy,
                lookahead_depth: 1,
                lookahead_top_k: 8,
                lookahead_weight: 1.0,
                criterion: Criterion::Auto,
                max_depth: None,

                min_samples_split: None,
                min_samples_leaf: None,

                physical_cores: Some(1),
                n_trees: None,
                max_features: MaxFeatures::Auto,
                seed: None,
                canary_filter: CanaryFilter::default(),
                compute_oob: false,
                learning_rate: None,
                bootstrap: false,
                top_gradient_fraction: None,
                other_gradient_fraction: None,
                missing_value_strategy: MissingValueStrategyConfig::heuristic(),
                histogram_bins: None,
            },
            Criterion::Gini,
        ),
        (
            &classification_table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Oblivious,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Greedy,
                lookahead_depth: 1,
                lookahead_top_k: 8,
                lookahead_weight: 1.0,
                criterion: Criterion::Auto,
                max_depth: None,

                min_samples_split: None,
                min_samples_leaf: None,

                physical_cores: Some(1),
                n_trees: None,
                max_features: MaxFeatures::Auto,
                seed: None,
                canary_filter: CanaryFilter::default(),
                compute_oob: false,
                learning_rate: None,
                bootstrap: false,
                top_gradient_fraction: None,
                other_gradient_fraction: None,
                missing_value_strategy: MissingValueStrategyConfig::heuristic(),
                histogram_bins: None,
            },
            Criterion::Gini,
        ),
    ] {
        let model = train(table, config).unwrap();
        assert_eq!(model.criterion(), expected_criterion);
    }
}

#[test]
fn unified_train_parallel_matches_single_core_across_supported_tree_types() {
    let classification_table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let regression_table = DenseTable::with_canaries(
        vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ],
        vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
        0,
    )
    .unwrap();

    for config in [
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Randomized,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Oblivious,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    ] {
        let single_core = train(&regression_table, config.clone()).unwrap();
        let parallel = train(
            &regression_table,
            TrainConfig {
                max_depth: None,

                min_samples_split: None,
                min_samples_leaf: None,
                physical_cores: Some(2),
                ..config
            },
        )
        .unwrap();

        assert_eq!(
            single_core.predict_table(&regression_table),
            parallel.predict_table(&regression_table)
        );
    }

    for config in [
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Id3,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Entropy,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::C45,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Entropy,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Randomized,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Oblivious,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    ] {
        let single_core = train(&classification_table, config.clone()).unwrap();
        let parallel = train(
            &classification_table,
            TrainConfig {
                max_depth: None,

                min_samples_split: None,
                min_samples_leaf: None,
                physical_cores: Some(2),
                ..config
            },
        )
        .unwrap();

        assert_eq!(
            single_core.predict_table(&classification_table),
            parallel.predict_table(&classification_table)
        );
    }
}

#[test]
fn unified_train_rejects_zero_physical_cores() {
    let table = DenseTable::new(vec![vec![0.0], vec![1.0]], vec![0.0, 1.0]).unwrap();

    let err = train(
        &table,
        TrainConfig {
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,
            physical_cores: Some(0),
            ..TrainConfig::default()
        },
    )
    .unwrap_err();

    assert!(matches!(
        err,
        TrainError::InvalidPhysicalCoreCount { requested: 0, .. }
    ));
}

#[test]
fn unified_train_caps_physical_cores_to_available_hardware() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();

    let single_core = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();
    let overprovisioned = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(usize::MAX),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();

    assert_eq!(
        single_core.predict_table(&table),
        overprovisioned.predict_table(&table)
    );
}

#[test]
fn ir_exports_regression_tree_with_training_binning() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![1.0, 10.0],
            vec![0.0, 20.0],
            vec![1.0, 30.0],
        ],
        vec![1.0, 3.0, 5.0, 7.0],
        2,
    )
    .unwrap();

    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();

    let ir = model.to_ir();

    assert_eq!(ir.ir_version, "1.0.0");
    assert_eq!(ir.model.algorithm, "dt");
    assert_eq!(ir.model.tree_type, "cart");
    assert_eq!(ir.input_schema.feature_count, 2);
    assert_eq!(ir.training_metadata.canaries, 2);
    assert!(matches!(
        &ir.preprocessing.numeric_binning.features[0],
        ir::FeatureBinning::Binary { feature_index: 0 }
    ));
    assert!(matches!(
        &ir.preprocessing.numeric_binning.features[1],
        ir::FeatureBinning::Numeric {
            feature_index: 1,
            ..
        }
    ));
    let ir::TreeDefinition::NodeTree { nodes, .. } = &ir.model.trees[0] else {
        panic!("target mean should export as a node tree");
    };
    assert!(nodes.iter().any(|node| matches!(
        node,
        ir::NodeTreeNode::Leaf {
            leaf: ir::LeafPayload::RegressionValue { value },
            ..
        } if value.is_finite()
    )));
}

#[test]
fn ir_exports_classifier_with_multiway_postprocessing() {
    let table = DenseTable::with_canaries(
        vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
        vec![2.0, 4.0, 6.0, 8.0],
        0,
    )
    .unwrap();

    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Id3,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Entropy,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();

    let ir = model.to_ir();

    assert_eq!(ir.model.representation, "node_tree");
    assert_eq!(ir.output_schema.class_order, Some(vec![2.0, 4.0, 6.0, 8.0]));
    assert!(matches!(
        &ir.postprocessing.steps[0],
        ir::PostprocessingStep::MapClassIndexToLabel { labels }
            if labels == &vec![2.0, 4.0, 6.0, 8.0]
    ));
    let ir::TreeDefinition::NodeTree { nodes, .. } = &ir.model.trees[0] else {
        panic!("id3 should export as a node tree");
    };
    assert!(
        nodes
            .iter()
            .any(|node| matches!(node, ir::NodeTreeNode::MultiwayBranch { .. }))
    );
}

#[test]
fn ir_exports_oblivious_regressor_with_msb_leaf_indexing() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 1.0, 1.0, 2.0],
        0,
    )
    .unwrap();

    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Oblivious,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();

    let ir = model.to_ir();

    let ir::TreeDefinition::ObliviousLevels {
        depth,
        leaf_indexing,
        leaves,
        ..
    } = &ir.model.trees[0]
    else {
        panic!("oblivious regressor should export as oblivious_levels");
    };

    assert_eq!(*depth, 2);
    assert_eq!(leaf_indexing.bit_order, "msb_first");
    assert_eq!(leaves.len(), 4);

    let json = model.to_ir_json().unwrap();
    let parsed: ModelPackageIr = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.model.tree_type, "oblivious");
}

#[test]
fn serialized_model_round_trips_through_deserialize() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        2,
    )
    .unwrap();

    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();

    let serialized = model.serialize().unwrap();
    let restored = Model::deserialize(&serialized).unwrap();

    assert_eq!(model.algorithm(), restored.algorithm());
    assert_eq!(model.task(), restored.task());
    assert_eq!(model.tree_type(), restored.tree_type());
    assert_eq!(model.criterion(), restored.criterion());
    assert_eq!(model.predict_table(&table), restored.predict_table(&table));
}

#[test]
fn optimized_model_matches_base_model_and_ir_for_standard_classifier() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0, 10.0, 20.0],
            vec![0.0, 1.0, 10.0, 20.0],
            vec![1.0, 0.0, 10.0, 20.0],
            vec![1.0, 1.0, 10.0, 20.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();
    let optimized = model.optimize_inference(Some(1)).unwrap();

    assert_eq!(model.to_ir_json().unwrap(), optimized.to_ir_json().unwrap());
    assert_eq!(model.serialize().unwrap(), optimized.serialize().unwrap());
    assert_eq!(model.used_feature_indices(), vec![0, 1]);
    assert_eq!(optimized.used_feature_indices(), vec![0, 1]);
    assert_eq!(optimized.used_feature_count(), 2);
    assert_predictions_close(
        &model.predict_table(&table),
        &optimized.predict_table(&table),
    );
    let model_preds = model
        .predict_rows(vec![vec![0.0, 1.0, 10.0, 20.0], vec![1.0, 1.0, 10.0, 20.0]])
        .unwrap();
    let optimized_preds = optimized
        .predict_rows(vec![vec![0.0, 1.0, 10.0, 20.0], vec![1.0, 1.0, 10.0, 20.0]])
        .unwrap();
    assert_predictions_close(model_preds.as_slice(), optimized_preds.as_slice());
}

#[test]
fn optimized_model_matches_base_model_and_ir_for_oblivious_regressor() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 1.0, 1.0, 2.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Oblivious,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();
    let optimized = model.optimize_inference(Some(2)).unwrap();

    assert_eq!(model.to_ir_json().unwrap(), optimized.to_ir_json().unwrap());
    assert_predictions_close(
        &model.predict_table(&table),
        &optimized.predict_table(&table),
    );
    let model_preds = model
        .predict_named_columns(BTreeMap::from([
            ("f0".to_string(), vec![0.0, 1.0]),
            ("f1".to_string(), vec![1.0, 1.0]),
        ]))
        .unwrap();
    let optimized_preds = optimized
        .predict_named_columns(BTreeMap::from([
            ("f0".to_string(), vec![0.0, 1.0]),
            ("f1".to_string(), vec![1.0, 1.0]),
        ]))
        .unwrap();
    assert_predictions_close(model_preds.as_slice(), optimized_preds.as_slice());
}

#[test]
fn optimized_oblivious_model_matches_base_on_large_batch() {
    let rows = (0..32)
        .map(|idx| vec![f64::from((idx % 2) as u8), f64::from(((idx / 2) % 2) as u8)])
        .collect::<Vec<_>>();
    let targets = rows.iter().map(|row| row[0] + row[1]).collect::<Vec<_>>();
    let table = DenseTable::with_canaries(rows.clone(), targets, 0).unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Oblivious,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();
    let optimized = model.optimize_inference(Some(2)).unwrap();

    assert_predictions_close(
        &model.predict_rows(rows.clone()).unwrap(),
        &optimized.predict_rows(rows).unwrap(),
    );
}

#[test]
fn optimized_cart_model_batch_and_single_row_predictions_match() {
    let rows = vec![
        vec![0.0],
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![4.0],
        vec![5.0],
    ];
    let targets = vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0];
    let table = DenseTable::with_canaries(rows.clone(), targets, 0).unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();
    let optimized = model.optimize_inference(Some(1)).unwrap();

    let batch_preds = optimized.predict_rows(rows.clone()).unwrap();
    let single_row_preds = rows
        .iter()
        .map(|row| optimized.predict_rows(vec![row.clone()]).unwrap()[0])
        .collect::<Vec<_>>();
    let base_preds = model.predict_rows(rows).unwrap();

    assert_predictions_close(&batch_preds, &single_row_preds);
    assert_predictions_close(&batch_preds, &base_preds);
}

#[test]
fn optimized_oblivious_model_batch_and_single_row_predictions_match() {
    let rows = (0..64)
        .map(|idx| vec![f64::from((idx % 2) as u8), f64::from(((idx / 2) % 2) as u8)])
        .collect::<Vec<_>>();
    let targets = rows
        .iter()
        .map(|row| row[0] + row[1] * 0.5)
        .collect::<Vec<_>>();
    let table = DenseTable::with_canaries(rows.clone(), targets, 0).unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Oblivious,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();
    let optimized = model.optimize_inference(Some(1)).unwrap();

    let batch_preds = optimized.predict_rows(rows.clone()).unwrap();
    let single_row_preds = rows
        .iter()
        .map(|row| optimized.predict_rows(vec![row.clone()]).unwrap()[0])
        .collect::<Vec<_>>();
    let base_preds = model.predict_rows(rows).unwrap();

    assert_predictions_close(&batch_preds, &single_row_preds);
    assert_predictions_close(&batch_preds, &base_preds);
}

#[test]
fn compiled_artifact_round_trips_for_binary_classifier_runtime() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0, 10.0, 20.0],
            vec![0.0, 1.0, 10.0, 20.0],
            vec![1.0, 0.0, 10.0, 20.0],
            vec![1.0, 1.0, 10.0, 20.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();
    let optimized = model.optimize_inference(Some(1)).unwrap();
    let compiled = optimized.serialize_compiled().unwrap();
    let restored = OptimizedModel::deserialize_compiled(&compiled, Some(1)).unwrap();
    let rows = vec![
        vec![0.0, 0.0, 10.0, 20.0],
        vec![0.0, 1.0, 10.0, 20.0],
        vec![1.0, 0.0, 10.0, 20.0],
        vec![1.0, 1.0, 10.0, 20.0],
    ];

    assert_eq!(&compiled[..4], &COMPILED_ARTIFACT_MAGIC);
    assert_eq!(
        optimized.serialize().unwrap(),
        restored.serialize().unwrap()
    );
    assert_eq!(
        optimized.to_ir_json().unwrap(),
        restored.to_ir_json().unwrap()
    );
    assert_eq!(optimized.used_feature_indices(), vec![0, 1]);
    assert_eq!(restored.used_feature_indices(), vec![0, 1]);
    assert_predictions_close(
        &optimized.predict_rows(rows.clone()).unwrap(),
        &restored.predict_rows(rows).unwrap(),
    );
}

#[test]
fn optimized_model_projects_ensemble_inputs_to_used_features() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0, 5.0, 9.0],
            vec![0.0, 1.0, 5.0, 9.0],
            vec![1.0, 0.0, 5.0, 9.0],
            vec![1.0, 1.0, 5.0, 9.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Rf,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            n_trees: Some(8),
            max_features: MaxFeatures::Count(2),
            seed: Some(7),
            physical_cores: Some(1),
            ..TrainConfig::default()
        },
    )
    .unwrap();
    let optimized = model.optimize_inference(Some(1)).unwrap();

    assert_eq!(model.used_feature_indices(), vec![0, 1]);
    assert_eq!(optimized.used_feature_indices(), vec![0, 1]);
    assert_eq!(optimized.used_feature_count(), 2);
    assert_predictions_close(
        &model
            .predict_rows(vec![vec![0.0, 0.0, 5.0, 9.0], vec![1.0, 1.0, 5.0, 9.0]])
            .unwrap(),
        &optimized
            .predict_rows(vec![vec![0.0, 0.0, 5.0, 9.0], vec![1.0, 1.0, 5.0, 9.0]])
            .unwrap(),
    );
}

#[test]
fn compiled_artifact_round_trips_for_oblivious_regressor_runtime() {
    let rows = (0..32)
        .map(|idx| vec![f64::from((idx % 2) as u8), f64::from(((idx / 2) % 2) as u8)])
        .collect::<Vec<_>>();
    let targets = rows.iter().map(|row| row[0] + row[1] * 0.5).collect();
    let table = DenseTable::with_canaries(rows.clone(), targets, 0).unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Oblivious,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();
    let optimized = model.optimize_inference(Some(1)).unwrap();
    let compiled = optimized.serialize_compiled().unwrap();
    let restored = OptimizedModel::deserialize_compiled(&compiled, Some(2)).unwrap();

    assert_eq!(
        optimized.serialize().unwrap(),
        restored.serialize().unwrap()
    );
    assert_predictions_close(
        &optimized.predict_rows(rows.clone()).unwrap(),
        &restored.predict_rows(rows).unwrap(),
    );
}

#[test]
fn compiled_artifact_round_trips_for_boosted_binary_classifier_runtime() {
    let table = DenseTable::with_options(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.2, 0.1],
            vec![0.8, 0.9],
        ],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        0,
        NumericBins::fixed(8).unwrap(),
    )
    .unwrap();
    let model = Model::GradientBoostedTrees(
        GradientBoostedTrees::train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Gbm,
                task: Task::Classification,
                tree_type: TreeType::Cart,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Greedy,
                lookahead_depth: 1,
                lookahead_top_k: 8,
                lookahead_weight: 1.0,
                criterion: Criterion::SecondOrder,
                n_trees: Some(16),
                learning_rate: Some(0.2),
                max_depth: Some(2),
                ..TrainConfig::default()
            },
            Parallelism::sequential(),
        )
        .unwrap(),
    );
    let optimized = model.optimize_inference(Some(1)).unwrap();
    let compiled = optimized.serialize_compiled().unwrap();
    let restored = OptimizedModel::deserialize_compiled(&compiled, Some(1)).unwrap();
    let rows = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.2, 0.1],
        vec![0.8, 0.9],
    ];

    assert_eq!(
        optimized.predict_rows(rows.clone()).unwrap(),
        restored.predict_rows(rows.clone()).unwrap()
    );
    assert_predictions_close(
        &optimized
            .predict_proba_rows(rows.clone())
            .unwrap()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
        &restored
            .predict_proba_rows(rows)
            .unwrap()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
    );
}

#[test]
fn compiled_artifact_rejects_invalid_header() {
    let err = OptimizedModel::deserialize_compiled(b"bad", Some(1)).unwrap_err();
    assert!(matches!(
        err,
        CompiledArtifactError::ArtifactTooShort { .. }
    ));

    let err =
        OptimizedModel::deserialize_compiled(b"NOPE\x01\0\x01\0payload", Some(1)).unwrap_err();
    assert!(matches!(err, CompiledArtifactError::InvalidMagic(_)));
}

#[test]
fn optimized_model_rejects_zero_physical_cores() {
    let table = DenseTable::with_canaries(vec![vec![0.0]], vec![1.0], 0).unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();

    let err = model.optimize_inference(Some(0)).unwrap_err();

    assert!(matches!(
        err,
        OptimizeError::InvalidPhysicalCoreCount { requested: 0, .. }
    ));
}

#[test]
fn model_predicts_from_raw_rows_without_building_a_training_table() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();

    let preds = model
        .predict_rows(vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ])
        .unwrap();

    assert_eq!(preds, vec![0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn model_predicts_from_named_columns() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();

    let preds = model
        .predict_named_columns(BTreeMap::from([
            ("f0".to_string(), vec![0.0, 0.0, 1.0, 1.0]),
            ("f1".to_string(), vec![0.0, 1.0, 0.0, 1.0]),
        ]))
        .unwrap();

    assert_eq!(preds, vec![0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn model_rejects_missing_named_feature() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();

    let err = model
        .predict_named_columns(BTreeMap::from([("f0".to_string(), vec![0.0, 1.0])]))
        .unwrap_err();

    assert!(matches!(err, PredictError::MissingFeature(feature) if feature == "f1"));
}

#[test]
fn optimized_classifier_preserves_missing_routing() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0],
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![f64::NAN],
            vec![f64::NAN],
        ],
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,
            min_samples_split: None,
            min_samples_leaf: None,
            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();
    let optimized = model.optimize_inference(Some(1)).unwrap();
    let rows = vec![vec![0.0], vec![1.0], vec![f64::NAN]];

    assert_eq!(
        model.predict_rows(rows.clone()).unwrap(),
        optimized.predict_rows(rows).unwrap()
    );
}

#[test]
fn optimized_regressor_preserves_missing_routing() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0],
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![f64::NAN],
            vec![f64::NAN],
        ],
        vec![0.0, 0.0, 10.0, 10.0, 0.0, 0.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,
            min_samples_split: None,
            min_samples_leaf: None,
            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();
    let optimized = model.optimize_inference(Some(1)).unwrap();
    let rows = vec![vec![0.0], vec![1.0], vec![f64::NAN]];

    assert_eq!(
        model.predict_rows(rows.clone()).unwrap(),
        optimized.predict_rows(rows).unwrap()
    );
}

#[test]
fn optimized_missing_feature_configuration_can_skip_missing_checks() {
    let table = DenseTable::with_canaries(
        vec![vec![0.0], vec![0.0], vec![1.0]],
        vec![0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,
            min_samples_split: None,
            min_samples_leaf: None,
            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();

    let missing_aware = model.optimize_inference(Some(1)).unwrap();
    let missing_disabled = model
        .optimize_inference_with_missing_features(Some(1), Some(Vec::new()))
        .unwrap();

    assert_eq!(
        missing_aware.predict_rows(vec![vec![f64::NAN]]).unwrap(),
        vec![0.0]
    );
    assert_ne!(
        missing_aware.predict_rows(vec![vec![f64::NAN]]).unwrap(),
        missing_disabled.predict_rows(vec![vec![f64::NAN]]).unwrap()
    );
}

#[test]
fn model_rejects_unexpected_named_feature() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();

    let err = model
        .predict_named_columns(BTreeMap::from([
            ("f0".to_string(), vec![0.0, 1.0]),
            ("f1".to_string(), vec![0.0, 1.0]),
            ("f2".to_string(), vec![0.0, 1.0]),
        ]))
        .unwrap_err();

    assert!(matches!(err, PredictError::UnexpectedFeature(feature) if feature == "f2"));
}

#[test]
fn model_rejects_invalid_binary_value_during_inference() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();

    let err = model.predict_rows(vec![vec![0.5, 1.0]]).unwrap_err();

    assert!(matches!(
        err,
        PredictError::InvalidBinaryValue {
            feature_index: 0,
            row_index: 0,
            ..
        }
    ));
}

#[test]
fn model_predicts_from_sparse_binary_columns() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();

    let preds = model
        .predict_sparse_binary_columns(4, 2, vec![vec![2, 3], vec![1, 3]])
        .unwrap();

    assert_eq!(preds, vec![0.0, 0.0, 0.0, 1.0]);
}

#[cfg(feature = "polars")]
#[test]
fn model_predicts_from_polars_dataframe() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();
    let df = DataFrame::new(vec![
        Series::new("f0".into(), &[0.0, 0.0, 1.0, 1.0]).into(),
        Series::new("f1".into(), &[0.0, 1.0, 0.0, 1.0]).into(),
    ])
    .unwrap();

    let preds = model.predict_polars_dataframe(&df).unwrap();

    assert_eq!(preds, vec![0.0, 0.0, 0.0, 1.0]);
}

#[cfg(feature = "polars")]
#[test]
fn model_predicts_from_polars_lazyframe() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();
    let df = DataFrame::new(vec![
        Series::new("f0".into(), &[0.0, 0.0, 1.0, 1.0]).into(),
        Series::new("f1".into(), &[0.0, 1.0, 0.0, 1.0]).into(),
    ])
    .unwrap();

    let preds = model.predict_polars_lazyframe(&df.lazy()).unwrap();

    assert_eq!(preds, vec![0.0, 0.0, 0.0, 1.0]);
}

#[cfg(feature = "polars")]
#[test]
fn model_and_optimized_model_predict_large_polars_lazyframes_in_batches() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();
    let optimized = model.clone().optimize_inference(Some(1)).unwrap();
    let n_rows = 20_003usize;
    let f0: Vec<f64> = [0.0, 0.0, 1.0, 1.0]
        .iter()
        .copied()
        .cycle()
        .take(n_rows)
        .collect();
    let f1: Vec<f64> = [0.0, 1.0, 0.0, 1.0]
        .iter()
        .copied()
        .cycle()
        .take(n_rows)
        .collect();
    let expected: Vec<f64> = [0.0, 0.0, 0.0, 1.0]
        .iter()
        .copied()
        .cycle()
        .take(n_rows)
        .collect();
    let df = DataFrame::new(vec![
        Series::new("f0".into(), f0).into(),
        Series::new("f1".into(), f1).into(),
    ])
    .unwrap();

    let preds = model.predict_polars_lazyframe(&df.clone().lazy()).unwrap();
    let optimized_preds = optimized.predict_polars_lazyframe(&df.lazy()).unwrap();

    assert_eq!(preds, expected);
    assert_eq!(optimized_preds, expected);
}

#[cfg(feature = "polars")]
#[test]
fn model_rejects_polars_nulls() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap();
    let df = DataFrame::new(vec![
        Series::new("f0".into(), &[Some(0.0), None]).into(),
        Series::new("f1".into(), &[Some(0.0), Some(1.0)]).into(),
    ])
    .unwrap();

    let err = model.predict_polars_dataframe(&df).unwrap_err();

    assert!(
        matches!(err, PredictError::NullValue { feature, row_index } if feature == "f0" && row_index == 1)
    );
}

#[test]
fn ir_serializes_node_stats_for_standard_and_oblivious_trees() {
    let classifier_table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0],
        0,
    )
    .unwrap();
    let classifier = train(
        &classifier_table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap()
    .to_ir();

    let ir::TreeDefinition::NodeTree { nodes, .. } = &classifier.model.trees[0] else {
        panic!("classifier should export as node_tree");
    };
    assert!(nodes.iter().all(|node| match node {
        ir::NodeTreeNode::Leaf { stats, .. } => stats.sample_count > 0,
        ir::NodeTreeNode::BinaryBranch { stats, .. }
        | ir::NodeTreeNode::MultiwayBranch { stats, .. } => {
            stats.sample_count > 0 && stats.impurity.is_some() && stats.gain.is_some()
        }
    }));

    let regressor_table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 1.0, 1.0, 2.0],
        0,
    )
    .unwrap();
    let regressor = train(
        &regressor_table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Oblivious,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,

            physical_cores: Some(1),
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
        },
    )
    .unwrap()
    .to_ir();

    let ir::TreeDefinition::ObliviousLevels { levels, leaves, .. } = &regressor.model.trees[0]
    else {
        panic!("regressor should export as oblivious_levels");
    };
    assert!(levels.iter().all(|level| {
        level.stats.sample_count > 0 && level.stats.impurity.is_some() && level.stats.gain.is_some()
    }));
    assert!(leaves.iter().all(|leaf| leaf.stats.sample_count > 0));
}

#[test]
fn generated_json_schema_matches_checked_in_schema() {
    let generated = Model::json_schema_json_pretty().unwrap();
    let checked_in = include_str!("../schema/forestfire-ir.schema.json");
    assert_eq!(generated.trim_end(), checked_in.trim_end());
}

#[test]
fn lookahead_depth_trains_across_tree_families() {
    let classification_table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![2.0, 1.0],
        ],
        vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        0,
    )
    .unwrap();
    let regression_table = DenseTable::with_canaries(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![2.0, 1.0],
        ],
        vec![0.0, 0.5, 1.0, 1.5, 2.0, 1.0],
        0,
    )
    .unwrap();

    for (task, algorithm, tree_type, table) in [
        (
            Task::Classification,
            TrainAlgorithm::Dt,
            TreeType::Id3,
            &classification_table as &dyn forestfire_data::TableAccess,
        ),
        (
            Task::Classification,
            TrainAlgorithm::Dt,
            TreeType::C45,
            &classification_table as &dyn forestfire_data::TableAccess,
        ),
        (
            Task::Classification,
            TrainAlgorithm::Dt,
            TreeType::Cart,
            &classification_table as &dyn forestfire_data::TableAccess,
        ),
        (
            Task::Classification,
            TrainAlgorithm::Dt,
            TreeType::Randomized,
            &classification_table as &dyn forestfire_data::TableAccess,
        ),
        (
            Task::Classification,
            TrainAlgorithm::Dt,
            TreeType::Oblivious,
            &classification_table as &dyn forestfire_data::TableAccess,
        ),
        (
            Task::Regression,
            TrainAlgorithm::Dt,
            TreeType::Cart,
            &regression_table as &dyn forestfire_data::TableAccess,
        ),
        (
            Task::Regression,
            TrainAlgorithm::Dt,
            TreeType::Randomized,
            &regression_table as &dyn forestfire_data::TableAccess,
        ),
        (
            Task::Regression,
            TrainAlgorithm::Dt,
            TreeType::Oblivious,
            &regression_table as &dyn forestfire_data::TableAccess,
        ),
        (
            Task::Regression,
            TrainAlgorithm::Gbm,
            TreeType::Cart,
            &regression_table as &dyn forestfire_data::TableAccess,
        ),
        (
            Task::Regression,
            TrainAlgorithm::Gbm,
            TreeType::Randomized,
            &regression_table as &dyn forestfire_data::TableAccess,
        ),
        (
            Task::Regression,
            TrainAlgorithm::Gbm,
            TreeType::Oblivious,
            &regression_table as &dyn forestfire_data::TableAccess,
        ),
    ] {
        let model = train(
            table,
            TrainConfig {
                algorithm,
                task,
                tree_type,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Greedy,
                lookahead_depth: 2,
                lookahead_top_k: 8,
                lookahead_weight: 1.0,
                criterion: Criterion::Auto,
                max_depth: Some(3),
                min_samples_split: Some(2),
                min_samples_leaf: Some(1),
                physical_cores: Some(1),
                n_trees: Some(3),
                max_features: MaxFeatures::Auto,
                seed: Some(7),
                canary_filter: CanaryFilter::default(),
                compute_oob: false,
                learning_rate: Some(0.1),
                bootstrap: false,
                top_gradient_fraction: None,
                other_gradient_fraction: None,
                missing_value_strategy: MissingValueStrategyConfig::heuristic(),
                histogram_bins: None,
            },
        )
        .unwrap();
        assert_eq!(model.tree_type(), tree_type);
    }
}
