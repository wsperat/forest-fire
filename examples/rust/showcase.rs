use std::collections::BTreeMap;
use std::error::Error;

use forestfire_core::{
    BuilderStrategy, CanaryFilter, Criterion, MaxFeatures, MissingValueStrategyConfig, Model,
    OptimizedModel, SplitStrategy, Task, TrainAlgorithm, TrainConfig, TreeType, train,
};
use forestfire_data::{NumericBins, Table};

fn print_section(title: &str) {
    println!("\n== {title} ==");
}

fn regression_rows() -> (Vec<Vec<f64>>, Vec<f64>) {
    (
        (0..8).map(|value| vec![value as f64]).collect(),
        vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0],
    )
}

fn classification_rows() -> (Vec<Vec<f64>>, Vec<f64>) {
    (
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![2.0, 1.0],
        ],
        vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    )
}

fn show_regression_models() -> Result<(), Box<dyn Error>> {
    let (x, y) = regression_rows();
    let table = Table::with_options(x.clone(), y, 0, NumericBins::Auto)?;
    let configs = [
        (TreeType::Cart, Criterion::Mean),
        (TreeType::Cart, Criterion::Median),
        (TreeType::Randomized, Criterion::Mean),
        (TreeType::Oblivious, Criterion::Mean),
    ];

    print_section("Regression Models");
    for (tree_type, criterion) in configs {
        let model = train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Regression,
                tree_type,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Greedy,
                lookahead_depth: 1,
                lookahead_top_k: 8,
                lookahead_weight: 1.0,
                beam_width: 4,
                criterion,
                max_depth: None,

                min_samples_split: None,
                min_samples_leaf: None,
                physical_cores: None,
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
        )?;
        let preds = model.predict_rows(x.clone())?;
        println!("{tree_type:?} / {criterion:?} -> {:?}", &preds[..4]);
    }
    Ok(())
}

fn show_classification_models() -> Result<(), Box<dyn Error>> {
    let (x, y) = classification_rows();
    let table = Table::with_options(x.clone(), y, 0, NumericBins::Fixed(64))?;
    let configs = [
        (TreeType::Id3, Criterion::Entropy),
        (TreeType::C45, Criterion::Entropy),
        (TreeType::Cart, Criterion::Gini),
        (TreeType::Cart, Criterion::Entropy),
        (TreeType::Randomized, Criterion::Gini),
        (TreeType::Oblivious, Criterion::Gini),
    ];

    print_section("Classification Models");
    for (tree_type, criterion) in configs {
        let model = train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Greedy,
                lookahead_depth: 1,
                lookahead_top_k: 8,
                lookahead_weight: 1.0,
                beam_width: 4,
                criterion,
                max_depth: None,

                min_samples_split: None,
                min_samples_leaf: None,
                physical_cores: None,
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
        )?;
        let preds = model.predict_rows(x.clone())?;
        println!("{tree_type:?} / {criterion:?} -> {preds:?}");
    }
    Ok(())
}

fn show_tables() -> Result<(), Box<dyn Error>> {
    let (classification_x, classification_y) = classification_rows();
    let dense_x = classification_x
        .iter()
        .map(|row| vec![row[0], row[1] + 0.25])
        .collect();

    let dense = Table::with_options(dense_x, classification_y.clone(), 2, NumericBins::Auto)?;
    let sparse = Table::with_options(classification_x, classification_y, 1, NumericBins::Fixed(8))?;

    print_section("Training Tables");
    println!("dense kind  -> {:?}", dense.kind());
    println!("sparse kind -> {:?}", sparse.kind());
    Ok(())
}

fn show_inference_and_optimized_runtime() -> Result<(), Box<dyn Error>> {
    let (x, y) = classification_rows();
    let table = Table::with_options(x.clone(), y, 0, NumericBins::Auto)?;
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
            beam_width: 4,
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,
            physical_cores: None,
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
    )?;
    let optimized = model.optimize_inference(Some(1))?;

    let rows = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 0.0],
    ];
    let named_columns = BTreeMap::from([
        ("f0".to_string(), vec![0.0, 0.0, 1.0, 2.0]),
        ("f1".to_string(), vec![0.0, 1.0, 1.0, 0.0]),
    ]);

    print_section("Inference Inputs");
    println!("raw rows      -> {:?}", model.predict_rows(rows.clone())?);
    println!(
        "named columns -> {:?}",
        model.predict_named_columns(named_columns.clone())?
    );
    println!("optimized     -> {:?}", optimized.predict_rows(rows)?);
    println!(
        "optimized named-> {:?}",
        optimized.predict_named_columns(named_columns)?
    );

    let compiled = optimized.serialize_compiled()?;
    let restored = OptimizedModel::deserialize_compiled(&compiled, Some(1))?;
    println!("compiled size -> {}", compiled.len());
    println!(
        "compiled pred -> {:?}",
        restored.predict_rows(vec![vec![0.0, 0.0], vec![1.0, 1.0]])?
    );
    Ok(())
}

fn show_oblique_models() -> Result<(), Box<dyn Error>> {
    let x = vec![
        vec![-2.0, 1.0],
        vec![1.0, -2.0],
        vec![-1.0, 2.0],
        vec![2.0, -1.0],
        vec![-3.0, 1.0],
        vec![1.0, -3.0],
        vec![-1.0, 3.0],
        vec![3.0, -1.0],
    ];
    let y = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let table = Table::with_options(x.clone(), y, 0, NumericBins::Fixed(64))?;

    let axis = train(
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
            beam_width: 4,
            criterion: Criterion::Gini,
            ..TrainConfig::default()
        },
    )?;
    let oblique_beam = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::Oblique,
            builder: BuilderStrategy::Beam,
            lookahead_depth: 2,
            lookahead_top_k: 4,
            lookahead_weight: 0.5,
            beam_width: 2,
            criterion: Criterion::Gini,
            canary_filter: CanaryFilter::TopFraction(0.95),
            ..TrainConfig::default()
        },
    )?;

    print_section("Oblique Splits");
    println!("axis used     -> {:?}", axis.used_feature_indices());
    println!("oblique used  -> {:?}", oblique_beam.used_feature_indices());
    println!("axis pred     -> {:?}", axis.predict_rows(x[..4].to_vec())?);
    println!(
        "oblique pred  -> {:?}",
        oblique_beam.predict_rows(x[..4].to_vec())?
    );
    Ok(())
}

fn show_builder_strategies() -> Result<(), Box<dyn Error>> {
    let x = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        vec![2.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        vec![2.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    ];
    let y = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let table = Table::with_options(x, y, 0, NumericBins::Fixed(64))?;

    print_section("Builder Strategies");
    for (label, config) in [
        (
            "greedy",
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Cart,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Greedy,
                criterion: Criterion::Gini,
                ..TrainConfig::default()
            },
        ),
        (
            "lookahead",
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Cart,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Lookahead,
                lookahead_depth: 2,
                lookahead_top_k: 4,
                lookahead_weight: 0.5,
                criterion: Criterion::Gini,
                ..TrainConfig::default()
            },
        ),
        (
            "beam",
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Cart,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Beam,
                lookahead_depth: 2,
                lookahead_top_k: 4,
                lookahead_weight: 0.5,
                beam_width: 2,
                criterion: Criterion::Gini,
                ..TrainConfig::default()
            },
        ),
        (
            "optimal",
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Cart,
                split_strategy: SplitStrategy::AxisAligned,
                builder: BuilderStrategy::Optimal,
                criterion: Criterion::Gini,
                max_depth: Some(4),
                canary_filter: CanaryFilter::TopFraction(0.95),
                ..TrainConfig::default()
            },
        ),
    ] {
        let model = train(&table, config)?;
        println!("{label:>9} -> {:?}", model.tree_structure(0)?);
    }
    Ok(())
}

fn show_optimal_builder() -> Result<(), Box<dyn Error>> {
    let x = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0, 1.0, 0.0],
        vec![2.0, 0.0, 0.0, 1.0, 1.0],
        vec![2.0, 1.0, 1.0, 1.0, 0.0],
    ];
    let y = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let table = Table::with_options(x, y, 2, NumericBins::Fixed(64))?;

    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Optimal,
            criterion: Criterion::Gini,
            max_depth: Some(4),
            canary_filter: CanaryFilter::TopFraction(0.95),
            ..TrainConfig::default()
        },
    )?;

    print_section("Optimal Builder");
    println!("{:?}", model.tree_structure(0)?);
    Ok(())
}

fn show_missing_value_routing() -> Result<(), Box<dyn Error>> {
    let x = vec![
        vec![0.0, f64::NAN, 1.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, f64::NAN, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 0.0],
    ];
    let y = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let table = Table::with_options(x.clone(), y, 0, NumericBins::Auto)?;

    let heuristic = train(
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
            beam_width: 4,
            criterion: Criterion::Gini,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            ..TrainConfig::default()
        },
    )?;

    let optimized = heuristic.optimize_inference_with_missing_features(Some(1), Some(vec![1]))?;

    print_section("Missing Value Routing");
    println!(
        "base pred     -> {:?}",
        heuristic.predict_rows(x[..4].to_vec())?
    );
    println!(
        "optimized pred-> {:?}",
        optimized.predict_rows(x[..4].to_vec())?
    );
    Ok(())
}

fn show_serialization() -> Result<(), Box<dyn Error>> {
    let (x, y) = regression_rows();
    let table = Table::with_options(x.clone(), y, 0, NumericBins::Auto)?;
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
            beam_width: 4,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,
            physical_cores: None,
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
    )?;

    let serialized = model.serialize()?;
    let restored = Model::deserialize(&serialized)?;

    print_section("Serialization");
    println!("json bytes    -> {}", serialized.len());
    println!(
        "restored pred -> {:?}",
        restored.predict_rows(x[..3].to_vec())?
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    show_regression_models()?;
    show_classification_models()?;
    show_tables()?;
    show_inference_and_optimized_runtime()?;
    show_oblique_models()?;
    show_builder_strategies()?;
    show_optimal_builder()?;
    show_missing_value_routing()?;
    show_serialization()?;
    Ok(())
}
