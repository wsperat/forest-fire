use std::collections::BTreeMap;
use std::error::Error;

use forestfire_core::{
    Criterion, MaxFeatures, Model, OptimizedModel, Task, TrainAlgorithm, TrainConfig, TreeType,
    train,
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
                criterion,
                max_depth: None,

                min_samples_split: None,
                min_samples_leaf: None,
                physical_cores: None,
                n_trees: None,
                max_features: MaxFeatures::Auto,
                seed: None,
                compute_oob: false,
                learning_rate: None,
                bootstrap: false,
                top_gradient_fraction: None,
                other_gradient_fraction: None,
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
                criterion,
                max_depth: None,

                min_samples_split: None,
                min_samples_leaf: None,
                physical_cores: None,
                n_trees: None,
                max_features: MaxFeatures::Auto,
                seed: None,
                compute_oob: false,
                learning_rate: None,
                bootstrap: false,
                top_gradient_fraction: None,
                other_gradient_fraction: None,
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
            criterion: Criterion::Gini,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,
            physical_cores: None,
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
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

fn show_serialization() -> Result<(), Box<dyn Error>> {
    let (x, y) = regression_rows();
    let table = Table::with_options(x.clone(), y, 0, NumericBins::Auto)?;
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Cart,
            criterion: Criterion::Mean,
            max_depth: None,

            min_samples_split: None,
            min_samples_leaf: None,
            physical_cores: None,
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
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
    show_serialization()?;
    Ok(())
}
