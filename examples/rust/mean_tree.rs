use std::error::Error;

use forestfire_core::{Model, TrainConfig, TreeType, train};
use forestfire_data::DenseTable;

fn main() -> Result<(), Box<dyn Error>> {
    let x = vec![vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, -1.0]];
    let y = vec![5.0, 7.0, 6.0];
    let table = DenseTable::new(x, y)?;

    let model = train(
        &table,
        TrainConfig {
            tree_type: TreeType::TargetMean,
            ..TrainConfig::default()
        },
    )?;
    let Model::TargetMean(model) = model else {
        panic!("expected target mean model");
    };
    println!("mean = {}", model.mean);

    let preds = model.predict_table(&table);
    println!("preds = {:?}", preds);
    Ok(())
}
