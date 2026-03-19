use forestfire_core::{Model, TrainConfig, TreeType, train};
use forestfire_data::DenseTable;

fn main() {
    let x = vec![vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, -1.0]];
    let y = vec![5.0, 7.0, 6.0];
    let table = DenseTable::new(x, y).unwrap();

    let model = train(
        &table,
        TrainConfig {
            tree_type: TreeType::TargetMean,
            ..TrainConfig::default()
        },
    )
    .unwrap();
    let Model::TargetMean(model) = model else {
        panic!("expected target mean model");
    };
    println!("mean = {}", model.mean);

    let preds = model.predict_table(&table);
    println!("preds = {:?}", preds);
}
