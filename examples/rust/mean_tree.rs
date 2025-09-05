use forestfire_core::TargetMeanTree;
use forestfire_data::DenseDataset;

fn main() {
    let x = vec![vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, -1.0]];
    let y = vec![5.0, 7.0, 6.0];
    let ds = DenseDataset::new(x, y).unwrap();

    let model = TargetMeanTree::train(&ds).unwrap();
    println!("mean = {}", model.mean);

    let preds = model.predict_dataset(&ds);
    println!("preds = {:?}", preds);
}
