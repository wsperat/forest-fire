use forestfire_core::TargetMeanTree;
use forestfire_data::DenseDataset;

#[test]
fn mean_tree_smoke() {
    let x = vec![vec![0.], vec![1.], vec![2.]];
    let y = vec![2.0, 4.0, 6.0];
    let ds = DenseDataset::new(x, y).unwrap();

    let m = TargetMeanTree::train(&ds).unwrap();
    assert!((m.mean - 4.0).abs() < 1e-12);

    let preds = m.predict_dataset(&ds);
    assert_eq!(preds, vec![4.0, 4.0, 4.0]);
}
