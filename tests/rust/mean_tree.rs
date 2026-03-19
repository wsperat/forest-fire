use forestfire_core::train;
use forestfire_data::DenseTable;

#[test]
fn mean_tree_smoke() {
    let x = vec![vec![0.], vec![1.], vec![2.]];
    let y = vec![2.0, 4.0, 6.0];
    let table = DenseTable::new(x, y).unwrap();

    let m = train(&table).unwrap();
    assert!((m.mean - 4.0).abs() < 1e-12);

    let preds = m.predict_table(&table);
    assert_eq!(preds, vec![4.0, 4.0, 4.0]);
}
