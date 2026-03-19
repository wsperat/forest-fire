use forestfire_core::train;
use forestfire_data::DenseTable;

fn main() {
    let x = vec![vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, -1.0]];
    let y = vec![5.0, 7.0, 6.0];
    let table = DenseTable::new(x, y).unwrap();

    let model = train(&table).unwrap();
    println!("mean = {}", model.mean);

    let preds = model.predict_table(&table);
    println!("preds = {:?}", preds);
}
