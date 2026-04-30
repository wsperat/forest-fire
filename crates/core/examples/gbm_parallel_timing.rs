//! Timing benchmark for GBM parallel training.
//!
//! Measures sequential vs parallel wall-clock time across several dataset
//! configurations to identify where parallelism actually pays off.
use forestfire_core::{Task, TrainAlgorithm, TrainConfig, TreeType, train};
use forestfire_data::{NumericBins, Table};
use std::time::Instant;

fn make_table(n_rows: usize, n_features: usize) -> Table {
    let x: Vec<Vec<f64>> = (0..n_rows)
        .map(|i| {
            (0..n_features)
                .map(|j| ((i * 7 + j * 3) as f64).sin())
                .collect()
        })
        .collect();
    let y: Vec<f64> = (0..n_rows)
        .map(|i| x[i][0] * 2.0 - x[i][1] + x[i][2] * 0.5)
        .collect();
    Table::with_options(x, y, 0, NumericBins::Fixed(64)).unwrap()
}

fn timed(table: &Table, config: TrainConfig, rounds: usize) -> f64 {
    train(table, config.clone()).unwrap(); // warm up
    let start = Instant::now();
    for _ in 0..rounds {
        train(table, config.clone()).unwrap();
    }
    start.elapsed().as_millis() as f64 / rounds as f64
}

fn bench(
    label: &str,
    n_rows: usize,
    n_features: usize,
    n_trees: usize,
    max_depth: usize,
    rounds: usize,
) {
    let table = make_table(n_rows, n_features);
    let cpus = num_cpus::get();
    let base = TrainConfig {
        algorithm: TrainAlgorithm::Gbm,
        task: Task::Regression,
        tree_type: TreeType::Cart,
        n_trees: Some(n_trees),
        max_depth: Some(max_depth),
        ..TrainConfig::default()
    };
    let seq_ms = timed(
        &table,
        TrainConfig {
            physical_cores: Some(1),
            ..base.clone()
        },
        rounds,
    );
    let par_ms = timed(&table, base, rounds);
    println!(
        "{label} [{n_rows} rows · {n_features} feat · depth {max_depth} · {n_trees} trees · {cpus} CPUs]: \
         seq={seq_ms:.0}ms  par={par_ms:.0}ms  speedup={:.2}x",
        seq_ms / par_ms
    );
}

fn main() {
    println!("=== GBM parallel training benchmark ===\n");
    bench("small ", 10_000, 30, 20, 4, 5);
    bench("medium", 50_000, 50, 20, 5, 3);
    bench("large ", 100_000, 80, 10, 6, 3);
    bench("wide  ", 20_000, 200, 10, 5, 3);
    bench("deep  ", 30_000, 50, 10, 8, 3);
}
