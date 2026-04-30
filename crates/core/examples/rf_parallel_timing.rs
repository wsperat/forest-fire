//! Timing benchmark for RF parallel training.
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
    let y: Vec<f64> = (0..n_rows).map(|i| x[i][0] * 2.0 - x[i][1]).collect();
    Table::with_options(x, y, 0, NumericBins::Fixed(64)).unwrap()
}

fn timed(table: &Table, config: TrainConfig, rounds: usize) -> f64 {
    train(table, config.clone()).unwrap();
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
    depth: usize,
    rounds: usize,
) {
    let table = make_table(n_rows, n_features);
    let cpus = num_cpus::get();
    let base = TrainConfig {
        algorithm: TrainAlgorithm::Rf,
        task: Task::Regression,
        tree_type: TreeType::Cart,
        n_trees: Some(n_trees),
        max_depth: Some(depth),

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
    let speedup = seq_ms / par_ms;
    println!(
        "{label:<9} [{n_rows:>6} rows · {n_features:>3} feat · depth {depth} · {n_trees:>3} trees · {cpus} CPUs]: seq={seq_ms:.0}ms  par={par_ms:.0}ms  speedup={speedup:.2}x"
    );
}

fn main() {
    println!("=== RF parallel training benchmark ===\n");
    bench("small", 10_000, 30, 100, 8, 3);
    bench("medium", 50_000, 50, 100, 8, 3);
    bench("large", 100_000, 80, 100, 8, 2);
    bench("wide", 20_000, 200, 100, 8, 3);
    bench("few_trees", 50_000, 50, 10, 8, 5);
}
