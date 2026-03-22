# Benchmarks

ForestFire includes benchmark scripts for:

- training
- prediction
- optimized inference

## Tasks

- `task benchmark-inference`
- `task benchmark-training-rf`
- `task benchmark-training-extra-trees`
- `task benchmark-training-gbm`
- `task benchmark-prediction-rf`
- `task benchmark-prediction-extra-trees`
- `task benchmark-prediction-gbm`

## Output

Benchmark artifacts are written under `docs/benchmarks/`.

Examples:

- `training_benchmark_results_<family>_<problem>.json`
- `prediction_benchmark_results_<family>_<problem>.json`
- `training_library_comparison_<family>_<problem>.png`
- `prediction_library_comparison_<family>_<problem>.png`
- `predict_proba_library_comparison_<family>_<problem>.png`

## Comparisons

The benchmark scripts compare ForestFire against:

- scikit-learn
- LightGBM
- XGBoost

depending on the learner family being benchmarked.
