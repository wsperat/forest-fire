from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    BACKEND_FITTERS,
    BenchmarkConfig,
    BenchmarkResult,
    average_runtime,
    dump_results,
    ensure_output_dir,
    format_result_line,
    generate_dataset,
    log,
    plot_library_comparison,
)


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Benchmark prediction time against ForestFire, sklearn, LightGBM, and XGBoost."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument(
        "--family",
        choices=("random_forest", "extra_trees"),
        default="random_forest",
    )
    parser.add_argument(
        "--problem",
        choices=("classification", "regression"),
        default="classification",
    )
    parser.add_argument("--train-rows", type=int, default=10_000)
    parser.add_argument("--predict-rows", type=int, default=10_000)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--physical-cores", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measurement-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    raw = parser.parse_args()
    return BenchmarkConfig(
        output_dir=raw.output_dir,
        family=raw.family,
        problem=raw.problem,
        train_rows=raw.train_rows,
        predict_rows=raw.predict_rows,
        n_features=raw.n_features,
        n_estimators=raw.n_estimators,
        max_depth=raw.max_depth,
        min_samples_split=raw.min_samples_split,
        min_samples_leaf=raw.min_samples_leaf,
        max_features=raw.max_features,
        physical_cores=raw.physical_cores,
        warmup_runs=raw.warmup_runs,
        measurement_runs=raw.measurement_runs,
        seed=raw.seed,
    )


def main() -> None:
    config = parse_args()
    ensure_output_dir(config.output_dir)
    log(
        "prediction benchmark start | "
        f"family={config.family} | problem={config.problem} | "
        f"train_rows={config.train_rows} | predict_rows={config.predict_rows} | "
        f"features={config.n_features} | estimators={config.n_estimators}"
    )
    X_train, y_train, X_predict = generate_dataset(
        config.problem,
        config.train_rows,
        config.predict_rows,
        config.n_features,
        config.seed,
    )
    log("dataset generated")

    results: list[BenchmarkResult] = []
    for backend, fitter in BACKEND_FITTERS.items():
        log(f"fitting backend={backend}")
        try:
            model = fitter(config, X_train, y_train)
            log(f"predicting backend={backend}")
            predict_seconds = average_runtime(
                lambda: model.predict(X_predict),
                config.warmup_runs,
                config.measurement_runs,
            )
            predict_proba_seconds = None
            if config.problem == "classification" and hasattr(model, "predict_proba"):
                log(f"predict_proba backend={backend}")
                predict_proba_seconds = average_runtime(
                    lambda: model.predict_proba(X_predict),
                    config.warmup_runs,
                    config.measurement_runs,
                )
            result = BenchmarkResult(
                benchmark="prediction",
                backend=backend,
                family=config.family,
                problem=config.problem,
                n_estimators=config.n_estimators,
                train_rows=config.train_rows,
                predict_rows=config.predict_rows,
                n_features=config.n_features,
                max_depth=config.max_depth,
                min_samples_split=config.min_samples_split,
                min_samples_leaf=config.min_samples_leaf,
                max_features=config.max_features,
                predict_seconds=predict_seconds,
                predict_proba_seconds=predict_proba_seconds,
            )
        except Exception as exc:
            result = BenchmarkResult(
                benchmark="prediction",
                backend=backend,
                family=config.family,
                problem=config.problem,
                n_estimators=config.n_estimators,
                train_rows=config.train_rows,
                predict_rows=config.predict_rows,
                n_features=config.n_features,
                max_depth=config.max_depth,
                min_samples_split=config.min_samples_split,
                min_samples_leaf=config.min_samples_leaf,
                max_features=config.max_features,
                status="error",
                note=str(exc),
            )
        print(format_result_line(result))
        results.append(result)

    log("writing prediction benchmark json")
    dump_results(config.output_dir / "prediction_benchmark_results.json", results)
    log("writing prediction comparison plot")
    plot_library_comparison(
        results,
        metric="predict_seconds",
        title=(
            f"Prediction benchmark | {config.family} | {config.problem} | "
            f"{config.predict_rows:,} rows | {config.n_estimators} estimators"
        ),
        ylabel="predict time (seconds)",
        output_path=config.output_dir / "prediction_library_comparison.png",
    )
    if config.problem == "classification":
        log("writing predict_proba comparison plot")
        plot_library_comparison(
            results,
            metric="predict_proba_seconds",
            title=(
                f"Predict_proba benchmark | {config.family} | {config.problem} | "
                f"{config.predict_rows:,} rows | {config.n_estimators} estimators"
            ),
            ylabel="predict_proba time (seconds)",
            output_path=config.output_dir / "predict_proba_library_comparison.png",
        )
    log("prediction benchmark complete")


if __name__ == "__main__":
    main()
