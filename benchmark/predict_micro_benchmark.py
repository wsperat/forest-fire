from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_FEATURE_GRID,
    DEFAULT_PREDICTION_MICRO_BATCH_GRID,
    DEFAULT_PREDICTION_MICRO_TRAIN_ROWS,
    BenchmarkConfig,
    PredictionMicroBenchmarkResult,
    average_runtime,
    cleanup_large_objects,
    combo_seed,
    default_parallelism,
    dump_results,
    ensure_output_dir,
    forestfire_fit,
    format_result_line,
    generate_dataset,
    log,
    plot_micro_grid,
    write_prediction_micro_summary_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ForestFire prediction runtimes independently by fixing "
            "training complexity and varying prediction batch size."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument(
        "--family",
        choices=("random_forest", "gradient_boosting"),
        default="random_forest",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=("classification", "regression"),
        default=("classification", "regression"),
    )
    parser.add_argument(
        "--batch-grid", nargs="+", type=int, default=DEFAULT_PREDICTION_MICRO_BATCH_GRID
    )
    parser.add_argument(
        "--feature-grid", nargs="+", type=int, default=DEFAULT_FEATURE_GRID
    )
    parser.add_argument(
        "--train-rows", type=int, default=DEFAULT_PREDICTION_MICRO_TRAIN_ROWS
    )
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--physical-cores", type=int, default=default_parallelism())
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measurement-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def build_config(
    args: argparse.Namespace, problem: str, n_features: int
) -> BenchmarkConfig:
    return BenchmarkConfig(
        output_dir=args.output_dir,
        family=args.family,
        problem=problem,
        rows=args.train_rows,
        n_features=n_features,
        n_estimators=args.n_estimators,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
        max_features=args.max_features,
        physical_cores=args.physical_cores,
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs,
        seed=combo_seed(args.seed, problem, args.train_rows, n_features),
    )


def main() -> None:
    from forestfire import OptimizedModel

    args = parse_args()
    ensure_output_dir(args.output_dir)
    log(
        "prediction micro-benchmark start | "
        f"family={args.family} | problems={list(args.problems)} | "
        f"train_rows={args.train_rows} | batches={list(args.batch_grid)} | "
        f"features={list(args.feature_grid)} | estimators={args.n_estimators} | "
        f"physical_cores={args.physical_cores}"
    )

    for problem in args.problems:
        result_prefix = f"{args.family}_{problem}"
        results: list[PredictionMicroBenchmarkResult] = []

        for n_features in args.feature_grid:
            config = build_config(args, problem, n_features)
            log(
                "training reference model | "
                f"problem={problem} | train_rows={args.train_rows} | "
                f"features={n_features}"
            )
            X_train, y_train = generate_dataset(
                problem, args.train_rows, n_features, config.seed
            )
            model = forestfire_fit(config, X_train, y_train)
            optimized = model.optimize_inference(physical_cores=args.physical_cores)
            compiled_bytes = optimized.serialize_compiled()
            compiled = OptimizedModel.deserialize_compiled(
                compiled_bytes, physical_cores=args.physical_cores
            )

            for batch_rows in args.batch_grid:
                log(
                    "prediction batch generation | "
                    f"problem={problem} | batch_rows={batch_rows} | "
                    f"features={n_features}"
                )
                X_predict, _ = generate_dataset(
                    problem,
                    batch_rows,
                    n_features,
                    combo_seed(config.seed, problem, batch_rows, n_features),
                )

                phases = {
                    "predict_base": lambda: model.predict(X_predict),
                    "predict_optimized": lambda: optimized.predict(X_predict),
                    "predict_compiled": lambda: compiled.predict(X_predict),
                }

                for phase, fn in phases.items():
                    log(
                        "phase timing | "
                        f"phase={phase} | problem={problem} | "
                        f"batch_rows={batch_rows} | features={n_features}"
                    )
                    try:
                        seconds = average_runtime(
                            fn,
                            config.warmup_runs,
                            config.measurement_runs,
                        )
                        result = PredictionMicroBenchmarkResult(
                            benchmark="prediction_micro",
                            family=config.family,
                            problem=config.problem,
                            phase=phase,
                            train_rows=args.train_rows,
                            predict_rows=batch_rows,
                            n_features=n_features,
                            n_estimators=config.n_estimators,
                            seconds=seconds,
                        )
                    except Exception as exc:
                        result = PredictionMicroBenchmarkResult(
                            benchmark="prediction_micro",
                            family=config.family,
                            problem=config.problem,
                            phase=phase,
                            train_rows=args.train_rows,
                            predict_rows=batch_rows,
                            n_features=n_features,
                            n_estimators=config.n_estimators,
                            status="error",
                            note=str(exc),
                        )
                    print(format_result_line(result))
                    results.append(result)

                dump_results(
                    args.output_dir / f"prediction_micro_results_{result_prefix}.json",
                    results,
                )
                plot_micro_grid(
                    results,
                    feature_grid=args.feature_grid,
                    x_values=args.batch_grid,
                    title_prefix=(
                        "Prediction micro-benchmark | "
                        f"{args.family} | {problem} | trained on {args.train_rows} rows"
                    ),
                    ylabel="time (seconds)",
                    output_path=args.output_dir
                    / f"prediction_micro_{result_prefix}.png",
                    x_label="prediction rows",
                    x_attr="predict_rows",
                )
                write_prediction_micro_summary_markdown(
                    results,
                    title=(
                        "Prediction micro summary | "
                        f"{args.family} | {problem} | trained on {args.train_rows} rows"
                    ),
                    output_path=args.output_dir
                    / f"prediction_micro_summary_{result_prefix}.md",
                )
                cleanup_large_objects(X_predict)

            cleanup_large_objects(
                compiled_bytes, compiled, optimized, model, X_train, y_train
            )

        log(f"prediction micro-benchmark complete | problem={problem}")


if __name__ == "__main__":
    main()
