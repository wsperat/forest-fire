from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_FEATURE_GRID,
    DEFAULT_TRAINING_MICRO_ROW_GRID,
    BenchmarkConfig,
    TrainingMicroBenchmarkResult,
    average_runtime,
    cleanup_large_objects,
    combo_seed,
    default_parallelism,
    dump_results,
    ensure_output_dir,
    forestfire_max_features,
    format_result_line,
    generate_dataset,
    log,
    plot_micro_grid,
    write_training_micro_summary_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ForestFire training phases independently so table "
            "construction and learner fit can be inspected separately."
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
        "--rows-grid", nargs="+", type=int, default=DEFAULT_TRAINING_MICRO_ROW_GRID
    )
    parser.add_argument(
        "--feature-grid", nargs="+", type=int, default=DEFAULT_FEATURE_GRID
    )
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--physical-cores", type=int, default=default_parallelism())
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measurement-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def build_config(
    args: argparse.Namespace, problem: str, rows: int, n_features: int
) -> BenchmarkConfig:
    return BenchmarkConfig(
        output_dir=args.output_dir,
        family=args.family,
        problem=problem,
        rows=rows,
        n_features=n_features,
        n_estimators=args.n_estimators,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
        max_features=args.max_features,
        physical_cores=args.physical_cores,
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs,
        seed=combo_seed(args.seed, problem, rows, n_features),
    )


def forestfire_algorithm(family: str) -> str:
    return "rf" if family == "random_forest" else "gbm"


def forestfire_train_kwargs(config: BenchmarkConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "task": "classification"
        if config.problem == "classification"
        else "regression",
        "algorithm": forestfire_algorithm(config.family),
        "tree_type": "cart",
        "n_trees": config.n_estimators,
        "max_depth": None,
        "min_samples_split": None,
        "min_samples_leaf": None,
        "max_features": forestfire_max_features(config.max_features),
        "physical_cores": config.physical_cores,
        "seed": config.seed,
    }
    if config.family == "gradient_boosting":
        kwargs["learning_rate"] = 0.1
        kwargs["top_gradient_fraction"] = 0.2
        kwargs["other_gradient_fraction"] = 0.1
    return kwargs


def main() -> None:
    from forestfire import Table, train

    args = parse_args()
    ensure_output_dir(args.output_dir)
    log(
        "training micro-benchmark start | "
        f"family={args.family} | problems={list(args.problems)} | "
        f"rows={list(args.rows_grid)} | features={list(args.feature_grid)} | "
        f"estimators={args.n_estimators} | physical_cores={args.physical_cores}"
    )

    for problem in args.problems:
        result_prefix = f"{args.family}_{problem}"
        results: list[TrainingMicroBenchmarkResult] = []

        for rows in args.rows_grid:
            for n_features in args.feature_grid:
                config = build_config(args, problem, rows, n_features)
                log(
                    "dataset generation | "
                    f"problem={problem} | rows={rows} | features={n_features}"
                )
                X, y = generate_dataset(problem, rows, n_features, config.seed)
                train_kwargs = forestfire_train_kwargs(config)
                table = Table(X, y, canaries=2, bins="auto")

                def fit_from_table() -> Any:
                    return train(table, **train_kwargs)

                phases = {
                    "table_build": lambda: Table(X, y, canaries=2, bins="auto"),
                    "fit_from_table": fit_from_table,
                    "fit_end_to_end": lambda: train(
                        X, y, canaries=2, bins="auto", **train_kwargs
                    ),
                }

                for phase, fn in phases.items():
                    log(
                        "phase timing | "
                        f"phase={phase} | problem={problem} | "
                        f"rows={rows} | features={n_features}"
                    )
                    try:
                        seconds = average_runtime(
                            fn,
                            config.warmup_runs,
                            config.measurement_runs,
                        )
                        result = TrainingMicroBenchmarkResult(
                            benchmark="training_micro",
                            family=config.family,
                            problem=config.problem,
                            phase=phase,
                            rows=config.rows,
                            n_features=config.n_features,
                            n_estimators=config.n_estimators,
                            seconds=seconds,
                        )
                    except Exception as exc:
                        result = TrainingMicroBenchmarkResult(
                            benchmark="training_micro",
                            family=config.family,
                            problem=config.problem,
                            phase=phase,
                            rows=config.rows,
                            n_features=config.n_features,
                            n_estimators=config.n_estimators,
                            status="error",
                            note=str(exc),
                        )
                    print(format_result_line(result))
                    results.append(result)

                dump_results(
                    args.output_dir / f"training_micro_results_{result_prefix}.json",
                    results,
                )
                plot_micro_grid(
                    results,
                    feature_grid=args.feature_grid,
                    x_values=args.rows_grid,
                    title_prefix=f"Training micro-benchmark | {args.family} | {problem}",
                    ylabel="time (seconds)",
                    output_path=args.output_dir / f"training_micro_{result_prefix}.png",
                    x_label="training rows",
                    x_attr="rows",
                )
                write_training_micro_summary_markdown(
                    results,
                    title=f"Training micro summary | {args.family} | {problem}",
                    output_path=args.output_dir
                    / f"training_micro_summary_{result_prefix}.md",
                )
                cleanup_large_objects(table, X, y)

        log(f"training micro-benchmark complete | problem={problem}")


if __name__ == "__main__":
    main()
