from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import (
    DEFAULT_FEATURE_GRID,
    DEFAULT_ROW_GRID,
    PREDICT_BACKEND_FITTERS,
    BenchmarkConfig,
    BenchmarkResult,
    average_runtime,
    cleanup_large_objects,
    combo_seed,
    default_parallelism,
    dump_results,
    ensure_output_dir,
    forestfire_reference_fit,
    format_result_line,
    generate_dataset,
    log,
    plot_grid_comparison,
    with_reference_complexity,
    write_summary_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark random-forest or gradient-boosting prediction across a "
            "row/column grid with learnable synthetic data."
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
    parser.add_argument("--rows-grid", nargs="+", type=int, default=DEFAULT_ROW_GRID)
    parser.add_argument(
        "--feature-grid", nargs="+", type=int, default=DEFAULT_FEATURE_GRID
    )
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--physical-cores", type=int, default=default_parallelism())
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measurement-runs", type=int, default=3)
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


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)
    log(
        "prediction benchmark grid start | "
        f"family={args.family} | problems={list(args.problems)} | "
        f"rows={list(args.rows_grid)} | features={list(args.feature_grid)} | "
        f"estimators={args.n_estimators} | physical_cores={args.physical_cores}"
    )

    results: list[BenchmarkResult] = []
    for problem in args.problems:
        result_prefix = f"{args.family}_{problem}"
        for rows in args.rows_grid:
            for n_features in args.feature_grid:
                config = build_config(args, problem, rows, n_features)
                log(
                    "dataset generation | "
                    f"family={config.family} | problem={problem} | "
                    f"rows={rows} | features={n_features}"
                )
                X, y = generate_dataset(problem, rows, n_features, config.seed)
                log(
                    "reference fit | "
                    f"backend=forestfire | problem={problem} | "
                    f"rows={rows} | features={n_features}"
                )
                (
                    reference_model,
                    reference_max_leaves,
                    reference_n_estimators,
                ) = forestfire_reference_fit(config, X, y)
                log(
                    "reference complexity | "
                    f"problem={problem} | rows={rows} | "
                    f"features={n_features} | max_leaves={reference_max_leaves} | "
                    f"trained_trees={reference_n_estimators}"
                )
                benchmark_config = with_reference_complexity(
                    config, reference_max_leaves, reference_n_estimators
                )

                for backend, fitter in PREDICT_BACKEND_FITTERS.items():
                    log(
                        "fit+predict | "
                        f"backend={backend} | problem={problem} | "
                        f"rows={rows} | features={n_features}"
                    )
                    try:
                        if backend == "forestfire":
                            model = reference_model
                        elif backend == "forestfire_optimized":
                            optimize_cores = max(1, os.cpu_count() or 1)
                            model = reference_model.optimize_inference(
                                physical_cores=optimize_cores
                            )
                        elif (
                            benchmark_config.family == "gradient_boosting"
                            and benchmark_config.n_estimators == 0
                        ):
                            raise RuntimeError(
                                "ForestFire stopped before the first boosting "
                                "stage; no comparable non-ForestFire GBM fit exists."
                            )
                        else:
                            model = fitter(benchmark_config, X, y)
                        predict_seconds = average_runtime(
                            lambda: model.predict(X),
                            benchmark_config.warmup_runs,
                            benchmark_config.measurement_runs,
                        )
                        result = BenchmarkResult(
                            benchmark="prediction",
                            backend=backend,
                            family=benchmark_config.family,
                            problem=benchmark_config.problem,
                            n_estimators=benchmark_config.n_estimators,
                            rows=benchmark_config.rows,
                            n_features=benchmark_config.n_features,
                            max_depth=benchmark_config.max_depth,
                            min_samples_split=benchmark_config.min_samples_split,
                            min_samples_leaf=benchmark_config.min_samples_leaf,
                            max_features=benchmark_config.max_features,
                            reference_max_leaves=reference_max_leaves,
                            predict_seconds=predict_seconds,
                        )
                    except Exception as exc:
                        note = str(exc)
                        result = BenchmarkResult(
                            benchmark="prediction",
                            backend=backend,
                            family=benchmark_config.family,
                            problem=benchmark_config.problem,
                            n_estimators=benchmark_config.n_estimators,
                            rows=benchmark_config.rows,
                            n_features=benchmark_config.n_features,
                            max_depth=benchmark_config.max_depth,
                            min_samples_split=benchmark_config.min_samples_split,
                            min_samples_leaf=benchmark_config.min_samples_leaf,
                            max_features=benchmark_config.max_features,
                            reference_max_leaves=reference_max_leaves,
                            status=(
                                "skipped"
                                if "no comparable non-ForestFire GBM fit exists" in note
                                else "error"
                            ),
                            note=note,
                        )
                    print(format_result_line(result))
                    results.append(result)

                family_problem_results = [
                    result
                    for result in results
                    if result.family == args.family and result.problem == problem
                ]
                dump_results(
                    args.output_dir / f"prediction_grid_results_{result_prefix}.json",
                    family_problem_results,
                )
                plot_grid_comparison(
                    family_problem_results,
                    metric="predict_seconds",
                    row_grid=args.rows_grid,
                    feature_grid=args.feature_grid,
                    title_prefix=f"Prediction time | {args.family} | {problem}",
                    ylabel="predict time (seconds)",
                    output_path=args.output_dir
                    / f"prediction_grid_{result_prefix}.png",
                )
                write_summary_markdown(
                    family_problem_results,
                    metric="predict_seconds",
                    title=f"Prediction summary | {args.family} | {problem}",
                    output_path=args.output_dir
                    / f"prediction_summary_{result_prefix}.md",
                )
                cleanup_large_objects(X, y)

        log(f"writing prediction artifacts | problem={problem}")
        family_problem_results = [
            result
            for result in results
            if result.family == args.family and result.problem == problem
        ]
        dump_results(
            args.output_dir / f"prediction_grid_results_{result_prefix}.json",
            family_problem_results,
        )
        plot_grid_comparison(
            family_problem_results,
            metric="predict_seconds",
            row_grid=args.rows_grid,
            feature_grid=args.feature_grid,
            title_prefix=f"Prediction time | {args.family} | {problem}",
            ylabel="predict time (seconds)",
            output_path=args.output_dir / f"prediction_grid_{result_prefix}.png",
        )
        write_summary_markdown(
            family_problem_results,
            metric="predict_seconds",
            title=f"Prediction summary | {args.family} | {problem}",
            output_path=args.output_dir / f"prediction_summary_{result_prefix}.md",
        )

    log("prediction benchmark grid complete")


if __name__ == "__main__":
    main()
