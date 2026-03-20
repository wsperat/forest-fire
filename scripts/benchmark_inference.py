from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

# Keep plotting/cache writes inside the workspace so the benchmark task is
# reproducible in sandboxed and CI-like environments.
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", ".cache")

import matplotlib.pyplot as plt
import numpy as np
from forestfire import train
from numpy.typing import NDArray

DEFAULT_TREE_TYPES = ("cart", "oblivious")
DEFAULT_DEPTHS = (8, 16, 32, 64, 128, 256)
DEFAULT_ROW_COUNTS = (10_000, 100_000, 1_000_000, 10_000_000, 100_000_000)
DEFAULT_TRAINING_ROWS = 32_768
DEFAULT_PREDICTION_BATCH_ROWS = 65_536


@dataclass(frozen=True)
class BenchmarkArgs:
    output_dir: Path
    tree_types: tuple[str, ...]
    depths: tuple[int, ...]
    row_counts: tuple[int, ...]
    training_rows: int
    prediction_batch_rows: int
    warmup_runs: int
    measurement_runs: int
    min_batch_seconds: float


@dataclass(frozen=True)
class BenchmarkResult:
    tree_type: str
    requested_depth: int
    measured_depth: int
    row_count: int
    baseline_seconds: float
    optimized_single_core_seconds: float
    optimized_parallel_seconds: float

    @property
    def optimized_single_core_speedup(self) -> float:
        return self.baseline_seconds / self.optimized_single_core_seconds

    @property
    def optimized_parallel_speedup(self) -> float:
        return self.baseline_seconds / self.optimized_parallel_seconds


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args() -> BenchmarkArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ForestFire inference for non-optimized and optimized models, "
            "then store plots and raw results."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/benchmarks"),
        help="Directory where plots and raw JSON results will be written.",
    )
    parser.add_argument(
        "--tree-types",
        type=parse_csv_strings,
        default=DEFAULT_TREE_TYPES,
        help="Comma-separated tree types to benchmark.",
    )
    parser.add_argument(
        "--depths",
        type=parse_csv_ints,
        default=DEFAULT_DEPTHS,
        help="Comma-separated requested synthetic tree depths.",
    )
    parser.add_argument(
        "--row-counts",
        type=parse_csv_ints,
        default=DEFAULT_ROW_COUNTS,
        help="Comma-separated total row counts to score during benchmarking.",
    )
    parser.add_argument(
        "--training-rows",
        type=int,
        default=DEFAULT_TRAINING_ROWS,
        help="Row count to materialize for model training.",
    )
    parser.add_argument(
        "--prediction-batch-rows",
        type=int,
        default=DEFAULT_PREDICTION_BATCH_ROWS,
        help="Maximum row count to materialize in each timed prediction batch.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Warmup runs per model/input pair before timing begins.",
    )
    parser.add_argument(
        "--measurement-runs",
        type=int,
        default=7,
        help="Timed runs per model/input pair.",
    )
    parser.add_argument(
        "--min-batch-seconds",
        type=float,
        default=0.05,
        help="Minimum target duration for one timed batch before averaging.",
    )
    raw_args = parser.parse_args()
    return BenchmarkArgs(
        output_dir=raw_args.output_dir,
        tree_types=tuple(raw_args.tree_types),
        depths=tuple(raw_args.depths),
        row_counts=tuple(raw_args.row_counts),
        training_rows=raw_args.training_rows,
        prediction_batch_rows=raw_args.prediction_batch_rows,
        warmup_runs=raw_args.warmup_runs,
        measurement_runs=raw_args.measurement_runs,
        min_batch_seconds=raw_args.min_batch_seconds,
    )


def build_binary_depth_dataset(
    row_count: int,
    depth: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if depth <= 0:
        raise ValueError("depth must be positive")

    row_indices = np.arange(row_count, dtype=np.uint64)[:, None]
    feature_indices = np.arange(depth, dtype=np.uint64)[None, :]
    mixed = row_indices * np.uint64(0x9E3779B97F4A7C15)
    shifted = feature_indices * np.uint64(0xBF58476D1CE4E5B9)
    features = ((mixed ^ shifted) >> np.uint64(63)).astype(np.float64)

    prefix = min(depth, 12)
    weighted_sum = np.sum(
        features[:, :prefix] * (np.arange(prefix, dtype=np.float64) + 1.0),
        axis=1,
    )
    threshold = weighted_sum.mean()
    target = (weighted_sum >= threshold).astype(np.float64)
    return features, target


def criterion_for_tree_type(tree_type: str) -> str:
    if tree_type in {"id3", "c45"}:
        return "entropy"
    return "gini"


def measured_depth(model_json: str) -> int:
    ir = json.loads(model_json)
    tree = ir["model"]["trees"][0]
    representation = tree["representation"]
    if representation == "oblivious_levels":
        return int(tree["depth"])
    return max(int(node["depth"]) for node in tree["nodes"])


def build_prediction_batch(
    row_count: int,
    depth: int,
    offset: int,
) -> NDArray[np.float64]:
    row_indices = np.arange(offset, offset + row_count, dtype=np.uint64)[:, None]
    feature_indices = np.arange(depth, dtype=np.uint64)[None, :]
    mixed = row_indices * np.uint64(0x9E3779B97F4A7C15)
    shifted = feature_indices * np.uint64(0xBF58476D1CE4E5B9)
    return ((mixed ^ shifted) >> np.uint64(63)).astype(np.float64)


def median_prediction_seconds(
    predict_fn: Callable[[], None],
    warmup_runs: int,
    measurement_runs: int,
    min_batch_seconds: float,
) -> float:
    for _ in range(warmup_runs):
        predict_fn()

    inner_loops = 1
    while True:
        start = perf_counter()
        for _ in range(inner_loops):
            predict_fn()
        elapsed = perf_counter() - start
        if elapsed >= min_batch_seconds or inner_loops >= 1 << 18:
            break
        inner_loops *= 2

    timings: list[float] = []
    for _ in range(measurement_runs):
        start = perf_counter()
        for _ in range(inner_loops):
            predict_fn()
        elapsed = perf_counter() - start
        timings.append(elapsed / inner_loops)

    return float(np.median(np.array(timings, dtype=np.float64)))


def benchmark_case(
    tree_type: str, depth: int, row_count: int, args: BenchmarkArgs
) -> BenchmarkResult:
    training_features, target = build_binary_depth_dataset(args.training_rows, depth)
    model = train(
        training_features,
        target,
        task="classification",
        tree_type=tree_type,
        criterion=criterion_for_tree_type(tree_type),
        canaries=0,
        physical_cores=1,
    )
    optimized_single_core = model.optimize_inference(physical_cores=1)
    optimized_parallel = model.optimize_inference()

    validation_rows = min(row_count, args.prediction_batch_rows)
    validation_features = build_prediction_batch(validation_rows, depth, 0)
    baseline_preds = model.predict(validation_features)
    optimized_single_core_preds = optimized_single_core.predict(validation_features)
    optimized_parallel_preds = optimized_parallel.predict(validation_features)

    if not np.array_equal(baseline_preds, optimized_single_core_preds):
        raise ValueError(
            "Optimized single-core predictions differ from baseline predictions."
        )
    if not np.array_equal(baseline_preds, optimized_parallel_preds):
        raise ValueError(
            "Optimized parallel predictions differ from baseline predictions."
        )

    def predict_baseline() -> None:
        model.predict(validation_features)

    def predict_optimized_single_core() -> None:
        optimized_single_core.predict(validation_features)

    def predict_optimized_parallel() -> None:
        optimized_parallel.predict(validation_features)

    batch_count = row_count // validation_rows
    if row_count % validation_rows:
        batch_count += 1

    baseline_batch_seconds = median_prediction_seconds(
        predict_baseline,
        args.warmup_runs,
        args.measurement_runs,
        args.min_batch_seconds,
    )
    optimized_single_core_batch_seconds = median_prediction_seconds(
        predict_optimized_single_core,
        args.warmup_runs,
        args.measurement_runs,
        args.min_batch_seconds,
    )
    optimized_parallel_batch_seconds = median_prediction_seconds(
        predict_optimized_parallel,
        args.warmup_runs,
        args.measurement_runs,
        args.min_batch_seconds,
    )

    return BenchmarkResult(
        tree_type=tree_type,
        requested_depth=depth,
        measured_depth=measured_depth(model.to_ir_json()),
        row_count=row_count,
        baseline_seconds=baseline_batch_seconds * batch_count,
        optimized_single_core_seconds=optimized_single_core_batch_seconds * batch_count,
        optimized_parallel_seconds=optimized_parallel_batch_seconds * batch_count,
    )


def subset_for_tree_type(
    results: Sequence[BenchmarkResult],
    tree_type: str,
) -> list[BenchmarkResult]:
    return [result for result in results if result.tree_type == tree_type]


def plot_runtime(
    results: Sequence[BenchmarkResult], tree_type: str, output_dir: Path
) -> None:
    depths = sorted({result.requested_depth for result in results})
    fig, axes = plt.subplots(
        len(depths),
        1,
        figsize=(9, 3.6 * len(depths)),
        sharex=True,
    )
    if len(depths) == 1:
        axes = [axes]

    for axis, depth in zip(axes, depths):
        subset = sorted(
            [result for result in results if result.requested_depth == depth],
            key=lambda result: result.row_count,
        )
        row_counts = [result.row_count for result in subset]
        axis.plot(
            row_counts,
            [result.baseline_seconds * 1000.0 for result in subset],
            marker="o",
            label="baseline",
        )
        axis.plot(
            row_counts,
            [result.optimized_single_core_seconds * 1000.0 for result in subset],
            marker="o",
            label="optimized (1 core)",
        )
        axis.plot(
            row_counts,
            [result.optimized_parallel_seconds * 1000.0 for result in subset],
            marker="o",
            label="optimized (all cores)",
        )
        measured = sorted({result.measured_depth for result in subset})
        measured_label = ", ".join(str(value) for value in measured)
        axis.set_title(
            f"{tree_type} | requested depth {depth} | measured depth {measured_label}"
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_ylabel("prediction time (ms)")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(loc="best")

    axes[-1].set_xlabel("row count")
    fig.suptitle(f"ForestFire inference runtime benchmark: {tree_type}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"{tree_type}_runtime.png", dpi=180)
    plt.close(fig)


def plot_speedup(
    results: Sequence[BenchmarkResult], tree_type: str, output_dir: Path
) -> None:
    depths = sorted({result.requested_depth for result in results})
    fig, axes = plt.subplots(
        len(depths),
        1,
        figsize=(9, 3.6 * len(depths)),
        sharex=True,
    )
    if len(depths) == 1:
        axes = [axes]

    for axis, depth in zip(axes, depths):
        subset = sorted(
            [result for result in results if result.requested_depth == depth],
            key=lambda result: result.row_count,
        )
        row_counts = [result.row_count for result in subset]
        axis.plot(
            row_counts,
            [result.optimized_single_core_speedup for result in subset],
            marker="o",
            label="optimized (1 core)",
        )
        axis.plot(
            row_counts,
            [result.optimized_parallel_speedup for result in subset],
            marker="o",
            label="optimized (all cores)",
        )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_ylabel("speedup vs baseline")
        axis.set_title(f"{tree_type} | requested depth {depth}")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(loc="best")

    axes[-1].set_xlabel("row count")
    fig.suptitle(f"ForestFire optimized inference speedup: {tree_type}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"{tree_type}_speedup.png", dpi=180)
    plt.close(fig)


def save_results(results: Sequence[BenchmarkResult], output_dir: Path) -> None:
    payload = {
        "results": [asdict(result) for result in results],
    }
    (output_dir / "inference_benchmark_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkResult] = []
    for tree_type in args.tree_types:
        for depth in args.depths:
            for row_count in args.row_counts:
                print(
                    f"benchmarking tree_type={tree_type} depth={depth} rows={row_count}",
                    flush=True,
                )
                results.append(benchmark_case(tree_type, depth, row_count, args))

    save_results(results, args.output_dir)
    for tree_type in args.tree_types:
        tree_results = subset_for_tree_type(results, tree_type)
        plot_runtime(tree_results, tree_type, args.output_dir)
        plot_speedup(tree_results, tree_type, args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
