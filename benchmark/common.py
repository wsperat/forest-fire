from __future__ import annotations

import gc
import importlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Callable, Sequence

os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", ".cache")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

DEFAULT_ROW_GRID = tuple(2**exponent for exponent in range(13, 20, 2))
DEFAULT_FEATURE_GRID = (8, 16, 32, 64, 128, 256)


def default_parallelism() -> int:
    # Benchmarks should use the full machine by default; callers can still
    # override this from the CLI when they want to pin comparisons to fewer
    # cores.
    return max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class BenchmarkConfig:
    output_dir: Path
    family: str
    problem: str
    rows: int
    n_features: int
    n_estimators: int
    max_depth: int | None
    min_samples_split: int | None
    min_samples_leaf: int | None
    max_features: str
    physical_cores: int
    warmup_runs: int
    measurement_runs: int
    seed: int
    reference_max_leaves: int | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: str
    backend: str
    family: str
    problem: str
    n_estimators: int
    rows: int
    n_features: int
    max_depth: int | None
    min_samples_split: int | None
    min_samples_leaf: int | None
    max_features: str
    reference_max_leaves: int | None = None
    fit_seconds: float | None = None
    predict_seconds: float | None = None
    status: str = "ok"
    note: str | None = None


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_results(path: Path, results: list[BenchmarkResult]) -> None:
    path.write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def average_runtime(
    fn: Callable[[], Any], warmup_runs: int, measurement_runs: int
) -> float:
    for _ in range(warmup_runs):
        fn()
    total = 0.0
    for _ in range(measurement_runs):
        start = perf_counter()
        fn()
        total += perf_counter() - start
    return total / measurement_runs


def load_optional(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


def informative_view(
    X: NDArray[np.float64], requested_index: int
) -> NDArray[np.float64]:
    return X[:, requested_index % X.shape[1]]


def generate_dataset(
    problem: str,
    rows: int,
    n_features: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)

    if problem == "classification":
        # Build a genuinely clusterable classification task: five Gaussian
        # components in n dimensions, with the component-to-class assignment
        # randomized per benchmark cell while still ensuring both classes
        # appear.
        cluster_count = 5
        centers = rng.normal(loc=0.0, scale=3.0, size=(cluster_count, n_features))
        class_assignment = np.array([0, 0, 0, 1, 1], dtype=np.float64)
        rng.shuffle(class_assignment)
        component_ids = rng.integers(0, cluster_count, size=rows)
        X = (
            centers[component_ids]
            + rng.normal(loc=0.0, scale=1.0, size=(rows, n_features))
        ).astype(np.float64)
        y = class_assignment[component_ids].astype(np.float64)
    elif problem == "regression":
        # Use a separable n-dimensional square wave so the target remains
        # strongly non-linear but still structured enough for tree ensembles to
        # learn.
        X = rng.uniform(-np.pi, np.pi, size=(rows, n_features)).astype(np.float64)
        frequencies = 1.0 + (np.arange(n_features, dtype=np.float64) % 4.0)
        phases = np.linspace(0.0, np.pi, num=n_features, endpoint=False)
        square_wave = np.where(np.sin(X * frequencies + phases) >= 0.0, 1.0, -1.0)
        weights = np.linspace(1.0, 2.0, num=n_features, dtype=np.float64)
        weights /= np.linalg.norm(weights)
        y = (square_wave @ weights + 0.05 * rng.normal(size=rows)).astype(np.float64)
    else:
        raise ValueError(f"Unsupported problem: {problem}")

    return X, y


def sklearn_max_features(problem: str, max_features: str) -> str | float | int | None:
    if max_features in {"sqrt", "log2"}:
        return max_features
    if max_features == "all":
        return None if problem == "classification" else 1.0
    if max_features == "third":
        return 1.0 / 3.0
    return int(max_features)


def sklearn_hist_max_features(n_features: int, max_features: str) -> float | None:
    if max_features == "all":
        return 1.0
    if max_features == "sqrt":
        return float(min(1.0, np.sqrt(n_features) / n_features))
    if max_features == "log2":
        return float(min(1.0, np.log2(n_features) / n_features))
    if max_features == "third":
        return float(min(1.0, max(1, n_features // 3) / n_features))
    return float(min(1.0, int(max_features) / n_features))


def forestfire_max_features(max_features: str) -> str | int | None:
    if max_features == "all":
        return "all"
    if max_features in {"sqrt", "third", "auto"}:
        return max_features
    return int(max_features)


def lightgbm_feature_fraction(n_features: int, max_features: str) -> float:
    if max_features == "all":
        return 1.0
    if max_features == "sqrt":
        return float(min(1.0, np.sqrt(n_features) / n_features))
    if max_features == "third":
        return float(min(1.0, max(1, n_features // 3) / n_features))
    return float(min(1.0, int(max_features) / n_features))


def xgboost_colsample(n_features: int, max_features: str) -> float:
    return lightgbm_feature_fraction(n_features, max_features)


def catboost_rsm(n_features: int, max_features: str) -> float:
    return lightgbm_feature_fraction(n_features, max_features)


def with_reference_complexity(
    config: BenchmarkConfig,
    reference_max_leaves: int | None,
    reference_n_estimators: int | None,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        output_dir=config.output_dir,
        family=config.family,
        problem=config.problem,
        rows=config.rows,
        n_features=config.n_features,
        n_estimators=(
            reference_n_estimators
            if config.family == "gradient_boosting"
            and reference_n_estimators is not None
            else config.n_estimators
        ),
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        physical_cores=config.physical_cores,
        warmup_runs=config.warmup_runs,
        measurement_runs=config.measurement_runs,
        seed=config.seed,
        reference_max_leaves=reference_max_leaves,
    )


def backend_reference_max_leaves(config: BenchmarkConfig) -> int:
    reference_max_leaves = config.reference_max_leaves or 1
    if config.family == "gradient_boosting":
        # Histogram GBM backends such as LightGBM require at least two leaves,
        # so a ForestFire stump becomes the minimal valid non-trivial leaf
        # budget for cross-library comparison.
        return max(2, reference_max_leaves)
    return reference_max_leaves


def forestfire_non_binding_max_depth() -> int:
    # ForestFire does not currently expose an unbounded depth mode, so the
    # benchmark uses a large sentinel that should not bind before the reference
    # leaf budget does.
    return 64


def forestfire_fit(
    config: BenchmarkConfig,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> Any:
    from forestfire import train

    algorithm = "rf" if config.family == "random_forest" else "gbm"
    kwargs: dict[str, Any] = {
        "task": "classification"
        if config.problem == "classification"
        else "regression",
        "algorithm": algorithm,
        "tree_type": "cart",
        "n_trees": config.n_estimators,
        "max_depth": forestfire_non_binding_max_depth(),
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": forestfire_max_features(config.max_features),
        "canaries": 2,
        "physical_cores": config.physical_cores,
        "seed": config.seed,
    }
    if config.family == "gradient_boosting":
        kwargs["learning_rate"] = 0.1
        kwargs["top_gradient_fraction"] = 0.2
        kwargs["other_gradient_fraction"] = 0.1
    return train(X, y, **kwargs)


def reference_leaf_budget(model: Any) -> int:
    tree_count = int(model.tree_count)
    if tree_count == 0:
        # Canary-driven GBM can legitimately stop before adding the first tree.
        # Treat that as a one-leaf ensemble so the comparison backends still get
        # a valid, minimal complexity budget.
        return 1
    return max(
        int(model.tree_structure(tree_index=tree_index)["leaf_count"])
        for tree_index in range(tree_count)
    )


def reference_tree_count(model: Any) -> int:
    return int(model.tree_count)


def forestfire_reference_fit(
    config: BenchmarkConfig,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[Any, int, int]:
    model = forestfire_fit(config, X, y)
    return model, reference_leaf_budget(model), reference_tree_count(model)


def forestfire_reference_fit_timed(
    config: BenchmarkConfig,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[Any, int, int, float]:
    start = perf_counter()
    model = forestfire_fit(config, X, y)
    fit_seconds = perf_counter() - start
    return (
        model,
        reference_leaf_budget(model),
        reference_tree_count(model),
        fit_seconds,
    )


def forestfire_optimized_fit(
    config: BenchmarkConfig,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> Any:
    model = forestfire_fit(config, X, y)
    optimize_cores = max(1, os.cpu_count() or 1)
    return model.optimize_inference(physical_cores=optimize_cores)


def sklearn_fit(
    config: BenchmarkConfig,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> Any:
    sklearn_ensemble = load_optional("sklearn.ensemble")
    if sklearn_ensemble is None:
        raise ModuleNotFoundError("scikit-learn is not installed")

    max_features = sklearn_max_features(config.problem, config.max_features)
    if config.family == "gradient_boosting":
        hist_max_features = sklearn_hist_max_features(
            config.n_features, config.max_features
        )
        estimator_cls = (
            sklearn_ensemble.HistGradientBoostingClassifier
            if config.problem == "classification"
            else sklearn_ensemble.HistGradientBoostingRegressor
        )
        estimator = estimator_cls(
            max_iter=config.n_estimators,
            learning_rate=0.1,
            max_depth=None,
            max_leaf_nodes=backend_reference_max_leaves(config),
            min_samples_leaf=1,
            max_features=hist_max_features,
            random_state=config.seed,
        )
        return estimator.fit(X, y)

    estimator_cls = (
        sklearn_ensemble.RandomForestClassifier
        if config.problem == "classification"
        else sklearn_ensemble.RandomForestRegressor
    )
    estimator = estimator_cls(
        n_estimators=config.n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_leaf_nodes=config.reference_max_leaves,
        max_features=max_features,
        n_jobs=config.physical_cores,
        random_state=config.seed,
    )
    return estimator.fit(X, y)


def lightgbm_fit(
    config: BenchmarkConfig,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> Any:
    lightgbm = load_optional("lightgbm")
    if lightgbm is None:
        raise ModuleNotFoundError("lightgbm is not installed")

    estimator_cls = (
        lightgbm.LGBMClassifier
        if config.problem == "classification"
        else lightgbm.LGBMRegressor
    )
    kwargs: dict[str, Any] = {
        "n_estimators": config.n_estimators,
        "max_depth": -1,
        "num_leaves": backend_reference_max_leaves(config),
        "min_child_samples": 1,
        "feature_fraction": 1.0,
        "feature_fraction_bynode": lightgbm_feature_fraction(
            config.n_features, config.max_features
        ),
        "n_jobs": config.physical_cores,
        "num_threads": config.physical_cores,
        "random_state": config.seed,
        "verbosity": -1,
    }
    if config.family == "gradient_boosting":
        kwargs["boosting_type"] = "gbdt"
        kwargs["learning_rate"] = 0.1
        kwargs["bagging_fraction"] = 1.0
        kwargs["bagging_freq"] = 0
    else:
        kwargs["boosting_type"] = "rf"
        kwargs["bagging_fraction"] = 0.8
        kwargs["bagging_freq"] = 1
    estimator = estimator_cls(**kwargs)
    return estimator.fit(X, y)


def xgboost_fit(
    config: BenchmarkConfig,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> Any:
    xgboost = load_optional("xgboost")
    if xgboost is None:
        raise ModuleNotFoundError("xgboost is not installed")

    estimator_cls = (
        xgboost.XGBClassifier
        if config.problem == "classification"
        else xgboost.XGBRegressor
    )
    kwargs: dict[str, Any] = {
        # `lossguide` plus a large depth sentinel leaves the reference leaf
        # budget as the main complexity control.
        "max_depth": forestfire_non_binding_max_depth(),
        "tree_method": "hist",
        "grow_policy": "lossguide",
        "max_leaves": backend_reference_max_leaves(config),
        "min_child_weight": 0.0,
        "gamma": 0.0,
        "colsample_bynode": xgboost_colsample(config.n_features, config.max_features),
        "n_jobs": config.physical_cores,
        "random_state": config.seed,
        "verbosity": 0,
    }
    if config.family == "gradient_boosting":
        kwargs["n_estimators"] = config.n_estimators
        kwargs["learning_rate"] = 0.1
        kwargs["subsample"] = 1.0
    else:
        kwargs["n_estimators"] = 1
        kwargs["num_parallel_tree"] = config.n_estimators
        kwargs["subsample"] = 0.8
    estimator = estimator_cls(**kwargs)
    return estimator.fit(X, y)


def catboost_fit(
    config: BenchmarkConfig,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> Any:
    if config.family == "random_forest":
        raise ValueError(
            "CatBoost does not expose a direct random-forest benchmark mode"
        )

    catboost = load_optional("catboost")
    if catboost is None:
        raise ModuleNotFoundError("catboost is not installed")

    estimator_cls = (
        catboost.CatBoostClassifier
        if config.problem == "classification"
        else catboost.CatBoostRegressor
    )
    estimator = estimator_cls(
        iterations=config.n_estimators,
        # CatBoost does not provide a clean "no depth limit" setting here, so
        # the benchmark uses a large non-binding depth together with max_leaves.
        depth=16,
        grow_policy="Lossguide",
        max_leaves=backend_reference_max_leaves(config),
        min_data_in_leaf=1,
        learning_rate=0.1,
        rsm=catboost_rsm(config.n_features, config.max_features),
        random_seed=config.seed,
        thread_count=config.physical_cores,
        verbose=False,
    )
    return estimator.fit(X, y)


TRAIN_BACKEND_FITTERS: dict[
    str, Callable[[BenchmarkConfig, NDArray[np.float64], NDArray[np.float64]], Any]
] = {
    "forestfire": forestfire_fit,
    "sklearn": sklearn_fit,
    "lightgbm": lightgbm_fit,
    "xgboost": xgboost_fit,
    "catboost": catboost_fit,
}

PREDICT_BACKEND_FITTERS: dict[
    str, Callable[[BenchmarkConfig, NDArray[np.float64], NDArray[np.float64]], Any]
] = {
    "forestfire": forestfire_fit,
    "forestfire_optimized": forestfire_optimized_fit,
    "sklearn": sklearn_fit,
    "lightgbm": lightgbm_fit,
    "xgboost": xgboost_fit,
    "catboost": catboost_fit,
}


def format_result_line(result: BenchmarkResult) -> str:
    fields = [
        result.backend,
        result.family,
        result.problem,
        f"rows={result.rows}",
        f"features={result.n_features}",
        result.status,
    ]
    if result.fit_seconds is not None:
        fields.append(f"fit={result.fit_seconds:.6f}s")
    if result.predict_seconds is not None:
        fields.append(f"predict={result.predict_seconds:.6f}s")
    if result.reference_max_leaves is not None:
        fields.append(f"max_leaves={result.reference_max_leaves}")
    if result.note:
        fields.append(result.note)
    return " | ".join(fields)


def log(message: str) -> None:
    print(f"[benchmark] {message}", flush=True)


def successful_results(
    results: list[BenchmarkResult], metric: str
) -> list[BenchmarkResult]:
    return [
        result
        for result in results
        if result.status == "ok" and getattr(result, metric) is not None
    ]


def backend_color(backend: str) -> str:
    return {
        "forestfire": "#d55e00",
        "forestfire_optimized": "#e69f00",
        "sklearn": "#0072b2",
        "lightgbm": "#009e73",
        "xgboost": "#cc79a7",
        "catboost": "#56b4e9",
    }.get(backend, "#4c4c4c")


def plot_grid_comparison(
    results: list[BenchmarkResult],
    metric: str,
    row_grid: Sequence[int],
    feature_grid: Sequence[int],
    title_prefix: str,
    ylabel: str,
    output_path: Path,
) -> None:
    filtered = successful_results(results, metric)
    if not filtered:
        return

    backend_order = list(dict.fromkeys(result.backend for result in filtered))
    n_cols = 3
    n_rows = (len(feature_grid) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.0 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for index, feature_count in enumerate(feature_grid):
        ax = axes[index // n_cols][index % n_cols]
        ax.set_title(f"{feature_count} features")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("rows")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", axis="both", alpha=0.25)

        feature_results = [
            result for result in filtered if result.n_features == feature_count
        ]
        for backend in backend_order:
            backend_results = sorted(
                (result for result in feature_results if result.backend == backend),
                key=lambda result: result.rows,
            )
            if not backend_results:
                continue
            x_values = [result.rows for result in backend_results]
            y_values = [float(getattr(result, metric)) for result in backend_results]
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2.0,
                color=backend_color(backend),
                label=backend,
            )

    for index in range(len(feature_grid), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.985),
            ncol=1,
        )
    fig.suptitle(title_prefix)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_markdown(
    results: list[BenchmarkResult],
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    filtered = successful_results(results, metric)
    if not filtered:
        return

    lines = [f"# {title}", ""]
    lines.append(
        "This summary is generated from the measured benchmark grid rather than "
        "being handwritten commentary."
    )
    lines.append("")

    fastest_counts: Counter[str] = Counter()
    by_cell: dict[tuple[int, int], list[BenchmarkResult]] = defaultdict(list)
    for result in filtered:
        by_cell[(result.rows, result.n_features)].append(result)
    for cell_results in by_cell.values():
        fastest = min(cell_results, key=lambda result: float(getattr(result, metric)))
        fastest_counts[fastest.backend] += 1

    lines.append("## Fastest backend by grid cell")
    lines.append("")
    for backend, count in sorted(
        fastest_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"- `{backend}` is fastest in {count} grid cell(s)")
    lines.append("")

    by_backend: dict[str, list[float]] = defaultdict(list)
    for result in filtered:
        by_backend[result.backend].append(float(getattr(result, metric)))

    lines.append("## Median measured time")
    lines.append("")
    for backend, values in sorted(by_backend.items()):
        lines.append(f"- `{backend}` median: `{median(values):.6f}s`")
    lines.append("")

    lines.append("## Scaling from smallest to largest row count")
    lines.append("")
    feature_counts = sorted({result.n_features for result in filtered})
    for backend, values in sorted(by_backend.items()):
        ratios: list[float] = []
        for feature_count in feature_counts:
            feature_results = sorted(
                (
                    result
                    for result in filtered
                    if result.backend == backend and result.n_features == feature_count
                ),
                key=lambda result: result.rows,
            )
            if len(feature_results) < 2:
                continue
            smallest = float(getattr(feature_results[0], metric))
            largest = float(getattr(feature_results[-1], metric))
            if smallest > 0.0:
                ratios.append(largest / smallest)
        if ratios:
            lines.append(f"- `{backend}` median growth ratio: `{median(ratios):.2f}x`")
    lines.append("")

    optimized_pairs: list[float] = []
    grouped: defaultdict[tuple[int, int], dict[str, BenchmarkResult]] = defaultdict(
        dict
    )
    for result in filtered:
        grouped[(result.rows, result.n_features)][result.backend] = result
    for cell in grouped.values():
        base = cell.get("forestfire")
        optimized = cell.get("forestfire_optimized")
        if base is None or optimized is None:
            continue
        base_value = float(getattr(base, metric))
        optimized_value = float(getattr(optimized, metric))
        if optimized_value > 0.0:
            optimized_pairs.append(base_value / optimized_value)

    if optimized_pairs:
        lines.append("## ForestFire optimized-vs-base")
        lines.append("")
        lines.append(
            "- `forestfire_optimized` median speedup over base: "
            f"`{median(optimized_pairs):.2f}x`"
        )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def combo_seed(base_seed: int, problem: str, rows: int, n_features: int) -> int:
    problem_offset = 0 if problem == "classification" else 10_000_019
    return base_seed + problem_offset + rows * 17 + n_features * 101


def cleanup_large_objects(*objects: object) -> None:
    for obj in objects:
        del obj
    gc.collect()
