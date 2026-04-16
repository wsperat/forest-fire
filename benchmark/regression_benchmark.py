# mypy: disable-error-code="import-not-found,import-untyped"

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from common import (
    BenchmarkConfig,
    backend_color,
    catboost_fit,
    ensure_output_dir,
    lightgbm_fit,
    log,
    reference_leaf_budget,
    sklearn_fit,
    xgboost_fit,
)
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def parse_canary_filter(value: str) -> int | float:
    try:
        parsed_int = int(value)
    except ValueError:
        pass
    else:
        if str(parsed_int) == value:
            return parsed_int

    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--filter must be an integer or float."
        ) from exc


@dataclass(frozen=True)
class RegressionBenchmarkResult:
    family: str
    backend: str
    train_seconds: float
    predict_seconds: float
    rmse: float
    r2: float
    train_rows: int
    test_rows: int
    noise: float
    n_estimators: int
    reference_max_leaves: int | None
    note: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark and visualize random-forest and gradient-boosting "
            "regressors on a one-dimensional non-linear synthetic task."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument("--train-rows", type=int, default=4096)
    parser.add_argument("--test-rows", type=int, default=2048)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-features", type=str, default="all")
    parser.add_argument("--canaries", type=int, default=3)
    parser.add_argument("--filter", type=parse_canary_filter, default=3)
    parser.add_argument("--physical-cores", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--grid-resolution", type=int, default=250)
    return parser.parse_args()


def generate_regression_curve(
    rows: int, noise: float, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-4.0, 4.0, size=(rows, 1)).astype(np.float64)
    x0 = X[:, 0]
    y = (
        0.9 * np.sin(1.4 * x0)
        + 0.35 * np.cos(0.55 * x0 * x0)
        + 0.12 * x0
        + noise * rng.normal(size=rows)
    ).astype(np.float64)
    return X, y


def forestfire_fit_with_tree_type(
    family: str,
    tree_type: str,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    args: argparse.Namespace,
) -> Any:
    algorithm = "rf" if family == "random_forest" else "gbm"
    kwargs: dict[str, Any] = {
        "task": "regression",
        "algorithm": algorithm,
        "tree_type": tree_type,
        "n_trees": args.n_estimators,
        "max_depth": 64,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": args.max_features,
        "canaries": args.canaries,
        "filter": args.filter,
        "physical_cores": args.physical_cores,
        "seed": args.seed,
    }
    if family == "gradient_boosting":
        kwargs["learning_rate"] = 0.1
        kwargs["top_gradient_fraction"] = 0.2
        kwargs["other_gradient_fraction"] = 0.1
    from forestfire import train

    return train(X, y, **kwargs)


def build_external_config(
    family: str,
    n_estimators: int,
    n_features: int,
    max_features: str,
    physical_cores: int,
    seed: int,
    reference_max_leaves: int | None,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        output_dir=Path("docs/benchmarks"),
        family=family,
        problem="regression",
        rows=0,
        n_features=n_features,
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
        max_features=max_features,
        physical_cores=physical_cores,
        warmup_runs=0,
        measurement_runs=1,
        seed=seed,
        reference_max_leaves=reference_max_leaves,
    )


def prediction_column(model: Any, X: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(model.predict(X), dtype=np.float64)


def prediction_grid(X: NDArray[np.float64], resolution: int) -> NDArray[np.float64]:
    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    return np.linspace(x_min, x_max, resolution, dtype=np.float64).reshape(-1, 1)


def fit_timed(fn: Any) -> tuple[Any, float]:
    start = perf_counter()
    model = fn()
    return model, perf_counter() - start


def predict_timed(
    model: Any, X: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    start = perf_counter()
    predictions = prediction_column(model, X)
    return predictions, perf_counter() - start


def model_note(model: Any) -> str | None:
    tree_count = getattr(model, "tree_count", None)
    if tree_count == 0:
        return "No trees were added. The ensemble stayed at its base prediction."
    return None


def plot_prediction_grid(
    family: str,
    train_X: NDArray[np.float64],
    train_y: NDArray[np.float64],
    grid_X: NDArray[np.float64],
    grid_predictions: dict[str, NDArray[np.float64]],
    results: list[RegressionBenchmarkResult],
    output_path: Path,
) -> None:
    backend_order = [result.backend for result in results]
    n_cols = 3
    n_rows = (len(backend_order) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 3.8 * n_rows),
        squeeze=False,
        layout="constrained",
    )
    result_by_backend = {result.backend: result for result in results}
    x_sorted = np.argsort(train_X[:, 0])
    train_x_sorted = train_X[x_sorted, 0]
    train_y_sorted = train_y[x_sorted]
    grid_x = grid_X[:, 0]

    for index, backend in enumerate(backend_order):
        ax = axes[index // n_cols][index % n_cols]
        predictions = grid_predictions[backend]
        ax.scatter(
            train_X[:, 0],
            train_y,
            s=10,
            color="#56748f",
            linewidths=0.2,
            edgecolors="black",
            alpha=0.55,
            zorder=1,
        )
        ax.plot(
            grid_x, predictions, color=backend_color(backend), linewidth=2.0, zorder=2
        )
        ax.plot(
            train_x_sorted,
            train_y_sorted,
            color="black",
            linewidth=0.8,
            alpha=0.2,
            zorder=0,
        )
        result = result_by_backend[backend]
        ax.set_title(f"{backend}\nrmse={result.rmse:.3f} | r2={result.r2:.3f}")
        if result.note:
            ax.text(
                0.03,
                0.03,
                result.note,
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            )
        ax.tick_params(labelsize=8)

    for index in range(len(backend_order), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(
    family: str,
    results: list[RegressionBenchmarkResult],
    output_path: Path,
    plot_name: str,
) -> None:
    lines = [f"# regression curve summary | {family}", ""]
    lines.append(
        "This benchmark uses a one-dimensional non-linear synthetic regression task "
        "to compare fitted prediction curves."
    )
    lines.append("")
    lines.append(f"Prediction plot: `{plot_name}`")
    lines.append("")
    lines.append("| backend | train | predict | rmse | r2 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for result in results:
        lines.append(
            f"| `{result.backend}` | `{result.train_seconds:.6f}s` | "
            f"`{result.predict_seconds:.6f}s` | `{result.rmse:.4f}` | "
            f"`{result.r2:.4f}` |"
        )
        if result.note:
            lines.append(f"  note: {result.note}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def family_backends(family: str) -> list[str]:
    if family == "random_forest":
        return [
            "forestfire_cart",
            "forestfire_randomized",
            "forestfire_oblivious",
            "sklearn",
            "lightgbm",
            "xgboost",
        ]
    return [
        "forestfire_cart",
        "forestfire_randomized",
        "forestfire_oblivious",
        "sklearn",
        "lightgbm",
        "xgboost",
        "catboost",
    ]


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    total_rows = args.train_rows + args.test_rows
    X, y = generate_regression_curve(total_rows, args.noise, args.seed)
    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        train_size=args.train_rows,
        test_size=args.test_rows,
        random_state=args.seed,
    )
    grid_X = prediction_grid(X, args.grid_resolution)

    benchmark_families = ["random_forest", "gradient_boosting"]
    for family in benchmark_families:
        log(
            "regression curve benchmark start | "
            f"family={family} | train_rows={args.train_rows} | "
            f"test_rows={args.test_rows} | noise={args.noise}"
        )
        cart_model, _ = fit_timed(
            lambda: forestfire_fit_with_tree_type(
                family, "cart", train_X, train_y, args
            )
        )
        reference_max_leaves = reference_leaf_budget(cart_model)
        config = build_external_config(
            family=family,
            n_estimators=args.n_estimators,
            n_features=train_X.shape[1],
            max_features=args.max_features,
            physical_cores=args.physical_cores,
            seed=args.seed,
            reference_max_leaves=reference_max_leaves,
        )

        fitters: dict[str, Any] = {
            "forestfire_cart": lambda: forestfire_fit_with_tree_type(
                family, "cart", train_X, train_y, args
            ),
            "forestfire_randomized": lambda: forestfire_fit_with_tree_type(
                family, "randomized", train_X, train_y, args
            ),
            "forestfire_oblivious": lambda: forestfire_fit_with_tree_type(
                family, "oblivious", train_X, train_y, args
            ),
            "sklearn": lambda: sklearn_fit(config, train_X, train_y),
            "lightgbm": lambda: lightgbm_fit(config, train_X, train_y),
            "xgboost": lambda: xgboost_fit(config, train_X, train_y),
        }
        if family == "gradient_boosting":
            fitters["catboost"] = lambda: catboost_fit(config, train_X, train_y)

        results: list[RegressionBenchmarkResult] = []
        grid_predictions: dict[str, NDArray[np.float64]] = {}
        for backend in family_backends(family):
            log(f"regression curve fit | family={family} | backend={backend}")
            model, train_seconds = fit_timed(fitters[backend])
            predictions, predict_seconds = predict_timed(model, test_X)
            grid_predictions[backend] = prediction_column(model, grid_X)
            rmse = float(np.sqrt(mean_squared_error(test_y, predictions)))
            results.append(
                RegressionBenchmarkResult(
                    family=family,
                    backend=backend,
                    train_seconds=train_seconds,
                    predict_seconds=predict_seconds,
                    rmse=rmse,
                    r2=float(r2_score(test_y, predictions)),
                    train_rows=args.train_rows,
                    test_rows=args.test_rows,
                    noise=args.noise,
                    n_estimators=args.n_estimators,
                    reference_max_leaves=reference_max_leaves,
                    note=model_note(model),
                )
            )

        json_path = args.output_dir / f"regression_curve_results_{family}.json"
        png_path = args.output_dir / f"regression_curve_{family}.png"
        md_path = args.output_dir / f"regression_curve_summary_{family}.md"
        json_path.write_text(
            json.dumps([asdict(result) for result in results], indent=2) + "\n",
            encoding="utf-8",
        )
        plot_prediction_grid(
            family=family,
            train_X=train_X,
            train_y=train_y,
            grid_X=grid_X,
            grid_predictions=grid_predictions,
            results=results,
            output_path=png_path,
        )
        write_summary(family, results, md_path, png_path.name)
        log(f"regression curve benchmark complete | family={family}")


if __name__ == "__main__":
    main()
