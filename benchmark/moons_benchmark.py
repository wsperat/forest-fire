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
    catboost_fit,
    ensure_output_dir,
    lightgbm_fit,
    log,
    reference_leaf_budget,
    sklearn_fit,
    xgboost_fit,
)
from matplotlib.colors import Normalize
from numpy.typing import NDArray
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, log_loss
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
class MoonsBenchmarkResult:
    family: str
    backend: str
    train_seconds: float
    predict_seconds: float
    accuracy: float
    log_loss: float
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
            "classifiers on sklearn.make_moons."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument("--train-rows", type=int, default=4096)
    parser.add_argument("--test-rows", type=int, default=2048)
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--canaries", type=int, default=5)
    parser.add_argument("--filter", type=parse_canary_filter, default=0.95)
    parser.add_argument("--physical-cores", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--grid-resolution", type=int, default=250)
    return parser.parse_args()


def forestfire_fit_with_tree_type(
    family: str,
    tree_type: str,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    args: argparse.Namespace,
) -> Any:
    algorithm = "rf" if family == "random_forest" else "gbm"
    kwargs: dict[str, Any] = {
        "task": "classification",
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
        problem="classification",
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


def probability_column(model: Any, X: NDArray[np.float64]) -> NDArray[np.float64]:
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Expected binary classification probabilities.")
    return probabilities[:, 1]


def mesh_grid(
    X: NDArray[np.float64], resolution: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    mesh = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    return xx, yy, mesh


def fit_timed(fn: Any) -> tuple[Any, float]:
    start = perf_counter()
    model = fn()
    return model, perf_counter() - start


def predict_timed(
    model: Any, X: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    start = perf_counter()
    probabilities = probability_column(model, X)
    return probabilities, perf_counter() - start


def model_note(model: Any) -> str | None:
    tree_count = getattr(model, "tree_count", None)
    if tree_count == 0:
        return (
            "No trees were added. The ensemble stayed at its base score, so "
            "the probability surface is constant."
        )
    return None


def plot_probability_grid(
    family: str,
    train_X: NDArray[np.float64],
    train_y: NDArray[np.float64],
    xx: NDArray[np.float64],
    yy: NDArray[np.float64],
    mesh_probabilities: dict[str, NDArray[np.float64]],
    results: list[MoonsBenchmarkResult],
    output_path: Path,
) -> None:
    backend_order = [result.backend for result in results]
    n_cols = 3
    n_rows = (len(backend_order) + n_cols - 1) // n_cols

    # layout="constrained" ensures the colorbar doesn't overlap subplots
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 4.4 * n_rows),
        squeeze=False,
        layout="constrained",
    )
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = "RdBu_r"

    result_by_backend = {result.backend: result for result in results}
    last_im = None

    for index, backend in enumerate(backend_order):
        ax = axes[index // n_cols][index % n_cols]
        probabilities = mesh_probabilities[backend].reshape(xx.shape)

        # 1. Background probability fill (bottom layer)
        last_im = ax.contourf(
            xx, yy, probabilities, levels=50, cmap=cmap, norm=norm, alpha=0.3, zorder=1
        )

        # 2. Training points (middle layer)
        ax.scatter(
            train_X[:, 0],
            train_X[:, 1],
            c=train_y,
            cmap=cmap,
            norm=norm,
            s=14,
            linewidths=0.2,
            edgecolors="black",
            alpha=0.8,
            zorder=2,
        )

        # 3. Decision boundary line (top layer, now in black)
        ax.contour(
            xx,
            yy,
            probabilities,
            levels=[0.5],
            colors=["black"],  # Changed to black for high contrast
            linewidths=2.0,
            zorder=3,
        )

        result = result_by_backend[backend]
        ax.set_title(
            f"{backend}\nacc={result.accuracy:.3f} | logloss={result.log_loss:.3f}"
        )

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
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for index in range(len(backend_order), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    # 4. Global Colorbar on the right
    if last_im:
        cbar = fig.colorbar(last_im, ax=axes, location="right", shrink=0.7, aspect=30)
        cbar.set_label("Probability $P(y=1)$")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(
    family: str,
    results: list[MoonsBenchmarkResult],
    output_path: Path,
    plot_name: str,
) -> None:
    lines = [f"# make_moons summary | {family}", ""]
    lines.append(
        "This benchmark uses sklearn.make_moons to compare probability "
        "surfaces on a two-dimensional non-linear classification task."
    )
    lines.append("")
    lines.append(f"Probability plot: `{plot_name}`")
    lines.append("")
    lines.append("| backend | train | predict | accuracy | log loss |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for result in results:
        lines.append(
            f"| `{result.backend}` | `{result.train_seconds:.6f}s` | "
            f"`{result.predict_seconds:.6f}s` | `{result.accuracy:.4f}` | "
            f"`{result.log_loss:.4f}` |"
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
    X, y = make_moons(
        n_samples=total_rows,
        noise=args.noise,
        random_state=args.seed,
    )
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        train_size=args.train_rows,
        test_size=args.test_rows,
        random_state=args.seed,
        stratify=y,
    )
    xx, yy, mesh = mesh_grid(X, args.grid_resolution)

    benchmark_families = ["random_forest", "gradient_boosting"]
    for family in benchmark_families:
        log(
            "make_moons benchmark start | "
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

        results: list[MoonsBenchmarkResult] = []
        mesh_probabilities: dict[str, NDArray[np.float64]] = {}
        for backend in family_backends(family):
            log(f"make_moons fit | family={family} | backend={backend}")
            model, train_seconds = fit_timed(fitters[backend])
            probabilities, predict_seconds = predict_timed(model, test_X)
            mesh_probabilities[backend] = probability_column(model, mesh)
            predictions = (probabilities >= 0.5).astype(np.float64)
            results.append(
                MoonsBenchmarkResult(
                    family=family,
                    backend=backend,
                    train_seconds=train_seconds,
                    predict_seconds=predict_seconds,
                    accuracy=float(accuracy_score(test_y, predictions)),
                    log_loss=float(log_loss(test_y, probabilities)),
                    train_rows=args.train_rows,
                    test_rows=args.test_rows,
                    noise=args.noise,
                    n_estimators=args.n_estimators,
                    reference_max_leaves=reference_max_leaves,
                    note=model_note(model),
                )
            )

        json_path = args.output_dir / f"moons_results_{family}.json"
        png_path = args.output_dir / f"moons_probabilities_{family}.png"
        md_path = args.output_dir / f"moons_summary_{family}.md"
        json_path.write_text(
            json.dumps([asdict(result) for result in results], indent=2) + "\n",
            encoding="utf-8",
        )
        plot_probability_grid(
            family=family,
            train_X=train_X,
            train_y=train_y,
            xx=xx,
            yy=yy,
            mesh_probabilities=mesh_probabilities,
            results=results,
            output_path=png_path,
        )
        write_summary(family, results, md_path, png_path.name)
        log(f"make_moons benchmark complete | family={family}")


if __name__ == "__main__":
    main()
