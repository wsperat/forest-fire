from __future__ import annotations

import importlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

os.environ.setdefault("XDG_CACHE_HOME", ".cache")

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BenchmarkConfig:
    output_dir: Path
    family: str
    problem: str
    train_rows: int
    predict_rows: int
    n_features: int
    n_estimators: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    max_features: str
    physical_cores: int
    warmup_runs: int
    measurement_runs: int
    seed: int


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: str
    backend: str
    family: str
    problem: str
    n_estimators: int
    train_rows: int
    predict_rows: int
    n_features: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    max_features: str
    fit_seconds: float | None = None
    predict_seconds: float | None = None
    predict_proba_seconds: float | None = None
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


def generate_dataset(
    problem: str,
    train_rows: int,
    predict_rows: int,
    n_features: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    sklearn_datasets = load_optional("sklearn.datasets")
    if sklearn_datasets is None:
        raise ModuleNotFoundError(
            "scikit-learn is required for benchmark dataset generation"
        )

    if problem == "classification":
        total_rows = train_rows + predict_rows
        X, y = sklearn_datasets.make_classification(
            n_samples=total_rows,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=max(0, n_features // 4),
            n_repeated=0,
            n_classes=2,
            random_state=seed,
        )
        y = y.astype(np.float64)
    elif problem == "regression":
        total_rows = train_rows + predict_rows
        X, y = sklearn_datasets.make_regression(
            n_samples=total_rows,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            noise=0.2,
            random_state=seed,
        )
        y = y.astype(np.float64)
    else:
        raise ValueError(f"Unsupported problem: {problem}")

    X = X.astype(np.float64)
    return X[:train_rows], y[:train_rows], X[train_rows:]


def sklearn_max_features(problem: str, max_features: str) -> str | float | int:
    if max_features in {"sqrt", "log2"}:
        return max_features
    if max_features == "all":
        return (
            1.0 if problem == "regression" else None
        )  # sklearn RF classifier treats None as all
    if max_features == "third":
        return max(1, 1 / 3)
    return int(max_features)


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
        return min(1.0, np.sqrt(n_features) / n_features)
    if max_features == "third":
        return min(1.0, max(1, n_features // 3) / n_features)
    return min(1.0, int(max_features) / n_features)


def xgboost_colsample_bynode(n_features: int, max_features: str) -> float:
    return lightgbm_feature_fraction(n_features, max_features)


def forestfire_fit(
    config: BenchmarkConfig,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
) -> Any:
    from forestfire import train

    tree_type = "cart" if config.family == "random_forest" else "randomized"
    task = "classification" if config.problem == "classification" else "regression"
    return train(
        X_train,
        y_train,
        task=task,
        algorithm="rf",
        tree_type=tree_type,
        n_trees=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=forestfire_max_features(config.max_features),
        canaries=0,
        physical_cores=config.physical_cores,
        seed=config.seed,
    )


def sklearn_fit(
    config: BenchmarkConfig,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
) -> Any:
    sklearn_ensemble = load_optional("sklearn.ensemble")
    if sklearn_ensemble is None:
        raise ModuleNotFoundError("scikit-learn is not installed")

    max_features = sklearn_max_features(config.problem, config.max_features)
    if config.problem == "classification":
        estimator_cls = (
            sklearn_ensemble.RandomForestClassifier
            if config.family == "random_forest"
            else sklearn_ensemble.ExtraTreesClassifier
        )
    else:
        estimator_cls = (
            sklearn_ensemble.RandomForestRegressor
            if config.family == "random_forest"
            else sklearn_ensemble.ExtraTreesRegressor
        )

    estimator = estimator_cls(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=max_features,
        n_jobs=config.physical_cores,
        random_state=config.seed,
    )
    return estimator.fit(X_train, y_train)


def lightgbm_fit(
    config: BenchmarkConfig,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
) -> Any:
    lightgbm = load_optional("lightgbm")
    if lightgbm is None:
        raise ModuleNotFoundError("lightgbm is not installed")

    estimator_cls = (
        lightgbm.LGBMClassifier
        if config.problem == "classification"
        else lightgbm.LGBMRegressor
    )
    estimator = estimator_cls(
        boosting_type="rf",
        extra_trees=config.family == "extra_trees",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_child_samples=config.min_samples_leaf,
        feature_fraction=lightgbm_feature_fraction(
            config.n_features, config.max_features
        ),
        bagging_fraction=0.8,
        bagging_freq=1,
        n_jobs=config.physical_cores,
        random_state=config.seed,
        verbosity=-1,
    )
    return estimator.fit(X_train, y_train)


def xgboost_fit(
    config: BenchmarkConfig,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
) -> Any:
    if config.family == "extra_trees":
        raise ValueError(
            "XGBoost does not expose a direct extra-trees random-forest mode"
        )

    xgboost = load_optional("xgboost")
    if xgboost is None:
        raise ModuleNotFoundError("xgboost is not installed")

    estimator_cls = (
        xgboost.XGBClassifier
        if config.problem == "classification"
        else xgboost.XGBRegressor
    )
    estimator = estimator_cls(
        n_estimators=1,
        num_parallel_tree=config.n_estimators,
        max_depth=config.max_depth,
        tree_method="hist",
        subsample=0.8,
        colsample_bynode=xgboost_colsample_bynode(
            config.n_features, config.max_features
        ),
        n_jobs=config.physical_cores,
        random_state=config.seed,
        verbosity=0,
    )
    return estimator.fit(X_train, y_train)


BACKEND_FITTERS: dict[
    str, Callable[[BenchmarkConfig, NDArray[np.float64], NDArray[np.float64]], Any]
] = {
    "forestfire": forestfire_fit,
    "sklearn": sklearn_fit,
    "lightgbm": lightgbm_fit,
    "xgboost": xgboost_fit,
}


def format_result_line(result: BenchmarkResult) -> str:
    fields = [result.backend, result.family, result.problem, result.status]
    if result.fit_seconds is not None:
        fields.append(f"fit={result.fit_seconds:.6f}s")
    if result.predict_seconds is not None:
        fields.append(f"predict={result.predict_seconds:.6f}s")
    if result.predict_proba_seconds is not None:
        fields.append(f"predict_proba={result.predict_proba_seconds:.6f}s")
    if result.note:
        fields.append(result.note)
    return " | ".join(fields)
