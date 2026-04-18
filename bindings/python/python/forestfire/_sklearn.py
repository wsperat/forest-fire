# mypy: disable-error-code="import-not-found,import-untyped,misc,no-redef,attr-defined"

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._api import Model, train

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
except ImportError:

    class BaseEstimator:
        def get_params(self, deep: bool = True) -> dict[str, Any]:
            del deep
            return {name: getattr(self, name) for name in self._parameter_names()}

        def set_params(self, **params: Any) -> BaseEstimator:
            valid_params = set(self._parameter_names())
            for key, value in params.items():
                if key not in valid_params:
                    raise ValueError(
                        f"Invalid parameter '{key}' for estimator {type(self).__name__}."
                    )
                setattr(self, key, value)
            return self

        @classmethod
        def _parameter_names(cls) -> list[str]:
            names = []
            for name in cls.__init__.__code__.co_varnames[
                1 : cls.__init__.__code__.co_argcount
            ]:
                if name != "self":
                    names.append(name)
            return names

    class ClassifierMixin:
        def score(self, x: Any, y: Any) -> float:
            predictions = np.asarray(self.predict(x))
            targets = np.asarray(y)
            return float(np.mean(predictions == targets))

    class RegressorMixin:
        def score(self, x: Any, y: Any) -> float:
            predictions = np.asarray(self.predict(x), dtype=np.float64)
            targets = np.asarray(y, dtype=np.float64)
            residual = np.sum((targets - predictions) ** 2)
            total = np.sum((targets - np.mean(targets)) ** 2)
            if total == 0.0:
                return 1.0 if residual == 0.0 else 0.0
            return float(1.0 - (residual / total))


def _resolve_n_jobs(n_jobs: int | None) -> int | None:
    if n_jobs is None or n_jobs == -1:
        return None
    if n_jobs == 0:
        raise ValueError("n_jobs must be positive, -1, or None.")
    return n_jobs


def _infer_n_features(x: Any) -> int | None:
    shape = getattr(x, "shape", None)
    if shape is not None and len(shape) >= 2:
        columns = shape[1]
        if isinstance(columns, int):
            return columns

    if isinstance(x, list) and x:
        first_row = x[0]
        if isinstance(first_row, (list, tuple)):
            return len(first_row)
    return None


def _validate_sample_weight(sample_weight: Any) -> None:
    if sample_weight is not None:
        raise ValueError("sample_weight is not supported by ForestFire estimators.")


class _ForestFireEstimator(BaseEstimator, ABC):
    model_: Model

    def predict(self, x: Any) -> Any:
        self._require_fitted()
        return self.model_.predict(x)

    def _require_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise ValueError(f"{type(self).__name__} is not fitted yet.")

    @abstractmethod
    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> "_ForestFireEstimator":
        raise NotImplementedError


class _ForestFireClassifier(_ForestFireEstimator, ClassifierMixin, ABC):
    classes_: NDArray[Any]

    def predict_proba(self, x: Any) -> NDArray[np.float64]:
        self._require_fitted()
        return np.asarray(self.model_.predict_proba(x), dtype=np.float64)

    def _finalize_fit(self, x: Any, y: Any, model: Model) -> None:
        self.model_ = model
        self.classes_ = np.unique(np.asarray(y))
        n_features = _infer_n_features(x)
        if n_features is not None:
            self.n_features_in_ = n_features


class _ForestFireRegressor(_ForestFireEstimator, RegressorMixin, ABC):
    def _finalize_fit(self, x: Any, model: Model) -> None:
        self.model_ = model
        n_features = _infer_n_features(x)
        if n_features is not None:
            self.n_features_in_ = n_features


class _TreeClassifierBase(_ForestFireClassifier):
    _tree_type: str

    def __init__(
        self,
        criterion: str = "auto",
        canaries: int = 0,
        bins: str | int = "auto",
        histogram_bins: str | int | None = None,
        max_depth: int = 8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_jobs: int | None = None,
        random_state: int | None = None,
        missing_value_strategy: str | dict[str, str] = "heuristic",
        categorical_strategy: str | None = None,
        categorical_features: str | list[str | int] | None = None,
        target_smoothing: float = 20.0,
    ) -> None:
        self.criterion = criterion
        self.canaries = canaries
        self.bins = bins
        self.histogram_bins = histogram_bins
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.missing_value_strategy = missing_value_strategy
        self.categorical_strategy = categorical_strategy
        self.categorical_features = categorical_features
        self.target_smoothing = target_smoothing

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> "_TreeClassifierBase":
        _validate_sample_weight(sample_weight)
        model = train(
            x,
            y,
            algorithm="dt",
            task="classification",
            tree_type=self._tree_type,
            criterion=self.criterion,
            canaries=self.canaries,
            bins=self.bins,
            histogram_bins=self.histogram_bins,
            physical_cores=_resolve_n_jobs(self.n_jobs),
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            seed=self.random_state,
            missing_value_strategy=self.missing_value_strategy,
            categorical_strategy=self.categorical_strategy,
            categorical_features=self.categorical_features,
            target_smoothing=self.target_smoothing,
        )
        self._finalize_fit(x, y, model)
        return self


class _TreeRegressorBase(_ForestFireRegressor):
    _tree_type: str

    def __init__(
        self,
        criterion: str = "auto",
        canaries: int = 0,
        bins: str | int = "auto",
        histogram_bins: str | int | None = None,
        max_depth: int = 8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_jobs: int | None = None,
        random_state: int | None = None,
        missing_value_strategy: str | dict[str, str] = "heuristic",
        categorical_strategy: str | None = None,
        categorical_features: str | list[str | int] | None = None,
        target_smoothing: float = 20.0,
    ) -> None:
        self.criterion = criterion
        self.canaries = canaries
        self.bins = bins
        self.histogram_bins = histogram_bins
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.missing_value_strategy = missing_value_strategy
        self.categorical_strategy = categorical_strategy
        self.categorical_features = categorical_features
        self.target_smoothing = target_smoothing

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> "_TreeRegressorBase":
        _validate_sample_weight(sample_weight)
        model = train(
            x,
            y,
            algorithm="dt",
            task="regression",
            tree_type=self._tree_type,
            criterion=self.criterion,
            canaries=self.canaries,
            bins=self.bins,
            histogram_bins=self.histogram_bins,
            physical_cores=_resolve_n_jobs(self.n_jobs),
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            seed=self.random_state,
            missing_value_strategy=self.missing_value_strategy,
            categorical_strategy=self.categorical_strategy,
            categorical_features=self.categorical_features,
            target_smoothing=self.target_smoothing,
        )
        self._finalize_fit(x, model)
        return self


class _ForestClassifierBase(_ForestFireClassifier):
    _tree_type: str

    def __init__(
        self,
        n_estimators: int = 1000,
        criterion: str = "auto",
        canaries: int = 0,
        bins: str | int = "auto",
        histogram_bins: str | int | None = None,
        max_depth: int = 8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | None = None,
        n_jobs: int | None = None,
        random_state: int | None = None,
        compute_oob: bool = False,
        missing_value_strategy: str | dict[str, str] = "heuristic",
        categorical_strategy: str | None = None,
        categorical_features: str | list[str | int] | None = None,
        target_smoothing: float = 20.0,
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.canaries = canaries
        self.bins = bins
        self.histogram_bins = histogram_bins
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.compute_oob = compute_oob
        self.missing_value_strategy = missing_value_strategy
        self.categorical_strategy = categorical_strategy
        self.categorical_features = categorical_features
        self.target_smoothing = target_smoothing

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> "_ForestClassifierBase":
        _validate_sample_weight(sample_weight)
        model = train(
            x,
            y,
            algorithm="rf",
            task="classification",
            tree_type=self._tree_type,
            criterion=self.criterion,
            canaries=self.canaries,
            bins=self.bins,
            histogram_bins=self.histogram_bins,
            physical_cores=_resolve_n_jobs(self.n_jobs),
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_trees=self.n_estimators,
            max_features=self.max_features,
            seed=self.random_state,
            compute_oob=self.compute_oob,
            missing_value_strategy=self.missing_value_strategy,
            categorical_strategy=self.categorical_strategy,
            categorical_features=self.categorical_features,
            target_smoothing=self.target_smoothing,
        )
        self._finalize_fit(x, y, model)
        return self


class _ForestRegressorBase(_ForestFireRegressor):
    _tree_type: str

    def __init__(
        self,
        n_estimators: int = 1000,
        criterion: str = "auto",
        canaries: int = 0,
        bins: str | int = "auto",
        histogram_bins: str | int | None = None,
        max_depth: int = 8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | None = None,
        n_jobs: int | None = None,
        random_state: int | None = None,
        compute_oob: bool = False,
        missing_value_strategy: str | dict[str, str] = "heuristic",
        categorical_strategy: str | None = None,
        categorical_features: str | list[str | int] | None = None,
        target_smoothing: float = 20.0,
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.canaries = canaries
        self.bins = bins
        self.histogram_bins = histogram_bins
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.compute_oob = compute_oob
        self.missing_value_strategy = missing_value_strategy
        self.categorical_strategy = categorical_strategy
        self.categorical_features = categorical_features
        self.target_smoothing = target_smoothing

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> "_ForestRegressorBase":
        _validate_sample_weight(sample_weight)
        model = train(
            x,
            y,
            algorithm="rf",
            task="regression",
            tree_type=self._tree_type,
            criterion=self.criterion,
            canaries=self.canaries,
            bins=self.bins,
            histogram_bins=self.histogram_bins,
            physical_cores=_resolve_n_jobs(self.n_jobs),
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_trees=self.n_estimators,
            max_features=self.max_features,
            seed=self.random_state,
            compute_oob=self.compute_oob,
            missing_value_strategy=self.missing_value_strategy,
            categorical_strategy=self.categorical_strategy,
            categorical_features=self.categorical_features,
            target_smoothing=self.target_smoothing,
        )
        self._finalize_fit(x, model)
        return self


class _GBMClassifierBase(_ForestFireClassifier):
    _tree_type: str

    def __init__(
        self,
        n_estimators: int = 100,
        canaries: int = 0,
        bins: str | int = "auto",
        histogram_bins: str | int | None = None,
        max_depth: int = 8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        learning_rate: float | None = None,
        bootstrap: bool = False,
        top_gradient_fraction: float | None = None,
        other_gradient_fraction: float | None = None,
        n_jobs: int | None = None,
        random_state: int | None = None,
        missing_value_strategy: str | dict[str, str] = "heuristic",
        categorical_strategy: str | None = None,
        categorical_features: str | list[str | int] | None = None,
        target_smoothing: float = 20.0,
    ) -> None:
        self.n_estimators = n_estimators
        self.canaries = canaries
        self.bins = bins
        self.histogram_bins = histogram_bins
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.bootstrap = bootstrap
        self.top_gradient_fraction = top_gradient_fraction
        self.other_gradient_fraction = other_gradient_fraction
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.missing_value_strategy = missing_value_strategy
        self.categorical_strategy = categorical_strategy
        self.categorical_features = categorical_features
        self.target_smoothing = target_smoothing

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> "_GBMClassifierBase":
        _validate_sample_weight(sample_weight)
        model = train(
            x,
            y,
            algorithm="gbm",
            task="classification",
            tree_type=self._tree_type,
            canaries=self.canaries,
            bins=self.bins,
            histogram_bins=self.histogram_bins,
            physical_cores=_resolve_n_jobs(self.n_jobs),
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_trees=self.n_estimators,
            seed=self.random_state,
            learning_rate=self.learning_rate,
            bootstrap=self.bootstrap,
            top_gradient_fraction=self.top_gradient_fraction,
            other_gradient_fraction=self.other_gradient_fraction,
            missing_value_strategy=self.missing_value_strategy,
            categorical_strategy=self.categorical_strategy,
            categorical_features=self.categorical_features,
            target_smoothing=self.target_smoothing,
        )
        self._finalize_fit(x, y, model)
        return self


class _GBMRegressorBase(_ForestFireRegressor):
    _tree_type: str

    def __init__(
        self,
        n_estimators: int = 100,
        canaries: int = 0,
        bins: str | int = "auto",
        histogram_bins: str | int | None = None,
        max_depth: int = 8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        learning_rate: float | None = None,
        bootstrap: bool = False,
        top_gradient_fraction: float | None = None,
        other_gradient_fraction: float | None = None,
        n_jobs: int | None = None,
        random_state: int | None = None,
        missing_value_strategy: str | dict[str, str] = "heuristic",
        categorical_strategy: str | None = None,
        categorical_features: str | list[str | int] | None = None,
        target_smoothing: float = 20.0,
    ) -> None:
        self.n_estimators = n_estimators
        self.canaries = canaries
        self.bins = bins
        self.histogram_bins = histogram_bins
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.bootstrap = bootstrap
        self.top_gradient_fraction = top_gradient_fraction
        self.other_gradient_fraction = other_gradient_fraction
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.missing_value_strategy = missing_value_strategy
        self.categorical_strategy = categorical_strategy
        self.categorical_features = categorical_features
        self.target_smoothing = target_smoothing

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> "_GBMRegressorBase":
        _validate_sample_weight(sample_weight)
        model = train(
            x,
            y,
            algorithm="gbm",
            task="regression",
            tree_type=self._tree_type,
            canaries=self.canaries,
            bins=self.bins,
            histogram_bins=self.histogram_bins,
            physical_cores=_resolve_n_jobs(self.n_jobs),
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_trees=self.n_estimators,
            seed=self.random_state,
            learning_rate=self.learning_rate,
            bootstrap=self.bootstrap,
            top_gradient_fraction=self.top_gradient_fraction,
            other_gradient_fraction=self.other_gradient_fraction,
            missing_value_strategy=self.missing_value_strategy,
            categorical_strategy=self.categorical_strategy,
            categorical_features=self.categorical_features,
            target_smoothing=self.target_smoothing,
        )
        self._finalize_fit(x, model)
        return self
