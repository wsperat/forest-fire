from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _as_float(value: Any) -> float:
    if _is_missing(value):
        return float("nan")
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    return float(value)


def _is_numeric_like(value: Any) -> bool:
    if _is_missing(value):
        return True
    if isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)):
        return True
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _to_category_key(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return str(value)


@dataclass
class TabularInput:
    rows: list[list[Any]]
    column_names: list[str]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_features(self) -> int:
        return len(self.column_names)


def extract_tabular_input(x: Any, *, allow_single_row: bool = False) -> TabularInput:
    if hasattr(x, "collect") and hasattr(x, "slice"):
        return extract_tabular_input(x.collect(), allow_single_row=allow_single_row)

    if hasattr(x, "to_dict"):
        try:
            columns = x.to_dict(orient="list")
            if isinstance(columns, dict):
                return _from_named_columns(columns)
        except TypeError:
            pass
        try:
            columns = x.to_dict(as_series=False)
            if isinstance(columns, dict):
                return _from_named_columns(columns)
        except TypeError:
            pass

    if hasattr(x, "to_pydict"):
        columns = x.to_pydict()
        if isinstance(columns, dict):
            return _from_named_columns(columns)

    if hasattr(x, "__array__"):
        array = np.asarray(x, dtype=object)
        if array.ndim == 1:
            if not allow_single_row:
                raise ValueError("Expected a two-dimensional input matrix.")
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("Expected a two-dimensional input matrix.")
        return TabularInput(
            rows=array.tolist(),
            column_names=[f"f{i}" for i in range(array.shape[1])],
        )

    if isinstance(x, list):
        if not x:
            return TabularInput(rows=[], column_names=[])
        if isinstance(x[0], (list, tuple)):
            width = len(x[0])
            rows = [list(row) for row in x]
            for index, row in enumerate(rows):
                if len(row) != width:
                    raise ValueError(
                        f"Ragged row at index {index}: expected {width} columns, found {len(row)}."
                    )
            return TabularInput(
                rows=rows,
                column_names=[f"f{i}" for i in range(width)],
            )
        if allow_single_row:
            row = list(x)
            return TabularInput(
                rows=[row], column_names=[f"f{i}" for i in range(len(row))]
            )

    raise ValueError("Input could not be interpreted as a supported tabular matrix.")


def _from_named_columns(columns: dict[str, Iterable[Any]]) -> TabularInput:
    column_names = list(columns.keys())
    values = [list(columns[name]) for name in column_names]
    lengths = {len(column) for column in values}
    if len(lengths) > 1:
        raise ValueError("Columnar inputs must have columns of equal length.")
    n_rows = lengths.pop() if lengths else 0
    rows = [
        [values[col][row] for col in range(len(column_names))] for row in range(n_rows)
    ]
    return TabularInput(rows=rows, column_names=column_names)


def encode_target_for_categorical(y: Any) -> tuple[np.ndarray, str, list[Any]]:
    array = np.asarray(y)
    if array.ndim != 1:
        raise ValueError("Targets must be one-dimensional.")
    if array.dtype.kind in ("U", "S", "O", "b", "i"):
        labels, encoded = np.unique(array, return_inverse=True)
        if len(labels) <= 2:
            return encoded.astype(float), "binary", labels.tolist()
        return encoded.astype(float), "multiclass", labels.tolist()
    return array.astype(float), "continuous", []


def resolve_categorical_indices(
    data: TabularInput,
    categorical_features: str | list[str | int] | None,
) -> set[int]:
    if categorical_features is None:
        categorical = set()
        for feature_index in range(data.n_features):
            column = [row[feature_index] for row in data.rows]
            if any(not _is_numeric_like(value) for value in column):
                categorical.add(feature_index)
        return categorical

    if categorical_features == "all":
        return set(range(data.n_features))

    resolved = set[int]()
    name_to_index = {name: index for index, name in enumerate(data.column_names)}
    for feature in categorical_features:
        if isinstance(feature, int):
            resolved.add(feature)
            continue
        if feature not in name_to_index:
            raise ValueError(f"Unknown categorical feature '{feature}'.")
        resolved.add(name_to_index[feature])
    return resolved


@dataclass
class OneHotSpec:
    feature_index: int
    categories: list[str]


@dataclass
class TargetEncodingSpec:
    feature_index: int
    classes: list[float] | None
    priors: list[float]
    mapping: dict[str, list[float]]


@dataclass
class FisherEncodingSpec:
    feature_index: int
    mapping: dict[str, float]


class CategoricalPreprocessor:
    def transform(self, x: Any) -> np.ndarray:
        raise NotImplementedError


class IdentityPreprocessor(CategoricalPreprocessor):
    def transform(self, x: Any) -> np.ndarray:
        data = extract_tabular_input(x, allow_single_row=True)
        return np.asarray(
            [[_as_float(value) for value in row] for row in data.rows],
            dtype=np.float64,
        )


class EncodedCategoricalPreprocessor(CategoricalPreprocessor):
    def __init__(
        self,
        column_names: list[str],
        numeric_features: list[int],
        one_hot_specs: list[OneHotSpec],
        target_specs: list[TargetEncodingSpec],
        fisher_specs: list[FisherEncodingSpec],
    ) -> None:
        self.column_names = column_names
        self.numeric_features = numeric_features
        self.one_hot_specs = one_hot_specs
        self.target_specs = target_specs
        self.fisher_specs = fisher_specs

    def transform(self, x: Any) -> np.ndarray:
        data = extract_tabular_input(x, allow_single_row=True)
        if data.column_names != self.column_names:
            raise ValueError("Prediction columns do not match the training columns.")
        encoded_rows: list[list[float]] = []
        for row in data.rows:
            encoded: list[float] = []
            for feature_index in self.numeric_features:
                encoded.append(_as_float(row[feature_index]))
            for spec in self.one_hot_specs:
                category = _to_category_key(row[spec.feature_index])
                for known in spec.categories:
                    encoded.append(float(category == known))
                encoded.append(
                    float(category is None or category not in spec.categories)
                )
            for spec in self.target_specs:
                category = _to_category_key(row[spec.feature_index])
                encoded.extend(spec.mapping.get(category, spec.priors))
            for spec in self.fisher_specs:
                category = _to_category_key(row[spec.feature_index])
                encoded.append(spec.mapping.get(category, float("nan")))
            encoded_rows.append(encoded)
        return np.asarray(encoded_rows, dtype=np.float64)


def fit_categorical_preprocessor(
    x: Any,
    y: Any,
    *,
    strategy: str | None,
    categorical_features: str | list[str | int] | None,
    target_smoothing: float,
) -> tuple[CategoricalPreprocessor, np.ndarray]:
    data = extract_tabular_input(x)
    if strategy is None:
        return IdentityPreprocessor(), np.asarray(
            [[_as_float(value) for value in row] for row in data.rows],
            dtype=np.float64,
        )

    resolved_strategy = strategy.lower()
    supported = {"one_hot", "target", "fisher"}
    if resolved_strategy not in supported:
        raise ValueError(
            f"Unsupported categorical_strategy '{strategy}'. Expected one of {sorted(supported)}."
        )

    categorical = resolve_categorical_indices(data, categorical_features)
    if not categorical:
        return IdentityPreprocessor(), np.asarray(
            [[_as_float(value) for value in row] for row in data.rows],
            dtype=np.float64,
        )

    numeric_features = [i for i in range(data.n_features) if i not in categorical]
    y_encoded, target_kind, classes = encode_target_for_categorical(y)
    one_hot_specs: list[OneHotSpec] = []
    target_specs: list[TargetEncodingSpec] = []
    fisher_specs: list[FisherEncodingSpec] = []

    for feature_index in sorted(categorical):
        column = [row[feature_index] for row in data.rows]
        observed_categories = [
            category
            for category in dict.fromkeys(_to_category_key(value) for value in column)
            if category is not None
        ]
        if resolved_strategy == "one_hot":
            one_hot_specs.append(
                OneHotSpec(feature_index=feature_index, categories=observed_categories)
            )
            continue

        if resolved_strategy == "target":
            target_specs.append(
                _fit_target_spec(
                    feature_index,
                    column,
                    y_encoded,
                    target_kind,
                    classes,
                    target_smoothing,
                )
            )
            continue

        fisher_specs.append(
            _fit_fisher_spec(
                feature_index,
                column,
                y_encoded,
                target_kind,
                classes,
                target_smoothing,
            )
        )

    preprocessor = EncodedCategoricalPreprocessor(
        column_names=data.column_names,
        numeric_features=numeric_features,
        one_hot_specs=one_hot_specs,
        target_specs=target_specs,
        fisher_specs=fisher_specs,
    )
    return preprocessor, preprocessor.transform(x)


def _fit_target_spec(
    feature_index: int,
    column: list[Any],
    y_encoded: np.ndarray,
    target_kind: str,
    classes: list[Any],
    smoothing: float,
) -> TargetEncodingSpec:
    buckets: dict[str, list[int]] = {}
    for row_index, value in enumerate(column):
        category = _to_category_key(value)
        if category is None:
            continue
        buckets.setdefault(category, []).append(row_index)

    if target_kind == "continuous":
        global_mean = float(np.mean(y_encoded)) if len(y_encoded) else 0.0
        mapping = {
            category: [
                (float(np.sum(y_encoded[rows])) + smoothing * global_mean)
                / (len(rows) + smoothing)
            ]
            for category, rows in buckets.items()
        }
        return TargetEncodingSpec(
            feature_index=feature_index,
            classes=None,
            priors=[global_mean],
            mapping=mapping,
        )

    class_indices = np.unique(y_encoded.astype(int))
    priors = [
        (
            float(np.sum(y_encoded == class_index))
            + smoothing * (1.0 / len(class_indices))
        )
        / (len(y_encoded) + smoothing)
        for class_index in class_indices
    ]
    mapping: dict[str, list[float]] = {}
    for category, rows in buckets.items():
        encoded = []
        row_targets = y_encoded[rows].astype(int)
        for class_index, prior in zip(class_indices, priors):
            encoded.append(
                (float(np.sum(row_targets == class_index)) + smoothing * prior)
                / (len(rows) + smoothing)
            )
        mapping[category] = encoded
    return TargetEncodingSpec(
        feature_index=feature_index,
        classes=[float(class_index) for class_index in class_indices],
        priors=priors,
        mapping=mapping,
    )


def _fit_fisher_spec(
    feature_index: int,
    column: list[Any],
    y_encoded: np.ndarray,
    target_kind: str,
    classes: list[Any],
    smoothing: float,
) -> FisherEncodingSpec:
    target_spec = _fit_target_spec(
        feature_index,
        column,
        y_encoded,
        target_kind,
        classes,
        smoothing,
    )
    ordered = sorted(
        target_spec.mapping.items(),
        key=lambda item: tuple(item[1]),
    )
    mapping = {
        category: float(rank) for rank, (category, _values) in enumerate(ordered)
    }
    return FisherEncodingSpec(feature_index=feature_index, mapping=mapping)
