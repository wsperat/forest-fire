from typing import Any

import numpy as np
from numpy.typing import NDArray

class Table:
    kind: str
    n_rows: int
    n_features: int
    canaries: int

    def __init__(
        self,
        x: Any,
        y: Any | None = None,
        canaries: int = 2,
        bins: str | int = "auto",
    ) -> None: ...

def train(
    x: Table | Any,
    y: Any | None = None,
    algorithm: str = "dt",
    task: str = "auto",
    tree_type: str = "cart",
    criterion: str = "auto",
    canaries: int = 2,
    bins: str | int = "auto",
    physical_cores: int | None = None,
    max_depth: int | None = None,
    min_samples_split: int | None = None,
    min_samples_leaf: int | None = None,
    n_trees: int | None = None,
    max_features: str | int | None = None,
    seed: int | None = None,
    compute_oob: bool = False,
    learning_rate: float | None = None,
    bootstrap: bool = False,
    top_gradient_fraction: float | None = None,
    other_gradient_fraction: float | None = None,
) -> "Model": ...

class OptimizedModel:
    algorithm: str
    task: str
    criterion: str
    tree_type: str
    canaries: int
    max_depth: int | None
    min_samples_split: int | None
    min_samples_leaf: int | None
    n_trees: int | None
    max_features: int | None
    seed: int | None
    compute_oob: bool
    oob_score: float | None
    learning_rate: float | None
    bootstrap: bool
    top_gradient_fraction: float | None
    other_gradient_fraction: float | None
    tree_count: int
    used_feature_count: int
    used_feature_indices: list[int]

    @classmethod
    def deserialize_compiled(
        cls,
        serialized: bytes,
        physical_cores: int | None = None,
    ) -> "OptimizedModel": ...
    def predict(self, x: Table | Any) -> Any: ...
    def predict_proba(self, x: Table | Any) -> NDArray[np.float64]: ...
    def tree_structure(self, tree_index: int = 0) -> dict[str, Any]: ...
    def tree_prediction_stats(self, tree_index: int = 0) -> dict[str, Any]: ...
    def tree_node(self, node_index: int, tree_index: int = 0) -> dict[str, Any]: ...
    def tree_level(self, level_index: int, tree_index: int = 0) -> dict[str, Any]: ...
    def tree_leaf(self, leaf_index: int, tree_index: int = 0) -> dict[str, Any]: ...
    def to_dataframe(self, tree_index: int | None = None) -> Any: ...
    def to_ir_json(self, pretty: bool = False) -> str: ...
    def serialize(self, pretty: bool = False) -> str: ...
    def serialize_compiled(self) -> bytes: ...

class Model:
    algorithm: str
    task: str
    criterion: str
    tree_type: str
    canaries: int
    max_depth: int | None
    min_samples_split: int | None
    min_samples_leaf: int | None
    n_trees: int | None
    max_features: int | None
    seed: int | None
    compute_oob: bool
    oob_score: float | None
    learning_rate: float | None
    bootstrap: bool
    top_gradient_fraction: float | None
    other_gradient_fraction: float | None
    tree_count: int
    used_feature_count: int
    used_feature_indices: list[int]

    @classmethod
    def deserialize(cls, serialized: str) -> "Model": ...
    def optimize_inference(
        self,
        physical_cores: int | None = None,
    ) -> "OptimizedModel": ...
    def predict(self, x: Table | Any) -> Any: ...
    def predict_proba(self, x: Table | Any) -> NDArray[np.float64]: ...
    def tree_structure(self, tree_index: int = 0) -> dict[str, Any]: ...
    def tree_prediction_stats(self, tree_index: int = 0) -> dict[str, Any]: ...
    def tree_node(self, node_index: int, tree_index: int = 0) -> dict[str, Any]: ...
    def tree_level(self, level_index: int, tree_index: int = 0) -> dict[str, Any]: ...
    def tree_leaf(self, leaf_index: int, tree_index: int = 0) -> dict[str, Any]: ...
    def to_dataframe(self, tree_index: int | None = None) -> Any: ...
    def to_ir_json(self, pretty: bool = False) -> str: ...
    def serialize(self, pretty: bool = False) -> str: ...
