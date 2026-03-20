# bindings/python/forestfire.pyi
from __future__ import annotations

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
    ) -> None: ...

def train(
    x: Table | Any,
    y: Any | None = None,
    algorithm: str = "dt",
    task: str = "regression",
    tree_type: str = "target_mean",
    criterion: str = "auto",
    canaries: int = 2,
    physical_cores: int | None = None,
) -> "Model": ...

class OptimizedModel:
    algorithm: str
    task: str
    criterion: str
    tree_type: str
    mean_: float | None

    @classmethod
    def deserialize_compiled(
        cls,
        serialized: bytes,
        physical_cores: int | None = None,
    ) -> "OptimizedModel": ...
    def predict(self, x: Table | Any) -> NDArray[np.float64]: ...
    def to_ir_json(self, pretty: bool = False) -> str: ...
    def serialize(self, pretty: bool = False) -> str: ...
    def serialize_compiled(self) -> bytes: ...

class Model:
    algorithm: str
    task: str
    criterion: str
    tree_type: str
    mean_: float | None

    @classmethod
    def deserialize(cls, serialized: str) -> "Model": ...
    def optimize_inference(
        self,
        physical_cores: int | None = None,
    ) -> "OptimizedModel": ...
    def predict(self, x: Table | Any) -> NDArray[np.float64]: ...
    def to_ir_json(self, pretty: bool = False) -> str: ...
    def serialize(self, pretty: bool = False) -> str: ...
