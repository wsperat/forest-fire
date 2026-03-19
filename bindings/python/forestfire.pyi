# bindings/python/forestfire.pyi
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

def train(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    algorithm: str = "dt",
    task: str = "regression",
    tree_type: str = "target_mean",
    criterion: str = "auto",
    canaries: int = 2,
) -> "Model": ...

class Model:
    algorithm: str
    task: str
    criterion: str
    tree_type: str
    mean_: float | None

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
