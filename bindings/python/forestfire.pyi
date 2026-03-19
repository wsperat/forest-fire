# bindings/python/forestfire.pyi
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

def train(x: NDArray[np.float64], y: NDArray[np.float64]) -> "TargetMeanTree": ...

class TargetMeanTree:
    mean_: float

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
