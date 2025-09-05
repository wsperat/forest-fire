# bindings/python/forestfire.pyi
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

class TargetMeanTree:
    mean_: float

    @classmethod
    def fit(
        cls, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> "TargetMeanTree": ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
