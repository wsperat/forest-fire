from __future__ import annotations

from typing import Any

from . import _core


class Model:
    def __init__(
        self,
        inner: Any,
        *,
        categorical_strategy: str | None = None,
    ) -> None:
        self._inner = inner
        self.categorical_strategy = categorical_strategy

    @classmethod
    def deserialize(cls, serialized: str) -> "Model":
        try:
            inner = _core.Model.deserialize(serialized)
            categorical_strategy = None
        except ValueError:
            inner = _core.CategoricalModel.deserialize(serialized)
            categorical_strategy = "categorical"
        return cls(inner, categorical_strategy=categorical_strategy)

    def predict(self, x: Any) -> Any:
        return self._inner.predict(x)

    def predict_proba(self, x: Any) -> Any:
        return self._inner.predict_proba(x)

    def optimize_inference(
        self,
        physical_cores: int | None = None,
        missing_features: list[int] | None = None,
    ) -> "OptimizedModel":
        return OptimizedModel(
            self._inner.optimize_inference(physical_cores, missing_features),
            categorical_strategy=self.categorical_strategy,
        )

    def to_ir_json(self, pretty: bool = False) -> str:
        return self._inner.to_ir_json(pretty)

    def serialize(self, pretty: bool = False) -> str:
        return self._inner.serialize(pretty)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class OptimizedModel:
    def __init__(
        self,
        inner: Any,
        *,
        categorical_strategy: str | None = None,
    ) -> None:
        self._inner = inner
        self.categorical_strategy = categorical_strategy

    @classmethod
    def deserialize_compiled(
        cls, serialized: bytes, physical_cores: int | None = None
    ) -> "OptimizedModel":
        try:
            inner = _core.OptimizedModel.deserialize_compiled(
                serialized, physical_cores
            )
            categorical_strategy = None
        except ValueError:
            inner = _core.CategoricalOptimizedModel.deserialize_compiled(
                serialized, physical_cores
            )
            categorical_strategy = "categorical"
        return cls(inner, categorical_strategy=categorical_strategy)

    def predict(self, x: Any) -> Any:
        return self._inner.predict(x)

    def predict_proba(self, x: Any) -> Any:
        return self._inner.predict_proba(x)

    def to_ir_json(self, pretty: bool = False) -> str:
        return self._inner.to_ir_json(pretty)

    def serialize(self, pretty: bool = False) -> str:
        return self._inner.serialize(pretty)

    def serialize_compiled(self) -> bytes:
        return self._inner.serialize_compiled()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


Table = _core.Table


def train(
    x: Any,
    y: Any = None,
    *,
    categorical_strategy: str | None = None,
    categorical_features: str | list[str | int] | None = None,
    target_smoothing: float = 20.0,
    **kwargs: Any,
) -> Model:
    return Model(
        _core.train(
            x,
            y,
            categorical_strategy=categorical_strategy,
            categorical_features=categorical_features,
            target_smoothing=target_smoothing,
            **kwargs,
        ),
        categorical_strategy=categorical_strategy,
    )
