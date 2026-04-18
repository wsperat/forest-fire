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
        return cls(_core.Model.deserialize(serialized))

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
        if self.categorical_strategy is not None:
            raise ValueError(
                "IR export is not supported for models trained with native categorical transforms."
            )
        return self._inner.to_ir_json(pretty)

    def serialize(self, pretty: bool = False) -> str:
        if self.categorical_strategy is not None:
            raise ValueError(
                "Serialization is not supported for models trained with native categorical transforms."
            )
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
        return cls(
            _core.OptimizedModel.deserialize_compiled(serialized, physical_cores)
        )

    def predict(self, x: Any) -> Any:
        return self._inner.predict(x)

    def predict_proba(self, x: Any) -> Any:
        return self._inner.predict_proba(x)

    def to_ir_json(self, pretty: bool = False) -> str:
        if self.categorical_strategy is not None:
            raise ValueError(
                "IR export is not supported for models trained with native categorical transforms."
            )
        return self._inner.to_ir_json(pretty)

    def serialize(self, pretty: bool = False) -> str:
        if self.categorical_strategy is not None:
            raise ValueError(
                "Serialization is not supported for models trained with native categorical transforms."
            )
        return self._inner.serialize(pretty)

    def serialize_compiled(self) -> bytes:
        if self.categorical_strategy is not None:
            raise ValueError(
                "Compiled serialization is not supported for models trained with native categorical transforms."
            )
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
