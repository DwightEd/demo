from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _tuple(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class SourceConfig:
    """Location and cohort selector for one audited residual manifest."""

    manifest: Path
    hidden_dir: Path | None = None
    response_generator: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", Path(self.manifest).expanduser())
        if self.hidden_dir is not None:
            object.__setattr__(self, "hidden_dir", Path(self.hidden_dir).expanduser())


@dataclass(frozen=True)
class RunConfig:
    """Validated numerical protocol for one joint token-times-layer run."""

    offsets: tuple[int, ...] = (-2, -1, 0, 1)
    layers: str | tuple[int, ...] = "all"
    max_pairs: int = 0
    rank: int = 16
    folds: int = 5
    bootstrap: int = 2000
    seed: int = 17
    ridge_alpha: float = 1.0

    def __post_init__(self) -> None:
        offsets = _tuple(self.offsets)
        object.__setattr__(self, "offsets", offsets)
        if len(offsets) < 2 or any(right - left != 1 for left, right in zip(offsets, offsets[1:])):
            raise ValueError("offsets must contain at least two consecutive increasing integers")
        if self.layers != "all":
            values = self.layers.split(",") if isinstance(self.layers, str) else self.layers
            layers = _tuple(value for value in values if str(value).strip())
            object.__setattr__(self, "layers", layers)
            if len(layers) < 2:
                raise ValueError("at least two layers are required")
            if len(set(layers)) != len(layers):
                raise ValueError("layers must be unique")
        for name in ("rank", "folds", "bootstrap"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_pairs < 0:
            raise ValueError("max_pairs cannot be negative")
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha cannot be negative")
