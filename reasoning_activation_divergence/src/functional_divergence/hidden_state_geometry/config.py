from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawFunctionalConfig:
    pca_dim: int = 16
    time_basis: int = 3
    layer_basis: int = 3
    positions_per_chain: int = 32
    l2: float = 1.0
    restarts: int = 3
    max_iter: int = 500
    null_repeats: int = 3

    def __post_init__(self) -> None:
        integer_fields = (
            "pca_dim",
            "time_basis",
            "layer_basis",
            "positions_per_chain",
            "restarts",
            "max_iter",
            "null_repeats",
        )
        if any(int(getattr(self, name)) < 1 for name in integer_fields):
            raise ValueError("dimensions, basis widths, restarts, and max_iter must be positive")
        if self.positions_per_chain < self.pca_dim:
            raise ValueError("positions_per_chain must be at least pca_dim")
        if self.l2 <= 0:
            raise ValueError("l2 must be positive")
