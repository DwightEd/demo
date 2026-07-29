from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WriteGeometry:
    write_mass: float
    resultant_norm: float
    cancellation: float
    effective_sources: float
    dominance: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.write_mass,
                self.resultant_norm,
                self.cancellation,
                self.effective_sources,
                self.dominance,
            ],
            dtype=np.float64,
        )


def summarize_write_group(writes: np.ndarray) -> WriteGeometry:
    """Summarize path cancellation without assigning semantic meaning to it."""

    array = np.asarray(writes, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("writes must have shape [sources, rank]")
    if not np.isfinite(array).all():
        raise ValueError("writes must contain only finite values")
    norms = np.linalg.norm(array, axis=1)
    write_mass = float(norms.sum())
    resultant_norm = float(np.linalg.norm(array.sum(axis=0)))
    squared_mass = float(np.square(norms).sum())
    if write_mass <= 0.0:
        return WriteGeometry(0.0, 0.0, 0.0, 0.0, 0.0)
    cancellation = float(
        np.clip(1.0 - resultant_norm / write_mass, 0.0, 1.0)
    )
    effective_sources = (
        write_mass * write_mass / squared_mass if squared_mass > 0.0 else 0.0
    )
    return WriteGeometry(
        write_mass=write_mass,
        resultant_norm=resultant_norm,
        cancellation=cancellation,
        effective_sources=float(effective_sources),
        dominance=float(norms.max() / write_mass),
    )
