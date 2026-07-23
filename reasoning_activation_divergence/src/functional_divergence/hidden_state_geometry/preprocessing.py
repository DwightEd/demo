from __future__ import annotations

import numpy as np


def group_balanced_weights(groups: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.mean()


class FiniteStandardizer:
    def __init__(self) -> None:
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray, sample_weight: np.ndarray) -> "FiniteStandardizer":
        x = np.asarray(values, dtype=np.float64)
        weight = np.asarray(sample_weight, dtype=np.float64)
        if x.ndim != 2 or weight.shape != (len(x),):
            raise ValueError("standardizer expects a matrix and one weight per row")
        center = np.zeros(x.shape[1], dtype=np.float64)
        for column in range(x.shape[1]):
            mask = np.isfinite(x[:, column])
            if not np.any(mask):
                continue
            order = np.argsort(x[mask, column], kind="stable")
            observed = x[mask, column][order]
            observed_weight = weight[mask][order]
            index = np.searchsorted(
                np.cumsum(observed_weight), observed_weight.sum() / 2.0, side="left"
            )
            center[column] = observed[min(int(index), len(observed) - 1)]
        filled = np.where(np.isfinite(x), x, center)
        scale = np.sqrt(np.average((filled - center) ** 2, axis=0, weights=weight))
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        self.center_, self.scale_ = center, scale
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("standardizer is not fitted")
        x = np.asarray(values, dtype=np.float64)
        return (np.where(np.isfinite(x), x, self.center_) - self.center_) / self.scale_
