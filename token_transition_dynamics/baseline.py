from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _RobustStatistics:
    median: np.ndarray
    scale: np.ndarray
    used_position_fallback: bool


class CorrectOnlyBaseline:
    """Domain/layer/position norms fitted without error labels."""

    used_error_labels = False

    def __init__(self, *, position_bins: int, min_bin_samples: int) -> None:
        self.position_bins = position_bins
        self.min_bin_samples = min_bin_samples
        self.statistics_: dict[tuple[str, int, int], _RobustStatistics] | None = None

    @staticmethod
    def _statistics(values: np.ndarray, used_fallback: bool) -> _RobustStatistics:
        median = np.median(values, axis=0)
        mad = 1.4826 * np.median(np.abs(values - median), axis=0)
        floor = 1e-6 * np.maximum(np.abs(median), 1.0)
        return _RobustStatistics(median, np.maximum(mad, floor), used_fallback)

    def fit(
        self,
        features: np.ndarray,
        domains: np.ndarray,
        position_bins: np.ndarray,
    ) -> CorrectOnlyBaseline:
        values = np.asarray(features, dtype=np.float64)
        domain_values = np.asarray(domains, dtype=object)
        bins = np.asarray(position_bins, dtype=np.int64)
        if values.ndim != 3 or domain_values.shape != (values.shape[0],):
            raise ValueError("features and domains are not row aligned")
        if bins.shape != (values.shape[0],):
            raise ValueError("features and position bins are not row aligned")

        fitted: dict[tuple[str, int, int], _RobustStatistics] = {}
        for domain_value in np.unique(domain_values):
            domain = str(domain_value)
            domain_rows = domain_values == domain_value
            for layer in range(values.shape[1]):
                all_values = values[domain_rows, layer]
                if all_values.shape[0] < self.min_bin_samples:
                    raise ValueError(
                        f"domain {domain!r}, layer index {layer} has only "
                        f"{all_values.shape[0]} correct training windows"
                    )
                fallback = self._statistics(all_values, used_fallback=True)
                for position_bin in range(self.position_bins):
                    selected = domain_rows & (bins == position_bin)
                    fitted[(domain, layer, position_bin)] = (
                        self._statistics(values[selected, layer], used_fallback=False)
                        if np.sum(selected) >= self.min_bin_samples
                        else fallback
                    )
        self.statistics_ = fitted
        return self

    def standardize(
        self,
        features: np.ndarray,
        domains: np.ndarray,
        position_bins: np.ndarray,
    ) -> np.ndarray:
        if self.statistics_ is None:
            raise RuntimeError("fit must be called before standardize")
        values = np.asarray(features, dtype=np.float64)
        domain_values = np.asarray(domains, dtype=object)
        bins = np.asarray(position_bins, dtype=np.int64)
        standardized = np.empty_like(values)
        for row in range(values.shape[0]):
            for layer in range(values.shape[1]):
                statistics = self.statistics_[(str(domain_values[row]), layer, int(bins[row]))]
                standardized[row, layer] = (
                    values[row, layer] - statistics.median
                ) / statistics.scale
        return standardized

    @property
    def fallback_bin_count(self) -> int:
        if self.statistics_ is None:
            raise RuntimeError("fit must be called before reading fit diagnostics")
        return sum(value.used_position_fallback for value in self.statistics_.values())
