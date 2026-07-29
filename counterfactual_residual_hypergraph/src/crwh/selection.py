from __future__ import annotations

import numpy as np

from .contracts import (
    _validated_finite_real,
    _validated_int,
)


def select_review_candidates(
    *,
    unsupervised_anomaly_percentile: np.ndarray,
    semisupervised_probability: np.ndarray,
    unsupervised_threshold: float,
    semisupervised_threshold: float,
    budget: int,
) -> np.ndarray:
    """Select thresholded detector disagreements for optional LLM review."""

    unsupervised = np.asarray(
        unsupervised_anomaly_percentile, dtype=np.float64
    )
    semisupervised = np.asarray(semisupervised_probability, dtype=np.float64)
    if unsupervised.ndim != 1 or unsupervised.shape != semisupervised.shape:
        raise ValueError("score arrays must be aligned one-dimensional arrays")
    if not np.isfinite(unsupervised).all() or not np.isfinite(
        semisupervised
    ).all():
        raise ValueError("score arrays must be finite")
    if np.any((unsupervised < 0.0) | (unsupervised > 1.0)) or np.any(
        (semisupervised < 0.0) | (semisupervised > 1.0)
    ):
        raise ValueError("percentiles and probabilities must lie in [0, 1]")
    unsupervised_threshold = _validated_finite_real(
        unsupervised_threshold,
        name="unsupervised_threshold",
        minimum=0.0,
    )
    semisupervised_threshold = _validated_finite_real(
        semisupervised_threshold,
        name="semisupervised_threshold",
        minimum=0.0,
    )
    if unsupervised_threshold > 1.0 or semisupervised_threshold > 1.0:
        raise ValueError("review thresholds must lie in [0, 1]")
    budget = min(
        _validated_int(budget, name="budget", minimum=0),
        len(unsupervised),
    )
    unsupervised_decision = unsupervised >= unsupervised_threshold
    semisupervised_decision = semisupervised >= semisupervised_threshold
    candidates = np.flatnonzero(
        unsupervised_decision != semisupervised_decision
    )
    if budget == 0 or not len(candidates):
        return np.empty(0, dtype=np.int64)

    def confidence(values: np.ndarray, threshold: float) -> np.ndarray:
        positive_scale = max(1.0 - threshold, 1e-12)
        negative_scale = max(threshold, 1e-12)
        return np.where(
            values >= threshold,
            (values - threshold) / positive_scale,
            (threshold - values) / negative_scale,
        )

    priority = 0.5 * (
        confidence(unsupervised, unsupervised_threshold)
        + confidence(semisupervised, semisupervised_threshold)
    )
    order = np.argsort(-priority[candidates], kind="stable")
    return candidates[order[:budget]]
