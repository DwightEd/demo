from __future__ import annotations

from typing import Mapping

import numpy as np


def _normalized_probability(values: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim < 1:
        raise ValueError("a probability tensor must have a hypothesis axis")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    mass = probability.sum(axis=-1, keepdims=True)
    if np.any(mass <= eps):
        raise ValueError("probability mass must be positive")
    return probability / mass


def mask_to_belief(mask: np.ndarray) -> np.ndarray:
    """Return the uniform categorical belief over a feasible support."""

    support = np.asarray(mask, dtype=bool)
    if support.ndim < 1:
        raise ValueError("a support mask must have a hypothesis axis")
    count = support.sum(axis=-1, keepdims=True)
    if np.any(count == 0):
        raise ValueError("a feasible support cannot be empty")
    return support.astype(np.float64) / count


def masked_belief_update(
    prior: np.ndarray,
    condition_mask: np.ndarray,
    *,
    epsilon_prior: float = 0.0,
) -> np.ndarray:
    """Apply an exact finite-space conditioning operator.

    ``epsilon_prior`` is zero for the reference process. A small positive value
    is useful only for deliberately mismatched null operators whose support may
    have zero mass under a decoded prior.
    """

    probability = _normalized_probability(prior)
    support = np.asarray(condition_mask, dtype=bool)
    if probability.shape != support.shape:
        raise ValueError(
            f"prior and condition mask must have the same shape, got "
            f"{probability.shape} and {support.shape}"
        )
    if epsilon_prior < 0.0:
        raise ValueError("epsilon_prior must be non-negative")
    weighted = (probability + float(epsilon_prior)) * support
    mass = weighted.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("condition has zero mass under the prior")
    return weighted / mass


def categorical_entropy(probability: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    p = _normalized_probability(probability, eps=eps)
    return -np.sum(np.where(p > 0.0, p * np.log(np.clip(p, eps, None)), 0.0), axis=-1)


def fisher_rao_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Categorical Fisher-Rao geodesic distance in radians."""

    p = _normalized_probability(first)
    q = _normalized_probability(second)
    if p.shape != q.shape:
        raise ValueError(f"belief shapes differ: {p.shape} versus {q.shape}")
    affinity = np.sum(np.sqrt(p * q), axis=-1)
    return 2.0 * np.arccos(np.clip(affinity, 0.0, 1.0))


def support_log_odds(
    probability: np.ndarray,
    support_mask: np.ndarray,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    p = _normalized_probability(probability)
    support = np.asarray(support_mask, dtype=bool)
    if p.shape != support.shape:
        raise ValueError("probability and support mask shapes differ")
    inside = np.sum(p * support, axis=-1)
    outside = np.sum(p * ~support, axis=-1)
    return np.log(np.clip(inside, eps, None)) - np.log(np.clip(outside, eps, None))


def transition_diagnostics(
    before: np.ndarray,
    after: np.ndarray,
    condition_mask: np.ndarray,
    *,
    epsilon_prior: float = 1e-12,
) -> Mapping[str, float | np.ndarray]:
    """Measure contraction, constraint support, and operator consistency."""

    p_before = _normalized_probability(before)
    p_after = _normalized_probability(after)
    support = np.asarray(condition_mask, dtype=bool)
    expected_after = masked_belief_update(
        p_before,
        support,
        epsilon_prior=epsilon_prior,
    )
    contraction = categorical_entropy(p_before) - categorical_entropy(p_after)
    margin_before = support_log_odds(p_before, support)
    margin_after = support_log_odds(p_after, support)
    support_gain = margin_after - margin_before
    unsupported = np.maximum(contraction, 0.0) * np.maximum(-support_gain, 0.0)
    diagnostics: dict[str, float | np.ndarray] = {
        "contraction": contraction,
        "support_margin_before": margin_before,
        "support_margin_after": margin_after,
        "support_gain": support_gain,
        "transport_residual": fisher_rao_distance(p_after, expected_after),
        "unsupported_contraction": unsupported,
    }
    if p_before.ndim == 1:
        return {name: float(np.asarray(value)) for name, value in diagnostics.items()}
    return diagnostics
