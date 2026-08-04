from __future__ import annotations

from typing import Sequence

import numpy as np

from .geometry import join_fourier, query_distribution_from_fourier


EPS = 1e-12


def softmax(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.maximum(weights.sum(axis=axis, keepdims=True), EPS)


def categorical_nll(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or target.shape != (len(values),):
        raise ValueError("probabilities and labels are misaligned")
    if np.any(target < 0) or np.any(target >= values.shape[1]):
        raise ValueError("categorical label is outside the probability support")
    return -np.log(np.clip(values[np.arange(len(values)), target], EPS, 1.0))


def fourier_query_distributions(
    predictions: np.ndarray,
    frequencies: np.ndarray,
    queries: np.ndarray,
    *,
    modulus: int,
) -> np.ndarray:
    coordinates = np.asarray(predictions, dtype=np.float64)
    query_values = np.asarray(queries, dtype=np.int64)
    if coordinates.ndim != 2 or query_values.ndim != 2:
        raise ValueError("predictions and queries must be matrices")
    if len(coordinates) != len(query_values):
        raise ValueError("predictions and queries are misaligned")
    result = np.empty((len(coordinates), int(modulus)), dtype=np.float64)
    for index, (row, query) in enumerate(zip(coordinates, query_values, strict=True)):
        phi = join_fourier(row, len(frequencies))
        result[index] = query_distribution_from_fourier(
            phi,
            frequencies,
            query,
            int(modulus),
        )
    return result


def evaluate_fourier_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    frequencies: np.ndarray,
    future_queries: np.ndarray,
    future_labels: np.ndarray,
    *,
    modulus: int,
) -> dict[str, float | np.ndarray]:
    predicted = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim != 2:
        raise ValueError("Fourier predictions and targets must be aligned matrices")
    distributions = fourier_query_distributions(
        predicted,
        frequencies,
        future_queries,
        modulus=modulus,
    )
    nll = categorical_nll(distributions, future_labels)
    squared_error = np.mean((predicted - truth) ** 2, axis=1)
    total_variance = float(np.mean((truth - truth.mean(axis=0)) ** 2))
    mean_mse = float(np.mean(squared_error))
    return {
        "fourier_mse": mean_mse,
        "fourier_r2": float(1.0 - mean_mse / max(total_variance, EPS)),
        "future_nll_nats": float(np.mean(nll)),
        "future_accuracy": float(
            np.mean(np.argmax(distributions, axis=1) == np.asarray(future_labels))
        ),
        "row_fourier_mse": squared_error.astype(np.float64),
        "row_future_nll_nats": nll.astype(np.float64),
        "future_probabilities": distributions.astype(np.float64),
    }


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("Jensen-Shannon inputs must be aligned vectors")
    p = p / np.maximum(p.sum(), EPS)
    q = q / np.maximum(q.sum(), EPS)
    midpoint = 0.5 * (p + q)
    kl_left = np.sum(p * (np.log(np.clip(p, EPS, None)) - np.log(midpoint)))
    kl_right = np.sum(q * (np.log(np.clip(q, EPS, None)) - np.log(midpoint)))
    return float(0.5 * (kl_left + kl_right))


def paired_alias_js(
    probabilities: np.ndarray,
    pair_ids: Sequence[int] | np.ndarray,
    branches: Sequence[int] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    groups = np.asarray(pair_ids, dtype=np.int64)
    branch_values = np.asarray(branches, dtype=np.int64)
    if len(values) != len(groups) or len(values) != len(branch_values):
        raise ValueError("alias probability arrays are misaligned")
    result: list[float] = []
    for group in np.unique(groups):
        local = np.flatnonzero(groups == group)
        left = local[branch_values[local] == 0]
        right = local[branch_values[local] == 1]
        if len(left) != 1 or len(right) != 1:
            raise ValueError("each alias pair must contain one row per branch")
        result.append(jensen_shannon_divergence(values[left[0]], values[right[0]]))
    return np.asarray(result, dtype=np.float64)


def paired_target_identification_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    pair_ids: Sequence[int] | np.ndarray,
    branches: Sequence[int] | np.ndarray,
) -> float:
    predicted = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    groups = np.asarray(pair_ids, dtype=np.int64)
    branch_values = np.asarray(branches, dtype=np.int64)
    if predicted.shape != truth.shape or predicted.ndim != 2:
        raise ValueError("paired predictions and targets must be aligned matrices")
    if len(predicted) != len(groups) or len(predicted) != len(branch_values):
        raise ValueError("paired target metadata is misaligned")
    outcomes: list[float] = []
    for group in np.unique(groups):
        local = np.flatnonzero(groups == group)
        left = local[branch_values[local] == 0]
        right = local[branch_values[local] == 1]
        if len(left) != 1 or len(right) != 1:
            raise ValueError("each target-identification pair needs two branches")
        for own, opposite in ((left[0], right[0]), (right[0], left[0])):
            own_error = float(np.mean((predicted[own] - truth[own]) ** 2))
            opposite_error = float(np.mean((predicted[own] - truth[opposite]) ** 2))
            outcomes.append(1.0 if own_error < opposite_error else (0.5 if own_error == opposite_error else 0.0))
    return float(np.mean(outcomes))


def cluster_bootstrap_mean(
    values: np.ndarray,
    groups: Sequence[int] | np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    scores = np.asarray(values, dtype=np.float64)
    group_values = np.asarray(groups, dtype=np.int64)
    if scores.shape != (len(group_values),):
        raise ValueError("bootstrap score must have one value per group row")
    finite = np.isfinite(scores)
    scores = scores[finite]
    group_values = group_values[finite]
    unique = np.unique(group_values)
    if len(unique) < 1:
        return {
            "point": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "groups": 0,
        }
    group_means = np.asarray(
        [scores[group_values == group].mean() for group in unique], dtype=np.float64
    )
    point = float(group_means.mean())
    if int(draws) <= 0 or len(group_means) == 1:
        low = high = point
    else:
        rng = np.random.default_rng(int(seed))
        sampled = rng.integers(0, len(group_means), size=(int(draws), len(group_means)))
        estimates = group_means[sampled].mean(axis=1)
        low, high = np.quantile(estimates, [0.025, 0.975]).tolist()
    return {
        "point": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "groups": int(len(group_means)),
    }


def conditional_usable_bits(
    baseline_nll: np.ndarray,
    augmented_nll: np.ndarray,
    pair_ids: Sequence[int] | np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    baseline = np.asarray(baseline_nll, dtype=np.float64)
    augmented = np.asarray(augmented_nll, dtype=np.float64)
    if baseline.shape != augmented.shape:
        raise ValueError("baseline and augmented NLL arrays are misaligned")
    row_bits = (baseline - augmented) / np.log(2.0)
    summary = cluster_bootstrap_mean(row_bits, pair_ids, draws=draws, seed=seed)
    summary["row_bits"] = row_bits
    return summary
