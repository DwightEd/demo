"""Numerical definitions for token-causal residual direction dispersion.

All public functions are NumPy-only so the audit can run without loading a model.
The core object is a trailing-window field indexed by token, block, and scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DispersionConfig:
    """Configuration for a causal multi-scale dispersion field."""

    windows: tuple[int, ...] = (4, 8, 16, 32)
    min_tokens: int = 3
    decay: float = 0.0
    eps: float = 1e-8
    rank_stride: int = 1

    def __post_init__(self) -> None:
        windows = tuple(int(window) for window in self.windows)
        if not windows or any(window < 2 for window in windows):
            raise ValueError("windows must contain integers >= 2")
        if len(set(windows)) != len(windows):
            raise ValueError("windows must be unique")
        if self.min_tokens < 2:
            raise ValueError("min_tokens must be >= 2")
        if self.decay < 0:
            raise ValueError("decay must be non-negative")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.rank_stride < 1:
            raise ValueError("rank_stride must be >= 1")
        object.__setattr__(self, "windows", tuple(sorted(windows)))


def _as_finite_float(name: str, value: np.ndarray, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        array = array.astype(np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def block_writes_from_states(
    states: np.ndarray,
    layers: Iterable[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive block writes from states shaped ``[token, depth, hidden]``.

    Exact block writes require consecutive depth snapshots. Sparse layer selections
    are rejected rather than mislabeled as individual block updates.
    """

    states = _as_finite_float("states", states, ndim=3)
    depth_count = states.shape[1]
    if depth_count < 2:
        raise ValueError("states need at least two consecutive depth snapshots")
    layer_ids = (
        np.arange(depth_count, dtype=np.int64)
        if layers is None
        else np.asarray(tuple(layers), dtype=np.int64)
    )
    if layer_ids.shape != (depth_count,):
        raise ValueError("layers must have one id per state depth")
    if not np.all(np.diff(layer_ids) == 1):
        raise ValueError(
            "exact block writes require consecutive layer ids; sparse depths are unsupported"
        )
    return states[:, 1:, :] - states[:, :-1, :], layer_ids[1:]


def depth_deltas_from_states(
    states: np.ndarray,
    layers: Iterable[int] | np.ndarray,
    *,
    allow_sparse: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return adjacent-observation deltas and their source/target depths.

    With consecutive depths these are individual block writes. Sparse depths are
    accepted only via an explicit pilot opt-in and represent multi-block interval
    deltas such as ``h[10] - h[8]``.
    """

    states = _as_finite_float("states", states, ndim=3)
    layer_ids = np.asarray(tuple(layers), dtype=np.int64)
    if states.shape[1] < 2:
        raise ValueError("states need at least two depth snapshots")
    if layer_ids.shape != (states.shape[1],):
        raise ValueError("layers must have one id per state depth")
    gaps = np.diff(layer_ids)
    if np.any(gaps <= 0):
        raise ValueError("layer ids must be strictly increasing")
    if np.any(gaps != 1) and not allow_sparse:
        raise ValueError(
            "exact block writes require consecutive layer ids; use the explicit "
            "sparse-depth pilot mode for interval deltas"
        )
    return states[:, 1:, :] - states[:, :-1, :], layer_ids[:-1], layer_ids[1:]


def _trailing_weights(length: int, decay: float) -> np.ndarray:
    if decay == 0:
        return np.full(length, 1.0 / length, dtype=np.float64)
    ages = np.arange(length - 1, -1, -1, dtype=np.float64)
    weights = np.exp(-decay * ages)
    return weights / weights.sum()


def _window_statistics(
    vectors: np.ndarray,
    weights: np.ndarray,
    min_tokens: int,
    eps: float,
    compute_rank: bool = True,
) -> dict[str, float]:
    norms = np.linalg.norm(vectors, axis=-1)
    valid = norms > eps
    valid_count = int(valid.sum())
    unconditional_mean_norm = float(np.sum(weights * norms))
    if valid_count < min_tokens:
        return {
            "resultant": np.nan,
            "pair_dispersion": np.nan,
            "effective_rank": np.nan,
            "scatter_trace": np.nan,
            "identity_error": np.nan,
            "mean_write_norm": unconditional_mean_norm,
            "valid_tokens": float(valid_count),
        }

    units = vectors[valid] / norms[valid, None]
    local_weights = weights[valid]
    local_weights = local_weights / local_weights.sum()
    mean_direction = np.sum(local_weights[:, None] * units, axis=0)
    resultant = float(np.linalg.norm(mean_direction))

    self_weight = float(np.square(local_weights).sum())
    denominator = 1.0 - self_weight
    mean_pair_cosine = (
        (resultant * resultant - self_weight) / denominator
        if denominator > eps
        else np.nan
    )
    pair_dispersion = float(1.0 - np.clip(mean_pair_cosine, -1.0, 1.0))

    expected_trace = float(max(0.0, 1.0 - resultant * resultant))
    if compute_rank:
        centered = units - mean_direction[None, :]
        weighted_centered = np.sqrt(local_weights)[:, None] * centered
        # Nonzero covariance eigenvalues equal those of this small token-token Gram.
        gram = weighted_centered @ weighted_centered.T
        eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
        scatter_trace = float(eigenvalues.sum())
        if scatter_trace > eps:
            probabilities = eigenvalues[eigenvalues > eps] / scatter_trace
            effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        else:
            # Convention for a zero-scatter window; rank is not uniquely defined there.
            effective_rank = 1.0
        identity_error = float(abs(scatter_trace - expected_trace))
    else:
        scatter_trace = expected_trace
        effective_rank = np.nan
        identity_error = np.nan
    return {
        "resultant": resultant,
        "pair_dispersion": pair_dispersion,
        "effective_rank": effective_rank,
        "scatter_trace": scatter_trace,
        "identity_error": identity_error,
        "mean_write_norm": unconditional_mean_norm,
        "valid_tokens": float(valid_count),
    }


def compute_dispersion_field(
    writes: np.ndarray,
    config: DispersionConfig | None = None,
) -> dict[str, np.ndarray]:
    """Compute trailing-window statistics with axes ``[token, block, scale]``.

    At token ``t`` only writes at positions ``<= t`` are used. The returned value
    is therefore a post-token diagnostic and can be shifted by one position when
    used to predict the next token.
    """

    writes = _as_finite_float("writes", writes, ndim=3)
    config = config or DispersionConfig()
    token_count, block_count, _ = writes.shape
    shape = (token_count, block_count, len(config.windows))
    names = (
        "resultant",
        "pair_dispersion",
        "effective_rank",
        "scatter_trace",
        "identity_error",
        "mean_write_norm",
        "valid_tokens",
    )
    output = {name: np.full(shape, np.nan, dtype=np.float64) for name in names}

    for scale_index, window in enumerate(config.windows):
        for token_index in range(token_count):
            start = max(0, token_index - window + 1)
            length = token_index - start + 1
            if length < config.min_tokens:
                continue
            weights = _trailing_weights(length, config.decay)
            for block_index in range(block_count):
                stats = _window_statistics(
                    writes[start : token_index + 1, block_index, :],
                    weights,
                    config.min_tokens,
                    config.eps,
                    compute_rank=(token_index % config.rank_stride == 0),
                )
                for name, value in stats.items():
                    output[name][token_index, block_index, scale_index] = value

    pair_dispersion = output["pair_dispersion"]
    temporal_jump = np.full_like(pair_dispersion, np.nan)
    temporal_jump[1:] = pair_dispersion[1:] - pair_dispersion[:-1]
    output["temporal_jump"] = temporal_jump
    output["windows"] = np.asarray(config.windows, dtype=np.int64)
    return output


def residual_arc_length(writes: np.ndarray, eps: float = 1e-8) -> dict[str, np.ndarray]:
    """Return raw causal arc length and a retrospective normalized phase."""

    writes = _as_finite_float("writes", writes, ndim=3)
    per_token = np.mean(np.linalg.norm(writes, axis=-1), axis=1)
    cumulative = np.cumsum(per_token)
    total = float(cumulative[-1]) if cumulative.size else 0.0
    phase = cumulative / total if total > eps else np.zeros_like(cumulative)
    return {
        "per_token_write_length": per_token,
        "cumulative_arc_length": cumulative,
        "retrospective_arc_phase": phase,
    }


def component_conflict(
    attention_writes: np.ndarray,
    mlp_writes: np.ndarray,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Measure antagonism and cancellation between attention and MLP writes."""

    attention = _as_finite_float("attention_writes", attention_writes, ndim=3)
    mlp = _as_finite_float("mlp_writes", mlp_writes, ndim=3)
    if attention.shape != mlp.shape:
        raise ValueError("attention_writes and mlp_writes must have equal shapes")
    attention_norm = np.linalg.norm(attention, axis=-1)
    mlp_norm = np.linalg.norm(mlp, axis=-1)
    denominator = attention_norm * mlp_norm
    cosine = np.full(attention_norm.shape, np.nan, dtype=np.float64)
    valid = denominator > eps
    cosine[valid] = np.sum(attention * mlp, axis=-1)[valid] / denominator[valid]
    antagonism = np.maximum(0.0, -cosine)
    cancellation = 1.0 - (
        np.linalg.norm(attention + mlp, axis=-1)
        / np.maximum(attention_norm + mlp_norm, eps)
    )
    cancellation[~valid] = np.nan
    return {
        "cosine": cosine,
        "antagonism": antagonism,
        "cancellation": cancellation,
    }
