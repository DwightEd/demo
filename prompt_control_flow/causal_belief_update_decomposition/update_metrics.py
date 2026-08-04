from __future__ import annotations

import numpy as np


def _broadcast_targets(targets: np.ndarray, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float64)
    if values.shape[-1:] != shape[-1:]:
        raise ValueError(f"{name} width does not match component writes")
    while values.ndim < len(shape):
        values = np.expand_dims(values, axis=-2)
    try:
        return np.broadcast_to(values, shape)
    except ValueError as error:
        raise ValueError(f"{name} cannot be broadcast to component writes") from error


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1e-12,
    )


def component_update_metrics(
    writes: np.ndarray,
    true_updates: np.ndarray,
    opposite_updates: np.ndarray,
) -> dict[str, np.ndarray]:
    """Measure a residual write in an analytic belief-update coordinate chart.

    ``target_progress`` is signed target coverage: one denotes a write whose
    projection onto the required update has the required magnitude.  The
    relative target error additionally penalizes orthogonal and over-sized
    components.  Alignment margin distinguishes the true update from the
    matched opposite-branch update without pretending to measure magnitude.
    """

    values = np.asarray(writes, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] < 1:
        raise ValueError("component writes must end in a non-empty coordinate axis")
    if not np.isfinite(values).all():
        raise ValueError("component writes contain non-finite values")
    true = _broadcast_targets(true_updates, values.shape, name="true updates")
    opposite = _broadcast_targets(
        opposite_updates,
        values.shape,
        name="opposite updates",
    )
    if not np.isfinite(true).all() or not np.isfinite(opposite).all():
        raise ValueError("belief-update targets contain non-finite values")
    target_norm_sq = np.sum(true * true, axis=-1)
    if np.any(target_norm_sq <= 1e-12):
        raise ValueError("true belief updates must be non-zero")
    target_norm = np.sqrt(target_norm_sq)
    alignment_true = _cosine(values, true)
    alignment_opposite = _cosine(values, opposite)
    return {
        "alignment_true": alignment_true,
        "alignment_opposite": alignment_opposite,
        "alignment_margin": alignment_true - alignment_opposite,
        "target_progress": np.sum(values * true, axis=-1) / target_norm_sq,
        "target_error": np.linalg.norm(true - values, axis=-1) / target_norm,
    }


def component_reconstruction_error(
    attention_writes: np.ndarray,
    mlp_writes: np.ndarray,
    block_deltas: np.ndarray,
) -> np.ndarray:
    """Return relative error for ``block_delta = attention + MLP``."""

    attention = np.asarray(attention_writes, dtype=np.float64)
    mlp = np.asarray(mlp_writes, dtype=np.float64)
    block = np.asarray(block_deltas, dtype=np.float64)
    if attention.shape != mlp.shape or attention.shape != block.shape:
        raise ValueError("attention, MLP, and block writes must have identical shapes")
    if attention.ndim < 2:
        raise ValueError("component writes must end in a hidden-state axis")
    discrepancy = np.linalg.norm(block - attention - mlp, axis=-1)
    scale = np.maximum(np.linalg.norm(block, axis=-1), 1e-12)
    return discrepancy / scale
