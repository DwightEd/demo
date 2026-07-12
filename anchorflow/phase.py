from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np


EPS = 1e-9


def _as_feature_matrix(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("features must have shape [time, channels]")
    return x


def _robust_location_scale(history: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(history, axis=0)
    mad = np.nanmedian(np.abs(history - center), axis=0) * 1.4826
    std = np.nanstd(history, axis=0)
    scale = np.where(np.isfinite(mad) & (mad > ridge), mad, std)
    scale = np.where(np.isfinite(scale) & (scale > ridge), scale, float(ridge))
    center = np.where(np.isfinite(center), center, 0.0)
    return center, scale


def causal_change_scores(
    features: np.ndarray,
    *,
    min_history: int = 4,
    recent_window: int = 1,
    history_window: Optional[int] = None,
    ridge: float = 1e-3,
    clip: float = 30.0,
) -> Dict[str, np.ndarray]:
    """Boundary-free causal change scores over a natural token/window stream.

    At time ``t`` the reference distribution contains only rows strictly before
    the current recent window.  Appending future rows therefore cannot alter any
    previously emitted score.
    """
    X = _as_feature_matrix(features)
    T, F = X.shape
    score = np.full(T, np.nan)
    signed_mean = np.full(T, np.nan)
    max_channel = np.full(T, np.nan)
    direction_jump = np.full(T, np.nan)
    w = max(1, int(recent_window))
    mh = max(2, int(min_history))

    for t in range(T):
        start = max(0, t - w + 1)
        if start < mh:
            continue
        h0 = 0 if history_window is None else max(0, start - int(history_window))
        hist = X[h0:start]
        current_block = X[start : t + 1]
        good_current = np.isfinite(current_block)
        if not good_current.any():
            continue
        current = np.nanmean(current_block, axis=0)
        center, scale = _robust_location_scale(hist, float(ridge))
        valid = np.isfinite(current)
        if not valid.any():
            continue
        z = np.zeros(F, float)
        z[valid] = np.clip((current[valid] - center[valid]) / scale[valid], -clip, clip)
        score[t] = float(np.sqrt(np.mean(z[valid] ** 2)))
        signed_mean[t] = float(np.mean(z[valid]))
        max_channel[t] = float(np.max(np.abs(z[valid])))

        prev_start = max(0, start - w)
        prev = np.nanmean(X[prev_start:start], axis=0)
        if np.isfinite(prev).all() and np.isfinite(current).all():
            pn = np.linalg.norm(prev)
            cn = np.linalg.norm(current)
            if pn > EPS and cn > EPS:
                direction_jump[t] = float(1.0 - np.dot(prev, current) / (pn * cn))

    return {
        "change_score": score,
        "signed_change": signed_mean,
        "max_channel_change": max_channel,
        "direction_jump": direction_jump,
    }


def calibrate_chain_fpr_threshold(
    correct_chain_scores: Iterable[Sequence[float]],
    *,
    target_fpr: float = 0.05,
) -> float:
    """Calibrate on one maximum per correct chain, not pooled token positions."""
    if not 0.0 < float(target_fpr) < 1.0:
        raise ValueError("target_fpr must lie in (0, 1)")
    maxima = []
    for seq in correct_chain_scores:
        v = np.asarray(seq, float)
        v = v[np.isfinite(v)]
        if v.size:
            maxima.append(float(np.max(v)))
    if not maxima:
        return float("nan")
    # 'higher' makes the finite-sample threshold conservative under ties.
    try:
        q = float(np.quantile(maxima, 1.0 - target_fpr, method="higher"))
    except TypeError:  # NumPy < 1.22
        q = float(np.quantile(maxima, 1.0 - target_fpr, interpolation="higher"))
    # Events use >=. Move one representable value above the empirical quantile
    # so tied calibration maxima cannot make the achieved FPR anti-conservative.
    return float(np.nextafter(q, np.inf))


def causal_boundary_events(
    score: Sequence[float],
    threshold: float,
    *,
    refractory: int = 1,
) -> np.ndarray:
    """Online threshold crossings with no future-looking local-maximum test."""
    s = np.asarray(score, float)
    out = np.zeros(len(s), dtype=bool)
    last = -10**9
    gap = max(1, int(refractory))
    for t, value in enumerate(s):
        if np.isfinite(value) and value >= float(threshold) and t - last >= gap:
            out[t] = True
            last = t
    return out
