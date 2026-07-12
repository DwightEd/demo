from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


EPS = 1e-9


def _row_unit(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, float)
    if a.ndim != 2:
        raise ValueError("expected a matrix [time, hidden_dim]")
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(norms, EPS)


def causal_window_mean(states: np.ndarray, window: int = 1) -> np.ndarray:
    """Causal rolling mean; row ``t`` never uses a state after ``t``."""
    x = np.asarray(states, float)
    if x.ndim != 2:
        raise ValueError("states must have shape [time, hidden_dim]")
    w = max(1, int(window))
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(len(x)):
        block = x[max(0, t - w + 1) : t + 1]
        good = np.isfinite(block).all(axis=1)
        if good.any():
            out[t] = np.mean(block[good], axis=0)
    return out


def compact_hidden_lookback(
    response_hidden: np.ndarray,
    anchor_vectors: np.ndarray,
    *,
    tau: float = 0.07,
    top_k: int = 3,
    window: int = 1,
) -> Dict[str, np.ndarray]:
    """Compact prompt lookback from response states to semantic anchor vectors.

    The result retains only interpretable online summaries plus the small
    ``[time, anchors]`` transport matrix.  No response-response future state is
    consulted, so the features can be computed during generation.
    """
    H = causal_window_mean(response_hidden, window=window)
    A = np.asarray(anchor_vectors, float)
    if A.ndim != 2:
        raise ValueError("anchor_vectors must have shape [anchors, hidden_dim]")
    if H.shape[1] != A.shape[1]:
        raise ValueError("response and anchor hidden dimensions differ")
    T, K = len(H), len(A)
    if K == 0:
        empty = np.full(T, np.nan)
        return {
            "similarity": np.empty((T, 0), dtype=float),
            "transport": np.empty((T, 0), dtype=float),
            "max_similarity": empty.copy(),
            "topk_similarity": empty.copy(),
            "anchor_entropy": empty.copy(),
            "anchor_coverage": empty.copy(),
            "detach": empty.copy(),
            "transport_shift": empty.copy(),
            "detach_jump": empty.copy(),
        }

    valid_h = np.isfinite(H).all(axis=1) & (np.linalg.norm(np.nan_to_num(H), axis=1) > EPS)
    valid_a = np.isfinite(A).all(axis=1) & (np.linalg.norm(np.nan_to_num(A), axis=1) > EPS)
    if not valid_a.all():
        raise ValueError("anchor_vectors contain a missing or zero vector")
    Au = _row_unit(A)
    sim = np.full((T, K), np.nan)
    P = np.full((T, K), np.nan)
    if valid_h.any():
        sim[valid_h] = _row_unit(H[valid_h]) @ Au.T
        logits = sim[valid_h] / max(float(tau), EPS)
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        P[valid_h] = weights / np.maximum(weights.sum(axis=1, keepdims=True), EPS)

    k = max(1, min(int(top_k), K))
    top = np.sort(sim, axis=1)[:, -k:]
    entropy = -np.sum(P * np.log(np.maximum(P, EPS)), axis=1)
    entropy_norm = entropy / max(float(np.log(K)), EPS) if K > 1 else np.zeros(T)
    coverage = np.exp(entropy) / float(K)
    max_sim = np.max(sim, axis=1)
    detach = 1.0 - max_sim
    shift = np.full(T, np.nan)
    detach_jump = np.full(T, np.nan)
    if T > 1:
        shift[1:] = np.linalg.norm(P[1:] - P[:-1], axis=1)
        detach_jump[1:] = detach[1:] - detach[:-1]
    return {
        "similarity": sim,
        "transport": P,
        "max_similarity": max_sim,
        "topk_similarity": np.mean(top, axis=1),
        "anchor_entropy": entropy_norm,
        "anchor_coverage": coverage,
        "detach": detach,
        "transport_shift": shift,
        "detach_jump": detach_jump,
    }


def compact_attention_lookback(
    attention: np.ndarray,
    prompt_mask: Sequence[bool],
    *,
    query_mask: Optional[Sequence[bool]] = None,
    top_k: int = 3,
) -> Dict[str, np.ndarray]:
    """Reduce attention to prompt mass, agreement, persistence, and churn.

    Accepted layouts are ``[query,key]``, ``[head,query,key]``, or
    ``[layer,head,query,key]``.  Top-k churn compares only the current query to
    the immediately previous query, so all summaries remain causal.
    """
    a = np.asarray(attention, float)
    if a.ndim == 2:
        a = a[None, None, :, :]
    elif a.ndim == 3:
        a = a[None, :, :, :]
    elif a.ndim != 4:
        raise ValueError("attention must be [Q,K], [H,Q,K], or [L,H,Q,K]")
    layers, heads, q, k = a.shape
    pm = np.asarray(prompt_mask, bool)
    if pm.shape != (k,):
        raise ValueError("prompt_mask must match the attention key dimension")
    if query_mask is not None:
        qm = np.asarray(query_mask, bool)
        if qm.shape != (q,):
            raise ValueError("query_mask must match the attention query dimension")
        a = a[:, :, qm, :]
        q = int(qm.sum())

    prompt_units = np.clip(a[:, :, :, pm], 0.0, None)
    prompt = np.nanmean(prompt_units, axis=(0, 1))
    if prompt.shape[1] == 0:
        missing = np.full(q, np.nan)
        return {
            "prompt_mass": np.zeros(q, dtype=float),
            "prompt_concentration": missing.copy(),
            "prompt_peak": missing.copy(),
            "head_agreement": missing.copy(),
            "layer_persistence": missing.copy(),
            "topk_churn": missing.copy(),
        }
    mass = np.sum(prompt, axis=1)
    cond = prompt / np.maximum(mass[:, None], EPS)
    ent = -np.sum(cond * np.log(np.maximum(cond, EPS)), axis=1)
    denom = np.log(max(2, prompt.shape[1]))
    concentration = 1.0 - ent / denom
    concentration[mass <= EPS] = np.nan
    peak = np.max(prompt, axis=1)
    peak[mass <= EPS] = np.nan

    units = prompt_units.reshape((layers * heads, q, prompt.shape[1]))
    head_agreement = np.full(q, np.nan)
    for t in range(q):
        V = units[:, t, :]
        good = np.isfinite(V).all(axis=1) & (np.linalg.norm(V, axis=1) > EPS)
        V = V[good]
        if len(V) < 2:
            continue
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        cosine = V @ V.T
        tri = cosine[np.triu_indices(len(V), k=1)]
        head_agreement[t] = float(np.clip(np.mean(tri), 0.0, 1.0))

    layer_persistence = np.full(q, np.nan)
    if layers > 1:
        LV = np.nanmean(prompt_units, axis=1)
        for t in range(q):
            vals = []
            for ell in range(1, layers):
                x, y = LV[ell - 1, t], LV[ell, t]
                nx, ny = np.linalg.norm(x), np.linalg.norm(y)
                if np.isfinite(x).all() and np.isfinite(y).all() and nx > EPS and ny > EPS:
                    vals.append(float(np.dot(x, y) / (nx * ny)))
            if vals:
                layer_persistence[t] = float(np.clip(np.mean(vals), 0.0, 1.0))

    churn = np.full(q, np.nan)
    kk = max(1, min(int(top_k), prompt.shape[1]))
    top = np.argsort(-prompt, axis=1)[:, :kk]
    for t in range(1, q):
        prev, cur = set(top[t - 1].tolist()), set(top[t].tolist())
        union = prev | cur
        churn[t] = 1.0 - len(prev & cur) / max(1, len(union))
    return {
        "prompt_mass": mass,
        "prompt_concentration": concentration,
        "prompt_peak": peak,
        "head_agreement": head_agreement,
        "layer_persistence": layer_persistence,
        "topk_churn": churn,
    }
