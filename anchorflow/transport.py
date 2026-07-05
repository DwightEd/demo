from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np

from .anchor_repr import AnchorBank
from .data import Trace, unit


EPS = 1e-9


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(x, float)
    z = z - np.nanmax(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.maximum(EPS, np.nansum(ez, axis=axis, keepdims=True))


def compute_transport(step_vectors: np.ndarray, anchor_vectors: np.ndarray, *, tau: float = 0.07) -> Dict[str, np.ndarray]:
    H = np.asarray([unit(v) for v in np.asarray(step_vectors, float)], float)
    A = np.asarray([unit(v) for v in np.asarray(anchor_vectors, float)], float)
    sim = H @ A.T
    P = softmax(sim / max(float(tau), EPS), axis=1)
    return {"similarity": sim, "transport": P, "mass": P}


def _kind_mask(anchors, kinds: Iterable[str]) -> np.ndarray:
    wanted = set(kinds)
    return np.asarray([a.kind in wanted for a in anchors], bool)


def _mass(P: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if P.size == 0 or not mask.any():
        return np.full(P.shape[0], np.nan)
    return P[:, mask].sum(axis=1)


def _entropy(P: np.ndarray) -> np.ndarray:
    K = max(1, P.shape[1])
    h = -np.sum(P * np.log(np.maximum(P, EPS)), axis=1)
    return h / max(np.log(K), EPS)


def _delta_l2(Z: np.ndarray) -> np.ndarray:
    out = np.full(Z.shape[0], np.nan)
    if Z.shape[0] > 1:
        out[1:] = np.linalg.norm(Z[1:] - Z[:-1], axis=1)
    return out


def _causal_z(x: np.ndarray, warmup: int = 2) -> np.ndarray:
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    for t in range(warmup, len(v)):
        hist = v[:t]
        hist = hist[np.isfinite(hist)]
        if len(hist) >= 2 and np.isfinite(v[t]):
            out[t] = (v[t] - hist.mean()) / (hist.std() + EPS)
    return out


def _leaky_cusum(z: np.ndarray, *, lam: float = 0.8, kref: float = 0.25) -> np.ndarray:
    out = np.zeros(len(z), float)
    c = 0.0
    for t, val in enumerate(np.asarray(z, float)):
        x = 0.0 if not np.isfinite(val) else float(val)
        c = max(0.0, lam * c + x - kref)
        out[t] = c
    return out


def transport_features(trace: Trace, bank: AnchorBank, *, prefix: str = "af", tau: float = 0.07) -> Dict[str, np.ndarray]:
    T = trace.n_steps
    if trace.stepvec is None or len(trace.stepvec) < T or bank.vectors.size == 0:
        return {f"{prefix}_available": np.zeros(T, float)}

    tr = compute_transport(trace.stepvec[:T], bank.vectors, tau=tau)
    sim = tr["similarity"]
    P = tr["transport"]

    goal = _kind_mask(bank.anchors, ["goal"])
    number = _kind_mask(bank.anchors, ["number"])
    constraint = _kind_mask(bank.anchors, ["constraint"])
    entity = _kind_mask(bank.anchors, ["entity"])
    core = goal | number | constraint

    entropy = _entropy(P)
    coverage = np.exp(entropy * np.log(max(1, P.shape[1]))) / max(1, P.shape[1])
    max_sim = np.nanmax(sim, axis=1)
    mean_top2 = np.nanmean(np.sort(sim, axis=1)[:, -min(2, sim.shape[1]) :], axis=1)
    jump = _delta_l2(P)
    detach = 1.0 - max_sim
    core_mass = _mass(P, core)
    core_detach = 1.0 - core_mass
    phase_score = np.nan_to_num(jump, nan=0.0) + np.maximum(0.0, _causal_z(detach))

    out = {
        f"{prefix}_available": np.ones(T, float),
        f"{prefix}_anchor_entropy": entropy,
        f"{prefix}_anchor_coverage": coverage,
        f"{prefix}_max_sim": max_sim,
        f"{prefix}_mean_top2_sim": mean_top2,
        f"{prefix}_detach": detach,
        f"{prefix}_goal_mass": _mass(P, goal),
        f"{prefix}_number_mass": _mass(P, number),
        f"{prefix}_constraint_mass": _mass(P, constraint),
        f"{prefix}_entity_mass": _mass(P, entity),
        f"{prefix}_core_mass": core_mass,
        f"{prefix}_core_detach": core_detach,
        f"{prefix}_transport_jump": jump,
        f"{prefix}_cz_detach": _causal_z(detach),
        f"{prefix}_cz_jump": _causal_z(jump),
        f"{prefix}_phase_score": phase_score,
        f"{prefix}_phase_cusum": _leaky_cusum(phase_score),
    }
    return out


def add_transport_features(
    traces: Sequence[Trace],
    banks: Sequence[AnchorBank],
    *,
    prefix: str = "af",
    tau: float = 0.07,
) -> List[str]:
    made = set()
    for trace, bank in zip(traces, banks):
        feats = transport_features(trace, bank, prefix=prefix, tau=tau)
        for name, values in feats.items():
            trace.features[name] = np.asarray(values, float)
            made.add(name)
    return sorted(made)
