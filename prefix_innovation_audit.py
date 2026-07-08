#!/usr/bin/env python3
"""Prefix-innovation audit for first-error reasoning steps.

The unit of analysis is a transition, not an isolated step:

    previous prefix  ->  current step

For each transition t-1 -> t, the script asks:

1. How much of the current step's token-cloud energy lies outside the previous
   step / prefix subspace?
2. How much of the mean-state displacement is normal to those subspaces?
3. Does hidden alignment to the question vector drop at the transition?
4. Which hidden dimensions are over-activated relative to the chain prefix?

This is the direct "did the reasoning leave the previous trajectory?" test.  It
is intentionally separate from within-step kappa/spread and from coarse
question/prefix anchor posteriors.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from audit_utils import finite_json
from multisample_temporal_rupture_audit import descriptive, within_pair_auroc
from premise_constraint_audit import bootstrap_within_increment
from second_moment_dynamics_audit import oof_scores
from token_stream_geometry_audit import chain_lengths, load_token_matrix, source_info


EPS = 1e-12


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


@dataclass
class TransitionRow:
    chain_idx: int
    problem_id: int
    step_idx: int
    prev_step_idx: int
    gold_error_step: int
    phase: str
    y_first_error: int
    features: Dict[str, float]


class DimAccumulator:
    """Online means for prefix-relative activation z-vectors."""

    def __init__(self) -> None:
        self.err_n = 0
        self.ctrl_n = 0
        self.err_sum: Optional[np.ndarray] = None
        self.ctrl_sum: Optional[np.ndarray] = None
        self.err_abs_sum: Optional[np.ndarray] = None
        self.ctrl_abs_sum: Optional[np.ndarray] = None

    def add(self, z: np.ndarray, y: int) -> None:
        v = np.asarray(z, dtype=np.float64)
        if v.ndim != 1 or v.size == 0 or not np.isfinite(v).any():
            return
        v = np.where(np.isfinite(v), v, 0.0)
        if self.err_sum is None:
            self.err_sum = np.zeros_like(v)
            self.ctrl_sum = np.zeros_like(v)
            self.err_abs_sum = np.zeros_like(v)
            self.ctrl_abs_sum = np.zeros_like(v)
        if y == 1:
            self.err_n += 1
            self.err_sum += v
            self.err_abs_sum += np.abs(v)
        elif y == 0:
            self.ctrl_n += 1
            self.ctrl_sum += v
            self.ctrl_abs_sum += np.abs(v)

    def summary(self, *, top_k: int) -> Dict[str, Any]:
        if self.err_sum is None or self.err_n == 0 or self.ctrl_n == 0:
            return {"ok": False, "err_n": int(self.err_n), "ctrl_n": int(self.ctrl_n), "top_positive": [], "top_abs": []}
        err = self.err_sum / max(1, self.err_n)
        ctrl = self.ctrl_sum / max(1, self.ctrl_n)
        err_abs = self.err_abs_sum / max(1, self.err_n)
        ctrl_abs = self.ctrl_abs_sum / max(1, self.ctrl_n)
        delta = err - ctrl
        abs_delta = err_abs - ctrl_abs

        def pack(order: np.ndarray, key: str) -> List[Dict[str, float]]:
            out = []
            for j in order[:top_k]:
                out.append(
                    {
                        "dim": int(j),
                        "err_mean_z": float(err[j]),
                        "ctrl_mean_z": float(ctrl[j]),
                        "delta_mean_z": float(delta[j]),
                        "err_mean_abs_z": float(err_abs[j]),
                        "ctrl_mean_abs_z": float(ctrl_abs[j]),
                        "delta_abs_z": float(abs_delta[j]),
                        "rank_key": key,
                    }
                )
            return out

        pos_order = np.argsort(delta)[::-1]
        abs_order = np.argsort(abs_delta)[::-1]
        return {
            "ok": True,
            "err_n": int(self.err_n),
            "ctrl_n": int(self.ctrl_n),
            "top_positive": pack(pos_order, "delta_mean_z"),
            "top_abs": pack(abs_order, "delta_abs_z"),
        }


def scalar_str(x: Any) -> str:
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(x)


def auroc(score: Iterable[float], y: Iterable[int]) -> float:
    s = np.asarray(score, dtype=np.float64)
    yy = np.asarray(y, dtype=int)
    m = np.isfinite(s)
    s, yy = s[m], yy[m]
    p = int((yy == 1).sum())
    n = int((yy == 0).sum())
    if p == 0 or n == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[yy == 1].sum() - p * (p + 1) / 2.0) / (p * n))


def safe_mean(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def safe_std(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1)) if a.size > 1 else (0.0 if a.size else float("nan"))


def normalize_rows(H: np.ndarray, *, center: Optional[np.ndarray] = None, unitize: bool = True) -> np.ndarray:
    X = np.asarray(H, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]
    if X.shape[0] == 0:
        return np.empty((0, H.shape[1] if np.asarray(H).ndim == 2 else 0), dtype=np.float64)
    if center is not None and np.asarray(center).shape[-1] == X.shape[1]:
        X = X - np.asarray(center, dtype=np.float64)[None, :]
    if unitize:
        norms = np.linalg.norm(X, axis=1)
        ok = norms > EPS
        X = X[ok] / np.maximum(norms[ok, None], EPS)
    return X


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        return float("nan")
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx <= EPS or ny <= EPS:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def basis_from_rows(A: np.ndarray, *, rank: int) -> np.ndarray:
    X = np.asarray(A, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    if X.shape[0] == 1:
        v = X[0]
        n = float(np.linalg.norm(v))
        return (v / n)[:, None] if n > EPS else np.empty((X.shape[1], 0), dtype=np.float64)
    X = X - np.mean(X, axis=0, keepdims=True) if X.shape[0] > 2 else X
    try:
        _u, _s, vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.empty((X.shape[1], 0), dtype=np.float64)
    k = int(min(rank, vt.shape[0]))
    V = vt[:k].T if k > 0 else np.empty((X.shape[1], 0), dtype=np.float64)
    n = np.linalg.norm(V, axis=0)
    ok = n > EPS
    return V[:, ok] / np.maximum(n[ok][None, :], EPS)


def projection_fraction(X: np.ndarray, V: np.ndarray) -> float:
    if X.ndim != 2 or V.ndim != 2 or X.shape[0] == 0 or V.shape[1] == 0 or X.shape[1] != V.shape[0]:
        return float("nan")
    denom = float(np.sum(X * X))
    if denom <= EPS:
        return float("nan")
    P = X @ V
    return float(np.sum(P * P) / denom)


def off_subspace_energy(X: np.ndarray, V: np.ndarray) -> float:
    frac = projection_fraction(X, V)
    return float(1.0 - frac) if np.isfinite(frac) else float("nan")


def exp_weights(n: int, beta: float) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    if n == 1 or abs(beta) <= EPS:
        return np.ones(n, dtype=np.float64) / n
    pos = np.linspace(0.0, 1.0, n)
    z = beta * pos
    z -= float(z.max())
    w = np.exp(z)
    return w / max(float(w.sum()), EPS)


def pooled_mean(H: np.ndarray, *, beta: float) -> np.ndarray:
    X = np.asarray(H, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    w = exp_weights(X.shape[0], beta)
    return w @ X


def prefix_z_vector(current_step: np.ndarray, prefix: np.ndarray, *, beta: float, std_floor_frac: float) -> np.ndarray:
    cur = pooled_mean(current_step, beta=beta)
    P = np.asarray(prefix, dtype=np.float64)
    if cur.size == 0 or P.ndim != 2 or P.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    mu = np.nanmean(P, axis=0)
    sd = np.nanstd(P, axis=0)
    finite = sd[np.isfinite(sd) & (sd > EPS)]
    floor = float(np.median(finite) * std_floor_frac) if finite.size else 1.0
    denom = np.maximum(sd, max(floor, EPS))
    return (cur - mu) / denom


def z_features(z: np.ndarray, *, top_k: int, z_thresh: float) -> Dict[str, float]:
    v = np.asarray(z, dtype=np.float64)
    v = v[np.isfinite(v)]
    out = {
        "prefix_z_max": float("nan"),
        "prefix_z_min": float("nan"),
        "prefix_z_abs_max": float("nan"),
        "prefix_z_top_abs_mean": float("nan"),
        "prefix_z_top_pos_mean": float("nan"),
        "prefix_z_l2": float("nan"),
        "prefix_z_n_gt": float("nan"),
    }
    if v.size == 0:
        return out
    k = int(min(max(1, top_k), v.size))
    abs_sorted = np.sort(np.abs(v))[::-1]
    pos_sorted = np.sort(v)[::-1]
    out["prefix_z_max"] = float(np.max(v))
    out["prefix_z_min"] = float(np.min(v))
    out["prefix_z_abs_max"] = float(abs_sorted[0])
    out["prefix_z_top_abs_mean"] = float(np.mean(abs_sorted[:k]))
    out["prefix_z_top_pos_mean"] = float(np.mean(pos_sorted[:k]))
    out["prefix_z_l2"] = float(np.linalg.norm(v) / math.sqrt(max(1, v.size)))
    out["prefix_z_n_gt"] = float(np.sum(v > z_thresh))
    return out


def slice_steps(H: np.ndarray, lengths: np.ndarray) -> List[np.ndarray]:
    steps: List[np.ndarray] = []
    cur = 0
    n = int(H.shape[0])
    for ln in np.asarray(lengths, dtype=int):
        hi = min(n, cur + max(0, int(ln)))
        if hi > cur:
            steps.append(np.asarray(H[cur:hi], dtype=np.float64))
        else:
            steps.append(np.empty((0, H.shape[1]), dtype=np.float64))
        cur = hi
    return steps


def select_qvec(data: np.lib.npyio.NpzFile, idx: int, layer_used: int, dim: int) -> Optional[np.ndarray]:
    if "qvec" not in data.files:
        return None
    q = data["qvec"]
    try:
        qi = np.asarray(q[idx], dtype=np.float64)
    except Exception:
        qi = np.asarray(q, dtype=np.float64)
    if qi.ndim == 1 and qi.size == dim:
        return qi
    if qi.ndim == 2 and qi.shape[1] == dim:
        layers = [int(x) for x in data["sv_layers"]] if "sv_layers" in data.files else []
        if layer_used in layers:
            return qi[layers.index(layer_used)]
        return qi[min(qi.shape[0] - 1, qi.shape[0] // 2)]
    return None


def step_entropy_trace(data: np.lib.npyio.NpzFile, idx: int, lengths: np.ndarray) -> np.ndarray:
    T = len(lengths)
    out = np.full(T, np.nan, dtype=np.float64)
    if "sv_out_entropy" in data.files:
        try:
            v = np.asarray(data["sv_out_entropy"][idx], dtype=np.float64).reshape(-1)
            out[: min(T, v.size)] = v[:T]
            return out
        except Exception:
            pass
    if "tok_U_D" in data.files:
        try:
            tok = np.asarray(data["tok_U_D"][idx], dtype=np.float64).reshape(-1)
            cur = 0
            for t, ln in enumerate(lengths):
                hi = min(tok.size, cur + int(ln))
                if hi > cur:
                    out[t] = float(np.nanmean(tok[cur:hi]))
                cur = hi
        except Exception:
            pass
    return out


def phase_for(gold: int, t: int) -> str:
    if gold < 0:
        return "correct_chain"
    if t < gold:
        return "pre_error"
    if t == gold:
        return "first_error"
    return "post_error"


def y_for_phase(phase: str, control_pool: str) -> int:
    if phase == "first_error":
        return 1
    if phase == "post_error":
        return -1
    if control_pool == "correct_chain" and phase != "correct_chain":
        return -1
    if control_pool == "pre_error" and phase != "pre_error":
        return -1
    return 0


def transition_features(
    prev_raw: np.ndarray,
    current_raw: np.ndarray,
    prefix_raw: np.ndarray,
    *,
    qvec: Optional[np.ndarray],
    chain_center: Optional[np.ndarray],
    rank: int,
    beta: float,
    unitize: bool,
    z_top_k: int,
    z_thresh: float,
    std_floor_frac: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    prev = normalize_rows(prev_raw, center=chain_center, unitize=unitize)
    cur = normalize_rows(current_raw, center=chain_center, unitize=unitize)
    prefix = normalize_rows(prefix_raw, center=chain_center, unitize=unitize)
    Vprev = basis_from_rows(prev, rank=rank)
    Vprefix = basis_from_rows(prefix, rank=rank)

    mu_prev = pooled_mean(prev, beta=beta)
    mu_cur = pooled_mean(cur, beta=beta)
    delta = mu_cur - mu_prev if mu_prev.size == mu_cur.size and mu_cur.size else np.empty(0)
    dmat = delta[None, :] if delta.size else np.empty((0, 0))
    off_prev = off_subspace_energy(cur, Vprev)
    off_prefix = off_subspace_energy(cur, Vprefix)
    delta_off_prev = off_subspace_energy(dmat, Vprev)
    delta_off_prefix = off_subspace_energy(dmat, Vprefix)

    z = prefix_z_vector(current_raw, prefix_raw, beta=beta, std_floor_frac=std_floor_frac)
    feats = {
        "off_prev_subspace": off_prev,
        "off_prefix_subspace": off_prefix,
        "in_prev_subspace": float(1.0 - off_prev) if np.isfinite(off_prev) else float("nan"),
        "in_prefix_subspace": float(1.0 - off_prefix) if np.isfinite(off_prefix) else float("nan"),
        "innovation_norm": float(np.linalg.norm(delta)) if delta.size else float("nan"),
        "innovation_off_prev": delta_off_prev,
        "innovation_off_prefix": delta_off_prefix,
        "innovation_in_prev": float(1.0 - delta_off_prev) if np.isfinite(delta_off_prev) else float("nan"),
        "innovation_in_prefix": float(1.0 - delta_off_prefix) if np.isfinite(delta_off_prefix) else float("nan"),
        "subspace_rank_prev": float(Vprev.shape[1]),
        "subspace_rank_prefix": float(Vprefix.shape[1]),
        "mean_step_cos_prev": cosine(mu_cur, mu_prev),
    }
    feats.update(z_features(z, top_k=z_top_k, z_thresh=z_thresh))
    if qvec is not None and mu_cur.size == np.asarray(qvec).size:
        feats["q_align_current"] = cosine(mu_cur, qvec)
        feats["q_align_prev"] = cosine(mu_prev, qvec)
        feats["q_align_drop"] = feats["q_align_prev"] - feats["q_align_current"]
        feats["innovation_q_cos"] = cosine(delta, qvec) if delta.size else float("nan")
    else:
        feats["q_align_current"] = float("nan")
        feats["q_align_prev"] = float("nan")
        feats["q_align_drop"] = float("nan")
        feats["innovation_q_cos"] = float("nan")
    return feats, z


def build_rows(path: str, args: argparse.Namespace) -> Tuple[List[TransitionRow], Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    if "gold_error_step" not in data.files:
        raise SystemExit("prefix_innovation_audit requires gold_error_step")
    source, layer_i, layer_used = source_info(data, path, args)
    gold = data["gold_error_step"].astype(int)
    problem_ids = data["problem_ids"].astype(int) if "problem_ids" in data.files else np.arange(len(gold))
    N = len(gold) if args.max_chains <= 0 else min(len(gold), int(args.max_chains))
    rows: List[TransitionRow] = []
    dim_acc = DimAccumulator()
    skipped = {"missing_hidden": 0, "too_few_steps": 0, "first_error_at_step0": 0}

    iterator = range(N)
    if not args.no_progress:
        iterator = tqdm(iterator, desc="prefix-innovation rows", unit="chain")
    for idx in iterator:
        H = load_token_matrix(data, path, args, idx=idx, source=source, layer_i=layer_i)
        if H is None or np.asarray(H).ndim != 2 or np.asarray(H).shape[0] == 0:
            skipped["missing_hidden"] += 1
            continue
        H = np.asarray(H, dtype=np.float64)
        lengths, _ranges = chain_lengths(data, idx, H.shape[0], source)
        if lengths.size < 2:
            skipped["too_few_steps"] += 1
            continue
        if int(gold[idx]) == 0:
            skipped["first_error_at_step0"] += 1
        step_mats = slice_steps(H, lengths)
        T = len(step_mats)
        chain_center = np.mean(H, axis=0) if args.center_chain else None
        qvec = select_qvec(data, idx, layer_used, H.shape[1])
        entropy_trace = step_entropy_trace(data, idx, lengths)
        for t in range(1, T):
            prefix_raw = np.concatenate(step_mats[:t], axis=0)
            feats, z = transition_features(
                step_mats[t - 1],
                step_mats[t],
                prefix_raw,
                qvec=qvec,
                chain_center=chain_center,
                rank=args.rank,
                beta=args.beta,
                unitize=not args.raw_hidden,
                z_top_k=args.z_top_k,
                z_thresh=args.z_thresh,
                std_floor_frac=args.std_floor_frac,
            )
            phase = phase_for(int(gold[idx]), t)
            y = y_for_phase(phase, args.control_pool)
            feats["n_tok_current"] = float(step_mats[t].shape[0])
            feats["n_tok_prefix"] = float(prefix_raw.shape[0])
            feats["logN_current"] = float(math.log1p(max(0, step_mats[t].shape[0])))
            feats["logN_prefix"] = float(math.log1p(max(0, prefix_raw.shape[0])))
            feats["pos"] = float(t / max(1, T - 1))
            feats["entropy_current"] = float(entropy_trace[t]) if t < entropy_trace.size else float("nan")
            feats["risk_off_prefix"] = feats["off_prefix_subspace"]
            feats["risk_off_prev"] = feats["off_prev_subspace"]
            feats["risk_innovation_off_prefix"] = feats["innovation_off_prefix"]
            feats["risk_q_drop"] = feats["q_align_drop"]
            feats["risk_prefix_z"] = feats["prefix_z_top_abs_mean"]
            feats["risk_combined_off_z"] = (
                feats["off_prefix_subspace"] * feats["prefix_z_top_abs_mean"]
                if np.isfinite(feats["off_prefix_subspace"]) and np.isfinite(feats["prefix_z_top_abs_mean"])
                else float("nan")
            )
            if y >= 0:
                dim_acc.add(z, y)
            rows.append(
                TransitionRow(
                    chain_idx=int(idx),
                    problem_id=int(problem_ids[idx]),
                    step_idx=int(t),
                    prev_step_idx=int(t - 1),
                    gold_error_step=int(gold[idx]),
                    phase=phase,
                    y_first_error=int(y),
                    features=feats,
                )
            )

    meta = {
        "npz": path,
        "source": source,
        "layer": int(layer_used),
        "layer_index": int(layer_i),
        "n_chains_seen": int(N),
        "n_rows": int(len(rows)),
        "skipped": skipped,
        "rank": int(args.rank),
        "projection": "raw_hidden" if args.raw_hidden else "centered_unit_hidden",
        "control_pool": args.control_pool,
        "dim_activation": dim_acc.summary(top_k=args.dim_top_k),
        "has_qvec": bool("qvec" in data.files),
    }
    return rows, meta


def feature_array(rows: Sequence[TransitionRow], name: str) -> np.ndarray:
    return np.asarray([r.features.get(name, float("nan")) for r in rows], dtype=np.float64)


def row_arrays(rows: Sequence[TransitionRow]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray([r.y_first_error for r in rows], dtype=int)
    problem_ids = np.asarray([r.problem_id for r in rows], dtype=int)
    chain_ids = np.asarray([r.chain_idx for r in rows], dtype=int)
    phases = np.asarray([r.phase for r in rows], dtype=object)
    return y, problem_ids, chain_ids, phases


def problem_groups(problem_ids: np.ndarray, y: np.ndarray, mask: np.ndarray, *, min_per_class: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for p in np.unique(problem_ids[mask]):
        idx = np.where(mask & (problem_ids == p))[0]
        if np.sum(y[idx] == 1) >= min_per_class and np.sum(y[idx] == 0) >= min_per_class:
            out.append(idx)
    return out


def eval_score(name: str, score: np.ndarray, y: np.ndarray, groups: Sequence[np.ndarray], mask: np.ndarray) -> Dict[str, Any]:
    s = np.asarray(score, dtype=np.float64)
    m = mask & np.isfinite(s)
    raw = auroc(s[m], y[m]) if m.any() else float("nan")
    sign = 1.0 if (not np.isfinite(raw) or raw >= 0.5) else -1.0
    ss = sign * s
    within, pairs = within_pair_auroc(groups, ss, y)
    return {
        "score": name,
        "n": int(m.sum()),
        "cross_auroc_error_high": float(auroc(ss[m], y[m])) if m.any() else float("nan"),
        "raw_cross_auroc": float(raw),
        "sign": float(sign),
        "within_pair_auroc_error_high": float(within),
        "within_pairs": int(pairs),
        "err_median": float(np.nanmedian(ss[m & (y == 1)])) if np.any(m & (y == 1)) else float("nan"),
        "ctrl_median": float(np.nanmedian(ss[m & (y == 0)])) if np.any(m & (y == 0)) else float("nan"),
    }


def design(rows: Sequence[TransitionRow], names: Sequence[str]) -> np.ndarray:
    X = np.full((len(rows), len(names)), np.nan, dtype=np.float64)
    for j, name in enumerate(names):
        X[:, j] = feature_array(rows, name)
    return X


def finite_names(rows: Sequence[TransitionRow], candidates: Sequence[str], mask: np.ndarray, *, min_coverage: float) -> List[str]:
    out = []
    for name in candidates:
        x = feature_array(rows, name)
        cov = float(np.mean(np.isfinite(x[mask]))) if mask.any() else 0.0
        if cov >= min_coverage:
            out.append(name)
    return out


def transition_summary(rows: Sequence[TransitionRow], names: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in names:
        err = [r.features.get(name, float("nan")) for r in rows if r.phase == "first_error"]
        pre = [r.features.get(name, float("nan")) for r in rows if r.phase == "pre_error"]
        ctrl = [r.features.get(name, float("nan")) for r in rows if r.phase == "correct_chain"]
        out[name] = {
            "first_error": descriptive(err),
            "pre_error": descriptive(pre),
            "correct_chain": descriptive(ctrl),
            "first_minus_correct_mean": safe_mean(err) - safe_mean(ctrl),
            "first_minus_pre_mean": safe_mean(err) - safe_mean(pre),
        }
    return out


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    rows, meta = build_rows(path, args)
    y, problem_ids, chain_ids, phases = row_arrays(rows)
    eval_mask = y >= 0
    groups = problem_groups(problem_ids, y, eval_mask, min_per_class=args.min_per_class)
    transition_candidates = [
        "off_prefix_subspace",
        "off_prev_subspace",
        "innovation_off_prefix",
        "innovation_off_prev",
        "innovation_norm",
        "mean_step_cos_prev",
        "q_align_current",
        "q_align_drop",
        "innovation_q_cos",
        "prefix_z_abs_max",
        "prefix_z_top_abs_mean",
        "prefix_z_top_pos_mean",
        "prefix_z_l2",
        "prefix_z_n_gt",
        "risk_combined_off_z",
    ]
    baseline_candidates = ["logN_current", "logN_prefix", "pos", "entropy_current", "n_tok_current", "n_tok_prefix"]
    transition_names = finite_names(rows, transition_candidates, eval_mask, min_coverage=args.min_feature_coverage)
    baseline_names = finite_names(rows, baseline_candidates, eval_mask, min_coverage=args.min_feature_coverage)

    single_rows: List[Dict[str, Any]] = []
    single_scores: Dict[str, np.ndarray] = {}
    for name in baseline_names + transition_names:
        vals = feature_array(rows, name)
        ev = eval_score(name, vals, y, groups, eval_mask)
        single_rows.append(ev)
        single_scores[name] = ev["sign"] * vals
    single_rows.sort(key=lambda r: np.nan_to_num(r["within_pair_auroc_error_high"], nan=-1.0), reverse=True)

    model_rows: List[Dict[str, Any]] = []
    model_scores: Dict[str, np.ndarray] = {}
    if eval_mask.sum() >= 20 and len(np.unique(y[eval_mask])) == 2:
        idx = np.where(eval_mask)[0]
        rr = [rows[i] for i in idx]
        yy = y[idx]
        gg = problem_ids[idx]
        if baseline_names:
            sb = oof_scores(design(rr, baseline_names), yy, gg, folds=args.folds, seed=args.seed)
            full = np.full(len(rows), np.nan)
            full[idx] = sb
            model_scores["OOF:baseline"] = full
            model_rows.append(eval_score("OOF:baseline", full, y, groups, eval_mask))
        if transition_names:
            st = oof_scores(design(rr, transition_names), yy, gg, folds=args.folds, seed=args.seed)
            full = np.full(len(rows), np.nan)
            full[idx] = st
            model_scores["OOF:transition"] = full
            model_rows.append(eval_score("OOF:transition", full, y, groups, eval_mask))
        if baseline_names and transition_names:
            sj = oof_scores(design(rr, baseline_names + transition_names), yy, gg, folds=args.folds, seed=args.seed)
            full = np.full(len(rows), np.nan)
            full[idx] = sj
            model_scores["OOF:baseline+transition"] = full
            model_rows.append(eval_score("OOF:baseline+transition", full, y, groups, eval_mask))
    model_rows.sort(key=lambda r: np.nan_to_num(r["within_pair_auroc_error_high"], nan=-1.0), reverse=True)

    best_baseline = max(
        [r for r in single_rows if r["score"] in baseline_names] + [r for r in model_rows if r["score"] == "OOF:baseline"],
        key=lambda r: np.nan_to_num(r["within_pair_auroc_error_high"], nan=-1.0),
        default=None,
    )
    best_transition = max(
        [r for r in single_rows if r["score"] in transition_names]
        + [r for r in model_rows if r["score"] in {"OOF:transition", "OOF:baseline+transition"}],
        key=lambda r: np.nan_to_num(r["within_pair_auroc_error_high"], nan=-1.0),
        default=None,
    )
    inc = {}
    score_bank = {**single_scores, **model_scores}
    if best_baseline and best_transition and best_baseline["score"] in score_bank and best_transition["score"] in score_bank:
        inc = bootstrap_within_increment(
            score_bank[best_transition["score"]],
            score_bank[best_baseline["score"]],
            groups=groups,
            y_err=y,
            n_boot=args.bootstrap,
            seed=args.seed + 17,
        )

    res = {
        "meta": {
            **meta,
            "eval_rows": int(eval_mask.sum()),
            "first_error_rows": int(np.sum(eval_mask & (y == 1))),
            "control_rows": int(np.sum(eval_mask & (y == 0))),
            "post_error_rows": int(np.sum(phases == "post_error")),
            "problem_groups": int(len(groups)),
            "within_pairs": int(sum(int(np.sum(y[g] == 1)) * int(np.sum(y[g] == 0)) for g in groups)),
            "baseline_features": baseline_names,
            "transition_features": transition_names,
        },
        "headline": {
            "best_baseline": best_baseline or {},
            "best_transition": best_transition or {},
            "increment_over_best_baseline": inc,
        },
        "single_scores": single_rows,
        "model_scores": model_rows,
        "transition_summary": transition_summary(rows, transition_names[:10]),
    }
    return res


def write_markdown(path: str, res: Mapping[str, Any]) -> None:
    meta = res["meta"]
    head = res["headline"]
    lines: List[str] = []
    lines.append(f"# Prefix Innovation Audit: {os.path.basename(str(meta['npz']))}\n")
    lines.append("## Summary\n")
    lines.append(
        f"- Rows {meta['eval_rows']} | first-error transitions {meta['first_error_rows']} | "
        f"controls {meta['control_rows']} | problems {meta['problem_groups']} | source {meta['source']} L{meta['layer']}."
    )
    bb = head.get("best_baseline") or {}
    bt = head.get("best_transition") or {}
    if bb:
        lines.append(f"- Best baseline `{bb['score']}` within {bb['within_pair_auroc_error_high']:.3f}, cross {bb['cross_auroc_error_high']:.3f}.")
    if bt:
        lines.append(f"- Best transition `{bt['score']}` within {bt['within_pair_auroc_error_high']:.3f}, cross {bt['cross_auroc_error_high']:.3f}.")
    inc = head.get("increment_over_best_baseline") or {}
    if inc:
        lines.append(f"- Increment over baseline: {inc.get('point')} CI [{inc.get('lo')}, {inc.get('hi')}] sig={inc.get('sig')}.")
    lines.append("")
    lines.append("## Model Scores\n")
    lines.append("| score | within | cross | pairs |")
    lines.append("|---|---:|---:|---:|")
    for r in res.get("model_scores", []):
        lines.append(f"| {r['score']} | {r['within_pair_auroc_error_high']:.3f} | {r['cross_auroc_error_high']:.3f} | {r['within_pairs']} |")
    lines.append("")
    lines.append("## Top Single Scores\n")
    lines.append("| score | within | cross | err median | ctrl median |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in res.get("single_scores", [])[:16]:
        lines.append(f"| {r['score']} | {r['within_pair_auroc_error_high']:.3f} | {r['cross_auroc_error_high']:.3f} | {r['err_median']:.3f} | {r['ctrl_median']:.3f} |")
    lines.append("")
    lines.append("## First-Error Transition Summary\n")
    lines.append("| feature | first mean | pre mean | correct mean | first-correct |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, st in res.get("transition_summary", {}).items():
        lines.append(
            f"| {name} | {st['first_error'].get('mean')} | {st['pre_error'].get('mean')} | "
            f"{st['correct_chain'].get('mean')} | {st.get('first_minus_correct_mean')} |"
        )
    dim = meta.get("dim_activation", {})
    if dim.get("ok"):
        lines.append("")
        lines.append("## Over-Activated Dimensions\n")
        lines.append("| rank | dim | err mean z | ctrl mean z | delta | err abs z | ctrl abs z | abs delta |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for i, r in enumerate(dim.get("top_positive", [])[:10], start=1):
            lines.append(
                f"| {i} | {r['dim']} | {r['err_mean_z']:.3f} | {r['ctrl_mean_z']:.3f} | "
                f"{r['delta_mean_z']:.3f} | {r['err_mean_abs_z']:.3f} | {r['ctrl_mean_abs_z']:.3f} | {r['delta_abs_z']:.3f} |"
            )
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append("- `off_prefix_subspace`: fraction of current step energy outside all prior-step subspace.")
    lines.append("- `innovation_off_prefix`: fraction of the mean displacement outside the prior-step subspace.")
    lines.append("- `prefix_z_*`: dimension-wise over-activation of the current step mean relative to the prefix token distribution.")
    lines.append("- These are transition features.  A positive result would support a step-flow break, not another isolated step-cloud scalar.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_outputs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    clean = finite_json(res)
    jpath = os.path.join(output_dir, stem + ".json")
    mpath = os.path.join(output_dir, stem + ".md")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    write_markdown(mpath, clean)
    return jpath, mpath


def _object_array(xs: Sequence[Any]) -> np.ndarray:
    out = np.empty(len(xs), dtype=object)
    out[:] = list(xs)
    return out


def make_selftest(path: str, *, seed: int = 0, n_problems: int = 16, samples_per_problem: int = 5, dim: int = 48) -> None:
    rng = np.random.default_rng(seed)
    layer = 16
    problem_ids: List[int] = []
    gold: List[int] = []
    clouds: List[np.ndarray] = []
    sizes: List[np.ndarray] = []
    qvecs: List[np.ndarray] = []
    ranges: List[np.ndarray] = []
    for p in range(n_problems):
        base = rng.normal(size=dim)
        base /= max(float(np.linalg.norm(base)), EPS)
        aux = rng.normal(size=dim)
        aux -= np.dot(aux, base) * base
        aux /= max(float(np.linalg.norm(aux)), EPS)
        bad = rng.normal(size=dim)
        bad -= np.dot(bad, base) * base + np.dot(bad, aux) * aux
        bad /= max(float(np.linalg.norm(bad)), EPS)
        bad_dim = 7
        for s in range(samples_per_problem):
            err = s >= samples_per_problem // 2
            problem_ids.append(p)
            gold.append(2 if err else -1)
            lens = np.array([6, 8, 8, 5], dtype=np.int32)
            centers = [
                base,
                0.8 * base + 0.2 * aux,
                0.65 * base + 0.35 * aux if not err else bad,
                0.6 * base + 0.4 * aux if not err else bad,
            ]
            chunks = []
            for t, c in enumerate(centers):
                cc = c / max(float(np.linalg.norm(c)), EPS)
                X = cc[None, :] + 0.025 * rng.normal(size=(int(lens[t]), dim))
                if err and t == 2:
                    X[:, bad_dim] += 2.5
                chunks.append(X)
            C = np.concatenate(chunks, axis=0)
            clouds.append(C[:, None, :].astype(np.float32))
            sizes.append(lens)
            lo = np.cumsum(np.r_[0, lens[:-1]])
            hi = lo + lens - 1
            ranges.append(np.stack([lo, hi], axis=1).astype(np.int32))
            qvecs.append(base[None, :].astype(np.float32))
    np.savez_compressed(
        path,
        problem_ids=np.asarray(problem_ids, dtype=np.int32),
        gold_error_step=np.asarray(gold, dtype=np.int32),
        sv_clouds=_object_array(clouds),
        cloud_sizes=_object_array(sizes),
        step_token_ranges=_object_array(ranges),
        cloud_layers=np.asarray([layer], dtype=np.int32),
        qvec=_object_array(qvecs),
        sv_layers=np.asarray([layer], dtype=np.int32),
    )


def assert_selftest(res: Mapping[str, Any]) -> None:
    best = res["headline"].get("best_transition") or {}
    if not best:
        raise AssertionError("no transition score produced")
    within = float(best.get("within_pair_auroc_error_high", float("nan")))
    if not np.isfinite(within) or within < 0.85:
        raise AssertionError(f"prefix innovation selftest too weak: {within}")
    dim = res["meta"].get("dim_activation", {})
    top = dim.get("top_positive", [{}])[0].get("dim") if dim.get("top_positive") else None
    if top != 7:
        raise AssertionError(f"expected over-activated dim 7, got {top}")


def print_result(res: Mapping[str, Any]) -> None:
    meta = res["meta"]
    print(f"===== prefix innovation audit | {os.path.basename(str(meta['npz']))} =====")
    print(
        f"rows {meta['eval_rows']} | first-error transitions {meta['first_error_rows']} | "
        f"controls {meta['control_rows']} | problems {meta['problem_groups']} | source {meta['source']} L{meta['layer']}"
    )
    bb = res["headline"].get("best_baseline") or {}
    bt = res["headline"].get("best_transition") or {}
    if bb:
        print(f"best baseline   {bb['score']} within={bb['within_pair_auroc_error_high']:.3f} cross={bb['cross_auroc_error_high']:.3f}")
    if bt:
        print(f"best transition {bt['score']} within={bt['within_pair_auroc_error_high']:.3f} cross={bt['cross_auroc_error_high']:.3f}")
    inc = res["headline"].get("increment_over_best_baseline") or {}
    if inc:
        print(f"increment over baseline: {inc.get('point')} CI=[{inc.get('lo')}, {inc.get('hi')}] sig={inc.get('sig')}")
    print("\nModel scores:")
    for r in res.get("model_scores", [])[:8]:
        print(f"  {r['score']:<28} within {r['within_pair_auroc_error_high']:.3f} cross {r['cross_auroc_error_high']:.3f}")
    print("\nTop single scores:")
    for r in res.get("single_scores", [])[:12]:
        print(f"  {r['score']:<28} within {r['within_pair_auroc_error_high']:.3f} cross {r['cross_auroc_error_high']:.3f}")
    dim = meta.get("dim_activation", {})
    if dim.get("ok"):
        print("\nTop over-activated dimensions:")
        for r in dim.get("top_positive", [])[:8]:
            print(f"  dim {r['dim']:<5d} delta_z {r['delta_mean_z']:+.3f} err {r['err_mean_z']:+.3f} ctrl {r['ctrl_mean_z']:+.3f}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="", help="input npz")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--nearest_layer", action="store_true")
    ap.add_argument("--hidden_dir", default="", help="override hidden shard directory for full_hidden npz files")
    ap.add_argument("--no_mmap", action="store_true")
    ap.add_argument("--control_pool", choices=["pre_and_correct", "pre_error", "correct_chain"], default="pre_and_correct")
    ap.add_argument("--rank", type=int, default=8, help="prefix/previous subspace rank")
    ap.add_argument("--beta", type=float, default=1.0, help="exp pooling strength inside a step")
    ap.add_argument("--raw_hidden", action="store_true", help="use raw hidden rows instead of centered unit directions for subspace features")
    ap.add_argument("--center_chain", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--z_top_k", type=int, default=16)
    ap.add_argument("--z_thresh", type=float, default=3.0)
    ap.add_argument("--std_floor_frac", type=float, default=0.10)
    ap.add_argument("--dim_top_k", type=int, default=20)
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--min_feature_coverage", type=float, default=0.60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs/prefix_innovation")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "prefix_innovation_selftest.npz")
            make_selftest(path, seed=args.seed)
            args.input = path
            args.no_progress = True
            res = run(path, args)
            assert_selftest(res)
            print_result(res)
            print("selftest passed")
            return
    if not args.input:
        raise SystemExit("--input is required unless --selftest is used")
    res = run(args.input, args)
    print_result(res)
    stem = os.path.splitext(os.path.basename(args.input))[0] + f"_L{res['meta']['layer']}_prefix_innovation"
    jpath, mpath = write_outputs(res, args.output_dir, stem)
    print(f"\nsaved: {jpath}")
    print(f"saved: {mpath}")


if __name__ == "__main__":
    main()
