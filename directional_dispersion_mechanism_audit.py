#!/usr/bin/env python3
"""Mechanism audit for low directional concentration in reasoning steps.

The validated project fact is that wrong reasoning steps often have lower
within-step directional concentration:

    u_i = h_i / ||h_i||
    kappa = ||sum_i w_i u_i||

This script does not ask whether another Gram/spectrum scalar beats kappa as a
detector.  It asks a narrower mechanism question:

    conditional on step length, position, and kappa, what morphology produced
    the low-kappa event?

For every step token cloud it decomposes the unit-direction second moment into
the rank-one consensus term plus centered residual scatter:

    A = sum_i w_i u_i u_i^T
    C = A - mu mu^T
    trace(C) = 1 - kappa^2

The trace is mathematically tied to kappa, so the only possible new mechanism
information is the shape of C and the token order/cluster structure around it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from multisample_temporal_rupture_audit import descriptive, finite_json, label_policy
from token_stream_geometry_audit import (
    chain_lengths,
    load_token_matrix,
    source_info,
)


EPS = 1e-12


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - progress bars are optional
    def tqdm(iterable, **_kwargs):
        return iterable


@dataclass
class StepRow:
    chain_idx: int
    problem_id: int
    step_idx: int
    gold_error_step: int
    y_err: int
    text: str
    features: Dict[str, float]
    taxonomy: Dict[str, Any]


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


def finite_quantile(x: Iterable[float], q: float) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if a.size else float("nan")


def exp_weights(n: int, beta: float) -> np.ndarray:
    """Exponentially emphasize later tokens inside a step."""
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    if n == 1 or abs(beta) <= EPS:
        return np.ones(n, dtype=np.float64) / n
    pos = np.linspace(0.0, 1.0, n)
    z = float(beta) * pos
    z -= float(z.max())
    w = np.exp(z)
    return w / max(float(w.sum()), EPS)


def unit_rows(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(H, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1)
    ok = np.isfinite(norms) & (norms > EPS) & np.isfinite(X).all(axis=1)
    X = X[ok]
    norms = norms[ok]
    if X.shape[0] == 0:
        return np.empty((0, X.shape[1] if X.ndim == 2 else 0), dtype=np.float64), np.empty(0, dtype=np.float64)
    return X / np.maximum(norms[:, None], EPS), norms


def unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    return x / max(float(np.linalg.norm(x)), EPS)


def weighted_mu_kappa(U: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, float]:
    mu = np.asarray(w, dtype=np.float64) @ np.asarray(U, dtype=np.float64)
    return mu, float(np.linalg.norm(mu))


def eig_from_weighted_rows(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues/vectors of Y Y^T, descending, for n_tokens << hidden_dim."""
    if Y.ndim != 2 or Y.shape[0] == 0:
        return np.empty(0, dtype=np.float64), np.empty((0, 0), dtype=np.float64)
    G = Y @ Y.T
    G = 0.5 * (G + G.T)
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1]
    vals = np.clip(vals[order], 0.0, None)
    vecs = vecs[:, order]
    return vals, vecs


def spectrum_shape(vals: np.ndarray, prefix: str, top_k: int) -> Dict[str, float]:
    x = np.asarray(vals, dtype=np.float64)
    x = x[np.isfinite(x) & (x > EPS)]
    out: Dict[str, float] = {
        f"{prefix}_trace": float(np.sum(x)) if x.size else float("nan"),
        f"{prefix}_lam1": float("nan"),
        f"{prefix}_lam2": float("nan"),
        f"{prefix}_lam1_frac": float("nan"),
        f"{prefix}_lam2_frac": float("nan"),
        f"{prefix}_gap12": float("nan"),
        f"{prefix}_eff_rank": float("nan"),
        f"{prefix}_entropy": float("nan"),
        f"{prefix}_participation": float("nan"),
    }
    for k in (2, 4, 8):
        if k <= top_k:
            out[f"{prefix}_top{k}_mass"] = float("nan")
    if x.size == 0:
        return out
    total = float(np.sum(x))
    p = x / max(total, EPS)
    ent = float(-np.sum(p * np.log(p + EPS)))
    out[f"{prefix}_lam1"] = float(x[0])
    out[f"{prefix}_lam2"] = float(x[1]) if x.size > 1 else 0.0
    out[f"{prefix}_lam1_frac"] = float(p[0])
    out[f"{prefix}_lam2_frac"] = float(p[1]) if p.size > 1 else 0.0
    out[f"{prefix}_gap12"] = float(p[0] - (p[1] if p.size > 1 else 0.0))
    out[f"{prefix}_entropy"] = ent
    out[f"{prefix}_eff_rank"] = float(np.exp(ent))
    out[f"{prefix}_participation"] = float(1.0 / max(float(np.sum(p * p)), EPS))
    csum = np.cumsum(p)
    for k in (2, 4, 8):
        if k <= top_k:
            out[f"{prefix}_top{k}_mass"] = float(csum[min(k, p.size) - 1])
    return out


def right_singular_direction(Y: np.ndarray, vals: np.ndarray, vecs: np.ndarray, j: int = 0) -> Optional[np.ndarray]:
    if vals.size <= j or vals[j] <= EPS or vecs.shape[1] <= j:
        return None
    v = Y.T @ vecs[:, j]
    v = v / math.sqrt(max(float(vals[j]), EPS))
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n <= EPS:
        return None
    return v / n


def sign_flip_rate(proj: np.ndarray) -> float:
    p = np.asarray(proj, dtype=np.float64)
    if p.size < 2:
        return float("nan")
    s = np.sign(p)
    keep = s != 0
    s = s[keep]
    if s.size < 2:
        return float("nan")
    return float(np.mean(s[1:] != s[:-1]))


def split_pair_cos_features(U: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    out = {
        "signed_cluster_balance": float("nan"),
        "signed_cluster_within_cos": float("nan"),
        "signed_cluster_between_cos": float("nan"),
        "signed_cluster_within_minus_between": float("nan"),
        "signed_clusterability": float("nan"),
    }
    y = np.asarray(labels, dtype=bool)
    if U.shape[0] < 4 or y.min() == y.max():
        return out
    frac = float(y.mean())
    out["signed_cluster_balance"] = float(2.0 * min(frac, 1.0 - frac))
    G = U @ U.T
    tri = np.triu_indices(U.shape[0], k=1)
    same = y[tri[0]] == y[tri[1]]
    if same.any():
        out["signed_cluster_within_cos"] = float(np.mean(G[tri][same]))
    if (~same).any():
        out["signed_cluster_between_cos"] = float(np.mean(G[tri][~same]))
    if np.isfinite(out["signed_cluster_within_cos"]) and np.isfinite(out["signed_cluster_between_cos"]):
        diff = out["signed_cluster_within_cos"] - out["signed_cluster_between_cos"]
        out["signed_cluster_within_minus_between"] = float(diff)
        out["signed_clusterability"] = float(max(0.0, diff) * out["signed_cluster_balance"])
    return out


def pairwise_cos_features(U: np.ndarray, *, max_pair_tokens: int) -> Dict[str, float]:
    out: Dict[str, float] = {
        "pair_cos_mean": float("nan"),
        "pair_cos_std": float("nan"),
        "pair_cos_q10": float("nan"),
        "pair_cos_q25": float("nan"),
        "pair_cos_q50": float("nan"),
        "pair_cos_q75": float("nan"),
        "pair_cos_q90": float("nan"),
        "pair_cos_frac_negative": float("nan"),
        "pair_cos_frac_below_025": float("nan"),
        "pair_cos_iqr": float("nan"),
    }
    n = U.shape[0]
    if n < 2:
        return out
    if n > max_pair_tokens:
        idx = np.linspace(0, n - 1, max_pair_tokens).round().astype(int)
        U = U[idx]
        n = U.shape[0]
    G = U @ U.T
    vals = G[np.triu_indices(n, k=1)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return out
    q10, q25, q50, q75, q90 = np.quantile(vals, [0.10, 0.25, 0.50, 0.75, 0.90])
    out.update(
        {
            "pair_cos_mean": float(np.mean(vals)),
            "pair_cos_std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "pair_cos_q10": float(q10),
            "pair_cos_q25": float(q25),
            "pair_cos_q50": float(q50),
            "pair_cos_q75": float(q75),
            "pair_cos_q90": float(q90),
            "pair_cos_frac_negative": float(np.mean(vals < 0.0)),
            "pair_cos_frac_below_025": float(np.mean(vals < 0.25)),
            "pair_cos_iqr": float(q75 - q25),
        }
    )
    return out


def chunk_kappa(U: np.ndarray, beta: float) -> Tuple[np.ndarray, float]:
    if U.shape[0] == 0:
        return np.zeros(U.shape[1] if U.ndim == 2 else 0), float("nan")
    w = exp_weights(U.shape[0], beta)
    return weighted_mu_kappa(U, w)


def early_late_features(U: np.ndarray, beta: float) -> Dict[str, float]:
    out = {
        "early_kappa": float("nan"),
        "late_kappa": float("nan"),
        "segment_kappa_mean": float("nan"),
        "segment_kappa_min": float("nan"),
        "early_late_cos": float("nan"),
        "early_late_shift": float("nan"),
        "ordered_shift": float("nan"),
    }
    if U.shape[0] < 4:
        return out
    mid = U.shape[0] // 2
    mu_e, ke = chunk_kappa(U[:mid], beta)
    mu_l, kl = chunk_kappa(U[mid:], beta)
    cos = float(np.dot(unit(mu_e), unit(mu_l))) if np.isfinite(ke) and np.isfinite(kl) else float("nan")
    shift = 1.0 - cos if np.isfinite(cos) else float("nan")
    seg_min = min(ke, kl) if np.isfinite(ke) and np.isfinite(kl) else float("nan")
    out.update(
        {
            "early_kappa": float(ke),
            "late_kappa": float(kl),
            "segment_kappa_mean": float(np.nanmean([ke, kl])),
            "segment_kappa_min": float(seg_min),
            "early_late_cos": cos,
            "early_late_shift": float(shift),
            "ordered_shift": float(shift * seg_min) if np.isfinite(shift) and np.isfinite(seg_min) else float("nan"),
        }
    )
    return out


def morphology_features(
    H: np.ndarray,
    *,
    beta: float,
    min_tokens: int,
    top_k: int,
    max_pair_tokens: int,
) -> Dict[str, float]:
    U, norms = unit_rows(H)
    n = int(U.shape[0])
    out: Dict[str, float] = {
        "n_tok": float(n),
        "logN": math.log1p(max(0, n)),
        "tok_norm_mean": float(np.mean(norms)) if norms.size else float("nan"),
        "tok_norm_cv": float(np.std(norms) / max(float(np.mean(norms)), EPS)) if norms.size else float("nan"),
    }
    if n < min_tokens:
        for name in (
            "kappa",
            "spread",
            "residual_energy",
            "res_identity_error",
            "axis_balance",
            "axis_separation",
            "axis_sign_flip_rate",
            "bipolarity",
        ):
            out[name] = float("nan")
        out.update(spectrum_shape(np.array([]), "res", top_k))
        out.update(pairwise_cos_features(U, max_pair_tokens=max_pair_tokens))
        out.update(early_late_features(U, beta))
        return out

    w = exp_weights(n, beta)
    mu, kappa = weighted_mu_kappa(U, w)
    Y = (U - mu[None, :]) * np.sqrt(w[:, None])
    vals, vecs = eig_from_weighted_rows(Y)
    res_trace = float(np.sum(vals))
    out["kappa"] = kappa
    out["spread"] = float(1.0 - kappa)
    out["residual_energy"] = float(1.0 - kappa * kappa)
    out["res_identity_error"] = float(abs(res_trace + kappa * kappa - 1.0))
    out.update(spectrum_shape(vals, "res", top_k))

    v1 = right_singular_direction(Y, vals, vecs, 0)
    if v1 is None:
        out.update(
            {
                "axis_balance": float("nan"),
                "axis_separation": float("nan"),
                "axis_abs_mean": float("nan"),
                "axis_sign_flip_rate": float("nan"),
                "bipolarity": float("nan"),
            }
        )
    else:
        proj = (U - mu[None, :]) @ v1
        pos = proj >= 0.0
        pos_mass = float(np.sum(w[pos]))
        neg_mass = float(np.sum(w[~pos]))
        balance = 2.0 * min(pos_mass, neg_mass)
        if pos.any() and (~pos).any():
            pmean = float(np.sum(w[pos] * proj[pos]) / max(pos_mass, EPS))
            nmean = float(np.sum(w[~pos] * proj[~pos]) / max(neg_mass, EPS))
            sep = pmean - nmean
        else:
            sep = float("nan")
        lam1_frac = out.get("res_lam1_frac", float("nan"))
        out.update(
            {
                "axis_balance": float(balance),
                "axis_separation": float(sep),
                "axis_abs_mean": float(np.sum(w * np.abs(proj))),
                "axis_sign_flip_rate": sign_flip_rate(proj),
                "bipolarity": float(lam1_frac * balance * abs(sep))
                if np.isfinite(lam1_frac) and np.isfinite(sep)
                else float("nan"),
            }
        )
        out.update(split_pair_cos_features(U, pos))

    out.update(pairwise_cos_features(U, max_pair_tokens=max_pair_tokens))
    out.update(early_late_features(U, beta))
    return out


def step_slices(H: np.ndarray, lengths: np.ndarray, ranges_obj: Any) -> List[np.ndarray]:
    X = np.asarray(H)
    if ranges_obj is not None:
        R = np.asarray(ranges_obj, dtype=int)
        if R.ndim == 2 and R.shape[1] >= 2 and R.shape[0] > 0:
            base = int(R[0, 0])
            out: List[np.ndarray] = []
            for lo0, hi0 in R:
                lo = max(0, int(lo0) - base)
                hi = min(X.shape[0], int(hi0) - base + 1)
                if hi > lo:
                    out.append(X[lo:hi])
            return out
    out = []
    cur = 0
    for s in np.asarray(lengths, dtype=int).reshape(-1):
        ss = int(s)
        if ss <= 0:
            continue
        out.append(X[cur : cur + ss])
        cur += ss
    return out


def object_to_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    try:
        return [str(v) for v in list(x)]
    except Exception:
        return []


def chain_policy_mask(data: np.lib.npyio.NpzFile, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, str]:
    n = len(data["problem_ids"]) if "problem_ids" in data.files else len(data["gold_error_step"])
    if args.policy == "gold_error_step" and "gold_error_step" in data.files:
        return (data["gold_error_step"].astype(int) >= 0).astype(int), np.ones(n, dtype=bool), "gold_error_step"
    if args.policy == "answer_format_ok" and "format_ok" not in data.files and "gold_error_step" in data.files:
        return (data["gold_error_step"].astype(int) >= 0).astype(int), np.ones(n, dtype=bool), "gold_error_step fallback"
    if args.policy in {"answer", "strict", "answer_format_ok"}:
        return label_policy(data, args.policy)
    raise ValueError(args.policy)


def step_label(gold: int, chain_y_err: int, step_idx: int, mode: str) -> Optional[int]:
    if mode == "chain_final":
        return int(chain_y_err)
    if gold < 0:
        return 0
    if mode == "first_error":
        if step_idx == gold:
            return 1
        if step_idx < gold:
            return 0
        return None
    if mode == "error_and_after":
        return int(step_idx >= gold)
    raise ValueError(mode)


def load_step_rows(path: str, args: argparse.Namespace) -> Tuple[List[StepRow], Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    if "problem_ids" in data.files:
        problem_ids = data["problem_ids"].astype(int)
    else:
        problem_ids = np.arange(len(data["gold_error_step"]), dtype=int)
    chain_y_err, chain_mask, policy_desc = chain_policy_mask(data, args)
    if args.label_mode == "auto":
        label_mode = "first_error" if "gold_error_step" in data.files else "chain_final"
    else:
        label_mode = args.label_mode
    if label_mode != "chain_final" and "gold_error_step" not in data.files:
        raise SystemExit(f"--label_mode={label_mode} requires gold_error_step")

    source, layer_i, layer_used = source_info(data, path, args)
    n_total = len(problem_ids)
    if args.max_chains:
        n_total = min(n_total, int(args.max_chains))
    rows: List[StepRow] = []
    skipped = {"policy": 0, "missing_hidden": 0, "bad_steps": 0, "label": 0}
    iterator = range(n_total)
    if not args.no_progress:
        iterator = tqdm(iterator, desc="step morphologies", unit="chain", dynamic_ncols=True)
    for i in iterator:
        if not chain_mask[i]:
            skipped["policy"] += 1
            continue
        H = load_token_matrix(data, path, args, idx=i, source=source, layer_i=layer_i)
        if H is None:
            skipped["missing_hidden"] += 1
            continue
        lengths, ranges_obj = chain_lengths(data, i, int(H.shape[0]), source)
        steps = step_slices(H, lengths, ranges_obj)
        if not steps:
            skipped["bad_steps"] += 1
            continue
        gold = int(data["gold_error_step"][i]) if "gold_error_step" in data.files else -1
        texts = object_to_str_list(data["steps_text"][i]) if "steps_text" in data.files else []
        T = len(steps)
        for t, step_H in enumerate(steps):
            y = step_label(gold, int(chain_y_err[i]), t, label_mode)
            if y is None:
                skipped["label"] += 1
                continue
            feats = morphology_features(
                step_H,
                beta=args.kappa_beta,
                min_tokens=args.min_tokens,
                top_k=args.top_k,
                max_pair_tokens=args.max_pair_tokens,
            )
            feats["pos"] = float(t / max(1, T - 1))
            feats["n_steps"] = float(T)
            rows.append(
                StepRow(
                    chain_idx=int(i),
                    problem_id=int(problem_ids[i]),
                    step_idx=int(t),
                    gold_error_step=int(gold),
                    y_err=int(y),
                    text=texts[t] if t < len(texts) else "",
                    features=feats,
                    taxonomy={},
                )
            )
    data.close()
    if len(rows) < 20 or len({r.y_err for r in rows}) < 2:
        raise SystemExit("not enough labeled step rows with both classes")
    meta = {
        "input": os.path.abspath(path),
        "basename": os.path.basename(path),
        "policy": args.policy,
        "policy_description": policy_desc,
        "label_mode": label_mode,
        "source": source,
        "layer": int(layer_used),
        "kappa_pooling": f"exp(beta={args.kappa_beta:g})",
        "n_rows": int(len(rows)),
        "n_error_rows": int(sum(r.y_err for r in rows)),
        "n_correct_rows": int(sum(1 - r.y_err for r in rows)),
        "n_chains": int(len(set(r.chain_idx for r in rows))),
        "n_problems": int(len(set(r.problem_id for r in rows))),
        "skipped": skipped,
        "controls": {
            "length_bins": int(args.length_bins),
            "kappa_bins": int(args.kappa_bins),
            "pos_bins": int(args.pos_bins),
            "matched_test": "pairwise error-vs-correct comparisons within finite (length, kappa, position) bins",
        },
    }
    return rows, meta


def feature_array(rows: Sequence[StepRow], name: str) -> np.ndarray:
    return np.asarray([r.features.get(name, float("nan")) for r in rows], dtype=np.float64)


def labels(rows: Sequence[StepRow]) -> np.ndarray:
    return np.asarray([r.y_err for r in rows], dtype=int)


def chain_groups(rows: Sequence[StepRow]) -> np.ndarray:
    return np.asarray([r.chain_idx for r in rows], dtype=int)


def quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, -1, dtype=int)
    m = np.isfinite(x)
    if n_bins <= 1 or m.sum() == 0:
        out[m] = 0
        return out
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    cuts = np.unique(np.quantile(x[m], qs))
    if cuts.size == 0:
        out[m] = 0
    else:
        out[m] = np.searchsorted(cuts, x[m], side="right")
    return out


def condition_keys(rows: Sequence[StepRow], *, length_bins: int, kappa_bins: int, pos_bins: int) -> np.ndarray:
    logn = feature_array(rows, "logN")
    kap = feature_array(rows, "kappa")
    pos = feature_array(rows, "pos")
    lb = quantile_bins(logn, length_bins)
    kb = quantile_bins(kap, kappa_bins)
    if pos_bins > 1:
        pb = quantile_bins(pos, pos_bins)
    else:
        pb = np.zeros(len(rows), dtype=int)
        pb[~np.isfinite(pos)] = -1
    key = lb.astype(np.int64)
    key = key * 100 + kb
    key = key * 100 + pb
    key[(lb < 0) | (kb < 0) | (pb < 0)] = -1
    return key


def conditional_pair_stats(
    score: np.ndarray,
    y: np.ndarray,
    keys: np.ndarray,
    *,
    min_pairs_per_bin: int = 1,
) -> Dict[str, Any]:
    s = np.asarray(score, dtype=np.float64)
    yy = np.asarray(y, dtype=int)
    kk = np.asarray(keys)
    total_conc = 0.0
    total_pairs = 0
    delta_num = 0.0
    delta_den = 0
    used_bins = 0
    for key in np.unique(kk[kk >= 0]):
        m = (kk == key) & np.isfinite(s)
        err = s[m & (yy == 1)]
        cor = s[m & (yy == 0)]
        pairs = int(err.size * cor.size)
        if pairs < min_pairs_per_bin:
            continue
        used_bins += 1
        total_pairs += pairs
        # Pairwise comparison is explicit to avoid rank edge cases inside very
        # small bins and to keep the statistic exactly matched to the design.
        cmp = err[:, None] - cor[None, :]
        total_conc += float(np.sum(cmp > 0.0) + 0.5 * np.sum(cmp == 0.0))
        delta_num += float(np.mean(err) - np.mean(cor)) * pairs
        delta_den += pairs
    auc = total_conc / total_pairs if total_pairs else float("nan")
    return {
        "conditional_auroc_error_high": float(auc),
        "conditional_best_direction": float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan"),
        "conditional_delta_error_minus_correct": float(delta_num / delta_den) if delta_den else float("nan"),
        "pairs": int(total_pairs),
        "bins": int(used_bins),
    }


def bootstrap_conditional(
    rows: Sequence[StepRow],
    score: np.ndarray,
    keys: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    y = labels(rows)
    point = conditional_pair_stats(score, y, keys)
    if not np.isfinite(point["conditional_delta_error_minus_correct"]) or n_boot <= 0:
        return {"point": point, "delta_ci": None, "auc_ci": None, "sig_delta": False}
    groups = chain_groups(rows)
    ug = np.unique(groups)
    by_group = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    deltas: List[float] = []
    aucs: List[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(ug, size=ug.size, replace=True)
        idx = np.concatenate([by_group[g] for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        st = conditional_pair_stats(score[idx], y[idx], keys[idx])
        if np.isfinite(st["conditional_delta_error_minus_correct"]):
            deltas.append(st["conditional_delta_error_minus_correct"])
        if np.isfinite(st["conditional_auroc_error_high"]):
            aucs.append(st["conditional_auroc_error_high"])
    delta_ci = None
    auc_ci = None
    sig = False
    if deltas:
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        delta_ci = [float(lo), float(hi)]
        sig = bool(lo > 0.0 or hi < 0.0)
    if aucs:
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        auc_ci = [float(lo), float(hi)]
    return {"point": point, "delta_ci": delta_ci, "auc_ci": auc_ci, "sig_delta": sig}


def score_feature(rows: Sequence[StepRow], name: str, keys: np.ndarray, args: argparse.Namespace, idx: int) -> Dict[str, Any]:
    s = feature_array(rows, name)
    y = labels(rows)
    m = np.isfinite(s)
    cross = auroc(s[m], y[m]) if m.sum() and len(np.unique(y[m])) == 2 else float("nan")
    boot = bootstrap_conditional(rows, s, keys, n_boot=args.bootstrap, seed=args.seed + idx)
    err = s[m & (y == 1)]
    cor = s[m & (y == 0)]
    return {
        "feature": name,
        "coverage": float(np.mean(m)),
        "cross_auroc_error_high": float(cross),
        "cross_best_direction": float(max(cross, 1.0 - cross)) if np.isfinite(cross) else float("nan"),
        "error": descriptive(err),
        "correct": descriptive(cor),
        "conditional": boot,
    }


def morphology_feature_names(rows: Sequence[StepRow], min_coverage: float) -> List[str]:
    names = sorted({k for r in rows for k in r.features})
    core_exclude = {"n_tok", "logN", "pos", "n_steps"}
    out = []
    for name in names:
        if name in core_exclude:
            continue
        cov = np.mean(np.isfinite(feature_array(rows, name)))
        if cov >= min_coverage:
            out.append(name)
    preferred = [
        "res_eff_rank",
        "res_participation",
        "res_lam1_frac",
        "res_gap12",
        "bipolarity",
        "axis_balance",
        "axis_separation",
        "axis_sign_flip_rate",
        "signed_clusterability",
        "signed_cluster_within_minus_between",
        "pair_cos_frac_negative",
        "pair_cos_frac_below_025",
        "pair_cos_iqr",
        "early_late_shift",
        "ordered_shift",
        "segment_kappa_min",
        "tok_norm_cv",
    ]
    return [x for x in preferred if x in out] + [x for x in out if x not in preferred]


def robust_z(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    m = np.isfinite(x)
    if not m.any():
        return out
    med = float(np.median(x[m]))
    mad = float(np.median(np.abs(x[m] - med)))
    scale = 1.4826 * mad if mad > EPS else float(np.std(x[m]))
    if scale <= EPS or not math.isfinite(scale):
        scale = 1.0
    out[m] = (x[m] - med) / scale
    return out


def robust_z_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    r = ref[np.isfinite(ref)]
    if r.size == 0:
        return robust_z(x)
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    scale = 1.4826 * mad if mad > EPS else float(np.std(r))
    if scale <= EPS or not math.isfinite(scale):
        scale = 1.0
    m = np.isfinite(x)
    out[m] = (x[m] - med) / scale
    return out


def assign_taxonomy(rows: Sequence[StepRow], low_kappa_q: float) -> Dict[str, Any]:
    kappa = feature_array(rows, "kappa")
    y = labels(rows)
    low_thr = finite_quantile(kappa, low_kappa_q)
    low = np.isfinite(kappa) & (kappa <= low_thr)
    ref = low & (y == 0)
    if ref.sum() < 10:
        ref = low
    metrics = {
        "high_rank_dispersion": feature_array(rows, "res_eff_rank"),
        "bipolar_split": feature_array(rows, "bipolarity"),
        "multi_cluster": feature_array(rows, "signed_clusterability"),
        "ordered_substep_shift": feature_array(rows, "ordered_shift"),
    }
    thresholds = {"low_kappa": float(low_thr)}
    # Robust z-scores are anchored to low-kappa correct controls when possible.
    # This makes a taxonomy flag mean "unusual among correct low-kappa steps",
    # not merely "large inside a mixture of correct and wrong low-kappa steps".
    global_z = {name: robust_z_against(v, v[ref]) for name, v in metrics.items()}
    for name, v in metrics.items():
        thresholds[name] = finite_quantile(v[ref], 0.70) if ref.any() else float("nan")

    for i, r in enumerate(rows):
        is_low = bool(np.isfinite(kappa[i]) and kappa[i] <= low_thr)
        flags = {}
        for name, v in metrics.items():
            thr = thresholds[name]
            flags[name] = bool(is_low and np.isfinite(v[i]) and np.isfinite(thr) and v[i] >= thr)
        if not is_low:
            primary = "not_low_kappa"
        else:
            best_name = "unclassified_low_kappa"
            best_score = -float("inf")
            for name in metrics:
                z = global_z[name][i]
                if flags[name] and np.isfinite(z) and z > best_score:
                    best_score = float(z)
                    best_name = name
            primary = best_name
        r.taxonomy = {
            "low_kappa": is_low,
            "primary": primary,
            **flags,
        }
    return {
        "thresholds": thresholds,
        "low_kappa_rate": float(np.mean(low)),
        "threshold_reference": "low-kappa correct controls" if (low & (y == 0)).sum() >= 10 else "all low-kappa rows",
    }


def enrichment_table(rows: Sequence[StepRow]) -> Dict[str, Any]:
    y = labels(rows)
    classes = sorted({str(r.taxonomy.get("primary", "unassigned")) for r in rows})
    flags = [
        "low_kappa",
        "high_rank_dispersion",
        "bipolar_split",
        "multi_cluster",
        "ordered_substep_shift",
    ]
    out: Dict[str, Any] = {"primary": {}, "flags": {}}
    for cls in classes:
        m = np.asarray([r.taxonomy.get("primary") == cls for r in rows], dtype=bool)
        out["primary"][cls] = enrichment_stats(m, y)
    for flag in flags:
        m = np.asarray([bool(r.taxonomy.get(flag, False)) for r in rows], dtype=bool)
        out["flags"][flag] = enrichment_stats(m, y)
    return out


def enrichment_stats(mask: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    y = np.asarray(y, dtype=int)
    e_in = int(np.sum(mask & (y == 1)))
    c_in = int(np.sum(mask & (y == 0)))
    e_out = int(np.sum((~mask) & (y == 1)))
    c_out = int(np.sum((~mask) & (y == 0)))
    err_rate_in = e_in / max(e_in + c_in, 1)
    err_rate_out = e_out / max(e_out + c_out, 1)
    odds = ((e_in + 0.5) * (c_out + 0.5)) / max((c_in + 0.5) * (e_out + 0.5), EPS)
    return {
        "n": int(mask.sum()),
        "error": e_in,
        "correct": c_in,
        "error_rate": float(err_rate_in),
        "outside_error_rate": float(err_rate_out),
        "odds_ratio": float(odds),
    }


def hypothesis_rows(scores: Mapping[str, Any]) -> Dict[str, Any]:
    mapping = {
        "H1a_high_rank_dispersion": "res_eff_rank",
        "H1b_bipolar_cancellation": "bipolarity",
        "H1c_multicluster_mixing": "signed_clusterability",
        "H1d_ordered_substep_shift": "ordered_shift",
    }
    return {h: scores.get(feat) for h, feat in mapping.items() if feat in scores}


def examples(rows: Sequence[StepRow], *, per_class: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    classes = [
        "high_rank_dispersion",
        "bipolar_split",
        "multi_cluster",
        "ordered_substep_shift",
        "unclassified_low_kappa",
    ]
    score_by_class = {
        "high_rank_dispersion": "res_eff_rank",
        "bipolar_split": "bipolarity",
        "multi_cluster": "signed_clusterability",
        "ordered_substep_shift": "ordered_shift",
        "unclassified_low_kappa": "spread",
    }
    for cls in classes:
        cand = [r for r in rows if r.taxonomy.get("primary") == cls]
        feat = score_by_class[cls]
        cand.sort(key=lambda r: np.nan_to_num(r.features.get(feat, float("nan")), nan=-1e9), reverse=True)
        for r in cand[:per_class]:
            out.append(
                {
                    "class": cls,
                    "chain_idx": r.chain_idx,
                    "problem_id": r.problem_id,
                    "step_idx": r.step_idx,
                    "gold_error_step": r.gold_error_step,
                    "y_err": r.y_err,
                    "text": r.text[:500],
                    "features": {
                        k: r.features.get(k)
                        for k in (
                            "n_tok",
                            "kappa",
                            "res_eff_rank",
                            "res_lam1_frac",
                            "bipolarity",
                            "signed_clusterability",
                            "ordered_shift",
                            "pair_cos_q10",
                        )
                    },
                }
            )
    return out


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    rows, meta = load_step_rows(path, args)
    tax_meta = assign_taxonomy(rows, args.low_kappa_q)
    keys = condition_keys(
        rows,
        length_bins=args.length_bins,
        kappa_bins=args.kappa_bins,
        pos_bins=args.pos_bins,
    )
    names = morphology_feature_names(rows, args.min_feature_coverage)
    score_rows: Dict[str, Any] = {}
    iterator = list(enumerate(names))
    if not args.no_progress:
        iterator = tqdm(iterator, desc="conditional feature tests", unit="feature", dynamic_ncols=True)
    for j, name in iterator:
        score_rows[name] = score_feature(rows, name, keys, args, j)
    ranked = sorted(
        score_rows.items(),
        key=lambda kv: np.nan_to_num(kv[1]["conditional"]["point"]["conditional_best_direction"], nan=-1.0),
        reverse=True,
    )
    identity = feature_array(rows, "res_identity_error")
    low = np.asarray([bool(r.taxonomy.get("low_kappa", False)) for r in rows], dtype=bool)
    y = labels(rows)
    res = {
        "meta": {
            **meta,
            "min_tokens": int(args.min_tokens),
            "top_k": int(args.top_k),
            "max_pair_tokens": int(args.max_pair_tokens),
            "low_kappa_q": float(args.low_kappa_q),
            "taxonomy": tax_meta,
            "identity_check": {
                "median_abs_trace_plus_kappa2_minus_1": float(np.nanmedian(identity)),
                "q90_abs_trace_plus_kappa2_minus_1": finite_quantile(identity, 0.90),
            },
            "method_notes": {
                "not_a_detector": "The headline is conditioned morphology, not classifier AUROC.",
                "conditioning": "Main pair tests compare error and correct steps only inside shared length/kappa/position bins.",
                "kappa_identity": "Residual scatter trace equals 1-kappa^2; only residual shape can be new.",
            },
        },
        "headline": {
            "hypotheses": hypothesis_rows(score_rows),
            "top_conditioned_features": {k: v for k, v in ranked[:20]},
            "taxonomy_enrichment": enrichment_table(rows),
            "low_kappa_error_rate": float(np.mean(y[low])) if low.any() else float("nan"),
            "non_low_kappa_error_rate": float(np.mean(y[~low])) if (~low).any() else float("nan"),
        },
        "feature_scores": score_rows,
        "examples": examples(rows, per_class=args.examples_per_class),
        "rows_for_csv": rows,
    }
    return res


def write_csvs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str]:
    rows: Sequence[StepRow] = res["rows_for_csv"]
    tax_path = os.path.join(output_dir, stem + ".taxonomy.csv")
    bin_path = os.path.join(output_dir, stem + ".features.csv")
    base_cols = ["chain_idx", "problem_id", "step_idx", "gold_error_step", "y_err", "primary", "low_kappa"]
    feat_cols = [
        "n_tok",
        "pos",
        "kappa",
        "spread",
        "res_eff_rank",
        "res_lam1_frac",
        "res_gap12",
        "bipolarity",
        "axis_balance",
        "signed_clusterability",
        "early_late_shift",
        "ordered_shift",
        "pair_cos_q10",
        "pair_cos_q50",
        "pair_cos_frac_negative",
    ]
    with open(tax_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_cols + feat_cols)
        w.writeheader()
        for r in rows:
            d = {
                "chain_idx": r.chain_idx,
                "problem_id": r.problem_id,
                "step_idx": r.step_idx,
                "gold_error_step": r.gold_error_step,
                "y_err": r.y_err,
                "primary": r.taxonomy.get("primary"),
                "low_kappa": r.taxonomy.get("low_kappa"),
            }
            d.update({c: r.features.get(c) for c in feat_cols})
            w.writerow(d)
    with open(bin_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "conditional_auc", "conditional_best", "conditional_delta", "delta_ci_lo", "delta_ci_hi", "pairs", "bins"])
        for name, row in res["feature_scores"].items():
            pt = row["conditional"]["point"]
            ci = row["conditional"].get("delta_ci") or [None, None]
            w.writerow(
                [
                    name,
                    pt.get("conditional_auroc_error_high"),
                    pt.get("conditional_best_direction"),
                    pt.get("conditional_delta_error_minus_correct"),
                    ci[0],
                    ci[1],
                    pt.get("pairs"),
                    pt.get("bins"),
                ]
            )
    return tax_path, bin_path


def write_markdown(path: str, res: Mapping[str, Any]) -> None:
    meta = res["meta"]
    head = res["headline"]

    def fmt(x: Any, signed: bool = False) -> str:
        try:
            v = float(x)
            if not math.isfinite(v):
                return ""
            return f"{v:+.3f}" if signed else f"{v:.3f}"
        except Exception:
            return ""

    lines = [
        f"# Directional Dispersion Mechanism Audit: `{meta['basename']}`",
        "",
        "## Headline",
        "",
        f"- Source: `{meta['source']}` at layer `{meta['layer']}`; label mode `{meta['label_mode']}`.",
        f"- Step rows: `{meta['n_rows']}`; error rows: `{meta['n_error_rows']}`; correct rows: `{meta['n_correct_rows']}`.",
        f"- Matched controls: length bins `{meta['controls']['length_bins']}`, kappa bins `{meta['controls']['kappa_bins']}`, position bins `{meta['controls']['pos_bins']}`.",
        f"- Low-kappa threshold: `{fmt(meta['taxonomy']['thresholds']['low_kappa'])}`; low-kappa error rate `{fmt(head['low_kappa_error_rate'])}` vs non-low `{fmt(head['non_low_kappa_error_rate'])}`.",
        f"- Identity check median `|trace(C)+kappa^2-1|`: `{fmt(meta['identity_check']['median_abs_trace_plus_kappa2_minus_1'])}`.",
        "",
        "## Hypothesis Tests",
        "",
        "| hypothesis | feature | cond. AUROC | best-dir | delta err-cor | delta CI | pairs | bins |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for h, row in head["hypotheses"].items():
        if not row:
            continue
        pt = row["conditional"]["point"]
        ci = row["conditional"].get("delta_ci") or [None, None]
        ci_txt = "" if ci[0] is None else f"[{fmt(ci[0], True)}, {fmt(ci[1], True)}]"
        lines.append(
            f"| `{h}` | `{row['feature']}` | {fmt(pt['conditional_auroc_error_high'])} | "
            f"{fmt(pt['conditional_best_direction'])} | {fmt(pt['conditional_delta_error_minus_correct'], True)} | "
            f"{ci_txt} | {pt['pairs']} | {pt['bins']} |"
        )
    lines += [
        "",
        "## Top Conditioned Morphology Features",
        "",
        "| feature | cond. AUROC | best-dir | delta err-cor | delta CI | cross best-dir | coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in list(head["top_conditioned_features"].items())[:16]:
        pt = row["conditional"]["point"]
        ci = row["conditional"].get("delta_ci") or [None, None]
        ci_txt = "" if ci[0] is None else f"[{fmt(ci[0], True)}, {fmt(ci[1], True)}]"
        lines.append(
            f"| `{name}` | {fmt(pt['conditional_auroc_error_high'])} | {fmt(pt['conditional_best_direction'])} | "
            f"{fmt(pt['conditional_delta_error_minus_correct'], True)} | {ci_txt} | "
            f"{fmt(row['cross_best_direction'])} | {fmt(row['coverage'])} |"
        )
    lines += [
        "",
        "## Low-Kappa Taxonomy Enrichment",
        "",
        "| class / flag | n | error | correct | error rate | outside rate | odds ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    primary = head["taxonomy_enrichment"]["primary"]
    for cls, row in sorted(primary.items(), key=lambda kv: np.nan_to_num(kv[1]["odds_ratio"], nan=-1), reverse=True):
        lines.append(
            f"| `{cls}` | {row['n']} | {row['error']} | {row['correct']} | "
            f"{fmt(row['error_rate'])} | {fmt(row['outside_error_rate'])} | {fmt(row['odds_ratio'])} |"
        )
    lines.append("")
    lines.append("## Interpretation Guardrails")
    lines.append("")
    lines.append("- `residual_energy` is not new signal: it is exactly `1-kappa^2` up to numerical error.")
    lines.append("- A positive morphology result means low-kappa errors differ in residual shape after matching length/kappa/position.")
    lines.append("- A null result means direction-only geometry is saturated by kappa and should hand off to source-aware anchor analysis.")
    lines.append("- `chain_final` label mode is descriptive only; first-error claims require `gold_error_step`.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_outputs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str, str, str, str]:
    os.makedirs(output_dir, exist_ok=True)
    clean = dict(res)
    rows = clean.pop("rows_for_csv", None)
    jpath = os.path.join(output_dir, stem + ".json")
    mpath = os.path.join(output_dir, stem + ".md")
    epath = os.path.join(output_dir, stem + ".examples.jsonl")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(finite_json(clean), f, ensure_ascii=False, indent=2)
    write_markdown(mpath, finite_json(clean))
    with open(epath, "w", encoding="utf-8") as f:
        for ex in clean.get("examples", []):
            f.write(json.dumps(finite_json(ex), ensure_ascii=False) + "\n")
    # Reattach rows for CSV writing; avoid serializing them to JSON.
    if rows is not None:
        tmp = dict(clean)
        tmp["rows_for_csv"] = rows
        tax_path, feat_path = write_csvs(tmp, output_dir, stem)
    else:
        tax_path, feat_path = "", ""
    return jpath, mpath, tax_path, feat_path, epath


def print_result(res: Mapping[str, Any]) -> None:
    meta = res["meta"]
    head = res["headline"]
    print(f"\n===== directional dispersion mechanism | {meta['basename']} =====")
    print(
        f"rows {meta['n_rows']} | err {meta['n_error_rows']} | chains {meta['n_chains']} | "
        f"source {meta['source']} L{meta['layer']} | label {meta['label_mode']}"
    )
    print(
        "low-kappa err-rate "
        f"{head['low_kappa_error_rate']:.3f} vs non-low {head['non_low_kappa_error_rate']:.3f} | "
        "identity median "
        f"{meta['identity_check']['median_abs_trace_plus_kappa2_minus_1']:.2e}"
    )
    print("\nHypotheses:")
    for h, row in head["hypotheses"].items():
        if not row:
            continue
        pt = row["conditional"]["point"]
        ci = row["conditional"].get("delta_ci")
        ci_txt = "" if not ci else f" CI [{ci[0]:+.3f},{ci[1]:+.3f}]"
        print(
            f"  {h:30s} {row['feature']:28s} cond {pt['conditional_auroc_error_high']:.3f} "
            f"best {pt['conditional_best_direction']:.3f} delta {pt['conditional_delta_error_minus_correct']:+.3f}{ci_txt}"
        )
    print("\nTop conditioned features:")
    for name, row in list(head["top_conditioned_features"].items())[:12]:
        pt = row["conditional"]["point"]
        print(
            f"  {name:36s} cond {pt['conditional_auroc_error_high']:.3f} "
            f"best {pt['conditional_best_direction']:.3f} delta {pt['conditional_delta_error_minus_correct']:+.3f}"
        )
    print("\nTaxonomy:")
    for cls, row in sorted(
        head["taxonomy_enrichment"]["primary"].items(),
        key=lambda kv: np.nan_to_num(kv[1]["odds_ratio"], nan=-1),
        reverse=True,
    ):
        print(
            f"  {cls:26s} n={row['n']:4d} err_rate={row['error_rate']:.3f} "
            f"outside={row['outside_error_rate']:.3f} OR={row['odds_ratio']:.2f}"
        )


def _make_cloud(
    rng: np.random.Generator,
    *,
    n: int,
    dim: int,
    kappa: float,
    mode: str,
) -> np.ndarray:
    e0 = np.zeros(dim, dtype=np.float64)
    e0[0] = 1.0
    res_scale = math.sqrt(max(0.0, 1.0 - kappa * kappa))
    rows: List[np.ndarray] = []
    if mode == "high_rank":
        rank = min(8, dim - 1)
        for i in range(n):
            r = np.zeros(dim)
            axis = 1 + ((i // 2) % rank)
            sign = 1.0 if i % 2 == 0 else -1.0
            r[axis] = sign
            rows.append(unit(kappa * e0 + res_scale * r + 0.01 * rng.normal(size=dim)))
    elif mode == "bipolar":
        e1 = np.zeros(dim)
        e1[1] = 1.0
        for i in range(n):
            sign = 1.0 if i % 2 == 0 else -1.0
            rows.append(unit(kappa * e0 + sign * res_scale * e1 + 0.01 * rng.normal(size=dim)))
    elif mode == "ordered":
        e1 = np.zeros(dim)
        e1[1] = 1.0
        e2 = np.zeros(dim)
        e2[2] = 1.0
        for i in range(n):
            r = e1 if i < n // 2 else e2
            rows.append(unit(kappa * e0 + res_scale * r + 0.01 * rng.normal(size=dim)))
    else:
        e1 = np.zeros(dim)
        e1[1] = 1.0
        for i in range(n):
            sign = 1.0 if i % 2 == 0 else -1.0
            rows.append(unit(kappa * e0 + sign * res_scale * e1 + 0.01 * rng.normal(size=dim)))
    return np.asarray(rows, dtype=np.float32)[:, None, :]


def _object_array(xs: Sequence[Any]) -> np.ndarray:
    out = np.empty(len(xs), dtype=object)
    out[:] = list(xs)
    return out


def make_selftest(path: str, *, seed: int = 0, n_chains: int = 90, dim: int = 48) -> None:
    rng = np.random.default_rng(seed)
    problem_ids: List[int] = []
    gold: List[int] = []
    is_correct: List[int] = []
    sv_clouds: List[np.ndarray] = []
    sizes: List[np.ndarray] = []
    steps_text: List[np.ndarray] = []
    for i in range(n_chains):
        err = i % 3 == 0
        T = 4
        g = 2 if err else -1
        chunks = []
        step_sizes = []
        texts = []
        for t in range(T):
            n = int(rng.integers(18, 24))
            step_sizes.append(n)
            if err and t == g:
                k = 0.42 + 0.010 * rng.normal()
                mode = "high_rank"
            elif (not err) and t == 2 and i % 6 == 1:
                k = 0.48 + 0.010 * rng.normal()
                mode = "ordered"
            elif (not err) and t == 2 and i % 6 == 4:
                k = 0.48 + 0.010 * rng.normal()
                mode = "bipolar"
            else:
                k = 0.50 + 0.012 * rng.normal()
                mode = "rank1"
            chunks.append(_make_cloud(rng, n=n, dim=dim, kappa=float(np.clip(k, 0.38, 0.55)), mode=mode))
            texts.append(f"Step {t}: synthetic {mode} morphology")
        problem_ids.append(i // 3)
        gold.append(g)
        is_correct.append(0 if err else 1)
        sv_clouds.append(np.concatenate(chunks, axis=0))
        sizes.append(np.asarray(step_sizes, dtype=np.int32))
        steps_text.append(np.asarray(texts, dtype=object))
    np.savez_compressed(
        path,
        problem_ids=np.asarray(problem_ids, dtype=np.int32),
        gold_error_step=np.asarray(gold, dtype=np.int32),
        is_correct=np.asarray(is_correct, dtype=np.int8),
        is_correct_strict=np.asarray(is_correct, dtype=np.int8),
        format_ok=np.ones(n_chains, dtype=bool),
        sv_clouds=_object_array(sv_clouds),
        cloud_sizes=_object_array(sizes),
        cloud_layers=np.asarray([16], dtype=np.int32),
        steps_text=_object_array(steps_text),
    )


def assert_selftest(res: Mapping[str, Any]) -> None:
    h = res["headline"]["hypotheses"]
    h1a = h["H1a_high_rank_dispersion"]["conditional"]["point"]
    if h1a["conditional_best_direction"] < 0.75:
        raise AssertionError("selftest failed: high-rank residual morphology was not recovered")
    delta = h1a["conditional_delta_error_minus_correct"]
    if not np.isfinite(delta) or delta <= 0.0:
        raise AssertionError("selftest failed: high-rank residual effect has wrong sign")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="")
    ap.add_argument("--policy", default="gold_error_step", choices=["gold_error_step", "answer", "strict", "answer_format_ok"])
    ap.add_argument("--label_mode", default="auto", choices=["auto", "first_error", "error_and_after", "chain_final"])
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--nearest_layer", action="store_true")
    ap.add_argument("--hidden_dir", default="", help="override hidden shard directory for full_hidden npz files")
    ap.add_argument("--no_mmap", action="store_true")
    ap.add_argument("--kappa_beta", type=float, default=1.0)
    ap.add_argument("--min_tokens", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--max_pair_tokens", type=int, default=96)
    ap.add_argument("--length_bins", type=int, default=4)
    ap.add_argument("--kappa_bins", type=int, default=4)
    ap.add_argument("--pos_bins", type=int, default=3)
    ap.add_argument("--low_kappa_q", type=float, default=0.30)
    ap.add_argument("--min_feature_coverage", type=float, default=0.70)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--examples_per_class", type=int, default=4)
    ap.add_argument("--output_dir", default="outputs/directional_dispersion_mechanism")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "directional_dispersion_selftest.npz")
            make_selftest(path, seed=args.seed)
            args.input = path
            args.layer = 16
            args.no_progress = True
            args.bootstrap = min(args.bootstrap, 50)
            res = run(path, args)
            assert_selftest(res)
            print_result(res)
            print("selftest passed")
        return
    if not args.input:
        raise SystemExit("--input is required unless --selftest is passed")
    res = run(args.input, args)
    stem = os.path.splitext(os.path.basename(args.input))[0] + "_directional_dispersion_mechanism"
    jpath, mpath, tax_path, feat_path, ex_path = write_outputs(res, args.output_dir, stem)
    print_result(res)
    print(f"\nsaved: {jpath}\nsaved: {mpath}\nsaved: {tax_path}\nsaved: {feat_path}\nsaved: {ex_path}")


if __name__ == "__main__":
    main()
