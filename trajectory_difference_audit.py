#!/usr/bin/env python3
"""Trajectory-difference audit for same-problem reasoning samples.

This is the first implementation of the research plan in
`md/insights/2026-07-06-localized-trajectory-difference-methods.md`.

The script is deliberately stricter than earlier one-off audits:

* same-problem paired AUROC is reported wherever a scalar score is used;
* learned classifiers use GroupKFold by problem, never random sample splits;
* dynamic trajectory features are compared against static spread/uncertainty
  and length controls;
* functional tests use within-problem error-minus-correct curves and
  cluster permutation, not isolated uncorrected p-values;
* online-style alarms are calibrated only on training-problem correct samples.

Input is the same multisample `.npz` family produced by `10_sample_and_extract.py`
and used by `multisample_*_audit.py`.
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

from multisample_temporal_rupture_audit import (
    add_mahal_sequences,
    auroc_signed,
    build_base_sequences,
    descriptive,
    finite_json,
    label_policy,
    paired_delta,
    problem_groups,
    safe_mean,
    signal_sign,
    within_pair_auroc,
)


EPS = 1e-12


DEFAULT_CHANNEL_PRIORITY = [
    "cloud_spread",
    "out_entropy",
    "out_committal",
    "tok_entropy",
    "tok_committal",
    "pr_mid",
    "ae_mid",
    "pr_deep",
    "ae_deep",
    "cloud_norm",
    "step_token_count",
]


@dataclass
class PolicyData:
    policy: str
    description: str
    y_err: np.ndarray
    mask: np.ndarray
    contrast_mask: np.ndarray
    groups: List[np.ndarray]
    problem_ids: np.ndarray
    seqs: List[Dict[str, np.ndarray]]
    channels: List[str]
    grid: np.ndarray
    tensor: np.ndarray
    channel_coverage: Dict[str, float]


def as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def robust_center_scale(x: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    if mad > EPS:
        return med, 1.4826 * mad
    sd = float(np.std(a))
    return med, sd if sd > EPS else 1.0


def group_folds(groups: np.ndarray, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    uniq = np.unique(groups)
    if uniq.size < 2:
        return []
    rng = np.random.default_rng(seed)
    uniq = np.array(uniq, copy=True)
    rng.shuffle(uniq)
    k = int(min(max(2, k), uniq.size))
    fold_of = {int(g): i % k for i, g in enumerate(uniq)}
    f = np.array([fold_of[int(g)] for g in groups], dtype=int)
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def contrastive_mask_from_groups(n: int, groups: Sequence[np.ndarray]) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    for idx in groups:
        out[np.asarray(idx, dtype=int)] = True
    return out


def resolve_channels(
    seqs: Sequence[Mapping[str, np.ndarray]],
    requested: Sequence[str],
    *,
    mask: np.ndarray,
    min_coverage: float,
    require: bool,
) -> Tuple[List[str], Dict[str, float]]:
    available = sorted({k for s in seqs for k in s.keys() if k != "step_pos"})
    if requested:
        candidates = list(dict.fromkeys(requested))
    else:
        candidates = [c for c in DEFAULT_CHANNEL_PRIORITY if c in available]
        extras = [c for c in available if c not in candidates]
        candidates.extend(extras)
    coverage: Dict[str, float] = {}
    selected: List[str] = []
    denom = max(1, int(mask.sum()))
    for ch in candidates:
        ok = 0
        for i, s in enumerate(seqs):
            if not mask[i] or ch not in s:
                continue
            v = as_float_array(s[ch])
            if v.size and np.isfinite(v).any():
                ok += 1
        cov = ok / denom
        coverage[ch] = float(cov)
        if cov >= min_coverage:
            selected.append(ch)
    missing_requested = [c for c in requested if c not in selected]
    if require and missing_requested:
        raise SystemExit(
            "required channels missing or below coverage threshold: "
            + ", ".join(f"{c}({coverage.get(c, 0.0):.2f})" for c in missing_requested)
        )
    if not selected:
        raise SystemExit(
            "no usable trajectory channels; lower --min_channel_coverage or pass channels present in the npz"
        )
    return selected, coverage


def interp_to_grid(v: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).reshape(-1)
    out = np.full(grid.shape, np.nan, dtype=np.float64)
    if x.size == 0:
        return out
    pos = np.linspace(0.0, 1.0, x.size)
    m = np.isfinite(x)
    if not m.any():
        return out
    if m.sum() == 1:
        out[:] = float(x[m][0])
        return out
    return np.interp(grid, pos[m], x[m])


def build_tensor(
    seqs: Sequence[Mapping[str, np.ndarray]],
    channels: Sequence[str],
    grid: np.ndarray,
) -> np.ndarray:
    X = np.full((len(seqs), len(channels), len(grid)), np.nan, dtype=np.float64)
    for i, s in enumerate(seqs):
        for c, ch in enumerate(channels):
            if ch not in s:
                continue
            sign = signal_sign(ch)
            X[i, c, :] = interp_to_grid(sign * as_float_array(s[ch]), grid)
    return X


def finite_mean(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(x, axis=axis)


def finite_std(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanstd(x, axis=axis, ddof=1)


def prepare_policy_data(
    data: np.lib.npyio.NpzFile,
    *,
    policy: str,
    bands: Sequence[str],
    requested_channels: Sequence[str],
    min_per_class: int,
    min_channel_coverage: float,
    require_channels: bool,
    grid_size: int,
    include_mahal: bool,
) -> PolicyData:
    problem_ids = data["problem_ids"].astype(int)
    y_err, mask, desc = label_policy(data, policy)
    groups = problem_groups(problem_ids, y_err, mask, min_per_class)
    contrast_mask = contrastive_mask_from_groups(len(problem_ids), groups)
    if not groups:
        raise SystemExit(f"policy {policy!r} has no contrastive same-problem groups")

    seqs = build_base_sequences(data, bands=bands)
    if include_mahal:
        add_mahal_sequences(data, seqs, mask=mask, y_err=y_err, bands=bands)
    channels, coverage = resolve_channels(
        seqs,
        requested_channels,
        mask=contrast_mask,
        min_coverage=min_channel_coverage,
        require=require_channels,
    )
    grid = np.linspace(0.0, 1.0, grid_size)
    tensor = build_tensor(seqs, channels, grid)
    return PolicyData(
        policy=policy,
        description=desc,
        y_err=y_err,
        mask=mask,
        contrast_mask=contrast_mask,
        groups=groups,
        problem_ids=problem_ids,
        seqs=seqs,
        channels=channels,
        grid=grid,
        tensor=tensor,
        channel_coverage=coverage,
    )


def per_problem_delta_tensor(pd: PolicyData) -> Tuple[np.ndarray, List[int]]:
    deltas: List[np.ndarray] = []
    pids: List[int] = []
    for idx in pd.groups:
        idx = np.asarray(idx, dtype=int)
        err = idx[pd.y_err[idx] == 1]
        cor = idx[pd.y_err[idx] == 0]
        if err.size == 0 or cor.size == 0:
            continue
        e = finite_mean(pd.tensor[err], axis=0)
        c = finite_mean(pd.tensor[cor], axis=0)
        d = e - c
        if np.isfinite(d).any():
            deltas.append(d)
            pids.append(int(pd.problem_ids[idx[0]]))
    if not deltas:
        return np.empty((0, len(pd.channels), len(pd.grid))), []
    return np.stack(deltas, axis=0), pids


def t_statistic(D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.sum(np.isfinite(D), axis=0)
    mu = finite_mean(D, axis=0)
    sd = finite_std(D, axis=0)
    t = mu / (sd / np.sqrt(np.maximum(n, 1)))
    t[(n < 2) | ~np.isfinite(t)] = 0.0
    return t, n


def contiguous_clusters(mask: np.ndarray) -> List[Tuple[int, int]]:
    clusters: List[Tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if not bool(mask[i]):
            i += 1
            continue
        j = i + 1
        while j < len(mask) and bool(mask[j]):
            j += 1
        clusters.append((i, j))
        i = j
    return clusters


def max_cluster_mass(t: np.ndarray, threshold: float) -> float:
    best = 0.0
    for c in range(t.shape[0]):
        m = np.abs(t[c]) >= threshold
        for a, b in contiguous_clusters(m):
            best = max(best, float(np.sum(np.abs(t[c, a:b]))))
    return best


def functional_cluster_test(
    pd: PolicyData,
    *,
    n_perm: int,
    threshold: float,
    alpha: float,
    seed: int,
) -> Dict[str, Any]:
    D, pids = per_problem_delta_tensor(pd)
    if D.shape[0] < 3:
        return {
            "n_problem_deltas": int(D.shape[0]),
            "error": "not enough paired problem deltas",
            "clusters": [],
        }
    t_obs, n_obs = t_statistic(D)
    rng = np.random.default_rng(seed)
    null_max = np.zeros(n_perm, dtype=np.float64)
    for b in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=(D.shape[0], 1, 1))
        t_b, _ = t_statistic(D * signs)
        null_max[b] = max_cluster_mass(t_b, threshold)

    clusters: List[Dict[str, Any]] = []
    for c, ch in enumerate(pd.channels):
        m = np.abs(t_obs[c]) >= threshold
        for a, b in contiguous_clusters(m):
            mass = float(np.sum(np.abs(t_obs[c, a:b])))
            p = float((1.0 + np.sum(null_max >= mass)) / (n_perm + 1.0))
            delta = finite_mean(D[:, c, a:b], axis=None)
            clusters.append(
                {
                    "channel": ch,
                    "start_bin": int(a),
                    "end_bin": int(b - 1),
                    "start_u": float(pd.grid[a]),
                    "end_u": float(pd.grid[b - 1]),
                    "center_u": float(np.mean(pd.grid[a:b])),
                    "mass": mass,
                    "p_cluster_fwer": p,
                    "mean_error_minus_correct": float(delta) if np.isfinite(delta) else None,
                    "endpoint_like": bool(np.mean(pd.grid[a:b]) >= 0.80),
                    "significant": bool(p <= alpha),
                    "n_problem_min": int(np.nanmin(n_obs[c, a:b])) if n_obs.size else 0,
                }
            )
    clusters.sort(key=lambda r: (r["p_cluster_fwer"], -r["mass"]))
    mean_delta = finite_mean(D, axis=0)
    return {
        "n_problem_deltas": int(D.shape[0]),
        "problem_ids": pids,
        "threshold_abs_t": float(threshold),
        "n_permutations": int(n_perm),
        "alpha": float(alpha),
        "channels": pd.channels,
        "grid": pd.grid.tolist(),
        "mean_delta_error_minus_correct": mean_delta,
        "t_stat": t_obs,
        "n_problem_by_cell": n_obs,
        "null_max_cluster_mass": {
            "mean": float(np.mean(null_max)),
            "q95": float(np.quantile(null_max, 0.95)),
            "max": float(np.max(null_max)),
        },
        "clusters": clusters,
        "significant_nonendpoint_clusters": [
            r for r in clusters if r["significant"] and not r["endpoint_like"]
        ],
    }


def path_scaler_fit(X: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    C = X.shape[1]
    center = np.zeros(C, dtype=np.float64)
    scale = np.ones(C, dtype=np.float64)
    for c in range(C):
        vals = X[train_idx, c, :].reshape(-1)
        center[c], scale[c] = robust_center_scale(vals)
    return center, scale


def path_scaler_apply(X: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    Z = np.array(X, dtype=np.float64, copy=True)
    for c in range(Z.shape[1]):
        Z[:, c, :] = (Z[:, c, :] - center[c]) / max(scale[c], EPS)
        Z[:, c, :] = np.where(np.isfinite(Z[:, c, :]), Z[:, c, :], 0.0)
    return Z


def path_signature_features(path_ct: np.ndarray, grid: np.ndarray, *, order: int) -> np.ndarray:
    """Compute lightweight path-signature features for one C x T path.

    The implementation uses the exact level-1/level-2 signature of a
    piecewise-linear path represented by its increments.  For order=3, it adds
    stable cubic summary interactions rather than a full C^3 tensor, keeping
    feature count modest for small-sample GroupKFold.
    """
    X = np.asarray(path_ct, dtype=np.float64)
    # Add time as a channel.  This lets signatures encode "when" changes happen.
    P = np.vstack([grid.reshape(1, -1), X])
    dP = np.diff(P, axis=1).T  # steps x channels
    d = P.shape[0]
    feats: List[float] = []
    # Basic summaries help the classifier stay interpretable and give the
    # signature a fair static baseline inside the same representation.
    feats.extend(np.nan_to_num(P[:, 0], nan=0.0).tolist())
    feats.extend(np.nan_to_num(P[:, -1], nan=0.0).tolist())
    feats.extend(np.nan_to_num(np.nanmean(P, axis=1), nan=0.0).tolist())
    feats.extend(np.nan_to_num(np.nanstd(P, axis=1), nan=0.0).tolist())
    if P.shape[1] >= 2:
        slope = P[:, -1] - P[:, 0]
        feats.extend(np.nan_to_num(slope, nan=0.0).tolist())
    s1 = np.zeros(d, dtype=np.float64)
    s2 = np.zeros((d, d), dtype=np.float64)
    for dx in dP:
        prev = s1.copy()
        s1 += dx
        if order >= 2:
            s2 += np.outer(prev, dx) + 0.5 * np.outer(dx, dx)
    feats.extend(s1.tolist())
    if order >= 2:
        feats.extend(s2.reshape(-1).tolist())
    if order >= 3:
        # Compact cubic descriptors: signed total variation and pairwise
        # variation interactions.  This is not a full level-3 signature, but it
        # captures higher-order trajectory shape without exploding dimensions.
        tv = np.sum(np.abs(dP), axis=0)
        signed_tv = np.sum(np.sign(dP) * (dP ** 2), axis=0)
        feats.extend(tv.tolist())
        feats.extend(signed_tv.tolist())
        feats.extend(np.outer(s1, tv).reshape(-1).tolist())
    return np.asarray(feats, dtype=np.float64)


def static_summary_features(X: np.ndarray, grid: np.ndarray, n_steps: np.ndarray) -> np.ndarray:
    N, C, T = X.shape
    out: List[np.ndarray] = []
    late = grid >= 0.60
    early = grid <= 0.40
    for c in range(C):
        A = X[:, c, :]
        out.append(finite_mean(A, axis=1))
        out.append(finite_mean(A[:, late], axis=1))
        out.append(finite_mean(A[:, early], axis=1))
        out.append(np.nanmax(A, axis=1))
        out.append(np.nanstd(A, axis=1))
        out.append(A[:, -1] - A[:, 0])
        out.append(np.nanmean(np.abs(np.diff(A, axis=1)), axis=1))
    out.append(np.asarray(n_steps, dtype=np.float64))
    out.append(np.log1p(np.asarray(n_steps, dtype=np.float64)))
    return np.vstack(out).T


def median_impute_fit(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    med[~np.isfinite(med)] = 0.0
    return med


def median_impute_apply(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    Z = np.asarray(X, dtype=np.float64).copy()
    bad = ~np.isfinite(Z)
    if bad.any():
        Z[bad] = np.take(med, np.where(bad)[1])
    return Z


def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd[sd < EPS] = 1.0
    return mu, sd


def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def logistic_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as e:  # pragma: no cover - dependency issue should be explicit
        raise SystemExit(f"scikit-learn is required for signature audit: {e}")

    y = np.asarray(y, dtype=int)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_meta: List[Dict[str, Any]] = []
    splits = group_folds(groups, folds, seed)
    if not splits:
        return oof, {"folds": [], "error": "not enough groups for GroupKFold"}
    for fold, (tr, te) in enumerate(splits):
        if len(np.unique(y[tr])) < 2:
            fold_meta.append({"fold": fold, "skipped": True, "reason": "single-class train"})
            continue
        med = median_impute_fit(X[tr])
        Xtr = median_impute_apply(X[tr], med)
        Xte = median_impute_apply(X[te], med)
        mu, sd = zscore_fit(Xtr)
        Xtr = zscore_apply(Xtr, mu, sd)
        Xte = zscore_apply(Xte, mu, sd)
        clf = LogisticRegression(
            penalty="l2",
            C=0.5,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=seed + fold,
        )
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
        fold_meta.append(
            {
                "fold": fold,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "n_features": int(X.shape[1]),
            }
        )
    return oof, {"folds": fold_meta}


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score
    except Exception:
        return float("nan")
    m = np.isfinite(score)
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(average_precision_score(y[m], score[m]))


def cluster_boot_increment(
    sf: np.ndarray,
    sb: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    sf = np.asarray(sf, dtype=np.float64)
    sb = np.asarray(sb, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    m = np.isfinite(sf) & np.isfinite(sb)
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return {"point": None, "lo": None, "hi": None, "sig": False}
    point = auroc_signed(sf[m & (y == 1)], sf[m & (y == 0)]) - auroc_signed(
        sb[m & (y == 1)], sb[m & (y == 0)]
    )
    rng = np.random.default_rng(seed)
    ug = np.unique(groups[m])
    by = {g: np.where(m & (groups == g))[0] for g in ug}
    vals: List[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([by[g] for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(
            auroc_signed(sf[idx[y[idx] == 1]], sf[idx[y[idx] == 0]])
            - auroc_signed(sb[idx[y[idx] == 1]], sb[idx[y[idx] == 0]])
        )
    if not vals:
        return {"point": float(point), "lo": None, "hi": None, "sig": False}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "point": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "sig": bool(lo > 0 or hi < 0),
    }


def scalar_score_row(
    name: str,
    score: np.ndarray,
    pd: PolicyData,
    *,
    mask: np.ndarray,
) -> Dict[str, Any]:
    m = mask & np.isfinite(score)
    err = score[m & (pd.y_err == 1)]
    cor = score[m & (pd.y_err == 0)]
    w, pairs = within_pair_auroc(pd.groups, score, pd.y_err)
    return {
        "name": name,
        "n": int(m.sum()),
        "n_error": int((m & (pd.y_err == 1)).sum()),
        "n_correct": int((m & (pd.y_err == 0)).sum()),
        "cross_auroc_error_high": auroc_signed(err, cor),
        "within_pair_auroc_error_high": w,
        "within_pairs": int(pairs),
        "aupr_error": average_precision(pd.y_err[m], score[m]) if m.any() else float("nan"),
        "paired_delta_error_minus_correct": paired_delta(pd.groups, score, pd.y_err),
        "error": descriptive(err),
        "correct": descriptive(cor),
    }


def signature_audit(
    pd: PolicyData,
    *,
    order: int,
    folds: int,
    seed: int,
    n_boot: int,
) -> Dict[str, Any]:
    idx = np.where(pd.contrast_mask)[0]
    if idx.size < 20:
        return {"error": "not enough contrastive samples", "n": int(idx.size)}
    y = pd.y_err[idx]
    groups = pd.problem_ids[idx]
    n_steps = np.array(
        [
            len(next(iter(pd.seqs[i].values()))) if pd.seqs[i] else pd.tensor.shape[2]
            for i in idx
        ],
        dtype=np.float64,
    )

    # Baseline uses static summaries of exactly the same selected channels plus
    # length controls.  This is the key anti-degradation comparison.
    Xraw = pd.tensor[idx]
    Xstatic = static_summary_features(Xraw, pd.grid, n_steps)

    # Signature features are computed fold-locally after path scaling.  Because
    # scaling affects signatures, we build OOF manually below.
    sig_oof = np.full(idx.size, np.nan, dtype=np.float64)
    sig_base_oof = np.full(idx.size, np.nan, dtype=np.float64)
    base_oof, base_meta = logistic_oof(Xstatic, y, groups, folds=folds, seed=seed)

    splits = group_folds(groups, folds, seed + 17)
    sig_meta: List[Dict[str, Any]] = []
    for fold, (tr, te) in enumerate(splits):
        if len(np.unique(y[tr])) < 2:
            sig_meta.append({"fold": fold, "skipped": True, "reason": "single-class train"})
            continue
        center, scale = path_scaler_fit(Xraw, tr)
        Z = path_scaler_apply(Xraw, center, scale)
        Xsig = np.vstack([path_signature_features(Z[j], pd.grid, order=order) for j in range(Z.shape[0])])
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception as e:
            raise SystemExit(f"scikit-learn is required for signature audit: {e}")
        Xtr_sig, Xte_sig = Xsig[tr], Xsig[te]
        med = median_impute_fit(Xtr_sig)
        Xtr_sig = median_impute_apply(Xtr_sig, med)
        Xte_sig = median_impute_apply(Xte_sig, med)
        mu, sd = zscore_fit(Xtr_sig)
        Xtr_sig = zscore_apply(Xtr_sig, mu, sd)
        Xte_sig = zscore_apply(Xte_sig, mu, sd)
        clf_sig = LogisticRegression(
            penalty="l2",
            C=0.5,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=seed + fold,
        )
        clf_sig.fit(Xtr_sig, y[tr])
        sig_oof[te] = clf_sig.predict_proba(Xte_sig)[:, 1]

        Xcombo = np.hstack([Xstatic, Xsig])
        Xtr, Xte = Xcombo[tr], Xcombo[te]
        med2 = median_impute_fit(Xtr)
        Xtr = median_impute_apply(Xtr, med2)
        Xte = median_impute_apply(Xte, med2)
        mu2, sd2 = zscore_fit(Xtr)
        Xtr = zscore_apply(Xtr, mu2, sd2)
        Xte = zscore_apply(Xte, mu2, sd2)
        clf_combo = LogisticRegression(
            penalty="l2",
            C=0.5,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=seed + 1000 + fold,
        )
        clf_combo.fit(Xtr, y[tr])
        sig_base_oof[te] = clf_combo.predict_proba(Xte)[:, 1]
        sig_meta.append(
            {
                "fold": fold,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "n_signature_features": int(Xsig.shape[1]),
                "n_static_features": int(Xstatic.shape[1]),
            }
        )

    # Expand scores back to full sample indexing for same-problem metrics.
    full_base = np.full(len(pd.y_err), np.nan)
    full_sig = np.full(len(pd.y_err), np.nan)
    full_combo = np.full(len(pd.y_err), np.nan)
    full_base[idx] = base_oof
    full_sig[idx] = sig_oof
    full_combo[idx] = sig_base_oof
    rows = [
        scalar_score_row("static_baseline", full_base, pd, mask=pd.contrast_mask),
        scalar_score_row("path_signature", full_sig, pd, mask=pd.contrast_mask),
        scalar_score_row("static_plus_signature", full_combo, pd, mask=pd.contrast_mask),
    ]
    inc_sig = cluster_boot_increment(full_sig, full_base, pd.y_err, pd.problem_ids, n_boot=n_boot, seed=seed)
    inc_combo = cluster_boot_increment(full_combo, full_base, pd.y_err, pd.problem_ids, n_boot=n_boot, seed=seed + 1)
    return {
        "order": int(order),
        "folds": int(folds),
        "n_samples": int(idx.size),
        "n_groups": int(np.unique(groups).size),
        "channels": pd.channels,
        "models": rows,
        "increment_over_static": {
            "path_signature": inc_sig,
            "static_plus_signature": inc_combo,
        },
        "base_cv_meta": base_meta,
        "signature_cv_meta": sig_meta,
        "anti_degradation": {
            "baseline_includes_length": True,
            "baseline_uses_same_channels_static_summaries": True,
            "split": "GroupKFold by problem_id",
        },
    }


def conformal_quantile(x: np.ndarray, eps: float) -> float:
    a = np.sort(np.asarray(x, dtype=np.float64)[np.isfinite(x)])
    if a.size == 0:
        return float("nan")
    # Split-conformal finite-sample conservative rank.
    k = int(math.ceil((a.size + 1) * (1.0 - eps))) - 1
    k = min(max(k, 0), a.size - 1)
    return float(a[k])


def reference_center_scale(
    X: np.ndarray,
    ref_idx: np.ndarray,
    *,
    robust: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    C, T = X.shape[1], X.shape[2]
    center = np.zeros((C, T), dtype=np.float64)
    scale = np.ones((C, T), dtype=np.float64)
    for c in range(C):
        for t in range(T):
            vals = X[ref_idx, c, t]
            if robust:
                center[c, t], scale[c, t] = robust_center_scale(vals)
            else:
                a = vals[np.isfinite(vals)]
                if a.size:
                    center[c, t] = float(np.mean(a))
                    sd = float(np.std(a))
                    scale[c, t] = sd if sd > EPS else 1.0
    return center, scale


def alarm_scores(X: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    Z = (X - center[None, :, :]) / np.maximum(scale[None, :, :], EPS)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    return np.mean(Z ** 2, axis=1)  # N x T


def conformal_alarm_audit(
    pd: PolicyData,
    *,
    folds: int,
    seed: int,
    eps_values: Sequence[float],
    robust: bool,
) -> Dict[str, Any]:
    idx = np.where(pd.contrast_mask)[0]
    y = pd.y_err[idx]
    groups = pd.problem_ids[idx]
    splits = group_folds(groups, folds, seed)
    if not splits:
        return {"error": "not enough groups for conformal alarm"}
    chain_max = {eps: np.full(len(pd.y_err), np.nan) for eps in eps_values}
    first_pos = {eps: np.full(len(pd.y_err), np.nan) for eps in eps_values}
    thresholds = {eps: [] for eps in eps_values}
    raw_max = np.full(len(pd.y_err), np.nan)
    raw_argpos = np.full(len(pd.y_err), np.nan)
    for fold, (tr_rel, te_rel) in enumerate(splits):
        tr = idx[tr_rel]
        te = idx[te_rel]
        ref = tr[pd.y_err[tr] == 0]
        if ref.size < 5:
            continue
        center, scale = reference_center_scale(pd.tensor, ref, robust=robust)
        S_train = alarm_scores(pd.tensor[tr], center, scale)
        S_test = alarm_scores(pd.tensor[te], center, scale)
        train_correct = tr[pd.y_err[tr] == 0]
        S_cal = alarm_scores(pd.tensor[train_correct], center, scale)
        cal_max = np.nanmax(S_cal, axis=1)
        raw_max[te] = np.nanmax(S_test, axis=1)
        raw_argpos[te] = np.nanargmax(S_test, axis=1) / max(1, S_test.shape[1] - 1)
        for eps in eps_values:
            thr = conformal_quantile(cal_max, eps)
            thresholds[eps].append({"fold": int(fold), "threshold": thr, "n_cal_correct": int(train_correct.size)})
            if not np.isfinite(thr):
                continue
            for local_j, global_i in enumerate(te):
                s = S_test[local_j]
                chain_max[eps][global_i] = float(np.nanmax(s))
                hit = np.where(s > thr)[0]
                if hit.size:
                    first_pos[eps][global_i] = float(hit[0] / max(1, s.size - 1))
    out: Dict[str, Any] = {
        "reference": "training-problem correct samples only",
        "robust_reference": bool(robust),
        "raw_score_cross_auroc": scalar_score_row("alarm_raw_max", raw_max, pd, mask=pd.contrast_mask),
        "raw_argpos_error": descriptive(raw_argpos[pd.contrast_mask & (pd.y_err == 1)]),
        "raw_argpos_correct": descriptive(raw_argpos[pd.contrast_mask & (pd.y_err == 0)]),
        "eps": {},
    }
    for eps in eps_values:
        fired = np.isfinite(first_pos[eps])
        m_err = pd.contrast_mask & (pd.y_err == 1)
        m_cor = pd.contrast_mask & (pd.y_err == 0)
        fpr = float(np.mean(fired[m_cor])) if m_cor.any() else float("nan")
        recall = float(np.mean(fired[m_err])) if m_err.any() else float("nan")
        pos_all = first_pos[eps][fired & pd.contrast_mask]
        endpoint_frac = float(np.mean(pos_all >= 0.80)) if pos_all.size else float("nan")
        out["eps"][str(eps)] = {
            "target_fpr": float(eps),
            "empirical_fpr": fpr,
            "error_recall": recall,
            "endpoint_alarm_fraction": endpoint_frac,
            "first_alarm_pos_error": descriptive(first_pos[eps][m_err & fired]),
            "first_alarm_pos_correct": descriptive(first_pos[eps][m_cor & fired]),
            "thresholds": thresholds[eps],
            "anti_degradation": {
                "calibration_excludes_test_problems": True,
                "calibration_uses_correct_only": True,
                "endpoint_fraction_reported": True,
            },
        }
    return out


def write_markdown(path: str, res: Mapping[str, Any]) -> None:
    lines: List[str] = []
    meta = res["meta"]
    lines.append("# Trajectory Difference Audit\n")
    lines.append(f"- input: `{meta['input']}`")
    lines.append(f"- policy: `{meta['policy']}`")
    lines.append(f"- channels: `{', '.join(meta['channels'])}`")
    lines.append(f"- contrastive problems: {meta['n_contrastive_problems']}")
    lines.append("")

    fn = res["functional_test"]
    lines.append("## Functional Cluster Test\n")
    lines.append(
        f"Problem-level paired deltas: {fn.get('n_problem_deltas')}; permutations: {fn.get('n_permutations')}"
    )
    clusters = fn.get("clusters", [])[:12]
    if clusters:
        lines.append("")
        lines.append("| channel | u-start | u-end | mean delta | mass | p | endpoint |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for r in clusters:
            lines.append(
                f"| {r['channel']} | {r['start_u']:.2f} | {r['end_u']:.2f} | "
                f"{(r['mean_error_minus_correct'] if r['mean_error_minus_correct'] is not None else float('nan')):.4g} | "
                f"{r['mass']:.3f} | {r['p_cluster_fwer']:.4f} | {r['endpoint_like']} |"
            )
    else:
        lines.append("No supra-threshold clusters.")
    lines.append("")

    lines.append("## Path Signature Classifier\n")
    lines.append("| model | cross AUROC | within AUROC | AUPR | n |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in res["signature_audit"].get("models", []):
        lines.append(
            f"| {row['name']} | {row['cross_auroc_error_high']:.3f} | "
            f"{row['within_pair_auroc_error_high']:.3f} | {row['aupr_error']:.3f} | {row['n']} |"
        )
    inc = res["signature_audit"].get("increment_over_static", {})
    lines.append("")
    lines.append("Increment over static baseline:")
    for k, v in inc.items():
        lines.append(f"- `{k}`: point={v.get('point')} CI=[{v.get('lo')}, {v.get('hi')}] sig={v.get('sig')}")
    lines.append("")

    lines.append("## Conformal Alarm\n")
    raw = res["conformal_alarm"].get("raw_score_cross_auroc", {})
    if raw:
        lines.append(
            f"Raw max alarm score: cross AUROC={raw.get('cross_auroc_error_high'):.3f}, "
            f"within AUROC={raw.get('within_pair_auroc_error_high'):.3f}"
        )
    lines.append("")
    lines.append("| eps | empirical FPR | error recall | endpoint alarm frac |")
    lines.append("|---:|---:|---:|---:|")
    for eps, row in res["conformal_alarm"].get("eps", {}).items():
        lines.append(
            f"| {eps} | {row['empirical_fpr']:.3f} | {row['error_recall']:.3f} | "
            f"{row['endpoint_alarm_fraction']:.3f} |"
        )
    lines.append("")
    lines.append("## Anti-Degradation Checks\n")
    lines.append("- Same-problem paired AUROC is reported for scalar scores.")
    lines.append("- Classifiers use GroupKFold by `problem_ids`.")
    lines.append("- Signature models are compared to a static baseline using the same channels plus length controls.")
    lines.append("- Conformal alarms calibrate on training-problem correct samples only.")
    lines.append("- Endpoint alarm fraction is reported to catch late-only detectors.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    requested = [x.strip() for x in args.channels.split(",") if x.strip()]
    bands = [x.strip() for x in args.bands.split(",") if x.strip()]
    pd = prepare_policy_data(
        data,
        policy=args.policy,
        bands=bands,
        requested_channels=requested,
        min_per_class=args.min_per_class,
        min_channel_coverage=args.min_channel_coverage,
        require_channels=args.require_channels,
        grid_size=args.grid,
        include_mahal=args.include_mahal,
    )
    functional = functional_cluster_test(
        pd,
        n_perm=args.permutations,
        threshold=args.cluster_t,
        alpha=args.alpha,
        seed=args.seed,
    )
    sig = signature_audit(
        pd,
        order=args.signature_order,
        folds=args.folds,
        seed=args.seed,
        n_boot=args.bootstrap,
    )
    eps_values = [float(x) for x in args.alarm_eps.split(",") if x.strip()]
    alarm = conformal_alarm_audit(
        pd,
        folds=args.folds,
        seed=args.seed,
        eps_values=eps_values,
        robust=not args.mean_alarm_reference,
    )
    miss_by_channel: Dict[str, float] = {}
    for ci, ch in enumerate(pd.channels):
        block = pd.tensor[pd.contrast_mask, ci, :]
        miss_by_channel[ch] = float((~np.isfinite(block)).mean()) if block.size else 1.0
    return {
        "meta": {
            "input": os.path.abspath(path),
            "basename": os.path.basename(path),
            "policy": args.policy,
            "description": pd.description,
            "n_samples_policy": int(pd.mask.sum()),
            "n_contrastive_samples": int(pd.contrast_mask.sum()),
            "n_error_contrastive": int(pd.y_err[pd.contrast_mask].sum()),
            "n_correct_contrastive": int(pd.contrast_mask.sum() - pd.y_err[pd.contrast_mask].sum()),
            "n_contrastive_problems": int(len(pd.groups)),
            "channels": pd.channels,
            "channel_coverage": pd.channel_coverage,
            "trajectory_missing_fraction_by_channel": miss_by_channel,
            "grid_size": int(args.grid),
            "bands": bands,
            "include_mahal": bool(args.include_mahal),
            "notes": {
                "no_random_splits": "All learned/OOD-style estimates use problem-grouped folds.",
                "no_oracle_same_problem_tube": "Same-problem grouping is used for evaluation, not for train-time test leakage.",
            },
        },
        "functional_test": functional,
        "signature_audit": sig,
        "conformal_alarm": alarm,
    }


def write_outputs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    jpath = os.path.join(output_dir, stem + ".json")
    mpath = os.path.join(output_dir, stem + ".md")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(finite_json(res), f, ensure_ascii=False, indent=2)
    write_markdown(mpath, finite_json(res))
    return jpath, mpath


def make_selftest(path: str, *, seed: int = 0, n_problems: int = 48, samples_per_problem: int = 6) -> None:
    rng = np.random.default_rng(seed)
    d = 24
    rows: List[Dict[str, Any]] = []
    base_dir = rng.normal(size=d)
    base_dir /= np.linalg.norm(base_dir)
    for p in range(n_problems):
        # Every problem has at least two correct and two error samples, with a
        # problem-specific offset so cross-problem difficulty alone is not enough.
        problem_vec = rng.normal(scale=0.25, size=d)
        for s in range(samples_per_problem):
            is_err = int(s >= samples_per_problem // 2)
            T = int(rng.integers(6, 11))
            sizes = rng.integers(4, 8, size=T)
            clouds: List[np.ndarray] = []
            entropy = np.zeros(T, dtype=np.float64)
            committal = np.zeros(T, dtype=np.float64)
            pr = np.zeros((T, 33), dtype=np.float64)
            ae = np.zeros((T, 33), dtype=np.float64)
            for t in range(T):
                u = t / max(1, T - 1)
                # Correct trajectories are smooth and become slightly more
                # confident.  Error trajectories undergo a mid/late dynamic
                # change: entropy rebound and token-cloud spread increase.
                err_phase = is_err and u >= 0.45
                noise_scale = 0.12 + (0.34 if err_phase else 0.0) + 0.03 * rng.random()
                center = base_dir + problem_vec + 0.05 * u * rng.normal(size=d)
                if err_phase:
                    center = center + 0.15 * (u - 0.45) * rng.normal(size=d)
                H = center[None, :] + noise_scale * rng.normal(size=(int(sizes[t]), d))
                clouds.append(H[:, None, :].astype(np.float32))
                entropy[t] = 0.55 - 0.10 * u + rng.normal(scale=0.025)
                committal[t] = 0.20 - 0.03 * u + rng.normal(scale=0.015)
                if err_phase:
                    entropy[t] += 0.32 + 0.30 * (u - 0.45)
                    committal[t] += 0.16 + 0.15 * (u - 0.45)
                pr[t, :] = 2.0 + 0.1 * rng.normal(size=33) + (0.35 if err_phase else 0.0)
                ae[t, :] = 0.9 + 0.08 * rng.normal(size=33) + (0.28 if err_phase else 0.0)
            rows.append(
                {
                    "problem_id": p,
                    "sample_idx": s,
                    "is_err": is_err,
                    "format_ok": True,
                    "n_steps": T,
                    "cloud_sizes": sizes.astype(np.int32),
                    "sv_clouds": np.concatenate(clouds, axis=0).astype(np.float32),
                    "sv_out_entropy": entropy.astype(np.float32),
                    "sv_out_committal": committal.astype(np.float32),
                    "sv_pr_step_exp": pr.astype(np.float32),
                    "sv_ae_step_exp": ae.astype(np.float32),
                }
            )
    np.savez(
        path,
        problem_ids=np.asarray([r["problem_id"] for r in rows], dtype=np.int32),
        sample_idx=np.asarray([r["sample_idx"] for r in rows], dtype=np.int32),
        is_correct=(1 - np.asarray([r["is_err"] for r in rows], dtype=np.int32)),
        is_correct_strict=(1 - np.asarray([r["is_err"] for r in rows], dtype=np.int32)),
        format_ok=np.asarray([r["format_ok"] for r in rows], dtype=bool),
        n_steps=np.asarray([r["n_steps"] for r in rows], dtype=np.int32),
        cloud_sizes=np.asarray([r["cloud_sizes"] for r in rows], dtype=object),
        sv_clouds=np.asarray([r["sv_clouds"] for r in rows], dtype=object),
        sv_out_entropy=np.asarray([r["sv_out_entropy"] for r in rows], dtype=object),
        sv_out_committal=np.asarray([r["sv_out_committal"] for r in rows], dtype=object),
        sv_pr_step_exp=np.asarray([r["sv_pr_step_exp"] for r in rows], dtype=object),
        sv_ae_step_exp=np.asarray([r["sv_ae_step_exp"] for r in rows], dtype=object),
        model_name=np.asarray("selftest"),
        prompt_style=np.asarray("selftest"),
        step_split=np.asarray("selftest"),
    )


def assert_selftest(res: Mapping[str, Any]) -> None:
    clusters = res["functional_test"].get("significant_nonendpoint_clusters", [])
    if not clusters:
        raise SystemExit("selftest failed: functional test found no significant non-endpoint cluster")
    rows = {r["name"]: r for r in res["signature_audit"].get("models", [])}
    sig_auc = rows.get("path_signature", {}).get("within_pair_auroc_error_high", float("nan"))
    combo_auc = rows.get("static_plus_signature", {}).get("within_pair_auroc_error_high", float("nan"))
    if not (np.isfinite(sig_auc) and sig_auc >= 0.80):
        raise SystemExit(f"selftest failed: path signature within AUROC too weak ({sig_auc})")
    if not (np.isfinite(combo_auc) and combo_auc >= sig_auc - 0.05):
        raise SystemExit(f"selftest failed: static+signature degraded unexpectedly ({combo_auc} vs {sig_auc})")
    eps_rows = res["conformal_alarm"].get("eps", {})
    recall = max(float(v.get("error_recall", 0.0)) for v in eps_rows.values()) if eps_rows else 0.0
    if recall < 0.60:
        raise SystemExit(f"selftest failed: conformal alarm recall too weak ({recall})")


def print_summary(res: Mapping[str, Any]) -> None:
    meta = res["meta"]
    print(
        f"\ntrajectory-difference audit | {meta['basename']} | policy={meta['policy']} | "
        f"contrastive problems={meta['n_contrastive_problems']} channels={','.join(meta['channels'])}"
    )
    clusters = res["functional_test"].get("clusters", [])[:5]
    print("\nFunctional clusters:")
    if not clusters:
        print("  none")
    for r in clusters:
        print(
            f"  {r['channel']:18s} u={r['start_u']:.2f}-{r['end_u']:.2f} "
            f"p={r['p_cluster_fwer']:.4f} mass={r['mass']:.2f} endpoint={r['endpoint_like']}"
        )
    print("\nSignature models:")
    for r in res["signature_audit"].get("models", []):
        print(
            f"  {r['name']:22s} cross={r['cross_auroc_error_high']:.3f} "
            f"within={r['within_pair_auroc_error_high']:.3f} aupr={r['aupr_error']:.3f}"
        )
    print("\nConformal alarm:")
    for eps, r in res["conformal_alarm"].get("eps", {}).items():
        print(
            f"  eps={eps:>4s} fpr={r['empirical_fpr']:.3f} "
            f"recall={r['error_recall']:.3f} endpoint_frac={r['endpoint_alarm_fraction']:.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="same-problem multisample npz")
    ap.add_argument("--policy", default="answer_format_ok", choices=["answer", "strict", "answer_format_ok"])
    ap.add_argument("--channels", default="", help="comma-separated channels; default uses available priority list")
    ap.add_argument("--require_channels", action="store_true", help="fail if any requested channel is absent/low coverage")
    ap.add_argument("--min_channel_coverage", type=float, default=0.50)
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--bands", default="mid,deep")
    ap.add_argument("--include_mahal", action="store_true", help="include global policy-fitted mahal channels; off by default")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--cluster_t", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--signature_order", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--alarm_eps", default="0.05,0.20")
    ap.add_argument("--mean_alarm_reference", action="store_true", help="use mean/std instead of robust median/MAD")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs/trajectory_difference_audit")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "trajectory_difference_selftest.npz")
            make_selftest(path, seed=args.seed)
            res = run(path, args)
            assert_selftest(res)
            stem = "trajectory_difference_selftest"
            jpath, mpath = write_outputs(res, args.output_dir, stem)
            print_summary(res)
            print(f"\nselftest passed; saved: {jpath} and {mpath}")
        return

    if not args.input:
        raise SystemExit("pass --input or --selftest")
    res = run(args.input, args)
    stem = os.path.splitext(os.path.basename(args.input))[0] + f"_{args.policy}"
    jpath, mpath = write_outputs(res, args.output_dir, stem)
    print_summary(res)
    print(f"\nsaved: {jpath}\nsaved: {mpath}")


if __name__ == "__main__":
    main()
