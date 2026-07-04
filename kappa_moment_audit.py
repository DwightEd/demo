#!/usr/bin/env python3
"""Kappa-conditioned second-moment audit.

Question:
  At matched first-moment concentration (kappa), does the residual scatter
  spectrum of a step's unit-token cloud distinguish correct structured
  branching from error-like diffuse confusion?

This is deliberately stricter than "does a second-moment score beat kappa":
it evaluates increments within kappa strata and over baselines that include
kappa, log length, and optionally entropy. It also has a synthetic self-test
where kappa is intentionally uninformative but scatter shape is decisive.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - exercised by CLI environment
    raise SystemExit("kappa_moment_audit.py needs scikit-learn") from exc


EPS = 1e-12


@dataclass
class StepRecord:
    chain: int
    group: int
    step: int
    y: int
    n_tok: int
    features: Dict[str, float]


# Shape features avoid deterministic kappa transforms where possible. The
# stricter kappa-bin baseline further guards against nonlinear kappa leakage.
SHAPE_FEATURES = [
    "A_e1",
    "A_e2",
    "A_e3",
    "A_e4",
    "A_e5",
    "A_gap12",
    "A_eff_rank",
    "A_entropy",
    "C_f1",
    "C_f2",
    "C_f3",
    "C_f4",
    "C_f5",
    "C_gap12_frac",
    "C_eff_rank",
    "C_entropy",
    "axis_residual_frac",
    "bipolarity",
]

ALL_MOMENT_FEATURES = [
    "kappa2",
    "trace_C",
    "A_lam1",
    "A_lam2",
    "A_gap12",
    "A_eff_rank",
    "A_entropy",
    "C_lam1",
    "C_lam2",
    "C_gap12",
    "C_eff_rank",
    "C_entropy",
    "C_lam1_frac",
    "C_gap12_frac",
    "axis_residual",
    "axis_residual_frac",
    "bipolarity",
] + [f"A_e{i}" for i in range(1, 6)] + [f"C_f{i}" for i in range(1, 6)]


def finite_json(obj):
    if isinstance(obj, dict):
        return {str(k): finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return finite_json(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def auroc(score: Sequence[float], y: Sequence[int]) -> float:
    s = np.asarray(score, float)
    yy = np.asarray(y, int)
    m = np.isfinite(s)
    s, yy = s[m], yy[m]
    pos = int((yy == 1).sum())
    neg = int((yy == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[yy == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def bdir(a: float) -> float:
    return float(max(a, 1.0 - a)) if np.isfinite(a) else float("nan")


def bucket_auc(score: Sequence[float], y: Sequence[int], length: Sequence[float], nb: int = 5) -> float:
    s = np.asarray(score, float)
    yy = np.asarray(y, int)
    nt = np.asarray(length, float)
    m = np.isfinite(s) & np.isfinite(nt)
    s, yy, nt = s[m], yy[m], nt[m]
    if len(s) == 0:
        return float("nan")
    edges = np.quantile(nt, np.linspace(0.0, 1.0, nb + 1))
    edges[-1] += 1e-9
    bins = np.clip(np.digitize(nt, edges[1:-1]), 0, nb - 1)
    num = den = 0.0
    for b in range(nb):
        mm = bins == b
        pos = int((yy[mm] == 1).sum())
        neg = int((yy[mm] == 0).sum())
        a = bdir(auroc(s[mm], yy[mm]))
        if np.isfinite(a) and pos and neg:
            num += a * pos * neg
            den += pos * neg
    return float(num / den) if den else float("nan")


def entropy_rank(eigs: np.ndarray) -> Tuple[float, float]:
    lam = np.asarray(eigs, float)
    lam = lam[lam > EPS]
    if lam.size == 0:
        return float("nan"), float("nan")
    p = lam / lam.sum()
    ent = float(-(p * np.log(p)).sum())
    return float(np.exp(ent)), ent


def exp_weights(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(n, 1), float)
    w = np.exp(np.arange(n, dtype=float) / max(n - 1, 1))
    return w / w.sum()


def moment_features(H: np.ndarray, *, min_tokens: int = 3) -> Optional[Dict[str, float]]:
    """Unit-token directional moment decomposition for one step."""
    H = np.asarray(H, np.float64)
    if H.ndim != 2 or H.shape[0] < min_tokens:
        return None
    norms = np.linalg.norm(H, axis=1)
    ok = norms > 1e-9
    if int(ok.sum()) < min_tokens:
        return None
    U = H[ok] / norms[ok, None]
    n = U.shape[0]
    w = exp_weights(n)
    sw = np.sqrt(w)

    m = w @ U
    kappa2 = float(np.dot(m, m))
    kappa2 = float(np.clip(kappa2, 0.0, 1.0))
    kappa = float(math.sqrt(kappa2))

    B = sw[:, None] * U
    K = (B @ B.T + (B @ B.T).T) * 0.5
    evA = np.linalg.eigvalsh(K)[::-1]
    evA = np.clip(evA, 0.0, None)
    if evA.sum() > EPS:
        evA = evA / evA.sum()

    centered = U - m[None, :]
    Bc = sw[:, None] * centered
    Kc = (Bc @ Bc.T + (Bc @ Bc.T).T) * 0.5
    evC = np.linalg.eigvalsh(Kc)[::-1]
    evC = np.clip(evC, 0.0, None)
    trace_C = float(evC.sum())
    trace_C_theory = float(max(0.0, 1.0 - kappa2))
    if abs(trace_C - trace_C_theory) > 1e-6:
        # Numerical drift can be slightly larger for low precision inputs.
        trace_C = trace_C_theory
    evC_frac = evC / trace_C if trace_C > EPS else np.zeros_like(evC)

    A_eff, A_ent = entropy_rank(evA)
    C_eff, C_ent = entropy_rank(evC_frac)

    def ev(arr: np.ndarray, idx: int) -> float:
        return float(arr[idx]) if idx < len(arr) else 0.0

    out: Dict[str, float] = {
        "n_tok": float(n),
        "kappa": kappa,
        "kappa2": kappa2,
        "trace_C": trace_C,
        "A_lam1": ev(evA, 0),
        "A_lam2": ev(evA, 1),
        "A_gap12": ev(evA, 0) - ev(evA, 1),
        "A_eff_rank": A_eff,
        "A_entropy": A_ent,
        "C_lam1": ev(evC, 0),
        "C_lam2": ev(evC, 1),
        "C_gap12": ev(evC, 0) - ev(evC, 1),
        "C_eff_rank": C_eff,
        "C_entropy": C_ent,
        "C_lam1_frac": ev(evC_frac, 0),
        "C_gap12_frac": ev(evC_frac, 0) - ev(evC_frac, 1),
        "axis_residual": max(0.0, ev(evA, 0) - kappa2),
        "axis_residual_frac": max(0.0, ev(evA, 0) - kappa2) / max(trace_C, EPS),
        "bipolarity": ev(evA, 0) * (1.0 - kappa),
    }
    for i in range(5):
        out[f"A_e{i + 1}"] = ev(evA, i)
        out[f"C_f{i + 1}"] = ev(evC_frac, i)
    return out


def impute_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float).copy()
    if X.ndim == 1:
        X = X[:, None]
    for j in range(X.shape[1]):
        col = X[:, j]
        good = np.isfinite(col)
        fill = float(np.nanmean(col[good])) if good.any() else 0.0
        col[~good] = fill
        X[:, j] = col
    return X


def oof_logit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5) -> np.ndarray:
    X = impute_matrix(X)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    pred = np.full(len(y), np.nan)
    uniq = np.unique(groups)
    n_splits = min(int(folds), len(uniq))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return pred
    splitter = GroupKFold(n_splits=n_splits)
    for tr, te in splitter.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        )
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return pred


def bootstrap_increment(
    full: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> Dict[str, float]:
    full = np.asarray(full, float)
    base = np.asarray(base, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    valid = np.isfinite(full) & np.isfinite(base)
    if valid.sum() == 0 or len(np.unique(y[valid])) < 2:
        return {"point": float("nan"), "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "sig": False}
    point = auroc(full[valid], y[valid]) - auroc(base[valid], y[valid])
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups[valid])
    by_group = {g: np.where(valid & (groups == g))[0] for g in uniq}
    vals = []
    for _ in range(int(n_boot)):
        sample_groups = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in sample_groups])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auroc(full[idx], y[idx]) - auroc(base[idx], y[idx]))
    if not vals:
        return {"point": float(point), "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "sig": False}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "point": float(point),
        "mean": float(np.mean(vals)),
        "lo": float(lo),
        "hi": float(hi),
        "sig": bool(lo > 0.0 or hi < 0.0),
    }


def matrix_from(records: List[StepRecord], names: Sequence[str]) -> np.ndarray:
    X = np.full((len(records), len(names)), np.nan)
    for i, r in enumerate(records):
        for j, nm in enumerate(names):
            if nm == "logN":
                X[i, j] = math.log1p(r.n_tok)
            elif nm == "n_tok":
                X[i, j] = r.n_tok
            else:
                X[i, j] = r.features.get(nm, float("nan"))
    return X


def vector_from(records: List[StepRecord], name: str) -> np.ndarray:
    return matrix_from(records, [name])[:, 0]


def kappa_bins(kappa: np.ndarray, nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(kappa)
    edges = np.quantile(kappa[finite], np.linspace(0.0, 1.0, nbins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        bins = np.zeros(len(kappa), int)
        return bins, edges
    edges[-1] += 1e-9
    bins = np.clip(np.digitize(kappa, edges[1:-1]), 0, len(edges) - 2)
    return bins, edges


def one_hot(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, int)
    if len(vals) == 0:
        return np.zeros((0, 0))
    k = int(vals.max()) + 1
    out = np.zeros((len(vals), k), float)
    out[np.arange(len(vals)), vals] = 1.0
    return out


def describe_subset(mask: np.ndarray, y: np.ndarray) -> Dict[str, int]:
    return {
        "n": int(mask.sum()),
        "n_error": int(y[mask].sum()),
        "n_correct": int((y[mask] == 0).sum()),
    }


def eval_feature_table(
    records: List[StepRecord],
    mask: np.ndarray,
    feature_names: Sequence[str],
    *,
    max_features: int = 12,
) -> List[Dict[str, float]]:
    y = np.array([r.y for r in records], int)
    nt = np.array([r.n_tok for r in records], float)
    rows = []
    for nm in feature_names:
        s = vector_from(records, nm)
        mm = mask & np.isfinite(s)
        if mm.sum() < 20 or len(np.unique(y[mm])) < 2:
            continue
        raw = auroc(s[mm], y[mm])
        rows.append(
            {
                "feature": nm,
                "auroc_bestdir": bdir(raw),
                "raw_auroc_high_is_error": raw,
                "bucket": bucket_auc(s[mm], y[mm], nt[mm]),
                "n": int(mm.sum()),
            }
        )
    rows.sort(key=lambda r: (np.nan_to_num(r["auroc_bestdir"], nan=-1.0)), reverse=True)
    return rows[:max_features]


def eval_increment(
    records: List[StepRecord],
    mask: np.ndarray,
    *,
    base_names: Sequence[str],
    add_names: Sequence[str],
    folds: int,
    n_boot: int,
    seed: int,
) -> Dict[str, object]:
    y = np.array([r.y for r in records], int)
    groups = np.array([r.group for r in records])
    nt = np.array([r.n_tok for r in records], float)
    if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
        return {"available": False, **describe_subset(mask, y)}
    Xb = matrix_from(records, base_names)[mask]
    Xa = matrix_from(records, list(base_names) + list(add_names))[mask]
    yy = y[mask]
    gg = groups[mask]
    sb = oof_logit(Xb, yy, gg, folds)
    sf = oof_logit(Xa, yy, gg, folds)
    valid = np.isfinite(sb) & np.isfinite(sf)
    if valid.sum() < 30 or len(np.unique(yy[valid])) < 2:
        return {"available": False, "reason": "insufficient_oof", **describe_subset(mask, y)}
    inc = bootstrap_increment(sf, sb, yy, gg, n_boot=n_boot, seed=seed)
    return {
        "available": True,
        **describe_subset(mask, y),
        "base_features": list(base_names),
        "add_features": list(add_names),
        "base_auroc": auroc(sb[valid], yy[valid]),
        "full_auroc": auroc(sf[valid], yy[valid]),
        "base_bucket": bucket_auc(sb[valid], yy[valid], nt[mask][valid]),
        "full_bucket": bucket_auc(sf[valid], yy[valid], nt[mask][valid]),
        "increment": inc,
    }


def analyze_records(
    records: List[StepRecord],
    *,
    folds: int = 5,
    n_boot: int = 500,
    nbins: int = 5,
    low_fracs: Sequence[float] = (0.1, 0.2, 0.3),
    include_entropy: bool = True,
    seed: int = 0,
) -> Dict[str, object]:
    if not records:
        raise SystemExit("no usable step records")

    y = np.array([r.y for r in records], int)
    nt = np.array([r.n_tok for r in records], float)
    kappa = vector_from(records, "kappa")
    groups = np.array([r.group for r in records])
    all_mask = np.ones(len(records), bool)

    bins, edges = kappa_bins(kappa, nbins)
    bin_oh = one_hot(bins)
    # Store bin features into records for simple matrix extraction.
    for i, r in enumerate(records):
        r.features["kappa_bin"] = float(bins[i])
        for b in range(bin_oh.shape[1]):
            r.features[f"kbin_{b}"] = float(bin_oh[i, b])

    shape_features = [f for f in SHAPE_FEATURES if f in records[0].features]
    all_features = [f for f in ALL_MOMENT_FEATURES if f in records[0].features]
    entropy_names = []
    if include_entropy:
        for nm in ("U_D", "U_C"):
            v = vector_from(records, nm)
            if np.isfinite(v).sum() >= 20:
                entropy_names.append(nm)

    base_k_len = ["kappa", "logN"]
    base_entropy = base_k_len + entropy_names
    bin_names = [f"kbin_{b}" for b in range(bin_oh.shape[1])]
    base_bin = bin_names + ["logN"]
    base_bin_entropy = base_bin + entropy_names

    result: Dict[str, object] = {
        "n_steps": int(len(records)),
        "n_error": int(y.sum()),
        "n_groups": int(len(np.unique(groups))),
        "kappa_edges": edges.tolist(),
        "shape_features": shape_features,
        "all_moment_features": all_features,
        "entropy_features": entropy_names,
        "overall": {},
        "by_kappa_bucket": [],
        "low_kappa_rescue": [],
        "feature_correlations": {},
    }

    result["overall"]["kappa"] = {
        "auroc_bestdir": bdir(auroc(-kappa, y)),
        "raw_auroc_low_is_error": auroc(-kappa, y),
        "bucket": bucket_auc(-kappa, y, nt),
    }
    result["overall"]["top_shape_features"] = eval_feature_table(records, all_mask, shape_features)
    result["overall"]["shape_over_kappa_logn"] = eval_increment(
        records, all_mask, base_names=base_k_len, add_names=shape_features,
        folds=folds, n_boot=n_boot, seed=seed,
    )
    result["overall"]["shape_over_kappa_bin_logn"] = eval_increment(
        records, all_mask, base_names=base_bin, add_names=shape_features,
        folds=folds, n_boot=n_boot, seed=seed + 1,
    )
    if entropy_names:
        result["overall"]["shape_over_kappa_logn_entropy"] = eval_increment(
            records, all_mask, base_names=base_entropy, add_names=shape_features,
            folds=folds, n_boot=n_boot, seed=seed + 2,
        )
        result["overall"]["shape_over_kappa_bin_logn_entropy"] = eval_increment(
            records, all_mask, base_names=base_bin_entropy, add_names=shape_features,
            folds=folds, n_boot=n_boot, seed=seed + 3,
        )

    for b in range(bin_oh.shape[1]):
        mask = bins == b
        row = {
            "bucket": int(b),
            "kappa_min": float(np.nanmin(kappa[mask])) if mask.any() else float("nan"),
            "kappa_max": float(np.nanmax(kappa[mask])) if mask.any() else float("nan"),
            **describe_subset(mask, y),
            "kappa_auroc_bestdir": bdir(auroc(-kappa[mask], y[mask])) if mask.any() else float("nan"),
            "top_shape_features": eval_feature_table(records, mask, shape_features, max_features=8),
            "shape_over_kappa_logn": eval_increment(
                records, mask, base_names=base_k_len, add_names=shape_features,
                folds=folds, n_boot=n_boot, seed=seed + 10 + b,
            ),
            "shape_over_kappa_bin_logn": eval_increment(
                records, mask, base_names=base_bin, add_names=shape_features,
                folds=folds, n_boot=n_boot, seed=seed + 20 + b,
            ),
        }
        if entropy_names:
            row["shape_over_kappa_logn_entropy"] = eval_increment(
                records, mask, base_names=base_entropy, add_names=shape_features,
                folds=folds, n_boot=n_boot, seed=seed + 30 + b,
            )
        result["by_kappa_bucket"].append(row)

    for frac in low_fracs:
        if not (0.0 < frac < 1.0):
            continue
        thr = float(np.nanquantile(kappa, frac))
        mask = kappa <= thr
        row = {
            "fraction": float(frac),
            "kappa_threshold": thr,
            **describe_subset(mask, y),
            "kappa_auroc_bestdir": bdir(auroc(-kappa[mask], y[mask])) if mask.any() else float("nan"),
            "top_shape_features": eval_feature_table(records, mask, shape_features, max_features=12),
            "shape_over_kappa_logn": eval_increment(
                records, mask, base_names=base_k_len, add_names=shape_features,
                folds=folds, n_boot=n_boot, seed=seed + int(frac * 1000),
            ),
            "shape_over_kappa_bin_logn": eval_increment(
                records, mask, base_names=base_bin, add_names=shape_features,
                folds=folds, n_boot=n_boot, seed=seed + int(frac * 1000) + 100,
            ),
        }
        if entropy_names:
            row["shape_over_kappa_logn_entropy"] = eval_increment(
                records, mask, base_names=base_entropy, add_names=shape_features,
                folds=folds, n_boot=n_boot, seed=seed + int(frac * 1000) + 200,
            )
        result["low_kappa_rescue"].append(row)

    for nm in shape_features:
        v = vector_from(records, nm)
        m = np.isfinite(v) & np.isfinite(kappa)
        if m.sum() >= 3 and np.std(v[m]) > 0 and np.std(kappa[m]) > 0:
            result["feature_correlations"][nm] = float(np.corrcoef(v[m], kappa[m])[0, 1])
        else:
            result["feature_correlations"][nm] = float("nan")

    return result


def load_npz_records(
    npz_path: str,
    *,
    layer: int,
    max_chains: int = 0,
    min_tokens: int = 3,
) -> Tuple[List[StepRecord], Dict[str, object]]:
    z = np.load(npz_path, allow_pickle=True)
    if "respcloud" not in z.files:
        raise SystemExit("need respcloud in npz; use full_*.npz with clouds_stored")
    cloud_layers = [int(x) for x in z["cloud_store_layers"]]
    if layer not in cloud_layers:
        raise SystemExit(f"layer {layer} not in cloud_store_layers={cloud_layers}")
    li = cloud_layers.index(layer)

    RC = z["respcloud"]
    SR = z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)
    pids = z["problem_ids"].astype(int) if "problem_ids" in z.files else np.arange(len(RC))
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None

    N = len(RC) if not max_chains else min(max_chains, len(RC))
    records: List[StepRecord] = []
    skipped = {
        "missing_cloud": 0,
        "too_few_steps": 0,
        "too_few_tokens": 0,
        "post_error": 0,
    }
    for i in range(N):
        if RC[i] is None:
            skipped["missing_cloud"] += 1
            continue
        rng = np.asarray(SR[i], int)
        if rng.ndim != 2 or rng.shape[0] < 1:
            skipped["too_few_steps"] += 1
            continue
        H = np.asarray(RC[i], np.float64)[:, li, :]
        a0 = int(rng[0, 0])
        k = int(ges[i])
        correct = k < 0
        ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        for j in range(rng.shape[0]):
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                skipped["post_error"] += 1
                continue
            lo = max(0, int(rng[j, 0]) - a0)
            hi = min(len(H), int(rng[j, 1]) - a0 + 1)
            feats = moment_features(H[lo:hi], min_tokens=min_tokens)
            if feats is None:
                skipped["too_few_tokens"] += 1
                continue
            if ud is not None:
                uhi = min(len(ud), hi)
                feats["U_D"] = float(np.nanmean(ud[lo:uhi])) if uhi > lo else float("nan")
            if uc is not None:
                uhi = min(len(uc), hi)
                feats["U_C"] = float(np.nanmean(uc[lo:uhi])) if uhi > lo else float("nan")
            records.append(
                StepRecord(
                    chain=i,
                    group=int(pids[i]),
                    step=j,
                    y=y,
                    n_tok=int(feats["n_tok"]),
                    features=feats,
                )
            )
    meta = {
        "npz_path": npz_path,
        "layer": layer,
        "n_chains_seen": N,
        "cloud_store_layers": cloud_layers,
        "skipped": skipped,
    }
    return records, meta


def random_unit(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    X = rng.normal(size=(n, d))
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def synthetic_records(
    *,
    n_per_class: int = 200,
    n_tokens: int = 48,
    dim: int = 64,
    seed: int = 0,
) -> List[StepRecord]:
    rng = np.random.default_rng(seed)
    records: List[StepRecord] = []

    axis = np.zeros(dim)
    axis[0] = 1.0

    def accept(feats: Optional[Dict[str, float]], target: float, tol: float = 0.02) -> bool:
        if feats is None:
            return False
        # Force paired synthetic classes into the same kappa band. This makes
        # kappa intentionally weak while scatter shape remains decisive.
        return abs(feats["kappa"] - target) <= tol

    def make_branch(target: float) -> Tuple[np.ndarray, Dict[str, float]]:
        for _ in range(2000):
            p_pos = np.clip((1.0 + target) / 2.0, 0.52, 0.70)
            signs = np.where(rng.random(n_tokens) < p_pos, 1.0, -1.0)
            if np.all(signs > 0) or np.all(signs < 0):
                continue
            rng.shuffle(signs)
            H = signs[:, None] * axis[None, :] + 0.18 * rng.normal(size=(n_tokens, dim))
            feats = moment_features(H)
            if accept(feats, target):
                return H, feats  # type: ignore[return-value]
        raise RuntimeError("failed to sample branch cloud in target kappa range")

    def make_iso(target: float) -> Tuple[np.ndarray, Dict[str, float]]:
        for _ in range(2000):
            H = random_unit(rng, n_tokens, dim)
            feats = moment_features(H)
            if accept(feats, target):
                return H, feats  # type: ignore[return-value]
        raise RuntimeError("failed to sample isotropic cloud in target kappa range")

    for i in range(n_per_class):
        # Correct structured branching: two tight lobes around +/- axis. The
        # mean cancels, but scatter has a strong dominant axis.
        target = float(rng.uniform(0.12, 0.18))
        _, feats = make_branch(target)
        records.append(StepRecord(chain=i, group=i, step=0, y=0, n_tok=n_tokens, features=feats))

        # Error isotropic confusion: same low-kappa regime, but no stable axis.
        _, feats = make_iso(target)
        records.append(
            StepRecord(
                chain=n_per_class + i,
                group=n_per_class + i,
                step=0,
                y=1,
                n_tok=n_tokens,
                features=feats,
            )
        )

    return records


def print_increment(label: str, entry: Dict[str, object]) -> None:
    if not entry.get("available"):
        reason = entry.get("reason", "insufficient")
        print(f"  {label:34s} unavailable ({reason}; n={entry.get('n')}, err={entry.get('n_error')})")
        return
    inc = entry["increment"]
    print(
        f"  {label:34s} {entry['base_auroc']:.3f}->{entry['full_auroc']:.3f} "
        f"inc {inc['point']:+.3f} [{inc['lo']:+.3f},{inc['hi']:+.3f}] "
        f"{'SIG' if inc['sig'] else 'ns'}"
    )


def print_summary(result: Dict[str, object], *, title: str) -> None:
    print(f"\n===== {title} =====")
    print(f"steps {result['n_steps']} | err {result['n_error']} | groups {result['n_groups']}")
    k = result["overall"]["kappa"]
    print(f"kappa baseline: AUROC {k['auroc_bestdir']:.3f} | bucket {k['bucket']:.3f}")
    print_increment("shape over [kappa+logN]", result["overall"]["shape_over_kappa_logn"])
    print_increment("shape over [kappa-bin+logN]", result["overall"]["shape_over_kappa_bin_logn"])
    if "shape_over_kappa_logn_entropy" in result["overall"]:
        print_increment("shape over [kappa+logN+U]", result["overall"]["shape_over_kappa_logn_entropy"])

    print("\nTop overall shape features:")
    for row in result["overall"]["top_shape_features"][:8]:
        print(f"  {row['feature']:20s} AUROC {row['auroc_bestdir']:.3f} bucket {row['bucket']:.3f}")

    print("\nLow-kappa rescue:")
    for row in result["low_kappa_rescue"]:
        print(
            f"  bottom {row['fraction']:.2f} k<= {row['kappa_threshold']:.3f} "
            f"n={row['n']} err={row['n_error']} kappaAUC={row['kappa_auroc_bestdir']:.3f}"
        )
        print_increment("    shape over [kappa+logN]", row["shape_over_kappa_logn"])
        print_increment("    shape over [kappa-bin+logN]", row["shape_over_kappa_bin_logn"])
        if "shape_over_kappa_logn_entropy" in row:
            print_increment("    shape over [kappa+logN+U]", row["shape_over_kappa_logn_entropy"])
        for feat in row["top_shape_features"][:5]:
            print(f"    {feat['feature']:18s} AUROC {feat['auroc_bestdir']:.3f} bucket {feat['bucket']:.3f}")


def resolve_npz(args: argparse.Namespace) -> str:
    if args.npz:
        return args.npz
    if not args.dataset:
        raise SystemExit("provide an npz path or --dataset")
    return os.path.join(args.data_dir, "features", f"full_{args.dataset}.npz")


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Kappa-conditioned second-moment audit")
    ap.add_argument("npz", nargs="?", help="full_*.npz path; optional if --dataset is set")
    ap.add_argument("--dataset", choices=["gsm8k", "math", "omnimath"], default=None)
    ap.add_argument("--data_dir", default="/gz-data/research/demo/data")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--nbins", type=int, default=5)
    ap.add_argument("--low_fracs", default="0.10,0.20,0.30")
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--min_tokens", type=int, default=3)
    ap.add_argument("--output_dir", default="outputs/kappa_moment_audit")
    ap.add_argument("--no_entropy", action="store_true", help="do not add U_D/U_C baselines even if present")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    low_fracs = parse_float_list(args.low_fracs)

    if args.selftest:
        records = synthetic_records()
        result = analyze_records(
            records,
            folds=args.folds,
            n_boot=max(100, min(args.boot, 300)),
            nbins=args.nbins,
            low_fracs=low_fracs,
            include_entropy=False,
        )
        print_summary(result, title="synthetic selftest")
        out = {"meta": {"selftest": True}, "result": result}
        os.makedirs(args.output_dir, exist_ok=True)
        out_file = os.path.join(args.output_dir, "selftest.json")
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(finite_json(out), fh, indent=2)
        print(f"\nsaved: {out_file}")
        return

    npz_path = resolve_npz(args)
    records, meta = load_npz_records(
        npz_path,
        layer=args.layer,
        max_chains=args.max_chains,
        min_tokens=args.min_tokens,
    )
    result = analyze_records(
        records,
        folds=args.folds,
        n_boot=args.boot,
        nbins=args.nbins,
        low_fracs=low_fracs,
        include_entropy=not args.no_entropy,
    )
    print_summary(result, title=f"{os.path.basename(npz_path)} | L{args.layer}")
    print(f"\nskipped: {meta['skipped']}")

    os.makedirs(args.output_dir, exist_ok=True)
    stem = args.dataset or os.path.splitext(os.path.basename(npz_path))[0]
    if args.max_chains:
        stem += f"_n{args.max_chains}"
    out_file = os.path.join(args.output_dir, f"{stem}_L{args.layer}.json")
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(finite_json({"meta": meta, "result": result}), fh, indent=2)
    print(f"saved: {out_file}")


if __name__ == "__main__":
    main()
