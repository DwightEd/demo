#!/usr/bin/env python3
"""Audit the normalized-sphere view of reasoning hidden states.

The hypothesis is not that the residual stream is literally a unit sphere.
Rather, because transformer blocks read normalized states, the direction
distribution of response-token hidden states may be the right first-order
geometry.  This script makes that claim falsifiable:

  G1 direction dominance
     Unit-direction features should beat norm-only features.

  G2 vMF proxy
     The mean resultant length R = ||mean_i unit(h_i)|| should recover the
     useful spread/kappa signal already seen in the project.

  G3 beyond one vMF
     If errors are not merely diffuse but structurally split or wrong-anchor
     concentrated, residual scatter / two-vMF gain should add over R.

  G4 mechanism hygiene
     Every comparison is reported against logN/pos and, when available,
     U_D_mean/anchor_loss so the result does not drift away from the existing
     effective signals.

The full audit needs full_*.npz plus per-chain hidden shards.  `--selftest`
exercises the same metrics on synthetic directional clouds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


@dataclass
class StepRow:
    chain_id: str
    problem_id: int
    step: int
    y: int
    features: Dict[str, float]


def _fn(cid: object) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(cid)) + ".npy"


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / max(n, EPS)


def safe_mean(x: Sequence[float]) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def safe_std(x: Sequence[float]) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return float(a.std()) if len(a) else float("nan")


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p = p[np.isfinite(p) & (p > 0)]
    s = float(p.sum())
    if s <= EPS:
        return float("nan")
    q = p / s
    return float(-(q * np.log(q + EPS)).sum())


def vmf_kappa_approx(R: float, d: int) -> float:
    """Banerjee-style approximation for vMF concentration from mean resultant.

    In this audit kappa_hat is a diagnostic, not a fitted probabilistic model.
    The raw R/resultant remains the primary scale-free signal.
    """
    if not np.isfinite(R) or d <= 1:
        return float("nan")
    R = float(np.clip(R, 0.0, 1.0 - 1e-7))
    return float(R * (d - R * R) / max(1e-7, 1.0 - R * R))


def debiased_resultant(R: float, n: int) -> float:
    """Finite-token correction: random unit directions have E[R^2] about 1/n."""
    if not np.isfinite(R) or n <= 1:
        return float("nan")
    num = max(0.0, R * R - 1.0 / n)
    den = max(EPS, 1.0 - 1.0 / n)
    return float(math.sqrt(num / den))


def centered_shape(U: np.ndarray) -> Dict[str, float]:
    n, d = U.shape
    if n < 3:
        return {
            "shape_top1_frac": float("nan"),
            "shape_eff_rank": float("nan"),
            "shape_entropy": float("nan"),
            "shape_totvar": float("nan"),
        }
    X = U - U.mean(axis=0, keepdims=True)
    # Work through the thin SVD; n is tiny compared with hidden dimension.
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    lam = (s * s) / max(1, n)
    lam = lam[lam > 1e-12]
    total = float(lam.sum())
    if total <= EPS:
        return {
            "shape_top1_frac": 0.0,
            "shape_eff_rank": 1.0,
            "shape_entropy": 0.0,
            "shape_totvar": 0.0,
        }
    p = lam / total
    h = entropy(p)
    return {
        "shape_top1_frac": float(p[0]),
        "shape_eff_rank": float(math.exp(h)),
        "shape_entropy": h,
        "shape_totvar": total,
    }


def two_vmf_gain(U: np.ndarray, *, n_iter: int = 8) -> Tuple[float, float, float]:
    """Cheap two-component spherical k-means gain.

    A single vMF cloud is summarized well by one resultant.  A split cloud can
    have a low global resultant even when each component is internally coherent.
    We measure how much concentration is recovered by two spherical components:

      gain = (||sum_{C1} u_i|| + ||sum_{C2} u_i|| - ||sum_i u_i||) / n

    The balanced version downweights one-token outlier splits.
    """
    n, d = U.shape
    if n < 4:
        return float("nan"), float("nan"), float("nan")
    X = U - U.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        axis = vt[0]
    except np.linalg.LinAlgError:
        axis = unit(U[0] - U[-1])
    proj = U @ axis
    labels = proj > np.median(proj)
    if labels.all() or (~labels).all():
        labels = proj > 0
    if labels.all() or (~labels).all():
        return 0.0, 0.0, 0.0
    for _ in range(n_iter):
        c0 = unit(U[~labels].sum(axis=0))
        c1 = unit(U[labels].sum(axis=0))
        score0 = U @ c0
        score1 = U @ c1
        new_labels = score1 > score0
        if new_labels.all() or (~new_labels).all():
            break
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    n1 = int(labels.sum())
    n0 = n - n1
    if n0 <= 0 or n1 <= 0:
        return 0.0, 0.0, 0.0
    global_s = float(np.linalg.norm(U.sum(axis=0)))
    split_s = float(np.linalg.norm(U[labels].sum(axis=0)) + np.linalg.norm(U[~labels].sum(axis=0)))
    gain = max(0.0, (split_s - global_s) / max(1, n))
    balance = min(n0, n1) / max(1, n)
    return float(gain), float(gain * 2.0 * balance), float(balance)


def step_sphere_features(H: np.ndarray, *, qvec: Optional[np.ndarray] = None) -> Dict[str, float]:
    H = np.asarray(H, float)
    if H.ndim != 2 or len(H) < 2:
        return {}
    norms = np.linalg.norm(H, axis=1)
    ok = np.isfinite(norms) & (norms > EPS)
    H = H[ok]
    norms = norms[ok]
    if len(H) < 2:
        return {}
    U = H / np.maximum(norms[:, None], EPS)
    n, d = U.shape
    mean_u = U.mean(axis=0)
    R = float(np.linalg.norm(mean_u))
    R_db = debiased_resultant(R, n)
    raw_mean = H.mean(axis=0)
    raw_mean_norm = float(np.linalg.norm(raw_mean))
    norm_mean = float(norms.mean())
    norm_std = float(norms.std())
    gain, gain_bal, split_balance = two_vmf_gain(U)
    out = {
        "n_tok_hidden": float(n),
        "unit_resultant": R,
        "unit_spread": 1.0 - R,
        "unit_resultant_debiased": R_db,
        "unit_spread_debiased": 1.0 - R_db if np.isfinite(R_db) else float("nan"),
        "vmf_kappa_hat": vmf_kappa_approx(R, d),
        "mean_norm": norm_mean,
        "norm_std": norm_std,
        "norm_cv": norm_std / max(norm_mean, EPS),
        "raw_mean_norm": raw_mean_norm,
        "raw_resultant_scaled": raw_mean_norm / max(norm_mean, EPS),
        "two_vmf_gain": gain,
        "two_vmf_gain_bal": gain_bal,
        "two_vmf_balance": split_balance,
    }
    out.update(centered_shape(U))
    out["bipolarity"] = out["shape_top1_frac"] * out["unit_spread"] if np.isfinite(out["shape_top1_frac"]) else float("nan")
    if qvec is not None:
        q = unit(qvec)
        out["q_align_unitmean"] = float(unit(mean_u) @ q) if R > EPS else float("nan")
        out["anchor_loss_unitmean"] = 1.0 - out["q_align_unitmean"] if np.isfinite(out["q_align_unitmean"]) else float("nan")
        out["q_align_rawmean"] = float(unit(raw_mean) @ q) if raw_mean_norm > EPS else float("nan")
    return out


def _layer_index(have: Sequence[int], requested: int) -> int:
    if not have:
        return 0
    arr = np.asarray([int(x) for x in have])
    return int(np.argmin(np.abs(arr - int(requested))))


def _obj_get(arr: object, i: int, default: object = None) -> object:
    if arr is None:
        return default
    try:
        return arr[i]
    except Exception:
        return default


def _hidden_candidates(
    hidden_dir: Optional[str],
    hidden_files: Optional[np.ndarray],
    ids: np.ndarray,
    i: int,
    dataset: str,
) -> List[str]:
    roots = [hidden_dir or ""]
    if dataset and hidden_dir:
        roots.append(os.path.join(hidden_dir, dataset))
    names: List[str] = []
    if hidden_files is not None:
        raw = str(_obj_get(hidden_files, i, ""))
        if raw:
            names.append(raw)
            if not os.path.isabs(raw) and dataset:
                names.append(os.path.join(dataset, raw))
    cid = _obj_get(ids, i, i)
    names.extend([_fn(cid)])
    if dataset:
        names.append(f"{dataset}-{i}.npy")
    names.append(f"{i}.npy")

    out: List[str] = []
    for nm in names:
        if os.path.isabs(nm):
            cand = nm
            if cand not in out:
                out.append(cand)
            continue
        for root in roots:
            cand = os.path.join(root, nm)
            if cand not in out:
                out.append(cand)
    return out


def _hidden_path(
    hidden_dir: Optional[str],
    hidden_files: Optional[np.ndarray],
    ids: np.ndarray,
    i: int,
    dataset: str,
) -> Optional[str]:
    for p in _hidden_candidates(hidden_dir, hidden_files, ids, i, dataset):
        if os.path.exists(p):
            return p
    return None


def _per_step_mean(values: Optional[np.ndarray], ranges: np.ndarray) -> np.ndarray:
    T = len(ranges)
    out = np.full(T, np.nan)
    if values is None:
        return out
    v = np.asarray(values, float)
    for t, (lo, hi) in enumerate(np.asarray(ranges, int)):
        lo = max(0, int(lo))
        hi = min(len(v) - 1, int(hi))
        if hi >= lo:
            out[t] = safe_mean(v[lo : hi + 1])
    return out


def load_rows(
    npz_path: str,
    *,
    hidden_dir: Optional[str],
    dataset: str = "",
    layer: int,
    max_chains: int = 0,
    min_tokens: int = 2,
) -> Tuple[List[StepRow], Dict[str, object]]:
    z = np.load(npz_path, allow_pickle=True)
    files = set(z.files)
    ids = z["ids"] if "ids" in files else np.arange(len(z["gold_error_step"]))
    ges = z["gold_error_step"].astype(int)
    groups = z["problem_ids"].astype(int) if "problem_ids" in files else np.arange(len(ges))
    ranges = z["step_token_ranges"]
    hidden_files = z["hidden_files"] if "hidden_files" in files else None
    if hidden_dir is None and "hidden_dir" in files:
        try:
            hidden_dir = str(np.asarray(z["hidden_dir"]).item())
        except Exception:
            hidden_dir = None
    hidden_layers = [int(x) for x in z["hidden_layers"]] if "hidden_layers" in files else []
    hcol = _layer_index(hidden_layers, layer)
    used_layer = hidden_layers[hcol] if hidden_layers else layer
    qvec = z["qvec"] if "qvec" in files else None
    sv_layers = [int(x) for x in z["sv_layers"]] if "sv_layers" in files else []
    q_layer_col = _layer_index(sv_layers, layer) if sv_layers else 0
    tok_ud = z["tok_U_D"] if "tok_U_D" in files else None
    stepcloud = z["stepcloud"] if "stepcloud" in files else None
    cloud_names = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in files else []
    cloud_layers = [int(x) for x in z["layers_used"]] if "layers_used" in files else []
    ccol = _layer_index(cloud_layers, layer) if cloud_layers else 0
    ri = cloud_names.index("resultant") if "resultant" in cloud_names else None

    n = len(ges) if not max_chains else min(int(max_chains), len(ges))
    rows: List[StepRow] = []
    skipped = {"missing_hidden": 0, "bad_hidden": 0, "short_step": 0, "post_error": 0}
    missing_examples: List[List[str]] = []
    for i in range(n):
        path = _hidden_path(hidden_dir, hidden_files, ids, i, dataset)
        if path is None:
            skipped["missing_hidden"] += 1
            if len(missing_examples) < 3:
                missing_examples.append(_hidden_candidates(hidden_dir, hidden_files, ids, i, dataset)[:6])
            continue
        try:
            Hfull = np.load(path, mmap_mode="r")
        except Exception:
            skipped["bad_hidden"] += 1
            continue
        if Hfull.ndim == 3:
            Hlayer = Hfull[:, hcol, :]
        elif Hfull.ndim == 2:
            Hlayer = Hfull
        else:
            skipped["bad_hidden"] += 1
            continue
        rng = np.asarray(ranges[i], int)
        if rng.ndim != 2 or len(rng) == 0:
            continue
        T = len(rng)
        a0 = int(rng[0, 0])
        n_tok = np.maximum(0, rng[:, 1] - rng[:, 0] + 1).astype(float)
        pos = np.arange(T, dtype=float) / max(1, T - 1)
        ud = _per_step_mean(np.asarray(tok_ud[i], float) if tok_ud is not None else None, rng)
        q_sel = None
        if qvec is not None:
            qraw = np.asarray(qvec[i], float)
            if qraw.ndim == 2:
                q_sel = qraw[min(q_layer_col, qraw.shape[0] - 1)]
            elif qraw.ndim == 1:
                q_sel = qraw
        precomp_R = np.full(T, np.nan)
        if stepcloud is not None and ri is not None:
            sc = np.asarray(stepcloud[i], float)
            if sc.ndim == 3 and sc.shape[0] >= T and ccol < sc.shape[1]:
                precomp_R = sc[:T, ccol, ri]

        gold = int(ges[i])
        for t, (lo_abs, hi_abs) in enumerate(rng):
            if gold >= 0 and t > gold:
                skipped["post_error"] += 1
                continue
            lo = max(0, int(lo_abs) - a0)
            hi = min(Hlayer.shape[0], int(hi_abs) - a0 + 1)
            if hi - lo < min_tokens:
                skipped["short_step"] += 1
                continue
            feats = step_sphere_features(np.asarray(Hlayer[lo:hi], dtype=np.float32), qvec=q_sel)
            if not feats:
                skipped["short_step"] += 1
                continue
            feats.update(
                {
                    "logN": float(np.log1p(n_tok[t])),
                    "pos": float(pos[t]),
                    "U_D_mean": float(ud[t]) if t < len(ud) else float("nan"),
                    "precomputed_resultant": float(precomp_R[t]) if t < len(precomp_R) else float("nan"),
                    "precomputed_spread": float(1.0 - precomp_R[t]) if t < len(precomp_R) and np.isfinite(precomp_R[t]) else float("nan"),
                }
            )
            rows.append(
                StepRow(
                    chain_id=str(_obj_get(ids, i, i)),
                    problem_id=int(groups[i]),
                    step=t,
                    y=int(gold >= 0 and t == gold),
                    features=feats,
                )
            )
    meta = {
        "npz": npz_path,
        "hidden_dir": hidden_dir,
        "dataset": dataset,
        "requested_layer": int(layer),
        "used_layer": int(used_layer),
        "n_chains_requested": int(n),
        "n_rows": len(rows),
        "n_error_rows": int(sum(r.y for r in rows)),
        "skipped": skipped,
        "missing_hidden_examples": missing_examples,
        "has_qvec": qvec is not None,
        "has_tok_U_D": tok_ud is not None,
        "has_precomputed_resultant": ri is not None,
    }
    return rows, meta


def matrix(rows: Sequence[StepRow], names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([[r.features.get(nm, np.nan) for nm in names] for r in rows], float)
    y = np.array([r.y for r in rows], int)
    g = np.array([r.problem_id for r in rows], int)
    return X, y, g


def auroc_score(y: np.ndarray, s: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        m = np.isfinite(s)
        if m.sum() < 3 or len(np.unique(y[m])) < 2:
            return float("nan")
        return float(roc_auc_score(y[m], s[m]))
    except Exception:
        return float("nan")


def aupr_score(y: np.ndarray, s: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score

        m = np.isfinite(s)
        if m.sum() < 3 or len(np.unique(y[m])) < 2:
            return float("nan")
        return float(average_precision_score(y[m], s[m]))
    except Exception:
        return float("nan")


def bestdir_auc(y: np.ndarray, s: np.ndarray) -> Tuple[float, str]:
    a = auroc_score(y, s)
    if not np.isfinite(a):
        return a, "na"
    return (a, "error_high") if a >= 0.5 else (1.0 - a, "error_low")


def feature_table(rows: Sequence[StepRow], names: Sequence[str], *, subset: Optional[np.ndarray] = None) -> List[Dict[str, object]]:
    if subset is None:
        use_rows = list(rows)
    else:
        use_rows = [r for r, m in zip(rows, subset) if bool(m)]
    out = []
    if not use_rows:
        return out
    y = np.array([r.y for r in use_rows], int)
    for nm in names:
        s = np.array([r.features.get(nm, np.nan) for r in use_rows], float)
        auc, direction = bestdir_auc(y, s)
        m = np.isfinite(s)
        out.append(
            {
                "feature": nm,
                "auroc_bestdir": auc,
                "raw_auroc": auroc_score(y, s),
                "direction": direction,
                "nonerr_mean": safe_mean(s[(y == 0) & m]),
                "err_mean": safe_mean(s[(y == 1) & m]),
                "n": int(m.sum()),
                "err": int(y[m].sum()),
            }
        )
    out.sort(key=lambda r: (float(r["auroc_bestdir"]) if np.isfinite(r["auroc_bestdir"]) else -1.0), reverse=True)
    return out


def oof_group(rows: Sequence[StepRow], names: Sequence[str], *, folds: int = 5, seed: int = 0) -> Dict[str, object]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import GroupKFold, StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        return {"error": f"sklearn unavailable: {e}", "features": list(names)}

    X, y, groups = matrix(rows, names)
    col_ok = np.isfinite(X).any(axis=0)
    names2 = [nm for nm, ok in zip(names, col_ok) if ok]
    X = X[:, col_ok]
    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "aupr": float("nan"), "features": names2, "n": int(len(y))}
    unique_groups = np.unique(groups)
    splits = []
    if len(unique_groups) >= 2:
        n_splits = min(int(folds), len(unique_groups))
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X, y, groups))
    else:
        n_splits = min(int(folds), int(np.bincount(y).min())) if len(np.unique(y)) == 2 else 0
        if n_splits >= 2:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(X, y))
    score = np.full(len(y), np.nan)
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=seed),
        )
        clf.fit(X[tr], y[tr])
        score[te] = clf.predict_proba(X[te])[:, 1]
    m = np.isfinite(score)
    if m.sum() < 3 or len(np.unique(y[m])) < 2:
        return {"auroc": float("nan"), "aupr": float("nan"), "features": names2, "n": int(m.sum())}
    return {
        "auroc": float(roc_auc_score(y[m], score[m])),
        "aupr": float(average_precision_score(y[m], score[m])),
        "features": names2,
        "n": int(m.sum()),
        "err": int(y[m].sum()),
        "scores": score,
    }


def bootstrap_increment(
    y: np.ndarray,
    groups: np.ndarray,
    base: np.ndarray,
    cand: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, float | bool]:
    point = auroc_score(y, cand) - auroc_score(y, base)
    if not n_boot:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"), "sig": False}
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_g = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(int(n_boot)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_g[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        b = auroc_score(y[idx], base[idx])
        c = auroc_score(y[idx], cand[idx])
        if np.isfinite(b) and np.isfinite(c):
            vals.append(c - b)
    if not vals:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"), "sig": False}
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0 or hi < 0)}


FEATURES = [
    "unit_spread",
    "unit_resultant",
    "unit_spread_debiased",
    "vmf_kappa_hat",
    "precomputed_spread",
    "precomputed_resultant",
    "mean_norm",
    "norm_std",
    "norm_cv",
    "raw_mean_norm",
    "raw_resultant_scaled",
    "shape_top1_frac",
    "shape_eff_rank",
    "shape_entropy",
    "shape_totvar",
    "two_vmf_gain",
    "two_vmf_gain_bal",
    "bipolarity",
    "anchor_loss_unitmean",
    "q_align_unitmean",
    "U_D_mean",
    "logN",
    "pos",
]


GROUPS = {
    "controls": ["logN", "pos"],
    "norm_only": ["mean_norm", "norm_std", "norm_cv", "logN", "pos"],
    "raw_magnitude_geometry": ["raw_mean_norm", "raw_resultant_scaled", "logN", "pos"],
    "unit_sphere": ["unit_spread", "logN", "pos"],
    "unit_plus_entropy": ["unit_spread", "U_D_mean", "logN", "pos"],
    "anchor_uncertainty_like": ["unit_spread", "anchor_loss_unitmean", "U_D_mean", "logN", "pos"],
    "shape_over_spread": ["unit_spread", "shape_top1_frac", "shape_eff_rank", "two_vmf_gain_bal", "bipolarity", "logN", "pos"],
    "shape_anchor_entropy": [
        "unit_spread",
        "shape_top1_frac",
        "shape_eff_rank",
        "two_vmf_gain_bal",
        "bipolarity",
        "anchor_loss_unitmean",
        "U_D_mean",
        "logN",
        "pos",
    ],
    "raw_plus_unit": ["unit_spread", "mean_norm", "norm_cv", "raw_resultant_scaled", "logN", "pos"],
}


def evaluate(rows: Sequence[StepRow], *, folds: int, n_boot: int, seed: int, high_spread_q: float, low_entropy_q: float) -> Dict[str, object]:
    y = np.array([r.y for r in rows], int)
    groups = np.array([r.problem_id for r in rows], int)
    feat_rows = feature_table(rows, FEATURES)
    group_rows = {name: oof_group(rows, feats, folds=folds, seed=seed) for name, feats in GROUPS.items()}
    # Remove large score arrays from the printable JSON after increment extraction.
    increments = []
    comparisons = [
        ("norm_only", "raw_magnitude_geometry"),
        ("norm_only", "unit_sphere"),
        ("unit_sphere", "unit_plus_entropy"),
        ("unit_plus_entropy", "anchor_uncertainty_like"),
        ("unit_sphere", "shape_over_spread"),
        ("anchor_uncertainty_like", "shape_anchor_entropy"),
        ("unit_sphere", "raw_plus_unit"),
    ]
    for base_name, cand_name in comparisons:
        if base_name not in group_rows or cand_name not in group_rows:
            continue
        base = np.asarray(group_rows[base_name].get("scores", np.full(len(y), np.nan)), float)
        cand = np.asarray(group_rows[cand_name].get("scores", np.full(len(y), np.nan)), float)
        m = np.isfinite(base) & np.isfinite(cand)
        if m.sum() >= 3 and len(np.unique(y[m])) >= 2:
            increments.append(
                {
                    "baseline": base_name,
                    "candidate": cand_name,
                    "increment": bootstrap_increment(y[m], groups[m], base[m], cand[m], n_boot=n_boot, seed=seed),
                }
            )
    for row in group_rows.values():
        row.pop("scores", None)

    spread = np.array([r.features.get("unit_spread", np.nan) for r in rows], float)
    high_mask = np.isfinite(spread) & (spread >= np.nanquantile(spread, high_spread_q))
    ud = np.array([r.features.get("U_D_mean", np.nan) for r in rows], float)
    lowe_mask = np.isfinite(ud) & (ud <= np.nanquantile(ud, low_entropy_q))
    return {
        "n_rows": int(len(rows)),
        "n_error_rows": int(y.sum()),
        "feature_table": feat_rows,
        "group_oof": group_rows,
        "increments": increments,
        "high_spread_q": float(high_spread_q),
        "high_spread_features": feature_table(rows, FEATURES, subset=high_mask),
        "low_entropy_q": float(low_entropy_q),
        "low_entropy_features": feature_table(rows, FEATURES, subset=lowe_mask),
    }


def print_rows(rows: Sequence[Dict[str, object]], label: str, *, top: int = 12) -> None:
    print(f"\n{label}:")
    for r in list(rows)[:top]:
        print(
            f"  {str(r['feature']):24s} AUROC {float(r['auroc_bestdir']):.3f} "
            f"{str(r['direction']):10s} nonerr {float(r['nonerr_mean']):+.4f} "
            f"err {float(r['err_mean']):+.4f} n={int(r['n'])} err={int(r['err'])}"
        )


def print_report(res: Dict[str, object]) -> None:
    meta = res.get("meta", {})
    print(f"\n===== sphere geometry audit | {os.path.basename(str(meta.get('npz', 'selftest')))} | L{meta.get('used_layer', meta.get('requested_layer', 'na'))} =====")
    print(
        f"rows {res['n_rows']} | err {res['n_error_rows']} | "
        f"qvec={meta.get('has_qvec', False)} U_D={meta.get('has_tok_U_D', False)} "
        f"precomputed_R={meta.get('has_precomputed_resultant', False)}"
    )
    if meta.get("skipped"):
        print(f"skipped {meta['skipped']}")
    print_rows(res["feature_table"], "Step/gold-error directional-sphere scores")
    print("\nOOF groups:")
    for name, r in res["group_oof"].items():
        print(f"  {name:26s} AUROC {float(r.get('auroc', float('nan'))):.3f} AUPR {float(r.get('aupr', float('nan'))):.3f} features={len(r.get('features', []))}")
    print("\nOOF increments:")
    for r in res["increments"]:
        inc = r["increment"]
        sig = "SIG" if inc.get("sig") else "ns"
        print(
            f"  {r['candidate']:24s} over {r['baseline']:24s} "
            f"{inc['point']:+.3f} [{inc['lo']:+.3f},{inc['hi']:+.3f}] {sig}"
        )
    print_rows(res["high_spread_features"], f"High-spread subset scores (q>={res['high_spread_q']:.2f})")
    print_rows(res["low_entropy_features"], f"Low-entropy/confident subset scores (q<={res['low_entropy_q']:.2f})")


def _orthogonal_noise(rng: np.random.Generator, base: np.ndarray, scale: float) -> np.ndarray:
    z = rng.normal(size=base.shape)
    z = z - (z @ base) * base
    return scale * unit(z)


def synth_rows(seed: int = 0, *, n_chains: int = 180, d: int = 96) -> List[StepRow]:
    rng = np.random.default_rng(seed)
    q = unit(rng.normal(size=d))
    rows: List[StepRow] = []
    for i in range(n_chains):
        is_err = i >= n_chains // 2
        gold = int(rng.integers(1, 4)) if is_err else -1
        mode = ["diffuse", "wrong_anchor", "split"][i % 3] if is_err else "healthy"
        T = 4
        for t in range(T):
            if gold >= 0 and t > gold:
                continue
            y = int(gold >= 0 and t == gold)
            n = int(rng.integers(6, 18))
            if not y:
                center = unit(q + _orthogonal_noise(rng, q, 0.35 + 0.10 * t))
                spread = 0.42 if t % 2 else 0.34
                entropy_val = 0.55 + 0.05 * rng.normal()
            elif mode == "diffuse":
                center = unit(q + _orthogonal_noise(rng, q, 0.70))
                spread = 0.95
                entropy_val = 0.85 + 0.05 * rng.normal()
            elif mode == "wrong_anchor":
                center = unit(q + _orthogonal_noise(rng, q, 1.55))
                spread = 0.35
                entropy_val = 0.42 + 0.04 * rng.normal()
            else:
                a = unit(q + _orthogonal_noise(rng, q, 1.1))
                b = unit(q + _orthogonal_noise(rng, q, 1.1))
                U = []
                for k in range(n):
                    c = a if k < n // 2 else b
                    U.append(unit(c + _orthogonal_noise(rng, c, 0.22)))
                U = np.asarray(U)
                norms = rng.lognormal(mean=0.0, sigma=0.15, size=n)
                H = U * norms[:, None]
                feats = step_sphere_features(H, qvec=q)
                feats.update({"logN": float(np.log1p(n)), "pos": t / (T - 1), "U_D_mean": entropy_val})
                rows.append(StepRow(chain_id=str(i), problem_id=i, step=t, y=y, features=feats))
                continue
            U = []
            for _ in range(n):
                U.append(unit(center + _orthogonal_noise(rng, center, spread)))
            U = np.asarray(U)
            # Norm is intentionally weakly informative in selftest: enough to
            # check norm baselines, not enough to define the mechanism.
            norm_shift = 0.03 if y and mode == "diffuse" else 0.0
            norms = rng.lognormal(mean=norm_shift, sigma=0.15, size=n)
            H = U * norms[:, None]
            feats = step_sphere_features(H, qvec=q)
            feats.update({"logN": float(np.log1p(n)), "pos": t / (T - 1), "U_D_mean": entropy_val})
            rows.append(StepRow(chain_id=str(i), problem_id=i, step=t, y=y, features=feats))
    return rows


def write_outputs(res: Dict[str, object], output_dir: str, stem: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    clean = json.loads(json.dumps(res, default=lambda x: float(x) if isinstance(x, np.floating) else str(x)))
    with open(os.path.join(output_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, f"{stem}.md"), "w", encoding="utf-8") as f:
        meta = clean.get("meta", {})
        f.write(f"# Sphere Geometry Audit: {stem}\n\n")
        f.write(f"- rows: {clean.get('n_rows')} error rows: {clean.get('n_error_rows')}\n")
        f.write(f"- layer: {meta.get('used_layer', meta.get('requested_layer'))}\n")
        f.write(f"- qvec: {meta.get('has_qvec')} U_D: {meta.get('has_tok_U_D')} precomputed_R: {meta.get('has_precomputed_resultant')}\n\n")
        f.write("## Result Analysis\n\n")
        f.write("This audit tests whether normalized direction geometry is useful beyond norm-only and whether mixture/shape features add over the current spread-anchor-entropy baseline.\n\n")
        f.write("## OOF Groups\n\n")
        for name, row in clean.get("group_oof", {}).items():
            f.write(f"- `{name}`: AUROC {row.get('auroc'):.4f} AUPR {row.get('aupr'):.4f}\n")
        f.write("\n## Increments\n\n")
        for row in clean.get("increments", []):
            inc = row["increment"]
            f.write(f"- `{row['candidate']}` over `{row['baseline']}`: {inc['point']:+.4f} [{inc['lo']:+.4f}, {inc['hi']:+.4f}] sig={inc['sig']}\n")
        f.write("\n## Follow-Up Research Direction\n\n")
        f.write("- Continue only if `unit_sphere` beats `norm_only` and shape/mixture features add in high-spread or low-entropy subsets.\n")
        f.write("- If shape adds, upgrade from single-vMF `kappa/resultant` to anchor-conditioned mixture-vMF or spherical transport.\n")
        f.write("- If shape does not add, treat the hypersphere framing as a useful explanation for existing `spread`, not a new mechanism.\n\n")
        f.write("## Optimization Suggestions\n\n")
        f.write("- Run across GSM8K/MATH/OmniMath and layers 10/14/18/22.\n")
        f.write("- Residualize decisive features over `U_D_mean`, `logN`, and `pos` before making mechanism claims.\n")
        f.write("- Add pre-LN/post-LN dumps only if raw-vs-unit results are ambiguous.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="Path to full_*.npz")
    ap.add_argument("--dataset", default=None, help="Dataset slug under --data_dir/features/full_<dataset>.npz")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--hidden_dir", default=None)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--min_tokens", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--high_spread_q", type=float, default=0.70)
    ap.add_argument("--low_entropy_q", type=float, default=0.30)
    ap.add_argument("--output_dir", default="outputs/sphere_geometry")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        rows = synth_rows(args.seed)
        meta = {
            "npz": "selftest",
            "requested_layer": args.layer,
            "used_layer": args.layer,
            "has_qvec": True,
            "has_tok_U_D": True,
            "has_precomputed_resultant": False,
            "skipped": {},
        }
        stem = "sphere_geometry_selftest"
    else:
        npz = args.npz
        if npz is None:
            if not args.dataset:
                raise SystemExit("pass --npz, --dataset, or --selftest")
            npz = os.path.join(args.data_dir, "features", f"full_{args.dataset}.npz")
        rows, meta = load_rows(
            npz,
            hidden_dir=args.hidden_dir,
            dataset=args.dataset or "",
            layer=args.layer,
            max_chains=args.max_chains,
            min_tokens=args.min_tokens,
        )
        if not rows:
            raise SystemExit(f"no rows loaded; check hidden_dir/npz. meta={meta}")
        stem = f"sphere_geometry_{args.dataset or os.path.splitext(os.path.basename(npz))[0]}_L{args.layer}"

    res = {"meta": meta, **evaluate(rows, folds=args.folds, n_boot=args.n_boot, seed=args.seed, high_spread_q=args.high_spread_q, low_entropy_q=args.low_entropy_q)}
    print_report(res)
    write_outputs(res, args.output_dir, stem)
    print(f"\nwrote {os.path.join(args.output_dir, stem + '.json')} and .md")


if __name__ == "__main__":
    main()
