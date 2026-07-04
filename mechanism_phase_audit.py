#!/usr/bin/env python3
"""Multi-channel mechanism audit for first-error phase transitions.

This script is intentionally broader than the geometry-only audits. It asks:

  1. Which internal channels change around the gold first-error step?
  2. Are the changes level shifts, step-to-step jumps, or causal deviations?
  3. Do geometry, uncertainty/logits, attention, and anchor/direction signals
     fail together or in different syndromes?

It consumes the existing full_*.npz schema when available:
  stepcloud       -> kappa/resultant + cloud spectrum features
  stepgeom        -> pooled-vector geometry features
  tok_U_D/U_C     -> token-level logits/uncertainty, pooled per step
  stepattn        -> optional attention sink/q/entropy features
  stepvec + qvec  -> optional step direction, q-anchor alignment, step jumps

The output is descriptive but testable: event-study tables, within-chain
gold-step localization, cross-fit increments, and syndrome counts. The goal is
to expose mechanism, not just chase one scalar.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise SystemExit("mechanism_phase_audit.py needs scikit-learn") from exc


EPS = 1e-9


@dataclass
class Chain:
    idx: int
    group: int
    gold: int
    correct: bool
    features: Dict[str, np.ndarray]  # each (T,)
    n_steps: int


def finite_json(obj):
    if isinstance(obj, dict):
        return {str(k): finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return finite_json(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def auroc(score, y) -> float:
    s = np.asarray(score, float)
    yy = np.asarray(y, int)
    m = np.isfinite(s)
    s, yy = s[m], yy[m]
    p = int((yy == 1).sum())
    n = int((yy == 0).sum())
    if p == 0 or n == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[yy == 1].sum() - p * (p + 1) / 2.0) / (p * n))


def bdir(a: float) -> float:
    return float(max(a, 1.0 - a)) if np.isfinite(a) else float("nan")


def safe_mean(x) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def safe_std(x) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return float(a.std()) if len(a) else float("nan")


def corr(a, b) -> float:
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) <= 0 or np.std(y[m]) <= 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def impute(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float).copy()
    if X.ndim == 1:
        X = X[:, None]
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.isfinite(col)
        fill = float(col[m].mean()) if m.any() else 0.0
        col[~m] = fill
        X[:, j] = col
    return X


def oof_logit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int) -> np.ndarray:
    X = impute(X)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    pred = np.full(len(y), np.nan)
    n_splits = min(int(folds), len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return pred
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        )
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return pred


def cluster_boot_increment(sf, sb, y, groups, *, n_boot=500, seed=0) -> Dict[str, object]:
    sf = np.asarray(sf, float)
    sb = np.asarray(sb, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    m = np.isfinite(sf) & np.isfinite(sb)
    if m.sum() < 30 or len(np.unique(y[m])) < 2:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "sig": False}
    point = auroc(sf[m], y[m]) - auroc(sb[m], y[m])
    rng = np.random.default_rng(seed)
    ug = np.unique(groups[m])
    by = {g: np.where(m & (groups == g))[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        chosen = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([by[g] for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auroc(sf[idx], y[idx]) - auroc(sb[idx], y[idx]))
    if not vals:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"), "sig": False}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0 or hi < 0)}


def step_ranges(rng: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rr = np.asarray(rng, int)
    n_tok = rr[:, 1] - rr[:, 0] + 1
    pos = np.arange(len(rr), dtype=float) / max(1, len(rr) - 1)
    return n_tok.astype(float), pos


def layer_index(layers: Sequence[int], layer: int, *, nearest: bool = False) -> Optional[int]:
    layers = [int(x) for x in layers]
    if layer in layers:
        return layers.index(layer)
    if nearest and layers:
        return int(np.argmin([abs(x - layer) for x in layers]))
    return None


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return x / max(float(np.linalg.norm(x)), EPS)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = unit(a)
    bb = unit(b)
    return float(np.dot(aa, bb))


def per_step_token_mean(arr: Optional[np.ndarray], ranges: np.ndarray) -> np.ndarray:
    T = len(ranges)
    out = np.full(T, np.nan)
    if arr is None:
        return out
    a = np.asarray(arr, float)
    a0 = int(ranges[0, 0])
    for t, (lo0, hi0) in enumerate(ranges):
        lo = max(0, int(lo0) - a0)
        hi = min(len(a), int(hi0) - a0 + 1)
        if hi > lo:
            out[t] = float(np.nanmean(a[lo:hi]))
    return out


def per_step_token_var(arr: Optional[np.ndarray], ranges: np.ndarray) -> np.ndarray:
    T = len(ranges)
    out = np.full(T, np.nan)
    if arr is None:
        return out
    a = np.asarray(arr, float)
    a0 = int(ranges[0, 0])
    for t, (lo0, hi0) in enumerate(ranges):
        lo = max(0, int(lo0) - a0)
        hi = min(len(a), int(hi0) - a0 + 1)
        if hi > lo:
            out[t] = float(np.nanvar(a[lo:hi]))
    return out


def delta(x: np.ndarray, *, reverse: bool = False) -> np.ndarray:
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    if len(v) >= 2:
        d = v[1:] - v[:-1]
        out[1:] = -d if reverse else d
    return out


def causal_z(x: np.ndarray, *, warmup: int = 2) -> np.ndarray:
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    for t in range(warmup, len(v)):
        hist = v[:t]
        hist = hist[np.isfinite(hist)]
        if len(hist) >= 2 and np.isfinite(v[t]):
            out[t] = (v[t] - hist.mean()) / (hist.std() + EPS)
    return out


def load_chains(npz_path: str, *, layer: int, max_chains: int = 0) -> Tuple[List[Chain], Dict[str, object]]:
    z = np.load(npz_path, allow_pickle=True)
    ges = z["gold_error_step"].astype(int)
    groups = z["problem_ids"].astype(int) if "problem_ids" in z.files else np.arange(len(ges))
    SR = z["step_token_ranges"]
    ST = z["steps_text"] if "steps_text" in z.files else None

    SC = z["stepcloud"] if "stepcloud" in z.files else None
    cn = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    cloud_layers = [int(x) for x in z["layers_used"]] if "layers_used" in z.files else []
    ci = layer_index(cloud_layers, layer)

    SG = z["stepgeom"] if "stepgeom" in z.files else None
    gn = [str(x) for x in z["geom_feature_names"]] if "geom_feature_names" in z.files else []
    gi = layer_index(cloud_layers, layer) if cloud_layers else None

    SA = z["stepattn"] if "stepattn" in z.files else None
    an = [str(x) for x in z["attn_names"]] if "attn_names" in z.files else []
    if SA is not None:
        has_attn = bool(np.asarray(z["attn_stored"]).item()) if "attn_stored" in z.files else True
    else:
        has_attn = False

    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None

    SV = z["stepvec"] if "stepvec" in z.files else None
    qvec = z["qvec"] if "qvec" in z.files else None
    sv_layers = [int(x) for x in z["sv_layers"]] if "sv_layers" in z.files else []
    svi = layer_index(sv_layers, layer, nearest=True) if SV is not None and sv_layers else None

    N = len(ges) if not max_chains else min(max_chains, len(ges))
    chains: List[Chain] = []
    missing = {"stepcloud": 0, "stepgeom": 0, "stepattn": 0, "stepvec": 0}

    for i in range(N):
        rng = np.asarray(SR[i], int)
        if rng.ndim != 2 or len(rng) == 0:
            continue
        T = len(rng)
        n_tok, pos = step_ranges(rng)
        feats: Dict[str, np.ndarray] = {
            "n_tok": n_tok,
            "logN": np.log1p(n_tok),
            "pos": pos,
        }

        if SC is not None and ci is not None and SC[i] is not None:
            sc = np.asarray(SC[i], float)
            for name in ("resultant", "resultant_unif", "resultant_bulk", "coherence", "cloud_D", "cloud_V", "cloud_C"):
                if name in cn and sc.ndim == 3 and sc.shape[0] >= T:
                    feats[name] = sc[:T, ci, cn.index(name)]
        else:
            missing["stepcloud"] += 1

        if SG is not None and gi is not None and SG[i] is not None:
            sg = np.asarray(SG[i], float)
            for name in ("norm", "pr", "ae", "ed_half", "e50", "e90", "ae_robust", "anom_k5", "anom_k10"):
                if name in gn and sg.ndim == 3 and sg.shape[0] >= T:
                    feats[f"geom_{name}"] = sg[:T, gi, gn.index(name)]
        else:
            missing["stepgeom"] += 1

        if has_attn and SA is not None and ci is not None and SA[i] is not None:
            sa = np.asarray(SA[i], float)
            for name in an:
                if sa.ndim == 3 and sa.shape[0] >= T:
                    feats[f"attn_{name}"] = sa[:T, ci, an.index(name)]
        else:
            missing["stepattn"] += 1

        ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        feats["U_D_mean"] = per_step_token_mean(ud, rng)
        feats["U_D_var"] = per_step_token_var(ud, rng)
        feats["U_C_mean"] = per_step_token_mean(uc, rng)
        feats["U_C_var"] = per_step_token_var(uc, rng)

        if SV is not None and qvec is not None and svi is not None and SV[i] is not None:
            sv = np.asarray(SV[i], float)
            qv = np.asarray(qvec[i], float) if np.asarray(qvec[i]).ndim == 2 else np.asarray(qvec, float)
            if sv.ndim == 3 and sv.shape[0] >= T:
                dirs = np.array([unit(sv[t, svi]) for t in range(T)])
                feats["q_align"] = np.array([cosine(dirs[t], qv[svi]) for t in range(T)])
                jump = np.full(T, np.nan)
                for t in range(1, T):
                    jump[t] = 1.0 - cosine(dirs[t], dirs[t - 1])
                feats["step_direction_jump"] = jump
        else:
            missing["stepvec"] += 1

        # Text density is a cheap content proxy when steps_text is present.
        if ST is not None and i < len(ST):
            txt = list(ST[i])
            dens = np.full(T, np.nan)
            for t in range(min(T, len(txt))):
                s = str(txt[t])
                dens[t] = 1.0 - sum(ch.isalpha() for ch in s) / max(1, len(s))
            feats["text_density"] = dens

        # Dynamic features: level, first difference, and causal deviation.
        base_names = list(feats.keys())
        for name in base_names:
            if name in ("n_tok", "logN", "pos"):
                continue
            v = np.asarray(feats[name], float)
            if name in ("resultant", "coherence", "resultant_unif", "resultant_bulk", "q_align"):
                feats[f"d_{name}_bad"] = delta(v, reverse=True)  # drop is bad
                feats[f"cz_{name}_bad"] = causal_z(-v)
            else:
                feats[f"d_{name}"] = delta(v)
                feats[f"cz_{name}"] = causal_z(v)

        # Mismatch terms: these are crude but useful mechanism probes.
        if "resultant" in feats and "U_D_mean" in feats:
            r = np.asarray(feats["resultant"], float)
            u = np.asarray(feats["U_D_mean"], float)
            feats["confident_geom_bad"] = (-r) * (-u)  # low kappa + low entropy
            feats["uncertain_geom_bad"] = (-r) * u
        if "resultant" in feats and "attn_q_frac" in feats:
            feats["flow_geometry_mismatch"] = (-feats["resultant"]) * (-feats["attn_q_frac"])
        if "q_align" in feats and "resultant" in feats:
            feats["coherent_anchor_drift"] = feats["resultant"] * (-feats["q_align"])

        chains.append(
            Chain(
                idx=i,
                group=int(groups[i]),
                gold=int(ges[i]),
                correct=bool(ges[i] < 0),
                features=feats,
                n_steps=T,
            )
        )

    meta = {
        "npz": npz_path,
        "layer": layer,
        "n_chains_seen": N,
        "cloud_layers": cloud_layers,
        "sv_layers": sv_layers,
        "cloud_features": cn,
        "geom_features": gn,
        "attn_features": an,
        "has_attention": bool(has_attn),
        "has_stepvec_qvec": bool(SV is not None and qvec is not None and svi is not None),
        "missing": missing,
    }
    return chains, meta


def flatten_labeled(chains: Sequence[Chain], names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    X, y, g, nt, keys = [], [], [], [], []
    for c in chains:
        for t in range(c.n_steps):
            if c.correct or t < c.gold:
                yy = 0
            elif t == c.gold:
                yy = 1
            else:
                continue
            X.append([c.features.get(nm, np.full(c.n_steps, np.nan))[t] for nm in names])
            y.append(yy)
            g.append(c.group)
            nt.append(c.features["n_tok"][t])
            keys.append((c.idx, t))
    return np.asarray(X, float), np.asarray(y, int), np.asarray(g), np.asarray(nt, float), keys


def available_feature_names(chains: Sequence[Chain], *, min_finite: int = 30) -> List[str]:
    names = sorted({k for c in chains for k in c.features.keys()})
    out = []
    for nm in names:
        vals = []
        for c in chains:
            if nm in c.features:
                vals.extend(np.asarray(c.features[nm], float).tolist())
        if np.isfinite(vals).sum() >= min_finite:
            out.append(nm)
    return out


def feature_table(chains: Sequence[Chain], names: Sequence[str], *, top: int = 25) -> List[Dict[str, object]]:
    rows = []
    for nm in names:
        X, y, _, nt, _ = flatten_labeled(chains, [nm])
        s = X[:, 0]
        if np.isfinite(s).sum() < 30 or len(np.unique(y[np.isfinite(s)])) < 2:
            continue
        raw = auroc(s, y)
        rows.append(
            {
                "feature": nm,
                "auroc_bestdir": bdir(raw),
                "raw_auroc_high_is_error": raw,
                "mean_correct": safe_mean(s[y == 0]),
                "mean_error": safe_mean(s[y == 1]),
                "n": int(np.isfinite(s).sum()),
            }
        )
    rows.sort(key=lambda r: np.nan_to_num(r["auroc_bestdir"], nan=-1.0), reverse=True)
    return rows[:top]


def within_chain_rank(chains: Sequence[Chain], feature: str, sign: float) -> Dict[str, object]:
    top1, exp1, pct = [], [], []
    for c in chains:
        if c.correct or c.gold < 0 or c.gold >= c.n_steps or feature not in c.features:
            continue
        s = sign * np.asarray(c.features[feature], float)
        # Use only steps up to the first error. Post-error steps are unjudged.
        m = np.isfinite(s)
        m[np.arange(c.n_steps) > c.gold] = False
        if not m[c.gold] or m.sum() < 2:
            continue
        cand = s[m]
        better = int((cand > s[c.gold]).sum())
        top1.append(float(better == 0))
        exp1.append(1.0 / m.sum())
        pct.append(better / (m.sum() - 1))
    return {
        "top1": safe_mean(top1),
        "expected_top1": safe_mean(exp1),
        "mean_pct": safe_mean(pct),
        "n": int(len(top1)),
        "sign": float(sign),
    }


def localization_table(chains: Sequence[Chain], names: Sequence[str], *, top: int = 25) -> List[Dict[str, object]]:
    flat_rows = feature_table(chains, names, top=len(names))
    sign = {}
    for r in flat_rows:
        sign[r["feature"]] = 1.0 if r["raw_auroc_high_is_error"] >= 0.5 else -1.0
    rows = []
    for nm in names:
        if nm not in sign:
            continue
        loc = within_chain_rank(chains, nm, sign[nm])
        if loc["n"] > 0 and np.isfinite(loc["top1"]):
            rows.append({"feature": nm, **loc})
    rows.sort(key=lambda r: (np.nan_to_num(r["top1"], nan=-1.0) - np.nan_to_num(r["expected_top1"], nan=0.0)), reverse=True)
    return rows[:top]


def event_study(chains: Sequence[Chain], names: Sequence[str], *, window: int = 3) -> Dict[str, object]:
    out: Dict[str, object] = {}
    err_chains = [c for c in chains if not c.correct and c.gold >= 0]
    for nm in names:
        rows = []
        for d in range(-window, window + 1):
            vals = []
            for c in err_chains:
                t = c.gold + d
                if 0 <= t < c.n_steps and nm in c.features:
                    vals.append(c.features[nm][t])
            rows.append({"delta": d, "mean": safe_mean(vals), "std": safe_std(vals), "n": int(np.isfinite(vals).sum())})
        pre = [r["mean"] for r in rows if r["delta"] < 0]
        at0 = next((r["mean"] for r in rows if r["delta"] == 0), float("nan"))
        out[nm] = {
            "trajectory": rows,
            "at_error_minus_pre_mean": float(at0 - safe_mean(pre)) if np.isfinite(at0) else float("nan"),
        }
    return out


def eval_groups(chains: Sequence[Chain], groups: Dict[str, Sequence[str]], *, folds: int, boot: int) -> Dict[str, object]:
    y_any = None
    g_any = None
    out = {}
    for label, names in groups.items():
        names = [nm for nm in names if nm]
        X, y, g, _, _ = flatten_labeled(chains, names)
        if X.shape[1] == 0:
            continue
        s = oof_logit(X, y, g, folds)
        m = np.isfinite(s)
        out[label] = {"features": list(names), "auroc": auroc(s[m], y[m]) if m.any() else float("nan")}
        y_any, g_any = y, g

    def inc(full_label: str, base_label: str):
        if full_label not in out or base_label not in out:
            return None
        Xf, y, g, _, _ = flatten_labeled(chains, out[full_label]["features"])
        Xb, _, _, _, _ = flatten_labeled(chains, out[base_label]["features"])
        sf = oof_logit(Xf, y, g, folds)
        sb = oof_logit(Xb, y, g, folds)
        return cluster_boot_increment(sf, sb, y, g, n_boot=boot)

    increments = {}
    for full, base in (("geom+uncertainty", "uncertainty"), ("all", "geom+uncertainty"), ("all", "confounds")):
        v = inc(full, base)
        if v is not None:
            increments[f"{full}_over_{base}"] = v
    out["increments"] = increments
    return out


def syndrome_counts(chains: Sequence[Chain]) -> Dict[str, object]:
    # Data-derived medians over labeled non-error steps.
    vals = {nm: [] for nm in ("resultant", "U_D_mean", "q_align", "attn_q_frac")}
    for c in chains:
        for t in range(c.n_steps):
            if not c.correct and t > c.gold:
                continue
            y = 1 if (not c.correct and t == c.gold) else 0
            if y != 0:
                continue
            for nm in vals:
                if nm in c.features and np.isfinite(c.features[nm][t]):
                    vals[nm].append(c.features[nm][t])
    med = {nm: safe_mean(v) if v else float("nan") for nm, v in vals.items()}
    # Use median where possible. For q/attn higher means more anchored.
    counts = {
        "diffuse_low_kappa": 0,
        "uncertain_entropy": 0,
        "confident_low_kappa": 0,
        "anchor_drift": 0,
        "flow_anchor_break": 0,
        "coherent_wrong_candidate": 0,
        "n_error_steps": 0,
    }
    for c in chains:
        if c.correct or c.gold < 0 or c.gold >= c.n_steps:
            continue
        t = c.gold
        counts["n_error_steps"] += 1
        r = c.features.get("resultant", np.full(c.n_steps, np.nan))[t]
        u = c.features.get("U_D_mean", np.full(c.n_steps, np.nan))[t]
        qa = c.features.get("q_align", np.full(c.n_steps, np.nan))[t]
        aq = c.features.get("attn_q_frac", np.full(c.n_steps, np.nan))[t]
        low_k = np.isfinite(r) and np.isfinite(med["resultant"]) and r <= med["resultant"]
        high_u = np.isfinite(u) and np.isfinite(med["U_D_mean"]) and u >= med["U_D_mean"]
        low_u = np.isfinite(u) and np.isfinite(med["U_D_mean"]) and u < med["U_D_mean"]
        low_q = np.isfinite(qa) and np.isfinite(med["q_align"]) and qa < med["q_align"]
        low_attn_q = np.isfinite(aq) and np.isfinite(med["attn_q_frac"]) and aq < med["attn_q_frac"]
        if low_k:
            counts["diffuse_low_kappa"] += 1
        if high_u:
            counts["uncertain_entropy"] += 1
        if low_k and low_u:
            counts["confident_low_kappa"] += 1
        if low_q:
            counts["anchor_drift"] += 1
        if low_attn_q:
            counts["flow_anchor_break"] += 1
        if (not low_k) and low_u and (low_q or low_attn_q):
            counts["coherent_wrong_candidate"] += 1
    counts["reference_medians"] = med
    return counts


def _object_array(xs: Sequence[object]) -> np.ndarray:
    out = np.empty(len(xs), dtype=object)
    out[:] = list(xs)
    return out


def make_selftest_npz(path: str, *, n_chains: int = 80, layer: int = 14, seed: int = 7) -> None:
    """Build a small synthetic full_*.npz-like file with known failure dynamics."""
    rng = np.random.default_rng(seed)
    layers = np.array([0, layer, layer + 8], dtype=int)
    sv_layers = np.array([layer], dtype=int)
    cloud_names = np.array(
        ["resultant", "resultant_unif", "resultant_bulk", "coherence", "cloud_D", "cloud_V", "cloud_C"],
        dtype=object,
    )
    geom_names = np.array(["norm", "pr", "ae", "ed_half", "e50", "e90", "ae_robust", "anom_k5", "anom_k10"], dtype=object)
    attn_names = np.array(["q_frac", "sink_frac", "attn_entropy"], dtype=object)

    gold, groups = [], []
    step_ranges, stepcloud, stepgeom, tok_ud, tok_uc = [], [], [], [], []
    stepattn, stepvec, steps_text, qvecs = [], [], [], []
    d = 24

    for i in range(n_chains):
        T = int(rng.integers(5, 9))
        is_error = (i % 5) in (0, 2)
        g = int(rng.integers(2, T - 1)) if is_error else -1
        gold.append(g)
        groups.append(i)

        lens = rng.integers(4, 10, size=T)
        lo = np.cumsum(np.r_[0, lens[:-1]])
        hi = lo + lens - 1
        ranges = np.stack([lo, hi], axis=1).astype(int)
        step_ranges.append(ranges)

        kappa = 0.76 + 0.04 * rng.normal(size=T)
        coh = 0.62 + 0.05 * rng.normal(size=T)
        ud_step = 0.22 + 0.04 * rng.normal(size=T)
        uc_step = 0.18 + 0.04 * rng.normal(size=T)
        q_frac = 0.42 + 0.04 * rng.normal(size=T)
        sink = 0.16 + 0.03 * rng.normal(size=T)
        attn_entropy = 1.0 + 0.08 * rng.normal(size=T)
        q_align = 0.78 + 0.04 * rng.normal(size=T)
        if is_error:
            kappa[g] -= 0.28
            coh[g] -= 0.22
            ud_step[g] += 0.38
            uc_step[g] += 0.22
            q_frac[g] -= 0.23
            sink[g] += 0.10
            attn_entropy[g] += 0.34
            q_align[g] -= 0.42
            if g + 1 < T:
                kappa[g + 1 :] -= 0.10
                ud_step[g + 1 :] += 0.12

        sc = np.zeros((T, len(layers), len(cloud_names)), float)
        sg = np.zeros((T, len(layers), len(geom_names)), float)
        sa = np.zeros((T, len(layers), len(attn_names)), float)
        for li, _ly in enumerate(layers):
            jitter = 0.01 * li
            sc[:, li, 0] = kappa - jitter
            sc[:, li, 1] = kappa - 0.03 - jitter
            sc[:, li, 2] = kappa - 0.05 - jitter
            sc[:, li, 3] = coh - jitter
            sc[:, li, 4] = 1.0 / np.clip(kappa, 0.05, 1.0)
            sc[:, li, 5] = 0.2 + 0.1 * (1 - kappa)
            sc[:, li, 6] = 0.3 + 0.1 * (1 - coh)

            sg[:, li, 0] = 8.0 + 0.3 * rng.normal(size=T)
            sg[:, li, 1] = 5.0 + 1.2 * (1 - kappa)
            sg[:, li, 2] = 0.25 + 0.6 * (1 - kappa)
            sg[:, li, 3] = sg[:, li, 2] + 0.02
            sg[:, li, 4] = 0.5 + 0.4 * (1 - kappa)
            sg[:, li, 5] = 0.7 + 0.4 * (1 - kappa)
            sg[:, li, 6] = sg[:, li, 2] + 0.01
            sg[:, li, 7] = 0.2 + 0.5 * (1 - kappa)
            sg[:, li, 8] = 0.3 + 0.5 * (1 - kappa)

            sa[:, li, 0] = q_frac - jitter
            sa[:, li, 1] = sink + jitter
            sa[:, li, 2] = attn_entropy + jitter

        q = unit(rng.normal(size=d))
        qv = q[None, :]
        sv = np.zeros((T, len(sv_layers), d), float)
        prev_dir = q
        for t in range(T):
            if is_error and t == g:
                off = unit(rng.normal(size=d))
                off = unit(off - np.dot(off, q) * q)
                cur = unit(0.35 * q + 0.65 * off + 0.05 * rng.normal(size=d))
            else:
                cur = unit(q + 0.12 * rng.normal(size=d))
            if t > 0 and not (is_error and t == g):
                cur = unit(0.75 * prev_dir + 0.25 * cur)
            sv[t, 0] = cur
            prev_dir = cur

        ud_tok = np.concatenate([rng.normal(ud_step[t], 0.025, size=int(lens[t])) for t in range(T)])
        uc_tok = np.concatenate([rng.normal(uc_step[t], 0.025, size=int(lens[t])) for t in range(T)])

        stepcloud.append(sc)
        stepgeom.append(sg)
        stepattn.append(sa)
        stepvec.append(sv)
        qvecs.append(qv)
        tok_ud.append(ud_tok)
        tok_uc.append(uc_tok)
        steps_text.append(np.array([f"step {t}: synthetic reasoning state" for t in range(T)], dtype=object))

    np.savez_compressed(
        path,
        gold_error_step=np.array(gold, dtype=int),
        problem_ids=np.array(groups, dtype=int),
        step_token_ranges=_object_array(step_ranges),
        steps_text=_object_array(steps_text),
        stepcloud=_object_array(stepcloud),
        cloud_feature_names=cloud_names,
        layers_used=layers,
        stepgeom=_object_array(stepgeom),
        geom_feature_names=geom_names,
        stepattn=_object_array(stepattn),
        attn_names=attn_names,
        attn_stored=np.array(True),
        tok_U_D=_object_array(tok_ud),
        tok_U_C=_object_array(tok_uc),
        stepvec=_object_array(stepvec),
        qvec=np.asarray(qvecs, float),
        sv_layers=sv_layers,
    )


def assert_selftest(res: Dict[str, object]) -> None:
    if res["n_error_chains"] < 10:
        raise SystemExit("selftest failed: too few synthetic error chains")
    event = res["event_study"].get("d_resultant_bad", {})
    jump = event.get("at_error_minus_pre_mean", float("nan"))
    if not np.isfinite(jump) or jump <= 0:
        raise SystemExit("selftest failed: did not recover kappa drop at error step")
    all_auc = res["group_oof"].get("all", {}).get("auroc", float("nan"))
    if not np.isfinite(all_auc) or all_auc < 0.8:
        raise SystemExit(f"selftest failed: all-channel AUROC too low ({all_auc})")


def run(npz: str, args: argparse.Namespace) -> Dict[str, object]:
    chains, meta = load_chains(npz, layer=args.layer, max_chains=args.max_chains)
    names = available_feature_names(chains, min_finite=args.min_finite)

    preferred_event = [
        "resultant",
        "d_resultant_bad",
        "cz_resultant_bad",
        "U_D_mean",
        "d_U_D_mean",
        "cz_U_D_mean",
        "U_D_var",
        "step_direction_jump",
        "q_align",
        "d_q_align_bad",
        "cz_q_align_bad",
        "attn_q_frac",
        "attn_sink_frac",
        "attn_attn_entropy",
        "flow_geometry_mismatch",
        "confident_geom_bad",
        "coherent_anchor_drift",
    ]
    preferred_event = [nm for nm in preferred_event if nm in names]

    geom = [nm for nm in ("resultant", "coherence", "cloud_D", "cloud_C", "geom_norm", "geom_pr", "geom_ae", "d_resultant_bad", "cz_resultant_bad", "step_direction_jump", "q_align", "d_q_align_bad", "cz_q_align_bad") if nm in names]
    uncertainty = [nm for nm in ("U_D_mean", "U_C_mean", "U_D_var", "U_C_var", "d_U_D_mean", "cz_U_D_mean", "d_U_D_var", "cz_U_D_var") if nm in names]
    attention = [nm for nm in names if nm.startswith("attn_")]
    mismatch = [nm for nm in ("flow_geometry_mismatch", "confident_geom_bad", "uncertain_geom_bad", "coherent_anchor_drift") if nm in names]
    confounds = [nm for nm in ("logN", "pos", "text_density") if nm in names]

    groups = {
        "confounds": confounds or ["logN", "pos"],
        "geom": confounds + geom,
        "uncertainty": confounds + uncertainty,
        "geom+uncertainty": confounds + geom + uncertainty,
        "attention": confounds + attention,
        "all": confounds + geom + uncertainty + attention + mismatch,
    }

    res = {
        "meta": meta,
        "n_chains": len(chains),
        "n_error_chains": int(sum(not c.correct for c in chains)),
        "available_features": names,
        "top_feature_table": feature_table(chains, names, top=args.top),
        "localization": localization_table(chains, names, top=args.top),
        "event_study": event_study(chains, preferred_event, window=args.event_window),
        "group_oof": eval_groups(chains, groups, folds=args.folds, boot=args.boot),
        "syndromes": syndrome_counts(chains),
    }
    return res


def print_result(res: Dict[str, object]) -> None:
    meta = res["meta"]
    print(f"\n===== mechanism phase audit | {os.path.basename(meta['npz'])} | L{meta['layer']} =====")
    print(f"chains {res['n_chains']} | error chains {res['n_error_chains']} | attention={meta['has_attention']} | qvec={meta['has_stepvec_qvec']}")
    print(f"available features: {len(res['available_features'])}")
    print("\nTop first-error features:")
    for r in res["top_feature_table"][:12]:
        print(f"  {r['feature']:24s} AUROC {r['auroc_bestdir']:.3f}  corr {r['mean_correct']:+.3f} err {r['mean_error']:+.3f}")
    print("\nWithin-chain localization:")
    for r in res["localization"][:12]:
        gain = r["top1"] - r["expected_top1"]
        print(f"  {r['feature']:24s} top1 {r['top1']:.3f} exp {r['expected_top1']:.3f} gain {gain:+.3f} n={r['n']} sign={r['sign']:+.0f}")
    print("\nOOF groups:")
    for k, v in res["group_oof"].items():
        if k == "increments":
            continue
        print(f"  {k:18s} AUROC {v['auroc']:.3f}  features={len(v['features'])}")
    print("\nGroup increments:")
    for k, inc in res["group_oof"].get("increments", {}).items():
        print(f"  {k:34s} {inc['point']:+.3f} [{inc['lo']:+.3f},{inc['hi']:+.3f}] {'SIG' if inc['sig'] else 'ns'}")
    print("\nSyndromes at gold error step:")
    syn = res["syndromes"]
    n = max(1, syn.get("n_error_steps", 0))
    for k, v in syn.items():
        if k in ("reference_medians", "n_error_steps"):
            continue
        print(f"  {k:26s} {v:4d}/{n} = {v/n:.2%}")
    print(f"  reference medians: {syn.get('reference_medians')}")


def resolve_npz(args: argparse.Namespace) -> str:
    if args.npz:
        return args.npz
    if not args.dataset:
        raise SystemExit("provide npz path or --dataset")
    return os.path.join(args.data_dir, "features", f"full_{args.dataset}.npz")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-channel phase-transition mechanism audit")
    ap.add_argument("npz", nargs="?")
    ap.add_argument("--dataset", choices=["gsm8k", "math", "omnimath"], default=None)
    ap.add_argument("--data_dir", default="/gz-data/research/demo/data")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--event_window", type=int, default=3)
    ap.add_argument("--min_finite", type=int, default=30)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--output_dir", default="outputs/mechanism_phase_audit")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            npz = os.path.join(td, "mechanism_selftest.npz")
            make_selftest_npz(npz, layer=args.layer)
            res = run(npz, args)
            assert_selftest(res)
            print_result(res)
            os.makedirs(args.output_dir, exist_ok=True)
            out_file = os.path.join(args.output_dir, f"selftest_L{args.layer}.json")
            with open(out_file, "w", encoding="utf-8") as fh:
                json.dump(finite_json(res), fh, indent=2)
            print(f"\nselftest passed; saved: {out_file}")
        return

    npz = resolve_npz(args)
    res = run(npz, args)
    print_result(res)
    os.makedirs(args.output_dir, exist_ok=True)
    stem = args.dataset or os.path.splitext(os.path.basename(npz))[0]
    if args.max_chains:
        stem += f"_n{args.max_chains}"
    out_file = os.path.join(args.output_dir, f"{stem}_L{args.layer}.json")
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(finite_json(res), fh, indent=2)
    print(f"\nsaved: {out_file}")


if __name__ == "__main__":
    main()
