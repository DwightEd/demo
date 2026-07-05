from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise SystemExit("anchorflow.eval needs scikit-learn") from exc

from .data import Trace, make_labels


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


def _arr(trace: Trace, name: str) -> np.ndarray:
    return np.asarray(trace.features.get(name, np.full(trace.n_steps, np.nan)), float)


def flatten_labeled(
    traces: Sequence[Trace],
    names: Sequence[str],
    *,
    high_spread_q: Optional[float] = None,
    confident_q: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    spread_vals = []
    unc_vals = []
    for tr in traces:
        y, mask = make_labels(tr)
        for t in range(tr.n_steps):
            if not mask[t]:
                continue
            s = _arr(tr, "spread")[t]
            u = _arr(tr, "U_D_mean")[t]
            if np.isfinite(s):
                spread_vals.append(s)
            if np.isfinite(u):
                unc_vals.append(u)
    spread_thr = float(np.quantile(spread_vals, high_spread_q)) if high_spread_q is not None and spread_vals else -float("inf")
    unc_thr = float(np.quantile(unc_vals, confident_q)) if confident_q is not None and unc_vals else float("inf")

    X, y_all, groups, keys = [], [], [], []
    for tr in traces:
        y, mask = make_labels(tr)
        spread = _arr(tr, "spread")
        unc = _arr(tr, "U_D_mean")
        for t in range(tr.n_steps):
            if not mask[t]:
                continue
            if np.isfinite(spread[t]) and spread[t] < spread_thr:
                continue
            if np.isfinite(unc[t]) and unc[t] > unc_thr:
                continue
            row = [_arr(tr, nm)[t] for nm in names]
            X.append(row)
            y_all.append(int(y[t]))
            groups.append(int(tr.problem_id))
            keys.append((tr.idx, t))
    return np.asarray(X, float), np.asarray(y_all, int), np.asarray(groups), keys


def impute(X: np.ndarray) -> np.ndarray:
    out = np.asarray(X, float).copy()
    if out.ndim == 1:
        out = out[:, None]
    for j in range(out.shape[1]):
        col = out[:, j]
        m = np.isfinite(col)
        fill = float(col[m].mean()) if m.any() else 0.0
        col[~m] = fill
        out[:, j] = col
    return out


def oof_logit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, folds: int) -> np.ndarray:
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
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
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
    for _ in range(int(n_boot)):
        chosen = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([by[g] for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auroc(sf[idx], y[idx]) - auroc(sb[idx], y[idx]))
    if not vals:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"), "sig": False}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0 or hi < 0)}


def feature_table(traces: Sequence[Trace], names: Sequence[str], *, top: int = 20, **flt) -> List[Dict[str, object]]:
    rows = []
    for nm in names:
        X, y, _, _ = flatten_labeled(traces, [nm], **flt)
        if X.size == 0:
            continue
        s = X[:, 0]
        m = np.isfinite(s)
        if m.sum() < 30 or len(np.unique(y[m])) < 2:
            continue
        raw = auroc(s[m], y[m])
        rows.append(
            {
                "feature": nm,
                "auroc_bestdir": bdir(raw),
                "raw_auroc_high_is_error": raw,
                "mean_non_error": safe_mean(s[(y == 0) & m]),
                "mean_gold_error": safe_mean(s[(y == 1) & m]),
                "n": int(m.sum()),
                "err": int(y[m].sum()),
            }
        )
    rows.sort(key=lambda r: np.nan_to_num(r["auroc_bestdir"], nan=-1), reverse=True)
    return rows[:top]


def group_table(
    traces: Sequence[Trace],
    groups_def: Dict[str, Sequence[str]],
    *,
    folds: int,
    n_boot: int,
    baseline: str,
) -> Dict[str, Dict[str, object]]:
    scored = {}
    for label, names in groups_def.items():
        X, y, g, _ = flatten_labeled(traces, names)
        if X.size == 0 or len(np.unique(y)) < 2:
            continue
        p = oof_logit(X, y, g, folds=folds)
        m = np.isfinite(p)
        if m.sum() < 30 or len(np.unique(y[m])) < 2:
            continue
        scored[label] = {"features": list(names), "score": p, "y": y, "groups": g, "auroc": auroc(p[m], y[m]), "n": int(m.sum())}
    out = {}
    base = scored.get(baseline)
    for label, val in scored.items():
        row = {"features": val["features"], "auroc": val["auroc"], "n": val["n"]}
        if base is not None and label != baseline:
            row["increment_vs_" + baseline] = cluster_boot_increment(
                val["score"], base["score"], val["y"], val["groups"], n_boot=n_boot, seed=31 + len(out)
            )
            row["baseline_auroc"] = base["auroc"]
        out[label] = row
    return out


def localization_table(traces: Sequence[Trace], names: Sequence[str], *, top: int = 20) -> List[Dict[str, object]]:
    rows = []
    for nm in names:
        hits_high = []
        hits_low = []
        exps = []
        for tr in traces:
            if tr.correct or tr.gold_error_step < 0 or tr.gold_error_step >= tr.n_steps:
                continue
            s = _arr(tr, nm)
            mask = np.isfinite(s)
            mask[np.arange(tr.n_steps) > tr.gold_error_step] = False
            if not mask[tr.gold_error_step] or mask.sum() < 2:
                continue
            ss = s[mask]
            raw_gold = s[tr.gold_error_step]
            hits_high.append(float(raw_gold >= np.nanmax(ss)))
            hits_low.append(float(raw_gold <= np.nanmin(ss)))
            exps.append(1.0 / float(mask.sum()))
        if hits_high:
            high = float(np.mean(hits_high))
            low = float(np.mean(hits_low))
            best = max(high, low)
            rows.append(
                {
                    "feature": nm,
                    "top1": best,
                    "direction": "high" if high >= low else "low",
                    "expected_top1": float(np.mean(exps)),
                    "gain": float(best - np.mean(exps)),
                    "n": int(len(hits_high)),
                }
            )
    rows.sort(key=lambda r: r["gain"], reverse=True)
    return rows[:top]
