#!/usr/bin/env python3
"""Difficulty-prior audit for reasoning error detection.

This is a deliberately modest first pass inspired by:

  The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via
  Hidden Representations

The question for this project is not "can we classify hard questions?"  The
useful question is:

  Does a generation-before prior D0 help distinguish hard-but-correct healthy
  divergence from real online reasoning breaks?

We therefore estimate a chain-level prior from qvec only:

  D0 = P(chain has an error | prompt/question hidden representation)

Then we test whether current step-level geometry signals improve after
conditioning on D0:

  feature_innov_t = feature_t - E[feature_t | D0, pos, logN, correct-history]

The script reports:
  - chain-level D0 AUROC;
  - step-level OOF groups vs the existing anchor_uncertainty baseline;
  - hard-correct false alarms at matched OOF thresholds;
  - within-chain first-error localization.

It is an audit, not the final method.  If D0 does not reduce hard-correct false
alarms or add to online/hazard scores, the prior should stay a calibration
baseline rather than become a mainline detector.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from anchorflow.data import Trace, load_traces, make_labels, unit

try:
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise SystemExit("difficulty_prior_audit.py needs scikit-learn") from exc


EPS = 1e-12


@dataclass
class FlatRows:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    keys: List[Tuple[int, int]]
    chain_correct: np.ndarray
    chain_prior: np.ndarray


def safe_mean(x: Sequence[float]) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    s = np.asarray(score, float)
    m = np.isfinite(s)
    if m.sum() < 3 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


def aupr(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    s = np.asarray(score, float)
    m = np.isfinite(s)
    if m.sum() < 3 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(average_precision_score(y[m], s[m]))


def bdir(a: float) -> float:
    return float(max(a, 1.0 - a)) if np.isfinite(a) else float("nan")


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


def chain_splits(y: np.ndarray, groups: np.ndarray, *, folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) >= 2:
        n_splits = min(int(folds), len(uniq))
        return list(GroupKFold(n_splits=n_splits).split(np.zeros((len(y), 1)), y, groups))
    counts = np.bincount(y) if len(y) else np.array([])
    n_splits = min(int(folds), int(counts.min())) if len(counts) >= 2 else 0
    if n_splits >= 2:
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros((len(y), 1)), y))
    return []


def qvec_matrix(traces: Sequence[Trace]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y, groups, norms = [], [], [], []
    for tr in traces:
        if tr.qvec is None:
            continue
        q = np.asarray(tr.qvec, float)
        if q.ndim != 1 or not np.isfinite(q).any():
            continue
        n = float(np.linalg.norm(q))
        X.append(unit(q))
        norms.append(n)
        y.append(int(not tr.correct))
        groups.append(int(tr.problem_id))
    return np.asarray(X, float), np.asarray(y, int), np.asarray(groups), np.asarray(norms, float)


def oof_difficulty_prior(
    traces: Sequence[Trace],
    *,
    folds: int,
    pca_dim: int,
    seed: int,
) -> Dict[str, object]:
    X, y, groups, norms = qvec_matrix(traces)
    score = np.full(len(traces), np.nan)
    norm_score = np.full(len(traces), np.nan)
    if len(y) < 3 or len(np.unique(y)) < 2:
        return {"score": score, "norm_score": norm_score, "summary": {"error": "not enough qvec/labels"}}

    # Map compact qvec rows back to trace indices.
    valid_trace_idx = [i for i, tr in enumerate(traces) if tr.qvec is not None and np.asarray(tr.qvec).ndim == 1]
    splits = chain_splits(y, groups, folds=folds, seed=seed)
    compact_score = np.full(len(y), np.nan)
    compact_norm_score = np.full(len(y), np.nan)
    for tr_idx, te_idx in splits:
        if len(np.unique(y[tr_idx])) < 2:
            continue
        n_comp = min(int(pca_dim), max(1, len(tr_idx) - 2), X.shape[1])
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PCA(n_components=n_comp, random_state=seed),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5, random_state=seed),
        )
        model.fit(X[tr_idx], y[tr_idx])
        compact_score[te_idx] = model.predict_proba(X[te_idx])[:, 1]

        norm_model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5, random_state=seed),
        )
        norm_model.fit(norms[tr_idx, None], y[tr_idx])
        compact_norm_score[te_idx] = norm_model.predict_proba(norms[te_idx, None])[:, 1]

    for compact_i, trace_i in enumerate(valid_trace_idx):
        score[trace_i] = compact_score[compact_i]
        norm_score[trace_i] = compact_norm_score[compact_i]

    summary = {
        "n": int(np.isfinite(compact_score).sum()),
        "err": int(y[np.isfinite(compact_score)].sum()),
        "qvec_prior_auroc": auroc(y, compact_score),
        "qvec_prior_aupr": aupr(y, compact_score),
        "qvec_norm_prior_auroc": auroc(y, compact_norm_score),
        "qvec_norm_prior_aupr": aupr(y, compact_norm_score),
        "mean_prior_error_chain": safe_mean(compact_score[y == 1]),
        "mean_prior_correct_chain": safe_mean(compact_score[y == 0]),
    }
    return {"score": score, "norm_score": norm_score, "summary": summary}


def add_prior_features(traces: Sequence[Trace], prior: np.ndarray, norm_prior: np.ndarray) -> None:
    for i, tr in enumerate(traces):
        T = tr.n_steps
        p = float(prior[i]) if i < len(prior) else float("nan")
        pn = float(norm_prior[i]) if i < len(norm_prior) else float("nan")
        tr.features["D0_error_prior"] = np.full(T, p)
        tr.features["D0_value_prior"] = np.full(T, 1.0 - p if np.isfinite(p) else np.nan)
        tr.features["D0_norm_prior"] = np.full(T, pn)
        for nm in ("spread", "anchor_loss", "U_D_mean"):
            v = np.asarray(tr.features.get(nm, np.full(T, np.nan)), float)
            tr.features[f"{nm}_x_D0"] = v * p if np.isfinite(p) else np.full(T, np.nan)


def flatten(traces: Sequence[Trace], names: Sequence[str]) -> FlatRows:
    X, y_all, groups, keys, corr, pri = [], [], [], [], [], []
    for i, tr in enumerate(traces):
        y, mask = make_labels(tr)
        d0 = np.asarray(tr.features.get("D0_error_prior", np.full(tr.n_steps, np.nan)), float)
        for t in range(tr.n_steps):
            if not mask[t]:
                continue
            X.append([np.asarray(tr.features.get(nm, np.full(tr.n_steps, np.nan)), float)[t] for nm in names])
            y_all.append(int(y[t]))
            groups.append(int(tr.problem_id))
            keys.append((i, t))
            corr.append(bool(tr.correct))
            pri.append(float(d0[t]) if t < len(d0) else float("nan"))
    return FlatRows(
        X=np.asarray(X, float),
        y=np.asarray(y_all, int),
        groups=np.asarray(groups),
        keys=keys,
        chain_correct=np.asarray(corr, bool),
        chain_prior=np.asarray(pri, float),
    )


def assign_flat_feature(traces: Sequence[Trace], keys: Sequence[Tuple[int, int]], values: np.ndarray, name: str) -> None:
    for tr in traces:
        tr.features[name] = np.full(tr.n_steps, np.nan)
    for (i, t), v in zip(keys, values):
        traces[i].features[name][t] = float(v)


def residualize_feature(
    traces: Sequence[Trace],
    feature: str,
    controls: Sequence[str],
    *,
    folds: int,
    seed: int,
    suffix: str = "_innov",
) -> str:
    rows = flatten(traces, [feature] + list(controls))
    if rows.X.size == 0:
        out_name = feature + suffix
        assign_flat_feature(traces, [], np.array([]), out_name)
        return out_name
    value = rows.X[:, 0]
    X = rows.X[:, 1:]
    resid = np.full(len(value), np.nan)
    zresid = np.full(len(value), np.nan)
    splits = chain_splits(rows.y, rows.groups, folds=folds, seed=seed)
    for tr_idx, te_idx in splits:
        fit_mask = (rows.y[tr_idx] == 0) & np.isfinite(value[tr_idx])
        fit_idx = tr_idx[fit_mask]
        if len(fit_idx) < max(20, X.shape[1] + 3):
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
        model.fit(X[fit_idx], value[fit_idx])
        pred_te = model.predict(X[te_idx])
        pred_fit = model.predict(X[fit_idx])
        resid[te_idx] = value[te_idx] - pred_te
        sd = float(np.nanstd(value[fit_idx] - pred_fit))
        zresid[te_idx] = resid[te_idx] / max(sd, EPS)
    out = feature + suffix
    assign_flat_feature(traces, rows.keys, resid, out)
    assign_flat_feature(traces, rows.keys, zresid, out + "_z")
    return out


def impute(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    return SimpleImputer(strategy="median").fit_transform(X)


def oof_step_logit(rows: FlatRows, *, folds: int, seed: int) -> np.ndarray:
    score = np.full(len(rows.y), np.nan)
    if rows.X.size == 0 or len(np.unique(rows.y)) < 2:
        return score
    splits = chain_splits(rows.y, rows.groups, folds=folds, seed=seed)
    for tr_idx, te_idx in splits:
        if len(np.unique(rows.y[tr_idx])) < 2:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5, random_state=seed),
        )
        model.fit(rows.X[tr_idx], rows.y[tr_idx])
        score[te_idx] = model.predict_proba(rows.X[te_idx])[:, 1]
    return score


def cluster_boot_increment(
    cand: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, object]:
    cand = np.asarray(cand, float)
    base = np.asarray(base, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    m = np.isfinite(cand) & np.isfinite(base)
    if m.sum() < 30 or len(np.unique(y[m])) < 2:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "sig": False}
    point = auroc(y[m], cand[m]) - auroc(y[m], base[m])
    if not n_boot:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"), "sig": False}
    rng = np.random.default_rng(seed)
    ug = np.unique(groups[m])
    by = {g: np.where(m & (groups == g))[0] for g in ug}
    vals = []
    for _ in range(int(n_boot)):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([by[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auroc(y[idx], cand[idx]) - auroc(y[idx], base[idx]))
    if not vals:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"), "sig": False}
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0 or hi < 0)}


def group_scores(
    traces: Sequence[Trace],
    groups_def: Dict[str, Sequence[str]],
    *,
    baseline: str,
    folds: int,
    n_boot: int,
    seed: int,
) -> Dict[str, Dict[str, object]]:
    raw: Dict[str, Dict[str, object]] = {}
    for label, names in groups_def.items():
        rows = flatten(traces, names)
        if rows.X.size == 0 or len(np.unique(rows.y)) < 2:
            continue
        score = oof_step_logit(rows, folds=folds, seed=seed + len(raw))
        m = np.isfinite(score)
        if m.sum() < 30 or len(np.unique(rows.y[m])) < 2:
            continue
        assign_flat_feature(traces, rows.keys, score, f"score__{label}")
        raw[label] = {
            "features": list(names),
            "score": score,
            "y": rows.y,
            "groups": rows.groups,
            "keys": rows.keys,
            "auroc": auroc(rows.y, score),
            "aupr": aupr(rows.y, score),
            "n": int(m.sum()),
            "err": int(rows.y[m].sum()),
        }
    out: Dict[str, Dict[str, object]] = {}
    base = raw.get(baseline)
    for label, row in raw.items():
        item = {
            "features": row["features"],
            "auroc": row["auroc"],
            "aupr": row["aupr"],
            "n": row["n"],
            "err": row["err"],
        }
        if base is not None and label != baseline:
            item["baseline_auroc"] = base["auroc"]
            item["increment_vs_" + baseline] = cluster_boot_increment(
                row["score"],
                base["score"],
                row["y"],
                row["groups"],
                n_boot=n_boot,
                seed=seed + 101 + len(out),
            )
        out[label] = item
    return out


def scalar_feature_table(traces: Sequence[Trace], names: Sequence[str], *, top: int = 20) -> List[Dict[str, object]]:
    rows = []
    for nm in names:
        flat = flatten(traces, [nm])
        if flat.X.size == 0:
            continue
        s = flat.X[:, 0]
        m = np.isfinite(s)
        if m.sum() < 30 or len(np.unique(flat.y[m])) < 2:
            continue
        raw = auroc(flat.y[m], s[m])
        rows.append(
            {
                "feature": nm,
                "auroc_bestdir": bdir(raw),
                "raw_auroc_high_is_error": raw,
                "nonerr_mean": safe_mean(s[(flat.y == 0) & m]),
                "err_mean": safe_mean(s[(flat.y == 1) & m]),
                "n": int(m.sum()),
                "err": int(flat.y[m].sum()),
            }
        )
    rows.sort(key=lambda r: np.nan_to_num(r["auroc_bestdir"], nan=-1.0), reverse=True)
    return rows[:top]


def online_alarm_table(traces: Sequence[Trace], score_names: Sequence[str], *, eps_list: Sequence[float]) -> List[Dict[str, object]]:
    rows = []
    correct_priors = []
    for tr in traces:
        if tr.correct:
            d0 = np.asarray(tr.features.get("D0_error_prior", []), float)
            if len(d0) and np.isfinite(d0[0]):
                correct_priors.append(d0[0])
    hard_thr = float(np.nanquantile(correct_priors, 0.70)) if correct_priors else float("inf")
    for nm in score_names:
        max_correct = []
        for tr in traces:
            y, mask = make_labels(tr)
            s = np.asarray(tr.features.get(nm, np.full(tr.n_steps, np.nan)), float)
            use = mask & np.isfinite(s)
            if tr.correct and use.any():
                max_correct.append(float(np.nanmax(s[use])))
        if not max_correct:
            continue
        for eps in eps_list:
            thr = float(np.quantile(max_correct, 1.0 - float(eps)))
            n_correct = false_alarm = hard_correct = hard_false = 0
            n_error = caught = early = 0
            delays = []
            for tr in traces:
                y, mask = make_labels(tr)
                s = np.asarray(tr.features.get(nm, np.full(tr.n_steps, np.nan)), float)
                use = mask & np.isfinite(s)
                hit_idx = np.where(use & (s > thr))[0]
                alarm = int(hit_idx[0]) if len(hit_idx) else -1
                d0 = np.asarray(tr.features.get("D0_error_prior", []), float)
                prior = float(d0[0]) if len(d0) and np.isfinite(d0[0]) else float("nan")
                if tr.correct:
                    n_correct += 1
                    false_alarm += int(alarm >= 0)
                    if np.isfinite(prior) and prior >= hard_thr:
                        hard_correct += 1
                        hard_false += int(alarm >= 0)
                elif 0 <= tr.gold_error_step < tr.n_steps:
                    n_error += 1
                    if alarm >= 0:
                        caught += 1
                        delay = alarm - tr.gold_error_step
                        delays.append(delay)
                        early += int(delay < 0)
            rows.append(
                {
                    "score": nm.replace("score__", ""),
                    "eps": float(eps),
                    "threshold": thr,
                    "fpr": false_alarm / max(1, n_correct),
                    "hard_correct_fpr": hard_false / max(1, hard_correct),
                    "hard_correct_n": int(hard_correct),
                    "recall": caught / max(1, n_error),
                    "median_delay": float(np.median(delays)) if delays else float("nan"),
                    "early_warn": early / max(1, caught),
                    "caught": int(caught),
                    "n_error": int(n_error),
                }
            )
    return rows


def localization_table(traces: Sequence[Trace], names: Sequence[str], *, top: int = 20) -> List[Dict[str, object]]:
    rows = []
    for nm in names:
        hits_hi, hits_lo, exps = [], [], []
        for tr in traces:
            if tr.correct or tr.gold_error_step < 0 or tr.gold_error_step >= tr.n_steps:
                continue
            s = np.asarray(tr.features.get(nm, np.full(tr.n_steps, np.nan)), float)
            y, mask = make_labels(tr)
            use = mask & np.isfinite(s)
            if not use[tr.gold_error_step] or use.sum() < 2:
                continue
            vals = s[use]
            if float(np.nanmax(vals) - np.nanmin(vals)) <= 1e-12:
                continue
            gold = s[tr.gold_error_step]
            hits_hi.append(float(gold >= np.nanmax(vals)))
            hits_lo.append(float(gold <= np.nanmin(vals)))
            exps.append(1.0 / float(use.sum()))
        if not hits_hi:
            continue
        hi = float(np.mean(hits_hi))
        lo = float(np.mean(hits_lo))
        best = max(hi, lo)
        exp = float(np.mean(exps))
        rows.append(
            {
                "feature": nm.replace("score__", ""),
                "top1": best,
                "direction": "high" if hi >= lo else "low",
                "expected_top1": exp,
                "gain": best - exp,
                "n": int(len(hits_hi)),
            }
        )
    rows.sort(key=lambda r: r["gain"], reverse=True)
    return rows[:top]


def add_simple_chain_z(traces: Sequence[Trace], names: Sequence[str]) -> None:
    for tr in traces:
        for nm in names:
            x = np.asarray(tr.features.get(nm, np.full(tr.n_steps, np.nan)), float)
            mu = safe_mean(x)
            sd = float(np.nanstd(x[np.isfinite(x)])) if np.isfinite(x).any() else float("nan")
            tr.features[f"cz_{nm}"] = (x - mu) / max(sd, EPS) if np.isfinite(sd) and sd > 0 else np.full(tr.n_steps, np.nan)


def run(traces: Sequence[Trace], args: argparse.Namespace, meta: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    meta = dict(meta or {})
    prior = oof_difficulty_prior(traces, folds=args.folds, pca_dim=args.prior_pca, seed=args.seed)
    add_prior_features(traces, np.asarray(prior["score"], float), np.asarray(prior["norm_score"], float))
    add_simple_chain_z(traces, ["spread", "anchor_loss", "U_D_mean"])

    controls = ["D0_error_prior", "logN", "pos"]
    innov = []
    for nm in ("spread", "anchor_loss", "U_D_mean", "d_spread", "d_anchor_loss", "cz_spread", "cz_anchor_loss", "cz_U_D_mean"):
        # Some traces may not contain all names; residualizer handles missing values.
        innov.append(residualize_feature(traces, nm, controls, folds=args.folds, seed=args.seed))
    for tr in traces:
        T = tr.n_steps
        vals = [np.asarray(tr.features.get(nm + "_z", np.full(T, np.nan)), float) for nm in ("spread_innov", "anchor_loss_innov", "U_D_mean_innov")]
        tr.features["geom_value_innov"] = vals[0] + vals[1] + vals[2]

    scalar_names = [
        "D0_error_prior",
        "spread",
        "anchor_loss",
        "U_D_mean",
        "d_spread",
        "step_direction_jump",
        "spread_innov",
        "anchor_loss_innov",
        "U_D_mean_innov",
        "geom_value_innov",
        "spread_x_D0",
        "anchor_loss_x_D0",
        "U_D_mean_x_D0",
        "logN",
        "pos",
    ]
    groups_def = {
        "controls": ["logN", "pos"],
        "D0_only": ["D0_error_prior", "logN", "pos"],
        "anchor_uncertainty": ["spread", "anchor_loss", "U_D_mean", "logN", "pos"],
        "anchor_plus_D0": ["spread", "anchor_loss", "U_D_mean", "D0_error_prior", "logN", "pos"],
        "anchor_D0_interactions": [
            "spread",
            "anchor_loss",
            "U_D_mean",
            "D0_error_prior",
            "spread_x_D0",
            "anchor_loss_x_D0",
            "U_D_mean_x_D0",
            "logN",
            "pos",
        ],
        "innovation_only": ["spread_innov", "anchor_loss_innov", "U_D_mean_innov", "D0_error_prior", "logN", "pos"],
        "hazard_value": [
            "spread",
            "anchor_loss",
            "U_D_mean",
            "d_spread",
            "d_anchor_loss",
            "spread_innov",
            "anchor_loss_innov",
            "U_D_mean_innov",
            "geom_value_innov",
            "D0_error_prior",
            "logN",
            "pos",
        ],
    }
    groups = group_scores(
        traces,
        groups_def,
        baseline="anchor_uncertainty",
        folds=args.folds,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    score_names = [f"score__{k}" for k in ("anchor_uncertainty", "anchor_plus_D0", "anchor_D0_interactions", "innovation_only", "hazard_value")]
    res = {
        "meta": meta,
        "n_chains": int(len(traces)),
        "n_error_chains": int(sum(not tr.correct for tr in traces)),
        "difficulty_prior": prior["summary"],
        "step_features": scalar_feature_table(traces, scalar_names, top=args.top),
        "group_oof": groups,
        "online_alarms": online_alarm_table(traces, score_names, eps_list=args.eps_list),
        "localization": localization_table(traces, scalar_names + score_names, top=args.top),
        "notes": {
            "D0": "qvec-only OOF estimate of chain-level error risk before generation",
            "innovation": "feature - E[feature | D0, logN, pos] fit on non-error steps only",
            "validation_focus": "look for hard-correct FPR reduction and increment over anchor_uncertainty, not D0-only AUROC",
        },
    }
    return res


def print_report(res: Dict[str, object]) -> None:
    meta = res.get("meta", {})
    print(f"\n===== difficulty prior audit | {os.path.basename(str(meta.get('npz', 'selftest')))} | L{meta.get('layer', 'na')} =====")
    print(f"chains {res['n_chains']} | error chains {res['n_error_chains']}")
    d0 = res["difficulty_prior"]
    print(
        "D0 qvec prior: "
        f"AUROC {float(d0.get('qvec_prior_auroc', float('nan'))):.3f} "
        f"AUPR {float(d0.get('qvec_prior_aupr', float('nan'))):.3f} | "
        f"norm-only AUROC {float(d0.get('qvec_norm_prior_auroc', float('nan'))):.3f}"
    )
    print("\nStep/gold-error scalar scores:")
    for r in res["step_features"][:12]:
        print(
            f"  {r['feature']:24s} AUROC {r['auroc_bestdir']:.3f} "
            f"nonerr {r['nonerr_mean']:+.4f} err {r['err_mean']:+.4f}"
        )
    print("\nOOF groups:")
    for name, row in res["group_oof"].items():
        print(f"  {name:24s} AUROC {row['auroc']:.3f} AUPR {row['aupr']:.3f} n={row['n']} err={row['err']}")
        inc = row.get("increment_vs_anchor_uncertainty")
        if inc:
            sig = "SIG" if inc.get("sig") else "ns"
            print(f"      inc {inc['point']:+.3f} [{inc['lo']:+.3f},{inc['hi']:+.3f}] {sig}")
    print("\nOOF online alarms:")
    for r in res["online_alarms"]:
        print(
            f"  {r['score']:24s} eps {r['eps']:.2f} FPR {r['fpr']:.3f} "
            f"hardFPR {r['hard_correct_fpr']:.3f} recall {r['recall']:.3f} "
            f"delay {r['median_delay']:+.1f} early {r['early_warn']:.3f}"
        )
    print("\nWithin-chain localization:")
    for r in res["localization"][:12]:
        print(
            f"  {r['feature']:24s} top1 {r['top1']:.3f} exp {r['expected_top1']:.3f} "
            f"gain {r['gain']:+.3f} dir {r['direction']}"
        )


def _object_array(xs: Sequence[object]) -> np.ndarray:
    out = np.empty(len(xs), dtype=object)
    out[:] = list(xs)
    return out


def make_selftest(seed: int = 0, n: int = 240) -> List[Trace]:
    rng = np.random.default_rng(seed)
    d = 96
    axis = unit(rng.normal(size=d))
    traces: List[Trace] = []
    for i in range(n):
        prior_true = float(rng.beta(2.0, 2.0))
        hard_correct = prior_true > 0.72 and rng.random() < 0.15
        easy_break = prior_true < 0.32 and rng.random() < 0.08
        error = (rng.random() < prior_true * 0.85) or easy_break
        if hard_correct:
            error = False
        T = int(rng.integers(4, 7))
        gold = int(rng.integers(1, T)) if error else -1
        q = unit(axis * (prior_true - 0.5) * 2.8 + 0.25 * rng.normal(size=d))
        pos = np.arange(T, dtype=float) / max(1, T - 1)
        logn = np.log1p(rng.integers(6, 20, size=T).astype(float))
        spread = 0.24 + 0.28 * prior_true + 0.04 * rng.normal(size=T)
        anchor = 0.18 + 0.22 * prior_true + 0.04 * rng.normal(size=T)
        unc = 0.45 + 0.18 * prior_true + 0.04 * rng.normal(size=T)
        if error:
            spread[gold] += 0.28 + 0.10 * easy_break
            anchor[gold] += 0.24
            unc[gold] += 0.05 if not easy_break else -0.12
        spread = np.clip(spread, 0.02, 0.95)
        anchor = np.clip(anchor, 0.02, 1.2)
        unc = np.clip(unc, 0.02, 1.2)
        feats = {
            "logN": logn,
            "pos": pos,
            "spread": spread,
            "resultant": 1.0 - spread,
            "anchor_loss": anchor,
            "q_align": 1.0 - anchor,
            "U_D_mean": unc,
            "d_spread": np.r_[np.nan, np.diff(spread)],
            "d_anchor_loss": np.r_[np.nan, np.diff(anchor)],
            "step_direction_jump": np.r_[np.nan, np.abs(np.diff(spread)) + 0.05 * rng.random(T - 1)],
        }
        rngs = np.c_[np.arange(T) * 8, np.arange(T) * 8 + 7]
        traces.append(
            Trace(
                idx=i,
                chain_id=f"selftest-{i}",
                problem_id=i,
                dataset="selftest",
                correct=not error,
                gold_error_step=gold,
                step_token_ranges=rngs,
                steps_text=[f"Step {j}" for j in range(T)],
                response_text="",
                prompt_text="",
                features=feats,
                stepvec=None,
                qvec=q,
                sv_layers=[14],
                hidden_path=None,
                layer=14,
            )
        )
    return traces


def write_outputs(res: Dict[str, object], output_dir: str, stem: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    clean = finite_json(res)
    with open(os.path.join(output_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, stem + ".md"), "w", encoding="utf-8") as f:
        f.write(f"# Difficulty Prior Audit: {stem}\n\n")
        f.write("## Result Analysis\n\n")
        d0 = clean.get("difficulty_prior", {})
        f.write(f"- D0 qvec prior AUROC: {d0.get('qvec_prior_auroc')}; norm-only AUROC: {d0.get('qvec_norm_prior_auroc')}.\n")
        f.write("- The key test is whether D0-conditioned innovation reduces hard-correct false alarms and adds over `anchor_uncertainty`.\n\n")
        f.write("## OOF Groups\n\n")
        for name, row in clean.get("group_oof", {}).items():
            f.write(f"- `{name}`: AUROC {row.get('auroc')} AUPR {row.get('aupr')}\n")
            inc = row.get("increment_vs_anchor_uncertainty")
            if inc:
                f.write(f"  - increment vs anchor_uncertainty: {inc.get('point')} [{inc.get('lo')}, {inc.get('hi')}] sig={inc.get('sig')}\n")
        f.write("\n## Follow-Up Research Direction\n\n")
        f.write("- If `hazard_value` or `anchor_D0_interactions` beats `anchor_uncertainty`, upgrade D0 into a value/hazard module.\n")
        f.write("- If AUROC is flat but hard-correct FPR falls, use D0 as calibration and threshold adaptation rather than a detector.\n")
        f.write("- If D0 fails both, keep it as a baseline and redirect to richer prompt-anchor/attention traces.\n\n")
        f.write("## Optimization Suggestions\n\n")
        f.write("- Replace logistic D0 with TD-style value learning only after this audit shows calibration utility.\n")
        f.write("- Use prompt-span hidden vectors when available; qvec-only is a cheap first proxy.\n")
        f.write("- Evaluate fixed-broke intervention gains after the online alarm table improves.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--prior_pca", type=int, default=32)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--eps_list", type=float, nargs="*", default=[0.05, 0.10, 0.20])
    ap.add_argument("--output_dir", default="outputs/difficulty_prior")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        traces = make_selftest(args.seed)
        meta = {"npz": "selftest", "dataset": "selftest", "layer": 14}
        stem = "difficulty_prior_selftest"
    else:
        npz = args.npz
        if npz is None:
            if not args.dataset:
                raise SystemExit("pass --dataset, npz path, or --selftest")
            npz = os.path.join(args.data_dir, "features", f"full_{args.dataset}.npz")
        traces, meta = load_traces(npz, dataset=args.dataset or "", layer=args.layer, max_chains=args.max_chains)
        if not traces:
            raise SystemExit(f"no traces loaded from {npz}")
        stem = f"difficulty_prior_{args.dataset or os.path.splitext(os.path.basename(npz))[0]}_L{args.layer}"

    res = run(traces, args, meta=meta)
    print_report(res)
    write_outputs(res, args.output_dir, stem)
    print(f"\nwrote {os.path.join(args.output_dir, stem + '.json')} and .md")


if __name__ == "__main__":
    main()
