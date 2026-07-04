#!/usr/bin/env python3
"""Chain-aware dynamics audit for real-time reasoning-error detection.

Earlier audits mostly flattened all steps across all chains. This script keeps
each reasoning trace as a sequence and asks a sharper question:

  Wrong reasoning is not merely more divergent. Correct hard reasoning can also
  diverge. Does the divergence recover, stay anchored to the question, and look
  like a healthy transition?

The script consumes the existing full_*.npz features through
mechanism_phase_audit.load_chains, then adds chain-level dynamic scores:

  recoverability           offline analysis: does spread reconverge soon?
  anchored divergence      spread plus loss of question-anchor alignment
  confident divergence     spread with low uncertainty
  healthy transition       cross-fit model trained on correct chains only
  online alarms            conformal-style risk curve over correct chains

It is an analysis scaffold, not the final LRSM. The goal is to make the next
state-modeling step honest: first test which sequence hypotheses are real.
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
    raise SystemExit("chain_dynamics_audit.py needs scikit-learn") from exc

from mechanism_phase_audit import Chain, auroc, bdir, finite_json, load_chains, safe_mean, safe_std


EPS = 1e-9


@dataclass
class TransitionModel:
    obs: List[str]
    controls: List[str]
    y_mu: np.ndarray
    y_sd: np.ndarray
    c_mu: np.ndarray
    c_sd: np.ndarray
    beta: np.ndarray
    cov_inv: np.ndarray
    score_mu: float
    score_sd: float


def arr(c: Chain, name: str) -> np.ndarray:
    return np.asarray(c.features.get(name, np.full(c.n_steps, np.nan)), float)


def causal_z(x: np.ndarray, *, warmup: int = 2) -> np.ndarray:
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    for t in range(warmup, len(v)):
        hist = v[:t]
        hist = hist[np.isfinite(hist)]
        if len(hist) >= 2 and np.isfinite(v[t]):
            out[t] = (v[t] - hist.mean()) / (hist.std() + EPS)
    return out


def chain_z(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, float)
    m = np.isfinite(v)
    out = np.full(len(v), np.nan)
    if m.sum() >= 2:
        out[m] = (v[m] - v[m].mean()) / (v[m].std() + EPS)
    return out


def delta(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    if len(v) >= 2:
        out[1:] = v[1:] - v[:-1]
    return out


def leaky_cusum(z: np.ndarray, *, lam: float = 0.8, kref: float = 0.25) -> np.ndarray:
    out = np.zeros(len(z), float)
    c = 0.0
    for t, val in enumerate(np.asarray(z, float)):
        x = 0.0 if not np.isfinite(val) else float(val)
        c = max(0.0, lam * c + x - kref)
        out[t] = c
    return out


def future_recovery(spread: np.ndarray, *, horizon: int = 1) -> np.ndarray:
    """Positive means current spread drops within the next horizon steps."""
    s = np.asarray(spread, float)
    out = np.full(len(s), np.nan)
    for t in range(len(s)):
        nxt = s[t + 1 : min(len(s), t + 1 + horizon)]
        nxt = nxt[np.isfinite(nxt)]
        if np.isfinite(s[t]) and len(nxt):
            out[t] = s[t] - float(np.nanmin(nxt))
    return out


def add_chain_dynamic_features(chains: Sequence[Chain], *, recovery_horizon: int, lam: float, kref: float) -> None:
    for c in chains:
        if "resultant" in c.features:
            spread = 1.0 - arr(c, "resultant")
        elif "coherence" in c.features:
            spread = 1.0 - arr(c, "coherence")
        else:
            spread = np.full(c.n_steps, np.nan)
        c.features["spread"] = spread
        c.features["d_spread"] = delta(spread)
        c.features["cz_spread"] = causal_z(spread)
        c.features["spread_cusum"] = leaky_cusum(c.features["cz_spread"], lam=lam, kref=kref)
        c.features["next_recovery_1"] = future_recovery(spread, horizon=1)
        c.features[f"next_recovery_{recovery_horizon}"] = future_recovery(spread, horizon=recovery_horizon)

        if "q_align" in c.features:
            anchor_loss = 1.0 - arr(c, "q_align")
        else:
            anchor_loss = np.full(c.n_steps, np.nan)
        c.features["anchor_loss"] = anchor_loss
        c.features["d_anchor_loss"] = delta(anchor_loss)
        c.features["cz_anchor_loss"] = causal_z(anchor_loss)

        if "U_D_mean" in c.features:
            uncertainty = arr(c, "U_D_mean")
        elif "U_C_mean" in c.features:
            uncertainty = arr(c, "U_C_mean")
        else:
            uncertainty = np.full(c.n_steps, np.nan)
        c.features["uncertainty"] = uncertainty
        c.features["cz_uncertainty"] = causal_z(uncertainty)

        z_spread = chain_z(spread)
        z_anchor = chain_z(anchor_loss)
        z_unc = chain_z(uncertainty)
        c.features["unanchored_divergence"] = z_spread + z_anchor
        c.features["confident_divergence"] = z_spread - z_unc
        c.features["unanchored_cusum"] = leaky_cusum(
            c.features["cz_spread"] + c.features["cz_anchor_loss"], lam=lam, kref=kref
        )
        c.features["confident_cusum"] = leaky_cusum(
            c.features["cz_spread"] - c.features["cz_uncertainty"], lam=lam, kref=kref
        )


def finite_count(chains: Sequence[Chain], name: str) -> int:
    return int(sum(np.isfinite(arr(c, name)).sum() for c in chains))


def choose_obs(chains: Sequence[Chain], requested: Optional[str], *, min_finite: int) -> List[str]:
    if requested:
        return [x.strip() for x in requested.split(",") if x.strip()]
    candidates = ["spread", "anchor_loss", "uncertainty", "step_direction_jump", "geom_ae", "cloud_D"]
    return [nm for nm in candidates if finite_count(chains, nm) >= min_finite]


def zscore_matrix(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.asarray(X, float) - mu) / np.maximum(sd, EPS)


def collect_transition_rows(chains: Sequence[Chain], idxs: Sequence[int], obs: Sequence[str], controls: Sequence[str]):
    prev, cur, ctrl = [], [], []
    for i in idxs:
        c = chains[i]
        if not c.correct:
            continue
        for t in range(1, c.n_steps):
            p = np.array([arr(c, nm)[t - 1] for nm in obs], float)
            y = np.array([arr(c, nm)[t] for nm in obs], float)
            u = np.array([arr(c, nm)[t] for nm in controls], float)
            if np.all(np.isfinite(p)) and np.all(np.isfinite(y)) and np.all(np.isfinite(u)):
                prev.append(p)
                cur.append(y)
                ctrl.append(u)
    return np.asarray(prev, float), np.asarray(cur, float), np.asarray(ctrl, float)


def fit_transition_model(
    chains: Sequence[Chain],
    idxs: Sequence[int],
    obs: Sequence[str],
    controls: Sequence[str],
    *,
    ridge: float,
) -> Optional[TransitionModel]:
    prev, cur, ctrl = collect_transition_rows(chains, idxs, obs, controls)
    m = len(obs)
    if len(cur) < max(20, 4 * (m + len(controls) + 1)):
        return None
    all_y = np.vstack([prev, cur])
    y_mu = np.nanmean(all_y, axis=0)
    y_sd = np.nanstd(all_y, axis=0) + EPS
    c_mu = np.nanmean(ctrl, axis=0) if len(controls) else np.zeros(0)
    c_sd = np.nanstd(ctrl, axis=0) + EPS if len(controls) else np.ones(0)
    Zprev = zscore_matrix(prev, y_mu, y_sd)
    Zcur = zscore_matrix(cur, y_mu, y_sd)
    Zctrl = zscore_matrix(ctrl, c_mu, c_sd) if len(controls) else np.zeros((len(cur), 0))
    X = np.column_stack([Zprev, Zctrl, np.ones(len(cur))])
    reg = ridge * np.eye(X.shape[1])
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(X.T @ X + reg, X.T @ Zcur)
    resid = Zcur - X @ beta
    cov = np.cov(resid, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = 0.5 * (cov + cov.T) + ridge * np.eye(m)
    cov_inv = np.linalg.pinv(cov)
    raw_scores = np.einsum("ij,jk,ik->i", resid, cov_inv, resid)
    return TransitionModel(
        obs=list(obs),
        controls=list(controls),
        y_mu=y_mu,
        y_sd=y_sd,
        c_mu=c_mu,
        c_sd=c_sd,
        beta=beta,
        cov_inv=cov_inv,
        score_mu=float(np.nanmean(raw_scores)),
        score_sd=float(np.nanstd(raw_scores) + EPS),
    )


def score_chain_transition(c: Chain, model: TransitionModel) -> np.ndarray:
    out = np.full(c.n_steps, np.nan)
    for t in range(1, c.n_steps):
        p = np.array([arr(c, nm)[t - 1] for nm in model.obs], float)
        y = np.array([arr(c, nm)[t] for nm in model.obs], float)
        u = np.array([arr(c, nm)[t] for nm in model.controls], float)
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(y)) and np.all(np.isfinite(u))):
            continue
        zp = zscore_matrix(p[None], model.y_mu, model.y_sd)[0]
        zy = zscore_matrix(y[None], model.y_mu, model.y_sd)[0]
        zu = zscore_matrix(u[None], model.c_mu, model.c_sd)[0] if len(model.controls) else np.zeros(0)
        x = np.r_[zp, zu, 1.0]
        resid = zy - x @ model.beta
        out[t] = float(resid @ model.cov_inv @ resid)
    return out


def standardized_cusum(score: np.ndarray, model: TransitionModel, *, lam: float, kref: float) -> np.ndarray:
    z = (np.asarray(score, float) - model.score_mu) / max(model.score_sd, EPS)
    return leaky_cusum(z, lam=lam, kref=kref)


def first_alarm(v: np.ndarray, threshold: float) -> int:
    x = np.asarray(v, float)
    hit = np.where(np.isfinite(x) & (x > threshold))[0]
    return int(hit[0]) if len(hit) else -1


def fit_crossfit_transition(
    chains: Sequence[Chain],
    obs: Sequence[str],
    *,
    folds: int,
    ridge: float,
    eps_list: Sequence[float],
    lam: float,
    kref: float,
) -> Dict[str, object]:
    idx = np.arange(len(chains))
    groups = np.array([c.group for c in chains])
    n_splits = min(int(folds), len(np.unique(groups)))
    transition = [np.full(c.n_steps, np.nan) for c in chains]
    transition_cusum = [np.full(c.n_steps, np.nan) for c in chains]
    fold_of: Dict[int, int] = {}
    calibration: Dict[int, Dict[str, np.ndarray]] = {}
    controls = [nm for nm in ("logN", "pos") if finite_count(chains, nm) >= len(chains)]
    if n_splits < 2 or not obs:
        return {"obs": list(obs), "controls": controls, "folds": 0, "online": []}

    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(idx[:, None], idx, groups)):
        model = fit_transition_model(chains, tr, obs, controls, ridge=ridge)
        if model is None:
            continue
        train_single, train_cusum = [], []
        for i in tr:
            if not chains[i].correct:
                continue
            s = score_chain_transition(chains[i], model)
            csm = standardized_cusum(s, model, lam=lam, kref=kref)
            if np.isfinite(s).any():
                train_single.append(float(np.nanmax(s)))
                train_cusum.append(float(np.nanmax(csm)))
        calibration[fold] = {
            "single": np.asarray(train_single, float),
            "cusum": np.asarray(train_cusum, float),
        }
        for i in te:
            s = score_chain_transition(chains[i], model)
            csm = standardized_cusum(s, model, lam=lam, kref=kref)
            transition[i] = s
            transition_cusum[i] = csm
            fold_of[i] = fold

    for i, c in enumerate(chains):
        c.features["transition_surprise"] = transition[i]
        c.features["transition_cusum"] = transition_cusum[i]

    online_rows = []
    for method in ("single", "cusum"):
        for eps in eps_list:
            thresholds = {}
            for fold, cal in calibration.items():
                vals = cal[method]
                vals = vals[np.isfinite(vals)]
                thresholds[fold] = float(np.quantile(vals, 1.0 - eps)) if len(vals) else float("inf")
            n_correct = false_alarm = n_error = caught = early = 0
            delays = []
            for i, c in enumerate(chains):
                if i not in fold_of:
                    continue
                score = c.features["transition_surprise"] if method == "single" else c.features["transition_cusum"]
                alarm = first_alarm(score, thresholds.get(fold_of[i], float("inf")))
                if c.correct:
                    n_correct += 1
                    false_alarm += int(alarm >= 0)
                else:
                    n_error += 1
                    if alarm >= 0:
                        caught += 1
                        d = alarm - c.gold
                        delays.append(d)
                        early += int(d < 0)
            online_rows.append(
                {
                    "method": method,
                    "eps": float(eps),
                    "fpr": false_alarm / max(1, n_correct),
                    "recall": caught / max(1, n_error),
                    "median_delay": safe_mean([np.median(delays)]) if delays else float("nan"),
                    "early_warn": early / max(1, caught),
                    "caught": int(caught),
                    "n_error": int(n_error),
                    "n_correct": int(n_correct),
                }
            )
    return {"obs": list(obs), "controls": controls, "folds": int(n_splits), "online": online_rows}


def flatten_labeled(chains: Sequence[Chain], names: Sequence[str], *, high_spread_q: Optional[float] = None):
    spread_vals = []
    if high_spread_q is not None:
        for c in chains:
            v = arr(c, "spread")
            for t in range(c.n_steps):
                if c.correct or t <= c.gold:
                    if np.isfinite(v[t]):
                        spread_vals.append(v[t])
        threshold = float(np.quantile(spread_vals, high_spread_q)) if spread_vals else float("inf")
    else:
        threshold = -float("inf")

    X, y, g, nt, keys = [], [], [], [], []
    for c in chains:
        spread = arr(c, "spread")
        for t in range(c.n_steps):
            if not np.isfinite(spread[t]) or spread[t] < threshold:
                continue
            if c.correct or t < c.gold:
                yy = 0
            elif t == c.gold:
                yy = 1
            else:
                continue
            X.append([arr(c, nm)[t] for nm in names])
            y.append(yy)
            g.append(c.group)
            nt.append(arr(c, "n_tok")[t] if "n_tok" in c.features else np.nan)
            keys.append((c.idx, t))
    return np.asarray(X, float), np.asarray(y, int), np.asarray(g), np.asarray(nt, float), keys


def feature_table(chains: Sequence[Chain], names: Sequence[str], *, high_spread_q: Optional[float] = None, top: int = 20):
    rows = []
    for nm in names:
        X, y, _, _, _ = flatten_labeled(chains, [nm], high_spread_q=high_spread_q)
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


def oof_logit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int) -> np.ndarray:
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.isfinite(col)
        fill = float(col[m].mean()) if m.any() else 0.0
        col[~m] = fill
        X[:, j] = col
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


def group_table(chains: Sequence[Chain], groups: Dict[str, Sequence[str]], *, folds: int):
    out = {}
    for label, names in groups.items():
        names = [nm for nm in names if finite_count(chains, nm) >= 30]
        if not names:
            continue
        X, y, g, _, _ = flatten_labeled(chains, names)
        if X.shape[1] == 0:
            continue
        s = oof_logit(X, y, g, folds)
        m = np.isfinite(s)
        out[label] = {
            "features": list(names),
            "auroc": auroc(s[m], y[m]) if m.any() else float("nan"),
            "n": int(m.sum()),
        }
    return out


def within_chain_rank(chains: Sequence[Chain], feature: str, sign: float):
    top1, exp1, pct = [], [], []
    for c in chains:
        if c.correct or c.gold < 0 or c.gold >= c.n_steps:
            continue
        s = sign * arr(c, feature)
        m = np.isfinite(s)
        m[np.arange(c.n_steps) > c.gold] = False
        if not m[c.gold] or m.sum() < 2:
            continue
        cand = s[m]
        better = int((cand > s[c.gold]).sum())
        top1.append(float(better == 0))
        exp1.append(1.0 / m.sum())
        pct.append(better / max(1, m.sum() - 1))
    return {
        "top1": safe_mean(top1),
        "expected_top1": safe_mean(exp1),
        "mean_pct": safe_mean(pct),
        "n": int(len(top1)),
        "sign": float(sign),
    }


def localization_table(chains: Sequence[Chain], names: Sequence[str], *, top: int = 20):
    det = {r["feature"]: r for r in feature_table(chains, names, top=len(names))}
    rows = []
    for nm, r in det.items():
        sign = 1.0 if r["raw_auroc_high_is_error"] >= 0.5 else -1.0
        loc = within_chain_rank(chains, nm, sign)
        if loc["n"] > 0:
            rows.append({"feature": nm, **loc})
    rows.sort(key=lambda r: np.nan_to_num(r["top1"], nan=-1) - np.nan_to_num(r["expected_top1"], nan=0), reverse=True)
    return rows[:top]


def event_study(chains: Sequence[Chain], names: Sequence[str], *, window: int):
    out = {}
    err = [c for c in chains if not c.correct and c.gold >= 0]
    for nm in names:
        rows = []
        for d in range(-window, window + 1):
            vals = []
            for c in err:
                t = c.gold + d
                if 0 <= t < c.n_steps:
                    vals.append(arr(c, nm)[t])
            rows.append({"delta": d, "mean": safe_mean(vals), "std": safe_std(vals), "n": int(np.isfinite(vals).sum())})
        pre = [r["mean"] for r in rows if r["delta"] < 0]
        at0 = next((r["mean"] for r in rows if r["delta"] == 0), float("nan"))
        out[nm] = {
            "trajectory": rows,
            "at_error_minus_pre_mean": float(at0 - safe_mean(pre)) if np.isfinite(at0) else float("nan"),
        }
    return out


def resolve_npz(args: argparse.Namespace) -> str:
    if args.npz:
        return args.npz
    if not args.dataset:
        raise SystemExit("provide npz path or --dataset")
    return os.path.join(args.data_dir, "features", f"full_{args.dataset}.npz")


def run(npz: str, args: argparse.Namespace) -> Dict[str, object]:
    chains, meta = load_chains(npz, layer=args.layer, max_chains=args.max_chains)
    add_chain_dynamic_features(chains, recovery_horizon=args.recovery_horizon, lam=args.lam, kref=args.kref)
    obs = choose_obs(chains, args.obs, min_finite=args.min_finite)
    transition = fit_crossfit_transition(
        chains,
        obs,
        folds=args.folds,
        ridge=args.ridge,
        eps_list=[float(x) for x in args.eps_list.split(",")],
        lam=args.lam,
        kref=args.kref,
    )

    static = ["spread", "resultant", "uncertainty", "anchor_loss", "step_direction_jump", "logN", "pos"]
    dynamic = [
        "d_spread",
        "cz_spread",
        "spread_cusum",
        "next_recovery_1",
        f"next_recovery_{args.recovery_horizon}",
        "unanchored_divergence",
        "confident_divergence",
        "unanchored_cusum",
        "confident_cusum",
        "transition_surprise",
        "transition_cusum",
    ]
    groups = {
        "static": ["spread", "logN", "pos"],
        "anchor_uncertainty": ["spread", "anchor_loss", "uncertainty", "logN", "pos"],
        "dynamic_online": ["spread", "d_spread", "cz_spread", "spread_cusum", "unanchored_cusum", "confident_cusum", "transition_surprise", "transition_cusum", "logN", "pos"],
        "offline_recovery": ["spread", "next_recovery_1", f"next_recovery_{args.recovery_horizon}", "logN", "pos"],
    }
    all_names = list(dict.fromkeys(static + dynamic))
    res = {
        "meta": {**meta, "chain_dynamic_layer": args.layer},
        "hypotheses": {
            "recoverability": "correct divergence should often reconverge soon; error divergence should recover less",
            "anchored_divergence": "healthy divergence stays anchored to the question/prompt; wrong divergence drifts",
            "healthy_transition": "errors should be surprising under a transition model trained only on correct chains",
            "online_detection": "alarms are evaluated per chain, not by shuffling all steps",
        },
        "n_chains": len(chains),
        "n_error_chains": int(sum(not c.correct for c in chains)),
        "transition_model": transition,
        "overall_features": feature_table(chains, all_names, top=args.top),
        "high_spread_features": feature_table(chains, all_names, high_spread_q=args.high_spread_q, top=args.top),
        "localization": localization_table(chains, all_names, top=args.top),
        "group_oof": group_table(chains, groups, folds=args.folds),
        "event_study": event_study(
            chains,
            [nm for nm in ("spread", "d_spread", "next_recovery_1", "anchor_loss", "uncertainty", "unanchored_divergence", "confident_divergence", "transition_surprise", "transition_cusum") if finite_count(chains, nm) >= 20],
            window=args.event_window,
        ),
    }
    return res


def print_rows(rows, *, label: str, key: str = "feature", n: int = 12) -> None:
    print(f"\n{label}:")
    for r in rows[:n]:
        print(
            f"  {r[key]:26s} AUROC {r['auroc_bestdir']:.3f} "
            f"nonerr {r['mean_non_error']:+.3f} err {r['mean_gold_error']:+.3f} n={r['n']} err={r['err']}"
        )


def print_result(res: Dict[str, object]) -> None:
    meta = res["meta"]
    print(f"\n===== chain dynamics audit | {os.path.basename(meta['npz'])} | L{meta['layer']} =====")
    print(f"chains {res['n_chains']} | error chains {res['n_error_chains']}")
    tm = res["transition_model"]
    print(f"transition obs={tm.get('obs')} controls={tm.get('controls')} folds={tm.get('folds')}")
    print_rows(res["overall_features"], label="Overall step/gold-error scores")
    print_rows(res["high_spread_features"], label="High-divergence subset scores")

    print("\nWithin-chain localization:")
    for r in res["localization"][:12]:
        gain = r["top1"] - r["expected_top1"]
        print(f"  {r['feature']:26s} top1 {r['top1']:.3f} exp {r['expected_top1']:.3f} gain {gain:+.3f} n={r['n']}")

    print("\nOOF groups:")
    for k, v in res["group_oof"].items():
        print(f"  {k:20s} AUROC {v['auroc']:.3f} n={v['n']} features={len(v['features'])}")

    print("\nOnline transition alarms:")
    for r in tm.get("online", []):
        print(
            f"  {r['method']:6s} eps {r['eps']:.2f} FPR {r['fpr']:.3f} "
            f"recall {r['recall']:.3f} delay {r['median_delay']:+.1f} early {r['early_warn']:.3f}"
        )


def _object_array(xs: Sequence[object]) -> np.ndarray:
    out = np.empty(len(xs), dtype=object)
    out[:] = list(xs)
    return out


def make_selftest_npz(path: str, *, n_chains: int = 90, layer: int = 14, seed: int = 21) -> None:
    rng = np.random.default_rng(seed)
    layers = np.array([layer], dtype=int)
    sv_layers = np.array([layer], dtype=int)
    cloud_names = np.array(["resultant", "coherence", "cloud_D"], dtype=object)
    geom_names = np.array(["ae"], dtype=object)
    gold, pids, ranges_all = [], [], []
    stepcloud, stepgeom, tok_ud, tok_uc, stepvec, qvecs, texts = [], [], [], [], [], [], []
    d = 24
    for i in range(n_chains):
        T = int(rng.integers(6, 9))
        is_error = (i % 3) == 0
        g = int(rng.integers(3, T - 1)) if is_error else -1
        gold.append(g)
        pids.append(i)
        lens = rng.integers(4, 8, size=T)
        lo = np.cumsum(np.r_[0, lens[:-1]])
        ranges = np.stack([lo, lo + lens - 1], axis=1).astype(int)
        ranges_all.append(ranges)

        spread = 0.28 + 0.03 * rng.normal(size=T)
        uncertainty = 0.30 + 0.04 * rng.normal(size=T)
        anchor_loss = 0.12 + 0.03 * rng.normal(size=T)
        # Correct chains can still have a hard divergent step, but it recovers and stays anchored.
        if not is_error:
            h = int(rng.integers(2, T - 1))
            spread[h] += 0.24
            uncertainty[h] += 0.18
            if h + 1 < T:
                spread[h + 1] -= 0.12
        else:
            spread[g] += 0.34
            anchor_loss[g] += 0.34
            uncertainty[g] -= 0.12
            if g + 1 < T:
                spread[g + 1 :] += 0.20
                anchor_loss[g + 1 :] += 0.22
        spread = np.clip(spread, 0.02, 0.95)
        resultant = 1.0 - spread
        q_align = np.clip(1.0 - anchor_loss, -1.0, 1.0)

        sc = np.zeros((T, 1, len(cloud_names)), float)
        sc[:, 0, 0] = resultant
        sc[:, 0, 1] = resultant - 0.03
        sc[:, 0, 2] = 1.0 + spread
        sg = np.zeros((T, 1, len(geom_names)), float)
        sg[:, 0, 0] = spread
        q = rng.normal(size=d)
        q = q / np.linalg.norm(q)
        qv = q[None, :]
        sv = np.zeros((T, 1, d), float)
        prev = q
        for t in range(T):
            off = rng.normal(size=d)
            off = off - np.dot(off, q) * q
            off = off / max(np.linalg.norm(off), EPS)
            cur = q_align[t] * q + max(0.0, 1.0 - q_align[t]) * off + 0.03 * rng.normal(size=d)
            if t > 0:
                cur = 0.65 * prev + 0.35 * cur
            cur = cur / max(np.linalg.norm(cur), EPS)
            sv[t, 0] = cur
            prev = cur
        ud = np.concatenate([rng.normal(uncertainty[t], 0.02, size=int(lens[t])) for t in range(T)])
        uc = np.concatenate([rng.normal(uncertainty[t] * 0.8, 0.02, size=int(lens[t])) for t in range(T)])
        stepcloud.append(sc)
        stepgeom.append(sg)
        tok_ud.append(ud)
        tok_uc.append(uc)
        stepvec.append(sv)
        qvecs.append(qv)
        texts.append(np.array([f"synthetic step {t}" for t in range(T)], dtype=object))
    np.savez_compressed(
        path,
        gold_error_step=np.array(gold, dtype=int),
        problem_ids=np.array(pids, dtype=int),
        step_token_ranges=_object_array(ranges_all),
        steps_text=_object_array(texts),
        stepcloud=_object_array(stepcloud),
        cloud_feature_names=cloud_names,
        layers_used=layers,
        stepgeom=_object_array(stepgeom),
        geom_feature_names=geom_names,
        tok_U_D=_object_array(tok_ud),
        tok_U_C=_object_array(tok_uc),
        stepvec=_object_array(stepvec),
        qvec=np.asarray(qvecs, float),
        sv_layers=sv_layers,
    )


def assert_selftest(res: Dict[str, object]) -> None:
    rows = {r["feature"]: r for r in res["overall_features"]}
    if rows.get("transition_surprise", {}).get("auroc_bestdir", 0.0) < 0.8:
        raise SystemExit("selftest failed: transition_surprise did not recover injected transition failure")
    high = {r["feature"]: r for r in res["high_spread_features"]}
    if high.get("unanchored_divergence", {}).get("auroc_bestdir", 0.0) < 0.8:
        raise SystemExit("selftest failed: unanchored_divergence did not separate high-spread failures")
    loc = {r["feature"]: r for r in res["localization"]}
    if loc.get("transition_surprise", {}).get("top1", 0.0) < 0.6:
        raise SystemExit("selftest failed: transition_surprise did not localize gold steps")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chain-aware dynamics audit")
    ap.add_argument("npz", nargs="?")
    ap.add_argument("--dataset", choices=["gsm8k", "math", "omnimath"], default=None)
    ap.add_argument("--data_dir", default="/gz-data/research/demo/data")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--obs", default=None, help="comma-separated transition observations; default auto-select")
    ap.add_argument("--min_finite", type=int, default=50)
    ap.add_argument("--recovery_horizon", type=int, default=2)
    ap.add_argument("--high_spread_q", type=float, default=0.70)
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--kref", type=float, default=0.25)
    ap.add_argument("--eps_list", default="0.05,0.10,0.20")
    ap.add_argument("--event_window", type=int, default=3)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--output_dir", default="outputs/chain_dynamics")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            npz = os.path.join(td, "chain_dynamics_selftest.npz")
            make_selftest_npz(npz, layer=args.layer)
            res = run(npz, args)
            assert_selftest(res)
            print_result(res)
            os.makedirs(args.output_dir, exist_ok=True)
            out_file = os.path.join(args.output_dir, f"selftest_L{args.layer}.json")
            with open(out_file, "w", encoding="utf-8") as fh:
                json.dump(finite_json(res), fh, indent=2, ensure_ascii=False)
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
        json.dump(finite_json(res), fh, indent=2, ensure_ascii=False)
    print(f"\nsaved: {out_file}")


if __name__ == "__main__":
    main()
