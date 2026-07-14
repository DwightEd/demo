#!/usr/bin/env python3
"""Chain-aware dynamics audit for real-time reasoning-error detection.

Earlier audits mostly flattened all steps across all chains. This script keeps
each reasoning trace as a sequence and asks a sharper question:

  Wrong reasoning is not merely more divergent. Correct hard reasoning can also
  diverge. Does the divergence recover, stay anchored to the question, and look
  like a healthy transition?

The script consumes the existing full_*.npz features through
audit_utils.load_chains, then adds chain-level dynamic scores:

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
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise SystemExit("chain_dynamics_audit.py needs scikit-learn") from exc

from audit_utils import (
    Chain,
    auroc,
    bdir,
    cluster_boot_increment,
    finite_json,
    load_chains,
    safe_mean,
    safe_std,
)


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


CORE_TRANSITION_OBS = [
    ("spread",),
    ("spread", "uncertainty"),
    ("spread", "anchor_loss"),
    ("spread", "anchor_loss", "uncertainty"),
    ("spread", "anchor_loss", "uncertainty", "step_direction_jump"),
]


# Frozen before the cross-dataset replication run.  These are the original
# scalar/dynamic hypotheses, reported in their expected direction instead of
# selecting the better direction independently on every benchmark.
PREDECLARED_REPLICATION_FEATURES = (
    "spread",
    "d_spread",
    "step_direction_jump",
    "transition_surprise__spread",
    "transition_cusum__spread",
    "transition_surprise__spread_anchor_unc",
    "transition_cusum__spread_anchor_unc",
)


# These comparisons are fixed before the additive-value run.  The component
# models answer what the existing ``anchor_uncertainty`` shorthand contains;
# the augmentations ask whether one dynamic score adds information beyond that
# full baseline on exactly the same held-out rows and GroupKFold splits.
PREDECLARED_COMPONENT_MODELS = {
    "controls": ("logN", "pos"),
    "controls+spread": ("logN", "pos", "spread"),
    "controls+anchor": ("logN", "pos", "anchor_loss"),
    "controls+uncertainty": ("logN", "pos", "uncertainty"),
    "anchor_uncertainty": ("logN", "pos", "spread", "anchor_loss", "uncertainty"),
    "without_spread": ("logN", "pos", "anchor_loss", "uncertainty"),
    "without_anchor": ("logN", "pos", "spread", "uncertainty"),
    "without_uncertainty": ("logN", "pos", "spread", "anchor_loss"),
}


PREDECLARED_ADDITIVE_SIGNALS = (
    ("depth.raw_update.relative_norm", "depth_band_update_relative_norm"),
    (
        "depth.raw_update.relative_norm.length_residual",
        "depth_band_update_relative_norm_resid_ctrl",
    ),
    (
        "depth.prompt_conditioned_update.relative_norm",
        "depth_band_prompt_conditioned_norm",
    ),
    (
        "depth.prompt_conditioned_update.relative_norm.length_residual",
        "depth_band_prompt_conditioned_norm_resid_ctrl",
    ),
    ("temporal.d_spread.raw", "d_spread"),
    ("temporal.d_spread.length_residual", "d_spread_resid_ctrl"),
    ("temporal.direction_jump.raw", "step_direction_jump"),
    ("temporal.direction_jump.length_residual", "step_direction_jump_resid_ctrl"),
    ("transition.joint_surprise.raw", "transition_surprise__spread_anchor_unc"),
    (
        "transition.joint_surprise.length_residual",
        "transition_surprise__spread_anchor_unc_resid_ctrl",
    ),
    ("transition.joint_cusum.raw", "transition_cusum__spread_anchor_unc"),
    (
        "transition.joint_cusum.length_residual",
        "transition_cusum__spread_anchor_unc_resid_ctrl",
    ),
)


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


def lag_burst(x: np.ndarray, *, lag: int = 1) -> np.ndarray:
    """Causal jump over a short lag; positive values are local uncertainty bursts."""
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    for t in range(max(1, lag), len(v)):
        if np.isfinite(v[t]) and np.isfinite(v[t - lag]):
            out[t] = float(v[t] - v[t - lag])
    return out


def causal_rebound(x: np.ndarray, *, warmup: int = 1) -> np.ndarray:
    """Causal rise above the best previous low point."""
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    for t in range(warmup, len(v)):
        hist = v[:t]
        hist = hist[np.isfinite(hist)]
        if len(hist) and np.isfinite(v[t]):
            out[t] = float(v[t] - np.min(hist))
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

        for nm in ("unc_entropy", "unc_committal", "unc_epistemic"):
            if nm not in c.features:
                continue
            v = arr(c, nm)
            c.features[f"d_{nm}"] = delta(v)
            c.features[f"cz_{nm}"] = causal_z(v)
            c.features[f"edyn_{nm}_burst"] = lag_burst(v, lag=1)
            c.features[f"edyn_{nm}_rebound"] = causal_rebound(v)
            c.features[f"edyn_{nm}_cusum"] = leaky_cusum(c.features[f"cz_{nm}"], lam=lam, kref=kref)

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


def rolling_pattern_arrays(x: np.ndarray, *, window: int) -> Dict[str, np.ndarray]:
    """Causal window shape around a score, not a future-looking recovery label."""
    v = np.asarray(x, float)
    n = len(v)
    z = causal_z(v)
    drift = np.full(n, np.nan)
    rise = np.full(n, np.nan)
    volatility = np.full(n, np.nan)
    persistence = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - window + 1)
        seg = v[lo : t + 1]
        sf = seg[np.isfinite(seg)]
        if len(sf) == 0 or not np.isfinite(v[t]):
            continue
        drift[t] = float(v[t] - sf[0])
        volatility[t] = float(np.std(sf)) if len(sf) >= 2 else 0.0
        if len(sf) >= 2:
            d = np.diff(sf)
            rise[t] = float(np.mean(np.maximum(d, 0.0)))
        else:
            rise[t] = 0.0
        zseg = z[lo : t + 1]
        zseg = zseg[np.isfinite(zseg)]
        if len(zseg):
            persistence[t] = float(np.mean(zseg > 1.0))
    return {
        "drift": drift,
        "rise": rise,
        "vol": volatility,
        "persist": persistence,
    }


def add_trajectory_pattern_features(
    chains: Sequence[Chain],
    sources: Sequence[str],
    *,
    window: int,
    min_finite: int,
) -> List[str]:
    """Add short causal trajectory-shape summaries for sequence-level tests."""
    made: List[str] = []
    sources = [nm for nm in dict.fromkeys(sources) if finite_count(chains, nm) >= min_finite]
    for nm in sources:
        slug = feature_slug(nm)
        names = {k: f"pat_{slug}_{k}_w{window}" for k in ("drift", "rise", "vol", "persist")}
        for c in chains:
            pats = rolling_pattern_arrays(arr(c, nm), window=window)
            for k, out_name in names.items():
                c.features[out_name] = pats[k]
        for out_name in names.values():
            if finite_count(chains, out_name) >= min_finite:
                made.append(out_name)
    return made


def choose_obs(chains: Sequence[Chain], requested: Optional[str], *, min_finite: int) -> List[str]:
    if requested:
        return [x.strip() for x in requested.split(",") if x.strip()]
    candidates = [
        "spread",
        "anchor_loss",
        "uncertainty",
        "unc_entropy",
        "unc_committal",
        "unc_epistemic",
        "step_direction_jump",
        "geom_ae",
        "cloud_D",
    ]
    return [nm for nm in candidates if finite_count(chains, nm) >= min_finite]


def short_name(name: str) -> str:
    return {
        "spread": "spread",
        "anchor_loss": "anchor",
        "uncertainty": "unc",
        "unc_entropy": "UD",
        "unc_committal": "UC",
        "unc_epistemic": "UE",
        "step_direction_jump": "jump",
        "geom_ae": "ae",
        "cloud_D": "cloudD",
    }.get(name, name.replace(" ", "_"))


def feature_slug(name: str) -> str:
    slug = name
    slug = slug.replace("transition_surprise__", "ts__")
    slug = slug.replace("transition_cusum__", "tc__")
    out = []
    for ch in slug:
        out.append(ch if ch.isalnum() else "_")
    return "_".join([p for p in "".join(out).split("_") if p])


def obs_label(obs: Sequence[str]) -> str:
    return "_".join(short_name(x) for x in obs)


def parse_obs_grid(spec: Optional[str]) -> List[List[str]]:
    if not spec:
        return [list(x) for x in CORE_TRANSITION_OBS]
    out = []
    for block in spec.split(";"):
        names = [x.strip() for x in block.split(",") if x.strip()]
        if names:
            out.append(names)
    return out


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
    prefix: str,
    alias: bool,
    folds: int,
    ridge: float,
    eps_list: Sequence[float],
    lam: float,
    kref: float,
) -> Dict[str, object]:
    idx = np.arange(len(chains))
    groups = np.array([c.group for c in chains])
    n_splits = min(int(folds), len(np.unique(groups)))
    surprise_feature = f"transition_surprise__{prefix}"
    cusum_feature = f"transition_cusum__{prefix}"
    transition = [np.full(c.n_steps, np.nan) for c in chains]
    transition_cusum = [np.full(c.n_steps, np.nan) for c in chains]
    fold_of: Dict[int, int] = {}
    calibration: Dict[int, Dict[str, np.ndarray]] = {}
    controls = [nm for nm in ("logN", "pos") if finite_count(chains, nm) >= len(chains)]
    if n_splits < 2 or not obs:
        return {
            "prefix": prefix,
            "obs": list(obs),
            "controls": controls,
            "folds": 0,
            "surprise_feature": surprise_feature,
            "cusum_feature": cusum_feature,
            "online": [],
        }

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
        c.features[surprise_feature] = transition[i]
        c.features[cusum_feature] = transition_cusum[i]
        if alias:
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
                score = c.features[surprise_feature] if method == "single" else c.features[cusum_feature]
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
    return {
        "prefix": prefix,
        "obs": list(obs),
        "controls": controls,
        "folds": int(n_splits),
        "surprise_feature": surprise_feature,
        "cusum_feature": cusum_feature,
        "support": {
            "surprise_finite": finite_count(chains, surprise_feature),
            "cusum_finite": finite_count(chains, cusum_feature),
        },
        "online": online_rows,
    }


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


def cluster_boot_auc_ci(
    score: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Tuple[float, List[float]]:
    """Cluster-bootstrap a fixed-direction AUROC over complete chains/problems."""

    score = np.asarray(score, float)
    labels = np.asarray(labels, int)
    groups = np.asarray(groups)
    finite = np.isfinite(score)
    point = auroc(score[finite], labels[finite])
    unique = np.unique(groups[finite])
    if n_boot <= 0 or unique.size < 2 or not np.isfinite(point):
        return float(point), [float("nan"), float("nan")]

    by_group = {group: np.where(finite & (groups == group))[0] for group in unique}
    rng = np.random.default_rng(int(seed))
    values: List[float] = []
    for _ in range(int(n_boot)):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        index = np.concatenate([by_group[group] for group in chosen])
        if np.unique(labels[index]).size < 2:
            continue
        value = auroc(score[index], labels[index])
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return float(point), [float("nan"), float("nan")]
    low, high = np.percentile(np.asarray(values, float), [2.5, 97.5])
    return float(point), [float(low), float(high)]


def fixed_direction_feature_row(
    chains: Sequence[Chain],
    feature: str,
    *,
    n_boot: int,
    seed: int,
) -> Optional[Dict[str, object]]:
    """Evaluate one predeclared score with higher values fixed as higher risk."""

    X, labels, groups, _, _ = flatten_labeled(chains, [feature])
    if X.size == 0:
        return None
    score = np.asarray(X[:, 0], float)
    finite = np.isfinite(score)
    if finite.sum() < 30 or np.unique(labels[finite]).size < 2:
        return None
    auc, ci95 = cluster_boot_auc_ci(
        score,
        labels,
        groups,
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "feature": feature,
        "expected_direction": "higher_is_error",
        "auroc_high_is_error": auc,
        "ci95": ci95,
        "coverage": float(finite.mean()),
        "n": int(finite.sum()),
        "errors": int(labels[finite].sum()),
        "mean_non_error": safe_mean(score[(labels == 0) & finite]),
        "mean_gold_error": safe_mean(score[(labels == 1) & finite]),
    }


def _fold_local_impute(
    train: np.ndarray,
    test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Median-impute using training rows only.

    The previous implementation filled missing values once before GroupKFold,
    which leaked held-out feature marginals.  This helper keeps every fitted
    quantity inside the training fold.
    """

    x_train = np.asarray(train, float).copy()
    x_test = np.asarray(test, float).copy()
    for j in range(x_train.shape[1]):
        finite = np.isfinite(x_train[:, j])
        fill = float(np.median(x_train[finite, j])) if finite.any() else 0.0
        x_train[~np.isfinite(x_train[:, j]), j] = fill
        x_test[~np.isfinite(x_test[:, j]), j] = fill
    return x_train, x_test


def oof_logit_details(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return leakage-safe OOF probabilities and standardized fold coefficients."""

    X = np.asarray(X, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    pred = np.full(len(y), np.nan)
    coefficients: List[np.ndarray] = []
    n_splits = min(int(folds), len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return pred, np.empty((0, X.shape[1]), float)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        x_train, x_test = _fold_local_impute(X[tr], X[te])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        clf.fit(x_train, y[tr])
        pred[te] = clf.predict_proba(x_test)[:, 1]
        coefficients.append(np.asarray(clf[-1].coef_[0], float))
    return pred, np.asarray(coefficients, float)


def oof_logit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int) -> np.ndarray:
    pred, _ = oof_logit_details(X, y, groups, folds)
    return pred


def candidate_label(c: Chain, t: int) -> Optional[int]:
    if c.correct or t < c.gold:
        return 0
    if t == c.gold:
        return 1
    return None


def add_control_residuals(
    chains: Sequence[Chain],
    names: Sequence[str],
    controls: Sequence[str],
    *,
    folds: int,
    ridge: float,
    suffix: str = "_resid_ctrl",
) -> Dict[str, object]:
    """Cross-fit feature residuals after removing logN/position from non-error steps.

    The fit uses only y=0 candidate steps in the training fold, then scores held-out
    chains. This keeps gold-error steps from defining their own nuisance baseline.
    """
    names = [nm for nm in dict.fromkeys(names) if finite_count(chains, nm) >= 30]
    controls = [nm for nm in controls if finite_count(chains, nm) >= 30]
    idx = np.arange(len(chains))
    groups = np.array([c.group for c in chains])
    n_splits = min(int(folds), len(np.unique(groups)))
    made = []
    if n_splits < 2 or not names or not controls:
        return {"features": made, "controls": list(controls), "folds": 0}

    for nm in names:
        out_name = f"{nm}{suffix}"
        for c in chains:
            c.features[out_name] = np.full(c.n_steps, np.nan)
        for tr, te in GroupKFold(n_splits=n_splits).split(idx[:, None], idx, groups):
            Xtr, ytr = [], []
            for i in tr:
                c = chains[i]
                for t in range(c.n_steps):
                    lab = candidate_label(c, t)
                    if lab != 0:
                        continue
                    x = np.array([arr(c, ctl)[t] for ctl in controls], float)
                    val = arr(c, nm)[t]
                    if np.isfinite(val) and np.all(np.isfinite(x)):
                        Xtr.append(x)
                        ytr.append(val)
            if len(ytr) < max(20, 4 * (len(controls) + 1)):
                continue
            Xtr = np.asarray(Xtr, float)
            ytr = np.asarray(ytr, float)
            mu = Xtr.mean(axis=0)
            sd = Xtr.std(axis=0) + EPS
            Ztr = (Xtr - mu) / sd
            Dtr = np.column_stack([Ztr, np.ones(len(Ztr))])
            reg = ridge * np.eye(Dtr.shape[1])
            reg[-1, -1] = 0.0
            beta = np.linalg.solve(Dtr.T @ Dtr + reg, Dtr.T @ ytr)
            for i in te:
                c = chains[i]
                for t in range(c.n_steps):
                    if candidate_label(c, t) is None:
                        continue
                    x = np.array([arr(c, ctl)[t] for ctl in controls], float)
                    val = arr(c, nm)[t]
                    if np.isfinite(val) and np.all(np.isfinite(x)):
                        z = (x - mu) / sd
                        pred = float(np.r_[z, 1.0] @ beta)
                        c.features[out_name][t] = float(val - pred)
        if finite_count(chains, out_name) >= 30:
            made.append(out_name)
    return {"features": made, "controls": list(controls), "folds": int(n_splits)}


def group_score(chains: Sequence[Chain], names: Sequence[str], *, folds: int) -> Optional[Dict[str, object]]:
    names = [nm for nm in names if finite_count(chains, nm) >= 30]
    if not names:
        return None
    X, y, g, _, _ = flatten_labeled(chains, names)
    if X.shape[1] == 0:
        return None
    s = oof_logit(X, y, g, folds)
    m = np.isfinite(s)
    return {
        "features": list(names),
        "score": s,
        "y": y,
        "groups": g,
        "auroc": auroc(s[m], y[m]) if m.any() else float("nan"),
        "n": int(m.sum()),
    }


def group_table(chains: Sequence[Chain], groups: Dict[str, Sequence[str]], *, folds: int):
    out = {}
    for label, names in groups.items():
        scored = group_score(chains, names, folds=folds)
        if scored is None:
            continue
        out[label] = {
            "features": scored["features"],
            "auroc": scored["auroc"],
            "n": scored["n"],
        }
    return out


def group_increment_table(
    chains: Sequence[Chain],
    groups: Dict[str, Sequence[str]],
    *,
    baseline: str,
    folds: int,
    n_boot: int,
    seed: int = 0,
):
    scored = {}
    for label, names in groups.items():
        val = group_score(chains, names, folds=folds)
        if val is not None and val["n"] >= 30:
            scored[label] = val
    if baseline not in scored:
        return []
    base = scored[baseline]
    rows = []
    for j, (label, val) in enumerate(scored.items()):
        if label == baseline:
            continue
        inc = cluster_boot_increment(
            val["score"],
            base["score"],
            val["y"],
            val["groups"],
            n_boot=n_boot,
            seed=seed + j,
        )
        rows.append(
            {
                "group": label,
                "baseline": baseline,
                "auroc": val["auroc"],
                "baseline_auroc": base["auroc"],
                "increment": inc,
                "features": val["features"],
                "n": val["n"],
            }
        )
    rows.sort(key=lambda r: np.nan_to_num(r["increment"]["point"], nan=-999), reverse=True)
    return rows


def auprc(score: np.ndarray, labels: np.ndarray) -> float:
    score = np.asarray(score, float)
    labels = np.asarray(labels, int)
    finite = np.isfinite(score)
    if finite.sum() == 0 or np.unique(labels[finite]).size < 2:
        return float("nan")
    return float(average_precision_score(labels[finite], score[finite]))


def cluster_boot_metric_increment(
    augmented: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    metric,
    n_boot: int,
    seed: int,
) -> Dict[str, object]:
    """Cluster-bootstrap a paired metric difference on shared OOF rows."""

    augmented = np.asarray(augmented, float)
    baseline = np.asarray(baseline, float)
    labels = np.asarray(labels, int)
    groups = np.asarray(groups)
    finite = np.isfinite(augmented) & np.isfinite(baseline)
    if finite.sum() < 30 or np.unique(labels[finite]).size < 2:
        return {
            "point": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "supports_positive_increment": False,
        }
    point = float(metric(augmented[finite], labels[finite]) - metric(baseline[finite], labels[finite]))
    unique = np.unique(groups[finite])
    if n_boot <= 0 or unique.size < 2:
        return {
            "point": point,
            "lo": float("nan"),
            "hi": float("nan"),
            "supports_positive_increment": False,
        }
    by_group = {group: np.where(finite & (groups == group))[0] for group in unique}
    rng = np.random.default_rng(int(seed))
    values: List[float] = []
    for _ in range(int(n_boot)):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        index = np.concatenate([by_group[group] for group in chosen])
        if np.unique(labels[index]).size < 2:
            continue
        value = float(metric(augmented[index], labels[index]) - metric(baseline[index], labels[index]))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {
            "point": point,
            "lo": float("nan"),
            "hi": float("nan"),
            "supports_positive_increment": False,
        }
    low, high = np.percentile(np.asarray(values, float), [2.5, 97.5])
    return {
        "point": point,
        "lo": float(low),
        "hi": float(high),
        "supports_positive_increment": bool(low > 0.0),
    }


def _model_metrics(score: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    score = np.asarray(score, float)
    labels = np.asarray(labels, int)
    finite = np.isfinite(score)
    return {
        "auroc": auroc(score[finite], labels[finite]),
        "auprc": auprc(score[finite], labels[finite]),
        "n": int(finite.sum()),
        "errors": int(labels[finite].sum()) if finite.any() else 0,
        "coverage": float(finite.mean()) if len(finite) else 0.0,
    }


def build_fixed_mechanism_increment_audit(
    chains: Sequence[Chain],
    *,
    folds: int,
    n_boot: int,
    seed: int = 20260714,
) -> Dict[str, object]:
    """Isolate baseline components and fixed dynamic additions without selection.

    All models share candidate rows, GroupKFold splits, fold-local imputation,
    scaling, and logistic-regression capacity.  Therefore the paired increment
    reflects the one added signal rather than a split or sample-set change.
    """

    required_baseline = list(PREDECLARED_COMPONENT_MODELS["anchor_uncertainty"])
    missing_baseline = [name for name in required_baseline if finite_count(chains, name) < 30]
    if missing_baseline:
        return {
            "ready": False,
            "reason": "missing baseline features",
            "missing": missing_baseline,
        }

    additive = [
        (label, feature)
        for label, feature in PREDECLARED_ADDITIVE_SIGNALS
        if finite_count(chains, feature) >= 30
    ]
    all_features = list(
        dict.fromkeys(
            required_baseline
            + [name for names in PREDECLARED_COMPONENT_MODELS.values() for name in names]
            + [feature for _, feature in additive]
        )
    )
    X, labels, groups, _, _ = flatten_labeled(chains, all_features)
    if X.size == 0 or np.unique(labels).size < 2:
        return {"ready": False, "reason": "no labeled candidate rows"}
    column = {name: index for index, name in enumerate(all_features)}

    predictions: Dict[str, np.ndarray] = {}
    coefficients: Dict[str, np.ndarray] = {}
    model_rows: List[Dict[str, object]] = []

    def score_model(name: str, features: Sequence[str]) -> np.ndarray:
        indices = [column[feature] for feature in features]
        score, coef = oof_logit_details(X[:, indices], labels, groups, folds)
        predictions[name] = score
        coefficients[name] = coef
        model_rows.append(
            {
                "model": name,
                "features": list(features),
                **_model_metrics(score, labels),
            }
        )
        return score

    for name, features in PREDECLARED_COMPONENT_MODELS.items():
        score_model(name, features)

    baseline_name = "anchor_uncertainty"
    baseline_score = predictions[baseline_name]
    component_rows: List[Dict[str, object]] = []
    reduced_by_component = {
        "spread": "without_spread",
        "anchor": "without_anchor",
        "uncertainty": "without_uncertainty",
    }
    component_model_by_name = {
        "spread": "controls+spread",
        "anchor": "controls+anchor",
        "uncertainty": "controls+uncertainty",
    }
    for index, (component, reduced_name) in enumerate(reduced_by_component.items()):
        reduced_score = predictions[reduced_name]
        component_model = component_model_by_name[component]
        component_score = predictions[component_model]
        controls_score = predictions["controls"]
        component_rows.append(
            {
                "component": component,
                "full_model": baseline_name,
                "reduced_model": reduced_name,
                "component_model": component_model,
                "controls_model": "controls",
                "component_over_controls_auroc": cluster_boot_metric_increment(
                    component_score,
                    controls_score,
                    labels,
                    groups,
                    metric=auroc,
                    n_boot=n_boot,
                    seed=seed + 20 + index,
                ),
                "component_over_controls_auprc": cluster_boot_metric_increment(
                    component_score,
                    controls_score,
                    labels,
                    groups,
                    metric=auprc,
                    n_boot=n_boot,
                    seed=seed + 40 + index,
                ),
                "auroc_increment": cluster_boot_metric_increment(
                    baseline_score,
                    reduced_score,
                    labels,
                    groups,
                    metric=auroc,
                    n_boot=n_boot,
                    seed=seed + index,
                ),
                "auprc_increment": cluster_boot_metric_increment(
                    baseline_score,
                    reduced_score,
                    labels,
                    groups,
                    metric=auprc,
                    n_boot=n_boot,
                    seed=seed + 100 + index,
                ),
            }
        )

    additive_rows: List[Dict[str, object]] = []
    for index, (label, feature) in enumerate(additive):
        model_name = f"anchor_uncertainty+{label}"
        features = required_baseline + [feature]
        eligible = np.isfinite(X[:, column[feature]])
        if eligible.sum() < 30 or np.unique(labels[eligible]).size < 2:
            continue
        eligible_X = X[eligible]
        eligible_labels = labels[eligible]
        eligible_groups = groups[eligible]
        baseline_indices = [column[name] for name in required_baseline]
        augmented_indices = [column[name] for name in features]
        eligible_baseline, _ = oof_logit_details(
            eligible_X[:, baseline_indices],
            eligible_labels,
            eligible_groups,
            folds,
        )
        augmented_score, coef = oof_logit_details(
            eligible_X[:, augmented_indices],
            eligible_labels,
            eligible_groups,
            folds,
        )
        model_rows.append(
            {
                "model": model_name,
                "features": list(features),
                **_model_metrics(augmented_score, eligible_labels),
            }
        )
        signal_coef = coef[:, -1] if coef.ndim == 2 and coef.shape[1] else np.asarray([], float)
        additive_rows.append(
            {
                "signal": label,
                "feature": feature,
                "baseline_model": baseline_name,
                "augmented_model": model_name,
                "eligible_rows": int(eligible.sum()),
                "eligible_coverage": float(eligible.mean()),
                "baseline": _model_metrics(eligible_baseline, eligible_labels),
                "augmented": _model_metrics(augmented_score, eligible_labels),
                "auroc_increment": cluster_boot_metric_increment(
                    augmented_score,
                    eligible_baseline,
                    eligible_labels,
                    eligible_groups,
                    metric=auroc,
                    n_boot=n_boot,
                    seed=seed + 1000 + index,
                ),
                "auprc_increment": cluster_boot_metric_increment(
                    augmented_score,
                    eligible_baseline,
                    eligible_labels,
                    eligible_groups,
                    metric=auprc,
                    n_boot=n_boot,
                    seed=seed + 2000 + index,
                ),
                "standardized_coefficient": {
                    "median": safe_mean([np.median(signal_coef)]) if signal_coef.size else float("nan"),
                    "positive_fraction": float(np.mean(signal_coef > 0.0)) if signal_coef.size else float("nan"),
                    "fold_values": signal_coef.tolist(),
                },
            }
        )

    return {
        "ready": True,
        "candidate_rows": int(len(labels)),
        "errors": int(labels.sum()),
        "folds": int(min(int(folds), len(np.unique(groups)))),
        "baseline_definition": {
            "name": baseline_name,
            "features": required_baseline,
            "note": "supervised OOF logistic baseline; anchor_loss is only one component",
        },
        "model_scores": model_rows,
        "unique_component_value": component_rows,
        "additive_value": additive_rows,
    }


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


def build_predeclared_replication(
    chains: Sequence[Chain],
    features: Sequence[str],
    *,
    n_boot: int,
    seed: int = 13,
) -> List[Dict[str, object]]:
    """Build selection-free replication rows for raw and nuisance-residual scores."""

    rows: List[Dict[str, object]] = []
    for index, feature in enumerate(features):
        raw = fixed_direction_feature_row(
            chains,
            feature,
            n_boot=n_boot,
            seed=seed + 101 * index,
        )
        if raw is None:
            continue
        residual_name = f"{feature}_resid_ctrl"
        residual = fixed_direction_feature_row(
            chains,
            residual_name,
            n_boot=n_boot,
            seed=seed + 10_000 + 101 * index,
        )
        raw_localization = within_chain_rank(chains, feature, sign=1.0)
        residual_localization = (
            within_chain_rank(chains, residual_name, sign=1.0)
            if residual is not None
            else {}
        )
        rows.append(
            {
                "feature": feature,
                "expected_direction": "higher_is_error",
                "raw": raw,
                "nuisance_residual": residual or {},
                "within_chain": raw_localization,
                "within_chain_residual": residual_localization,
            }
        )
    return rows


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
    if args.obs:
        obs_grid = [choose_obs(chains, args.obs, min_finite=args.min_finite)]
    else:
        obs_grid = []
        for obs in parse_obs_grid(args.obs_grid):
            ok = [nm for nm in obs if finite_count(chains, nm) >= args.min_finite]
            if len(ok) == len(obs):
                obs_grid.append(ok)
        auto = choose_obs(chains, None, min_finite=args.min_finite)
        if auto and auto not in obs_grid:
            obs_grid.append(auto)
    # Keep the central hypothesis first if available: divergence + anchor + uncertainty.
    preferred = ["spread", "anchor_loss", "uncertainty"]
    obs_grid = sorted(obs_grid, key=lambda x: 0 if x == preferred else 1)

    transition_models = {}
    transition_feature_names: List[str] = []
    eps_values = [float(x) for x in args.eps_list.split(",")]
    for i, obs in enumerate(obs_grid):
        prefix = obs_label(obs)
        transition = fit_crossfit_transition(
            chains,
            obs,
            prefix=prefix,
            alias=(i == 0),
            folds=args.folds,
            ridge=args.ridge,
            eps_list=eps_values,
            lam=args.lam,
            kref=args.kref,
        )
        transition_models[prefix] = transition
        transition_feature_names.extend([transition["surprise_feature"], transition["cusum_feature"]])

    if transition_feature_names and "transition_surprise" not in transition_feature_names:
        transition_feature_names.extend(["transition_surprise", "transition_cusum"])

    uncertainty_family = [
        "unc_entropy",
        "unc_committal",
        "unc_epistemic",
    ]
    uncertainty_dynamics = []
    for nm in uncertainty_family:
        uncertainty_dynamics.extend(
            [
                f"d_{nm}",
                f"cz_{nm}",
                f"edyn_{nm}_burst",
                f"edyn_{nm}_rebound",
                f"edyn_{nm}_cusum",
            ]
        )
    uncertainty_dynamics = [
        nm for nm in uncertainty_dynamics if finite_count(chains, nm) >= args.min_finite
    ]
    static = [
        "spread",
        "resultant",
        "uncertainty",
        "anchor_loss",
        "step_direction_jump",
        "logN",
        "pos",
    ] + uncertainty_family
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
    ] + uncertainty_dynamics + transition_feature_names

    residual_source = [
        "spread",
        "step_direction_jump",
        "depth_band_update_relative_norm",
        "depth_band_prompt_conditioned_norm",
        "depth_band_state_rewire",
        "unanchored_divergence",
        "confident_divergence",
        "d_spread",
        "cz_spread",
        "spread_cusum",
    ] + uncertainty_dynamics + transition_feature_names
    residual_info = add_control_residuals(
        chains,
        residual_source,
        ["logN", "pos"],
        folds=args.folds,
        ridge=args.ridge,
    )
    residual_names = residual_info["features"]
    pattern_sources = [
        "spread",
        "anchor_loss",
        "uncertainty",
        "unc_entropy",
        "unc_committal",
        "unc_epistemic",
        "unanchored_divergence",
        "confident_divergence",
        "transition_surprise",
    ]
    pattern_names = add_trajectory_pattern_features(
        chains,
        pattern_sources,
        window=args.pattern_window,
        min_finite=args.min_finite,
    )

    groups = {
        "static": ["spread", "logN", "pos"],
        "anchor_uncertainty": ["spread", "anchor_loss", "uncertainty", "logN", "pos"],
        "explicit_uncertainty": ["spread", "anchor_loss", "unc_entropy", "unc_committal", "unc_epistemic", "logN", "pos"],
        "uncertainty_dynamics": ["spread", "anchor_loss", "logN", "pos"] + uncertainty_dynamics,
        "dynamic_online": [
            "spread",
            "d_spread",
            "cz_spread",
            "spread_cusum",
            "unanchored_cusum",
            "confident_cusum",
            "transition_surprise",
            "transition_cusum",
            "logN",
            "pos",
        ]
        + uncertainty_dynamics,
        "transition_ablation": ["spread", "logN", "pos"] + transition_feature_names,
        "control_residual": residual_names,
        "trajectory_pattern": ["spread", "anchor_loss", "uncertainty", "logN", "pos"] + pattern_names,
        "sequence_state": [
            "spread",
            "anchor_loss",
            "uncertainty",
            "d_spread",
            "cz_spread",
            "unanchored_cusum",
            "confident_cusum",
            "transition_surprise",
            "transition_cusum",
            "logN",
            "pos",
        ]
        + pattern_names,
        "offline_recovery": ["spread", "next_recovery_1", f"next_recovery_{args.recovery_horizon}", "logN", "pos"],
    }
    all_names = list(dict.fromkeys(static + dynamic + residual_names + pattern_names))
    causal_names = [
        nm
        for nm in dict.fromkeys(
            [
                "d_spread",
                "cz_spread",
                "unanchored_cusum",
                "confident_cusum",
                "transition_surprise",
                "transition_cusum",
            ]
            + pattern_names
        )
        if finite_count(chains, nm) >= 30
    ]
    predeclared_replication = build_predeclared_replication(
        chains,
        PREDECLARED_REPLICATION_FEATURES,
        n_boot=args.n_boot,
        seed=13,
    )
    fixed_mechanism_increment = build_fixed_mechanism_increment_audit(
        chains,
        folds=args.folds,
        n_boot=args.n_boot,
    )
    group_scores = group_table(chains, groups, folds=args.folds)
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
        "transition_models": transition_models,
        "primary_transition_model": next(iter(transition_models.values()), {}),
        "control_residualization": residual_info,
        "trajectory_pattern_features": pattern_names,
        "predeclared_replication": predeclared_replication,
        "fixed_mechanism_increment": fixed_mechanism_increment,
        "overall_features": feature_table(chains, all_names, top=args.top),
        "high_spread_features": feature_table(chains, all_names, high_spread_q=args.high_spread_q, top=args.top),
        "control_residual_features": feature_table(chains, residual_names, top=args.top),
        "transition_ablation_features": feature_table(chains, transition_feature_names, top=args.top),
        "localization": localization_table(chains, all_names, top=args.top),
        "residual_localization": localization_table(chains, residual_names, top=args.top),
        "causal_pattern_localization": localization_table(chains, causal_names, top=args.top),
        "group_oof": group_scores,
        "group_increments_vs_anchor_uncertainty": group_increment_table(
            chains,
            groups,
            baseline="anchor_uncertainty",
            folds=args.folds,
            n_boot=args.n_boot,
        ),
        "event_study": event_study(
            chains,
            [
                nm
                for nm in (
                    "spread",
                    "d_spread",
                    "next_recovery_1",
                    "anchor_loss",
                    "uncertainty",
                    "unc_entropy",
                    "unc_committal",
                    "unc_epistemic",
                    "unanchored_divergence",
                    "confident_divergence",
                    "transition_surprise",
                    "transition_cusum",
                )
                + tuple(uncertainty_dynamics[:8])
                + tuple(pattern_names[:8])
                if finite_count(chains, nm) >= 20
            ],
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
    tm = res.get("primary_transition_model", {})
    print(f"primary transition obs={tm.get('obs')} controls={tm.get('controls')} folds={tm.get('folds')}")
    if res.get("transition_models"):
        print("transition ablations:")
        for k, v in res["transition_models"].items():
            sup = v.get("support", {})
            print(
                f"  {k:36s} obs={v.get('obs')} "
                f"finite={sup.get('surprise_finite', 0)}"
            )
    print_rows(res["overall_features"], label="Overall step/gold-error scores")
    print_rows(res["high_spread_features"], label="High-divergence subset scores")
    print_rows(res.get("transition_ablation_features", []), label="Transition ablation scores")
    print_rows(res.get("control_residual_features", []), label="Control-residualized scores")

    print("\nPredeclared replication signals (higher fixed as error):")
    for row in res.get("predeclared_replication", []):
        raw = row.get("raw", {})
        residual = row.get("nuisance_residual", {})
        raw_loc = row.get("within_chain", {})
        residual_loc = row.get("within_chain_residual", {})
        print(
            f"  {row['feature']:42s} raw {float(raw.get('auroc_high_is_error', np.nan)):.3f} "
            f"resid {float(residual.get('auroc_high_is_error', np.nan)):.3f} "
            f"top1 {float(raw_loc.get('top1', np.nan)):.3f}/"
            f"{float(residual_loc.get('top1', np.nan)):.3f}"
        )

    fixed = res.get("fixed_mechanism_increment", {})
    if fixed.get("ready"):
        print("\nFixed component value inside anchor_uncertainty:")
        for row in fixed.get("unique_component_value", []):
            auc_inc = row["auroc_increment"]
            pr_inc = row["auprc_increment"]
            print(
                f"  {row['component']:12s} AUROC {auc_inc['point']:+.3f} "
                f"[{auc_inc['lo']:+.3f},{auc_inc['hi']:+.3f}] "
                f"AUPRC {pr_inc['point']:+.3f} [{pr_inc['lo']:+.3f},{pr_inc['hi']:+.3f}]"
            )
        print("\nFixed additions beyond anchor_uncertainty:")
        for row in fixed.get("additive_value", []):
            auc_inc = row["auroc_increment"]
            pr_inc = row["auprc_increment"]
            coef = row["standardized_coefficient"]
            print(
                f"  {row['signal']:48s} AUROC {auc_inc['point']:+.3f} "
                f"[{auc_inc['lo']:+.3f},{auc_inc['hi']:+.3f}] "
                f"AUPRC {pr_inc['point']:+.3f} "
                f"coef {coef['median']:+.3f} sign+ {coef['positive_fraction']:.2f}"
            )

    print("\nWithin-chain localization:")
    for r in res["localization"][:12]:
        gain = r["top1"] - r["expected_top1"]
        print(f"  {r['feature']:26s} top1 {r['top1']:.3f} exp {r['expected_top1']:.3f} gain {gain:+.3f} n={r['n']}")

    print("\nResidualized localization:")
    for r in res.get("residual_localization", [])[:8]:
        gain = r["top1"] - r["expected_top1"]
        print(f"  {r['feature']:26s} top1 {r['top1']:.3f} exp {r['expected_top1']:.3f} gain {gain:+.3f} n={r['n']}")

    print("\nCausal/pattern localization:")
    for r in res.get("causal_pattern_localization", [])[:8]:
        gain = r["top1"] - r["expected_top1"]
        print(f"  {r['feature']:26s} top1 {r['top1']:.3f} exp {r['expected_top1']:.3f} gain {gain:+.3f} n={r['n']}")

    print("\nOOF groups:")
    for k, v in res["group_oof"].items():
        print(f"  {k:20s} AUROC {v['auroc']:.3f} n={v['n']} features={len(v['features'])}")

    print("\nOOF increments vs anchor_uncertainty:")
    for r in res.get("group_increments_vs_anchor_uncertainty", [])[:8]:
        inc = r["increment"]
        sig = "SIG" if inc.get("sig") else "ns"
        print(
            f"  {r['group']:20s} {r['baseline_auroc']:.3f}->{r['auroc']:.3f} "
            f"inc {inc['point']:+.3f} [{inc['lo']:+.3f},{inc['hi']:+.3f}] {sig}"
        )

    print("\nOnline transition alarms:")
    for label, model in res.get("transition_models", {}).items():
        for r in model.get("online", []):
            print(
                f"  {label:24s} {r['method']:6s} eps {r['eps']:.2f} FPR {r['fpr']:.3f} "
                f"recall {r['recall']:.3f} delay {r['median_delay']:+.1f} early {r['early_warn']:.3f}"
            )


def _object_array(xs: Sequence[object]) -> np.ndarray:
    out = np.empty(len(xs), dtype=object)
    out[:] = list(xs)
    return out


def make_selftest_npz(path: str, *, n_chains: int = 90, layer: int = 14, seed: int = 21) -> None:
    rng = np.random.default_rng(seed)
    layers = np.array([layer], dtype=int)
    sv_layers = np.array([layer - 2, layer], dtype=int)
    cloud_names = np.array(["resultant", "coherence", "cloud_D"], dtype=object)
    geom_names = np.array(["ae"], dtype=object)
    gold, pids, ranges_all = [], [], []
    stepcloud, stepgeom, tok_ud, tok_uc, tok_ue, tok_ue_offsets, stepvec, qvecs, texts = [], [], [], [], [], [], [], [], []
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
        qv = np.stack([q, q], axis=0)
        sv = np.zeros((T, 2, d), float)
        prev = q
        for t in range(T):
            off = rng.normal(size=d)
            off = off - np.dot(off, q) * q
            off = off / max(np.linalg.norm(off), EPS)
            cur = q_align[t] * q + max(0.0, 1.0 - q_align[t]) * off + 0.03 * rng.normal(size=d)
            if t > 0:
                cur = 0.65 * prev + 0.35 * cur
            cur = cur / max(np.linalg.norm(cur), EPS)
            previous_depth = 0.82 * cur + 0.18 * q + 0.01 * rng.normal(size=d)
            previous_depth = previous_depth / max(np.linalg.norm(previous_depth), EPS)
            sv[t, 0] = previous_depth
            sv[t, 1] = cur
            prev = cur
        ud = np.concatenate([rng.normal(uncertainty[t], 0.02, size=int(lens[t])) for t in range(T)])
        uc = np.concatenate([rng.normal(uncertainty[t] * 0.8, 0.02, size=int(lens[t])) for t in range(T)])
        ue_full = np.concatenate([rng.normal(uncertainty[t] * 1.2, 0.03, size=int(lens[t])) for t in range(T)])
        ue_off = np.arange(0, len(ue_full), 2, dtype=int)
        stepcloud.append(sc)
        stepgeom.append(sg)
        tok_ud.append(ud)
        tok_uc.append(uc)
        tok_ue.append(ue_full[ue_off])
        tok_ue_offsets.append(ue_off)
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
        tok_U_E=_object_array(tok_ue),
        tok_U_E_offsets=_object_array(tok_ue_offsets),
        stepvec=_object_array(stepvec),
        qvec=np.asarray(qvecs, float),
        sv_layers=sv_layers,
    )


def assert_selftest(res: Dict[str, object]) -> None:
    rows = {r["feature"]: r for r in res["overall_features"]}
    if rows.get("transition_surprise", {}).get("auroc_bestdir", 0.0) < 0.8:
        raise SystemExit("selftest failed: transition_surprise did not recover injected transition failure")
    high = {r["feature"]: r for r in res["high_spread_features"]}
    if high.get("transition_surprise", {}).get("auroc_bestdir", 0.0) < 0.8:
        raise SystemExit("selftest failed: transition_surprise did not separate high-spread failures")
    resid = {r["feature"]: r for r in res.get("control_residual_features", [])}
    best_resid = max(
        [r.get("auroc_bestdir", 0.0) for name, r in resid.items() if name.startswith("transition_surprise")],
        default=0.0,
    )
    if best_resid < 0.8:
        raise SystemExit("selftest failed: control-residualized transition_surprise is too weak")
    loc = {r["feature"]: r for r in res["localization"]}
    if loc.get("transition_cusum", {}).get("top1", 0.0) < 0.6:
        raise SystemExit("selftest failed: transition_cusum did not localize gold steps")


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
    ap.add_argument(
        "--obs_grid",
        default=None,
        help="semicolon-separated transition obs sets, e.g. 'spread;spread,uncertainty;spread,anchor_loss,uncertainty'",
    )
    ap.add_argument("--min_finite", type=int, default=50)
    ap.add_argument("--recovery_horizon", type=int, default=2)
    ap.add_argument("--high_spread_q", type=float, default=0.70)
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--kref", type=float, default=0.25)
    ap.add_argument("--eps_list", default="0.05,0.10,0.20")
    ap.add_argument("--pattern_window", type=int, default=3)
    ap.add_argument("--event_window", type=int, default=3)
    ap.add_argument("--n_boot", type=int, default=200)
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
