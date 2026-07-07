#!/usr/bin/env python3
"""Joint audit for directional consensus and residual-rank trajectory.

This script studies the two signals that currently matter most in the project:

1. consensus loss: `spread = 1 - kappa`
2. high-dimensional residual dispersion: `res_eff_rank`

It answers two questions:

* Do the two signals jointly define a stronger failure state than either alone?
* How do the two signals evolve along a reasoning trajectory, especially around
  the gold first-error step?

Unlike `directional_dispersion_mechanism_audit.py`, this script deliberately
keeps pre-error, first-error, and post-error steps in memory.  It uses gold
first-error labels only for evaluation, not for computing the geometric
features.
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

import directional_dispersion_mechanism_audit as ddm
from multisample_temporal_rupture_audit import descriptive, finite_json


EPS = 1e-12


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


@dataclass
class SignalRow:
    chain_idx: int
    problem_id: int
    step_idx: int
    gold_error_step: int
    phase: str
    y_first_error: int
    features: Dict[str, float]


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


def z_against(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    med, scale = robust_center_scale(ref)
    all_vals = np.asarray(x, dtype=np.float64)
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size > 1:
        scale = max(scale, 0.25 * float(np.std(finite, ddof=1)))
    out = (np.asarray(x, dtype=np.float64) - med) / max(scale, EPS)
    out[~np.isfinite(out)] = np.nan
    return out


def feature_array(rows: Sequence[SignalRow], name: str) -> np.ndarray:
    return np.asarray([r.features.get(name, float("nan")) for r in rows], dtype=np.float64)


def y_array(rows: Sequence[SignalRow]) -> np.ndarray:
    return np.asarray([r.y_first_error for r in rows], dtype=int)


def group_array(rows: Sequence[SignalRow]) -> np.ndarray:
    return np.asarray([r.chain_idx for r in rows], dtype=int)


def phase_for(gold: int, step_idx: int) -> str:
    if gold < 0:
        return "correct_chain"
    if step_idx < gold:
        return "pre_error"
    if step_idx == gold:
        return "first_error"
    return "post_error"


def load_rows(path: str, args: argparse.Namespace) -> Tuple[List[SignalRow], Dict[str, Any]]:
    # Keep every step so that trajectory/cascade summaries are meaningful.
    ddm_args = argparse.Namespace(**vars(args))
    ddm_args.label_mode = "error_and_after" if args.policy == "gold_error_step" else "chain_final"
    base_rows, meta = ddm.load_step_rows(path, ddm_args)
    rows: List[SignalRow] = []
    for r in base_rows:
        phase = phase_for(r.gold_error_step, r.step_idx)
        if args.control_pool == "correct_chain" and phase == "pre_error":
            eligible_control = False
        elif args.control_pool == "pre_error" and phase == "correct_chain":
            eligible_control = False
        else:
            eligible_control = phase in {"correct_chain", "pre_error"}
        y = int(phase == "first_error")
        if phase == "post_error":
            # Post-error rows are used for trajectory/cascade summaries, not as
            # negatives in first-error detection.
            y_eval = -1
        elif y == 0 and not eligible_control:
            y_eval = -1
        else:
            y_eval = y
        feats = dict(r.features)
        feats["spread"] = 1.0 - feats.get("kappa", float("nan"))
        rows.append(
            SignalRow(
                chain_idx=r.chain_idx,
                problem_id=r.problem_id,
                step_idx=r.step_idx,
                gold_error_step=r.gold_error_step,
                phase=phase,
                y_first_error=y_eval,
                features=feats,
            )
        )
    meta = {
        **meta,
        "control_pool": args.control_pool,
        "trajectory_rows": int(len(rows)),
        "n_first_error": int(sum(r.phase == "first_error" for r in rows)),
        "n_pre_error": int(sum(r.phase == "pre_error" for r in rows)),
        "n_post_error": int(sum(r.phase == "post_error" for r in rows)),
        "n_correct_chain_steps": int(sum(r.phase == "correct_chain" for r in rows)),
    }
    return rows, meta


def design_matrix(rows: Sequence[SignalRow], controls: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    cols: List[np.ndarray] = [np.ones(len(rows), dtype=np.float64)]
    names = ["intercept"]
    for name in controls:
        x = feature_array(rows, name)
        cols.append(x)
        names.append(name)
        if name in {"logN", "kappa", "spread", "pos"}:
            cols.append(x * x)
            names.append(f"{name}^2")
    X = np.column_stack(cols)
    return X, names


def ridge_residual(
    rows: Sequence[SignalRow],
    target: str,
    controls: Sequence[str],
    ref_mask: np.ndarray,
    *,
    ridge: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y = feature_array(rows, target)
    X, names = design_matrix(rows, controls)
    m = ref_mask & np.isfinite(y) & np.isfinite(X).all(axis=1)
    out = np.full(len(rows), np.nan, dtype=np.float64)
    if m.sum() < len(names) + 5:
        return out, {"target": target, "controls": list(controls), "n_fit": int(m.sum()), "ok": False}
    Xfit = X[m].copy()
    yfit = y[m]
    means = np.nanmean(Xfit[:, 1:], axis=0)
    stds = np.nanstd(Xfit[:, 1:], axis=0)
    stds = np.where(stds > EPS, stds, 1.0)
    Xs = X.copy()
    Xs[:, 1:] = (Xs[:, 1:] - means[None, :]) / stds[None, :]
    Xfit = Xs[m]
    xtx = Xfit.T @ Xfit
    pen = float(ridge) * np.eye(xtx.shape[0])
    pen[0, 0] = 0.0
    beta = np.linalg.pinv(xtx + pen) @ Xfit.T @ yfit
    pred = Xs @ beta
    ok = np.isfinite(y) & np.isfinite(pred)
    out[ok] = y[ok] - pred[ok]
    return out, {
        "target": target,
        "controls": list(controls),
        "design": names,
        "n_fit": int(m.sum()),
        "ok": True,
    }


def add_joint_features(rows: Sequence[SignalRow], args: argparse.Namespace) -> Dict[str, Any]:
    phase = np.asarray([r.phase for r in rows], dtype=object)
    control_ref = np.isin(phase, ["correct_chain", "pre_error"])
    if args.residual_ref == "correct_chain":
        control_ref = phase == "correct_chain"
    if args.residual_ref == "pre_error":
        control_ref = phase == "pre_error"
    spread_resid, spread_info = ridge_residual(
        rows,
        "spread",
        ["logN", "pos", "n_steps"],
        control_ref,
        ridge=args.ridge,
    )
    rank_resid, rank_info = ridge_residual(
        rows,
        "res_eff_rank",
        ["logN", "kappa", "pos", "n_steps"],
        control_ref,
        ridge=args.ridge,
    )
    spread = feature_array(rows, "spread")
    rank = feature_array(rows, "res_eff_rank")
    z_spread = z_against(spread, spread[control_ref])
    z_rank = z_against(rank, rank[control_ref])
    z_spread_resid = z_against(spread_resid, spread_resid[control_ref])
    z_rank_resid = z_against(rank_resid, rank_resid[control_ref])
    joint_raw = z_spread + z_rank
    joint_strict = z_spread_resid + z_rank_resid
    joint_interaction = np.minimum(z_spread_resid, z_rank_resid)
    for i, r in enumerate(rows):
        r.features["spread_resid_lenpos"] = float(spread_resid[i])
        r.features["rank_resid_lenkappapos"] = float(rank_resid[i])
        r.features["z_spread"] = float(z_spread[i])
        r.features["z_rank"] = float(z_rank[i])
        r.features["z_spread_resid"] = float(z_spread_resid[i])
        r.features["z_rank_resid"] = float(z_rank_resid[i])
        r.features["joint_raw_zsum"] = float(joint_raw[i])
        r.features["joint_strict_zsum"] = float(joint_strict[i])
        r.features["joint_strict_min"] = float(joint_interaction[i])
    return {
        "reference": args.residual_ref,
        "n_reference": int(control_ref.sum()),
        "spread_residual": spread_info,
        "rank_residual": rank_info,
        "joint_definition": {
            "joint_raw_zsum": "z(spread) + z(res_eff_rank) against control steps",
            "joint_strict_zsum": "z(spread residualized over length/position) + z(res_eff_rank residualized over length/kappa/position)",
            "joint_strict_min": "min(z_spread_resid, z_rank_resid), high only when both are high",
        },
    }


def eval_mask(rows: Sequence[SignalRow]) -> np.ndarray:
    return np.asarray([r.y_first_error >= 0 for r in rows], dtype=bool)


def score_stats(rows: Sequence[SignalRow], score_name: str) -> Dict[str, Any]:
    y = y_array(rows)
    m = eval_mask(rows)
    s = feature_array(rows, score_name)
    mm = m & np.isfinite(s)
    yy = y[mm]
    ss = s[mm]
    err = ss[yy == 1]
    cor = ss[yy == 0]
    a = auroc(ss, yy) if len(np.unique(yy)) == 2 else float("nan")
    return {
        "score": score_name,
        "n": int(mm.sum()),
        "n_error": int(np.sum(yy == 1)),
        "n_control": int(np.sum(yy == 0)),
        "auroc_error_high": float(a),
        "best_direction": float(max(a, 1.0 - a)) if np.isfinite(a) else float("nan"),
        "error": descriptive(err),
        "control": descriptive(cor),
        "delta_error_minus_control": float(np.nanmean(err) - np.nanmean(cor)) if err.size and cor.size else float("nan"),
    }


def bootstrap_auc_increment(
    rows: Sequence[SignalRow],
    new_name: str,
    base_name: str,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    y = y_array(rows)
    m = eval_mask(rows)
    s_new = feature_array(rows, new_name)
    s_base = feature_array(rows, base_name)
    finite = m & np.isfinite(s_new) & np.isfinite(s_base)
    point = auroc(s_new[finite], y[finite]) - auroc(s_base[finite], y[finite])
    groups = group_array(rows)
    ug = np.unique(groups[finite])
    by = {g: np.where(finite & (groups == g))[0] for g in ug}
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(ug, size=ug.size, replace=True)
        idx = np.concatenate([by[g] for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auroc(s_new[idx], y[idx]) - auroc(s_base[idx], y[idx]))
    if not vals:
        return {"new": new_name, "base": base_name, "point": float(point), "lo": None, "hi": None, "sig": False}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "new": new_name,
        "base": base_name,
        "point": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "sig": bool(lo > 0.0 or hi < 0.0),
    }


def quadrant_name(high_spread: bool, high_rank: bool) -> str:
    if high_spread and high_rank:
        return "dual_high_spread_high_rank"
    if high_spread:
        return "consensus_loss_only"
    if high_rank:
        return "rank_dispersion_only"
    return "low_spread_low_rank"


def odds_stats(mask: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    y = np.asarray(y, dtype=int)
    e_in = int(np.sum(mask & (y == 1)))
    c_in = int(np.sum(mask & (y == 0)))
    e_out = int(np.sum((~mask) & (y == 1)))
    c_out = int(np.sum((~mask) & (y == 0)))
    rate = e_in / max(e_in + c_in, 1)
    out_rate = e_out / max(e_out + c_out, 1)
    odds = ((e_in + 0.5) * (c_out + 0.5)) / max((c_in + 0.5) * (e_out + 0.5), EPS)
    recall = e_in / max(int(np.sum(y == 1)), 1)
    fpr = c_in / max(int(np.sum(y == 0)), 1)
    return {
        "n": int(mask.sum()),
        "error": e_in,
        "control": c_in,
        "error_rate": float(rate),
        "outside_error_rate": float(out_rate),
        "odds_ratio": float(odds),
        "recall": float(recall),
        "control_fpr": float(fpr),
    }


def quadrant_analysis(rows: Sequence[SignalRow], args: argparse.Namespace) -> Dict[str, Any]:
    m = eval_mask(rows)
    y = y_array(rows)[m]
    spread = feature_array(rows, "z_spread_resid")[m]
    rank = feature_array(rows, "z_rank_resid")[m]
    controls = y == 0
    spread_thr = finite_quantile(spread[controls], args.quadrant_q)
    rank_thr = finite_quantile(rank[controls], args.quadrant_q)
    high_spread = spread >= spread_thr
    high_rank = rank >= rank_thr
    names = np.asarray([quadrant_name(bool(a), bool(b)) for a, b in zip(high_spread, high_rank)], dtype=object)
    out = {
        "threshold_source": f"control quantile q={args.quadrant_q:g}",
        "z_spread_resid_threshold": float(spread_thr),
        "z_rank_resid_threshold": float(rank_thr),
        "quadrants": {},
        "flags": {
            "high_spread_resid": odds_stats(high_spread, y),
            "high_rank_resid": odds_stats(high_rank, y),
            "dual_high": odds_stats(high_spread & high_rank, y),
            "either_high": odds_stats(high_spread | high_rank, y),
        },
    }
    for name in sorted(set(names.tolist())):
        out["quadrants"][name] = odds_stats(names == name, y)
    return out


def build_chain_index(rows: Sequence[SignalRow]) -> Dict[int, List[SignalRow]]:
    by: Dict[int, List[SignalRow]] = {}
    for r in rows:
        by.setdefault(r.chain_idx, []).append(r)
    for vals in by.values():
        vals.sort(key=lambda r: r.step_idx)
    return by


def summarize_values(vals: Sequence[float]) -> Dict[str, Any]:
    return descriptive([float(v) for v in vals if np.isfinite(v)])


def trajectory_profiles(rows: Sequence[SignalRow], args: argparse.Namespace) -> Dict[str, Any]:
    signals = [
        "spread",
        "res_eff_rank",
        "z_spread_resid",
        "z_rank_resid",
        "joint_strict_zsum",
    ]
    by_chain = build_chain_index(rows)
    rel_bins = list(range(-int(args.pre_window), int(args.post_window) + 1))
    rel_out: Dict[str, Dict[str, Any]] = {sig: {} for sig in signals}
    for sig in signals:
        for rel in rel_bins:
            vals = []
            for chain_rows in by_chain.values():
                if not chain_rows:
                    continue
                gold = chain_rows[0].gold_error_step
                if gold < 0:
                    continue
                for r in chain_rows:
                    if r.step_idx - gold == rel:
                        vals.append(r.features.get(sig, float("nan")))
            rel_out[sig][str(rel)] = summarize_values(vals)

    phase_out: Dict[str, Dict[str, Any]] = {sig: {} for sig in signals}
    for sig in signals:
        for phase in ("correct_chain", "pre_error", "first_error", "post_error"):
            vals = [r.features.get(sig, float("nan")) for r in rows if r.phase == phase]
            phase_out[sig][phase] = summarize_values(vals)

    pos_out: Dict[str, Dict[str, Any]] = {sig: {} for sig in signals}
    edges = np.linspace(0.0, 1.0, int(args.profile_bins) + 1)
    for sig in signals:
        for bi in range(len(edges) - 1):
            lo, hi = edges[bi], edges[bi + 1]
            vals = []
            for r in rows:
                pos = r.features.get("pos", float("nan"))
                if not np.isfinite(pos):
                    continue
                if bi == len(edges) - 2:
                    keep = lo <= pos <= hi
                else:
                    keep = lo <= pos < hi
                if keep:
                    vals.append(r.features.get(sig, float("nan")))
            pos_out[sig][f"{lo:.2f}-{hi:.2f}"] = summarize_values(vals)
    return {
        "relative_to_first_error": rel_out,
        "by_phase": phase_out,
        "by_normalized_position": pos_out,
    }


def transition_stats(rows: Sequence[SignalRow], args: argparse.Namespace) -> Dict[str, Any]:
    signals = ["spread", "res_eff_rank", "z_spread_resid", "z_rank_resid", "joint_strict_zsum"]
    by_chain = build_chain_index(rows)
    out: Dict[str, Any] = {}
    for sig in signals:
        first_jump = []
        post_jump = []
        control_jumps = []
        for chain_rows in by_chain.values():
            vals = {r.step_idx: r.features.get(sig, float("nan")) for r in chain_rows}
            if not chain_rows:
                continue
            gold = chain_rows[0].gold_error_step
            if gold >= 0:
                if (gold - 1) in vals and gold in vals:
                    first_jump.append(vals[gold] - vals[gold - 1])
                if gold in vals and (gold + 1) in vals:
                    post_jump.append(vals[gold + 1] - vals[gold])
            else:
                ordered = [r for r in chain_rows if np.isfinite(r.features.get(sig, float("nan")))]
                for a, b in zip(ordered[:-1], ordered[1:]):
                    control_jumps.append(b.features[sig] - a.features[sig])
        out[sig] = {
            "pre_to_first_error": summarize_values(first_jump),
            "first_to_post_error": summarize_values(post_jump),
            "correct_chain_adjacent": summarize_values(control_jumps),
            "pre_to_first_minus_correct_mean": safe_mean(first_jump) - safe_mean(control_jumps),
        }
    return out


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    rows, meta = load_rows(path, args)
    residual_meta = add_joint_features(rows, args)
    score_names = [
        "spread",
        "res_eff_rank",
        "spread_resid_lenpos",
        "rank_resid_lenkappapos",
        "joint_raw_zsum",
        "joint_strict_zsum",
        "joint_strict_min",
    ]
    scores = {name: score_stats(rows, name) for name in score_names}
    increments = {
        "joint_strict_vs_spread": bootstrap_auc_increment(
            rows, "joint_strict_zsum", "spread", n_boot=args.bootstrap, seed=args.seed + 1
        ),
        "joint_strict_vs_rank": bootstrap_auc_increment(
            rows, "joint_strict_zsum", "res_eff_rank", n_boot=args.bootstrap, seed=args.seed + 2
        ),
        "joint_strict_vs_rank_resid": bootstrap_auc_increment(
            rows, "joint_strict_zsum", "rank_resid_lenkappapos", n_boot=args.bootstrap, seed=args.seed + 3
        ),
        "joint_raw_vs_spread": bootstrap_auc_increment(
            rows, "joint_raw_zsum", "spread", n_boot=args.bootstrap, seed=args.seed + 4
        ),
    }
    quad = quadrant_analysis(rows, args)
    profiles = trajectory_profiles(rows, args)
    transitions = transition_stats(rows, args)
    best_score = max(scores.items(), key=lambda kv: np.nan_to_num(kv[1]["auroc_error_high"], nan=-1.0))
    return {
        "meta": {
            **meta,
            "input": os.path.abspath(path),
            "basename": os.path.basename(path),
            "joint": residual_meta,
            "evaluation": {
                "positive": "gold first-error step",
                "controls": args.control_pool,
                "post_error_rows": "excluded from first-error AUROC, included in trajectory/cascade profiles",
            },
        },
        "headline": {
            "best_score": best_score[0],
            "best_score_row": best_score[1],
            "joint_increments": increments,
            "quadrants": quad,
        },
        "scores": scores,
        "trajectory_profiles": profiles,
        "transition_stats": transitions,
        "rows_for_csv": rows,
    }


def write_csvs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str, str]:
    rows: Sequence[SignalRow] = res["rows_for_csv"]
    row_path = os.path.join(output_dir, stem + ".rows.csv")
    profile_path = os.path.join(output_dir, stem + ".profiles.csv")
    transition_path = os.path.join(output_dir, stem + ".transitions.csv")
    cols = [
        "chain_idx",
        "problem_id",
        "step_idx",
        "gold_error_step",
        "phase",
        "y_first_error",
        "n_tok",
        "pos",
        "kappa",
        "spread",
        "res_eff_rank",
        "spread_resid_lenpos",
        "rank_resid_lenkappapos",
        "z_spread_resid",
        "z_rank_resid",
        "joint_strict_zsum",
    ]
    with open(row_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            d = {
                "chain_idx": r.chain_idx,
                "problem_id": r.problem_id,
                "step_idx": r.step_idx,
                "gold_error_step": r.gold_error_step,
                "phase": r.phase,
                "y_first_error": r.y_first_error,
            }
            for c in cols:
                if c not in d:
                    d[c] = r.features.get(c)
            w.writerow(d)
    with open(profile_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["view", "signal", "bin", "n", "mean", "median", "q25", "q75"])
        for view, sigs in res["trajectory_profiles"].items():
            for sig, bins in sigs.items():
                for b, stat in bins.items():
                    w.writerow(
                        [
                            view,
                            sig,
                            b,
                            stat.get("n", 0),
                            stat.get("mean"),
                            stat.get("median"),
                            stat.get("q25"),
                            stat.get("q75"),
                        ]
                    )
    with open(transition_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal", "transition", "n", "mean", "median", "q25", "q75"])
        for sig, transitions in res["transition_stats"].items():
            for name, stat in transitions.items():
                if isinstance(stat, dict):
                    w.writerow([sig, name, stat.get("n", 0), stat.get("mean"), stat.get("median"), stat.get("q25"), stat.get("q75")])
    return row_path, profile_path, transition_path


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
        f"# Kappa-Rank Joint Trajectory Audit: `{meta['basename']}`",
        "",
        "## Headline",
        "",
        f"- Source: `{meta['source']}` layer `{meta['layer']}`.",
        f"- Rows: `{meta['trajectory_rows']}`; first-error rows `{meta['n_first_error']}`; pre-error `{meta['n_pre_error']}`; post-error `{meta['n_post_error']}`; correct-chain steps `{meta['n_correct_chain_steps']}`.",
        f"- Best first-error score: `{head['best_score']}` AUROC `{fmt(head['best_score_row']['auroc_error_high'])}`.",
        f"- Joint strict definition: `{meta['joint']['joint_definition']['joint_strict_zsum']}`.",
        "",
        "## Joint Scores",
        "",
        "| score | AUROC | best-dir | err median | ctrl median | delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(res["scores"].items(), key=lambda kv: np.nan_to_num(kv[1]["auroc_error_high"], nan=-1), reverse=True):
        lines.append(
            f"| `{name}` | {fmt(row['auroc_error_high'])} | {fmt(row['best_direction'])} | "
            f"{fmt(row['error'].get('median'))} | {fmt(row['control'].get('median'))} | "
            f"{fmt(row['delta_error_minus_control'], True)} |"
        )
    lines += [
        "",
        "## Increment Tests",
        "",
        "| comparison | point | CI | sig |",
        "|---|---:|---:|---:|",
    ]
    for name, row in head["joint_increments"].items():
        ci = "" if row.get("lo") is None else f"[{fmt(row['lo'], True)}, {fmt(row['hi'], True)}]"
        lines.append(f"| `{name}` | {fmt(row['point'], True)} | {ci} | {row.get('sig')} |")
    lines += [
        "",
        "## Quadrants",
        "",
        "| quadrant | n | error | control | error rate | odds | recall | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(head["quadrants"]["quadrants"].items(), key=lambda kv: np.nan_to_num(kv[1]["odds_ratio"], nan=-1), reverse=True):
        lines.append(
            f"| `{name}` | {row['n']} | {row['error']} | {row['control']} | {fmt(row['error_rate'])} | "
            f"{fmt(row['odds_ratio'])} | {fmt(row['recall'])} | {fmt(row['control_fpr'])} |"
        )
    lines += [
        "",
        "## Phase Profile",
        "",
        "| signal | correct | pre-error | first-error | post-error |",
        "|---|---:|---:|---:|---:|",
    ]
    phase = res["trajectory_profiles"]["by_phase"]
    for sig in ("spread", "res_eff_rank", "z_spread_resid", "z_rank_resid", "joint_strict_zsum"):
        bins = phase[sig]
        lines.append(
            f"| `{sig}` | {fmt(bins['correct_chain'].get('median'))} | {fmt(bins['pre_error'].get('median'))} | "
            f"{fmt(bins['first_error'].get('median'))} | {fmt(bins['post_error'].get('median'))} |"
        )
    lines += [
        "",
        "## First-Error Transition",
        "",
        "| signal | pre->first mean | correct adjacent mean | difference | first->post mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for sig, row in res["transition_stats"].items():
        pre = row["pre_to_first_error"].get("mean")
        ctrl = row["correct_chain_adjacent"].get("mean")
        post = row["first_to_post_error"].get("mean")
        lines.append(
            f"| `{sig}` | {fmt(pre, True)} | {fmt(ctrl, True)} | "
            f"{fmt(row.get('pre_to_first_minus_correct_mean'), True)} | {fmt(post, True)} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `spread` is the consensus-loss signal.",
        "- `rank_resid_lenkappapos` is the residual-rank signal after continuous controls for length, kappa, position, and number of steps.",
        "- `dual_high_spread_high_rank` is the most interpretable joint failure state: consensus is weak and the residual scatter is unusually high-rank.",
        "- Post-error rows are not included in first-error AUROC; they only diagnose whether the signal cascades after the first wrong step.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_outputs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str, str, str, str]:
    os.makedirs(output_dir, exist_ok=True)
    clean = dict(res)
    rows = clean.pop("rows_for_csv")
    jpath = os.path.join(output_dir, stem + ".json")
    mpath = os.path.join(output_dir, stem + ".md")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(finite_json(clean), f, ensure_ascii=False, indent=2)
    write_markdown(mpath, finite_json(clean))
    tmp = dict(clean)
    tmp["rows_for_csv"] = rows
    row_path, profile_path, transition_path = write_csvs(tmp, output_dir, stem)
    return jpath, mpath, row_path, profile_path, transition_path


def print_result(res: Mapping[str, Any]) -> None:
    meta = res["meta"]
    head = res["headline"]
    print(f"\n===== kappa-rank joint trajectory | {meta['basename']} =====")
    print(
        f"rows {meta['trajectory_rows']} | first-error {meta['n_first_error']} | pre {meta['n_pre_error']} | "
        f"post {meta['n_post_error']} | correct-step {meta['n_correct_chain_steps']} | source {meta['source']} L{meta['layer']}"
    )
    print(f"best score {head['best_score']} AUROC {head['best_score_row']['auroc_error_high']:.3f}")
    print("\nScores:")
    for name, row in sorted(res["scores"].items(), key=lambda kv: np.nan_to_num(kv[1]["auroc_error_high"], nan=-1), reverse=True):
        print(
            f"  {name:28s} AUROC {row['auroc_error_high']:.3f} "
            f"delta {row['delta_error_minus_control']:+.3f}"
        )
    print("\nJoint increments:")
    for name, row in head["joint_increments"].items():
        ci = "" if row.get("lo") is None else f" [{row['lo']:+.3f},{row['hi']:+.3f}]"
        print(f"  {name:28s} {row['point']:+.3f}{ci}")
    print("\nQuadrants:")
    for name, row in sorted(head["quadrants"]["quadrants"].items(), key=lambda kv: np.nan_to_num(kv[1]["odds_ratio"], nan=-1), reverse=True):
        print(
            f"  {name:32s} n={row['n']:4d} err_rate={row['error_rate']:.3f} "
            f"OR={row['odds_ratio']:.2f} recall={row['recall']:.3f} FPR={row['control_fpr']:.3f}"
        )
    print("\nFirst-error transition means:")
    for sig, row in res["transition_stats"].items():
        print(
            f"  {sig:24s} pre->first {row['pre_to_first_error'].get('mean', float('nan')):+.3f} "
            f"ctrl {row['correct_chain_adjacent'].get('mean', float('nan')):+.3f} "
            f"first->post {row['first_to_post_error'].get('mean', float('nan')):+.3f}"
        )


def make_selftest(path: str, *, seed: int = 0) -> None:
    ddm.make_selftest(path, seed=seed, n_chains=75, dim=48)


def assert_selftest(res: Mapping[str, Any]) -> None:
    scores = res["scores"]
    if scores["joint_strict_zsum"]["auroc_error_high"] < 0.85:
        raise AssertionError("selftest failed: joint strict score did not recover injected failures")
    quad = res["headline"]["quadrants"]["quadrants"].get("dual_high_spread_high_rank", {})
    if quad.get("odds_ratio", 0.0) < 5.0:
        raise AssertionError("selftest failed: dual-high quadrant was not enriched")
    trans = res["transition_stats"]["joint_strict_zsum"]
    if trans["pre_to_first_error"].get("mean", 0.0) <= trans["correct_chain_adjacent"].get("mean", 0.0):
        raise AssertionError("selftest failed: first-error transition did not exceed correct transitions")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="")
    ap.add_argument("--policy", default="gold_error_step", choices=["gold_error_step", "answer", "strict", "answer_format_ok"])
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--nearest_layer", action="store_true")
    ap.add_argument("--hidden_dir", default="")
    ap.add_argument("--no_mmap", action="store_true")
    ap.add_argument("--kappa_beta", type=float, default=1.0)
    ap.add_argument("--min_tokens", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--max_pair_tokens", type=int, default=96)
    ap.add_argument("--length_bins", type=int, default=4)
    ap.add_argument("--kappa_bins", type=int, default=4)
    ap.add_argument("--pos_bins", type=int, default=3)
    ap.add_argument("--control_pool", default="pre_and_correct", choices=["pre_and_correct", "correct_chain", "pre_error"])
    ap.add_argument("--residual_ref", default="pre_and_correct", choices=["pre_and_correct", "correct_chain", "pre_error"])
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--quadrant_q", type=float, default=0.75)
    ap.add_argument("--pre_window", type=int, default=4)
    ap.add_argument("--post_window", type=int, default=4)
    ap.add_argument("--profile_bins", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs/kappa_rank_joint_trajectory")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "kappa_rank_joint_selftest.npz")
            make_selftest(path, seed=args.seed)
            args.input = path
            args.layer = 16
            args.no_progress = True
            args.bootstrap = min(args.bootstrap, 40)
            res = run(path, args)
            assert_selftest(res)
            print_result(res)
            print("selftest passed")
        return
    if not args.input:
        raise SystemExit("--input is required unless --selftest is passed")
    res = run(args.input, args)
    stem = os.path.splitext(os.path.basename(args.input))[0] + "_kappa_rank_joint_trajectory"
    paths = write_outputs(res, args.output_dir, stem)
    print_result(res)
    print("\nsaved:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
