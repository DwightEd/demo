#!/usr/bin/env python3
"""Trajectory-relative audit for online reasoning-break detection.

This script checks the gap that step-level first-error AUROC can hide:

1. Compare masked first-error evaluation with full-trajectory evaluation.
2. Report false alarms and false localizations on fully correct chains.
3. Test a trajectory-relative detector that separates slow effort from fast
   breaks, calibrated only on correct training chains.

It is intentionally light-weight: no learned deep sequence model, no labels from
post-error segments, and no intervention hooks.  The goal is to decide whether
the next detector should be a prefix-state / changepoint module rather than
another flattened step classifier.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.model_selection import GroupKFold
except ImportError as exc:  # pragma: no cover
    raise SystemExit("trajectory_relative_audit.py needs scikit-learn") from exc

from audit_utils import Chain, auroc, finite_json, load_chains, make_selftest_npz, safe_mean, safe_std


EPS = 1e-9
BASE_DETECTORS = ["raw_spread", "anchor_uncertainty", "traj_break", "traj_break_cusum", "confident_break"]


def arr(c: Chain, name: str) -> np.ndarray:
    return np.asarray(c.features.get(name, np.full(c.n_steps, np.nan)), float)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(x, float), 0.0)


def robust_scale(x: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)) * 1.4826)
    if not np.isfinite(mad) or mad < EPS:
        mad = float(np.std(v) + EPS)
    return med, max(mad, EPS)


def causal_ema_prev(x: np.ndarray, *, alpha: float) -> np.ndarray:
    """EMA of past values only; current value never enters its own baseline."""
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    state = np.nan
    for t, val in enumerate(v):
        out[t] = state
        if not np.isfinite(val):
            continue
        if np.isfinite(state):
            state = alpha * state + (1.0 - alpha) * float(val)
        else:
            state = float(val)
    return out


def leaky_cusum(z: np.ndarray, *, lam: float, kref: float) -> np.ndarray:
    out = np.zeros(len(z), float)
    c = 0.0
    for t, val in enumerate(np.asarray(z, float)):
        x = 0.0 if not np.isfinite(val) else float(val)
        c = max(0.0, lam * c + x - kref)
        out[t] = c
    return out


def add_basic_features(chains: Sequence[Chain]) -> None:
    for c in chains:
        if "resultant" in c.features:
            spread = 1.0 - arr(c, "resultant")
        elif "coherence" in c.features:
            spread = 1.0 - arr(c, "coherence")
        else:
            spread = np.full(c.n_steps, np.nan)
        c.features["spread"] = spread
        c.features["anchor_loss"] = 1.0 - arr(c, "q_align") if "q_align" in c.features else np.full(c.n_steps, np.nan)
        if "U_D_mean" in c.features:
            c.features["uncertainty"] = arr(c, "U_D_mean")
        elif "U_C_mean" in c.features:
            c.features["uncertainty"] = arr(c, "U_C_mean")
        else:
            c.features["uncertainty"] = np.full(c.n_steps, np.nan)
        c.features["raw_spread"] = spread
        c.features["anchor_uncertainty"] = spread + c.features["anchor_loss"] + c.features["uncertainty"]


def add_trajectory_innovations(chains: Sequence[Chain], *, alpha: float) -> None:
    for c in chains:
        for name in ("spread", "anchor_loss", "uncertainty", "step_direction_jump"):
            v = arr(c, name)
            slow = causal_ema_prev(v, alpha=alpha)
            c.features[f"slow_{name}"] = slow
            c.features[f"innov_{name}"] = v - slow


def collect_correct_null(chains: Sequence[Chain], idxs: Sequence[int], names: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    params = {}
    for nm in names:
        vals: List[float] = []
        for i in idxs:
            c = chains[i]
            if not c.correct:
                continue
            vals.extend(arr(c, nm).tolist())
        params[nm] = robust_scale(vals)
    return params


def z_feature(c: Chain, name: str, params: Dict[str, Tuple[float, float]]) -> np.ndarray:
    med, scale = params.get(name, (0.0, 1.0))
    return (arr(c, name) - med) / max(scale, EPS)


def score_chain_relative(
    c: Chain,
    params: Dict[str, Tuple[float, float]],
    *,
    lam: float,
    kref: float,
    risk_center: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    zs = z_feature(c, "innov_spread", params)
    za = z_feature(c, "innov_anchor_loss", params)
    zu = z_feature(c, "innov_uncertainty", params)
    zj = z_feature(c, "innov_step_direction_jump", params)

    anchor_break = relu(zs) + relu(za)
    confident_break = relu(zs) + relu(za) + relu(zs - zu)
    effort_break = relu(zs) + relu(za) + 0.5 * relu(zu)
    jump_break = 0.35 * relu(zj)
    traj_break = effort_break + 0.5 * confident_break + jump_break
    traj_break[~np.isfinite(traj_break)] = np.nan
    confident_break[~np.isfinite(confident_break)] = np.nan

    if risk_center is None:
        mu, sd = 0.0, 1.0
    else:
        mu, sd = risk_center
    traj_break_cusum = leaky_cusum((traj_break - mu) / max(sd, EPS), lam=lam, kref=kref)
    return {
        "traj_break": traj_break,
        "traj_break_cusum": traj_break_cusum,
        "confident_break": confident_break,
    }


def assign_score(c: Chain, name: str, score: np.ndarray) -> None:
    c.features[name] = np.asarray(score, float)


def finite_max(x: np.ndarray) -> float:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    return float(np.max(v)) if len(v) else float("nan")


def finite_argmax(x: np.ndarray) -> int:
    v = np.asarray(x, float)
    if not np.isfinite(v).any():
        return -1
    return int(np.nanargmax(v))


def split_indices(chains: Sequence[Chain], folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(chains))
    groups = np.asarray([c.group for c in chains])
    n_splits = min(int(folds), len(np.unique(groups)))
    if n_splits < 2:
        return []
    return list(GroupKFold(n_splits=n_splits).split(idx[:, None], idx, groups))


def fit_crossfit_relative_scores(
    chains: Sequence[Chain],
    *,
    folds: int,
    alpha: float,
    lam: float,
    kref: float,
) -> Dict[str, object]:
    add_trajectory_innovations(chains, alpha=alpha)
    splits = split_indices(chains, folds)
    if not splits:
        return {"folds": 0, "detectors": BASE_DETECTORS}

    fold_of: Dict[int, int] = {}
    calibration: Dict[int, Dict[str, np.ndarray]] = {}
    innov_names = ["innov_spread", "innov_anchor_loss", "innov_uncertainty", "innov_step_direction_jump"]

    for fold, (tr, te) in enumerate(splits):
        params = collect_correct_null(chains, tr, innov_names)

        # Center CUSUM by the correct-chain null risk under this fold.
        null_risk = []
        for i in tr:
            if not chains[i].correct:
                continue
            sc = score_chain_relative(chains[i], params, lam=lam, kref=kref, risk_center=(0.0, 1.0))
            null_risk.extend(sc["traj_break"].tolist())
        mu, sd = robust_scale(null_risk)

        cal_by_detector: Dict[str, List[float]] = {nm: [] for nm in BASE_DETECTORS}
        for i in tr:
            if not chains[i].correct:
                continue
            rel = score_chain_relative(chains[i], params, lam=lam, kref=kref, risk_center=(mu, sd))
            for nm, score in rel.items():
                cal_by_detector[nm].append(finite_max(score))
            for nm in ("raw_spread", "anchor_uncertainty"):
                cal_by_detector[nm].append(finite_max(arr(chains[i], nm)))

        calibration[fold] = {nm: np.asarray(vals, float) for nm, vals in cal_by_detector.items()}
        for i in te:
            rel = score_chain_relative(chains[i], params, lam=lam, kref=kref, risk_center=(mu, sd))
            for nm, score in rel.items():
                assign_score(chains[i], nm, score)
            fold_of[i] = fold

    return {"folds": len(splits), "fold_of": fold_of, "calibration": calibration, "detectors": BASE_DETECTORS}


def flatten_gold(chains: Sequence[Chain], score_name: str, *, mask_post_error: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, score, groups = [], [], []
    for c in chains:
        s = arr(c, score_name)
        for t in range(c.n_steps):
            if (not c.correct) and c.gold >= 0 and mask_post_error and t > c.gold:
                continue
            if not np.isfinite(s[t]):
                continue
            y.append(int((not c.correct) and t == c.gold))
            score.append(float(s[t]))
            groups.append(int(c.group))
    return np.asarray(score, float), np.asarray(y, int), np.asarray(groups)


def step_tables(chains: Sequence[Chain], detectors: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    for nm in detectors:
        sm, ym, _ = flatten_gold(chains, nm, mask_post_error=True)
        sf, yf, _ = flatten_gold(chains, nm, mask_post_error=False)
        rows.append(
            {
                "detector": nm,
                "masked_gold_auroc": auroc(sm, ym),
                "full_gold_auroc": auroc(sf, yf),
                "masked_n": int(len(ym)),
                "full_n": int(len(yf)),
                "err": int(np.sum(ym)),
            }
        )
    rows.sort(key=lambda r: np.nan_to_num(r["masked_gold_auroc"], nan=-1.0), reverse=True)
    return rows


def localization_table(chains: Sequence[Chain], detectors: Sequence[str], *, mask_post_error: bool) -> List[Dict[str, object]]:
    rows = []
    for nm in detectors:
        top1, exp, pct = [], [], []
        for c in chains:
            if c.correct or c.gold < 0 or c.gold >= c.n_steps:
                continue
            s = arr(c, nm)
            m = np.isfinite(s)
            if mask_post_error:
                m[np.arange(c.n_steps) > c.gold] = False
            if not m[c.gold] or m.sum() < 2:
                continue
            cand = s[m]
            better = int((cand > s[c.gold]).sum())
            top1.append(float(better == 0))
            exp.append(1.0 / m.sum())
            pct.append(better / max(1, m.sum() - 1))
        rows.append(
            {
                "detector": nm,
                "top1": safe_mean(top1),
                "expected_top1": safe_mean(exp),
                "gain": safe_mean(top1) - safe_mean(exp),
                "mean_rank_pct": safe_mean(pct),
                "n": int(len(top1)),
                "mask_post_error": bool(mask_post_error),
            }
        )
    rows.sort(key=lambda r: np.nan_to_num(r["top1"], nan=-1.0), reverse=True)
    return rows


def profile_table(chains: Sequence[Chain], detectors: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    for nm in detectors:
        correct_vals, pre_vals, gold_vals, post_vals = [], [], [], []
        for c in chains:
            s = arr(c, nm)
            if c.correct:
                correct_vals.extend(s[np.isfinite(s)].tolist())
            elif 0 <= c.gold < c.n_steps:
                pre_vals.extend(s[np.arange(c.n_steps) < c.gold].tolist())
                gold_vals.append(float(s[c.gold]))
                post_vals.extend(s[np.arange(c.n_steps) > c.gold].tolist())
        rows.append(
            {
                "detector": nm,
                "correct_mean": safe_mean(correct_vals),
                "pre_error_mean": safe_mean(pre_vals),
                "gold_mean": safe_mean(gold_vals),
                "post_error_mean": safe_mean(post_vals),
                "post_minus_gold": safe_mean(post_vals) - safe_mean(gold_vals),
            }
        )
    return rows


def correct_false_localization(chains: Sequence[Chain], detectors: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    correct = [c for c in chains if c.correct]
    for nm in detectors:
        max_scores, arg_pos, arg_step = [], [], []
        for c in correct:
            s = arr(c, nm)
            t = finite_argmax(s)
            if t < 0:
                continue
            max_scores.append(float(s[t]))
            arg_step.append(float(t))
            arg_pos.append(float(t / max(1, c.n_steps - 1)))
        rows.append(
            {
                "detector": nm,
                "correct_chains": int(len(arg_pos)),
                "max_score_mean": safe_mean(max_scores),
                "max_score_std": safe_std(max_scores),
                "argmax_step_mean": safe_mean(arg_step),
                "argmax_pos_mean": safe_mean(arg_pos),
                "argmax_pos_std": safe_std(arg_pos),
            }
        )
    return rows


def online_alarm_table(chains: Sequence[Chain], fit: Dict[str, object], *, eps_list: Sequence[float]) -> List[Dict[str, object]]:
    fold_of: Dict[int, int] = fit.get("fold_of", {})
    calibration: Dict[int, Dict[str, np.ndarray]] = fit.get("calibration", {})
    rows = []
    for nm in fit.get("detectors", BASE_DETECTORS):
        for eps in eps_list:
            thresholds = {}
            for fold, cal in calibration.items():
                vals = np.asarray(cal.get(nm, []), float)
                vals = vals[np.isfinite(vals)]
                thresholds[fold] = float(np.quantile(vals, 1.0 - eps)) if len(vals) else float("inf")
            n_correct = false_alarm = n_error = caught = early = late = 0
            delays, false_pos = [], []
            for i, c in enumerate(chains):
                if i not in fold_of:
                    continue
                score = arr(c, nm)
                thr = thresholds.get(fold_of[i], float("inf"))
                hit = np.where(np.isfinite(score) & (score > thr))[0]
                alarm = int(hit[0]) if len(hit) else -1
                if c.correct:
                    n_correct += 1
                    false_alarm += int(alarm >= 0)
                    if alarm >= 0:
                        false_pos.append(alarm / max(1, c.n_steps - 1))
                elif 0 <= c.gold < c.n_steps:
                    n_error += 1
                    if alarm >= 0:
                        caught += 1
                        d = alarm - c.gold
                        delays.append(d)
                        early += int(d < 0)
                        late += int(d > 0)
            rows.append(
                {
                    "detector": nm,
                    "eps": float(eps),
                    "fpr": false_alarm / max(1, n_correct),
                    "recall": caught / max(1, n_error),
                    "median_delay": float(np.median(delays)) if delays else float("nan"),
                    "early_rate": early / max(1, caught),
                    "late_rate": late / max(1, caught),
                    "false_alarm_pos_mean": safe_mean(false_pos),
                    "caught": int(caught),
                    "n_error": int(n_error),
                    "false_alarm": int(false_alarm),
                    "n_correct": int(n_correct),
                }
            )
    rows.sort(key=lambda r: (r["eps"], -np.nan_to_num(r["recall"], nan=-1.0), np.nan_to_num(r["fpr"], nan=9.0)))
    return rows


def run(npz: str, args: argparse.Namespace) -> Dict[str, object]:
    chains, meta = load_chains(npz, layer=args.layer, max_chains=args.max_chains)
    add_basic_features(chains)
    fit = fit_crossfit_relative_scores(
        chains,
        folds=args.folds,
        alpha=args.slow_alpha,
        lam=args.lam,
        kref=args.kref,
    )
    detectors = list(fit.get("detectors", BASE_DETECTORS))
    return {
        "meta": {**meta, "audit": "trajectory_relative", "layer": args.layer},
        "n_chains": int(len(chains)),
        "n_error_chains": int(sum(not c.correct for c in chains)),
        "n_correct_chains": int(sum(c.correct for c in chains)),
        "detectors": detectors,
        "fit": {"folds": fit.get("folds", 0), "eps_list": list(args.eps_list)},
        "step_masked_vs_full": step_tables(chains, detectors),
        "localization_masked": localization_table(chains, detectors, mask_post_error=True),
        "localization_full": localization_table(chains, detectors, mask_post_error=False),
        "correct_false_localization": correct_false_localization(chains, detectors),
        "online_alarms": online_alarm_table(chains, fit, eps_list=args.eps_list),
        "trajectory_profiles": profile_table(chains, detectors),
        "notes": {
            "masked_gold": "correct all steps + error pre-first-error + gold step; post-error skipped",
            "full_gold": "correct all steps + all error-chain steps; only gold step is positive, post-error is evaluated as non-gold",
            "online": "thresholds are calibrated from correct training-chain max scores, then first crossing is evaluated on held-out full trajectories",
            "trajectory_relative": "fast innovation = current score - causal EMA of previous scores; null scaling uses correct training chains only",
        },
    }


def print_table(title: str, rows: Sequence[Dict[str, object]], *, limit: int = 12) -> None:
    print(f"\n{title}:")
    for r in rows[:limit]:
        if "masked_gold_auroc" in r:
            print(
                f"  {r['detector']:22s} masked {float(r['masked_gold_auroc']):.3f} "
                f"full {float(r['full_gold_auroc']):.3f} n {r['masked_n']}->{r['full_n']}"
            )
        elif "top1" in r:
            print(
                f"  {r['detector']:22s} top1 {float(r['top1']):.3f} "
                f"exp {float(r['expected_top1']):.3f} gain {float(r['gain']):+.3f} n={r['n']}"
            )
        elif "fpr" in r:
            print(
                f"  {r['detector']:22s} eps {float(r['eps']):.2f} FPR {float(r['fpr']):.3f} "
                f"recall {float(r['recall']):.3f} delay {float(r['median_delay']):+.1f} "
                f"early {float(r['early_rate']):.3f} false_pos {float(r['false_alarm_pos_mean']):.3f}"
            )
        elif "argmax_pos_mean" in r:
            print(
                f"  {r['detector']:22s} correct max {float(r['max_score_mean']):+.3f} "
                f"argpos {float(r['argmax_pos_mean']):.3f} +/- {float(r['argmax_pos_std']):.3f}"
            )
        elif "post_minus_gold" in r:
            print(
                f"  {r['detector']:22s} correct {float(r['correct_mean']):+.3f} "
                f"pre {float(r['pre_error_mean']):+.3f} gold {float(r['gold_mean']):+.3f} "
                f"post {float(r['post_error_mean']):+.3f}"
            )


def print_report(res: Dict[str, object]) -> None:
    meta = res.get("meta", {})
    print(
        f"\n===== trajectory-relative audit | {os.path.basename(str(meta.get('npz', 'selftest')))} | "
        f"L{meta.get('layer', 'na')} ====="
    )
    print(
        f"chains {res['n_chains']} | error {res['n_error_chains']} | "
        f"correct {res['n_correct_chains']} | folds {res['fit']['folds']}"
    )
    print_table("Masked vs full first-error step AUROC", res["step_masked_vs_full"])
    print_table("Within-error-chain localization masked", res["localization_masked"])
    print_table("Within-error-chain localization full", res["localization_full"])
    print_table("Correct-chain false localization", res["correct_false_localization"])
    print_table("Online full-trajectory alarms", res["online_alarms"], limit=20)
    print_table("Trajectory profiles", res["trajectory_profiles"])


def write_outputs(res: Dict[str, object], output_dir: str, stem: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    clean = finite_json(res)
    with open(os.path.join(output_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, stem + ".md"), "w", encoding="utf-8") as f:
        f.write(f"# Trajectory-Relative Audit: {stem}\n\n")
        f.write("## Result Analysis\n\n")
        f.write("- This audit separates masked first-error localization from full-trajectory online monitoring.\n")
        f.write("- `traj_break` and `traj_break_cusum` use only causal within-chain history plus correct-chain null calibration.\n")
        f.write("- Correct-chain false localization is reported explicitly, so a high top1 on error chains is not enough.\n\n")
        f.write("## Follow-Up Research Direction\n\n")
        f.write("- If full localization drops but online recall remains useful, report the method as a monitor rather than a first-error localizer.\n")
        f.write("- If trajectory-relative scores reduce correct-chain false alarms, upgrade them into a prefix-state detector.\n")
        f.write("- If raw spread matches trajectory-relative scores, the current features are too weak and we need richer anchor/key/attention signals.\n\n")
        f.write("## Optimization Suggestions\n\n")
        f.write("- Replace scalar EMA with a low-dimensional metastable state model only after this audit shows nontrivial gains.\n")
        f.write("- Add a detector/actuator interface so the online alarm rows can drive `repath`, `compress`, or key re-anchoring.\n")
        f.write("- Always report masked, full, and correct-chain false-localization metrics together.\n")


def run_selftest(args: argparse.Namespace) -> Dict[str, object]:
    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "trajectory_relative_selftest.npz")
        make_selftest_npz(npz, n_chains=90, layer=args.layer, seed=args.seed)
        return run(npz, args)


def assert_selftest(res: Dict[str, object]) -> None:
    rows = {r["detector"]: r for r in res["step_masked_vs_full"]}
    if rows.get("traj_break", {}).get("masked_gold_auroc", 0.0) < 0.75:
        raise SystemExit("selftest failed: traj_break did not detect synthetic gold steps")
    alarms = [r for r in res["online_alarms"] if r["detector"] == "traj_break_cusum" and abs(float(r["eps"]) - 0.20) < 1e-6]
    if alarms and float(alarms[0].get("recall", 0.0)) < 0.6:
        raise SystemExit("selftest failed: trajectory CUSUM online recall is too low")


def parse_eps(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Trajectory-relative full-chain reasoning-break audit")
    ap.add_argument("npz", nargs="?", default=None)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--slow_alpha", type=float, default=0.65)
    ap.add_argument("--lam", type=float, default=0.80)
    ap.add_argument("--kref", type=float, default=0.25)
    ap.add_argument("--eps_list", default="0.05,0.10,0.20")
    ap.add_argument("--output_dir", default="outputs/trajectory_relative")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    args.eps_list = parse_eps(args.eps_list)

    if args.selftest:
        res = run_selftest(args)
        stem = "trajectory_relative_selftest"
        assert_selftest(res)
    else:
        npz = args.npz
        if npz is None:
            if not args.dataset:
                raise SystemExit("pass npz path, --dataset, or --selftest")
            npz = os.path.join(args.data_dir, "features", f"full_{args.dataset}.npz")
        res = run(npz, args)
        stem = f"trajectory_relative_{args.dataset or os.path.splitext(os.path.basename(npz))[0]}_L{args.layer}"

    print_report(res)
    write_outputs(res, args.output_dir, stem)
    print(f"\nwrote {os.path.join(args.output_dir, stem + '.json')} and .md")


if __name__ == "__main__":
    main()
