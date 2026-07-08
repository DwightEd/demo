#!/usr/bin/env python3
"""Audit phase-transition patterns in step-level reasoning signals.

Previous response-level scripts compressed a chain into rough summaries such as
mean, min, slope, or CUSUM.  This script keeps the step order and asks the more
diagnostic question:

    Is the prefix stable, and does the first wrong step create a local break?

It works on precomputed step-level signals from `stepcloud`, so it is cheap.
Heavy hidden-state geometry should still be computed in the upstream GPU-enabled
audits; this script analyzes the resulting scalar trajectories.
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


EPS = 1e-12


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


@dataclass
class EventRow:
    chain_idx: int
    problem_id: int
    step_idx: int
    gold_error_step: int
    phase: str
    y_first_error: int
    y_chain_error: int
    features: Dict[str, float]


def finite_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return finite_json(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def descriptive(vals: Iterable[float]) -> Dict[str, Any]:
    x = np.asarray([float(v) for v in vals], dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
            "q95": float("nan"),
        }
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
        "q95": float(np.quantile(x, 0.95)),
    }


def finite_quantile(vals: Iterable[float], q: float) -> float:
    x = np.asarray([float(v) for v in vals], dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")


def safe_mean(vals: Iterable[float]) -> float:
    x = np.asarray([float(v) for v in vals], dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def auroc(score: Iterable[float], y: Iterable[int]) -> float:
    s = np.asarray(list(score), dtype=np.float64)
    yy = np.asarray(list(y), dtype=int)
    m = np.isfinite(s)
    s, yy = s[m], yy[m]
    p = int(np.sum(yy == 1))
    n = int(np.sum(yy == 0))
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
    return float((np.sum(ranks[yy == 1]) - p * (p + 1) / 2.0) / (p * n))


def bootstrap_auc_increment(
    chain_rows: Sequence[Mapping[str, Any]],
    score_a: str,
    score_b: str,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows = list(chain_rows)
    valid = [r for r in rows if np.isfinite(r.get(score_a, np.nan)) and np.isfinite(r.get(score_b, np.nan))]
    if not valid:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    base_a = auroc([r[score_a] for r in valid], [r["y_chain_error"] for r in valid])
    base_b = auroc([r[score_b] for r in valid], [r["y_chain_error"] for r in valid])
    deltas: List[float] = []
    n = len(valid)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        aa = [valid[int(i)][score_a] for i in idx]
        bb = [valid[int(i)][score_b] for i in idx]
        yy = [valid[int(i)]["y_chain_error"] for i in idx]
        da = auroc(aa, yy)
        db = auroc(bb, yy)
        if np.isfinite(da) and np.isfinite(db):
            deltas.append(float(da - db))
    return {
        "score_a": score_a,
        "score_b": score_b,
        "auc_a": float(base_a),
        "auc_b": float(base_b),
        "delta": float(base_a - base_b) if np.isfinite(base_a) and np.isfinite(base_b) else float("nan"),
        "ci_low": finite_quantile(deltas, 0.025),
        "ci_high": finite_quantile(deltas, 0.975),
        "n_boot": int(len(deltas)),
    }


def robust_center_scale(x: np.ndarray, min_scale: float) -> Tuple[float, float]:
    z = np.asarray(x, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0, 1.0
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med)))
    if mad > EPS:
        return med, max(1.4826 * mad, float(min_scale))
    sd = float(np.std(z))
    return med, max(sd, float(min_scale))


def finite_slope(x: np.ndarray) -> float:
    y = np.asarray(x, dtype=np.float64)
    m = np.isfinite(y)
    y = y[m]
    if y.size < 3:
        return float("nan")
    t = np.arange(y.size, dtype=np.float64)
    t -= float(np.mean(t))
    yc = y - float(np.mean(y))
    den = float(np.sum(t * t))
    return float(np.sum(t * yc) / den) if den > EPS else float("nan")


def phase_for(gold: int, step_idx: int) -> str:
    if gold < 0:
        return "correct_chain"
    if step_idx < gold:
        return "pre_error"
    if step_idx == gold:
        return "first_error"
    return "post_error"


def select_layer_index(data: Mapping[str, Any], n_layers: int, args: argparse.Namespace) -> Tuple[int, int]:
    layers = None
    if "layers_used" in data:
        layers = [int(x) for x in data["layers_used"]]
    elif "cloud_store_layers" in data:
        layers = [int(x) for x in data["cloud_store_layers"]]
    if layers and len(layers) == n_layers:
        arr = np.asarray(layers, dtype=int)
        if args.nearest_layer:
            idx = int(np.argmin(np.abs(arr - int(args.layer))))
            return idx, int(arr[idx])
        if int(args.layer) not in set(layers):
            raise SystemExit(f"layer {args.layer} not present; available {layers}. Use --nearest_layer to choose closest.")
        idx = layers.index(int(args.layer))
        return idx, int(args.layer)
    if n_layers == 1:
        return 0, int(args.layer)
    if 0 <= int(args.layer) < n_layers and not args.layer_is_id:
        return int(args.layer), int(args.layer)
    raise SystemExit("cannot map requested layer to stepcloud layer axis")


def get_signal_sequences(data: Mapping[str, Any], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if "stepcloud" not in data or "cloud_feature_names" not in data:
        raise SystemExit("trajectory_phase_transition_audit requires stepcloud and cloud_feature_names")
    if "gold_error_step" not in data:
        raise SystemExit("trajectory_phase_transition_audit requires gold_error_step")
    names = [str(x) for x in data["cloud_feature_names"]]
    if "resultant" not in names:
        raise SystemExit(f"cloud_feature_names has no resultant: {names}")
    result_idx = names.index("resultant")
    stepcloud = data["stepcloud"]
    first = np.asarray(stepcloud[0])
    if first.ndim < 2:
        raise SystemExit("stepcloud rows must have shape (steps, layers, features)")
    layer_idx, layer_used = select_layer_index(data, int(first.shape[1]), args)
    gold = np.asarray(data["gold_error_step"], dtype=int)
    problem_ids = np.asarray(data["problem_ids"], dtype=int) if "problem_ids" in data else np.arange(len(gold), dtype=int)
    max_n = len(gold) if args.max_chains <= 0 else min(len(gold), int(args.max_chains))
    seqs: List[Dict[str, Any]] = []
    iterator = range(max_n)
    if not args.no_progress:
        iterator = tqdm(iterator, desc="phase-transition sequences", unit="chain")
    for idx in iterator:
        sc = np.asarray(stepcloud[idx], dtype=np.float64)
        if sc.ndim != 3 or layer_idx >= sc.shape[1] or result_idx >= sc.shape[2]:
            continue
        kappa = sc[:, layer_idx, result_idx]
        spread = 1.0 - kappa
        if not np.isfinite(spread).any():
            continue
        signals: Dict[str, np.ndarray] = {"spread": spread}
        if "tok_U_D" in data and "step_token_ranges" in data:
            ent = np.asarray(data["tok_U_D"][idx], dtype=np.float64)
            rng = np.asarray(data["step_token_ranges"][idx], dtype=int)
            if rng.ndim == 2 and rng.shape[0] >= spread.size:
                a0 = int(rng[0, 0])
                step_ent = []
                for j in range(spread.size):
                    lo = max(0, int(rng[j, 0]) - a0)
                    hi = min(ent.size, int(rng[j, 1]) - a0 + 1)
                    val = float(np.nanmean(ent[lo:hi])) if hi > lo else float("nan")
                    step_ent.append(val)
                signals["entropy_mean"] = np.asarray(step_ent, dtype=np.float64)
        seqs.append(
            {
                "chain_idx": int(idx),
                "problem_id": int(problem_ids[idx]),
                "gold_error_step": int(gold[idx]),
                "n_steps": int(spread.size),
                "signals": signals,
            }
        )
    meta = {
        "layer_requested": int(args.layer),
        "layer_used": int(layer_used),
        "n_chains_loaded": int(len(seqs)),
        "n_error_chains": int(sum(1 for s in seqs if int(s["gold_error_step"]) >= 0)),
        "available_signals": sorted({k for s in seqs for k in s["signals"].keys()}),
    }
    return seqs, meta


def prefix_event_features(seq: np.ndarray, step_idx: int, args: argparse.Namespace) -> Dict[str, float]:
    x = np.asarray(seq, dtype=np.float64)
    cur = float(x[step_idx]) if step_idx < x.size else float("nan")
    prev = float(x[step_idx - 1]) if step_idx > 0 and step_idx - 1 < x.size else float("nan")
    prefix = x[:step_idx]
    prefix = prefix[np.isfinite(prefix)]
    min_scale = max(float(args.scale_floor), float(args.std_floor_frac) * float(np.nanstd(x)))
    center, scale = robust_center_scale(prefix, min_scale=min_scale)
    mean = float(np.mean(prefix)) if prefix.size else float("nan")
    std = float(np.std(prefix, ddof=1)) if prefix.size > 1 else 0.0
    slope = finite_slope(prefix)
    jump = cur - prev if np.isfinite(cur) and np.isfinite(prev) else float("nan")
    level = cur - center if np.isfinite(cur) else float("nan")
    level_z = level / max(scale, EPS) if np.isfinite(level) else float("nan")
    jump_z = jump / max(scale, EPS) if np.isfinite(jump) else float("nan")
    prefix_cv = std / max(abs(mean), float(args.scale_floor)) if np.isfinite(mean) else float("nan")
    break_z = max(0.0, level_z if np.isfinite(level_z) else 0.0) + max(0.0, jump_z if np.isfinite(jump_z) else 0.0)
    shock_z = max(level_z if np.isfinite(level_z) else float("-inf"), jump_z if np.isfinite(jump_z) else float("-inf"))
    return {
        "current": cur,
        "prev": prev,
        "prefix_center": center,
        "prefix_mean": mean,
        "prefix_std": std,
        "prefix_scale": scale,
        "prefix_slope": slope,
        "prefix_cv": prefix_cv,
        "level": level,
        "jump": jump,
        "level_z": level_z,
        "jump_z": jump_z,
        "break_z": break_z,
        "shock_z": shock_z,
    }


def build_event_rows(seqs: Sequence[Mapping[str, Any]], signal: str, args: argparse.Namespace) -> List[EventRow]:
    rows: List[EventRow] = []
    for item in seqs:
        signals = item["signals"]
        if signal not in signals:
            continue
        seq = np.asarray(signals[signal], dtype=np.float64)
        gold = int(item["gold_error_step"])
        y_chain = int(gold >= 0)
        for step_idx in range(int(args.min_prefix), seq.size):
            if not np.isfinite(seq[step_idx]):
                continue
            phase = phase_for(gold, step_idx)
            y_first = int(phase == "first_error")
            feats = prefix_event_features(seq, step_idx, args)
            feats.update(
                {
                    "signal_value": feats["current"],
                    "n_steps": float(seq.size),
                    "pos": float(step_idx / max(seq.size - 1, 1)),
                }
            )
            rows.append(
                EventRow(
                    chain_idx=int(item["chain_idx"]),
                    problem_id=int(item["problem_id"]),
                    step_idx=int(step_idx),
                    gold_error_step=gold,
                    phase=phase,
                    y_first_error=y_first,
                    y_chain_error=y_chain,
                    features=feats,
                )
            )
    return rows


def event_value(row: EventRow, name: str) -> float:
    return float(row.features.get(name, float("nan")))


def event_mask_for_controls(rows: Sequence[EventRow]) -> np.ndarray:
    return np.asarray([r.phase in {"correct_chain", "pre_error"} for r in rows], dtype=bool)


def event_detection(rows: Sequence[EventRow]) -> Dict[str, Any]:
    yy = np.asarray([r.y_first_error for r in rows], dtype=int)
    controls = event_mask_for_controls(rows)
    eval_mask = (yy == 1) | controls
    out: Dict[str, Any] = {}
    for name in ["signal_value", "level_z", "jump_z", "break_z", "shock_z", "prefix_slope", "prefix_std"]:
        score = np.asarray([event_value(r, name) for r in rows], dtype=np.float64)
        out[name] = {
            "first_error_auroc": auroc(score[eval_mask], yy[eval_mask]),
            "first_error_mean": safe_mean(score[yy == 1]),
            "control_mean": safe_mean(score[controls]),
            "first_error_minus_control": safe_mean(score[yy == 1]) - safe_mean(score[controls]),
        }
    return out


def build_chain_summaries(rows: Sequence[EventRow], signal: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    by_chain: Dict[int, List[EventRow]] = {}
    for row in rows:
        by_chain.setdefault(row.chain_idx, []).append(row)
    summaries: List[Dict[str, Any]] = []
    for chain_idx, vals in by_chain.items():
        vals = sorted(vals, key=lambda r: r.step_idx)
        if not vals:
            continue
        out: Dict[str, Any] = {
            "chain_idx": int(chain_idx),
            "problem_id": int(vals[0].problem_id),
            "gold_error_step": int(vals[0].gold_error_step),
            "y_chain_error": int(vals[0].y_chain_error),
            "signal": signal,
            "n_events": int(len(vals)),
        }
        for name in ["signal_value", "level_z", "jump_z", "break_z", "shock_z"]:
            scores = np.asarray([event_value(v, name) for v in vals], dtype=np.float64)
            finite = np.isfinite(scores)
            out[f"max_{name}"] = float(np.max(scores[finite])) if np.any(finite) else float("nan")
            out[f"mean_{name}"] = float(np.mean(scores[finite])) if np.any(finite) else float("nan")
            if np.any(finite):
                mx = int(np.nanargmax(scores))
                out[f"argmax_{name}"] = int(vals[mx].step_idx)
            else:
                out[f"argmax_{name}"] = -1
        gold = int(vals[0].gold_error_step)
        if gold >= 0:
            gold_rows = [v for v in vals if v.step_idx == gold]
            for name in ["level_z", "jump_z", "break_z", "shock_z"]:
                if gold_rows:
                    gs = event_value(gold_rows[0], name)
                    scores = np.asarray([event_value(v, name) for v in vals], dtype=np.float64)
                    finite = np.isfinite(scores)
                    if np.isfinite(gs) and np.any(finite):
                        rank = 1 + int(np.sum(scores[finite] > gs))
                        out[f"gold_{name}"] = float(gs)
                        out[f"gold_{name}_rank"] = int(rank)
                        out[f"gold_{name}_top1"] = int(rank == 1)
                        out[f"gold_{name}_percentile"] = float(np.mean(scores[finite] <= gs))
                    else:
                        out[f"gold_{name}"] = float("nan")
                        out[f"gold_{name}_rank"] = -1
                        out[f"gold_{name}_top1"] = 0
                        out[f"gold_{name}_percentile"] = float("nan")
                else:
                    out[f"gold_{name}"] = float("nan")
                    out[f"gold_{name}_rank"] = -1
                    out[f"gold_{name}_top1"] = 0
                    out[f"gold_{name}_percentile"] = float("nan")
        summaries.append(out)
    return summaries


def response_detection(chain_rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    y = [int(r["y_chain_error"]) for r in chain_rows]
    score_names = [
        "max_signal_value",
        "mean_signal_value",
        "max_level_z",
        "max_jump_z",
        "max_break_z",
        "max_shock_z",
    ]
    for name in score_names:
        out[name] = {"chain_auroc": auroc([float(r.get(name, float("nan"))) for r in chain_rows], y)}
    out["increments"] = {
        "max_break_z_vs_max_signal": bootstrap_auc_increment(
            chain_rows,
            "max_break_z",
            "max_signal_value",
            n_boot=args.bootstrap,
            seed=args.seed + 11,
        ),
        "max_shock_z_vs_max_signal": bootstrap_auc_increment(
            chain_rows,
            "max_shock_z",
            "max_signal_value",
            n_boot=args.bootstrap,
            seed=args.seed + 12,
        ),
        "max_jump_z_vs_max_signal": bootstrap_auc_increment(
            chain_rows,
            "max_jump_z",
            "max_signal_value",
            n_boot=args.bootstrap,
            seed=args.seed + 13,
        ),
    }
    return out


def aligned_profiles(rows: Sequence[EventRow], args: argparse.Namespace) -> Dict[str, Any]:
    rels = list(range(-int(args.pre_window), int(args.post_window) + 1))
    out: Dict[str, Dict[str, Any]] = {}
    for name in ["signal_value", "level_z", "jump_z", "break_z", "shock_z", "prefix_std", "prefix_slope"]:
        out[name] = {}
        for rel in rels:
            vals = [
                event_value(r, name)
                for r in rows
                if r.gold_error_step >= 0 and r.step_idx - r.gold_error_step == rel
            ]
            out[name][str(rel)] = descriptive(vals)
    return out


def classify_first_error_modes(rows: Sequence[EventRow], args: argparse.Namespace) -> Dict[str, Any]:
    first_rows = [r for r in rows if r.phase == "first_error"]
    control_rows = [r for r in rows if r.phase in {"correct_chain", "pre_error"}]
    level_thr = finite_quantile([event_value(r, "level_z") for r in control_rows], args.event_q)
    jump_thr = finite_quantile([event_value(r, "jump_z") for r in control_rows], args.event_q)
    break_thr = finite_quantile([event_value(r, "break_z") for r in control_rows], args.event_q)
    pre_std_thr = finite_quantile([event_value(r, "prefix_std") for r in control_rows], args.stable_q)
    pre_mean_thr = finite_quantile([event_value(r, "prefix_mean") for r in control_rows], args.event_q)
    pre_slope_thr = finite_quantile([event_value(r, "prefix_slope") for r in control_rows], args.event_q)
    modes: Dict[str, List[EventRow]] = {
        "stable_prefix_break": [],
        "gradual_drift": [],
        "persistently_unstable_prefix": [],
        "isolated_jump": [],
        "no_clear_geometry_event": [],
    }
    for r in first_rows:
        level = event_value(r, "level_z")
        jump = event_value(r, "jump_z")
        br = event_value(r, "break_z")
        pstd = event_value(r, "prefix_std")
        pmean = event_value(r, "prefix_mean")
        pslope = event_value(r, "prefix_slope")
        if np.isfinite(pstd) and np.isfinite(br) and pstd <= pre_std_thr and br >= break_thr:
            modes["stable_prefix_break"].append(r)
        elif np.isfinite(pslope) and np.isfinite(jump) and pslope >= pre_slope_thr and jump < jump_thr:
            modes["gradual_drift"].append(r)
        elif np.isfinite(pmean) and pmean >= pre_mean_thr:
            modes["persistently_unstable_prefix"].append(r)
        elif (np.isfinite(level) and level >= level_thr) or (np.isfinite(jump) and jump >= jump_thr):
            modes["isolated_jump"].append(r)
        else:
            modes["no_clear_geometry_event"].append(r)
    total = max(len(first_rows), 1)
    return {
        "thresholds": {
            "event_q": float(args.event_q),
            "stable_q": float(args.stable_q),
            "level_z": float(level_thr),
            "jump_z": float(jump_thr),
            "break_z": float(break_thr),
            "prefix_std": float(pre_std_thr),
            "prefix_mean": float(pre_mean_thr),
            "prefix_slope": float(pre_slope_thr),
        },
        "modes": {
            name: {
                "n": int(len(vals)),
                "fraction_of_first_errors": float(len(vals) / total),
                "mean_level_z": safe_mean(event_value(v, "level_z") for v in vals),
                "mean_jump_z": safe_mean(event_value(v, "jump_z") for v in vals),
                "mean_break_z": safe_mean(event_value(v, "break_z") for v in vals),
            }
            for name, vals in modes.items()
        },
    }


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    seqs, meta = get_signal_sequences(data, args)
    signal_names = [s.strip() for s in args.signals.split(",") if s.strip()]
    signal_names = [s for s in signal_names if s in meta["available_signals"]]
    if not signal_names:
        raise SystemExit(f"no requested signals are available; requested {args.signals}, available {meta['available_signals']}")
    signal_results: Dict[str, Any] = {}
    all_event_rows: Dict[str, List[EventRow]] = {}
    all_chain_rows: Dict[str, List[Dict[str, Any]]] = {}
    for sig in signal_names:
        rows = build_event_rows(seqs, sig, args)
        chains = build_chain_summaries(rows, sig, args)
        signal_results[sig] = {
            "event_detection": event_detection(rows),
            "response_detection": response_detection(chains, args),
            "aligned_profiles": aligned_profiles(rows, args),
            "first_error_modes": classify_first_error_modes(rows, args),
            "gold_event_ranks": {
                name: {
                    "top1_rate": safe_mean(int(r.get(f"gold_{name}_top1", 0)) for r in chains if int(r.get("y_chain_error", 0)) == 1),
                    "mean_percentile": safe_mean(float(r.get(f"gold_{name}_percentile", float("nan"))) for r in chains if int(r.get("y_chain_error", 0)) == 1),
                    "mean_rank": safe_mean(float(r.get(f"gold_{name}_rank", float("nan"))) for r in chains if int(r.get("y_chain_error", 0)) == 1 and int(r.get(f"gold_{name}_rank", -1)) > 0),
                }
                for name in ["level_z", "jump_z", "break_z", "shock_z"]
            },
            "n_event_rows": int(len(rows)),
            "n_chain_rows": int(len(chains)),
        }
        all_event_rows[sig] = rows
        all_chain_rows[sig] = chains
    return {
        "meta": {
            **meta,
            "input": os.path.abspath(path),
            "basename": os.path.basename(path),
            "min_prefix": int(args.min_prefix),
            "signals": signal_names,
            "read": {
                "level_z": "current risk relative to the previous prefix distribution",
                "jump_z": "current minus previous step, normalized by prefix scale",
                "break_z": "positive level_z plus positive jump_z",
                "shock_z": "max(level_z, jump_z)",
            },
        },
        "signals": signal_results,
        "event_rows": all_event_rows,
        "chain_rows": all_chain_rows,
    }


def write_csvs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str]:
    event_path = os.path.join(output_dir, stem + ".events.csv")
    chain_path = os.path.join(output_dir, stem + ".chains.csv")
    event_cols = [
        "signal",
        "chain_idx",
        "problem_id",
        "step_idx",
        "gold_error_step",
        "phase",
        "y_first_error",
        "y_chain_error",
        "signal_value",
        "prefix_mean",
        "prefix_std",
        "prefix_slope",
        "level_z",
        "jump_z",
        "break_z",
        "shock_z",
        "pos",
        "n_steps",
    ]
    with open(event_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=event_cols)
        w.writeheader()
        for sig, rows in res["event_rows"].items():
            for r in rows:
                row = {
                    "signal": sig,
                    "chain_idx": r.chain_idx,
                    "problem_id": r.problem_id,
                    "step_idx": r.step_idx,
                    "gold_error_step": r.gold_error_step,
                    "phase": r.phase,
                    "y_first_error": r.y_first_error,
                    "y_chain_error": r.y_chain_error,
                }
                for c in event_cols:
                    if c not in row and c != "signal":
                        row[c] = r.features.get(c)
                w.writerow(row)
    chain_cols = [
        "signal",
        "chain_idx",
        "problem_id",
        "gold_error_step",
        "y_chain_error",
        "n_events",
        "max_signal_value",
        "mean_signal_value",
        "max_level_z",
        "max_jump_z",
        "max_break_z",
        "max_shock_z",
        "argmax_break_z",
        "gold_break_z",
        "gold_break_z_rank",
        "gold_break_z_top1",
        "gold_break_z_percentile",
    ]
    with open(chain_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=chain_cols)
        w.writeheader()
        for sig, rows in res["chain_rows"].items():
            for row in rows:
                w.writerow({c: row.get(c) for c in chain_cols})
    return event_path, chain_path


def markdown_report(res: Mapping[str, Any]) -> str:
    meta = res["meta"]
    lines = [
        f"# Trajectory Phase Transition Audit: `{meta['basename']}`",
        "",
        "## Setup",
        "",
        f"- Chains loaded: {meta['n_chains_loaded']} | error chains: {meta['n_error_chains']}",
        f"- Layer used: {meta['layer_used']} | min prefix: {meta['min_prefix']}",
        f"- Signals: {', '.join(meta['signals'])}",
        "",
        "## Main Read",
        "",
        "Old response aggregation loses the local event structure.  This audit tests whether the first wrong step is a prefix-relative break.",
        "",
    ]
    for sig, block in res["signals"].items():
        lines.extend([f"## Signal: `{sig}`", ""])
        lines.append("### Event Detection")
        lines.append("")
        lines.append("| event score | first-error AUROC | first mean | control mean | diff |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, row in block["event_detection"].items():
            lines.append(
                f"| `{name}` | {row['first_error_auroc']:.3f} | {row['first_error_mean']:.3f} | "
                f"{row['control_mean']:.3f} | {row['first_error_minus_control']:+.3f} |"
            )
        lines.extend(["", "### Response Detection", ""])
        lines.append("| chain score | AUROC |")
        lines.append("|---|---:|")
        for name, row in block["response_detection"].items():
            if name == "increments":
                continue
            lines.append(f"| `{name}` | {row['chain_auroc']:.3f} |")
        lines.extend(["", "### Response Increments", ""])
        lines.append("| comparison | delta | 95% CI |")
        lines.append("|---|---:|---|")
        for name, row in block["response_detection"].get("increments", {}).items():
            lines.append(f"| `{name}` | {row['delta']:+.3f} | [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] |")
        lines.extend(["", "### Gold First-Error Rank In Chain", ""])
        lines.append("| score | top1 rate | mean percentile | mean rank |")
        lines.append("|---|---:|---:|---:|")
        for name, row in block["gold_event_ranks"].items():
            lines.append(
                f"| `{name}` | {row['top1_rate']:.3f} | {row['mean_percentile']:.3f} | {row['mean_rank']:.3f} |"
            )
        lines.extend(["", "### First-Error Modes", ""])
        lines.append("| mode | n | frac | mean level_z | mean jump_z | mean break_z |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name, row in block["first_error_modes"]["modes"].items():
            lines.append(
                f"| `{name}` | {row['n']} | {row['fraction_of_first_errors']:.3f} | "
                f"{row['mean_level_z']:.3f} | {row['mean_jump_z']:.3f} | {row['mean_break_z']:.3f} |"
            )
        lines.extend(["", "### Aligned Profile Around First Error", ""])
        lines.append("| rel step | signal mean | level_z mean | jump_z mean | break_z mean |")
        lines.append("|---:|---:|---:|---:|---:|")
        profiles = block["aligned_profiles"]
        rels = sorted(profiles["signal_value"].keys(), key=lambda x: int(x))
        for rel in rels:
            lines.append(
                f"| {rel} | {profiles['signal_value'][rel]['mean']:.3f} | "
                f"{profiles['level_z'][rel]['mean']:.3f} | {profiles['jump_z'][rel]['mean']:.3f} | "
                f"{profiles['break_z'][rel]['mean']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str, str, str]:
    os.makedirs(output_dir, exist_ok=True)
    clean = finite_json({k: v for k, v in res.items() if k not in {"event_rows", "chain_rows"}})
    jpath = os.path.join(output_dir, stem + ".json")
    mpath = os.path.join(output_dir, stem + ".md")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=output_dir) as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
        tmpj = f.name
    os.replace(tmpj, jpath)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=output_dir) as f:
        f.write(markdown_report(res))
        tmpm = f.name
    os.replace(tmpm, mpath)
    epath, cpath = write_csvs(res, output_dir, stem)
    return jpath, mpath, epath, cpath


def make_selftest(path: str, *, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = 36
    t = 6
    stepcloud = np.zeros((n, t, 1, 1), dtype=np.float32)
    gold = np.full(n, -1, dtype=np.int32)
    problem_ids = np.arange(n, dtype=np.int32)
    ranges = np.zeros((n, t, 2), dtype=np.int32)
    ent = np.zeros((n, t * 4), dtype=np.float32)
    for i in range(n):
        err = i < n // 2
        k = 3 if err else -1
        gold[i] = k
        spread = 0.18 + 0.015 * rng.normal(size=t)
        if err:
            spread[k] = 0.55 + 0.03 * rng.normal()
            spread[k + 1 :] = 0.38 + 0.03 * rng.normal(size=t - k - 1)
        else:
            spread += 0.015 * rng.normal(size=t)
        stepcloud[i, :, 0, 0] = 1.0 - np.clip(spread, 0.02, 0.95)
        for j in range(t):
            lo = j * 4
            hi = lo + 3
            ranges[i, j] = [lo, hi]
            ent[i, lo : hi + 1] = 0.2 + 0.02 * rng.normal(size=4)
    np.savez(
        path,
        stepcloud=stepcloud,
        cloud_feature_names=np.asarray(["resultant"], dtype=object),
        layers_used=np.asarray([16], dtype=np.int32),
        gold_error_step=gold,
        problem_ids=problem_ids,
        step_token_ranges=ranges,
        tok_U_D=ent,
    )


def assert_selftest(res: Mapping[str, Any]) -> None:
    spread = res["signals"]["spread"]
    if spread["event_detection"]["break_z"]["first_error_auroc"] < 0.95:
        raise AssertionError("selftest failed: break_z should localize synthetic first-error steps")
    if spread["response_detection"]["max_break_z"]["chain_auroc"] < 0.95:
        raise AssertionError("selftest failed: max_break_z should detect synthetic wrong chains")
    if spread["gold_event_ranks"]["break_z"]["top1_rate"] < 0.8:
        raise AssertionError("selftest failed: first-error break should be top-ranked in most wrong chains")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="?", help="input .npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--nearest_layer", action="store_true")
    ap.add_argument("--layer_is_id", action="store_true", help="treat --layer as a raw stepcloud axis when no layer list exists")
    ap.add_argument("--signals", default="spread,entropy_mean", help="comma-separated risk-high signals to audit")
    ap.add_argument("--min_prefix", type=int, default=2)
    ap.add_argument("--pre_window", type=int, default=3)
    ap.add_argument("--post_window", type=int, default=3)
    ap.add_argument("--event_q", type=float, default=0.90)
    ap.add_argument("--stable_q", type=float, default=0.50)
    ap.add_argument("--scale_floor", type=float, default=1e-3)
    ap.add_argument("--std_floor_frac", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs/trajectory_phase_transition")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.selftest:
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, "trajectory_phase_transition_selftest.npz")
        make_selftest(path, seed=args.seed)
        args.input = path
        args.layer = 16
        args.nearest_layer = True
        args.bootstrap = min(args.bootstrap, 50)
    if not args.input:
        raise SystemExit("pass input .npz or --selftest")
    res = run(args.input, args)
    if args.selftest:
        assert_selftest(res)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    paths = write_outputs(res, args.output_dir, "trajectory_phase_transition_" + stem)
    print(f"===== trajectory phase transition | {os.path.basename(args.input)} =====")
    meta = res["meta"]
    print(
        f"chains {meta['n_chains_loaded']} | err {meta['n_error_chains']} | "
        f"L{meta['layer_used']} | signals {','.join(meta['signals'])}"
    )
    for sig, block in res["signals"].items():
        print(f"\nSignal {sig}:")
        for name in ["signal_value", "level_z", "jump_z", "break_z", "shock_z"]:
            row = block["event_detection"][name]
            print(
                f"  event {name:14s} first-AUROC {row['first_error_auroc']:.3f} "
                f"diff {row['first_error_minus_control']:+.3f}"
            )
        print("  response:")
        for name in ["max_signal_value", "max_level_z", "max_jump_z", "max_break_z", "max_shock_z"]:
            row = block["response_detection"][name]
            print(f"    {name:18s} AUROC {row['chain_auroc']:.3f}")
        ranks = block["gold_event_ranks"]["break_z"]
        print(
            f"  gold break_z rank: top1 {ranks['top1_rate']:.3f} "
            f"mean-percentile {ranks['mean_percentile']:.3f} mean-rank {ranks['mean_rank']:.2f}"
        )
        print("  modes:")
        for mode, row in block["first_error_modes"]["modes"].items():
            print(f"    {mode:30s} n={row['n']:4d} frac={row['fraction_of_first_errors']:.3f}")
    print("\noutputs:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
