#!/usr/bin/env python3
"""Step-native audit for the core spectral-geometry hypotheses.

This script is deliberately narrower than the broad mechanism audit. It targets
the exact spectral anchors:

  H1: reasoning-state alpha compression.
  H2: base-vs-instruct spectral reversal.
  H3: token/layer spectral cascade and step-boundary punctuation.

The current ProcessBench full_*.npz data can test step-native alpha dynamics,
cross-layer cascade decay, and first-error boundary punctuation. It cannot, by
itself, test reasoning-vs-factual recall or base-vs-instruct reversal because
those require matched task/model hidden states. The JSON output records that
gap explicitly instead of smuggling it into the error-localization data.
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

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hidden_io import _fn  # noqa: E402
from nts.signals.alpha import spectral_alpha  # noqa: E402


EPS = 1e-9


@dataclass
class ChainSpec:
    idx: int
    chain_id: str
    problem_id: int
    gold: int
    correct: bool
    ranges: np.ndarray
    hidden_path: str


@dataclass
class ChainAlpha:
    idx: int
    problem_id: int
    gold: int
    correct: bool
    layers: List[int]
    alpha: np.ndarray  # (T, L)
    n_tok: np.ndarray  # (T,)
    metrics: Dict[str, np.ndarray]


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


def safe_mean(xs) -> float:
    a = np.asarray(xs, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def safe_std(xs) -> float:
    a = np.asarray(xs, float)
    a = a[np.isfinite(a)]
    return float(a.std()) if len(a) else float("nan")


def corr(x, y) -> float:
    a = np.asarray(x, float)
    b = np.asarray(y, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) <= 0 or np.std(b[m]) <= 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


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


def nanmean_axis(A: np.ndarray, axis: int) -> np.ndarray:
    X = np.asarray(A, float)
    m = np.isfinite(X)
    count = m.sum(axis=axis)
    total = np.where(m, X, 0.0).sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def nanstd_axis(A: np.ndarray, axis: int) -> np.ndarray:
    X = np.asarray(A, float)
    mean = np.expand_dims(nanmean_axis(X, axis), axis)
    m = np.isfinite(X)
    count = m.sum(axis=axis)
    var_total = np.where(m, (X - mean) ** 2, 0.0).sum(axis=axis)
    return np.sqrt(np.divide(var_total, count, out=np.full_like(var_total, np.nan, dtype=float), where=count > 0))


def _unit_rows(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, float)
    return H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), EPS)


def kappa(H: np.ndarray) -> float:
    if len(H) < 2:
        return float("nan")
    return float(np.linalg.norm(_unit_rows(H).mean(axis=0)))


def layer_columns(hidden_layers: Sequence[int], requested: Optional[Sequence[int]]) -> Tuple[List[int], List[int]]:
    have = [int(x) for x in hidden_layers]
    if not requested:
        return have, list(range(len(have)))
    layers: List[int] = []
    cols: List[int] = []
    for L in requested:
        if int(L) in have:
            layers.append(int(L))
            cols.append(have.index(int(L)))
        else:
            j = int(np.argmin([abs(int(L) - h) for h in have]))
            layers.append(have[j])
            cols.append(j)
    # Deduplicate while keeping order.
    seen = set()
    out_l, out_c = [], []
    for L, c in zip(layers, cols):
        if L not in seen:
            seen.add(L)
            out_l.append(L)
            out_c.append(c)
    return out_l, out_c


def relative_ranges(ranges: np.ndarray) -> np.ndarray:
    rr = np.asarray(ranges, int)
    if rr.ndim != 2 or rr.shape[1] != 2 or len(rr) == 0:
        return np.zeros((0, 2), int)
    a0 = int(rr[0, 0])
    return np.column_stack([np.maximum(0, rr[:, 0] - a0), np.maximum(0, rr[:, 1] - a0 + 1)])


def load_specs(npz_path: str, hidden_dir: str, *, max_chains: int = 0) -> Tuple[List[ChainSpec], Dict[str, object]]:
    z = np.load(npz_path, allow_pickle=True)
    ids = z["ids"] if "ids" in z.files else np.arange(len(z["gold_error_step"]))
    ges = z["gold_error_step"].astype(int)
    pids = z["problem_ids"].astype(int) if "problem_ids" in z.files else np.arange(len(ges))
    ranges = z["step_token_ranges"]
    N = len(ges) if not max_chains else min(max_chains, len(ges))
    specs: List[ChainSpec] = []
    missing = 0
    for i in range(N):
        cid = str(ids[i])
        hp = os.path.join(hidden_dir, _fn(cid))
        if not os.path.exists(hp):
            missing += 1
            continue
        specs.append(
            ChainSpec(
                idx=i,
                chain_id=cid,
                problem_id=int(pids[i]),
                gold=int(ges[i]),
                correct=bool(ges[i] < 0),
                ranges=np.asarray(ranges[i], int),
                hidden_path=hp,
            )
        )
    meta = {
        "npz": npz_path,
        "hidden_dir": hidden_dir,
        "n_seen": int(N),
        "n_missing_hidden": int(missing),
        "hidden_layers": [int(x) for x in z["hidden_layers"]] if "hidden_layers" in z.files else [],
        "can_test_reasoning_vs_fact_recall": False,
        "can_test_base_vs_instruct_reversal": False,
        "missing_for_h1_full_test": "matched factual-recall hidden states",
        "missing_for_h2_reversal_test": "matched base and instruction-tuned hidden states over the same reasoning/factual tasks",
    }
    return specs, meta


def compute_chain_alpha(spec: ChainSpec, layers: Sequence[int], cols: Sequence[int], *, min_tokens: int) -> Optional[ChainAlpha]:
    Hall = np.load(spec.hidden_path, mmap_mode="r")
    rel = relative_ranges(spec.ranges)
    if len(rel) == 0:
        return None
    T, L = len(rel), len(cols)
    A = np.full((T, L), np.nan)
    N = np.full(T, np.nan)
    K = np.full((T, L), np.nan)
    for t, (lo, hi) in enumerate(rel):
        lo, hi = int(lo), int(hi)
        N[t] = max(0, hi - lo)
        if hi <= lo:
            continue
        for j, col in enumerate(cols):
            H = np.asarray(Hall[lo:hi, col, :], dtype=np.float32)
            if len(H) >= min_tokens:
                A[t, j] = spectral_alpha(H)
            K[t, j] = kappa(H)

    alpha_mean = nanmean_axis(A, axis=1)
    alpha_std_layer = nanstd_axis(A, axis=1)
    kappa_mean = nanmean_axis(K, axis=1)
    dA = np.full_like(A, np.nan)
    if T >= 2:
        dA[1:] = A[1:] - A[:-1]
    d_mean = nanmean_axis(dA, axis=1)
    phase_jump = np.sqrt(nanmean_axis(dA * dA, axis=1))
    phase_desync = nanstd_axis(dA, axis=1)
    # User-facing convention: their anchor reads lower alpha as stronger compression.
    compression_score = -alpha_mean
    metrics = {
        "alpha_mean": alpha_mean,
        "alpha_std_layer": alpha_std_layer,
        "compression_score_low_alpha": compression_score,
        "d_alpha_mean": d_mean,
        "phase_jump_l2": phase_jump,
        "phase_desync_layer_std": phase_desync,
        "kappa_mean": kappa_mean,
        "n_tok": N,
    }
    return ChainAlpha(
        idx=spec.idx,
        problem_id=spec.problem_id,
        gold=spec.gold,
        correct=spec.correct,
        layers=[int(x) for x in layers],
        alpha=A,
        n_tok=N,
        metrics=metrics,
    )


def flatten_metric(chains: Sequence[ChainAlpha], metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s, y, g, nt = [], [], [], []
    for c in chains:
        v = c.metrics[metric]
        for t in range(len(v)):
            if c.correct:
                yy = 0
            elif t < c.gold:
                yy = 0
            elif t == c.gold:
                yy = 1
            else:
                continue
            s.append(v[t])
            y.append(yy)
            g.append(c.problem_id)
            nt.append(c.n_tok[t])
    return np.asarray(s, float), np.asarray(y, int), np.asarray(g), np.asarray(nt, float)


def detection_table(chains: Sequence[ChainAlpha]) -> List[Dict[str, object]]:
    rows = []
    for metric in (
        "alpha_mean",
        "compression_score_low_alpha",
        "d_alpha_mean",
        "phase_jump_l2",
        "phase_desync_layer_std",
        "alpha_std_layer",
        "kappa_mean",
    ):
        s, y, _, nt = flatten_metric(chains, metric)
        m = np.isfinite(s)
        if m.sum() < 30 or len(np.unique(y[m])) < 2:
            continue
        raw = auroc(s[m], y[m])
        rows.append(
            {
                "metric": metric,
                "auroc_bestdir_gold_error": bdir(raw),
                "raw_auroc_high_is_error": raw,
                "mean_non_error": safe_mean(s[(y == 0) & m]),
                "mean_gold_error": safe_mean(s[(y == 1) & m]),
                "n": int(m.sum()),
            }
        )
    rows.sort(key=lambda r: np.nan_to_num(r["auroc_bestdir_gold_error"], nan=-1), reverse=True)
    return rows


def event_study(chains: Sequence[ChainAlpha], metrics: Sequence[str], *, window: int) -> Dict[str, object]:
    out: Dict[str, object] = {}
    err = [c for c in chains if not c.correct and c.gold >= 0]
    for metric in metrics:
        rows = []
        for d in range(-window, window + 1):
            vals = []
            for c in err:
                t = c.gold + d
                if 0 <= t < len(c.metrics[metric]):
                    vals.append(c.metrics[metric][t])
            rows.append({"delta": d, "mean": safe_mean(vals), "std": safe_std(vals), "n": int(np.isfinite(vals).sum())})
        pre = [r["mean"] for r in rows if r["delta"] < 0]
        at0 = next((r["mean"] for r in rows if r["delta"] == 0), float("nan"))
        out[metric] = {
            "trajectory": rows,
            "at_error_minus_pre_mean": float(at0 - safe_mean(pre)) if np.isfinite(at0) else float("nan"),
        }
    return out


def within_chain_rank(chains: Sequence[ChainAlpha], metric: str, sign: float) -> Dict[str, object]:
    top1, exp1, pct = [], [], []
    for c in chains:
        if c.correct or c.gold < 0 or c.gold >= len(c.metrics[metric]):
            continue
        s = sign * np.asarray(c.metrics[metric], float)
        m = np.isfinite(s)
        m[np.arange(len(s)) > c.gold] = False
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


def localization_table(chains: Sequence[ChainAlpha]) -> List[Dict[str, object]]:
    det = {r["metric"]: r for r in detection_table(chains)}
    rows = []
    for metric, row in det.items():
        sign = 1.0 if row["raw_auroc_high_is_error"] >= 0.5 else -1.0
        loc = within_chain_rank(chains, metric, sign)
        if loc["n"] > 0:
            rows.append({"metric": metric, **loc})
    rows.sort(key=lambda r: np.nan_to_num(r["top1"], nan=-1) - np.nan_to_num(r["expected_top1"], nan=0), reverse=True)
    return rows


def cascade_profile(chains: Sequence[ChainAlpha]) -> Dict[str, object]:
    if not chains:
        return {}
    layers = chains[0].layers
    L = len(layers)
    if L < 2:
        return {"error": "need at least two layers"}
    pair_rows = []
    for i in range(L):
        for j in range(i + 1, L):
            xi, xj = [], []
            for c in chains:
                if len(c.alpha) < 2:
                    continue
                dA = c.alpha[1:] - c.alpha[:-1]
                xi.extend(dA[:, i])
                xj.extend(dA[:, j])
            rho = corr(xi, xj)
            pair_rows.append({"layer_i": layers[i], "layer_j": layers[j], "distance": abs(layers[j] - layers[i]), "rho": rho})
    by_dist: Dict[int, List[float]] = {}
    for r in pair_rows:
        by_dist.setdefault(int(r["distance"]), []).append(r["rho"])
    dist_rows = []
    for d in sorted(by_dist):
        vals = np.asarray(by_dist[d], float)
        vals = vals[np.isfinite(vals)]
        dist_rows.append(
            {
                "distance": int(d),
                "mean_rho": safe_mean(vals),
                "mean_abs_rho": safe_mean(np.abs(vals)),
                "n_pairs": int(len(vals)),
            }
        )
    ds = np.array([r["distance"] for r in dist_rows], float)
    rs = np.array([r["mean_abs_rho"] for r in dist_rows], float)
    m = np.isfinite(ds) & np.isfinite(rs) & (rs > 0)
    if m.sum() >= 2:
        slope, intercept = np.polyfit(ds[m], np.log(rs[m]), 1)
        exp_decay = {
            "log_abs_rho_slope_per_layer": float(slope),
            "decay_lambda_if_negative": float(-slope) if slope < 0 else float("nan"),
            "intercept": float(intercept),
        }
    else:
        exp_decay = {"log_abs_rho_slope_per_layer": float("nan"), "decay_lambda_if_negative": float("nan"), "intercept": float("nan")}
    return {"pairs": pair_rows, "rho_by_distance": dist_rows, "exp_decay_fit_abs_rho": exp_decay}


def run(npz: str, hidden_dir: str, args: argparse.Namespace) -> Dict[str, object]:
    specs, meta = load_specs(npz, hidden_dir, max_chains=args.max_chains)
    if not meta["hidden_layers"]:
        raise SystemExit(f"{npz}: missing hidden_layers")
    layers, cols = layer_columns(meta["hidden_layers"], args.layers)
    chains: List[ChainAlpha] = []
    for k, spec in enumerate(specs):
        ca = compute_chain_alpha(spec, layers, cols, min_tokens=args.min_tokens)
        if ca is not None:
            chains.append(ca)
        if args.progress and (k + 1) % args.progress == 0:
            print(f"  processed {k + 1}/{len(specs)} chains")
    meta["layers_used"] = layers
    metrics = ["alpha_mean", "compression_score_low_alpha", "d_alpha_mean", "phase_jump_l2", "phase_desync_layer_std"]
    return {
        "meta": meta,
        "hypotheses": {
            "H1_alpha_compression": {
                "status": "partial",
                "what_this_run_tests": "step-native alpha dynamics inside reasoning chains and around first-error boundaries",
                "not_tested": "reasoning-vs-factual recall alpha shift; requires matched factual recall hidden states",
                "orientation_note": "existing spectral_alpha returns a raw exponent; compression_score_low_alpha = -alpha_mean follows the anchor convention lower alpha = stronger compression",
            },
            "H2_instruction_reversal": {
                "status": "not_testable_from_this_npz",
                "required_data": "matched base and instruction-tuned hidden states for reasoning and factual-recall prompts",
            },
            "H3_cascade_and_punctuation": {
                "status": "tested",
                "what_this_run_tests": "cross-layer correlations of step-to-step alpha changes, layer-distance decay, and gold-error boundary localization",
            },
        },
        "n_chains": len(chains),
        "n_error_chains": int(sum(not c.correct for c in chains)),
        "detection": detection_table(chains),
        "localization": localization_table(chains),
        "event_study": event_study(chains, metrics, window=args.event_window),
        "cascade": cascade_profile(chains),
    }


def print_result(res: Dict[str, object]) -> None:
    meta = res["meta"]
    print(f"\n===== spectral hypothesis audit | {os.path.basename(meta['npz'])} =====")
    print(f"chains {res['n_chains']} | error chains {res['n_error_chains']} | layers {meta['layers_used']}")
    print("H1 alpha compression: partial; needs factual recall contrast for full test")
    print("H2 instruction reversal: not testable from this npz")
    print("H3 cascade/punctuation: tested below")

    print("\nStep/gold-error spectral signals:")
    for r in res["detection"]:
        print(
            f"  {r['metric']:28s} AUROC {r['auroc_bestdir_gold_error']:.3f} "
            f"nonerr {r['mean_non_error']:+.4f} err {r['mean_gold_error']:+.4f}"
        )

    print("\nWithin-chain boundary punctuation:")
    for r in res["localization"]:
        gain = r["top1"] - r["expected_top1"]
        print(f"  {r['metric']:28s} top1 {r['top1']:.3f} exp {r['expected_top1']:.3f} gain {gain:+.3f} n={r['n']}")

    print("\nCascade layer-distance profile:")
    for r in res["cascade"].get("rho_by_distance", []):
        print(f"  distance {r['distance']:2d}: mean rho {r['mean_rho']:+.3f} |abs| {r['mean_abs_rho']:.3f} n={r['n_pairs']}")
    fit = res["cascade"].get("exp_decay_fit_abs_rho", {})
    if fit:
        print(
            f"  exp fit log|rho| slope/layer {fit.get('log_abs_rho_slope_per_layer', float('nan')):+.4f}; "
            f"lambda {fit.get('decay_lambda_if_negative', float('nan')):.4f}"
        )


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str, str]:
    if args.npz:
        npz = args.npz
    else:
        if not args.dataset:
            raise SystemExit("provide --dataset or npz path")
        npz = os.path.join(args.data_dir, "features", f"full_{args.dataset}.npz")
    if args.hidden_dir:
        hidden = args.hidden_dir
    else:
        stem = args.dataset or os.path.basename(npz).replace("full_", "").replace(".npz", "")
        hidden = os.path.join(args.data_dir, "hidden", stem)
    stem = args.dataset or os.path.splitext(os.path.basename(npz))[0]
    return npz, hidden, stem


def make_cloud(n: int, d: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    r = min(n, d)
    qn, _ = np.linalg.qr(rng.normal(size=(n, r)))
    qd, _ = np.linalg.qr(rng.normal(size=(d, r)))
    s = np.arange(1, r + 1, dtype=float) ** (-alpha)
    H = qn @ np.diag(s) @ qd.T
    H += 0.005 * rng.normal(size=H.shape)
    return H.astype(np.float32)


def make_selftest_data(root: str, *, layer_values: Sequence[int] = (10, 14, 18, 22), seed: int = 11) -> Tuple[str, str]:
    rng = np.random.default_rng(seed)
    features = os.path.join(root, "features")
    hidden = os.path.join(root, "hidden", "selftest")
    os.makedirs(features, exist_ok=True)
    os.makedirs(hidden, exist_ok=True)
    n_chains, d = 60, 32
    ids, ges, pids, ranges_all = [], [], [], []
    for i in range(n_chains):
        cid = f"selftest-{i}"
        T = int(rng.integers(5, 8))
        lens = rng.integers(8, 14, size=T)
        starts = np.cumsum(np.r_[0, lens[:-1]])
        ranges = np.stack([starts, starts + lens - 1], axis=1).astype(int)
        err = (i % 3) == 0
        gold = int(rng.integers(2, T - 1)) if err else -1
        ids.append(cid)
        ges.append(gold)
        pids.append(i)
        ranges_all.append(ranges)
        layers = []
        for t in range(T):
            # Raw spectral_alpha is larger for faster singular-value decay.
            base_alpha = 1.35 + 0.04 * rng.normal()
            if err and t == gold:
                base_alpha += 0.55
            step_layers = []
            common = 0.08 * rng.normal()
            for li, _L in enumerate(layer_values):
                # Adjacent layers share more fluctuation than distant layers.
                local = common * math.exp(-0.18 * li) + 0.03 * rng.normal()
                step_layers.append(make_cloud(int(lens[t]), d, base_alpha + local, rng))
            layers.append(np.stack(step_layers, axis=1))
        Hall = np.concatenate(layers, axis=0)
        np.save(os.path.join(hidden, _fn(cid)), Hall)
    obj_ranges = np.empty(len(ranges_all), dtype=object)
    obj_ranges[:] = ranges_all
    npz = os.path.join(features, "full_selftest.npz")
    np.savez_compressed(
        npz,
        ids=np.asarray(ids, dtype=object),
        gold_error_step=np.asarray(ges, dtype=int),
        problem_ids=np.asarray(pids, dtype=int),
        step_token_ranges=obj_ranges,
        hidden_layers=np.asarray(layer_values, dtype=int),
        hidden_stored=np.array(True),
    )
    return npz, hidden


def assert_selftest(res: Dict[str, object]) -> None:
    loc = {r["metric"]: r for r in res["localization"]}
    jump = loc.get("phase_jump_l2", {})
    if jump.get("top1", 0.0) < 0.8:
        raise SystemExit(f"selftest failed: phase_jump_l2 top1 too low ({jump.get('top1')})")
    cas = res["cascade"].get("exp_decay_fit_abs_rho", {})
    slope = cas.get("log_abs_rho_slope_per_layer", float("nan"))
    if not np.isfinite(slope):
        raise SystemExit("selftest failed: cascade slope not finite")


def main() -> None:
    ap = argparse.ArgumentParser(description="Step-native spectral hypothesis audit")
    ap.add_argument("npz", nargs="?")
    ap.add_argument("--dataset", choices=["gsm8k", "math", "omnimath"], default=None)
    ap.add_argument("--data_dir", default="/gz-data/research/demo/data")
    ap.add_argument("--hidden_dir", default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--min_tokens", type=int, default=5)
    ap.add_argument("--event_window", type=int, default=3)
    ap.add_argument("--progress", type=int, default=50)
    ap.add_argument("--output_dir", default="outputs/spectral_hypothesis")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            npz, hidden = make_selftest_data(td)
            res = run(npz, hidden, args)
            assert_selftest(res)
            print_result(res)
            os.makedirs(args.output_dir, exist_ok=True)
            out_file = os.path.join(args.output_dir, "selftest.json")
            with open(out_file, "w", encoding="utf-8") as fh:
                json.dump(finite_json(res), fh, indent=2, ensure_ascii=False)
            print(f"\nselftest passed; saved: {out_file}")
        return

    npz, hidden, stem = resolve_paths(args)
    res = run(npz, hidden, args)
    print_result(res)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.max_chains:
        stem += f"_n{args.max_chains}"
    out_file = os.path.join(args.output_dir, f"{stem}_spectral_hypotheses.json")
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(finite_json(res), fh, indent=2, ensure_ascii=False)
    print(f"\nsaved: {out_file}")


if __name__ == "__main__":
    main()
