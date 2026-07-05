#!/usr/bin/env python3
"""Constrained-manifold health audit for step-level reasoning failures.

Inspired by "Reasoning emerges from constrained inference manifolds in large
language models", this script tests a minimal operationalization of the paper's
main constraint before we invest in a full AnchorFlow implementation:

  healthy reasoning = compact dynamics + non-degenerate information volume
                      + preserved prompt/question anchoring

It deliberately reuses the existing full_*.npz step-native features.  The goal
is not to claim the same model-level label-free diagnostic as that paper, but
to ask whether a step-level proxy of "constrained, informative, anchored
manifold" improves first-error detection over the current scalar baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from chain_dynamics_audit import make_selftest_npz
from mechanism_phase_audit import (
    Chain,
    auroc,
    bdir,
    cluster_boot_increment,
    finite_json,
    flatten_labeled,
    load_chains,
    localization_table,
    oof_logit,
    safe_mean,
)


EPS = 1e-9


def arr(c: Chain, name: str) -> np.ndarray:
    return np.asarray(c.features.get(name, np.full(c.n_steps, np.nan)), float)


def zscore(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    m = np.isfinite(v)
    if m.sum() >= 2:
        out[m] = (v[m] - v[m].mean()) / (v[m].std() + EPS)
    return out


def delta(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    if len(v) >= 2:
        out[1:] = v[1:] - v[:-1]
    return out


def leaky_cusum(x: np.ndarray, *, lam: float, kref: float) -> np.ndarray:
    out = np.zeros(len(x), float)
    c = 0.0
    for t, val in enumerate(np.asarray(x, float)):
        u = 0.0 if not np.isfinite(val) else float(val)
        c = max(0.0, lam * c + u - kref)
        out[t] = c
    return out


def finite_count(chains: Sequence[Chain], name: str) -> int:
    n = 0
    for c in chains:
        if name in c.features:
            n += int(np.isfinite(c.features[name]).sum())
    return n


def choose_volume_feature(chains: Sequence[Chain]) -> Optional[str]:
    for name in ("cloud_V", "cloud_D", "cloud_C", "geom_pr", "geom_ae"):
        if finite_count(chains, name) >= 30:
            return name
    return None


def add_manifold_health_features(chains: Sequence[Chain], *, lam: float, kref: float) -> Dict[str, object]:
    volume_feature = choose_volume_feature(chains)
    added = set()
    for c in chains:
        if "resultant" in c.features:
            compression = arr(c, "resultant")
            spread = 1.0 - compression
        elif "coherence" in c.features:
            compression = arr(c, "coherence")
            spread = 1.0 - compression
        else:
            compression = np.full(c.n_steps, np.nan)
            spread = np.full(c.n_steps, np.nan)

        if volume_feature is not None:
            raw_volume = arr(c, volume_feature)
            # Volume proxies can be signed or scale-shifted across extractors.
            # log1p(abs(.)) keeps the non-degeneracy reading monotone in magnitude
            # without assuming a particular feature family.
            volume = np.log1p(np.maximum(0.0, raw_volume - np.nanmin(raw_volume[np.isfinite(raw_volume)]) if np.isfinite(raw_volume).any() else raw_volume))
        else:
            raw_volume = np.full(c.n_steps, np.nan)
            volume = np.full(c.n_steps, np.nan)

        if "q_align" in c.features:
            anchor = arr(c, "q_align")
        else:
            anchor = np.full(c.n_steps, np.nan)
        anchor_loss = 1.0 - anchor

        uncertainty = arr(c, "U_D_mean") if "U_D_mean" in c.features else np.full(c.n_steps, np.nan)

        z_spread = zscore(spread)
        z_comp = zscore(compression)
        z_volume = zscore(volume)
        z_anchor = zscore(anchor)
        z_anchor_loss = zscore(anchor_loss)
        z_unc = zscore(uncertainty)

        # Higher health is good.  Higher *_bad scores are expected to indicate
        # first-error risk.
        health = z_comp + z_volume + z_anchor
        health_break = -health
        diffuse_drift = z_spread + z_anchor_loss
        information_starvation = z_comp - z_volume
        anchored_volume_loss = z_anchor_loss - z_volume
        constrained_manifold_bad = z_spread + z_anchor_loss - z_volume
        confident_manifold_bad = constrained_manifold_bad - z_unc
        uncertain_manifold_bad = constrained_manifold_bad + z_unc
        health_jump_bad = -delta(health)
        phase_break = np.abs(delta(health)) + np.maximum(0.0, delta(diffuse_drift))
        health_cusum = leaky_cusum(health_break, lam=lam, kref=kref)
        phase_cusum = leaky_cusum(phase_break, lam=lam, kref=kref)

        new_features = {
            "mh_compression": compression,
            "mh_spread": spread,
            "mh_volume_raw": raw_volume,
            "mh_volume": volume,
            "mh_anchor": anchor,
            "mh_anchor_loss": anchor_loss,
            "mh_health": health,
            "mh_health_break": health_break,
            "mh_diffuse_drift": diffuse_drift,
            "mh_information_starvation": information_starvation,
            "mh_anchored_volume_loss": anchored_volume_loss,
            "mh_constrained_bad": constrained_manifold_bad,
            "mh_confident_bad": confident_manifold_bad,
            "mh_uncertain_bad": uncertain_manifold_bad,
            "mh_health_jump_bad": health_jump_bad,
            "mh_phase_break": phase_break,
            "mh_health_cusum": health_cusum,
            "mh_phase_cusum": phase_cusum,
        }
        for k, v in new_features.items():
            c.features[k] = np.asarray(v, float)
            added.add(k)
    return {
        "volume_feature": volume_feature,
        "features": sorted(added),
        "interpretation": {
            "mh_health": "compact + non-degenerate volume + anchored; lower is worse",
            "mh_constrained_bad": "spread plus anchor loss minus volume; higher is worse",
            "mh_information_starvation": "compressed but volume-poor; higher is worse",
            "mh_phase_break": "step-to-step health/constraint change; higher is worse",
        },
    }


def flatten_filtered(
    chains: Sequence[Chain],
    names: Sequence[str],
    *,
    high_spread_q: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    threshold = None
    if high_spread_q is not None:
        vals = []
        for c in chains:
            s = arr(c, "mh_spread")
            for t in range(c.n_steps):
                if c.correct or t <= c.gold:
                    if np.isfinite(s[t]):
                        vals.append(s[t])
        threshold = float(np.quantile(vals, high_spread_q)) if vals else float("inf")

    X, y, g = [], [], []
    for c in chains:
        for t in range(c.n_steps):
            if c.correct or t < c.gold:
                yy = 0
            elif t == c.gold:
                yy = 1
            else:
                continue
            if threshold is not None:
                sp = arr(c, "mh_spread")[t]
                if not np.isfinite(sp) or sp < threshold:
                    continue
            X.append([arr(c, nm)[t] for nm in names])
            y.append(yy)
            g.append(c.group)
    return np.asarray(X, float), np.asarray(y, int), np.asarray(g)


def feature_table(chains: Sequence[Chain], names: Sequence[str], *, top: int, high_spread_q: Optional[float] = None):
    rows = []
    for nm in names:
        X, y, _ = flatten_filtered(chains, [nm], high_spread_q=high_spread_q)
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
                "mean_non_error": safe_mean(s[y == 0]),
                "mean_gold_error": safe_mean(s[y == 1]),
                "n": int(m.sum()),
                "err": int(y[m].sum()),
            }
        )
    rows.sort(key=lambda r: np.nan_to_num(r["auroc_bestdir"], nan=-1), reverse=True)
    return rows[:top]


def group_score(chains: Sequence[Chain], names: Sequence[str], *, folds: int):
    names = [nm for nm in names if finite_count(chains, nm) >= 30]
    if not names:
        return None
    X, y, groups, _, _ = flatten_labeled(chains, names)
    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return None
    pred = oof_logit(X, y, groups, folds)
    m = np.isfinite(pred)
    return {
        "features": list(names),
        "score": pred,
        "y": y,
        "groups": groups,
        "auroc": auroc(pred[m], y[m]) if m.any() else float("nan"),
        "n": int(m.sum()),
    }


def group_table(chains: Sequence[Chain], groups: Dict[str, Sequence[str]], *, folds: int, n_boot: int):
    scored = {}
    for label, names in groups.items():
        val = group_score(chains, names, folds=folds)
        if val is not None:
            scored[label] = val
    out = {}
    base = scored.get("anchor_uncertainty")
    for label, val in scored.items():
        row = {
            "features": val["features"],
            "auroc": val["auroc"],
            "n": val["n"],
        }
        if base is not None and label != "anchor_uncertainty":
            row["increment_vs_anchor_uncertainty"] = cluster_boot_increment(
                val["score"],
                base["score"],
                val["y"],
                val["groups"],
                n_boot=n_boot,
                seed=17 + len(out),
            )
            row["baseline_auroc"] = base["auroc"]
        out[label] = row
    return out


def resolve_npz(args: argparse.Namespace) -> str:
    if args.npz:
        return args.npz
    if not args.dataset:
        raise SystemExit("provide npz path or --dataset")
    return os.path.join(args.data_dir, "features", f"full_{args.dataset}.npz")


def run(npz: str, args: argparse.Namespace) -> Dict[str, object]:
    chains, meta = load_chains(npz, layer=args.layer, max_chains=args.max_chains)
    mh = add_manifold_health_features(chains, lam=args.lam, kref=args.kref)
    mh_names = mh["features"]
    core_mh = [
        "mh_health_break",
        "mh_constrained_bad",
        "mh_information_starvation",
        "mh_anchored_volume_loss",
        "mh_confident_bad",
        "mh_phase_break",
        "mh_health_cusum",
        "mh_phase_cusum",
    ]
    groups = {
        "static": ["mh_spread", "logN", "pos"],
        "anchor_uncertainty": ["mh_spread", "mh_anchor_loss", "U_D_mean", "logN", "pos"],
        "manifold_health": core_mh + ["logN", "pos"],
        "manifold_plus_anchor_uncertainty": ["mh_spread", "mh_anchor_loss", "U_D_mean"] + core_mh + ["logN", "pos"],
        "confident_manifold": ["mh_constrained_bad", "mh_confident_bad", "U_D_mean", "logN", "pos"],
    }
    all_names = list(dict.fromkeys(["resultant", "q_align", "U_D_mean", "logN", "pos"] + mh_names))
    res = {
        "meta": {
            **meta,
            "npz": npz,
            "manifold_layer": args.layer,
            "claim": "healthy steps should be compact, anchored, and non-degenerate in information volume",
        },
        "n_chains": len(chains),
        "n_error_chains": int(sum(not c.correct for c in chains)),
        "manifold_health_features": mh,
        "overall_features": feature_table(chains, all_names, top=args.top),
        "high_spread_features": feature_table(chains, all_names, top=args.top, high_spread_q=args.high_spread_q),
        "localization": localization_table(chains, all_names, top=args.top),
        "group_oof": group_table(chains, groups, folds=args.folds, n_boot=args.n_boot),
    }
    return res


def print_rows(rows, title: str, n: int = 12) -> None:
    print(f"\n{title}:")
    for r in rows[:n]:
        print(
            f"  {r['feature']:28s} AUROC {r['auroc_bestdir']:.3f} "
            f"nonerr {r['mean_non_error']:+.3f} err {r['mean_gold_error']:+.3f} n={r['n']} err={r['err']}"
        )


def print_result(res: Dict[str, object]) -> None:
    meta = res["meta"]
    print(f"\n===== constrained manifold health | {os.path.basename(meta['npz'])} | L{meta['layer']} =====")
    print(f"chains {res['n_chains']} | error chains {res['n_error_chains']}")
    mh = res["manifold_health_features"]
    print(f"volume proxy: {mh.get('volume_feature')}")
    print_rows(res["overall_features"], "Overall step/gold-error scores")
    print_rows(res["high_spread_features"], "High-spread subset scores")

    print("\nWithin-chain localization:")
    for r in res["localization"][:10]:
        gain = r["top1"] - r["expected_top1"]
        print(f"  {r['feature']:28s} top1 {r['top1']:.3f} exp {r['expected_top1']:.3f} gain {gain:+.3f} n={r['n']}")

    print("\nOOF groups:")
    for k, v in res["group_oof"].items():
        line = f"  {k:32s} AUROC {v['auroc']:.3f} n={v['n']} features={len(v['features'])}"
        inc = v.get("increment_vs_anchor_uncertainty")
        if inc is not None:
            sig = "SIG" if inc.get("sig") else "ns"
            line += f" inc {inc['point']:+.3f} [{inc['lo']:+.3f},{inc['hi']:+.3f}] {sig}"
        print(line)


def run_selftest(args: argparse.Namespace) -> Dict[str, object]:
    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "manifold_health_selftest.npz")
        make_selftest_npz(npz, n_chains=90, layer=args.layer)
        return run(npz, args)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit constrained-manifold health features")
    ap.add_argument("npz", nargs="?")
    ap.add_argument("--dataset", choices=["gsm8k", "math", "omnimath"], default=None)
    ap.add_argument("--data_dir", default="/gz-data/research/demo/data")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--high_spread_q", type=float, default=0.70)
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--kref", type=float, default=0.25)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--output_dir", default="outputs/manifold_health")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    res = run_selftest(args) if args.selftest else run(resolve_npz(args), args)
    print_result(res)
    os.makedirs(args.output_dir, exist_ok=True)
    stem = "selftest" if args.selftest else (args.dataset or os.path.splitext(os.path.basename(resolve_npz(args)))[0])
    if args.max_chains:
        stem += f"_n{args.max_chains}"
    out_path = os.path.join(args.output_dir, f"{stem}_L{args.layer}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(finite_json(res), fh, indent=2, ensure_ascii=False)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
