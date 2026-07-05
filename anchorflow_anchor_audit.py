#!/usr/bin/env python3
"""Phase-1 AnchorFlow audit.

This is the first code step toward the top-conference version of the project:
replace single qvec cosine with a multi-anchor transport field, then test it
honestly against the current anchor_uncertainty baseline.

Important limitation: current full_*.npz files often do not store prompt text or
prompt-span hidden states. In that case the script uses a deterministic
q-partition fallback anchor bank and reports the mode explicitly. That fallback
is a plumbing/ablation scaffold, not the final semantic AnchorFlow claim.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Dict, List, Sequence

import numpy as np

from anchorflow.anchor_repr import build_anchor_bank
from anchorflow.anchors import anchor_coverage_stats, anchors_to_jsonable, fallback_anchors, parse_anchors
from anchorflow.data import Trace, load_traces
from anchorflow.eval import (
    auroc,
    bdir,
    feature_table,
    finite_json,
    flatten_labeled,
    group_table,
    localization_table,
    safe_mean,
)
from anchorflow.residualize import crossfit_residualize
from anchorflow.transport import add_transport_features
from audit_utils import make_selftest_npz


def resolve_npz(args: argparse.Namespace) -> str:
    if args.npz:
        return args.npz
    if not args.dataset:
        raise SystemExit("provide npz path or --dataset")
    return os.path.join(args.data_dir, "features", f"full_{args.dataset}.npz")


def add_anchorflow_features(
    traces: Sequence[Trace],
    *,
    max_anchors: int,
    tau: float,
    seed: int,
) -> Dict[str, object]:
    anchor_lists = []
    real_banks = []
    random_banks = []
    shuffled_banks = []
    mode_counts: Dict[str, int] = {}
    examples = []
    for tr in traces:
        anchors = parse_anchors(tr.prompt_text, max_anchors=max_anchors) if tr.prompt_text else fallback_anchors()
        anchor_lists.append(anchors)
        bank = build_anchor_bank(tr, anchors, max_anchors=max_anchors, seed=seed)
        rb = build_anchor_bank(tr, anchors, max_anchors=max_anchors, random=True, seed=seed + 101)
        sb = build_anchor_bank(tr, anchors, max_anchors=max_anchors, shuffle_kinds=True, seed=seed + 202)
        real_banks.append(bank)
        random_banks.append(rb)
        shuffled_banks.append(sb)
        mode_counts[bank.mode] = mode_counts.get(bank.mode, 0) + 1
        if len(examples) < 8:
            examples.append({"chain_id": tr.chain_id, "mode": bank.mode, "anchors": anchors_to_jsonable(anchors[:8])})

    made = []
    made += add_transport_features(traces, real_banks, prefix="af", tau=tau)
    made += add_transport_features(traces, random_banks, prefix="afr", tau=tau)
    made += add_transport_features(traces, shuffled_banks, prefix="afs", tau=tau)
    return {
        "features": sorted(set(made)),
        "anchor_stats": anchor_coverage_stats(anchor_lists),
        "bank_modes": mode_counts,
        "examples": examples,
        "limitation": (
            "If bank_modes is q_partition_fallback, this run validates the transport/eval scaffold; "
            "semantic claims require prompt text plus prompt-span hidden anchors."
        ),
    }


def residual_feature_rows(traces: Sequence[Trace], names: Sequence[str], *, folds: int, top: int) -> List[Dict[str, object]]:
    rows = []
    for nm in names:
        X, y, groups, _ = flatten_labeled(traces, [nm])
        C, yc, gc, _ = flatten_labeled(traces, ["logN", "pos"])
        if len(X) == 0 or len(C) != len(X) or not np.array_equal(y, yc) or not np.array_equal(groups, gc):
            continue
        s = X[:, 0]
        r = crossfit_residualize(s, C, groups, folds=folds)
        m = np.isfinite(r)
        if m.sum() < 30 or len(np.unique(y[m])) < 2:
            continue
        raw = auroc(r[m], y[m])
        rows.append(
            {
                "feature": nm,
                "resid_auroc_bestdir": bdir(raw),
                "raw_resid_auroc_high_is_error": raw,
                "resid_mean_non_error": safe_mean(r[(y == 0) & m]),
                "resid_mean_gold_error": safe_mean(r[(y == 1) & m]),
                "n": int(m.sum()),
                "err": int(y[m].sum()),
            }
        )
    rows.sort(key=lambda r: np.nan_to_num(r["resid_auroc_bestdir"], nan=-1), reverse=True)
    return rows[:top]


def run(npz: str, args: argparse.Namespace) -> Dict[str, object]:
    traces, meta = load_traces(
        npz,
        dataset=args.dataset or "",
        layer=args.layer,
        max_chains=args.max_chains,
        hidden_dir=args.hidden_dir,
    )
    af_meta = add_anchorflow_features(traces, max_anchors=args.max_anchors, tau=args.tau, seed=args.seed)

    af_core = [
        "af_detach",
        "af_core_detach",
        "af_anchor_entropy",
        "af_transport_jump",
        "af_cz_detach",
        "af_cz_jump",
        "af_phase_score",
        "af_phase_cusum",
    ]
    random_core = [x.replace("af_", "afr_", 1) for x in af_core]
    shuffled_core = [x.replace("af_", "afs_", 1) for x in af_core]
    baseline = ["spread", "anchor_loss", "U_D_mean", "logN", "pos"]
    groups = {
        "length_pos": ["logN", "pos"],
        "spread_uncertainty": ["spread", "U_D_mean", "logN", "pos"],
        "anchor_uncertainty": baseline,
        "transport_only": af_core + ["logN", "pos"],
        "transport_plus_anchor_uncertainty": af_core + baseline,
        "random_transport": random_core + ["logN", "pos"],
        "shuffled_kind_transport": shuffled_core + ["logN", "pos"],
    }
    all_features = list(
        dict.fromkeys(
            [
                "spread",
                "anchor_loss",
                "U_D_mean",
                "q_align",
                "logN",
                "pos",
            ]
            + af_core
            + ["af_goal_mass", "af_number_mass", "af_constraint_mass", "af_core_mass"]
            + random_core
            + shuffled_core
        )
    )

    res = {
        "meta": {
            **meta,
            "npz": npz,
            "layer": args.layer,
            "claim": "multi-anchor transport should reveal constraint-flow breaks beyond qvec cosine",
        },
        "n_chains": len(traces),
        "n_error_chains": int(sum(not tr.correct for tr in traces)),
        "anchorflow": af_meta,
        "overall_features": feature_table(traces, all_features, top=args.top),
        "high_spread_features": feature_table(traces, all_features, top=args.top, high_spread_q=args.high_spread_q),
        "confident_features": feature_table(traces, all_features, top=args.top, confident_q=args.confident_q),
        "residual_features": residual_feature_rows(traces, all_features, folds=args.folds, top=args.top),
        "localization": localization_table(traces, all_features, top=args.top),
        "group_oof": group_table(traces, groups, folds=args.folds, n_boot=args.n_boot, baseline="anchor_uncertainty"),
    }
    return res


def run_selftest(args: argparse.Namespace) -> Dict[str, object]:
    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "anchorflow_selftest.npz")
        make_selftest_npz(npz, n_chains=90, layer=args.layer)
        return run(npz, args)


def print_rows(rows, title: str, key: str = "auroc_bestdir", n: int = 12) -> None:
    print(f"\n{title}:")
    for r in rows[:n]:
        if key == "resid_auroc_bestdir":
            print(
                f"  {r['feature']:28s} resid-AUROC {r[key]:.3f} "
                f"nonerr {r['resid_mean_non_error']:+.3f} err {r['resid_mean_gold_error']:+.3f}"
            )
        else:
            print(
                f"  {r['feature']:28s} AUROC {r[key]:.3f} "
                f"nonerr {r['mean_non_error']:+.3f} err {r['mean_gold_error']:+.3f}"
            )


def print_result(res: Dict[str, object]) -> None:
    meta = res["meta"]
    print(f"\n===== AnchorFlow anchor audit | {os.path.basename(meta['npz'])} | L{meta['layer']} =====")
    print(f"chains {res['n_chains']} | error chains {res['n_error_chains']}")
    print(f"bank modes: {res['anchorflow']['bank_modes']}")
    print(f"anchor stats: {res['anchorflow']['anchor_stats']}")
    print_rows(res["overall_features"], "Overall step/gold-error scores")
    print_rows(res["high_spread_features"], "High-spread subset scores")
    print_rows(res["confident_features"], "Confident/low-entropy subset scores")
    print_rows(res["residual_features"], "Residualized over [logN,pos]", key="resid_auroc_bestdir")

    print("\nWithin-chain localization:")
    for r in res["localization"][:10]:
        print(
            f"  {r['feature']:28s} top1 {r['top1']:.3f} exp {r['expected_top1']:.3f} "
            f"gain {r['gain']:+.3f} dir={r['direction']} n={r['n']}"
        )

    print("\nOOF groups:")
    for k, v in res["group_oof"].items():
        line = f"  {k:36s} AUROC {v['auroc']:.3f} n={v['n']} features={len(v['features'])}"
        inc = v.get("increment_vs_anchor_uncertainty")
        if inc is not None:
            sig = "SIG" if inc.get("sig") else "ns"
            line += f" inc {inc['point']:+.3f} [{inc['lo']:+.3f},{inc['hi']:+.3f}] {sig}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit phase-1 AnchorFlow multi-anchor transport features")
    ap.add_argument("npz", nargs="?")
    ap.add_argument("--dataset", choices=["gsm8k", "math", "omnimath"], default=None)
    ap.add_argument("--data_dir", default="/gz-data/research/demo/data")
    ap.add_argument("--hidden_dir", default=None)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--max_anchors", type=int, default=24)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--high_spread_q", type=float, default=0.70)
    ap.add_argument("--confident_q", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--output_dir", default="outputs/anchorflow_anchor")
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
