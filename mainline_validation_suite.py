#!/usr/bin/env python3
"""Batch runner for the main reasoning-flow validation line.

This is deliberately lighter than the token HGN branch.  It treats the current
main claim as an audit matrix:

  1. strong scalar baseline: anchor_uncertainty
  2. healthy-vs-bad divergence: high-spread subset and transition surprise
  3. online usefulness: per-chain alarms with FPR/recall/delay
  4. mechanism hygiene: residualized localization and increments over baseline

The script reuses chain_dynamics_audit.run instead of duplicating the core
statistics, then writes a compact JSON/Markdown summary for paper-facing
triage across datasets and layers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from chain_dynamics_audit import (
    assert_selftest,
    finite_json,
    make_selftest_npz,
    run as run_chain_dynamics,
)


def parse_csv(text: str, *, cast=str) -> List:
    return [cast(x.strip()) for x in str(text).split(",") if x.strip()]


def bdir(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(max(x, 1.0 - x))


def safe_get(d: Dict, path: Sequence, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def best_by(rows: Sequence[Dict], *, score_fn, default=None):
    vals = [r for r in rows if score_fn(r) is not None and np.isfinite(score_fn(r))]
    if not vals:
        return default
    return max(vals, key=score_fn)


def best_online_alarm(res: Dict, *, max_fpr: float) -> Optional[Dict]:
    rows = []
    for label, model in res.get("transition_models", {}).items():
        for row in model.get("online", []):
            if float(row.get("fpr", 1.0)) <= max_fpr:
                rows.append({**row, "transition_model": label})
    return best_by(rows, score_fn=lambda r: float(r.get("recall", float("nan"))))


def increment_row(res: Dict, group: str) -> Optional[Dict]:
    for row in res.get("group_increments_vs_anchor_uncertainty", []):
        if row.get("group") == group:
            return row
    return None


def summarize_result(dataset: str, layer: int, res: Dict, *, max_fpr: float) -> Dict[str, object]:
    group_oof = res.get("group_oof", {})
    anchor = group_oof.get("anchor_uncertainty", {})
    sequence = group_oof.get("sequence_state", {})
    dynamic = group_oof.get("dynamic_online", {})
    transition = group_oof.get("transition_ablation", {})

    seq_inc = increment_row(res, "sequence_state")
    dyn_inc = increment_row(res, "dynamic_online")
    trans_inc = increment_row(res, "transition_ablation")

    high = best_by(
        res.get("high_spread_features", []),
        score_fn=lambda r: bdir(float(r.get("auroc", r.get("auroc_bestdir", float("nan"))))),
    )
    resid_loc = best_by(
        res.get("residual_localization", []),
        score_fn=lambda r: float(r.get("top1", float("nan"))) - float(r.get("expected_top1", float("nan"))),
    )
    causal_loc = best_by(
        res.get("causal_pattern_localization", []),
        score_fn=lambda r: float(r.get("top1", float("nan"))) - float(r.get("expected_top1", float("nan"))),
    )
    alarm = best_online_alarm(res, max_fpr=max_fpr)

    def inc_point(row):
        return safe_get(row or {}, ["increment", "point"], float("nan"))

    def inc_sig(row):
        return bool(safe_get(row or {}, ["increment", "sig"], False))

    return {
        "dataset": dataset,
        "layer": int(layer),
        "chains": int(res.get("n_chains", 0)),
        "error_chains": int(res.get("n_error_chains", 0)),
        "anchor_uncertainty_auroc": float(anchor.get("auroc", float("nan"))),
        "sequence_state_auroc": float(sequence.get("auroc", float("nan"))),
        "sequence_state_increment": float(inc_point(seq_inc)),
        "sequence_state_increment_sig": inc_sig(seq_inc),
        "dynamic_online_auroc": float(dynamic.get("auroc", float("nan"))),
        "dynamic_online_increment": float(inc_point(dyn_inc)),
        "dynamic_online_increment_sig": inc_sig(dyn_inc),
        "transition_ablation_auroc": float(transition.get("auroc", float("nan"))),
        "transition_ablation_increment": float(inc_point(trans_inc)),
        "transition_ablation_increment_sig": inc_sig(trans_inc),
        "best_high_spread_feature": None if high is None else high.get("feature"),
        "best_high_spread_auroc": float(high.get("auroc_bestdir", float("nan"))) if high else float("nan"),
        "best_residual_loc_feature": None if resid_loc is None else resid_loc.get("feature"),
        "best_residual_loc_gain": (
            float(resid_loc.get("top1", float("nan"))) - float(resid_loc.get("expected_top1", float("nan")))
            if resid_loc
            else float("nan")
        ),
        "best_causal_loc_feature": None if causal_loc is None else causal_loc.get("feature"),
        "best_causal_loc_gain": (
            float(causal_loc.get("top1", float("nan"))) - float(causal_loc.get("expected_top1", float("nan")))
            if causal_loc
            else float("nan")
        ),
        "online_alarm": alarm or {},
        "predeclared_replication": list(res.get("predeclared_replication", [])),
        "recommendation": recommendation(anchor, sequence, seq_inc, alarm),
    }


def recommendation(anchor: Dict, sequence: Dict, seq_inc: Optional[Dict], alarm: Optional[Dict]) -> str:
    a = float(anchor.get("auroc", float("nan")))
    s = float(sequence.get("auroc", float("nan")))
    inc = safe_get(seq_inc or {}, ["increment", "point"], float("nan"))
    sig = bool(safe_get(seq_inc or {}, ["increment", "sig"], False))
    recall = float((alarm or {}).get("recall", float("nan")))
    if np.isfinite(inc) and inc > 0.015 and sig:
        return "promote_sequence_state"
    if np.isfinite(recall) and recall >= 0.45 and np.isfinite(a) and a >= 0.75:
        return "intervention_ready_but_keep_simple_detector"
    if np.isfinite(s) and np.isfinite(a) and s <= a + 0.005:
        return "do_not_overfit_sequence_model"
    return "needs_signal_redesign"


def make_chain_args(args: argparse.Namespace, *, dataset: str, layer: int) -> SimpleNamespace:
    return SimpleNamespace(
        npz=None,
        dataset=dataset,
        data_dir=args.data_dir,
        max_chains=args.max_chains,
        layer=layer,
        folds=args.folds,
        controls=args.controls,
        ridge=args.ridge,
        obs=args.obs,
        obs_grid=args.obs_grid,
        min_finite=args.min_finite,
        recovery_horizon=args.recovery_horizon,
        high_spread_q=args.high_spread_q,
        lam=args.lam,
        kref=args.kref,
        eps_list=args.eps_list,
        pattern_window=args.pattern_window,
        event_window=args.event_window,
        n_boot=args.n_boot,
        top=args.top,
        output_dir=args.output_dir,
    )


def resolve_npz(data_dir: str, dataset: str) -> str:
    return os.path.join(data_dir, "features", f"full_{dataset}.npz")


def _number(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def flatten_replication_rows(summaries: Sequence[Dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary in summaries:
        for signal in summary.get("predeclared_replication", []):
            raw = signal.get("raw", {}) or {}
            residual = signal.get("nuisance_residual", {}) or {}
            raw_loc = signal.get("within_chain", {}) or {}
            residual_loc = signal.get("within_chain_residual", {}) or {}
            raw_ci = raw.get("ci95", [float("nan"), float("nan")])
            residual_ci = residual.get("ci95", [float("nan"), float("nan")])
            rows.append(
                {
                    "dataset": summary["dataset"],
                    "layer": int(summary["layer"]),
                    "feature": signal["feature"],
                    "expected_direction": signal.get("expected_direction", "higher_is_error"),
                    "raw_auc": _number(raw.get("auroc_high_is_error")),
                    "raw_ci_low": _number(raw_ci[0] if len(raw_ci) > 0 else np.nan),
                    "raw_ci_high": _number(raw_ci[1] if len(raw_ci) > 1 else np.nan),
                    "residual_auc": _number(residual.get("auroc_high_is_error")),
                    "residual_ci_low": _number(
                        residual_ci[0] if len(residual_ci) > 0 else np.nan
                    ),
                    "residual_ci_high": _number(
                        residual_ci[1] if len(residual_ci) > 1 else np.nan
                    ),
                    "raw_top1": _number(raw_loc.get("top1")),
                    "raw_expected_top1": _number(raw_loc.get("expected_top1")),
                    "residual_top1": _number(residual_loc.get("top1")),
                    "residual_expected_top1": _number(
                        residual_loc.get("expected_top1")
                    ),
                    "n": int(raw.get("n", 0) or 0),
                    "errors": int(raw.get("errors", 0) or 0),
                }
            )
    return rows


def aggregate_replication(
    summaries: Sequence[Dict],
    expected_datasets: Sequence[str],
) -> List[Dict[str, object]]:
    rows = flatten_replication_rows(summaries)
    expected = list(dict.fromkeys(str(name) for name in expected_datasets))
    grouped: Dict[tuple[int, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["layer"]), str(row["feature"])), []).append(row)

    output: List[Dict[str, object]] = []
    for (layer, feature), values in sorted(grouped.items()):
        by_dataset = {str(row["dataset"]): row for row in values}
        observed = [name for name in expected if name in by_dataset]
        selected = [by_dataset[name] for name in observed]
        raw_auc = np.asarray([row["raw_auc"] for row in selected], float)
        residual_auc = np.asarray([row["residual_auc"] for row in selected], float)
        raw_low = np.asarray([row["raw_ci_low"] for row in selected], float)
        residual_low = np.asarray([row["residual_ci_low"] for row in selected], float)
        complete = bool(len(observed) == len(expected) and len(expected) > 0)
        finite_raw = bool(raw_auc.size and np.all(np.isfinite(raw_auc)))
        finite_residual = bool(residual_auc.size and np.all(np.isfinite(residual_auc)))
        output.append(
            {
                "layer": int(layer),
                "feature": feature,
                "expected_datasets": expected,
                "observed_datasets": observed,
                "complete": complete,
                "raw_macro_auc": float(np.mean(raw_auc)) if finite_raw else float("nan"),
                "raw_min_auc": float(np.min(raw_auc)) if finite_raw else float("nan"),
                "residual_macro_auc": (
                    float(np.mean(residual_auc)) if finite_residual else float("nan")
                ),
                "residual_min_auc": (
                    float(np.min(residual_auc)) if finite_residual else float("nan")
                ),
                "raw_direction_consistent": bool(
                    complete and finite_raw and np.all(raw_auc > 0.5)
                ),
                "residual_direction_consistent": bool(
                    complete and finite_residual and np.all(residual_auc > 0.5)
                ),
                "raw_ci_replication": bool(
                    complete and raw_low.size and np.all(np.isfinite(raw_low)) and np.all(raw_low > 0.5)
                ),
                "residual_ci_replication": bool(
                    complete
                    and residual_low.size
                    and np.all(np.isfinite(residual_low))
                    and np.all(residual_low > 0.5)
                ),
            }
        )
    return output


def write_replication_csv(path: str, summaries: Sequence[Dict]) -> None:
    rows = flatten_replication_rows(summaries)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: str,
    summaries: Sequence[Dict],
    replication: Sequence[Dict[str, object]] = (),
) -> None:
    lines = [
        "# Mainline Validation Summary",
        "",
        "This table is the triage layer over `chain_dynamics_audit.py`: anchor baseline, sequence-state increment, high-divergence behavior, residualized localization, and online alarm readiness.",
        "",
        "| dataset | L | chains | anchor | sequence | seq inc | high-spread best | residual loc gain | online alarm | recommendation |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for s in summaries:
        alarm = s.get("online_alarm", {}) or {}
        alarm_txt = ""
        if alarm:
            alarm_txt = (
                f"{alarm.get('transition_model','?')}/{alarm.get('method','?')} "
                f"FPR {float(alarm.get('fpr', math.nan)):.2f} "
                f"R {float(alarm.get('recall', math.nan)):.2f} "
                f"D {float(alarm.get('median_delay', math.nan)):+.1f}"
            )
        lines.append(
            "| {dataset} | {layer} | {chains} | {anchor:.3f} | {seq:.3f} | {inc:+.3f}{sig} | {hf} {ha:.3f} | {lg:+.3f} | {alarm} | {rec} |".format(
                dataset=s["dataset"],
                layer=s["layer"],
                chains=s["chains"],
                anchor=s["anchor_uncertainty_auroc"],
                seq=s["sequence_state_auroc"],
                inc=s["sequence_state_increment"],
                sig="*" if s["sequence_state_increment_sig"] else "",
                hf=s.get("best_high_spread_feature") or "",
                ha=s["best_high_spread_auroc"],
                lg=s["best_residual_loc_gain"],
                alarm=alarm_txt,
                rec=s["recommendation"],
            )
        )
    lines.extend(
        [
            "",
            "Legend: `seq inc` is OOF AUROC increment over `anchor_uncertainty`; `*` means the cluster bootstrap interval excluded zero.",
            "",
            "Decision rule:",
            "- promote sequence/state modeling only when sequence increment is positive and stable;",
            "- otherwise keep the detector simple and invest in richer constraint anchors, attention/logit traces, or intervention design;",
            "- online alarm rows are for real-time guard feasibility, not final paper evidence by themselves.",
        ]
    )
    replication_rows = flatten_replication_rows(summaries)
    if replication_rows:
        lines.extend(
            [
                "",
                "## Frozen Cross-Dataset Replication",
                "",
                "Every signal is evaluated in its predeclared direction (`higher = error`). "
                "No benchmark-specific sign flip or best-feature selection is used.",
                "",
                "| dataset | L | signal | raw AUROC [CI95] | nuisance-residual AUROC [CI95] | residual top1 / random |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        for row in replication_rows:
            lines.append(
                "| {dataset} | {layer} | {feature} | {raw:.3f} [{rlo:.3f}, {rhi:.3f}] | "
                "{resid:.3f} [{slo:.3f}, {shi:.3f}] | {top1:.3f} / {expected:.3f} |".format(
                    dataset=row["dataset"],
                    layer=row["layer"],
                    feature=row["feature"],
                    raw=row["raw_auc"],
                    rlo=row["raw_ci_low"],
                    rhi=row["raw_ci_high"],
                    resid=row["residual_auc"],
                    slo=row["residual_ci_low"],
                    shi=row["residual_ci_high"],
                    top1=row["residual_top1"],
                    expected=row["residual_expected_top1"],
                )
            )
    if replication:
        lines.extend(
            [
                "",
                "## Cross-Dataset Gate",
                "",
                "`direction` requires AUROC above 0.5 on every requested dataset. "
                "`CI replication` is the stronger gate: every dataset-level 95% CI must lie above 0.5.",
                "",
                "| L | signal | observed | raw macro/min | residual macro/min | raw direction / CI | residual direction / CI |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in replication:
            lines.append(
                "| {layer} | {feature} | {observed}/{expected} | {raw_macro:.3f}/{raw_min:.3f} | "
                "{res_macro:.3f}/{res_min:.3f} | {raw_dir}/{raw_ci} | {res_dir}/{res_ci} |".format(
                    layer=row["layer"],
                    feature=row["feature"],
                    observed=len(row["observed_datasets"]),
                    expected=len(row["expected_datasets"]),
                    raw_macro=row["raw_macro_auc"],
                    raw_min=row["raw_min_auc"],
                    res_macro=row["residual_macro_auc"],
                    res_min=row["residual_min_auc"],
                    raw_dir=int(row["raw_direction_consistent"]),
                    raw_ci=int(row["raw_ci_replication"]),
                    res_dir=int(row["residual_direction_consistent"]),
                    res_ci=int(row["residual_ci_replication"]),
                )
            )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def print_summary(summaries: Sequence[Dict]) -> None:
    print("\n===== mainline validation suite =====", flush=True)
    print(
        f"{'dataset':10s} {'L':>3s} {'anchor':>7s} {'seq':>7s} {'inc':>8s} "
        f"{'high-spread':>18s} {'loc+':>7s} {'alarm':>24s} {'recommendation':>30s}",
        flush=True,
    )
    for s in summaries:
        alarm = s.get("online_alarm", {}) or {}
        alarm_txt = "-"
        if alarm:
            alarm_txt = f"{alarm.get('method','?')} R{float(alarm.get('recall', math.nan)):.2f}/F{float(alarm.get('fpr', math.nan)):.2f}"
        print(
            f"{s['dataset']:10s} {s['layer']:3d} "
            f"{s['anchor_uncertainty_auroc']:7.3f} {s['sequence_state_auroc']:7.3f} "
            f"{s['sequence_state_increment']:+8.3f} "
            f"{str(s.get('best_high_spread_feature') or '-')[:18]:>18s} "
            f"{s['best_residual_loc_gain']:+7.3f} "
            f"{alarm_txt[:24]:>24s} {s['recommendation']:>30s}",
            flush=True,
        )


def print_replication(replication: Sequence[Dict[str, object]]) -> None:
    if not replication:
        return
    print("\n===== frozen cross-dataset replication =====", flush=True)
    print(
        f"{'L':>3s} {'signal':42s} {'seen':>5s} {'raw':>11s} {'residual':>11s} "
        f"{'raw gate':>9s} {'resid gate':>10s}",
        flush=True,
    )
    for row in replication:
        raw_gate = (
            "CI"
            if row["raw_ci_replication"]
            else "dir"
            if row["raw_direction_consistent"]
            else "fail"
        )
        residual_gate = (
            "CI"
            if row["residual_ci_replication"]
            else "dir"
            if row["residual_direction_consistent"]
            else "fail"
        )
        print(
            f"{int(row['layer']):3d} {str(row['feature'])[:42]:42s} "
            f"{len(row['observed_datasets']):2d}/{len(row['expected_datasets']):2d} "
            f"{float(row['raw_macro_auc']):.3f}/{float(row['raw_min_auc']):.3f} "
            f"{float(row['residual_macro_auc']):.3f}/{float(row['residual_min_auc']):.3f} "
            f"{raw_gate:>9s} {residual_gate:>10s}",
            flush=True,
        )


def run_suite(args: argparse.Namespace) -> Dict[str, object]:
    datasets = parse_csv(args.datasets, cast=str)
    layers = parse_csv(args.layers, cast=int)
    os.makedirs(args.output_dir, exist_ok=True)

    full_results: Dict[str, object] = {}
    summaries: List[Dict] = []
    partial_json_path = os.path.join(args.output_dir, "mainline_validation_partial.json")
    partial_md_path = os.path.join(args.output_dir, "mainline_validation_partial.md")
    for dataset in datasets:
        for layer in layers:
            npz = resolve_npz(args.data_dir, dataset)
            chain_args = make_chain_args(args, dataset=dataset, layer=layer)
            t0 = time.time()
            print(f"\n[mainline] running {dataset} L{layer}: {npz}", flush=True)
            res = run_chain_dynamics(npz, chain_args)
            key = f"{dataset}_L{layer}"
            full_results[key] = res
            summary = summarize_result(dataset, layer, res, max_fpr=args.max_alarm_fpr)
            summaries.append(summary)
            replication = aggregate_replication(summaries, datasets)
            print(f"[mainline] finished {dataset} L{layer} in {time.time() - t0:.1f}s", flush=True)
            print_summary([summary])
            partial = {
                "meta": {
                    "datasets": datasets,
                    "layers": layers,
                    "data_dir": args.data_dir,
                    "folds": args.folds,
                    "max_chains": args.max_chains,
                    "max_alarm_fpr": args.max_alarm_fpr,
                    "partial": True,
                },
                "summaries": summaries,
                "cross_dataset_replication": replication,
                "results": full_results if args.keep_full_results else {},
            }
            with open(partial_json_path, "w", encoding="utf-8") as fh:
                json.dump(finite_json(partial), fh, indent=2, ensure_ascii=False)
            write_markdown(partial_md_path, summaries, replication)
            write_replication_csv(
                os.path.join(args.output_dir, "cross_dataset_replication_partial.csv"),
                summaries,
            )
            print(f"[mainline] partial saved: {partial_json_path}", flush=True)

    replication = aggregate_replication(summaries, datasets)
    out = {
        "meta": {
            "datasets": datasets,
            "layers": layers,
            "data_dir": args.data_dir,
            "folds": args.folds,
            "max_chains": args.max_chains,
            "max_alarm_fpr": args.max_alarm_fpr,
            "claim": "reasoning failures are online breaks in anchored constraint flow, not merely high spread",
        },
        "summaries": summaries,
        "cross_dataset_replication": replication,
        "results": full_results if args.keep_full_results else {},
    }
    json_path = os.path.join(args.output_dir, "mainline_validation_summary.json")
    md_path = os.path.join(args.output_dir, "mainline_validation_summary.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(finite_json(out), fh, indent=2, ensure_ascii=False)
    write_markdown(md_path, summaries, replication)
    csv_path = os.path.join(args.output_dir, "cross_dataset_replication.csv")
    write_replication_csv(csv_path, summaries)
    out["saved_json"] = json_path
    out["saved_markdown"] = md_path
    out["saved_replication_csv"] = csv_path
    return out


def run_selftest(args: argparse.Namespace) -> Dict[str, object]:
    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "mainline_suite_selftest.npz")
        make_selftest_npz(npz, layer=14)
        chain_args = make_chain_args(args, dataset="selftest", layer=14)
        chain_args.max_chains = 0
        chain_args.folds = min(args.folds, 3)
        chain_args.n_boot = min(args.n_boot, 50)
        res = run_chain_dynamics(npz, chain_args)
        assert_selftest(res)
        summary = summarize_result("selftest", 14, res, max_fpr=args.max_alarm_fpr)
        replication = aggregate_replication([summary], ["selftest"])
        os.makedirs(args.output_dir, exist_ok=True)
        out = {
            "summaries": [summary],
            "cross_dataset_replication": replication,
            "results": {"selftest_L14": res},
        }
        path = os.path.join(args.output_dir, "mainline_validation_selftest.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(finite_json(out), fh, indent=2, ensure_ascii=False)
        write_markdown(
            os.path.join(args.output_dir, "mainline_validation_selftest.md"),
            [summary],
            replication,
        )
        csv_path = os.path.join(args.output_dir, "cross_dataset_replication_selftest.csv")
        write_replication_csv(csv_path, [summary])
        out["saved_json"] = path
        out["saved_replication_csv"] = csv_path
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the main reasoning-flow validation matrix")
    ap.add_argument("--datasets", default="gsm8k,math,omnimath")
    ap.add_argument("--layers", default="14")
    ap.add_argument("--data_dir", default="/gz-data/research/demo/data")
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--controls", default="logN,pos")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--obs", default=None)
    ap.add_argument("--obs_grid", default=None)
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
    ap.add_argument("--max_alarm_fpr", type=float, default=0.20)
    ap.add_argument("--keep_full_results", action="store_true")
    ap.add_argument("--output_dir", default="outputs/mainline_validation")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    out = run_selftest(args) if args.selftest else run_suite(args)
    print_summary(out["summaries"])
    print_replication(out.get("cross_dataset_replication", []))
    print(f"\nsaved json: {out['saved_json']}")
    if "saved_markdown" in out:
        print(f"saved markdown: {out['saved_markdown']}")
    if "saved_replication_csv" in out:
        print(f"saved replication csv: {out['saved_replication_csv']}")


if __name__ == "__main__":
    main()
