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
        "fixed_mechanism_increment": dict(res.get("fixed_mechanism_increment", {})),
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


def flatten_component_rows(summaries: Sequence[Dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary in summaries:
        audit = summary.get("fixed_mechanism_increment", {}) or {}
        model_lookup = {
            str(row["model"]): row for row in audit.get("model_scores", [])
        }
        for row in audit.get("unique_component_value", []):
            auc_inc = row.get("auroc_increment", {}) or {}
            pr_inc = row.get("auprc_increment", {}) or {}
            component_auc_inc = row.get("component_over_controls_auroc", {}) or {}
            component_pr_inc = row.get("component_over_controls_auprc", {}) or {}
            full = model_lookup.get(str(row.get("full_model")), {})
            reduced = model_lookup.get(str(row.get("reduced_model")), {})
            component_model = model_lookup.get(str(row.get("component_model")), {})
            controls_model = model_lookup.get(str(row.get("controls_model")), {})
            rows.append(
                {
                    "dataset": summary["dataset"],
                    "layer": int(summary["layer"]),
                    "component": row["component"],
                    "controls_auroc": _number(controls_model.get("auroc")),
                    "component_model_auroc": _number(component_model.get("auroc")),
                    "component_over_controls_auroc_delta": _number(
                        component_auc_inc.get("point")
                    ),
                    "component_over_controls_auroc_low": _number(
                        component_auc_inc.get("lo")
                    ),
                    "component_over_controls_auroc_high": _number(
                        component_auc_inc.get("hi")
                    ),
                    "controls_auprc": _number(controls_model.get("auprc")),
                    "component_model_auprc": _number(component_model.get("auprc")),
                    "component_over_controls_auprc_delta": _number(
                        component_pr_inc.get("point")
                    ),
                    "full_auroc": _number(full.get("auroc")),
                    "reduced_auroc": _number(reduced.get("auroc")),
                    "auroc_delta": _number(auc_inc.get("point")),
                    "auroc_delta_low": _number(auc_inc.get("lo")),
                    "auroc_delta_high": _number(auc_inc.get("hi")),
                    "full_auprc": _number(full.get("auprc")),
                    "reduced_auprc": _number(reduced.get("auprc")),
                    "auprc_delta": _number(pr_inc.get("point")),
                    "auprc_delta_low": _number(pr_inc.get("lo")),
                    "auprc_delta_high": _number(pr_inc.get("hi")),
                }
            )
    return rows


def flatten_additive_rows(summaries: Sequence[Dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary in summaries:
        audit = summary.get("fixed_mechanism_increment", {}) or {}
        for row in audit.get("additive_value", []):
            baseline = row.get("baseline", {}) or {}
            augmented = row.get("augmented", {}) or {}
            auc_inc = row.get("auroc_increment", {}) or {}
            pr_inc = row.get("auprc_increment", {}) or {}
            coefficient = row.get("standardized_coefficient", {}) or {}
            rows.append(
                {
                    "dataset": summary["dataset"],
                    "layer": int(summary["layer"]),
                    "signal": row["signal"],
                    "feature": row["feature"],
                    "baseline_auroc": _number(baseline.get("auroc")),
                    "augmented_auroc": _number(augmented.get("auroc")),
                    "auroc_delta": _number(auc_inc.get("point")),
                    "auroc_delta_low": _number(auc_inc.get("lo")),
                    "auroc_delta_high": _number(auc_inc.get("hi")),
                    "baseline_auprc": _number(baseline.get("auprc")),
                    "augmented_auprc": _number(augmented.get("auprc")),
                    "auprc_delta": _number(pr_inc.get("point")),
                    "auprc_delta_low": _number(pr_inc.get("lo")),
                    "auprc_delta_high": _number(pr_inc.get("hi")),
                    "coefficient_median": _number(coefficient.get("median")),
                    "coefficient_positive_fraction": _number(
                        coefficient.get("positive_fraction")
                    ),
                    "eligible_coverage": _number(row.get("eligible_coverage")),
                    "n": int(augmented.get("n", 0) or 0),
                    "errors": int(augmented.get("errors", 0) or 0),
                }
            )
    return rows


def aggregate_increment_rows(
    rows: Sequence[Dict[str, object]],
    expected_datasets: Sequence[str],
    *,
    key: str,
) -> List[Dict[str, object]]:
    expected = list(dict.fromkeys(str(name) for name in expected_datasets))
    grouped: Dict[tuple[int, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["layer"]), str(row[key])), []).append(row)

    output: List[Dict[str, object]] = []
    for (layer, name), values in sorted(grouped.items()):
        by_dataset = {str(row["dataset"]): row for row in values}
        observed = [dataset for dataset in expected if dataset in by_dataset]
        selected = [by_dataset[dataset] for dataset in observed]
        auc_delta = np.asarray([row["auroc_delta"] for row in selected], float)
        auc_low = np.asarray([row["auroc_delta_low"] for row in selected], float)
        pr_delta = np.asarray([row["auprc_delta"] for row in selected], float)
        pr_low = np.asarray([row["auprc_delta_low"] for row in selected], float)
        complete = bool(len(observed) == len(expected) and len(expected) > 0)
        finite_auc = bool(auc_delta.size and np.all(np.isfinite(auc_delta)))
        finite_pr = bool(pr_delta.size and np.all(np.isfinite(pr_delta)))
        output.append(
            {
                "layer": int(layer),
                key: name,
                "expected_datasets": expected,
                "observed_datasets": observed,
                "complete": complete,
                "auroc_delta_macro": (
                    float(np.mean(auc_delta)) if finite_auc else float("nan")
                ),
                "auroc_delta_min": (
                    float(np.min(auc_delta)) if finite_auc else float("nan")
                ),
                "auprc_delta_macro": (
                    float(np.mean(pr_delta)) if finite_pr else float("nan")
                ),
                "auprc_delta_min": (
                    float(np.min(pr_delta)) if finite_pr else float("nan")
                ),
                "auroc_direction_consistent": bool(
                    complete and finite_auc and np.all(auc_delta > 0.0)
                ),
                "auroc_ci_replication": bool(
                    complete
                    and auc_low.size
                    and np.all(np.isfinite(auc_low))
                    and np.all(auc_low > 0.0)
                ),
                "auprc_direction_consistent": bool(
                    complete and finite_pr and np.all(pr_delta > 0.0)
                ),
                "auprc_ci_replication": bool(
                    complete
                    and pr_low.size
                    and np.all(np.isfinite(pr_low))
                    and np.all(pr_low > 0.0)
                ),
            }
        )
    return output


def write_rows_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
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
    component_replication: Sequence[Dict[str, object]] = (),
    additive_replication: Sequence[Dict[str, object]] = (),
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

    component_rows = flatten_component_rows(summaries)
    if component_rows:
        lines.extend(
            [
                "",
                "## What Is Inside `anchor_uncertainty`?",
                "",
                "`anchor_uncertainty` is a supervised OOF logistic model over spread, "
                "anchor loss, uncertainty, step length, and relative position. It is not "
                "the question-vector cosine alone. Each row below removes one component "
                "while holding rows, folds, preprocessing, and model capacity fixed.",
                "",
                "| dataset | L | component | controls/component AUROC | component gain [CI95] | full/without AUROC | unique AUROC delta [CI95] | unique AUPRC delta [CI95] |",
                "|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in component_rows:
            lines.append(
                "| {dataset} | {layer} | {component} | {controls:.3f}/{component_auc:.3f} | "
                "{component_delta:+.3f} [{component_lo:+.3f}, {component_hi:+.3f}] | "
                "{full:.3f}/{reduced:.3f} | {delta:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                "{pr:+.3f} [{prlo:+.3f}, {prhi:+.3f}] |".format(
                    dataset=row["dataset"],
                    layer=row["layer"],
                    component=row["component"],
                    controls=row["controls_auroc"],
                    component_auc=row["component_model_auroc"],
                    component_delta=row["component_over_controls_auroc_delta"],
                    component_lo=row["component_over_controls_auroc_low"],
                    component_hi=row["component_over_controls_auroc_high"],
                    full=row["full_auroc"],
                    reduced=row["reduced_auroc"],
                    delta=row["auroc_delta"],
                    lo=row["auroc_delta_low"],
                    hi=row["auroc_delta_high"],
                    pr=row["auprc_delta"],
                    prlo=row["auprc_delta_low"],
                    prhi=row["auprc_delta_high"],
                )
            )

    additive_rows = flatten_additive_rows(summaries)
    if additive_rows:
        lines.extend(
            [
                "",
                "## Fixed Additive Value Beyond `anchor_uncertainty`",
                "",
                "Each augmented model adds exactly one predeclared temporal or transition "
                "score to the full baseline. Positive CI replication is required; a strong "
                "standalone score is not evidence of incremental information.",
                "",
                "| dataset | L | added signal | coverage | base/aug AUROC | AUROC delta [CI95] | AUPRC delta [CI95] | coef median / positive folds |",
                "|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in additive_rows:
            lines.append(
                "| {dataset} | {layer} | {signal} | {coverage:.3f} | {base:.3f}/{aug:.3f} | "
                "{delta:+.3f} [{lo:+.3f}, {hi:+.3f}] | {pr:+.3f} [{prlo:+.3f}, {prhi:+.3f}] | "
                "{coef:+.3f} / {sign:.2f} |".format(
                    dataset=row["dataset"],
                    layer=row["layer"],
                    signal=row["signal"],
                    coverage=row["eligible_coverage"],
                    base=row["baseline_auroc"],
                    aug=row["augmented_auroc"],
                    delta=row["auroc_delta"],
                    lo=row["auroc_delta_low"],
                    hi=row["auroc_delta_high"],
                    pr=row["auprc_delta"],
                    prlo=row["auprc_delta_low"],
                    prhi=row["auprc_delta_high"],
                    coef=row["coefficient_median"],
                    sign=row["coefficient_positive_fraction"],
                )
            )

    if component_replication or additive_replication:
        lines.extend(
            [
                "",
                "## Cross-Dataset Increment Gates",
                "",
                "The CI gate passes only when every requested dataset has a paired "
                "cluster-bootstrap lower bound above zero.",
                "",
                "| family | item | observed | AUROC delta macro/min | AUROC dir/CI | AUPRC delta macro/min | AUPRC dir/CI |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for family, key, rows in (
            ("component", "component", component_replication),
            ("addition", "signal", additive_replication),
        ):
            for row in rows:
                lines.append(
                    "| {family} | {item} | {observed}/{expected} | {auc_macro:+.3f}/{auc_min:+.3f} | "
                    "{auc_dir}/{auc_ci} | {pr_macro:+.3f}/{pr_min:+.3f} | {pr_dir}/{pr_ci} |".format(
                        family=family,
                        item=row[key],
                        observed=len(row["observed_datasets"]),
                        expected=len(row["expected_datasets"]),
                        auc_macro=row["auroc_delta_macro"],
                        auc_min=row["auroc_delta_min"],
                        auc_dir=int(row["auroc_direction_consistent"]),
                        auc_ci=int(row["auroc_ci_replication"]),
                        pr_macro=row["auprc_delta_macro"],
                        pr_min=row["auprc_delta_min"],
                        pr_dir=int(row["auprc_direction_consistent"]),
                        pr_ci=int(row["auprc_ci_replication"]),
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


def print_increment_replication(
    component_replication: Sequence[Dict[str, object]],
    additive_replication: Sequence[Dict[str, object]],
) -> None:
    if not component_replication and not additive_replication:
        return
    print("\n===== fixed additive-value replication =====", flush=True)
    print(
        f"{'family':10s} {'item':48s} {'seen':>5s} {'dAUC macro/min':>17s} "
        f"{'AUC gate':>9s} {'dAUPRC macro/min':>18s} {'PR gate':>9s}",
        flush=True,
    )
    for family, key, rows in (
        ("component", "component", component_replication),
        ("addition", "signal", additive_replication),
    ):
        for row in rows:
            auc_gate = (
                "CI"
                if row["auroc_ci_replication"]
                else "dir"
                if row["auroc_direction_consistent"]
                else "fail"
            )
            pr_gate = (
                "CI"
                if row["auprc_ci_replication"]
                else "dir"
                if row["auprc_direction_consistent"]
                else "fail"
            )
            print(
                f"{family:10s} {str(row[key])[:48]:48s} "
                f"{len(row['observed_datasets']):2d}/{len(row['expected_datasets']):2d} "
                f"{float(row['auroc_delta_macro']):+.3f}/{float(row['auroc_delta_min']):+.3f} "
                f"{auc_gate:>9s} "
                f"{float(row['auprc_delta_macro']):+.3f}/{float(row['auprc_delta_min']):+.3f} "
                f"{pr_gate:>9s}",
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
            component_rows = flatten_component_rows(summaries)
            additive_rows = flatten_additive_rows(summaries)
            component_replication = aggregate_increment_rows(
                component_rows,
                datasets,
                key="component",
            )
            additive_replication = aggregate_increment_rows(
                additive_rows,
                datasets,
                key="signal",
            )
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
                "cross_dataset_component_value": component_replication,
                "cross_dataset_additive_value": additive_replication,
                "results": full_results if args.keep_full_results else {},
            }
            with open(partial_json_path, "w", encoding="utf-8") as fh:
                json.dump(finite_json(partial), fh, indent=2, ensure_ascii=False)
            write_markdown(
                partial_md_path,
                summaries,
                replication,
                component_replication,
                additive_replication,
            )
            write_replication_csv(
                os.path.join(args.output_dir, "cross_dataset_replication_partial.csv"),
                summaries,
            )
            write_rows_csv(
                os.path.join(args.output_dir, "mechanism_component_ablation_partial.csv"),
                component_rows,
            )
            write_rows_csv(
                os.path.join(args.output_dir, "transition_additive_value_partial.csv"),
                additive_rows,
            )
            print(f"[mainline] partial saved: {partial_json_path}", flush=True)

    replication = aggregate_replication(summaries, datasets)
    component_rows = flatten_component_rows(summaries)
    additive_rows = flatten_additive_rows(summaries)
    component_replication = aggregate_increment_rows(
        component_rows,
        datasets,
        key="component",
    )
    additive_replication = aggregate_increment_rows(
        additive_rows,
        datasets,
        key="signal",
    )
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
        "cross_dataset_component_value": component_replication,
        "cross_dataset_additive_value": additive_replication,
        "results": full_results if args.keep_full_results else {},
    }
    json_path = os.path.join(args.output_dir, "mainline_validation_summary.json")
    md_path = os.path.join(args.output_dir, "mainline_validation_summary.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(finite_json(out), fh, indent=2, ensure_ascii=False)
    write_markdown(
        md_path,
        summaries,
        replication,
        component_replication,
        additive_replication,
    )
    csv_path = os.path.join(args.output_dir, "cross_dataset_replication.csv")
    write_replication_csv(csv_path, summaries)
    component_csv_path = os.path.join(args.output_dir, "mechanism_component_ablation.csv")
    additive_csv_path = os.path.join(args.output_dir, "transition_additive_value.csv")
    write_rows_csv(component_csv_path, component_rows)
    write_rows_csv(additive_csv_path, additive_rows)
    out["saved_json"] = json_path
    out["saved_markdown"] = md_path
    out["saved_replication_csv"] = csv_path
    out["saved_component_csv"] = component_csv_path
    out["saved_additive_csv"] = additive_csv_path
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
        component_rows = flatten_component_rows([summary])
        additive_rows = flatten_additive_rows([summary])
        component_replication = aggregate_increment_rows(
            component_rows,
            ["selftest"],
            key="component",
        )
        additive_replication = aggregate_increment_rows(
            additive_rows,
            ["selftest"],
            key="signal",
        )
        os.makedirs(args.output_dir, exist_ok=True)
        out = {
            "summaries": [summary],
            "cross_dataset_replication": replication,
            "cross_dataset_component_value": component_replication,
            "cross_dataset_additive_value": additive_replication,
            "results": {"selftest_L14": res},
        }
        path = os.path.join(args.output_dir, "mainline_validation_selftest.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(finite_json(out), fh, indent=2, ensure_ascii=False)
        write_markdown(
            os.path.join(args.output_dir, "mainline_validation_selftest.md"),
            [summary],
            replication,
            component_replication,
            additive_replication,
        )
        csv_path = os.path.join(args.output_dir, "cross_dataset_replication_selftest.csv")
        component_csv_path = os.path.join(args.output_dir, "mechanism_component_ablation_selftest.csv")
        additive_csv_path = os.path.join(args.output_dir, "transition_additive_value_selftest.csv")
        write_replication_csv(csv_path, [summary])
        write_rows_csv(component_csv_path, component_rows)
        write_rows_csv(additive_csv_path, additive_rows)
        out["saved_json"] = path
        out["saved_replication_csv"] = csv_path
        out["saved_component_csv"] = component_csv_path
        out["saved_additive_csv"] = additive_csv_path
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
    print_increment_replication(
        out.get("cross_dataset_component_value", []),
        out.get("cross_dataset_additive_value", []),
    )
    print(f"\nsaved json: {out['saved_json']}")
    if "saved_markdown" in out:
        print(f"saved markdown: {out['saved_markdown']}")
    if "saved_replication_csv" in out:
        print(f"saved replication csv: {out['saved_replication_csv']}")
    if "saved_component_csv" in out:
        print(f"saved component csv: {out['saved_component_csv']}")
    if "saved_additive_csv" in out:
        print(f"saved additive csv: {out['saved_additive_csv']}")


if __name__ == "__main__":
    main()
