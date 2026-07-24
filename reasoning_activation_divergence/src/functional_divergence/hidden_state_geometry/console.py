"""Concise foreground reports for hidden-state geometry runs.

Detailed provenance, fold diagnostics, and optimizer records remain in the
JSON artifacts. This module intentionally selects only the quantities needed
to decide whether a completed run is worth inspecting further.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _number(value: Any) -> float:
    return float(value)


def _signed(value: Any) -> str:
    return f"{_number(value):+.4f}"


def _joined(values: Any) -> str:
    return ", ".join(str(value) for value in values)


def _increment_status(increment: Mapping[str, Any]) -> tuple[str, str]:
    if _number(increment["ci_low"]) > 0:
        return (
            "better_on_evaluated_domains",
            "conditional bootstrap CI entirely > 0; candidate model better",
        )
    if _number(increment["ci_high"]) < 0:
        return (
            "worse_on_evaluated_domains",
            "conditional bootstrap CI entirely < 0; candidate model worse",
        )
    return "uncertain", "conditional bootstrap CI includes 0"


def _inference_label(scope: Any) -> str:
    if scope == "conditional_test_problem_group_cluster_bootstrap":
        return "conditional test problem-group bootstrap"
    return str(scope).replace("_", " ")


def format_run_summary(result: Mapping[str, Any], output_dir: str | Path) -> str:
    """Format the decision-relevant summary of a completed experiment."""
    data = result["data"]
    execution = result["execution"]
    method = result["method"]
    cap = int(execution["max_records_per_domain"])
    cap_text = "all eligible records/domain" if cap == 0 else f"{cap}/domain"
    lines = [
        "Run complete; detailed artifacts are on disk.",
        f"output_dir: {Path(output_dir)}",
        f"run_id: {result['run_id']}",
        (
            f"records: {data['records']} | domains: {_joined(data['domains'])} | "
            f"method: {method['name']} | cap: {cap_text}"
        ),
        f"bootstrap_replicates: {execution['bootstrap_replicates']}",
        "Increment convention: positive = candidate model lower NLL (better).",
    ]
    for task_name, task in result["tasks"].items():
        lines.append(f"\n{task_name}: rows={task['rows']} | events={task['events']}")
        summary = task["summary"]
        lines.append("  domain-macro metrics (problem-group balanced):")
        for arm_name, scores in summary["arms"].items():
            macro = scores["macro"]
            lines.append(
                f"    {arm_name}: AUROC={_number(macro['auroc']):.4f} | "
                f"AUPRC={_number(macro['auprc']):.4f} | "
                f"NLL={_number(macro['nll_nats']):.4f}"
            )
        increments = summary["increments"]
        scopes = {increment["inference_scope"] for increment in increments.values()}
        shared_scope = next(iter(scopes)) if len(scopes) == 1 else None
        if shared_scope is not None:
            lines.append(f"  increment inference: {_inference_label(shared_scope)}")
        for name, increment in increments.items():
            status, explanation = _increment_status(increment)
            scope = (
                ""
                if shared_scope is not None
                else f" | inference: {_inference_label(increment['inference_scope'])}"
            )
            lines.append(
                f"  {name}: {_signed(increment['point'])} "
                f"[95% CI {_signed(increment['ci_low'])}, "
                f"{_signed(increment['ci_high'])}] | {status} ({explanation}){scope}"
            )
    return "\n".join(lines)


def format_preflight_summary(result: Mapping[str, Any]) -> str:
    """Format a compact domain/provenance report after shard validation."""
    lines = [
        (
            f"preflight: run_id={result['run_id']} | "
            f"validated shards={result['shards_validated']}"
        )
    ]
    for domain in result["domains"]:
        lines.append(
            f"{domain['dataset']}: selected={domain['selected_records']} | "
            f"error={domain['error_records']} | correct={domain['correct_records']} | "
            f"layers={len(domain['layers'])} | hidden_dim={domain['hidden_dimension']} | "
            f"generator={_joined(domain['response_generators'])} | "
            f"observer={_joined(domain['observer_models'])}"
        )
    return "\n".join(lines)
