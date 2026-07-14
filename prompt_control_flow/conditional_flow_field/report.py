from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluate import finite_json
from .evaluate import (
    ConditionalFlowFieldValidationConfig,
    evaluate_conditional_flow_field,
)
from .schema import ConditionalFlowFieldResult


def _format(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def _save_scores(result: ConditionalFlowFieldResult, path: Path) -> None:
    dataset = result.dataset
    transition = np.empty(len(result.transition_scores), dtype=object)
    for row, values in enumerate(result.transition_scores):
        transition[row] = np.asarray(values, dtype=np.float32)
    np.savez_compressed(
        path,
        original_indices=dataset.original_indices,
        problem_ids=dataset.problem_ids,
        sample_idx=dataset.sample_idx,
        y_error=dataset.y_error,
        is_correct=dataset.is_correct,
        n_steps=dataset.n_steps,
        response_chars=dataset.response_chars,
        layer_ids=dataset.layer_ids,
        transition_score_names=np.asarray(result.transition_score_names, dtype=object),
        transition_scores=transition,
        chain_score_names=np.asarray(result.chain_score_names, dtype=object),
        chain_scores=np.asarray(result.chain_scores, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(finite_json(result.metadata))),
    )


def _save_chain_csv(result: ConditionalFlowFieldResult, path: Path) -> None:
    dataset = result.dataset
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row",
                "original_index",
                "problem_id",
                "sample_idx",
                "y_error",
                "is_correct",
                "n_steps",
                "response_chars",
                *result.chain_score_names,
            ]
        )
        for row in range(dataset.n_samples):
            writer.writerow(
                [
                    row,
                    int(dataset.original_indices[row]),
                    dataset.problem_ids[row],
                    int(dataset.sample_idx[row]),
                    int(dataset.y_error[row]),
                    int(dataset.is_correct[row]),
                    int(dataset.n_steps[row]),
                    int(dataset.response_chars[row]),
                    *[float(value) for value in result.chain_scores[row]],
                ]
            )


def _markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    gate1 = summary["geometry_existence_gate"]
    gate2 = summary["error_excursion_gate"]
    lines = [
        "# Conditional Spherical Feasible-Flow Field",
        "",
        "This geometry-only audit uses a nonparametric proper score on the unit sphere. "
        "It does not read logits or train a correctness classifier.",
        "",
        "## Data",
        "",
        f"- Samples: `{dataset['samples']}`",
        f"- Errors / correct: `{dataset['errors']}` / `{dataset['correct']}`",
        f"- Problems: `{dataset['problems']}`",
        f"- Layers: `{dataset['layers']}`",
        "",
        "## Gate 1: Does A Conditional Direction Distribution Exist?",
        "",
        f"**Pass: `{gate1['pass']}`**",
        "",
        f"Score coverage: `{_format(gate1['score_coverage'])}`.",
        "",
        "| correct-target contrast | mean difference | CI95 | problems |",
        "|---|---:|---|---:|",
    ]
    for name, value in gate1["contrasts_on_correct_targets"].items():
        lines.append(
            f"| {name} | {_format(value['mean_difference'])} | "
            f"{value['ci95']} | {value['problems']} |"
        )
    lines.extend(
        [
            "",
            "## Gate 2: Do Errors Make Persistent Low-Density Excursions?",
            "",
            f"**Pass: `{gate2['pass']}`**",
            "",
            f"Primary preregistered score: `{gate2['primary_score']}`.",
            "",
            "| score | coverage | pooled AUROC | within-problem AUROC | CI95 | problems |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    for name, value in summary["response_diagnostics"].items():
        lines.append(
            f"| {name} | {_format(value['coverage'])} | "
            f"{_format(value['pooled_auroc'])} | "
            f"{_format(value['within_problem_auroc_equal_weight'])} | "
            f"{value['within_problem_ci95']} | {value['within_problem_problems']} |"
        )
    decision = summary["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Status: **`{decision['status']}`**",
            "- Advance to output-Fisher/JVP extraction: "
            f"`{decision['advance_to_output_fisher_extraction']}`",
            "- Logits remain blocked unless both geometry gates pass.",
            "",
        ]
    )
    return "\n".join(lines)


def write_conditional_flow_field_report(
    result: ConditionalFlowFieldResult,
    output_dir: str | Path,
    cfg: ConditionalFlowFieldValidationConfig,
) -> tuple[dict[str, Any], dict[str, str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate_conditional_flow_field(result, cfg)
    score_path = output_dir / "conditional_flow_field_scores.npz"
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    chain_path = output_dir / "chain_scores.csv"
    _save_scores(result, score_path)
    _save_chain_csv(result, chain_path)
    clean = finite_json(summary)
    json_path.write_text(
        json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(_markdown(clean), encoding="utf-8")
    return summary, {
        "scores_npz": str(score_path),
        "summary_json": str(json_path),
        "summary_md": str(markdown_path),
        "chain_scores": str(chain_path),
    }
