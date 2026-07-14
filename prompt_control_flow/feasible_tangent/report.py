from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluate import finite_json
from .evaluate import FeasibleTangentValidationConfig, evaluate_feasible_tangent
from .schema import FeasibleTangentResult


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def _save_scores(result: FeasibleTangentResult, path: Path) -> None:
    dataset = result.dataset
    packed = np.empty(len(result.transition_scores), dtype=object)
    for index, values in enumerate(result.transition_scores):
        packed[index] = np.asarray(values, dtype=np.float32)
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
        transition_scores=packed,
        chain_score_names=np.asarray(result.chain_score_names, dtype=object),
        chain_scores=np.asarray(result.chain_scores, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(finite_json(result.metadata))),
    )


def _save_chain_csv(result: FeasibleTangentResult, path: Path) -> None:
    dataset = result.dataset
    header = [
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
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
    gate2 = summary["error_escape_gate"]
    lines = [
        "# Same-Problem Feasible-Tangent Gate",
        "",
        "This audit is geometry-only. It neither reads logits nor trains a label classifier.",
        "",
        "## Data",
        "",
        f"- Samples: `{dataset['samples']}`",
        f"- Errors / correct: `{dataset['errors']}` / `{dataset['correct']}`",
        f"- Problems: `{dataset['problems']}`",
        f"- Layers: `{dataset['layers']}`",
        "- The unavailable prompt-to-first-step transition is excluded.",
        "",
        "## Gate 1: Does A Low-Rank Feasible Tangent Exist?",
        "",
        f"**Pass: `{gate1['pass']}`**",
        "",
        "Rank-supported coverage (problem-equal): "
        f"`{_format(gate1['rank_support_equal_problem_mean'])}` "
        f"CI `{gate1['rank_support_ci95']}`.",
        "",
        "| correct-target contrast | mean null - primary | CI95 | problems |",
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
            "## Gate 2: Do Errors Show Persistent Normal Escape?",
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
            f"| {name} | {_format(value['coverage'])} | {_format(value['pooled_auroc'])} | "
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
            "- Advance to exact output cotangent extraction: "
            f"`{decision['advance_to_output_cotangent_extraction']}`",
            "- No output sensitivity claim is permitted unless both gates pass.",
            "",
        ]
    )
    return "\n".join(lines)


def write_feasible_tangent_report(
    result: FeasibleTangentResult,
    output_dir: str | Path,
    cfg: FeasibleTangentValidationConfig,
) -> tuple[dict[str, Any], dict[str, str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate_feasible_tangent(result, cfg)
    score_path = output_dir / "feasible_tangent_scores.npz"
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    chain_path = output_dir / "chain_scores.csv"
    _save_scores(result, score_path)
    _save_chain_csv(result, chain_path)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(finite_json(summary), handle, ensure_ascii=False, indent=2)
    markdown_path.write_text(_markdown(finite_json(summary)), encoding="utf-8")
    return summary, {
        "scores_npz": str(score_path),
        "summary_json": str(json_path),
        "summary_md": str(markdown_path),
        "chain_scores": str(chain_path),
    }
