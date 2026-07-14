from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _finite_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _ci(metric: Mapping[str, Any], *, bits: bool = False) -> str:
    suffix = "_bits" if bits else ""
    point = metric.get(f"point{suffix}")
    low = metric.get(f"ci_low{suffix}")
    high = metric.get(f"ci_high{suffix}")
    if point is None or low is None or high is None:
        return "NA"
    return f"{float(point):+.4f} [{float(low):+.4f}, {float(high):+.4f}]"


def _markdown(report: Mapping[str, Any]) -> str:
    preflight = report["preflight"]
    geometry = preflight["geometry"]
    coupling = report["bidirectional_coupling"]
    response = report["response_detection"]
    online = report["online_response_detection"]
    forecast = report["future_output_forecast"]
    lines = [
        "# Output-Conditioned Geometric Predictive Information Audit",
        "",
        "## Frozen Research Question",
        "",
        str(report["research_question"]),
        "",
        "The reported conditional usable information is a cross-fitted model-relative quantity, not an unrestricted mutual-information estimate.",
        "",
        "## Preflight",
        "",
        f"- Joined chains: `{preflight['num_joined_chains']}`",
        f"- Error responses: `{preflight['num_errors']}`",
        f"- Output features: `{preflight['output_features']}`",
        f"- Geometry features: `{preflight['geometry_features']}`",
        f"- Geometry tier: `{geometry['tier']}`",
        f"- State source: `{geometry['state_source']}`",
        f"- Layers: `{geometry['layers']}`",
        "",
        "## Bidirectional Coupling",
        "",
        f"Cross-fitted output-to-geometry explained variance: `{float(coupling['output_to_geometry_overall_r2']):.4f}`.",
        "",
        "This is the fraction of standardized geometry recoverable from causal output history under the fixed ridge family. The remaining chart is the geometry tested for incremental prediction.",
        "",
        "## Response Detection",
        "",
        "| prefix | output AUROC | controls+geometry AUROC | output+residual geometry AUROC | usable bits | delta AUROC | null delta AUROC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint, summary in sorted(
        response.items(), key=lambda item: float(item[0])
    ):
        increment = summary["increment"]
        lines.append(
            "| "
            + " | ".join(
                (
                    checkpoint,
                    f"{float(summary['output_only']['auroc']):.4f}",
                    f"{float(summary['controls_plus_geometry']['auroc']):.4f}",
                    f"{float(summary['output_plus_geometry']['auroc']):.4f}",
                    _ci(increment["conditional_usable_information"], bits=True),
                    _ci(increment["delta_auroc"]),
                    _ci(increment["delta_auroc_vs_null"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Shared Online Prefix Detector",
            "",
            f"- Prefix rows: `{online['n_rows']}`",
            f"- Output-only AUROC: `{float(online['output_only']['auroc']):.4f}`",
            f"- Controls + geometry AUROC: `{float(online['controls_plus_geometry']['auroc']):.4f}`",
            f"- Output + residual geometry AUROC: `{float(online['output_plus_geometry']['auroc']):.4f}`",
            f"- Conditional usable information (bits): `{_ci(online['increment']['conditional_usable_information'], bits=True)}`",
            f"- AUROC increment: `{_ci(online['increment']['delta_auroc'])}`",
            "",
            "The shared detector is evaluated on every observed prefix and never uses the eventual response length to select its input. Relative checkpoints above are retrospective diagnostic slices.",
            "",
            "## Future Output Forecast",
            "",
            f"- MSE space: `{forecast['mse_space']}`",
            f"- Output-only MSE: `{float(forecast['output_only_mse']):.6f}`",
            f"- Controls + geometry MSE: `{float(forecast['controls_plus_geometry_mse']):.6f}`",
            f"- Output + residual geometry MSE: `{float(forecast['output_plus_geometry_mse']):.6f}`",
            f"- Partial R2: `{_ci(forecast['increment']['partial_r2'])}`",
            f"- Partial R2 versus length-matched null: `{_ci(forecast['increment']['partial_r2_vs_null'])}`",
            f"- Gaussian conditional information (bits): `{_ci(forecast['increment']['gaussian_conditional_information_bits'])}`",
            "",
            "## Output Baseline Saturation Ladder",
            "",
            "| output tier | output AUROC | +geometry AUROC | usable bits | future partial R2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for tier in ("scalar", "distribution", "full_compact"):
        value = report["output_saturation_ladder"][tier]
        response_tier = value["response"]
        forecast_tier = value["future_output"]
        lines.append(
            f"| {tier} | {float(response_tier['output_only']['auroc']):.4f} | "
            f"{float(response_tier['output_plus_geometry']['auroc']):.4f} | "
            f"{_ci(response_tier['increment']['conditional_usable_information'], bits=True)} | "
            f"{_ci(forecast_tier['increment']['partial_r2'])} |"
        )
    lines.extend(
        [
            "",
            "## Geometry Group Ablation",
            "",
            "| group | response usable bits | response delta AUROC | future partial R2 |",
            "|---|---:|---:|---:|",
        ]
    )
    for group, value in report["geometry_group_ablation"].items():
        response_group = value["response"]["increment"]
        forecast_group = value["future_output"]["increment"]
        lines.append(
            f"| {group} | "
            f"{_ci(response_group['conditional_usable_information'], bits=True)} | "
            f"{_ci(response_group['delta_auroc'])} | "
            f"{_ci(forecast_group['partial_r2'])} |"
        )
    gate = report["decision_gate"]
    lines.extend(
        [
            "",
            "## Decision Gate",
            "",
            f"- Mechanism supported: `{gate['mechanism_supported']}`",
            f"- Detector increment supported: `{gate['detector_increment_supported']}`",
            f"- Confirmatory ready: `{gate['confirmatory_ready']}`",
            "",
        ]
    )
    for condition, passed in gate["conditions"].items():
        lines.append(f"- `{condition}`: `{passed}`")
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "A positive response increment alone is not a mechanism result. The mechanism claim requires residual geometry to predict future output change and to beat the matched null. A negative result falsifies the proposed geometry family, not the existence of every possible internal signal.",
            "",
        ]
    )
    return "\n".join(lines)


def save_audit_report(report: Mapping[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = _finite_json(report)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(clean, handle, indent=2, ensure_ascii=False)
    (output_dir / "summary.md").write_text(_markdown(clean), encoding="utf-8")

    rows = report["bidirectional_coupling"]["per_feature"]
    with (output_dir / "geometry_explained_by_output.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("feature", "group", "r2_from_output")
        )
        writer.writeheader()
        writer.writerows(rows)

    importance_rows: list[dict[str, Any]] = []
    for row in report["online_response_detection"]["feature_importance"]:
        importance_rows.append({"task": "online_response", **row})
    for checkpoint, summary in report["response_detection"].items():
        for row in summary["feature_importance"]:
            importance_rows.append({"task": f"response_{checkpoint}", **row})
    for row in report["future_output_forecast"]["feature_importance"]:
        importance_rows.append({"task": "future_output", **row})
    with (output_dir / "conditional_geometry_importance.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task", "rank", "feature", "group", "importance"),
        )
        writer.writeheader()
        writer.writerows(importance_rows)
