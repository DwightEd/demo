from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .dataset import (
    BinaryTask,
    ForecastTask,
    build_forecast_task,
    build_online_response_task,
    build_response_tasks,
    join_trace_and_geometry,
    select_geometry_groups,
    select_output_tier,
)
from .geometry_features import load_geometry_collection
from .gates import evaluate_claim_gate
from .metrics import (
    ranked_feature_importance,
    summarize_binary_increment,
    summarize_forecast_increment,
)
from .models import (
    CrossFitConfig,
    binary_task_bootstrap_seed,
    crossfit_binary_increment,
    crossfit_forecast_increment,
    crossfit_geometry_explainability,
)
from .report import save_audit_report
from .schema import TraceArtifact


@dataclass(frozen=True)
class OCGPIAuditConfig:
    checkpoints: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    history: int = 2
    horizon: int = 1
    bootstrap: int = 1000
    include_legacy_geometry: bool = True
    compute_device: str = "cpu"
    geometry_batch_size: int = 32
    allow_model_mismatch: bool = False
    label_policy: str = "process_error"
    importance_limit: int = 30
    crossfit: CrossFitConfig = CrossFitConfig()

    def validate(self) -> None:
        if not self.checkpoints or any(
            not 0.0 < value <= 1.0 for value in self.checkpoints
        ):
            raise ValueError("checkpoints must be non-empty and lie in (0, 1]")
        if self.history < 1 or self.horizon < 1:
            raise ValueError("history and horizon must be positive")
        if self.bootstrap < 10:
            raise ValueError("bootstrap must be at least 10")
        if self.geometry_batch_size < 1:
            raise ValueError("geometry_batch_size must be positive")
        if self.label_policy not in {"process_error", "final_answer"}:
            raise ValueError("label_policy must be `process_error` or `final_answer`")
        self.crossfit.validate()


def _primary_groups(groups: Sequence[str]) -> tuple[str, ...]:
    unique = tuple(dict.fromkeys(str(group) for group in groups))
    primary = tuple(group for group in unique if group != "final_control")
    return primary or unique


def _model_identity_matches(left: str, right: str) -> bool:
    left = str(left).strip().replace("\\", "/").rstrip("/").lower()
    right = str(right).strip().replace("\\", "/").rstrip("/").lower()
    if not left or not right:
        return False
    return left == right or left.rsplit("/", 1)[-1] == right.rsplit("/", 1)[-1]


def _binary_run(
    task: BinaryTask,
    cfg: OCGPIAuditConfig,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    result = crossfit_binary_increment(
        task.x_output,
        task.x_geometry,
        task.y,
        task.groups,
        task.nuisance_values,
        cfg.crossfit,
    )
    summary = summarize_binary_increment(
        result,
        n_boot=cfg.bootstrap,
        seed=binary_task_bootstrap_seed(cfg.crossfit.seed, task.checkpoint),
    )
    summary["checkpoint"] = float(task.checkpoint)
    summary["feature_importance"] = ranked_feature_importance(
        task.geometry_names,
        task.geometry_groups,
        result.feature_importance,
        limit=cfg.importance_limit,
    )
    predictions = {
        "chain_idx": task.chain_idx,
        "problem_id": task.groups,
        "label": task.y,
        "output_only": result.base_probability,
        "controls_plus_geometry": result.geometry_probability,
        "output_plus_geometry": result.full_probability,
        "length_matched_null": result.null_probability,
    }
    if task.step_idx is not None:
        predictions["step_idx"] = task.step_idx
    return summary, predictions


def _forecast_run(
    task: ForecastTask,
    cfg: OCGPIAuditConfig,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    result = crossfit_forecast_increment(
        task.x_output,
        task.x_geometry,
        task.target,
        task.groups,
        task.nuisance_values,
        cfg.crossfit,
    )
    summary = summarize_forecast_increment(
        result,
        n_boot=cfg.bootstrap,
        seed=cfg.crossfit.seed + 101,
    )
    summary["history"] = int(cfg.history)
    summary["horizon"] = int(cfg.horizon)
    summary["target_features"] = list(task.target_names)
    summary["feature_importance"] = ranked_feature_importance(
        task.geometry_names,
        task.geometry_groups,
        result.feature_importance,
        limit=cfg.importance_limit,
    )
    predictions = {
        "chain_idx": task.chain_idx,
        "problem_id": task.groups,
        "step_idx": task.step_idx,
        "target": result.target,
        "output_only": result.base_prediction,
        "controls_plus_geometry": result.geometry_prediction,
        "output_plus_geometry": result.full_prediction,
        "length_matched_null": result.null_prediction,
        "standardized_target": result.standardized_target,
        "standardized_output_only": result.standardized_base_prediction,
        "standardized_controls_plus_geometry": result.standardized_geometry_prediction,
        "standardized_output_plus_geometry": result.standardized_full_prediction,
        "standardized_length_matched_null": result.standardized_null_prediction,
    }
    return summary, predictions


def _compact_model_summary(summary: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"feature_importance", "target_features"}
    }


def run_ocgpi_audit(
    *,
    trace_path: str | Path,
    geometry_path: str | Path,
    output_dir: str | Path,
    cfg: OCGPIAuditConfig,
) -> dict[str, object]:
    cfg.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = TraceArtifact.load(trace_path)
    geometry = load_geometry_collection(
        geometry_path,
        compute_device=cfg.compute_device,
        geometry_batch_size=cfg.geometry_batch_size,
        include_legacy_geometry=cfg.include_legacy_geometry,
    )
    trace_model = str(trace.metadata.get("model", ""))
    geometry_model = str(geometry.preflight.get("model_name", ""))
    model_match = _model_identity_matches(trace_model, geometry_model)
    if (
        trace_model
        and geometry_model
        and not model_match
        and not cfg.allow_model_mismatch
    ):
        raise ValueError(
            f"observer model mismatch: trace={trace_model!r}, geometry={geometry_model!r}; "
            "use the same checkpoint or pass the explicit unsafe-ablation flag"
        )
    collection = join_trace_and_geometry(trace, geometry, label_policy=cfg.label_policy)
    collection.preflight["trace_model"] = trace_model
    collection.preflight["geometry_model"] = geometry_model
    collection.preflight["observer_model_match"] = bool(model_match)
    primary_groups = _primary_groups(collection.geometry_groups)

    response_tasks = build_response_tasks(collection, checkpoints=cfg.checkpoints)
    response_summaries: dict[str, object] = {}
    prediction_payload: dict[str, np.ndarray] = {}
    for checkpoint, raw_task in response_tasks.items():
        task = select_geometry_groups(raw_task, primary_groups)
        summary, predictions = _binary_run(task, cfg)
        key = f"{checkpoint:.2f}"
        response_summaries[key] = summary
        for name, value in predictions.items():
            prediction_payload[f"response_{key}_{name}"] = value

    online_raw_task = build_online_response_task(collection)
    online_task = select_geometry_groups(online_raw_task, primary_groups)
    online_summary, online_predictions = _binary_run(online_task, cfg)
    for name, value in online_predictions.items():
        prediction_payload[f"online_response_{name}"] = value

    raw_forecast = build_forecast_task(
        collection,
        history=cfg.history,
        horizon=cfg.horizon,
    )
    forecast_task = select_geometry_groups(raw_forecast, primary_groups)
    forecast_summary, forecast_predictions = _forecast_run(forecast_task, cfg)
    for name, value in forecast_predictions.items():
        prediction_payload[f"forecast_{name}"] = value

    final_task = select_geometry_groups(
        response_tasks[max(cfg.checkpoints)], primary_groups
    )

    saturation_ladder: dict[str, object] = {
        "full_compact": {
            "response": _compact_model_summary(online_summary),
            "future_output": _compact_model_summary(forecast_summary),
        }
    }
    for tier in ("scalar", "distribution"):
        tier_response, _ = _binary_run(select_output_tier(online_task, tier), cfg)
        tier_forecast, _ = _forecast_run(select_output_tier(forecast_task, tier), cfg)
        saturation_ladder[tier] = {
            "response": _compact_model_summary(tier_response),
            "future_output": _compact_model_summary(tier_forecast),
        }

    explainability = crossfit_geometry_explainability(
        final_task.x_output,
        final_task.x_geometry,
        final_task.groups,
        cfg.crossfit,
        stratify_y=final_task.y,
    )
    explainability_rows = [
        {
            "feature": name,
            "group": group,
            "r2_from_output": float(value),
        }
        for name, group, value in zip(
            final_task.geometry_names,
            final_task.geometry_groups,
            explainability.feature_r2,
        )
    ]
    explainability_rows.sort(key=lambda row: float(row["r2_from_output"]), reverse=True)

    group_ablation: dict[str, object] = {}
    for group in tuple(dict.fromkeys(collection.geometry_groups)):
        response_group_task = select_geometry_groups(online_raw_task, (group,))
        forecast_group_task = select_geometry_groups(raw_forecast, (group,))
        response_group, _ = _binary_run(response_group_task, cfg)
        forecast_group, _ = _forecast_run(forecast_group_task, cfg)
        group_ablation[group] = {
            "response": response_group,
            "future_output": forecast_group,
        }

    report: dict[str, object] = {
        "method": "Output-Conditioned Geometric Predictive Information (OC-GPI)",
        "schema_version": "ocgpi_audit_v1",
        "research_question": (
            "Does internal reasoning geometry carry usable information about future output drift "
            "and response failure after conditioning on the causal output-distribution history?"
        ),
        "config": {
            **asdict(cfg),
            "crossfit": asdict(cfg.crossfit),
        },
        "preflight": collection.preflight,
        "primary_geometry_groups": list(primary_groups),
        "bidirectional_coupling": {
            "output_to_geometry_overall_r2": float(explainability.overall_r2),
            "per_feature": explainability_rows,
        },
        "response_detection": response_summaries,
        "online_response_detection": online_summary,
        "future_output_forecast": forecast_summary,
        "output_saturation_ladder": saturation_ladder,
        "geometry_group_ablation": group_ablation,
    }
    report["decision_gate"] = evaluate_claim_gate(
        online_summary,
        forecast_summary,
        collection.preflight,
    )
    np.savez_compressed(output_dir / "oof_predictions.npz", **prediction_payload)
    save_audit_report(report, output_dir)
    return report
