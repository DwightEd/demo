from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .evaluate import auprc, auroc, finite_json
from .first_error_geometry import GEOMETRY_NAMES, compute_geometry_fields
from .flow_signature_audit import (
    FlowAuditConfig,
    bootstrap_same_problem_auc_delta,
    crossfit_residualize_score,
    evaluate_score,
    score_conditional_support,
)
from .flow_signature_data import FlowTrajectoryDataset


STATIC_SUMMARY_NAMES = ("mean", "max", "std", "late_mean")


@dataclass(frozen=True)
class MultisampleGeometryConfig:
    phase_points: int = 16
    late_fraction: float = 1.0 / 3.0
    batch_size: int = 64
    compute_device: str = "cuda"
    folds: int = 5
    bootstrap: int = 1000
    permutations: int = 500
    min_correct_support: int = 2
    seed: int = 0

    def validate(self) -> None:
        if self.phase_points < 4:
            raise ValueError("phase_points must be at least 4")
        if not 0.0 < self.late_fraction <= 1.0:
            raise ValueError("late_fraction must lie in (0, 1]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass
class GeometryProfiles:
    profiles: np.ndarray
    static_features: np.ndarray
    valid_features: np.ndarray
    phase_grid: np.ndarray
    metric_names: tuple[str, ...]
    summary_names: tuple[str, ...]


def _subset_dataset(dataset: FlowTrajectoryDataset, indices: np.ndarray) -> FlowTrajectoryDataset:
    idx = np.asarray(indices, dtype=np.int64)
    return FlowTrajectoryDataset(
        source_path=dataset.source_path,
        vector_key=dataset.vector_key,
        trajectories=[dataset.trajectories[int(i)] for i in idx],
        original_indices=dataset.original_indices[idx],
        problem_ids=dataset.problem_ids[idx],
        sample_idx=dataset.sample_idx[idx],
        y_error=dataset.y_error[idx],
        is_correct=dataset.is_correct[idx],
        n_steps=dataset.n_steps[idx],
        response_chars=dataset.response_chars[idx],
        layer_ids=dataset.layer_ids.copy(),
        hidden_dim=dataset.hidden_dim,
        label_policy=dataset.label_policy,
        skipped=dict(dataset.skipped),
        metadata=dict(dataset.metadata),
    )


def _interpolate_finite(values: np.ndarray, source_phase: np.ndarray, target_phase: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(target_phase.shape, np.nan, dtype=np.float32)
    if int(np.sum(finite)) == 1:
        return np.full(target_phase.shape, float(values[finite][0]), dtype=np.float32)
    return np.interp(target_phase, source_phase[finite], values[finite]).astype(np.float32)


def build_geometry_profiles(
    fields: Sequence[np.ndarray],
    *,
    phase_points: int,
    late_fraction: float = 1.0 / 3.0,
) -> GeometryProfiles:
    """Align ragged local-geometry fields on normalized reasoning phase.

    Missing endpoint values are extended with the nearest finite observation.
    This keeps short trajectories in the response audit while the saved
    ``valid_features`` mask makes their coverage explicit.
    """

    if not fields:
        raise ValueError("no geometry fields")
    first = np.asarray(fields[0])
    if first.ndim != 3 or first.shape[2] != len(GEOMETRY_NAMES):
        raise ValueError("fields must have shape [step, layer, geometry_metric]")
    n_layers = int(first.shape[1])
    if any(np.asarray(item).ndim != 3 or np.asarray(item).shape[1:] != first.shape[1:] for item in fields):
        raise ValueError("geometry field shapes are inconsistent")

    phase_grid = np.linspace(0.0, 1.0, int(phase_points), dtype=np.float32)
    profiles = np.full(
        (len(fields), phase_grid.size, n_layers, len(GEOMETRY_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    static = np.full(
        (len(fields), n_layers, len(GEOMETRY_NAMES), len(STATIC_SUMMARY_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    valid = np.zeros((len(fields), n_layers, len(GEOMETRY_NAMES)), dtype=bool)
    late_start = 1.0 - float(late_fraction)

    for row, item in enumerate(fields):
        array = np.asarray(item, dtype=np.float32)
        source_phase = np.arange(array.shape[0], dtype=np.float32) / max(array.shape[0] - 1, 1)
        for layer in range(n_layers):
            for metric in range(len(GEOMETRY_NAMES)):
                values = array[:, layer, metric]
                finite = np.isfinite(values)
                if not np.any(finite):
                    continue
                valid[row, layer, metric] = True
                profiles[row, :, layer, metric] = _interpolate_finite(
                    values,
                    source_phase,
                    phase_grid,
                )
                observed = values[finite].astype(np.float64)
                observed_phase = source_phase[finite]
                late = observed[observed_phase >= late_start]
                if late.size == 0:
                    late = observed[-1:]
                static[row, layer, metric] = np.asarray(
                    [
                        np.mean(observed),
                        np.max(observed),
                        np.std(observed),
                        np.mean(late),
                    ],
                    dtype=np.float32,
                )
    return GeometryProfiles(
        profiles=profiles,
        static_features=static,
        valid_features=valid,
        phase_grid=phase_grid,
        metric_names=tuple(GEOMETRY_NAMES),
        summary_names=STATIC_SUMMARY_NAMES,
    )


def _scatter(values: np.ndarray, indices: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(values)
    shape = (int(size),) + array.shape[1:]
    output = np.full(shape, np.nan, dtype=np.float32)
    output[np.asarray(indices, dtype=np.int64)] = array.astype(np.float32, copy=False)
    return output


def _profile_mean(profile: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(profile, dtype=np.float64), axis=1)


def _profile_late_mean(profile: np.ndarray) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64)
    start = max(0, int(math.floor(2 * values.shape[1] / 3)))
    return np.mean(values[:, start:], axis=1)


def _conditional_scores(
    features: np.ndarray,
    valid: np.ndarray,
    dataset: FlowTrajectoryDataset,
    cfg: FlowAuditConfig,
) -> dict[str, np.ndarray]:
    indices = np.where(np.asarray(valid, dtype=bool))[0]
    empty = np.full(dataset.n_samples, np.nan, dtype=np.float32)
    if indices.size < 4:
        return {
            "global_mean": empty.copy(),
            "global_late": empty.copy(),
            "support_mean": empty.copy(),
            "support_late": empty.copy(),
        }
    subset = _subset_dataset(dataset, indices)
    scored = score_conditional_support(np.asarray(features)[indices], subset.y_error, subset.problem_ids, cfg)
    return {
        "global_mean": _scatter(_profile_mean(scored["global_profile"]), indices, dataset.n_samples),
        "global_late": _scatter(_profile_late_mean(scored["global_profile"]), indices, dataset.n_samples),
        "support_mean": _scatter(_profile_mean(scored["support_profile"]), indices, dataset.n_samples),
        "support_late": _scatter(_profile_late_mean(scored["support_profile"]), indices, dataset.n_samples),
    }


def _bh_qvalues(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    result = np.full(p.shape, np.nan, dtype=np.float64)
    finite = np.where(np.isfinite(p))[0]
    if finite.size == 0:
        return result
    order = finite[np.argsort(p[finite])]
    adjusted = np.empty(order.size, dtype=np.float64)
    running = 1.0
    for rank in range(order.size - 1, -1, -1):
        value = p[order[rank]] * order.size / (rank + 1)
        running = min(running, value)
        adjusted[rank] = min(1.0, running)
    result[order] = adjusted
    return result


def _common_finite_scores(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    common = np.isfinite(left) & np.isfinite(right)
    return (
        np.where(common, left, np.nan),
        np.where(common, right, np.nan),
        float(np.mean(common)),
    )


def _contrastive_index_groups(dataset: FlowTrajectoryDataset) -> list[tuple[Any, np.ndarray, np.ndarray]]:
    groups: list[tuple[Any, np.ndarray, np.ndarray]] = []
    for problem in np.unique(dataset.problem_ids):
        indices = np.where(dataset.problem_ids == problem)[0]
        errors = indices[dataset.y_error[indices] == 1]
        correct = indices[dataset.y_error[indices] == 0]
        if errors.size and correct.size:
            groups.append((problem, errors, correct))
    return groups


def _quick_score_row(
    name: str,
    score: np.ndarray,
    dataset: FlowTrajectoryDataset,
    groups: Sequence[tuple[Any, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    values = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(values)
    concordance = 0.0
    pairs = 0
    eligible = 0
    for _, errors, correct in groups:
        err = values[errors[np.isfinite(values[errors])]]
        cor = values[correct[np.isfinite(values[correct])]]
        if not err.size or not cor.size:
            continue
        difference = err[:, None] - cor[None, :]
        concordance += float(np.sum(difference > 0) + 0.5 * np.sum(difference == 0))
        pairs += int(difference.size)
        eligible += 1
    return {
        "name": name,
        "n": int(np.sum(finite)),
        "coverage": float(np.mean(finite)),
        "n_error": int(np.sum(finite & (dataset.y_error == 1))),
        "cross_problem_auroc": auroc(dataset.y_error, values),
        "cross_problem_auprc": auprc(dataset.y_error, values),
        "same_problem_auroc": concordance / pairs if pairs else float("nan"),
        "same_problem_pairs": pairs,
        "same_problem_problems": eligible,
        "same_problem_ci95": [float("nan"), float("nan")],
        "same_problem_permutation_p": float("nan"),
        "error_mean": (
            float(np.nanmean(values[dataset.y_error == 1]))
            if np.any(finite & (dataset.y_error == 1))
            else float("nan")
        ),
        "correct_mean": (
            float(np.nanmean(values[dataset.y_error == 0]))
            if np.any(finite & (dataset.y_error == 0))
            else float("nan")
        ),
    }


def _crossfit_residualize_matrix(
    scores: np.ndarray,
    controls: np.ndarray,
    problem_ids: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    """Residualize many exploratory scores without rebuilding group folds."""

    values = np.asarray(scores, dtype=np.float64)
    nuisance = np.asarray(controls, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("scores must have shape [sample, score]")
    output = np.full(values.shape, np.nan, dtype=np.float64)
    problems = np.unique(problem_ids).copy()
    if problems.size < 2:
        return output
    rng = np.random.default_rng(int(seed))
    rng.shuffle(problems)
    n_folds = min(max(2, int(folds)), int(problems.size))
    fold_map = {value: index % n_folds for index, value in enumerate(problems.tolist())}
    assignment = np.asarray([fold_map[value] for value in problem_ids.tolist()], dtype=np.int64)
    nuisance_finite = np.isfinite(nuisance).all(axis=1)
    for fold in range(n_folds):
        train_base = (assignment != fold) & nuisance_finite
        test_base = (assignment == fold) & nuisance_finite
        for column in range(values.shape[1]):
            train = np.where(train_base & np.isfinite(values[:, column]))[0]
            test = np.where(test_base & np.isfinite(values[:, column]))[0]
            if train.size < nuisance.shape[1] + 2 or test.size == 0:
                continue
            center = nuisance[train].mean(axis=0)
            scale = nuisance[train].std(axis=0)
            scale[scale < 1e-8] = 1.0
            x_train = np.column_stack([np.ones(train.size), (nuisance[train] - center) / scale])
            x_test = np.column_stack([np.ones(test.size), (nuisance[test] - center) / scale])
            x_train_tensor = torch.as_tensor(x_train, dtype=torch.float64)
            target_tensor = torch.as_tensor(values[train, column], dtype=torch.float64)
            gram = x_train_tensor.T @ x_train_tensor
            ridge = 1e-8 * torch.eye(gram.shape[0], dtype=torch.float64)
            beta = torch.linalg.solve(
                gram + ridge,
                x_train_tensor.T @ target_tensor,
            ).numpy()
            output[test, column] = values[test, column] - x_test @ beta
    return output


def _phase_effect_rows(
    profiles: GeometryProfiles,
    dataset: FlowTrajectoryDataset,
    groups: Sequence[tuple[Any, np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_pos, layer_id in enumerate(dataset.layer_ids.tolist()):
        for metric_pos, metric in enumerate(profiles.metric_names):
            for phase_pos, phase in enumerate(profiles.phase_grid.tolist()):
                differences: list[float] = []
                values = profiles.profiles[:, phase_pos, layer_pos, metric_pos]
                for _, error_indices, correct_indices in groups:
                    err = values[error_indices[np.isfinite(values[error_indices])]]
                    cor = values[correct_indices[np.isfinite(values[correct_indices])]]
                    if err.size and cor.size:
                        differences.append(float(np.mean(err) - np.mean(cor)))
                diff = np.asarray(differences, dtype=np.float64)
                std = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")
                rows.append(
                    {
                        "metric": metric,
                        "layer": int(layer_id),
                        "phase": float(phase),
                        "contrastive_problems": int(diff.size),
                        "paired_difference": float(np.mean(diff)) if diff.size else float("nan"),
                        "paired_effect_dz": (
                            float(np.mean(diff) / std) if np.isfinite(std) and std > 1e-12 else float("nan")
                        ),
                        "positive_fraction": float(np.mean(diff > 0)) if diff.size else float("nan"),
                    }
                )
    return rows


def run_multisample_geometry_audit(
    dataset: FlowTrajectoryDataset,
    cfg: MultisampleGeometryConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run response-level, same-problem geometry diagnostics.

    The primary score is a two-sided distance from the phase-conditioned
    correct support of the same problem. It is an oracle diagnostic, not a
    single-sample online detector. A cross-problem support score and static
    geometry summaries are reported as controls.
    """

    cfg.validate()
    started = time.perf_counter()
    fields = compute_geometry_fields(
        dataset.trajectories,
        device=cfg.compute_device,
        batch_size=cfg.batch_size,
    )
    profiles = build_geometry_profiles(
        fields,
        phase_points=cfg.phase_points,
        late_fraction=cfg.late_fraction,
    )
    audit_cfg = FlowAuditConfig(
        folds=cfg.folds,
        bootstrap=cfg.bootstrap,
        permutations=cfg.permutations,
        min_correct_support=cfg.min_correct_support,
        seed=cfg.seed,
        compute_device=cfg.compute_device,
    )
    dynamic_valid = np.all(profiles.valid_features, axis=(1, 2))
    static_valid = np.isfinite(profiles.static_features).all(axis=(1, 2, 3))
    dynamic_features = profiles.profiles.reshape(dataset.n_samples, cfg.phase_points, -1)
    static_features = profiles.static_features.reshape(dataset.n_samples, 1, -1)
    dynamic = _conditional_scores(dynamic_features, dynamic_valid, dataset, audit_cfg)
    static = _conditional_scores(static_features, static_valid, dataset, audit_cfg)

    controls = np.column_stack(
        [
            np.log1p(dataset.n_steps.astype(np.float64)),
            np.log1p(dataset.response_chars.astype(np.float64)),
        ]
    )
    score_map: dict[str, np.ndarray] = {
        "control.log1p_n_steps": controls[:, 0],
        "control.log1p_response_chars": controls[:, 1],
        "geometry.dynamic_global_mean": dynamic["global_mean"],
        "geometry.dynamic_global_late": dynamic["global_late"],
        "geometry.dynamic_support_mean": dynamic["support_mean"],
        "geometry.dynamic_support_late": dynamic["support_late"],
        "geometry.static_global": static["global_mean"],
        "geometry.static_support": static["support_mean"],
    }
    for name in tuple(score_map):
        if not name.startswith("geometry."):
            continue
        score_map[f"{name}.length_residual"] = crossfit_residualize_score(
            score_map[name], controls, dataset.problem_ids, audit_cfg
        )

    headline_order = list(score_map)
    headline = [evaluate_score(name, score_map[name], dataset, audit_cfg) for name in headline_order]
    q_values = _bh_qvalues([row["same_problem_permutation_p"] for row in headline])
    for row, q_value in zip(headline, q_values):
        row["same_problem_bh_q"] = float(q_value)

    contrastive_groups = _contrastive_index_groups(dataset)
    exploratory_names: list[str] = []
    exploratory_meta: list[tuple[str, int, str]] = []
    exploratory_values: list[np.ndarray] = []
    for layer_pos, layer_id in enumerate(dataset.layer_ids.tolist()):
        for metric_pos, metric in enumerate(profiles.metric_names):
            for summary_pos, summary in enumerate(profiles.summary_names):
                raw = profiles.static_features[:, layer_pos, metric_pos, summary_pos]
                name = f"explore.{metric}.layer_{layer_id}.{summary}"
                exploratory_names.append(name)
                exploratory_meta.append((metric, int(layer_id), summary))
                exploratory_values.append(raw)
    exploratory_matrix = np.column_stack(exploratory_values)
    exploratory_residual = _crossfit_residualize_matrix(
        exploratory_matrix,
        controls,
        dataset.problem_ids,
        folds=cfg.folds,
        seed=cfg.seed + 991,
    )
    exploratory: list[dict[str, Any]] = []
    for column, (name, meta) in enumerate(zip(exploratory_names, exploratory_meta)):
        metric, layer_id, summary = meta
        row = _quick_score_row(name, exploratory_matrix[:, column], dataset, contrastive_groups)
        row.update({"metric": metric, "layer": layer_id, "summary": summary, "variant": "raw"})
        exploratory.append(row)
        residual_name = f"{name}.length_residual"
        residual_row = _quick_score_row(
            residual_name,
            exploratory_residual[:, column],
            dataset,
            contrastive_groups,
        )
        residual_row.update(
            {"metric": metric, "layer": layer_id, "summary": summary, "variant": "length_residual"}
        )
        exploratory.append(residual_row)

    dynamic_common, static_common, delta_coverage = _common_finite_scores(
        score_map["geometry.dynamic_support_mean"],
        score_map["geometry.static_support"],
    )
    dynamic_static_delta = bootstrap_same_problem_auc_delta(
        dynamic_common,
        static_common,
        dataset,
        audit_cfg,
    )
    dynamic_residual_common, static_residual_common, residual_delta_coverage = _common_finite_scores(
        score_map["geometry.dynamic_support_mean.length_residual"],
        score_map["geometry.static_support.length_residual"],
    )
    dynamic_static_residual_delta = bootstrap_same_problem_auc_delta(
        dynamic_residual_common,
        static_residual_common,
        dataset,
        audit_cfg,
    )
    dynamic_static_delta["common_coverage"] = delta_coverage
    dynamic_static_residual_delta["common_coverage"] = residual_delta_coverage
    phase_rows = _phase_effect_rows(profiles, dataset, contrastive_groups)
    contrastive = len(contrastive_groups)
    report: dict[str, Any] = {
        "meta": {
            "source": dataset.source_path,
            "vector_key": dataset.vector_key,
            "label_policy": dataset.label_policy,
            "samples": dataset.n_samples,
            "errors": int(np.sum(dataset.y_error == 1)),
            "correct": int(np.sum(dataset.y_error == 0)),
            "problems": int(np.unique(dataset.problem_ids).size),
            "contrastive_problems": contrastive,
            "layers": dataset.layer_ids.tolist(),
            "phase_points": int(cfg.phase_points),
            "dynamic_complete_coverage": float(np.mean(dynamic_valid)),
            "static_complete_coverage": float(np.mean(static_valid)),
            "skipped": dataset.skipped,
            "runtime_seconds": time.perf_counter() - started,
            "compute_device": cfg.compute_device,
        },
        "claim_scope": {
            "primary": "response-level same-problem trajectory separability",
            "not_supported": "first-error localization; the multisample files do not contain gold_error_step",
            "support_score": "oracle diagnostic using labeled same-problem correct references",
            "global_score": "problem-held-out correct support; no target-problem labels",
        },
        "headline_scores": headline,
        "dynamic_minus_static": dynamic_static_delta,
        "dynamic_minus_static_length_residual": dynamic_static_residual_delta,
        "exploratory_scores": exploratory,
        "phase_effects": phase_rows,
    }
    packed = {
        "problem_ids": dataset.problem_ids,
        "sample_idx": dataset.sample_idx,
        "y_error": dataset.y_error,
        "is_correct": dataset.is_correct,
        "n_steps": dataset.n_steps,
        "response_chars": dataset.response_chars,
        "layer_ids": dataset.layer_ids,
        "phase_grid": profiles.phase_grid,
        "geometry_names": np.asarray(profiles.metric_names, dtype=object),
        "score_names": np.asarray(headline_order, dtype=object),
        "scores": np.column_stack([score_map[name] for name in headline_order]).astype(np.float32),
        "valid_features": profiles.valid_features,
    }
    packed["profiles"] = profiles.profiles
    return report, packed


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns and not isinstance(row[key], (list, dict)):
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    meta = report["meta"]
    lines = [
        "# Same-Problem Multisample Geometry Audit",
        "",
        f"- Samples: `{meta['samples']}` (`{meta['errors']}` errors, `{meta['correct']}` correct)",
        f"- Problems: `{meta['problems']}`; contrastive: `{meta['contrastive_problems']}`",
        f"- Layers: `{meta['layers']}`",
        f"- Dynamic complete coverage: `{meta['dynamic_complete_coverage']:.3f}`",
        "",
        "The primary support scores are diagnostic: they use labeled correct samples from the same problem. "
        "They test whether a problem-conditioned geometric trajectory is separable, not whether a single-pass "
        "online detector is deployable.",
        "",
        "## Headline Scores",
        "",
        "| score | coverage | same-problem AUROC | pairs | problems | CI95 | permutation p | BH q | cross AUROC |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in report["headline_scores"]:
        ci = row["same_problem_ci95"]
        lines.append(
            f"| {row['name']} | {_fmt(row['coverage'], 3)} | {_fmt(row['same_problem_auroc'])} | "
            f"{row['same_problem_pairs']} | {row['same_problem_problems']} | "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(row['same_problem_permutation_p'])} | "
            f"{_fmt(row['same_problem_bh_q'])} | {_fmt(row['cross_problem_auroc'])} |"
        )
    lines.extend(
        [
            "",
            "## Dynamic Increment Over Static Geometry",
            "",
            f"- Raw paired-AUROC delta: `{_fmt(report['dynamic_minus_static']['point'])}` "
            f"CI `{report['dynamic_minus_static']['ci95']}`.",
            f"- Length-residualized delta: `{_fmt(report['dynamic_minus_static_length_residual']['point'])}` "
            f"CI `{report['dynamic_minus_static_length_residual']['ci95']}`.",
            "",
            "## Strongest Exploratory Scalar Summaries",
            "",
            "These rows are exploratory and receive no confirmatory p-value. They must not replace the fixed "
            "dynamic-versus-static comparison above.",
            "",
            "| score | variant | coverage | same-problem AUROC | pairs | cross AUROC |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    exploratory = sorted(
        report["exploratory_scores"],
        key=lambda row: abs(float(row["same_problem_auroc"]) - 0.5)
        if np.isfinite(row["same_problem_auroc"])
        else -1.0,
        reverse=True,
    )
    for row in exploratory[:30]:
        lines.append(
            f"| {row['name']} | {row['variant']} | {_fmt(row['coverage'], 3)} | "
            f"{_fmt(row['same_problem_auroc'])} | {row['same_problem_pairs']} | "
            f"{_fmt(row['cross_problem_auroc'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- No `gold_error_step` is available, so this audit cannot localize the first wrong step.",
            "- A support score working while the global score fails means the geometry is problem-conditioned, "
            "not deployable from a universal healthy manifold.",
            "- Dynamic geometry is useful only if its paired-AUROC increment over static geometry has a positive CI.",
            "- Length-residualized results are required because response length varies within a problem.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_phase_heatmap(path: Path, report: Mapping[str, Any]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    rows = report["phase_effects"]
    layers = report["meta"]["layers"]
    phases = sorted({float(row["phase"]) for row in rows})
    metrics = list(GEOMETRY_NAMES)
    values = np.asarray(
        [abs(float(row["paired_effect_dz"])) for row in rows if np.isfinite(row["paired_effect_dz"])],
        dtype=np.float64,
    )
    limit = max(0.1, float(np.percentile(values, 95))) if values.size else 1.0
    figure, axes = plt.subplots(len(metrics), 1, figsize=(12, 2.7 * len(metrics)), constrained_layout=True)
    for axis, metric in zip(np.atleast_1d(axes), metrics):
        matrix = np.full((len(layers), len(phases)), np.nan, dtype=np.float64)
        selected = [row for row in rows if row["metric"] == metric]
        layer_pos = {int(value): i for i, value in enumerate(layers)}
        phase_pos = {float(value): i for i, value in enumerate(phases)}
        for row in selected:
            matrix[layer_pos[int(row["layer"])], phase_pos[float(row["phase"])]] = row["paired_effect_dz"]
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit, origin="lower")
        axis.set_title(metric)
        axis.set_ylabel("layer")
        axis.set_yticks(np.arange(len(layers)))
        axis.set_yticklabels(layers)
        ticks = np.linspace(0, len(phases) - 1, min(6, len(phases))).astype(int)
        axis.set_xticks(ticks)
        axis.set_xticklabels([f"{phases[i]:.2f}" for i in ticks])
        axis.set_xlabel("normalized reasoning phase")
        figure.colorbar(image, ax=axis, label="same-problem paired effect dz")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def write_multisample_geometry_outputs(
    report: Mapping[str, Any],
    packed: Mapping[str, np.ndarray],
    *,
    output: str | Path,
    output_dir: str | Path,
    keep_profiles: bool = False,
    render_plots: bool = True,
) -> dict[str, str]:
    output = Path(output)
    output_dir = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(packed)
    if not keep_profiles:
        payload.pop("profiles", None)
    np.savez_compressed(output, **payload)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    exploratory_path = output_dir / "exploratory_scores.csv"
    phase_path = output_dir / "phase_effects.csv"
    json_path.write_text(json.dumps(finite_json(report), indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(markdown_path, report)
    _write_csv(exploratory_path, report["exploratory_scores"])
    _write_csv(phase_path, report["phase_effects"])
    paths = {
        "scores": str(output),
        "json": str(json_path),
        "markdown": str(markdown_path),
        "exploratory_csv": str(exploratory_path),
        "phase_csv": str(phase_path),
    }
    if render_plots:
        plot_path = output_dir / "same_problem_phase_effects.png"
        if _write_phase_heatmap(plot_path, report):
            paths["phase_plot"] = str(plot_path)
    return paths
