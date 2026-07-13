from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from .conditional_tangent import LEGACY_METRIC_NAMES, ConditionalTangentResult
from .evaluate import auprc, auroc, finite_json


EPS = 1e-8
HIGH_IS_ERROR = {
    "qpt_escape_ratio": True,
    "qpt_isotropic_distance": True,
    "qpt_nearest_direction_distance": True,
    "qpt_reference_topr_energy": False,
    "qpt_reference_effective_rank": True,
    "phase_only_escape_ratio": True,
    "shuffled_question_escape_ratio": True,
    "global_escape_ratio": True,
    "random_escape_ratio": True,
    "question_conditioning_excess": True,
    "window_normal_energy": True,
    "normal_persistence": True,
    "coherent_normal_drift": True,
    "output_transverse_energy": True,
    "output_tangent_energy": True,
    "output_transverse_fraction": True,
    "output_normal_alignment": True,
    "output_tangent_alignment": True,
    "output_alignment_excess": True,
    "update_speed": True,
    "direction_spread": True,
    "direction_resultant_jl": False,
    "direction_spec_entropy_raw": True,
    "direction_spec_entropy_norm": True,
    "direction_effective_rank_norm": True,
}
PRIMARY_RESPONSE_METRICS = (
    "qpt_escape_ratio",
    "coherent_normal_drift",
    "output_transverse_energy",
    "output_normal_alignment",
)


@dataclass(frozen=True)
class ConditionalTangentValidationConfig:
    folds: int = 5
    bootstrap: int = 1000
    length_bins: int = 5
    event_offsets: tuple[int, ...] = (-2, -1, 0, 1, 2)
    min_coverage: float = 0.80
    random_seed: int = 17


@dataclass
class _FlatStepAxis:
    chain_rows: np.ndarray
    step_indices: np.ndarray
    labels: np.ndarray
    problem_ids: np.ndarray
    chain_ids: np.ndarray
    controls: np.ndarray
    step_lengths: np.ndarray
    causal_step_clock: np.ndarray


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                names.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _flat_first_error_axis(result: ConditionalTangentResult) -> _FlatStepAxis:
    source = result.dataset.source
    chain_rows: list[int] = []
    step_indices: list[int] = []
    labels: list[int] = []
    problem_ids: list[Any] = []
    chain_ids: list[int] = []
    controls: list[np.ndarray] = []
    lengths: list[float] = []
    positions: list[float] = []
    for row in range(source.n_samples):
        gold = int(source.gold_error_step[row])
        limit = source.trajectories[row].shape[0] if gold < 0 else min(
            source.trajectories[row].shape[0], gold + 1
        )
        for step in range(limit):
            chain_rows.append(row)
            step_indices.append(step)
            labels.append(int(gold >= 0 and step == gold))
            problem_ids.append(source.problem_ids[row])
            chain_ids.append(row)
            controls.append(result.axis.controls[row][step])
            lengths.append(float(source.step_lengths[row][step]))
            positions.append(float(result.axis.controls[row][step, 0]))
    return _FlatStepAxis(
        chain_rows=np.asarray(chain_rows, dtype=np.int64),
        step_indices=np.asarray(step_indices, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        problem_ids=np.asarray(problem_ids),
        chain_ids=np.asarray(chain_ids, dtype=np.int64),
        controls=np.asarray(controls, dtype=np.float64),
        step_lengths=np.asarray(lengths, dtype=np.float64),
        causal_step_clock=np.asarray(positions, dtype=np.float64),
    )


def _flatten_field(
    values: Sequence[np.ndarray],
    flat: _FlatStepAxis,
    *,
    layer: int,
    metric: int | None = None,
) -> np.ndarray:
    output = np.full(flat.labels.size, np.nan, dtype=np.float64)
    for index, (row, step) in enumerate(zip(flat.chain_rows, flat.step_indices)):
        value = values[int(row)]
        output[index] = (
            value[int(step), int(layer)]
            if metric is None
            else value[int(step), int(layer), int(metric)]
        )
    return output


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if np.sum(valid) < 3:
        return float("nan")
    x = rankdata(left[valid], method="average")
    y = rankdata(right[valid], method="average")
    if np.std(x) <= EPS or np.std(y) <= EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _orient(score: np.ndarray, metric: str) -> np.ndarray:
    direction = 1.0 if HIGH_IS_ERROR.get(metric, True) else -1.0
    return direction * np.asarray(score, dtype=np.float64)


def _length_bucket_auc(
    labels: np.ndarray,
    score: np.ndarray,
    lengths: np.ndarray,
    bins: int,
) -> tuple[float, int]:
    valid = np.isfinite(score) & np.isfinite(lengths)
    y, s, n = labels[valid], score[valid], lengths[valid]
    if y.size == 0:
        return float("nan"), 0
    edges = np.unique(np.quantile(n, np.linspace(0.0, 1.0, max(2, int(bins)) + 1)))
    if edges.size < 2:
        return auroc(y, s), 1
    assignment = np.clip(np.digitize(n, edges[1:-1]), 0, edges.size - 2)
    numerator = 0.0
    denominator = 0
    eligible = 0
    for bucket in range(edges.size - 1):
        mask = assignment == bucket
        pos = int(np.sum(y[mask] == 1))
        neg = int(np.sum(y[mask] == 0))
        value = auroc(y[mask], s[mask])
        if pos and neg and np.isfinite(value):
            pairs = pos * neg
            numerator += float(value) * pairs
            denominator += pairs
            eligible += 1
    return (numerator / denominator if denominator else float("nan")), eligible


def _grouped_oof_logit(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
) -> np.ndarray:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for grouped OOF validation") from exc
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(labels, dtype=np.int64)
    g = np.asarray(groups)
    output = np.full(y.size, np.nan, dtype=np.float64)
    unique = np.unique(g)
    if unique.size < 2:
        return output
    splitter = GroupKFold(n_splits=max(2, min(int(folds), int(unique.size))))
    for train, test in splitter.split(x, y, g):
        if np.unique(y[train]).size < 2:
            continue
        mean = np.nanmean(x[train], axis=0)
        mean[~np.isfinite(mean)] = 0.0
        x_train = np.where(np.isfinite(x[train]), x[train], mean)
        x_test = np.where(np.isfinite(x[test]), x[test], mean)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
        )
        model.fit(x_train, y[train])
        output[test] = model.predict_proba(x_test)[:, 1]
    return output


def _cluster_bootstrap_delta(
    full: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    valid = np.isfinite(full) & np.isfinite(baseline)
    full, baseline = full[valid], baseline[valid]
    labels, groups = labels[valid], groups[valid]
    point = auroc(labels, full) - auroc(labels, baseline)
    unique = np.unique(groups)
    if int(draws) <= 0 or unique.size < 2:
        return {"point": point, "ci95": [float("nan"), float("nan")], "groups": int(unique.size)}
    rng = np.random.default_rng(int(seed))
    by_group = {group: np.where(groups == group)[0] for group in unique}
    samples: list[float] = []
    for _ in range(int(draws)):
        selected = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([by_group[group] for group in selected])
        full_auc = auroc(labels[indices], full[indices])
        base_auc = auroc(labels[indices], baseline[indices])
        if np.isfinite(full_auc) and np.isfinite(base_auc):
            samples.append(float(full_auc - base_auc))
    interval = np.quantile(samples, [0.025, 0.975]) if samples else [np.nan, np.nan]
    return {
        "point": float(point),
        "ci95": [float(interval[0]), float(interval[1])],
        "groups": int(unique.size),
    }


def _bootstrap_mean(values: np.ndarray, *, draws: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if draws <= 0 or values.size == 1:
        value = float(np.mean(values))
        return value, value
    rng = np.random.default_rng(int(seed))
    index = rng.integers(0, values.size, size=(int(draws), values.size))
    means = np.mean(values[index], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _sign_flip_p(values: np.ndarray, *, permutations: int, seed: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or permutations <= 0:
        return float("nan")
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(int(seed))
    exceed = 0
    remaining = int(permutations)
    while remaining:
        take = min(remaining, 512)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(take, values.size))
        exceed += int(np.sum(np.abs(np.mean(signs * values[None, :], axis=1)) >= observed))
        remaining -= take
    return float((exceed + 1) / (int(permutations) + 1))


def _bh_qvalues(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    output = np.full(p.shape, np.nan, dtype=np.float64)
    valid = np.where(np.isfinite(p))[0]
    if valid.size == 0:
        return output
    order = valid[np.argsort(p[valid])]
    adjusted = p[order] * valid.size / np.arange(1, valid.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _one_step_metric_row(
    *,
    family: str,
    variant: str,
    metric: str,
    layer: int,
    score: np.ndarray,
    flat: _FlatStepAxis,
    baseline_oof: np.ndarray,
    cfg: ConditionalTangentValidationConfig,
    seed: int,
) -> dict[str, Any]:
    oriented = _orient(score, metric)
    valid = np.isfinite(oriented)
    raw_auc = auroc(flat.labels, score)
    bucket_auc, eligible_bins = _length_bucket_auc(
        flat.labels,
        oriented,
        flat.step_lengths,
        cfg.length_bins,
    )
    row: dict[str, Any] = {
        "family": family,
        "variant": variant,
        "metric": metric,
        "layer": int(layer),
        "n": int(np.sum(valid)),
        "positives": int(np.sum(flat.labels[valid] == 1)),
        "coverage": float(np.mean(valid)),
        "signed_auroc": auroc(flat.labels, oriented),
        "raw_high_auroc": raw_auc,
        "best_direction_auroc": max(raw_auc, 1.0 - raw_auc)
        if np.isfinite(raw_auc)
        else float("nan"),
        "length_bucket_auroc": bucket_auc,
        "eligible_length_bins": int(eligible_bins),
        "spearman_log_step_len": _rank_correlation(
            oriented, np.log1p(flat.step_lengths)
        ),
        "spearman_causal_step_clock": _rank_correlation(
            oriented, flat.causal_step_clock
        ),
    }
    if variant == "raw" and np.mean(valid) >= cfg.min_coverage:
        full = _grouped_oof_logit(
            np.column_stack([flat.controls, oriented]),
            flat.labels,
            flat.problem_ids,
            folds=cfg.folds,
        )
        row["control_oof_auroc"] = auroc(flat.labels, baseline_oof)
        row["control_plus_metric_oof_auroc"] = auroc(flat.labels, full)
        row["oof_increment"] = _cluster_bootstrap_delta(
            full,
            baseline_oof,
            flat.labels,
            flat.problem_ids,
            draws=cfg.bootstrap,
            seed=seed,
        )
    return row


def _add_step_persistence_increment(
    row: dict[str, Any],
    *,
    persistence_score: np.ndarray,
    instantaneous_escape: np.ndarray,
    flat: _FlatStepAxis,
    cfg: ConditionalTangentValidationConfig,
    seed: int,
) -> None:
    """Test memory in normal motion beyond the current escape magnitude."""

    persistence = _orient(persistence_score, "coherent_normal_drift")
    instantaneous = _orient(instantaneous_escape, "qpt_escape_ratio")
    jointly_valid = np.isfinite(persistence) & np.isfinite(instantaneous)
    row["nested_input_coverage"] = float(np.mean(jointly_valid))
    if np.mean(jointly_valid) < cfg.min_coverage:
        return
    instantaneous_oof = _grouped_oof_logit(
        np.column_stack([flat.controls, instantaneous]),
        flat.labels,
        flat.problem_ids,
        folds=cfg.folds,
    )
    full_oof = _grouped_oof_logit(
        np.column_stack([flat.controls, instantaneous, persistence]),
        flat.labels,
        flat.problem_ids,
        folds=cfg.folds,
    )
    row["instantaneous_escape_oof_auroc"] = auroc(
        flat.labels, instantaneous_oof
    )
    row["instantaneous_plus_persistence_oof_auroc"] = auroc(
        flat.labels, full_oof
    )
    row["persistence_beyond_instantaneous"] = _cluster_bootstrap_delta(
        full_oof,
        instantaneous_oof,
        flat.labels,
        flat.problem_ids,
        draws=cfg.bootstrap,
        seed=seed,
    )


def _metric_rows(
    result: ConditionalTangentResult,
    flat: _FlatStepAxis,
    cfg: ConditionalTangentValidationConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline = _grouped_oof_logit(
        flat.controls,
        flat.labels,
        flat.problem_ids,
        folds=cfg.folds,
    )
    qpt_index = result.metric_names.index("qpt_escape_ratio")
    for variant, collection in (
        ("raw", result.fields),
        ("nuisance_residual", result.residual_fields),
    ):
        for metric_index, metric in enumerate(result.metric_names):
            for layer_pos, layer in enumerate(result.axis.layer_ids):
                score = _flatten_field(
                    collection,
                    flat,
                    layer=layer_pos,
                    metric=metric_index,
                )
                row = _one_step_metric_row(
                        family="conditional_tangent",
                        variant=variant,
                        metric=metric,
                        layer=int(layer),
                        score=score,
                        flat=flat,
                        baseline_oof=baseline,
                        cfg=cfg,
                        seed=cfg.random_seed + 1000 + metric_index * 97 + layer_pos,
                    )
                if variant == "raw" and metric == "coherent_normal_drift":
                    instantaneous = _flatten_field(
                        result.fields,
                        flat,
                        layer=layer_pos,
                        metric=qpt_index,
                    )
                    _add_step_persistence_increment(
                        row,
                        persistence_score=score,
                        instantaneous_escape=instantaneous,
                        flat=flat,
                        cfg=cfg,
                        seed=cfg.random_seed + 19000 + layer_pos,
                    )
                rows.append(row)
    for metric in LEGACY_METRIC_NAMES:
        raw_values = result.legacy_fields.get(metric)
        residual_values = result.legacy_residual_fields.get(metric)
        for variant, collection in (
            ("raw", raw_values),
            ("nuisance_residual", residual_values),
        ):
            if collection is None:
                continue
            for layer_pos, layer in enumerate(result.axis.layer_ids):
                score = _flatten_field(collection, flat, layer=layer_pos)
                rows.append(
                    _one_step_metric_row(
                        family="legacy_directional",
                        variant=variant,
                        metric=metric,
                        layer=int(layer),
                        score=score,
                        flat=flat,
                        baseline_oof=baseline,
                        cfg=cfg,
                        seed=cfg.random_seed + 3000 + layer_pos,
                    )
                )
    return rows


def _event_rows(
    result: ConditionalTangentResult,
    cfg: ConditionalTangentValidationConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    collections: list[tuple[str, str, Sequence[np.ndarray], int | None]] = []
    for variant, values in (
        ("raw", result.fields),
        ("nuisance_residual", result.residual_fields),
    ):
        for metric_index, metric in enumerate(result.metric_names):
            collections.append((variant, metric, values, metric_index))
    for metric in LEGACY_METRIC_NAMES:
        raw = result.legacy_fields.get(metric)
        residual = result.legacy_residual_fields.get(metric)
        if raw is not None:
            collections.append(("raw", metric, raw, None))
        if residual is not None:
            collections.append(("nuisance_residual", metric, residual, None))

    for variant, metric, values, metric_index in collections:
        direction = 1.0 if HIGH_IS_ERROR.get(metric, True) else -1.0
        for layer_pos, layer in enumerate(result.axis.layer_ids):
            for offset in cfg.event_offsets:
                errors: list[float] = []
                controls: list[float] = []
                for match in result.matches:
                    error_step = int(match.error_step + offset)
                    control_step = int(match.control_step + offset)
                    if error_step < 0 or control_step < 0:
                        continue
                    if error_step >= values[match.error_row].shape[0]:
                        continue
                    if control_step >= values[match.control_row].shape[0]:
                        continue
                    if metric_index is None:
                        error_value = values[match.error_row][error_step, layer_pos]
                        control_value = values[match.control_row][control_step, layer_pos]
                    else:
                        error_value = values[match.error_row][
                            error_step, layer_pos, metric_index
                        ]
                        control_value = values[match.control_row][
                            control_step, layer_pos, metric_index
                        ]
                    if np.isfinite(error_value) and np.isfinite(control_value):
                        errors.append(direction * float(error_value))
                        controls.append(direction * float(control_value))
                error_array = np.asarray(errors, dtype=np.float64)
                control_array = np.asarray(controls, dtype=np.float64)
                difference = error_array - control_array
                low, high = _bootstrap_mean(
                    difference,
                    draws=cfg.bootstrap,
                    seed=cfg.random_seed + 4000 + layer_pos * 31 + int(offset),
                )
                rows.append(
                    {
                        "variant": variant,
                        "metric": metric,
                        "layer": int(layer),
                        "offset": int(offset),
                        "n_pairs": int(difference.size),
                        "pair_coverage": float(
                            difference.size / max(len(result.matches), 1)
                        ),
                        "error_mean": float(np.mean(error_array))
                        if error_array.size
                        else float("nan"),
                        "control_mean": float(np.mean(control_array))
                        if control_array.size
                        else float("nan"),
                        "paired_difference": float(np.mean(difference))
                        if difference.size
                        else float("nan"),
                        "difference_ci_low": low,
                        "difference_ci_high": high,
                        "matched_auroc": auroc(
                            np.r_[
                                np.ones(error_array.size),
                                np.zeros(control_array.size),
                            ],
                            np.r_[error_array, control_array],
                        ),
                        "sign_flip_p": _sign_flip_p(
                            difference,
                            permutations=cfg.bootstrap,
                            seed=cfg.random_seed + 5000 + layer_pos * 37,
                        )
                        if int(offset) == 0
                        else float("nan"),
                    }
                )
    offset_zero = [index for index, row in enumerate(rows) if row["offset"] == 0]
    qvalues = _bh_qvalues([rows[index]["sign_flip_p"] for index in offset_zero])
    for index, value in zip(offset_zero, qvalues):
        rows[index]["bh_q"] = float(value)
    return rows


def _aggregate(values: np.ndarray, kind: str) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    if kind == "mean":
        return float(np.mean(finite))
    if kind == "max":
        return float(np.max(finite))
    if kind == "top20_mean":
        count = max(1, int(math.ceil(0.20 * finite.size)))
        return float(np.mean(np.partition(finite, finite.size - count)[-count:]))
    if kind == "last":
        return float(finite[-1])
    raise ValueError(kind)


def _response_rows(
    result: ConditionalTangentResult,
    cfg: ConditionalTangentValidationConfig,
) -> list[dict[str, Any]]:
    source = result.dataset.source
    labels = (source.gold_error_step >= 0).astype(np.int64)
    groups = source.problem_ids
    mean_length = np.asarray(
        [np.mean(value) for value in source.step_lengths], dtype=np.float64
    )
    max_length = np.asarray(
        [np.max(value) for value in source.step_lengths], dtype=np.float64
    )
    baseline_x = np.column_stack(
        [
            np.log1p(source.n_steps.astype(np.float64)),
            np.log1p(mean_length),
            np.log1p(max_length),
        ]
    )
    baseline = _grouped_oof_logit(
        baseline_x, labels, groups, folds=cfg.folds
    )
    qpt_index = result.metric_names.index("qpt_escape_ratio")
    rows: list[dict[str, Any]] = []
    collections: list[tuple[str, str, Sequence[np.ndarray], int | None]] = []
    for variant, values in (
        ("raw", result.fields),
        ("nuisance_residual", result.residual_fields),
    ):
        for metric_index, metric in enumerate(result.metric_names):
            collections.append((variant, metric, values, metric_index))
    for metric in LEGACY_METRIC_NAMES:
        raw = result.legacy_fields.get(metric)
        residual = result.legacy_residual_fields.get(metric)
        if raw is not None:
            collections.append(("raw", metric, raw, None))
        if residual is not None:
            collections.append(("nuisance_residual", metric, residual, None))

    for variant, metric, values, metric_index in collections:
        direction = 1.0 if HIGH_IS_ERROR.get(metric, True) else -1.0
        for layer_pos, layer in enumerate(result.axis.layer_ids):
            for aggregation in ("mean", "max", "top20_mean", "last"):
                score = np.full(source.n_samples, np.nan, dtype=np.float64)
                for row in range(source.n_samples):
                    item = (
                        values[row][:, layer_pos]
                        if metric_index is None
                        else values[row][:, layer_pos, metric_index]
                    )
                    score[row] = _aggregate(
                        direction * np.asarray(item, dtype=np.float64),
                        aggregation,
                    )
                valid = np.isfinite(score)
                record: dict[str, Any] = {
                    "variant": variant,
                    "metric": metric,
                    "layer": int(layer),
                    "aggregation": aggregation,
                    "n": int(np.sum(valid)),
                    "errors": int(np.sum(labels[valid] == 1)),
                    "coverage": float(np.mean(valid)),
                    "auroc": auroc(labels, score),
                    "auprc": auprc(labels, score),
                    "spearman_num_steps": _rank_correlation(score, source.n_steps),
                    "spearman_mean_step_len": _rank_correlation(
                        score, mean_length
                    ),
                }
                if (
                    metric in PRIMARY_RESPONSE_METRICS
                    and aggregation in {"mean", "max", "top20_mean"}
                    and np.mean(valid) >= cfg.min_coverage
                ):
                    full = _grouped_oof_logit(
                        np.column_stack([baseline_x, score]),
                        labels,
                        groups,
                        folds=cfg.folds,
                    )
                    record["control_oof_auroc"] = auroc(labels, baseline)
                    record["control_plus_metric_oof_auroc"] = auroc(labels, full)
                    record["oof_increment"] = _cluster_bootstrap_delta(
                        full,
                        baseline,
                        labels,
                        groups,
                        draws=cfg.bootstrap,
                        seed=cfg.random_seed + 7000 + layer_pos,
                    )
                if (
                    variant == "raw"
                    and metric == "coherent_normal_drift"
                    and aggregation in {"mean", "max", "top20_mean"}
                ):
                    instantaneous = np.full(
                        source.n_samples, np.nan, dtype=np.float64
                    )
                    for row in range(source.n_samples):
                        item = result.fields[row][
                            :, layer_pos, qpt_index
                        ]
                        instantaneous[row] = _aggregate(
                            _orient(item, "qpt_escape_ratio"),
                            aggregation,
                        )
                    jointly_valid = np.isfinite(score) & np.isfinite(instantaneous)
                    record["nested_input_coverage"] = float(
                        np.mean(jointly_valid)
                    )
                    if np.mean(jointly_valid) >= cfg.min_coverage:
                        instantaneous_oof = _grouped_oof_logit(
                            np.column_stack([baseline_x, instantaneous]),
                            labels,
                            groups,
                            folds=cfg.folds,
                        )
                        full_oof = _grouped_oof_logit(
                            np.column_stack(
                                [baseline_x, instantaneous, score]
                            ),
                            labels,
                            groups,
                            folds=cfg.folds,
                        )
                        record["instantaneous_escape_oof_auroc"] = auroc(
                            labels, instantaneous_oof
                        )
                        record[
                            "instantaneous_plus_persistence_oof_auroc"
                        ] = auroc(labels, full_oof)
                        record[
                            "persistence_beyond_instantaneous"
                        ] = _cluster_bootstrap_delta(
                            full_oof,
                            instantaneous_oof,
                            labels,
                            groups,
                            draws=cfg.bootstrap,
                            seed=cfg.random_seed + 17000 + layer_pos,
                        )
                if (
                    variant == "raw"
                    and metric
                    in {"output_transverse_energy", "output_normal_alignment"}
                    and aggregation in {"mean", "max", "top20_mean"}
                ):
                    instantaneous = np.full(
                        source.n_samples, np.nan, dtype=np.float64
                    )
                    for row in range(source.n_samples):
                        item = result.fields[row][:, layer_pos, qpt_index]
                        instantaneous[row] = _aggregate(
                            _orient(item, "qpt_escape_ratio"),
                            aggregation,
                        )
                    jointly_valid = np.isfinite(score) & np.isfinite(instantaneous)
                    record["output_nested_input_coverage"] = float(
                        np.mean(jointly_valid)
                    )
                    if np.mean(jointly_valid) >= cfg.min_coverage:
                        escape_oof = _grouped_oof_logit(
                            np.column_stack([baseline_x, instantaneous]),
                            labels,
                            groups,
                            folds=cfg.folds,
                        )
                        full_oof = _grouped_oof_logit(
                            np.column_stack(
                                [baseline_x, instantaneous, score]
                            ),
                            labels,
                            groups,
                            folds=cfg.folds,
                        )
                        record["escape_only_oof_auroc"] = auroc(
                            labels, escape_oof
                        )
                        record["escape_plus_output_oof_auroc"] = auroc(
                            labels, full_oof
                        )
                        record["output_beyond_escape"] = _cluster_bootstrap_delta(
                            full_oof,
                            escape_oof,
                            labels,
                            groups,
                            draws=cfg.bootstrap,
                            seed=cfg.random_seed + 23000 + layer_pos,
                        )
                rows.append(record)
    return rows


def _rank_rows(result: ConditionalTangentResult) -> list[dict[str, Any]]:
    source = result.dataset.source
    rows: list[dict[str, Any]] = []
    collections: list[tuple[str, Sequence[np.ndarray], int | None]] = [
        (metric, result.fields, index)
        for index, metric in enumerate(result.metric_names)
    ]
    collections.extend(
        (metric, values, None) for metric, values in result.legacy_fields.items()
    )
    for metric, values, metric_index in collections:
        direction = 1.0 if HIGH_IS_ERROR.get(metric, True) else -1.0
        for layer_pos, layer in enumerate(result.axis.layer_ids):
            ranks: list[float] = []
            top1: list[float] = []
            candidate_counts: list[int] = []
            eligible = 0
            for row, gold_value in enumerate(source.gold_error_step):
                gold = int(gold_value)
                if gold < 0 or gold >= values[row].shape[0]:
                    continue
                eligible += 1
                score = (
                    values[row][: gold + 1, layer_pos]
                    if metric_index is None
                    else values[row][: gold + 1, layer_pos, metric_index]
                )
                score = direction * np.asarray(score, dtype=np.float64)
                if not np.isfinite(score[gold]):
                    continue
                finite = score[np.isfinite(score)]
                greater = int(np.sum(finite > score[gold]))
                equal = int(np.sum(finite == score[gold]))
                ranks.append(1.0 + greater + 0.5 * max(equal - 1, 0))
                top1.append(1.0 / equal if greater == 0 and equal else 0.0)
                candidate_counts.append(int(finite.size))
            rows.append(
                {
                    "metric": metric,
                    "layer": int(layer),
                    "n": int(len(ranks)),
                    "eligible_chains": int(eligible),
                    "coverage": float(len(ranks) / max(eligible, 1)),
                    "top1": float(np.mean(top1)) if top1 else float("nan"),
                    "mean_rank": float(np.mean(ranks))
                    if ranks
                    else float("nan"),
                    "mean_candidates": float(np.mean(candidate_counts))
                    if candidate_counts
                    else float("nan"),
                }
            )
    return rows


def _hypothesis_gates(
    result: ConditionalTangentResult,
    metric_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
    cfg: ConditionalTangentValidationConfig,
) -> dict[str, Any]:
    source = result.dataset.source
    primary_index = result.metric_names.index("qpt_escape_ratio")
    shuffle_index = result.metric_names.index("shuffled_question_escape_ratio")
    phase_index = result.metric_names.index("phase_only_escape_ratio")
    shuffle_gain: list[float] = []
    phase_gain: list[float] = []
    for row in range(source.n_samples):
        if int(source.gold_error_step[row]) >= 0:
            continue
        primary = result.fields[row][:, :, primary_index]
        shuffled = result.fields[row][:, :, shuffle_index]
        phase = result.fields[row][:, :, phase_index]
        difference = shuffled - primary
        finite = difference[np.isfinite(difference)]
        if finite.size:
            shuffle_gain.append(float(np.mean(finite)))
        difference = phase - primary
        finite = difference[np.isfinite(difference)]
        if finite.size:
            phase_gain.append(float(np.mean(finite)))
    shuffle_array = np.asarray(shuffle_gain, dtype=np.float64)
    phase_array = np.asarray(phase_gain, dtype=np.float64)
    shuffle_ci = _bootstrap_mean(
        shuffle_array, draws=cfg.bootstrap, seed=cfg.random_seed + 8001
    )
    phase_ci = _bootstrap_mean(
        phase_array, draws=cfg.bootstrap, seed=cfg.random_seed + 8002
    )

    primary_events = [
        row
        for row in event_rows
        if row["variant"] == "nuisance_residual"
        and row["metric"] == "qpt_escape_ratio"
        and int(row["offset"]) == 0
        and float(row["pair_coverage"]) >= cfg.min_coverage
    ]
    gate_qvalues = _bh_qvalues(
        [float(row.get("sign_flip_p", np.nan)) for row in primary_events]
    )
    for row, value in zip(primary_events, gate_qvalues):
        row["gate_bh_q_across_layers"] = float(value)
    best_event = max(
        primary_events,
        key=lambda row: _finite(row["paired_difference"]),
        default=None,
    )
    persistence_rows = [
        row
        for row in metric_rows
        if row["variant"] == "raw"
        and row["metric"] == "coherent_normal_drift"
        and isinstance(row.get("persistence_beyond_instantaneous"), Mapping)
    ]
    best_persistence = max(
        persistence_rows,
        key=lambda row: _finite(
            row["persistence_beyond_instantaneous"].get("point")
        ),
        default=None,
    )
    persistence_response_rows = [
        row
        for row in response_rows
        if row["variant"] == "raw"
        and row["metric"] == "coherent_normal_drift"
        and isinstance(row.get("persistence_beyond_instantaneous"), Mapping)
    ]
    best_persistence_response = max(
        persistence_response_rows,
        key=lambda row: _finite(
            row["persistence_beyond_instantaneous"].get("point")
        ),
        default=None,
    )
    exact_kinds = {
        "pullback_fisher_cotangent",
        "downstream_logit_margin_gradient",
        "exact_downstream_cotangent",
    }
    output_kind = result.dataset.output_cotangent_kind
    output_event_rows = [
        row
        for row in event_rows
        if row["variant"] == "nuisance_residual"
        and row["metric"] == "output_normal_alignment"
        and int(row["offset"]) == 0
        and float(row["pair_coverage"]) >= cfg.min_coverage
    ]
    output_qvalues = _bh_qvalues(
        [float(row.get("sign_flip_p", np.nan)) for row in output_event_rows]
    )
    for row, value in zip(output_event_rows, output_qvalues):
        row["output_gate_bh_q_across_layers"] = float(value)
    best_output_event = max(
        output_event_rows,
        key=lambda row: _finite(row["paired_difference"]),
        default=None,
    )
    output_rows = [
        row
        for row in response_rows
        if row["variant"] == "raw"
        and row["metric"] == "output_normal_alignment"
        and isinstance(row.get("output_beyond_escape"), Mapping)
    ]
    best_output = max(
        output_rows,
        key=lambda row: _finite(row["output_beyond_escape"].get("point")),
        default=None,
    )
    return {
        "question_conditioning": {
            "shuffle_minus_primary_mean": float(np.mean(shuffle_array))
            if shuffle_array.size
            else float("nan"),
            "shuffle_minus_primary_ci95": list(shuffle_ci),
            "phase_only_minus_primary_mean": float(np.mean(phase_array))
            if phase_array.size
            else float("nan"),
            "phase_only_minus_primary_ci95": list(phase_ci),
            "correct_chains": int(shuffle_array.size),
            "unit_of_analysis": "one equal-weight mean per held-out correct chain",
            "pass": bool(shuffle_ci[0] > 0 and phase_ci[0] > 0),
        },
        "first_error_escape": {
            "best_layer_row": best_event,
            "pass": bool(
                best_event is not None
                and float(best_event["difference_ci_low"]) > 0
                and float(best_event.get("gate_bh_q_across_layers", 1.0)) < 0.05
            ),
        },
        "persistence_increment": {
            "best_step_layer_row": best_persistence,
            "best_response_row": best_persistence_response,
            "comparison": (
                "controls + instantaneous qpt escape versus controls + "
                "instantaneous qpt escape + coherent normal drift"
            ),
            "pass": bool(
                best_persistence_response is not None
                and float(
                    best_persistence_response[
                        "persistence_beyond_instantaneous"
                    ]["ci95"][0]
                )
                > 0
            ),
        },
        "output_sensitivity": {
            "cotangent_available": bool(result.dataset.output_cotangents is not None),
            "cotangent_kind": output_kind,
            "kind_is_exact": bool(output_kind in exact_kinds),
            "best_first_error_event": best_output_event,
            "best_response_row": best_output,
            "comparison": (
                "controls + instantaneous qpt escape versus controls + "
                "instantaneous qpt escape + output-normal alignment"
            ),
            "status": (
                "not_tested_missing_cotangent"
                if result.dataset.output_cotangents is None
                else "diagnostic_only_unverified_cotangent_kind"
                if output_kind not in exact_kinds
                else "tested_exact_cotangent"
            ),
            "pass": bool(
                output_kind in exact_kinds
                and best_output_event is not None
                and float(best_output_event["difference_ci_low"]) > 0
                and float(
                    best_output_event.get(
                        "output_gate_bh_q_across_layers", 1.0
                    )
                )
                < 0.05
                and best_output is not None
                and float(best_output["output_beyond_escape"]["ci95"][0]) > 0
            ),
        },
    }


def build_validation_summary(
    result: ConditionalTangentResult,
    cfg: ConditionalTangentValidationConfig,
) -> dict[str, Any]:
    flat = _flat_first_error_axis(result)
    metric_rows = _metric_rows(result, flat, cfg)
    event_rows = _event_rows(result, cfg)
    response_rows = _response_rows(result, cfg)
    rank_rows = _rank_rows(result)
    source = result.dataset.source
    return {
        "method": "question_conditioned_feasible_tangent_escape",
        "metadata": {
            **result.metadata,
            **result.dataset.metadata,
            "source": source.source_path,
            "chains": source.n_samples,
            "error_chains": int(np.sum(source.gold_error_step >= 0)),
            "correct_chains": int(np.sum(source.gold_error_step < 0)),
            "first_error_rows": int(flat.labels.size),
            "first_error_positives": int(np.sum(flat.labels)),
            "layers": result.axis.layer_ids.tolist(),
            "validation_folds": int(cfg.folds),
            "validation_bootstrap": int(cfg.bootstrap),
            "length_bins": int(cfg.length_bins),
        },
        "hypothesis_gates": _hypothesis_gates(
            result, metric_rows, event_rows, response_rows, cfg
        ),
        "step_metrics": metric_rows,
        "event_metrics": event_rows,
        "response_metrics": response_rows,
        "first_error_ranks": rank_rows,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    number = _finite(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def _increment_text(row: Mapping[str, Any]) -> str:
    increment = row.get("oof_increment")
    if not isinstance(increment, Mapping):
        return "NA"
    interval = increment.get("ci95", [np.nan, np.nan])
    return (
        f"{_fmt(increment.get('point'))} "
        f"[{_fmt(interval[0])}, {_fmt(interval[1])}]"
    )


def _nested_increment_text(row: Mapping[str, Any]) -> str:
    increment = row.get("persistence_beyond_instantaneous")
    if not isinstance(increment, Mapping):
        return "NA"
    interval = increment.get("ci95", [np.nan, np.nan])
    return (
        f"{_fmt(increment.get('point'))} "
        f"[{_fmt(interval[0])}, {_fmt(interval[1])}]"
    )


def _output_increment_text(row: Mapping[str, Any]) -> str:
    increment = row.get("output_beyond_escape")
    if not isinstance(increment, Mapping):
        return "NA"
    interval = increment.get("ci95", [np.nan, np.nan])
    return (
        f"{_fmt(increment.get('point'))} "
        f"[{_fmt(interval[0])}, {_fmt(interval[1])}]"
    )


def render_markdown(
    summary: Mapping[str, Any],
    cfg: ConditionalTangentValidationConfig,
) -> str:
    meta = summary["metadata"]
    gates = summary["hypothesis_gates"]
    lines = [
        "# Question-Conditioned Feasible-Tangent Escape Audit",
        "",
        "## Core Hypothesis",
        "",
        r"For normalized update $\Delta z_t^{(\ell)}$ and a group-held-out, question/phase-conditioned feasible transition space $T_{q,\tau}^{(\ell)}$,",
        "",
        r"$$E_{\perp,t}^{(\ell)}=\left\|\left(I-P_{T_{q,\tau}^{(\ell)}}\right)\Delta z_t^{(\ell)}\right\|_2^2.$$",
        "",
        r"The strong claim additionally requires output sensitivity. The audit separates escape magnitude from $A_{\perp\to y}=\langle \hat g,\widehat{\Delta z_\perp}\rangle^2$ and tests whether this normal alignment adds signal beyond instantaneous escape.",
        "",
        f"- Source: {meta['source']}",
        f"- Chains: {meta['chains']} ({meta['error_chains']} error, {meta['correct_chains']} correct)",
        f"- Layers: {meta['layers']}",
        f"- Reference policy: {meta['reference_policy']}",
        f"- First-error rows: {meta['first_error_rows']} ({meta['first_error_positives']} positives; post-error steps excluded)",
        "",
        "## Pre-Registered Gates",
        "",
        "| gate | pass | decisive statistic |",
        "|---|---:|---|",
    ]
    gate = gates["question_conditioning"]
    lines.append(
        f"| question-conditioned reference | {gate['pass']} | "
        f"shuffle-primary {_fmt(gate['shuffle_minus_primary_mean'])} "
        f"[{_fmt(gate['shuffle_minus_primary_ci95'][0])}, "
        f"{_fmt(gate['shuffle_minus_primary_ci95'][1])}] |"
    )
    gate = gates["first_error_escape"]
    row = gate.get("best_layer_row") or {}
    lines.append(
        f"| matched first-error escape | {gate['pass']} | L{row.get('layer', 'NA')} "
        f"difference {_fmt(row.get('paired_difference'))} "
        f"[{_fmt(row.get('difference_ci_low'))}, {_fmt(row.get('difference_ci_high'))}], "
        f"layer-q {_fmt(row.get('gate_bh_q_across_layers'))} |"
    )
    gate = gates["persistence_increment"]
    row = gate.get("best_response_row") or {}
    lines.append(
        f"| persistent normal drift increment | {gate['pass']} | "
        f"L{row.get('layer', 'NA')} {row.get('aggregation', 'NA')} "
        f"{_nested_increment_text(row)} beyond instantaneous escape |"
    )
    gate = gates["output_sensitivity"]
    output_event = gate.get("best_first_error_event") or {}
    output_response = gate.get("best_response_row") or {}
    lines.append(
        f"| output-sensitive transverse escape | {gate['pass']} | "
        f"{gate['status']} ({gate.get('cotangent_kind')}), "
        f"event d={_fmt(output_event.get('paired_difference'))}, "
        f"q={_fmt(output_event.get('output_gate_bh_q_across_layers'))}; "
        f"response increment {_output_increment_text(output_response)} |"
    )

    event_rows = [
        row
        for row in summary["event_metrics"]
        if row["variant"] == "nuisance_residual"
        and int(row["offset"]) == 0
        and float(row["pair_coverage"]) >= cfg.min_coverage
    ]
    event_rows.sort(key=lambda row: _finite(row["paired_difference"]), reverse=True)
    lines.extend(
        [
            "",
            "## Matched First-Error Event",
            "",
            "| metric | layer | pairs | coverage | error-control | 95% CI | AUROC | q |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in event_rows[:24]:
        lines.append(
            f"| {row['metric']} | {row['layer']} | {row['n_pairs']} | "
            f"{_fmt(row['pair_coverage'])} | {_fmt(row['paired_difference'])} | "
            f"[{_fmt(row['difference_ci_low'])}, {_fmt(row['difference_ci_high'])}] | "
            f"{_fmt(row['matched_auroc'])} | {_fmt(row.get('bh_q'))} |"
        )

    geometry = [
        row
        for row in summary["step_metrics"]
        if row["family"] == "conditional_tangent"
        and row["variant"] == "raw"
        and row["metric"]
        in {
            "qpt_escape_ratio",
            "coherent_normal_drift",
            "output_transverse_energy",
            "output_normal_alignment",
        }
    ]
    geometry.sort(key=lambda row: _finite(row["signed_auroc"]), reverse=True)
    lines.extend(
        [
            "",
            "## Conditional-Tangent Step Diagnosis",
            "",
            "| metric | layer | coverage | signed AUC | length-bucket AUC | rho(log length) | controls delta | persistence delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in geometry:
        lines.append(
            f"| {row['metric']} | {row['layer']} | {_fmt(row['coverage'])} | "
            f"{_fmt(row['signed_auroc'])} | {_fmt(row['length_bucket_auroc'])} | "
            f"{_fmt(row['spearman_log_step_len'])} | {_increment_text(row)} | "
            f"{_nested_increment_text(row)} |"
        )

    legacy = [
        row
        for row in summary["step_metrics"]
        if row["family"] == "legacy_directional" and row["variant"] == "raw"
    ]
    legacy.sort(key=lambda row: _finite(row["signed_auroc"]), reverse=True)
    lines.extend(
        [
            "",
            "## Length Audit of Earlier Directional Signals",
            "",
            "Signed AUROC uses the pre-registered error direction. Best-direction AUROC is descriptive only.",
            "",
            "| metric | layer | coverage | signed AUC | best-dir AUC | length-bucket AUC | rho(log length) | strict OOF delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in legacy[:30]:
        lines.append(
            f"| {row['metric']} | {row['layer']} | {_fmt(row['coverage'])} | "
            f"{_fmt(row['signed_auroc'])} | {_fmt(row['best_direction_auroc'])} | "
            f"{_fmt(row['length_bucket_auroc'])} | "
            f"{_fmt(row['spearman_log_step_len'])} | {_increment_text(row)} |"
        )

    persistence_response = [
        row
        for row in summary["response_metrics"]
        if row["variant"] == "raw"
        and row["metric"] == "coherent_normal_drift"
        and isinstance(row.get("persistence_beyond_instantaneous"), Mapping)
    ]
    persistence_response.sort(
        key=lambda row: _finite(
            row["persistence_beyond_instantaneous"].get("point")
        ),
        reverse=True,
    )
    lines.extend(
        [
            "",
            "## Persistence Beyond Instantaneous Escape",
            "",
            "| layer | aggregation | coverage | instantaneous OOF AUC | + persistence OOF AUC | nested delta |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in persistence_response:
        lines.append(
            f"| {row['layer']} | {row['aggregation']} | "
            f"{_fmt(row.get('nested_input_coverage'))} | "
            f"{_fmt(row.get('instantaneous_escape_oof_auroc'))} | "
            f"{_fmt(row.get('instantaneous_plus_persistence_oof_auroc'))} | "
            f"{_nested_increment_text(row)} |"
        )

    responses = [
        row
        for row in summary["response_metrics"]
        if row["variant"] == "nuisance_residual"
        and float(row["coverage"]) >= cfg.min_coverage
    ]
    responses.sort(key=lambda row: _finite(row["auroc"]), reverse=True)
    lines.extend(
        [
            "",
            "## Response Diagnosis",
            "",
            "| metric | layer | aggregation | coverage | AUROC | AUPRC | rho(n steps) | strict OOF delta |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in responses[:30]:
        lines.append(
            f"| {row['metric']} | {row['layer']} | {row['aggregation']} | "
            f"{_fmt(row['coverage'])} | {_fmt(row['auroc'])} | "
            f"{_fmt(row['auprc'])} | {_fmt(row['spearman_num_steps'])} | "
            f"{_increment_text(row)} |"
        )

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- PCA/SVD is not the contribution; it is only the estimator of a group-held-out conditional transition space.",
            "- First-error localization is secondary. Matched events, response diagnosis, structural nulls, and strict length-controlled increments are primary.",
            "- Missing cotangents leave the output-sensitive clause untested. Entropy, NLL, gradient norm, or raw unembedding overlap are not substitutes.",
            "- Causal interpretation still requires an intervention along the learned normal direction.",
        ]
    )
    return "\n".join(lines) + "\n"


def _plot_event_heatmaps(summary: Mapping[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = (
        "qpt_escape_ratio",
        "coherent_normal_drift",
        "output_transverse_energy",
        "direction_spread",
        "direction_spec_entropy_norm",
    )
    rows = summary["event_metrics"]
    layers = [int(value) for value in summary["metadata"]["layers"]]
    offsets = sorted({int(row["offset"]) for row in rows})
    for metric in candidates:
        selected = [
            row
            for row in rows
            if row["variant"] == "nuisance_residual" and row["metric"] == metric
        ]
        if not selected:
            continue
        lookup = {
            (int(row["layer"]), int(row["offset"])): float(
                row["paired_difference"]
            )
            for row in selected
        }
        matrix = np.asarray(
            [
                [lookup.get((layer, offset), np.nan) for offset in offsets]
                for layer in layers
            ],
            dtype=np.float64,
        )
        finite = np.abs(matrix[np.isfinite(matrix)])
        bound = max(
            float(np.quantile(finite, 0.95)) if finite.size else 1.0,
            1e-3,
        )
        fig, ax = plt.subplots(
            figsize=(
                max(7.5, 0.7 * len(offsets)),
                max(3.5, 0.42 * len(layers)),
            )
        )
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
        )
        ax.set_xticks(np.arange(len(offsets)), labels=offsets)
        ax.set_yticks(np.arange(len(layers)), labels=layers)
        ax.set_xlabel("offset from first error")
        ax.set_ylabel("layer")
        ax.set_title(metric)
        if 0 in offsets:
            ax.axvline(
                offsets.index(0),
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
        fig.colorbar(image, ax=ax, label="paired error-control residual")
        fig.tight_layout()
        fig.savefig(output_dir / f"event_{metric}.png", dpi=180)
        plt.close(fig)


def save_result_npz(
    result: ConditionalTangentResult,
    path: str | Path,
    *,
    include_normal_vectors: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "original_indices": result.axis.original_indices,
        "problem_ids": result.axis.problem_ids,
        "is_correct": result.axis.is_correct,
        "gold_error_step": result.axis.event_indices,
        "layer_ids": result.axis.layer_ids,
        "metric_names": np.asarray(result.metric_names, dtype=object),
        "fields": np.asarray(result.fields, dtype=object),
        "residual_fields": np.asarray(result.residual_fields, dtype=object),
        "legacy_metric_names": np.asarray(
            list(result.legacy_fields), dtype=object
        ),
        "metadata_json": np.asarray(
            json.dumps(
                finite_json(
                    {**result.metadata, **result.dataset.metadata}
                ),
                ensure_ascii=True,
            )
        ),
    }
    for name, values in result.legacy_fields.items():
        payload[f"legacy_{name}"] = np.asarray(values, dtype=object)
    for name, values in result.legacy_residual_fields.items():
        payload[f"legacy_residual_{name}"] = np.asarray(values, dtype=object)
    if include_normal_vectors:
        payload["normal_vectors"] = np.asarray(
            result.normal_vectors, dtype=object
        )
        payload["normal_vector_semantics"] = np.asarray(
            "normalized_update_minus_question_phase_tangent_projection",
            dtype=object,
        )
    np.savez_compressed(path, **payload)


def write_validation_report(
    result: ConditionalTangentResult,
    output_dir: str | Path,
    cfg: ConditionalTangentValidationConfig,
    *,
    render_plots: bool = True,
    score_output: str | Path | None = None,
    include_normal_vectors: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_validation_summary(result, cfg)
    summary_json = output_dir / "conditional_tangent_summary.json"
    summary_md = output_dir / "conditional_tangent_summary.md"
    step_csv = output_dir / "step_length_audit.csv"
    event_csv = output_dir / "first_error_event_curves.csv"
    response_csv = output_dir / "response_diagnosis.csv"
    rank_csv = output_dir / "first_error_ranks.csv"
    summary_json.write_text(
        json.dumps(finite_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(render_markdown(summary, cfg), encoding="utf-8")
    _write_rows(step_csv, summary["step_metrics"])
    _write_rows(event_csv, summary["event_metrics"])
    _write_rows(response_csv, summary["response_metrics"])
    _write_rows(rank_csv, summary["first_error_ranks"])
    if render_plots:
        _plot_event_heatmaps(summary, output_dir)
    paths = {
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "step_csv": str(step_csv),
        "event_csv": str(event_csv),
        "response_csv": str(response_csv),
        "rank_csv": str(rank_csv),
    }
    if score_output:
        save_result_npz(
            result,
            score_output,
            include_normal_vectors=include_normal_vectors,
        )
        paths["score_npz"] = str(score_output)
    return summary, paths
