from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .ghost import GhostMahalanobisEnsemble, GhostTraceEmbedding


@dataclass(frozen=True)
class EvaluationConfig:
    split_seed: int = 17
    validation_ratio: float = 0.2
    test_ratio: float = 0.2
    threshold_quantile: float = 0.95
    shrinkage: float = 0.1
    regularization: float = 1e-8
    bootstrap_replicates: int = 1000
    bootstrap_confidence: float = 0.95

    def validate(self) -> None:
        if (
            isinstance(self.split_seed, bool)
            or not isinstance(self.split_seed, Integral)
            or int(self.split_seed) < 0
        ):
            raise ValueError("split_seed must be a non-negative integer")
        if (
            not 0.0 < float(self.validation_ratio) < 1.0
            or not 0.0 < float(self.test_ratio) < 1.0
            or float(self.validation_ratio) + float(self.test_ratio) >= 1.0
        ):
            raise ValueError("validation/test ratios are invalid")
        for value, name in (
            (self.threshold_quantile, "threshold_quantile"),
            (self.bootstrap_confidence, "bootstrap_confidence"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
                or not 0.0 < float(value) < 1.0
            ):
                raise ValueError(f"{name} must lie strictly inside (0, 1)")
        if (
            isinstance(self.bootstrap_replicates, bool)
            or not isinstance(self.bootstrap_replicates, Integral)
            or int(self.bootstrap_replicates) < 1
        ):
            raise ValueError("bootstrap_replicates must be positive")


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    positives, negatives = int(positive.sum()), int((~positive).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    rank_sum = float(ranks[positive].sum())
    return float(
        (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    positive_count = int(labels.sum())
    if not positive_count or positive_count == len(labels):
        return None
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = np.asarray(scores)[order]
    true_positive = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < len(labels):
        stop = start + 1
        while stop < len(labels) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        true_positive += int(sorted_labels[start:stop].sum())
        recall = true_positive / positive_count
        precision = true_positive / stop
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(average_precision)


def _detection_metrics(
    labels: np.ndarray,
    ranking_scores: np.ndarray,
    threshold_scores: np.ndarray,
    *,
    threshold: float,
    calibration_count: int,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    ranking_scores = np.asarray(ranking_scores, dtype=np.float64)
    threshold_scores = np.asarray(threshold_scores, dtype=np.float64)
    if (
        labels.ndim != 1
        or ranking_scores.shape != labels.shape
        or threshold_scores.shape != labels.shape
        or not len(labels)
        or not np.isin(labels, [0, 1]).all()
        or not np.isfinite(ranking_scores).all()
        or not np.isfinite(threshold_scores).all()
        or np.any((threshold_scores < 0.0) | (threshold_scores > 1.0))
    ):
        raise ValueError("binary labels and aligned finite anomaly scores are required")
    if calibration_count < 1:
        raise ValueError("calibration_count must be positive")
    predicted = threshold_scores >= threshold
    positive = labels == 1
    tp = int((predicted & positive).sum())
    tn = int((~predicted & ~positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    roc_tpr, roc_fpr = _max_tpr_at_fpr(
        labels,
        ranking_scores,
        maximum_fpr=0.05,
    )
    cutoff_rank = int(np.ceil(threshold * calibration_count))
    grid_implied_tail = (
        calibration_count - cutoff_rank + 1
    ) / (calibration_count + 1)
    minimum_grid_tail = 1.0 / (calibration_count + 1)
    return {
        "n": int(len(labels)),
        "positives": int(positive.sum()),
        "prevalence": float(positive.mean()),
        "ranking_score_semantics": "unquantized_or_pre_final_ecdf_anomaly_score",
        "threshold_score_semantics": "normal_calibration_empirical_anomaly_percentile",
        "auroc": _rank_auc(labels, ranking_scores),
        "aupr": _average_precision(labels, ranking_scores),
        "threshold": float(threshold),
        "calibration_count": int(calibration_count),
        "calibration_percentile_grid_step": float(1.0 / calibration_count),
        "minimum_rank_grid_tail_rate": float(minimum_grid_tail),
        "grid_implied_tail_rate_at_percentile_cutoff": float(grid_implied_tail),
        "nominal_tail_rate": float(1.0 - threshold),
        "threshold_metrics_scope": (
            "finite_sample_diagnostic_not_an_fpr_guarantee"
        ),
        "accuracy": float(np.mean(predicted == positive)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "mcc": float(mcc),
        "fixed_threshold_achieved_test_fpr": float(1.0 - specificity),
        "max_tpr_at_test_fpr_at_most_0_05": roc_tpr,
        "achieved_test_fpr_for_roc_operating_point": roc_fpr,
    }


def _max_tpr_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    maximum_fpr: float,
) -> tuple[float | None, float | None]:
    positive = labels == 1
    positive_count = int(positive.sum())
    negative_count = int((~positive).sum())
    if not positive_count or not negative_count:
        return None, None
    best_tpr = 0.0
    best_fpr = 0.0
    thresholds = np.concatenate(
        (
            np.asarray([np.inf]),
            np.unique(np.asarray(scores, dtype=np.float64))[::-1],
        )
    )
    for cutoff in thresholds:
        predicted = scores >= cutoff
        fpr = float(np.mean(predicted[~positive]))
        if fpr <= maximum_fpr:
            tpr = float(np.mean(predicted[positive]))
            if tpr > best_tpr or (tpr == best_tpr and fpr > best_fpr):
                best_tpr, best_fpr = tpr, fpr
    return best_tpr, best_fpr


def _bootstrap_indices_by_group(
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> list[np.ndarray]:
    grouped = {
        group: np.flatnonzero(groups == group)
        for group in sorted(set(str(value) for value in groups))
    }
    group_ids = tuple(grouped)
    rng = np.random.default_rng(seed)
    indices = []
    for _ in range(replicates):
        sampled = rng.integers(0, len(group_ids), size=len(group_ids))
        indices.append(
            np.concatenate([grouped[group_ids[int(index)]] for index in sampled])
        )
    return indices


def _bootstrap_interval(
    values: Sequence[float],
    *,
    confidence: float,
    total: int,
) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "lower": None,
            "upper": None,
            "defined_replicates": 0,
            "total_replicates": total,
        }
    tail = 0.5 * (1.0 - confidence)
    return {
        "lower": float(np.quantile(finite, tail)),
        "upper": float(np.quantile(finite, 1.0 - tail)),
        "defined_replicates": int(len(finite)),
        "total_replicates": total,
    }


def _group_bootstrap(
    labels: np.ndarray,
    mid_scores: np.ndarray,
    final_scores: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    confidence: float,
    seed: int,
    candidate_name: str = "mid_fused",
) -> tuple[dict[str, object], dict[str, object]]:
    sampled_indices = _bootstrap_indices_by_group(
        groups,
        replicates=replicates,
        seed=seed,
    )
    mid_auc, mid_ap, differences = [], [], []
    for indices in sampled_indices:
        auc = _rank_auc(labels[indices], mid_scores[indices])
        final_auc = _rank_auc(labels[indices], final_scores[indices])
        average_precision = _average_precision(labels[indices], mid_scores[indices])
        if auc is not None:
            mid_auc.append(auc)
        if average_precision is not None:
            mid_ap.append(average_precision)
        if auc is not None and final_auc is not None:
            differences.append(auc - final_auc)
    mid_report = {
        "resampling_scope": (
            "conditional_on_fixed_split_fit_and_calibration_test_group_bootstrap"
        ),
        "groups": int(len(set(str(value) for value in groups))),
        "replicates": int(replicates),
        "confidence": float(confidence),
        "auroc": _bootstrap_interval(
            mid_auc,
            confidence=confidence,
            total=replicates,
        ),
        "aupr": _bootstrap_interval(
            mid_ap,
            confidence=confidence,
            total=replicates,
        ),
    }
    difference_report = {
        "estimand": f"test_auroc_{candidate_name}_minus_final_layer",
        "resampling_scope": (
            "conditional_on_fixed_split_fit_and_calibration_test_group_bootstrap"
        ),
        "point": (
            None
            if _rank_auc(labels, mid_scores) is None
            or _rank_auc(labels, final_scores) is None
            else float(
                _rank_auc(labels, mid_scores) - _rank_auc(labels, final_scores)
            )
        ),
        "groups": int(len(set(str(value) for value in groups))),
        "replicates": int(replicates),
        "confidence": float(confidence),
        "interval": _bootstrap_interval(
            differences,
            confidence=confidence,
            total=replicates,
        ),
    }
    return mid_report, difference_report


def _normal_tail_scores(
    reference: np.ndarray,
    values: np.ndarray,
) -> dict[str, np.ndarray]:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    queries = np.asarray(values, dtype=np.float64)
    if not len(ordered) or not np.isfinite(queries).all():
        raise ValueError("normal nuisance calibration requires finite values")
    left_ranks = np.searchsorted(ordered, queries, side="left")
    right_ranks = np.searchsorted(ordered, queries, side="right")
    midrank = 0.5 * (left_ranks + right_ranks) / len(ordered)
    right = midrank
    left_tail = 1.0 - midrank
    two_sided = np.clip(2.0 * np.abs(midrank - 0.5), 0.0, 1.0)
    return {
        "upper_tail": right,
        "lower_tail": left_tail,
        "two_sided": two_sided,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_score_rows(
    path: Path,
    rows: Sequence[dict[str, object]],
    *,
    include_label: bool,
) -> None:
    fieldnames = [
        "trace_id",
        "problem_id",
        "generator_model",
        "token_count",
        "response_token_count",
        "step_count",
    ]
    if include_label:
        fieldnames.extend(("label", "first_error"))
    score_fields = sorted(
        key
        for key in rows[0]
        if key not in set(fieldnames).union({"label", "first_error"})
    )
    fieldnames.extend(score_fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def evaluate_ghost(
    traces: Sequence[GhostTraceEmbedding],
    *,
    output_dir: str | Path,
    config: EvaluationConfig,
    embedding_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fit on train-normal, calibrate on validation-normal, score held-out test."""

    config.validate()
    examples = tuple(traces)
    if len(examples) < 12:
        raise ValueError("GHOST evaluation requires at least 12 traces")
    if len({trace.trace_id for trace in examples}) != len(examples):
        raise ValueError("trace IDs must be unique")
    layer_depths = examples[0].layer_depths
    if any(trace.layer_depths != layer_depths for trace in examples):
        raise ValueError("every trace must share one layer-depth contract")
    if len(layer_depths) < 2:
        raise ValueError("GHOST evaluation requires middle and final layers")

    from hypergraph.attention.splitting import (
        FixedHoldoutConfig,
        FixedHoldoutSplitter,
        TraceMeta,
    )

    split = FixedHoldoutSplitter(
        FixedHoldoutConfig(
            seed=int(config.split_seed),
            validation_ratio=float(config.validation_ratio),
            test_ratio=float(config.test_ratio),
        )
    ).split(
        [
            TraceMeta(
                trace_id=trace.trace_id,
                group_id=trace.problem_id,
                group_is_fallback=False,
                split=None,
                response_label=trace.response_label,
                gold_step=trace.first_error,
                num_steps=trace.step_count,
                num_response_tokens=trace.response_token_count,
                generator_model=trace.generator_model,
            )
            for trace in examples
        ]
    )
    fit_indices = tuple(
        index
        for index in split.train.indices
        if examples[index].response_label == 0
    )
    calibration_indices = tuple(
        index
        for index in split.validation.indices
        if examples[index].response_label == 0
    )
    test_indices = tuple(split.test.indices)
    if len(fit_indices) < 2:
        raise ValueError("training partition contains fewer than two normal references")
    if len(calibration_indices) < 2:
        raise ValueError(
            "validation partition contains fewer than two normal calibration traces"
        )
    test_labels = np.asarray(
        [examples[index].response_label for index in test_indices],
        dtype=np.int64,
    )
    if set(test_labels.tolist()) != {0, 1}:
        raise ValueError("held-out test partition must contain both response classes")

    fit_traces = tuple(examples[index] for index in fit_indices)
    calibration_traces = tuple(examples[index] for index in calibration_indices)
    test_traces = tuple(examples[index] for index in test_indices)
    fit_groups = {trace.problem_id for trace in fit_traces}
    calibration_groups = {trace.problem_id for trace in calibration_traces}
    test_groups = {trace.problem_id for trace in test_traces}
    intersections = {
        "fit_calibration": sorted(fit_groups & calibration_groups),
        "fit_test": sorted(fit_groups & test_groups),
        "calibration_test": sorted(calibration_groups & test_groups),
    }
    if any(intersections.values()):
        raise ValueError(f"problem group leakage detected: {intersections}")
    test_generator_names = sorted(
        {trace.generator_model for trace in test_traces}
    )
    generator_coverage = {
        generator: {
            "fit_normal": sum(
                trace.generator_model == generator for trace in fit_traces
            ),
            "calibration_normal": sum(
                trace.generator_model == generator
                for trace in calibration_traces
            ),
            "test_normal": sum(
                trace.generator_model == generator
                and trace.response_label == 0
                for trace in test_traces
            ),
            "test_error": sum(
                trace.generator_model == generator
                and trace.response_label == 1
                for trace in test_traces
            ),
        }
        for generator in test_generator_names
    }
    missing_fit_generator_reference = [
        generator
        for generator, counts in generator_coverage.items()
        if counts["fit_normal"] == 0
    ]
    missing_calibration_generator_reference = [
        generator
        for generator, counts in generator_coverage.items()
        if counts["calibration_normal"] == 0
    ]
    within_generator_eligible = [
        generator
        for generator, counts in generator_coverage.items()
        if counts["test_normal"] > 0 and counts["test_error"] > 0
    ]
    generator_confound_status = (
        "coverage_complete_but_split_not_generator_stratified"
        if (
            not missing_fit_generator_reference
            and not missing_calibration_generator_reference
            and len(within_generator_eligible) == len(generator_coverage)
        )
        else "uncontrolled_use_within_generator_metrics_before_pooled_claim"
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    split_payload = split.manifest()
    split_payload.update(
        {
            "fit_normal_trace_ids": [trace.trace_id for trace in fit_traces],
            "fit_normal_problem_ids": sorted(fit_groups),
            "calibration_normal_trace_ids": [
                trace.trace_id for trace in calibration_traces
            ],
            "calibration_normal_problem_ids": sorted(calibration_groups),
            "test_trace_ids": [trace.trace_id for trace in test_traces],
            "test_problem_ids": sorted(test_groups),
            "generator_coverage": generator_coverage,
            "unused_train_error_trace_ids": [
                examples[index].trace_id
                for index in split.train.indices
                if examples[index].response_label == 1
            ],
            "unused_validation_error_trace_ids": [
                examples[index].trace_id
                for index in split.validation.indices
                if examples[index].response_label == 1
            ],
        }
    )
    _write_json(destination / "split.json", split_payload)
    audit = {
        "schema_version": "ghost_fit_audit_v1",
        "method_scope": "normal_reference_one_class_not_fully_label_free",
        "fit_trace_count": len(fit_traces),
        "fit_positive_count": sum(trace.response_label for trace in fit_traces),
        "calibration_trace_count": len(calibration_traces),
        "calibration_positive_count": sum(
            trace.response_label for trace in calibration_traces
        ),
        "test_trace_count": len(test_traces),
        "problem_group_intersections": intersections,
        "error_samples_used_for_fit_or_calibration": 0,
        "training_epochs": 0,
        "generator_coverage": generator_coverage,
        "test_generators_without_fit_normal_reference": (
            missing_fit_generator_reference
        ),
        "test_generators_without_calibration_normal_reference": (
            missing_calibration_generator_reference
        ),
        "within_generator_both_class_eligible": within_generator_eligible,
        "generator_confound_status": generator_confound_status,
        "layer_depth_semantics": {
            "depth_d": "output of d transformer blocks; HF hidden_states index d",
            "block_index": "depth_d minus 1",
        },
    }
    _write_json(destination / "fit-audit.json", audit)

    score_rows = [
        {
            "trace_id": trace.trace_id,
            "problem_id": trace.problem_id,
            "generator_model": trace.generator_model,
            "token_count": trace.token_count,
            "response_token_count": trace.response_token_count,
            "step_count": trace.step_count,
        }
        for trace in test_traces
    ]
    calibration_count = len(calibration_traces)
    scored_by_representation = {}
    for representation in ("response_mean", "response_last"):
        detector = GhostMahalanobisEnsemble(
            mid_depths=layer_depths[:-1],
            final_depth=layer_depths[-1],
            shrinkage=float(config.shrinkage),
            regularization=float(config.regularization),
        ).fit(fit_traces, representation=representation)
        detector.calibrate(calibration_traces)
        scored = detector.score(test_traces)
        scored_by_representation[representation] = scored
        detector.save(destination / f"reference-model-{representation}.npz")
        for row_index, row in enumerate(score_rows):
            row[f"{representation}_mid_fused_rank_score"] = float(
                scored.mid_fused_score[row_index]
            )
            row[f"{representation}_mid_fused_percentile"] = float(
                scored.mid_fused_percentile[row_index]
            )
            row[f"{representation}_final_distance"] = float(
                scored.layer_distances[row_index, -1]
            )
            row[f"{representation}_final_percentile"] = float(
                scored.final_percentile[row_index]
            )
            for layer_index, depth in enumerate(layer_depths):
                row[f"{representation}_depth_{depth}_distance"] = float(
                    scored.layer_distances[row_index, layer_index]
                )
                row[f"{representation}_depth_{depth}_percentile"] = float(
                    scored.layer_percentiles[row_index, layer_index]
                )

    calibration_token_counts = np.asarray(
        [trace.response_token_count for trace in calibration_traces],
        dtype=np.float64,
    )
    calibration_step_counts = np.asarray(
        [trace.step_count for trace in calibration_traces],
        dtype=np.float64,
    )
    test_token_counts = np.asarray(
        [trace.response_token_count for trace in test_traces],
        dtype=np.float64,
    )
    test_step_counts = np.asarray(
        [trace.step_count for trace in test_traces],
        dtype=np.float64,
    )
    nuisance_scores = {
        "response_token_count": _normal_tail_scores(
            calibration_token_counts,
            test_token_counts,
        ),
        "step_count": _normal_tail_scores(
            calibration_step_counts,
            test_step_counts,
        ),
    }
    for nuisance_name, tails in nuisance_scores.items():
        for tail_name, values in tails.items():
            for row_index, row in enumerate(score_rows):
                row[f"nuisance_{nuisance_name}_{tail_name}"] = float(
                    values[row_index]
                )

    unlabeled_path = destination / "anomaly-scores-test.csv"
    _write_score_rows(unlabeled_path, score_rows, include_label=False)
    score_hash = hashlib.sha256(unlabeled_path.read_bytes()).hexdigest()
    audit["unlabeled_score_sha256"] = score_hash
    audit["frozen_score_artifact_written_before_metric_computation"] = True
    audit["test_labels_used_for_predeclared_split_stratification"] = True
    _write_json(destination / "fit-audit.json", audit)

    metrics: dict[str, object] = {
        "schema_version": "ghost_metrics_v1",
        "result_kind": "real_processbench_ghost_style_one_class",
        "primary_representation": "response_mean",
        "primary_layer_rule": "equal_mean_of_six_mid_layer_calibration_percentiles",
        "labels_used_for": (
            "upstream cohort construction (profile-dependent), predeclared split "
            "stratification, normal-reference identification, and final evaluation"
        ),
        "evaluation_config": asdict(config),
        "bootstrap_scope": (
            "conditional_on_fixed_split_fit_and_calibration_test_group_bootstrap"
        ),
        "not_probability": True,
    }
    test_groups_array = np.asarray(
        [trace.problem_id for trace in test_traces],
        dtype=str,
    )
    generators = np.asarray(
        [trace.generator_model for trace in test_traces],
        dtype=str,
    )
    for representation_index, representation in enumerate(
        ("response_mean", "response_last")
    ):
        scored = scored_by_representation[representation]
        mid_metrics = _detection_metrics(
            test_labels,
            scored.mid_fused_score,
            scored.mid_fused_percentile,
            threshold=float(config.threshold_quantile),
            calibration_count=calibration_count,
        )
        final_metrics = _detection_metrics(
            test_labels,
            scored.layer_distances[:, -1],
            scored.final_percentile,
            threshold=float(config.threshold_quantile),
            calibration_count=calibration_count,
        )
        bootstrap, difference = _group_bootstrap(
            test_labels,
            scored.mid_fused_score,
            scored.layer_distances[:, -1],
            test_groups_array,
            replicates=int(config.bootstrap_replicates),
            confidence=float(config.bootstrap_confidence),
            seed=int(config.split_seed) + 1000 + representation_index,
        )
        mid_metrics["problem_group_bootstrap"] = bootstrap
        per_layer = {
            str(depth): _detection_metrics(
                test_labels,
                scored.layer_distances[:, layer_index],
                scored.layer_percentiles[:, layer_index],
                threshold=float(config.threshold_quantile),
                calibration_count=calibration_count,
            )
            for layer_index, depth in enumerate(layer_depths)
        }
        per_layer_differences = {}
        for layer_index, depth in enumerate(layer_depths[:-1]):
            _, layer_difference = _group_bootstrap(
                test_labels,
                scored.layer_distances[:, layer_index],
                scored.layer_distances[:, -1],
                test_groups_array,
                replicates=int(config.bootstrap_replicates),
                confidence=float(config.bootstrap_confidence),
                seed=(
                    int(config.split_seed)
                    + 2000
                    + 100 * representation_index
                    + layer_index
                ),
                candidate_name=f"depth_{depth}",
            )
            per_layer_differences[str(depth)] = layer_difference
        generator_metrics = {}
        for generator in sorted(set(generators)):
            mask = generators == generator
            generator_metrics[generator] = _detection_metrics(
                test_labels[mask],
                scored.mid_fused_score[mask],
                scored.mid_fused_percentile[mask],
                threshold=float(config.threshold_quantile),
                calibration_count=calibration_count,
            )
        defined_auc = [
            report["auroc"]
            for report in generator_metrics.values()
            if report["auroc"] is not None
        ]
        defined_ap = [
            report["aupr"]
            for report in generator_metrics.values()
            if report["aupr"] is not None
        ]
        generator_macro = {
            "total_generators": len(generator_metrics),
            "eligible_both_class_generators": len(defined_auc),
            "defined_auroc_generators": len(defined_auc),
            "defined_aupr_generators": len(defined_ap),
            "auroc": (
                float(np.mean(defined_auc)) if defined_auc else None
            ),
            "aupr": float(np.mean(defined_ap)) if defined_ap else None,
        }
        metrics[representation] = {
            "mid_fused": mid_metrics,
            "final_layer": final_metrics,
            "per_layer": per_layer,
            "generator_stratified_mid_fused": generator_metrics,
            "generator_macro_mid_fused": generator_macro,
            "mid_minus_final_bootstrap": difference,
            "per_mid_layer_minus_final_bootstrap": per_layer_differences,
        }

    nuisance_values = {
        "response_token_count": (
            calibration_token_counts,
            test_token_counts,
        ),
        "step_count": (
            calibration_step_counts,
            test_step_counts,
        ),
    }
    metrics["nuisance_only"] = {}
    for nuisance_name, (normal_values, held_out_values) in nuisance_values.items():
        rankings = {
            "upper_tail": held_out_values,
            "lower_tail": -held_out_values,
            "two_sided": nuisance_scores[nuisance_name]["two_sided"],
        }
        metrics["nuisance_only"][nuisance_name] = {
            tail_name: _detection_metrics(
                test_labels,
                rankings[tail_name],
                nuisance_scores[nuisance_name][tail_name],
                threshold=float(config.threshold_quantile),
                calibration_count=calibration_count,
            )
            for tail_name in ("upper_tail", "lower_tail", "two_sided")
        }
    labeled_rows = [
        {
            **row,
            "label": trace.response_label,
            "first_error": trace.first_error,
        }
        for row, trace in zip(score_rows, test_traces)
    ]
    _write_score_rows(
        destination / "scores-test.csv",
        labeled_rows,
        include_label=True,
    )
    _write_json(destination / "metrics.json", metrics)

    primary = metrics["response_mean"]
    summary: dict[str, object] = {
        "status": "success",
        "schema_version": "ghost_summary_v1",
        "result_kind": "real_processbench_ghost_style_one_class",
        "method": "GHOST-inspired abstract-level mid-layer Mahalanobis",
        "training_epochs": 0,
        "evaluation_config": asdict(config),
        "observer_mode": "teacher_forced_fixed_processbench_response",
        "scientific_boundary": (
            "The supplied abstract does not specify exact probe layers, pooling, "
            "covariance shrinkage, or fusion. This is a preregistered abstract-level "
            "reimplementation, not a claim of exact paper reproduction."
        ),
        "normal_reference_boundary": (
            "ProcessBench label=-1 identifies fit/calibration normal samples; "
            "no error sample enters geometry or score calibration."
        ),
        "label_use_boundary": (
            "Labels may define a balanced smoke/pilot cohort, stratify the fixed "
            "problem-group split, identify normal references, and compute final "
            "metrics. They never enter hidden-state geometry, fusion weights, or "
            "the fixed threshold."
        ),
        "comparison_estimand": (
            "fixed_equal_weight_six_mid_layer_ensemble_vs_single_normalized_"
            "final_layer_control; not an isolated causal layer-depth effect"
        ),
        "final_layer_normalization_boundary": (
            "Middle probes are raw decoder-block residual outputs, whereas the "
            "final AutoModel state is after the model final RMSNorm. The comparison "
            "may therefore include a final-normalization effect."
        ),
        "bootstrap_boundary": (
            "Intervals condition on one fixed split, fitted geometry, and "
            "calibration set; they resample held-out problem groups only."
        ),
        "generator_confound_status": generator_confound_status,
        "generator_confound_boundary": (
            "The shared problem splitter is not generator-stratified. Treat pooled "
            "metrics as potentially style-confounded when generator coverage is "
            "incomplete or within-generator metrics are undefined."
        ),
        "layer_depths": list(layer_depths),
        "mid_depths": list(layer_depths[:-1]),
        "final_depth": layer_depths[-1],
        "fit_normal_traces": len(fit_traces),
        "calibration_normal_traces": len(calibration_traces),
        "test_traces": len(test_traces),
        "test_prevalence": float(test_labels.mean()),
        "threshold_resolution": {
            "calibration_count": calibration_count,
            "minimum_rank_grid_tail_rate": primary["mid_fused"][
                "minimum_rank_grid_tail_rate"
            ],
            "grid_implied_tail_rate_at_percentile_cutoff": primary[
                "mid_fused"
            ]["grid_implied_tail_rate_at_percentile_cutoff"],
            "nominal_tail_rate": primary["mid_fused"]["nominal_tail_rate"],
            "scope": primary["mid_fused"]["threshold_metrics_scope"],
        },
        "primary_test_mid_fused": primary["mid_fused"],
        "primary_test_final_layer": primary["final_layer"],
        "primary_mid_minus_final_bootstrap": primary[
            "mid_minus_final_bootstrap"
        ],
        "embedding_metadata": dict(embedding_metadata or {}),
        "artifacts": {
            "split": "split.json",
            "fit_audit": "fit-audit.json",
            "unlabeled_scores": "anomaly-scores-test.csv",
            "labeled_scores": "scores-test.csv",
            "metrics": "metrics.json",
            "response_mean_model": "reference-model-response_mean.npz",
            "response_last_model": "reference-model-response_last.npz",
        },
    }
    _write_json(destination / "summary.json", summary)
    return summary
