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
    if not positive_count:
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
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int | None]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if (
        labels.ndim != 1
        or scores.shape != labels.shape
        or not len(labels)
        or not np.isin(labels, [0, 1]).all()
        or not np.isfinite(scores).all()
        or np.any((scores < 0.0) | (scores > 1.0))
    ):
        raise ValueError("binary labels and anomaly percentiles must be aligned")
    predicted = scores >= threshold
    positive = labels == 1
    tp = int((predicted & positive).sum())
    tn = int((~predicted & ~positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    negative_scores = scores[~positive]
    if len(negative_scores) and positive.any():
        fpr_threshold = float(
            np.quantile(negative_scores, 0.95, method="higher")
        )
        tpr_at_fpr_5 = float(np.mean(scores[positive] > fpr_threshold))
    else:
        tpr_at_fpr_5 = None
    return {
        "n": int(len(labels)),
        "positives": int(positive.sum()),
        "prevalence": float(positive.mean()),
        "score_semantics": "normal_calibration_empirical_anomaly_percentile",
        "auroc": _rank_auc(labels, scores),
        "aupr": _average_precision(labels, scores),
        "threshold": float(threshold),
        "accuracy": float(np.mean(predicted == positive)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "mcc": float(mcc),
        "tpr_at_empirical_5pct_fpr": tpr_at_fpr_5,
    }


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
        "estimand": "test_auroc_mid_fused_minus_final_layer",
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


def _normal_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    if not len(ordered):
        raise ValueError("normal nuisance calibration cannot be empty")
    return np.searchsorted(ordered, values, side="right") / len(ordered)


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
            "label": trace.response_label,
            "first_error": trace.first_error,
        }
        for trace in test_traces
    ]
    metrics: dict[str, object] = {
        "schema_version": "ghost_metrics_v1",
        "result_kind": "real_processbench_ghost_style_one_class",
        "primary_representation": "response_mean",
        "primary_layer_rule": "equal_mean_of_six_mid_layer_calibration_percentiles",
        "labels_used_for": "split/reference identification and final evaluation only",
        "not_probability": True,
    }
    test_groups_array = np.asarray(
        [trace.problem_id for trace in test_traces],
        dtype=str,
    )
    for representation_index, representation in enumerate(
        ("response_mean", "response_last")
    ):
        detector = GhostMahalanobisEnsemble(
            mid_depths=layer_depths[:-1],
            final_depth=layer_depths[-1],
            shrinkage=float(config.shrinkage),
            regularization=float(config.regularization),
        ).fit(fit_traces, representation=representation)
        detector.calibrate(calibration_traces)
        scored = detector.score(test_traces)
        detector.save(destination / f"reference-model-{representation}.npz")
        for row_index, row in enumerate(score_rows):
            row[f"{representation}_mid_fused"] = float(
                scored.mid_fused_percentile[row_index]
            )
            row[f"{representation}_final"] = float(
                scored.final_percentile[row_index]
            )
            for layer_index, depth in enumerate(layer_depths):
                row[f"{representation}_depth_{depth}"] = float(
                    scored.layer_percentiles[row_index, layer_index]
                )

        mid_metrics = _detection_metrics(
            test_labels,
            scored.mid_fused_percentile,
            threshold=float(config.threshold_quantile),
        )
        final_metrics = _detection_metrics(
            test_labels,
            scored.final_percentile,
            threshold=float(config.threshold_quantile),
        )
        bootstrap, difference = _group_bootstrap(
            test_labels,
            scored.mid_fused_percentile,
            scored.final_percentile,
            test_groups_array,
            replicates=int(config.bootstrap_replicates),
            confidence=float(config.bootstrap_confidence),
            seed=int(config.split_seed) + 1000 + representation_index,
        )
        mid_metrics["problem_group_bootstrap"] = bootstrap
        per_layer = {
            str(depth): _detection_metrics(
                test_labels,
                scored.layer_percentiles[:, layer_index],
                threshold=float(config.threshold_quantile),
            )
            for layer_index, depth in enumerate(layer_depths)
        }
        generator_metrics = {}
        generators = np.asarray(
            [trace.generator_model for trace in test_traces],
            dtype=str,
        )
        for generator in sorted(set(generators)):
            mask = generators == generator
            generator_metrics[generator] = _detection_metrics(
                test_labels[mask],
                scored.mid_fused_percentile[mask],
                threshold=float(config.threshold_quantile),
            )
        metrics[representation] = {
            "mid_fused": mid_metrics,
            "final_layer": final_metrics,
            "per_layer": per_layer,
            "generator_stratified_mid_fused": generator_metrics,
            "mid_minus_final_bootstrap": difference,
        }

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
    metrics["nuisance_only"] = {
        "response_token_count": _detection_metrics(
            test_labels,
            _normal_cdf(calibration_token_counts, test_token_counts),
            threshold=float(config.threshold_quantile),
        ),
        "step_count": _detection_metrics(
            test_labels,
            _normal_cdf(calibration_step_counts, test_step_counts),
            threshold=float(config.threshold_quantile),
        ),
    }
    unlabeled_path = destination / "anomaly-scores-test.csv"
    _write_score_rows(unlabeled_path, score_rows, include_label=False)
    score_hash = hashlib.sha256(unlabeled_path.read_bytes()).hexdigest()
    audit["unlabeled_score_sha256"] = score_hash
    audit["test_labels_attached_after_unlabeled_score_artifact"] = True
    _write_json(destination / "fit-audit.json", audit)
    _write_score_rows(
        destination / "scores-test.csv",
        score_rows,
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
        "layer_depths": list(layer_depths),
        "mid_depths": list(layer_depths[:-1]),
        "final_depth": layer_depths[-1],
        "fit_normal_traces": len(fit_traces),
        "calibration_normal_traces": len(calibration_traces),
        "test_traces": len(test_traces),
        "test_prevalence": float(test_labels.mean()),
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
