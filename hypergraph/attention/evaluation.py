from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = labels == 1
    negative_count = int((~positives).sum())
    positive_count = int(positives.sum())
    if not positive_count or not negative_count:
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
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive_count = int(labels.sum())
    if not positive_count:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels, sorted_scores = labels[order], scores[order]
    true_positive = false_positive = 0
    previous_recall = average_precision = 0.0
    start = 0
    while start < len(labels):
        stop = start + 1
        while stop < len(labels) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        true_positive += int((sorted_labels[start:stop] == 1).sum())
        false_positive += int((sorted_labels[start:stop] == 0).sum())
        recall = true_positive / positive_count
        precision = true_positive / (true_positive + false_positive)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(average_precision)


@dataclass(frozen=True)
class BinaryReport:
    n: int
    positives: int
    prevalence: float
    auroc: float | None
    aupr: float | None
    threshold: float
    accuracy: float
    sensitivity: float
    specificity: float
    balanced_accuracy: float
    mcc: float
    brier: float
    ece: float

    @classmethod
    def from_scores(
        cls,
        *,
        labels: np.ndarray,
        scores: np.ndarray,
        threshold: float = 0.5,
        calibration_bins: int = 10,
    ) -> "BinaryReport":
        labels = np.asarray(labels)
        scores = np.asarray(scores, dtype=np.float64)
        if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
            raise ValueError("labels and scores must be aligned non-empty vectors")
        if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
            raise ValueError("labels must be binary and scores must be finite")
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("scores must be probabilities in [0, 1]")
        if not 0.0 <= threshold <= 1.0 or calibration_bins <= 0:
            raise ValueError("threshold and calibration_bins are invalid")
        labels = labels.astype(np.int64, copy=False)
        predictions = scores >= threshold
        positive = labels == 1
        tp = int((predictions & positive).sum())
        tn = int((~predictions & ~positive).sum())
        fp = int((predictions & ~positive).sum())
        fn = int((~predictions & positive).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else 0.0

        bin_ids = np.minimum(
            (scores * calibration_bins).astype(int), calibration_bins - 1
        )
        ece = 0.0
        for bin_id in range(calibration_bins):
            mask = bin_ids == bin_id
            if mask.any():
                ece += float(mask.mean()) * abs(
                    float(scores[mask].mean()) - float(labels[mask].mean())
                )

        return cls(
            n=len(labels),
            positives=int(positive.sum()),
            prevalence=float(positive.mean()),
            auroc=_rank_auc(labels, scores),
            aupr=_average_precision(labels, scores),
            threshold=float(threshold),
            accuracy=float((predictions == positive).mean()),
            sensitivity=float(sensitivity),
            specificity=float(specificity),
            balanced_accuracy=float(0.5 * (sensitivity + specificity)),
            mcc=float(mcc),
            brier=float(np.mean((scores - labels) ** 2)),
            ece=float(ece),
        )

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionRow:
    """One auditable held-out response prediction and its step hazards."""

    trace_id: str
    problem_id: str
    label: int
    probability: float
    first_error: int
    predicted_step: int
    step_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        steps = np.asarray(self.step_probabilities, dtype=np.float64)
        if not self.trace_id or not self.problem_id:
            raise ValueError("trace_id and problem_id are required")
        if self.label not in (0, 1) or self.label != int(self.first_error >= 0):
            raise ValueError("label and first_error are inconsistent")
        if not np.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        if steps.ndim != 1 or not np.isfinite(steps).all():
            raise ValueError("step_probabilities must be a finite vector")
        if np.any((steps < 0.0) | (steps > 1.0)):
            raise ValueError("step probabilities must lie in [0, 1]")
        if len(steps):
            if self.first_error >= len(steps):
                raise ValueError("first_error lies outside step_probabilities")
            if not 0 <= self.predicted_step < len(steps):
                raise ValueError("predicted_step lies outside step_probabilities")
            if self.predicted_step != int(np.argmax(steps)):
                raise ValueError(
                    "predicted_step must select the largest step probability"
                )
        elif self.predicted_step != -1:
            raise ValueError(
                "predicted_step must be -1 when step scores are unavailable"
            )
        object.__setattr__(
            self, "step_probabilities", tuple(float(value) for value in steps)
        )


@dataclass(frozen=True)
class BootstrapInterval:
    lower: float | None
    upper: float | None
    defined_replicates: int
    total_replicates: int

    @classmethod
    def from_values(
        cls,
        values: Sequence[float],
        *,
        total_replicates: int,
        confidence: float,
    ) -> "BootstrapInterval":
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            return cls(None, None, 0, total_replicates)
        tail = 0.5 * (1.0 - confidence)
        lower, upper = np.quantile(finite, [tail, 1.0 - tail])
        return cls(float(lower), float(upper), len(finite), total_replicates)


@dataclass(frozen=True)
class GroupedBootstrapReport:
    """Uncertainty from resampling problem IDs, preserving within-problem traces."""

    groups: int
    replicates: int
    confidence: float
    auroc: BootstrapInterval
    aupr: BootstrapInterval

    @classmethod
    def from_predictions(
        cls,
        predictions: Sequence[PredictionRow],
        *,
        replicates: int,
        confidence: float,
        seed: int,
    ) -> "GroupedBootstrapReport":
        if not predictions:
            raise ValueError("bootstrap requires held-out predictions")
        if replicates <= 0 or not 0.0 < confidence < 1.0:
            raise ValueError("bootstrap replicates and confidence are invalid")
        grouped: dict[str, list[PredictionRow]] = {}
        for row in predictions:
            grouped.setdefault(row.problem_id, []).append(row)
        group_ids = tuple(sorted(grouped))
        rng = np.random.default_rng(seed)
        aucs: list[float] = []
        aprs: list[float] = []
        for _ in range(replicates):
            sampled = rng.integers(0, len(group_ids), size=len(group_ids))
            rows = [row for index in sampled for row in grouped[group_ids[int(index)]]]
            labels = np.asarray([row.label for row in rows], dtype=np.int64)
            scores = np.asarray([row.probability for row in rows], dtype=np.float64)
            auc = _rank_auc(labels, scores)
            apr = _average_precision(labels, scores)
            if auc is not None:
                aucs.append(auc)
            if apr is not None:
                aprs.append(apr)
        return cls(
            groups=len(group_ids),
            replicates=replicates,
            confidence=float(confidence),
            auroc=BootstrapInterval.from_values(
                aucs, total_replicates=replicates, confidence=confidence
            ),
            aupr=BootstrapInterval.from_values(
                aprs, total_replicates=replicates, confidence=confidence
            ),
        )


@dataclass(frozen=True)
class LocalizationReport:
    error_traces: int
    top1: float | None
    mean_rank: float | None
    mean_reciprocal_rank: float | None

    @classmethod
    def from_traces(
        cls, first_errors: list[int], step_scores: list[np.ndarray]
    ) -> "LocalizationReport":
        if len(first_errors) != len(step_scores):
            raise ValueError("first-error labels and step scores must align")
        ranks: list[float] = []
        for first_error, scores in zip(first_errors, step_scores):
            if first_error < 0:
                continue
            values = np.asarray(scores, dtype=np.float64)
            if values.ndim != 1 or first_error >= len(values):
                raise ValueError("first-error index lies outside the step scores")
            if not np.isfinite(values).all():
                raise ValueError("step scores must be finite")
            gold = values[first_error]
            better = int((values > gold).sum())
            tied = int((values == gold).sum())
            ranks.append(1.0 + better + 0.5 * (tied - 1))
        if not ranks:
            return cls(0, None, None, None)
        rank_array = np.asarray(ranks)
        return cls(
            error_traces=len(ranks),
            top1=float(np.mean(rank_array == 1.0)),
            mean_rank=float(rank_array.mean()),
            mean_reciprocal_rank=float(np.mean(1.0 / rank_array)),
        )
