from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..evaluation import BinaryReport, GroupedBootstrapReport, PredictionRow
from ..splitting import SplitAssignment
from .data import CausalTrace


@dataclass(frozen=True)
class NuisanceBaselineResult:
    validation: BinaryReport
    test: BinaryReport
    validation_uncertainty: GroupedBootstrapReport
    test_uncertainty: GroupedBootstrapReport
    validation_predictions: tuple[PredictionRow, ...]
    test_predictions: tuple[PredictionRow, ...]
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray


class NuisanceLogisticBaseline:
    """Train-only length/step baseline with a stable Newton solver."""

    def __init__(self, *, l2: float = 1e-3, max_iter: int = 100) -> None:
        if l2 < 0 or max_iter <= 0:
            raise ValueError("l2 and max_iter are invalid")
        self.l2 = float(l2)
        self.max_iter = int(max_iter)

    def fit(
        self,
        traces: Sequence[CausalTrace],
        split: SplitAssignment,
        *,
        bootstrap_replicates: int = 2000,
        bootstrap_confidence: float = 0.95,
        bootstrap_seed: int = 17,
    ) -> NuisanceBaselineResult:
        raw = np.asarray([self._features(trace) for trace in traces])
        labels = np.asarray([trace.response_label for trace in traces], dtype=np.int64)
        train = np.asarray(split.train.indices)
        mean = raw[train].mean(axis=0)
        scale = raw[train].std(axis=0)
        scale[scale < 1e-8] = 1.0
        features = np.column_stack((np.ones(len(raw)), (raw - mean) / scale))
        weights = self._newton(features[train], labels[train])
        probabilities = self._sigmoid(features @ weights)
        validation = np.asarray(split.validation.indices)
        test = np.asarray(split.test.indices)
        validation_predictions = self._prediction_rows(
            traces, validation, probabilities
        )
        test_predictions = self._prediction_rows(traces, test, probabilities)
        return NuisanceBaselineResult(
            validation=BinaryReport.from_scores(
                labels=labels[validation], scores=probabilities[validation]
            ),
            test=BinaryReport.from_scores(
                labels=labels[test], scores=probabilities[test]
            ),
            validation_uncertainty=GroupedBootstrapReport.from_predictions(
                validation_predictions,
                replicates=bootstrap_replicates,
                confidence=bootstrap_confidence,
                seed=bootstrap_seed,
            ),
            test_uncertainty=GroupedBootstrapReport.from_predictions(
                test_predictions,
                replicates=bootstrap_replicates,
                confidence=bootstrap_confidence,
                seed=bootstrap_seed + 1,
            ),
            validation_predictions=validation_predictions,
            test_predictions=test_predictions,
            weights=weights,
            mean=mean,
            scale=scale,
        )

    @staticmethod
    def _prediction_rows(
        traces: Sequence[CausalTrace],
        indices: np.ndarray,
        probabilities: np.ndarray,
    ) -> tuple[PredictionRow, ...]:
        return tuple(
            PredictionRow(
                trace_id=traces[int(index)].trace_id,
                problem_id=traces[int(index)].problem_id,
                label=traces[int(index)].response_label,
                probability=float(probabilities[int(index)]),
                first_error=traces[int(index)].labels.first_error,
                predicted_step=-1,
                step_probabilities=(),
            )
            for index in indices
        )

    @staticmethod
    def _features(trace: CausalTrace) -> tuple[float, float, float, float]:
        steps = trace.labels.num_steps
        return (
            np.log1p(trace.prompt_tokens),
            np.log1p(trace.response_tokens),
            np.log1p(steps),
            trace.response_tokens / steps,
        )

    def _newton(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        weights = np.zeros(features.shape[1], dtype=np.float64)
        penalty = np.eye(features.shape[1]) * self.l2
        penalty[0, 0] = 0.0
        for _ in range(self.max_iter):
            probability = self._sigmoid(features @ weights)
            gradient = features.T @ (probability - labels) + penalty @ weights
            variance = np.maximum(probability * (1.0 - probability), 1e-8)
            hessian = (features.T * variance) @ features + penalty
            step = np.linalg.solve(hessian, gradient)
            weights -= step
            if np.linalg.norm(step) < 1e-8:
                break
        return weights

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        positive = value >= 0
        result = np.empty_like(value, dtype=np.float64)
        result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
        exponential = np.exp(value[~positive])
        result[~positive] = exponential / (1.0 + exponential)
        return result
