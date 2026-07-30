from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np


def select_mid_depths(
    *,
    num_hidden_layers: int,
    start_fraction: float = 0.25,
    end_fraction: float = 0.50,
    count: int = 6,
) -> tuple[int, ...]:
    """Select one-based block-output depths on a fixed middle-depth grid.

    A Hugging Face ``hidden_states`` tuple has the embedding output at index 0,
    so block depth ``d`` corresponds to ``hidden_states[d]`` and to zero-based
    block index ``d - 1``.  Keeping depth as the public coordinate prevents the
    common off-by-one ambiguity.
    """

    if (
        isinstance(num_hidden_layers, bool)
        or not isinstance(num_hidden_layers, Integral)
        or int(num_hidden_layers) < 1
    ):
        raise ValueError("num_hidden_layers must be a positive integer")
    if isinstance(count, bool) or not isinstance(count, Integral) or int(count) < 1:
        raise ValueError("count must be a positive integer")
    if (
        isinstance(start_fraction, bool)
        or not isinstance(start_fraction, Real)
        or isinstance(end_fraction, bool)
        or not isinstance(end_fraction, Real)
    ):
        raise ValueError("depth fractions must be finite real numbers")
    start_fraction, end_fraction = float(start_fraction), float(end_fraction)
    if (
        not np.isfinite(start_fraction)
        or not np.isfinite(end_fraction)
        or not 0.0 < start_fraction < end_fraction <= 1.0
    ):
        raise ValueError("depth fractions must satisfy 0 < start < end <= 1")

    layers, count = int(num_hidden_layers), int(count)
    start = max(1, int(np.ceil(start_fraction * layers)))
    stop = min(layers, int(np.floor(end_fraction * layers)))
    if stop < start:
        raise ValueError("the requested middle-depth interval contains no blocks")
    depths = tuple(
        int(np.floor(value + 0.5))
        for value in np.linspace(start, stop, num=count)
    )
    if len(set(depths)) != count:
        raise ValueError(
            "the model is too shallow for the requested number of unique probes"
        )
    return depths


def response_token_positions(
    step_ranges: np.ndarray,
    *,
    token_count: int | None = None,
) -> np.ndarray:
    ranges = np.asarray(step_ranges)
    if (
        ranges.ndim != 2
        or ranges.shape[1] != 2
        or not len(ranges)
        or not np.issubdtype(ranges.dtype, np.integer)
    ):
        raise ValueError("step_ranges must have integer shape [steps, 2]")
    ranges = ranges.astype(np.int64, copy=False)
    if np.any(ranges[:, 0] < 0) or np.any(ranges[:, 1] <= ranges[:, 0]):
        raise ValueError("every step range must be non-empty and non-negative")
    if np.any(ranges[1:, 0] < ranges[:-1, 1]):
        raise ValueError("step ranges must be ordered and non-overlapping")
    if token_count is not None:
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, Integral)
            or int(token_count) < 1
        ):
            raise ValueError("token_count must be a positive integer")
        if int(ranges[-1, 1]) > int(token_count):
            raise ValueError("a step range lies outside the hidden-state token axis")
    return np.concatenate(
        [np.arange(start, stop, dtype=np.int64) for start, stop in ranges]
    )


def pool_response_hidden(
    hidden: np.ndarray,
    step_ranges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return equal-trace response mean and final-response-token states."""

    values = np.asarray(hidden)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("hidden must have shape [tokens, features]")
    if not np.isfinite(values).all():
        raise ValueError("hidden states must be finite")
    positions = response_token_positions(step_ranges, token_count=values.shape[0])
    selected = values[positions]
    return selected.mean(axis=0), selected[-1].copy(), positions


@dataclass(frozen=True)
class GhostTraceEmbedding:
    trace_id: str
    problem_id: str
    response_label: int
    first_error: int
    generator_model: str
    layer_depths: tuple[int, ...]
    response_mean: np.ndarray
    response_last: np.ndarray
    token_count: int
    response_token_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise ValueError("trace_id is required")
        if not isinstance(self.problem_id, str) or not self.problem_id.strip():
            raise ValueError("problem_id is required")
        if not isinstance(self.generator_model, str) or not self.generator_model.strip():
            raise ValueError("generator_model is required")
        if self.response_label not in (0, 1):
            raise ValueError("response_label must be 0 or 1")
        if (
            isinstance(self.first_error, bool)
            or not isinstance(self.first_error, Integral)
            or (self.response_label == 0 and int(self.first_error) != -1)
            or (self.response_label == 1 and int(self.first_error) < 0)
        ):
            raise ValueError("first_error and response_label are inconsistent")
        depths = tuple(int(depth) for depth in self.layer_depths)
        if (
            not depths
            or any(depth < 1 for depth in depths)
            or any(right <= left for left, right in zip(depths, depths[1:]))
        ):
            raise ValueError("layer_depths must be strictly increasing positive depths")
        mean = np.asarray(self.response_mean)
        last = np.asarray(self.response_last)
        if (
            mean.ndim != 2
            or last.shape != mean.shape
            or mean.shape[0] != len(depths)
            or mean.shape[1] < 1
        ):
            raise ValueError(
                "response representations must have shape [layer_depths, features]"
            )
        if not np.isfinite(mean).all() or not np.isfinite(last).all():
            raise ValueError("response representations must be finite")
        if (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, Integral)
            or isinstance(self.response_token_count, bool)
            or not isinstance(self.response_token_count, Integral)
            or int(self.token_count) < 1
            or not 1 <= int(self.response_token_count) <= int(self.token_count)
        ):
            raise ValueError("token counts are invalid")
        object.__setattr__(self, "layer_depths", depths)
        object.__setattr__(self, "response_mean", mean)
        object.__setattr__(self, "response_last", last)
        object.__setattr__(self, "first_error", int(self.first_error))
        object.__setattr__(self, "token_count", int(self.token_count))
        object.__setattr__(
            self, "response_token_count", int(self.response_token_count)
        )

    def representation(self, name: str) -> np.ndarray:
        if name == "response_mean":
            return self.response_mean
        if name == "response_last":
            return self.response_last
        raise ValueError("representation must be response_mean or response_last")


class LowRankShrunkMahalanobis:
    """Exact full-covariance shrinkage scored through the Woodbury identity.

    The implementation avoids an unreported random projection and never forms
    the feature-by-feature inverse.  It is therefore practical for 4096-wide
    LLM states when the normal reference cohort is comparatively small.
    """

    def __init__(
        self,
        *,
        shrinkage: float = 0.1,
        regularization: float = 1e-8,
    ) -> None:
        if (
            isinstance(shrinkage, bool)
            or not isinstance(shrinkage, Real)
            or not np.isfinite(float(shrinkage))
            or not 0.0 <= float(shrinkage) <= 1.0
        ):
            raise ValueError("shrinkage must be finite and lie in [0, 1]")
        if (
            isinstance(regularization, bool)
            or not isinstance(regularization, Real)
            or not np.isfinite(float(regularization))
            or float(regularization) <= 0.0
        ):
            raise ValueError("regularization must be finite and positive")
        self.shrinkage = float(shrinkage)
        self.regularization = float(regularization)
        self.mean_: np.ndarray | None = None
        self.centered_reference_: np.ndarray | None = None
        self.cholesky_: np.ndarray | None = None
        self.beta_: float | None = None
        self.lambda_: float | None = None

    def fit(self, reference: np.ndarray) -> "LowRankShrunkMahalanobis":
        values = np.asarray(reference, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("reference must have shape [at least 2 rows, features]")
        if not np.isfinite(values).all():
            raise ValueError("reference must contain only finite values")
        mean = values.mean(axis=0)
        centered = values - mean
        denominator = values.shape[0] - 1
        isotropic_scale = float(np.sum(centered * centered) / denominator / values.shape[1])
        target_scale = max(isotropic_scale, self.regularization)
        beta = (1.0 - self.shrinkage) / denominator
        diagonal = self.shrinkage * target_scale + self.regularization
        gram = np.eye(values.shape[0], dtype=np.float64)
        if beta:
            gram += (beta / diagonal) * (centered @ centered.T)
        self.mean_ = mean
        self.centered_reference_ = centered
        self.cholesky_ = np.linalg.cholesky(gram)
        self.beta_ = float(beta)
        self.lambda_ = float(diagonal)
        return self

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        if (
            self.mean_ is None
            or self.centered_reference_ is None
            or self.cholesky_ is None
            or self.beta_ is None
            or self.lambda_ is None
        ):
            raise RuntimeError("fit must be called before score_samples")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.mean_.shape[0]:
            raise ValueError("values must align with the fitted feature dimension")
        if not np.isfinite(array).all():
            raise ValueError("values must contain only finite values")
        differences = array - self.mean_
        squared_norm = np.einsum("ni,ni->n", differences, differences)
        scores = squared_norm / self.lambda_
        if self.beta_:
            projected = differences @ self.centered_reference_.T
            solved = np.linalg.solve(self.cholesky_, projected.T)
            correction = np.einsum("in,in->n", solved, solved)
            scores -= (self.beta_ / (self.lambda_**2)) * correction
        return np.maximum(scores, 0.0)


def _empirical_percentile(
    sorted_reference: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(sorted_reference, dtype=np.float64)
    scores = np.asarray(values, dtype=np.float64)
    if reference.ndim != 1 or not len(reference):
        raise ValueError("sorted_reference must be a non-empty vector")
    if scores.ndim != 1:
        raise ValueError("values must be a vector")
    return np.searchsorted(reference, scores, side="right") / len(reference)


@dataclass(frozen=True)
class GhostScores:
    trace_ids: tuple[str, ...]
    layer_depths: tuple[int, ...]
    layer_distances: np.ndarray
    layer_percentiles: np.ndarray
    mid_fused_percentile: np.ndarray
    final_percentile: np.ndarray


class GhostMahalanobisEnsemble:
    """Fixed six-mid-layer, normal-reference Mahalanobis detector."""

    def __init__(
        self,
        *,
        mid_depths: Sequence[int],
        final_depth: int,
        shrinkage: float = 0.1,
        regularization: float = 1e-8,
    ) -> None:
        middle = tuple(int(depth) for depth in mid_depths)
        if (
            not middle
            or len(set(middle)) != len(middle)
            or any(depth < 1 for depth in middle)
            or tuple(sorted(middle)) != middle
        ):
            raise ValueError("mid_depths must be unique increasing positive depths")
        if (
            isinstance(final_depth, bool)
            or not isinstance(final_depth, Integral)
            or int(final_depth) <= middle[-1]
        ):
            raise ValueError("final_depth must be an integer after every middle depth")
        self.mid_depths = middle
        self.final_depth = int(final_depth)
        self.layer_depths = (*middle, self.final_depth)
        self.shrinkage = float(shrinkage)
        self.regularization = float(regularization)
        self.models_: tuple[LowRankShrunkMahalanobis, ...] | None = None
        self.reference_layer_scores_: tuple[np.ndarray, ...] | None = None
        self.reference_fused_scores_: np.ndarray | None = None
        self.representation_: str | None = None
        self.reference_trace_ids_: tuple[str, ...] | None = None

    def _matrix(
        self,
        traces: Sequence[GhostTraceEmbedding],
        *,
        representation: str,
    ) -> np.ndarray:
        if not traces:
            raise ValueError("at least one trace is required")
        for trace in traces:
            if trace.layer_depths != self.layer_depths:
                raise ValueError("trace layer depths do not match the detector")
        values = np.stack(
            [trace.representation(representation) for trace in traces],
            axis=0,
        )
        if values.ndim != 3:
            raise AssertionError("validated trace representations must be rank three")
        return values

    def fit(
        self,
        references: Sequence[GhostTraceEmbedding],
        *,
        representation: str,
    ) -> "GhostMahalanobisEnsemble":
        traces = tuple(references)
        if len(traces) < 2:
            raise ValueError("at least two normal reference traces are required")
        if any(trace.response_label != 0 for trace in traces):
            raise ValueError("every fit trace must be an explicitly normal reference")
        if len({trace.trace_id for trace in traces}) != len(traces):
            raise ValueError("normal reference trace IDs must be unique")
        values = self._matrix(traces, representation=representation)
        models = []
        distances = np.empty((len(traces), len(self.layer_depths)), dtype=np.float64)
        sorted_scores = []
        for layer_index in range(len(self.layer_depths)):
            model = LowRankShrunkMahalanobis(
                shrinkage=self.shrinkage,
                regularization=self.regularization,
            ).fit(values[:, layer_index, :])
            layer_scores = model.score_samples(values[:, layer_index, :])
            distances[:, layer_index] = layer_scores
            models.append(model)
            sorted_scores.append(np.sort(layer_scores))
        layer_percentiles = np.column_stack(
            [
                _empirical_percentile(sorted_scores[index], distances[:, index])
                for index in range(len(self.layer_depths))
            ]
        )
        fused = layer_percentiles[:, : len(self.mid_depths)].mean(axis=1)
        self.models_ = tuple(models)
        self.reference_layer_scores_ = tuple(sorted_scores)
        self.reference_fused_scores_ = np.sort(fused)
        self.representation_ = representation
        self.reference_trace_ids_ = tuple(trace.trace_id for trace in traces)
        return self

    def score(
        self,
        traces: Sequence[GhostTraceEmbedding],
    ) -> GhostScores:
        if (
            self.models_ is None
            or self.reference_layer_scores_ is None
            or self.reference_fused_scores_ is None
            or self.representation_ is None
        ):
            raise RuntimeError("fit must be called before score")
        examples = tuple(traces)
        values = self._matrix(examples, representation=self.representation_)
        distances = np.column_stack(
            [
                model.score_samples(values[:, index, :])
                for index, model in enumerate(self.models_)
            ]
        )
        percentiles = np.column_stack(
            [
                _empirical_percentile(reference, distances[:, index])
                for index, reference in enumerate(self.reference_layer_scores_)
            ]
        )
        fused = percentiles[:, : len(self.mid_depths)].mean(axis=1)
        return GhostScores(
            trace_ids=tuple(trace.trace_id for trace in examples),
            layer_depths=self.layer_depths,
            layer_distances=distances,
            layer_percentiles=percentiles,
            mid_fused_percentile=_empirical_percentile(
                self.reference_fused_scores_,
                fused,
            ),
            final_percentile=percentiles[:, -1],
        )
