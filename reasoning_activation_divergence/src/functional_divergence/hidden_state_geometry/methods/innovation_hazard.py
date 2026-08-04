from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd

from ...progress import NullProgress
from ..data import load_step_end_states
from ..method import FoldInput, MethodFoldResult
from ..model import RegularizedLogistic
from ..preprocessing import FiniteStandardizer, domain_group_balanced_weights
from ..registry import ContrastSpec, register_method
from ..tasks import TaskExample, nuisance_features, visible_output_steps

ARM_COMPONENTS = {
    "nuisance": ("nuisance",),
    "output_only": ("nuisance", "output"),
    "hidden_only": ("nuisance", "hidden"),
    "innovation_only": ("nuisance", "innovation"),
    "output_plus_hidden": ("nuisance", "output", "hidden"),
    "output_plus_innovation": ("nuisance", "output", "innovation"),
}


@dataclass(frozen=True)
class InnovationHazardConfig:
    source_layer: int = 14
    destination_layer: int = 16
    rank: int = 8
    normal_ridge_alpha: float = 10.0
    covariance_shrinkage: float = 0.1
    l2: float = 0.1
    max_iter: int = 2000

    def __post_init__(self) -> None:
        integer_fields = ("source_layer", "destination_layer", "rank", "max_iter")
        if any(
            isinstance(getattr(self, name), (bool, np.bool_))
            or not isinstance(getattr(self, name), (int, np.integer))
            for name in integer_fields
        ):
            raise ValueError("layers, rank, and max_iter must be integers")
        if self.source_layer == self.destination_layer:
            raise ValueError("source and destination layers must differ")
        if self.rank < 1 or self.max_iter < 1:
            raise ValueError("rank and max_iter must be positive")
        if not np.isfinite(self.normal_ridge_alpha) or self.normal_ridge_alpha <= 0:
            raise ValueError("normal_ridge_alpha must be finite and positive")
        if not np.isfinite(self.l2) or self.l2 <= 0:
            raise ValueError("l2 must be finite and positive")
        if (
            not np.isfinite(self.covariance_shrinkage)
            or not 0.0 <= self.covariance_shrinkage <= 1.0
        ):
            raise ValueError("covariance_shrinkage must lie in [0, 1]")


@dataclass(frozen=True)
class _BoundaryRows:
    nuisance: np.ndarray
    output: np.ndarray
    source: np.ndarray
    destination: np.ndarray
    domains: np.ndarray


@dataclass(frozen=True)
class _Projection:
    center: np.ndarray
    components: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.center) @ self.components.T) / self.scale


@dataclass(frozen=True)
class _HazardArm:
    probability: np.ndarray
    model: RegularizedLogistic
    scaler: FiniteStandardizer


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(np.asarray(values, dtype=np.float64), axis=0, weights=weights)


def _fit_projection(
    values: np.ndarray,
    weights: np.ndarray,
    rank: int,
    seed: int,
) -> _Projection:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) <= rank or x.shape[1] < rank:
        raise ValueError(
            f"rank={rank} needs more than {rank} normal rows and at least {rank} hidden dimensions"
        )
    center = _weighted_mean(x, weights)
    centered = x - center
    normalized_weights = weights / weights.sum()
    weighted = centered * np.sqrt(normalized_weights)[:, None]
    _, _, components = randomized_svd(
        weighted,
        n_components=rank,
        n_iter=5,
        random_state=seed,
    )
    encoded = centered @ components.T
    scale = np.sqrt(np.average(encoded**2, axis=0, weights=weights))
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return _Projection(center=center, components=components, scale=scale)


def _finite_column_mean(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    total = np.where(finite, values, 0.0).sum(axis=0)
    return np.divide(
        total,
        count,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=count > 0,
    )


def _past_output(example: TaskExample) -> np.ndarray:
    values = np.asarray(visible_output_steps(example), dtype=np.float64)
    if values.ndim != 2 or len(values) < 1:
        raise ValueError("strict-prefix rows need at least one completed output step")
    return np.concatenate([values[-1], _finite_column_mean(values)])


def _layer_index(example: TaskExample, layer: int) -> int:
    matches = np.flatnonzero(example.sample.layer_ids == layer)
    if len(matches) != 1:
        raise ValueError(
            f"chain {example.sample.chain_id}: layer {layer} is not stored exactly once"
        )
    return int(matches[0])


def _boundary_rows(
    examples: tuple[TaskExample, ...],
    source_layer: int,
    destination_layer: int,
) -> _BoundaryRows:
    nuisance_rows = []
    output_rows = []
    source_rows = []
    destination_rows = []
    domains = []
    last_sample_key: tuple[str, int, str] | None = None
    last_states: np.ndarray | None = None
    last_source_index: int | None = None
    last_destination_index: int | None = None

    for example in examples:
        if (
            example.boundary_step is None
            or example.visible_steps != example.boundary_step
        ):
            raise ValueError("innovation_hazard only accepts strict_prefix examples")
        chain_id = int(example.sample.chain_id)
        sample_key = (
            example.sample.dataset,
            chain_id,
            str(example.sample.state_path),
        )
        if sample_key != last_sample_key:
            last_states = load_step_end_states(example.sample)
            last_source_index = _layer_index(example, source_layer)
            last_destination_index = _layer_index(example, destination_layer)
            last_sample_key = sample_key
        if (
            last_states is None
            or last_source_index is None
            or last_destination_index is None
        ):
            raise RuntimeError("boundary-state cache was not initialized")
        state_index = example.visible_steps - 1
        _, nuisance = nuisance_features(example)
        nuisance_rows.append(nuisance)
        output_rows.append(_past_output(example))
        source_rows.append(last_states[state_index, last_source_index])
        destination_rows.append(last_states[state_index, last_destination_index])
        domains.append(example.sample.dataset)

    return _BoundaryRows(
        nuisance=np.asarray(nuisance_rows, dtype=np.float64),
        output=np.asarray(output_rows, dtype=np.float64),
        source=np.asarray(source_rows, dtype=np.float32),
        destination=np.asarray(destination_rows, dtype=np.float32),
        domains=np.asarray(domains, dtype=object),
    )


def _design_matrices(
    rows: _BoundaryRows,
    hidden: np.ndarray,
    innovation: np.ndarray,
) -> dict[str, np.ndarray]:
    components = {
        "nuisance": rows.nuisance,
        "output": rows.output,
        "hidden": hidden,
        "innovation": innovation,
    }
    return {
        arm: np.column_stack([components[name] for name in names])
        for arm, names in ARM_COMPONENTS.items()
    }


def _optimizer_diagnostics(model: RegularizedLogistic) -> dict[str, Any]:
    return {
        "iterations": model.iterations_,
        "objective": model.objective_,
        "gradient_inf_norm": model.gradient_inf_norm_,
        "message": model.message_,
    }


@register_method(
    "innovation_hazard",
    contrasts=(
        ContrastSpec(
            "output_summary_given_nuisance_nll",
            "nuisance",
            "output_only",
            "past output-summary increment beyond position and length controls",
        ),
        ContrastSpec(
            "hidden_given_output_summary_nll",
            "output_only",
            "output_plus_hidden",
            "destination-state increment beyond past output summaries",
        ),
        ContrastSpec(
            "innovation_given_output_summary_nll",
            "output_only",
            "output_plus_innovation",
            "normal-flow innovation increment beyond past output summaries",
        ),
        ContrastSpec(
            "innovation_vs_capacity_matched_hidden_nll",
            "output_plus_hidden",
            "output_plus_innovation",
            "equal-rank structured innovation versus destination hidden state",
        ),
    ),
    arm_definitions={
        "nuisance": "boundary position and completed-prefix length controls",
        "output_only": "nuisance plus last and running-mean past output summaries",
        "hidden_only": "nuisance plus rank-q destination-layer state",
        "innovation_only": "nuisance plus rank-q whitened normal-flow residual",
        "output_plus_hidden": "past output summaries plus rank-q destination state",
        "output_plus_innovation": (
            "past output summaries plus rank-q whitened normal-flow residual"
        ),
    },
    default_config=InnovationHazardConfig,
)
class InnovationHazard:
    """Prospective first-error hazard from deviations from normal cross-layer flow."""

    def __init__(self, config: InnovationHazardConfig | Mapping[str, Any]) -> None:
        if isinstance(config, Mapping):
            config = InnovationHazardConfig(**dict(config))
        if not isinstance(config, InnovationHazardConfig):
            raise TypeError(
                "innovation_hazard config must be a config object or mapping"
            )
        self.config = config

    def _fit_arm(
        self,
        train: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        test: np.ndarray,
    ) -> _HazardArm:
        scaler = FiniteStandardizer().fit(train, weights)
        train_scaled = scaler.transform(train)
        test_scaled = scaler.transform(test)
        model = RegularizedLogistic(
            l2=self.config.l2,
            max_iter=self.config.max_iter,
        ).fit(train_scaled, labels, weights)
        return _HazardArm(model.predict_proba(test_scaled), model, scaler)

    def fit_predict(self, fold: FoldInput) -> MethodFoldResult:
        if fold.task_name != "strict_prefix":
            raise ValueError("innovation_hazard requires the strict_prefix task")
        reporter = fold.progress or NullProgress()
        reporter.stage("encode", f"{fold.task_name}: prospective boundary states")
        train_rows = _boundary_rows(
            fold.train_examples,
            self.config.source_layer,
            self.config.destination_layer,
        )
        test_rows = _boundary_rows(
            fold.test_examples,
            self.config.source_layer,
            self.config.destination_layer,
        )
        weights = domain_group_balanced_weights(
            train_rows.domains,
            fold.train_groups,
        )
        normal = fold.train_labels == 0
        normal_weights = domain_group_balanced_weights(
            train_rows.domains[normal],
            fold.train_groups[normal],
        )
        if int(normal.sum()) <= self.config.rank:
            raise ValueError(
                "too few correct-prefix rows for the requested innovation rank"
            )

        reporter.stage("fit", f"{fold.task_name}: normal cross-layer flow")
        source_projection = _fit_projection(
            train_rows.source[normal],
            normal_weights,
            self.config.rank,
            fold.seed,
        )
        destination_projection = _fit_projection(
            train_rows.destination[normal],
            normal_weights,
            self.config.rank,
            fold.seed + 1,
        )
        source_train = source_projection.transform(train_rows.source)
        source_test = source_projection.transform(test_rows.source)
        destination_train = destination_projection.transform(train_rows.destination)
        destination_test = destination_projection.transform(test_rows.destination)

        normal_context = np.column_stack(
            [train_rows.nuisance[normal], train_rows.output[normal]]
        )
        context_scaler = FiniteStandardizer().fit(normal_context, normal_weights)
        train_context = context_scaler.transform(
            np.column_stack([train_rows.nuisance, train_rows.output])
        )
        test_context = context_scaler.transform(
            np.column_stack([test_rows.nuisance, test_rows.output])
        )
        normal_design = np.column_stack([source_train[normal], train_context[normal]])
        flow = Ridge(alpha=self.config.normal_ridge_alpha, fit_intercept=True).fit(
            normal_design,
            destination_train[normal],
            sample_weight=normal_weights,
        )
        train_residual = destination_train - flow.predict(
            np.column_stack([source_train, train_context])
        )
        test_residual = destination_test - flow.predict(
            np.column_stack([source_test, test_context])
        )
        residual_center = _weighted_mean(train_residual[normal], normal_weights)
        centered_normal = train_residual[normal] - residual_center
        covariance = (
            centered_normal.T
            @ (centered_normal * normal_weights[:, None])
            / normal_weights.sum()
        )
        diagonal = np.diag(np.diag(covariance))
        shrinkage = self.config.covariance_shrinkage
        covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        positive = eigenvalue[eigenvalue > 0]
        scale = float(np.median(positive)) if len(positive) else 1.0
        floor = max(scale * 1e-6, 1e-8)
        eigenvalue = np.maximum(eigenvalue, floor)
        whitener = eigenvector @ np.diag(1.0 / np.sqrt(eigenvalue))
        innovation_train = (train_residual - residual_center) @ whitener
        innovation_test = (test_residual - residual_center) @ whitener

        train_designs = _design_matrices(
            train_rows,
            destination_train,
            innovation_train,
        )
        test_designs = _design_matrices(
            test_rows,
            destination_test,
            innovation_test,
        )
        reporter.stage("fit", f"{fold.task_name}: discrete first-error hazard arms")
        fitted = {
            arm: self._fit_arm(
                train_designs[arm],
                fold.train_labels,
                weights,
                test_designs[arm],
            )
            for arm in ARM_COMPONENTS
        }

        factors: dict[str, np.ndarray] = {
            "source_projection.center": source_projection.center,
            "source_projection.components": source_projection.components,
            "source_projection.scale": source_projection.scale,
            "destination_projection.center": destination_projection.center,
            "destination_projection.components": destination_projection.components,
            "destination_projection.scale": destination_projection.scale,
            "normal_flow.coefficients": np.asarray(flow.coef_),
            "normal_flow.intercept": np.asarray(flow.intercept_),
            "normal_flow.residual_center": residual_center,
            "normal_flow.whitener": whitener,
            "normal_context.center": np.asarray(context_scaler.center_),
            "normal_context.scale": np.asarray(context_scaler.scale_),
        }
        for arm, result in fitted.items():
            factors[f"{arm}.coefficients"] = result.model.coefficients
            factors[f"{arm}.scaler_center"] = np.asarray(result.scaler.center_)
            factors[f"{arm}.scaler_scale"] = np.asarray(result.scaler.scale_)

        return MethodFoldResult(
            probabilities={arm: result.probability for arm, result in fitted.items()},
            diagnostics={
                "target": "discrete_time_first_error_hazard",
                "source_state_timing": "end_of_step_t_minus_1",
                "post_error_policy": "censored",
                "step0_policy": "left_truncated_without_prompt_end_state",
                "layers": [
                    self.config.source_layer,
                    self.config.destination_layer,
                ],
                "rank": self.config.rank,
                "normal_bank": "outer_train_correct_prefix_rows_only",
                "normal_bank_rows": int(normal.sum()),
                "normal_flow_predictors": (
                    "source_state_plus_online_nuisance_and_past_output"
                ),
                "training_weights": "equal domain, then equal problem group",
                "arm_feature_dimensions": {
                    arm: matrix.shape[1] for arm, matrix in train_designs.items()
                },
                "comparison_design": (
                    "capacity_matched_destination_state_vs_whitened_flow_innovation"
                ),
                "converged": {
                    arm: result.model.converged_ for arm, result in fitted.items()
                },
                "optimizer": {
                    arm: _optimizer_diagnostics(result.model)
                    for arm, result in fitted.items()
                },
            },
            factors=factors,
        )
