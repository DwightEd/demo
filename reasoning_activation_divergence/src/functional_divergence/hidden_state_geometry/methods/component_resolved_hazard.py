from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from ...progress import NullProgress
from ..component_contract import ComponentStepArtifact
from ..component_features import load_component_step_selection
from ..method import FoldInput, MethodFoldResult
from ..model import RegularizedLogistic
from ..preprocessing import FiniteStandardizer, domain_group_balanced_weights
from ..registry import ContrastSpec, register_method
from ..tasks import TaskExample, nuisance_features, visible_output_steps

SOURCE_BUCKETS = ("prompt", "earlier_steps", "previous_step", "current_step")
COMPONENT_CHANNELS = ("attention", "mlp", "propagation")

ARM_COMPONENTS = {
    "nuisance": ("nuisance",),
    "output_only": ("nuisance", "output"),
    "hidden_post": ("nuisance", "output", "hidden_post"),
    "attention_only": ("nuisance", "output", "attention"),
    "mlp_only": ("nuisance", "output", "mlp"),
    "propagation_only": ("nuisance", "output", "propagation"),
    "components_all": ("nuisance", "output", "components"),
    "components_plus_hidden": ("nuisance", "output", "components", "hidden_post"),
    "all_minus_attention": ("nuisance", "output", "all_minus_attention"),
    "all_minus_mlp": ("nuisance", "output", "all_minus_mlp"),
    "all_minus_propagation": ("nuisance", "output", "all_minus_propagation"),
}


@dataclass(frozen=True)
class ComponentResolvedHazardConfig:
    attention_rank: int = 4
    mlp_rank: int = 4
    propagation_rank: int = 4
    pre_context_rank: int = 4
    normal_ridge_alpha: float = 10.0
    l2: float = 0.1
    max_iter: int = 2000

    def __post_init__(self) -> None:
        integer_fields = (
            "attention_rank",
            "mlp_rank",
            "propagation_rank",
            "pre_context_rank",
            "max_iter",
        )
        if any(
            isinstance(getattr(self, name), (bool, np.bool_))
            or not isinstance(getattr(self, name), (int, np.integer))
            for name in integer_fields
        ):
            raise ValueError("ranks and max_iter must be integers")
        if any(getattr(self, name) < 1 for name in integer_fields):
            raise ValueError("ranks and max_iter must be positive")
        if not np.isfinite(self.normal_ridge_alpha) or self.normal_ridge_alpha <= 0:
            raise ValueError("normal_ridge_alpha must be finite and positive")
        if not np.isfinite(self.l2) or self.l2 <= 0:
            raise ValueError("l2 must be finite and positive")

    @property
    def total_component_rank(self) -> int:
        return int(self.attention_rank + self.mlp_rank + self.propagation_rank)


@dataclass(frozen=True)
class _ComponentSchema:
    residual_layers: tuple[int, ...]
    attention_layers: tuple[int, ...]
    mlp_layers: tuple[int, ...]
    hidden_size: int
    output_feature_names: tuple[str, ...]

    @property
    def attention_raw_dim(self) -> int:
        return len(SOURCE_BUCKETS) * len(self.attention_layers) * self.hidden_size

    @property
    def mlp_raw_dim(self) -> int:
        return len(self.mlp_layers) * self.hidden_size

    @property
    def residual_raw_dim(self) -> int:
        return len(self.residual_layers) * self.hidden_size

    @property
    def output_dim(self) -> int:
        return 2 * len(self.output_feature_names)


@dataclass(frozen=True)
class _ComponentRows:
    nuisance: np.ndarray
    output: np.ndarray
    attention: np.ndarray
    mlp: np.ndarray
    propagation: np.ndarray
    hidden_post: np.ndarray
    pre_context: np.ndarray
    domains: np.ndarray
    schema: _ComponentSchema


@dataclass(frozen=True)
class _Projection:
    center: np.ndarray
    components: np.ndarray
    scale: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.center.shape[0]:
            raise ValueError("projection input dimension does not match fitted center")
        return ((x - self.center) @ self.components.T) / self.scale


@dataclass(frozen=True)
class _Innovation:
    train: np.ndarray
    test: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    residual_center: np.ndarray
    residual_scale: np.ndarray


@dataclass(frozen=True)
class _FittedArm:
    probability: np.ndarray
    model: RegularizedLogistic
    scaler: FiniteStandardizer


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(np.asarray(values, dtype=np.float64), axis=0, weights=weights)


def _fit_weighted_projection(
    values: np.ndarray,
    weights: np.ndarray,
    rank: int,
    name: str,
) -> _Projection:
    x = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or weight.shape != (len(x),) or len(x) == 0:
        raise ValueError(f"{name} projection expects aligned non-empty rows")
    if np.any(weight < 0) or weight.sum() <= 0:
        raise ValueError(f"{name} projection weights must be nonnegative and nonzero")
    if min(x.shape) < rank:
        raise ValueError(
            f"{name} rank={rank} requires at least {rank} rows and raw dimensions"
        )
    center = _weighted_mean(x, weight)
    centered = x - center
    normalized = weight / weight.sum()
    weighted = centered * np.sqrt(normalized)[:, None]
    _, singular_values, components = np.linalg.svd(weighted, full_matrices=False)
    selected = components[:rank]
    encoded = centered @ selected.T
    scale = np.sqrt(np.average(encoded**2, axis=0, weights=weight))
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    variance = singular_values**2
    selected_variance = variance[:rank]
    total_variance = float(variance.sum())
    ratio = (
        selected_variance / total_variance
        if total_variance > 0
        else np.zeros(rank, dtype=np.float64)
    )
    return _Projection(
        center=center,
        components=selected,
        scale=scale,
        explained_variance=selected_variance,
        explained_variance_ratio=ratio,
    )


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


def _current_and_running_output(example: TaskExample) -> np.ndarray:
    values = np.asarray(visible_output_steps(example), dtype=np.float64)
    if values.ndim != 2 or len(values) < 1:
        raise ValueError("post_step rows need at least one completed output step")
    return np.concatenate([values[-1], _finite_column_mean(values)])


def _schema_for(
    artifact: ComponentStepArtifact,
    example: TaskExample,
) -> _ComponentSchema:
    hidden_size = int(artifact.resid_boundary.shape[2])
    return _ComponentSchema(
        residual_layers=tuple(int(value) for value in artifact.residual_layers),
        attention_layers=tuple(int(value) for value in artifact.attention_layers),
        mlp_layers=tuple(int(value) for value in artifact.mlp_layers),
        hidden_size=hidden_size,
        output_feature_names=tuple(example.sample.output_feature_names),
    )


def _empty_rows(schema: _ComponentSchema) -> _ComponentRows:
    return _ComponentRows(
        nuisance=np.zeros((0, 3), dtype=np.float64),
        output=np.zeros((0, schema.output_dim), dtype=np.float64),
        attention=np.zeros((0, schema.attention_raw_dim), dtype=np.float64),
        mlp=np.zeros((0, schema.mlp_raw_dim), dtype=np.float64),
        propagation=np.zeros((0, schema.residual_raw_dim), dtype=np.float64),
        hidden_post=np.zeros((0, schema.residual_raw_dim), dtype=np.float64),
        pre_context=np.zeros((0, schema.residual_raw_dim), dtype=np.float64),
        domains=np.asarray([], dtype=object),
        schema=schema,
    )


def _attention_source_buckets(
    artifact: ComponentStepArtifact,
    current_step: int,
) -> np.ndarray:
    """Aggregate residual-space attention writes into fixed semantic source buckets."""
    step = int(current_step)
    if step < 0 or step >= artifact.attn_msg_resid_by_source.shape[0]:
        raise ValueError("current_step is outside the component artifact")
    messages = np.asarray(artifact.attn_msg_resid_by_source[step], dtype=np.float64)
    source_ids = np.asarray(artifact.source_step_id[step], dtype=np.int64)
    mask = np.asarray(artifact.source_mask[step], dtype=bool)
    buckets = np.zeros(
        (len(SOURCE_BUCKETS), messages.shape[0], messages.shape[2]),
        dtype=np.float64,
    )
    for block, source_step in enumerate(source_ids):
        if not mask[block]:
            continue
        source = int(source_step)
        if source == -1:
            bucket = 0
        elif source < step - 1:
            bucket = 1
        elif source == step - 1:
            bucket = 2
        elif source == step:
            bucket = 3
        else:
            raise ValueError("source_step_id cannot reference a future step")
        buckets[bucket] += messages[:, block, :]
    return buckets


def _collect_component_rows(
    examples: tuple[TaskExample, ...],
    expected_schema: _ComponentSchema | None = None,
) -> _ComponentRows:
    if not examples:
        if expected_schema is None:
            raise ValueError("component_resolved_hazard needs at least one row")
        return _empty_rows(expected_schema)

    nuisance_rows = []
    output_rows = []
    attention_rows = []
    mlp_rows = []
    propagation_rows = []
    hidden_post_rows = []
    pre_context_rows = []
    domains = []
    schema = expected_schema
    for example in examples:
        selection = load_component_step_selection(example)
        artifact = selection.artifact
        current_schema = _schema_for(artifact, example)
        if schema is None:
            schema = current_schema
        elif current_schema != schema:
            raise ValueError(
                "component schema/layer grid mismatch across fold rows: "
                f"{current_schema!r} != {schema!r}"
            )
        step = selection.current_step
        _, nuisance = nuisance_features(example)
        nuisance_rows.append(nuisance)
        output_rows.append(_current_and_running_output(example))
        attention_rows.append(_attention_source_buckets(artifact, step).reshape(-1))
        mlp_rows.append(
            np.asarray(artifact.mlp_out_step[step], dtype=np.float64).reshape(-1)
        )
        propagation_rows.append(
            (
                np.asarray(artifact.resid_boundary[step + 1], dtype=np.float64)
                - np.asarray(artifact.resid_boundary[step], dtype=np.float64)
            ).reshape(-1)
        )
        hidden_post_rows.append(
            np.asarray(artifact.resid_boundary[step + 1], dtype=np.float64).reshape(-1)
        )
        pre_context_rows.append(
            np.asarray(artifact.resid_boundary[step], dtype=np.float64).reshape(-1)
        )
        domains.append(example.sample.dataset)

    if schema is None:
        raise RuntimeError("component schema was not initialized")
    return _ComponentRows(
        nuisance=np.asarray(nuisance_rows, dtype=np.float64),
        output=np.asarray(output_rows, dtype=np.float64),
        attention=np.asarray(attention_rows, dtype=np.float64),
        mlp=np.asarray(mlp_rows, dtype=np.float64),
        propagation=np.asarray(propagation_rows, dtype=np.float64),
        hidden_post=np.asarray(hidden_post_rows, dtype=np.float64),
        pre_context=np.asarray(pre_context_rows, dtype=np.float64),
        domains=np.asarray(domains, dtype=object),
        schema=schema,
    )


def _fit_normal_map(
    train_channel: np.ndarray,
    test_channel: np.ndarray,
    train_context: np.ndarray,
    test_context: np.ndarray,
    normal_mask: np.ndarray,
    normal_weights: np.ndarray,
    alpha: float,
) -> _Innovation:
    channel = np.asarray(train_channel, dtype=np.float64)
    test = np.asarray(test_channel, dtype=np.float64)
    context = np.asarray(train_context, dtype=np.float64)
    test_context = np.asarray(test_context, dtype=np.float64)
    if channel.ndim != 2 or test.shape[1] != channel.shape[1]:
        raise ValueError("normal map expects aligned train/test channel matrices")
    coefficients = np.zeros((channel.shape[1], context.shape[1]), dtype=np.float64)
    intercept = np.zeros(channel.shape[1], dtype=np.float64)
    normal_context = context[normal_mask]
    for index in range(channel.shape[1]):
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(
            normal_context,
            channel[normal_mask, index],
            sample_weight=normal_weights,
        )
        coefficients[index] = np.asarray(model.coef_, dtype=np.float64)
        intercept[index] = float(model.intercept_)
    train_prediction = context @ coefficients.T + intercept
    test_prediction = test_context @ coefficients.T + intercept
    train_residual = channel - train_prediction
    test_residual = test - test_prediction
    residual_center = _weighted_mean(train_residual[normal_mask], normal_weights)
    centered_normal = train_residual[normal_mask] - residual_center
    residual_scale = np.sqrt(
        np.average(centered_normal**2, axis=0, weights=normal_weights)
    )
    residual_scale[~np.isfinite(residual_scale) | (residual_scale < 1e-8)] = 1.0
    return _Innovation(
        train=(train_residual - residual_center) / residual_scale,
        test=(test_residual - residual_center) / residual_scale,
        coefficients=coefficients,
        intercept=intercept,
        residual_center=residual_center,
        residual_scale=residual_scale,
    )


def _component_slots(
    attention: np.ndarray,
    mlp: np.ndarray,
    propagation: np.ndarray,
    *,
    zero_attention: bool = False,
    zero_mlp: bool = False,
    zero_propagation: bool = False,
) -> np.ndarray:
    return np.column_stack(
        [
            np.zeros_like(attention) if zero_attention else attention,
            np.zeros_like(mlp) if zero_mlp else mlp,
            np.zeros_like(propagation) if zero_propagation else propagation,
        ]
    )


def _design_matrices(
    rows: _ComponentRows,
    attention: np.ndarray,
    mlp: np.ndarray,
    propagation: np.ndarray,
    hidden_post: np.ndarray,
) -> dict[str, np.ndarray]:
    controls = np.column_stack([rows.nuisance, rows.output])
    components = _component_slots(attention, mlp, propagation)
    zeros_attention = np.zeros_like(attention)
    zeros_mlp = np.zeros_like(mlp)
    zeros_propagation = np.zeros_like(propagation)
    return {
        "nuisance": rows.nuisance,
        "output_only": controls,
        "hidden_post": np.column_stack([controls, hidden_post]),
        "attention_only": np.column_stack(
            [controls, _component_slots(attention, zeros_mlp, zeros_propagation)]
        ),
        "mlp_only": np.column_stack(
            [controls, _component_slots(zeros_attention, mlp, zeros_propagation)]
        ),
        "propagation_only": np.column_stack(
            [controls, _component_slots(zeros_attention, zeros_mlp, propagation)]
        ),
        "components_all": np.column_stack([controls, components]),
        "components_plus_hidden": np.column_stack([controls, components, hidden_post]),
        "all_minus_attention": np.column_stack(
            [
                controls,
                _component_slots(
                    attention,
                    mlp,
                    propagation,
                    zero_attention=True,
                ),
            ]
        ),
        "all_minus_mlp": np.column_stack(
            [
                controls,
                _component_slots(attention, mlp, propagation, zero_mlp=True),
            ]
        ),
        "all_minus_propagation": np.column_stack(
            [
                controls,
                _component_slots(
                    attention,
                    mlp,
                    propagation,
                    zero_propagation=True,
                ),
            ]
        ),
    }


def _optimizer_diagnostics(model: RegularizedLogistic) -> dict[str, Any]:
    return {
        "iterations": model.iterations_,
        "objective": model.objective_,
        "gradient_inf_norm": model.gradient_inf_norm_,
        "message": model.message_,
    }


def _projection_factors(prefix: str, projection: _Projection) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_projection.center": projection.center,
        f"{prefix}_projection.components": projection.components,
        f"{prefix}_projection.scale": projection.scale,
        f"{prefix}_projection.explained_variance": projection.explained_variance,
        f"{prefix}_projection.explained_variance_ratio": (
            projection.explained_variance_ratio
        ),
    }


def _normal_map_factors(prefix: str, innovation: _Innovation) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_normal_map.coefficients": innovation.coefficients,
        f"{prefix}_normal_map.intercept": innovation.intercept,
        f"{prefix}_normal_map.innovation_center": innovation.residual_center,
        f"{prefix}_normal_map.innovation_scale": innovation.residual_scale,
    }


@register_method(
    "component_resolved_hazard",
    contrasts=(
        ContrastSpec(
            "output_summary_given_nuisance_nll",
            "nuisance",
            "output_only",
            "output-summary diagnostic predictive increment beyond position controls",
        ),
        ContrastSpec(
            "components_all_vs_capacity_matched_hidden_nll",
            "hidden_post",
            "components_all",
            (
                "component innovation diagnostic predictive increment versus "
                "capacity-matched post-step hidden state"
            ),
        ),
        ContrastSpec(
            "components_given_capacity_matched_hidden_nll",
            "hidden_post",
            "components_plus_hidden",
            (
                "component innovation diagnostic predictive increment beyond "
                "capacity-matched post-step hidden state"
            ),
        ),
        ContrastSpec(
            "attention_innovation_given_output_nll",
            "output_only",
            "attention_only",
            (
                "attention-message innovation diagnostic predictive increment "
                "beyond output"
            ),
        ),
        ContrastSpec(
            "mlp_innovation_given_output_nll",
            "output_only",
            "mlp_only",
            ("MLP-update innovation diagnostic predictive increment beyond output"),
        ),
        ContrastSpec(
            "propagation_innovation_given_output_nll",
            "output_only",
            "propagation_only",
            (
                "residual-propagation innovation diagnostic predictive increment "
                "beyond output"
            ),
        ),
        ContrastSpec(
            "attention_full_vs_all_minus_nll",
            "all_minus_attention",
            "components_all",
            (
                "attention slot diagnostic predictive increment in fixed-width "
                "full component arm"
            ),
        ),
        ContrastSpec(
            "mlp_full_vs_all_minus_nll",
            "all_minus_mlp",
            "components_all",
            (
                "MLP slot diagnostic predictive increment in fixed-width "
                "full component arm"
            ),
        ),
        ContrastSpec(
            "propagation_full_vs_all_minus_nll",
            "all_minus_propagation",
            "components_all",
            (
                "propagation slot diagnostic predictive increment in fixed-width "
                "full component arm"
            ),
        ),
    ),
    arm_definitions={
        "nuisance": "step index, prefix token count, and current step token length",
        "output_only": "nuisance plus current and running-mean output summaries",
        "hidden_post": (
            "nuisance, output summaries, and rank-matched post-step hidden baseline"
        ),
        "attention_only": (
            "nuisance, output summaries, attention-message innovation slot, and "
            "zeroed non-attention component slots"
        ),
        "mlp_only": (
            "nuisance, output summaries, MLP-update innovation slot, and zeroed "
            "non-MLP component slots"
        ),
        "propagation_only": (
            "nuisance, output summaries, residual-propagation innovation slot, and "
            "zeroed non-propagation component slots"
        ),
        "components_all": (
            "nuisance, output summaries, and attention/MLP/propagation innovation slots"
        ),
        "components_plus_hidden": (
            "nuisance, output summaries, all component innovation slots, and "
            "rank-matched post-step hidden baseline"
        ),
        "all_minus_attention": (
            "components_all layout with the attention innovation slot fixed to zero"
        ),
        "all_minus_mlp": (
            "components_all layout with the MLP innovation slot fixed to zero"
        ),
        "all_minus_propagation": (
            "components_all layout with the propagation innovation slot fixed to zero"
        ),
    },
    default_config=ComponentResolvedHazardConfig,
)
class ComponentResolvedHazard:
    """Post-step first-error diagnosis from component-channel innovations."""

    def __init__(
        self,
        config: ComponentResolvedHazardConfig | Mapping[str, Any],
    ) -> None:
        if isinstance(config, Mapping):
            config = ComponentResolvedHazardConfig(**dict(config))
        if not isinstance(config, ComponentResolvedHazardConfig):
            raise TypeError(
                "component_resolved_hazard config must be a config object or mapping"
            )
        self.config = config

    def _fit_arm(
        self,
        train: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        test: np.ndarray,
    ) -> _FittedArm:
        scaler = FiniteStandardizer().fit(train, weights)
        train_scaled = scaler.transform(train)
        test_scaled = scaler.transform(test)
        model = RegularizedLogistic(
            l2=self.config.l2,
            max_iter=self.config.max_iter,
        ).fit(train_scaled, labels, weights)
        return _FittedArm(model.predict_proba(test_scaled), model, scaler)

    def fit_predict(self, fold: FoldInput) -> MethodFoldResult:
        if fold.task_name != "post_step":
            raise ValueError("component_resolved_hazard requires the post_step task")
        reporter = fold.progress or NullProgress()
        reporter.stage("encode", f"{fold.task_name}: component-resolved rows")
        train_rows = _collect_component_rows(fold.train_examples)
        test_rows = _collect_component_rows(fold.test_examples, train_rows.schema)
        weights = domain_group_balanced_weights(train_rows.domains, fold.train_groups)
        normal = fold.train_labels == 0
        if not np.any(normal):
            raise ValueError("normal-update maps need label-0 training rows")
        normal_weights = domain_group_balanced_weights(
            train_rows.domains[normal],
            fold.train_groups[normal],
        )

        reporter.stage("fit", f"{fold.task_name}: fold-local component projections")
        attention_projection = _fit_weighted_projection(
            train_rows.attention,
            weights,
            self.config.attention_rank,
            "attention",
        )
        mlp_projection = _fit_weighted_projection(
            train_rows.mlp,
            weights,
            self.config.mlp_rank,
            "mlp",
        )
        propagation_projection = _fit_weighted_projection(
            train_rows.propagation,
            weights,
            self.config.propagation_rank,
            "propagation",
        )
        hidden_projection = _fit_weighted_projection(
            train_rows.hidden_post,
            weights,
            self.config.total_component_rank,
            "hidden_post",
        )
        pre_context_projection = _fit_weighted_projection(
            train_rows.pre_context,
            weights,
            self.config.pre_context_rank,
            "pre_context",
        )

        attention_train = attention_projection.transform(train_rows.attention)
        attention_test = attention_projection.transform(test_rows.attention)
        mlp_train = mlp_projection.transform(train_rows.mlp)
        mlp_test = mlp_projection.transform(test_rows.mlp)
        propagation_train = propagation_projection.transform(train_rows.propagation)
        propagation_test = propagation_projection.transform(test_rows.propagation)
        hidden_train = hidden_projection.transform(train_rows.hidden_post)
        hidden_test = hidden_projection.transform(test_rows.hidden_post)
        pre_context_train = pre_context_projection.transform(train_rows.pre_context)
        pre_context_test = pre_context_projection.transform(test_rows.pre_context)

        reporter.stage("fit", f"{fold.task_name}: label-0 normal update maps")
        normal_context_raw = np.column_stack(
            [
                train_rows.nuisance[normal],
                train_rows.output[normal],
                pre_context_train[normal],
            ]
        )
        context_scaler = FiniteStandardizer().fit(normal_context_raw, normal_weights)
        train_context = context_scaler.transform(
            np.column_stack([train_rows.nuisance, train_rows.output, pre_context_train])
        )
        test_context = context_scaler.transform(
            np.column_stack([test_rows.nuisance, test_rows.output, pre_context_test])
        )
        attention_innovation = _fit_normal_map(
            attention_train,
            attention_test,
            train_context,
            test_context,
            normal,
            normal_weights,
            self.config.normal_ridge_alpha,
        )
        mlp_innovation = _fit_normal_map(
            mlp_train,
            mlp_test,
            train_context,
            test_context,
            normal,
            normal_weights,
            self.config.normal_ridge_alpha,
        )
        propagation_innovation = _fit_normal_map(
            propagation_train,
            propagation_test,
            train_context,
            test_context,
            normal,
            normal_weights,
            self.config.normal_ridge_alpha,
        )

        train_designs = _design_matrices(
            train_rows,
            attention_innovation.train,
            mlp_innovation.train,
            propagation_innovation.train,
            hidden_train,
        )
        test_designs = _design_matrices(
            test_rows,
            attention_innovation.test,
            mlp_innovation.test,
            propagation_innovation.test,
            hidden_test,
        )
        reporter.stage("fit", f"{fold.task_name}: component-resolved logistic arms")
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
            "training_weights": np.asarray(weights, dtype=np.float64),
            "normal_training_weights": np.asarray(normal_weights, dtype=np.float64),
            "normal_context.center": np.asarray(context_scaler.center_),
            "normal_context.scale": np.asarray(context_scaler.scale_),
        }
        for name, projection in (
            ("attention", attention_projection),
            ("mlp", mlp_projection),
            ("propagation", propagation_projection),
            ("hidden_post", hidden_projection),
            ("pre_context", pre_context_projection),
        ):
            factors.update(_projection_factors(name, projection))
        for name, innovation in (
            ("attention", attention_innovation),
            ("mlp", mlp_innovation),
            ("propagation", propagation_innovation),
        ):
            factors.update(_normal_map_factors(name, innovation))
        for arm, result in fitted.items():
            factors[f"{arm}.coefficients"] = result.model.coefficients
            factors[f"{arm}.scaler_center"] = np.asarray(result.scaler.center_)
            factors[f"{arm}.scaler_scale"] = np.asarray(result.scaler.scale_)

        total_component_rank = self.config.total_component_rank
        return MethodFoldResult(
            probabilities={arm: result.probability for arm, result in fitted.items()},
            diagnostics={
                "target": "post_step_first_error_diagnosis",
                "target_timing": "post_step_after_current_step",
                "no_post_error_policy": "censored_by_post_step_task",
                "source_bucket_order": SOURCE_BUCKETS,
                "source_bucket_definitions": {
                    "prompt": "source_step_id == -1",
                    "earlier_steps": "source_step_id < current_step - 1",
                    "previous_step": "source_step_id == current_step - 1",
                    "current_step": "source_step_id == current_step",
                },
                "mechanism_signal": (
                    "residual-space attention message writes, not raw attention weights"
                ),
                "ranks": {
                    "attention": self.config.attention_rank,
                    "mlp": self.config.mlp_rank,
                    "propagation": self.config.propagation_rank,
                    "pre_context": self.config.pre_context_rank,
                    "hidden_post": total_component_rank,
                    "total_component": total_component_rank,
                },
                "transform_fit_scope": "outer_training_fold_only",
                "projection_fit_rows": len(fold.train_examples),
                "projection_fit_weights": "domain/problem-group balanced training rows",
                "normal_bank": "outer_train_label_0_rows_only",
                "normal_bank_rows": int(normal.sum()),
                "normal_map_predictors": (
                    "nuisance plus current/running output plus projected "
                    "pre-boundary context"
                ),
                "training_weights": "equal domain, then equal problem group",
                "raw_feature_dimensions": {
                    "nuisance": int(train_rows.nuisance.shape[1]),
                    "output": int(train_rows.output.shape[1]),
                    "attention": int(train_rows.attention.shape[1]),
                    "mlp": int(train_rows.mlp.shape[1]),
                    "propagation": int(train_rows.propagation.shape[1]),
                    "hidden_post": int(train_rows.hidden_post.shape[1]),
                    "pre_context": int(train_rows.pre_context.shape[1]),
                },
                "projected_feature_dimensions": {
                    "attention": int(attention_train.shape[1]),
                    "mlp": int(mlp_train.shape[1]),
                    "propagation": int(propagation_train.shape[1]),
                    "hidden_post": int(hidden_train.shape[1]),
                    "pre_context": int(pre_context_train.shape[1]),
                    "components_all": total_component_rank,
                },
                "arm_feature_dimensions": {
                    arm: int(matrix.shape[1]) for arm, matrix in train_designs.items()
                },
                "capacity_matched_component_slots": {
                    "components_all": total_component_rank,
                    "hidden_post": total_component_rank,
                    "all_minus_attention": total_component_rank,
                    "all_minus_mlp": total_component_rank,
                    "all_minus_propagation": total_component_rank,
                },
                "zero_slot_ablation_policy": (
                    "all-minus and individual component arms keep fixed component "
                    "slots and set excluded slots to exact zero"
                ),
                "projection_parameter_factors": [
                    f"{name}_projection.components"
                    for name in (
                        "attention",
                        "mlp",
                        "propagation",
                        "hidden_post",
                        "pre_context",
                    )
                ],
                "normal_map_parameter_factors": [
                    f"{name}_normal_map.coefficients" for name in COMPONENT_CHANNELS
                ],
                "claim_limitation": (
                    "observational predictive decomposition; activation patching "
                    "required for causal attribution"
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
