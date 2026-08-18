from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from ...progress import NullProgress
from ..method import FoldInput, MethodFoldResult
from ..model import RegularizedLogistic
from ..preprocessing import FiniteStandardizer, domain_group_balanced_weights
from ..registry import ContrastSpec, register_method
from ..representation import ChainBalancedPCA
from ..tasks import (
    TaskExample,
    load_visible_step_token_states,
    nuisance_features,
    visible_output_steps,
)


def _lagged_transitions(
    sequence: np.ndarray,
    *,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return x[t-1:t-order] and x[t] without exposing x[t] as input."""
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("token sequence must be a finite [token, state] matrix")
    if order < 1:
        raise ValueError("history order must be positive")
    if len(values) <= order:
        return (
            np.empty((0, order, values.shape[1]), dtype=np.float64),
            np.empty((0, values.shape[1]), dtype=np.float64),
        )
    targets = values[order:]
    lags = np.stack(
        [values[order - lag : len(values) - lag] for lag in range(1, order + 1)],
        axis=1,
    )
    return lags, targets


def _shuffle_older_lags(
    lags: np.ndarray,
    *,
    seed: int,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """History null: retain x[t-1], break older-state/target alignment."""
    values = np.asarray(lags, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 1:
        raise ValueError("lags must be [transition, history, state]")
    shuffled = values.copy()
    if len(values) < 2 or values.shape[1] < 2:
        return shuffled
    group_values = (
        np.zeros(len(values), dtype=np.int64)
        if groups is None
        else np.asarray(groups)
    )
    if group_values.shape != (len(values),):
        raise ValueError("shuffle groups must align with transitions")
    rng = np.random.default_rng(seed)
    for group in np.unique(group_values):
        indices = np.flatnonzero(group_values == group)
        if len(indices) < 2:
            continue
        identity = np.arange(len(indices))
        for lag in range(1, values.shape[1]):
            order = rng.permutation(len(indices))
            if np.array_equal(order, identity):
                order = np.roll(order, lag % len(indices) or 1)
            shuffled[indices, lag] = values[indices[order], lag]
    return shuffled


def _sample_transitions(
    sequence: np.ndarray,
    *,
    order: int,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    lags, targets = _lagged_transitions(sequence, order=order)
    if maximum > 0 and len(targets) > maximum:
        indices = np.linspace(0, len(targets) - 1, maximum, dtype=np.int64)
        return lags[indices], targets[indices]
    return lags, targets


def _transition_matrix(
    sequences: tuple[np.ndarray, ...],
    *,
    order: int,
    transitions_per_chain: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lag_rows = []
    target_rows = []
    chain_rows = []
    state_dim: int | None = None
    for chain, sequence in enumerate(sequences):
        values = np.asarray(sequence, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("all token sequences must be finite matrices")
        if state_dim is None:
            state_dim = int(values.shape[1])
        elif values.shape[1] != state_dim:
            raise ValueError("token state dimension differs across chains")
        lags, targets = _sample_transitions(
            values,
            order=order,
            maximum=transitions_per_chain,
        )
        if len(targets):
            lag_rows.append(lags)
            target_rows.append(targets)
            chain_rows.append(np.full(len(targets), chain, dtype=np.int64))
    if not target_rows:
        raise ValueError(f"no chain has more than history_order={order} tokens")
    return (
        np.concatenate(lag_rows),
        np.concatenate(target_rows),
        np.concatenate(chain_rows),
    )


def _chain_balanced_weights(chain_rows: np.ndarray) -> np.ndarray:
    _, inverse, count = np.unique(chain_rows, return_inverse=True, return_counts=True)
    weights = 1.0 / count[inverse]
    return weights / weights.mean()


@dataclass(frozen=True)
class TokenDynamics:
    order: int
    ar1: Ridge
    ordered: Ridge
    shuffled: Ridge
    target_variance: float
    residual_scale: float
    seed: int

    def evaluate(
        self,
        sequences: tuple[np.ndarray, ...],
        *,
        transitions_per_chain: int,
        seed: int,
    ) -> dict[str, float | int]:
        lags, targets, chain_rows = _transition_matrix(
            sequences,
            order=self.order,
            transitions_per_chain=transitions_per_chain,
        )
        shuffled = _shuffle_older_lags(lags, seed=seed, groups=chain_rows)
        weights = _chain_balanced_weights(chain_rows)
        ar1_row_mse = np.mean((targets - self.ar1.predict(lags[:, 0])) ** 2, axis=1)
        ordered_row_mse = np.mean(
            (targets - self.ordered.predict(lags.reshape(len(lags), -1))) ** 2,
            axis=1,
        )
        shuffled_row_mse = np.mean(
            (targets - self.shuffled.predict(shuffled.reshape(len(lags), -1))) ** 2,
            axis=1,
        )
        ar1_mse = float(np.average(ar1_row_mse, weights=weights))
        ordered_mse = float(np.average(ordered_row_mse, weights=weights))
        shuffled_mse = float(np.average(shuffled_row_mse, weights=weights))
        denominator = self.target_variance
        return {
            "transitions": len(targets),
            "ar1_nmse": ar1_mse / denominator,
            "ordered_nmse": ordered_mse / denominator,
            "shuffled_nmse": shuffled_mse / denominator,
            "history_gain_nmse": (ar1_mse - ordered_mse) / denominator,
            "history_order_gain_nmse": (shuffled_mse - ordered_mse) / denominator,
        }

    def prefix_features(
        self,
        sequence: np.ndarray,
        *,
        recent_window: int,
    ) -> np.ndarray:
        values = np.asarray(sequence, dtype=np.float64)
        if recent_window < 1:
            raise ValueError("recent_window must be positive")
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("token sequence must be a finite matrix")
        if len(values) < 2:
            return np.zeros(8, dtype=np.float64)

        ar1_error = np.mean(
            (values[1:] - self.ar1.predict(values[:-1])) ** 2,
            axis=1,
        )
        recent_ar1 = ar1_error[-recent_window:] / self.residual_scale
        logged = np.log1p(recent_ar1)
        slope = 0.0
        if len(logged) > 1:
            position = np.arange(len(logged), dtype=np.float64)
            slope = float(np.polyfit(position, logged, deg=1)[0])
        features = [
            float(logged[-1]),
            float(np.mean(logged)),
            float(np.max(logged)),
            slope,
        ]

        lags, targets = _lagged_transitions(values, order=self.order)
        if not len(targets):
            return np.asarray([*features, 0.0, 0.0, 0.0, 0.0])
        common_ar1 = np.mean((targets - self.ar1.predict(lags[:, 0])) ** 2, axis=1)
        ordered = np.mean(
            (targets - self.ordered.predict(lags.reshape(len(lags), -1))) ** 2,
            axis=1,
        )
        shuffled_lags = _shuffle_older_lags(lags, seed=self.seed + len(values))
        shuffled = np.mean(
            (
                targets
                - self.shuffled.predict(shuffled_lags.reshape(len(lags), -1))
            )
            ** 2,
            axis=1,
        )
        history_gain = (common_ar1 - ordered)[-recent_window:] / self.residual_scale
        shuffled_gain = (common_ar1 - shuffled)[-recent_window:] / self.residual_scale
        return np.asarray(
            [
                *features,
                float(np.mean(history_gain)),
                float(history_gain[-1]),
                float(np.mean(shuffled_gain)),
                float(shuffled_gain[-1]),
            ],
            dtype=np.float64,
        )


def fit_token_dynamics(
    sequences: tuple[np.ndarray, ...],
    *,
    order: int,
    ridge_alpha: float,
    transitions_per_chain: int,
    seed: int,
) -> TokenDynamics:
    if order < 2:
        raise ValueError("history_order must be at least two for a history audit")
    if ridge_alpha <= 0 or transitions_per_chain < 0:
        raise ValueError("ridge_alpha must be positive and transition cap nonnegative")
    lags, targets, chain_rows = _transition_matrix(
        sequences,
        order=order,
        transitions_per_chain=transitions_per_chain,
    )
    shuffled_lags = _shuffle_older_lags(lags, seed=seed, groups=chain_rows)
    weights = _chain_balanced_weights(chain_rows)
    ar1 = Ridge(alpha=ridge_alpha).fit(lags[:, 0], targets, sample_weight=weights)
    ordered = Ridge(alpha=ridge_alpha).fit(
        lags.reshape(len(lags), -1), targets, sample_weight=weights
    )
    shuffled = Ridge(alpha=ridge_alpha).fit(
        shuffled_lags.reshape(len(lags), -1), targets, sample_weight=weights
    )
    target_center = np.average(targets, axis=0, weights=weights)
    target_variance = max(
        float(np.average(np.mean((targets - target_center) ** 2, axis=1), weights=weights)),
        1e-12,
    )
    residual = np.mean((targets - ar1.predict(lags[:, 0])) ** 2, axis=1)
    residual_scale = max(float(np.median(residual)), target_variance * 1e-8, 1e-12)
    return TokenDynamics(
        order=order,
        ar1=ar1,
        ordered=ordered,
        shuffled=shuffled,
        target_variance=target_variance,
        residual_scale=residual_scale,
        seed=seed,
    )


@dataclass(frozen=True)
class TokenPredictiveStateConfig:
    pca_dim: int = 4
    positions_per_chain: int = 16
    history_order: int = 4
    transitions_per_chain: int = 64
    recent_window: int = 8
    dynamics_ridge_alpha: float = 10.0
    hazard_l2: float = 0.1
    hazard_max_iter: int = 2000

    def __post_init__(self) -> None:
        integers = (
            "pca_dim",
            "positions_per_chain",
            "history_order",
            "transitions_per_chain",
            "recent_window",
            "hazard_max_iter",
        )
        if any(
            isinstance(getattr(self, name), (bool, np.bool_))
            or not isinstance(getattr(self, name), (int, np.integer))
            for name in integers
        ):
            raise ValueError("token predictive-state integer options must be integers")
        if self.pca_dim < 1 or self.positions_per_chain < self.pca_dim:
            raise ValueError("positions_per_chain must be at least the positive PCA dimension")
        if self.history_order < 3:
            raise ValueError("history_order must be at least three for an order null")
        if self.transitions_per_chain < 0:
            raise ValueError("transitions_per_chain cannot be negative")
        if self.recent_window < 1 or self.hazard_max_iter < 1:
            raise ValueError("recent_window and hazard_max_iter must be positive")
        if self.dynamics_ridge_alpha <= 0 or self.hazard_l2 <= 0:
            raise ValueError("dynamics and hazard regularization must be positive")


@dataclass(frozen=True)
class _HazardFit:
    probability: np.ndarray
    model: RegularizedLogistic
    scaler: FiniteStandardizer


def _chain_key(example: TaskExample) -> tuple[str, int]:
    return example.sample.dataset, int(example.sample.chain_id)


def _latest_examples(
    examples: Sequence[TaskExample],
) -> dict[tuple[str, int], TaskExample]:
    latest: dict[tuple[str, int], TaskExample] = {}
    for example in examples:
        key = _chain_key(example)
        if key not in latest or example.visible_steps > latest[key].visible_steps:
            latest[key] = example
    return latest


def _visible_token_tensor(example: TaskExample) -> np.ndarray:
    return np.concatenate(load_visible_step_token_states(example), axis=0)


def _project_chains(
    examples: Sequence[TaskExample],
    projector: ChainBalancedPCA,
    *,
    reporter: Any,
    description: str,
) -> dict[tuple[str, int], tuple[np.ndarray, ...]]:
    latest = _latest_examples(examples)
    cache: dict[tuple[str, int], tuple[np.ndarray, ...]] = {}
    tracked = reporter.track(latest.items(), total=len(latest), description=description)
    for key, example in tracked:
        raw_steps = load_visible_step_token_states(example)
        lengths = np.asarray([len(step) for step in raw_steps], dtype=np.int64)
        projected = projector.transform(np.concatenate(raw_steps, axis=0))
        flattened = projected.reshape(projected.shape[0], -1)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        cache[key] = tuple(
            np.ascontiguousarray(
                flattened[offsets[index] : offsets[index + 1]], dtype=np.float32
            )
            for index in range(len(raw_steps))
        )
    return cache


def _prefix_sequence(
    example: TaskExample,
    cache: Mapping[tuple[str, int], tuple[np.ndarray, ...]],
) -> np.ndarray:
    steps = cache[_chain_key(example)][: example.visible_steps]
    if len(steps) != example.visible_steps:
        raise RuntimeError("projected chain is shorter than the visible prefix")
    return np.concatenate(steps, axis=0)


def _context_features(example: TaskExample) -> np.ndarray:
    _, nuisance = nuisance_features(example)
    output = np.asarray(visible_output_steps(example), dtype=np.float64)
    current = output[-1]
    if len(output) == 1:
        earlier = np.zeros(output.shape[1], dtype=np.float64)
        has_earlier = 0.0
    else:
        earlier = output[:-1].mean(axis=0)
        has_earlier = 1.0
    values = np.concatenate([nuisance, current, earlier, [has_earlier]])
    if not np.isfinite(values).all():
        raise ValueError("online context contains non-finite values")
    return values


def _row_matrices(
    examples: tuple[TaskExample, ...],
    cache: Mapping[tuple[str, int], tuple[np.ndarray, ...]],
    dynamics: TokenDynamics,
    *,
    recent_window: int,
) -> dict[str, np.ndarray]:
    context = np.stack([_context_features(example) for example in examples])
    dynamics_features = np.stack(
        [
            dynamics.prefix_features(
                _prefix_sequence(example, cache), recent_window=recent_window
            )
            for example in examples
        ]
    )
    ar1 = dynamics_features[:, :4]
    ordered = dynamics_features[:, 4:6]
    shuffled = dynamics_features[:, 6:8]
    return {
        "output_only": context,
        "ar1_innovation": np.column_stack([context, ar1]),
        "ordered_history": np.column_stack([context, ar1, ordered]),
        "shuffled_history": np.column_stack([context, ar1, shuffled]),
    }


def _fit_hazard(
    train: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    test: np.ndarray,
    *,
    l2: float,
    max_iter: int,
) -> _HazardFit:
    scaler = FiniteStandardizer().fit(train, weights)
    model = RegularizedLogistic(l2=l2, max_iter=max_iter).fit(
        scaler.transform(train), labels, weights
    )
    return _HazardFit(model.predict_proba(scaler.transform(test)), model, scaler)


@register_method(
    "token_predictive_state",
    contrasts=(
        ContrastSpec(
            "ar1_innovation_given_output_nll",
            "output_only",
            "ar1_innovation",
            "recent token-transition innovation beyond completed-output controls",
        ),
        ContrastSpec(
            "ordered_history_given_ar1_nll",
            "ar1_innovation",
            "ordered_history",
            "ordered finite-history prediction gain beyond first-order innovation",
        ),
        ContrastSpec(
            "history_order_nll",
            "shuffled_history",
            "ordered_history",
            "ordered older-token history versus a capacity-matched alignment null",
        ),
    ),
    arm_definitions={
        "output_only": "online position/length and completed-step entropy/NLL controls",
        "ar1_innovation": "output controls plus recent AR(1) token residual summaries",
        "ordered_history": (
            "AR(1) arm plus gain from the ordered AR(p) conditional-mean model"
        ),
        "shuffled_history": (
            "same dimension as ordered_history, using an older-token alignment null"
        ),
    },
    default_config=TokenPredictiveStateConfig,
)
class TokenPredictiveState:
    """Approximate token-state Markov audit and prospective first-error probe."""

    def __init__(self, config: TokenPredictiveStateConfig | Mapping[str, Any]) -> None:
        if isinstance(config, Mapping):
            config = TokenPredictiveStateConfig(**dict(config))
        if not isinstance(config, TokenPredictiveStateConfig):
            raise TypeError("token_predictive_state config must be a config or mapping")
        self.config = config

    def fit_predict(self, fold: FoldInput) -> MethodFoldResult:
        if fold.task_name != "strict_prefix":
            raise ValueError("token_predictive_state requires the strict_prefix task")
        reporter = fold.progress or NullProgress()
        reporter.stage("projection", f"{fold.task_name}: outer-train token-state PCA")
        projector = ChainBalancedPCA(
            dim=self.config.pca_dim,
            positions_per_chain=self.config.positions_per_chain,
            seed=fold.seed,
        ).fit(
            fold.train_examples,
            progress=reporter,
            state_loader=_visible_token_tensor,
        )
        train_cache = _project_chains(
            fold.train_examples,
            projector,
            reporter=reporter,
            description="train token-state chains",
        )
        test_cache = _project_chains(
            fold.test_examples,
            projector,
            reporter=reporter,
            description="test token-state chains",
        )
        train_sequences = tuple(
            np.concatenate(steps, axis=0) for steps in train_cache.values()
        )
        test_sequences = tuple(
            np.concatenate(steps, axis=0) for steps in test_cache.values()
        )

        reporter.stage("fit", f"{fold.task_name}: label-free token dynamics")
        dynamics = fit_token_dynamics(
            train_sequences,
            order=self.config.history_order,
            ridge_alpha=self.config.dynamics_ridge_alpha,
            transitions_per_chain=self.config.transitions_per_chain,
            seed=fold.seed,
        )
        transition_diagnostics = dynamics.evaluate(
            test_sequences,
            transitions_per_chain=self.config.transitions_per_chain,
            seed=fold.seed + 1,
        )
        train_designs = _row_matrices(
            fold.train_examples,
            train_cache,
            dynamics,
            recent_window=self.config.recent_window,
        )
        test_designs = _row_matrices(
            fold.test_examples,
            test_cache,
            dynamics,
            recent_window=self.config.recent_window,
        )
        domains = np.asarray(
            [example.sample.dataset for example in fold.train_examples], dtype=object
        )
        weights = domain_group_balanced_weights(domains, fold.train_groups)
        reporter.stage("fit", f"{fold.task_name}: explicit first-error hazard")
        fitted = {
            arm: _fit_hazard(
                train,
                fold.train_labels,
                weights,
                test_designs[arm],
                l2=self.config.hazard_l2,
                max_iter=self.config.hazard_max_iter,
            )
            for arm, train in train_designs.items()
        }

        if projector.model is None:
            raise RuntimeError("token-state projector is unavailable after fit")
        factors: dict[str, np.ndarray] = {
            "pca.components": np.asarray(projector.model.components_),
            "pca.mean": projector.mean_,
            "dynamics.ar1.coefficients": np.asarray(dynamics.ar1.coef_),
            "dynamics.ar1.intercept": np.asarray(dynamics.ar1.intercept_),
            "dynamics.ordered.coefficients": np.asarray(dynamics.ordered.coef_),
            "dynamics.ordered.intercept": np.asarray(dynamics.ordered.intercept_),
            "dynamics.shuffled.coefficients": np.asarray(dynamics.shuffled.coef_),
            "dynamics.shuffled.intercept": np.asarray(dynamics.shuffled.intercept_),
        }
        for arm, fit in fitted.items():
            factors[f"hazard.{arm}.coefficients"] = fit.model.coefficients
            factors[f"hazard.{arm}.center"] = np.asarray(fit.scaler.center_)
            factors[f"hazard.{arm}.scale"] = np.asarray(fit.scaler.scale_)

        return MethodFoldResult(
            probabilities={arm: fit.probability for arm, fit in fitted.items()},
            diagnostics={
                "analysis_unit": "token_transition",
                "state_representation": (
                    "all_stored_layers_times_outer_train_hidden_dimension_PCA"
                ),
                "projection_dim_per_layer": self.config.pca_dim,
                "projected_token_state_dim": int(train_sequences[0].shape[1]),
                "history_order": self.config.history_order,
                "recent_window": self.config.recent_window,
                "dynamics_model": "ridge_linear_conditional_mean_AR1_and_ARp",
                "dynamics_fit_scope": (
                    "outer_train_unique_at_risk_chains_label_free_loss"
                ),
                "censoring_uses_first_error_labels": True,
                "dynamics_chain_weighting": "equal_total_weight_per_chain",
                "test_transition_diagnostics": transition_diagnostics,
                "hazard_model": "regularized_logistic_no_neural_network",
                "hazard_target": "completed_prefix_predicts_next_step_first_error",
                "hazard_training_weights": "equal_domain_then_equal_problem_group",
                "arm_feature_dimensions": {
                    arm: int(matrix.shape[1]) for arm, matrix in train_designs.items()
                },
                "strict_markov_claim": False,
                "interpretation": (
                    "predictive-state quantification; not proof that the LLM is Markov, "
                    "linear, Gaussian, or causally governed by the fitted dynamics"
                ),
                "uses_future_tokens": False,
                "uses_post_error_states": False,
            },
            factors=factors,
        )
