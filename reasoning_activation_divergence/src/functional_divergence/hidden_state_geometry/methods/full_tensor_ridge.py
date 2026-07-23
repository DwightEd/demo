from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ...progress import NullProgress, ProgressReporter
from ..features import EncodedRows, FunctionalFeatureBuilder
from ..method import FoldInput, MethodFoldResult
from ..model import RegularizedLogistic
from ..preprocessing import (
    FiniteStandardizer,
    domain_group_balanced_weights,
    group_balanced_weights,
)
from ..registry import ContrastSpec, register_method


ARM_COMPONENTS = {
    "nuisance": ("nuisance",),
    "output_only": ("nuisance", "output"),
    "hidden_only": ("nuisance", "hidden"),
    "output_plus_hidden": ("nuisance", "output", "hidden"),
}


@dataclass(frozen=True)
class FullTensorRidgeConfig:
    pca_dim: int = 16
    time_basis: int = 3
    layer_basis: int = 3
    positions_per_chain: int = 32
    l2_grid: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    max_iter: int = 500
    selection_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        try:
            raw_grid = tuple(self.l2_grid)
        except TypeError as exc:
            raise ValueError("l2_grid must be an iterable of positive numbers") from exc
        if any(isinstance(value, (bool, np.bool_)) for value in raw_grid):
            raise ValueError("l2_grid cannot contain booleans")
        grid = tuple(sorted({float(value) for value in raw_grid}))
        object.__setattr__(self, "l2_grid", grid)
        integer_fields = (
            "pca_dim",
            "time_basis",
            "layer_basis",
            "positions_per_chain",
            "max_iter",
        )
        if any(
            isinstance(getattr(self, name), (bool, np.bool_))
            or not isinstance(getattr(self, name), (int, np.integer))
            for name in integer_fields
        ):
            raise ValueError("dimensions, basis widths, and max_iter must be integers")
        if any(getattr(self, name) < 1 for name in integer_fields):
            raise ValueError("dimensions, basis widths, and max_iter must be positive")
        if self.positions_per_chain < self.pca_dim:
            raise ValueError("positions_per_chain must be at least pca_dim")
        if not grid or any(not np.isfinite(value) or value <= 0 for value in grid):
            raise ValueError("l2_grid must contain finite positive values")
        if not np.isfinite(self.selection_tolerance) or self.selection_tolerance < 0:
            raise ValueError("selection_tolerance must be finite and nonnegative")


@dataclass(frozen=True)
class InnerDomainFold:
    held_domain: str
    train: np.ndarray
    validation: np.ndarray


@dataclass(frozen=True)
class _FittedArm:
    model: RegularizedLogistic
    scaler: FiniteStandardizer
    probability: np.ndarray


def _domains(examples: tuple) -> np.ndarray:
    return np.asarray([example.sample.dataset for example in examples], dtype=object)


def inner_lodo_splits(fold: FoldInput) -> tuple[InnerDomainFold, ...]:
    domains = _domains(fold.train_examples)
    unique_domains = sorted(str(value) for value in np.unique(domains))
    if len(unique_domains) < 2:
        raise ValueError(
            "full_tensor_ridge needs at least two outer-training domains "
            "for multi-value l2 selection"
        )
    splits = []
    for held_domain in unique_domains:
        validation = np.flatnonzero(domains == held_domain)
        train = np.flatnonzero(domains != held_domain)
        for name, indices in (("train", train), ("validation", validation)):
            if len(np.unique(fold.train_labels[indices])) != 2:
                raise ValueError(
                    f"inner held domain {held_domain}: {name} rows need both classes"
                )
        overlap = set(fold.train_groups[train]).intersection(
            fold.train_groups[validation]
        )
        if overlap:
            raise ValueError(
                f"inner held domain {held_domain}: problem-group leakage "
                f"for {sorted(str(value) for value in overlap)[:5]}"
            )
        splits.append(InnerDomainFold(held_domain, train, validation))
    return tuple(splits)


def _design_matrices(rows: EncodedRows) -> dict[str, np.ndarray]:
    hidden = rows.hidden.reshape(len(rows.hidden), -1)
    components = {
        "nuisance": rows.nuisance,
        "output": rows.output,
        "hidden": hidden,
    }
    return {
        arm: np.column_stack([components[name] for name in names])
        for arm, names in ARM_COMPONENTS.items()
    }


def _binary_nll(
    labels: np.ndarray, probability: np.ndarray, weights: np.ndarray
) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    loss = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    return float(np.average(loss, weights=weights))


def _grid_edge(value: float, grid: tuple[float, ...]) -> str:
    if len(grid) == 1:
        return "fixed"
    if value == grid[0]:
        return "lower"
    if value == grid[-1]:
        return "upper"
    return "none"


@register_method(
    "full_tensor_ridge",
    contrasts=(
        ContrastSpec(
            "output_summary_given_nuisance_nll",
            "nuisance",
            "output_only",
            "stored entropy/NLL summary increment beyond length and position",
        ),
        ContrastSpec(
            "hidden_given_nuisance_nll",
            "nuisance",
            "hidden_only",
            "full functional hidden tensor increment beyond nuisance controls",
        ),
        ContrastSpec(
            "hidden_given_output_summary_nll",
            "output_only",
            "output_plus_hidden",
            "full functional hidden tensor increment beyond output summaries",
        ),
    ),
    arm_definitions={
        "nuisance": "length and boundary-position controls",
        "output_only": "nuisance plus stored entropy/NLL step summaries",
        "hidden_only": "nuisance plus every encoded time-layer-channel tensor cell",
        "output_plus_hidden": (
            "nuisance, output summaries, and every encoded hidden tensor cell"
        ),
    },
    default_config=FullTensorRidgeConfig,
)
class FullTensorRidge:
    """Convex ridge probe over every encoded time-layer-channel cell."""

    def __init__(self, config: FullTensorRidgeConfig | Mapping[str, Any]) -> None:
        if isinstance(config, Mapping):
            config = FullTensorRidgeConfig(**dict(config))
        if not isinstance(config, FullTensorRidgeConfig):
            raise TypeError("full_tensor_ridge config must be a config object or mapping")
        self.config = config

    def _builder(self, seed: int) -> FunctionalFeatureBuilder:
        return FunctionalFeatureBuilder(
            pca_dim=self.config.pca_dim,
            time_basis=self.config.time_basis,
            layer_basis=self.config.layer_basis,
            positions_per_chain=self.config.positions_per_chain,
            seed=seed,
        )

    def _fit_arm(
        self,
        train: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        domains: np.ndarray,
        test: np.ndarray,
        l2: float,
    ) -> _FittedArm:
        weights = domain_group_balanced_weights(domains, groups)
        scaler = FiniteStandardizer().fit(train, weights)
        train_scaled = scaler.transform(train)
        test_scaled = scaler.transform(test)
        model = RegularizedLogistic(
            l2=l2, max_iter=self.config.max_iter
        ).fit(train_scaled, labels, weights)
        return _FittedArm(model, scaler, model.predict_proba(test_scaled))

    def _select_l2(
        self,
        fold: FoldInput,
        reporter: ProgressReporter,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, Any]]:
        if len(self.config.l2_grid) == 1:
            value = self.config.l2_grid[0]
            return (
                {arm: value for arm in ARM_COMPONENTS},
                {arm: {} for arm in ARM_COMPONENTS},
                {
                    "scheme": "fixed_single_l2",
                    "held_domains": [],
                    "feature_fit_scope": "outer_train_only",
                    "projection_train_domains": {},
                },
            )

        scores = {
            arm: {l2: [] for l2 in self.config.l2_grid}
            for arm in ARM_COMPONENTS
        }
        projection_domains = {}
        splits = inner_lodo_splits(fold)
        for split_index, split in enumerate(splits):
            reporter.stage(
                "select",
                f"{fold.task_name}: inner held domain {split.held_domain}",
            )
            train_examples = tuple(fold.train_examples[index] for index in split.train)
            validation_examples = tuple(
                fold.train_examples[index] for index in split.validation
            )
            builder = self._builder(fold.seed + split_index + 1).fit(
                train_examples, reporter=reporter
            )
            train = builder.transform(
                train_examples,
                reporter=reporter,
                projection_description=f"{split.held_domain} inner-train projected chains",
                encoding_description=f"{split.held_domain} inner-train examples",
            )
            validation = builder.transform(
                validation_examples,
                reporter=reporter,
                projection_description=f"{split.held_domain} validation projected chains",
                encoding_description=f"{split.held_domain} validation examples",
            )
            if builder.projector is None:
                raise RuntimeError("inner feature builder lost its fitted projector")
            projection_domains[split.held_domain] = sorted(
                {dataset for dataset, _ in builder.projector.sampled_rows_per_chain}
            )
            train_designs = _design_matrices(train.rows)
            validation_designs = _design_matrices(validation.rows)
            train_domains = _domains(train_examples)
            train_weights = domain_group_balanced_weights(
                train_domains, fold.train_groups[split.train]
            )
            validation_weights = group_balanced_weights(
                fold.train_groups[split.validation]
            )
            for arm in ARM_COMPONENTS:
                scaler = FiniteStandardizer().fit(
                    train_designs[arm], train_weights
                )
                train_scaled = scaler.transform(train_designs[arm])
                validation_scaled = scaler.transform(validation_designs[arm])
                grid = reporter.track(
                    self.config.l2_grid,
                    total=len(self.config.l2_grid),
                    description=f"{split.held_domain} {arm} ridge",
                )
                for l2 in grid:
                    try:
                        model = RegularizedLogistic(
                            l2=l2, max_iter=self.config.max_iter
                        ).fit(
                            train_scaled,
                            fold.train_labels[split.train],
                            train_weights,
                        )
                    except RuntimeError as exc:
                        raise RuntimeError(
                            f"{arm} failed for inner held domain "
                            f"{split.held_domain}, l2={l2}"
                        ) from exc
                    probability = model.predict_proba(validation_scaled)
                    scores[arm][l2].append(
                        _binary_nll(
                            fold.train_labels[split.validation],
                            probability,
                            validation_weights,
                        )
                    )

        mean_scores = {
            arm: {
                l2: float(np.mean(domain_scores))
                for l2, domain_scores in arm_scores.items()
            }
            for arm, arm_scores in scores.items()
        }
        selected = {}
        for arm, arm_scores in mean_scores.items():
            best = min(arm_scores.values())
            eligible = [
                l2
                for l2, score in arm_scores.items()
                if score <= best + self.config.selection_tolerance
            ]
            selected[arm] = max(eligible)
        serialized_scores = {
            arm: {f"{l2:.12g}": score for l2, score in arm_scores.items()}
            for arm, arm_scores in mean_scores.items()
        }
        return (
            selected,
            serialized_scores,
            {
                "scheme": "inner_leave_one_training_domain_out",
                "held_domains": [split.held_domain for split in splits],
                "feature_fit_scope": "each_inner_train_only",
                "projection_train_domains": projection_domains,
                "score": "equal-inner-domain mean of group-balanced NLL",
                "tie_break": "largest l2 within selection_tolerance",
            },
        )

    def fit_predict(self, fold: FoldInput) -> MethodFoldResult:
        reporter = fold.progress or NullProgress()
        encoded = self._builder(fold.seed).build(fold)
        train_designs = _design_matrices(encoded.train.rows)
        test_designs = _design_matrices(encoded.test.rows)
        selected_l2, inner_scores, selection = self._select_l2(fold, reporter)
        train_domains = _domains(fold.train_examples)

        reporter.stage("fit", f"{fold.task_name}: selected full-tensor ridge arms")
        fitted = {}
        arms = reporter.track(
            ARM_COMPONENTS, total=len(ARM_COMPONENTS), description="final ridge arms"
        )
        for arm in arms:
            fitted[arm] = self._fit_arm(
                train_designs[arm],
                fold.train_labels,
                fold.train_groups,
                train_domains,
                test_designs[arm],
                selected_l2[arm],
            )

        tensor_shape = encoded.train.rows.hidden.shape[1:]
        nuisance_dim = encoded.train.rows.nuisance.shape[1]
        output_dim = encoded.train.rows.output.shape[1]
        factors = {}
        for arm, result in fitted.items():
            if result.scaler.center_ is None or result.scaler.scale_ is None:
                raise RuntimeError(f"{arm} scaler unexpectedly missing after fit")
            factors[f"{arm}.coefficients"] = result.model.coefficients
            factors[f"{arm}.scaler_center"] = np.asarray(result.scaler.center_)
            factors[f"{arm}.scaler_scale"] = np.asarray(result.scaler.scale_)
            factors[f"{arm}.selected_l2"] = np.asarray(selected_l2[arm])
            if "hidden" in ARM_COMPONENTS[arm]:
                start = nuisance_dim + (
                    output_dim if "output" in ARM_COMPONENTS[arm] else 0
                )
                raw_coefficient = (
                    result.model.coefficients[1:] / result.scaler.scale_
                )
                factors[f"{arm}.hidden_tensor_coefficient"] = raw_coefficient[
                    start : start + int(np.prod(tensor_shape))
                ].reshape(tensor_shape)
        projector = encoded.projector
        if projector.model is None:
            raise RuntimeError("projector unexpectedly missing after fit")
        selected_at_grid_edge = {
            arm: _grid_edge(value, self.config.l2_grid)
            for arm, value in selected_l2.items()
        }
        factors.update(
            {
                "pca_components": np.asarray(projector.model.components_),
                "pca_mean": projector.mean_,
                "pca_explained_variance": np.asarray(
                    projector.model.explained_variance_
                ),
            }
        )
        return MethodFoldResult(
            probabilities={
                arm: result.probability for arm, result in fitted.items()
            },
            diagnostics={
                "projection_dim": self.config.pca_dim,
                "projection_training_rows": projector.training_rows,
                "projection_explained_variance": float(
                    projector.explained_variance_ratio_.sum()
                ),
                "tensor_shape": list(tensor_shape),
                "flattened_hidden_dim": int(np.prod(tensor_shape)),
                "arm_feature_dimensions": {
                    arm: matrix.shape[1] for arm, matrix in train_designs.items()
                },
                "selected_l2": selected_l2,
                "selected_at_grid_edge": selected_at_grid_edge,
                "inner_cv_scores": inner_scores,
                "selection": selection,
                "training_weights": "equal domain, then equal problem group",
                "comparison_design": (
                    "outer_lodo_fixed_l2_full_tensor_ridge"
                    if len(self.config.l2_grid) == 1
                    else "outer_lodo_inner_lodo_full_tensor_ridge"
                ),
                "regularization_selection": "independent_by_arm",
                "risk_difference_scope": (
                    "predictive risk difference between independently tuned feature "
                    "models, not a coefficient-level conditional effect"
                ),
                "temporal_encoding": (
                    "current_plus_history_mean"
                    if fold.task_name == "strict_prefix"
                    else f"low_frequency_dct_width_{self.config.time_basis}"
                ),
                "axis_order_controls_in_this_method": False,
                "coefficient_coordinate_scope": (
                    "fold_local_whitened_pca_not_cross_fold_aligned"
                ),
                "converged": {
                    arm: result.model.converged_ for arm, result in fitted.items()
                },
            },
            factors=factors,
        )
