from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...progress import NullProgress, ProgressReporter
from ..config import RawFunctionalConfig
from ..method import FoldInput, MethodFoldResult
from ..model import RankOneLogistic, RegularizedLogistic
from ..preprocessing import (
    FiniteStandardizer,
    group_balanced_weights,
)
from ..registry import ContrastSpec, RandomizationSpec, register_method
from ..representation import ChainBalancedPCA, FunctionalEncoder
from ..tasks import load_visible_states, nuisance_features


def _key(row) -> tuple[str, int]:
    return row.sample.dataset, row.sample.chain_id


@dataclass(frozen=True)
class _EncodedRows:
    nuisance: np.ndarray
    output: np.ndarray
    hidden: np.ndarray


@dataclass(frozen=True)
class _TensorArm:
    probability: np.ndarray
    factor: np.ndarray
    parameters: np.ndarray
    converged: bool
    used_baseline: bool
    objective: float | None
    baseline_objective: float | None
    signal_restarts_converged: int


def _projection_cache(rows, encoder, reporter: ProgressReporter, description: str):
    latest = {}
    for row in rows:
        key = _key(row)
        if key not in latest or row.visible_steps > latest[key].visible_steps:
            latest[key] = row
    cache = {}
    tracked = reporter.track(latest.items(), total=len(latest), description=description)
    for key, row in tracked:
        cache[key] = encoder.projector.transform(load_visible_states(row))
    return cache


def _encode(rows, encoder, cache, reporter: ProgressReporter, description: str) -> _EncodedRows:
    encoded = []
    tracked = reporter.track(rows, total=len(rows), description=description)
    for row in tracked:
        projected = cache[_key(row)]
        encoded.append(
            (
                nuisance_features(row)[1],
                encoder.output_features(row),
                encoder.hidden_tensor(row, projected=projected),
            )
        )
    columns = tuple(np.stack(values, axis=0) for values in zip(*encoded))
    return _EncodedRows(*columns)


def _encode_nulls(rows, encoder, cache, reporter, description):
    time, layer = [], []
    tracked = reporter.track(rows, total=len(rows), description=description)
    for row in tracked:
        projected = cache[_key(row)]
        time.append(encoder.hidden_tensor(row, null="time", projected=projected))
        layer.append(encoder.hidden_tensor(row, null="layer", projected=projected))
    return np.stack(time, axis=0), np.stack(layer, axis=0)


@register_method(
    "raw_functional_probe",
    contrasts=(
        ContrastSpec(
            "hidden_given_output_summary_nll",
            "output_only",
            "output_plus_hidden",
            "hidden increment beyond nuisance plus entropy/NLL summaries",
        ),
    ),
    randomizations=(
        RandomizationSpec(
            "time_axis_order",
            "output_plus_time_null_r",
            "output_plus_hidden",
            3,
            "exploratory sample-keyed time permutations; prefixes shorter than three steps excluded",
        ),
        RandomizationSpec(
            "layer_axis_order",
            "output_plus_layer_null_r",
            "output_plus_hidden",
            1,
            "multiple sample-keyed layer permutations",
        ),
    ),
    arm_definitions={
        "nuisance": "length and boundary-position controls",
        "output_only": "nuisance plus stored entropy/NLL step summaries",
        "hidden_only": "nuisance plus raw-hidden functional tensor",
        "output_plus_hidden": "nuisance, output summaries, and raw-hidden tensor",
        "output_plus_time_null_r*": "joint arms over sample-keyed time permutations",
        "output_plus_layer_null_r*": "joint arms over sample-keyed layer permutations",
    },
    default_config=RawFunctionalConfig,
)
class RawFunctionalProbe:
    """Raw-hidden discriminant; no normal/correct trajectory is fitted."""

    def __init__(self, config: RawFunctionalConfig) -> None:
        self.config = config

    def _tensor_arm(
        self,
        train_tensor: np.ndarray,
        test_tensor: np.ndarray,
        train_static: np.ndarray,
        test_static: np.ndarray,
        fold: FoldInput,
        constrained_probability: np.ndarray,
        baseline_parameters: np.ndarray,
    ) -> _TensorArm:
        variation = float(np.nanmax(np.std(train_tensor, axis=0)))
        if variation < 1e-10:
            parameters = np.concatenate(
                [
                    baseline_parameters,
                    *(np.zeros(size, dtype=np.float64) for size in train_tensor.shape[1:]),
                ]
            )
            return _TensorArm(
                probability=constrained_probability.copy(),
                factor=np.zeros(train_tensor.shape[1:]),
                parameters=parameters,
                converged=True,
                used_baseline=True,
                objective=None,
                baseline_objective=None,
                signal_restarts_converged=0,
            )
        weights = group_balanced_weights(fold.train_groups)
        model = RankOneLogistic(
            l2=self.config.l2,
            restarts=self.config.restarts,
            max_iter=self.config.max_iter,
            seed=fold.seed,
        ).fit(
            train_tensor,
            fold.train_labels,
            static=train_static,
            sample_weight=weights,
            baseline_parameters=baseline_parameters,
        )
        return _TensorArm(
            probability=model.predict_proba(test_tensor, test_static),
            factor=model.coefficient_tensor,
            parameters=model.model_parameters,
            converged=model.converged_,
            used_baseline=model.used_baseline_,
            objective=model.objective_,
            baseline_objective=model.baseline_objective_,
            signal_restarts_converged=model.signal_restarts_converged_,
        )

    def fit_predict(self, fold: FoldInput) -> MethodFoldResult:
        reporter = fold.progress or NullProgress()
        reporter.stage("projection", fold.task_name)
        projector = ChainBalancedPCA(
            dim=self.config.pca_dim,
            positions_per_chain=self.config.positions_per_chain,
            seed=fold.seed,
        ).fit(fold.train_examples, progress=reporter)
        encoder = FunctionalEncoder(
            projector,
            time_basis=self.config.time_basis,
            layer_basis=self.config.layer_basis,
            null_seed=fold.seed,
        )
        reporter.stage("encode", fold.task_name)
        train_cache = _projection_cache(
            fold.train_examples, encoder, reporter, "train projected chains"
        )
        test_cache = _projection_cache(
            fold.test_examples, encoder, reporter, "test projected chains"
        )
        train = _encode(
            fold.train_examples, encoder, train_cache, reporter, "train examples"
        )
        test = _encode(
            fold.test_examples, encoder, test_cache, reporter, "test examples"
        )
        combined_train = np.column_stack([train.nuisance, train.output])
        combined_test = np.column_stack([test.nuisance, test.output])

        reporter.stage("fit", f"{fold.task_name}: static controls")
        weights = group_balanced_weights(fold.train_groups)
        nuisance_scaler = FiniteStandardizer().fit(train.nuisance, weights)
        output_scaler = FiniteStandardizer().fit(combined_train, weights)
        nuisance_train = nuisance_scaler.transform(train.nuisance)
        nuisance_test = nuisance_scaler.transform(test.nuisance)
        output_train = output_scaler.transform(combined_train)
        output_test = output_scaler.transform(combined_test)
        nuisance_model = RegularizedLogistic(
            l2=self.config.l2, max_iter=self.config.max_iter
        ).fit(nuisance_train, fold.train_labels, weights)
        output_model = RegularizedLogistic(
            l2=self.config.l2, max_iter=self.config.max_iter
        ).fit(output_train, fold.train_labels, weights)
        nuisance_probability = nuisance_model.predict_proba(nuisance_test)
        output_probability = output_model.predict_proba(output_test)

        reporter.stage("fit", f"{fold.task_name}: hidden_only")
        train_tensor = train.hidden
        test_tensor = test.hidden
        hidden_arm = self._tensor_arm(
            train_tensor,
            test_tensor,
            nuisance_train,
            nuisance_test,
            fold,
            nuisance_probability,
            nuisance_model.coefficients,
        )
        reporter.stage("fit", f"{fold.task_name}: output_plus_hidden")
        joint_arm = self._tensor_arm(
            train_tensor,
            test_tensor,
            output_train,
            output_test,
            fold,
            output_probability,
            output_model.coefficients,
        )
        null_results = {}
        null_factors = {}
        null_parameters = {}
        null_convergence = {}
        null_optimization = {}
        null_seeds = []
        for repeat in range(self.config.null_repeats):
            null_seed = fold.seed + 7919 * (repeat + 1)
            null_seeds.append(null_seed)
            null_encoder = replace(encoder, null_seed=null_seed)
            train_nulls = _encode_nulls(
                fold.train_examples,
                null_encoder,
                train_cache,
                reporter,
                f"train null r{repeat}",
            )
            test_nulls = _encode_nulls(
                fold.test_examples,
                null_encoder,
                test_cache,
                reporter,
                f"test null r{repeat}",
            )
            for name, train_null, test_null in zip(
                ("time", "layer"), train_nulls, test_nulls
            ):
                key = f"output_plus_{name}_null_r{repeat}"
                reporter.stage("fit", f"{fold.task_name}: {name}_null_r{repeat}")
                arm = self._tensor_arm(
                    train_null,
                    test_null,
                    output_train,
                    output_test,
                    fold,
                    output_probability,
                    output_model.coefficients,
                )
                null_results[key] = arm.probability
                null_factors[key] = arm.factor
                null_parameters[f"{key}.parameters"] = arm.parameters
                null_convergence[f"{name}_r{repeat}"] = arm.converged
                null_optimization[f"{name}_r{repeat}"] = {
                    "used_baseline": arm.used_baseline,
                    "objective": arm.objective,
                    "baseline_objective": arm.baseline_objective,
                    "signal_restarts_converged": arm.signal_restarts_converged,
                }
        if projector.model is None:
            raise RuntimeError("projector unexpectedly missing after fit")
        return MethodFoldResult(
            probabilities={
                "nuisance": nuisance_probability,
                "output_only": output_probability,
                "hidden_only": hidden_arm.probability,
                "output_plus_hidden": joint_arm.probability,
                **null_results,
            },
            diagnostics={
                "projection_dim": self.config.pca_dim,
                "projection_training_rows": projector.training_rows,
                "projection_explained_variance": float(
                    projector.explained_variance_ratio_.sum()
                ),
                "hidden_converged": hidden_arm.converged,
                "joint_converged": joint_arm.converged,
                "null_converged": null_convergence,
                "optimization": {
                    "hidden_only": {
                        "used_baseline": hidden_arm.used_baseline,
                        "objective": hidden_arm.objective,
                        "baseline_objective": hidden_arm.baseline_objective,
                        "signal_restarts_converged": hidden_arm.signal_restarts_converged,
                    },
                    "output_plus_hidden": {
                        "used_baseline": joint_arm.used_baseline,
                        "objective": joint_arm.objective,
                        "baseline_objective": joint_arm.baseline_objective,
                        "signal_restarts_converged": joint_arm.signal_restarts_converged,
                    },
                    "nulls": null_optimization,
                },
                "null_scheme": "sample_keyed_axis_permutation",
                "null_seeds": null_seeds,
                "comparison_design": "nested_same_objective_scaler_and_weights",
                "tensor_shape": list(train_tensor.shape[1:]),
            },
            factors={
                "pca_components": np.asarray(projector.model.components_),
                "pca_mean": projector.mean_,
                "pca_explained_variance": np.asarray(
                    projector.model.explained_variance_
                ),
                "nuisance_baseline": nuisance_model.coefficients,
                "output_baseline": output_model.coefficients,
                "nuisance_scaler_center": np.asarray(nuisance_scaler.center_),
                "nuisance_scaler_scale": np.asarray(nuisance_scaler.scale_),
                "output_scaler_center": np.asarray(output_scaler.center_),
                "output_scaler_scale": np.asarray(output_scaler.scale_),
                "hidden_only": hidden_arm.factor,
                "hidden_only.parameters": hidden_arm.parameters,
                "output_plus_hidden": joint_arm.factor,
                "output_plus_hidden.parameters": joint_arm.parameters,
                **null_factors,
                **null_parameters,
            },
        )
