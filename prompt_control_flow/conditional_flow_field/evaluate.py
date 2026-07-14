from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping

import numpy as np

from ..evaluate import auprc, auroc
from ..feasible_tangent.data import problem_key
from ..flow_signature_audit import FlowAuditConfig, crossfit_residualize_score
from .schema import ConditionalFlowFieldResult


@dataclass(frozen=True)
class ConditionalFlowFieldValidationConfig:
    folds: int = 5
    bootstrap: int = 2000
    permutations: int = 1000
    min_score_coverage: float = 0.80
    min_problem_count: int = 100
    random_seed: int = 17


def _pair_auc(error: np.ndarray, correct: np.ndarray) -> float:
    difference = error[:, None] - correct[None, :]
    return float(np.mean((difference > 0) + 0.5 * (difference == 0)))


def _problem_auc_values(
    score: np.ndarray,
    labels: np.ndarray,
    problem_ids: np.ndarray,
) -> dict[Hashable, float]:
    output: dict[Hashable, float] = {}
    for raw_problem in np.unique(problem_ids):
        index = np.where(problem_ids == raw_problem)[0]
        local = np.asarray(score[index], dtype=np.float64)
        finite = np.isfinite(local)
        error = local[finite & (labels[index] == 1)]
        correct = local[finite & (labels[index] == 0)]
        if error.size and correct.size:
            output[problem_key(raw_problem)] = _pair_auc(error, correct)
    return output


def _bootstrap_mean(
    values: Mapping[Hashable, float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    point = float(np.mean(array)) if array.size else float("nan")
    if array.size < 2 or draws <= 0:
        return point, (float("nan"), float("nan"))
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    samples = np.asarray(
        [np.mean(rng.choice(array, size=array.size, replace=True)) for _ in range(draws)]
    )
    return point, tuple(float(x) for x in np.percentile(samples, [2.5, 97.5]))


def _bootstrap_problem_delta(
    first: Mapping[Hashable, float],
    second: Mapping[Hashable, float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    keys = sorted(set(first) & set(second), key=str)
    difference = np.asarray([first[key] - second[key] for key in keys], dtype=np.float64)
    point = float(np.mean(difference)) if difference.size else float("nan")
    if difference.size < 2 or draws <= 0:
        ci = (float("nan"), float("nan"))
    else:
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        samples = np.asarray(
            [
                np.mean(rng.choice(difference, size=difference.size, replace=True))
                for _ in range(draws)
            ]
        )
        ci = tuple(float(x) for x in np.percentile(samples, [2.5, 97.5]))
    return {"delta": point, "ci95": list(ci), "problems": len(keys)}


def _within_problem_permutation_p(
    score: np.ndarray,
    labels: np.ndarray,
    problem_ids: np.ndarray,
    observed: float,
    *,
    permutations: int,
    seed: int,
) -> float:
    if permutations <= 0 or not np.isfinite(observed):
        return float("nan")
    groups = []
    for raw_problem in np.unique(problem_ids):
        index = np.where(problem_ids == raw_problem)[0]
        index = index[np.isfinite(score[index])]
        if np.any(labels[index] == 0) and np.any(labels[index] == 1):
            groups.append(index)
    if not groups:
        return float("nan")
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    null = np.empty(int(permutations), dtype=np.float64)
    for permutation in range(int(permutations)):
        values = []
        for index in groups:
            permuted = rng.permutation(labels[index])
            values.append(_pair_auc(score[index[permuted == 1]], score[index[permuted == 0]]))
        null[permutation] = np.mean(values)
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def _evaluate_score(
    name: str,
    score: np.ndarray,
    result: ConditionalFlowFieldResult,
    cfg: ConditionalFlowFieldValidationConfig,
) -> tuple[dict[str, Any], dict[Hashable, float]]:
    dataset = result.dataset
    values = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(values)
    problem_auc = _problem_auc_values(values, dataset.y_error, dataset.problem_ids)
    point, ci = _bootstrap_mean(
        problem_auc,
        draws=cfg.bootstrap,
        seed=cfg.random_seed + sum(ord(ch) for ch in name),
    )
    return (
        {
            "name": name,
            "n": int(finite.sum()),
            "coverage": float(finite.mean()),
            "errors": int(np.sum(finite & (dataset.y_error == 1))),
            "pooled_auroc": auroc(dataset.y_error, values),
            "pooled_auprc": auprc(dataset.y_error, values),
            "within_problem_auroc_equal_weight": point,
            "within_problem_ci95": list(ci),
            "within_problem_problems": len(problem_auc),
            "within_problem_permutation_p": _within_problem_permutation_p(
                values,
                dataset.y_error,
                dataset.problem_ids,
                point,
                permutations=cfg.permutations,
                seed=cfg.random_seed + 31 + sum(ord(ch) for ch in name),
            ),
        },
        problem_auc,
    )


def _correct_contrast(
    first: np.ndarray,
    second: np.ndarray,
    result: ConditionalFlowFieldResult,
) -> dict[Hashable, float]:
    dataset = result.dataset
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    output: dict[Hashable, float] = {}
    for raw_problem in np.unique(dataset.problem_ids):
        index = np.where(
            (dataset.problem_ids == raw_problem) & (dataset.y_error == 0)
        )[0]
        finite = difference[index][np.isfinite(difference[index])]
        if finite.size:
            output[problem_key(raw_problem)] = float(np.mean(finite))
    return output


def _contrast_summary(
    values: Mapping[Hashable, float],
    *,
    cfg: ConditionalFlowFieldValidationConfig,
    seed_offset: int,
) -> dict[str, Any]:
    point, ci = _bootstrap_mean(
        values,
        draws=cfg.bootstrap,
        seed=cfg.random_seed + seed_offset,
    )
    return {"mean_difference": point, "ci95": list(ci), "problems": len(values)}


def evaluate_conditional_flow_field(
    result: ConditionalFlowFieldResult,
    cfg: ConditionalFlowFieldValidationConfig,
) -> dict[str, Any]:
    dataset = result.dataset
    score = {name: result.chain_score(name) for name in result.chain_score_names}
    step = np.log1p(dataset.n_steps)
    chars = np.log1p(dataset.response_chars)
    controls = np.column_stack([step, chars, step**2, chars**2, step * chars])
    residual_cfg = FlowAuditConfig(
        folds=cfg.folds,
        bootstrap=0,
        permutations=0,
        seed=cfg.random_seed,
        compute_device="cpu",
    )

    gate1_contrasts = {
        "shuffle_minus_phase": _contrast_summary(
            _correct_contrast(
                score["shuffle_energy_mean"], score["phase_energy_mean"], result
            ),
            cfg=cfg,
            seed_offset=101,
        ),
        "wrong_problem_minus_phase": _contrast_summary(
            _correct_contrast(
                score["wrong_problem_energy_mean"], score["phase_energy_mean"], result
            ),
            cfg=cfg,
            seed_offset=103,
        ),
        "phase_minus_state": _contrast_summary(
            _correct_contrast(
                score["phase_energy_mean"], score["state_energy_mean"], result
            ),
            cfg=cfg,
            seed_offset=107,
        ),
    }
    primary_finite = np.isfinite(score["phase_energy_mean"])
    correct_problem_count = gate1_contrasts["shuffle_minus_phase"]["problems"]
    gate1_conditions = {
        "score_coverage_at_least_threshold": float(primary_finite.mean())
        >= cfg.min_score_coverage,
        "correct_problem_count_at_least_minimum": correct_problem_count
        >= cfg.min_problem_count,
        "time_shuffle_contrast_ci_above_zero": gate1_contrasts[
            "shuffle_minus_phase"
        ]["ci95"][0]
        > 0.0,
        "wrong_problem_contrast_ci_above_zero": gate1_contrasts[
            "wrong_problem_minus_phase"
        ]["ci95"][0]
        > 0.0,
        "donor_count_is_target_label_independent": bool(
            result.metadata["donor_count_is_target_label_independent"]
        ),
    }
    gate1_pass = all(gate1_conditions.values())

    diagnostic_names = (
        "phase_energy_mean",
        "phase_energy_late",
        "phase_calibrated_mean",
        "phase_calibrated_late",
        "phase_calibrated_free_energy",
        "phase_calibrated_positive_area",
        "phase_calibrated_cusum",
        "state_calibrated_free_energy",
        "shuffle_calibrated_free_energy",
        "wrong_problem_calibrated_free_energy",
    )
    diagnostics: dict[str, Any] = {}
    auc_values: dict[str, dict[Hashable, float]] = {}
    for name, values in {
        "control.log1p_n_steps": step,
        "control.log1p_response_chars": chars,
    }.items():
        diagnostics[name], auc_values[name] = _evaluate_score(name, values, result, cfg)
    for name in diagnostic_names:
        diagnostics[name], auc_values[name] = _evaluate_score(name, score[name], result, cfg)
        residual_name = f"{name}.length_residual"
        residual = crossfit_residualize_score(
            score[name], controls, dataset.problem_ids, residual_cfg
        )
        diagnostics[residual_name], auc_values[residual_name] = _evaluate_score(
            residual_name, residual, result, cfg
        )

    primary_name = "phase_calibrated_free_energy.length_residual"
    shuffle_name = "shuffle_calibrated_free_energy.length_residual"
    wrong_name = "wrong_problem_calibrated_free_energy.length_residual"
    primary = diagnostics[primary_name]
    primary_vs_shuffle = _bootstrap_problem_delta(
        auc_values[primary_name],
        auc_values[shuffle_name],
        draws=cfg.bootstrap,
        seed=cfg.random_seed + 127,
    )
    primary_vs_wrong = _bootstrap_problem_delta(
        auc_values[primary_name],
        auc_values[wrong_name],
        draws=cfg.bootstrap,
        seed=cfg.random_seed + 131,
    )
    gate2_conditions = {
        "response_score_coverage_at_least_threshold": primary["coverage"]
        >= cfg.min_score_coverage,
        "contrastive_problem_count_at_least_minimum": primary[
            "within_problem_problems"
        ]
        >= cfg.min_problem_count,
        "length_residual_auc_ci_above_chance": primary[
            "within_problem_ci95"
        ][0]
        > 0.5,
        "primary_beats_time_shuffle_ci_above_zero": primary_vs_shuffle["ci95"][0]
        > 0.0,
        "primary_beats_wrong_problem_ci_above_zero": primary_vs_wrong["ci95"][0]
        > 0.0,
    }
    gate2_pass = all(gate2_conditions.values())
    return {
        "method": "conditional_spherical_feasible_flow_field",
        "claim_order": [
            "same_problem_phase_conditioned_direction_distribution_exists",
            "errors_have_persistent_low_density_excursions",
            "only_then_extract_output_fisher_cotangents",
        ],
        "dataset": {
            "samples": dataset.n_samples,
            "errors": int(dataset.y_error.sum()),
            "correct": int((dataset.y_error == 0).sum()),
            "problems": int(np.unique(dataset.problem_ids).size),
            "layers": dataset.layer_ids.tolist(),
            "label_policy": dataset.label_policy,
        },
        "geometry_existence_gate": {
            "pass": gate1_pass,
            "conditions": gate1_conditions,
            "score_coverage": float(primary_finite.mean()),
            "contrasts_on_correct_targets": gate1_contrasts,
            "state_conditioning_subgate": {
                "pass": gate1_contrasts["phase_minus_state"]["ci95"][0] > 0.0,
                "note": "state-window matching is secondary to the preregistered phase field",
            },
        },
        "error_excursion_gate": {
            "pass": gate2_pass,
            "primary_score": primary_name,
            "conditions": gate2_conditions,
            "primary_minus_time_shuffle_auc": primary_vs_shuffle,
            "primary_minus_wrong_problem_auc": primary_vs_wrong,
        },
        "response_diagnostics": diagnostics,
        "decision": {
            "advance_to_output_fisher_extraction": bool(gate1_pass and gate2_pass),
            "status": (
                "advance_to_output_fisher_extraction"
                if gate1_pass and gate2_pass
                else "blocked_until_conditional_flow_field_passes"
            ),
            "logits_used_in_this_audit": False,
            "classifier_trained_in_this_audit": False,
        },
        "scoring_metadata": result.metadata,
    }
