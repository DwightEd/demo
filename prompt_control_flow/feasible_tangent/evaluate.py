from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping

import numpy as np

from ..evaluate import auprc, auroc
from ..flow_signature_audit import FlowAuditConfig, crossfit_residualize_score
from .data import problem_key
from .schema import FeasibleTangentResult


@dataclass(frozen=True)
class FeasibleTangentValidationConfig:
    folds: int = 5
    bootstrap: int = 2000
    permutations: int = 1000
    min_rank_coverage: float = 0.80
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
    score = np.asarray(score, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    output: dict[Hashable, float] = {}
    for raw_problem in np.unique(problem_ids):
        index = np.where(problem_ids == raw_problem)[0]
        finite = np.isfinite(score[index])
        error = score[index[finite & (labels[index] == 1)]]
        correct = score[index[finite & (labels[index] == 0)]]
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
    samples = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        samples[draw] = np.mean(rng.choice(array, size=array.size, replace=True))
    low, high = np.percentile(samples, [2.5, 97.5])
    return point, (float(low), float(high))


def _bootstrap_problem_delta(
    first: Mapping[Hashable, float],
    second: Mapping[Hashable, float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    keys = sorted(set(first) & set(second), key=str)
    differences = np.asarray([first[key] - second[key] for key in keys], dtype=np.float64)
    point = float(np.mean(differences)) if differences.size else float("nan")
    if differences.size < 2 or draws <= 0:
        ci = (float("nan"), float("nan"))
    else:
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        samples = np.empty(int(draws), dtype=np.float64)
        for draw in range(int(draws)):
            samples[draw] = np.mean(
                rng.choice(differences, size=differences.size, replace=True)
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
        finite = np.isfinite(score[index])
        index = index[finite]
        if np.any(labels[index] == 0) and np.any(labels[index] == 1):
            groups.append(index)
    if not groups:
        return float("nan")
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    null = []
    for _ in range(int(permutations)):
        problem_values = []
        for index in groups:
            permuted = rng.permutation(labels[index])
            error = score[index[permuted == 1]]
            correct = score[index[permuted == 0]]
            if error.size and correct.size:
                problem_values.append(_pair_auc(error, correct))
        if problem_values:
            null.append(float(np.mean(problem_values)))
    if not null:
        return float("nan")
    return float((1 + np.sum(np.asarray(null) >= observed)) / (len(null) + 1))


def _evaluate_response_score(
    name: str,
    score: np.ndarray,
    result: FeasibleTangentResult,
    cfg: FeasibleTangentValidationConfig,
) -> tuple[dict[str, Any], dict[Hashable, float]]:
    dataset = result.dataset
    score = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(score)
    values = _problem_auc_values(score, dataset.y_error, dataset.problem_ids)
    point, ci = _bootstrap_mean(
        values,
        draws=cfg.bootstrap,
        seed=cfg.random_seed + sum(ord(ch) for ch in name),
    )
    return (
        {
            "name": name,
            "n": int(finite.sum()),
            "coverage": float(finite.mean()),
            "errors": int(np.sum(finite & (dataset.y_error == 1))),
            "pooled_auroc": auroc(dataset.y_error, score),
            "pooled_auprc": auprc(dataset.y_error, score),
            "within_problem_auroc_equal_weight": point,
            "within_problem_ci95": list(ci),
            "within_problem_problems": len(values),
            "within_problem_permutation_p": _within_problem_permutation_p(
                score,
                dataset.y_error,
                dataset.problem_ids,
                point,
                permutations=cfg.permutations,
                seed=cfg.random_seed + 31 + sum(ord(ch) for ch in name),
            ),
        },
        values,
    )


def _problem_correct_contrast(
    first: np.ndarray,
    second: np.ndarray,
    result: FeasibleTangentResult,
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


def _problem_correct_mean(
    score: np.ndarray,
    result: FeasibleTangentResult,
) -> dict[Hashable, float]:
    dataset = result.dataset
    output: dict[Hashable, float] = {}
    for raw_problem in np.unique(dataset.problem_ids):
        index = np.where(
            (dataset.problem_ids == raw_problem) & (dataset.y_error == 0)
        )[0]
        finite = np.asarray(score[index], dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            output[problem_key(raw_problem)] = float(np.mean(finite))
    return output


def _contrast_summary(
    values: Mapping[Hashable, float],
    *,
    cfg: FeasibleTangentValidationConfig,
    seed_offset: int,
) -> dict[str, Any]:
    point, ci = _bootstrap_mean(
        values,
        draws=cfg.bootstrap,
        seed=cfg.random_seed + seed_offset,
    )
    return {"mean_difference": point, "ci95": list(ci), "problems": len(values)}


def evaluate_feasible_tangent(
    result: FeasibleTangentResult,
    cfg: FeasibleTangentValidationConfig,
) -> dict[str, Any]:
    """Evaluate geometry existence first and error detection second."""

    dataset = result.dataset
    score = {name: result.chain_score(name) for name in result.chain_score_names}
    controls = np.column_stack(
        [
            np.log1p(dataset.n_steps),
            np.log1p(dataset.response_chars),
            np.square(np.log1p(dataset.n_steps)),
            np.square(np.log1p(dataset.response_chars)),
            np.log1p(dataset.n_steps) * np.log1p(dataset.response_chars),
        ]
    )
    residual_cfg = FlowAuditConfig(
        folds=cfg.folds,
        bootstrap=0,
        permutations=0,
        seed=cfg.random_seed,
        compute_device="cpu",
    )
    residual_names = (
        "primary_escape_mean",
        "primary_escape_late",
        "primary_coherent_escape",
        "primary_late_coherent_escape",
        "primary_normal_persistence",
        "phase_coherent_escape",
        "shuffle_coherent_escape",
        "wrong_problem_coherent_escape",
    )
    residual = {
        name: crossfit_residualize_score(
            score[name], controls, dataset.problem_ids, residual_cfg
        )
        for name in residual_names
    }

    primary = score["primary_escape_mean"]
    gate1_contrasts = {
        "phase_minus_primary": _contrast_summary(
            _problem_correct_contrast(score["phase_escape_mean"], primary, result),
            cfg=cfg,
            seed_offset=101,
        ),
        "shuffle_minus_primary": _contrast_summary(
            _problem_correct_contrast(score["shuffle_escape_mean"], primary, result),
            cfg=cfg,
            seed_offset=103,
        ),
        "wrong_problem_minus_primary": _contrast_summary(
            _problem_correct_contrast(
                score["wrong_problem_escape_mean"], primary, result
            ),
            cfg=cfg,
            seed_offset=107,
        ),
        "random_minus_primary": _contrast_summary(
            _problem_correct_contrast(score["random_escape_mean"], primary, result),
            cfg=cfg,
            seed_offset=109,
        ),
    }
    rank_problem = _problem_correct_mean(score["rank_support_rate"], result)
    rank_support, rank_ci = _bootstrap_mean(
        rank_problem,
        draws=cfg.bootstrap,
        seed=cfg.random_seed + 113,
    )
    primary_problem = _problem_correct_mean(primary, result)
    gate1_conditions = {
        "correct_problem_count_at_least_minimum": len(primary_problem)
        >= cfg.min_problem_count,
        "rank_support_at_least_threshold": rank_support >= cfg.min_rank_coverage,
        "time_shuffle_contrast_ci_above_zero": gate1_contrasts[
            "shuffle_minus_primary"
        ]["ci95"][0]
        > 0.0,
        "wrong_problem_contrast_ci_above_zero": gate1_contrasts[
            "wrong_problem_minus_primary"
        ]["ci95"][0]
        > 0.0,
    }
    gate1_pass = all(gate1_conditions.values())

    diagnostics: dict[str, Any] = {}
    auc_values: dict[str, dict[Hashable, float]] = {}
    control_scores = {
        "control.log1p_n_steps": np.log1p(dataset.n_steps),
        "control.log1p_response_chars": np.log1p(dataset.response_chars),
    }
    for name, values in control_scores.items():
        diagnostics[name], auc_values[name] = _evaluate_response_score(
            name, values, result, cfg
        )
    for name in (
        "primary_escape_mean",
        "primary_escape_late",
        "primary_coherent_escape",
        "primary_late_coherent_escape",
        "primary_normal_persistence",
        "phase_coherent_escape",
        "shuffle_coherent_escape",
        "wrong_problem_coherent_escape",
    ):
        diagnostics[name], auc_values[name] = _evaluate_response_score(
            name, score[name], result, cfg
        )
        residual_name = f"{name}.length_residual"
        diagnostics[residual_name], auc_values[residual_name] = _evaluate_response_score(
            residual_name, residual[name], result, cfg
        )

    primary_name = "primary_coherent_escape.length_residual"
    shuffle_name = "shuffle_coherent_escape.length_residual"
    wrong_name = "wrong_problem_coherent_escape.length_residual"
    primary_diag = diagnostics[primary_name]
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
        "response_score_coverage_at_least_threshold": primary_diag["coverage"]
        >= cfg.min_score_coverage,
        "contrastive_problem_count_at_least_minimum": primary_diag[
            "within_problem_problems"
        ]
        >= cfg.min_problem_count,
        "length_residual_auc_ci_above_chance": primary_diag[
            "within_problem_ci95"
        ][0]
        > 0.5,
        "primary_beats_time_shuffle_ci_above_zero": primary_vs_shuffle["ci95"][0]
        > 0.0,
    }
    gate2_pass = all(gate2_conditions.values())
    advance = gate1_pass and gate2_pass

    return {
        "method": "same_problem_feasible_tangent_gate",
        "claim_order": [
            "healthy_low_rank_tangent_exists",
            "errors_show_persistent_normal_escape",
            "only_then_test_output_cotangent_coupling",
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
            "rank_support_equal_problem_mean": rank_support,
            "rank_support_ci95": list(rank_ci),
            "rank_support_problems": len(rank_problem),
            "contrasts_on_correct_targets": gate1_contrasts,
            "state_conditioning_subgate": {
                "pass": gate1_contrasts["phase_minus_primary"]["ci95"][0] > 0.0,
                "note": "descriptive subgate; phase-only may be sufficient for some tasks",
            },
        },
        "error_escape_gate": {
            "pass": gate2_pass,
            "primary_score": primary_name,
            "conditions": gate2_conditions,
            "primary_minus_time_shuffle_auc": primary_vs_shuffle,
            "primary_minus_wrong_problem_auc": primary_vs_wrong,
        },
        "response_diagnostics": diagnostics,
        "decision": {
            "advance_to_output_cotangent_extraction": advance,
            "status": (
                "advance_to_exact_output_cotangent"
                if advance
                else "blocked_until_feasible_tangent_passes"
            ),
            "logits_used_in_this_audit": False,
            "classifier_trained_in_this_audit": False,
        },
        "scoring_metadata": result.metadata,
    }
