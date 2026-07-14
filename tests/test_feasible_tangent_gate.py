from __future__ import annotations

import json

import numpy as np
import torch

from prompt_control_flow.feasible_tangent import (
    FeasibleTangentConfig,
    FeasibleTangentValidationConfig,
    evaluate_feasible_tangent,
    run_feasible_tangent_gate,
    write_feasible_tangent_report,
)
from prompt_control_flow.feasible_tangent.data import feasible_tangent_preflight
from prompt_control_flow.feasible_tangent.scoring import _fit_primary
from prompt_control_flow.flow_signature_data import load_flow_trajectory_dataset


def _orthogonal_component(value: np.ndarray, basis: np.ndarray) -> np.ndarray:
    residual = value - basis @ (basis.T @ value)
    return residual / np.linalg.norm(residual)


def _orthonormal_columns(values: np.ndarray) -> np.ndarray:
    columns = []
    for column in np.asarray(values, dtype=np.float64).T:
        vector = column.copy()
        for previous in columns:
            vector -= previous * np.dot(previous, vector)
        vector /= np.linalg.norm(vector)
        columns.append(vector)
    return np.stack(columns, axis=1)


def _write_synthetic_multisample(path) -> None:
    rng = np.random.default_rng(13)
    n_problem = 8
    n_correct = 7
    n_error = 2
    transitions = 5
    hidden = 24
    trajectories = []
    problem_ids = []
    sample_idx = []
    is_correct = []
    responses = []
    for problem in range(n_problem):
        phase_basis = _orthonormal_columns(
            rng.normal(size=(hidden, transitions))
        )
        anchor = rng.normal(size=hidden)
        anchor /= np.linalg.norm(anchor)
        error_direction = _orthogonal_component(
            rng.normal(size=hidden), phase_basis
        )
        for sample in range(n_correct + n_error):
            erroneous = sample >= n_correct
            state = anchor + 0.01 * rng.normal(size=hidden)
            chain = [state.copy()]
            for transition in range(transitions):
                direction = phase_basis[:, transition] + 0.005 * rng.normal(size=hidden)
                if erroneous and transition >= 2:
                    direction = 0.10 * direction + error_direction
                direction /= np.linalg.norm(direction)
                state = state + 0.35 * direction
                chain.append(state.copy())
            chain_array = np.asarray(chain, dtype=np.float32)
            trajectories.append(
                np.stack(
                    [chain_array, 1.1 * chain_array, 0.9 * chain_array],
                    axis=1,
                )
            )
            problem_ids.append(problem)
            sample_idx.append(sample)
            is_correct.append(int(not erroneous))
            responses.append("x" * (80 + 3 * sample + problem))
    np.savez_compressed(
        path,
        sv_vec_step_exp=np.stack(trajectories),
        problem_ids=np.asarray(problem_ids),
        sample_idx=np.asarray(sample_idx),
        is_correct=np.asarray(is_correct),
        responses=np.asarray(responses, dtype=object),
        layers_used=np.asarray([14, 16, 18]),
        model_name=np.asarray("synthetic-observer"),
    )


def test_adaptive_rank_rejects_forced_low_rank_support() -> None:
    references = [np.eye(6, dtype=np.float32)[:4]]
    target = np.asarray([[1, 0, 0, 0, 0, 0]], dtype=np.float32)
    score = _fit_primary(
        references,
        target,
        FeasibleTangentConfig(
            device="cpu",
            max_rank=2,
            min_donors=4,
            max_donors=4,
            rank_energy=0.90,
        ),
        device=torch.device("cpu"),
    )
    assert not bool(score.supported[0])
    assert score.selected_rank[0] == 2
    assert score.captured_energy[0] < 0.9


def test_same_problem_tangent_and_escape_gates(tmp_path) -> None:
    source = tmp_path / "same_problem.npz"
    _write_synthetic_multisample(source)
    dataset = load_flow_trajectory_dataset(
        source,
        vector_key="sv_vec_step_exp",
        layers="all",
        label_policy="answer",
    )
    config = FeasibleTangentConfig(
        device="cpu",
        batch_size=13,
        layer_batch_size=2,
        phase_sigma=0.15,
        causal_time_scale=4.0,
        rank_energy=0.90,
        max_rank=2,
        min_donors=4,
        max_donors=6,
        wrong_problem_draws=2,
        random_seed=5,
    )
    preflight = feasible_tangent_preflight(dataset, config)
    assert preflight["contrastive_problems"] == 8
    assert preflight["correct_leave_one_out_eligible_problems"] == 8

    result = run_feasible_tangent_gate(dataset, config)
    assert result.metadata["target_label_used_in_scoring"] is False
    assert result.metadata["phase_uses_final_response_length"] is False
    assert result.metadata["rank_supported_fraction"] > 0.95

    correct = dataset.y_error == 0
    error = dataset.y_error == 1
    primary = result.chain_score("primary_escape_mean")
    shuffle = result.chain_score("shuffle_escape_mean")
    wrong = result.chain_score("wrong_problem_escape_mean")
    coherent = result.chain_score("primary_coherent_escape")
    assert np.nanmean(primary[correct]) < np.nanmean(shuffle[correct])
    assert np.nanmean(primary[correct]) < np.nanmean(wrong[correct])
    assert np.nanmean(coherent[error]) > np.nanmean(coherent[correct])

    validation = FeasibleTangentValidationConfig(
        folds=4,
        bootstrap=100,
        permutations=25,
        min_rank_coverage=0.8,
        min_score_coverage=0.8,
        min_problem_count=6,
        random_seed=7,
    )
    summary = evaluate_feasible_tangent(result, validation)
    assert summary["geometry_existence_gate"]["pass"] is True
    assert (
        summary["response_diagnostics"][
            "primary_coherent_escape.length_residual"
        ]["within_problem_auroc_equal_weight"]
        > 0.8
    )

    output = tmp_path / "report"
    _, paths = write_feasible_tangent_report(result, output, validation)
    assert (output / "feasible_tangent_scores.npz").exists()
    assert (output / "chain_scores.csv").exists()
    assert "geometry_existence_gate" in json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert paths["summary_md"].endswith("summary.md")
