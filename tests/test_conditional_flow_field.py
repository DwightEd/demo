from __future__ import annotations

import json

import numpy as np
import torch

from prompt_control_flow.conditional_flow_field import (
    ConditionalFlowFieldConfig,
    ConditionalFlowFieldValidationConfig,
    conditional_flow_field_preflight,
    evaluate_conditional_flow_field,
    run_conditional_flow_field,
    spherical_energy_score,
    write_conditional_flow_field_report,
)
from prompt_control_flow.flow_signature_data import load_flow_trajectory_dataset


def _orthogonal(value: np.ndarray, columns: np.ndarray) -> np.ndarray:
    residual = value - columns @ (columns.T @ value)
    return residual / np.linalg.norm(residual)


def _orthonormal(values: np.ndarray) -> np.ndarray:
    columns = []
    for column in np.asarray(values, dtype=np.float64).T:
        vector = column.copy()
        for previous in columns:
            vector -= previous * np.dot(previous, vector)
        vector /= np.linalg.norm(vector)
        columns.append(vector)
    return np.stack(columns, axis=1)


def _write_synthetic(path) -> None:
    rng = np.random.default_rng(29)
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
        phase = _orthonormal(rng.normal(size=(hidden, transitions)))
        error_direction = _orthogonal(rng.normal(size=hidden), phase)
        anchor = rng.normal(size=hidden)
        anchor /= np.linalg.norm(anchor)
        for sample in range(n_correct + n_error):
            erroneous = sample >= n_correct
            state = anchor + 0.002 * rng.normal(size=hidden)
            chain = [state.copy()]
            for transition in range(transitions):
                direction = phase[:, transition] + 0.004 * rng.normal(size=hidden)
                if erroneous and transition >= 2:
                    direction = 0.05 * direction + error_direction
                direction /= np.linalg.norm(direction)
                state = state + 0.4 * direction
                chain.append(state.copy())
            chain = np.asarray(chain, dtype=np.float32)
            trajectories.append(np.stack([chain, 1.1 * chain], axis=1))
            problem_ids.append(problem)
            sample_idx.append(sample)
            is_correct.append(int(not erroneous))
            responses.append("x" * (120 + problem + sample))
    np.savez_compressed(
        path,
        sv_vec_step_exp=np.stack(trajectories),
        problem_ids=np.asarray(problem_ids),
        sample_idx=np.asarray(sample_idx),
        is_correct=np.asarray(is_correct),
        responses=np.asarray(responses, dtype=object),
        layers_used=np.asarray([14, 16]),
        model_name=np.asarray("synthetic-observer"),
    )


def test_spherical_energy_score_is_rotation_invariant_and_detects_excursion() -> None:
    rng = np.random.default_rng(7)
    references = np.zeros((2, 6, 8), dtype=np.float32)
    references[:, :, 0] = 1.0
    references += 0.01 * rng.normal(size=references.shape).astype(np.float32)
    targets = np.zeros((2, 8), dtype=np.float32)
    targets[0, 0] = 1.0
    targets[1, 1] = 1.0
    rotation = _orthonormal(rng.normal(size=(8, 8))).astype(np.float32)

    first, first_z = spherical_energy_score(
        torch.as_tensor(targets), torch.as_tensor(references)
    )
    target_rotated = torch.as_tensor(targets) @ torch.as_tensor(rotation)
    reference_rotated = torch.as_tensor(references) @ torch.as_tensor(rotation)
    second, second_z = spherical_energy_score(
        target_rotated,
        reference_rotated,
    )
    assert first[1] > first[0]
    assert first_z[1] > first_z[0]
    np.testing.assert_allclose(first.numpy(), second.numpy(), atol=2e-5)
    np.testing.assert_allclose(first_z.numpy(), second_z.numpy(), rtol=1e-3, atol=1e-3)

    identical_target = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    identical_refs = identical_target[:, None, :].repeat(1, 4, 1)
    identical_energy, identical_z = spherical_energy_score(
        identical_target, identical_refs
    )
    torch.testing.assert_close(identical_energy, torch.zeros_like(identical_energy))
    torch.testing.assert_close(identical_z, torch.zeros_like(identical_z))


def test_conditional_flow_field_gates_and_matched_donors(tmp_path) -> None:
    source = tmp_path / "field.npz"
    _write_synthetic(source)
    dataset = load_flow_trajectory_dataset(
        source,
        vector_key="sv_vec_step_exp",
        layers="all",
        label_policy="answer",
    )
    scoring = ConditionalFlowFieldConfig(
        device="cpu",
        batch_size=17,
        min_donors=4,
        max_donors=6,
        state_window=1,
        wrong_problem_draws=2,
        random_seed=11,
    )
    preflight = conditional_flow_field_preflight(dataset, scoring)
    assert preflight["eligible_contrastive_problems"] == 8
    assert preflight["donor_count_is_target_label_independent"] is True

    result = run_conditional_flow_field(dataset, scoring)
    assert result.metadata["donor_count_correct_mean"] == 6.0
    assert result.metadata["donor_count_error_mean"] == 6.0
    assert result.metadata["target_label_used_as_predictor"] is False
    correct = dataset.y_error == 0
    error = dataset.y_error == 1
    phase = result.chain_score("phase_energy_mean")
    shuffle = result.chain_score("shuffle_energy_mean")
    wrong = result.chain_score("wrong_problem_energy_mean")
    risk = result.chain_score("phase_calibrated_free_energy")
    assert np.nanmean(phase[correct]) < np.nanmean(shuffle[correct])
    assert np.nanmean(phase[correct]) < np.nanmean(wrong[correct])
    assert np.nanmean(risk[error]) > np.nanmean(risk[correct])

    validation = ConditionalFlowFieldValidationConfig(
        folds=4,
        bootstrap=100,
        permutations=25,
        min_score_coverage=0.8,
        min_problem_count=6,
        random_seed=13,
    )
    summary = evaluate_conditional_flow_field(result, validation)
    assert summary["geometry_existence_gate"]["pass"] is True
    assert (
        summary["response_diagnostics"][
            "phase_calibrated_free_energy.length_residual"
        ]["within_problem_auroc_equal_weight"]
        > 0.8
    )

    output = tmp_path / "report"
    _, paths = write_conditional_flow_field_report(result, output, validation)
    assert (output / "conditional_flow_field_scores.npz").exists()
    assert (output / "chain_scores.csv").exists()
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert "error_excursion_gate" in saved
    assert paths["summary_md"].endswith("summary.md")
