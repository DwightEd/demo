from __future__ import annotations

import numpy as np

from prompt_control_flow.layer_time_evaluate import (
    LayerTimeValidationConfig,
    evaluate_layer_time_geometry,
    residualize_step_signal,
)
from prompt_control_flow.layer_time_geometry import LAYER_TIME_FIELD_NAMES


def _synthetic_metrics() -> dict[str, np.ndarray]:
    n_problems, samples_per_problem = 6, 2
    n_chains = n_problems * samples_per_problem
    n_steps, n_layers = 5, 3
    field = np.full(
        (n_chains, n_steps, n_layers, len(LAYER_TIME_FIELD_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    names = {name: i for i, name in enumerate(LAYER_TIME_FIELD_NAMES)}
    correct = np.tile(np.asarray([1, 0], dtype=np.int64), n_problems)
    gold = np.where(correct == 0, 2, -1).astype(np.int64)
    rng = np.random.default_rng(9)
    for chain in range(n_chains):
        for step in range(n_steps):
            field[chain, step, :, names["lid"]] = 4.0
            field[chain, step, :, names["fiber_rank"]] = 4.0
            field[chain, step, :, names["depth_neighbor_rewire"]] = 0.1
            field[chain, step, :, names["rank_singularity"]] = 0.0
            if step == 0:
                continue
            field[chain, step, :, names["time_neighbor_rewire"]] = 0.1
            field[chain, step, :, names["depth_tangent_drift"]] = 0.05
            field[chain, step, :, names["time_tangent_drift"]] = 0.05
            transport = 0.08 + 0.01 * rng.normal(size=n_layers)
            field[chain, step, :, names["plaquette_transport_residual"]] = transport
            curvature = np.full(n_layers, 0.1, dtype=np.float32)
            if correct[chain] == 0 and step == 2:
                curvature += 0.8
            field[chain, step, :, names["plaquette_holonomy"]] = curvature
            field[chain, step, :, names["plaquette_wilson_curvature"]] = curvature
            field[chain, step, :, names["plaquette_reliable_wilson"]] = curvature
    rel_pos = np.tile(np.linspace(0.0, 1.0, n_steps), (n_chains, 1))
    step_len = np.full((n_chains, n_steps), 8.0)
    return {
        "chain_idx": np.arange(n_chains, dtype=np.int64),
        "problem_id": np.repeat(np.arange(n_problems), samples_per_problem),
        "gold_error_step": gold,
        "is_correct": correct,
        "n_steps": np.full(n_chains, n_steps, dtype=np.int64),
        "step_scores": np.stack([step_len, rel_pos], axis=2),
        "step_score_names": np.asarray(["step_len", "rel_pos"]),
        "layer_time_geometry_field": field,
        "layer_time_geometry_field_names": np.asarray(LAYER_TIME_FIELD_NAMES),
        "layer_time_geometry_layers": np.asarray([1, 2, 3]),
    }


def test_claim_driven_layer_time_evaluation_detects_local_curvature_event() -> None:
    summary = evaluate_layer_time_geometry(
        _synthetic_metrics(),
        LayerTimeValidationConfig(bootstrap=100, random_seed=3),
    )
    primary = summary["observables"]["plaquette_reliable_wilson"]
    assert primary["event_adjusted"]["peak_offset"] == 0
    assert primary["event_adjusted"]["local_peak_contrast"]["mean"] > 0.5
    assert primary["same_problem"]["eligible_problems"] == 6
    assert primary["same_problem"]["mean_problem_delta"] > 0.0
    assert primary["same_problem"]["pair_micro_auroc"] == 1.0
    assert summary["conditional_wilson_event"]["peak_offset"] == 0
    assert summary["claim_gate"]["event_peak_at_zero"]


def test_problem_fixed_effect_residualization_removes_position_trend() -> None:
    n_chains, n_steps = 4, 6
    rel_pos = np.tile(np.linspace(0.0, 1.0, n_steps), (n_chains, 1))
    signal = 3.0 * rel_pos + np.arange(n_chains)[:, None] * 7.0
    residual = residualize_step_signal(
        signal,
        n_steps=np.full(n_chains, n_steps),
        problem_id=np.arange(n_chains),
        controls=(rel_pos, np.ones_like(rel_pos)),
    )
    assert np.nanmax(np.abs(residual)) < 1e-10
