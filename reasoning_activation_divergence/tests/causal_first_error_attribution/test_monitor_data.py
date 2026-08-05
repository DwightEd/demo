from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from functional_divergence.causal_first_error_attribution.monitor_data import (
    load_processbench_monitor_data,
)


def _write_domain(
    root: Path,
    domain: str,
    *,
    pre_states: np.ndarray,
    chain_ids: np.ndarray,
    first_errors: np.ndarray,
    step_ranges: np.ndarray,
    n_steps: np.ndarray,
    point_chain_ids: np.ndarray,
    point_step_ids: np.ndarray,
    step_scores: np.ndarray,
) -> Path:
    geometry = root / domain / "geometry"
    geometry.mkdir(parents=True)
    state_path = geometry / "trace.states.pre.test.npy"
    np.save(state_path, np.asarray(pre_states, dtype=np.float32))
    path = geometry / "trace.npz"
    np.savez_compressed(
        path,
        chain_idx=np.asarray(chain_ids, dtype=np.int64),
        gold_error_step=np.asarray(first_errors, dtype=np.int64),
        n_steps=np.asarray(n_steps, dtype=np.int64),
        step_token_ranges=np.asarray(step_ranges, dtype=np.int64),
        prompt_token_counts=np.asarray([3] * len(chain_ids), dtype=np.int64),
        problem_group_id=np.asarray(
            [f"{domain}-problem-{value}" for value in chain_ids], dtype=object
        ),
        problem_ids=np.asarray(
            [f"problem_sha256:{domain}-{value}" for value in chain_ids],
            dtype=object,
        ),
        dataset=np.asarray([domain] * len(chain_ids), dtype=object),
        step_scores=np.asarray(step_scores, dtype=np.float32),
        step_score_names=np.asarray(["token_entropy", "token_nll"]),
        step_pre_state_memmap_path=np.asarray(state_path.name, dtype=object),
        step_pre_state_memmap_count=np.asarray(len(pre_states), dtype=np.int64),
        step_pre_state_vector_chain_idx=np.asarray(point_chain_ids, dtype=np.int64),
        step_pre_state_vector_step_idx=np.asarray(point_step_ids, dtype=np.int64),
        step_layer_state_vector_layers=np.asarray([1, 2, 3], dtype=np.int64),
        state_pooling_kind=np.asarray("boundary_token", dtype=object),
    )
    return path


def test_monitor_risk_set_includes_step_zero_and_excludes_post_error(tmp_path) -> None:
    # chain 10 first fails at step 1; chain 11 is fully correct.
    states = np.arange(5 * 3 * 4, dtype=np.float32).reshape(5, 3, 4)
    _write_domain(
        tmp_path,
        "gsm8k",
        pre_states=states,
        chain_ids=np.asarray([10, 11]),
        first_errors=np.asarray([1, -1]),
        step_ranges=np.asarray(
            [
                [[3, 4], [5, 7], [8, 10]],
                [[3, 3], [4, 6], [-1, -1]],
            ]
        ),
        n_steps=np.asarray([3, 2]),
        point_chain_ids=np.asarray([10, 10, 10, 11, 11]),
        point_step_ids=np.asarray([0, 1, 2, 0, 1]),
        step_scores=np.asarray(
            [
                [[0.1, 1.1], [0.2, 1.2], [999.0, 999.0]],
                [[0.3, 1.3], [0.4, 1.4], [np.nan, np.nan]],
            ]
        ),
    )

    data = load_processbench_monitor_data(tmp_path, ("gsm8k",))
    observed = [
        (row.chain_id, row.candidate_step, row.label, row.decision_position)
        for row in data.rows
    ]

    assert observed == [
        ("gsm8k::10", 0, 0, 2),
        ("gsm8k::10", 1, 1, 4),
        ("gsm8k::11", 0, 0, 2),
        ("gsm8k::11", 1, 0, 3),
    ]
    np.testing.assert_array_equal(data.state(0), states[0])
    np.testing.assert_array_equal(data.state(1), states[1])
    assert data.layer_ids.tolist() == [1, 2, 3]


def test_output_context_uses_only_completed_steps(tmp_path) -> None:
    states = np.ones((2, 3, 4), dtype=np.float32)
    trace = _write_domain(
        tmp_path,
        "math",
        pre_states=states,
        chain_ids=np.asarray([7]),
        first_errors=np.asarray([1]),
        step_ranges=np.asarray([[[3, 4], [5, 8]]]),
        n_steps=np.asarray([2]),
        point_chain_ids=np.asarray([7, 7]),
        point_step_ids=np.asarray([0, 1]),
        step_scores=np.asarray([[[0.5, 1.5], [100.0, 200.0]]]),
    )
    original = load_processbench_monitor_data(tmp_path, ("math",))

    with np.load(trace, allow_pickle=True) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload["step_scores"] = payload["step_scores"].copy()
    payload["step_scores"][0, 1] = [-999.0, -999.0]
    np.savez_compressed(trace, **payload)
    changed = load_processbench_monitor_data(tmp_path, ("math",))

    # Candidate step 1 can see step 0, but never its own output summary.
    np.testing.assert_array_equal(
        original.rows[1].output_context, changed.rows[1].output_context
    )
    np.testing.assert_array_equal(
        original.rows[1].output_context,
        np.asarray([0.5, 1.5, 0.5, 1.5, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        original.rows[0].output_context,
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_monitor_loader_rejects_missing_or_duplicate_boundary_states(tmp_path) -> None:
    states = np.ones((2, 3, 4), dtype=np.float32)
    _write_domain(
        tmp_path,
        "omnimath",
        pre_states=states,
        chain_ids=np.asarray([3]),
        first_errors=np.asarray([1]),
        step_ranges=np.asarray([[[3, 4], [5, 6]]]),
        n_steps=np.asarray([2]),
        point_chain_ids=np.asarray([3, 3]),
        point_step_ids=np.asarray([0, 0]),
        step_scores=np.asarray([[[0.1, 0.2], [0.3, 0.4]]]),
    )

    with pytest.raises(ValueError, match="duplicate pre-step state"):
        load_processbench_monitor_data(tmp_path, ("omnimath",))


def test_monitor_loader_requires_the_causal_pre_step_view(tmp_path) -> None:
    geometry = tmp_path / "gsm8k" / "geometry"
    geometry.mkdir(parents=True)
    np.savez_compressed(
        geometry / "trace.npz",
        chain_idx=np.asarray([1]),
        gold_error_step=np.asarray([-1]),
        n_steps=np.asarray([1]),
    )

    with pytest.raises(ValueError, match="step_pre_state_memmap_path"):
        load_processbench_monitor_data(tmp_path, ("gsm8k",))
