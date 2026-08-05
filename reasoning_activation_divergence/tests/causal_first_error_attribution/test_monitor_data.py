from __future__ import annotations

from pathlib import Path
import json

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
    selected = root / domain / "selected"
    selected.mkdir(parents=True)
    state_path = geometry / "trace.states.pre.test.npy"
    np.save(state_path, np.asarray(pre_states, dtype=np.float32))
    np.savez_compressed(
        selected / "trace.npz",
        chain_idx=np.asarray(chain_ids, dtype=np.int64),
        gold_error_step=np.asarray(first_errors, dtype=np.int64),
        n_steps=np.asarray(n_steps, dtype=np.int64),
        step_token_ranges=np.asarray(step_ranges, dtype=np.int64),
        dataset=np.asarray([domain] * len(chain_ids), dtype=object),
        step_scores=np.asarray(step_scores, dtype=np.float32),
        step_score_names=np.asarray(["token_entropy", "token_nll"]),
    )
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
        step_scores=np.empty((*np.asarray(step_scores).shape[:2], 0), dtype=np.float32),
        step_score_names=np.asarray([], dtype=object),
        step_pre_state_memmap_path=np.asarray(state_path.name, dtype=object),
        step_pre_state_memmap_count=np.asarray(len(pre_states), dtype=np.int64),
        step_pre_state_vector_chain_idx=np.asarray(point_chain_ids, dtype=np.int64),
        step_pre_state_vector_step_idx=np.asarray(point_step_ids, dtype=np.int64),
        step_layer_state_vector_layers=np.asarray([1, 2, 3], dtype=np.int64),
        state_representation_kind=np.asarray("hidden_state", dtype=object),
        hidden_state_token_semantics=np.asarray(
            "h_i_after_reading_token_i", dtype=object
        ),
        step_prediction_position_shift=np.asarray(-1, dtype=np.int8),
        metadata_json=np.asarray(
            [
                json.dumps(
                    {
                        "step_pre_state_temporal_semantics": (
                            "causal_before_first_step_token"
                        )
                    }
                )
                for _ in chain_ids
            ],
            dtype=object,
        ),
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


def test_monitor_history_contains_only_same_chain_states_through_candidate(
    tmp_path,
) -> None:
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
        step_scores=np.zeros((2, 3, 2), dtype=np.float32),
    )

    data = load_processbench_monitor_data(tmp_path, ("gsm8k",))

    np.testing.assert_array_equal(data.history(1), states[[0, 1]])
    np.testing.assert_array_equal(data.history(1, max_steps=1), states[[1]])
    np.testing.assert_array_equal(data.history(2), states[[3]])
    np.testing.assert_array_equal(data.boundary_pair(0), states[[0, 0]])
    np.testing.assert_array_equal(data.boundary_pair(1), states[[0, 1]])
    np.testing.assert_array_equal(data.boundary_pair(2), states[[3, 3]])


def test_output_context_uses_only_completed_steps(tmp_path) -> None:
    states = np.ones((2, 3, 4), dtype=np.float32)
    _write_domain(
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

    output_trace = tmp_path / "math" / "selected" / "trace.npz"
    with np.load(output_trace, allow_pickle=True) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload["step_scores"] = payload["step_scores"].copy()
    payload["step_scores"][0, 1] = [-999.0, -999.0]
    np.savez_compressed(output_trace, **payload)
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


def test_monitor_loader_rejects_a_stable_problem_hash_spanning_domains(
    tmp_path,
) -> None:
    for domain in ("gsm8k", "math"):
        trace = _write_domain(
            tmp_path,
            domain,
            pre_states=np.ones((1, 3, 4), dtype=np.float32),
            chain_ids=np.asarray([1]),
            first_errors=np.asarray([-1]),
            step_ranges=np.asarray([[[3, 4]]]),
            n_steps=np.asarray([1]),
            point_chain_ids=np.asarray([1]),
            point_step_ids=np.asarray([0]),
            step_scores=np.asarray([[[0.1, 0.2]]]),
        )
        with np.load(trace, allow_pickle=True) as archive:
            payload = {name: np.asarray(archive[name]) for name in archive.files}
        payload["problem_ids"] = np.asarray(["problem_sha256:shared"])
        np.savez_compressed(trace, **payload)

    with pytest.raises(ValueError, match="problem hash spans LODO domains"):
        load_processbench_monitor_data(tmp_path, ("gsm8k", "math"))


def test_monitor_loader_rejects_a_forged_noncausal_pre_step_view(tmp_path) -> None:
    trace = _write_domain(
        tmp_path,
        "gsm8k",
        pre_states=np.ones((1, 3, 4), dtype=np.float32),
        chain_ids=np.asarray([1]),
        first_errors=np.asarray([-1]),
        step_ranges=np.asarray([[[3, 4]]]),
        n_steps=np.asarray([1]),
        point_chain_ids=np.asarray([1]),
        point_step_ids=np.asarray([0]),
        step_scores=np.asarray([[[0.1, 0.2]]]),
    )
    with np.load(trace, allow_pickle=True) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload["metadata_json"] = np.asarray(
        [json.dumps({"step_pre_state_temporal_semantics": "post_step_end"})],
        dtype=object,
    )
    np.savez_compressed(trace, **payload)

    with pytest.raises(ValueError, match="causal pre-step semantics"):
        load_processbench_monitor_data(tmp_path, ("gsm8k",))


def test_monitor_loader_rejects_an_unordered_layer_axis(tmp_path) -> None:
    trace = _write_domain(
        tmp_path,
        "math",
        pre_states=np.ones((1, 3, 4), dtype=np.float32),
        chain_ids=np.asarray([1]),
        first_errors=np.asarray([-1]),
        step_ranges=np.asarray([[[3, 4]]]),
        n_steps=np.asarray([1]),
        point_chain_ids=np.asarray([1]),
        point_step_ids=np.asarray([0]),
        step_scores=np.asarray([[[0.1, 0.2]]]),
    )
    with np.load(trace, allow_pickle=True) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload["step_layer_state_vector_layers"] = np.asarray([1, 3, 2])
    np.savez_compressed(trace, **payload)

    with pytest.raises(ValueError, match="strictly increasing"):
        load_processbench_monitor_data(tmp_path, ("math",))


def test_smoke_limit_samples_complete_error_and_correct_problem_groups(
    tmp_path,
) -> None:
    _write_domain(
        tmp_path,
        "gsm8k",
        pre_states=np.ones((6, 3, 4), dtype=np.float32),
        chain_ids=np.asarray([1, 2, 3]),
        first_errors=np.asarray([1, 1, -1]),
        step_ranges=np.asarray([[[3, 4], [5, 6]]] * 3),
        n_steps=np.asarray([2, 2, 2]),
        point_chain_ids=np.repeat(np.asarray([1, 2, 3]), 2),
        point_step_ids=np.tile(np.asarray([0, 1]), 3),
        step_scores=np.asarray([[[0.1, 0.2], [0.3, 0.4]]] * 3),
    )

    data = load_processbench_monitor_data(
        tmp_path, ("gsm8k",), max_chains_per_domain=2
    )

    selected = {row.chain_id for row in data.rows}
    assert len(selected) == 2
    assert any(row.first_error_step == -1 for row in data.rows)
    assert any(row.first_error_step >= 0 for row in data.rows)
