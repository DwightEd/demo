from __future__ import annotations

from dataclasses import replace

import numpy as np

from functional_divergence.hidden_state_geometry.contracts import ChainSample
from functional_divergence.hidden_state_geometry.tasks import (
    build_strict_prefix_task,
    build_whole_chain_task,
    load_visible_states,
    nuisance_features,
    visible_output_steps,
)


def _sample(tmp_path, *, chain_id: int, gold: int, n_steps: int = 4) -> ChainSample:
    state = np.empty((n_steps * 2, 2, 5), dtype=np.float32)
    for token in range(len(state)):
        state[token] = token
    path = tmp_path / f"chain_{chain_id}.npy"
    np.save(path, state)
    ranges = np.asarray(
        [[10 + 2 * step, 11 + 2 * step] for step in range(n_steps)], dtype=np.int64
    )
    output = np.arange(n_steps * 2, dtype=np.float32).reshape(n_steps, 2)
    return ChainSample(
        chain_id=chain_id,
        manifest_row=chain_id,
        problem_group=str(chain_id),
        dataset="gsm8k",
        generator="Llama-3.1-8B",
        observer_model="Llama-3.1-8B",
        state_path=path,
        state_count=len(state),
        response_start=10,
        step_ranges=ranges,
        layer_ids=np.asarray([8, 10]),
        output_steps=output,
        output_feature_names=("token_entropy", "token_nll"),
        first_error_step=gold,
    )


def test_strict_prefix_uses_step_start_boundary_and_stops_at_first_error(tmp_path):
    error = _sample(tmp_path, chain_id=1, gold=3)
    correct = _sample(tmp_path, chain_id=2, gold=-1)

    task = build_strict_prefix_task((error, correct))

    error_rows = [row for row in task.examples if row.sample.chain_id == 1]
    assert [row.boundary_step for row in error_rows] == [1, 2, 3]
    assert task.labels[:3].tolist() == [0, 0, 1]
    # Predicting step 2 reads h[a_2 - 1], which is the end of completed step 1.
    assert load_visible_states(error_rows[1])[-1, 0, 0] == 3.0
    assert visible_output_steps(error_rows[1]).shape == (2, 2)


def test_future_and_post_error_mutation_cannot_change_strict_prefix(tmp_path):
    original = _sample(tmp_path, chain_id=3, gold=2)
    task = build_strict_prefix_task((original,))
    event_row = task.examples[-1]
    before_hidden = load_visible_states(event_row).copy()
    before_output = visible_output_steps(event_row).copy()

    shard = np.load(original.state_path)
    shard[4:] += 10_000
    np.save(original.state_path, shard)
    original.output_steps[2:] += 10_000

    assert np.array_equal(load_visible_states(event_row), before_hidden)
    assert np.array_equal(visible_output_steps(event_row), before_output)


def test_step_zero_error_is_left_truncated_and_not_faked_from_response_state(tmp_path):
    task = build_strict_prefix_task((_sample(tmp_path, chain_id=4, gold=0),))

    assert len(task.examples) == 0
    assert task.left_truncated_step0_errors == 1


def test_whole_chain_is_separate_retrospective_task_and_sees_post_error(tmp_path):
    sample = _sample(tmp_path, chain_id=5, gold=1)
    task = build_whole_chain_task((sample,))
    before = load_visible_states(task.examples[0]).copy()
    shard = np.load(sample.state_path)
    shard[-1] += 99
    np.save(sample.state_path, shard)

    assert task.claim_scope == "retrospective_information_ceiling"
    assert not np.array_equal(load_visible_states(task.examples[0]), before)


def test_strict_prefix_nuisance_never_uses_final_length_or_relative_progress(tmp_path):
    sample = _sample(tmp_path, chain_id=6, gold=3)
    task = build_strict_prefix_task((sample,))

    names, values = nuisance_features(task.examples[1])

    assert names == ("step_index", "prefix_token_count", "previous_step_length")
    assert values.shape == (3,)
