from __future__ import annotations

import numpy as np

from functional_divergence.causal_first_error_attribution.trace_data import (
    load_decision_prefix,
)


def _object_vector(values) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def test_trace_prefix_crops_future_and_marks_separator_tokens(tmp_path) -> None:
    path = tmp_path / "trace.npz"
    np.savez_compressed(
        path,
        full_input_ids=np.asarray([[10, 11, 99, 20, 21, 98, 30, 31, 0]]),
        full_attention_mask=np.asarray([[1, 1, 1, 1, 1, 1, 1, 1, 0]]),
        prompt_token_counts=np.asarray([2]),
        step_token_ranges=np.asarray([[[3, 4], [6, 7]]]),
        n_steps=np.asarray([2]),
        gold_error_step=np.asarray([1]),
        dataset=np.asarray(["gsm8k"]),
    )

    prefix = load_decision_prefix(path, record_index=0, decision_position=6)

    assert prefix.input_ids.tolist() == [10, 11, 99, 20, 21, 98, 30]
    assert prefix.source_step_ids.tolist() == [-1, -1, -2, 0, 0, -2, 1]
    assert prefix.decision_position == 6
    assert 31 not in prefix.input_ids


def test_controlled_trace_does_not_require_a_gold_error_label(tmp_path) -> None:
    path = tmp_path / "trace.npz"
    np.savez_compressed(
        path,
        full_input_ids=np.asarray([[10, 11, 20]]),
        full_attention_mask=np.asarray([[1, 1, 1]]),
        prompt_token_counts=np.asarray([1]),
        step_token_ranges=np.asarray([[[1, 2]]]),
        n_steps=np.asarray([1]),
        dataset=np.asarray(["controlled_math"]),
    )

    prefix = load_decision_prefix(path, record_index=0, decision_position=1)

    assert prefix.first_error_step == -2
    assert prefix.dataset == "controlled_math"


def test_trace_prefix_accepts_production_ragged_object_vectors(tmp_path) -> None:
    path = tmp_path / "trace.npz"
    np.savez_compressed(
        path,
        full_input_ids=_object_vector([np.asarray([10, 11, 99, 20, 21, 98, 30, 31])]),
        full_attention_mask=_object_vector([np.ones(8, dtype=np.int8)]),
        prompt_token_counts=np.asarray([2]),
        step_token_ranges=_object_vector([np.asarray([[3, 4], [6, 7]])]),
        n_steps=np.asarray([2]),
        gold_error_step=np.asarray([1]),
        dataset=np.asarray(["gsm8k"]),
    )

    prefix = load_decision_prefix(path, record_index=0, decision_position=6)

    assert prefix.input_ids.tolist() == [10, 11, 99, 20, 21, 98, 30]
    assert prefix.source_step_ids.tolist() == [-1, -1, -2, 0, 0, -2, 1]
    assert prefix.first_error_step == 1
