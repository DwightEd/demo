from __future__ import annotations

import numpy as np
import pytest

from crwh.adapters import from_attnhyper, from_mvht_view


def test_attnhyper_adapter_preserves_incidence_and_marks_attention_baseline() -> None:
    payload = {
        "source_id": "legacy-1",
        "response_idx": 2,
        "x": np.arange(12, dtype=np.float64).reshape(4, 3),
        "he_incidence_index": np.asarray([[0, 1, 2, 1, 3], [0, 0, 0, 1, 1]]),
        "he_attr": np.asarray([[0.4, 0.7, 0.0], [0.3, 0.5, 1.0]]),
        "y_token": np.asarray([-1, -1, 0, 1]),
        "token_ids": np.asarray([10, 11, 101, 102]),
    }

    graph = from_attnhyper(payload, view_name="legacy", num_heads=2)

    assert np.array_equal(graph.incidence, payload["he_incidence_index"])
    assert np.array_equal(graph.receivers, np.asarray([2, 3]))
    assert np.array_equal(graph.response_token_ids, np.asarray([101, 102]))
    assert np.array_equal(graph.token_labels, np.asarray([0, 1]))
    assert set(graph.edge_kind.tolist()) == {"adapted_attention_ablation"}


def test_mvht_adapter_exposes_either_subspace_as_a_view() -> None:
    common = {
        "x_S": np.ones((4, 2)),
        "x_R": 2.0 * np.ones((4, 2)),
        "he_index_S": np.asarray([[0, 2, 1, 3], [0, 0, 1, 1]]),
        "he_index_R": np.asarray([[1, 2, 0, 3], [0, 0, 1, 1]]),
        "he_attr_S": np.asarray([[0.2, 0.5, 0.0], [0.3, 0.6, 1.0]]),
        "he_attr_R": np.asarray([[0.4, 0.7, 0.0], [0.5, 0.8, 1.0]]),
        "response_mask": np.asarray([False, False, True, True]),
        "token_ids": np.asarray([10, 11, 101, 102]),
    }

    semantic = from_mvht_view(
        common,
        sample_id="mvht-1",
        view="semantic",
        num_heads=2,
    )
    reasoning = from_mvht_view(
        common,
        sample_id="mvht-1",
        view="reasoning",
        num_heads=2,
    )

    assert np.all(semantic.node_features == 1.0)
    assert np.all(reasoning.node_features == 2.0)
    assert semantic.view_name == "mvht_semantic"
    assert reasoning.view_name == "mvht_reasoning"


def test_legacy_adapter_rejects_hyperedges_without_response_receiver() -> None:
    payload = {
        "source_id": "bad",
        "response_idx": 2,
        "x": np.ones((4, 2)),
        "he_incidence_index": np.asarray([[0, 1], [0, 0]]),
        "he_attr": np.asarray([[0.2, 0.5, 0.0]]),
        "y_token": np.asarray([-1, -1, 0, 1]),
        "token_ids": np.asarray([10, 11, 101, 102]),
    }

    with pytest.raises(ValueError, match="response receiver"):
        from_attnhyper(payload)


def test_legacy_adapter_does_not_silently_use_positions_as_token_ids() -> None:
    payload = {
        "source_id": "missing-token-ids",
        "response_idx": 2,
        "x": np.ones((4, 2)),
        "he_incidence_index": np.asarray([[0, 2], [0, 0]]),
        "he_attr": np.asarray([[0.2, 0.5, 0.0]]),
        "y_token": np.asarray([-1, -1, 0, 1]),
    }

    with pytest.raises(ValueError, match="token_ids"):
        from_attnhyper(payload)
