from __future__ import annotations

import numpy as np
import pytest

from functional_divergence.causal_first_error_attribution.contracts import (
    ONSET_TRACE_SCHEMA,
    OnsetTraceArtifact,
)


def _artifact() -> OnsetTraceArtifact:
    mass = np.full((2, 2, 4), 0.25, dtype=np.float32)
    return OnsetTraceArtifact(
        input_ids=np.asarray([101, 102, 201, 202], dtype=np.int32),
        decision_position=3,
        source_token_positions=np.arange(4, dtype=np.int32),
        source_step_ids=np.asarray([-1, -1, 0, 0], dtype=np.int16),
        selected_layers=np.asarray([2, 4], dtype=np.int16),
        wrong_token_id=9,
        correct_token_id=8,
        logits_topk_ids=np.asarray([8, 9, 7], dtype=np.int32),
        logits_topk_values=np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
        resid_pre_block=np.zeros((2, 4), dtype=np.float32),
        attn_head_output=np.zeros((2, 2, 2), dtype=np.float32),
        attn_branch_output=np.zeros((2, 4), dtype=np.float32),
        mlp_output=np.zeros((2, 4), dtype=np.float32),
        attn_edge_margin_proxy=np.zeros((2, 2, 4), dtype=np.float32),
        attn_edge_mass=mass,
        metadata={
            "schema": ONSET_TRACE_SCHEMA,
            "case_id": "gsm8k-7",
            "model_name": "llama",
            "model_revision_or_unknown": "unknown",
            "tokenizer_name": "llama",
            "tokenizer_revision_or_unknown": "unknown",
            "source_trace_fingerprint": "trace-sha",
            "pair_file_fingerprint": "pair-sha",
            "extraction_config": {"layers": [2, 4]},
            "attention_reconstruction_max_relative_error": 0.0,
        },
    )


def test_onset_trace_roundtrip_preserves_head_source_graph(tmp_path) -> None:
    artifact = _artifact()
    path = tmp_path / "case_7.onset_trace_v1.npz"

    artifact.save(path)
    loaded = OnsetTraceArtifact.load(path)

    assert loaded.decision_position == 3
    assert loaded.metadata["case_id"] == "gsm8k-7"
    np.testing.assert_array_equal(loaded.source_token_positions, np.arange(4))
    assert loaded.attn_edge_margin_proxy.shape == (2, 2, 4)


def test_onset_trace_rejects_tokens_after_decision_boundary() -> None:
    artifact = _artifact()
    leaked = OnsetTraceArtifact(
        **{**artifact.__dict__, "decision_position": 2}
    )

    with pytest.raises(ValueError, match="final observable token"):
        leaked.validate()

