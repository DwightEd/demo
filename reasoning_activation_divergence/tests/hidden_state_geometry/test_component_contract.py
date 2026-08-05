from __future__ import annotations

import json

import numpy as np
import pytest
from functional_divergence.hidden_state_geometry.component_contract import (
    ComponentStepArtifact,
)


def _metadata() -> dict[str, str]:
    return {
        "schema": "component_step_v1",
        "sample_id": "chain-7",
        "dataset": "gsm8k",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "model_revision": "rev-model",
        "tokenizer_name": "meta-llama/Llama-3.1-8B-Instruct",
        "tokenizer_revision": "rev-tokenizer",
        "extractor_commit": "abc123",
        "source_trace_sha256": "0" * 64,
        "generation_config_sha256": "1" * 64,
    }


def _valid_artifact() -> ComponentStepArtifact:
    attn_msg = np.zeros((3, 2, 3, 4), dtype=np.float16)
    attn_msg[:, :, 0, :] = 0.25
    attn_msg[:, :, 1, :] = 0.5
    return ComponentStepArtifact(
        input_ids=np.asarray([101, 102, 103, 104, 105], dtype=np.int32),
        step_token_start=np.asarray([1, 2, 4], dtype=np.int32),
        step_token_end=np.asarray([2, 4, 5], dtype=np.int32),
        boundary_token_end=np.asarray([1, 2, 4, 5], dtype=np.int32),
        first_error_step=1,
        step_label=np.asarray([0, 1, -1], dtype=np.int8),
        source_step_id=np.asarray(
            [
                [-1, 0, -32768],
                [-1, 0, 1],
                [-1, 1, 2],
            ],
            dtype=np.int16,
        ),
        source_mask=np.asarray(
            [
                [True, True, False],
                [True, True, True],
                [True, True, True],
            ],
            dtype=bool,
        ),
        resid_boundary=np.full((4, 2, 4), 1.0, dtype=np.float16),
        attn_msg_resid_by_source=attn_msg,
        attn_out_step=np.sum(attn_msg, axis=2).astype(np.float16),
        mlp_out_step=np.full((3, 2, 4), 0.125, dtype=np.float16),
        residual_layers=np.asarray([8, 12], dtype=np.int16),
        attention_layers=np.asarray([8, 12], dtype=np.int16),
        mlp_layers=np.asarray([8, 12], dtype=np.int16),
        metadata=_metadata(),
    )


def test_round_trip_uses_compressed_npz_with_scalar_metadata(tmp_path):
    artifact = _valid_artifact()
    path = tmp_path / "chain_7.component_step_v1.npz"

    artifact.save(path)
    loaded = ComponentStepArtifact.load(path)

    assert path.is_file()
    assert np.array_equal(loaded.input_ids, artifact.input_ids)
    assert loaded.first_error_step == artifact.first_error_step
    assert np.array_equal(loaded.source_step_id, artifact.source_step_id)
    assert np.array_equal(
        loaded.attn_msg_resid_by_source, artifact.attn_msg_resid_by_source
    )
    assert loaded.metadata == artifact.metadata
    with np.load(path, allow_pickle=False) as archive:
        assert "metadata_json" in archive.files
        assert np.asarray(archive["metadata_json"]).shape == ()
        assert json.loads(str(archive["metadata_json"].item()))["schema"] == (
            "component_step_v1"
        )


def test_step_ranges_allow_unassigned_separator_tokens_between_steps():
    artifact = _valid_artifact()
    separated = ComponentStepArtifact(
        **{
            **artifact.__dict__,
            "step_token_start": np.asarray([1, 3, 4], dtype=np.int32),
        }
    )

    separated.validate()


def test_raw_attention_weights_cannot_substitute_for_required_message_array(tmp_path):
    artifact = _valid_artifact()
    path = tmp_path / "raw_attention_only.npz"
    payload = {
        "input_ids": artifact.input_ids,
        "step_token_start": artifact.step_token_start,
        "step_token_end": artifact.step_token_end,
        "boundary_token_end": artifact.boundary_token_end,
        "first_error_step": np.asarray(artifact.first_error_step, dtype=np.int32),
        "step_label": artifact.step_label,
        "source_step_id": artifact.source_step_id,
        "source_mask": artifact.source_mask,
        "resid_boundary": artifact.resid_boundary,
        "attn_weight_mass_by_source": np.ones((3, 2, 1, 3), dtype=np.float16),
        "attn_out_step": artifact.attn_out_step,
        "mlp_out_step": artifact.mlp_out_step,
        "residual_layers": artifact.residual_layers,
        "attention_layers": artifact.attention_layers,
        "mlp_layers": artifact.mlp_layers,
        "metadata_json": np.asarray(json.dumps(artifact.metadata, sort_keys=True)),
    }
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="attn_msg_resid_by_source"):
        ComponentStepArtifact.load(path)


def test_mismatched_attention_output_sum_is_rejected():
    artifact = _valid_artifact()
    bad = ComponentStepArtifact(
        **{
            **artifact.__dict__,
            "attn_out_step": artifact.attn_out_step.copy(),
        }
    )
    bad.attn_out_step[0, 0, 0] += np.float16(1.0)

    with pytest.raises(ValueError, match="attn_out_step"):
        bad.validate()


def test_future_source_step_is_rejected():
    artifact = _valid_artifact()
    bad_source_step_id = artifact.source_step_id.copy()
    bad_source_step_id[0, 1] = 1
    bad = ComponentStepArtifact(
        **{
            **artifact.__dict__,
            "source_step_id": bad_source_step_id,
        }
    )

    with pytest.raises(ValueError, match="future"):
        bad.validate()


def test_zero_label_after_first_error_is_rejected():
    artifact = _valid_artifact()
    bad = ComponentStepArtifact(
        **{
            **artifact.__dict__,
            "step_label": np.asarray([0, 1, 0], dtype=np.int8),
        }
    )

    with pytest.raises(ValueError, match="post-error"):
        bad.validate()


def test_first_error_step_must_match_the_step_labels():
    artifact = _valid_artifact()
    bad = ComponentStepArtifact(
        **{
            **artifact.__dict__,
            "first_error_step": -1,
        }
    )

    with pytest.raises(ValueError, match="first_error_step"):
        bad.validate()


def test_every_step_needs_unique_source_blocks():
    artifact = _valid_artifact()
    bad_source_step_id = artifact.source_step_id.copy()
    bad_source_step_id[1, 2] = 0
    bad = ComponentStepArtifact(
        **{
            **artifact.__dict__,
            "source_step_id": bad_source_step_id,
        }
    )

    with pytest.raises(ValueError, match="unique"):
        bad.validate()


def test_padded_attention_messages_must_be_zero():
    artifact = _valid_artifact()
    bad_messages = artifact.attn_msg_resid_by_source.copy()
    bad_messages[0, :, 2, :] = 0.25
    bad = ComponentStepArtifact(
        **{
            **artifact.__dict__,
            "attn_msg_resid_by_source": bad_messages,
        }
    )

    with pytest.raises(ValueError, match="padded"):
        bad.validate()


def test_shape_mismatch_is_rejected():
    artifact = _valid_artifact()
    bad = ComponentStepArtifact(
        **{
            **artifact.__dict__,
            "mlp_out_step": np.full((3, 2, 5), 0.125, dtype=np.float16),
        }
    )

    with pytest.raises(ValueError, match="mlp_out_step"):
        bad.validate()
