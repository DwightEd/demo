from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from functional_divergence.hidden_state_geometry.component_contract import (
    ComponentStepArtifact,
)
from functional_divergence.hidden_state_geometry.component_features import (
    load_component_step_selection,
)
from functional_divergence.hidden_state_geometry.contracts import ChainSample
from functional_divergence.hidden_state_geometry.tasks import (
    TaskExample,
    build_post_step_task,
)


def _sample(
    tmp_path,
    *,
    chain_id: int = 7,
    gold: int = 1,
    dataset: str = "gsm8k",
    component_path=None,
) -> ChainSample:
    state_path = tmp_path / f"chain_{chain_id}.npy"
    np.save(state_path, np.zeros((6, 1, 2), dtype=np.float32))
    return ChainSample(
        chain_id=chain_id,
        manifest_row=chain_id,
        problem_group=str(chain_id),
        dataset=dataset,
        generator="Llama-3.1-8B",
        observer_model="Llama-3.1-8B",
        state_path=state_path,
        state_count=6,
        response_start=4,
        step_ranges=np.asarray([[4, 5], [6, 7], [8, 9]], dtype=np.int64),
        layer_ids=np.asarray([8]),
        output_steps=np.zeros((3, 2), dtype=np.float32),
        output_feature_names=("token_entropy", "token_nll"),
        first_error_step=gold,
        component_path=component_path,
    )


def _metadata(chain_id: int, dataset: str) -> dict[str, str]:
    return {
        "schema": "component_step_v1",
        "sample_id": f"chain-{chain_id}",
        "dataset": dataset,
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "model_revision": "rev-model",
        "tokenizer_name": "meta-llama/Llama-3.1-8B-Instruct",
        "tokenizer_revision": "rev-tokenizer",
        "extractor_commit": "abc123",
        "source_trace_sha256": "0" * 64,
        "generation_config_sha256": "1" * 64,
    }


def _step_labels(gold: int) -> np.ndarray:
    if gold == -1:
        return np.zeros(3, dtype=np.int8)
    labels = np.full(3, -1, dtype=np.int8)
    labels[:gold] = 0
    labels[gold] = 1
    return labels


def _artifact(
    *,
    chain_id: int = 7,
    dataset: str = "gsm8k",
    gold: int = 1,
    boundary_token_end=None,
) -> ComponentStepArtifact:
    boundaries = np.asarray(
        [4, 6, 8, 10] if boundary_token_end is None else boundary_token_end,
        dtype=np.int32,
    )
    source_step_id = np.asarray(
        [
            [-1, 0, -32768],
            [-1, 0, 1],
            [-1, 1, 2],
        ],
        dtype=np.int16,
    )
    source_mask = np.asarray(
        [
            [True, True, False],
            [True, True, True],
            [True, True, True],
        ],
        dtype=bool,
    )
    attn_msg = np.zeros((3, 1, 3, 2), dtype=np.float16)
    for step in range(3):
        for source in range(3):
            if source_mask[step, source]:
                attn_msg[step, :, source, :] = np.float16(0.25)
    return ComponentStepArtifact(
        input_ids=np.arange(10, dtype=np.int32),
        step_token_start=boundaries[:-1],
        step_token_end=boundaries[1:],
        boundary_token_end=boundaries,
        first_error_step=gold,
        step_label=_step_labels(gold),
        source_step_id=source_step_id,
        source_mask=source_mask,
        resid_boundary=np.zeros((4, 1, 2), dtype=np.float16),
        attn_msg_resid_by_source=attn_msg,
        attn_out_step=np.sum(attn_msg, axis=2).astype(np.float16),
        mlp_out_step=np.zeros((3, 1, 2), dtype=np.float16),
        residual_layers=np.asarray([8], dtype=np.int16),
        attention_layers=np.asarray([8], dtype=np.int16),
        mlp_layers=np.asarray([8], dtype=np.int16),
        metadata=_metadata(chain_id, dataset),
    )


def _write_artifact(sample: ChainSample, artifact: ComponentStepArtifact) -> None:
    assert sample.component_path is not None
    artifact.save(sample.component_path)


def test_load_component_step_selection_accepts_aligned_post_step_example(tmp_path):
    component_path = tmp_path / "chain_7.component_step_v1.npz"
    sample = _sample(tmp_path, component_path=component_path)
    _write_artifact(sample, _artifact())
    task = build_post_step_task((sample,))
    event_example = task.examples[1]

    selection = load_component_step_selection(event_example)

    assert selection.current_step == 1
    assert selection.artifact.first_error_step == 1


def test_component_loader_rejects_non_post_step_examples(tmp_path):
    component_path = tmp_path / "chain_7.component_step_v1.npz"
    sample = _sample(tmp_path, component_path=component_path)
    _write_artifact(sample, _artifact())
    example = TaskExample(sample, visible_steps=2, boundary_step=1)

    with pytest.raises(ValueError, match="post_step"):
        load_component_step_selection(example)


def test_component_loader_requires_component_path_and_existing_file(tmp_path):
    sample = _sample(tmp_path, component_path=None)
    example = build_post_step_task((sample,)).examples[0]

    with pytest.raises(ValueError, match="component_path"):
        load_component_step_selection(example)

    missing = replace(sample, component_path=tmp_path / "missing.component_step_v1.npz")
    missing_example = build_post_step_task((missing,)).examples[0]

    with pytest.raises(FileNotFoundError, match="missing.component_step_v1.npz"):
        load_component_step_selection(missing_example)


def test_component_loader_rejects_metadata_and_first_error_mismatches(tmp_path):
    dataset_path = tmp_path / "dataset_mismatch.component_step_v1.npz"
    dataset_sample = _sample(tmp_path, component_path=dataset_path)
    _write_artifact(dataset_sample, _artifact(dataset="math"))
    dataset_example = build_post_step_task((dataset_sample,)).examples[0]

    with pytest.raises(ValueError, match="metadata dataset"):
        load_component_step_selection(dataset_example)

    chain_path = tmp_path / "chain_mismatch.component_step_v1.npz"
    chain_sample = _sample(tmp_path, component_path=chain_path)
    _write_artifact(chain_sample, _artifact(chain_id=8))
    chain_example = build_post_step_task((chain_sample,)).examples[0]

    with pytest.raises(ValueError, match="metadata chain id"):
        load_component_step_selection(chain_example)

    gold_path = tmp_path / "gold_mismatch.component_step_v1.npz"
    gold_sample = _sample(tmp_path, component_path=gold_path)
    _write_artifact(gold_sample, _artifact(gold=2))
    gold_example = build_post_step_task((gold_sample,)).examples[0]

    with pytest.raises(ValueError, match="first_error_step"):
        load_component_step_selection(gold_example)


def test_component_loader_rejects_boundary_and_label_semantics_mismatches(tmp_path):
    component_path = tmp_path / "boundary_mismatch.component_step_v1.npz"
    sample = _sample(tmp_path, component_path=component_path)
    _write_artifact(sample, _artifact(boundary_token_end=[5, 6, 8, 10]))
    example = build_post_step_task((sample,)).examples[0]

    with pytest.raises(ValueError, match="boundary_token_end"):
        load_component_step_selection(example)

    aligned_path = tmp_path / "aligned.component_step_v1.npz"
    aligned = replace(sample, component_path=aligned_path)
    _write_artifact(aligned, _artifact())
    post_error_example = TaskExample(
        aligned,
        visible_steps=3,
        boundary_step=2,
        task_name="post_step",
    )

    with pytest.raises(ValueError, match="post-error"):
        load_component_step_selection(post_error_example)
