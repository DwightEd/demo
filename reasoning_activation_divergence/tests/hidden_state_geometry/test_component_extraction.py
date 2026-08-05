from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from functional_divergence.hidden_state_geometry import component_extraction
from functional_divergence.hidden_state_geometry.component_contract import (
    SOURCE_STEP_PADDING,
    ComponentStepArtifact,
)
from functional_divergence.hidden_state_geometry.component_extract_cli import (
    build_parser as build_component_parser,
)
from functional_divergence.hidden_state_geometry.component_extraction import (
    ComponentExtractionConfig,
    ComponentTraceExtractor,
    _build_component_layout,
    load_trace_replay_record,
    validate_replay_fidelity,
)
from functional_divergence.hidden_state_geometry.component_replay import (
    _expand_grouped_query_values,
    _residual_writes_from_source_contexts,
    _source_contexts_for_boundary_queries,
    _source_spans_for_step,
    _validate_model_for_replay,
)
from functional_divergence.hidden_state_geometry.contracts import (
    ChainSample,
    TraceSource,
)


def _sample(tmp_path, *, chain_id: int = 11, component_path=None) -> ChainSample:
    state_path = tmp_path / f"chain_{chain_id}.npy"
    np.save(state_path, np.zeros((3, 1, 4), dtype=np.float32))
    return ChainSample(
        chain_id=chain_id,
        manifest_row=1,
        problem_group="problem-11",
        dataset="gsm8k",
        generator="Meta-Llama-3.1-8B-Instruct",
        observer_model="Meta-Llama-3.1-8B-Instruct",
        state_path=state_path,
        state_count=3,
        response_start=2,
        step_ranges=np.asarray([[2, 3], [4, 4]], dtype=np.int64),
        layer_ids=np.asarray([1], dtype=np.int64),
        output_steps=np.zeros((2, 2), dtype=np.float32),
        output_feature_names=("token_entropy", "token_nll"),
        first_error_step=1,
        component_path=component_path,
    )


def _trace(
    tmp_path,
    *,
    attention_mask=None,
    gold=None,
    dataset=None,
    padded_ranges=False,
    model_revision="main",
    tokenizer_revision="main",
) -> TraceSource:
    selected = tmp_path / "gsm8k" / "selected"
    selected.mkdir(parents=True)
    trace = selected / "trace.npz"
    if padded_ranges:
        ranges = np.asarray(
            [
                [[1, 1], [-1, -1], [-1, -1]],
                [[2, 3], [4, 4], [-1, -1]],
            ],
            dtype=np.int64,
        )
    else:
        ranges = np.empty(2, dtype=object)
        ranges[0] = np.asarray([[1, 1]], dtype=np.int64)
        ranges[1] = np.asarray([[2, 3], [4, 4]], dtype=np.int64)
    np.savez(
        trace,
        chain_idx=np.asarray([10, 11], dtype=np.int64),
        full_input_ids=np.asarray(
            [[901, 902, 0, 0, 0, 0], [101, 102, 201, 202, 203, 0]],
            dtype=np.int64,
        ),
        full_attention_mask=np.asarray(
            (
                [[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0]]
                if attention_mask is None
                else attention_mask
            ),
            dtype=np.int64,
        ),
        prompt_token_counts=np.asarray([1, 2], dtype=np.int64),
        n_steps=np.asarray([1, 2], dtype=np.int64),
        step_token_ranges=ranges,
        gold_error_step=np.asarray([-1, 1] if gold is None else gold, dtype=np.int64),
        dataset=np.asarray(
            ["gsm8k", "gsm8k"] if dataset is None else dataset, dtype=object
        ),
        source_model=np.asarray(["meta-llama/Llama-3.1-8B-Instruct"] * 2, dtype=object),
        source_model_revision=np.asarray([model_revision] * 2, dtype=object),
        source_tokenizer=np.asarray(
            ["meta-llama/Llama-3.1-8B-Instruct"] * 2, dtype=object
        ),
        source_tokenizer_revision=np.asarray([tokenizer_revision] * 2, dtype=object),
    )
    return TraceSource(
        "gsm8k",
        selected / "trace.raw_residual_stream.npz",
        "observer_teacher_forcing_replay",
        exact_trace=trace,
        component_dir=selected / "component_step_v1",
    )


def _metadata(
    sample: ChainSample,
    config: ComponentExtractionConfig,
    source_sha: str,
) -> dict[str, str]:
    return {
        "schema": "component_step_v1",
        "sample_id": f"chain-{sample.chain_id}",
        "chain_id": f"chain-{sample.chain_id}",
        "dataset": sample.dataset,
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "tokenizer_name": config.tokenizer_name,
        "tokenizer_revision": config.tokenizer_revision,
        "extractor_commit": config.extractor_commit,
        "source_trace_sha256": source_sha,
        "generation_config_sha256": config.replay_config_sha256(),
    }


def _artifact(
    sample: ChainSample,
    config: ComponentExtractionConfig,
    source_sha: str,
) -> ComponentStepArtifact:
    layout = _build_component_layout(sample)
    attn_msg = np.zeros((2, 1, 3, 4), dtype=np.float16)
    attn_msg[:, :, 0, :] = np.float16(0.25)
    attn_msg[:, :, 1, :] = np.float16(0.5)
    return ComponentStepArtifact(
        input_ids=np.asarray([101, 102, 201, 202, 203], dtype=np.int32),
        step_token_start=layout.step_token_start,
        step_token_end=layout.step_token_end,
        boundary_token_end=layout.boundary_token_end,
        first_error_step=sample.first_error_step,
        step_label=layout.step_label,
        source_step_id=layout.source_step_id,
        source_mask=layout.source_mask,
        resid_boundary=np.zeros((3, 1, 4), dtype=np.float16),
        attn_msg_resid_by_source=attn_msg,
        attn_out_step=np.sum(attn_msg, axis=2).astype(np.float16),
        mlp_out_step=np.zeros((2, 1, 4), dtype=np.float16),
        residual_layers=np.asarray(config.layers, dtype=np.int16),
        attention_layers=np.asarray(config.layers, dtype=np.int16),
        mlp_layers=np.asarray(config.layers, dtype=np.int16),
        metadata=_metadata(sample, config, source_sha),
    )


def test_source_spans_cover_prompt_and_completed_steps_exactly() -> None:
    spans = _source_spans_for_step(
        prompt_end=3,
        boundary_token_end=np.asarray([3, 5, 7, 10], dtype=np.int64),
        current_step=2,
    )

    assert [(span.source_step_id, span.start, span.end) for span in spans] == [
        (-1, 0, 3),
        (0, 3, 5),
        (1, 5, 7),
        (2, 7, 10),
    ]
    covered = [token for span in spans for token in range(span.start, span.end)]
    assert covered == list(range(10))


def test_component_layout_builds_boundaries_labels_and_source_grid(tmp_path) -> None:
    sample = _sample(tmp_path)

    layout = _build_component_layout(sample)

    assert layout.step_token_start.tolist() == [2, 4]
    assert layout.step_token_end.tolist() == [4, 5]
    assert layout.boundary_token_end.tolist() == [2, 4, 5]
    assert layout.step_label.tolist() == [0, 1]
    assert layout.source_step_id.tolist() == [
        [-1, 0, SOURCE_STEP_PADDING],
        [-1, 0, 1],
    ]
    assert layout.source_mask.tolist() == [[True, True, False], [True, True, True]]


def test_final_normalized_hidden_depth_is_not_treated_as_a_block_output() -> None:
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[object(), object()]),
        config=SimpleNamespace(_attn_implementation="eager"),
    )

    with pytest.raises(ValueError, match="final normalized state"):
        _validate_model_for_replay(model, (2,))


def test_component_cli_makes_replay_fidelity_an_explicit_audit() -> None:
    parser = build_component_parser()
    required = [
        "--data-root",
        "/data",
        "--model-dir",
        "/model",
        "--component-layers",
        "8,12",
    ]

    assert parser.parse_args(required).verify_replay_fidelity is False
    assert (
        parser.parse_args([*required, "--verify-replay-fidelity"])
        .verify_replay_fidelity
        is True
    )


def test_load_trace_replay_record_joins_by_chain_and_rejects_interior_padding(
    tmp_path,
) -> None:
    source = _trace(tmp_path)
    sample = _sample(tmp_path)
    config = ComponentExtractionConfig(
        layers=(1,),
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="auto",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_revision="auto",
        extractor_commit="abc123",
    )

    record = load_trace_replay_record(source.exact_trace, sample, config)

    assert record.input_ids.tolist() == [101, 102, 201, 202, 203]
    assert record.attention_mask.tolist() == [1, 1, 1, 1, 1]
    assert record.trace_path == source.exact_trace
    assert record.provenance["model_revision"] == "main"
    assert record.provenance["tokenizer_revision"] == "main"

    bad_source = _trace(
        tmp_path / "bad",
        attention_mask=[[1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 0]],
    )
    with pytest.raises(ValueError, match="interior padding"):
        load_trace_replay_record(bad_source.exact_trace, sample)


def test_load_trace_replay_record_trims_step_range_padding_by_n_steps(tmp_path) -> None:
    source = _trace(tmp_path, padded_ranges=True)
    sample = _sample(tmp_path)

    record = load_trace_replay_record(source.exact_trace, sample)

    np.testing.assert_array_equal(record.step_token_ranges, sample.step_ranges)


def test_auto_revision_records_missing_trace_revision_without_weakening_explicit_match(
    tmp_path,
) -> None:
    source = _trace(tmp_path, model_revision="", tokenizer_revision="")
    sample = _sample(tmp_path)
    auto = ComponentExtractionConfig(
        layers=(1,),
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="auto",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_revision="auto",
        extractor_commit="abc123",
    )

    record = load_trace_replay_record(source.exact_trace, sample, auto)

    assert record.provenance["model_revision"] == "unavailable_in_source_trace"
    assert record.provenance["tokenizer_revision"] == "unavailable_in_source_trace"

    explicit = ComponentExtractionConfig(
        layers=(1,),
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="main",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_revision="main",
        extractor_commit="abc123",
    )
    with pytest.raises(ValueError, match="provenance mismatch for model_revision"):
        load_trace_replay_record(source.exact_trace, sample, explicit)


def test_load_trace_replay_record_rejects_alignment_mismatches(tmp_path) -> None:
    sample = _sample(tmp_path)
    bad_gold = _trace(tmp_path / "gold", gold=[-1, -1])
    with pytest.raises(ValueError, match="first-error"):
        load_trace_replay_record(bad_gold.exact_trace, sample)

    bad_dataset = _trace(tmp_path / "dataset", dataset=["gsm8k", "math"])
    with pytest.raises(ValueError, match="dataset"):
        load_trace_replay_record(bad_dataset.exact_trace, sample)


def test_gqa_values_expand_and_residual_writes_reconstruct_attention_output() -> None:
    torch = pytest.importorskip("torch")
    values = torch.tensor([[[[1.0], [10.0]], [[2.0], [20.0]], [[3.0], [30.0]]]])
    expanded = _expand_grouped_query_values(values, num_heads=4)
    assert expanded.shape == (1, 3, 4, 1)
    assert expanded[0, :, 0, 0].tolist() == [1.0, 2.0, 3.0]
    assert expanded[0, :, 1, 0].tolist() == [1.0, 2.0, 3.0]
    assert expanded[0, :, 2, 0].tolist() == [10.0, 20.0, 30.0]
    assert expanded[0, :, 3, 0].tolist() == [10.0, 20.0, 30.0]

    attention = torch.zeros((1, 4, 3, 3), dtype=torch.float32)
    attention[0, 0, 2] = torch.tensor([0.2, 0.3, 0.5])
    attention[0, 1, 2] = torch.tensor([0.5, 0.2, 0.3])
    attention[0, 2, 2] = torch.tensor([0.1, 0.7, 0.2])
    attention[0, 3, 2] = torch.tensor([0.4, 0.4, 0.2])
    contexts = _source_contexts_for_boundary_queries(
        attention,
        expanded,
        target_indices=torch.tensor([2]),
        source_spans=[[(-1, 0, 1), (0, 1, 3)]],
    )
    weight = torch.eye(4, dtype=torch.float32)
    writes = _residual_writes_from_source_contexts(contexts, weight, bias=None)

    actual_by_head = torch.sum(
        attention[0, :, 2, :, None] * expanded[0].transpose(0, 1),
        dim=1,
    )
    assert torch.allclose(writes.sum(dim=1)[0], actual_by_head.reshape(-1))


def test_aligned_existing_component_artifact_from_an_older_commit_is_skipped(
    tmp_path,
) -> None:
    source = _trace(tmp_path)
    sample = _sample(
        tmp_path,
        component_path=source.component_dir / "chain_11.component_step_v1.npz",
    )
    old_config = ComponentExtractionConfig(
        layers=(1,),
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="main",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_revision="main",
        extractor_commit="abc123",
    )
    config = ComponentExtractionConfig(
        **{**old_config.__dict__, "extractor_commit": "new456"}
    )
    extractor = ComponentTraceExtractor(config)
    source_sha = extractor.source_trace_sha256(source.exact_trace)
    _artifact(sample, old_config, source_sha).save(sample.component_path)

    result = extractor.extract(model=object(), samples=(sample,), sources=(source,))

    assert result.written == ()
    assert result.skipped == (sample.component_path,)


def test_component_extraction_does_not_require_hidden_shard_fidelity_by_default(
    tmp_path, monkeypatch
) -> None:
    source = _trace(tmp_path)
    sample = _sample(
        tmp_path,
        component_path=source.component_dir / "chain_11.component_step_v1.npz",
    )
    config = ComponentExtractionConfig(
        layers=(1, 2),
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="main",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_revision="main",
        extractor_commit="abc123",
        verify_replay_fidelity=False,
    )
    monkeypatch.setattr(
        component_extraction,
        "_replay_components",
        lambda *args, **kwargs: (
            np.zeros((3, 2, 4), dtype=np.float32),
            np.zeros((2, 2, 3, 4), dtype=np.float32),
            np.zeros((2, 2, 4), dtype=np.float32),
            np.zeros((2, 2, 4), dtype=np.float32),
            0.0,
        ),
    )

    result = ComponentTraceExtractor(config).extract(
        model=object(), samples=(sample,), sources=(source,)
    )

    assert result.written == (sample.component_path,)
    artifact = ComponentStepArtifact.load(sample.component_path)
    assert artifact.residual_layers.tolist() == [1, 2]
    assert artifact.metadata["replay_fidelity_checked"] is False


def test_replay_fidelity_failure_is_hard_error(tmp_path) -> None:
    sample = _sample(tmp_path)
    np.save(sample.state_path, np.zeros((3, 1, 4), dtype=np.float32))
    replayed = np.ones((2, 1, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="replay fidelity"):
        validate_replay_fidelity(
            sample,
            layers=np.asarray([1], dtype=np.int64),
            replayed_step_states=replayed,
            max_relative_error=1e-6,
        )
