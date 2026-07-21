from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hypergraph.attention import (
    EDGE_ATTR_NAMES,
    EXTENDED_EDGE_ATTR_NAMES,
    AttentionHypergraphConfig,
    build_attention_hypergraph,
    validate_attention_hypergraph,
)
from hypergraph.attention.extract import (
    _write_extraction_manifest,
    build_parser as build_extraction_parser,
    canonical_record,
    char_spans_to_token_ranges,
    classify_model_commit_source,
    estimate_attention_gib,
    extraction_code_sha256,
    extract_trace,
    fingerprint_extraction_config,
    generator_matches_model,
    parse_max_memory,
    prepare_empty_output_dir,
    require_exact_replay_inputs,
    resolve_loaded_commit,
    response_and_char_spans,
    select_dataset_shard,
    validate_sequence_length,
    verify_chunked_equivalence,
)


def _write_audited_observer_cohort(tmp_path, generators, *, model_names=None):
    """Write a complete tiny extraction cohort with production-format provenance."""

    from hypergraph.attention.data import TRACE_CONTRACT

    attention, token_ids, response_idx = _trace()
    input_sha = "a" * 64
    scope = {
        "input_sha256": input_sha,
        "input_num_rows": len(generators),
        "pre_shard_num_rows": len(generators),
        "selected_num_rows": len(generators),
        "requested_limit": None,
        "num_shards": 1,
        "shard_index": 0,
        "skip_invalid": False,
        "max_seq_len": 2048,
        "max_attention_gib": 24.0,
        "allow_large_attention": False,
    }
    scope_json = json.dumps(scope, sort_keys=True, separators=(",", ":"))
    scope_fingerprint = hashlib.sha256(scope_json.encode("utf-8")).hexdigest()
    output = tmp_path / "observer-traces"
    output.mkdir()
    if model_names is None:
        model_names = ["/models/Meta-Llama-3.1-8B-Instruct"] * len(generators)
    for index, (generator, model_name) in enumerate(zip(generators, model_names)):
        method = {
            "trace_contract": TRACE_CONTRACT,
            "model_name": model_name,
            "model_commit_hash": None,
            "model_commit_source": "unavailable",
            "tokenizer_name": model_name,
            "prompt_style": "plain",
            "replay_mode": "observer",
            "dtype": "bfloat16",
            "attention_storage_dtype": "float32",
            "activation_layer": None,
            "allow_unverified_generator_weights": False,
            "query_chunk_size": 0,
            "attention_layers": [14],
            "attention_heads": [0],
            "num_model_layers": 32,
            "num_model_heads": 32,
        }
        method_json = json.dumps(method, sort_keys=True, separators=(",", ":"))
        method_fingerprint = hashlib.sha256(method_json.encode("utf-8")).hexdigest()
        np.savez_compressed(
            output / f"trace-{index:02d}.npz",
            attention=attention,
            token_ids=token_ids,
            response_idx=np.asarray(response_idx),
            response_y=np.asarray(float(index % 2), np.float32),
            sample_id=np.asarray(f"trace-{index}"),
            problem_id=np.asarray(f"problem-{index}"),
            attention_layers=np.asarray([14], np.int64),
            attention_heads=np.asarray([0], np.int64),
            num_model_layers=np.asarray(32, np.int64),
            num_model_heads=np.asarray(32, np.int64),
            activation_layer=np.asarray(-1, np.int64),
            model_name=np.asarray(model_name),
            model_commit_hash=np.asarray(""),
            model_commit_source=np.asarray("unavailable"),
            tokenizer_name=np.asarray(model_name),
            extraction_dtype=np.asarray("bfloat16"),
            attention_storage_dtype=np.asarray("float32"),
            extraction_fingerprint=np.asarray(method_fingerprint),
            extraction_method_json=np.asarray(method_json),
            extraction_scope_fingerprint=np.asarray(scope_fingerprint),
            extraction_scope_json=np.asarray(scope_json),
            source_input_sha256=np.asarray(input_sha),
            source_row_index=np.asarray(index, np.int64),
            extraction_forward_mode=np.asarray("full"),
            chunk_equivalence_status=np.asarray("not_applicable"),
            chunk_equivalence_json=np.asarray('{"status":"not_applicable"}'),
            prompt_style=np.asarray("plain"),
            replay_mode=np.asarray("observer"),
            step_alignment_policy=np.asarray("positive_character_overlap_v1"),
            replay_fidelity=np.asarray("observer_counterfactual"),
            unverified_generator_weights_explicitly_allowed=np.asarray(False),
            prompt_provenance=np.asarray("frozen_plain_observer"),
            generator_model=np.asarray(generator),
            generator_model_commit=np.asarray(""),
            prompt_add_special_tokens=np.asarray(True),
            rendered_prompt_sha256=np.asarray(f"prompt-{index}"),
            response_text_sha256=np.asarray(f"response-{index}"),
            trace_contract=np.asarray(TRACE_CONTRACT),
        )
    return output


def _trace():
    attention = np.zeros((1, 1, 5, 5), np.float32)
    np.fill_diagonal(attention[0, 0], [0.9, 0.8, 0.7, 0.5, 0.4])
    attention[0, 0, 3, :3] = [0.10, 0.04, 0.20]
    attention[0, 0, 4, :4] = [0.01, 0.06, 0.07, 0.08]
    return attention, np.arange(5, dtype=np.int64), 3


def _edge_members(graph):
    nodes, edge_ids = graph.he_index
    return [nodes[edge_ids == edge].tolist() for edge in range(graph.num_hyperedges)]


def test_faithful_attention_rows_are_the_hypergraph_topology():
    attention, token_ids, response_idx = _trace()
    graph = build_attention_hypergraph(attention, token_ids, response_idx)

    assert graph.num_nodes == 5
    assert graph.construction_config.target_alignment == "same_index_post_emission"
    assert graph.num_hyperedges == 2
    assert _edge_members(graph) == [[0, 1, 2, 3], [1, 2, 3, 4]]
    np.testing.assert_array_equal(graph.he_receiver, [3, 4])
    np.testing.assert_array_equal(graph.he_count, [4, 4])
    np.testing.assert_allclose(graph.x[:, 0], [0.9, 0.8, 0.7, 0.5, 0.4])
    np.testing.assert_array_equal(graph.he_mark, [[1, 0], [1, 0]])
    np.testing.assert_allclose(graph.he_weight, 1.0)
    assert graph.he_attr.shape == (2, 3)
    assert graph.edge_attr_names == EDGE_ATTR_NAMES
    np.testing.assert_allclose(graph.he_attr[:, 2], 0.0)
    assert graph.propagation_mode == "symmetric"
    validate_attention_hypergraph(graph)


def test_top_k_and_relation_scope_are_explicit_ablation_switches():
    attention, token_ids, response_idx = _trace()
    top_one = build_attention_hypergraph(
        attention,
        token_ids,
        response_idx,
        config=AttentionHypergraphConfig(top_k=1),
    )
    assert _edge_members(top_one) == [[2, 3], [3, 4]]
    # Validation is self-describing and must not silently fall back to default
    # threshold/top-k settings.
    validate_attention_hypergraph(top_one)

    response_only = build_attention_hypergraph(
        attention,
        token_ids,
        response_idx,
        config=AttentionHypergraphConfig(source_scope="response_only"),
    )
    assert _edge_members(response_only) == [[3, 4]]
    np.testing.assert_array_equal(response_only.he_mark, [[0, 1]])


def test_length_robust_sparsifiers_are_explicit_and_can_rescue_subthreshold_rows():
    with pytest.raises(ValueError, match="positive integer"):
        AttentionHypergraphConfig(source_selection="top_k_only", top_k=1.5)
    attention, token_ids, response_idx = _trace()
    attention[0, 0, 3, :3] = [0.003, 0.004, 0.005]
    attention[0, 0, 4, :4] = [0.002, 0.003, 0.004, 0.005]
    faithful = build_attention_hypergraph(attention, token_ids, response_idx)
    assert faithful.num_hyperedges == 0

    with pytest.raises(ValueError, match="requires top_k"):
        AttentionHypergraphConfig(source_selection="threshold_fallback_topk")
    original_fallback = build_attention_hypergraph(
        attention,
        token_ids,
        response_idx,
        config=AttentionHypergraphConfig(
            threshold=0.05,
            source_selection="threshold_fallback_topk",
            top_k=2,
            min_sources=2,
        ),
    )
    assert _edge_members(original_fallback) == [[1, 2, 3], [2, 3, 4]]

    top_k = build_attention_hypergraph(
        attention,
        token_ids,
        response_idx,
        config=AttentionHypergraphConfig(source_selection="top_k_only", top_k=1),
    )
    assert _edge_members(top_k) == [[2, 3], [3, 4]]

    all_zero = attention.copy()
    all_zero[0, 0, 3, :3] = 0.0
    all_zero[0, 0, 4, :4] = 0.0
    zero_top_k = build_attention_hypergraph(
        all_zero,
        token_ids,
        response_idx,
        config=AttentionHypergraphConfig(source_selection="top_k_only", top_k=1),
    )
    assert _edge_members(zero_top_k) == [[0, 3], [0, 4]]

    attention, token_ids, response_idx = _trace()
    mass = build_attention_hypergraph(
        attention,
        token_ids,
        response_idx,
        config=AttentionHypergraphConfig(
            source_selection="cumulative_mass", cumulative_mass=0.8
        ),
    )
    assert _edge_members(mass) == [[0, 2, 3], [1, 2, 3, 4]]


def test_length_attributes_are_opt_in_not_part_of_the_faithful_baseline():
    attention, token_ids, response_idx = _trace()
    graph = build_attention_hypergraph(
        attention,
        token_ids,
        response_idx,
        config=AttentionHypergraphConfig(edge_attr_mode="extended"),
    )
    assert graph.edge_attr_names == EXTENDED_EDGE_ATTR_NAMES
    assert graph.he_attr.shape == (2, 6)
    np.testing.assert_allclose(graph.he_attr[:, 4], [4 / 5, 4 / 5])
    validate_attention_hypergraph(graph)


def test_hidden_content_and_attention_diagonal_are_independent_node_feature_controls():
    attention, token_ids, response_idx = _trace()
    activation = np.arange(15, dtype=np.float32).reshape(5, 3)
    attention_only = build_attention_hypergraph(
        attention,
        token_ids,
        response_idx,
        config=AttentionHypergraphConfig(node_feature_mode="attention_diagonal"),
    )
    hidden_only = build_attention_hypergraph(
        attention,
        token_ids,
        response_idx,
        activation=activation,
        config=AttentionHypergraphConfig(node_feature_mode="activation_only"),
    )
    np.testing.assert_array_equal(hidden_only.x, activation)

    combined = build_attention_hypergraph(
        attention,
        token_ids,
        response_idx,
        activation=activation,
        config=AttentionHypergraphConfig(node_feature_mode="diagonal_plus_activation"),
    )
    assert combined.x.shape == (5, 4)
    np.testing.assert_array_equal(combined.x[:, 1:], activation)
    assert hidden_only.he_attr.shape[1] == 3
    assert combined.he_attr.shape[1] == 3
    np.testing.assert_array_equal(hidden_only.he_index, attention_only.he_index)
    np.testing.assert_array_equal(combined.he_index, attention_only.he_index)
    np.testing.assert_allclose(hidden_only.he_attr, attention_only.he_attr)
    np.testing.assert_allclose(combined.he_attr, attention_only.he_attr)
    with pytest.raises(ValueError, match="requires aligned activation"):
        build_attention_hypergraph(
            attention,
            token_ids,
            response_idx,
            config=AttentionHypergraphConfig(node_feature_mode="activation_only"),
        )

def test_extracted_layer_head_ids_stay_global_but_faithful_edge_attr_is_local():
    attention, token_ids, response_idx = _trace()
    graph = build_attention_hypergraph(
        attention,
        token_ids,
        response_idx,
        attention_layer_ids=np.asarray([5]),
        attention_head_ids=np.asarray([3]),
        num_model_layers=8,
        num_model_heads=4,
        config=AttentionHypergraphConfig(selected_layers=(5,), selected_heads=(3,)),
    )
    np.testing.assert_array_equal(graph.he_layer, [5, 5])
    np.testing.assert_array_equal(graph.he_head, [3, 3])
    # The original computes this field from the flattened stored attention
    # tensor. Global model ids remain available separately in he_layer/he_head.
    np.testing.assert_allclose(graph.he_attr[:, 2], 0.0)
    validate_attention_hypergraph(graph)


def test_attention_weights_remain_separate_from_membership():
    attention, token_ids, response_idx = _trace()
    graph = build_attention_hypergraph(
        attention,
        token_ids,
        response_idx,
        config=AttentionHypergraphConfig(incidence_weight_mode="attention"),
    )
    np.testing.assert_allclose(graph.he_weight, graph.he_attention)
    # Receiver self-attention is retained because the faithful local method
    # adds the centre before computing edge statistics.
    np.testing.assert_allclose(graph.he_attention[:4], [0.10, 0.04, 0.20, 0.50])


def test_first_error_step_is_not_expanded_into_token_labels():
    attention, token_ids, response_idx = _trace()
    graph = build_attention_hypergraph(
        attention,
        token_ids,
        response_idx,
        step_ranges=np.asarray([[3, 4], [4, 5]], np.int64),
        gold_step=1,
        step_loss_mask=np.asarray([True, True]),
        response_y=1,
    )
    assert graph.token_y is None
    assert graph.gold_step == 1
    np.testing.assert_array_equal(graph.step_ranges, [[3, 4], [4, 5]])


def test_exact_token_labels_cannot_mark_prompt_tokens_positive():
    attention, token_ids, response_idx = _trace()
    with pytest.raises(ValueError, match="Prompt tokens|prompt tokens"):
        build_attention_hypergraph(
            attention,
            token_ids,
            response_idx,
            token_y=np.asarray([1, 0, 0, 0, 0], np.float32),
        )


def test_step_ranges_must_use_the_full_prompt_response_axis():
    attention, token_ids, response_idx = _trace()
    with pytest.raises(ValueError, match="within the response"):
        build_attention_hypergraph(
            attention,
            token_ids,
            response_idx,
            step_ranges=np.asarray([[0, 1], [3, 5]], np.int64),
            gold_step=1,
        )

    with pytest.raises(ValueError, match="integer offsets"):
        build_attention_hypergraph(
            attention,
            token_ids,
            response_idx,
            step_ranges=np.asarray([[3.2, 4.0]]),
            gold_step=0,
        )


def test_step_response_labels_and_masks_must_be_consistent():
    attention, token_ids, response_idx = _trace()
    with pytest.raises(ValueError, match="response_y conflicts"):
        build_attention_hypergraph(
            attention,
            token_ids,
            response_idx,
            step_ranges=np.asarray([[3, 4], [4, 5]]),
            gold_step=1,
            response_y=0,
        )
    with pytest.raises(ValueError, match="cannot hide"):
        build_attention_hypergraph(
            attention,
            token_ids,
            response_idx,
            step_ranges=np.asarray([[3, 4], [4, 5]]),
            gold_step=1,
            step_loss_mask=np.asarray([True, False]),
        )


def test_response_must_be_non_empty():
    attention, token_ids, _ = _trace()
    with pytest.raises(ValueError, match="response_idx"):
        build_attention_hypergraph(attention, token_ids, len(token_ids))


def test_exact_axis_extractor_keeps_bos_prompt_response_and_steps_aligned():
    response, char_spans = response_and_char_spans(["a", "b"], prompt_chars=2)
    assert response == "a\n\nb"
    np.testing.assert_array_equal(char_spans, [[2, 3], [5, 6]])
    offsets = np.asarray(
        [
            [0, 0],  # BOS / special token
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],  # separator newline, intentionally outside either step
            [4, 5],  # second separator newline
            [5, 6],
        ],
        np.int64,
    )
    response_idx, ranges = char_spans_to_token_ranges(offsets, 2, char_spans)
    assert response_idx == 3
    np.testing.assert_array_equal(ranges, [[3, 4], [6, 7]])


def test_exact_axis_extractor_rejects_tokens_crossing_response_boundary():
    offsets = np.asarray([[0, 0], [0, 1], [1, 3], [3, 4]], np.int64)
    with pytest.raises(ValueError, match="prompt/response"):
        char_spans_to_token_ranges(offsets, 2, np.asarray([[2, 4]], np.int64))


def test_step_alignment_accepts_tokens_that_absorb_only_the_separator():
    char_spans = np.asarray([[2, 3], [5, 6]], np.int64)
    offsets = np.asarray(
        [
            [0, 0],
            [0, 1],
            [1, 2],
            [2, 4],  # first-step text plus the first separator newline
            [4, 5],  # separator-only token
            [5, 6],
        ],
        np.int64,
    )

    response_idx, ranges = char_spans_to_token_ranges(offsets, 2, char_spans)

    assert response_idx == 3
    np.testing.assert_array_equal(ranges, [[3, 4], [5, 6]])


def test_step_alignment_assigns_leading_separator_token_by_text_overlap():
    char_spans = np.asarray([[2, 3], [5, 6]], np.int64)
    offsets = np.asarray(
        [
            [0, 0],
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 6],  # separator plus second-step text
        ],
        np.int64,
    )

    _, ranges = char_spans_to_token_ranges(offsets, 2, char_spans)

    np.testing.assert_array_equal(ranges, [[3, 4], [4, 5]])


def test_step_alignment_rejects_one_token_covering_two_step_texts():
    offsets = np.asarray([[0, 0], [0, 1], [1, 2], [2, 6]], np.int64)
    with pytest.raises(ValueError, match="multiple reasoning steps"):
        char_spans_to_token_ranges(
            offsets,
            2,
            np.asarray([[2, 3], [5, 6]], np.int64),
        )


def test_processbench_record_preserves_step_granularity_and_memory_estimate():
    record = canonical_record(
        {"id": "q1", "problem": "1+1?", "steps": ["1+1=2", "done"], "label": 0},
        0,
    )
    assert record["gold_step"] == 0
    assert record["response_y"] == 1.0
    assert record["steps"] == ["1+1=2", "done"]
    assert estimate_attention_gib(2, 4, 8, 2) == pytest.approx(1024 / 1024**3)

    with pytest.raises(ValueError, match="refusing to invent -1"):
        canonical_record({"problem": "1+1?", "steps": ["1+1=2"]}, 1)
    with pytest.raises(ValueError, match="ambiguous gold step aliases"):
        canonical_record(
            {
                "problem": "1+1?",
                "steps": ["1+1=2"],
                "label": -1,
                "gold_step": 0,
            },
            2,
        )
    with pytest.raises(ValueError, match="conflicts with gold_step"):
        canonical_record(
            {
                "problem": "1+1?",
                "steps": ["1+1=2"],
                "label": -1,
                "response_y": 1,
            },
            3,
        )


def test_accelerated_extraction_configuration_is_strict_and_shards_are_disjoint():
    code_hashes = extraction_code_sha256()
    assert "hypergraph/attention/extract.py" in code_hashes
    assert "hypergraph/attention/selective_capture.py" in code_hashes
    assert "hypergraph/attention/trace_contract.py" in code_hashes
    assert "utils/step_boundaries.py" in code_hashes
    assert "hypergraph/attention/data.py" not in code_hashes
    assert all(len(digest) == 64 for digest in code_hashes.values())
    assert parse_max_memory("0=22GiB,1=22GiB,cpu=64GiB") == {
        0: "22GiB",
        1: "22GiB",
        "cpu": "64GiB",
    }
    with pytest.raises(ValueError, match="device=value"):
        parse_max_memory("22GiB")
    with pytest.raises(ValueError, match="CUDA indices"):
        parse_max_memory("cuda:0=22GiB")

    rows = [{"id": index} for index in range(7)]
    shard0, pre0 = select_dataset_shard(rows, limit=6, num_shards=2, shard_index=0)
    shard1, pre1 = select_dataset_shard(rows, limit=6, num_shards=2, shard_index=1)
    assert pre0 == pre1 == 6
    assert [index for index, _ in shard0] == [0, 2, 4]
    assert [index for index, _ in shard1] == [1, 3, 5]

    common = {
        "input_path": "/data/a.json",
        "input_sha256": "a" * 64,
        "input_num_rows": 6,
        "pre_shard_num_rows": 6,
        "selected_num_rows": 3,
        "requested_limit": 6,
        "num_shards": 2,
        "shard_index": 0,
        "skip_invalid": False,
        "max_seq_len": 2048,
        "max_attention_gib": 24.0,
        "allow_large_attention": False,
        "model_name": "org/model",
        "query_chunk_size": 64,
        "device": "cuda:0",
        "device_map": "none",
        "archive_compression": "none",
    }
    method0, scope0, _, _ = fingerprint_extraction_config(common)
    method1, scope1, _, _ = fingerprint_extraction_config(
        {**common, "shard_index": 1, "device": "cuda:1"}
    )
    assert method0 == method1
    assert scope0 != scope1
    method_chunk, _, _, _ = fingerprint_extraction_config(
        {**common, "query_chunk_size": 128}
    )
    assert method_chunk != method0
    with pytest.raises(ValueError, match="must be an integer"):
        canonical_record(
            {"problem": "x", "steps": ["one", "two"], "label": 0.5}, 1
        )
    with pytest.raises(ValueError, match="integer token IDs"):
        canonical_record(
            {
                "problem": "x",
                "steps": ["one"],
                "label": -1,
                "prompt_token_ids": [1.2],
            },
            2,
        )


def test_trace_request_gate_preserves_legacy_evidence_and_separates_v2_hashes(
    tmp_path,
):
    from hypergraph.attention.pipeline_guard import (
        TRACE_REQUEST_SCHEMA,
        validate_or_initialize_trace_request,
    )

    request = {"input": "/data/gsm8k.json", "layer": "14", "limit": ""}
    legacy_path = tmp_path / "legacy" / "pipeline_request.json"
    legacy_path.parent.mkdir()
    legacy_payload = {**request, "method_code_sha256": "a" * 64}
    legacy_bytes = (
        json.dumps(legacy_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    legacy_path.write_bytes(legacy_bytes)

    accepted = validate_or_initialize_trace_request(
        legacy_path,
        request=request,
        extraction_code_sha256="b" * 64,
    )
    assert accepted["mode"] == "validated_legacy_without_rewrite"
    assert accepted["legacy_method_code_sha256"] == "a" * 64
    assert legacy_path.read_bytes() == legacy_bytes
    with pytest.raises(ValueError, match="legacy trace request mismatch"):
        validate_or_initialize_trace_request(
            legacy_path,
            request={**request, "layer": "15"},
            extraction_code_sha256="b" * 64,
        )

    v2_path = tmp_path / "v2" / "pipeline_request.json"
    initialized = validate_or_initialize_trace_request(
        v2_path,
        request=request,
        extraction_code_sha256="c" * 64,
    )
    assert initialized["mode"] == "initialized_v2"
    stored = json.loads(v2_path.read_text(encoding="utf-8"))
    assert stored["schema"] == TRACE_REQUEST_SCHEMA
    validated = validate_or_initialize_trace_request(
        v2_path,
        request=request,
        extraction_code_sha256="c" * 64,
    )
    assert validated["mode"] == "validated_v2"
    assert validated["request_file_sha256"] == initialized["request_file_sha256"]
    with pytest.raises(ValueError, match="trace request mismatch"):
        validate_or_initialize_trace_request(
            v2_path,
            request=request,
            extraction_code_sha256="d" * 64,
        )


def test_generator_cohort_is_materialized_before_limit_with_source_row_audit(tmp_path):
    from hypergraph.attention.cohort import materialize_generator_cohort

    rows = [
        {
            "id": "q0",
            "problem": "zero",
            "steps": ["ok"],
            "label": -1,
            "generator": "Qwen2-7B-Instruct",
        },
        {
            "id": "q1",
            "problem": "one",
            "steps": ["bad"],
            "label": 0,
            "generator": "Llama-3.1-8B-Instruct",
        },
        {
            "id": "q2",
            "problem": "two",
            "steps": ["ok"],
            "label": -1,
            "generator": "Llama-3.1-8B-Instruct",
        },
    ]
    source = tmp_path / "gsm8k.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "cohorts" / "llama.json"
    report = materialize_generator_cohort(
        source,
        output,
        generator_model="llama-3.1-8b-instruct",
    )
    assert report["num_input_rows"] == 3
    assert report["num_selected_rows"] == 2
    assert report["source_row_indices"] == [1, 2]
    assert report["response_label_counts_selected"] == {
        "negative": 1,
        "positive": 1,
    }
    assert [row["id"] for row in json.loads(output.read_text(encoding="utf-8"))] == [
        "q1",
        "q2",
    ]
    same = materialize_generator_cohort(
        source,
        output,
        generator_model="llama-3.1-8b-instruct",
    )
    assert same["cohort_sha256"] == report["cohort_sha256"]
    with pytest.raises(ValueError, match="different paths"):
        materialize_generator_cohort(
            source,
            output,
            report_path=output,
            generator_model="llama-3.1-8b-instruct",
        )


def test_shard_audit_enforces_identity_membership_uniqueness_and_completeness(tmp_path):
    from hypergraph.attention.shards import (
        audit_extraction_manifests,
        audit_scope_records,
    )

    input_sha = "a" * 64

    def scope(shard_index, *, sha=input_sha):
        value = {
            "input_sha256": sha,
            "input_num_rows": 4,
            "pre_shard_num_rows": 4,
            "selected_num_rows": 2,
            "requested_limit": None,
            "num_shards": 2,
            "shard_index": shard_index,
            "skip_invalid": False,
            "max_seq_len": 2048,
            "max_attention_gib": 24.0,
            "allow_large_attention": False,
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return value, hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    records = []
    for shard_index, row_indices in ((0, (0, 2)), (1, (1, 3))):
        value, fingerprint = scope(shard_index)
        for row_index in row_indices:
            records.append(
                {
                    "trace_id": f"trace-{row_index}",
                    "scope": value,
                    "source_input_sha256": input_sha,
                    "source_row_index": row_index,
                    "extraction_scope_fingerprint": fingerprint,
                    "status": "ok",
                }
            )
    report = audit_scope_records(records)
    assert report["complete"]
    assert report["cohorts"][0]["observed_shards"] == [0, 1]

    with pytest.raises(ValueError, match="incomplete"):
        audit_scope_records(records[:-1])
    incomplete = audit_scope_records(records[:-1], allow_incomplete=True)
    assert not incomplete["complete"]
    with pytest.raises(ValueError, match="duplicate source row"):
        audit_scope_records([*records, {**records[0], "trace_id": "renamed-copy"}])
    with pytest.raises(ValueError, match="not a member of shard"):
        audit_scope_records(
            [{**records[0], "scope": scope(1)[0], "extraction_scope_fingerprint": scope(1)[1]}]
        )
    with pytest.raises(ValueError, match="does not match scope"):
        audit_scope_records([{**records[0], "extraction_scope_fingerprint": "b" * 64}])

    other_sha = "b" * 64
    other_records = []
    for record in records:
        shard_index = int(record["scope"]["shard_index"])
        other_scope, other_fingerprint = scope(shard_index, sha=other_sha)
        other_records.append(
            {
                **record,
                "trace_id": "other-" + record["trace_id"],
                "scope": other_scope,
                "source_input_sha256": other_sha,
                "extraction_scope_fingerprint": other_fingerprint,
            }
        )
    with pytest.raises(ValueError, match="multiple source input"):
        audit_scope_records([*records, *other_records])
    assert audit_scope_records(
        [*records, *other_records], allow_multiple_inputs=True
    )["complete"]

    skipped = [{**record} for record in records]
    skipped[-1]["status"] = "skipped"
    with pytest.raises(ValueError, match="incomplete"):
        audit_scope_records(skipped)
    assert not audit_scope_records(skipped, allow_incomplete=True)["complete"]

    shard_dir = tmp_path / "shard0"
    shard_dir.mkdir()
    manifest_scope, manifest_fingerprint = scope(0)
    manifest = {
        "extraction_config": manifest_scope,
        "extraction_scope_fingerprint": manifest_fingerprint,
        "traces": [
            {
                "index": 0,
                "sample_id": "trace-0",
                "file": "trace-0.npz",
                "status": "ok",
            }
        ],
    }
    (shard_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="missing trace file"):
        audit_extraction_manifests([str(shard_dir)], allow_incomplete=True)


def test_release_metrics_reject_nonfinite_predictions_and_keep_json_booleans():
    from hypergraph.attention.train import (
        _binary_metrics,
        _finite_json,
        _tie_aware_localization_rank,
        _trace_metrics_by_generator,
    )

    assert _finite_json({"flag": True}) == {"flag": True}
    with pytest.raises(RuntimeError, match="NaN or infinity"):
        _binary_metrics([0, 1, 1], [0.1, np.nan, 0.9])
    with pytest.raises(ValueError, match="aligned"):
        _binary_metrics([0, 1], [0.1])
    with pytest.raises(RuntimeError, match="NaN or infinity"):
        _tie_aware_localization_rank([0.1, np.nan, 0.9], 2, [True, True, True])
    grouped = _trace_metrics_by_generator(
        [
            {"generator_model": "A", "label": 0, "score": 0.1},
            {"generator_model": "A", "label": 1, "score": 0.9},
            {"generator_model": "B", "label": 1, "score": 0.8},
        ]
    )
    assert grouped["A"]["auroc"] == pytest.approx(1.0)
    assert grouped["B"]["n"] == 1
    assert grouped["B"]["auroc"] is None


def test_dual_gpu_script_keeps_worker_identity_options_protected():
    script = (
        __import__("pathlib")
        .Path("hypergraph/attention/scripts/extract_dual_gpu.sh")
        .read_text(encoding="utf-8")
    )
    assert "GPU0 and GPU1 must name two different" in script
    assert "--output_dir|--output_dir=*" in script
    assert "--num-shards|--num-shards=*" in script
    data_worker = script.index("CUDA_VISIBLE_DEVICES=")
    passthrough = script.index('"$@"', data_worker)
    protected_output = script.index("--output_dir", passthrough)
    assert passthrough < protected_output


def test_strict_response_pipeline_uses_exact_full_forward_by_default():
    extractor = Path(
        "hypergraph/attention/scripts/extract_dual_gpu.sh"
    ).read_text(encoding="utf-8")
    pipeline = Path(
        "hypergraph/attention/scripts/run_single_layer_response_pipeline.sh"
    ).read_text(encoding="utf-8")
    all_datasets = Path(
        "hypergraph/attention/scripts/run_all_processbench_response_pipeline.sh"
    ).read_text(encoding="utf-8")
    fixed_aggregate = Path(
        "hypergraph/attention/aggregate_fixed.py"
    ).read_text(encoding="utf-8")

    assert 'MODE="${MODE:-model_parallel}"' in extractor
    assert 'QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-0}"' in extractor
    assert 'MAX_SEQ_LEN="${MAX_SEQ_LEN:-0}"' in extractor
    assert 'MODE="${MODE:-model_parallel}"' in pipeline
    assert 'QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-0}"' in pipeline
    assert 'MAX_SEQ_LEN="${MAX_SEQ_LEN:-0}"' in pipeline
    assert 'ACTIVATION_LAYER="${ACTIVATION_LAYER:-}"' in pipeline
    assert 'ACTIVATION_LAYER="${ACTIVATION_LAYER:-$((LAYER + 1))}"' in pipeline
    assert 'SEQ_POLICY_SUFFIX="_nocap"' in pipeline
    assert 'TRACE_VARIANT_SUFFIX="${SEQ_POLICY_SUFFIX}${ACTIVATION_TRACE_SUFFIX}"' in pipeline
    assert 'FULL_DATASET_TRACE_ROOT="${REPO_ROOT}/data/attention_traces/${DATASET_TAG}_llama31_layer${LAYER}${TRACE_VARIANT_SUFFIX}"' in pipeline
    assert 'EXTRACT_ACTIVATION_ARGS+=(--activation_layer "${ACTIVATION_LAYER}")' in pipeline
    assert '"activation_layer=${ACTIVATION_LAYER}"' in pipeline
    assert "hidden-state mismatch" in pipeline
    assert "QUERY_CHUNK_SIZE must be a non-negative integer" in pipeline
    assert "strict pipeline requires QUERY_CHUNK_SIZE=0" in pipeline
    assert "EXTRACTION_CODE_SHA256 TRAINING_CODE_SHA256" in pipeline
    assert "hypergraph.attention.pipeline_guard" in pipeline
    assert "hypergraph.attention.cohort" in pipeline
    assert 'TRACE_INPUT_MODE="materialized_matched_generator"' in pipeline
    assert 'TRACE_EXTRACTION_LIMIT=""' in pipeline
    assert 'COHORT_ARGS+=(--limit "${LIMIT}")' in pipeline
    assert '"cohort_report_sha256=${COHORT_REPORT_SHA256}"' in pipeline
    assert "validated_legacy_without_rewrite) TRACE_REQUEST_KIND=\"legacy\"" in pipeline
    assert '"legacy_monolithic_method_code_sha256=${LEGACY_METHOD_CODE_SHA256}"' in pipeline
    assert 'PREFLIGHT_CANDIDATE="$(mktemp ' in pipeline
    assert '--output "${PREFLIGHT_CANDIDATE}"' in pipeline
    assert '"current_extraction_validation_code_sha256=${EXTRACTION_CODE_SHA256}"' in pipeline
    assert "seed must be a canonical non-negative integer" in pipeline
    assert "--seeds contains a duplicate value" in pipeline
    assert "--objective response_bce" in pipeline
    assert 'COHORT_SUFFIX="_observer_all"' in pipeline
    assert '--generator-model "${GENERATOR_MODEL}"' in pipeline
    assert 'MODE="${MODE:-model_parallel}"' in all_datasets
    assert '--mode "${MODE}"' in all_datasets
    assert '--generator-model "${GENERATOR_MODEL}"' in all_datasets
    assert 'cohort_suffix="_observer_all"' in all_datasets
    assert 'NODE_FEATURE_MODE="${NODE_FEATURE_MODE:-attention_diagonal}"' in all_datasets
    assert 'node_variant_suffix="_node_attention_hidden_hs${ACTIVATION_LAYER}"' in all_datasets
    assert 'seq_policy_suffix="_nocap"' in all_datasets
    assert 'SUMMARY_ROOT="${REPO_ROOT}/results/attention_hypergraph/' in all_datasets
    assert 'root = repo / "results" / "attention_hypergraph"' in all_datasets
    assert "all_processbench_fixed_holdout_request_v1" in all_datasets
    assert '"preflight_sha256": preflight_sha256' in all_datasets
    assert 'SPLIT_MODE="${SPLIT_MODE:-fixed_holdout}"' in pipeline
    assert 'THRESHOLD="${THRESHOLD:-0.05}"' in pipeline
    assert 'TRACE_EQUIVALENCE_THRESHOLD="${TRACE_EQUIVALENCE_THRESHOLD:-0.01}"' in pipeline
    assert '"chunk_equivalence_threshold=${TRACE_EQUIVALENCE_THRESHOLD}"' in pipeline
    assert '--chunk-equivalence-threshold "${TRACE_EQUIVALENCE_THRESHOLD}"' in pipeline
    assert 'SOURCE_SELECTION="${SOURCE_SELECTION:-threshold_fallback_topk}"' in pipeline
    assert 'TOP_K="${TOP_K:-16}"' in pipeline
    assert 'MIN_SOURCES="${MIN_SOURCES:-2}"' in pipeline
    assert 'TOPOLOGY_HEADS="${TOPOLOGY_HEADS:-0}"' in pipeline
    assert "keeps the original 3-D hyperedge attributes" in pipeline
    assert '--selected-heads "${TOPOLOGY_HEADS}"' in pipeline
    assert '--split-mode fixed_holdout' in pipeline
    assert '"partition_group_ids"' not in pipeline  # written by train.py, not fabricated in shell
    assert "-m hypergraph.attention.aggregate_fixed" in pipeline
    assert '"final_test": dict(test)' in fixed_aggregate
    assert '"generator_final_test"' in all_datasets
    assert '"macro_final_test"' in all_datasets
    assert '"node_feature_mode": node_feature_mode' in all_datasets
    assert '"activation_layer": None if not activation_layer else int(activation_layer)' in all_datasets
    assert '"edge_attributes": ["attention_mean", "attention_max", "flattened_head_normalized"]' in all_datasets
    assert "fold${fold}" not in pipeline
    assert "pooled_oof_test" not in pipeline


def test_artifact_layout_separates_intermediate_data_from_results():
    root = Path(__file__).resolve().parents[1]
    single = (root / "hypergraph/attention/scripts/run_single_layer_response_pipeline.sh").read_text()
    dual = (root / "hypergraph/attention/scripts/extract_dual_gpu.sh").read_text()
    migration = (root / "hypergraph/attention/scripts/migrate_artifacts_layout.sh").read_text()

    assert '${REPO_ROOT}/data/attention_cohorts' in single
    assert '${REPO_ROOT}/data/attention_traces' in single
    assert '${REPO_ROOT}/results/attention_hypergraph' in single
    assert 'data/attention_traces/llama31_8b_observer' in dual
    assert 'outputs/attention_cohorts' in migration
    assert 'data/attention_cohorts' in migration
    assert 'outputs/attention_traces' in migration
    assert 'data/attention_traces' in migration
    assert 'outputs/attention_hypergraph' in migration
    assert 'results/attention_hypergraph' in migration
    assert 'destination already exists; refusing to merge automatically' in migration


def test_artifact_layout_migration_moves_only_owned_directories(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")

    root = Path(__file__).resolve().parents[1]
    script = root / "hypergraph/attention/scripts/migrate_artifacts_layout.sh"
    owned = {
        "outputs/attention_cohorts": "data/attention_cohorts",
        "outputs/attention_traces": "data/attention_traces",
        "outputs/attention_hypergraph": "results/attention_hypergraph",
    }
    for source in owned:
        source_path = tmp_path / source
        source_path.mkdir(parents=True)
        (source_path / "sentinel.txt").write_text(source, encoding="utf-8")
    unrelated = tmp_path / "outputs/residual_flow"
    unrelated.mkdir(parents=True)
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")

    environment = dict(os.environ, REPO_ROOT_OVERRIDE=str(tmp_path))
    subprocess.run([bash, str(script)], check=True, env=environment, capture_output=True)

    for source, destination in owned.items():
        assert not (tmp_path / source).exists()
        assert (tmp_path / destination / "sentinel.txt").read_text(encoding="utf-8") == source
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_artifact_layout_migration_refuses_destination_collision(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")

    root = Path(__file__).resolve().parents[1]
    script = root / "hypergraph/attention/scripts/migrate_artifacts_layout.sh"
    (tmp_path / "outputs/attention_cohorts").mkdir(parents=True)
    (tmp_path / "outputs/attention_traces").mkdir(parents=True)
    (tmp_path / "data/attention_cohorts").mkdir(parents=True)
    environment = dict(os.environ, REPO_ROOT_OVERRIDE=str(tmp_path))

    completed = subprocess.run(
        [bash, str(script)], env=environment, capture_output=True, text=True
    )
    assert completed.returncode != 0
    assert "refusing to merge automatically" in completed.stderr
    assert (tmp_path / "outputs/attention_cohorts").is_dir()
    assert (tmp_path / "outputs/attention_traces").is_dir()


def test_extraction_has_no_artificial_sequence_cap_by_default():
    args = build_extraction_parser().parse_args(
        [
            "--input",
            "/data/gsm8k.json",
            "--output_dir",
            "/tmp/traces",
            "--model",
            "/models/llama",
        ]
    )
    assert args.max_seq_len == 0

    validate_sequence_length(4096, max_seq_len=0, model_context_limit=131072)
    with pytest.raises(ValueError, match="configured max_seq_len"):
        validate_sequence_length(4096, max_seq_len=2048, model_context_limit=131072)
    with pytest.raises(ValueError, match="model context window"):
        validate_sequence_length(131073, max_seq_len=0, model_context_limit=131072)


def test_extraction_progress_uses_tqdm(monkeypatch):
    from hypergraph.attention import extract as extraction

    observed = {}

    def fake_tqdm(iterable, **kwargs):
        observed.update(kwargs)
        return iterable

    monkeypatch.setattr(extraction, "tqdm", fake_tqdm)
    rows = [(0, {"id": "sample"})]
    assert list(
        extraction._extraction_progress(rows, shard_index=0, num_shards=2)
    ) == rows
    assert observed["desc"] == "extract shard 1/2"
    assert observed["unit"] == "trace"
    assert observed["dynamic_ncols"] is True


def test_full_forward_captures_only_requested_decoder_layers():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    class SelfAttention(torch.nn.Module):
        def __init__(self, layer_index):
            super().__init__()
            self.layer_index = int(layer_index)
            self.output_attention_flags = []

        def forward(self, hidden_states, *, output_attentions=False, **kwargs):
            del kwargs
            self.output_attention_flags.append(bool(output_attentions))
            tokens = int(hidden_states.shape[1])
            attention = (
                torch.full(
                    (1, 3, tokens, tokens),
                    float(self.layer_index + 1),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                if output_attentions
                else None
            )
            return hidden_states + float(self.layer_index + 1), attention

    class DecoderLayer(torch.nn.Module):
        def __init__(self, layer_index):
            super().__init__()
            self.self_attn = SelfAttention(layer_index)

        def forward(self, hidden_states, *, output_attentions=False, **kwargs):
            updated, _ = self.self_attn(
                hidden_states,
                output_attentions=output_attentions,
                **kwargs,
            )
            return (updated,)

    class SelectiveModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([DecoderLayer(i) for i in range(4)])
            self.top_level_attention_flags = []

        def forward(
            self,
            *,
            input_ids,
            attention_mask,
            output_attentions,
            output_hidden_states,
            use_cache,
            return_dict,
        ):
            del attention_mask, output_hidden_states, use_cache, return_dict
            self.top_level_attention_flags.append(bool(output_attentions))
            hidden = input_ids.to(dtype=torch.float32)[..., None]
            for layer in self.layers:
                hidden = layer(
                    hidden, output_attentions=bool(output_attentions)
                )[0]
            return SimpleNamespace(attentions=None, hidden_states=None)

    model = SelectiveModel()
    trace = extract_trace(
        model,
        torch,
        {
            "token_ids": np.arange(5, dtype=np.int64),
            "attention_mask": np.ones(5, dtype=np.int64),
        },
        device="cpu",
        attention_layers=(1, 3),
        attention_heads=(0, 2),
        activation_layer=None,
        storage_dtype="float32",
        query_chunk_size=0,
    )

    assert model.top_level_attention_flags == [False]
    assert [layer.self_attn.output_attention_flags for layer in model.layers] == [
        [False],
        [True],
        [False],
        [True],
    ]
    assert trace["attention"].shape == (2, 2, 5, 5)
    assert np.all(trace["attention"][0] == 2.0)
    assert np.all(trace["attention"][1] == 4.0)


def test_selected_capture_matches_full_huggingface_llama_attention():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = transformers.LlamaModel(config).eval()
    token_ids = np.asarray([1, 7, 3, 9, 2], dtype=np.int64)
    tokenized = {
        "token_ids": token_ids,
        "attention_mask": np.ones_like(token_ids),
    }
    selected = extract_trace(
        model,
        torch,
        tokenized,
        device="cpu",
        attention_layers=(1,),
        attention_heads=(0, 2),
        activation_layer=None,
        storage_dtype="float32",
        query_chunk_size=0,
    )["attention"]
    with torch.inference_mode():
        full = model(
            input_ids=torch.as_tensor(token_ids, dtype=torch.long)[None],
            attention_mask=torch.ones((1, len(token_ids)), dtype=torch.long),
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    expected = full.attentions[1][0, [0, 2]].float().numpy()
    np.testing.assert_allclose(selected[0], expected, atol=0, rtol=0)


def test_dual_gpu_wrapper_options_match_extraction_cli():
    parser = build_extraction_parser()
    args = parser.parse_args(
        [
            "--input",
            "/data/gsm8k.json",
            "--output_dir",
            "/tmp/shard0",
            "--model",
            "/models/llama",
            "--model_class",
            "base",
            "--replay_mode",
            "observer",
            "--prompt_style",
            "plain",
            "--dtype",
            "auto",
            "--storage_dtype",
            "float32",
            "--query_chunk_size",
            "64",
            "--verify_chunked_equivalence",
            "--archive_compression",
            "none",
            "--max_seq_len",
            "2048",
            "--max_attention_gib",
            "24",
            "--attention_layers",
            "14",
            "--attention_heads",
            "all",
            "--chunk_equivalence_threshold",
            "0.01",
            "--device",
            "cuda:0",
            "--num_shards",
            "2",
            "--shard_index",
            "0",
        ]
    )

    assert args.replay_mode == "observer"
    assert args.prompt_style == "plain"
    assert args.attention_layers == "14"
    assert args.num_shards == 2
    assert args.shard_index == 0


def test_extraction_cli_accepts_hyphenated_prompt_and_replay_aliases():
    args = build_extraction_parser().parse_args(
        [
            "--input",
            "/data/gsm8k.json",
            "--output_dir",
            "/tmp/shard0",
            "--model",
            "/models/llama",
            "--replay-mode",
            "observer",
            "--prompt-style",
            "chat",
        ]
    )

    assert args.replay_mode == "observer"
    assert args.prompt_style == "chat"


def test_extractor_rejects_empty_steps_and_nonempty_output_dirs(tmp_path):
    with pytest.raises(ValueError, match="silently renumber gold_step"):
        canonical_record(
            {"problem": "x", "steps": ["first", "  ", "third"], "label": 2},
            0,
        )
    output = tmp_path / "extract"
    assert prepare_empty_output_dir(output) == output.resolve()
    (output / "stale.npz").write_bytes(b"stale")
    with pytest.raises(FileExistsError, match="stale traces"):
        prepare_empty_output_dir(output)


def test_extraction_manifest_checkpoint_is_atomic_and_self_describing(tmp_path):
    destination = tmp_path / "manifest.json"
    _write_extraction_manifest(
        destination,
        extraction_config={"input_sha256": "a" * 64},
        extraction_fingerprint="method-fingerprint",
        extraction_scope_fingerprint="scope-fingerprint",
        traces=[{"index": 0, "status": "ok"}],
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["chunk_equivalence_policy"] == "per_trace_prefix"
    assert payload["traces"] == [{"index": 0, "status": "ok"}]
    assert not destination.with_suffix(".json.tmp").exists()


def test_cached_trace_is_bound_to_the_verified_graph_selector():
    from types import SimpleNamespace

    from hypergraph.attention.train import _validate_chunk_graph_contract

    trace = SimpleNamespace(
        metadata={
            "extraction_forward_mode": "cached_query_chunks",
            "chunk_equivalence_json": json.dumps(
                {"status": "prefix_pass", "topology_threshold": 0.01}
            ),
        }
    )
    assert not _validate_chunk_graph_contract(
        trace, AttentionHypergraphConfig(), allow_unverified=False
    )
    with pytest.raises(ValueError, match="not topology-gated"):
        _validate_chunk_graph_contract(
            trace,
            AttentionHypergraphConfig(top_k=2),
            allow_unverified=False,
        )
    assert _validate_chunk_graph_contract(
        trace,
        AttentionHypergraphConfig(source_selection="top_k_only", top_k=2),
        allow_unverified=True,
    )

    trace.metadata["chunk_equivalence_json"] = json.dumps(
        {"status": "prefix_pass", "topology_threshold": None}
    )
    with pytest.raises(ValueError, match="not topology-gated"):
        _validate_chunk_graph_contract(
            trace, AttentionHypergraphConfig(), allow_unverified=False
        )


def test_same_generator_replay_requires_the_stored_generation_axis():
    assert generator_matches_model("meta/Llama-3.1-8B", "D:/models/Llama-3.1-8B")
    assert not generator_matches_model("Llama-3.1-8B", "Llama-3.1-70B")
    assert not generator_matches_model("org-a/model-x", "org-b/model-x")
    with pytest.raises(ValueError, match="rendered_prompt"):
        require_exact_replay_inputs({"generator_model": "Llama-3.1-8B"})

    require_exact_replay_inputs(
        {
            "generator_model": "Llama-3.1-8B",
            "rendered_prompt": "prompt",
            "response_text": "answer",
            "prompt_token_ids": [1, 2],
            "response_token_ids": [3],
        }
    )

    full_commit = "a" * 40
    assert resolve_loaded_commit("a" * 8, full_commit, full_commit) == full_commit
    assert (
        classify_model_commit_source(
            is_local_model=True,
            requested_commit=full_commit,
            model_commit=None,
            tokenizer_commit=None,
        )
        == "local_declared_commit"
    )
    assert (
        classify_model_commit_source(
            is_local_model=False,
            requested_commit=full_commit,
            model_commit=full_commit,
            tokenizer_commit=full_commit,
        )
        == "remote_resolved_model_commit"
    )
    assert (
        classify_model_commit_source(
            is_local_model=True,
            requested_commit=None,
            model_commit=None,
            tokenizer_commit=full_commit,
        )
        == "local_tokenizer_metadata_only"
    )
    with pytest.raises(ValueError, match="conflict"):
        resolve_loaded_commit("a" * 40, "b" * 40, None)
    with pytest.raises(ValueError, match="immutable hexadecimal"):
        resolve_loaded_commit("moving-main", None, None)


def test_representation_fingerprint_separates_observer_source_from_method():
    from types import SimpleNamespace

    from hypergraph.attention.data import (
        trace_representation_fingerprint,
        trace_source_provenance,
    )

    method = {
        "trace_contract": "exact_prompt_response_attention_v1",
        "model_name": "/models/Meta-Llama-3.1-8B-Instruct",
        "model_commit_source": "unavailable",
        "tokenizer_name": "/models/Meta-Llama-3.1-8B-Instruct",
        "prompt_style": "plain",
        "replay_mode": "observer",
        "replay_fidelity": "observer_counterfactual",
        "prompt_provenance": "frozen_plain_observer",
        "prompt_add_special_tokens": True,
        "extraction_dtype": "bfloat16",
        "attention_storage_dtype": "float32",
        "extraction_fingerprint": "method-fingerprint",
    }

    def make_trace(generator, **updates):
        metadata = dict(method, generator_model=generator, **updates)
        return SimpleNamespace(
            metadata=metadata,
            attention_layer_ids=np.asarray([14], dtype=np.int64),
            attention_head_ids=np.arange(32, dtype=np.int64),
            num_model_layers=32,
            num_model_heads=32,
        )

    qwen = make_trace("Qwen2-7B-Instruct")
    llama = make_trace("Llama-3.1-8B-Instruct")
    assert trace_representation_fingerprint(qwen) == trace_representation_fingerprint(
        llama
    )
    assert trace_source_provenance(qwen) != trace_source_provenance(llama)

    other_observer = make_trace(
        "Qwen2-7B-Instruct", model_name="/models/other-observer"
    )
    assert trace_representation_fingerprint(qwen) != trace_representation_fingerprint(
        other_observer
    )

    incomplete = make_trace("Qwen2-7B-Instruct", replay_fidelity="")
    incomplete_other = make_trace("Llama-3.1-8B-Instruct", replay_fidelity="")
    assert trace_representation_fingerprint(
        incomplete
    ) != trace_representation_fingerprint(incomplete_other)

    same_generator = make_trace(
        "Qwen2-7B-Instruct",
        replay_mode="same_generator",
        replay_fidelity="token_axis_verified_weights_unverified",
    )
    same_generator_other = make_trace(
        "Llama-3.1-8B-Instruct",
        replay_mode="same_generator",
        replay_fidelity="token_axis_verified_weights_unverified",
    )
    assert trace_representation_fingerprint(
        same_generator
    ) != trace_representation_fingerprint(same_generator_other)


def test_dataset_generator_match_uses_only_the_curated_meta_llama_alias():
    from hypergraph.attention.pipeline_guard import (
        dataset_generator_matches_observer,
    )

    observer = "/models/Meta-Llama-3.1-8B-Instruct"
    assert dataset_generator_matches_observer("Llama-3.1-8B-Instruct", observer)
    assert dataset_generator_matches_observer("Meta-Llama-3.1-8B-Instruct", observer)
    assert not dataset_generator_matches_observer("Qwen2-7B-Instruct", observer)


def test_strict_preflight_accepts_cross_generator_observer_and_filters_before_limit(
    tmp_path,
):
    from hypergraph.attention.train import main

    trace_dir = _write_audited_observer_cohort(
        tmp_path,
        [
            "Qwen2-7B-Instruct",
            "Llama-3.1-8B-Instruct",
            "Qwen2-7B-Instruct",
            "Llama-3.1-8B-Instruct",
        ],
    )
    all_report = tmp_path / "all-preflight.json"
    assert (
        main(
            [
                "inspect",
                str(trace_dir),
                "--objective",
                "response_bce",
                "--allow-observer-traces",
                "--output",
                str(all_report),
            ]
        )
        == 0
    )
    all_payload = json.loads(all_report.read_text(encoding="utf-8"))
    assert all_payload["inspection_mode"] == "supervised_cohort_gate"
    assert all_payload["cohort_gate_passed"] is True
    assert all_payload["cohort_audit"]["representation_fingerprint"]
    assert all_payload["cohort_audit"]["source_provenance"][
        "generator_model_counts"
    ] == {"Llama-3.1-8B-Instruct": 2, "Qwen2-7B-Instruct": 2}
    assert all_payload["selection"][
        "generator_response_label_counts_selected"
    ] == {
        "Llama-3.1-8B-Instruct": {
            "negative": 0,
            "positive": 2,
            "unlabeled": 0,
        },
        "Qwen2-7B-Instruct": {
            "negative": 2,
            "positive": 0,
            "unlabeled": 0,
        },
    }

    matched_report = tmp_path / "matched-preflight.json"
    assert (
        main(
            [
                "inspect",
                str(trace_dir),
                "--objective",
                "response_bce",
                "--allow-observer-traces",
                "--generator-model",
                "Llama-3.1-8B-Instruct",
                "--limit",
                "1",
                "--output",
                str(matched_report),
            ]
        )
        == 0
    )
    selection = json.loads(matched_report.read_text(encoding="utf-8"))["selection"]
    assert selection["num_input_traces"] == 4
    assert selection["num_matched_before_limit"] == 2
    assert selection["num_selected"] == 1
    assert selection["num_excluded_generator"] == 2
    assert selection["num_excluded_limit"] == 1
    assert selection["generator_distribution_selected"] == {
        "Llama-3.1-8B-Instruct": 1
    }


def test_limit_rejects_multiple_storage_ordered_trace_roots():
    from types import SimpleNamespace

    from hypergraph.attention.train import (
        _iter_input_traces,
        _validate_limit_input_order,
    )

    args = SimpleNamespace(inputs=["shard0", "shard1"], limit=3)
    with pytest.raises(SystemExit, match="storage-order dependent"):
        next(iter(_iter_input_traces(args)))
    sharded = SimpleNamespace(
        metadata={"extraction_scope_json": json.dumps({"num_shards": 2})}
    )
    with pytest.raises(SystemExit, match="sharded extraction scope"):
        _validate_limit_input_order(
            SimpleNamespace(inputs=["parent"], limit=3), sharded
        )


def test_strict_preflight_still_rejects_different_observer_representations(tmp_path):
    from hypergraph.attention.train import main

    trace_dir = _write_audited_observer_cohort(
        tmp_path,
        ["Qwen2-7B-Instruct", "Llama-3.1-8B-Instruct"],
        model_names=[
            "/models/Meta-Llama-3.1-8B-Instruct",
            "/models/another-observer",
        ],
    )
    with pytest.raises(SystemExit, match="different observer/template/layer"):
        main(
            [
                "inspect",
                str(trace_dir),
                "--objective",
                "response_bce",
                "--allow-observer-traces",
            ]
        )


def test_training_replay_provenance_is_a_strict_state_machine():
    from hypergraph.attention.data import TRACE_CONTRACT
    from hypergraph.attention.train import _audit_replay_provenance

    commit = "a" * 40
    verified = {
        "trace_contract": TRACE_CONTRACT,
        "model_name": "org/model-x",
        "model_commit_hash": commit,
        "model_commit_source": "remote_resolved_model_commit",
        "tokenizer_name": "org/model-x",
        "prompt_style": "plain",
        "replay_mode": "same_generator",
        "replay_fidelity": "weight_and_token_verified_replay",
        "unverified_generator_weights_explicitly_allowed": False,
        "prompt_provenance": "stored_rendered_prompt",
        "generator_model": "org/model-x",
        "generator_model_commit": commit,
        "prompt_add_special_tokens": False,
        "extraction_dtype": "bfloat16",
        "attention_storage_dtype": "float32",
        "extraction_fingerprint": "fingerprint",
    }
    scope_json = json.dumps(
        {
            "input_sha256": "input-hash",
            "input_num_rows": 1,
            "pre_shard_num_rows": 1,
            "selected_num_rows": 1,
            "requested_limit": None,
            "num_shards": 1,
            "shard_index": 0,
            "skip_invalid": False,
            "max_seq_len": 2048,
            "max_attention_gib": 24.0,
            "allow_large_attention": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    audit_metadata_base = {
        "rendered_prompt_sha256": "prompt-hash",
        "response_text_sha256": "response-hash",
        "extraction_scope_fingerprint": hashlib.sha256(
            scope_json.encode("utf-8")
        ).hexdigest(),
        "extraction_scope_json": scope_json,
        "source_input_sha256": "input-hash",
        "source_row_index": 0,
        "extraction_forward_mode": "full",
        "chunk_equivalence_status": "not_applicable",
        "chunk_equivalence_json": json.dumps(
            {"status": "not_applicable"}, separators=(",", ":")
        ),
    }

    def bound_inputs(provenance):
        method_config = {
            "trace_contract": provenance.get("trace_contract"),
            "model_name": provenance.get("model_name"),
            "model_commit_hash": provenance.get("model_commit_hash"),
            "model_commit_source": provenance.get("model_commit_source"),
            "tokenizer_name": provenance.get("tokenizer_name"),
            "prompt_style": provenance.get("prompt_style"),
            "replay_mode": provenance.get("replay_mode"),
            "dtype": provenance.get("extraction_dtype"),
            "attention_storage_dtype": provenance.get("attention_storage_dtype"),
            "activation_layer": None,
            "allow_unverified_generator_weights": bool(
                provenance.get("unverified_generator_weights_explicitly_allowed")
            ),
            "query_chunk_size": 0,
        }
        method_json = json.dumps(
            method_config, sort_keys=True, separators=(",", ":")
        )
        bound_provenance = dict(
            provenance,
            extraction_fingerprint=hashlib.sha256(
                method_json.encode("utf-8")
            ).hexdigest(),
        )
        metadata = dict(audit_metadata_base, extraction_method_json=method_json)
        return bound_provenance, metadata

    verified, audit_metadata = bound_inputs(verified)
    audit = _audit_replay_provenance(verified, audit_metadata)
    assert audit.complete and not audit.observer and not audit.unverified_weights

    bad = dict(verified, replay_mode="typo")
    with pytest.raises(ValueError, match="unknown replay_mode"):
        _audit_replay_provenance(*bound_inputs(bad))
    bad = dict(verified, trace_contract="unknown_axis_v0")
    with pytest.raises(ValueError, match="unsupported trace_contract"):
        _audit_replay_provenance(*bound_inputs(bad))
    bad = dict(verified, replay_mode="observer")
    with pytest.raises(ValueError, match="inconsistent"):
        _audit_replay_provenance(*bound_inputs(bad))
    bad = dict(verified, generator_model="other/model-y")
    with pytest.raises(ValueError, match="does not match model_name"):
        _audit_replay_provenance(*bound_inputs(bad))
    bad = dict(verified, generator_model_commit="b" * 40)
    with pytest.raises(ValueError, match="does not match model_commit_hash"):
        _audit_replay_provenance(*bound_inputs(bad))
    bad = dict(verified, generator_model_commit="moving-main", model_commit_hash="moving-main")
    with pytest.raises(ValueError, match="immutable hexadecimal"):
        _audit_replay_provenance(*bound_inputs(bad))
    bad = dict(verified, model_commit_source="local_declared_commit")
    with pytest.raises(ValueError, match="resolved/pinned"):
        _audit_replay_provenance(*bound_inputs(bad))
    bad = dict(verified, model_commit_source="local_model_metadata_commit")
    with pytest.raises(ValueError, match="resolved/pinned"):
        _audit_replay_provenance(*bound_inputs(bad))

    missing_commit = dict(verified)
    missing_commit.pop("model_commit_hash")
    with pytest.raises(ValueError, match="requires model_commit_hash"):
        _audit_replay_provenance(*bound_inputs(missing_commit))
    missing_legacy_commit = dict(missing_commit)
    missing_legacy_commit.pop("model_commit_source")
    missing_legacy_commit.pop("replay_fidelity")
    assert not _audit_replay_provenance(*bound_inputs(missing_legacy_commit)).complete

    unverified = dict(verified)
    unverified["replay_fidelity"] = "token_axis_verified_weights_unverified"
    unverified["model_commit_source"] = "local_declared_commit"
    unverified["unverified_generator_weights_explicitly_allowed"] = True
    audit = _audit_replay_provenance(*bound_inputs(unverified))
    assert audit.complete and audit.unverified_weights

    tampered_scope = dict(audit_metadata, source_input_sha256="other-input")
    with pytest.raises(ValueError, match="does not match extraction scope"):
        _audit_replay_provenance(verified, tampered_scope)

    tampered_method = dict(verified, model_name="other/model")
    with pytest.raises(ValueError, match="disagrees with extraction method"):
        _audit_replay_provenance(tampered_method, audit_metadata)


def test_trace_loader_and_inspect_build_cli_share_one_contract(tmp_path):
    from hypergraph.attention.data import load_trace_file
    from hypergraph.attention.train import main

    attention, token_ids, response_idx = _trace()
    trace_path = tmp_path / "trace.npz"
    np.savez_compressed(
        trace_path,
        attention=attention,
        token_ids=token_ids,
        response_idx=np.asarray(response_idx),
        step_ranges=np.asarray([[3, 4], [4, 5]], np.int64),
        gold_step=np.asarray(1),
        response_y=np.asarray(1.0, np.float32),
        sample_id=np.asarray("trace-1"),
        problem_id=np.asarray("problem-1"),
    )
    loaded = load_trace_file(trace_path)
    assert len(loaded) == 1
    assert loaded[0].trace_id == "trace-1"
    assert loaded[0].group_id == "problem-1"
    assert loaded[0].activation is None

    report = tmp_path / "inspect.json"
    assert main(["inspect", str(trace_path), "--output", str(report)]) == 0
    assert report.exists()
    inspected = __import__("json").loads(report.read_text(encoding="utf-8"))
    assert "receiver_edge_coverage" in inspected["summary"]
    assert len(inspected["summary"]["position_bin_edges_per_token"]) == 5
    graph_dir = tmp_path / "graphs"
    assert main(["build", str(trace_path), "--output", str(graph_dir)]) == 0
    assert (graph_dir / "manifest.json").exists()
    graph_files = list(graph_dir.glob("*.npz"))
    assert len(graph_files) == 1
    with np.load(graph_files[0], allow_pickle=False) as archive:
        assert "construction_config_json" in archive.files
        np.testing.assert_array_equal(archive["attention_layer_ids"], [0])
        np.testing.assert_array_equal(archive["attention_head_ids"], [0])


def test_response_only_token_labels_get_an_exact_full_axis_mask(tmp_path):
    from hypergraph.attention.data import load_trace_file

    attention, token_ids, response_idx = _trace()
    trace_path = tmp_path / "token-labels.npz"
    np.savez_compressed(
        trace_path,
        attention=attention,
        token_ids=token_ids,
        response_idx=np.asarray(response_idx),
        token_y=np.asarray([0.0, 1.0], np.float32),
        trace_id=np.asarray("trace-mask"),
        group_id=np.asarray("problem-mask"),
        split=np.asarray("train"),
    )
    trace = load_trace_file(trace_path)[0]
    np.testing.assert_array_equal(trace.token_y, [-100, -100, -100, 0, 1])
    np.testing.assert_array_equal(trace.token_label_mask, [False, False, False, True, True])
    graph = build_attention_hypergraph(**trace.builder_kwargs())
    np.testing.assert_array_equal(graph.token_label_mask, trace.token_label_mask)
    assert graph.trace_id == "trace-mask"
    assert graph.group_id == "problem-mask"
    assert graph.split == "train"


def test_partial_token_labels_never_fabricate_a_negative_response_label():
    from hypergraph.attention.data import TraceFormatError, canonicalize_trace

    base = {
        "attention": np.zeros((1, 1, 5, 5), dtype=np.float32),
        "token_ids": np.arange(5, dtype=np.int64),
        "response_idx": 2,
    }
    partial_clean = canonicalize_trace(
        {
            **base,
            "token_y": np.asarray([-100, -100, 0, -100, -100], np.float32),
        }
    )
    assert partial_clean.response_y is None

    partial_positive = canonicalize_trace(
        {
            **base,
            "token_y": np.asarray([-100, -100, 0, 1, -100], np.float32),
        }
    )
    assert partial_positive.response_y == 1.0

    fully_observed_clean = canonicalize_trace(
        {
            **base,
            "token_y": np.asarray([-100, -100, 0, 0, 0], np.float32),
        }
    )
    assert fully_observed_clean.response_y == 0.0

    with pytest.raises(TraceFormatError, match="exact positive token"):
        canonicalize_trace(
            {
                **base,
                "token_y": np.asarray([-100, -100, 1, 0, -100], np.float32),
                "response_y": 0,
            }
        )
    with pytest.raises(ValueError, match="exact positive token"):
        build_attention_hypergraph(
            base["attention"],
            base["token_ids"],
            base["response_idx"],
            token_y=np.asarray([-100, -100, 1, 0, -100], np.float32),
            response_y=0,
        )


def test_label_and_metadata_aliases_are_strict_and_risk_mask_is_rejected():
    from hypergraph.attention.data import TraceFormatError, canonicalize_trace

    base = {
        "attention": np.zeros((1, 1, 5, 5), dtype=np.float32),
        "token_ids": np.arange(5, dtype=np.int64),
        "response_idx": 2,
    }
    conflicts = [
        {"response_y": 0, "is_incorrect": 1},
        {"split": "train", "partition": "test"},
        {"problem_id": "p1", "group_id": "p2"},
    ]
    for conflict in conflicts:
        with pytest.raises(TraceFormatError, match="ambiguous"):
            canonicalize_trace({**base, **conflict})

    for invalid in ("False", "0", "no"):
        with pytest.raises(TraceFormatError, match="actual bool or numeric 0/1"):
            canonicalize_trace({**base, "is_correct": invalid})

    with pytest.raises(TraceFormatError, match="risk_mask is not accepted"):
        canonicalize_trace(
            {
                **base,
                "step_ranges": np.asarray([[2, 3], [3, 5]], np.int64),
                "gold_step": 0,
                "risk_mask": np.asarray([True, False]),
            }
        )


def test_first_error_training_refuses_future_leaking_symmetric_forward(tmp_path):
    from hypergraph.attention.train import main

    with pytest.raises(SystemExit, match="future context"):
        main(
            [
                "train",
                str(tmp_path / "not-needed-for-protocol-check.npz"),
                "--objective",
                "step_bce",
                "--output",
                str(tmp_path / "run"),
            ]
        )

    with pytest.raises(SystemExit, match="z-score uses mean/variance from future"):
        main(
            [
                "train",
                str(tmp_path / "not-needed-for-normalization-check.npz"),
                "--objective",
                "step_bce",
                "--propagation-mode",
                "receiver",
                "--output",
                str(tmp_path / "run-2"),
            ]
        )

    with pytest.raises(SystemExit, match="raw query-key attention"):
        main(
            [
                "train",
                str(tmp_path / "not-needed-for-pairwise-scale-check.npz"),
                "--objective",
                "response_bce",
                "--message-operator",
                "pairwise",
                "--propagation-mode",
                "receiver",
                "--output",
                str(tmp_path / "run-3"),
            ]
        )


def test_split_protocol_never_discards_or_resplits_official_partitions():
    from argparse import Namespace
    from types import SimpleNamespace

    from hypergraph.attention.train import _explicit_split, _group_cv_split

    traces = [
        SimpleNamespace(trace_id="a", group_id="a", split="train"),
        SimpleNamespace(trace_id="b", group_id="b", split="validation"),
        SimpleNamespace(trace_id="c", group_id="c", split="test"),
        SimpleNamespace(trace_id="d", group_id="d", split="other"),
    ]
    explicit_args = Namespace(
        train_split="train", val_split="validation", test_split="test"
    )
    with pytest.raises(SystemExit, match="silently discard"):
        _explicit_split(traces, explicit_args)

    duplicate_args = Namespace(
        train_split="train", val_split="train", test_split="test"
    )
    with pytest.raises(SystemExit, match="distinct"):
        _explicit_split(traces[:3], duplicate_args)

    cv_args = Namespace(allow_resplit_official_data=False)
    with pytest.raises(SystemExit, match="official split metadata"):
        _group_cv_split(traces[:3], cv_args)


def test_fixed_holdout_is_deterministic_group_disjoint_and_single_test():
    from argparse import Namespace

    from hypergraph.attention.train import _TraceMeta, _fixed_holdout_split

    traces = [
        _TraceMeta(
            trace_id=f"trace-{index}",
            group_id=f"problem-{index // 2}",
            group_is_fallback=False,
            split=None,
            response_label=(index // 2) % 2,
            gold_step=(-1 if (index // 2) % 2 == 0 else (index // 7) % 4),
            num_steps=4,
            num_response_tokens=20 + (index % 17),
            generator_model="model",
        )
        for index in range(120)
    ]
    args = Namespace(
        allow_resplit_official_data=False,
        allow_trace_as_group=False,
        split_seed=41,
        val_ratio=0.1,
        test_ratio=0.2,
    )

    first = _fixed_holdout_split(traces, args)
    second = _fixed_holdout_split(traces, args)
    train, val, test, info = first

    assert first == second
    assert info["mode"] == "fixed_holdout"
    assert info["split_seed"] == 41
    assert set(info["partition_trace_ids"]) == {"train", "validation", "test"}
    assert set(info["partition_group_ids"]) == {"train", "validation", "test"}
    manifest_groups = [
        set(info["partition_group_ids"][name])
        for name in ("train", "validation", "test")
    ]
    assert not (manifest_groups[0] & manifest_groups[1])
    assert not (manifest_groups[0] & manifest_groups[2])
    assert not (manifest_groups[1] & manifest_groups[2])
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert sorted(train + val + test) == list(range(len(traces)))
    for left, right in ((train, val), (train, test), (val, test)):
        assert {
            traces[index].group_id for index in left
        }.isdisjoint({traces[index].group_id for index in right})
    assert 0.05 <= len(val) / len(traces) <= 0.15
    assert 0.15 <= len(test) / len(traces) <= 0.25
    for partition in (train, val, test):
        assert {traces[index].response_label for index in partition} == {0, 1}


def test_fixed_holdout_aggregation_reads_one_test_and_never_scans_folds(tmp_path):
    import json

    from hypergraph.attention.aggregate_fixed import aggregate_fixed_run

    root = tmp_path / "experiment"
    run = root / "fixed_seed17"
    run.mkdir(parents=True)
    split = {
        "mode": "fixed_holdout",
        "split_seed": 17,
        "partition_group_ids": {
            "train": ["p0"],
            "validation": ["p1"],
            "test": ["p2"],
        },
    }
    (run / "results.json").write_text(
        json.dumps(
            {
                "best_epoch": 3,
                "validation_monitor": {"name": "aupr", "value": 0.7},
                "partition_sizes": {"train": 7, "val": 1, "test": 2},
                "metrics": {
                    "test": {
                        "n": 2,
                        "positives": 1,
                        "auroc": 0.75,
                        "aupr": 0.8,
                        "accuracy_0.5": 0.5,
                    }
                },
                "trace_detection_by_generator": {"test": {"model": {"n": 2}}},
                "resolved": {"split": split},
            }
        ),
        encoding="utf-8",
    )
    (run / "predictions_test.csv").write_text(
        "trace_id,label,probability\na,0,0.2\nb,1,0.8\n", encoding="utf-8"
    )

    summary = aggregate_fixed_run(root, run, write_outputs=True)

    assert summary["schema"] == "fixed_holdout_response_test_v1"
    assert summary["final_test"]["auroc"] == 0.75
    assert (root / "aggregate_results.json").is_file()
    assert (root / "predictions_test.csv").is_file()
    assert json.loads((root / "split_manifest.json").read_text())["mode"] == "fixed_holdout"

    (run / "predictions_test.csv").unlink()
    with pytest.raises(FileNotFoundError, match="fixed held-out prediction"):
        aggregate_fixed_run(root, run, write_outputs=False)


def test_json_config_is_strict_and_never_overrides_explicit_cli(tmp_path):
    import json

    from hypergraph.attention.train import _load_config, build_parser

    parser = build_parser()
    config_path = tmp_path / "config.json"

    def parsed(document, *extra):
        config_path.write_text(json.dumps(document), encoding="utf-8")
        argv = [
            "train",
            "trace.npz",
            "--config",
            str(config_path),
            "--objective",
            "response_bce",
            "--output",
            str(tmp_path / "run"),
            *extra,
        ]
        args = parser.parse_args(argv)
        _load_config(args, argv, parser)
        return args

    args = parsed(
        {
            "threshold": 0.9,
            "propagation_mode": "symmetric",
            "allow_offline_full_context": False,
            "include_center": True,
        },
        "--threshold=0.05",
        "--propagation-mode=receiver",
        "--allow-offline-symmetric-step",
        "--no-include-center",
    )
    assert args.threshold == 0.05
    assert args.propagation_mode == "receiver"
    assert args.allow_offline_full_context is True
    assert args.include_center is False

    with pytest.raises(SystemExit, match="exact JSON integer"):
        parsed({"top_k": 1.5})
    with pytest.raises(SystemExit, match="JSON boolean"):
        parsed({"include_center": "false"})
    with pytest.raises(SystemExit, match="must be one of"):
        parsed({"scheduler": "typo"})
    with pytest.raises(SystemExit, match="protected"):
        parsed({"inputs": ["other-dataset.npz"]})
    with pytest.raises(SystemExit, match="duplicates"):
        parsed({"graph": {"top-k": 2}, "top_k": 3})

    config_path.write_text('{"top_k": 2, "top_k": 3}', encoding="utf-8")
    duplicate_argv = [
        "train",
        "trace.npz",
        "--config",
        str(config_path),
        "--objective",
        "response_bce",
        "--output",
        str(tmp_path / "run-duplicate"),
    ]
    duplicate_args = parser.parse_args(duplicate_argv)
    with pytest.raises(SystemExit, match="duplicate JSON key"):
        _load_config(duplicate_args, duplicate_argv, parser)


def test_localization_rank_does_not_use_gold_to_prune_later_steps():
    import json

    from hypergraph.attention.train import (
        _first_crossing_metrics,
        _select_first_crossing_threshold,
        _tie_aware_localization_rank,
    )

    rank, top1 = _tie_aware_localization_rank(
        [0.1, 0.8, 0.9], gold_step=1, valid_mask=[True, True, True]
    )
    assert rank == 2.0
    assert top1 == 0.0

    rows = [
        {
            "gold_step": -1,
            "step_probabilities_json": json.dumps([0.1, 0.2]),
            "valid_steps_json": json.dumps([1, 1]),
        },
        {
            "gold_step": 1,
            "step_probabilities_json": json.dumps([0.2, 0.8, 0.9]),
            "valid_steps_json": json.dumps([1, 1, 1]),
        },
    ]
    threshold, validation = _select_first_crossing_threshold(rows)
    evaluated = _first_crossing_metrics(rows, threshold)
    assert validation == evaluated
    assert evaluated["first_error_exact_accuracy"] == 1.0
    assert evaluated["correct_trace_false_alarm_rate"] == 0.0


def test_objectives_keep_token_step_and_response_granularity_separate():
    torch = pytest.importorskip("torch")
    from hypergraph.attention.objectives import (
        make_first_error_targets,
        pool_token_logits_to_response,
        pool_token_logits_to_steps,
        step_bce,
        token_bce,
    )

    logits = torch.tensor([-3.0, -2.0, -1.0, 0.0, 2.0])
    steps = pool_token_logits_to_steps(logits, [[3, 4], [4, 5]])
    response = pool_token_logits_to_response(logits, 3)
    torch.testing.assert_close(steps, torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(response, torch.tensor(1.0))

    targets, risk = make_first_error_targets(2, 1)
    assert step_bce(steps, targets, risk_mask=risk).ndim == 0
    with pytest.raises(TypeError):
        token_bce(logits, torch.zeros(5))


def test_cached_query_chunks_reconstruct_the_full_causal_attention_axis():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    class DeterministicModel:
        def __call__(
            self,
            *,
            input_ids,
            attention_mask,
            output_attentions,
            output_hidden_states,
            use_cache,
            return_dict,
            past_key_values=None,
        ):
            del output_attentions, output_hidden_states, return_dict, past_key_values
            query_length = int(input_ids.shape[1])
            key_length = int(attention_mask.shape[1])
            start = key_length - query_length
            blocks = []
            for layer in range(2):
                block = torch.zeros((1, 2, query_length, key_length))
                for query in range(query_length):
                    absolute_query = start + query
                    weights = torch.arange(1, absolute_query + 2, dtype=torch.float32)
                    weights = weights / weights.sum()
                    block[0, :, query, : absolute_query + 1] = weights
                blocks.append(block * (1.0 + layer * 0.1))
            positions = torch.arange(start, key_length, dtype=torch.float32)
            hidden = torch.stack([positions, positions + 1, positions + 2], dim=-1)[None]
            return SimpleNamespace(
                attentions=tuple(blocks),
                hidden_states=(hidden, hidden + 10, hidden + 20),
                past_key_values=key_length if use_cache else None,
            )

    tokenized = {
        "token_ids": np.arange(6, dtype=np.int64),
        "attention_mask": np.ones(6, dtype=np.int64),
    }
    kwargs = {
        "model": DeterministicModel(),
        "torch": torch,
        "tokenized": tokenized,
        "device": "cpu",
        "attention_layers": (0, 1),
        "attention_heads": (0, 1),
        "activation_layer": 2,
        "storage_dtype": "float32",
    }
    full = extract_trace(**kwargs, query_chunk_size=0)
    chunked = extract_trace(**kwargs, query_chunk_size=2)
    np.testing.assert_allclose(chunked["attention"], full["attention"], atol=0, rtol=0)
    np.testing.assert_allclose(chunked["activation"], full["activation"], atol=0, rtol=0)
    gate = verify_chunked_equivalence(
        kwargs["model"],
        torch,
        tokenized,
        device="cpu",
        attention_layers=(0, 1),
        attention_heads=(0, 1),
        query_chunk_size=2,
        activation_layer=2,
        prefix_tokens=4,
        atol=0.0,
        topology_threshold=0.01,
    )
    assert gate["status"] == "prefix_pass"
    assert gate["topology_disagreements"] == 0
    assert gate["activation_max_abs_error"] == 0


def test_faithful_and_directed_message_passing_are_separate_switches():
    torch = pytest.importorskip("torch")
    from hypergraph.attention.model import HyperCharmLayer

    graph = {
        "he_index": torch.tensor([[0, 1, 2], [0, 0, 0]]),
        "he_attr": torch.zeros((1, 3)),
        "he_mark": torch.tensor([[1.0, 0.0]]),
        "he_weight": torch.ones(3),
        "he_attention": torch.tensor([0.2, 0.3, 0.5]),
        "he_receiver": torch.tensor([2]),
    }
    x = torch.randn(3, 4)
    torch.manual_seed(0)
    symmetric = HyperCharmLayer(4, 3, 8, directed_receiver_only=False).eval()
    assert any(isinstance(module, torch.nn.LayerNorm) for module in symmetric.node2edge)
    torch.manual_seed(0)
    directed = HyperCharmLayer(4, 3, 8, directed_receiver_only=True).eval()
    directed.load_state_dict(symmetric.state_dict())
    with torch.no_grad():
        sym = symmetric(x=x, **graph)
        rec = directed(x=x, **graph)

    # Receiver-only propagation leaves non-receivers at their residual values;
    # the faithful symmetric baseline updates all incidence members.
    torch.testing.assert_close(rec[:2], x[:2])
    assert not torch.allclose(sym[:2], x[:2])

    with pytest.raises(ValueError, match="requires receiver-only"):
        HyperCharmLayer(4, 3, 8, message_operator="pairwise")
    pairwise = HyperCharmLayer(
        4,
        3,
        8,
        directed_receiver_only=True,
        message_operator="pairwise",
    ).eval()
    pairwise.load_state_dict(directed.state_dict())
    with torch.no_grad():
        pair = pairwise(x=x, **graph)
    assert not torch.allclose(pair[2], rec[2])


def test_step_supervision_mask_cannot_prune_deployable_evaluation_candidates():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from hypergraph.attention.train import _TraceMeta, _evaluate

    class FixedModel:
        def eval(self):
            return self

        def __call__(self, graph):
            del graph
            return torch.tensor([0.0, 1.0, 10.0])

    graph = SimpleNamespace(
        step_ranges=np.asarray([[0, 1], [1, 2], [2, 3]], np.int64),
        gold_step=1,
        step_loss_mask=np.asarray([True, True, False]),
    )
    trace = _TraceMeta(
        trace_id="masked",
        group_id="g",
        group_is_fallback=False,
        split="test",
        response_label=1,
        gold_step=1,
        num_steps=3,
        num_response_tokens=3,
    )
    _, rows = _evaluate(
        FixedModel(),
        [graph],
        [trace],
        [0],
        "step_bce",
        pooling="mean",
        temperature=1.0,
    )
    row = rows[0]
    assert row["n_scored"] == 2
    assert row["n_localization_candidates"] == 3
    assert row["score"] == pytest.approx(float(torch.sigmoid(torch.tensor(10.0))))
    assert row["valid_steps_json"] == "[1,1,1]"
    assert row["supervision_steps_json"] == "[1,1,0]"


def test_model_rejects_graph_propagation_metadata_mismatch():
    pytest.importorskip("torch")
    from hypergraph.attention.model import HyperCHARMToken

    attention, token_ids, response_idx = _trace()
    graph = build_attention_hypergraph(
        attention,
        token_ids,
        response_idx,
        config=AttentionHypergraphConfig(propagation_mode="receiver"),
    )
    model = HyperCHARMToken(
        node_dim=graph.x.shape[1],
        hedge_dim=graph.he_attr.shape[1],
        hidden_dim=8,
        num_layers=1,
        directed_receiver_only=False,
    )
    with pytest.raises(ValueError, match="propagation does not match"):
        model(graph)
