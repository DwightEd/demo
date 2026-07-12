from __future__ import annotations

from prompt_control_flow.teacher_forcing import prepare_teacher_forcing_trace
from prompt_control_flow.data import load_chain_records
from prompt_control_flow.cli.extract_mechanisms import model_identity_matches
import numpy as np
import json
import pytest
from utils.step_boundaries import TokenAlignmentError


class BosMockTokenizer:
    """Adds a fake BOS only when a caller incorrectly requests specials."""

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False, **_):
        ids = [ord(ch) + 5 for ch in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        if add_special_tokens:
            ids = [999] + ids
            offsets = [(0, 0)] + offsets
        out = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def test_fallback_alignment_never_introduces_bos_axis_shift() -> None:
    prompt = "P:"
    response = "first\n\nsecond"
    trace = prepare_teacher_forcing_trace(
        BosMockTokenizer(),
        prompt,
        response,
        steps=["first", "second"],
        max_seq_len=100,
    )
    assert trace["input_ids"][0] != 999
    assert trace["response_start_token"] == len(prompt)
    assert trace["step_token_ranges"] == [
        (len(prompt), len(prompt) + len("first") - 1),
        (len(prompt) + len("first\n\n"), len(prompt + response) - 1),
    ]


def test_exact_artifact_axis_is_replayed_without_calling_tokenizer() -> None:
    class ForbiddenTokenizer:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("exact token artifacts must not be re-tokenized")

    trace = prepare_teacher_forcing_trace(
        ForbiddenTokenizer(),
        "ignored prompt",
        "ignored response",
        steps=["ignored-1", "ignored-2"],
        max_seq_len=6,
        exact_input_ids=[101, 102, 201, 202, 203, 204],
        exact_attention_mask=[1, 1, 1, 1, 1, 1],
        exact_token_offsets=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        exact_step_token_ranges=[(2, 3), (4, 5)],
        exact_response_start_token=2,
    )
    assert trace["input_ids"] == [101, 102, 201, 202, 203, 204]
    assert trace["step_token_ranges"] == [(2, 3), (4, 5)]
    assert trace["response_start_token"] == 2


def test_max_seq_len_cannot_silently_drop_a_reasoning_step() -> None:
    class ForbiddenTokenizer:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("exact token artifacts must not be re-tokenized")

    with pytest.raises(TokenAlignmentError, match="refusing to change the chain/step axis"):
        prepare_teacher_forcing_trace(
            ForbiddenTokenizer(),
            "ignored prompt",
            "ignored response",
            steps=["ignored-1", "ignored-2"],
            max_seq_len=5,
            exact_input_ids=[101, 102, 201, 202, 203, 204],
            exact_attention_mask=[1, 1, 1, 1, 1, 1],
            exact_token_offsets=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
            exact_step_token_ranges=[(2, 3), (4, 5)],
            exact_response_start_token=2,
        )


def test_exact_npz_loader_accepts_scalar_model_metadata_and_kept_axis(tmp_path) -> None:
    path = tmp_path / "exact_sample.npz"
    obj = lambda value: np.asarray([np.asarray(value)], dtype=object)
    np.savez(
        path,
        trace_schema_version=np.asarray("exact_generation_trace_v1"),
        trace_token_add_special_tokens=np.asarray(False),
        token_offset_convention=np.asarray("char_half_open"),
        step_token_range_convention=np.asarray("inclusive"),
        span_range_convention=np.asarray("half_open"),
        problems=np.asarray(["q"], dtype=object),
        problem_ids=np.asarray([7]),
        sample_idx=np.asarray([0]),
        responses=np.asarray(["a\n\nb\n\nc"], dtype=object),
        steps_text=np.asarray([["a", "b", "c"]], dtype=object),
        kept_steps=obj([0, 2]),
        is_correct=np.asarray([1]),
        prompts=np.asarray(["prompt"], dtype=object),
        input_ids=obj([10, 11, 12, 13, 14]),
        attention_mask=obj([1, 1, 1, 1, 1]),
        token_offsets=np.asarray([np.asarray([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])], dtype=object),
        step_token_ranges=np.asarray([np.asarray([(2, 2), (4, 4)])], dtype=object),
        response_token_ranges=np.asarray([(2, 5)]),
        model_name=np.asarray("model-x"),
        dataset=np.asarray("internal:toy"),
        model_sampling_metadata_json=np.asarray(
            json.dumps({"tokenizer_name": "org/model-x", "model_revision": "abc"})
        ),
    )
    rows = load_chain_records(path)
    assert len(rows) == 1
    assert rows[0].steps == ["a", "c"]
    assert rows[0].generator == "model-x"
    assert rows[0].dataset == "internal:toy"
    assert rows[0].exact_input_ids == [10, 11, 12, 13, 14]
    assert rows[0].exact_step_token_ranges == [(2, 2), (4, 4)]
    assert rows[0].source_tokenizer == "org/model-x"
    assert rows[0].source_model_revision == "abc"


def test_partial_exact_npz_fails_closed_instead_of_retokenizing(tmp_path) -> None:
    path = tmp_path / "partial_exact.npz"
    np.savez(
        path,
        trace_schema_version=np.asarray("exact_generation_trace_v1"),
        steps_text=np.asarray([["s"]], dtype=object),
        responses=np.asarray(["s"], dtype=object),
        prompts=np.asarray(["p"], dtype=object),
        input_ids=np.asarray([np.asarray([1, 2])], dtype=object),
    )
    with pytest.raises(TokenAlignmentError, match="exact trace is incomplete"):
        load_chain_records(path)


def test_exact_model_identity_guard_accepts_path_alias_only() -> None:
    assert model_identity_matches("meta-llama/Llama-3.1-8B-Instruct", "D:/models/Llama-3.1-8B-Instruct")
    assert not model_identity_matches("org/model-a", "org/model-b")
    assert not model_identity_matches("org-a/model-x", "org-b/model-x")
