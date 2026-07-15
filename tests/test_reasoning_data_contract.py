from __future__ import annotations

import json

import numpy as np

from prompt_control_flow.data_contract import inspect_reasoning_artifact


def _objects(values):
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out


def _exact_trace_payload(n: int = 2) -> dict[str, np.ndarray]:
    return {
        "trace_schema_version": np.asarray("exact_generation_trace_v1"),
        "trace_token_add_special_tokens": np.asarray(False),
        "token_offset_convention": np.asarray("char_half_open"),
        "step_token_range_convention": np.asarray("inclusive"),
        "span_range_convention": np.asarray("half_open"),
        "prompts": _objects(["prompt"] * n),
        "responses": _objects(["answer"] * n),
        "steps_text": _objects([["answer"]] * n),
        "input_ids": _objects([np.asarray([1, 2, 3])] * n),
        "attention_mask": _objects([np.asarray([1, 1, 1])] * n),
        "token_offsets": _objects([np.asarray([(0, 1), (1, 2), (2, 3)])] * n),
        "step_token_ranges": _objects([np.asarray([(2, 2)])] * n),
        "response_token_ranges": np.asarray([(2, 3)] * n),
    }


def test_exact_self_generation_trace_is_not_confused_with_observer_trace(tmp_path):
    path = tmp_path / "self_trace.npz"
    payload = _exact_trace_payload()
    payload.update(
        problem_ids=np.asarray([7, 7]),
        sample_idx=np.asarray([0, 1]),
        final_answer_correct=np.asarray([1, 0]),
        process_correct=np.asarray([-1, -1]),
        trace_semantics=np.asarray("generation_matched_teacher_forcing"),
        generated_token_ids_stored=np.asarray(True),
        generated_token_ids=_objects([np.asarray([3]), np.asarray([3])]),
        sv_clouds=_objects(
            [np.zeros((1, 1, 4), dtype=np.float16)] * 2
        ),
        cloud_sizes=_objects([np.asarray([1]), np.asarray([1])]),
        cloud_layers=np.asarray([16]),
    )
    np.savez(path, **payload)

    report = inspect_reasoning_artifact(path)
    assert report["evidence_tier"] == "generation_matched_self_trace"
    assert report["capabilities"]["generation_matched_self_state"] is True
    assert report["capabilities"]["exact_observer_teacher_forcing"] is False
    assert report["capabilities"]["confirmatory_causal_intervention"] is True
    assert report["process_label_counts"]["known"] == 0
    assert report["same_problem"]["contrastive_problems"] == 1


def test_processbench_observer_trace_keeps_process_and_answer_labels_distinct(tmp_path):
    path = tmp_path / "observer_trace.npz"
    payload = _exact_trace_payload()
    payload.update(
        problem_ids=np.asarray([1, 2]),
        sample_idx=np.asarray([0, 0]),
        gold_error_step=np.asarray([0, -1]),
        process_correct=np.asarray([0, 1]),
        final_answer_correct=np.asarray([1, 1]),
        trace_semantics=np.asarray("benchmark_observer_teacher_forcing"),
    )
    np.savez(path, **payload)

    report = inspect_reasoning_artifact(path)
    assert report["evidence_tier"] == "exact_benchmark_observer_trace"
    assert report["process_label_counts"]["error"] == 1
    assert report["final_answer_label_counts"]["error"] == 0
    assert report["capabilities"]["benchmark_first_error_diagnosis"] is True
    assert report["capabilities"]["generation_matched_self_state"] is False


def test_legacy_same_problem_hidden_is_exploratory_not_exact(tmp_path):
    path = tmp_path / "legacy_same_problem.npz"
    np.savez(
        path,
        problem_ids=np.asarray([4, 4]),
        sample_idx=np.asarray([0, 1]),
        is_correct=np.asarray([1, 0]),
        responses=_objects(["a", "b"]),
        steps_text=_objects([["a"], ["b"]]),
        sv_clouds=_objects([np.zeros((1, 1, 4), dtype=np.float16)] * 2),
        cloud_sizes=_objects([np.asarray([1]), np.asarray([1])]),
        cloud_layers=np.asarray([16]),
    )

    report = inspect_reasoning_artifact(path)
    assert report["evidence_tier"] == "legacy_same_problem_trace"
    assert report["exact_trace_complete"] is False
    assert report["raw_response_token_hidden_stored"] is True
    assert report["capabilities"]["confirmatory_causal_intervention"] is False


def test_processbench_source_audit_catches_bad_rows_and_label_mismatch(tmp_path):
    path = tmp_path / "gsm8k.jsonl"
    good = {
        "id": "x",
        "generator": "model",
        "problem": "q",
        "steps": ["wrong", "right answer"],
        "final_answer_correct": True,
        "label": 0,
    }
    path.write_text(json.dumps(good) + "\n{bad json\n", encoding="utf-8")

    report = inspect_reasoning_artifact(path)
    assert report["ready"] is False
    assert report["valid_records"] == 1
    assert len(report["malformed_records"]) == 1
    assert report["process_error_with_correct_final_answer"] == 1
    assert report["original_generation_prompt_available"] is False


def test_processbench_json_list_is_supported(tmp_path):
    path = tmp_path / "subset.json"
    row = {
        "id": "x",
        "generator": "model",
        "problem": "q",
        "steps": ["correct"],
        "final_answer_correct": True,
        "label": -1,
    }
    path.write_text(json.dumps([row]), encoding="utf-8")
    report = inspect_reasoning_artifact(path)
    assert report["ready"] is True
    assert report["source_records"] == 1
    assert report["process_correct"] == 1
