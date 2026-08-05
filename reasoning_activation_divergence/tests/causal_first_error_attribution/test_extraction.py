from __future__ import annotations

import json

import numpy as np

from functional_divergence.causal_first_error_attribution import extraction


def test_controlled_pair_extracts_recipient_and_counterfactual_graphs(
    tmp_path, monkeypatch
) -> None:
    selected = tmp_path / "controlled_math" / "selected"
    pair_dir = selected / "causal_first_error_v1"
    pair_dir.mkdir(parents=True)
    np.savez_compressed(
        selected / "trace.npz",
        full_input_ids=np.asarray([[10, 11, 20], [10, 12, 21]]),
        full_attention_mask=np.ones((2, 3), dtype=np.int8),
        prompt_token_counts=np.asarray([1, 1]),
        step_token_ranges=np.asarray([[[1, 2]], [[1, 2]]]),
        n_steps=np.asarray([1, 1]),
        dataset=np.asarray(["controlled_math", "controlled_math"]),
    )
    pair = {
        "schema_version": "onset_pair_v1",
        "case_id": "controlled-1",
        "dataset": "controlled_math",
        "problem_hash": "template:sum",
        "error_chain_id": "condition-a",
        "error_trace_record": 0,
        "first_error_step": 0,
        "error_step_token_start": 1,
        "error_step_token_end": 3,
        "pair_kind": "controlled_root",
        "correctness_verifier": "python_exact",
        "verification_evidence": "exact targets",
        "decision_position": 1,
        "counterfactual_decision_position": 1,
        "counterfactual_trace_record": 1,
        "template_id": "arithmetic-op",
        "condition_id": "sum",
        "counterfactual_condition_id": "product",
        "intervention_variable": "operator",
        "donor_alignment": {"source": [0, 1], "target": [0, 1]},
        "target_token_ids": [20],
        "counterfactual_target_token_ids": [21],
    }
    pair_path = pair_dir / "onset_pairs_v1.jsonl"
    pair_path.write_text(json.dumps(pair) + "\n", encoding="utf-8")
    calls = []

    class FakeArtifact:
        def save(self, path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("trace", encoding="utf-8")

    class FakeExtractor:
        def __init__(self, **_kwargs) -> None:
            pass

        def extract(self, **kwargs):
            calls.append(kwargs)
            return FakeArtifact()

    monkeypatch.setattr(extraction, "DecisionTraceExtractor", FakeExtractor)
    real_sha256 = extraction.file_sha256
    fingerprinted = []

    def counting_sha256(path):
        fingerprinted.append(path)
        return real_sha256(path)

    monkeypatch.setattr(extraction, "file_sha256", counting_sha256)
    config = extraction.OnsetTraceExtractionConfig(
        data_root=tmp_path,
        domains=("controlled_math",),
        layers=(1,),
        model_name="fake",
        model_revision="test",
        tokenizer_name="fake",
        tokenizer_revision="test",
    )

    report = extraction.OnsetTraceExtraction(config).run(object())

    assert report["selected_pairs"] == 1
    assert report["selected_traces"] == 2
    assert {call["metadata"]["trajectory_role"] for call in calls} == {
        "recipient",
        "counterfactual",
    }
    assert {tuple(call["input_ids"]) for call in calls} == {
        (10, 11),
        (10, 12),
    }
    names = {path.rsplit("/", 1)[-1] for path in report["written"]}
    assert names == {
        "case_controlled-1.recipient.onset_trace_v1.npz",
        "case_controlled-1.counterfactual.onset_trace_v1.npz",
    }
    assert fingerprinted.count(selected / "trace.npz") == 1
    assert fingerprinted.count(pair_path) == 1
