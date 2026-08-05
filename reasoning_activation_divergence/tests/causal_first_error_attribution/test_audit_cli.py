from __future__ import annotations

import json

import numpy as np

from functional_divergence.causal_first_error_attribution.audit import PairAuditor
from functional_divergence.causal_first_error_attribution.main import main


def _write_trace(path) -> None:
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        gold_error_step=np.asarray([2, -1, 0], dtype=np.int32),
        problem_ids=np.asarray(
            ["problem_sha256:a", "problem_sha256:b", "problem_sha256:c"],
            dtype=object,
        ),
    )


def test_processbench_audit_reports_root_cause_unidentifiable_without_pairs(
    tmp_path,
) -> None:
    _write_trace(tmp_path / "gsm8k" / "selected" / "trace.npz")

    report = PairAuditor(tmp_path, ("gsm8k",)).run()

    assert report["natural_chains"] == 3
    assert report["error_chains"] == 2
    assert report["correct_chains"] == 1
    assert report["valid_target_correction_pairs"] == 0
    assert report["valid_controlled_root_pairs"] == 0
    assert report["root_cause_claim"] == "not_identifiable_from_available_data"
    assert report["model_required"] is False
    assert report["next_required_artifact"].endswith(
        "gsm8k/selected/causal_first_error_v1/onset_pairs_v1.jsonl"
    )


def test_audit_cli_finishes_without_loading_a_model(tmp_path, capsys) -> None:
    _write_trace(tmp_path / "gsm8k" / "selected" / "trace.npz")
    output = tmp_path / "audit.json"

    main(
        [
            "audit",
            "--data-root",
            str(tmp_path),
            "--domains",
            "gsm8k",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    text = capsys.readouterr().out
    assert report["root_cause_claim"] == "not_identifiable_from_available_data"
    assert "root-cause analysis: not identifiable" in text
    assert "next required artifact:" in text


def test_audit_accepts_a_controlled_trace_without_processbench_labels(tmp_path) -> None:
    selected = tmp_path / "controlled_math" / "selected"
    selected.mkdir(parents=True)
    np.savez_compressed(
        selected / "trace.npz",
        full_input_ids=np.asarray([[1, 2, 3], [1, 2, 4]]),
    )
    pair_dir = selected / "causal_first_error_v1"
    pair_dir.mkdir()
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
        "target_token_ids": [3],
        "counterfactual_target_token_ids": [4],
    }
    (pair_dir / "onset_pairs_v1.jsonl").write_text(
        json.dumps(pair) + "\n", encoding="utf-8"
    )

    report = PairAuditor(tmp_path, ("controlled_math",)).run()

    assert report["natural_chains"] == 0
    assert report["valid_controlled_root_pairs"] == 1
    assert report["root_cause_claim"] == "controlled_pairs_available"
