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

