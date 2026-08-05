from __future__ import annotations

import json

import numpy as np
import pytest

from functional_divergence.causal_first_error_attribution.contracts import OnsetPair
from functional_divergence.causal_first_error_attribution.pairs import (
    first_divergence,
    load_pre_decision_prefix,
)


def _target_pair() -> dict[str, object]:
    return {
        "schema_version": "onset_pair_v1",
        "case_id": "gsm8k-7",
        "dataset": "gsm8k",
        "problem_hash": "problem_sha256:abc",
        "error_chain_id": 7,
        "error_trace_record": 3,
        "first_error_step": 1,
        "error_step_token_start": 12,
        "error_step_token_end": 18,
        "pair_kind": "target_correction",
        "correctness_verifier": "human",
        "verification_evidence": "checked correction",
        "corrected_step_text": "12 + 8 = 20",
        "first_divergent_token_index": 14,
        "wrong_token_id": 91,
        "correct_token_id": 92,
        "same_prefix_before_divergence": True,
        "generation_config_status": "stored",
    }


def test_same_prefix_minimal_correction_is_not_a_root_cause_donor() -> None:
    pair = OnsetPair.from_mapping(_target_pair())

    assert pair.decision_position == 13
    assert pair.root_cause_eligible is False
    assert pair.allowed_claim == "target_reference_only"
    with pytest.raises(ValueError, match="controlled_root"):
        pair.require_root_cause_eligibility()


def test_first_divergence_uses_only_prefix_before_target() -> None:
    error = np.asarray([10, 11, 12, 13, 90, 91], dtype=np.int64)
    corrected = np.asarray([10, 11, 12, 13, 92, 93], dtype=np.int64)
    changed_after_divergence = np.asarray([10, 11, 12, 13, 92, 777], dtype=np.int64)

    divergence, decision = first_divergence(error, corrected, step_start=2)
    changed_divergence, changed_decision = first_divergence(
        error, changed_after_divergence, step_start=2
    )

    assert (divergence, decision) == (4, 3)
    assert (changed_divergence, changed_decision) == (4, 3)
    np.testing.assert_array_equal(
        load_pre_decision_prefix(error, decision), np.asarray([10, 11, 12, 13])
    )


def test_pair_loader_rejects_unverified_correction(tmp_path) -> None:
    payload = _target_pair()
    payload["verification_evidence"] = ""
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="verification_evidence"):
        OnsetPair.from_mapping(payload)

