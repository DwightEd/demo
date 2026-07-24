from __future__ import annotations

from pathlib import Path

from functional_divergence.hidden_state_geometry.console import (
    format_preflight_summary,
    format_run_summary,
)


def _result(point: float, ci_low: float, ci_high: float) -> dict:
    return {
        "run_id": "20260724T160000Z_deadbeef",
        "execution": {"max_records_per_domain": 32, "bootstrap_replicates": 200},
        "method": {"name": "full_tensor_ridge"},
        "data": {"records": 128, "domains": ["gsm8k", "math"]},
        "tasks": {
            "whole_chain": {
                "rows": 128,
                "events": 89,
                "summary": {
                    "arms": {
                        "hidden_only": {
                            "macro": {"auroc": 0.6977, "auprc": 0.8456, "nll_nats": 0.6003}
                        }
                    },
                    "increments": {
                        "hidden_given_output_summary_nll": {
                            "point": point,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "inference_scope": "conditional_test_problem_group_cluster_bootstrap",
                        }
                    },
                },
            }
        },
    }


def test_run_summary_reports_evaluated_domain_better_increment_without_diagnostics():
    text = format_run_summary(_result(0.04, 0.01, 0.08), "/tmp/output")

    assert f"output_dir: {Path('/tmp/output')}" in text
    assert "run_id: 20260724T160000Z_deadbeef" in text
    assert "records: 128 | domains: gsm8k, math | method: full_tensor_ridge | cap: 32/domain" in text
    assert "whole_chain: rows=128 | events=89" in text
    assert "bootstrap_replicates: 200" in text
    assert "domain-macro metrics (problem-group balanced):" in text
    assert "hidden_only: AUROC=0.6977 | AUPRC=0.8456 | NLL=0.6003" in text
    assert "increment inference: conditional test problem-group bootstrap" in text
    assert "hidden_given_output_summary_nll: +0.0400 [95% CI +0.0100, +0.0800] | better_on_evaluated_domains" in text
    assert "conditional bootstrap CI entirely > 0" in text
    assert "positive = candidate model lower NLL (better)" in text
    assert "optimizer" not in text
    assert "iterations" not in text
    assert "gradient_inf_norm" not in text
    assert "fold_diagnostics" not in text


def test_run_summary_marks_negative_nonoverlapping_increment_worse_on_evaluated_domains():
    text = format_run_summary(_result(-0.04, -0.08, -0.01), "/tmp/output")

    assert "[95% CI -0.0800, -0.0100] | worse_on_evaluated_domains" in text
    assert "candidate model worse" in text


def test_run_summary_marks_interval_crossing_zero_uncertain():
    text = format_run_summary(_result(0.02, -0.01, 0.05), "/tmp/output")

    assert "[95% CI -0.0100, +0.0500] | uncertain" in text
    assert "CI includes 0" in text


def test_preflight_summary_reports_domain_provenance_without_json_blob():
    text = format_preflight_summary(
        {
            "run_id": "20260724T160000Z_deadbeef",
            "shards_validated": 128,
            "domains": [
                {
                    "dataset": "gsm8k",
                    "selected_records": 32,
                    "error_records": 21,
                    "correct_records": 11,
                    "layers": [0, 1, 2],
                    "hidden_dimension": 4096,
                    "response_generators": ["llama3.1-8b"],
                    "observer_models": ["llama3.1-8b"],
                }
            ],
        }
    )

    assert "preflight: run_id=20260724T160000Z_deadbeef | validated shards=128" in text
    assert "gsm8k: selected=32 | error=21 | correct=11 | layers=3 | hidden_dim=4096" in text
    assert "generator=llama3.1-8b | observer=llama3.1-8b" in text
    assert "{" not in text
