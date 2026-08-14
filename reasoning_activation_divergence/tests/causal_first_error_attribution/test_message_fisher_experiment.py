from __future__ import annotations

import numpy as np

from functional_divergence.causal_first_error_attribution.message_fisher_experiment import (
    first_error_boundary_pairs,
    paired_fisher_summary,
)


def test_first_error_pairs_use_only_consecutive_future_free_boundaries(tmp_path) -> None:
    trace = tmp_path / "trace.npz"
    np.savez_compressed(
        trace,
        full_input_ids=np.asarray(
            [
                [10, 11, 20, 21, 30, 31, 0],
                [10, 12, 40, 41, 0, 0, 0],
                [10, 13, 50, 51, 60, 61, 0],
            ]
        ),
        full_attention_mask=np.asarray(
            [
                [1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
            ],
            dtype=np.int8,
        ),
        prompt_token_counts=np.asarray([2, 2, 2]),
        step_token_ranges=np.asarray(
            [
                [[2, 3], [4, 5]],
                [[2, 3], [-1, -1]],
                [[2, 3], [4, 5]],
            ]
        ),
        n_steps=np.asarray([2, 1, 2]),
        gold_error_step=np.asarray([1, 0, -1]),
        chain_idx=np.asarray([101, 102, 103]),
    )

    pairs = first_error_boundary_pairs(trace, max_cases=0, seed=17)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.record_index == 0
    assert pair.chain_id == 101
    assert pair.first_error_step == 1
    assert pair.control_step == 0
    assert pair.control_decision_position == 1
    assert pair.event_decision_position == 3


def test_paired_summary_bootstraps_event_minus_control_by_case() -> None:
    rows = []
    for case, domain, control, event in (
        ("a", "gsm8k", 1.0, 3.0),
        ("b", "gsm8k", 2.0, 5.0),
        ("c", "math", 4.0, 8.0),
        ("d", "math", 3.0, 8.0),
    ):
        rows.extend(
            (
                {
                    "case_id": case,
                    "domain": domain,
                    "boundary_role": "previous_correct",
                    "layer": 8,
                    "largest_eigenvalue": control,
                },
                {
                    "case_id": case,
                    "domain": domain,
                    "boundary_role": "first_error",
                    "layer": 8,
                    "largest_eigenvalue": event,
                },
            )
        )

    summary = paired_fisher_summary(
        rows,
        metrics=("largest_eigenvalue",),
        bootstrap_repeats=500,
        seed=9,
    )

    result = summary["pooled"]["8"]["largest_eigenvalue"]
    assert result["n_pairs"] == 4
    assert result["mean_event_minus_control"] == 3.5
    assert result["ci_low"] > 0.0
    assert result["positive_fraction"] == 1.0
