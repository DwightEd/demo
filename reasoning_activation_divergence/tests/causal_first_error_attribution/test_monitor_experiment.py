from __future__ import annotations

import numpy as np

from functional_divergence.causal_first_error_attribution.monitor_data import (
    MonitorBoundaryRow,
)
from functional_divergence.causal_first_error_attribution.monitor_experiment import (
    correct_chain_threshold,
    evaluate_boundary_scores,
    paired_problem_bootstrap,
)


def _row(
    chain: str,
    problem: str,
    domain: str,
    step: int,
    gold: int,
) -> MonitorBoundaryRow:
    return MonitorBoundaryRow(
        chain_id=chain,
        problem_hash=f"{domain}::{problem}",
        sibling_group=f"{domain}::{problem}",
        domain=domain,
        candidate_step=step,
        first_error_step=gold,
        decision_position=step + 2,
        label=int(step == gold),
        nuisance=np.zeros(5, dtype=np.float32),
        output_context=np.zeros(5, dtype=np.float32),
        store_index=0,
        state_index=0,
    )


def test_false_alarm_threshold_is_selected_on_correct_chain_maxima() -> None:
    rows = [
        _row("correct-a", "p1", "gsm8k", 0, -1),
        _row("correct-a", "p1", "gsm8k", 1, -1),
        _row("correct-b", "p2", "gsm8k", 0, -1),
        _row("correct-b", "p2", "gsm8k", 1, -1),
        _row("error", "p3", "gsm8k", 0, 1),
        _row("error", "p3", "gsm8k", 1, 1),
    ]
    scores = np.asarray([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])

    threshold = correct_chain_threshold(
        rows, np.arange(len(rows)), scores, target_false_alarm_rate=0.5
    )

    assert threshold == 0.8


def test_boundary_evaluation_reports_localization_and_problem_balanced_rows() -> None:
    rows = [
        _row("error-a", "p1", "math", 0, 1),
        _row("error-a", "p1", "math", 1, 1),
        _row("correct-a", "p2", "math", 0, -1),
        _row("correct-a", "p2", "math", 1, -1),
    ]
    scores = np.asarray([0.1, 0.9, 0.1, 0.2])

    report = evaluate_boundary_scores(
        rows, np.arange(len(rows)), scores, false_alarm_threshold=0.5
    )

    assert report["top1_localization"] == 1.0
    assert report["mrr"] == 1.0
    assert report["correct_chain_false_alarm_rate"] == 0.0
    assert report["row_auroc"] == 1.0
    assert report["problem_groups"] == 2


def test_paired_bootstrap_tests_graph_against_each_structural_control() -> None:
    rows = []
    candidate = []
    baseline = []
    for domain in ("gsm8k", "math"):
        for problem in range(4):
            rows.extend(
                [
                    _row(f"{domain}-{problem}", f"p{problem}", domain, 0, 1),
                    _row(f"{domain}-{problem}", f"p{problem}", domain, 1, 1),
                ]
            )
            candidate.extend([0.1, 0.9])
            baseline.extend([0.9, 0.1])
    indices = np.arange(len(rows))

    report = paired_problem_bootstrap(
        rows,
        indices,
        candidate_scores=np.asarray(candidate),
        baseline_scores=np.asarray(baseline),
        candidate_thresholds={"gsm8k": 0.5, "math": 0.5},
        baseline_thresholds={"gsm8k": 0.5, "math": 0.5},
        repeats=100,
        seed=11,
    )

    assert report["top1_gain"]["point"] == 1.0
    assert report["top1_gain"]["ci_low"] > 0.0
    assert report["nll_improvement"]["ci_low"] > 0.0
