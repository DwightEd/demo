from __future__ import annotations

import numpy as np

from functional_divergence.causal_first_error_attribution.evaluation import (
    MonitorRow,
    build_pre_onset_view,
    localization_metrics,
    lodo_monitor_splits,
)


def test_pre_onset_task_never_reads_current_step_or_post_state() -> None:
    input_ids = np.asarray([10, 11, 20, 21, 30, 31], dtype=np.int64)
    starts = np.asarray([2, 4], dtype=np.int64)
    boundary = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    components = np.asarray([[5.0, 6.0], [7.0, 8.0]])

    original = build_pre_onset_view(
        input_ids,
        starts,
        candidate_step=1,
        boundary_residual=boundary,
        completed_step_components=components,
    )
    changed_tokens = input_ids.copy()
    changed_tokens[4:] = 999
    changed_boundary = boundary.copy()
    changed_boundary[0] = [-100.0, -100.0]
    changed_components = components.copy()
    changed_components[1] = [1000.0, 1000.0]
    changed = build_pre_onset_view(
        changed_tokens,
        starts,
        candidate_step=1,
        boundary_residual=changed_boundary,
        completed_step_components=changed_components,
    )

    np.testing.assert_array_equal(original.prefix_input_ids, changed.prefix_input_ids)
    np.testing.assert_array_equal(original.residual_state, changed.residual_state)
    np.testing.assert_array_equal(original.past_components, changed.past_components)
    assert original.decision_position == 3


def test_problem_and_counterfactual_siblings_never_cross_folds() -> None:
    rows = []
    for domain in ("gsm8k", "math"):
        for chain in range(2):
            for step in range(2):
                rows.append(
                    MonitorRow(
                        chain_id=f"{domain}-{chain}",
                        problem_hash=f"{domain}-problem-{chain}",
                        sibling_group=f"{domain}-sibling-{chain}",
                        domain=domain,
                        candidate_step=step,
                        first_error_step=1,
                        score=float(step),
                    )
                )

    for split in lodo_monitor_splits(rows):
        train_siblings = {rows[index].sibling_group for index in split.train}
        test_siblings = {rows[index].sibling_group for index in split.test}
        train_problems = {rows[index].problem_hash for index in split.train}
        test_problems = {rows[index].problem_hash for index in split.test}
        assert not train_siblings.intersection(test_siblings)
        assert not train_problems.intersection(test_problems)


def test_localization_ranks_first_error_within_chain_not_over_rows() -> None:
    rows = [
        MonitorRow("error-a", "p-a", "s-a", "gsm8k", 0, 2, 0.1),
        MonitorRow("error-a", "p-a", "s-a", "gsm8k", 1, 2, 0.2),
        MonitorRow("error-a", "p-a", "s-a", "gsm8k", 2, 2, 0.9),
        MonitorRow("error-b", "p-b", "s-b", "gsm8k", 0, 1, 0.8),
        MonitorRow("error-b", "p-b", "s-b", "gsm8k", 1, 1, 0.7),
        MonitorRow("correct", "p-c", "s-c", "gsm8k", 0, -1, 0.1),
        MonitorRow("correct", "p-c", "s-c", "gsm8k", 1, -1, 0.2),
    ]

    metrics = localization_metrics(rows, false_alarm_threshold=0.5)

    assert metrics["error_chains"] == 2
    assert metrics["top1_localization"] == 0.5
    assert metrics["mrr"] == 0.75
    assert metrics["correct_chain_false_alarm_rate"] == 0.0
    assert metrics["correct_step_false_alarm_rate"] == 0.0

