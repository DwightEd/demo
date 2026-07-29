from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from crwh.cohort import select_balanced_unique


@dataclass(frozen=True)
class Candidate:
    trace_id: str
    problem_id: str
    response_label: int


def _candidate_pool() -> list[Candidate]:
    candidates = [
        Candidate(
            trace_id=f"trace-{label}-{index}",
            problem_id=f"problem-{label}-{index}",
            response_label=label,
        )
        for label in (0, 1)
        for index in range(8)
    ]
    candidates.extend(
        (
            Candidate("duplicate-normal", "problem-0-0", 0),
            Candidate("duplicate-error", "problem-1-0", 1),
        )
    )
    return candidates


def test_select_balanced_unique_balances_labels_and_deduplicates_problems() -> None:
    selected = select_balanced_unique(_candidate_pool(), limit=8, seed=17)

    class_counts = {
        label: sum(item.response_label == label for item in selected)
        for label in (0, 1)
    }
    assert len(selected) == 8
    assert class_counts == {0: 4, 1: 4}
    assert len({item.problem_id for item in selected}) == len(selected)


def test_select_balanced_unique_is_seeded_and_input_order_independent() -> None:
    candidates = _candidate_pool()

    first = select_balanced_unique(candidates, limit=8, seed=91)
    repeated = select_balanced_unique(candidates, limit=8, seed=91)
    reversed_input = select_balanced_unique(
        list(reversed(candidates)),
        limit=8,
        seed=91,
    )

    first_ids = tuple(item.trace_id for item in first)
    assert tuple(item.trace_id for item in repeated) == first_ids
    assert tuple(item.trace_id for item in reversed_input) == first_ids


def test_select_balanced_unique_rejects_insufficient_unique_class_members() -> None:
    candidates = [
        Candidate("normal-a", "one-normal-problem", 0),
        Candidate("normal-b", "one-normal-problem", 0),
        Candidate("error-a", "error-problem-a", 1),
        Candidate("error-b", "error-problem-b", 1),
        Candidate("error-c", "error-problem-c", 1),
        Candidate("error-d", "error-problem-d", 1),
    ]

    with pytest.raises(ValueError):
        select_balanced_unique(candidates, limit=6, seed=17)


def test_select_balanced_unique_allocates_all_shared_groups_without_reuse() -> None:
    candidates = [
        Candidate(
            trace_id=f"shared-{problem}-{label}",
            problem_id=f"shared-problem-{problem}",
            response_label=label,
        )
        for problem in range(8)
        for label in (0, 1)
    ]

    selected = select_balanced_unique(candidates, limit=6, seed=23)
    reversed_selected = select_balanced_unique(
        list(reversed(candidates)),
        limit=6,
        seed=23,
    )

    assert {
        label: sum(item.response_label == label for item in selected)
        for label in (0, 1)
    } == {0: 3, 1: 3}
    assert len({item.problem_id for item in selected}) == len(selected)
    assert tuple(item.trace_id for item in reversed_selected) == tuple(
        item.trace_id for item in selected
    )


def test_select_balanced_unique_fills_exclusive_class_deficits_from_shared_groups() -> None:
    candidates = [
        Candidate("normal-a", "normal-problem-a", 0),
        Candidate("normal-b", "normal-problem-b", 0),
        Candidate("error-a", "error-problem-a", 1),
    ]
    candidates.extend(
        Candidate(
            trace_id=f"shared-{problem}-{label}",
            problem_id=f"shared-problem-{problem}",
            response_label=label,
        )
        for problem in range(4)
        for label in (0, 1)
    )

    selected = select_balanced_unique(candidates, limit=6, seed=31)
    reversed_selected = select_balanced_unique(
        list(reversed(candidates)),
        limit=6,
        seed=31,
    )

    assert {
        label: sum(item.response_label == label for item in selected)
        for label in (0, 1)
    } == {0: 3, 1: 3}
    assert len({item.problem_id for item in selected}) == len(selected)
    assert tuple(item.trace_id for item in reversed_selected) == tuple(
        item.trace_id for item in selected
    )


def test_select_balanced_unique_rejects_shared_groups_insufficient_for_both_quotas() -> None:
    candidates = [
        Candidate(
            trace_id=f"shared-{problem}-{label}",
            problem_id=f"shared-problem-{problem}",
            response_label=label,
        )
        for problem in range(3)
        for label in (0, 1)
    ]

    with pytest.raises(ValueError, match="insufficient unique problem groups"):
        select_balanced_unique(candidates, limit=4, seed=17)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("trace_id", " trace-id"),
        ("trace_id", "trace-id "),
        ("trace_id", "trace\x00id"),
        ("problem_id", " problem-id"),
        ("problem_id", "problem-id "),
        ("problem_id", "problem\x00id"),
    ),
)
def test_select_balanced_unique_rejects_unsafe_identifiers(
    field: str,
    invalid_value: str,
) -> None:
    candidates = _candidate_pool()
    candidates[0] = replace(candidates[0], **{field: invalid_value})

    with pytest.raises(ValueError):
        select_balanced_unique(candidates, limit=8, seed=17)
