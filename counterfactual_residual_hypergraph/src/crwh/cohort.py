from __future__ import annotations

import hashlib
from collections import defaultdict
from numbers import Integral
from typing import Iterable, Protocol, TypeVar


class BinaryCohortCandidate(Protocol):
    trace_id: str
    problem_id: str
    response_label: int


CandidateT = TypeVar("CandidateT", bound=BinaryCohortCandidate)


def _rank(candidate: BinaryCohortCandidate, *, seed: int, label: int) -> bytes:
    identity = (
        f"{seed}\0{label}\0{candidate.problem_id}\0{candidate.trace_id}"
    ).encode("utf-8")
    return hashlib.sha256(identity).digest()


def select_balanced_unique(
    candidates: Iterable[CandidateT],
    *,
    limit: int,
    seed: int,
) -> tuple[CandidateT, ...]:
    """Select an order-independent binary cohort with unique problem groups."""

    if isinstance(limit, bool) or not isinstance(limit, Integral):
        raise ValueError("limit must be an even integer")
    if int(limit) < 2 or int(limit) % 2:
        raise ValueError("limit must be an even integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    limit, seed = int(limit), int(seed)

    by_group: dict[str, dict[int, CandidateT]] = defaultdict(dict)
    trace_ids: set[str] = set()
    for candidate in candidates:
        trace_id = getattr(candidate, "trace_id", None)
        problem_id = getattr(candidate, "problem_id", None)
        label = getattr(candidate, "response_label", None)
        if not isinstance(trace_id, str) or not trace_id.strip():
            raise ValueError("candidate trace_id must be a non-empty string")
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise ValueError("candidate problem_id must be a non-empty string")
        if isinstance(label, bool) or not isinstance(label, Integral):
            raise ValueError("candidate response_label must be 0 or 1")
        label = int(label)
        if label not in (0, 1):
            raise ValueError("candidate response_label must be 0 or 1")
        if trace_id in trace_ids:
            raise ValueError(f"duplicate trace_id: {trace_id}")
        trace_ids.add(trace_id)

        previous = by_group[problem_id].get(label)
        if previous is None or _rank(
            candidate, seed=seed, label=label
        ) < _rank(previous, seed=seed, label=label):
            by_group[problem_id][label] = candidate

    quota = limit // 2
    selected: list[CandidateT] = []
    used_groups: set[str] = set()
    deficits: dict[int, int] = {}
    for label in (0, 1):
        exclusive = [
            (group_id, choices[label])
            for group_id, choices in by_group.items()
            if set(choices) == {label}
        ]
        exclusive.sort(key=lambda item: _rank(item[1], seed=seed, label=label))
        chosen = exclusive[:quota]
        selected.extend(candidate for _, candidate in chosen)
        used_groups.update(group_id for group_id, _ in chosen)
        deficits[label] = quota - len(chosen)

    shared = [
        (group_id, choices)
        for group_id, choices in by_group.items()
        if set(choices) == {0, 1} and group_id not in used_groups
    ]
    if sum(deficits.values()) > len(shared):
        counts = {
            label: sum(label in choices for choices in by_group.values())
            for label in (0, 1)
        }
        raise ValueError(
            "insufficient unique problem groups for a balanced cohort: "
            f"need {quota} per class, available={counts}"
        )

    for label in (0, 1):
        shared.sort(
            key=lambda item: _rank(item[1][label], seed=seed, label=label)
        )
        chosen = shared[: deficits[label]]
        selected.extend(choices[label] for _, choices in chosen)
        chosen_groups = {group_id for group_id, _ in chosen}
        used_groups.update(chosen_groups)
        shared = [
            item for item in shared if item[0] not in chosen_groups
        ]

    class_counts = {
        label: sum(int(candidate.response_label) == label for candidate in selected)
        for label in (0, 1)
    }
    if len(selected) != limit or class_counts != {0: quota, 1: quota}:
        raise ValueError(
            "insufficient unique problem groups for a balanced cohort: "
            f"need {quota} per class, selected={class_counts}"
        )
    selected.sort(
        key=lambda candidate: _rank(
            candidate,
            seed=seed,
            label=int(candidate.response_label),
        )
    )
    return tuple(selected)
