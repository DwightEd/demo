from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreOnsetView:
    decision_position: int
    prefix_input_ids: np.ndarray
    residual_state: np.ndarray
    past_components: np.ndarray


def build_pre_onset_view(
    input_ids: np.ndarray,
    step_token_start: np.ndarray,
    *,
    candidate_step: int,
    boundary_residual: np.ndarray,
    completed_step_components: np.ndarray,
) -> PreOnsetView:
    """Build one future-free boundary view; step zero remains identifiable."""
    tokens = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    starts = np.asarray(step_token_start, dtype=np.int64).reshape(-1)
    residual = np.asarray(boundary_residual)
    components = np.asarray(completed_step_components)
    step = int(candidate_step)
    if starts.size < 1 or np.any(np.diff(starts) <= 0):
        raise ValueError("step_token_start must be a non-empty increasing vector")
    if not (0 <= step < len(starts)):
        raise ValueError("candidate_step lies outside step_token_start")
    if starts[0] < 1 or starts[-1] >= len(tokens):
        raise ValueError("step starts must follow a prompt and lie inside input_ids")
    if residual.ndim != 2 or residual.shape[0] != len(starts):
        raise ValueError("boundary_residual must have shape [S,D]")
    if components.ndim < 2 or components.shape[0] != len(starts):
        raise ValueError("completed_step_components must start with step axis S")
    decision = int(starts[step]) - 1
    return PreOnsetView(
        decision_position=decision,
        prefix_input_ids=tokens[: decision + 1].copy(),
        residual_state=residual[step].copy(),
        past_components=components[:step].copy(),
    )


@dataclass(frozen=True)
class MonitorRow:
    chain_id: str
    problem_hash: str
    sibling_group: str
    domain: str
    candidate_step: int
    first_error_step: int
    score: float

    def __post_init__(self) -> None:
        if not all((self.chain_id, self.problem_hash, self.sibling_group, self.domain)):
            raise ValueError("monitor row identifiers cannot be empty")
        if self.candidate_step < 0 or self.first_error_step < -1:
            raise ValueError("monitor step indices are invalid")
        if not np.isfinite(self.score):
            raise ValueError("monitor score must be finite")


@dataclass(frozen=True)
class MonitorSplit:
    held_domain: str
    train: np.ndarray
    test: np.ndarray


def lodo_monitor_splits(
    rows: list[MonitorRow] | tuple[MonitorRow, ...],
) -> tuple[MonitorSplit, ...]:
    values = list(rows)
    if not values:
        raise ValueError("at least one monitor row is required")
    domains = sorted({row.domain for row in values})
    if len(domains) < 2:
        raise ValueError("LODO monitoring requires at least two domains")
    for attribute in ("problem_hash", "sibling_group"):
        owners: defaultdict[str, set[str]] = defaultdict(set)
        for row in values:
            owners[str(getattr(row, attribute))].add(row.domain)
        leaked = sorted(name for name, owner in owners.items() if len(owner) > 1)
        if leaked:
            raise ValueError(f"{attribute} spans held domains: {leaked[:3]}")
    row_domains = np.asarray([row.domain for row in values], dtype=object)
    return tuple(
        MonitorSplit(
            held_domain=domain,
            train=np.flatnonzero(row_domains != domain),
            test=np.flatnonzero(row_domains == domain),
        )
        for domain in domains
    )


def localization_metrics(
    rows: list[MonitorRow] | tuple[MonitorRow, ...],
    *,
    false_alarm_threshold: float,
) -> dict[str, Any]:
    """Rank the onset inside each chain instead of pooling independent rows."""
    grouped: defaultdict[str, list[MonitorRow]] = defaultdict(list)
    for row in rows:
        grouped[row.chain_id].append(row)
    reciprocal_ranks: list[float] = []
    top1: list[float] = []
    correct_step_predictions: list[bool] = []
    correct_chain_predictions: list[bool] = []
    for chain_rows in grouped.values():
        gold_values = {row.first_error_step for row in chain_rows}
        if len(gold_values) != 1:
            raise ValueError("first_error_step must be constant within each chain")
        candidates = [row.candidate_step for row in chain_rows]
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidate_step must be unique within each chain")
        gold = gold_values.pop()
        if gold == -1:
            predictions = [row.score >= false_alarm_threshold for row in chain_rows]
            correct_step_predictions.extend(predictions)
            correct_chain_predictions.append(any(predictions))
            continue
        eligible = [row for row in chain_rows if row.candidate_step <= gold]
        target = [row for row in eligible if row.candidate_step == gold]
        if len(target) != 1:
            raise ValueError("an error chain must contain its first-error candidate")
        ordered = sorted(eligible, key=lambda row: (-row.score, row.candidate_step))
        rank = next(
            index
            for index, row in enumerate(ordered, start=1)
            if row.candidate_step == gold
        )
        reciprocal_ranks.append(1.0 / rank)
        top1.append(float(rank == 1))
    if not reciprocal_ranks:
        raise ValueError("localization requires at least one error chain")
    return {
        "error_chains": len(reciprocal_ranks),
        "correct_chains": len(correct_chain_predictions),
        "top1_localization": float(np.mean(top1)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "correct_chain_false_alarm_rate": (
            float(np.mean(correct_chain_predictions))
            if correct_chain_predictions
            else float("nan")
        ),
        "correct_step_false_alarm_rate": (
            float(np.mean(correct_step_predictions))
            if correct_step_predictions
            else float("nan")
        ),
    }


__all__ = [
    "MonitorRow",
    "MonitorSplit",
    "PreOnsetView",
    "build_pre_onset_view",
    "localization_metrics",
    "lodo_monitor_splits",
]

