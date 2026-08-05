from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .contracts import OnsetPair


def first_divergence(
    error_token_ids: np.ndarray,
    corrected_token_ids: np.ndarray,
    *,
    step_start: int,
) -> tuple[int, int]:
    """Return the first differing token and its last observable prefix position."""
    error = np.asarray(error_token_ids, dtype=np.int64).reshape(-1)
    corrected = np.asarray(corrected_token_ids, dtype=np.int64).reshape(-1)
    start = int(step_start)
    if start < 1 or start > min(len(error), len(corrected)):
        raise ValueError("step_start must follow a non-empty observable prefix")
    if not np.array_equal(error[:start], corrected[:start]):
        raise ValueError("error and corrected traces differ before the target step")
    shared = min(len(error), len(corrected))
    differences = np.flatnonzero(error[start:shared] != corrected[start:shared])
    if differences.size:
        divergent = start + int(differences[0])
    elif len(error) != len(corrected):
        divergent = shared
    else:
        raise ValueError("error and corrected traces contain no divergent token")
    return divergent, divergent - 1


def load_pre_decision_prefix(token_ids: np.ndarray, decision_position: int) -> np.ndarray:
    """Copy only tokens observable when predicting the target token."""
    values = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    position = int(decision_position)
    if position < 0 or position >= len(values):
        raise ValueError("decision_position lies outside token_ids")
    return values[: position + 1].copy()


def load_pair_file(path: str | Path) -> tuple[tuple[OnsetPair, ...], tuple[str, ...]]:
    """Load valid pairs and return human-readable validation errors separately."""
    source = Path(path)
    if not source.is_file():
        return (), ()
    valid: list[OnsetPair] = []
    errors: list[str] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("JSON value must be an object")
            valid.append(OnsetPair.from_mapping(payload))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(f"line {line_number}: {exc}")
    case_ids = [pair.case_id for pair in valid]
    if len(set(case_ids)) != len(case_ids):
        errors.append("case_id values must be unique within one pair file")
        valid = []
    return tuple(valid), tuple(errors)


def pairs_of_kind(pairs: Iterable[OnsetPair], kind: str) -> tuple[OnsetPair, ...]:
    return tuple(pair for pair in pairs if pair.pair_kind == kind)
