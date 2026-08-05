from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DecisionPrefix:
    input_ids: np.ndarray
    source_step_ids: np.ndarray
    decision_position: int
    first_error_step: int
    dataset: str


def _record_scalar(archive, name: str, row: int, default: int | str):
    if name not in archive.files:
        return default
    values = np.asarray(archive[name])
    if values.ndim == 0:
        return values.item()
    return values.reshape(-1)[row]


def load_decision_prefix(
    trace_path: str | Path,
    *,
    record_index: int,
    decision_position: int,
) -> DecisionPrefix:
    """Load one trace row and crop it before any replay reaches the model."""
    path = Path(trace_path).expanduser()
    with np.load(path, allow_pickle=True) as archive:
        required = {
            "full_input_ids",
            "full_attention_mask",
            "prompt_token_counts",
            "step_token_ranges",
            "n_steps",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path}: trace is missing {missing}")
        inputs = np.asarray(archive["full_input_ids"])
        masks = np.asarray(archive["full_attention_mask"])
        row = int(record_index)
        if inputs.ndim != 2 or masks.shape != inputs.shape:
            raise ValueError("full input ids and attention mask must have shape [N,T]")
        if not (0 <= row < inputs.shape[0]):
            raise IndexError("record_index lies outside trace rows")
        token_row = np.asarray(inputs[row], dtype=np.int64).reshape(-1)
        mask_row = np.asarray(masks[row], dtype=np.int64).reshape(-1)
        zeros = np.flatnonzero(mask_row == 0)
        valid_count = int(zeros[0]) if zeros.size else len(mask_row)
        if not np.all(mask_row[:valid_count] == 1) or not np.all(
            mask_row[valid_count:] == 0
        ):
            raise ValueError("attention mask contains interior padding")
        q = int(decision_position)
        if not (0 <= q < valid_count):
            raise ValueError("decision_position lies outside the unpadded trace")

        prompt_count = int(_record_scalar(archive, "prompt_token_counts", row, -1))
        step_count = int(_record_scalar(archive, "n_steps", row, -1))
        if not (1 <= prompt_count <= valid_count) or step_count < 1:
            raise ValueError("trace contains invalid prompt or step counts")
        ranges = np.asarray(archive["step_token_ranges"][row], dtype=np.int64)
        if ranges.ndim != 2 or ranges.shape[1] != 2 or len(ranges) < step_count:
            raise ValueError("step_token_ranges must contain [start,end] rows")
        source_steps = np.full(valid_count, -2, dtype=np.int16)
        source_steps[:prompt_count] = -1
        for step, (start, end) in enumerate(ranges[:step_count]):
            start, end = int(start), int(end)
            if not (prompt_count <= start <= end < valid_count):
                raise ValueError("step token range lies outside the response trace")
            if np.any(source_steps[start : end + 1] >= 0):
                raise ValueError("step token ranges overlap")
            source_steps[start : end + 1] = step
        first_error = int(_record_scalar(archive, "gold_error_step", row, -2))
        dataset = str(_record_scalar(archive, "dataset", row, ""))
    return DecisionPrefix(
        input_ids=token_row[: q + 1].astype(np.int32, copy=True),
        source_step_ids=source_steps[: q + 1].copy(),
        decision_position=q,
        first_error_step=first_error,
        dataset=dataset,
    )


__all__ = ["DecisionPrefix", "load_decision_prefix"]
