from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ReplayPlan:
    trigger_index: int
    cut_index: int
    prefix: Tuple[Any, ...]
    removed_suffix: Tuple[Any, ...]
    repair_instruction: Tuple[Any, ...]
    model_input: Tuple[Any, ...]


@dataclass(frozen=True)
class TextReplayPlan:
    trigger_step: int
    cut_char: int
    prefix: str
    removed_suffix: str
    model_input: str


def build_micro_replay(
    tokens: Sequence[Any],
    trigger_index: int,
    *,
    rollback: int = 0,
    repair_instruction: Sequence[Any] = (),
    context_limit: Optional[int] = None,
) -> ReplayPlan:
    """Pure token-level micro-replay planner.

    ``trigger_index`` is the first token/window judged unsafe.  The returned
    model input keeps only the safe prefix, optionally rolls back extra tokens,
    and appends a repair instruction.  It never calls or mutates a model.
    """
    original = tuple(tokens)
    trigger = int(trigger_index)
    if trigger < 0 or trigger > len(original):
        raise ValueError("trigger_index lies outside the token sequence")
    cut = max(0, trigger - max(0, int(rollback)))
    prefix = original[:cut]
    if context_limit is not None:
        limit = max(0, int(context_limit))
        prefix = prefix[-limit:] if limit else tuple()
    instruction = tuple(repair_instruction)
    return ReplayPlan(
        trigger_index=trigger,
        cut_index=cut,
        prefix=prefix,
        removed_suffix=original[cut:],
        repair_instruction=instruction,
        model_input=prefix + instruction,
    )


def apply_micro_replay(plan: ReplayPlan, continuation: Sequence[Any]) -> Tuple[Any, ...]:
    """Compose the preserved prefix and a generated continuation, without mutation."""
    return tuple(plan.prefix) + tuple(continuation)


def build_text_micro_replay(
    text: str,
    spans: Sequence[Sequence[int]],
    trigger_step: int,
    *,
    rollback_steps: int = 0,
    repair_instruction: str = "\n[Re-evaluate the flagged step using the original constraints.]\n",
) -> TextReplayPlan:
    """Pure text-level equivalent using half-open step/window character spans."""
    ss = np.asarray(spans, int)
    if ss.ndim != 2 or ss.shape[1] != 2:
        raise ValueError("spans must have shape [steps, 2]")
    t = int(trigger_step)
    if t < 0 or t >= len(ss):
        raise ValueError("trigger_step lies outside spans")
    keep_step = max(0, t - max(0, int(rollback_steps)))
    cut = int(ss[keep_step, 0])
    if cut < 0 or cut > len(text):
        raise ValueError("span lies outside text")
    prefix = str(text[:cut])
    return TextReplayPlan(
        trigger_step=t,
        cut_char=cut,
        prefix=prefix,
        removed_suffix=str(text[cut:]),
        model_input=prefix + str(repair_instruction),
    )


def select_low_risk_candidate(
    candidates: Sequence[Any],
    risk_scores: Sequence[float],
) -> tuple[Any, int]:
    """Deterministically select the finite minimum-risk replay continuation."""
    if len(candidates) != len(risk_scores) or len(candidates) == 0:
        raise ValueError("candidates and risk_scores must be non-empty and aligned")
    risk = np.asarray(risk_scores, float)
    finite = np.where(np.isfinite(risk))[0]
    if finite.size == 0:
        raise ValueError("no finite candidate risk score")
    idx = int(finite[np.argmin(risk[finite])])
    return candidates[idx], idx
