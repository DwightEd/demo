from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np

from utils.step_boundaries import TokenAlignmentError


@dataclass
class ForwardCache:
    input_ids: "object"
    offset_mapping: Sequence[Tuple[int, int]]
    prompt_len_tokens: int
    response_start_token: int
    step_token_ranges: list[tuple[int, int]]
    hidden_states: list["object"] | None = None
    attentions: list["object"] | None = None
    logits: "object" | None = None
    seq_len: int = 0
    replay_kind: str = ""


def build_prompt_response(problem: str, steps: Sequence[str]) -> tuple[str, str]:
    prompt = f"Problem: {problem}\n\nSolution:\n\n"
    response = "\n\n".join(str(s) for s in steps)
    return prompt, response


def prompt_token_indices(offsets: Sequence[Tuple[int, int]], prompt_len_chars: int) -> np.ndarray:
    idx = []
    for i, (a, b) in enumerate(offsets):
        if a == b == 0 and i > 0:
            continue
        if b <= prompt_len_chars and b > a:
            idx.append(i)
    return np.asarray(idx, dtype=np.int64)


def response_start_token(offsets: Sequence[Tuple[int, int]], prompt_len_chars: int) -> int:
    for i, (a, b) in enumerate(offsets):
        if b > prompt_len_chars and b > a:
            return int(i)
    return int(len(offsets))


def run_teacher_forcing(
    model,
    tokenizer,
    prompt: str,
    response: str,
    *,
    output_hidden_states: bool,
    output_attentions: bool,
    output_logits: bool = True,
    max_seq_len: int = 4096,
    steps: Sequence[str] | None = None,
    exact_input_ids: Sequence[int] | None = None,
    exact_attention_mask: Sequence[int] | None = None,
    exact_token_offsets: Sequence[Tuple[int, int]] | None = None,
    exact_step_token_ranges: Sequence[Tuple[int, int]] | None = None,
    exact_response_start_token: int | None = None,
) -> ForwardCache:
    """Run one teacher-forced forward pass and align response steps.

    Large tensors are detached to CPU immediately.  The extraction framework
    computes compact mechanism scores and never persists full attentions.
    """

    import torch

    trace = prepare_teacher_forcing_trace(
        tokenizer,
        prompt,
        response,
        steps=steps,
        max_seq_len=max_seq_len,
        exact_input_ids=exact_input_ids,
        exact_attention_mask=exact_attention_mask,
        exact_token_offsets=exact_token_offsets,
        exact_step_token_ranges=exact_step_token_ranges,
        exact_response_start_token=exact_response_start_token,
    )
    offsets = trace["token_offsets"]
    resp_start = int(trace["response_start_token"])
    ranges = trace["step_token_ranges"]
    enc = {
        "input_ids": torch.tensor([trace["input_ids"]], dtype=torch.long),
        "attention_mask": torch.tensor([trace["attention_mask"]], dtype=torch.long),
    }
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(
            **enc,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            use_cache=False,
            return_dict=True,
        )

    input_ids = enc["input_ids"][0].detach().cpu()
    seq_len = int(input_ids.shape[0])
    safe_ranges = [(int(a), int(b)) for (a, b) in ranges if b < seq_len and b >= a]

    hidden = None
    if output_hidden_states and getattr(out, "hidden_states", None) is not None:
        hidden = [h[0].detach().float().cpu() for h in out.hidden_states]

    attn = None
    if output_attentions and getattr(out, "attentions", None) is not None:
        attn = [a[0].detach().float().cpu() for a in out.attentions]

    logits = None
    if output_logits and getattr(out, "logits", None) is not None:
        logits = out.logits[0].detach().float().cpu()

    del out
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ForwardCache(
        input_ids=input_ids,
        offset_mapping=offsets,
        prompt_len_tokens=resp_start,
        response_start_token=resp_start,
        step_token_ranges=safe_ranges,
        hidden_states=hidden,
        attentions=attn,
        logits=logits,
        seq_len=seq_len,
        replay_kind=str(trace["replay_kind"]),
    )


def prepare_teacher_forcing_trace(
    tokenizer,
    prompt: str,
    response: str,
    *,
    steps: Sequence[str] | None,
    max_seq_len: int,
    exact_input_ids: Sequence[int] | None = None,
    exact_attention_mask: Sequence[int] | None = None,
    exact_token_offsets: Sequence[Tuple[int, int]] | None = None,
    exact_step_token_ranges: Sequence[Tuple[int, int]] | None = None,
    exact_response_start_token: int | None = None,
) -> dict[str, Any]:
    """Prepare one token axis for both step alignment and model replay.

    If an extraction artifact supplies exact arrays, they are replayed directly
    and never decoded/re-tokenized.  Otherwise the shared no-special-token
    alignment builder creates the axis once.  In either route, step ranges and
    hidden states index the exact same sequence.
    """

    supplied = exact_input_ids is not None
    optional = [
        exact_attention_mask,
        exact_token_offsets,
        exact_step_token_ranges,
        exact_response_start_token,
    ]
    if supplied and any(value is None for value in optional):
        raise TokenAlignmentError(
            "exact_input_ids requires attention_mask, token_offsets, step_token_ranges, "
            "and response_start_token"
        )
    if supplied:
        ids = [int(x) for x in exact_input_ids] if exact_input_ids is not None else []
        mask = [int(x) for x in exact_attention_mask] if exact_attention_mask is not None else []
        offsets = (
            [(int(a), int(b)) for a, b in exact_token_offsets]
            if exact_token_offsets is not None
            else []
        )
        ranges = (
            [(int(a), int(b)) for a, b in exact_step_token_ranges]
            if exact_step_token_ranges is not None
            else []
        )
        response_start = int(exact_response_start_token)
        if len(ids) != len(mask) or len(ids) != len(offsets):
            raise TokenAlignmentError("exact input_ids, attention_mask, and token_offsets disagree")
        if any(value not in (0, 1) for value in mask):
            raise TokenAlignmentError("exact attention_mask must be binary")
        if any(a < 0 or b < a for a, b in offsets):
            raise TokenAlignmentError("exact token_offsets contains an invalid half-open span")
        visible_offsets = [(a, b) for a, b in offsets if b > a]
        if any(
            visible_offsets[i][0] < visible_offsets[i - 1][0]
            for i in range(1, len(visible_offsets))
        ):
            raise TokenAlignmentError("exact token_offsets is not monotone")
        if not 0 <= response_start <= len(ids):
            raise TokenAlignmentError("exact response_start_token is outside input_ids")
        if any(a < response_start or b < a or b >= len(ids) for a, b in ranges):
            raise TokenAlignmentError("an exact step token range is outside input_ids")
        if any(ranges[i][0] <= ranges[i - 1][1] for i in range(1, len(ranges))):
            raise TokenAlignmentError("exact step token ranges must be strictly ordered and non-overlapping")
        if steps is not None and len(ranges) != len(steps):
            raise TokenAlignmentError(
                f"exact step ranges ({len(ranges)}) do not match kept step strings ({len(steps)})"
            )
    else:
        try:
            from utils.step_boundaries import build_exact_trace_alignment
        except Exception:  # pragma: no cover - package import fallback
            from ..utils.step_boundaries import build_exact_trace_alignment  # type: ignore
        trace = build_exact_trace_alignment(
            tokenizer,
            prompt,
            response,
            list(steps) if steps is not None else None,
        )
        ids = [int(x) for x in trace["input_ids"]]
        mask = [int(x) for x in trace["attention_mask"]]
        offsets = [(int(a), int(b)) for a, b in trace["token_offsets"]]
        ranges = [(int(a), int(b)) for a, b in trace["all_step_token_ranges"]]
        response_start = int(trace["response_token_range"][0])

    limit = len(ids) if int(max_seq_len) <= 0 else min(len(ids), int(max_seq_len))
    if limit < response_start:
        raise ValueError(
            f"max_seq_len={max_seq_len} truncates the prompt at token {response_start}"
        )
    ids = ids[:limit]
    mask = mask[:limit]
    offsets = offsets[:limit]
    truncated_ranges = [(a, b) for a, b in ranges if b >= limit]
    if truncated_ranges:
        raise TokenAlignmentError(
            f"max_seq_len={max_seq_len} truncates {len(truncated_ranges)} reasoning "
            "step range(s); refusing to change the chain/step axis"
        )
    ranges = [(a, b) for a, b in ranges if b < limit]
    if not ranges:
        raise ValueError("no complete reasoning step remains on the teacher-forcing axis")
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "token_offsets": offsets,
        "response_start_token": response_start,
        "step_token_ranges": ranges,
        "replay_kind": "exact_artifact_ids" if supplied else "single_axis_no_special_fallback",
    }
