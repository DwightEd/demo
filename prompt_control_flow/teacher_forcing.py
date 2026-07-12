from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


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
) -> ForwardCache:
    """Run one teacher-forced forward pass and align response steps.

    Large tensors are detached to CPU immediately.  The extraction framework
    computes compact mechanism scores and never persists full attentions.
    """

    import torch

    try:
        from utils.step_boundaries import find_step_token_ranges
    except Exception:  # pragma: no cover - package import fallback
        from ..utils.step_boundaries import find_step_token_ranges  # type: ignore

    full_text = prompt + response
    enc_offsets = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_len,
    )
    offsets = list(enc_offsets["offset_mapping"])
    resp_start = response_start_token(offsets, len(prompt))
    ranges = find_step_token_ranges(tokenizer, prompt, response, list(steps) if steps is not None else None)

    enc = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_len,
    )
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
    )
