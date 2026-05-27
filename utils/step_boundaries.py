"""Find token RANGES (start, end) for each reasoning step.

The (step × layer) low-rank analysis needs the full token cloud H_j^(l)
∈ R^{n_j × d} at every step. We therefore need the token ranges occupied by
each step, not just the last token. This is the key change from the prior
last-token-only extraction.

For demo simplicity, step boundaries come from the pre-parsed `steps` list in
ProcessBench (so we sidestep the boundary-classifier issue entirely).
"""

from __future__ import annotations

import re
from typing import List, Tuple


def split_response_into_steps(response_text: str) -> List[str]:
    """Heuristic split when steps are not pre-parsed.

    Splits on blank lines or 'Step N:' / 'Step N.' headers.
    """
    parts = re.split(r"\n\s*\n|\bStep\s+\d+\s*[:\.]", response_text)
    return [p.strip() for p in parts if p.strip()]


def find_step_token_ranges(
    tokenizer,
    prompt_text: str,
    response_text: str,
    steps_text: List[str] | None = None,
) -> List[Tuple[int, int]]:
    """Return token (start, end) ranges (inclusive) for each step.

    Args:
        tokenizer: a *fast* HF tokenizer (offset_mapping required).
        prompt_text: prompt prefix (question + instructions).
        response_text: model response text.
        steps_text: pre-parsed step strings (e.g. from ProcessBench); if None,
            falls back to the heuristic split.

    Returns:
        ranges: list of (start_tok_idx, end_tok_idx) tuples, both inclusive,
                indexing into the tokenization of `prompt_text + response_text`.
                Steps whose text cannot be located are silently dropped.
    """
    if steps_text is None:
        steps_text = split_response_into_steps(response_text)
    if not steps_text:
        return []

    full_text = prompt_text + response_text
    encoding = tokenizer(
        full_text, return_offsets_mapping=True, add_special_tokens=True,
    )
    offsets = encoding["offset_mapping"]  # list[(char_start, char_end)]
    n_tokens = len(offsets)

    response_start_char = len(prompt_text)
    cursor_in_response = 0
    ranges: List[Tuple[int, int]] = []

    for step in steps_text:
        idx = response_text.find(step, cursor_in_response)
        if idx == -1:
            continue
        start_char = response_start_char + idx
        end_char = response_start_char + idx + len(step)

        start_tok = None
        end_tok = None
        for t, (a, b) in enumerate(offsets):
            if a == b == 0 and t > 0:
                # special token with empty offset; skip
                continue
            if b <= start_char:
                continue
            if a >= end_char:
                break
            if start_tok is None:
                start_tok = t
            end_tok = t

        if start_tok is not None and end_tok is not None and end_tok >= start_tok:
            ranges.append((start_tok, end_tok))
        cursor_in_response = idx + len(step)

    return ranges
