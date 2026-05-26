"""Find step-representative token indices in a tokenized response.

For demo simplicity, we use the LAST TOKEN of each step (the token before the
step separator '\\n' or before the next step prefix like 'Step 2:').

Production version should use a PRM800K-trained classifier; we keep this simple
for the demo.
"""

import re


def split_response_into_steps(response_text):
    """Split a response into step text strings using simple heuristics.

    Recognizes 'Step k:' prefixes (case-insensitive) and '\\n\\n' separators.
    Each returned step is the text from one separator to the next.

    Args:
        response_text: full response text from the model

    Returns:
        list[str]: text of each step (in order)
    """
    # Strategy: split on double newline or 'Step N:' pattern
    # ProcessBench format is usually 'Step 1: ...\n\nStep 2: ...'
    parts = re.split(r"\n\s*\n|\bStep\s+\d+\s*[:\.]", response_text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def find_step_token_indices(tokenizer, prompt_text, response_text, steps_text=None):
    """Find token indices that mark the end of each reasoning step.

    Algorithm:
        1. Tokenize full prompt+response with offset_mapping
        2. For each step, locate its end character position in response
        3. Find token whose offset contains (or is closest before) that char

    Args:
        tokenizer: a fast HuggingFace tokenizer
        prompt_text: the prompt (question + instructions) text
        response_text: the generated response text
        steps_text: optional list of step strings (if pre-parsed, e.g., from ProcessBench)

    Returns:
        list[int]: token indices (in full prompt+response sequence) of the last
                   token of each step
    """
    if steps_text is None:
        steps_text = split_response_into_steps(response_text)

    if len(steps_text) == 0:
        return []

    # Tokenize full sequence with offsets
    full_text = prompt_text + response_text
    encoding = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = encoding["offset_mapping"]  # list of (start, end)

    # For each step, find its end character position in response_text
    response_start_char = len(prompt_text)
    cumulative_pos_in_response = 0
    step_token_indices = []

    for step in steps_text:
        # Find this step text within response (handle minor whitespace differences)
        idx = response_text.find(step, cumulative_pos_in_response)
        if idx == -1:
            # Fallback: skip this step
            continue
        end_char_in_response = idx + len(step)
        end_char_in_full = response_start_char + end_char_in_response

        # Find the token whose end >= end_char (last token of the step)
        last_token_idx = None
        for tok_idx, (start, end) in enumerate(offsets):
            if start <= end_char_in_full <= end:
                last_token_idx = tok_idx
                break
            if start > end_char_in_full:
                last_token_idx = max(0, tok_idx - 1)
                break
        if last_token_idx is None:
            last_token_idx = len(offsets) - 1

        step_token_indices.append(last_token_idx)
        cumulative_pos_in_response = end_char_in_response

    return step_token_indices
