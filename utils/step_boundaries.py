"""Exact text/token alignment for reasoning traces.

The trace contract deliberately tokenizes the rendered prompt and response as
two segments with ``add_special_tokens=False`` and concatenates their token
IDs.  This keeps a generation prompt byte-for-byte and token-for-token
identical when the same sequence is replayed with teacher forcing.  Character
offsets are stored on the combined ``rendered_prompt + response`` text axis;
token ranges are explicit about whether they are half-open or inclusive.

This module is intentionally free of torch imports so alignment can be tested
with a small mock tokenizer on CPU-only machines.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


TRACE_SCHEMA_VERSION = "exact_generation_trace_v1"
TOKEN_OFFSET_CONVENTION = "char_half_open"
STEP_TOKEN_RANGE_CONVENTION = "inclusive"
SPAN_RANGE_CONVENTION = "half_open"


class TokenAlignmentError(ValueError):
    """Raised as soon as a text/token trace ceases to be exact."""


def split_response_into_steps(response_text: str) -> List[str]:
    """Heuristic split when steps are not pre-parsed.

    Splits on blank lines or ``Step N:`` / ``Step N.`` headers.  Returned
    strings remain verbatim substrings of ``response_text`` after trimming
    their surrounding whitespace.
    """

    parts = re.split(r"\n\s*\n|\bStep\s+\d+\s*[:\.]", response_text)
    return [p.strip() for p in parts if p.strip()]


def _tolist(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _flat_ints(value: Any, name: str) -> List[int]:
    raw = _tolist(value)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = [raw]
    if len(raw) == 1 and isinstance(raw[0], (list, tuple)):
        raw = list(raw[0])
    if any(isinstance(x, (list, tuple)) for x in raw):
        raise TokenAlignmentError(f"{name} must contain exactly one token sequence")
    return [int(x) for x in raw]


def _offsets(value: Any, name: str = "offset_mapping") -> List[Tuple[int, int]]:
    raw = _tolist(value)
    if raw is None:
        raise TokenAlignmentError(f"tokenizer did not return {name}")
    if not isinstance(raw, list):
        raise TokenAlignmentError(f"{name} must be a list")
    # A tokenizer called with return_tensors may leave one batch dimension.
    if (len(raw) == 1 and isinstance(raw[0], (list, tuple)) and raw[0]
            and isinstance(raw[0][0], (list, tuple))):
        raw = list(raw[0])
    out: List[Tuple[int, int]] = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise TokenAlignmentError(f"invalid {name} entry: {pair!r}")
        a, b = int(pair[0]), int(pair[1])
        if a < 0 or b < a:
            raise TokenAlignmentError(f"invalid {name} span {(a, b)}")
        out.append((a, b))
    return out


def _encode_segment(tokenizer, text: str, name: str) -> Dict[str, List[Any]]:
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = _flat_ints(encoding["input_ids"], f"{name}_input_ids")
    mask = _flat_ints(encoding.get("attention_mask", [1] * len(ids)),
                      f"{name}_attention_mask")
    offsets = _offsets(encoding.get("offset_mapping"), f"{name}_offset_mapping")
    if len(ids) != len(mask) or len(ids) != len(offsets):
        raise TokenAlignmentError(
            f"{name} token fields disagree: ids={len(ids)}, mask={len(mask)}, "
            f"offsets={len(offsets)}"
        )
    if any(x not in (0, 1) for x in mask):
        raise TokenAlignmentError(f"{name}_attention_mask must be binary")
    return {"input_ids": ids, "attention_mask": mask, "offset_mapping": offsets}


def _assert_same_tokens(expected: Sequence[int], actual: Sequence[int], name: str) -> None:
    expected = [int(x) for x in expected]
    actual = [int(x) for x in actual]
    if expected == actual:
        return
    limit = min(len(expected), len(actual))
    mismatch = next((i for i in range(limit) if expected[i] != actual[i]), limit)
    exp = expected[mismatch] if mismatch < len(expected) else "<end>"
    got = actual[mismatch] if mismatch < len(actual) else "<end>"
    raise TokenAlignmentError(
        f"{name} token mismatch at position {mismatch}: expected {exp}, got {got}; "
        f"lengths {len(expected)} != {len(actual)}"
    )


def trim_trailing_generation_tokens(
    token_ids: Any,
    *,
    pad_token_id: Any = None,
    eos_token_id: Any = None,
) -> Tuple[List[int], List[int]]:
    """Split visible response IDs from trailing EOS/padding IDs.

    The returned second list preserves exactly what was removed, so an artifact
    can retain the complete ``generate()`` tail while teacher forcing replays
    only the token sequence represented by ``response_text``.
    """

    ids = _flat_ints(token_ids, "generated_token_ids")
    terminal: set[int] = set()
    for value in (pad_token_id, eos_token_id):
        if value is None:
            continue
        raw = _tolist(value)
        if isinstance(raw, list):
            terminal.update(int(x) for x in raw)
        else:
            terminal.add(int(raw))
    cut = len(ids)
    while cut > 0 and ids[cut - 1] in terminal:
        cut -= 1
    return ids[:cut], ids[cut:]


def char_span_to_token_range(
    offsets: Sequence[Tuple[int, int]],
    start_char: int,
    end_char: int,
    *,
    name: str,
) -> Tuple[int, int]:
    """Map a half-open character span to a half-open token range."""

    if start_char < 0 or end_char <= start_char:
        raise TokenAlignmentError(f"invalid {name} character span {(start_char, end_char)}")
    hits = [
        i for i, (a, b) in enumerate(offsets)
        if b > a and b > start_char and a < end_char
    ]
    if not hits:
        raise TokenAlignmentError(
            f"{name} character span {(start_char, end_char)} maps to no token"
        )
    if hits != list(range(hits[0], hits[-1] + 1)):
        raise TokenAlignmentError(f"{name} maps to a non-contiguous token range")
    return hits[0], hits[-1] + 1


def _step_spans(
    response_text: str,
    steps_text: Sequence[str],
    offsets: Sequence[Tuple[int, int]],
    response_start_char: int,
    *,
    fail_on_unmatched: bool,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    token_ranges: List[Tuple[int, int]] = []
    char_spans: List[Tuple[int, int]] = []
    cursor = 0
    for step_i, raw_step in enumerate(steps_text):
        step = str(raw_step)
        idx = response_text.find(step, cursor)
        if idx < 0:
            if fail_on_unmatched:
                raise TokenAlignmentError(
                    f"step {step_i} is not a verbatim response substring after char {cursor}: "
                    f"{step!r}"
                )
            continue
        start = response_start_char + idx
        end = start + len(step)
        try:
            tok_start, tok_end = char_span_to_token_range(
                offsets, start, end, name=f"step[{step_i}]"
            )
        except TokenAlignmentError:
            if fail_on_unmatched:
                raise
            cursor = idx + len(step)
            continue
        token_ranges.append((tok_start, tok_end - 1))  # legacy inclusive convention
        char_spans.append((start, end))
        cursor = idx + len(step)
    return token_ranges, char_spans


def build_exact_trace_alignment(
    tokenizer,
    rendered_prompt: str,
    response_text: str,
    steps_text: Sequence[str] | None = None,
    *,
    prompt_token_ids: Any = None,
    prompt_attention_mask: Any = None,
    response_token_ids: Any = None,
    response_attention_mask: Any = None,
    question_text: str | None = None,
    fail_on_unmatched: bool = True,
) -> Dict[str, Any]:
    """Build and validate the canonical exact-generation token trace.

    When token IDs from generation are provided, both prompt and response are
    re-tokenized only for offsets and immediately compared to the supplied
    IDs.  A mismatch raises :class:`TokenAlignmentError`; it is never silently
    repaired by a second, different tokenization.
    """

    prompt = str(rendered_prompt)
    response = str(response_text)
    p_enc = _encode_segment(tokenizer, prompt, "prompt")
    r_enc = _encode_segment(tokenizer, response, "response")

    if prompt_token_ids is not None:
        supplied = _flat_ints(prompt_token_ids, "supplied_prompt_token_ids")
        _assert_same_tokens(supplied, p_enc["input_ids"], "prompt")
        p_ids = supplied
    else:
        p_ids = p_enc["input_ids"]
    if response_token_ids is not None:
        supplied = _flat_ints(response_token_ids, "supplied_response_token_ids")
        _assert_same_tokens(supplied, r_enc["input_ids"], "response")
        r_ids = supplied
    else:
        r_ids = r_enc["input_ids"]

    p_mask = (_flat_ints(prompt_attention_mask, "supplied_prompt_attention_mask")
              if prompt_attention_mask is not None else p_enc["attention_mask"])
    r_mask = (_flat_ints(response_attention_mask, "supplied_response_attention_mask")
              if response_attention_mask is not None else r_enc["attention_mask"])
    if len(p_mask) != len(p_ids) or len(r_mask) != len(r_ids):
        raise TokenAlignmentError("supplied attention mask length does not match token IDs")

    shift = len(prompt)
    offsets = list(p_enc["offset_mapping"]) + [
        (a + shift, b + shift) for a, b in r_enc["offset_mapping"]
    ]
    ids = list(p_ids) + list(r_ids)
    mask = list(p_mask) + list(r_mask)
    if len(ids) != len(offsets):
        raise TokenAlignmentError("combined token IDs and offsets have different lengths")

    if question_text is None:
        q_char = (-1, -1)
        q_tok = (-1, -1)
        question = ""
    else:
        question = str(question_text)
        q_start = prompt.rfind(question)
        if q_start < 0:
            raise TokenAlignmentError("target question is not present in the rendered prompt")
        q_char = (q_start, q_start + len(question))
        q_tok = char_span_to_token_range(offsets, *q_char, name="question")
        if q_tok[1] > len(p_ids):
            raise TokenAlignmentError("question token span escapes the prompt token span")

    if steps_text is None:
        steps = split_response_into_steps(response)
    else:
        steps = [str(x) for x in steps_text]
    step_ranges, step_char_spans = _step_spans(
        response,
        steps,
        offsets,
        len(prompt),
        fail_on_unmatched=fail_on_unmatched,
    )

    trace: Dict[str, Any] = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "rendered_prompt": prompt,
        "response_text": response,
        "question_text": question,
        "steps_text": steps,
        "prompt_token_ids": list(p_ids),
        "prompt_attention_mask": list(p_mask),
        "response_token_ids": list(r_ids),
        "response_attention_mask": list(r_mask),
        "full_input_ids": list(ids),
        "full_attention_mask": list(mask),
        "full_token_offsets": list(offsets),
        "input_ids": ids,
        "attention_mask": mask,
        "token_offsets": offsets,
        "prompt_char_span": (0, len(prompt)),
        "response_char_span": (len(prompt), len(prompt) + len(response)),
        "question_char_span": q_char,
        "question_token_range": q_tok,
        "response_token_range": (len(p_ids), len(ids)),
        "all_step_token_ranges": step_ranges,
        "all_step_char_spans": step_char_spans,
        "model_input_truncated": False,
        "original_input_token_count": len(ids),
    }
    assert_trace_alignment(trace)
    return trace


def truncate_trace_alignment(trace: Dict[str, Any], max_length: int | None) -> Dict[str, Any]:
    """Return a trace whose model-input fields match an exact truncation."""

    out = dict(trace)
    original = len(_flat_ints(trace["input_ids"], "input_ids"))
    if max_length is None or int(max_length) <= 0 or original <= int(max_length):
        out["model_input_truncated"] = False
        out["original_input_token_count"] = original
        assert_trace_alignment(out)
        return out

    limit = int(max_length)
    prompt_len = len(_flat_ints(trace["prompt_token_ids"], "prompt_token_ids"))
    if limit < prompt_len:
        raise TokenAlignmentError(
            f"max_length={limit} truncates the generation prompt ({prompt_len} tokens)"
        )
    out["input_ids"] = _flat_ints(trace["input_ids"], "input_ids")[:limit]
    out["attention_mask"] = _flat_ints(trace["attention_mask"], "attention_mask")[:limit]
    out["token_offsets"] = _offsets(trace["token_offsets"], "token_offsets")[:limit]
    out["response_token_range"] = (prompt_len, limit)
    out["model_input_truncated"] = True
    out["original_input_token_count"] = original
    assert_trace_alignment(out)
    return out


def attach_trace_time_axis(
    trace: Dict[str, Any],
    kept_steps: Sequence[int],
    kept_step_token_ranges: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    """Attach the step/time axis actually used by downstream feature arrays."""

    out = dict(trace)
    ks = [int(x) for x in _tolist(kept_steps)]
    kr = [(int(a), int(b)) for a, b in _tolist(kept_step_token_ranges)]
    if len(ks) != len(kr):
        raise TokenAlignmentError(
            f"kept_steps ({len(ks)}) and step_token_ranges ({len(kr)}) disagree"
        )
    out["kept_steps"] = ks
    out["step_token_ranges"] = kr
    out["time_axis_original_step_indices"] = list(ks)
    out["time_axis_positions"] = list(range(len(ks)))
    out["time_axis_token_ranges"] = list(kr)
    out["time_axis_length"] = len(ks)
    assert_trace_alignment(out)
    return out


def assert_trace_alignment(trace: Dict[str, Any]) -> None:
    """Fail fast if a trace no longer obeys the exact-token contract."""

    ids = _flat_ints(trace["input_ids"], "input_ids")
    mask = _flat_ints(trace["attention_mask"], "attention_mask")
    offsets = _offsets(trace["token_offsets"], "token_offsets")
    p_ids = _flat_ints(trace["prompt_token_ids"], "prompt_token_ids")
    p_mask = _flat_ints(trace["prompt_attention_mask"], "prompt_attention_mask")
    r_ids = _flat_ints(trace["response_token_ids"], "response_token_ids")
    if len(ids) != len(mask) or len(ids) != len(offsets):
        raise TokenAlignmentError("model input IDs, mask, and offsets must have equal length")
    if len(p_ids) != len(p_mask):
        raise TokenAlignmentError("prompt token IDs and prompt attention mask disagree")
    _assert_same_tokens(p_ids, ids[:len(p_ids)], "model-input prompt prefix")
    _assert_same_tokens(p_mask, mask[:len(p_mask)], "model-input prompt mask")

    response_range = tuple(int(x) for x in trace["response_token_range"])
    if response_range[0] != len(p_ids) or not (
        response_range[0] <= response_range[1] <= len(ids)
    ):
        raise TokenAlignmentError(f"invalid response_token_range {response_range}")
    used_response = ids[response_range[0]:response_range[1]]
    _assert_same_tokens(r_ids[:len(used_response)], used_response,
                        "model-input response prefix")

    full_ids = _flat_ints(trace.get("full_input_ids", p_ids + r_ids), "full_input_ids")
    full_mask = _flat_ints(
        trace.get("full_attention_mask", p_mask + [1] * len(r_ids)),
        "full_attention_mask",
    )
    full_offsets = _offsets(
        trace.get("full_token_offsets", trace["token_offsets"]),
        "full_token_offsets",
    )
    if len(full_ids) != len(full_mask) or len(full_ids) != len(full_offsets):
        raise TokenAlignmentError("full token IDs, mask, and offsets must have equal length")
    _assert_same_tokens(p_ids + r_ids, full_ids, "full prompt-response input")
    _assert_same_tokens(ids, full_ids[:len(ids)], "truncated model-input prefix")

    q_char = tuple(int(x) for x in trace.get("question_char_span", (-1, -1)))
    q_tok = tuple(int(x) for x in trace.get("question_token_range", (-1, -1)))
    if q_char != (-1, -1):
        if not (0 <= q_char[0] < q_char[1] <= len(trace["rendered_prompt"])):
            raise TokenAlignmentError(f"invalid question_char_span {q_char}")
        if not (0 <= q_tok[0] < q_tok[1] <= len(p_ids)):
            raise TokenAlignmentError(f"invalid question_token_range {q_tok}")

    steps = [str(x) for x in trace.get("steps_text", [])]
    all_ranges = [
        (int(a), int(b)) for a, b in _tolist(trace.get("all_step_token_ranges", []))
    ]
    all_char_spans = [
        (int(a), int(b)) for a, b in _tolist(trace.get("all_step_char_spans", []))
    ]
    if len(steps) != len(all_ranges) or len(steps) != len(all_char_spans):
        raise TokenAlignmentError(
            "step strings, full token ranges, and character spans must have equal length"
        )
    for a, b in all_ranges:
        if not (0 <= a <= b < len(full_ids)):
            raise TokenAlignmentError(f"full step token range {(a, b)} is outside full input")

    if "kept_steps" in trace or "step_token_ranges" in trace:
        ks = [int(x) for x in _tolist(trace.get("kept_steps", []))]
        kr = [(int(a), int(b)) for a, b in _tolist(trace.get("step_token_ranges", []))]
        if len(ks) != len(kr):
            raise TokenAlignmentError("kept step indices and token ranges disagree")
        for step_i, (a, b) in zip(ks, kr):
            if not (0 <= a <= b < len(ids)):
                raise TokenAlignmentError(f"kept step token range {(a, b)} is outside model input")
            if not 0 <= step_i < len(all_ranges) or (a, b) != all_ranges[step_i]:
                raise TokenAlignmentError(
                    f"kept step {step_i} range {(a, b)} disagrees with the full step axis"
                )
        if int(trace.get("time_axis_length", len(ks))) != len(ks):
            raise TokenAlignmentError("time_axis_length does not match kept_steps")

    if trace.get("token_uncertainty_token_range") is not None:
        a, b = (int(x) for x in trace["token_uncertainty_token_range"])
        if not (0 <= a < b <= len(ids)):
            raise TokenAlignmentError(
                f"invalid token_uncertainty_token_range {(a, b)}"
            )


def _object_vector(values: Sequence[Any]) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    out[:] = list(values)
    return out


def trace_records_to_npz(records: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Convert validated trace records to the shared NPZ artifact schema."""

    records = list(records)
    if not records:
        raise TokenAlignmentError("cannot serialize an empty trace record list")
    for record in records:
        assert_trace_alignment(record)

    payload: Dict[str, np.ndarray] = {
        "trace_schema_version": np.array(TRACE_SCHEMA_VERSION),
        "trace_token_add_special_tokens": np.array(False),
        "token_offset_convention": np.array(TOKEN_OFFSET_CONVENTION),
        "step_token_range_convention": np.array(STEP_TOKEN_RANGE_CONVENTION),
        "span_range_convention": np.array(SPAN_RANGE_CONVENTION),
        "hidden_state_token_semantics": np.array("h_i_after_reading_token_i"),
        "logit_prediction_semantics": np.array("logits_i_predict_token_i_plus_1"),
        "step_prediction_position_shift": np.array(-1, dtype=np.int8),
        "time_axis_kind": np.array("kept_step_index"),
        "prompts": _object_vector([r["rendered_prompt"] for r in records]),
        "responses": _object_vector([r["response_text"] for r in records]),
        "questions": _object_vector([r.get("question_text", "") for r in records]),
        "steps_text": _object_vector([list(r.get("steps_text", [])) for r in records]),
        "prompt_token_ids": _object_vector([
            np.asarray(r["prompt_token_ids"], dtype=np.int64) for r in records
        ]),
        "prompt_attention_mask": _object_vector([
            np.asarray(r["prompt_attention_mask"], dtype=np.int8) for r in records
        ]),
        "response_token_ids": _object_vector([
            np.asarray(r["response_token_ids"], dtype=np.int64) for r in records
        ]),
        "response_attention_mask": _object_vector([
            np.asarray(r["response_attention_mask"], dtype=np.int8) for r in records
        ]),
        "input_ids": _object_vector([
            np.asarray(r["input_ids"], dtype=np.int64) for r in records
        ]),
        "attention_mask": _object_vector([
            np.asarray(r["attention_mask"], dtype=np.int8) for r in records
        ]),
        "token_offsets": _object_vector([
            np.asarray(r["token_offsets"], dtype=np.int64) for r in records
        ]),
        "full_input_ids": _object_vector([
            np.asarray(r.get("full_input_ids", r["input_ids"]), dtype=np.int64)
            for r in records
        ]),
        "full_attention_mask": _object_vector([
            np.asarray(r.get("full_attention_mask", r["attention_mask"]), dtype=np.int8)
            for r in records
        ]),
        "full_token_offsets": _object_vector([
            np.asarray(r.get("full_token_offsets", r["token_offsets"]), dtype=np.int64)
            for r in records
        ]),
        "question_char_spans": np.asarray([
            r["question_char_span"] for r in records
        ], dtype=np.int64),
        "question_token_ranges": np.asarray([
            r["question_token_range"] for r in records
        ], dtype=np.int64),
        "response_char_spans": np.asarray([
            r["response_char_span"] for r in records
        ], dtype=np.int64),
        "response_token_ranges": np.asarray([
            r["response_token_range"] for r in records
        ], dtype=np.int64),
        "kept_steps": _object_vector([
            np.asarray(r.get("kept_steps", []), dtype=np.int32) for r in records
        ]),
        "step_token_ranges": _object_vector([
            np.asarray(r.get("step_token_ranges", []), dtype=np.int32).reshape(-1, 2)
            for r in records
        ]),
        "all_step_token_ranges": _object_vector([
            np.asarray(r.get("all_step_token_ranges", []), dtype=np.int32).reshape(-1, 2)
            for r in records
        ]),
        "all_step_char_spans": _object_vector([
            np.asarray(r.get("all_step_char_spans", []), dtype=np.int64).reshape(-1, 2)
            for r in records
        ]),
        "time_axis_original_step_indices": _object_vector([
            np.asarray(r.get("time_axis_original_step_indices", []), dtype=np.int32)
            for r in records
        ]),
        "time_axis_positions": _object_vector([
            np.asarray(r.get("time_axis_positions", []), dtype=np.int32)
            for r in records
        ]),
        "time_axis_token_ranges": _object_vector([
            np.asarray(r.get("time_axis_token_ranges", []), dtype=np.int32).reshape(-1, 2)
            for r in records
        ]),
        "time_axis_lengths": np.asarray([
            int(r.get("time_axis_length", len(r.get("kept_steps", [])))) for r in records
        ], dtype=np.int32),
        "prompt_token_counts": np.asarray([
            len(r["prompt_token_ids"]) for r in records
        ], dtype=np.int32),
        "input_token_counts": np.asarray([
            len(r["input_ids"]) for r in records
        ], dtype=np.int32),
        "original_input_token_counts": np.asarray([
            int(r.get("original_input_token_count", len(r["input_ids"]))) for r in records
        ], dtype=np.int32),
        "model_input_truncated": np.asarray([
            bool(r.get("model_input_truncated", False)) for r in records
        ], dtype=np.bool_),
    }

    have_generated = [r.get("generated_token_ids") is not None for r in records]
    if any(have_generated) and not all(have_generated):
        raise TokenAlignmentError("generated token IDs must be present for all or no records")
    payload["generated_token_ids_stored"] = np.array(all(have_generated))
    if all(have_generated):
        payload["generated_token_ids"] = _object_vector([
            np.asarray(r["generated_token_ids"], dtype=np.int64) for r in records
        ])
        payload["generation_terminal_token_ids"] = _object_vector([
            np.asarray(r.get("generation_terminal_token_ids", []), dtype=np.int64)
            for r in records
        ])

    have_tok_axis = [r.get("token_uncertainty_token_range") is not None for r in records]
    if any(have_tok_axis) and not all(have_tok_axis):
        raise TokenAlignmentError(
            "token uncertainty ranges must be present for all or no records"
        )
    payload["token_uncertainty_axis_stored"] = np.array(all(have_tok_axis))
    if all(have_tok_axis):
        payload["token_uncertainty_token_ranges"] = np.asarray([
            r["token_uncertainty_token_range"] for r in records
        ], dtype=np.int32)

    have_prompt_hidden = [r.get("prompt_hidden") is not None for r in records]
    if any(have_prompt_hidden) and not all(have_prompt_hidden):
        raise TokenAlignmentError("prompt hidden states must be present for all or no records")
    payload["prompt_hidden_stored"] = np.array(all(have_prompt_hidden))
    if all(have_prompt_hidden):
        layer_sets = [tuple(int(x) for x in r["prompt_hidden_layers"]) for r in records]
        if any(x != layer_sets[0] for x in layer_sets[1:]):
            raise TokenAlignmentError("prompt hidden layer sets differ across records")
        payload["prompt_hidden_layers"] = np.asarray(layer_sets[0], dtype=np.int32)
        payload["prompt_hidden"] = _object_vector([
            np.asarray(r["prompt_hidden"], dtype=np.float16) for r in records
        ])
    return payload


def find_step_token_ranges(
    tokenizer,
    prompt_text: str,
    response_text: str,
    steps_text: List[str] | None = None,
    *,
    fail_on_unmatched: bool = True,
) -> List[Tuple[int, int]]:
    """Return inclusive step token ranges on the exact no-special-token axis.

    This compatibility wrapper now uses :func:`build_exact_trace_alignment`;
    unlike the old implementation, an unlocatable step raises by default.
    """

    trace = build_exact_trace_alignment(
        tokenizer,
        prompt_text,
        response_text,
        steps_text,
        fail_on_unmatched=fail_on_unmatched,
    )
    return list(trace["all_step_token_ranges"])
