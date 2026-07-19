#!/usr/bin/env python3
"""Extract exact-axis attention traces for the faithful hypergraph path.

The legacy geometry archives are not valid inputs here: they retain response
hidden states and aggregated attention summaries, but not the full
prompt/response token axis or token-to-token attention rows.  This extractor
rebuilds the autoregressive axis as prompt IDs followed by visible response
IDs, derives every boundary from the corresponding fast-tokenizer offsets, and
writes one trace per sample.  Strict same-generator replay validates this axis
against stored generation artifacts; reconstructed or cross-model passes are
explicitly marked as observer traces.

Dense attention is intentionally a first implementation target because it is
auditable and exactly reproduces the original thresholding rule.  It is also
quadratic in sequence length.  The CLI estimates the forward-pass attention
output tensor lower bound (not peak eager memory) and refuses unexpectedly
large samples unless explicitly allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .data import (
    VERIFIED_MODEL_COMMIT_SOURCES,
    TRACE_CONTRACT,
    commit_hashes_match,
    is_immutable_commit_hash,
    model_identity_matches,
)
from utils.step_boundaries import trim_trailing_generation_tokens


PLAIN_PROMPT_TEMPLATE = "Problem: {question}\n\nSolution:\n\n"

EXTRACTION_SCOPE_KEYS = (
    "input_sha256",
    "input_num_rows",
    "pre_shard_num_rows",
    "selected_num_rows",
    "requested_limit",
    "num_shards",
    "shard_index",
    "skip_invalid",
    "max_seq_len",
    "max_attention_gib",
    "allow_large_attention",
)
EXTRACTION_RUNTIME_ONLY_KEYS = (
    "input_path",
    *EXTRACTION_SCOPE_KEYS,
    "device",
    "device_map",
    "max_memory",
    "input_device",
    "low_cpu_mem_usage",
    "offload_folder",
    "archive_compression",
    "extraction_fingerprint",
    "extraction_scope_fingerprint",
)


def _row_alias(
    row: Mapping[str, Any],
    keys: Sequence[str],
    *,
    row_index: int,
    field_name: str,
) -> Tuple[Any, Optional[str]]:
    """Return one JSON field while rejecting ambiguous aliases."""

    present = [key for key in keys if key in row and row[key] is not None]
    if len(present) > 1:
        raise ValueError(
            f"row {row_index} has ambiguous {field_name} aliases {present}; "
            "provide exactly one canonical field"
        )
    if not present:
        return None, None
    key = present[0]
    return row[key], key


def _binary_record_scalar(value: Any, *, row_index: int, field_name: str) -> float:
    """Parse a JSON binary label without string/truthiness coercion."""

    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if np.isfinite(number) and number in (0.0, 1.0):
            return number
    raise ValueError(
        f"row {row_index} {field_name} must be an actual bool or numeric 0/1, "
        f"got {value!r}"
    )


def _integer_token_vector(value: Any, *, name: str, row_index: int) -> Optional[List[int]]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"row {row_index} {name} must contain exactly one token sequence")
    if not np.issubdtype(array.dtype, np.integer):
        if (
            not np.issubdtype(array.dtype, np.number)
            or not np.isfinite(array).all()
            or not np.equal(array, np.floor(array)).all()
        ):
            raise ValueError(f"row {row_index} {name} must contain integer token IDs")
    return np.asarray(array, dtype=np.int64).tolist()


def parse_indices(text: str, size: int, *, name: str) -> Tuple[int, ...]:
    """Parse ``all`` or a comma-separated, zero-based index selection."""

    if str(text).strip().lower() in {"", "all", "*"}:
        return tuple(range(int(size)))
    values = tuple(int(piece.strip()) for piece in str(text).split(",") if piece.strip())
    if not values:
        raise ValueError(f"{name} selected no indices")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate indices")
    invalid = [value for value in values if value < 0 or value >= int(size)]
    if invalid:
        raise ValueError(f"{name} has out-of-range indices {invalid}; valid range is 0..{size - 1}")
    return values


def load_processbench_rows(path: str) -> List[Mapping[str, Any]]:
    """Load a ProcessBench-style JSON array, ``{"data": [...]}``, or JSONL."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"input dataset does not exist: {source}")
    raw = source.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if not stripped:
        return []
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict) and isinstance(value.get("data"), list):
            rows = value["data"]
        elif value is not None:
            rows = [value]
        else:
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("every dataset row must be a JSON object")
    return list(rows)


def canonical_record(row: Mapping[str, Any], index: int) -> Dict[str, Any]:
    """Normalize a ProcessBench row without inventing finer-grained labels."""

    question, _ = _row_alias(
        row,
        ("problem", "question"),
        row_index=index,
        field_name="question",
    )
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"row {index} has no non-empty problem/question")
    raw_steps = row.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise ValueError(f"row {index} has no step sequence")
    steps = [str(step) for step in raw_steps]
    if not steps or any(not step.strip() for step in steps):
        raise ValueError(
            f"row {index} contains an empty reasoning step; refusing to drop it because "
            "that would silently renumber gold_step"
        )
    raw_gold_step, _ = _row_alias(
        row,
        ("label", "gold_step", "gold_error_step", "first_error_step"),
        row_index=index,
        field_name="gold step",
    )
    if raw_gold_step is None:
        raise ValueError(
            f"row {index} has no gold step; refusing to invent -1 (fully correct)"
        )
    if isinstance(raw_gold_step, (bool, np.bool_)):
        raise ValueError(f"row {index} gold step must be an integer, not boolean")
    try:
        gold_number = float(raw_gold_step)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {index} gold step is not numeric: {raw_gold_step!r}") from exc
    if not np.isfinite(gold_number) or gold_number != math.floor(gold_number):
        raise ValueError(f"row {index} gold step must be an integer, got {raw_gold_step!r}")
    gold_step = int(gold_number)
    if gold_step < -1 or gold_step >= len(steps):
        raise ValueError(
            f"row {index} gold step {gold_step} is invalid for {len(steps)} steps"
        )
    question = question.strip()
    sample_id_value, _ = _row_alias(
        row, ("id", "sample_id"), row_index=index, field_name="sample id"
    )
    sample_id = str(index if sample_id_value is None else sample_id_value)
    explicit_problem_id, _ = _row_alias(
        row,
        ("problem_id", "question_id"),
        row_index=index,
        field_name="problem id",
    )
    problem_id = (
        str(explicit_problem_id)
        if explicit_problem_id is not None
        else "question-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:20]
    )
    split_value, _ = _row_alias(
        row, ("split", "partition"), row_index=index, field_name="split"
    )
    generator_model, _ = _row_alias(
        row,
        ("generator", "generator_model", "source_model", "model"),
        row_index=index,
        field_name="generator model",
    )
    generator_model_commit, _ = _row_alias(
        row,
        ("generator_model_commit", "generator_commit_hash", "model_commit_hash"),
        row_index=index,
        field_name="generator commit",
    )
    rendered_prompt = row.get("rendered_prompt")
    response_text, _ = _row_alias(
        row,
        ("response_text", "response"),
        row_index=index,
        field_name="response text",
    )
    if rendered_prompt is not None and not isinstance(rendered_prompt, str):
        raise ValueError(f"row {index} rendered_prompt must be a string")
    if response_text is not None and not isinstance(response_text, str):
        raise ValueError(f"row {index} response_text/response must be a string")
    response_value, response_key = _row_alias(
        row,
        ("response_y", "response_label", "is_hallucinated", "is_incorrect"),
        row_index=index,
        field_name="response label",
    )
    if response_value is not None and row.get("is_correct") is not None:
        raise ValueError(
            f"row {index} has both {response_key!r} and 'is_correct'; provide one "
            "canonical response label"
        )
    if response_value is not None:
        response_y = _binary_record_scalar(
            response_value, row_index=index, field_name=response_key or "response_y"
        )
    elif row.get("is_correct") is not None:
        response_y = 1.0 - _binary_record_scalar(
            row["is_correct"], row_index=index, field_name="is_correct"
        )
    else:
        response_y = float(gold_step >= 0)
    derived_response_y = float(gold_step >= 0)
    if response_y != derived_response_y:
        raise ValueError(
            f"row {index} response label {response_y:g} conflicts with "
            f"gold_step={gold_step}"
        )

    response_token_ids, _ = _row_alias(
        row,
        ("response_token_ids", "generated_token_ids"),
        row_index=index,
        field_name="response token ids",
    )
    return {
        "sample_id": sample_id,
        "problem_id": problem_id,
        "question": question,
        "steps": steps,
        "gold_step": gold_step,
        "response_y": response_y,
        "split": None if split_value is None else str(split_value).lower(),
        "generator_model": None if generator_model is None else str(generator_model),
        "generator_model_commit": (
            None if generator_model_commit is None else str(generator_model_commit)
        ),
        "rendered_prompt": rendered_prompt,
        "response_text": response_text,
        "prompt_token_ids": _integer_token_vector(
            row.get("prompt_token_ids"), name="prompt_token_ids", row_index=index
        ),
        "response_token_ids": _integer_token_vector(
            response_token_ids,
            name="response_token_ids",
            row_index=index,
        ),
    }


def render_prompt(tokenizer, question: str, prompt_style: str) -> str:
    if prompt_style == "plain":
        return PLAIN_PROMPT_TEMPLATE.format(question=question)
    if prompt_style == "chat":
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("the tokenizer does not expose apply_chat_template")
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(f"unsupported prompt_style {prompt_style!r}")


def response_and_char_spans(
    steps: Sequence[str],
    prompt_chars: int,
    *,
    response_text: Optional[str] = None,
    separator: str = "\n\n",
) -> Tuple[str, np.ndarray]:
    """Return exact response text and sequential, absolute step character spans."""

    response = separator.join(str(step) for step in steps) if response_text is None else str(response_text)
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for step_index, raw_step in enumerate(steps):
        step = str(raw_step)
        start_in_response = response.find(step, cursor)
        if start_in_response < 0:
            raise ValueError(
                f"step {step_index} is not a verbatim, ordered substring of response_text"
            )
        stop_in_response = start_in_response + len(step)
        spans.append(
            (int(prompt_chars) + start_in_response, int(prompt_chars) + stop_in_response)
        )
        cursor = stop_in_response
    return response, np.asarray(spans, dtype=np.int64)


def _content_token_offsets(offsets: np.ndarray) -> Iterable[Tuple[int, int, int]]:
    for token_index, (start, stop) in enumerate(np.asarray(offsets, np.int64)):
        start_i, stop_i = int(start), int(stop)
        if stop_i > start_i:  # special tokens conventionally use (0, 0)
            yield token_index, start_i, stop_i


def _assert_no_crossing_boundary(offsets: np.ndarray, boundary: int, *, label: str) -> None:
    crossing = [
        token_index
        for token_index, start, stop in _content_token_offsets(offsets)
        if start < int(boundary) < stop
    ]
    if crossing:
        raise ValueError(
            f"token(s) {crossing} cross the {label} character boundary {boundary}; "
            "the token axis is not exactly separable"
        )


def char_spans_to_token_ranges(
    offsets: np.ndarray,
    response_char_start: int,
    step_char_spans: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """Map character spans to the same full token axis used by the model.

    Boundaries crossed by a tokenizer token are rejected rather than rounded.
    This is stricter than the legacy prefix-length arithmetic and prevents BOS
    or prompt/response shifts from silently corrupting graph labels.
    """

    offsets = np.asarray(offsets, np.int64)
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError("offset_mapping must have shape (tokens, 2)")
    _assert_no_crossing_boundary(
        offsets, int(response_char_start), label="prompt/response"
    )
    for step_index, (start, stop) in enumerate(np.asarray(step_char_spans, np.int64)):
        _assert_no_crossing_boundary(offsets, int(start), label=f"step-{step_index} start")
        _assert_no_crossing_boundary(offsets, int(stop), label=f"step-{step_index} end")

    response_tokens = [
        token_index
        for token_index, start, stop in _content_token_offsets(offsets)
        if start >= int(response_char_start)
    ]
    if not response_tokens:
        raise ValueError("response maps to no model token")
    response_idx = int(response_tokens[0])

    ranges: List[Tuple[int, int]] = []
    for step_index, (char_start, char_stop) in enumerate(
        np.asarray(step_char_spans, np.int64)
    ):
        members = [
            token_index
            for token_index, start, stop in _content_token_offsets(offsets)
            if start >= int(char_start) and stop <= int(char_stop)
        ]
        if not members:
            raise ValueError(f"step {step_index} maps to no complete model token")
        if members != list(range(members[0], members[-1] + 1)):
            raise ValueError(f"step {step_index} token range is not contiguous")
        ranges.append((members[0], members[-1] + 1))
    token_ranges = np.asarray(ranges, dtype=np.int64)
    if np.any(token_ranges[:, 0] < response_idx):
        raise ValueError("a reasoning step begins before response_idx")
    if len(token_ranges) > 1 and np.any(token_ranges[1:, 0] < token_ranges[:-1, 1]):
        raise ValueError("mapped step token ranges overlap")
    return response_idx, token_ranges


def tokenize_exact(
    tokenizer,
    prompt: str,
    steps: Sequence[str],
    *,
    add_special_tokens: Optional[bool] = True,
    response_text: Optional[str] = None,
    prompt_token_ids: Optional[Sequence[int]] = None,
    response_token_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Rebuild the generation axis as prompt tokens followed by response tokens.

    Segment-wise tokenization prevents a BPE merge across the generation
    boundary.  If stored generation IDs exist, they are checked exactly.
    """

    response, char_spans = response_and_char_spans(
        steps, len(prompt), response_text=response_text
    )
    full_text = prompt + response
    prompt_flags = (
        (False, True)
        if add_special_tokens is None and prompt_token_ids is not None
        else (bool(add_special_tokens),)
    )
    prompt_encoded = None
    expected_prompt_ids = (
        None
        if prompt_token_ids is None
        else np.asarray(prompt_token_ids, dtype=np.int64).reshape(-1)
    )
    for prompt_flag in prompt_flags:
        candidate = tokenizer(
            prompt,
            add_special_tokens=prompt_flag,
            return_attention_mask=True,
            return_offsets_mapping=True,
            truncation=False,
        )
        candidate_ids = np.asarray(candidate["input_ids"], dtype=np.int64).reshape(-1)
        if expected_prompt_ids is None or np.array_equal(candidate_ids, expected_prompt_ids):
            prompt_encoded = candidate
            prompt_add_special_tokens = prompt_flag
            break
    if prompt_encoded is None:
        raise ValueError(
            "stored prompt_token_ids match neither add_special_tokens=False nor True"
        )
    response_encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
        truncation=False,
    )
    if "offset_mapping" not in prompt_encoded or "offset_mapping" not in response_encoded:
        raise ValueError("a fast tokenizer with offset_mapping support is required")
    prompt_ids = np.asarray(prompt_encoded["input_ids"], dtype=np.int64).reshape(-1)
    response_ids = np.asarray(response_encoded["input_ids"], dtype=np.int64).reshape(-1)
    if expected_prompt_ids is not None and not np.array_equal(
        prompt_ids, expected_prompt_ids
    ):
        raise ValueError("stored prompt_token_ids do not match replay tokenization")
    expected_response_ids = (
        None
        if response_token_ids is None
        else np.asarray(response_token_ids, dtype=np.int64).reshape(-1)
    )
    if expected_response_ids is not None and not np.array_equal(
        response_ids, expected_response_ids
    ):
        raise ValueError(
            "stored response_token_ids do not match the text offset tokenization; "
            "the stored IDs remain authoritative, but this sample cannot receive exact "
            "character/step ranges without an explicit ID-to-character alignment"
        )
    # Supplied generation IDs are the source of truth.  Re-tokenization is used
    # only to prove that its offset mapping addresses that exact same axis.
    if expected_prompt_ids is not None:
        prompt_ids = expected_prompt_ids
    if expected_response_ids is not None:
        response_ids = expected_response_ids
    token_ids = np.concatenate([prompt_ids, response_ids])
    attention_mask = np.concatenate(
        [
            np.asarray(prompt_encoded["attention_mask"], dtype=np.int64).reshape(-1),
            np.asarray(response_encoded["attention_mask"], dtype=np.int64).reshape(-1),
        ]
    )
    prompt_offsets = np.asarray(prompt_encoded["offset_mapping"], dtype=np.int64).reshape(-1, 2)
    response_offsets = np.asarray(response_encoded["offset_mapping"], dtype=np.int64).reshape(-1, 2)
    response_offsets = response_offsets + np.asarray([len(prompt), len(prompt)], np.int64)
    offsets = np.concatenate([prompt_offsets, response_offsets], axis=0)
    mapped_response_idx, step_ranges = char_spans_to_token_ranges(
        offsets, len(prompt), char_spans
    )
    response_idx = int(len(prompt_ids))
    if mapped_response_idx != response_idx:
        raise ValueError("response_idx disagrees with the prompt/response token concatenation")
    return {
        "full_text": full_text,
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offsets,
        "response_idx": response_idx,
        "step_ranges": step_ranges,
        "prompt_add_special_tokens": bool(prompt_add_special_tokens),
    }


def estimate_attention_gib(
    num_layers: int, num_heads: int, sequence_length: int, bytes_per_value: int
) -> float:
    return estimate_attention_block_gib(
        num_layers,
        num_heads,
        sequence_length,
        sequence_length,
        bytes_per_value,
    )


def estimate_attention_block_gib(
    num_layers: int,
    num_heads: int,
    query_length: int,
    key_length: int,
    bytes_per_value: int,
) -> float:
    return (
        int(num_layers)
        * int(num_heads)
        * int(query_length)
        * int(key_length)
        * int(bytes_per_value)
        / float(1024**3)
    )


def save_trace_archive(
    destination: Path, payload: Mapping[str, Any], *, compression: str
) -> None:
    """Atomically save a trace, optionally avoiding costly ZIP compression."""

    if compression not in {"compressed", "none"}:
        raise ValueError("archive compression must be 'compressed' or 'none'")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        if compression == "compressed":
            np.savez_compressed(stream, **payload)
        else:
            np.savez(stream, **payload)
    os.replace(temporary, destination)


def _write_extraction_manifest(
    destination: Path,
    *,
    extraction_config: Mapping[str, Any],
    extraction_fingerprint: str,
    extraction_scope_fingerprint: str,
    traces: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically checkpoint extraction progress after every completed row."""

    payload = {
        "extraction_config": dict(extraction_config),
        "extraction_fingerprint": str(extraction_fingerprint),
        "extraction_scope_fingerprint": str(extraction_scope_fingerprint),
        "chunk_equivalence_policy": "per_trace_prefix",
        "traces": list(traces),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _safe_stem(value: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return f"{index:06d}_{cleaned[:80] or 'sample'}"


def generator_matches_model(generator: str, model_name: str) -> bool:
    """Conservative name match for local paths versus dataset generator tags."""

    return model_identity_matches(generator, model_name)


def resolve_loaded_commit(
    requested_commit: Optional[str],
    model_commit: Optional[str],
    tokenizer_commit: Optional[str],
) -> Optional[str]:
    """Return the loaded immutable revision and reject conflicting evidence."""

    values = [
        str(value).strip()
        for value in (requested_commit, model_commit, tokenizer_commit)
        if value not in (None, "")
    ]
    for value in values:
        if not is_immutable_commit_hash(value):
            raise ValueError(
                f"model revision {value!r} is not an immutable hexadecimal commit hash"
            )
    lowered = [value.lower() for value in values]
    for left_index, left in enumerate(lowered):
        for right in lowered[left_index + 1 :]:
            if not commit_hashes_match(left, right):
                raise ValueError(
                    f"requested/loaded model revisions conflict: {sorted(set(values))}"
                )
    if model_commit not in (None, ""):
        return str(model_commit).strip()
    if tokenizer_commit not in (None, ""):
        return str(tokenizer_commit).strip()
    if requested_commit not in (None, ""):
        return str(requested_commit).strip()
    return None


def classify_model_commit_source(
    *,
    is_local_model: bool,
    requested_commit: Optional[str],
    model_commit: Optional[str],
    tokenizer_commit: Optional[str],
) -> str:
    """Describe whether the recorded revision was resolved or merely declared."""

    has_model_commit = model_commit not in (None, "")
    has_tokenizer_commit = tokenizer_commit not in (None, "")
    if is_local_model:
        if has_model_commit:
            return "local_model_metadata_commit"
        if requested_commit not in (None, ""):
            return "local_declared_commit"
        if has_tokenizer_commit:
            return "local_tokenizer_metadata_only"
        return "unavailable"
    if has_model_commit:
        return "remote_resolved_model_commit"
    if requested_commit not in (None, ""):
        return "remote_pinned_requested_commit"
    if has_tokenizer_commit:
        return "remote_tokenizer_metadata_only"
    return "unavailable"


def require_exact_replay_inputs(record: Mapping[str, Any]) -> None:
    """Reject a same-generator claim unless the original token axis is stored."""

    required = (
        "generator_model",
        "rendered_prompt",
        "response_text",
        "prompt_token_ids",
        "response_token_ids",
    )
    missing = [key for key in required if record.get(key) is None]
    if missing:
        raise ValueError(
            "same_generator replay requires stored "
            + ", ".join(missing)
            + "; use --replay_mode observer for reconstructed teacher forcing"
        )


def prepare_empty_output_dir(path: str | Path) -> Path:
    """Create an extraction directory, refusing any pre-existing contents.

    A stale NPZ from another model, dataset slice, or ``--limit`` value is a
    scientifically dangerous input because the training loader discovers files
    independently of the extraction manifest.  A new extraction therefore gets
    a new empty directory; this helper never deletes old artifacts.
    """

    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing non-empty extraction directory {output_dir}; choose a new empty "
            "directory so stale traces cannot be mixed into this run"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extraction_code_sha256() -> Dict[str, str]:
    """Hash the source files that define extraction and trace semantics."""

    repository = Path(__file__).resolve().parents[2]
    package = Path(__file__).resolve().parent
    paths = [*sorted(package.glob("*.py")), repository / "utils" / "step_boundaries.py"]
    hashes: Dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required extraction source is missing: {path}")
        hashes[path.relative_to(repository).as_posix()] = _sha256_file(path)
    return hashes


def parse_max_memory(value: Optional[str]) -> Optional[Dict[Any, Any]]:
    """Parse ``0=22GiB,1=22GiB,cpu=64GiB`` for HF Accelerate dispatch."""

    if value is None or not str(value).strip():
        return None
    result: Dict[Any, Any] = {}
    for raw_item in str(value).split(","):
        if "=" not in raw_item:
            raise ValueError(
                "max_memory entries must use device=value, for example 0=22GiB"
            )
        raw_key, raw_limit = (part.strip() for part in raw_item.split("=", 1))
        key: Any = int(raw_key) if raw_key.isdigit() else raw_key.lower()
        if key != "cpu" and not isinstance(key, int):
            raise ValueError("max_memory devices must be CUDA indices or 'cpu'")
        if key in result:
            raise ValueError(f"duplicate max_memory device {raw_key!r}")
        if not re.fullmatch(r"[1-9][0-9]*(?:\.[0-9]+)?(?:GiB|MiB|GB|MB)", raw_limit):
            raise ValueError(
                f"invalid max_memory value {raw_limit!r}; use values such as 22GiB"
            )
        result[key] = raw_limit
    return result


def select_dataset_shard(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: Optional[int],
    num_shards: int,
    shard_index: int,
) -> Tuple[List[Tuple[int, Mapping[str, Any]]], int]:
    """Select a deterministic modulo shard while preserving original indices."""

    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard count/index")
    indexed = list(enumerate(rows))
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("limit must be positive")
        indexed = indexed[: int(limit)]
    pre_shard_count = len(indexed)
    return (
        [
            item
            for position, item in enumerate(indexed)
            if position % int(num_shards) == int(shard_index)
        ],
        pre_shard_count,
    )


def fingerprint_extraction_config(
    extraction_config: Mapping[str, Any],
) -> Tuple[str, str, str, str]:
    """Return method/scope fingerprints and their canonical JSON records."""

    missing = [key for key in EXTRACTION_SCOPE_KEYS if key not in extraction_config]
    if missing:
        raise ValueError(f"extraction config lacks scope keys: {missing}")
    scope_config = {
        key: extraction_config[key] for key in EXTRACTION_SCOPE_KEYS
    }
    scope_json = json.dumps(scope_config, sort_keys=True, separators=(",", ":"))
    scope_fingerprint = hashlib.sha256(scope_json.encode("utf-8")).hexdigest()

    method_config = dict(extraction_config)
    for key in EXTRACTION_RUNTIME_ONLY_KEYS:
        method_config.pop(key, None)
    method_json = json.dumps(method_config, sort_keys=True, separators=(",", ":"))
    method_fingerprint = hashlib.sha256(method_json.encode("utf-8")).hexdigest()
    return method_fingerprint, scope_fingerprint, method_json, scope_json


def _torch_dtype(torch, name: str, device: str):
    if name == "auto":
        if str(device).startswith("cuda"):
            supports_bf16 = bool(
                hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            )
            return torch.bfloat16 if supports_bf16 else torch.float16
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _model_input_device(model, fallback: str) -> str:
    """Return the actual embedding device after optional Accelerate dispatch."""

    try:
        device = model.get_input_embeddings().weight.device
        if str(device) != "meta":
            return str(device)
    except (AttributeError, StopIteration):
        pass
    return str(fallback)


def _resolved_model_device_map(model, fallback: str) -> Dict[str, str]:
    """Canonicalize actual model placement for numerical provenance."""

    mapping = getattr(model, "hf_device_map", None)
    if isinstance(mapping, Mapping) and mapping:
        return {
            str(key): str(value)
            for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
        }
    return {"model": str(fallback)}


def extract_trace(
    model,
    torch,
    tokenized: Mapping[str, Any],
    *,
    device: str,
    attention_layers: Sequence[int],
    attention_heads: Sequence[int],
    activation_layer: Optional[int],
    storage_dtype: str,
    query_chunk_size: int = 0,
) -> Dict[str, np.ndarray]:
    """Run teacher forcing and select auditable trace tensors.

    ``query_chunk_size=0`` performs the original full-sequence forward.  A
    positive value streams prefix chunks through the model KV cache and writes
    their causal attention rows into one CPU tensor.  The latter reduces the
    returned-attention peak from ``O(N^2)`` to ``O(chunk*N)`` while preserving
    the same full token axis in the saved artifact.
    """

    token_ids = np.asarray(tokenized["token_ids"], dtype=np.int64).reshape(-1)
    mask_values = np.asarray(tokenized["attention_mask"], dtype=np.int64).reshape(-1)
    if token_ids.shape != mask_values.shape or not len(token_ids):
        raise ValueError("token_ids and attention_mask must be aligned non-empty vectors")
    n_tokens = int(len(token_ids))
    want_hidden = activation_layer is not None
    storage_torch_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[storage_dtype]
    chunk_size = int(query_chunk_size)
    if chunk_size < 0:
        raise ValueError("query_chunk_size cannot be negative")

    if chunk_size == 0 or chunk_size >= n_tokens:
        input_ids = torch.as_tensor(token_ids, dtype=torch.long, device=device)[None]
        attention_mask = torch.as_tensor(
            mask_values, dtype=torch.long, device=device
        )[None]
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=want_hidden,
                use_cache=False,
                return_dict=True,
            )
        if output.attentions is None:
            raise RuntimeError(
                "model returned no attentions; load it with attn_implementation='eager'"
            )
        selected_attention = torch.stack(
            [
                output.attentions[layer][0, list(attention_heads)]
                .detach()
                .to("cpu", dtype=storage_torch_dtype)
                for layer in attention_layers
            ],
            dim=0,
        ).numpy()
        result: Dict[str, np.ndarray] = {"attention": selected_attention}
        if activation_layer is not None:
            if output.hidden_states is None or not 0 <= int(activation_layer) < len(
                output.hidden_states
            ):
                raise ValueError(
                    f"activation_layer {activation_layer} is outside returned hidden states"
                )
            result["activation"] = (
                output.hidden_states[int(activation_layer)][0]
                .detach()
                .to("cpu", dtype=storage_torch_dtype)
                .numpy()
            )
        del output
        return result

    storage_numpy_dtype = np.float16 if storage_dtype == "float16" else np.float32
    selected_attention = np.zeros(
        (len(attention_layers), len(attention_heads), n_tokens, n_tokens),
        dtype=storage_numpy_dtype,
    )
    selected_activation: Optional[np.ndarray] = None
    past_key_values = None
    for start in range(0, n_tokens, chunk_size):
        stop = min(n_tokens, start + chunk_size)
        chunk_ids = torch.as_tensor(
            token_ids[start:stop], dtype=torch.long, device=device
        )[None]
        prefix_mask = torch.as_tensor(
            mask_values[:stop], dtype=torch.long, device=device
        )[None]
        with torch.inference_mode():
            output = model(
                input_ids=chunk_ids,
                attention_mask=prefix_mask,
                past_key_values=past_key_values,
                output_attentions=True,
                output_hidden_states=want_hidden,
                use_cache=True,
                return_dict=True,
            )
        if output.attentions is None or output.past_key_values is None:
            raise RuntimeError(
                "chunked extraction requires returned attentions and past_key_values"
            )
        query_length = stop - start
        for output_layer, model_layer in enumerate(attention_layers):
            block = output.attentions[model_layer][0, list(attention_heads)]
            if tuple(block.shape[-2:]) != (query_length, stop):
                raise RuntimeError(
                    "cached attention block has unexpected query/key axes: "
                    f"expected {(query_length, stop)}, got {tuple(block.shape[-2:])}"
                )
            selected_attention[output_layer, :, start:stop, :stop] = (
                block.detach()
                .to("cpu", dtype=storage_torch_dtype)
                .numpy()
            )
        if activation_layer is not None:
            if output.hidden_states is None or not 0 <= int(activation_layer) < len(
                output.hidden_states
            ):
                raise ValueError(
                    f"activation_layer {activation_layer} is outside returned hidden states"
                )
            activation_block = (
                output.hidden_states[int(activation_layer)][0]
                .detach()
                .to("cpu", dtype=storage_torch_dtype)
                .numpy()
            )
            if activation_block.shape[0] != query_length:
                raise RuntimeError("cached hidden-state block has an unexpected token axis")
            if selected_activation is None:
                selected_activation = np.zeros(
                    (n_tokens, *activation_block.shape[1:]), dtype=storage_numpy_dtype
                )
            selected_activation[start:stop] = activation_block
        past_key_values = output.past_key_values
        del output

    result = {"attention": selected_attention}
    if selected_activation is not None:
        result["activation"] = selected_activation
    del past_key_values
    return result


def verify_chunked_equivalence(
    model,
    torch,
    tokenized: Mapping[str, Any],
    *,
    device: str,
    attention_layers: Sequence[int],
    attention_heads: Sequence[int],
    query_chunk_size: int,
    activation_layer: Optional[int] = None,
    prefix_tokens: int = 192,
    atol: float = 1e-4,
    topology_threshold: float = 0.01,
) -> Dict[str, Any]:
    """Compare cached chunks with a full forward on one real-model prefix."""

    n_tokens = int(len(tokenized["token_ids"]))
    chunk_size = int(query_chunk_size)
    if chunk_size <= 0 or n_tokens <= chunk_size:
        return {"status": "not_needed", "tokens": min(n_tokens, chunk_size)}
    verify_tokens = min(n_tokens, max(int(prefix_tokens), chunk_size + 1))
    prefix = {
        "token_ids": np.asarray(tokenized["token_ids"])[:verify_tokens],
        "attention_mask": np.asarray(tokenized["attention_mask"])[:verify_tokens],
    }
    common = {
        "model": model,
        "torch": torch,
        "tokenized": prefix,
        "device": device,
        "attention_layers": attention_layers,
        "attention_heads": attention_heads,
        "activation_layer": activation_layer,
        "storage_dtype": "float32",
    }
    full_trace = extract_trace(**common, query_chunk_size=0)
    chunked_trace = extract_trace(**common, query_chunk_size=chunk_size)
    full = full_trace["attention"]
    chunked = chunked_trace["attention"]
    if full.shape != chunked.shape or not np.isfinite(full).all() or not np.isfinite(
        chunked
    ).all():
        raise RuntimeError("chunk equivalence produced invalid or misaligned attention tensors")
    attention_max_abs = float(np.max(np.abs(full - chunked), initial=0.0))
    activation_max_abs = 0.0
    if activation_layer is not None:
        full_activation = full_trace.get("activation")
        chunked_activation = chunked_trace.get("activation")
        if (
            full_activation is None
            or chunked_activation is None
            or full_activation.shape != chunked_activation.shape
            or not np.isfinite(full_activation).all()
            or not np.isfinite(chunked_activation).all()
        ):
            raise RuntimeError(
                "chunk equivalence produced invalid or misaligned activation tensors"
            )
        activation_max_abs = float(
            np.max(np.abs(full_activation - chunked_activation), initial=0.0)
        )
    max_abs = max(attention_max_abs, activation_max_abs)
    topology_disagreements = int(
        np.count_nonzero(
            (full > float(topology_threshold))
            != (chunked > float(topology_threshold))
        )
    )
    result = {
        "status": "prefix_pass",
        "tokens": int(verify_tokens),
        "query_chunk_size": chunk_size,
        "max_abs_error": max_abs,
        "attention_max_abs_error": attention_max_abs,
        "activation_max_abs_error": activation_max_abs,
        "atol": float(atol),
        "topology_threshold": float(topology_threshold),
        "topology_disagreements": topology_disagreements,
    }
    if max_abs > float(atol) or topology_disagreements:
        result["status"] = "fail"
        raise RuntimeError(
            "cached query chunks failed the real-model equivalence gate: "
            f"max_abs={max_abs:.3g} (atol={atol}), "
            f"threshold_disagreements={topology_disagreements}"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="ProcessBench JSON/JSONL file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True, help="Hugging Face model name or path")
    parser.add_argument(
        "--model_commit_hash",
        default=None,
        help=(
            "immutable hexadecimal revision that pins remote model/tokenizer loading; "
            "for a local checkpoint it is only an unverified declaration"
        ),
    )
    parser.add_argument("--prompt_style", choices=("plain", "chat"), default="plain")
    parser.add_argument(
        "--replay_mode",
        choices=("same_generator", "observer"),
        default="same_generator",
        help=(
            "same_generator requires matching generator plus stored prompt/response text "
            "and token IDs; observer permits reconstructed/cross-model teacher forcing"
        ),
    )
    parser.add_argument("--attention_layers", default="all", help="zero-based block indices")
    parser.add_argument("--attention_heads", default="all", help="zero-based head indices")
    parser.add_argument(
        "--activation_layer",
        type=int,
        default=None,
        help="optional hidden_states index (embedding=0, first block=1)",
    )
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument(
        "--max_attention_gib",
        type=float,
        default=24.0,
        help=(
            "refuse a sample if the dense attention-output lower bound exceeds this; "
            "this is not a peak-memory estimate"
        ),
    )
    parser.add_argument("--allow_large_attention", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--device_map",
        "--device-map",
        dest="device_map",
        choices=("none", "auto", "balanced", "balanced_low_0", "sequential"),
        default="none",
        help="optional Hugging Face Accelerate model sharding across visible GPUs",
    )
    parser.add_argument(
        "--max_memory",
        "--max-memory",
        dest="max_memory",
        default=None,
        help="per-device dispatch caps, e.g. 0=22GiB,1=22GiB,cpu=64GiB",
    )
    parser.add_argument(
        "--offload_folder",
        "--offload-folder",
        dest="offload_folder",
        default=None,
        help="optional Accelerate CPU/disk offload directory",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        "--low-cpu-mem-usage",
        dest="low_cpu_mem_usage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--model_class",
        "--model-class",
        dest="model_class",
        choices=("base", "causal_lm"),
        default="base",
        help="base omits the unused LM head and is recommended for trace extraction",
    )
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    parser.add_argument(
        "--storage_dtype",
        choices=("float16", "float32"),
        default="float32",
        help="trace storage precision; float32 avoids threshold flips from extra quantization",
    )
    parser.add_argument(
        "--query_chunk_size",
        "--query-chunk-size",
        dest="query_chunk_size",
        type=int,
        default=0,
        help="0 uses one full forward; >0 streams exact causal rows with KV cache",
    )
    parser.add_argument(
        "--verify_chunked_equivalence",
        "--verify-chunked-equivalence",
        dest="verify_chunked_equivalence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compare full vs cached attention on every applicable real-model prefix",
    )
    parser.add_argument(
        "--chunk_verify_tokens",
        "--chunk-verify-tokens",
        dest="chunk_verify_tokens",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--chunk_equivalence_atol",
        "--chunk-equivalence-atol",
        dest="chunk_equivalence_atol",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--chunk_equivalence_threshold",
        "--chunk-equivalence-threshold",
        dest="chunk_equivalence_threshold",
        type=float,
        default=0.01,
        help="topology threshold that must have zero full/chunk membership flips",
    )
    parser.add_argument(
        "--archive_compression",
        "--archive-compression",
        dest="archive_compression",
        choices=("compressed", "none"),
        default="compressed",
        help="none is faster on local scratch but uses more disk",
    )
    parser.add_argument("--num_shards", "--num-shards", dest="num_shards", type=int, default=1)
    parser.add_argument("--shard_index", "--shard-index", dest="shard_index", type=int, default=0)
    parser.add_argument(
        "--allow_unverified_generator_weights",
        "--allow-unverified-generator-weights",
        dest="allow_unverified_generator_weights",
        action="store_true",
        help="unsafe diagnostic: exact token axis but generator weight revision is unverified",
    )
    parser.add_argument(
        "--skip_invalid",
        action="store_true",
        help="record and skip invalid/unalignable samples instead of failing immediately",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and int(args.limit) < 1:
        parser.error("--limit must be positive")
    if int(args.max_seq_len) < 1:
        parser.error("--max_seq_len must be positive")
    if not math.isfinite(float(args.max_attention_gib)) or float(args.max_attention_gib) <= 0:
        parser.error("--max_attention_gib must be finite and positive")
    if int(args.query_chunk_size) < 0:
        parser.error("--query_chunk_size cannot be negative")
    if int(args.chunk_verify_tokens) < 2:
        parser.error("--chunk_verify_tokens must be at least 2")
    if (
        not math.isfinite(float(args.chunk_equivalence_atol))
        or float(args.chunk_equivalence_atol) < 0
    ):
        parser.error("--chunk_equivalence_atol must be finite and non-negative")
    if (
        not math.isfinite(float(args.chunk_equivalence_threshold))
        or not 0.0 <= float(args.chunk_equivalence_threshold) <= 1.0
    ):
        parser.error("--chunk_equivalence_threshold must lie in [0,1]")
    if int(args.num_shards) < 1:
        parser.error("--num_shards must be positive")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        parser.error("--shard_index must lie in [0, num_shards)")
    try:
        max_memory = parse_max_memory(args.max_memory)
    except ValueError as exc:
        parser.error(str(exc))
    if max_memory is not None and args.device_map == "none":
        parser.error("--max_memory requires a non-'none' --device_map")
    if args.offload_folder and args.device_map == "none":
        parser.error("--offload_folder requires a non-'none' --device_map")
    try:
        import torch
        import transformers
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - remote extraction environment
        raise SystemExit("attention extraction requires torch and transformers") from exc
    try:
        import accelerate
    except ImportError as exc:  # pragma: no cover - remote extraction environment
        accelerate_version = None
        if args.device_map != "none":
            raise SystemExit(
                "--device_map requires the Hugging Face accelerate package"
            ) from exc
    else:
        accelerate_version = str(accelerate.__version__)

    output_dir = prepare_empty_output_dir(args.output_dir)
    input_path = Path(args.input).expanduser().resolve()
    input_sha256 = _sha256_file(input_path)
    all_rows = load_processbench_rows(str(input_path))
    input_num_rows = len(all_rows)
    indexed_rows, pre_shard_num_rows = select_dataset_shard(
        all_rows,
        limit=args.limit,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )
    if not indexed_rows:
        raise SystemExit("the selected limit/shard contains no rows")

    is_local_model = Path(args.model).expanduser().exists()
    revision_kwargs = (
        {"revision": str(args.model_commit_hash)}
        if args.model_commit_hash and not is_local_model
        else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, **revision_kwargs
    )
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit("exact-axis extraction requires a Hugging Face fast tokenizer")
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype_device = "cuda:0" if args.device_map != "none" and torch.cuda.is_available() else device
    dtype = _torch_dtype(torch, args.dtype, dtype_device)
    resolved_dtype = str(dtype).replace("torch.", "")
    model_class = AutoModel if args.model_class == "base" else AutoModelForCausalLM
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": bool(args.low_cpu_mem_usage),
        **revision_kwargs,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = str(args.device_map)
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
        if args.offload_folder:
            model_kwargs["offload_folder"] = str(
                Path(args.offload_folder).expanduser().resolve()
            )
            model_kwargs["offload_state_dict"] = True
    model = model_class.from_pretrained(args.model, **model_kwargs)
    if args.device_map == "none":
        model = model.to(device)
    model.eval()
    input_device = _model_input_device(model, device)
    resolved_device_map = _resolved_model_device_map(model, input_device)
    tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    try:
        resolved_model_commit = resolve_loaded_commit(
            args.model_commit_hash,
            getattr(model.config, "_commit_hash", None),
            tokenizer_commit,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    model_commit_source = classify_model_commit_source(
        is_local_model=is_local_model,
        requested_commit=args.model_commit_hash,
        model_commit=getattr(model.config, "_commit_hash", None),
        tokenizer_commit=tokenizer_commit,
    )
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    attention_layers = parse_indices(
        args.attention_layers, num_layers, name="attention_layers"
    )
    attention_heads = parse_indices(
        args.attention_heads, num_heads, name="attention_heads"
    )
    if args.activation_layer is not None and not 0 <= args.activation_layer <= num_layers:
        raise SystemExit(f"activation_layer must lie in 0..{num_layers}")

    extraction_config = {
        "trace_contract": TRACE_CONTRACT,
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "input_num_rows": int(input_num_rows),
        "pre_shard_num_rows": int(pre_shard_num_rows),
        "selected_num_rows": int(len(indexed_rows)),
        "requested_limit": None if args.limit is None else int(args.limit),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "skip_invalid": bool(args.skip_invalid),
        "max_seq_len": int(args.max_seq_len),
        "max_attention_gib": float(args.max_attention_gib),
        "allow_large_attention": bool(args.allow_large_attention),
        "model_name": str(args.model),
        "model_commit_hash": resolved_model_commit,
        "model_commit_source": model_commit_source,
        "tokenizer_name": str(getattr(tokenizer, "name_or_path", args.model)),
        "prompt_style": str(args.prompt_style),
        "replay_mode": str(args.replay_mode),
        "allow_unverified_generator_weights": bool(
            args.allow_unverified_generator_weights
        ),
        "dtype": resolved_dtype,
        "attention_implementation": "eager",
        "model_class": str(args.model_class),
        "code_sha256": extraction_code_sha256(),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "accelerate_version": accelerate_version,
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_device_names": [
            str(torch.cuda.get_device_name(index))
            for index in range(torch.cuda.device_count())
        ],
        "cuda_compute_capabilities": [
            list(torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        ],
        "cuda_matmul_allow_tf32": bool(
            torch.cuda.is_available() and torch.backends.cuda.matmul.allow_tf32
        ),
        "device": str(device),
        "device_map": str(args.device_map),
        "resolved_device_map": resolved_device_map,
        "max_memory": (
            None
            if max_memory is None
            else {str(key): value for key, value in max_memory.items()}
        ),
        "input_device": input_device,
        "low_cpu_mem_usage": bool(args.low_cpu_mem_usage),
        "offload_folder": None if args.offload_folder is None else str(args.offload_folder),
        "query_chunk_size": int(args.query_chunk_size),
        "verify_chunked_equivalence": bool(args.verify_chunked_equivalence),
        "chunk_verify_tokens": int(args.chunk_verify_tokens),
        "chunk_equivalence_atol": float(args.chunk_equivalence_atol),
        "chunk_equivalence_threshold": float(args.chunk_equivalence_threshold),
        "archive_compression": str(args.archive_compression),
        "attention_storage_dtype": str(args.storage_dtype),
        "attention_layers": list(attention_layers),
        "attention_heads": list(attention_heads),
        "num_model_layers": num_layers,
        "num_model_heads": num_heads,
        "activation_layer": args.activation_layer,
    }
    # Scope and machine/runtime placement are recorded separately.  They must
    # not prevent complementary shards from being trained together.
    (
        extraction_fingerprint,
        extraction_scope_fingerprint,
        extraction_method_json,
        extraction_scope_json,
    ) = fingerprint_extraction_config(extraction_config)
    extraction_config["extraction_fingerprint"] = extraction_fingerprint
    extraction_config["extraction_scope_fingerprint"] = (
        extraction_scope_fingerprint
    )

    manifest: List[Dict[str, Any]] = []
    for index, raw_row in indexed_rows:
        try:
            record = canonical_record(raw_row, index)
            if args.replay_mode == "same_generator":
                require_exact_replay_inputs(record)
                if not generator_matches_model(record["generator_model"], args.model):
                    raise ValueError(
                        f"dataset generator {record['generator_model']!r} does not match "
                        f"replay model {args.model!r}; use --replay_mode observer only for "
                        "a separately reported cross-model detector"
                    )
                generator_commit = record["generator_model_commit"]
                replay_commit = extraction_config["model_commit_hash"]
                if generator_commit and not is_immutable_commit_hash(generator_commit):
                    raise ValueError(
                        f"dataset generator commit {generator_commit!r} is not an immutable "
                        "hexadecimal commit hash"
                    )
                if replay_commit and not is_immutable_commit_hash(replay_commit):
                    raise ValueError(
                        f"replay model commit {replay_commit!r} is not an immutable hexadecimal "
                        "commit hash"
                    )
                if (
                    generator_commit
                    and replay_commit
                    and not commit_hashes_match(generator_commit, replay_commit)
                ):
                    raise ValueError(
                        f"dataset generator commit {generator_commit!r} does not match "
                        f"replay model commit {replay_commit!r}"
                    )
                weights_verified = bool(
                    generator_commit
                    and replay_commit
                    and extraction_config["model_commit_source"]
                    in VERIFIED_MODEL_COMMIT_SOURCES
                )
                if not weights_verified and not args.allow_unverified_generator_weights:
                    raise ValueError(
                        "same_generator replay cannot verify both generator and replay weight "
                        "revisions; provide generator_model_commit plus a remotely pinned or "
                        "model-config-resolved commit (a local CLI declaration is insufficient), "
                        "use --replay_mode observer, or explicitly opt into the separately "
                        "reported --allow-unverified-generator-weights diagnostic"
                    )
                replay_fidelity = (
                    "weight_and_token_verified_replay"
                    if weights_verified
                    else "token_axis_verified_weights_unverified"
                )
            else:
                replay_fidelity = "observer_counterfactual"
            if record["rendered_prompt"] is None:
                prompt = render_prompt(tokenizer, record["question"], args.prompt_style)
                prompt_provenance = f"frozen_{args.prompt_style}_observer"
            else:
                prompt = record["rendered_prompt"]
                prompt_provenance = "stored_rendered_prompt"
            visible_response_ids = record["response_token_ids"]
            terminal_response_ids: List[int] = []
            if visible_response_ids is not None:
                visible_response_ids, terminal_response_ids = trim_trailing_generation_tokens(
                    visible_response_ids,
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                )
            # Stored prompt IDs decide whether tokenizer specials were present.
            # Without them, chat templates already contain their own controls.
            tokenized = tokenize_exact(
                tokenizer,
                prompt,
                record["steps"],
                add_special_tokens=(
                    None
                    if record["prompt_token_ids"] is not None
                    else args.prompt_style != "chat"
                ),
                response_text=record["response_text"],
                prompt_token_ids=record["prompt_token_ids"],
                response_token_ids=visible_response_ids,
            )
            sequence_length = len(tokenized["token_ids"])
            if sequence_length > int(args.max_seq_len):
                raise ValueError(
                    f"sequence length {sequence_length} exceeds max_seq_len={args.max_seq_len}; "
                    "truncation is forbidden because it would corrupt step labels"
                )
            uses_query_chunks = bool(
                int(args.query_chunk_size) > 0
                and int(args.query_chunk_size) < sequence_length
            )
            compute_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
            query_block = (
                sequence_length
                if int(args.query_chunk_size) <= 0
                else min(sequence_length, int(args.query_chunk_size))
            )
            estimated_forward_gib = estimate_attention_block_gib(
                num_layers,
                num_heads,
                query_block,
                sequence_length,
                compute_bytes,
            )
            estimated_trace_gib = estimate_attention_gib(
                len(attention_layers),
                len(attention_heads),
                sequence_length,
                2 if args.storage_dtype == "float16" else 4,
            )
            equivalence_tokens = (
                min(
                    sequence_length,
                    max(
                        int(args.chunk_verify_tokens),
                        int(args.query_chunk_size) + 1,
                    ),
                )
                if uses_query_chunks and bool(args.verify_chunked_equivalence)
                else 0
            )
            estimated_equivalence_gib = (
                estimate_attention_block_gib(
                    num_layers,
                    num_heads,
                    equivalence_tokens,
                    equivalence_tokens,
                    compute_bytes,
                )
                if equivalence_tokens
                else 0.0
            )
            estimated_activation_gib = 0.0
            if args.activation_layer is not None:
                hidden_size = int(model.config.hidden_size)
                activation_elements = max(
                    (num_layers + 1) * query_block * hidden_size,
                    sequence_length * hidden_size,
                    (num_layers + 1) * equivalence_tokens * hidden_size,
                )
                estimated_activation_gib = (
                    activation_elements * compute_bytes / float(1024**3)
                )
            estimated_gib = max(
                estimated_forward_gib,
                estimated_trace_gib,
                estimated_equivalence_gib,
                estimated_activation_gib,
            )
            if estimated_gib > float(args.max_attention_gib) and not args.allow_large_attention:
                raise ValueError(
                    f"attention/activation tensor lower bound {estimated_gib:.2f} GiB exceeds "
                    f"max_attention_gib={args.max_attention_gib}"
                )
            if uses_query_chunks and bool(args.verify_chunked_equivalence):
                trace_chunk_equivalence = verify_chunked_equivalence(
                    model,
                    torch,
                    tokenized,
                    device=input_device,
                    attention_layers=attention_layers,
                    attention_heads=attention_heads,
                    query_chunk_size=int(args.query_chunk_size),
                    activation_layer=args.activation_layer,
                    prefix_tokens=int(args.chunk_verify_tokens),
                    atol=float(args.chunk_equivalence_atol),
                    topology_threshold=float(args.chunk_equivalence_threshold),
                )
                trace_chunk_status = str(trace_chunk_equivalence["status"])
            elif uses_query_chunks:
                trace_chunk_status = "disabled"
                trace_chunk_equivalence = {"status": trace_chunk_status}
            else:
                trace_chunk_status = "not_applicable"
                trace_chunk_equivalence = {"status": trace_chunk_status}
            trace = extract_trace(
                model,
                torch,
                tokenized,
                device=input_device,
                attention_layers=attention_layers,
                attention_heads=attention_heads,
                activation_layer=args.activation_layer,
                storage_dtype=str(args.storage_dtype),
                query_chunk_size=int(args.query_chunk_size),
            )
            stem = _safe_stem(record["sample_id"], index)
            destination = output_dir / f"{stem}.npz"
            payload = {
                "attention": trace["attention"],
                "token_ids": tokenized["token_ids"],
                "response_idx": np.asarray(tokenized["response_idx"], np.int64),
                "step_ranges": tokenized["step_ranges"],
                "gold_step": np.asarray(record["gold_step"], np.int64),
                "response_y": np.asarray(record["response_y"], np.float32),
                "sample_id": np.asarray(record["sample_id"]),
                "problem_id": np.asarray(record["problem_id"]),
                "attention_layers": np.asarray(attention_layers, np.int64),
                "attention_heads": np.asarray(attention_heads, np.int64),
                "num_model_layers": np.asarray(num_layers, np.int64),
                "num_model_heads": np.asarray(num_heads, np.int64),
                "activation_layer": np.asarray(
                    -1 if args.activation_layer is None else args.activation_layer,
                    np.int64,
                ),
                "model_name": np.asarray(args.model),
                "model_commit_hash": np.asarray(
                    ""
                    if extraction_config["model_commit_hash"] is None
                    else extraction_config["model_commit_hash"]
                ),
                "model_commit_source": np.asarray(
                    extraction_config["model_commit_source"]
                ),
                "tokenizer_name": np.asarray(extraction_config["tokenizer_name"]),
                "extraction_dtype": np.asarray(resolved_dtype),
                "attention_storage_dtype": np.asarray(args.storage_dtype),
                "extraction_fingerprint": np.asarray(extraction_fingerprint),
                "extraction_method_json": np.asarray(extraction_method_json),
                "extraction_scope_fingerprint": np.asarray(
                    extraction_scope_fingerprint
                ),
                "extraction_scope_json": np.asarray(extraction_scope_json),
                "source_input_sha256": np.asarray(input_sha256),
                "source_row_index": np.asarray(index, np.int64),
                "extraction_forward_mode": np.asarray(
                    "cached_query_chunks" if uses_query_chunks else "full"
                ),
                "chunk_equivalence_status": np.asarray(trace_chunk_status),
                "chunk_equivalence_json": np.asarray(
                    json.dumps(
                        trace_chunk_equivalence,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
                "prompt_style": np.asarray(args.prompt_style),
                "replay_mode": np.asarray(args.replay_mode),
                "replay_fidelity": np.asarray(replay_fidelity),
                "unverified_generator_weights_explicitly_allowed": np.asarray(
                    bool(
                        replay_fidelity == "token_axis_verified_weights_unverified"
                        and args.allow_unverified_generator_weights
                    )
                ),
                "prompt_provenance": np.asarray(prompt_provenance),
                "generator_model": np.asarray(record["generator_model"] or ""),
                "generator_model_commit": np.asarray(
                    record["generator_model_commit"] or ""
                ),
                "prompt_add_special_tokens": np.asarray(
                    tokenized["prompt_add_special_tokens"]
                ),
                "generation_terminal_token_ids": np.asarray(
                    terminal_response_ids, np.int64
                ),
                "generation_terminal_token_count": np.asarray(
                    len(terminal_response_ids), np.int64
                ),
                "rendered_prompt": np.asarray(prompt),
                "response_text": np.asarray(
                    tokenized["full_text"][len(prompt) :]
                ),
                "rendered_prompt_sha256": np.asarray(
                    hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                ),
                "response_text_sha256": np.asarray(
                    hashlib.sha256(
                        tokenized["full_text"][len(prompt) :].encode("utf-8")
                    ).hexdigest()
                ),
                "trace_contract": np.asarray(TRACE_CONTRACT),
            }
            if record["split"] is not None:
                payload["split"] = np.asarray(record["split"])
            if "activation" in trace:
                payload["activation"] = trace["activation"]
            save_trace_archive(
                destination, payload, compression=str(args.archive_compression)
            )
            manifest.append(
                {
                    "index": index,
                    "sample_id": record["sample_id"],
                    "problem_id": record["problem_id"],
                    "file": destination.name,
                    "tokens": sequence_length,
                    "steps": len(record["steps"]),
                    "gold_step": record["gold_step"],
                    "generator_model": record["generator_model"],
                    "replay_fidelity": replay_fidelity,
                    "model_commit_source": extraction_config["model_commit_source"],
                    "prompt_provenance": prompt_provenance,
                    "generation_terminal_token_count": len(terminal_response_ids),
                    "estimated_forward_attention_block_gib": estimated_forward_gib,
                    "estimated_dense_trace_gib": estimated_trace_gib,
                    "estimated_equivalence_attention_gib": estimated_equivalence_gib,
                    "estimated_activation_gib": estimated_activation_gib,
                    "extraction_fingerprint": extraction_fingerprint,
                    "extraction_scope_fingerprint": extraction_scope_fingerprint,
                    "extraction_forward_mode": (
                        "cached_query_chunks" if uses_query_chunks else "full"
                    ),
                    "chunk_equivalence_status": trace_chunk_status,
                    "chunk_equivalence": trace_chunk_equivalence,
                    "status": "ok",
                }
            )
            _write_extraction_manifest(
                output_dir / "manifest.json",
                extraction_config=extraction_config,
                extraction_fingerprint=extraction_fingerprint,
                extraction_scope_fingerprint=extraction_scope_fingerprint,
                traces=manifest,
            )
            del trace
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            failure = {
                "index": index,
                "sample_id": str(raw_row.get("id", index)),
                "status": "skipped" if args.skip_invalid else "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            manifest.append(failure)
            _write_extraction_manifest(
                output_dir / "manifest.json",
                extraction_config=extraction_config,
                extraction_fingerprint=extraction_fingerprint,
                extraction_scope_fingerprint=extraction_scope_fingerprint,
                traces=manifest,
            )
            if not args.skip_invalid:
                raise

    _write_extraction_manifest(
        output_dir / "manifest.json",
        extraction_config=extraction_config,
        extraction_fingerprint=extraction_fingerprint,
        extraction_scope_fingerprint=extraction_scope_fingerprint,
        traces=manifest,
    )
    completed = sum(item["status"] == "ok" for item in manifest)
    if completed == 0:
        raise SystemExit(
            f"extracted 0/{len(manifest)} valid traces; inspect manifest.json before retrying"
        )
    print(f"extracted {completed}/{len(manifest)} exact-axis attention traces -> {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
