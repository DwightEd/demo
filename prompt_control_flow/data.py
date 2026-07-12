from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

from utils.step_boundaries import (
    SPAN_RANGE_CONVENTION,
    STEP_TOKEN_RANGE_CONVENTION,
    TOKEN_OFFSET_CONVENTION,
    TRACE_SCHEMA_VERSION,
    TokenAlignmentError,
)


@dataclass
class ChainRecord:
    """A single reasoning chain with text and labels."""

    chain_idx: int
    problem_id: int
    problem: str
    steps: List[str]
    response: str
    gold_error_step: int = -1
    is_correct: Optional[int] = None
    sample_idx: Optional[int] = None
    generator: Optional[str] = None
    dataset: Optional[str] = None
    rendered_prompt: Optional[str] = None
    exact_input_ids: Optional[List[int]] = None
    exact_attention_mask: Optional[List[int]] = None
    exact_token_offsets: Optional[List[Tuple[int, int]]] = None
    exact_step_token_ranges: Optional[List[Tuple[int, int]]] = None
    exact_response_start_token: Optional[int] = None
    source_tokenizer: Optional[str] = None
    source_model_revision: Optional[str] = None
    source_tokenizer_revision: Optional[str] = None


def _as_list_of_str(x: Any) -> List[str]:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    if x is None:
        return []
    return [str(x)]


def _get_array(npz: np.lib.npyio.NpzFile, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in npz.files:
            return npz[name]
    return default


def _row_ints(array: Any, i: int) -> Optional[List[int]]:
    if array is None:
        return None
    value = array[i]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [int(x) for x in value]


def _row_pairs(array: Any, i: int) -> Optional[List[Tuple[int, int]]]:
    if array is None:
        return None
    value = array[i]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [(int(pair[0]), int(pair[1])) for pair in value]


def _scalar_or_row(array: Any, i: int, default: Any = None) -> Any:
    if array is None:
        return default
    value = np.asarray(array, dtype=object)
    if value.ndim == 0:
        return value.item()
    return value[i]


def is_processbench_full(path: str | Path, npz: np.lib.npyio.NpzFile | None = None) -> bool:
    name = Path(path).name.lower()
    if name.startswith("full_"):
        return True
    z = npz if npz is not None else np.load(path, allow_pickle=True)
    return ("gold_error_step" in z.files or "labels" in z.files) and "steps_text" in z.files


def is_multisample(path: str | Path, npz: np.lib.npyio.NpzFile | None = None) -> bool:
    name = Path(path).name.lower()
    if "multisample" in name:
        return True
    z = npz if npz is not None else np.load(path, allow_pickle=True)
    return "sample_idx" in z.files and ("is_correct" in z.files or "is_correct_strict" in z.files)


def _problem_id_from_record(raw_id: Any, fallback: int) -> int:
    text = str(raw_id)
    m = re.search(r"(\d+)$", text)
    return int(m.group(1)) if m else int(fallback)


def _dataset_from_path(path: Path) -> str:
    name = path.stem.lower()
    for suffix in ("_multisample_sv", "_features", "_sv", "_full"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.startswith("full_"):
        name = name[len("full_") :]
    return name


def _load_processbench_jsonl(path: Path, max_chains: int = 0) -> List[ChainRecord]:
    rows: List[ChainRecord] = []
    dataset = _dataset_from_path(path)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_chains and len(rows) >= int(max_chains):
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            steps = _as_list_of_str(rec.get("steps", rec.get("steps_text", [])))
            response = str(rec.get("response", "\n\n".join(steps)))
            gold = int(rec.get("label", rec.get("gold_error_step", -1)))
            final_correct = rec.get("final_answer_correct", None)
            is_correct = None if final_correct is None else int(bool(final_correct))
            rows.append(
                ChainRecord(
                    chain_idx=len(rows),
                    problem_id=_problem_id_from_record(rec.get("id", i), i),
                    problem=str(rec.get("problem", rec.get("question", ""))),
                    steps=steps,
                    response=response,
                    gold_error_step=gold,
                    is_correct=is_correct,
                    sample_idx=None,
                    generator=rec.get("generator"),
                    dataset=dataset,
                )
            )
    return rows


def load_chain_records(
    path: str | Path,
    max_chains: int = 0,
    input_format: str = "auto",
) -> List[ChainRecord]:
    """Load text chains from a ProcessBench full or multisample npz.

    This loader intentionally avoids inferring same-problem response labels from
    ProcessBench full data.  The caller chooses the evaluation mode later.
    """

    path = Path(path)
    fmt = input_format.lower()
    if fmt not in {"auto", "npz", "processbench_jsonl", "jsonl"}:
        raise ValueError(f"unknown input_format={input_format!r}")
    if fmt in {"processbench_jsonl", "jsonl"} or (fmt == "auto" and path.suffix.lower() == ".jsonl"):
        return _load_processbench_jsonl(path, max_chains=max_chains)

    z = np.load(path, allow_pickle=True)
    steps_arr = _get_array(z, ["steps_text", "steps"], None)
    if steps_arr is None:
        raise ValueError(f"{path}: expected `steps_text` or `steps` for prompt-flow extraction")

    n = int(len(steps_arr))
    if max_chains and max_chains > 0:
        n = min(n, int(max_chains))

    problems = _get_array(z, ["problems", "problem", "questions", "question"], None)
    responses = _get_array(z, ["responses", "response"], None)
    prompts = _get_array(z, ["prompts", "rendered_prompts", "rendered_prompt"], None)
    problem_ids = _get_array(z, ["problem_ids"], np.arange(len(steps_arr), dtype=np.int64))
    sample_idx = _get_array(z, ["sample_idx"], None)
    generators = _get_array(z, ["generator", "generators", "source_model", "model_name"], None)
    datasets = _get_array(z, ["dataset", "datasets", "subset"], None)

    gold = _get_array(z, ["gold_error_step_kept", "gold_error_step", "labels"], None)
    if gold is None:
        gold = np.full(len(steps_arr), -1, dtype=np.int64)

    correct = _get_array(z, ["is_correct_strict", "is_correct"], None)
    kept_steps = _get_array(z, ["kept_steps", "time_axis_original_step_indices"], None)
    exact_input_ids = _get_array(z, ["input_ids"], None)
    exact_attention_mask = _get_array(z, ["attention_mask"], None)
    exact_token_offsets = _get_array(z, ["token_offsets", "input_token_offsets"], None)
    exact_step_ranges = _get_array(z, ["step_token_ranges", "time_axis_token_ranges"], None)
    response_ranges = _get_array(z, ["response_token_ranges"], None)
    prompt_counts = _get_array(z, ["prompt_token_counts"], None)
    sampling_metadata: dict[str, Any] = {}
    if "model_sampling_metadata_json" in z.files:
        try:
            sampling_metadata = json.loads(str(np.asarray(z["model_sampling_metadata_json"]).item()))
        except Exception as exc:
            raise TokenAlignmentError(f"{path}: invalid model_sampling_metadata_json") from exc
    declares_exact_trace = "trace_schema_version" in z.files or exact_input_ids is not None
    exact_contract_parts = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": exact_input_ids,
        "attention_mask": exact_attention_mask,
        "token_offsets": exact_token_offsets,
        "step_token_ranges": exact_step_ranges,
        "response_start": response_ranges if response_ranges is not None else prompt_counts,
    }
    if declares_exact_trace:
        missing_exact = [name for name, value in exact_contract_parts.items() if value is None]
        if missing_exact:
            raise TokenAlignmentError(
                f"{path}: exact trace is incomplete; missing {missing_exact}. "
                "Refusing to silently re-tokenize a partial generation artifact."
            )
        conventions = {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "token_offset_convention": TOKEN_OFFSET_CONVENTION,
            "step_token_range_convention": STEP_TOKEN_RANGE_CONVENTION,
            "span_range_convention": SPAN_RANGE_CONVENTION,
        }
        for key, expected_value in conventions.items():
            if key not in z.files:
                raise TokenAlignmentError(f"{path}: exact trace is missing convention field {key}")
            actual = str(np.asarray(z[key]).item())
            if actual != expected_value:
                raise TokenAlignmentError(
                    f"{path}: {key}={actual!r}, expected {expected_value!r}"
                )
        if "trace_token_add_special_tokens" not in z.files:
            raise TokenAlignmentError(f"{path}: exact trace is missing trace_token_add_special_tokens")
        if bool(np.asarray(z["trace_token_add_special_tokens"]).item()):
            raise TokenAlignmentError(f"{path}: exact trace unexpectedly adds special tokens")

    rows: List[ChainRecord] = []
    for i in range(n):
        all_steps = _as_list_of_str(steps_arr[i])
        kept = _row_ints(kept_steps, i)
        if kept is None or len(all_steps) == len(kept):
            # Both exact writers may already serialize the kept time axis.
            steps = all_steps
        else:
            if any(j < 0 or j >= len(all_steps) for j in kept):
                raise ValueError(f"{path}: kept_steps[{i}] cannot index steps_text[{i}]")
            steps = [all_steps[j] for j in kept]
        response = str(responses[i]) if responses is not None else "\n\n".join(steps)
        problem = str(problems[i]) if problems is not None else ""
        response_start = None
        if response_ranges is not None:
            response_start = int(np.asarray(response_ranges[i]).reshape(-1)[0])
        elif prompt_counts is not None:
            response_start = int(prompt_counts[i])
        have_exact_bundle = declares_exact_trace and all(
            value is not None
            for value in (
                exact_input_ids,
                exact_attention_mask,
                exact_token_offsets,
                exact_step_ranges,
                response_start,
            )
        )
        rows.append(
            ChainRecord(
                chain_idx=i,
                problem_id=int(problem_ids[i]) if problem_ids is not None else i,
                problem=problem,
                steps=steps,
                response=response,
                gold_error_step=int(gold[i]) if gold is not None else -1,
                is_correct=int(correct[i]) if correct is not None else None,
                sample_idx=int(sample_idx[i]) if sample_idx is not None else None,
                generator=str(_scalar_or_row(generators, i)) if generators is not None else None,
                dataset=str(_scalar_or_row(datasets, i)) if datasets is not None else _dataset_from_path(path),
                rendered_prompt=str(prompts[i]) if prompts is not None else None,
                exact_input_ids=_row_ints(exact_input_ids, i) if have_exact_bundle else None,
                exact_attention_mask=_row_ints(exact_attention_mask, i) if have_exact_bundle else None,
                exact_token_offsets=_row_pairs(exact_token_offsets, i) if have_exact_bundle else None,
                exact_step_token_ranges=_row_pairs(exact_step_ranges, i) if have_exact_bundle else None,
                exact_response_start_token=response_start if have_exact_bundle else None,
                source_tokenizer=str(sampling_metadata.get("tokenizer_name", "")) or None,
                source_model_revision=str(sampling_metadata.get("model_revision", "")) or None,
                source_tokenizer_revision=str(sampling_metadata.get("tokenizer_revision", "")) or None,
            )
        )
    return rows
