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
from .replay_protocols import stable_problem_group_id


PROCESS_LABEL_UNAVAILABLE = -2


@dataclass
class ChainRecord:
    """A single reasoning chain with text and labels."""

    chain_idx: int
    problem_id: int
    problem: str
    steps: List[str]
    response: str
    gold_error_step: int = PROCESS_LABEL_UNAVAILABLE
    problem_group_id: str = ""
    process_correct: Optional[int] = None
    final_answer_correct: Optional[int] = None
    is_correct: Optional[int] = None
    sample_idx: Optional[int] = None
    # Model that generated the answer text, when ProcessBench exposes it.
    generator: Optional[str] = None
    dataset: Optional[str] = None
    rendered_prompt: Optional[str] = None
    exact_input_ids: Optional[List[int]] = None
    exact_attention_mask: Optional[List[int]] = None
    exact_token_offsets: Optional[List[Tuple[int, int]]] = None
    exact_step_token_ranges: Optional[List[Tuple[int, int]]] = None
    exact_response_start_token: Optional[int] = None
    exact_question_token_range: Optional[Tuple[int, int]] = None
    # Observer identity that produced the exact token/state replay artifact.
    source_model: Optional[str] = None
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


def _get_array(
    npz: np.lib.npyio.NpzFile, names: Iterable[str], default: Any = None
) -> Any:
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


def _row_numeric_id(array: Any, i: int, fallback: int) -> int:
    """Return a numeric row identifier without treating text groups as IDs."""

    if array is None:
        return int(fallback)
    value = _scalar_or_row(array, i, fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def is_processbench_full(
    path: str | Path, npz: np.lib.npyio.NpzFile | None = None
) -> bool:
    name = Path(path).name.lower()
    if name.startswith("full_"):
        return True
    z = npz if npz is not None else np.load(path, allow_pickle=True)
    return (
        "gold_error_step" in z.files or "labels" in z.files
    ) and "steps_text" in z.files


def is_multisample(path: str | Path, npz: np.lib.npyio.NpzFile | None = None) -> bool:
    name = Path(path).name.lower()
    if "multisample" in name:
        return True
    z = npz if npz is not None else np.load(path, allow_pickle=True)
    return "sample_idx" in z.files and (
        "is_correct" in z.files or "is_correct_strict" in z.files
    )


def _problem_id_from_record(raw_id: Any, fallback: int) -> int:
    text = str(raw_id)
    m = re.search(r"(\d+)$", text)
    return int(m.group(1)) if m else int(fallback)


def _optional_binary_label(value: Any, *, field: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return 1
        if normalized in {"false", "0"}:
            return 0
        raise ValueError(f"{field} must be binary, got {value!r}")
    numeric = int(value)
    if numeric not in {0, 1}:
        raise ValueError(f"{field} must be binary, got {value!r}")
    return numeric


def _validate_gold_error_step(value: int, n_steps: int, *, context: str) -> int:
    label = int(value)
    if label < PROCESS_LABEL_UNAVAILABLE or label >= int(n_steps):
        raise ValueError(
            f"{context}: gold_error_step={label} is outside -2, -1, or "
            f"[0, {n_steps - 1}]"
        )
    return label


def process_correct_from_gold(gold_error_step: int) -> int:
    """Map the tri-state first-error label to a tri-state correctness label."""

    label = int(gold_error_step)
    if label == PROCESS_LABEL_UNAVAILABLE:
        return -1
    if label == -1:
        return 1
    if label >= 0:
        return 0
    raise ValueError(f"unsupported gold_error_step={label}")


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
            if not steps or any(not step.strip() for step in steps):
                raise ValueError(f"{path}: row {i} contains an empty reasoning step")
            response = str(rec.get("response", "\n\n".join(steps)))
            if "label" not in rec and "gold_error_step" not in rec:
                raise ValueError(
                    f"{path}: ProcessBench row {i} has no process label"
                )
            gold = _validate_gold_error_step(
                int(rec.get("label", rec.get("gold_error_step"))),
                len(steps),
                context=f"{path}: row {i}",
            )
            final_correct = rec.get("final_answer_correct", None)
            process_correct = process_correct_from_gold(gold)
            final_correct_value = _optional_binary_label(
                final_correct, field="final_answer_correct"
            )
            problem = str(rec.get("problem", rec.get("question", "")))
            if not problem.strip():
                raise ValueError(f"{path}: ProcessBench row {i} has an empty problem")
            rows.append(
                ChainRecord(
                    chain_idx=len(rows),
                    problem_id=_problem_id_from_record(rec.get("id", i), i),
                    problem=problem,
                    steps=steps,
                    response=response,
                    gold_error_step=gold,
                    problem_group_id=stable_problem_group_id(problem),
                    process_correct=process_correct,
                    final_answer_correct=final_correct_value,
                    is_correct=process_correct,
                    sample_idx=None,
                    generator=rec.get("generator"),
                    dataset=dataset,
                )
            )
    return rows


def _iter_processbench_source(path: Path, subset: str):
    """Yield raw ProcessBench rows from canonical JSON or HF dataset sources."""

    candidate: Path | None = path if path.is_file() else None
    if candidate is None:
        for suffix in (".json", ".jsonl"):
            value = path / f"{subset}{suffix}"
            if value.is_file():
                candidate = value
                break
    if candidate is not None:
        text = candidate.read_text(encoding="utf-8")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            yield from parsed
            return
        if isinstance(parsed, dict):
            rows = parsed.get("data") or parsed.get("rows")
            if rows is None:
                raise ValueError(f"{candidate}: JSON object has no `data` or `rows`")
            yield from rows
            return
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - remote extraction dependency
        raise ImportError(
            "loading a ProcessBench dataset directory requires `datasets`"
        ) from exc
    yield from load_dataset(str(path), split=str(subset))


def _load_canonical_processbench(
    path: Path,
    *,
    subset: str,
    max_chains: int = 0,
) -> List[ChainRecord]:
    """Load canonical ProcessBench rows without mutating its annotated axis."""

    rows: List[ChainRecord] = []
    for raw in _iter_processbench_source(path, subset):
        steps = [str(value) for value in (raw.get("steps") or [])]
        if not steps or any(not value.strip() for value in steps):
            raise ValueError("canonical ProcessBench row contains an empty reasoning step")
        if max_chains and len(rows) >= int(max_chains):
            break
        index = len(rows)
        if "label" not in raw and "gold_error_step" not in raw:
            raise ValueError("canonical ProcessBench row has no process label")
        gold = _validate_gold_error_step(
            int(raw.get("label", raw.get("gold_error_step"))),
            len(steps),
            context=f"ProcessBench/{subset} row {index}",
        )
        final_correct = raw.get("final_answer_correct")
        process_correct = process_correct_from_gold(gold)
        final_correct_value = _optional_binary_label(
            final_correct, field="final_answer_correct"
        )
        response = str(raw.get("response") or "\n\n".join(steps))
        problem = str(raw.get("problem", raw.get("question", "")))
        if not problem.strip():
            raise ValueError("canonical ProcessBench row has an empty problem")
        rows.append(
            ChainRecord(
                chain_idx=int(raw.get("chain_idx", index)),
                problem_id=int(raw.get("problem_id", index)),
                problem=problem,
                steps=steps,
                response=response,
                gold_error_step=gold,
                problem_group_id=stable_problem_group_id(problem),
                process_correct=process_correct,
                final_answer_correct=final_correct_value,
                is_correct=process_correct,
                sample_idx=None,
                generator=raw.get("generator"),
                dataset=str(subset),
            )
        )
    return rows


def validate_records_against_reference(
    records: List[ChainRecord],
    reference_path: str | Path,
) -> dict[str, Any]:
    """Fail before model loading when replay records disagree with a saved artifact."""

    path = Path(reference_path)
    z = np.load(path, allow_pickle=True)
    if "steps_text" not in z.files:
        raise ValueError(f"{path}: reference has no `steps_text`")
    reference_count = int(len(z["steps_text"]))
    if len(records) > reference_count:
        raise ValueError(
            f"{path}: {len(records)} replay records exceed {reference_count} reference rows"
        )
    chain_idx = (
        np.asarray(z["chain_idx"], dtype=np.int64)
        if "chain_idx" in z.files
        else np.arange(reference_count, dtype=np.int64)
    )
    problem_ids = z["problem_ids"] if "problem_ids" in z.files else chain_idx
    reference_problems = _get_array(
        z, ["problems", "problem", "questions", "question"], None
    )
    gold = (
        np.asarray(z["gold_error_step"], dtype=np.int64)
        if "gold_error_step" in z.files
        else None
    )
    responses = z["responses"] if "responses" in z.files else None
    steps_text = z["steps_text"]
    mismatches: list[str] = []
    for row, record in enumerate(records):
        expected_steps = _as_list_of_str(steps_text[row])
        checks = {
            "chain_idx": int(record.chain_idx) == int(chain_idx[row]),
            "n_steps": len(record.steps) == len(expected_steps),
            "steps_text": list(record.steps) == expected_steps,
        }
        try:
            checks["problem_id"] = int(record.problem_id) == int(problem_ids[row])
        except (TypeError, ValueError):
            if reference_problems is not None:
                checks["problem_group_id"] = record.problem_group_id == stable_problem_group_id(
                    str(reference_problems[row])
                )
        if gold is not None:
            checks["gold_error_step"] = int(record.gold_error_step) == int(gold[row])
        if responses is not None:
            checks["response"] = str(record.response) == str(responses[row])
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            mismatches.append(
                f"row={row} chain={record.chain_idx} fields={','.join(failed)}"
            )
            if len(mismatches) >= 8:
                break
    if mismatches:
        raise ValueError(
            f"replay source disagrees with {path}: " + "; ".join(mismatches)
        )
    return {
        "reference_path": str(path),
        "checked_records": int(len(records)),
        "reference_records": reference_count,
        "exact_text_match": True,
        "gold_label_checked": bool(gold is not None),
    }


def load_chain_records(
    path: str | Path,
    max_chains: int = 0,
    input_format: str = "auto",
    subset: str | None = None,
) -> List[ChainRecord]:
    """Load text chains from a ProcessBench full or multisample npz.

    This loader intentionally avoids inferring same-problem response labels from
    ProcessBench full data.  The caller chooses the evaluation mode later.
    """

    path = Path(path)
    fmt = input_format.lower()
    if fmt not in {
        "auto",
        "npz",
        "processbench_jsonl",
        "processbench_source",
        "jsonl",
    }:
        raise ValueError(f"unknown input_format={input_format!r}")
    if fmt == "processbench_source":
        if not subset:
            raise ValueError("processbench_source requires an explicit subset")
        return _load_canonical_processbench(
            path,
            subset=str(subset),
            max_chains=max_chains,
        )
    if fmt in {"processbench_jsonl", "jsonl"} or (
        fmt == "auto" and path.suffix.lower() == ".jsonl"
    ):
        return _load_processbench_jsonl(path, max_chains=max_chains)

    z = np.load(path, allow_pickle=True)
    steps_arr = _get_array(z, ["steps_text", "steps"], None)
    if steps_arr is None:
        raise ValueError(
            f"{path}: expected `steps_text` or `steps` for prompt-flow extraction"
        )

    n = int(len(steps_arr))
    if max_chains and max_chains > 0:
        n = min(n, int(max_chains))

    problems = _get_array(z, ["problems", "problem", "questions", "question"], None)
    responses = _get_array(z, ["responses", "response"], None)
    prompts = _get_array(z, ["prompts", "rendered_prompts", "rendered_prompt"], None)
    chain_ids = _get_array(
        z, ["chain_idx"], np.arange(len(steps_arr), dtype=np.int64)
    )
    problem_ids = _get_array(z, ["problem_ids", "problem_id"], None)
    problem_group_ids = _get_array(
        z, ["problem_group_ids", "problem_group_id"], None
    )
    sample_idx = _get_array(z, ["sample_idx"], None)
    generators = _get_array(z, ["generator", "generators"], None)
    source_models = _get_array(
        z, ["source_model", "observer_model", "model_name"], None
    )
    source_tokenizers = _get_array(
        z, ["source_tokenizer", "observer_tokenizer", "tokenizer_name"], None
    )
    source_model_revisions = _get_array(z, ["source_model_revision"], None)
    source_tokenizer_revisions = _get_array(
        z, ["source_tokenizer_revision"], None
    )
    datasets = _get_array(z, ["dataset", "datasets", "subset"], None)

    gold = _get_array(z, ["gold_error_step_kept", "gold_error_step", "labels"], None)
    has_process_labels = gold is not None
    if not has_process_labels:
        gold = np.full(
            len(steps_arr), PROCESS_LABEL_UNAVAILABLE, dtype=np.int64
        )

    process_correct = _get_array(z, ["process_correct"], None)
    final_answer_correct = _get_array(
        z, ["final_answer_correct", "is_correct_strict"], None
    )
    compatibility_correct = _get_array(z, ["is_correct"], None)
    kept_steps = _get_array(z, ["kept_steps", "time_axis_original_step_indices"], None)
    exact_input_ids = _get_array(z, ["input_ids"], None)
    exact_attention_mask = _get_array(z, ["attention_mask"], None)
    exact_token_offsets = _get_array(z, ["token_offsets", "input_token_offsets"], None)
    exact_step_ranges = _get_array(
        z, ["step_token_ranges", "time_axis_token_ranges"], None
    )
    response_ranges = _get_array(z, ["response_token_ranges"], None)
    question_ranges = _get_array(z, ["question_token_ranges"], None)
    prompt_counts = _get_array(z, ["prompt_token_counts"], None)
    sampling_metadata: dict[str, Any] = {}
    if "model_sampling_metadata_json" in z.files:
        try:
            sampling_metadata = json.loads(
                str(np.asarray(z["model_sampling_metadata_json"]).item())
            )
        except Exception as exc:
            raise TokenAlignmentError(
                f"{path}: invalid model_sampling_metadata_json"
            ) from exc
    declares_exact_trace = (
        "trace_schema_version" in z.files or exact_input_ids is not None
    )
    exact_contract_parts = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": exact_input_ids,
        "attention_mask": exact_attention_mask,
        "token_offsets": exact_token_offsets,
        "step_token_ranges": exact_step_ranges,
        "response_start": (
            response_ranges if response_ranges is not None else prompt_counts
        ),
    }
    if declares_exact_trace:
        missing_exact = [
            name for name, value in exact_contract_parts.items() if value is None
        ]
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
                raise TokenAlignmentError(
                    f"{path}: exact trace is missing convention field {key}"
                )
            actual = str(np.asarray(z[key]).item())
            if actual != expected_value:
                raise TokenAlignmentError(
                    f"{path}: {key}={actual!r}, expected {expected_value!r}"
                )
        if "trace_token_add_special_tokens" not in z.files:
            raise TokenAlignmentError(
                f"{path}: exact trace is missing trace_token_add_special_tokens"
            )
        if bool(np.asarray(z["trace_token_add_special_tokens"]).item()):
            raise TokenAlignmentError(
                f"{path}: exact trace unexpectedly adds special tokens"
            )

    rows: List[ChainRecord] = []
    for i in range(n):
        all_steps = _as_list_of_str(steps_arr[i])
        kept = _row_ints(kept_steps, i)
        if kept is None or len(all_steps) == len(kept):
            # Both exact writers may already serialize the kept time axis.
            steps = all_steps
        else:
            if any(j < 0 or j >= len(all_steps) for j in kept):
                raise ValueError(
                    f"{path}: kept_steps[{i}] cannot index steps_text[{i}]"
                )
            steps = [all_steps[j] for j in kept]
        gold_value = int(gold[i])
        if has_process_labels:
            gold_value = _validate_gold_error_step(
                gold_value,
                len(steps),
                context=f"{path}: row {i}",
            )
        response = str(responses[i]) if responses is not None else "\n\n".join(steps)
        problem = str(problems[i]) if problems is not None else ""
        numeric_problem_id = _row_numeric_id(problem_ids, i, i)
        raw_group = (
            str(_scalar_or_row(problem_group_ids, i))
            if problem_group_ids is not None
            else (
                stable_problem_group_id(problem)
                if problem.strip()
                else f"problem_id:{numeric_problem_id}"
            )
        )
        process_value = (
            int(_scalar_or_row(process_correct, i))
            if process_correct is not None
            else (
                process_correct_from_gold(int(gold[i]))
                if has_process_labels
                else -1
            )
        )
        final_value = (
            int(_scalar_or_row(final_answer_correct, i))
            if final_answer_correct is not None
            else None
        )
        if process_value not in {-1, 0, 1}:
            raise ValueError(
                f"{path}: process_correct[{i}]={process_value} is not -1/0/1"
            )
        if final_value is not None and final_value not in {-1, 0, 1}:
            raise ValueError(
                f"{path}: final_answer_correct[{i}]={final_value} is not -1/0/1"
            )
        compatibility_value = (
            int(_scalar_or_row(compatibility_correct, i))
            if compatibility_correct is not None
            else (process_value if process_value >= 0 else None)
        )
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
        exact_ranges_row = (
            _row_pairs(exact_step_ranges, i) if have_exact_bundle else None
        )
        if exact_ranges_row is not None:
            exact_ranges_row = exact_ranges_row[: len(steps)]
            if len(exact_ranges_row) != len(steps):
                raise TokenAlignmentError(
                    f"{path}: exact step range count {len(exact_ranges_row)} does "
                    f"not match n_steps={len(steps)} in row {i}"
                )
            if any(a < 0 or b < a for a, b in exact_ranges_row):
                raise TokenAlignmentError(
                    f"{path}: invalid exact step token range in row {i}"
                )
        rows.append(
            ChainRecord(
                chain_idx=_row_numeric_id(chain_ids, i, i),
                problem_id=numeric_problem_id,
                problem=problem,
                steps=steps,
                response=response,
                gold_error_step=gold_value,
                problem_group_id=raw_group,
                process_correct=process_value,
                final_answer_correct=final_value,
                is_correct=compatibility_value,
                sample_idx=int(sample_idx[i]) if sample_idx is not None else None,
                generator=(
                    str(_scalar_or_row(generators, i))
                    if generators is not None
                    else None
                ),
                dataset=(
                    str(_scalar_or_row(datasets, i))
                    if datasets is not None
                    else _dataset_from_path(path)
                ),
                rendered_prompt=str(prompts[i]) if prompts is not None else None,
                exact_input_ids=(
                    _row_ints(exact_input_ids, i) if have_exact_bundle else None
                ),
                exact_attention_mask=(
                    _row_ints(exact_attention_mask, i) if have_exact_bundle else None
                ),
                exact_token_offsets=(
                    _row_pairs(exact_token_offsets, i) if have_exact_bundle else None
                ),
                exact_step_token_ranges=(
                    exact_ranges_row
                ),
                exact_response_start_token=(
                    response_start if have_exact_bundle else None
                ),
                exact_question_token_range=(
                    tuple(int(x) for x in np.asarray(question_ranges[i]).reshape(-1)[:2])
                    if have_exact_bundle and question_ranges is not None
                    else None
                ),
                source_model=(
                    str(_scalar_or_row(source_models, i))
                    if source_models is not None
                    else str(sampling_metadata.get("model_name", ""))
                )
                or None,
                source_tokenizer=(
                    str(_scalar_or_row(source_tokenizers, i))
                    if source_tokenizers is not None
                    else str(sampling_metadata.get("tokenizer_name", ""))
                )
                or None,
                source_model_revision=(
                    str(_scalar_or_row(source_model_revisions, i))
                    if source_model_revisions is not None
                    else str(sampling_metadata.get("model_revision", ""))
                )
                or None,
                source_tokenizer_revision=(
                    str(_scalar_or_row(source_tokenizer_revisions, i))
                    if source_tokenizer_revisions is not None
                    else str(sampling_metadata.get("tokenizer_revision", ""))
                )
                or None,
            )
        )
    return rows
