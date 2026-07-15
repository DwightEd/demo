from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from prompt_control_flow.schema import inspect_npz_schema


PROCESSBENCH_REQUIRED_FIELDS = {
    "id",
    "generator",
    "problem",
    "steps",
    "final_answer_correct",
    "label",
}

DATA_CONTRACT_VERSION = "reasoning_trace_contract_v2"


def _json_scalar(z: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    if key not in z.files:
        return {}
    try:
        value = str(np.asarray(z[key]).item())
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _scalar_string(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    try:
        return str(np.asarray(z[key]).item())
    except Exception:
        return default


def _scalar_bool(z: np.lib.npyio.NpzFile, key: str, default: bool = False) -> bool:
    if key not in z.files:
        return default
    try:
        return bool(np.asarray(z[key]).item())
    except Exception:
        return default


def _sample_count(z: np.lib.npyio.NpzFile) -> int:
    for key in (
        "ids",
        "chain_idx",
        "problem_ids",
        "n_steps",
        "responses",
        "gold_error_step",
        "is_correct",
    ):
        if key in z.files:
            try:
                return int(len(z[key]))
            except TypeError:
                continue
    return 0


def _small_vector(z: np.lib.npyio.NpzFile, keys: Iterable[str]) -> np.ndarray | None:
    for key in keys:
        if key not in z.files:
            continue
        try:
            value = np.asarray(z[key]).reshape(-1)
        except Exception:
            continue
        return value
    return None


def _label_counts(values: np.ndarray | None, *, error_when_nonnegative: bool) -> dict[str, int]:
    if values is None:
        return {"known": 0, "correct": 0, "error": 0, "unknown": 0}
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return {"known": 0, "correct": 0, "error": 0, "unknown": int(len(values))}
    finite = np.isfinite(numeric)
    if error_when_nonnegative:
        known = finite
        error = known & (numeric >= 0)
        correct = known & (numeric < 0)
    else:
        known = finite & np.isin(numeric, (0, 1))
        error = known & (numeric == 0)
        correct = known & (numeric == 1)
    return {
        "known": int(known.sum()),
        "correct": int(correct.sum()),
        "error": int(error.sum()),
        "unknown": int(len(numeric) - known.sum()),
    }


def _same_problem_counts(
    problem_ids: np.ndarray | None,
    labels: np.ndarray | None,
) -> dict[str, int]:
    if problem_ids is None:
        return {
            "problems": 0,
            "repeated_problems": 0,
            "contrastive_problems": 0,
            "max_samples_per_problem": 0,
        }
    groups: dict[str, list[int]] = defaultdict(list)
    numeric_labels: np.ndarray | None = None
    if labels is not None:
        try:
            numeric_labels = np.asarray(labels, dtype=np.int64)
        except (TypeError, ValueError):
            numeric_labels = None
    for i, value in enumerate(problem_ids):
        groups[str(value)].append(i)
    contrastive = 0
    if numeric_labels is not None and len(numeric_labels) == len(problem_ids):
        for indices in groups.values():
            values = numeric_labels[np.asarray(indices, dtype=np.int64)]
            if np.any(values == 0) and np.any(values == 1):
                contrastive += 1
    counts = [len(indices) for indices in groups.values()]
    return {
        "problems": int(len(groups)),
        "repeated_problems": int(sum(count > 1 for count in counts)),
        "contrastive_problems": int(contrastive),
        "max_samples_per_problem": int(max(counts, default=0)),
    }


def _infer_trace_semantics(
    *,
    files: set[str],
    sampling_metadata: Mapping[str, Any],
    exact_trace_complete: bool,
    has_same_problem_axis: bool,
    has_process_labels: bool,
) -> str:
    explicit = str(
        sampling_metadata.get("trace_semantics", "")
        or sampling_metadata.get("source_mode", "")
    )
    if explicit:
        if explicit == "sampled_generation_then_exact_teacher_forcing":
            return "generation_matched_teacher_forcing"
        if explicit == "teacher_forced_processbench":
            return "benchmark_observer_teacher_forcing"
        return explicit
    if exact_trace_complete and "generated_token_ids" in files:
        return "generation_matched_teacher_forcing"
    if exact_trace_complete and has_process_labels:
        return "benchmark_observer_teacher_forcing"
    if has_same_problem_axis:
        return "legacy_same_problem_states_without_exact_token_axis"
    if has_process_labels:
        return "legacy_benchmark_observer_states_without_exact_token_axis"
    return "unknown"


def inspect_reasoning_npz(path: str | Path) -> dict[str, Any]:
    """Classify an NPZ by the research claims its stored trace can support."""

    path = Path(path)
    schema = inspect_npz_schema(path)
    with np.load(path, allow_pickle=True) as z:
        files = set(str(key) for key in z.files)
        sampling_metadata = _json_scalar(z, "model_sampling_metadata_json")
        n = _sample_count(z)

        gold = _small_vector(z, ("gold_error_step", "labels"))
        process_correct = _small_vector(z, ("process_correct",))
        final_correct = _small_vector(
            z,
            ("final_answer_correct", "is_correct_strict", "is_correct"),
        )
        problem_ids = _small_vector(z, ("problem_ids",))
        truncated = _small_vector(z, ("model_input_truncated",))
        process_label_counts = (
            _label_counts(gold, error_when_nonnegative=True)
            if gold is not None
            else _label_counts(process_correct, error_when_nonnegative=False)
        )
        final_answer_label_counts = _label_counts(
            final_correct, error_when_nonnegative=False
        )
        has_process_labels = process_label_counts["known"] > 0
        has_final_answer_labels = final_answer_label_counts["known"] > 0
        same_problem = _same_problem_counts(problem_ids, final_correct)
        has_same_problem_axis = bool(
            "problem_ids" in files
            and "sample_idx" in files
            and same_problem["repeated_problems"] > 0
        )
        has_raw_response_hidden = bool(
            {"sv_clouds", "cloud_sizes", "cloud_layers"}.issubset(files)
            or schema["has_hidden_shards"]
        )
        has_prompt_hidden = bool(schema["has_prompt_hidden"])
        has_token_output_features = bool(
            {
                "sv_tok_entropy",
                "sv_tok_committal",
                "tok_U_D",
                "token_features",
                "chosen_token_logprobs",
                "token_entropy",
                "top1_top2_margin",
            }
            & files
        )
        has_generated_ids = bool(
            "generated_token_ids" in files
            and _scalar_bool(z, "generated_token_ids_stored", True)
        )
        exact_trace_complete = bool(schema["exact_trace_complete"])
        trace_semantics = _scalar_string(z, "trace_semantics") or _infer_trace_semantics(
            files=files,
            sampling_metadata=sampling_metadata,
            exact_trace_complete=exact_trace_complete,
            has_same_problem_axis=has_same_problem_axis,
            has_process_labels=has_process_labels,
        )
        data_contract_version = _scalar_string(
            z, "data_contract_version", "legacy_or_unspecified"
        )
        source_model = _scalar_string(z, "model_name") or str(
            sampling_metadata.get("model_name", "")
        )
        is_correct_semantics = _scalar_string(z, "is_correct_semantics")
        label_semantics = _scalar_string(z, "label_semantics") or str(
            sampling_metadata.get("label_semantics", "")
        )
        if truncated is None:
            truncation_rate = None
            has_truncated_inputs = False
        else:
            truncated_bool = np.asarray(truncated, dtype=bool)
            truncation_rate = float(np.mean(truncated_bool)) if len(truncated_bool) else 0.0
            has_truncated_inputs = bool(np.any(truncated_bool))
    generation_matched = bool(
        exact_trace_complete
        and has_generated_ids
        and trace_semantics == "generation_matched_teacher_forcing"
    )
    observer_trace = trace_semantics in {
        "benchmark_observer_teacher_forcing",
        "teacher_forced_processbench",
        "legacy_benchmark_observer_states_without_exact_token_axis",
    }

    if generation_matched:
        evidence_tier = "generation_matched_self_trace"
    elif observer_trace and exact_trace_complete:
        evidence_tier = "exact_benchmark_observer_trace"
    elif has_same_problem_axis:
        evidence_tier = "legacy_same_problem_trace"
    elif has_process_labels:
        evidence_tier = "legacy_benchmark_observer_trace"
    else:
        evidence_tier = "unclassified"

    warnings: list[str] = []
    if schema["exact_trace_declared"] and not exact_trace_complete:
        warnings.append("partial exact-token trace: fail closed instead of re-tokenizing")
    if not exact_trace_complete:
        warnings.append("exact prompt/input token axis is unavailable")
    if observer_trace:
        warnings.append(
            "ProcessBench observer states are not the original solution generator states"
        )
        warnings.append(
            "published ProcessBench responses were reformatted before expert annotation"
        )
    if has_process_labels and "final_answer_correct" not in files and "is_correct" in files:
        warnings.append(
            "legacy is_correct semantics may mean final-answer correctness; use gold_error_step for process-error labels"
        )
    if has_same_problem_axis and not has_process_labels:
        warnings.append(
            "same-problem artifact has response labels only and cannot localize the first error"
        )
    if generation_matched and not has_raw_response_hidden:
        warnings.append("exact generation trace has no raw hidden states for geometric intervention")
    if has_truncated_inputs:
        warnings.append(
            "some model inputs were truncated; exclude them from full-chain or causal claims"
        )

    if "process_correct" not in files and has_process_labels:
        warnings.append(
            "process_correct is implicit; map -1 to correct, nonnegative to "
            "error, and -2 to unavailable"
        )
    if "final_answer_correct" not in files and has_final_answer_labels:
        warnings.append(
            "final-answer labels are exposed only through a legacy is_correct alias"
        )

    capabilities = {
        "benchmark_first_error_diagnosis": bool(
            has_process_labels and schema["has_steps_text"]
        ),
        "response_process_error_diagnosis": bool(has_process_labels),
        "same_problem_response_contrast": bool(
            has_same_problem_axis and has_final_answer_labels
        ),
        "exact_observer_teacher_forcing": bool(observer_trace and exact_trace_complete),
        "generation_matched_self_state": generation_matched,
        "raw_response_hidden_geometry": has_raw_response_hidden,
        "prompt_hidden_geometry": has_prompt_hidden,
        "geometry_conditioned_on_output_features": bool(
            has_raw_response_hidden and has_token_output_features
        ),
        "confirmatory_causal_intervention": bool(
            generation_matched and has_raw_response_hidden and not has_truncated_inputs
        ),
    }

    return {
        "path": str(path),
        "data_contract_version": data_contract_version,
        "evidence_tier": evidence_tier,
        "trace_semantics": trace_semantics,
        "label_semantics": label_semantics or "legacy_or_unspecified",
        "is_correct_semantics": is_correct_semantics or "legacy_or_unspecified",
        "samples": n,
        "process_label_counts": process_label_counts,
        "final_answer_label_counts": final_answer_label_counts,
        "same_problem": same_problem,
        "source_model": source_model,
        "source_mode": str(sampling_metadata.get("source_mode", "")),
        "exact_trace_complete": exact_trace_complete,
        "generated_token_ids_stored": has_generated_ids,
        "raw_response_token_hidden_stored": has_raw_response_hidden,
        "prompt_token_hidden_stored": has_prompt_hidden,
        "token_output_features_stored": has_token_output_features,
        "model_input_truncation_rate": truncation_rate,
        "capabilities": capabilities,
        "warnings": warnings,
        "schema": schema,
        "ready": bool(evidence_tier != "unclassified"),
    }


def _contrastive_group_count(
    rows: list[dict[str, Any]],
    key_fn,
) -> tuple[int, int, int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    repeated = 0
    contrastive = 0
    max_count = 0
    for values in groups.values():
        max_count = max(max_count, len(values))
        repeated += int(len(values) > 1)
        labels = [int(value.get("label", -1)) for value in values]
        contrastive += int(any(label < 0 for label in labels) and any(label >= 0 for label in labels))
    return repeated, contrastive, max_count


def _summarize_processbench_rows(
    path: Path,
    parsed_rows: list[dict[str, Any]],
    *,
    malformed: list[dict[str, Any]],
    source_records: int,
) -> dict[str, Any]:
    missing_fields: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for value in parsed_rows:
        for field in PROCESSBENCH_REQUIRED_FIELDS - set(value):
            missing_fields[field] += 1
        rows.append(value)

    repeated, contrastive, max_per_problem = _contrastive_group_count(
        rows, lambda row: row.get("problem", "")
    )
    same_model_repeated, same_model_contrastive, max_same_model = (
        _contrastive_group_count(
            rows,
            lambda row: (row.get("problem", ""), row.get("generator", "")),
        )
    )
    labels = np.asarray([int(row.get("label", -1)) for row in rows], dtype=np.int64)
    final = np.asarray(
        [int(bool(row.get("final_answer_correct", False))) for row in rows],
        dtype=np.int8,
    )
    process_error_with_correct_answer = int(np.sum((labels >= 0) & (final == 1)))
    generators = Counter(str(row.get("generator", "")) for row in rows)
    ready = bool(not malformed and not missing_fields and rows)
    recommendation = (
        "source is structurally ready for benchmark-observer extraction"
        if ready
        else "repair or refresh the source before extraction; never skip malformed rows"
    )
    return {
        "path": str(path),
        "kind": "processbench_source",
        "source_records": int(source_records),
        "valid_records": int(len(rows)),
        "malformed_records": malformed,
        "missing_required_fields": dict(sorted(missing_fields.items())),
        "process_correct": int(np.sum(labels < 0)),
        "process_error": int(np.sum(labels >= 0)),
        "final_answer_correct": int(np.sum(final == 1)),
        "final_answer_error": int(np.sum(final == 0)),
        "process_error_with_correct_final_answer": process_error_with_correct_answer,
        "unique_problems": int(len({str(row.get("problem", "")) for row in rows})),
        "repeated_problems": int(repeated),
        "contrastive_problems": int(contrastive),
        "max_samples_per_problem": int(max_per_problem),
        "repeated_same_model_problems": int(same_model_repeated),
        "contrastive_same_model_problems": int(same_model_contrastive),
        "max_same_model_samples_per_problem": int(max_same_model),
        "generators": dict(sorted(generators.items())),
        "original_generation_prompt_available": False,
        "original_generation_token_trace_available": False,
        "published_solution_reformatted": True,
        "supported_estimand": "frozen_observer_detection_of_published_candidate_solution",
        "unsupported_estimand": "original_generator_online_internal_state",
        "ready": ready,
        "recommendation": recommendation,
    }


def inspect_processbench_jsonl(path: str | Path) -> dict[str, Any]:
    """Validate a raw ProcessBench JSONL without silently dropping bad rows."""

    path = Path(path)
    rows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    nonempty_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            nonempty_lines += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed.append(
                    {
                        "line": int(line_number),
                        "error": f"{exc.msg} at column {exc.colno}",
                    }
                )
                continue
            if not isinstance(value, dict):
                malformed.append(
                    {"line": int(line_number), "error": "record is not a JSON object"}
                )
                continue
            rows.append(value)
    return _summarize_processbench_rows(
        path,
        rows,
        malformed=malformed,
        source_records=nonempty_lines,
    )


def inspect_processbench_json(path: str | Path) -> dict[str, Any]:
    """Validate a JSON list (or a split-to-list mapping) as ProcessBench data."""

    path = Path(path)
    malformed: list[dict[str, Any]] = []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _summarize_processbench_rows(
            path,
            [],
            malformed=[{"record": 0, "error": f"{exc.msg} at position {exc.pos}"}],
            source_records=0,
        )
    if isinstance(value, dict) and PROCESSBENCH_REQUIRED_FIELDS.issubset(value):
        candidates = [value]
    elif isinstance(value, dict):
        candidates = [
            item
            for items in value.values()
            if isinstance(items, list)
            for item in items
        ]
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = []
        malformed.append({"record": 0, "error": "top-level JSON is not a list or split mapping"})
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(candidates):
        if isinstance(item, dict):
            rows.append(item)
        else:
            malformed.append({"record": int(index), "error": "record is not a JSON object"})
    return _summarize_processbench_rows(
        path,
        rows,
        malformed=malformed,
        source_records=len(candidates),
    )


def inspect_reasoning_artifact(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "kind": "missing", "ready": False}
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return {"kind": "reasoning_npz", **inspect_reasoning_npz(path)}
    if suffix == ".jsonl":
        return inspect_processbench_jsonl(path)
    if suffix == ".json":
        return inspect_processbench_json(path)
    return {
        "path": str(path),
        "kind": "unsupported",
        "ready": False,
        "recommendation": "use a ProcessBench JSONL or reasoning NPZ artifact",
    }
