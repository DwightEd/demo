from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..data import ChainRecord, load_chain_records
from ..flow_signature_data import FlowTrajectoryDataset
from ..teacher_forcing import prepare_teacher_forcing_trace


CUSTOM_ZEROSHOT_TEMPLATE = (
    "Solve the following grade-school math problem. Reason step by step, with "
    "one short step per line. Then end with a final line of exactly the form "
    "'#### <answer>' where <answer> is just the number.\n\n"
    "Problem: {question}"
)


@dataclass
class PullbackSource:
    dataset: FlowTrajectoryDataset
    records_by_original_index: dict[int, ChainRecord]
    prompt_style: str
    model_name: str
    dataset_provenance: str
    state_source: str


@dataclass(frozen=True)
class ProblemSourceSpec:
    dataset_format: str
    path: str
    subset: str
    split: str
    provenance: str

    def as_dict(self) -> dict[str, str]:
        return {
            "dataset_format": self.dataset_format,
            "path": self.path,
            "subset": self.subset,
            "split": self.split,
            "provenance": self.provenance,
        }


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").replace("$", "").replace("%", "").strip()
    text = text.rstrip(".")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _predicted_answer(text: str) -> float | None:
    marked = list(re.finditer(r"####\s*([^\n]+)", str(text)))
    if marked:
        value = _number(marked[-1].group(1))
        if value is not None:
            return value
    numbers = re.findall(r"[-+]?\$?\d[\d,]*\.?\d*", str(text))
    return _number(numbers[-1]) if numbers else None


def load_ordered_processbench_questions(
    path: str | Path,
    subset: str,
) -> list[str]:
    """Reproduce the ordered problem list used by 10_sample_and_extract.py."""

    from datasets import load_dataset

    dataset = load_dataset(str(path), split=str(subset))
    gold_fields = (
        "answer",
        "final_answer",
        "gt_answer",
        "ground_truth",
        "gold_answer",
    )
    problems: dict[str, float] = {}
    for example in dataset:
        problem = example.get("problem")
        if not problem:
            continue
        gold = None
        for field in gold_fields:
            if example.get(field) is not None:
                gold = _number(str(example[field]))
                if gold is not None:
                    break
        if gold is None:
            label = int(example.get("label", -1))
            final_correct = example.get("final_answer_correct", None)
            if label == -1 or final_correct is True:
                gold = _predicted_answer("\n".join(example.get("steps", []) or []))
        if gold is not None and str(problem) not in problems:
            problems[str(problem)] = float(gold)
    return list(problems)


def load_ordered_gsm8k_questions(
    path: str,
    subset: str,
    split: str,
) -> list[str]:
    """Reproduce the ordered direct-GSM8K problem axis used during sampling."""

    from datasets import load_dataset

    dataset = load_dataset(str(path), str(subset), split=str(split))
    questions: list[str] = []
    for example in dataset:
        question = example.get("question")
        answer = example.get("answer")
        if question and _number(str(answer)) is not None:
            questions.append(str(question))
    return questions


def _provenance_spec(value: str) -> ProblemSourceSpec | None:
    text = str(value).strip().replace("\\", "/")
    if ":" not in text:
        return None
    dataset_format, payload = text.split(":", 1)
    dataset_format = dataset_format.strip().lower()
    if dataset_format not in {"gsm8k", "processbench"} or "/" not in payload:
        return None
    path, subset = payload.rsplit("/", 1)
    if not path or not subset:
        return None
    return ProblemSourceSpec(
        dataset_format=dataset_format,
        path=path,
        subset=subset,
        split="test",
        provenance=text,
    )


def resolve_problem_source_spec(
    provenance: str,
    *,
    dataset_format: str = "auto",
    path: str = "",
    subset: str = "",
    split: str = "test",
) -> ProblemSourceSpec:
    """Resolve legacy prompt reconstruction without silently changing datasets."""

    requested_format = str(dataset_format).strip().lower()
    if requested_format not in {"auto", "gsm8k", "processbench"}:
        raise ValueError(f"unknown problem dataset format {dataset_format!r}")
    inferred = _provenance_spec(provenance)
    if requested_format == "auto" and inferred is not None:
        requested_format = inferred.dataset_format
    if requested_format == "auto":
        source_hint = str(path).replace("\\", "/").lower()
        if "processbench" in source_hint:
            requested_format = "processbench"
        elif source_hint:
            requested_format = "gsm8k"
        else:
            raise ValueError(
                "legacy artifact has no parseable `dataset` provenance. Pass "
                "--problem_format, --problem_source, and --problem_subset explicitly"
            )
    resolved_path = str(path).strip() or (inferred.path if inferred else "")
    resolved_subset = str(subset).strip() or (inferred.subset if inferred else "")
    if not resolved_path:
        resolved_path = (
            "data/hf_datasets/ProcessBench"
            if requested_format == "processbench"
            else "openai/gsm8k"
        )
    if not resolved_subset:
        resolved_subset = "gsm8k" if requested_format == "processbench" else "main"
    return ProblemSourceSpec(
        dataset_format=requested_format,
        path=resolved_path,
        subset=resolved_subset,
        split=str(split),
        provenance=str(provenance),
    )


def load_ordered_problem_questions(spec: ProblemSourceSpec) -> list[str]:
    if spec.dataset_format == "processbench":
        return load_ordered_processbench_questions(spec.path, spec.subset)
    if spec.dataset_format == "gsm8k":
        return load_ordered_gsm8k_questions(spec.path, spec.subset, spec.split)
    raise ValueError(f"unsupported problem source format {spec.dataset_format!r}")


def validate_problem_question_map(
    source: PullbackSource,
    ordered_questions: list[str],
    spec: ProblemSourceSpec,
) -> dict[str, Any]:
    required = []
    for original in source.dataset.original_indices:
        record = source.records_by_original_index[int(original)]
        if (
            record.exact_input_ids is None
            and not record.rendered_prompt
            and not record.problem
        ):
            required.append(int(record.problem_id))
    if not required:
        return {"required_problem_ids": 0, "question_map_size": len(ordered_questions)}
    invalid = sorted({value for value in required if value < 0 or value >= len(ordered_questions)})
    if invalid:
        raise ValueError(
            "reconstructed question map does not cover artifact problem IDs: "
            f"missing={invalid[:10]}, max_required={max(required)}, "
            f"map_size={len(ordered_questions)}, source={spec.as_dict()}. "
            "The NPZ and question source are not the same extraction cohort."
        )
    return {
        "required_problem_ids": len(set(required)),
        "maximum_required_problem_id": max(required),
        "question_map_size": len(ordered_questions),
    }


def _pool_raw_cloud_steps(cloud: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Pool raw token hidden states exactly like the replay step-exp operator."""

    cloud = np.asarray(cloud, dtype=np.float32)
    sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
    if cloud.ndim != 2:
        raise ValueError(f"expected raw token cloud [token, hidden], got {cloud.shape}")
    if sizes.size == 0 or np.any(sizes <= 0) or int(sizes.sum()) != cloud.shape[0]:
        raise ValueError("cloud_sizes does not partition the raw token cloud")
    rows: list[np.ndarray] = []
    start = 0
    for width in sizes.tolist():
        stop = start + int(width)
        local = cloud[start:stop]
        if int(width) == 1:
            rows.append(local[0])
        else:
            position = np.linspace(0.0, 1.0, int(width), dtype=np.float32)
            weight = np.exp(position - float(np.max(position)))
            weight /= float(np.sum(weight))
            rows.append(np.sum(local * weight[:, None], axis=0, dtype=np.float32))
        start = stop
    return np.stack(rows).astype(np.float32, copy=False)


def _raw_hidden_dataset_from_clouds(
    path: str | Path,
    *,
    vector_key: str,
    layer: int,
    label_policy: str,
    max_samples: int,
) -> FlowTrajectoryDataset:
    from ..directional_consensus import load_directional_cloud_dataset

    cloud = load_directional_cloud_dataset(
        path,
        vector_key=vector_key,
        cloud_layers=str(int(layer)),
        label_policy=label_policy,
        max_samples=max_samples,
    )
    if cloud.cloud_layer_ids.tolist() != [int(layer)]:
        raise ValueError(
            f"raw token clouds do not contain requested layer {layer}; "
            f"selected={cloud.cloud_layer_ids.tolist()}"
        )
    trajectories = []
    for values, sizes in zip(cloud.clouds, cloud.step_sizes):
        pooled = _pool_raw_cloud_steps(values[:, 0, :], sizes)
        trajectories.append(np.ascontiguousarray(pooled[:, None, :]))
    base = cloud.base
    metadata = dict(base.metadata)
    metadata.update(
        {
            "alignment_vector_key": base.vector_key,
            "state_source": "sv_clouds",
            "state_pooling": "raw_hidden_step_exp",
            "state_representation": "raw_hidden_state",
            "cloud_layer_ids": cloud.cloud_layer_ids.tolist(),
            "cloud_hidden_dim": int(cloud.cloud_hidden_dim),
        }
    )
    return FlowTrajectoryDataset(
        source_path=base.source_path,
        vector_key="sv_clouds:raw_hidden_step_exp",
        trajectories=trajectories,
        original_indices=base.original_indices.copy(),
        problem_ids=base.problem_ids.copy(),
        sample_idx=base.sample_idx.copy(),
        y_error=base.y_error.copy(),
        is_correct=base.is_correct.copy(),
        n_steps=base.n_steps.copy(),
        response_chars=base.response_chars.copy(),
        layer_ids=cloud.cloud_layer_ids.copy(),
        hidden_dim=int(cloud.cloud_hidden_dim),
        label_policy=base.label_policy,
        skipped={**base.skipped, **{f"cloud_{k}": v for k, v in cloud.skipped_clouds.items()}},
        metadata=metadata,
    )


def build_legacy_rendered_prompt(
    tokenizer,
    question: str,
    prompt_style: str,
) -> str:
    if str(prompt_style) != "custom_zeroshot":
        raise ValueError(
            "legacy prompt reconstruction currently supports only custom_zeroshot; "
            "use an exact-trace artifact for few-shot prompts"
        )
    messages = [
        {
            "role": "user",
            "content": CUSTOM_ZEROSHOT_TEMPLATE.format(question=str(question)),
        }
    ]
    return str(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    )


def load_pullback_source(
    path: str | Path,
    *,
    vector_key: str,
    layer: int,
    label_policy: str,
    max_samples: int,
) -> PullbackSource:
    # Residual interventions require ambient model hidden states.  The legacy
    # ``sv_vec_*`` arrays may live in a projected reasoning basis (467 dims in
    # the canonical artifact) and are therefore alignment metadata only.
    dataset = _raw_hidden_dataset_from_clouds(
        path,
        vector_key=vector_key,
        layer=int(layer),
        label_policy=label_policy,
        max_samples=max_samples,
    )
    records = load_chain_records(path, input_format="npz", max_chains=0)
    records_by_original = {int(record.chain_idx): record for record in records}
    missing = [
        int(index)
        for index in dataset.original_indices
        if int(index) not in records_by_original
    ]
    if missing:
        raise ValueError(f"text records are missing original rows {missing[:10]}")
    return PullbackSource(
        dataset=dataset,
        records_by_original_index=records_by_original,
        prompt_style=str(dataset.metadata.get("prompt_style", "")),
        model_name=str(dataset.metadata.get("model_name", "")),
        dataset_provenance=str(dataset.metadata.get("dataset_provenance", "")),
        state_source=str(dataset.metadata.get("state_source", "")),
    )


def source_preflight(source: PullbackSource) -> dict[str, Any]:
    records = [
        source.records_by_original_index[int(index)]
        for index in source.dataset.original_indices
    ]
    exact = sum(record.exact_input_ids is not None for record in records)
    rendered = sum(bool(record.rendered_prompt) for record in records)
    has_problem = sum(bool(record.problem) for record in records)
    needs_reconstruction = sum(
        record.exact_input_ids is None
        and not record.rendered_prompt
        and not record.problem
        for record in records
    )
    return {
        "path": source.dataset.source_path,
        "samples": source.dataset.n_samples,
        "errors": int(source.dataset.y_error.sum()),
        "correct": int((source.dataset.y_error == 0).sum()),
        "problems": int(np.unique(source.dataset.problem_ids).size),
        "layer": int(source.dataset.layer_ids[0]),
        "hidden_dim": int(source.dataset.hidden_dim),
        "prompt_style": source.prompt_style,
        "model_name": source.model_name,
        "dataset_provenance": source.dataset_provenance,
        "state_source": source.state_source,
        "state_representation": source.dataset.metadata.get(
            "state_representation", "unknown"
        ),
        "state_pooling": source.dataset.metadata.get("state_pooling", "unknown"),
        "alignment_vector_key": source.dataset.metadata.get(
            "alignment_vector_key", ""
        ),
        "exact_trace_records": exact,
        "rendered_prompt_records": rendered,
        "records_with_problem_text": has_problem,
        "legacy_problem_reconstruction_records": int(needs_reconstruction),
        "legacy_problem_reconstruction_required": bool(needs_reconstruction),
        "stored_state_key": source.dataset.vector_key,
        "residual_intervention_ready": bool(
            source.state_source == "sv_clouds"
            and source.dataset.metadata.get("state_representation")
            == "raw_hidden_state"
        ),
        "label_policy": source.dataset.label_policy,
        "skipped": source.dataset.skipped,
    }


def prepare_record_trace(
    record: ChainRecord,
    tokenizer,
    *,
    prompt_style: str,
    ordered_questions: list[str] | None,
    max_seq_len: int,
) -> tuple[dict[str, Any], str]:
    if record.rendered_prompt:
        prompt = str(record.rendered_prompt)
    elif record.problem:
        prompt = build_legacy_rendered_prompt(tokenizer, record.problem, prompt_style)
    elif record.exact_input_ids is not None:
        # Exact IDs, offsets, and step ranges define the complete replay axis.
        # ``prepare_teacher_forcing_trace`` does not tokenize this placeholder.
        prompt = ""
    else:
        if ordered_questions is None:
            raise ValueError(
                "legacy artifact has no prompt text; provide --problem_source and "
                "--problem_subset"
            )
        if record.problem_id < 0 or record.problem_id >= len(ordered_questions):
            raise ValueError(
                f"problem_id={record.problem_id} is outside reconstructed problem map"
            )
        prompt = build_legacy_rendered_prompt(
            tokenizer,
            ordered_questions[record.problem_id],
            prompt_style,
        )
    trace = prepare_teacher_forcing_trace(
        tokenizer,
        prompt,
        str(record.response),
        steps=record.steps,
        max_seq_len=int(max_seq_len),
        exact_input_ids=record.exact_input_ids,
        exact_attention_mask=record.exact_attention_mask,
        exact_token_offsets=record.exact_token_offsets,
        exact_step_token_ranges=record.exact_step_token_ranges,
        exact_response_start_token=record.exact_response_start_token,
    )
    kind = str(trace["replay_kind"])
    if record.exact_input_ids is None:
        kind = "legacy_reconstructed_prompt_and_retokenized_response"
    trace["replay_kind"] = kind
    return trace, prompt
