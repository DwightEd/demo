from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..data import ChainRecord, load_chain_records
from ..flow_signature_data import FlowTrajectoryDataset, load_flow_trajectory_dataset
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
    dataset = load_flow_trajectory_dataset(
        path,
        vector_key=vector_key,
        layers=str(int(layer)),
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
        "exact_trace_records": exact,
        "rendered_prompt_records": rendered,
        "records_with_problem_text": has_problem,
        "legacy_problem_reconstruction_records": int(needs_reconstruction),
        "legacy_problem_reconstruction_required": bool(needs_reconstruction),
        "stored_vector_key": source.dataset.vector_key,
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
