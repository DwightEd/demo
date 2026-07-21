from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from .contracts import FirstErrorLabels


@dataclass(frozen=True)
class ShardSpec:
    num_shards: int = 1
    shard_index: int = 0

    def __post_init__(self) -> None:
        if self.num_shards <= 0 or not 0 <= self.shard_index < self.num_shards:
            raise ValueError("shard_index must lie inside a positive shard count")

    def includes(self, global_index: int) -> bool:
        if global_index < 0:
            raise ValueError("global_index cannot be negative")
        return global_index % self.num_shards == self.shard_index

    @property
    def tag(self) -> str:
        return f"shard-{self.shard_index:03d}-of-{self.num_shards:03d}"


@dataclass(frozen=True)
class ProcessBenchRecord:
    trace_id: str
    problem_id: str
    question: str
    steps: tuple[str, ...]
    labels: FirstErrorLabels
    generator_model: str

    @classmethod
    def from_mapping(
        cls, row: Mapping[str, object], *, index: int
    ) -> "ProcessBenchRecord":
        question = row.get("problem", row.get("question"))
        steps = row.get("steps")
        first_error = row.get(
            "label",
            row.get(
                "gold_step", row.get("gold_error_step", row.get("first_error_step"))
            ),
        )
        generator = row.get(
            "generator", row.get("generator_model", row.get("source_model"))
        )
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"row {index} has no question")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise ValueError(f"row {index} has no reasoning steps")
        clean_steps = tuple(str(step) for step in steps)
        if not clean_steps or any(not step.strip() for step in clean_steps):
            raise ValueError(f"row {index} contains an empty reasoning step")
        if isinstance(first_error, bool) or not isinstance(
            first_error, (int, np.integer)
        ):
            raise ValueError(f"row {index} has no integer first-error label")
        if not isinstance(generator, str) or not generator.strip():
            raise ValueError(f"row {index} has no generator model")
        problem_id = row.get("problem_id", row.get("question_id"))
        if problem_id is None:
            digest = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:20]
            problem_id = f"question-{digest}"
        trace_id = row.get("id", row.get("sample_id", index))
        return cls(
            trace_id=str(trace_id),
            problem_id=str(problem_id),
            question=question.strip(),
            steps=clean_steps,
            labels=FirstErrorLabels(len(clean_steps), int(first_error)),
            generator_model=generator.strip(),
        )


class ProcessBenchReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def records(self) -> Iterator[ProcessBenchRecord]:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        text = self.path.read_text(encoding="utf-8")
        if self.path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload = json.loads(text)
            rows = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(
                "ProcessBench input must be a JSON array or {'data': [...]} "
            )
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"row {index} is not an object")
            yield ProcessBenchRecord.from_mapping(row, index=index)


@dataclass(frozen=True)
class RenderedTrace:
    text: str
    prompt_end_char: int
    step_char_spans: np.ndarray


class PlainReasoningRenderer:
    def __init__(self, *, separator: str = "\n\n") -> None:
        self.separator = separator

    def render(self, record: ProcessBenchRecord) -> RenderedTrace:
        prompt = record.question.rstrip() + self.separator
        response = self.separator.join(record.steps)
        spans: list[tuple[int, int]] = []
        cursor = len(prompt)
        for index, step in enumerate(record.steps):
            start = cursor
            stop = start + len(step)
            spans.append((start, stop))
            cursor = stop + (
                len(self.separator) if index + 1 < len(record.steps) else 0
            )
        return RenderedTrace(
            text=prompt + response,
            prompt_end_char=len(prompt),
            step_char_spans=np.asarray(spans, dtype=np.int64),
        )


@dataclass(frozen=True)
class TokenizedTrace:
    input_ids: np.ndarray
    prompt_end: int
    step_ranges: np.ndarray


class TokenizerAligner:
    """Map step text to one exact token axis; boundary whitespace is unlabelled."""

    def tokenize(self, tokenizer, rendered: RenderedTrace) -> TokenizedTrace:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("CCT extraction requires a fast tokenizer with offsets")
        encoded = tokenizer(
            rendered.text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors=None,
        )
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
        step_ranges = self.align_offsets(offsets, rendered.step_char_spans)
        prompt_end = int(step_ranges[0, 0])
        if prompt_end <= 0:
            raise ValueError("prompt maps to no model tokens")
        return TokenizedTrace(input_ids, prompt_end, step_ranges)

    @staticmethod
    def align_offsets(offsets: np.ndarray, spans: np.ndarray) -> np.ndarray:
        offsets = np.asarray(offsets, dtype=np.int64)
        spans = np.asarray(spans, dtype=np.int64)
        if offsets.ndim != 2 or offsets.shape[1] != 2:
            raise ValueError("offsets must have shape [tokens, 2]")
        if spans.ndim != 2 or spans.shape[1] != 2 or not len(spans):
            raise ValueError("step spans must have shape [steps, 2]")
        owners: dict[int, int] = {}
        ranges: list[tuple[int, int]] = []
        for step, (start, stop) in enumerate(spans):
            members = [
                token
                for token, (left, right) in enumerate(offsets)
                if right > left and left < stop and right > start
            ]
            if not members:
                raise ValueError(f"step {step} maps to no token")
            overlap = [token for token in members if token in owners]
            if overlap:
                raise ValueError(f"tokens {overlap} overlap multiple reasoning steps")
            if members != list(range(members[0], members[-1] + 1)):
                raise ValueError(f"step {step} token range is not contiguous")
            owners.update((token, step) for token in members)
            ranges.append((members[0], members[-1] + 1))
        if any(
            ranges[index][0] < ranges[index - 1][1] for index in range(1, len(ranges))
        ):
            raise ValueError("step token ranges overlap")
        return np.asarray(ranges, dtype=np.int64)
