from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ChainSample
from .data import load_step_end_states


@dataclass(frozen=True)
class TaskExample:
    sample: ChainSample
    visible_steps: int
    boundary_step: int | None


@dataclass(frozen=True)
class TaskDataset:
    name: str
    claim_scope: str
    examples: tuple[TaskExample, ...]
    labels: np.ndarray
    left_truncated_step0_errors: int = 0

    def __post_init__(self) -> None:
        if self.labels.shape != (len(self.examples),):
            raise ValueError("task labels must align with examples")

    @property
    def groups(self) -> np.ndarray:
        return np.asarray(
            [row.sample.cluster_group for row in self.examples],
            dtype=object,
        )

    @property
    def problem_hashes(self) -> np.ndarray:
        return np.asarray(
            [row.sample.problem_hash for row in self.examples], dtype=object
        )

    @property
    def domains(self) -> np.ndarray:
        return np.asarray([row.sample.dataset for row in self.examples], dtype=object)


def build_whole_chain_task(samples: tuple[ChainSample, ...]) -> TaskDataset:
    examples = tuple(TaskExample(sample, sample.n_steps, None) for sample in samples)
    labels = np.asarray([sample.first_error_step >= 0 for sample in samples], dtype=np.int8)
    return TaskDataset(
        name="whole_chain",
        claim_scope="retrospective_information_ceiling",
        examples=examples,
        labels=labels,
    )


def build_strict_prefix_task(samples: tuple[ChainSample, ...]) -> TaskDataset:
    examples: list[TaskExample] = []
    labels: list[int] = []
    left_truncated = 0
    for sample in samples:
        gold = int(sample.first_error_step)
        if gold >= sample.n_steps:
            raise ValueError(f"chain {sample.chain_id}: first error exceeds n_steps")
        if gold == 0:
            left_truncated += 1
            continue
        last_boundary = gold if gold > 0 else sample.n_steps - 1
        for step in range(1, last_boundary + 1):
            examples.append(TaskExample(sample, visible_steps=step, boundary_step=step))
            labels.append(int(gold == step))
    return TaskDataset(
        name="strict_prefix",
        claim_scope="prospective_first_error_association",
        examples=tuple(examples),
        labels=np.asarray(labels, dtype=np.int8),
        left_truncated_step0_errors=left_truncated,
    )


def load_visible_states(example: TaskExample) -> np.ndarray:
    return load_step_end_states(example.sample, example.visible_steps)


def visible_output_steps(example: TaskExample) -> np.ndarray:
    # Each summary describes an already completed step; current/future summaries
    # are never exposed to a strict-prefix row.
    return np.asarray(example.sample.output_steps[: example.visible_steps], dtype=np.float32)


def nuisance_features(example: TaskExample) -> tuple[tuple[str, ...], np.ndarray]:
    ranges = example.sample.step_ranges
    if example.boundary_step is not None:
        last = example.visible_steps - 1
        values = np.asarray(
            [
                float(example.boundary_step),
                float(ranges[last, 1] - example.sample.response_start + 1),
                float(ranges[last, 1] - ranges[last, 0] + 1),
            ],
            dtype=np.float64,
        )
        return ("step_index", "prefix_token_count", "previous_step_length"), values
    lengths = ranges[:, 1] - ranges[:, 0] + 1
    return (
        "total_steps",
        "total_response_tokens",
        "mean_step_length",
    ), np.asarray([len(ranges), lengths.sum(), lengths.mean()], dtype=np.float64)
