from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TraceMeta:
    trace_id: str
    group_id: str
    group_is_fallback: bool
    split: Optional[str]
    response_label: Optional[int]
    gold_step: Optional[int]
    num_steps: int
    num_response_tokens: int
    generator_model: Optional[str]


@dataclass(frozen=True)
class FixedHoldoutConfig:
    seed: int
    validation_ratio: float
    test_ratio: float
    allow_official_resplit: bool = False
    allow_trace_as_group: bool = False

    @property
    def train_ratio(self) -> float:
        return 1.0 - self.validation_ratio - self.test_ratio

    def validate(self) -> None:
        if (
            not 0.0 < self.validation_ratio < 1.0
            or not 0.0 < self.test_ratio < 1.0
            or self.train_ratio <= 0.0
        ):
            raise ValueError(
                "validation/test ratios must be positive and sum to less than one"
            )


@dataclass(frozen=True)
class BalanceStats:
    groups: int
    traces: int
    negative_traces: int
    positive_traces: int
    early_errors: int
    middle_errors: int
    late_errors: int
    response_tokens: int

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "BalanceStats":
        return cls(*(int(round(value)) for value in vector))


@dataclass(frozen=True)
class PartitionManifest:
    name: str
    indices: tuple[int, ...]
    trace_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    balance: BalanceStats


@dataclass(frozen=True)
class SplitAssignment:
    train: PartitionManifest
    validation: PartitionManifest
    test: PartitionManifest
    seed: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float

    def manifest(self) -> dict:
        partitions = (self.train, self.validation, self.test)
        return {
            "mode": "fixed_holdout",
            "split_seed": self.seed,
            "ratios": {
                "train": self.train_ratio,
                "validation": self.validation_ratio,
                "test": self.test_ratio,
            },
            "assignment": "greedy_group_stratified_response_error_position_length",
            "partition_balance": {
                partition.name: asdict(partition.balance) for partition in partitions
            },
            "partition_trace_ids": {
                partition.name: list(partition.trace_ids) for partition in partitions
            },
            "partition_group_ids": {
                partition.name: list(partition.group_ids) for partition in partitions
            },
        }


class FixedHoldoutSplitter:
    """Deterministic problem-disjoint split with label/position/length balance."""

    _NAMES = ("train", "validation", "test")

    def __init__(self, config: FixedHoldoutConfig) -> None:
        config.validate()
        self.config = config

    def split(self, traces: Sequence[TraceMeta]) -> SplitAssignment:
        self._validate_cohort(traces)
        groups: dict[str, list[int]] = {}
        for index, trace in enumerate(traces):
            groups.setdefault(trace.group_id, []).append(index)
        if len(groups) < 3:
            raise ValueError("fixed holdout requires at least three problem groups")

        items = [
            (group_id, indices, self._balance_vector(traces, indices))
            for group_id, indices in groups.items()
        ]
        rng = np.random.default_rng(self.config.seed)
        rng.shuffle(items)
        total = np.sum([item[2] for item in items], axis=0)
        ratios = np.asarray(
            [
                self.config.train_ratio,
                self.config.validation_ratio,
                self.config.test_ratio,
            ]
        )
        targets = ratios[:, None] * total[None, :]
        scales = np.maximum(targets, 1.0)
        global_scale = np.maximum(total, 1.0)
        items.sort(
            key=lambda item: (
                float(np.max(item[2] / global_scale)),
                float(item[2][1]),
            ),
            reverse=True,
        )

        indices: list[list[int]] = [[], [], []]
        observed = np.zeros_like(targets)
        for _, group_indices, vector in items:
            destination = min(
                range(3),
                key=lambda partition: self._assignment_cost(
                    observed, targets, scales, vector, partition
                ),
            )
            indices[destination].extend(group_indices)
            observed[destination] += vector
        indices = [sorted(partition) for partition in indices]
        if any(not partition for partition in indices):
            raise ValueError("fixed holdout assignment produced an empty partition")
        self._assert_group_disjoint(traces, indices)

        manifests = tuple(
            self._manifest(name, partition, traces, observed[position])
            for position, (name, partition) in enumerate(zip(self._NAMES, indices))
        )
        return SplitAssignment(
            train=manifests[0],
            validation=manifests[1],
            test=manifests[2],
            seed=self.config.seed,
            train_ratio=self.config.train_ratio,
            validation_ratio=self.config.validation_ratio,
            test_ratio=self.config.test_ratio,
        )

    def _validate_cohort(self, traces: Sequence[TraceMeta]) -> None:
        official = [trace.trace_id for trace in traces if trace.split is not None]
        if official and not self.config.allow_official_resplit:
            raise ValueError(f"found official split metadata on {len(official)} traces")
        fallback = [trace.trace_id for trace in traces if trace.group_is_fallback]
        if fallback and not self.config.allow_trace_as_group:
            raise ValueError(f"{len(fallback)} traces lack a real problem/question id")

    @staticmethod
    def _balance_vector(
        traces: Sequence[TraceMeta], indices: Sequence[int]
    ) -> np.ndarray:
        vector = np.zeros(8, dtype=np.float64)
        vector[:2] = (1.0, len(indices))
        for index in indices:
            trace = traces[index]
            if trace.response_label in (0, 1):
                vector[2 + int(trace.response_label)] += 1.0
            if trace.gold_step is not None and trace.gold_step >= 0 and trace.num_steps:
                relative = (trace.gold_step + 0.5) / trace.num_steps
                vector[4 + min(int(relative * 3), 2)] += 1.0
            vector[7] += trace.num_response_tokens
        return vector

    @staticmethod
    def _assignment_cost(
        observed: np.ndarray,
        targets: np.ndarray,
        scales: np.ndarray,
        vector: np.ndarray,
        partition: int,
    ) -> tuple[float, float, int]:
        proposed = observed.copy()
        proposed[partition] += vector
        fit = float(np.sum(((proposed - targets) / scales) ** 2))
        overflow = float(np.sum((np.maximum(proposed - targets, 0.0) / scales) ** 2))
        fullness = float(proposed[partition, 0] / max(targets[partition, 0], 1.0))
        return fit + 2.0 * overflow, fullness, partition

    @staticmethod
    def _assert_group_disjoint(
        traces: Sequence[TraceMeta], partitions: Sequence[Sequence[int]]
    ) -> None:
        groups = [{traces[index].group_id for index in part} for part in partitions]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("problem/group leakage detected across partitions")

    @staticmethod
    def _manifest(
        name: str,
        indices: Sequence[int],
        traces: Sequence[TraceMeta],
        balance: np.ndarray,
    ) -> PartitionManifest:
        return PartitionManifest(
            name=name,
            indices=tuple(indices),
            trace_ids=tuple(traces[index].trace_id for index in indices),
            group_ids=tuple(sorted({traces[index].group_id for index in indices})),
            balance=BalanceStats.from_vector(balance),
        )
