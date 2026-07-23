from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..progress import ProgressReporter
from .tasks import TaskExample


@dataclass(frozen=True)
class FoldInput:
    task_name: str
    train_examples: tuple[TaskExample, ...]
    train_labels: np.ndarray
    train_groups: np.ndarray
    test_examples: tuple[TaskExample, ...]
    seed: int
    progress: ProgressReporter | None = None

    def __post_init__(self) -> None:
        count = len(self.train_examples)
        if self.train_labels.shape != (count,) or self.train_groups.shape != (count,):
            raise ValueError("training labels/groups must align with examples")
        if len(np.unique(self.train_labels)) != 2:
            raise ValueError("each training fold needs both classes")


@dataclass(frozen=True)
class MethodFoldResult:
    probabilities: dict[str, np.ndarray]
    diagnostics: dict[str, Any]
    factors: dict[str, np.ndarray]


class DiscriminativeMethod(Protocol):
    def fit_predict(self, fold: FoldInput) -> MethodFoldResult: ...
