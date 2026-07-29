from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .contracts import (
    CounterfactualExample,
    _validated_finite_real,
    _validated_int,
)
from .model import MultiViewHypergraphDetector
from .objectives import SemiSupervisedObjective


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    max_grad_norm: float = 2.0
    supervised_weight: float = 1.0
    consistency_weight: float = 0.25
    contrastive_weight: float = 0.25
    contrastive_margin: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "epochs",
            _validated_int(self.epochs, name="epochs", minimum=1),
        )
        object.__setattr__(
            self,
            "seed",
            _validated_int(self.seed, name="seed", minimum=0),
        )
        for name in ("learning_rate", "max_grad_norm"):
            object.__setattr__(
                self,
                name,
                _validated_finite_real(
                    getattr(self, name),
                    name=name,
                    strictly_positive=True,
                ),
            )
        for name in (
            "weight_decay",
            "supervised_weight",
            "consistency_weight",
            "contrastive_weight",
            "contrastive_margin",
        ):
            object.__setattr__(
                self,
                name,
                _validated_finite_real(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                ),
            )


@dataclass(frozen=True)
class EpochStats:
    epoch: int
    total_loss: float
    supervised_loss: float
    consistency_loss: float
    contrastive_loss: float
    labeled_tokens: int
    active_examples: int
    skipped_examples: int


def _aligned_labels(example: CounterfactualExample) -> np.ndarray:
    labels = np.full(
        len(example.factual.response_nodes),
        -1,
        dtype=np.int64,
    )
    views = (example.factual, example.counterfactual, example.paraphrase)
    for view in views:
        if view is None or view.token_labels is None:
            continue
        visible = view.token_labels >= 0
        labels[visible] = view.token_labels[visible]
    return labels


def _has_active_objective(
    example: CounterfactualExample,
    config: TrainingConfig,
) -> bool:
    supervised = (
        config.supervised_weight > 0.0
        and np.any(_aligned_labels(example) >= 0)
    )
    consistency = (
        config.consistency_weight > 0.0
        and example.paraphrase is not None
    )
    contrastive = (
        config.contrastive_weight > 0.0
        and bool(np.any(example.contrastive_eligible))
    )
    return bool(supervised or consistency or contrastive)


def train_semisupervised(
    model: MultiViewHypergraphDetector,
    examples: Sequence[CounterfactualExample],
    *,
    config: TrainingConfig,
    device: torch.device | None = None,
) -> list[EpochStats]:
    if not examples:
        raise ValueError("examples cannot be empty")
    active_examples = tuple(
        example
        for example in examples
        if _has_active_objective(example, config)
    )
    if not active_examples:
        raise ValueError(
            "training data contain no active training objective"
        )
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    objective = SemiSupervisedObjective(
        supervised_weight=config.supervised_weight,
        consistency_weight=config.consistency_weight,
        contrastive_weight=config.contrastive_weight,
        contrastive_margin=config.contrastive_margin,
    )
    history: list[EpochStats] = []
    for epoch in range(config.epochs):
        model.train()
        totals = np.zeros(4, dtype=np.float64)
        labeled_tokens = 0
        for index in rng.permutation(len(active_examples)):
            example = active_examples[int(index)]
            labels_array = _aligned_labels(example)
            labels = torch.as_tensor(labels_array, dtype=torch.long, device=device)
            outputs = model(example)
            reference_mask = torch.as_tensor(
                example.contrastive_eligible,
                dtype=torch.bool,
                device=device,
            )
            losses = objective(
                outputs,
                labels=labels,
                context_reference_mask=reference_mask,
            )
            optimizer.zero_grad()
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            optimizer.step()
            totals += np.asarray(
                [
                    losses.total.item(),
                    losses.supervised.item(),
                    losses.consistency.item(),
                    losses.contrastive.item(),
                ]
            )
            labeled_tokens += int(torch.sum(labels >= 0).item())
        totals /= len(active_examples)
        history.append(
            EpochStats(
                epoch=epoch + 1,
                total_loss=float(totals[0]),
                supervised_loss=float(totals[1]),
                consistency_loss=float(totals[2]),
                contrastive_loss=float(totals[3]),
                labeled_tokens=labeled_tokens,
                active_examples=len(active_examples),
                skipped_examples=len(examples) - len(active_examples),
            )
        )
    return history
