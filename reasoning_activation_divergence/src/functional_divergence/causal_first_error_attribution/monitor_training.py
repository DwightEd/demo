from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .monitor_data import MonitorBoundaryRow, ProcessBenchMonitorData
from .monitor_models import (
    ContextMonitor,
    DepthGraphMonitor,
    LayerSetMonitor,
    fixed_layer_permutation,
)


MONITOR_ARMS = (
    "nuisance",
    "output_history",
    "layer_set",
    "depth_graph_shuffled",
    "depth_graph",
)


@dataclass(frozen=True)
class MonitorTrainingConfig:
    width: int = 64
    message_passing_steps: int = 2
    dropout: float = 0.1
    epochs: int = 20
    patience: int = 4
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    show_progress: bool = True

    def __post_init__(self) -> None:
        if min(
            self.width,
            self.message_passing_steps,
            self.epochs,
            self.patience,
            self.batch_size,
        ) < 1:
            raise ValueError("training dimensions and counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer parameters are invalid")


@dataclass(frozen=True)
class StateNormalizer:
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    scale: np.ndarray


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}::{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def build_inner_group_split(
    rows: Sequence[MonitorBoundaryRow],
    outer_train_indices: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a deterministic domain-stratified validation split by problem group."""

    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie in (0,0.5)")
    indices = np.asarray(outer_train_indices, dtype=np.int64).reshape(-1)
    if indices.size < 2:
        raise ValueError("outer training split is too small")
    domain_groups: dict[str, dict[str, list[int]]] = {}
    for index in indices:
        row = rows[int(index)]
        domain_groups.setdefault(row.domain, {}).setdefault(
            row.sibling_group, []
        ).append(int(index))

    validation_groups: set[str] = set()
    for domain, groups in sorted(domain_groups.items()):
        names = sorted(
            groups,
            key=lambda name: (_stable_fraction(f"{domain}::{name}", seed), name),
        )
        if len(names) < 2:
            continue
        validation_count = max(1, int(math.ceil(validation_fraction * len(names))))
        validation_count = min(validation_count, len(names) - 1)
        validation_groups.update(names[:validation_count])
    validation = np.asarray(
        [i for i in indices if rows[int(i)].sibling_group in validation_groups],
        dtype=np.int64,
    )
    train = np.asarray(
        [i for i in indices if rows[int(i)].sibling_group not in validation_groups],
        dtype=np.int64,
    )
    if train.size == 0 or validation.size == 0:
        raise ValueError("could not construct non-empty group-disjoint train/validation sets")
    return train, validation


def fit_state_normalizer(
    data: ProcessBenchMonitorData, indices: np.ndarray
) -> StateNormalizer:
    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        raise ValueError("state normalization requires training rows")
    shape = (len(data.layer_ids), data.hidden_size)
    total = np.zeros(shape, dtype=np.float64)
    square_total = np.zeros(shape, dtype=np.float64)
    for index in tqdm(selected, desc="fit state normalization", leave=False):
        state = data.state(int(index)).astype(np.float64, copy=False)
        total += state
        square_total += state * state
    mean = total / selected.size
    variance = np.maximum(square_total / selected.size - mean * mean, 0.0)
    scale = np.maximum(np.sqrt(variance), 1e-4)
    return StateNormalizer(
        mean=mean.astype(np.float32), scale=scale.astype(np.float32)
    )


def _context_matrix(data: ProcessBenchMonitorData, arm: str) -> np.ndarray:
    if arm == "nuisance":
        return data.nuisance
    if arm in MONITOR_ARMS:
        return np.concatenate([data.nuisance, data.output_context], axis=1)
    raise ValueError(f"unknown monitor arm {arm!r}; available={list(MONITOR_ARMS)}")


def fit_feature_normalizer(
    values: np.ndarray, indices: np.ndarray
) -> FeatureNormalizer:
    train = np.asarray(values[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
    if train.ndim != 2 or train.shape[0] == 0:
        raise ValueError("feature normalization requires a non-empty matrix")
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1e-4)
    return FeatureNormalizer(mean=mean.astype(np.float32), scale=scale.astype(np.float32))


def _group_balanced_weights(
    rows: Sequence[MonitorBoundaryRow], indices: np.ndarray
) -> np.ndarray:
    selected = [int(value) for value in np.asarray(indices, dtype=np.int64)]
    domains = sorted({rows[index].domain for index in selected})
    result = np.zeros(len(selected), dtype=np.float32)
    position = {index: pos for pos, index in enumerate(selected)}
    for domain in domains:
        domain_indices = [index for index in selected if rows[index].domain == domain]
        groups: dict[str, list[int]] = {}
        for index in domain_indices:
            groups.setdefault(rows[index].sibling_group, []).append(index)
        for members in groups.values():
            weight = 1.0 / (len(domains) * len(groups) * len(members))
            for index in members:
                result[position[index]] = weight
    return result * (len(result) / result.sum())


class _BoundaryDataset(Dataset):
    def __init__(
        self,
        data: ProcessBenchMonitorData,
        indices: np.ndarray,
        *,
        context: np.ndarray,
        feature_normalizer: FeatureNormalizer,
        state_normalizer: StateNormalizer | None,
        permutation: np.ndarray | None,
        weights: np.ndarray,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.context = np.asarray(context, dtype=np.float32)
        self.feature_normalizer = feature_normalizer
        self.state_normalizer = state_normalizer
        self.permutation = permutation
        self.weights = np.asarray(weights, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        row_index = int(self.indices[position])
        context = (
            self.context[row_index] - self.feature_normalizer.mean
        ) / self.feature_normalizer.scale
        if self.state_normalizer is None:
            state = np.zeros((1, 1), dtype=np.float32)
        else:
            state = (
                self.data.state(row_index) - self.state_normalizer.mean
            ) / self.state_normalizer.scale
            if self.permutation is not None:
                state = state[self.permutation]
        return (
            torch.from_numpy(np.asarray(state, dtype=np.float32)),
            torch.from_numpy(np.asarray(context, dtype=np.float32)),
            torch.tensor(self.data.rows[row_index].label, dtype=torch.float32),
            torch.tensor(self.weights[position], dtype=torch.float32),
        )


def _build_model(
    data: ProcessBenchMonitorData,
    arm: str,
    context_size: int,
    config: MonitorTrainingConfig,
) -> nn.Module:
    if arm in ("nuisance", "output_history"):
        return ContextMonitor(context_size=context_size, width=config.width)
    if arm == "layer_set":
        return LayerSetMonitor(
            hidden_size=data.hidden_size,
            context_size=context_size,
            width=config.width,
        )
    if arm in ("depth_graph", "depth_graph_shuffled"):
        return DepthGraphMonitor(
            hidden_size=data.hidden_size,
            layer_count=len(data.layer_ids),
            context_size=context_size,
            width=config.width,
            message_passing_steps=config.message_passing_steps,
            dropout=config.dropout,
        )
    raise ValueError(f"unknown monitor arm {arm!r}; available={list(MONITOR_ARMS)}")


@dataclass
class TrainedMonitor:
    arm: str
    model: nn.Module
    state_normalizer: StateNormalizer | None
    feature_normalizer: FeatureNormalizer
    context: np.ndarray
    permutation: np.ndarray | None
    validation_nll: float
    epochs_trained: int

    def predict(
        self,
        data: ProcessBenchMonitorData,
        indices: np.ndarray,
        *,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        dataset = _BoundaryDataset(
            data,
            selected,
            context=self.context,
            feature_normalizer=self.feature_normalizer,
            state_normalizer=self.state_normalizer,
            permutation=self.permutation,
            weights=np.ones(len(selected), dtype=np.float32),
        )
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
        target_device = torch.device(device)
        self.model.to(target_device)
        self.model.eval()
        scores = []
        with torch.inference_mode():
            for states, context, _, _ in loader:
                logits = self.model(states.to(target_device), context.to(target_device))
                scores.append(torch.sigmoid(logits).cpu().numpy())
        self.model.to(torch.device("cpu"))
        return (
            np.concatenate(scores).astype(np.float64)
            if scores
            else np.zeros(0, dtype=np.float64)
        )


def train_monitor_arm(
    data: ProcessBenchMonitorData,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    arm: str,
    config: MonitorTrainingConfig,
    seed: int,
    state_normalizer: StateNormalizer | None = None,
) -> TrainedMonitor:
    if arm not in MONITOR_ARMS:
        raise ValueError(f"unknown monitor arm {arm!r}; available={list(MONITOR_ARMS)}")
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    if train_indices.size == 0 or validation_indices.size == 0:
        raise ValueError("training and validation rows must both be non-empty")
    uses_state = arm not in ("nuisance", "output_history")
    if uses_state and state_normalizer is None:
        state_normalizer = fit_state_normalizer(data, train_indices)
    if not uses_state:
        state_normalizer = None

    context = _context_matrix(data, arm)
    feature_normalizer = fit_feature_normalizer(context, train_indices)
    permutation = (
        fixed_layer_permutation(len(data.layer_ids), seed)
        if arm == "depth_graph_shuffled"
        else None
    )
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = _build_model(data, arm, context.shape[1], config)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    train_data = _BoundaryDataset(
        data,
        train_indices,
        context=context,
        feature_normalizer=feature_normalizer,
        state_normalizer=state_normalizer,
        permutation=permutation,
        weights=_group_balanced_weights(data.rows, train_indices),
    )
    validation_data = _BoundaryDataset(
        data,
        validation_indices,
        context=context,
        feature_normalizer=feature_normalizer,
        state_normalizer=state_normalizer,
        permutation=permutation,
        weights=_group_balanced_weights(data.rows, validation_indices),
    )
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=config.batch_size, shuffle=False
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    epochs_trained = 0
    progress = tqdm(
        range(config.epochs),
        desc=f"train {arm}",
        unit="epoch",
        disable=not config.show_progress,
    )
    for epoch in progress:
        model.train()
        for states, features, labels, weights in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(states.to(device), features.to(device))
            losses = loss_function(logits, labels.to(device))
            loss = torch.sum(losses * weights.to(device)) / torch.sum(
                weights.to(device)
            )
            loss.backward()
            optimizer.step()

        model.eval()
        validation_total = 0.0
        validation_weight = 0.0
        with torch.inference_mode():
            for states, features, labels, weights in validation_loader:
                logits = model(states.to(device), features.to(device))
                losses = loss_function(logits, labels.to(device))
                validation_total += float(
                    torch.sum(losses * weights.to(device)).cpu()
                )
                validation_weight += float(torch.sum(weights).cpu())
        validation_loss = validation_total / validation_weight
        epochs_trained = epoch + 1
        progress.set_postfix(val_nll=f"{validation_loss:.4f}")
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    model.to(torch.device("cpu"))
    return TrainedMonitor(
        arm=arm,
        model=model,
        state_normalizer=state_normalizer,
        feature_normalizer=feature_normalizer,
        context=context,
        permutation=permutation,
        validation_nll=float(best_loss),
        epochs_trained=epochs_trained,
    )


__all__ = [
    "MONITOR_ARMS",
    "FeatureNormalizer",
    "MonitorTrainingConfig",
    "StateNormalizer",
    "TrainedMonitor",
    "build_inner_group_split",
    "fit_feature_normalizer",
    "fit_state_normalizer",
    "train_monitor_arm",
]
