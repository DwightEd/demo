from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ...progress import NullProgress
from ..method import FoldInput, MethodFoldResult
from ..preprocessing import FiniteStandardizer, domain_group_balanced_weights
from ..registry import ContrastSpec, register_method
from ..representation import ChainBalancedPCA
from ..tasks import (
    TaskExample,
    load_visible_step_token_states,
    load_visible_states,
    nuisance_features,
    visible_output_steps,
)
from .token_attention import TokenAttentionPool, history_attention_mass

STATE_ARMS = (
    "initial_state",
    "current_state",
    "ordered_history",
    "shuffled_history",
)


@dataclass(frozen=True)
class PredictiveStateConfig:
    pca_dim: int = 8
    positions_per_chain: int = 16
    sequence_unit: Literal["step_end", "token"] = "step_end"
    sequence_encoder: Literal["gru_final", "attention_pool"] = "gru_final"
    attention_heads: int = 4
    attention_queries: int = 4
    width: int = 32
    epochs: int = 20
    patience: int = 4
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    device: str = "cuda"
    show_progress: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            "pca_dim",
            "positions_per_chain",
            "width",
            "epochs",
            "patience",
            "batch_size",
            "attention_heads",
            "attention_queries",
        )
        if any(
            isinstance(getattr(self, name), (bool, np.bool_))
            or int(getattr(self, name)) < 1
            for name in integer_fields
        ):
            raise ValueError("dimensions and training counts must be positive integers")
        if self.positions_per_chain < self.pca_dim:
            raise ValueError("positions_per_chain must be at least pca_dim")
        if self.sequence_unit not in {"step_end", "token"}:
            raise ValueError("sequence_unit must be 'step_end' or 'token'")
        if self.sequence_encoder not in {"gru_final", "attention_pool"}:
            raise ValueError("sequence_encoder must be 'gru_final' or 'attention_pool'")
        if (
            self.sequence_encoder == "attention_pool"
            and self.width % self.attention_heads != 0
        ):
            raise ValueError("attention width must be divisible by attention_heads")
        if self.patience > self.epochs:
            raise ValueError("patience cannot exceed epochs")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer parameters are invalid")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must lie in (0, 0.5)")
        if not str(self.device).strip():
            raise ValueError("device cannot be empty")


def _chain_key(example: TaskExample) -> tuple[str, int]:
    return example.sample.dataset, int(example.sample.chain_id)


def _stable_permutation(example: TaskExample, length: int, seed: int) -> np.ndarray:
    if length < 1:
        return np.empty(0, dtype=np.int64)
    identity = np.arange(length, dtype=np.int64)
    if length == 1:
        return identity
    sample = example.sample
    payload = (
        f"{seed}|{sample.dataset}|{sample.chain_id}|{sample.manifest_row}|"
        f"{example.visible_steps}|{length}"
    ).encode()
    token = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    order = np.random.default_rng(token).permutation(length)
    if np.array_equal(order, identity):
        order = np.roll(order, 1 + token % (length - 1))
    return np.asarray(order, dtype=np.int64)


def _arm_sequence(
    sequence: np.ndarray | Sequence[np.ndarray],
    example: TaskExample,
    arm: str,
    *,
    seed: int,
    add_step_markers: bool = False,
) -> np.ndarray:
    """Select causal step blocks without pooling any tokens inside a block."""

    if isinstance(sequence, np.ndarray):
        values = np.asarray(sequence, dtype=np.float32)
        if values.ndim != 2 or len(values) != example.visible_steps:
            raise ValueError("projected sequence must align with the visible prefix")
        steps = tuple(values[index : index + 1] for index in range(len(values)))
    else:
        steps = tuple(np.asarray(step, dtype=np.float32) for step in sequence)
        if len(steps) != example.visible_steps:
            raise ValueError("projected step blocks must align with the visible prefix")
        if any(step.ndim != 2 or len(step) < 1 for step in steps):
            raise ValueError("each projected step block must be a non-empty matrix")
        if len({step.shape[1] for step in steps}) != 1:
            raise ValueError("projected step blocks must share one feature dimension")
    if arm == "initial_state":
        selected = steps[:1]
    elif arm == "current_state":
        selected = steps[-1:]
    elif arm == "ordered_history":
        selected = steps
    elif arm == "shuffled_history":
        if len(steps) <= 2:
            selected = steps
        else:
            order = _stable_permutation(example, len(steps) - 1, seed)
            selected = tuple(steps[int(index)] for index in order) + steps[-1:]
    else:
        raise ValueError(f"unknown state arm {arm!r}; available={list(STATE_ARMS)}")
    if not add_step_markers:
        return np.ascontiguousarray(np.concatenate(selected, axis=0), dtype=np.float32)
    marked = []
    for step in selected:
        markers = np.zeros((len(step), 2), dtype=np.float32)
        markers[0, 0] = 1.0
        markers[-1, 1] = 1.0
        marked.append(np.concatenate([step, markers], axis=1))
    return np.ascontiguousarray(np.concatenate(marked, axis=0), dtype=np.float32)


def _latest_examples(
    examples: Sequence[TaskExample],
) -> dict[tuple[str, int], TaskExample]:
    latest: dict[tuple[str, int], TaskExample] = {}
    for example in examples:
        key = _chain_key(example)
        if key not in latest or example.visible_steps > latest[key].visible_steps:
            latest[key] = example
    return latest


def _project_chains(
    examples: Sequence[TaskExample],
    projector: ChainBalancedPCA,
    *,
    reporter: Any,
    description: str,
    sequence_unit: str,
) -> dict[tuple[str, int], tuple[np.ndarray, ...]]:
    latest = _latest_examples(examples)
    cache: dict[tuple[str, int], tuple[np.ndarray, ...]] = {}
    tracked = reporter.track(
        latest.items(), total=len(latest), description=description
    )
    for key, example in tracked:
        if sequence_unit == "token":
            raw_steps = load_visible_step_token_states(example)
            lengths = np.asarray([len(step) for step in raw_steps], dtype=np.int64)
            projected = projector.transform(np.concatenate(raw_steps, axis=0))
            flattened = projected.reshape(projected.shape[0], -1)
            offsets = np.concatenate([[0], np.cumsum(lengths)])
            cache[key] = tuple(
                np.ascontiguousarray(
                    flattened[offsets[index] : offsets[index + 1]],
                    dtype=np.float32,
                )
                for index in range(len(raw_steps))
            )
        else:
            projected = projector.transform(load_visible_states(example))
            flattened = projected.reshape(projected.shape[0], -1)
            cache[key] = tuple(
                np.ascontiguousarray(flattened[index : index + 1], dtype=np.float32)
                for index in range(len(flattened))
            )
    return cache


def _visible_token_tensor(example: TaskExample) -> np.ndarray:
    return np.concatenate(load_visible_step_token_states(example), axis=0)


def _context_features(example: TaskExample) -> np.ndarray:
    _, nuisance = nuisance_features(example)
    output = np.asarray(visible_output_steps(example), dtype=np.float64)
    current = output[-1]
    if len(output) == 1:
        earlier = np.zeros(output.shape[1], dtype=np.float64)
        has_earlier = 0.0
    else:
        earlier = output[:-1].mean(axis=0)
        has_earlier = 1.0
    return np.concatenate([nuisance, current, earlier, [has_earlier]])


@dataclass(frozen=True)
class _PreparedRows:
    examples: tuple[TaskExample, ...]
    sequences: tuple[tuple[np.ndarray, ...], ...]
    context: np.ndarray


def _prepare_rows(
    examples: Sequence[TaskExample],
    cache: Mapping[tuple[str, int], tuple[np.ndarray, ...]],
) -> _PreparedRows:
    rows = tuple(examples)
    sequences = tuple(
        cache[_chain_key(example)][: example.visible_steps]
        for example in rows
    )
    if any(len(sequence) != example.visible_steps for sequence, example in zip(sequences, rows)):
        raise RuntimeError("a projected chain is shorter than an eligible prefix")
    context = np.stack([_context_features(example) for example in rows], axis=0)
    if not np.isfinite(context).all():
        raise ValueError("online context contains non-finite values")
    return _PreparedRows(rows, sequences, context)


def _state_sequences(
    rows: _PreparedRows,
    arm: str,
    *,
    config: PredictiveStateConfig,
    seed: int,
) -> tuple[np.ndarray, ...]:
    return tuple(
        _arm_sequence(
            sequence,
            example,
            arm,
            seed=seed,
            add_step_markers=config.sequence_encoder == "attention_pool",
        )
        for sequence, example in zip(rows.sequences, rows.examples)
    )


def _inner_group_split(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int8)
    group = np.asarray(groups)
    splitter = GroupShuffleSplit(
        n_splits=64,
        test_size=validation_fraction,
        random_state=seed,
    )
    for train, validation in splitter.split(np.zeros(len(y)), y, group):
        if len(np.unique(y[train])) == 2 and len(np.unique(y[validation])) == 2:
            return np.asarray(train, dtype=np.int64), np.asarray(
                validation, dtype=np.int64
            )
    raise ValueError("could not construct a two-class problem-group validation split")


class _SequenceDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[np.ndarray],
        context: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        self.sequences = tuple(np.asarray(value, dtype=np.float32) for value in sequences)
        self.context = np.asarray(context, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)
        count = len(self.sequences)
        if (
            self.context.shape[0] != count
            or self.labels.shape != (count,)
            or self.weights.shape != (count,)
        ):
            raise ValueError("sequence dataset fields must align")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.sequences[index]),
            torch.from_numpy(self.context[index]),
            torch.tensor(self.labels[index], dtype=torch.float32),
            torch.tensor(self.weights[index], dtype=torch.float32),
        )


def _collate(batch):
    sequences, context, labels, weights = zip(*batch)
    lengths = torch.tensor([len(value) for value in sequences], dtype=torch.int64)
    return (
        pad_sequence(sequences, batch_first=True),
        lengths,
        torch.stack(context),
        torch.stack(labels),
        torch.stack(weights),
    )


class _ContextMonitor(nn.Module):
    def __init__(self, context_dim: int, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_dim, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(
        self, states: torch.Tensor, lengths: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        del states, lengths
        return self.network(context).squeeze(-1)


class _GRUPrefixMonitor(nn.Module):
    def __init__(self, input_dim: int, context_dim: int, width: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, width, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(width + context_dim, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(
        self, states: torch.Tensor, lengths: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        packed = pack_padded_sequence(
            states,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.head(torch.cat([hidden[-1], context], dim=1)).squeeze(-1)


def _parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _device(config: PredictiveStateConfig) -> torch.device:
    target = torch.device(config.device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("predictive_state_monitor requested CUDA but CUDA is unavailable")
    return target


def _model(
    arm: str,
    *,
    input_dim: int,
    context_dim: int,
    config: PredictiveStateConfig,
) -> nn.Module:
    if arm == "output_only":
        return _ContextMonitor(context_dim, config.width)
    if arm in STATE_ARMS:
        if config.sequence_encoder == "attention_pool":
            return TokenAttentionPool(
                input_dim=input_dim,
                context_dim=context_dim,
                width=config.width,
                heads=config.attention_heads,
                queries=config.attention_queries,
            )
        return _GRUPrefixMonitor(input_dim, context_dim, config.width)
    raise ValueError(f"unknown arm {arm!r}")


def _loader(
    sequences: Sequence[np.ndarray],
    context: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    selected = np.asarray(indices, dtype=np.int64)
    weights = domain_group_balanced_weights(
        np.asarray(domains)[selected], np.asarray(groups)[selected]
    )
    dataset = _SequenceDataset(
        [sequences[int(index)] for index in selected],
        np.asarray(context)[selected],
        np.asarray(labels)[selected],
        weights,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=_collate,
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    for states, lengths, context, labels, weights in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(states.to(device), lengths, context.to(device))
        losses = loss_function(logits, labels.to(device))
        weight = weights.to(device)
        loss = torch.sum(losses * weight) / torch.sum(weight)
        loss.backward()
        optimizer.step()


def _nll(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    total = 0.0
    total_weight = 0.0
    model.eval()
    with torch.inference_mode():
        for states, lengths, context, labels, weights in loader:
            logits = model(states.to(device), lengths, context.to(device))
            losses = loss_function(logits, labels.to(device))
            total += float(torch.sum(losses * weights.to(device)).cpu())
            total_weight += float(torch.sum(weights).cpu())
    return total / total_weight


def _predict(
    model: nn.Module,
    sequences: Sequence[np.ndarray],
    context: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    count = len(sequences)
    loader = _loader(
        sequences,
        context,
        np.zeros(count, dtype=np.int8),
        np.full(count, "prediction", dtype=object),
        np.arange(count),
        np.arange(count),
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    scores = []
    model.eval()
    with torch.inference_mode():
        for states, lengths, features, _, _ in loader:
            logits = model(states.to(device), lengths, features.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


@dataclass(frozen=True)
class _FittedArm:
    probability: np.ndarray
    model: nn.Module
    scaler: FiniteStandardizer
    selected_epoch: int
    validation_nll: float


def _fit_arm(
    arm: str,
    train_rows: _PreparedRows,
    test_rows: _PreparedRows,
    labels: np.ndarray,
    groups: np.ndarray,
    inner_train: np.ndarray,
    validation: np.ndarray,
    *,
    config: PredictiveStateConfig,
    seed: int,
) -> _FittedArm:
    domains = np.asarray(
        [example.sample.dataset for example in train_rows.examples], dtype=object
    )
    state_arm = arm if arm in STATE_ARMS else "current_state"
    train_sequences = _state_sequences(
        train_rows,
        state_arm,
        config=config,
        seed=seed,
    )
    test_sequences = _state_sequences(
        test_rows,
        state_arm,
        config=config,
        seed=seed,
    )
    inner_weights = domain_group_balanced_weights(
        domains[inner_train], np.asarray(groups)[inner_train]
    )
    selection_scaler = FiniteStandardizer().fit(
        train_rows.context[inner_train], inner_weights
    )
    selection_context = selection_scaler.transform(train_rows.context)
    input_dim = int(train_sequences[0].shape[1])
    context_dim = int(selection_context.shape[1])
    target = _device(config)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    selection_model = _model(
        arm, input_dim=input_dim, context_dim=context_dim, config=config
    ).to(target)
    optimizer = torch.optim.AdamW(
        selection_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _loader(
        train_sequences,
        selection_context,
        labels,
        domains,
        groups,
        inner_train,
        batch_size=config.batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _loader(
        train_sequences,
        selection_context,
        labels,
        domains,
        groups,
        validation,
        batch_size=config.batch_size,
        shuffle=False,
        seed=seed,
    )
    best_nll = float("inf")
    best_epoch = 1
    stale = 0
    progress = tqdm(
        range(1, config.epochs + 1),
        desc=f"select {arm}",
        unit="epoch",
        disable=not config.show_progress,
        leave=False,
    )
    for epoch in progress:
        _train_epoch(selection_model, train_loader, optimizer, target)
        value = _nll(selection_model, validation_loader, target)
        progress.set_postfix(val_nll=f"{value:.4f}")
        if value < best_nll - 1e-6:
            best_nll = value
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    outer_weights = domain_group_balanced_weights(domains, np.asarray(groups))
    final_scaler = FiniteStandardizer().fit(train_rows.context, outer_weights)
    final_train_context = final_scaler.transform(train_rows.context)
    final_test_context = final_scaler.transform(test_rows.context)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    final_model = _model(
        arm,
        input_dim=input_dim,
        context_dim=final_train_context.shape[1],
        config=config,
    ).to(target)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    full_loader = _loader(
        train_sequences,
        final_train_context,
        labels,
        domains,
        groups,
        np.arange(len(labels)),
        batch_size=config.batch_size,
        shuffle=True,
        seed=seed,
    )
    for _ in range(best_epoch):
        _train_epoch(final_model, full_loader, final_optimizer, target)
    probability = _predict(
        final_model,
        test_sequences,
        final_test_context,
        batch_size=config.batch_size,
        device=target,
    )
    final_model.to(torch.device("cpu"))
    return _FittedArm(
        probability=probability,
        model=final_model,
        scaler=final_scaler,
        selected_epoch=best_epoch,
        validation_nll=best_nll,
    )


@register_method(
    "predictive_state_monitor",
    contrasts=(
        ContrastSpec(
            "initial_state_given_output_nll",
            "output_only",
            "initial_state",
            "early hidden-state increment beyond online output controls",
        ),
        ContrastSpec(
            "current_state_given_initial_nll",
            "initial_state",
            "current_state",
            "current boundary increment beyond a chain-static early-state control",
        ),
        ContrastSpec(
            "ordered_history_given_current_nll",
            "current_state",
            "ordered_history",
            "causal raw-history increment beyond the current hidden state",
        ),
        ContrastSpec(
            "history_order_nll",
            "shuffled_history",
            "ordered_history",
            "ordered history versus a same-capacity visible-past permutation",
        ),
        ContrastSpec(
            "same_model_history_access_nll",
            "ordered_model_current_ablation",
            "ordered_history",
            "ordered model with history retained versus the same model with history removed",
        ),
        ContrastSpec(
            "same_model_history_order_nll",
            "ordered_model_shuffled_ablation",
            "ordered_history",
            "ordered model on intact history versus the same model on permuted past steps",
        ),
    ),
    arm_definitions={
        "output_only": "online prefix counts and completed-step entropy/NLL summaries",
        "initial_state": "same configured encoder/head using only the first visible state block",
        "current_state": "same configured encoder/head using only the current state block",
        "ordered_history": "configured encoder over every visible completed-step state block",
        "shuffled_history": (
            "same configured encoder/head and current block, with only the past permuted"
        ),
        "ordered_model_current_ablation": (
            "trained ordered model evaluated after deleting all pre-current hidden tokens"
        ),
        "ordered_model_shuffled_ablation": (
            "trained ordered model evaluated after permuting complete past-step token blocks"
        ),
    },
    default_config=PredictiveStateConfig,
)
class PredictiveStateMonitor:
    """Capacity-matched test of hidden-state history for future first error."""

    def __init__(self, config: PredictiveStateConfig | Mapping[str, Any]) -> None:
        if isinstance(config, Mapping):
            config = PredictiveStateConfig(**dict(config))
        if not isinstance(config, PredictiveStateConfig):
            raise TypeError(
                "predictive_state_monitor config must be a config object or mapping"
            )
        self.config = config

    def fit_predict(self, fold: FoldInput) -> MethodFoldResult:
        if fold.task_name != "strict_prefix":
            raise ValueError("predictive_state_monitor requires the strict_prefix task")
        reporter = fold.progress or NullProgress()
        reporter.stage(
            "projection",
            f"{fold.task_name}: outer-train {self.config.sequence_unit} hidden PCA",
        )
        projector = ChainBalancedPCA(
            dim=self.config.pca_dim,
            positions_per_chain=self.config.positions_per_chain,
            seed=fold.seed,
        ).fit(
            fold.train_examples,
            progress=reporter,
            state_loader=(
                _visible_token_tensor
                if self.config.sequence_unit == "token"
                else load_visible_states
            ),
        )
        reporter.stage(
            "encode", f"{fold.task_name}: causal {self.config.sequence_unit} sequences"
        )
        train_cache = _project_chains(
            fold.train_examples,
            projector,
            reporter=reporter,
            description="train projected chains",
            sequence_unit=self.config.sequence_unit,
        )
        test_cache = _project_chains(
            fold.test_examples,
            projector,
            reporter=reporter,
            description="test projected chains",
            sequence_unit=self.config.sequence_unit,
        )
        train_rows = _prepare_rows(fold.train_examples, train_cache)
        test_rows = _prepare_rows(fold.test_examples, test_cache)
        inner_train, validation = _inner_group_split(
            fold.train_labels,
            fold.train_groups,
            validation_fraction=self.config.validation_fraction,
            seed=fold.seed,
        )

        reporter.stage("fit", f"{fold.task_name}: capacity-matched state arms")
        arms = {}
        for arm in ("output_only", *STATE_ARMS):
            arms[arm] = _fit_arm(
                arm,
                train_rows,
                test_rows,
                fold.train_labels,
                fold.train_groups,
                inner_train,
                validation,
                config=self.config,
                seed=fold.seed,
            )
        state_counts = {
            arm: _parameter_count(arms[arm].model) for arm in STATE_ARMS
        }
        if len(set(state_counts.values())) != 1:
            raise RuntimeError("state arms do not have identical parameter counts")
        ordered_model = arms["ordered_history"].model
        target = _device(self.config)
        ordered_model.to(target)
        intervention_context = arms["ordered_history"].scaler.transform(test_rows.context)
        current_view = _state_sequences(
            test_rows,
            "current_state",
            config=self.config,
            seed=fold.seed,
        )
        shuffled_view = _state_sequences(
            test_rows,
            "shuffled_history",
            config=self.config,
            seed=fold.seed,
        )
        intervention_probabilities = {
            "ordered_model_current_ablation": _predict(
                ordered_model,
                current_view,
                intervention_context,
                batch_size=self.config.batch_size,
                device=target,
            ),
            "ordered_model_shuffled_ablation": _predict(
                ordered_model,
                shuffled_view,
                intervention_context,
                batch_size=self.config.batch_size,
                device=target,
            ),
        }
        attention_mass = None
        attention_excess = None
        if isinstance(ordered_model, TokenAttentionPool):
            ordered_view = _state_sequences(
                test_rows,
                "ordered_history",
                config=self.config,
                seed=fold.seed,
            )
            current_lengths = np.asarray(
                [len(sequence[-1]) for sequence in test_rows.sequences],
                dtype=np.int64,
            )
            attention_mass = history_attention_mass(
                ordered_model,
                ordered_view,
                current_lengths,
                intervention_context,
                batch_size=self.config.batch_size,
                device=target,
            )
            ordered_lengths = np.asarray(
                [sum(len(step) for step in sequence) for sequence in test_rows.sequences],
                dtype=np.float64,
            )
            uniform_history_fraction = 1.0 - current_lengths / ordered_lengths
            attention_excess = attention_mass - uniform_history_fraction
        ordered_model.to(torch.device("cpu"))
        if projector.model is None:
            raise RuntimeError("projector unexpectedly missing after fit")
        factors: dict[str, np.ndarray] = {
            "pca_components": np.asarray(projector.model.components_),
            "pca_mean": projector.mean_,
        }
        for arm, result in arms.items():
            factors[f"{arm}.context_center"] = np.asarray(result.scaler.center_)
            factors[f"{arm}.context_scale"] = np.asarray(result.scaler.scale_)
            for name, value in result.model.state_dict().items():
                factors[f"{arm}.{name}"] = value.detach().cpu().numpy()
        if attention_mass is not None:
            factors["ordered_history.test_history_attention_mass"] = attention_mass
            factors[
                "ordered_history.test_history_attention_excess_over_token_fraction"
            ] = attention_excess
        probabilities = {
            name: result.probability for name, result in arms.items()
        }
        probabilities.update(intervention_probabilities)
        ordered_probability = probabilities["ordered_history"]
        current_ablation = probabilities["ordered_model_current_ablation"]
        shuffled_ablation = probabilities["ordered_model_shuffled_ablation"]
        eligible_history = np.asarray(
            [example.visible_steps > 1 for example in test_rows.examples], dtype=bool
        )
        history_mass_mean = (
            float(np.mean(attention_mass[eligible_history]))
            if attention_mass is not None and np.any(eligible_history)
            else None
        )
        history_attention_excess_mean = (
            float(np.mean(attention_excess[eligible_history]))
            if attention_excess is not None and np.any(eligible_history)
            else None
        )
        return MethodFoldResult(
            probabilities=probabilities,
            diagnostics={
                "projection_dim": self.config.pca_dim,
                "sequence_unit": self.config.sequence_unit,
                "sequence_encoder": self.config.sequence_encoder,
                "step_pooling": (
                    "none" if self.config.sequence_unit == "token" else "step_end"
                ),
                "all_visible_step_tokens_preserved": (
                    self.config.sequence_unit == "token"
                ),
                "projection_training_rows": projector.training_rows,
                "pca_fit_scope": "outer_train_unique_chains",
                "train_unique_chains": len(_latest_examples(fold.train_examples)),
                "test_unique_chains": len(_latest_examples(fold.test_examples)),
                "inner_train_rows": len(inner_train),
                "inner_validation_rows": len(validation),
                "state_arm_parameter_counts": state_counts,
                "output_only_parameter_count": _parameter_count(
                    arms["output_only"].model
                ),
                "selected_epochs": {
                    name: result.selected_epoch for name, result in arms.items()
                },
                "validation_nll": {
                    name: result.validation_nll for name, result in arms.items()
                },
                "target_alignment": (
                    "completed_prefix_predicts_next_step_first_error"
                ),
                "uses_final_response_length": False,
                "uses_post_error_states": False,
                "shuffle_preserves_current_state": True,
                "state_comparison_design": (
                    "identical_encoder_and_head_for_initial_current_ordered_shuffled_"
                    f"{self.config.sequence_unit}_sequences"
                ),
                "same_model_history_interventions": True,
                "same_model_history_ablation_max_abs_probability_change": float(
                    np.max(np.abs(ordered_probability - current_ablation))
                ),
                "same_model_history_shuffle_max_abs_probability_change": float(
                    np.max(np.abs(ordered_probability - shuffled_ablation))
                ),
                "ordered_history_attention_mass_mean": history_mass_mean,
                "ordered_history_attention_excess_over_token_fraction_mean": (
                    history_attention_excess_mean
                ),
                "attention_is_descriptive_not_causal_attribution": (
                    attention_mass is not None
                ),
            },
            factors=factors,
        )


__all__ = [
    "PredictiveStateConfig",
    "PredictiveStateMonitor",
]
