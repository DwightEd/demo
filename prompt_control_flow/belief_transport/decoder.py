from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class DecoderConfig:
    num_folds: int = 5
    epochs: int = 80
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    validation_fraction: float = 0.15
    patience: int = 10
    decoder_kind: str = "linear"
    hidden_dim: int = 256
    device: str = "auto"
    seed: int = 17

    def validate(self) -> None:
        if self.num_folds < 2:
            raise ValueError("num_folds must be at least 2")
        if min(self.epochs, self.batch_size, self.patience, self.hidden_dim) < 1:
            raise ValueError("training sizes must be positive")
        if min(self.learning_rate, self.weight_decay) <= 0.0:
            raise ValueError("learning_rate and weight_decay must be positive")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must lie in (0, 0.5)")
        if self.decoder_kind not in {"linear", "mlp"}:
            raise ValueError("decoder_kind must be linear or mlp")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass
class DecoderResult:
    predictions: np.ndarray
    fold_ids: np.ndarray
    fold_diagnostics: list[dict[str, Any]]


def build_group_folds(
    groups: np.ndarray,
    *,
    num_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic, row-balanced folds without splitting a problem."""

    group_values = np.asarray(groups)
    unique, inverse, counts = np.unique(group_values, return_inverse=True, return_counts=True)
    if len(unique) < int(num_folds):
        raise ValueError(
            f"need at least {int(num_folds)} unique groups, found {len(unique)}"
        )
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(unique))
    order = order[np.argsort(-counts[order], kind="stable")]
    fold_load = np.zeros(int(num_folds), dtype=np.int64)
    assignment = np.empty(len(unique), dtype=np.int64)
    for group_index in order:
        fold = int(np.argmin(fold_load))
        assignment[group_index] = fold
        fold_load[fold] += counts[group_index]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(int(num_folds)):
        test_mask = assignment[inverse] == fold
        test_index = np.flatnonzero(test_mask)
        train_index = np.flatnonzero(~test_mask)
        if len(test_index) == 0 or len(train_index) == 0:
            raise RuntimeError("group fold construction produced an empty split")
        folds.append((train_index, test_index))
    return folds


def problem_balanced_row_weights(groups: np.ndarray) -> np.ndarray:
    values = np.asarray(groups)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights / np.mean(weights)


def build_control_features(artifact, *, include_output: bool) -> np.ndarray:
    groups = np.asarray(artifact.problem_ids)
    max_prefix: dict[int, int] = {}
    for problem, prefix in zip(groups, artifact.prefix_index):
        max_prefix[int(problem)] = max(max_prefix.get(int(problem), 0), int(prefix))
    relative_prefix = np.asarray(
        [
            float(prefix) / max(max_prefix[int(problem)], 1)
            for problem, prefix in zip(groups, artifact.prefix_index)
        ],
        dtype=np.float64,
    )
    family_count = int(np.max(artifact.template_families, initial=0)) + 1
    family = np.eye(max(family_count, 1), dtype=np.float64)[artifact.template_families]
    columns = [
        np.asarray(artifact.prefix_index, dtype=np.float64)[:, None],
        relative_prefix[:, None],
        np.log1p(np.asarray(artifact.prompt_token_count, dtype=np.float64))[:, None],
        family,
    ]
    if include_output:
        columns.extend(
            [
                np.asarray(artifact.output_entropy, dtype=np.float64)[:, None],
                np.asarray(artifact.output_margin, dtype=np.float64)[:, None],
                np.asarray(artifact.output_topk_mass, dtype=np.float64)[:, None],
                np.asarray(artifact.output_logit_sketch, dtype=np.float64),
            ]
        )
    return np.concatenate(columns, axis=1)


def _validation_split(
    train_index: np.ndarray,
    groups: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_groups = np.unique(groups[train_index])
    if len(train_groups) < 3:
        raise ValueError("each outer training fold needs at least three problem groups")
    rng = np.random.default_rng(int(seed))
    shuffled = rng.permutation(train_groups)
    num_validation = max(1, int(round(len(shuffled) * float(fraction))))
    num_validation = min(num_validation, len(shuffled) - 1)
    validation_groups = set(shuffled[:num_validation].tolist())
    validation_mask = np.asarray(
        [value in validation_groups for value in groups[train_index]], dtype=bool
    )
    return train_index[~validation_mask], train_index[validation_mask]


def _standardize(
    fit_x: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    center = np.mean(fit_x, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(fit_x, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0)
    transformed = [((np.asarray(fit_x, dtype=np.float32) - center) / scale).astype(np.float32)]
    transformed.extend(
        ((np.asarray(value, dtype=np.float32) - center) / scale).astype(np.float32)
        for value in others
    )
    return tuple(transformed)


def _fit_one_fold(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    cfg: DecoderConfig,
    fold: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn

    torch.manual_seed(int(cfg.seed + fold))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed + fold))
    device = torch.device(
        "cuda"
        if cfg.device == "auto" and torch.cuda.is_available()
        else ("cpu" if cfg.device == "auto" else cfg.device)
    )
    fit_index, validation_index = _validation_split(
        train_index,
        groups,
        cfg.validation_fraction,
        cfg.seed + fold,
    )
    fit_x, validation_x, test_x = _standardize(
        features[fit_index],
        features[validation_index],
        features[test_index],
    )
    fit_target = np.asarray(targets[fit_index], dtype=np.float32)
    validation_target = np.asarray(targets[validation_index], dtype=np.float32)
    row_weight = problem_balanced_row_weights(groups)
    fit_weight = np.asarray(row_weight[fit_index], dtype=np.float32)
    validation_weight = np.asarray(row_weight[validation_index], dtype=np.float32)

    input_dim = int(features.shape[1])
    output_dim = int(targets.shape[1])
    if cfg.decoder_kind == "linear":
        model = nn.Linear(input_dim, output_dim)
    else:
        model = nn.Sequential(
            nn.Linear(input_dim, int(cfg.hidden_dim)),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(int(cfg.hidden_dim), output_dim),
        )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    x_fit = torch.as_tensor(fit_x, device=device)
    y_fit = torch.as_tensor(fit_target, device=device)
    w_fit = torch.as_tensor(fit_weight, device=device)
    x_validation = torch.as_tensor(validation_x, device=device)
    y_validation = torch.as_tensor(validation_target, device=device)
    w_validation = torch.as_tensor(validation_weight, device=device)
    x_test = torch.as_tensor(test_x, device=device)

    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    for epoch in range(int(cfg.epochs)):
        model.train()
        permutation = torch.randperm(len(x_fit), device=device)
        for start in range(0, len(permutation), int(cfg.batch_size)):
            index = permutation[start : start + int(cfg.batch_size)]
            logits = model(x_fit[index])
            per_row = -torch.sum(
                y_fit[index] * torch.log_softmax(logits.float(), dim=-1), dim=-1
            )
            loss = torch.sum(per_row * w_fit[index]) / torch.sum(w_fit[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_logits = model(x_validation)
            validation_rows = -torch.sum(
                y_validation
                * torch.log_softmax(validation_logits.float(), dim=-1),
                dim=-1,
            )
            validation_loss = float(
                (torch.sum(validation_rows * w_validation) / torch.sum(w_validation))
                .detach()
                .cpu()
            )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break
    if best_state is None:
        raise RuntimeError("belief decoder did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predictions = torch.softmax(model(x_test).float(), dim=-1).cpu().numpy()
    diagnostics = {
        "fold": int(fold),
        "fit_rows": int(len(fit_index)),
        "validation_rows": int(len(validation_index)),
        "test_rows": int(len(test_index)),
        "best_epoch": int(best_epoch),
        "validation_soft_ce": float(best_loss),
        "device": str(device),
    }
    return predictions.astype(np.float32), diagnostics


def cross_fit_belief_decoder(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    cfg: DecoderConfig,
) -> DecoderResult:
    cfg.validate()
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    group_values = np.asarray(groups)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) != len(group_values):
        raise ValueError("features, targets, and groups have incompatible shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("decoder inputs must be finite")
    target_mass = y.sum(axis=1, keepdims=True)
    if np.any(target_mass <= 0.0):
        raise ValueError("decoder targets must have positive probability mass")
    y = y / target_mass
    predictions = np.full_like(y, np.nan, dtype=np.float32)
    fold_ids = np.full(len(y), -1, dtype=np.int64)
    diagnostics: list[dict[str, Any]] = []
    folds = build_group_folds(
        group_values,
        num_folds=cfg.num_folds,
        seed=cfg.seed,
    )
    for fold, (train_index, test_index) in enumerate(folds):
        fold_prediction, fold_diagnostics = _fit_one_fold(
            x,
            y,
            group_values,
            train_index,
            test_index,
            cfg,
            fold,
        )
        predictions[test_index] = fold_prediction
        fold_ids[test_index] = fold
        diagnostics.append(fold_diagnostics)
    if np.any(fold_ids < 0) or not np.isfinite(predictions).all():
        raise RuntimeError("cross-fitting did not produce exactly one prediction per row")
    return DecoderResult(
        predictions=predictions,
        fold_ids=fold_ids,
        fold_diagnostics=diagnostics,
    )
