from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


SCALAR_TOKEN_FEATURES = (
    "out.entropy_norm",
    "out.chosen_logprob",
    "out.logit_margin",
    "out.top5_mass",
    "out.top20_mass",
    "out.chosen_log_rank",
    "out.js_velocity",
    "out.hellinger_velocity",
)


@dataclass(frozen=True)
class LogitTraceConfig:
    top_k: int = 64
    sketch_dim: int = 64
    token_chunk_size: int = 32
    eps: float = 1e-8

    def validate(self) -> None:
        if self.top_k < 20:
            raise ValueError("top_k must be at least 20 for exact top-20 mass")
        if self.sketch_dim < 4:
            raise ValueError("sketch_dim must be at least 4")
        if self.token_chunk_size < 1:
            raise ValueError("token_chunk_size must be positive")


def token_feature_names(sketch_dim: int) -> tuple[str, ...]:
    full = tuple(
        f"out.probability_count_sketch.{i:03d}" for i in range(int(sketch_dim))
    )
    topk = tuple(f"out.topk_count_sketch.{i:03d}" for i in range(int(sketch_dim)))
    return SCALAR_TOKEN_FEATURES + full + topk


def step_feature_names(sketch_dim: int) -> tuple[str, ...]:
    names: list[str] = []
    for name in SCALAR_TOKEN_FEATURES:
        names.extend(
            f"{name}.{summary}" for summary in ("mean", "max", "last", "slope")
        )
    for family in ("probability_count_sketch", "topk_count_sketch"):
        for i in range(int(sketch_dim)):
            base = f"out.{family}.{i:03d}"
            names.extend((f"{base}.mean", f"{base}.last"))
    names.append("control.log1p_step_tokens")
    return tuple(names)


def count_sketch_topk(
    top_ids: torch.Tensor,
    top_probabilities: torch.Tensor,
    sketch_dim: int,
) -> torch.Tensor:
    """Signed count-sketch of a top-k probability vector.

    The hash is fixed and label-free.  It preserves token identity information
    without storing a vocabulary-sized vector or learning from the test labels.
    """

    if top_ids.ndim != 2 or top_probabilities.shape != top_ids.shape:
        raise ValueError("top-k IDs and probabilities must have shape [batch, k]")
    dim = int(sketch_dim)
    ids = top_ids.to(torch.int64)
    bins = torch.remainder(ids * 1_103_515_245 + 12_345, 2_147_483_647)
    bins = torch.remainder(bins, dim)
    signs = torch.remainder(ids * 2_654_435_761 + 97, 2).to(top_probabilities.dtype)
    signs = signs.mul_(2.0).sub_(1.0)
    sketch = torch.zeros(
        (top_ids.shape[0], dim),
        dtype=top_probabilities.dtype,
        device=top_probabilities.device,
    )
    sketch.scatter_add_(1, bins, top_probabilities * signs)
    return sketch


def count_sketch_distribution(
    probability: torch.Tensor, sketch_dim: int
) -> torch.Tensor:
    """Count-sketch the complete vocabulary distribution, including the tail."""

    if probability.ndim != 2:
        raise ValueError("probability must have shape [batch, vocabulary]")
    dim = int(sketch_dim)
    ids = torch.arange(
        probability.shape[1], device=probability.device, dtype=torch.int64
    )
    bins = torch.remainder(ids * 1_103_515_245 + 12_345, 2_147_483_647)
    bins = torch.remainder(bins, dim)
    signs = torch.remainder(ids * 2_654_435_761 + 97, 2).to(probability.dtype)
    signs = signs.mul_(2.0).sub_(1.0)
    sketch = torch.zeros(
        (probability.shape[0], dim),
        dtype=probability.dtype,
        device=probability.device,
    )
    sketch.scatter_add_(
        1,
        bins[None, :].expand(probability.shape[0], -1),
        probability * signs[None, :],
    )
    return sketch


@torch.inference_mode()
def compact_features_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    cfg: LogitTraceConfig,
    *,
    previous_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert logits into finite, shift-invariant compact features.

    ``previous_logits`` contains the one distribution immediately preceding
    this chunk and is used only for the first velocity row.  The returned
    second value is the final logit row and can be carried into the next chunk.
    """

    cfg.validate()
    if (
        logits.ndim != 2
        or target_ids.ndim != 1
        or logits.shape[0] != target_ids.numel()
    ):
        raise ValueError("expected logits [tokens, vocab] and matching target_ids")
    values = logits.float()
    targets = target_ids.to(device=values.device, dtype=torch.long)
    log_prob = torch.log_softmax(values, dim=-1)
    probability = log_prob.exp()
    vocab = int(values.shape[-1])
    entropy = -(probability * log_prob).sum(dim=-1) / np.log(max(vocab, 2))
    chosen_logprob = log_prob.gather(1, targets[:, None]).squeeze(1)
    k = min(int(cfg.top_k), vocab)
    top_values, top_ids = torch.topk(values, k=k, dim=-1)
    top_prob = probability.gather(1, top_ids)
    margin = top_values[:, 0] - top_values[:, 1]
    top5_mass = top_prob[:, : min(5, k)].sum(dim=-1)
    top20_mass = top_prob[:, : min(20, k)].sum(dim=-1)
    chosen_value = values.gather(1, targets[:, None])
    chosen_rank = 1 + (values > chosen_value).sum(dim=-1)
    chosen_log_rank = torch.log(chosen_rank.to(values.dtype))

    if previous_logits is None:
        previous_probability = probability[:1]
    else:
        previous_probability = torch.softmax(previous_logits.float(), dim=-1)
    paired_previous = torch.cat([previous_probability, probability[:-1]], dim=0)
    mixture = 0.5 * (probability + paired_previous)
    eps = float(cfg.eps)
    js = 0.5 * (
        (
            probability
            * (
                torch.log(probability.clamp_min(eps))
                - torch.log(mixture.clamp_min(eps))
            )
        ).sum(dim=-1)
        + (
            paired_previous
            * (
                torch.log(paired_previous.clamp_min(eps))
                - torch.log(mixture.clamp_min(eps))
            )
        ).sum(dim=-1)
    )
    hellinger = torch.sqrt(
        0.5
        * (
            torch.sqrt(probability.clamp_min(0.0))
            - torch.sqrt(paired_previous.clamp_min(0.0))
        )
        .square()
        .sum(dim=-1)
    )
    if previous_logits is None:
        js[0] = 0.0
        hellinger[0] = 0.0
    probability_sketch = count_sketch_distribution(probability, cfg.sketch_dim)
    topk_sketch = count_sketch_topk(top_ids, top_prob, cfg.sketch_dim)
    scalar = torch.stack(
        (
            entropy,
            chosen_logprob,
            margin,
            top5_mass,
            top20_mass,
            chosen_log_rank,
            js,
            hellinger,
        ),
        dim=-1,
    )
    features = torch.cat([scalar, probability_sketch, topk_sketch], dim=-1)
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError("compact logit features contain a non-finite value")
    return features, values[-1:].detach()


def _linear_slope(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros(values.shape[1], dtype=np.float32)
    x = np.linspace(-1.0, 1.0, values.shape[0], dtype=np.float64)
    denom = float(np.dot(x, x))
    return ((x[:, None] * values).sum(axis=0) / max(denom, 1e-12)).astype(np.float32)


def aggregate_token_features_to_steps(
    token_features: np.ndarray,
    step_token_ranges: Sequence[tuple[int, int]],
    *,
    response_start_token: int,
    scalar_dim: int = len(SCALAR_TOKEN_FEATURES),
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate token features without losing the final output state.

    Ranges from teacher forcing use absolute sequence positions.  The returned
    ranges are inclusive and relative to the compact response-token trace.
    """

    features = np.asarray(token_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] < scalar_dim:
        raise ValueError("invalid token feature matrix")
    relative_ranges = np.asarray(
        [
            (int(a) - int(response_start_token), int(b) - int(response_start_token))
            for a, b in step_token_ranges
        ],
        dtype=np.int32,
    )
    rows: list[np.ndarray] = []
    for a, b in relative_ranges:
        if a < 0 or b < a or b >= len(features):
            raise ValueError("step range is outside the compact response trace")
        block = features[a : b + 1]
        scalar = block[:, :scalar_dim]
        sketch = block[:, scalar_dim:]
        scalar_parts = np.stack(
            [
                scalar.mean(axis=0),
                scalar.max(axis=0),
                scalar[-1],
                _linear_slope(scalar),
            ],
            axis=1,
        ).reshape(-1)
        sketch_parts = np.stack([sketch.mean(axis=0), sketch[-1]], axis=1).reshape(-1)
        rows.append(
            np.concatenate(
                [
                    scalar_parts,
                    sketch_parts,
                    np.asarray([np.log1p(len(block))], dtype=np.float32),
                ]
            )
        )
    result = np.stack(rows, axis=0).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("step aggregation contains a non-finite value")
    return result, relative_ranges
