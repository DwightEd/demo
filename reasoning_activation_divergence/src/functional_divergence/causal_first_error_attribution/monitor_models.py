from __future__ import annotations

import numpy as np
import torch
from torch import nn


def depth_neighbor_mean(values: torch.Tensor) -> torch.Tensor:
    """Aggregate immediate predecessor/successor messages on the depth chain."""

    if values.ndim != 3 or values.shape[1] < 2:
        raise ValueError("values must have shape [batch,layer,feature] with >=2 layers")
    result = torch.zeros_like(values)
    counts = torch.zeros(
        (1, values.shape[1], 1), dtype=values.dtype, device=values.device
    )
    result[:, 1:] += values[:, :-1]
    counts[:, 1:] += 1
    result[:, :-1] += values[:, 1:]
    counts[:, :-1] += 1
    return result / counts


def path_edge_overlap(permutation: np.ndarray) -> int:
    order = np.asarray(permutation, dtype=np.int64).reshape(-1)
    if len(np.unique(order)) != len(order) or sorted(order.tolist()) != list(
        range(len(order))
    ):
        raise ValueError("permutation must contain every layer position exactly once")
    return int(np.sum(np.abs(np.diff(order)) == 1))


def fixed_layer_permutation(layer_count: int, seed: int) -> np.ndarray:
    if int(layer_count) < 2:
        raise ValueError("layer_count must be at least two")
    rng = np.random.default_rng(int(seed))
    best = np.arange(int(layer_count), dtype=np.int64)
    best_overlap = path_edge_overlap(best)
    for _ in range(10_000):
        candidate = rng.permutation(int(layer_count)).astype(np.int64)
        overlap = path_edge_overlap(candidate)
        if overlap < best_overlap:
            best = candidate
            best_overlap = overlap
        if overlap == 0:
            return candidate
    return best


class ContextMonitor(nn.Module):
    """Nuisance/output-only baseline without access to hidden states."""

    def __init__(self, context_size: int, width: int = 32) -> None:
        super().__init__()
        if context_size < 1 or width < 1:
            raise ValueError("context_size and width must be positive")
        self.network = nn.Sequential(
            nn.Linear(context_size, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        del states
        if context.ndim != 2:
            raise ValueError("context must have shape [batch,feature]")
        return self.network(context).squeeze(-1)


class LayerSetMonitor(nn.Module):
    """Capacity control that sees all states but discards layer order and edges."""

    def __init__(self, hidden_size: int, context_size: int, width: int = 64) -> None:
        super().__init__()
        if min(hidden_size, context_size, width) < 1:
            raise ValueError("model dimensions must be positive")
        self.input_projection = nn.Linear(hidden_size, width)
        self.head = nn.Sequential(
            nn.Linear(2 * width + context_size, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError("states must have shape [batch,layer,hidden]")
        encoded = torch.nn.functional.gelu(self.input_projection(states))
        pooled = torch.cat([encoded.mean(dim=1), encoded.amax(dim=1)], dim=-1)
        return self.head(torch.cat([pooled, context], dim=-1)).squeeze(-1)


class _DepthMessageBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.self_update = nn.Linear(width, width)
        self.neighbor_update = nn.Linear(width, width, bias=False)
        self.normalization = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.self_update(values) + self.neighbor_update(
            depth_neighbor_mean(values)
        )
        update = self.dropout(torch.nn.functional.gelu(update))
        return self.normalization(values + update)


class DepthGraphMonitor(nn.Module):
    """Message-passing monitor on the ordered residual-depth graph."""

    def __init__(
        self,
        *,
        hidden_size: int,
        layer_count: int,
        context_size: int,
        width: int = 64,
        message_passing_steps: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(hidden_size, layer_count, context_size, width) < 1:
            raise ValueError("model dimensions must be positive")
        if message_passing_steps < 1:
            raise ValueError("message_passing_steps must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.input_projection = nn.Linear(hidden_size, width)
        self.layer_position = nn.Embedding(layer_count, width)
        self.depth_blocks = nn.ModuleList(
            [_DepthMessageBlock(width, dropout) for _ in range(message_passing_steps)]
        )
        self.pool_gate = nn.Linear(width, 1)
        self.head = nn.Sequential(
            nn.Linear(width + context_size, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError("states must have shape [batch,layer,hidden]")
        if states.shape[1] != self.layer_position.num_embeddings:
            raise ValueError("state layer count disagrees with the graph")
        positions = torch.arange(states.shape[1], device=states.device)
        values = self.input_projection(states) + self.layer_position(positions)[None]
        for block in self.depth_blocks:
            values = block(values)
        weights = torch.softmax(self.pool_gate(values).squeeze(-1), dim=1)
        pooled = torch.sum(values * weights.unsqueeze(-1), dim=1)
        return self.head(torch.cat([pooled, context], dim=-1)).squeeze(-1)


__all__ = [
    "ContextMonitor",
    "DepthGraphMonitor",
    "LayerSetMonitor",
    "depth_neighbor_mean",
    "fixed_layer_permutation",
    "path_edge_overlap",
]
