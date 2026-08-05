from __future__ import annotations

import torch
from torch import nn


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


class _LayerSetSummary(nn.Module):
    """Summarize a layer set without assigning meaning to layer order."""

    def __init__(self, hidden_size: int, width: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, width)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        encoded = torch.nn.functional.gelu(self.input_projection(states))
        return torch.cat([encoded.mean(dim=1), encoded.amax(dim=1)], dim=-1)


class _TwoBoundaryMonitor(nn.Module):
    """Shared-capacity base for tests of a directed boundary update."""

    def __init__(self, hidden_size: int, context_size: int, width: int) -> None:
        super().__init__()
        if min(hidden_size, context_size, width) < 1:
            raise ValueError("model dimensions must be positive")
        self.summary = _LayerSetSummary(hidden_size, width)
        self.head = nn.Sequential(
            nn.Linear(4 * width + context_size, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def _boundary_views(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if states.ndim != 4 or states.shape[1] != 2:
            raise ValueError(
                "states must have shape [batch,previous/current,layer,hidden]"
            )
        if context.ndim != 2 or context.shape[0] != states.shape[0]:
            raise ValueError("context must have shape [batch,feature]")
        first, second = self._boundary_views(states[:, 0], states[:, 1])
        pooled = torch.cat([self.summary(first), self.summary(second)], dim=-1)
        return self.head(torch.cat([pooled, context], dim=-1)).squeeze(-1)


class StaticLayerSetMonitor(_TwoBoundaryMonitor):
    """Same-capacity control containing the current boundary but no update."""

    def __init__(self, hidden_size: int, context_size: int, width: int = 64) -> None:
        super().__init__(hidden_size, context_size, width)

    def _boundary_views(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del previous
        return current, torch.zeros_like(current)


class TwoBoundaryBagMonitor(_TwoBoundaryMonitor):
    """Directionless control containing boundary level and change magnitude."""

    def __init__(self, hidden_size: int, context_size: int, width: int = 64) -> None:
        super().__init__(hidden_size, context_size, width)

    def _boundary_views(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (previous + current) / 2.0, torch.abs(current - previous)


class TwoBoundaryInnovationMonitor(_TwoBoundaryMonitor):
    """Current boundary plus the signed previous-to-current state update."""

    def __init__(self, hidden_size: int, context_size: int, width: int = 64) -> None:
        super().__init__(hidden_size, context_size, width)

    def _boundary_views(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return current, current - previous


__all__ = [
    "ContextMonitor",
    "StaticLayerSetMonitor",
    "TwoBoundaryBagMonitor",
    "TwoBoundaryInnovationMonitor",
]
