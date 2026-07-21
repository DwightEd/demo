from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import CausalHypergraph

try:  # pragma: no cover - depends on the runtime image
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("the causal constraint transport model requires PyTorch")


@dataclass(frozen=True)
class TensorHypergraph:
    incidence: Any
    receivers: Any
    edge_features: Any
    response_nodes: Any

    @classmethod
    def from_graph(
        cls, graph: CausalHypergraph, *, device: Any = None
    ) -> "TensorHypergraph":
        require_torch()
        return cls(
            incidence=torch.as_tensor(graph.incidence, dtype=torch.long, device=device),
            receivers=torch.as_tensor(graph.receivers, dtype=torch.long, device=device),
            edge_features=torch.as_tensor(
                graph.edge_features, dtype=torch.float32, device=device
            ),
            response_nodes=torch.as_tensor(
                graph.response_nodes, dtype=torch.long, device=device
            ),
        )


if torch is not None:

    class DirectedCausalLayer(nn.Module):
        """Aggregate source sets and update only their measured receivers."""

        def __init__(self, *, hidden_dim: int, edge_dim: int) -> None:
            super().__init__()
            if hidden_dim <= 0 or edge_dim <= 0:
                raise ValueError("hidden_dim and edge_dim must be positive")
            self.edge_encoder = nn.Sequential(
                nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, states, graph: TensorHypergraph):
            if states.ndim != 2:
                raise ValueError("states must have shape [nodes, hidden]")
            if graph.incidence.ndim != 2 or graph.incidence.shape[0] != 2:
                raise ValueError("incidence must have shape [2, memberships]")
            edges = graph.edge_features.shape[0]
            if graph.receivers.shape != (edges,):
                raise ValueError("receivers must align with edge features")
            if not edges:
                return states

            node_ids, edge_ids = graph.incidence
            is_source = node_ids != graph.receivers[edge_ids]
            source_nodes = node_ids[is_source]
            source_edges = edge_ids[is_source]
            source_sum = states.new_zeros((edges, states.shape[1]))
            source_sum.index_add_(0, source_edges, states[source_nodes])
            source_count = states.new_zeros(edges)
            source_count.index_add_(
                0, source_edges, torch.ones_like(source_edges, dtype=states.dtype)
            )
            if torch.any(source_count == 0):
                raise ValueError("every causal edge must contain at least one source")
            source_mean = source_sum / source_count[:, None]
            receiver_state = states[graph.receivers]
            edge_message = self.edge_encoder(
                torch.cat((source_mean, receiver_state, graph.edge_features), dim=1)
            )

            received = states.new_zeros(states.shape)
            received.index_add_(0, graph.receivers, edge_message)
            receiver_count = states.new_zeros(states.shape[0])
            receiver_count.index_add_(
                0,
                graph.receivers,
                torch.ones_like(graph.receivers, dtype=states.dtype),
            )
            active = receiver_count > 0
            received[active] /= receiver_count[active, None]
            gate = torch.sigmoid(self.gate(torch.cat((states, received), dim=1)))
            candidate = self.norm(states + gate * received)
            return torch.where(active[:, None], candidate, states)

    class ConstraintTransportDetector(nn.Module):
        """Encode a causal hypergraph and emit one first-error hazard per step."""

        def __init__(
            self,
            *,
            node_dim: int,
            edge_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
        ) -> None:
            super().__init__()
            if node_dim <= 0 or num_layers <= 0:
                raise ValueError("node_dim and num_layers must be positive")
            self.node_encoder = nn.Sequential(
                nn.Linear(node_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
            self.layers = nn.ModuleList(
                DirectedCausalLayer(hidden_dim=hidden_dim, edge_dim=edge_dim)
                for _ in range(num_layers)
            )
            self.hazard_head = nn.Linear(hidden_dim, 1)

        def forward(self, graph: CausalHypergraph):
            device = next(self.parameters()).device
            tensors = TensorHypergraph.from_graph(graph, device=device)
            states = self.node_encoder(
                torch.as_tensor(graph.node_features, dtype=torch.float32, device=device)
            )
            for layer in self.layers:
                states = layer(states, tensors)
            return self.hazard_head(states[tensors.response_nodes]).squeeze(-1)

else:  # pragma: no cover - keeps imports actionable without torch

    class DirectedCausalLayer:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            require_torch()

    class ConstraintTransportDetector:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            require_torch()
