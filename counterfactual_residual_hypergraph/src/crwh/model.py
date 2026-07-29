from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .contracts import (
    CounterfactualExample,
    ResidualWriteHypergraph,
    _validated_finite_real,
    _validated_int,
)


@dataclass(frozen=True)
class ModelOutputs:
    logits: torch.Tensor
    factual_embedding: torch.Tensor
    counterfactual_embedding: torch.Tensor
    paraphrase_embedding: torch.Tensor | None


class DirectedResidualHypergraphLayer(nn.Module):
    """HyperCHARM-style aggregation that updates explicit receivers only."""

    def __init__(self, *, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        hidden_dim = _validated_int(
            hidden_dim,
            name="hidden_dim",
            minimum=1,
        )
        edge_dim = _validated_int(edge_dim, name="edge_dim", minimum=1)
        self.edge_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        states: torch.Tensor,
        incidence: torch.Tensor,
        receivers: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        edge_count = edge_features.shape[0]
        if edge_count == 0:
            return states
        node_ids, edge_ids = incidence
        source_mask = node_ids != receivers[edge_ids]
        source_nodes = node_ids[source_mask]
        source_edges = edge_ids[source_mask]
        source_sum = states.new_zeros((edge_count, states.shape[1]))
        source_sum.index_add_(0, source_edges, states[source_nodes])
        source_count = states.new_zeros(edge_count)
        source_count.index_add_(
            0,
            source_edges,
            torch.ones_like(source_edges, dtype=states.dtype),
        )
        if torch.any(source_count == 0):
            raise ValueError("every hyperedge must contain a non-receiver source")
        source_mean = source_sum / source_count[:, None]
        receiver_state = states[receivers]
        edge_message = self.edge_encoder(
            torch.cat((source_mean, receiver_state, edge_features), dim=1)
        )

        received = states.new_zeros(states.shape)
        received.index_add_(0, receivers, edge_message)
        receiver_count = states.new_zeros(states.shape[0])
        receiver_count.index_add_(
            0,
            receivers,
            torch.ones_like(receivers, dtype=states.dtype),
        )
        active = receiver_count > 0
        received[active] /= receiver_count[active, None]
        gate = torch.sigmoid(self.gate(torch.cat((states, received), dim=1)))
        candidate = self.norm(states + gate * received)
        return torch.where(active[:, None], candidate, states)


class ResidualHypergraphEncoder(nn.Module):
    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        node_dim = _validated_int(node_dim, name="node_dim", minimum=1)
        edge_dim = _validated_int(edge_dim, name="edge_dim", minimum=1)
        hidden_dim = _validated_int(hidden_dim, name="hidden_dim", minimum=1)
        num_layers = _validated_int(
            num_layers,
            name="num_layers",
            minimum=1,
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            DirectedResidualHypergraphLayer(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
            )
            for _ in range(num_layers)
        )

    def forward(self, graph: ResidualWriteHypergraph) -> torch.Tensor:
        device = next(self.parameters()).device
        states = self.node_encoder(
            torch.as_tensor(
                graph.node_features,
                dtype=torch.float32,
                device=device,
            )
        )
        incidence = torch.as_tensor(
            graph.incidence, dtype=torch.long, device=device
        )
        receivers = torch.as_tensor(
            graph.receivers, dtype=torch.long, device=device
        )
        edge_features = torch.as_tensor(
            graph.edge_features, dtype=torch.float32, device=device
        )
        for layer in self.layers:
            states = layer(states, incidence, receivers, edge_features)
        response_nodes = torch.as_tensor(
            graph.response_nodes, dtype=torch.long, device=device
        )
        return states[response_nodes]


class MultiViewHypergraphDetector(nn.Module):
    """Shared encoder with MVHTR-inspired cross-view gated readout."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        dropout = _validated_finite_real(
            dropout,
            name="dropout",
            minimum=0.0,
        )
        if dropout > 1.0:
            raise ValueError("dropout must lie in [0, 1]")
        self.encoder = ResidualHypergraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.context_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, example: CounterfactualExample) -> ModelOutputs:
        factual = self.encoder(example.factual)
        counterfactual = self.encoder(example.counterfactual)
        paraphrase = (
            self.encoder(example.paraphrase)
            if example.paraphrase is not None
            else None
        )
        context_delta = torch.abs(factual - counterfactual)
        paraphrase_delta = (
            torch.abs(factual - paraphrase)
            if paraphrase is not None
            else torch.zeros_like(factual)
        )
        gate = torch.sigmoid(
            self.context_gate(torch.cat((context_delta, paraphrase_delta), dim=1))
        )
        fused = torch.cat(
            (
                factual,
                gate * context_delta,
                (1.0 - gate) * paraphrase_delta,
            ),
            dim=1,
        )
        return ModelOutputs(
            logits=self.readout(fused).squeeze(-1),
            factual_embedding=factual,
            counterfactual_embedding=counterfactual,
            paraphrase_embedding=paraphrase,
        )
