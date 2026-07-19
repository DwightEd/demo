"""Faithful attention-row HyperCHARM token classifier.

The default path deliberately mirrors the original local HyperCHARM design:

1. incidence members are averaged into a hyperedge with uniform weights;
2. the resulting hyperedge message is broadcast to every incidence member;
3. the model emits one logit per token.

Attention-weighted node-to-edge aggregation, receiver-only propagation, and
receiver/source interaction are explicit ablations.  They are not silently
enabled by the default configuration.

This module does not require PyTorch Geometric.  ``forward`` accepts the NumPy
``AttentionHypergraph`` dataclass, a mapping, or any object exposing the schema
fields used below.  Importing the module remains safe when PyTorch is absent.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Optional

try:  # pragma: no cover - availability depends on the execution environment
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


def require_torch() -> None:
    """Fail with an actionable message only when tensor functionality is used."""

    if torch is None:
        raise RuntimeError("the attention HyperCHARM model requires PyTorch")


_MISSING = object()


def _field(obj: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
    elif hasattr(obj, name):
        return getattr(obj, name)
    if default is _MISSING:
        raise KeyError(f"graph is missing required field {name!r}")
    return default


def to_torch_data(graph: Any, device: Optional[Any] = None) -> SimpleNamespace:
    """Convert one framework-neutral attention hypergraph to tensor fields.

    Extra supervision and audit fields are retained when present.  This helper
    intentionally returns a ``SimpleNamespace`` rather than a PyG ``Data`` so
    the faithful single-graph path has no optional PyG dependency.
    """

    require_torch()
    tensor_specs = {
        "x": torch.float32,
        "he_index": torch.long,
        "he_attr": torch.float32,
        "he_mark": torch.float32,
        "he_weight": torch.float32,
        "he_attention": torch.float32,
        "he_count": torch.float32,
        "he_receiver": torch.long,
        "he_layer": torch.long,
        "he_head": torch.long,
        "token_ids": torch.long,
        "token_y": torch.float32,
        "token_label_mask": torch.bool,
        "y": torch.float32,
        "response_y": torch.float32,
        "step_ranges": torch.long,
        "gold_step": torch.long,
        "step_loss_mask": torch.bool,
    }
    required = {
        "x",
        "he_index",
        "he_attr",
        "he_mark",
        "he_weight",
        "he_attention",
        "he_receiver",
    }
    values = {}
    for name, dtype in tensor_specs.items():
        value = _field(graph, name, None)
        if value is None:
            if name in required:
                raise KeyError(f"graph is missing required field {name!r}")
            continue
        values[name] = torch.as_tensor(value, dtype=dtype, device=device)

    response_start = _field(graph, "response_start", None)
    if response_start is None:
        response_start = _field(graph, "response_idx", None)
    if response_start is not None:
        response_start = torch.as_tensor(response_start, dtype=torch.long, device=device)
        values["response_start"] = response_start
        values["response_idx"] = response_start

    propagation_mode = _field(graph, "propagation_mode", None)
    if propagation_mode is not None:
        values["propagation_mode"] = str(propagation_mode)
    incidence_weight_mode = _field(graph, "incidence_weight_mode", None)
    if incidence_weight_mode is not None:
        values["incidence_weight_mode"] = str(incidence_weight_mode)
    return SimpleNamespace(**values)


if torch is not None:

    def make_mlp(
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        *,
        dropout: float = 0.0,
        norm: bool = False,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous = int(in_dim)
        for hidden in hidden_dims:
            layers.append(nn.Linear(previous, hidden))
            if norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous = hidden
        layers.append(nn.Linear(previous, out_dim))
        return nn.Sequential(*layers)


    class HyperCharmLayer(nn.Module):
        """One node-to-hyperedge-to-node message-passing layer.

        ``incidence_weighting='uniform'`` and
        ``directed_receiver_only=False`` are the faithful symmetric defaults.
        ``attention`` uses raw attention as a weighted sum, preserving selected
        attention mass. ``normalized_attention`` uses a convex weighted mean.
        These are distinct opt-in innovations; ``uniform`` remains faithful.
        """

        _WEIGHTING_MODES = frozenset(
            {"uniform", "attention", "normalized_attention"}
        )

        def __init__(
            self,
            node_dim: int,
            hedge_dim: int,
            hidden_dim: int,
            *,
            mark_dim: int = 2,
            residual: bool = True,
            directed_receiver_only: bool = False,
            incidence_weighting: str = "uniform",
            receiver_source_interaction: bool = False,
            mlp_norm: bool = True,
            message_operator: str = "hypergraph",
        ) -> None:
            super().__init__()
            if incidence_weighting not in self._WEIGHTING_MODES:
                raise ValueError(
                    "incidence_weighting must be uniform, attention, or "
                    "normalized_attention; "
                    f"got {incidence_weighting!r}"
                )
            self.residual = bool(residual)
            self.directed_receiver_only = bool(directed_receiver_only)
            self.incidence_weighting = incidence_weighting
            self.receiver_source_interaction = bool(receiver_source_interaction)
            if message_operator not in {"hypergraph", "pairwise"}:
                raise ValueError("message_operator must be 'hypergraph' or 'pairwise'")
            if message_operator == "pairwise" and not directed_receiver_only:
                raise ValueError(
                    "the directed pairwise attention baseline requires receiver-only propagation"
                )
            self.message_operator = str(message_operator)
            self.mark_dim = int(mark_dim)

            # The innovation path explicitly compares each incidence member to
            # the attention-row receiver.  The faithful path sees only the
            # member state and the original two-dimensional edge mark.
            interaction_factor = 4 if self.receiver_source_interaction else 1
            node2edge_dim = interaction_factor * node_dim + self.mark_dim
            self.node2edge = make_mlp(
                node2edge_dim, [hidden_dim], hidden_dim, norm=mlp_norm
            )
            self.edge2node = make_mlp(
                hedge_dim + hidden_dim, [hidden_dim], node_dim, norm=mlp_norm
            )
            self.output_norm = nn.LayerNorm(node_dim)

        def _validate_inputs(
            self, x, he_index, he_attr, he_mark, he_weight, he_attention, he_receiver
        ) -> None:
            if he_index.ndim != 2 or he_index.size(0) != 2:
                raise ValueError("he_index must have shape [2, num_incidences]")
            num_edges = int(he_attr.size(0))
            num_incidences = int(he_index.size(1))
            if he_mark.ndim != 2 or he_mark.shape != (num_edges, self.mark_dim):
                raise ValueError(
                    f"he_mark must have shape [{num_edges}, {self.mark_dim}]"
                )
            if he_weight.ndim != 1 or he_weight.numel() != num_incidences:
                raise ValueError(
                    "he_weight must have one scalar per incidence; expected "
                    f"{num_incidences}, got {he_weight.numel()}"
                )
            if he_attention.ndim != 1 or he_attention.numel() != num_incidences:
                raise ValueError(
                    "he_attention must have one raw query-key value per incidence"
                )
            if he_receiver.ndim != 1 or he_receiver.numel() != num_edges:
                raise ValueError("he_receiver must have one node index per hyperedge")
            if num_incidences:
                node_ids, edge_ids = he_index
                if int(node_ids.min()) < 0 or int(node_ids.max()) >= x.size(0):
                    raise ValueError("he_index contains an out-of-range node index")
                if int(edge_ids.min()) < 0 or int(edge_ids.max()) >= num_edges:
                    raise ValueError("he_index contains an out-of-range edge index")
            if num_edges and (
                int(he_receiver.min()) < 0 or int(he_receiver.max()) >= x.size(0)
            ):
                raise ValueError("he_receiver contains an out-of-range node index")

        def forward(
            self,
            x,
            he_index,
            he_attr,
            he_mark,
            he_weight,
            he_attention,
            he_receiver,
        ):
            self._validate_inputs(
                x, he_index, he_attr, he_mark, he_weight, he_attention, he_receiver
            )
            num_edges = int(he_attr.size(0))
            if num_edges == 0:
                # Required for a genuine feature-only/no-edge control.
                return x

            node_ids = he_index[0]
            edge_ids = he_index[1]
            source_state = x[node_ids]
            per_incidence_mark = he_mark[edge_ids]
            if self.receiver_source_interaction:
                receiver_state = x[he_receiver[edge_ids]]
                node2edge_input = torch.cat(
                    [
                        source_state,
                        receiver_state,
                        source_state - receiver_state,
                        source_state * receiver_state,
                        per_incidence_mark,
                    ],
                    dim=-1,
                )
            else:
                node2edge_input = torch.cat(
                    [source_state, per_incidence_mark], dim=-1
                )
            incidence_messages = self.node2edge(node2edge_input)

            if self.incidence_weighting == "uniform":
                weights = torch.ones_like(he_weight, dtype=incidence_messages.dtype)
            else:
                if not torch.isfinite(he_weight).all() or bool((he_weight < 0).any()):
                    raise ValueError("attention incidence weights must be finite and nonnegative")
                weights = he_weight.to(dtype=incidence_messages.dtype)

            if self.message_operator == "pairwise":
                # Same support, encoders, and parameter count as the hypergraph
                # path, but decode each query-key incidence before aggregation.
                # Retaining the centre/self incidence keeps this an operator
                # ablation rather than silently changing graph support.
                pair_edge_ids = edge_ids
                pair_attr = he_attr[pair_edge_ids].clone()
                if pair_attr.size(-1) >= 2:
                    pair_attention = he_attention.to(dtype=pair_attr.dtype)
                    pair_attr[:, 0] = pair_attention
                    pair_attr[:, 1] = pair_attention
                pair_messages = F.relu(
                    self.edge2node(
                        torch.cat(
                            [pair_attr, incidence_messages],
                            dim=-1,
                        )
                    )
                )
                pair_weights = weights
                edge_messages = torch.zeros(
                    (num_edges, x.size(-1)), dtype=x.dtype, device=x.device
                )
                edge_messages.index_add_(
                    0, pair_edge_ids, pair_messages * pair_weights.unsqueeze(-1)
                )
                if self.incidence_weighting != "attention":
                    pair_weight_sum = torch.zeros(
                        num_edges, dtype=x.dtype, device=x.device
                    )
                    pair_weight_sum.index_add_(0, pair_edge_ids, pair_weights)
                    edge_messages = edge_messages / (
                        pair_weight_sum + 1e-6
                    ).unsqueeze(-1)
            else:
                edge_sum = torch.zeros(
                    (num_edges, incidence_messages.size(-1)),
                    dtype=incidence_messages.dtype,
                    device=x.device,
                )
                edge_sum.index_add_(
                    0, edge_ids, incidence_messages * weights.unsqueeze(-1)
                )
                edge_weight_sum = torch.zeros(
                    num_edges, dtype=incidence_messages.dtype, device=x.device
                )
                edge_weight_sum.index_add_(0, edge_ids, weights)
                if self.incidence_weighting == "attention":
                    edge_state = edge_sum
                else:
                    # Faithful uniform mean or normalized-attention convex mean.
                    edge_state = edge_sum / (edge_weight_sum + 1e-6).unsqueeze(-1)

                edge_messages = F.relu(
                    self.edge2node(torch.cat([he_attr, edge_state], dim=-1))
                )
            out = torch.zeros_like(x)
            if self.directed_receiver_only:
                out.index_add_(0, he_receiver, edge_messages)
                degree = torch.bincount(
                    he_receiver, minlength=x.size(0)
                ).to(dtype=x.dtype, device=x.device)
            else:
                # Faithful HyperCHARM: every member receives its incident
                # hyperedge message, including the attention-row centre.
                out.index_add_(0, node_ids, edge_messages[edge_ids])
                degree = torch.bincount(
                    node_ids, minlength=x.size(0)
                ).to(dtype=x.dtype, device=x.device)
            out = out / (degree + 1e-6).unsqueeze(-1)
            out = self.output_norm(out)
            return x + out if self.residual else out


    class HyperCHARMToken(nn.Module):
        """Token-logit HyperCHARM with faithful defaults and explicit ablations."""

        def __init__(
            self,
            node_dim: int,
            hedge_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.1,
            residual: bool = True,
            *,
            mark_dim: int = 2,
            directed_receiver_only: bool = False,
            incidence_weighting: str = "uniform",
            receiver_source_interaction: bool = False,
            mlp_norm: bool = True,
            classifier_norm: bool = True,
            init_weights: bool = True,
            message_operator: str = "hypergraph",
        ) -> None:
            super().__init__()
            if node_dim < 1 or hedge_dim < 0 or hidden_dim < 1:
                raise ValueError("node_dim/hidden_dim must be positive and hedge_dim nonnegative")
            if num_layers < 0:
                raise ValueError("num_layers cannot be negative")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must lie in [0, 1)")

            self.node_dim = int(node_dim)
            self.hedge_dim = int(hedge_dim)
            self.propagation_mode = (
                "receiver" if bool(directed_receiver_only) else "symmetric"
            )
            self.incidence_weighting = str(incidence_weighting)
            self.message_operator = str(message_operator)
            self.in_proj = nn.Linear(node_dim, hidden_dim)
            self.layers = nn.ModuleList(
                [
                    HyperCharmLayer(
                        hidden_dim,
                        hedge_dim,
                        hidden_dim,
                        mark_dim=mark_dim,
                        residual=residual,
                        directed_receiver_only=directed_receiver_only,
                        incidence_weighting=incidence_weighting,
                        receiver_source_interaction=receiver_source_interaction,
                        mlp_norm=mlp_norm,
                        message_operator=message_operator,
                    )
                    for _ in range(num_layers)
                ]
            )
            classifier_hidden = max(8, hidden_dim // 2)
            classifier_layers: list[nn.Module] = [
                nn.Linear(hidden_dim, classifier_hidden)
            ]
            if classifier_norm:
                classifier_layers.append(nn.LayerNorm(classifier_hidden))
            classifier_layers.extend(
                [
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(classifier_hidden, 1),
                ]
            )
            self.pred = nn.Sequential(*classifier_layers)
            if init_weights:
                self._init_weights()

        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, data):
            parameter_device = self.in_proj.weight.device
            if not torch.is_tensor(_field(data, "x")):
                data = to_torch_data(data, device=parameter_device)

            graph_propagation = _field(data, "propagation_mode", None)
            if (
                graph_propagation is not None
                and str(graph_propagation) != self.propagation_mode
            ):
                raise ValueError(
                    "model propagation does not match graph metadata: "
                    f"model={self.propagation_mode!r}, graph={str(graph_propagation)!r}"
                )
            graph_weighting = _field(data, "incidence_weight_mode", None)
            if graph_weighting is not None:
                graph_weighting = str(graph_weighting)
                compatible = graph_weighting == self.incidence_weighting
                if not compatible:
                    raise ValueError(
                        "model incidence weighting does not match graph metadata: "
                        f"model={self.incidence_weighting!r}, graph={graph_weighting!r}"
                    )

            x = torch.as_tensor(
                _field(data, "x"), dtype=torch.float32, device=parameter_device
            )
            he_index = torch.as_tensor(
                _field(data, "he_index"), dtype=torch.long, device=parameter_device
            )
            he_attr = torch.as_tensor(
                _field(data, "he_attr"), dtype=torch.float32, device=parameter_device
            )
            he_mark = torch.as_tensor(
                _field(data, "he_mark"), dtype=torch.float32, device=parameter_device
            )
            he_weight = torch.as_tensor(
                _field(data, "he_weight"), dtype=torch.float32, device=parameter_device
            )
            he_attention = torch.as_tensor(
                _field(data, "he_attention"), dtype=torch.float32, device=parameter_device
            )
            he_receiver = torch.as_tensor(
                _field(data, "he_receiver"), dtype=torch.long, device=parameter_device
            )

            hidden = F.relu(self.in_proj(x))
            for layer in self.layers:
                hidden = layer(
                    hidden,
                    he_index,
                    he_attr,
                    he_mark,
                    he_weight,
                    he_attention,
                    he_receiver,
                )
            return self.pred(hidden).squeeze(-1)


    # Short alias for code that used the original class name.
    HyperCHARM = HyperCHARMToken


else:  # pragma: no cover - import-safe placeholders for CPU analysis setups

    class HyperCharmLayer:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            require_torch()


    class HyperCHARMToken:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            require_torch()


    HyperCHARM = HyperCHARMToken


def build_model(node_dim: int, hedge_dim: int, **kwargs) -> HyperCHARMToken:
    """Build a token-logit HyperCHARM from explicit dimensions and options."""

    require_torch()
    return HyperCHARMToken(node_dim=node_dim, hedge_dim=hedge_dim, **kwargs)


__all__ = [
    "HyperCHARM",
    "HyperCHARMToken",
    "HyperCharmLayer",
    "build_model",
    "require_torch",
    "to_torch_data",
]
