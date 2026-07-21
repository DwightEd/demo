from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import FirstErrorLabels, InterventionEffect, TransportInputs
from .contribution import OutputEffectiveTransport
from .data import CausalTrace
from .geometry import ConstraintBundleAnalyzer
from .hypergraph import CausalHypergraphBuilder


@dataclass(frozen=True)
class TraceIdentity:
    trace_id: str
    problem_id: str
    generator_model: str
    observer_model: str
    layer_id: int
    response_tokens: int


@dataclass(frozen=True)
class AssemblyInputs:
    identity: TraceIdentity
    node_features: np.ndarray
    transport: TransportInputs
    interventions: tuple[InterventionEffect, ...]
    labels: FirstErrorLabels


class CausalTraceAssembler:
    """Single audited path from model mechanisms to a trainable causal trace."""

    def __init__(
        self,
        *,
        bundle_energy: float = 0.9,
        min_effect: float = 0.0,
        min_synergy: float = 1e-6,
    ) -> None:
        self.transport = OutputEffectiveTransport()
        self.geometry = ConstraintBundleAnalyzer(energy=bundle_energy)
        self.graph_builder = CausalHypergraphBuilder(
            min_effect=min_effect, min_synergy=min_synergy
        )

    def assemble(self, inputs: AssemblyInputs) -> CausalTrace:
        if len(inputs.transport.receiver_positions) != inputs.labels.num_steps:
            raise ValueError(
                "transport queries must align one-to-one with reasoning steps"
            )
        contribution = self.transport.compute(inputs.transport)
        geometry = self.geometry.analyze(inputs.transport, contribution)
        node_features = self._attach_geometry(
            inputs.node_features,
            inputs.transport.receiver_positions,
            geometry.as_features(),
        )
        graph = self.graph_builder.build(
            node_features=node_features,
            contribution=contribution,
            interventions=inputs.interventions,
            response_nodes=inputs.transport.receiver_positions,
        )
        identity = inputs.identity
        return CausalTrace(
            trace_id=identity.trace_id,
            problem_id=identity.problem_id,
            generator_model=identity.generator_model,
            observer_model=identity.observer_model,
            layer_id=identity.layer_id,
            prompt_tokens=inputs.transport.prompt_end,
            response_tokens=identity.response_tokens,
            graph=graph,
            labels=inputs.labels,
        )

    @staticmethod
    def _attach_geometry(
        node_features: np.ndarray,
        receivers: np.ndarray,
        geometry_features: np.ndarray,
    ) -> np.ndarray:
        nodes = np.asarray(node_features)
        if nodes.ndim != 2 or len(receivers) != len(geometry_features):
            raise ValueError("node and geometry features do not align")
        if np.any((receivers < 0) | (receivers >= len(nodes))):
            raise ValueError("geometry receiver lies outside the node axis")
        geometry_nodes = np.zeros((len(nodes), geometry_features.shape[1] + 1))
        geometry_nodes[receivers, :-1] = geometry_features
        geometry_nodes[receivers, -1] = 1.0
        return np.concatenate((nodes, geometry_nodes), axis=1)
