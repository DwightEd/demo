from __future__ import annotations

from dataclasses import replace

import numpy as np

from .contracts import CausalHypergraph
from .data import CausalTrace


def _members(graph: CausalHypergraph, edge: int) -> list[int]:
    return graph.incidence[0, graph.incidence[1] == edge].tolist()


def _graph(
    template: CausalHypergraph,
    memberships: list[list[int]],
    receivers: list[int],
    features: list[np.ndarray],
    kinds: list[str],
) -> CausalHypergraph:
    columns = [
        (node, edge)
        for edge, edge_members in enumerate(memberships)
        for node in edge_members
    ]
    incidence = (
        np.asarray(columns, dtype=np.int64).T
        if columns
        else np.empty((2, 0), dtype=np.int64)
    )
    return CausalHypergraph(
        node_features=template.node_features,
        incidence=incidence,
        receivers=np.asarray(receivers, dtype=np.int64),
        edge_features=np.asarray(features, dtype=np.float64).reshape(-1, 4),
        edge_kind=np.asarray(kinds, dtype=str),
        response_nodes=template.response_nodes,
    )


class NoEdgeControl:
    name = "no_edge"

    def apply(self, trace: CausalTrace) -> CausalTrace:
        return replace(trace, graph=_graph(trace.graph, [], [], [], []))


class PairwiseControl:
    name = "pairwise"

    def apply(self, trace: CausalTrace) -> CausalTrace:
        memberships: list[list[int]] = []
        receivers: list[int] = []
        features: list[np.ndarray] = []
        for edge, receiver in enumerate(trace.graph.receivers):
            sources = [
                node for node in _members(trace.graph, edge) if node != int(receiver)
            ]
            if not sources:
                continue
            edge_feature = trace.graph.edge_features[edge].copy()
            edge_feature[:2] /= len(sources)
            edge_feature[2] = 0.0
            for source in sources:
                pair_feature = edge_feature.copy()
                pair_feature[3] = float(source < trace.prompt_tokens)
                memberships.append([source, int(receiver)])
                receivers.append(int(receiver))
                features.append(pair_feature)
        return replace(
            trace,
            graph=_graph(
                trace.graph,
                memberships,
                receivers,
                features,
                ["pair"] * len(receivers),
            ),
        )


class CausalCardinalityRewire:
    """Randomize sources while preserving receivers, edge count, and cardinality."""

    name = "causal_cardinality_rewire"

    def __init__(self, *, seed: int) -> None:
        self.seed = int(seed)

    def apply(self, trace: CausalTrace) -> CausalTrace:
        rng = np.random.default_rng(self.seed)
        memberships: list[list[int]] = []
        features: list[np.ndarray] = []
        for edge, receiver in enumerate(trace.graph.receivers):
            original = [
                node for node in _members(trace.graph, edge) if node != int(receiver)
            ]
            candidates = np.arange(int(receiver), dtype=np.int64)
            if len(candidates) < len(original):
                raise ValueError("not enough causal source nodes to rewire an edge")
            sources = rng.choice(candidates, size=len(original), replace=False).tolist()
            memberships.append([*sources, int(receiver)])
            edge_feature = trace.graph.edge_features[edge].copy()
            edge_feature[3] = np.mean(np.asarray(sources) < trace.prompt_tokens)
            features.append(edge_feature)
        return replace(
            trace,
            graph=_graph(
                trace.graph,
                memberships,
                trace.graph.receivers.tolist(),
                features,
                trace.graph.edge_kind.tolist(),
            ),
        )


class NoGeometryControl:
    name = "no_geometry"

    def __init__(self, *, geometry_columns: int = 6) -> None:
        if geometry_columns <= 0:
            raise ValueError("geometry_columns must be positive")
        self.geometry_columns = geometry_columns

    def apply(self, trace: CausalTrace) -> CausalTrace:
        features = trace.graph.node_features.copy()
        if features.shape[1] <= self.geometry_columns:
            raise ValueError("node features do not contain the geometry block")
        features[:, -(self.geometry_columns + 1) : -1] = 0.0
        graph = trace.graph
        return replace(
            trace,
            graph=CausalHypergraph(
                node_features=features,
                incidence=graph.incidence,
                receivers=graph.receivers,
                edge_features=graph.edge_features,
                edge_kind=graph.edge_kind,
                response_nodes=graph.response_nodes,
            ),
        )


class HiddenOnlyControl:
    name = "hidden_only"

    def apply(self, trace: CausalTrace) -> CausalTrace:
        return NoEdgeControl().apply(NoGeometryControl().apply(trace))
