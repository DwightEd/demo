from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import CounterfactualExample, ResidualWriteHypergraph


@dataclass(frozen=True)
class PairedTokenSignatures:
    factual_state: np.ndarray
    factual_relation: np.ndarray
    context_state_delta: np.ndarray
    context_relation_delta: np.ndarray
    paraphrase_state_delta: np.ndarray
    paraphrase_relation_delta: np.ndarray

    def combined(self) -> np.ndarray:
        return np.concatenate(
            (
                self.factual_state,
                self.factual_relation,
                self.context_state_delta,
                self.context_relation_delta,
                self.paraphrase_state_delta,
                self.paraphrase_relation_delta,
            ),
            axis=1,
        )


def relational_token_signatures(graph: ResidualWriteHypergraph) -> np.ndarray:
    """Pool edge statistics by their explicit response-query identity."""

    edge_width = graph.edge_features.shape[1]
    signatures = np.zeros(
        (len(graph.response_nodes), 2 * edge_width + 2),
        dtype=np.float64,
    )
    for query in range(len(graph.response_nodes)):
        selected = graph.edge_queries == query
        edge_features = graph.edge_features[selected]
        if not len(edge_features):
            continue
        signatures[query, :edge_width] = edge_features.mean(axis=0)
        signatures[query, edge_width : 2 * edge_width] = edge_features.max(axis=0)
        signatures[query, -2] = float(len(edge_features))
        signatures[query, -1] = (
            float(len(np.unique(graph.edge_heads[selected]))) / graph.num_heads
        )
    return signatures


def paired_token_signatures(
    example: CounterfactualExample,
) -> PairedTokenSignatures:
    factual_state = example.factual.node_features[
        example.factual.response_nodes
    ]
    counterfactual_state = example.counterfactual.node_features[
        example.counterfactual.response_nodes
    ]
    if factual_state.shape != counterfactual_state.shape:
        raise ValueError("paired response-state signatures must align")
    factual_relation = relational_token_signatures(example.factual)
    counterfactual_relation = relational_token_signatures(example.counterfactual)
    if factual_relation.shape != counterfactual_relation.shape:
        raise ValueError("paired relational signatures must have the same shape")
    paraphrase_state_delta = np.zeros_like(factual_state)
    paraphrase_relation_delta = np.zeros_like(factual_relation)
    if example.paraphrase is not None:
        paraphrase_state = example.paraphrase.node_features[
            example.paraphrase.response_nodes
        ]
        if paraphrase_state.shape != factual_state.shape:
            raise ValueError("paraphrase response-state signatures must align")
        paraphrase_state_delta = np.abs(factual_state - paraphrase_state)
        paraphrase_relation = relational_token_signatures(example.paraphrase)
        if paraphrase_relation.shape != factual_relation.shape:
            raise ValueError("paraphrase relational signatures must align")
        paraphrase_relation_delta = np.abs(
            factual_relation - paraphrase_relation
        )
    return PairedTokenSignatures(
        factual_state=factual_state,
        factual_relation=factual_relation,
        context_state_delta=np.abs(factual_state - counterfactual_state),
        context_relation_delta=np.abs(
            factual_relation - counterfactual_relation
        ),
        paraphrase_state_delta=paraphrase_state_delta,
        paraphrase_relation_delta=paraphrase_relation_delta,
    )
