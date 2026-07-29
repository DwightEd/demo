from __future__ import annotations

import numpy as np

from .contracts import (
    ResidualWriteHypergraph,
    ResidualWriteTrace,
    _validated_finite_real,
    _validated_int,
)
from .geometry import summarize_write_group


class ResidualWriteHypergraphBuilder:
    """Build attention-selected, residual-write-attributed hyperedges."""

    def __init__(
        self,
        *,
        attention_threshold: float = 0.05,
        min_sources: int = 1,
        reconstruction_tolerance: float = 1e-6,
        evidence_role: int = 0,
    ) -> None:
        self.attention_threshold = _validated_finite_real(
            attention_threshold,
            name="attention_threshold",
            minimum=0.0,
        )
        self.min_sources = _validated_int(
            min_sources,
            name="min_sources",
            minimum=1,
        )
        self.reconstruction_tolerance = _validated_finite_real(
            reconstruction_tolerance,
            name="reconstruction_tolerance",
            strictly_positive=True,
        )
        self.evidence_role = _validated_int(
            evidence_role,
            name="evidence_role",
        )

    def build(self, trace: ResidualWriteTrace) -> ResidualWriteHypergraph:
        reconstructed = trace.source_writes.sum(axis=2)
        absolute_error = np.linalg.norm(
            reconstructed - trace.expected_updates,
            axis=-1,
        )
        expected_norm = np.linalg.norm(trace.expected_updates, axis=-1)
        allowed_error = self.reconstruction_tolerance * (1.0 + expected_norm)
        if np.any(absolute_error > allowed_error):
            worst = float(np.max(absolute_error / (1.0 + expected_norm)))
            raise ValueError(
                "source-write reconstruction gate failed "
                f"(max scaled reconstruction error={worst:.3e})"
            )

        memberships: list[tuple[int, int]] = []
        receivers: list[int] = []
        edge_features: list[list[float]] = []
        edge_heads: list[int] = []
        edge_queries: list[int] = []
        edge_kinds: list[str] = []
        heads, queries, sources = trace.attention.shape

        for head in range(heads):
            for query in range(queries):
                receiver = int(trace.receiver_positions[query])
                eligible = np.arange(sources) <= receiver
                selected = np.flatnonzero(
                    eligible
                    & (trace.attention[head, query] > self.attention_threshold)
                )
                selected = selected[selected != receiver]
                if selected.size < self.min_sources:
                    continue

                edge_id = len(receivers)
                memberships.extend((int(source), edge_id) for source in selected)
                memberships.append((receiver, edge_id))
                receivers.append(receiver)
                edge_heads.append(head)
                edge_queries.append(query)
                edge_kinds.append("write")
                edge_features.append(
                    self._features(
                        trace=trace,
                        head=head,
                        query=query,
                        selected=selected,
                    )
                )

        incidence = (
            np.asarray(memberships, dtype=np.int64).T
            if memberships
            else np.empty((2, 0), dtype=np.int64)
        )
        width = len(ResidualWriteHypergraph.edge_feature_names)
        features = np.asarray(edge_features, dtype=np.float64).reshape(-1, width)
        return ResidualWriteHypergraph(
            sample_id=trace.sample_id,
            view_name=trace.view_name,
            perturbation_kind=trace.perturbation_kind,
            perturbation_id=trace.perturbation_id,
            perturbation_seed=trace.perturbation_seed,
            perturbation_validated=trace.perturbation_validated,
            provenance=trace.provenance,
            num_heads=heads,
            node_features=trace.node_features,
            incidence=incidence,
            receivers=np.asarray(receivers, dtype=np.int64),
            edge_features=features,
            edge_heads=np.asarray(edge_heads, dtype=np.int64),
            edge_queries=np.asarray(edge_queries, dtype=np.int64),
            edge_kind=np.asarray(edge_kinds, dtype=str),
            response_nodes=trace.receiver_positions,
            response_token_ids=trace.response_token_ids,
            output_directions=trace.output_directions,
            source_roles=trace.source_roles,
            token_labels=trace.token_labels,
        )

    def _features(
        self,
        *,
        trace: ResidualWriteTrace,
        head: int,
        query: int,
        selected: np.ndarray,
    ) -> list[float]:
        attention = trace.attention[head, query, selected]
        writes = trace.source_writes[head, query, selected]
        geometry = summarize_write_group(writes)
        write_mass = geometry.write_mass
        resultant = writes.sum(axis=0)
        direction = trace.output_directions[query]
        direction_norm = float(np.linalg.norm(direction))
        signed_support = (
            float(resultant @ direction) / (write_mass * direction_norm)
            if write_mass > 0.0 and direction_norm > 0.0
            else 0.0
        )
        head_fraction = head / max(trace.attention.shape[0] - 1, 1)
        receiver = int(trace.receiver_positions[query])
        self_attention = float(trace.attention[head, query, receiver])
        self_write = trace.source_writes[head, query, receiver]
        self_write_mass = float(np.linalg.norm(self_write))
        self_signed_support = (
            float(self_write @ direction) / (self_write_mass * direction_norm)
            if self_write_mass > 0.0 and direction_norm > 0.0
            else 0.0
        )
        return [
            float(attention.mean()),
            float(attention.max()),
            float(head_fraction),
            float(selected.size / trace.attention.shape[2]),
            float(
                np.mean(trace.source_roles[selected] == self.evidence_role)
            ),
            write_mass,
            geometry.resultant_norm,
            geometry.cancellation,
            geometry.effective_sources,
            geometry.dominance,
            float(signed_support),
            self_attention,
            self_write_mass,
            float(self_signed_support),
        ]
