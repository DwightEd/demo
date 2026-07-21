from __future__ import annotations

import numpy as np

from .contracts import CausalHypergraph, ContributionMap, InterventionEffect


class CausalHypergraphBuilder:
    """Promote source sets to hyperedges only after non-additive intervention."""

    def __init__(self, *, min_effect: float = 0.0, min_synergy: float = 1e-6) -> None:
        if min_effect < 0.0 or min_synergy <= 0.0:
            raise ValueError("min_effect must be non-negative and min_synergy positive")
        self.min_effect = float(min_effect)
        self.min_synergy = float(min_synergy)

    def build(
        self,
        *,
        node_features: np.ndarray,
        contribution: ContributionMap,
        interventions: tuple[InterventionEffect, ...],
        response_nodes: np.ndarray,
    ) -> CausalHypergraph:
        nodes = np.asarray(node_features)
        if nodes.ndim != 2:
            raise ValueError("node_features must have shape [nodes, features]")

        members: list[list[int]] = []
        receivers: list[int] = []
        features: list[list[float]] = []
        kinds: list[str] = []
        for effect in interventions:
            if effect.query_index >= len(contribution.receiver_positions):
                raise ValueError("intervention query does not exist")
            receiver = int(contribution.receiver_positions[effect.query_index])
            if any(source < 0 or source >= len(nodes) for source in effect.sources):
                raise ValueError("intervention contains an invalid source")
            if receiver in effect.sources:
                raise ValueError("an intervention source cannot equal its receiver")

            nonlinear = (
                len(effect.sources) > 1
                and abs(effect.joint_effect) >= self.min_effect
                and abs(effect.synergy) >= self.min_synergy
            )
            if nonlinear:
                self._append(
                    members,
                    receivers,
                    features,
                    kinds,
                    sources=effect.sources,
                    receiver=receiver,
                    signed_effect=effect.joint_effect,
                    synergy=effect.synergy,
                    prompt_end=int(contribution.prompt_mask.sum()),
                    kind="hyper",
                )
                continue

            for source, singleton in zip(effect.sources, effect.singleton_effects):
                if abs(float(singleton)) < self.min_effect:
                    continue
                self._append(
                    members,
                    receivers,
                    features,
                    kinds,
                    sources=(source,),
                    receiver=receiver,
                    signed_effect=float(singleton),
                    synergy=0.0,
                    prompt_end=int(contribution.prompt_mask.sum()),
                    kind="pair",
                )

        incidence_columns = [
            (node, edge)
            for edge, edge_members in enumerate(members)
            for node in edge_members
        ]
        incidence = (
            np.asarray(incidence_columns, dtype=np.int64).T
            if incidence_columns
            else np.empty((2, 0), dtype=np.int64)
        )
        return CausalHypergraph(
            node_features=nodes,
            incidence=incidence,
            receivers=np.asarray(receivers, dtype=np.int64),
            edge_features=np.asarray(features, dtype=np.float64).reshape(-1, 4),
            edge_kind=np.asarray(kinds, dtype=str),
            response_nodes=np.asarray(response_nodes, dtype=np.int64),
        )

    @staticmethod
    def _append(
        members: list[list[int]],
        receivers: list[int],
        features: list[list[float]],
        kinds: list[str],
        *,
        sources: tuple[int, ...],
        receiver: int,
        signed_effect: float,
        synergy: float,
        prompt_end: int,
        kind: str,
    ) -> None:
        members.append([*sources, receiver])
        receivers.append(receiver)
        features.append(
            [
                signed_effect,
                abs(signed_effect),
                synergy,
                sum(source < prompt_end for source in sources) / len(sources),
            ]
        )
        kinds.append(kind)
