from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .finite_field import enumerate_vectors
from .geometry import (
    direct_query_distribution,
    fourier_coordinates,
    split_fourier,
    uniform_belief,
)
from .world import PredictiveAliasWorld, render_alias_prompt


@dataclass(frozen=True)
class AliasObservation:
    row_id: int
    pair_id: int
    branch: int
    query_role: str
    template_family: int
    modulus: int
    query_vector: np.ndarray
    exact_query_distribution: np.ndarray
    support_mask: np.ndarray
    base_support_mask: np.ndarray
    exact_belief: np.ndarray
    fourier_coordinates: np.ndarray
    update_fourier: np.ndarray
    user_text: str
    branch_evidence_text: str


def build_alias_observations(
    worlds: Sequence[PredictiveAliasWorld],
) -> tuple[list[AliasObservation], np.ndarray]:
    if not worlds:
        raise ValueError("at least one predictive-alias world is required")
    signatures = {
        (int(world.modulus), int(world.num_variables)) for world in worlds
    }
    if len(signatures) != 1:
        raise ValueError("all worlds in one artifact must share modulus and dimension")
    modulus, dimension = signatures.pop()
    pair_ids = [int(world.pair_id) for world in worlds]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair_id must be unique")
    frequencies = enumerate_vectors(modulus, dimension)
    rows: list[AliasObservation] = []
    for world in sorted(worlds, key=lambda item: item.pair_id):
        base_support = world.base_support_mask()
        base_phi = fourier_coordinates(
            world.assignments,
            base_support,
            frequencies,
            modulus,
        )
        for branch in (0, 1):
            support = world.support_mask(branch)
            belief = uniform_belief(support)
            branch_phi = fourier_coordinates(
                world.assignments,
                support,
                frequencies,
                modulus,
            )
            split_branch = split_fourier(branch_phi)
            split_update = split_fourier(branch_phi - base_phi)
            for query_role, query in (
                ("current", world.current_query),
                ("future", world.future_query),
            ):
                rendered = render_alias_prompt(world, branch, query_role)
                distribution = direct_query_distribution(
                    world.assignments,
                    support,
                    query,
                    modulus,
                )
                rows.append(
                    AliasObservation(
                        row_id=len(rows),
                        pair_id=int(world.pair_id),
                        branch=int(branch),
                        query_role=query_role,
                        template_family=int(world.template_family),
                        modulus=modulus,
                        query_vector=np.asarray(query, dtype=np.int64),
                        exact_query_distribution=distribution.astype(np.float64),
                        support_mask=support.copy(),
                        base_support_mask=base_support.copy(),
                        exact_belief=belief,
                        fourier_coordinates=split_branch,
                        update_fourier=split_update,
                        user_text=rendered.user_text,
                        branch_evidence_text=rendered.branch_evidence_text,
                    )
                )
    return rows, frequencies
