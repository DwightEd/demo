from __future__ import annotations

import numpy as np

from .contracts import ConstraintGeometry, ContributionMap, TransportInputs


class ConstraintBundleAnalyzer:
    """Measure output-relevant escape from prompt-origin transport directions."""

    def __init__(self, *, energy: float = 0.9, eps: float = 1e-8) -> None:
        if not 0.0 < energy <= 1.0:
            raise ValueError("energy must lie in (0, 1]")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.energy = float(energy)
        self.eps = float(eps)

    def analyze(
        self, inputs: TransportInputs, contribution: ContributionMap
    ) -> ConstraintGeometry:
        if contribution.signed.shape[0] != inputs.attention.shape[1]:
            raise ValueError("contribution and transport inputs do not align")

        queries = inputs.attention.shape[1]
        rank = np.zeros(queries, dtype=np.int64)
        transverse_fraction = np.zeros(queries, dtype=np.float64)
        transverse_escape = np.zeros(queries, dtype=np.float64)

        for query in range(queries):
            source_writes = inputs.source_writes[query, : inputs.prompt_end]
            basis, rank[query] = self._basis(source_writes)
            update = inputs.residual_updates[query]
            parallel = (
                basis @ (basis.T @ update) if basis.size else np.zeros_like(update)
            )
            transverse = update - parallel
            direction = inputs.output_directions[query]
            transverse_fraction[query] = np.linalg.norm(transverse) / (
                np.linalg.norm(update) + self.eps
            )
            transverse_escape[query] = abs(float(transverse @ direction)) / (
                np.linalg.norm(update) * np.linalg.norm(direction) + self.eps
            )

        prompt_support = contribution.prompt_fraction
        return ConstraintGeometry(
            prompt_support=prompt_support,
            response_support=1.0 - prompt_support,
            effective_update=np.einsum(
                "qd,qd->q",
                inputs.residual_updates,
                inputs.output_directions,
                optimize=True,
            ),
            transverse_fraction=transverse_fraction,
            transverse_escape=transverse_escape,
            tangent_rank=rank,
        )

    def _basis(self, points: np.ndarray) -> tuple[np.ndarray, int]:
        if not points.size or np.linalg.norm(points) <= self.eps:
            return np.empty((points.shape[1], 0), dtype=points.dtype), 0
        _, singular_values, right = np.linalg.svd(points, full_matrices=False)
        energy = singular_values**2
        total = float(energy.sum())
        if total <= self.eps:
            return np.empty((points.shape[1], 0), dtype=points.dtype), 0
        rank = int(np.searchsorted(np.cumsum(energy) / total, self.energy) + 1)
        return right[:rank].T, rank
