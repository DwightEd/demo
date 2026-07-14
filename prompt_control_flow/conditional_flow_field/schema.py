from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..flow_signature_data import FlowTrajectoryDataset


TRANSITION_SCORE_NAMES = (
    "phase_energy",
    "phase_calibrated",
    "state_energy",
    "state_calibrated",
    "shuffle_energy",
    "shuffle_calibrated",
    "wrong_problem_energy",
    "wrong_problem_calibrated",
    "donor_count",
    "state_alignment_changed",
)


CHAIN_SCORE_NAMES = tuple(
    f"{variant}_{summary}"
    for variant in ("phase", "state", "shuffle", "wrong_problem")
    for summary in (
        "energy_mean",
        "energy_late",
        "calibrated_mean",
        "calibrated_late",
        "calibrated_free_energy",
        "calibrated_positive_area",
        "calibrated_cusum",
    )
) + (
    "donor_count",
    "state_alignment_changed_rate",
)


@dataclass(frozen=True)
class ConditionalFlowFieldConfig:
    """Configuration for the nonparametric spherical feasible-flow field."""

    device: str = "cuda"
    batch_size: int = 32
    min_donors: int = 6
    max_donors: int = 11
    state_window: int = 2
    wrong_problem_draws: int = 3
    late_fraction: float = 1.0 / 3.0
    free_energy_beta: float = 2.0
    cusum_drift: float = 0.5
    calibration_floor: float = 2e-2
    random_seed: int = 17


@dataclass
class ConditionalFlowFieldResult:
    dataset: FlowTrajectoryDataset
    transition_score_names: tuple[str, ...]
    transition_scores: list[np.ndarray]
    chain_score_names: tuple[str, ...]
    chain_scores: np.ndarray
    metadata: dict[str, Any]

    def chain_score(self, name: str) -> np.ndarray:
        try:
            position = self.chain_score_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return np.asarray(self.chain_scores[:, position], dtype=np.float64)
