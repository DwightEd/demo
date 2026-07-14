from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..flow_signature_data import FlowTrajectoryDataset


TRANSITION_SCORE_NAMES = (
    "primary_escape_raw",
    "primary_escape_supported",
    "phase_escape",
    "shuffle_escape",
    "wrong_problem_escape",
    "random_escape",
    "selected_rank",
    "captured_energy",
    "rank_supported",
    "state_match_cosine",
    "shuffle_changed_rate",
)


CHAIN_SCORE_NAMES = (
    "primary_escape_raw_mean",
    "primary_escape_mean",
    "primary_escape_late",
    "primary_escape_max",
    "primary_coherent_escape",
    "primary_late_coherent_escape",
    "primary_normal_persistence",
    "phase_escape_mean",
    "phase_coherent_escape",
    "shuffle_escape_mean",
    "shuffle_coherent_escape",
    "wrong_problem_escape_mean",
    "wrong_problem_coherent_escape",
    "random_escape_mean",
    "rank_support_rate",
    "mean_selected_rank",
    "state_match_cosine_mean",
    "shuffle_changed_rate_mean",
)


@dataclass(frozen=True)
class FeasibleTangentConfig:
    """Configuration for the geometry-only feasibility gate."""

    device: str = "cuda"
    batch_size: int = 32
    layer_batch_size: int = 2
    phase_sigma: float = 0.20
    causal_time_scale: float = 4.0
    rank_energy: float = 0.90
    max_rank: int = 4
    min_donors: int = 6
    max_donors: int = 12
    wrong_problem_draws: int = 3
    late_fraction: float = 1.0 / 3.0
    random_seed: int = 17


@dataclass
class FeasibleTangentResult:
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
