from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np
import torch

from ..conditional_flow_field.data import build_field_supports
from ..conditional_flow_field.schema import ConditionalFlowFieldConfig
from ..conditional_flow_field.scoring import spherical_energy_score
from ..feasible_tangent.data import (
    TransitionData,
    problem_key,
    select_donors,
    trajectory_transitions,
)
from ..flow_signature_data import FlowTrajectoryDataset
from .schema import CausalPullbackConfig, FieldWitnesses


EPS = 1e-7


def _stable_seed(*values: int) -> int:
    seed = 2166136261
    for value in values:
        seed ^= int(value) & 0xFFFFFFFF
        seed = (seed * 16777619) & 0xFFFFFFFF
    return int(seed)


def _unit(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= EPS:
        return np.zeros_like(array)
    return array / norm


def spherical_energy_witness(
    target: np.ndarray,
    references: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return the tangent gradient of the chordal energy score.

    The reference-spread correction in the energy score is constant with
    respect to the target.  The ambient gradient is therefore the average of
    unit vectors from each donor to the target.  Projection onto the tangent
    space of the unit sphere removes a radial scale change.
    """

    u = _unit(target)
    refs = np.asarray([_unit(row) for row in references], dtype=np.float32)
    difference = u[None, :] - refs
    distance = np.linalg.norm(difference, axis=1, keepdims=True)
    valid = distance[:, 0] > EPS
    if not np.any(valid):
        return np.zeros_like(u), 0.0
    ambient = np.mean(difference[valid] / distance[valid], axis=0)
    tangent = ambient - float(np.dot(ambient, u)) * u
    norm = float(np.linalg.norm(tangent))
    if not np.isfinite(norm) or norm <= EPS:
        return np.zeros_like(u), 0.0
    return np.asarray(tangent / norm, dtype=np.float32), norm


def _random_tangent(
    target: np.ndarray,
    primary: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    target = _unit(target)
    primary = _unit(primary)
    for _ in range(8):
        value = rng.normal(size=target.shape[0]).astype(np.float32)
        value -= float(np.dot(value, target)) * target
        if float(np.dot(primary, primary)) > EPS:
            value -= float(np.dot(value, primary)) * primary
        value = _unit(value)
        if float(np.dot(value, value)) > EPS:
            return value
    basis = np.zeros_like(target)
    basis[int(np.argmin(np.abs(target)))] = 1.0
    basis -= float(np.dot(basis, target)) * target
    return _unit(basis)


@dataclass
class ConditionalFieldBank:
    dataset: FlowTrajectoryDataset
    cfg: CausalPullbackConfig
    supports: dict[Hashable, object]
    transition_cache: dict[int, TransitionData] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        dataset: FlowTrajectoryDataset,
        cfg: CausalPullbackConfig,
    ) -> "ConditionalFieldBank":
        if len(dataset.layer_ids) != 1:
            raise ValueError(
                "causal pullback extraction currently requires exactly one selected layer"
            )
        if int(dataset.layer_ids[0]) != int(cfg.layer):
            raise ValueError(
                f"dataset selected layer {dataset.layer_ids.tolist()} does not match "
                f"configured layer {cfg.layer}"
            )
        field_cfg = ConditionalFlowFieldConfig(
            min_donors=cfg.min_donors,
            max_donors=cfg.max_donors,
            calibration_floor=cfg.calibration_floor,
            random_seed=cfg.random_seed,
        )
        return cls(
            dataset=dataset,
            cfg=cfg,
            supports=build_field_supports(dataset, field_cfg),
        )

    def transitions(self, index: int) -> TransitionData:
        index = int(index)
        if index not in self.transition_cache:
            self.transition_cache[index] = trajectory_transitions(
                self.dataset.trajectories[index]
            )
        return self.transition_cache[index]

    def eligible_target_indices(self) -> np.ndarray:
        """Return rows with enough same-problem correct reference donors."""

        eligible = []
        for index, raw_problem in enumerate(self.dataset.problem_ids):
            support = self.supports[problem_key(raw_problem)]
            if int(support.donor_count) >= int(self.cfg.min_donors):
                eligible.append(index)
        return np.asarray(eligible, dtype=np.int64)

    def _donors(self, target_index: int) -> tuple[int, ...]:
        problem = problem_key(self.dataset.problem_ids[int(target_index)])
        support = self.supports[problem]
        donor_count = int(support.donor_count)
        if donor_count < self.cfg.min_donors:
            return tuple()
        pool = [
            int(index)
            for index in support.correct_indices
            if int(index) != int(target_index)
        ]
        donors = select_donors(
            pool,
            count=donor_count,
            seed=_stable_seed(self.cfg.random_seed, target_index, 41),
        )
        if len(donors) != donor_count:
            raise RuntimeError("matched donor-count invariant was violated")
        return donors

    def witnesses(self, target_index: int) -> FieldWitnesses | None:
        target_index = int(target_index)
        donors = self._donors(target_index)
        if len(donors) < self.cfg.min_donors:
            return None
        target = self.transitions(target_index)
        donor_data = [self.transitions(index) for index in donors]
        transition_count = int(target.direction.shape[0])
        hidden_dim = int(target.direction.shape[-1])
        phase_refs = np.empty(
            (transition_count, len(donors), hidden_dim), dtype=np.float32
        )
        shuffle_refs = np.empty_like(phase_refs)
        for transition in range(transition_count):
            for donor_position, donor in enumerate(donor_data):
                width = int(donor.direction.shape[0])
                phase = min(transition, width - 1)
                if width > 1:
                    offset = 1 + _stable_seed(
                        self.cfg.random_seed,
                        target_index,
                        donor_position,
                        transition,
                    ) % (width - 1)
                    shuffled = (phase + offset) % width
                else:
                    shuffled = phase
                phase_refs[transition, donor_position] = donor.direction[phase, 0]
                shuffle_refs[transition, donor_position] = donor.direction[shuffled, 0]

        target_direction = np.asarray(target.direction[:, 0], dtype=np.float32)
        with torch.inference_mode():
            energy, calibrated = spherical_energy_score(
                torch.as_tensor(target_direction),
                torch.as_tensor(phase_refs),
                calibration_floor=self.cfg.calibration_floor,
            )
        field_direction = np.zeros_like(target_direction)
        shuffle_direction = np.zeros_like(target_direction)
        random_direction = np.zeros_like(target_direction)
        field_norm = np.zeros(transition_count, dtype=np.float32)
        shuffle_norm = np.zeros(transition_count, dtype=np.float32)
        for transition in range(transition_count):
            field_direction[transition], field_norm[transition] = spherical_energy_witness(
                target_direction[transition], phase_refs[transition]
            )
            shuffle_direction[transition], shuffle_norm[transition] = (
                spherical_energy_witness(
                    target_direction[transition], shuffle_refs[transition]
                )
            )
            random_direction[transition] = _random_tangent(
                target_direction[transition],
                field_direction[transition],
                seed=_stable_seed(
                    self.cfg.random_seed, target_index, transition, 97
                ),
            )
        result = FieldWitnesses(
            field_direction=field_direction,
            shuffle_direction=shuffle_direction,
            random_direction=random_direction,
            field_witness_norm=field_norm,
            shuffle_witness_norm=shuffle_norm,
            field_energy=energy.cpu().numpy().astype(np.float32),
            field_calibrated_energy=calibrated.cpu().numpy().astype(np.float32),
            donor_count=len(donors),
        )
        result.validate(hidden_dim)
        return result
