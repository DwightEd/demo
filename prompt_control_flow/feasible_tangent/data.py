from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

import numpy as np

from ..flow_signature_data import FlowTrajectoryDataset
from .schema import FeasibleTangentConfig


EPS = 1e-8


def problem_key(value: Any) -> Hashable:
    return value.item() if isinstance(value, np.generic) else value


@dataclass(frozen=True)
class TransitionData:
    predecessor: np.ndarray
    direction: np.ndarray
    phase: np.ndarray


@dataclass(frozen=True)
class ProblemSupport:
    problem_id: Hashable
    sample_indices: tuple[int, ...]
    correct_indices: tuple[int, ...]
    control: np.ndarray


def _unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return np.divide(
        array,
        np.maximum(norm, EPS),
        out=np.zeros_like(array),
        where=norm > EPS,
    )


def trajectory_transitions(trajectory: np.ndarray) -> TransitionData:
    """Return predecessor states and unit transition directions.

    The first prompt-to-step transition is intentionally absent because the
    multisample artifacts do not store a prompt anchor.  No synthetic origin
    is introduced.
    """

    states = np.asarray(trajectory, dtype=np.float32)
    if states.ndim != 3 or states.shape[0] < 2:
        raise ValueError(f"expected [step, layer, hidden] with >=2 steps, got {states.shape}")
    predecessor = _unit_rows(states[:-1])
    direction = _unit_rows(states[1:] - states[:-1])
    n_transition = int(direction.shape[0])
    # Causal step time does not reveal the eventual response length.
    phase = np.arange(n_transition, dtype=np.float32)
    return TransitionData(predecessor=predecessor, direction=direction, phase=phase)


def build_problem_supports(
    dataset: FlowTrajectoryDataset,
) -> dict[Hashable, ProblemSupport]:
    supports: dict[Hashable, ProblemSupport] = {}
    for raw_problem in np.unique(dataset.problem_ids):
        problem = problem_key(raw_problem)
        indices = np.where(dataset.problem_ids == raw_problem)[0]
        correct = indices[dataset.y_error[indices] == 0]
        control = np.asarray(
            [
                np.median(np.log1p(dataset.n_steps[indices])),
                np.median(np.log1p(dataset.response_chars[indices])),
            ],
            dtype=np.float64,
        )
        supports[problem] = ProblemSupport(
            problem_id=problem,
            sample_indices=tuple(int(x) for x in indices),
            correct_indices=tuple(int(x) for x in correct),
            control=control,
        )
    return supports


def wrong_problem_candidates(
    supports: dict[Hashable, ProblemSupport],
    *,
    minimum_correct: int,
) -> dict[Hashable, tuple[Hashable, ...]]:
    eligible = [
        support
        for support in supports.values()
        if len(support.correct_indices) >= int(minimum_correct)
    ]
    if not eligible:
        return {key: tuple() for key in supports}
    controls = np.stack([support.control for support in eligible])
    center = np.median(controls, axis=0)
    scale = 1.4826 * np.median(np.abs(controls - center), axis=0)
    fallback = np.std(controls, axis=0)
    scale = np.where(scale > EPS, scale, np.where(fallback > EPS, fallback, 1.0))
    standardized = (controls - center) / scale
    output: dict[Hashable, tuple[Hashable, ...]] = {}
    for target in supports.values():
        target_value = (target.control - center) / scale
        distance = np.linalg.norm(standardized - target_value[None, :], axis=1)
        order = np.argsort(distance, kind="stable")
        output[target.problem_id] = tuple(
            eligible[int(position)].problem_id
            for position in order
            if eligible[int(position)].problem_id != target.problem_id
        )
    return output


def select_donors(
    indices: Sequence[int],
    *,
    count: int,
    seed: int,
) -> tuple[int, ...]:
    values = np.asarray(sorted({int(x) for x in indices}), dtype=np.int64)
    if values.size <= int(count):
        return tuple(int(x) for x in values)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    chosen = np.sort(rng.choice(values, size=int(count), replace=False))
    return tuple(int(x) for x in chosen)


def feasible_tangent_preflight(
    dataset: FlowTrajectoryDataset,
    cfg: FeasibleTangentConfig,
) -> dict[str, Any]:
    supports = build_problem_supports(dataset)
    contrastive = 0
    donor_eligible = 0
    leave_one_out_eligible = 0
    for support in supports.values():
        indices = np.asarray(support.sample_indices, dtype=np.int64)
        labels = dataset.y_error[indices]
        contrastive += int(np.any(labels == 0) and np.any(labels == 1))
        donor_eligible += int(len(support.correct_indices) >= cfg.min_donors)
        leave_one_out_eligible += int(len(support.correct_indices) >= cfg.min_donors + 1)
    return {
        "path": dataset.source_path,
        "vector_key": dataset.vector_key,
        "samples": dataset.n_samples,
        "errors": int(dataset.y_error.sum()),
        "correct": int((dataset.y_error == 0).sum()),
        "problems": len(supports),
        "contrastive_problems": contrastive,
        "donor_eligible_problems": donor_eligible,
        "correct_leave_one_out_eligible_problems": leave_one_out_eligible,
        "layers": dataset.layer_ids.tolist(),
        "hidden_dim": dataset.hidden_dim,
        "min_donors": cfg.min_donors,
        "max_donors": cfg.max_donors,
        "rank_energy": cfg.rank_energy,
        "max_rank": cfg.max_rank,
        "causal_time_scale": cfg.causal_time_scale,
        "layer_batch_size": cfg.layer_batch_size,
        "skipped": dataset.skipped,
    }
