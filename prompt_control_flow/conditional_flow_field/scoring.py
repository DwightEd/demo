from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import torch

from ..feasible_tangent.data import (
    TransitionData,
    problem_key,
    select_donors,
    trajectory_transitions,
)
from ..flow_signature_data import FlowTrajectoryDataset
from .data import build_field_supports, wrong_problem_order
from .schema import (
    CHAIN_SCORE_NAMES,
    TRANSITION_SCORE_NAMES,
    ConditionalFlowFieldConfig,
    ConditionalFlowFieldResult,
)


EPS = 1e-7


@dataclass(frozen=True)
class _FieldJob:
    sample: int
    transition: int
    layer: int
    phase: np.ndarray
    state: np.ndarray
    shuffle: np.ndarray
    wrong: tuple[np.ndarray, ...]
    target: np.ndarray
    state_alignment_changed: float


def _effective_device(value: str) -> torch.device:
    if str(value).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(str(value))


def _stable_seed(*values: int) -> int:
    seed = 2166136261
    for value in values:
        seed ^= int(value) & 0xFFFFFFFF
        seed = (seed * 16777619) & 0xFFFFFFFF
    return int(seed)


def _unit(values: torch.Tensor) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(EPS)


@torch.inference_mode()
def spherical_energy_score(
    targets: torch.Tensor,
    references: torch.Tensor,
    *,
    calibration_floor: float = 2e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an empirical spherical energy score and healthy-calibrated score.

    For chordal distance ``d`` on the unit sphere, the score is

        mean_j d(target, ref_j) - 0.5 mean_{j != k} d(ref_j, ref_k).

    The second output calibrates the target against leave-one-out scores of
    the healthy references. No bandwidth or correctness classifier is fitted.
    """

    if targets.ndim != 2 or references.ndim != 3:
        raise ValueError("expected targets [batch, hidden] and references [batch, donor, hidden]")
    if targets.shape[0] != references.shape[0] or targets.shape[1] != references.shape[2]:
        raise ValueError("target/reference shapes are incompatible")
    donor_count = int(references.shape[1])
    if donor_count < 3:
        raise ValueError("spherical energy calibration needs at least three donors")

    target = _unit(targets.float())
    refs = _unit(references.float())
    target_cosine = torch.einsum("bd,bkd->bk", target, refs).clamp(-1.0, 1.0)
    target_distance = torch.sqrt((2.0 - 2.0 * target_cosine).clamp_min(0.0))
    pair_cosine = torch.einsum("bjd,bkd->bjk", refs, refs).clamp(-1.0, 1.0)
    pair_distance = torch.sqrt((2.0 - 2.0 * pair_cosine).clamp_min(0.0))
    eye = torch.eye(donor_count, dtype=torch.bool, device=refs.device)[None]
    pair_distance = pair_distance.masked_fill(eye, 0.0)
    row_sum = pair_distance.sum(dim=-1)
    total_pair = row_sum.sum(dim=-1)
    pair_mean = total_pair / float(donor_count * (donor_count - 1))
    energy = target_distance.mean(dim=-1) - 0.5 * pair_mean

    self_first = row_sum / float(donor_count - 1)
    self_pair = (total_pair[:, None] - 2.0 * row_sum) / float(
        (donor_count - 1) * (donor_count - 2)
    )
    self_energy = self_first - 0.5 * self_pair
    center = torch.median(self_energy, dim=-1).values
    absolute = torch.abs(self_energy - center[:, None])
    mad = 1.4826 * torch.median(absolute, dim=-1).values
    std = torch.std(self_energy, dim=-1, unbiased=False)
    scale = torch.where(mad > float(calibration_floor), mad, std)
    scale = scale.clamp_min(float(calibration_floor))
    calibrated = (energy - center) / scale
    return energy, calibrated


def _aligned_reference_sets(
    target: TransitionData,
    donors: Sequence[int],
    cache: dict[int, TransitionData],
    *,
    transition: int,
    layer: int,
    state_window: int,
    seed_token: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    phase_rows = []
    state_rows = []
    shuffle_rows = []
    changed = []
    target_state = target.predecessor[transition, layer]
    for donor_position, donor_index in enumerate(donors):
        donor = cache[int(donor_index)]
        width = int(donor.direction.shape[0])
        phase_index = min(int(transition), width - 1)
        lo = max(0, phase_index - int(state_window))
        hi = min(width, phase_index + int(state_window) + 1)
        candidates = donor.predecessor[lo:hi, layer]
        cosine = candidates @ target_state
        state_index = lo + int(np.argmax(cosine))
        if width > 1:
            offset = 1 + _stable_seed(seed_token, donor_position, transition, layer) % (
                width - 1
            )
            shuffle_index = (phase_index + offset) % width
        else:
            shuffle_index = phase_index
        phase_rows.append(donor.direction[phase_index, layer])
        state_rows.append(donor.direction[state_index, layer])
        shuffle_rows.append(donor.direction[shuffle_index, layer])
        changed.append(float(state_index != phase_index))
    return (
        np.asarray(phase_rows, dtype=np.float32),
        np.asarray(state_rows, dtype=np.float32),
        np.asarray(shuffle_rows, dtype=np.float32),
        float(np.mean(changed)),
    )


def _phase_reference_set(
    target: TransitionData,
    donors: Sequence[int],
    cache: dict[int, TransitionData],
    *,
    transition: int,
    layer: int,
) -> np.ndarray:
    rows = []
    for donor_index in donors:
        donor = cache[int(donor_index)]
        index = min(int(transition), int(donor.direction.shape[0]) - 1)
        rows.append(donor.direction[index, layer])
    return np.asarray(rows, dtype=np.float32)


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _layer_mean(values: np.ndarray, reducer) -> float:
    summaries = []
    for layer in range(values.shape[1]):
        local = _finite(values[:, layer])
        if local.size:
            summaries.append(float(reducer(local)))
    return float(np.mean(summaries)) if summaries else float("nan")


def _free_energy(values: np.ndarray, beta: float) -> float:
    local = _finite(values)
    if not local.size:
        return float("nan")
    maximum = float(np.max(beta * local))
    return float((maximum + np.log(np.mean(np.exp(beta * local - maximum)))) / beta)


def _cusum(values: np.ndarray, drift: float) -> float:
    local = _finite(values)
    if not local.size:
        return float("nan")
    running = 0.0
    maximum = 0.0
    for value in local:
        running = max(0.0, running + float(value) - float(drift))
        maximum = max(maximum, running)
    return maximum / np.sqrt(float(local.size))


def _summarize_variant(
    energy: np.ndarray,
    calibrated: np.ndarray,
    cfg: ConditionalFlowFieldConfig,
) -> tuple[float, ...]:
    late_start = max(0, energy.shape[0] - int(np.ceil(energy.shape[0] * cfg.late_fraction)))
    return (
        _layer_mean(energy, np.mean),
        _layer_mean(energy[late_start:], np.mean),
        _layer_mean(calibrated, np.mean),
        _layer_mean(calibrated[late_start:], np.mean),
        _layer_mean(calibrated, lambda x: _free_energy(x, cfg.free_energy_beta)),
        _layer_mean(calibrated, lambda x: np.mean(np.maximum(x, 0.0))),
        _layer_mean(calibrated, lambda x: _cusum(x, cfg.cusum_drift)),
    )


def _summarize_chain(
    scores: np.ndarray,
    cfg: ConditionalFlowFieldConfig,
) -> np.ndarray:
    metric = {name: i for i, name in enumerate(TRANSITION_SCORE_NAMES)}
    values = []
    for variant in ("phase", "state", "shuffle", "wrong_problem"):
        values.extend(
            _summarize_variant(
                scores[:, :, metric[f"{variant}_energy"]],
                scores[:, :, metric[f"{variant}_calibrated"]],
                cfg,
            )
        )
    values.append(float(np.nanmedian(scores[:, :, metric["donor_count"]])))
    values.append(float(np.nanmean(scores[:, :, metric["state_alignment_changed"]])))
    return np.asarray(values, dtype=np.float32)


def _select_wrong_problems(
    target: Hashable,
    order: dict[Hashable, tuple[Hashable, ...]],
    supports,
    *,
    donor_count: int,
    draws: int,
) -> tuple[Hashable, ...]:
    selected = []
    for candidate in order.get(target, tuple()):
        if len(supports[candidate].correct_indices) < donor_count:
            continue
        selected.append(candidate)
        if len(selected) == int(draws):
            break
    return tuple(selected)


def run_conditional_flow_field(
    dataset: FlowTrajectoryDataset,
    cfg: ConditionalFlowFieldConfig,
) -> ConditionalFlowFieldResult:
    """Score a leave-one-response-out spherical feasible-flow field."""

    if cfg.min_donors < 3:
        raise ValueError("min_donors must be at least three")
    if cfg.max_donors < cfg.min_donors:
        raise ValueError("max_donors must be >= min_donors")
    if cfg.state_window < 0:
        raise ValueError("state_window must be non-negative")
    if cfg.wrong_problem_draws < 1:
        raise ValueError("wrong_problem_draws must be positive")
    if cfg.free_energy_beta <= 0:
        raise ValueError("free_energy_beta must be positive")

    device = _effective_device(cfg.device)
    supports = build_field_supports(dataset, cfg)
    wrong_order = wrong_problem_order(dataset, supports)
    metric = {name: i for i, name in enumerate(TRANSITION_SCORE_NAMES)}
    transition_scores = [
        np.full(
            (trajectory.shape[0] - 1, trajectory.shape[1], len(TRANSITION_SCORE_NAMES)),
            np.nan,
            dtype=np.float32,
        )
        for trajectory in dataset.trajectories
    ]
    chain_scores = np.full(
        (dataset.n_samples, len(CHAIN_SCORE_NAMES)), np.nan, dtype=np.float32
    )
    cache: dict[int, TransitionData] = {}
    skipped = {
        "problem_not_donor_eligible": 0,
        "insufficient_wrong_problem_controls": 0,
        "zero_target_direction": 0,
    }
    scored_jobs = 0
    donor_counts_by_label = {"correct": [], "error": []}

    def transition_for(index: int) -> TransitionData:
        if int(index) not in cache:
            cache[int(index)] = trajectory_transitions(dataset.trajectories[int(index)])
        return cache[int(index)]

    for raw_problem in np.unique(dataset.problem_ids):
        problem = problem_key(raw_problem)
        support = supports[problem]
        if support.donor_count < cfg.min_donors:
            skipped["problem_not_donor_eligible"] += len(support.sample_indices)
            continue
        jobs: list[_FieldJob] = []

        def flush() -> None:
            nonlocal jobs, scored_jobs
            if not jobs:
                return
            targets = torch.as_tensor(
                np.stack([job.target for job in jobs]), dtype=torch.float32, device=device
            )

            def score_refs(name: str) -> tuple[np.ndarray, np.ndarray]:
                refs = torch.as_tensor(
                    np.stack([getattr(job, name) for job in jobs]),
                    dtype=torch.float32,
                    device=device,
                )
                energy, calibrated = spherical_energy_score(
                    targets, refs, calibration_floor=cfg.calibration_floor
                )
                return (
                    energy.cpu().numpy().astype(np.float32),
                    calibrated.cpu().numpy().astype(np.float32),
                )

            phase_energy, phase_calibrated = score_refs("phase")
            state_energy, state_calibrated = score_refs("state")
            shuffle_energy, shuffle_calibrated = score_refs("shuffle")
            wrong_energy_draws = []
            wrong_calibrated_draws = []
            for draw in range(cfg.wrong_problem_draws):
                refs = torch.as_tensor(
                    np.stack([job.wrong[draw] for job in jobs]),
                    dtype=torch.float32,
                    device=device,
                )
                energy, calibrated = spherical_energy_score(
                    targets, refs, calibration_floor=cfg.calibration_floor
                )
                wrong_energy_draws.append(energy.cpu().numpy())
                wrong_calibrated_draws.append(calibrated.cpu().numpy())
            wrong_energy = np.mean(np.stack(wrong_energy_draws), axis=0)
            wrong_calibrated = np.mean(np.stack(wrong_calibrated_draws), axis=0)
            for row, job in enumerate(jobs):
                output = transition_scores[job.sample][job.transition, job.layer]
                output[metric["phase_energy"]] = phase_energy[row]
                output[metric["phase_calibrated"]] = phase_calibrated[row]
                output[metric["state_energy"]] = state_energy[row]
                output[metric["state_calibrated"]] = state_calibrated[row]
                output[metric["shuffle_energy"]] = shuffle_energy[row]
                output[metric["shuffle_calibrated"]] = shuffle_calibrated[row]
                output[metric["wrong_problem_energy"]] = wrong_energy[row]
                output[metric["wrong_problem_calibrated"]] = wrong_calibrated[row]
                output[metric["donor_count"]] = job.phase.shape[0]
                output[metric["state_alignment_changed"]] = job.state_alignment_changed
            scored_jobs += len(jobs)
            jobs = []

        wrong_problems = _select_wrong_problems(
            problem,
            wrong_order,
            supports,
            donor_count=support.donor_count,
            draws=cfg.wrong_problem_draws,
        )
        if len(wrong_problems) < cfg.wrong_problem_draws:
            skipped["insufficient_wrong_problem_controls"] += len(support.sample_indices)
            continue

        for target_index in support.sample_indices:
            target_index = int(target_index)
            donor_pool = [
                int(index)
                for index in support.correct_indices
                if int(index) != target_index
            ]
            donor_indices = select_donors(
                donor_pool,
                count=support.donor_count,
                seed=_stable_seed(cfg.random_seed, target_index, 11),
            )
            if len(donor_indices) != support.donor_count:
                raise RuntimeError("target-independent donor count invariant was violated")
            label_name = "error" if dataset.y_error[target_index] else "correct"
            donor_counts_by_label[label_name].append(len(donor_indices))
            wrong_donor_groups = tuple(
                select_donors(
                    supports[wrong_problem].correct_indices,
                    count=support.donor_count,
                    seed=_stable_seed(cfg.random_seed, target_index, draw, 23),
                )
                for draw, wrong_problem in enumerate(wrong_problems)
            )
            for index in donor_indices:
                transition_for(index)
            for group in wrong_donor_groups:
                for index in group:
                    transition_for(index)
            target = transition_for(target_index)
            for transition in range(target.direction.shape[0]):
                for layer in range(target.direction.shape[1]):
                    target_direction = target.direction[transition, layer]
                    if float(np.dot(target_direction, target_direction)) <= EPS:
                        skipped["zero_target_direction"] += 1
                        continue
                    phase, state, shuffle, changed = _aligned_reference_sets(
                        target,
                        donor_indices,
                        cache,
                        transition=transition,
                        layer=layer,
                        state_window=cfg.state_window,
                        seed_token=_stable_seed(cfg.random_seed, target_index),
                    )
                    wrong = tuple(
                        _phase_reference_set(
                            target,
                            group,
                            cache,
                            transition=transition,
                            layer=layer,
                        )
                        for group in wrong_donor_groups
                    )
                    jobs.append(
                        _FieldJob(
                            sample=target_index,
                            transition=transition,
                            layer=layer,
                            phase=phase,
                            state=state,
                            shuffle=shuffle,
                            wrong=wrong,
                            target=target_direction,
                            state_alignment_changed=changed,
                        )
                    )
                    if len(jobs) >= cfg.batch_size:
                        flush()
        flush()
        for target_index in support.sample_indices:
            target_index = int(target_index)
            if np.isfinite(transition_scores[target_index]).any():
                chain_scores[target_index] = _summarize_chain(
                    transition_scores[target_index], cfg
                )
        cache.clear()

    correct_counts = np.asarray(donor_counts_by_label["correct"], dtype=np.int64)
    error_counts = np.asarray(donor_counts_by_label["error"], dtype=np.int64)
    metadata = {
        "method": "conditional_spherical_feasible_flow_field",
        "geometry_only": True,
        "uses_labels_for_healthy_donors": True,
        "target_label_used_as_predictor": False,
        "donor_count_is_target_label_independent": True,
        "primary_alignment": "same_problem_causal_phase",
        "state_alignment": "same_problem_local_window_predecessor_cosine",
        "distance": "unit_sphere_chordal",
        "proper_score": "empirical_energy_score",
        "layers": dataset.layer_ids.tolist(),
        "min_donors": cfg.min_donors,
        "max_donors": cfg.max_donors,
        "state_window": cfg.state_window,
        "wrong_problem_draws": cfg.wrong_problem_draws,
        "free_energy_beta": cfg.free_energy_beta,
        "cusum_drift": cfg.cusum_drift,
        "effective_device": str(device),
        "scored_transition_layer_jobs": scored_jobs,
        "donor_count_correct_mean": float(np.mean(correct_counts)) if correct_counts.size else float("nan"),
        "donor_count_error_mean": float(np.mean(error_counts)) if error_counts.size else float("nan"),
        "skipped": skipped,
    }
    return ConditionalFlowFieldResult(
        dataset=dataset,
        transition_score_names=TRANSITION_SCORE_NAMES,
        transition_scores=transition_scores,
        chain_score_names=CHAIN_SCORE_NAMES,
        chain_scores=chain_scores,
        metadata=metadata,
    )
