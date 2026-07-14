from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import torch

from ..flow_signature_data import FlowTrajectoryDataset
from .data import (
    ProblemSupport,
    TransitionData,
    build_problem_supports,
    problem_key,
    select_donors,
    trajectory_transitions,
    wrong_problem_candidates,
)
from .schema import (
    CHAIN_SCORE_NAMES,
    TRANSITION_SCORE_NAMES,
    FeasibleTangentConfig,
    FeasibleTangentResult,
)


EPS = 1e-8


@dataclass
class _TangentJob:
    sample: int
    transition: int
    layer: int
    target: np.ndarray
    primary: np.ndarray
    phase: np.ndarray
    shuffle: np.ndarray
    wrong: tuple[np.ndarray, ...]
    state_match_cosine: float
    shuffle_changed_rate: float


@dataclass
class _BasisScore:
    escape: np.ndarray
    residual: np.ndarray
    selected_rank: np.ndarray | None = None
    captured_energy: np.ndarray | None = None
    supported: np.ndarray | None = None


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


def _select_wrong_problems(
    target_problem: Hashable,
    candidates: dict[Hashable, tuple[Hashable, ...]],
    supports: dict[Hashable, ProblemSupport],
    *,
    donor_count: int,
    draws: int,
) -> tuple[Hashable, ...]:
    selected = []
    for candidate in candidates.get(target_problem, tuple()):
        if len(supports[candidate].correct_indices) < donor_count:
            continue
        selected.append(candidate)
        if len(selected) >= int(draws):
            break
    return tuple(selected)


@torch.inference_mode()
def _align_reference_grid(
    target: TransitionData,
    donor_indices: Sequence[int],
    transition_cache: dict[int, TransitionData],
    *,
    cfg: FeasibleTangentConfig,
    target_sample: int,
    device: torch.device,
    include_nulls: bool,
    layer_positions: Sequence[int],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    """Align all target transitions/layers to all donor transitions on device."""

    donors = [transition_cache[int(index)] for index in donor_indices]
    donor_count = len(donors)
    max_time = max(int(value.direction.shape[0]) for value in donors)
    selected_layers = np.asarray(layer_positions, dtype=np.int64)
    layer_count = int(selected_layers.size)
    if layer_count == 0:
        raise ValueError("layer_positions must not be empty")
    hidden = int(target.direction.shape[2])
    predecessor = torch.zeros(
        donor_count,
        max_time,
        layer_count,
        hidden,
        dtype=torch.float32,
        device=device,
    )
    direction = torch.zeros_like(predecessor)
    phase = torch.zeros(donor_count, max_time, dtype=torch.float32, device=device)
    valid = torch.zeros(donor_count, max_time, dtype=torch.bool, device=device)
    lengths = torch.empty(donor_count, dtype=torch.int64, device=device)
    for donor_position, donor in enumerate(donors):
        width = int(donor.direction.shape[0])
        predecessor[donor_position, :width] = torch.as_tensor(
            donor.predecessor[:, selected_layers, :],
            dtype=torch.float32,
            device=device,
        )
        direction[donor_position, :width] = torch.as_tensor(
            donor.direction[:, selected_layers, :],
            dtype=torch.float32,
            device=device,
        )
        phase[donor_position, :width] = torch.as_tensor(
            donor.phase, dtype=torch.float32, device=device
        )
        valid[donor_position, :width] = True
        lengths[donor_position] = width

    target_predecessor = torch.as_tensor(
        target.predecessor[:, selected_layers, :],
        dtype=torch.float32,
        device=device,
    )
    target_phase = torch.as_tensor(target.phase, dtype=torch.float32, device=device)
    cosine = torch.einsum("tld,ksld->tlks", target_predecessor, predecessor)
    phase_distance = (
        phase[None, None, :, :] - target_phase[:, None, None, None]
    ) / max(float(cfg.causal_time_scale), EPS)
    valid_grid = valid[None, None, :, :]
    aligned_score = cosine - 0.5 * torch.square(
        phase_distance / max(float(cfg.phase_sigma), EPS)
    )
    aligned_score = aligned_score.masked_fill(~valid_grid, -torch.inf)
    primary_index = torch.argmax(aligned_score, dim=-1)

    layer_index = torch.arange(layer_count, device=device)[:, None, None]
    layer_index = layer_index.expand(
        layer_count, target.direction.shape[0], donor_count
    )
    donor_index = torch.arange(donor_count, device=device)[None, None, :]
    donor_index = donor_index.expand(
        layer_count, target.direction.shape[0], donor_count
    )

    def gather(selected: torch.Tensor) -> torch.Tensor:
        selected_ltk = selected.permute(1, 0, 2)
        values = direction.permute(2, 0, 1, 3)[
            layer_index,
            donor_index,
            selected_ltk,
        ]
        return values.permute(1, 0, 2, 3)

    primary = gather(primary_index)
    selected_cosine = torch.gather(
        cosine, dim=-1, index=primary_index[..., None]
    ).squeeze(-1)

    if include_nulls:
        phase_cost = torch.abs(phase_distance).expand(
            target.direction.shape[0], layer_count, donor_count, max_time
        )
        phase_cost = phase_cost.masked_fill(~valid_grid, torch.inf)
        phase_index = torch.argmin(phase_cost, dim=-1)
        phase_reference = gather(phase_index)

        time_index = torch.arange(target.direction.shape[0], device=device)[:, None, None]
        layer_grid = torch.arange(layer_count, device=device)[None, :, None]
        donor_grid = torch.arange(donor_count, device=device)[None, None, :]
        token = (
            int(cfg.random_seed) * 1000003
            + int(target_sample) * 9176
            + time_index * 131
            + layer_grid * 29
            + donor_grid * 17
        )
        movable = lengths[None, None, :] > 1
        offset = torch.where(
            movable,
            1 + torch.remainder(token, (lengths[None, None, :] - 1).clamp_min(1)),
            torch.zeros_like(token),
        )
        shuffle_index = torch.remainder(
            primary_index + offset,
            lengths[None, None, :],
        )
        shuffle_reference = gather(shuffle_index)
        changed = torch.mean((shuffle_index != primary_index).to(torch.float32), dim=-1)
    else:
        phase_reference = None
        shuffle_reference = None
        changed = torch.full(
            selected_cosine.shape,
            float("nan"),
            dtype=torch.float32,
            device=device,
        )

    return (
        primary.cpu().numpy().astype(np.float32),
        None
        if phase_reference is None
        else phase_reference.cpu().numpy().astype(np.float32),
        None
        if shuffle_reference is None
        else shuffle_reference.cpu().numpy().astype(np.float32),
        torch.mean(selected_cosine, dim=-1).cpu().numpy().astype(np.float32),
        changed.cpu().numpy().astype(np.float32),
    )


def _pad_references(
    references: Sequence[np.ndarray],
    *,
    device: torch.device,
) -> torch.Tensor:
    max_donors = max(int(value.shape[0]) for value in references)
    hidden = int(references[0].shape[1])
    output = torch.zeros(
        (len(references), max_donors, hidden),
        dtype=torch.float32,
        device=device,
    )
    for row, value in enumerate(references):
        output[row, : value.shape[0]] = torch.as_tensor(
            value,
            dtype=torch.float32,
            device=device,
        )
    return output


@torch.inference_mode()
def _fit_primary(
    references: Sequence[np.ndarray],
    targets: np.ndarray,
    cfg: FeasibleTangentConfig,
    *,
    device: torch.device,
) -> _BasisScore:
    refs = _pad_references(references, device=device)
    target = torch.as_tensor(targets, dtype=torch.float32, device=device)
    gram = refs @ refs.transpose(1, 2)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = torch.flip(eigenvalues.clamp_min(0.0), dims=(1,))
    eigenvectors = torch.flip(eigenvectors, dims=(2,))
    max_rank = min(int(cfg.max_rank), int(refs.shape[1]), int(refs.shape[2]))
    values = eigenvalues[:, :max_rank]
    vectors = eigenvectors[:, :, :max_rank]
    total = eigenvalues.sum(dim=1).clamp_min(EPS)
    cumulative = torch.cumsum(values, dim=1) / total[:, None]
    reaches = cumulative >= float(cfg.rank_energy)
    first = torch.argmax(reaches.to(torch.int64), dim=1) + 1
    supported = torch.any(reaches, dim=1)
    selected_rank = torch.where(
        supported,
        first,
        torch.full_like(first, max_rank),
    )
    captured = cumulative.gather(1, (selected_rank - 1)[:, None]).squeeze(1)
    floor = torch.maximum(
        eigenvalues[:, :1] * 1e-6,
        torch.full_like(eigenvalues[:, :1], EPS),
    )
    valid_mode = values > floor
    basis = refs.transpose(1, 2) @ vectors
    basis = basis / torch.sqrt(values.clamp_min(EPS))[:, None, :]
    rank_mask = (
        torch.arange(max_rank, device=device)[None, :] < selected_rank[:, None]
    ) & valid_mode
    basis = basis * rank_mask[:, None, :]
    coefficient = torch.einsum("bdr,bd->br", basis, target)
    projection = torch.einsum("bdr,br->bd", basis, coefficient)
    residual = target - projection
    escape = torch.sum(residual.square(), dim=1).clamp(0.0, 1.25)
    return _BasisScore(
        escape=escape.cpu().numpy().astype(np.float32),
        residual=residual.cpu().numpy().astype(np.float32),
        selected_rank=selected_rank.cpu().numpy().astype(np.int64),
        captured_energy=captured.cpu().numpy().astype(np.float32),
        supported=supported.cpu().numpy().astype(bool),
    )


@torch.inference_mode()
def _fit_matched_rank(
    references: Sequence[np.ndarray],
    targets: np.ndarray,
    ranks: np.ndarray,
    *,
    device: torch.device,
) -> _BasisScore:
    refs = _pad_references(references, device=device)
    target = torch.as_tensor(targets, dtype=torch.float32, device=device)
    rank = torch.as_tensor(ranks, dtype=torch.int64, device=device)
    gram = refs @ refs.transpose(1, 2)
    values, vectors = torch.linalg.eigh(gram)
    values = torch.flip(values.clamp_min(0.0), dims=(1,))
    vectors = torch.flip(vectors, dims=(2,))
    max_rank = min(int(np.max(ranks)), int(refs.shape[1]), int(refs.shape[2]))
    values = values[:, :max_rank]
    vectors = vectors[:, :, :max_rank]
    floor = torch.maximum(values[:, :1] * 1e-6, torch.full_like(values[:, :1], EPS))
    valid_mode = values > floor
    basis = refs.transpose(1, 2) @ vectors
    basis = basis / torch.sqrt(values.clamp_min(EPS))[:, None, :]
    rank_mask = (
        torch.arange(max_rank, device=device)[None, :] < rank[:, None]
    ) & valid_mode
    basis = basis * rank_mask[:, None, :]
    coefficient = torch.einsum("bdr,bd->br", basis, target)
    projection = torch.einsum("bdr,br->bd", basis, coefficient)
    residual = target - projection
    escape = torch.sum(residual.square(), dim=1).clamp(0.0, 1.25)
    return _BasisScore(
        escape=escape.cpu().numpy().astype(np.float32),
        residual=residual.cpu().numpy().astype(np.float32),
    )


@torch.inference_mode()
def _score_random_subspace(
    targets: np.ndarray,
    ranks: np.ndarray,
    job_tokens: np.ndarray,
    cfg: FeasibleTangentConfig,
    *,
    device: torch.device,
) -> _BasisScore:
    target = torch.as_tensor(targets, dtype=torch.float32, device=device)
    rank = torch.as_tensor(ranks, dtype=torch.int64, device=device)
    max_rank = int(np.max(ranks))
    random_rows = []
    for token in np.asarray(job_tokens, dtype=np.int64):
        generator = torch.Generator(device=device)
        generator.manual_seed(_stable_seed(cfg.random_seed, int(token), 104729))
        random_rows.append(
            torch.randn(
                target.shape[1],
                max_rank,
                generator=generator,
                dtype=torch.float32,
                device=device,
            )
        )
    random = torch.stack(random_rows)
    basis, _ = torch.linalg.qr(random, mode="reduced")
    mask = torch.arange(max_rank, device=device)[None, :] < rank[:, None]
    basis = basis * mask[:, None, :]
    coefficient = torch.einsum("bdr,bd->br", basis, target)
    projection = torch.einsum("bdr,br->bd", basis, coefficient)
    residual = target - projection
    escape = torch.sum(residual.square(), dim=1).clamp(0.0, 1.25)
    return _BasisScore(
        escape=escape.cpu().numpy().astype(np.float32),
        residual=residual.cpu().numpy().astype(np.float32),
    )


def _residual_summary(
    residual: np.ndarray,
    support: np.ndarray,
    *,
    late_fraction: float,
) -> tuple[float, float, float]:
    values = np.asarray(residual, dtype=np.float32)
    support_mask = np.asarray(support, dtype=bool)
    coherent_by_layer = []
    late_coherent_by_layer = []
    persistence_by_layer = []
    for layer in range(values.shape[1]):
        valid = support_mask[:, layer] & np.isfinite(values[:, layer]).all(axis=1)
        local = values[valid, layer]
        if local.size == 0:
            continue
        mean = np.mean(local, axis=0)
        coherent_by_layer.append(float(np.dot(mean, mean)))
        energy = float(np.sum(local * local))
        summed = np.sum(local, axis=0)
        persistence_by_layer.append(
            float(np.dot(summed, summed) / max(local.shape[0] * energy, EPS))
        )
        late_start = max(0, values.shape[0] - int(np.ceil(values.shape[0] * late_fraction)))
        late_valid = valid[late_start:]
        late = values[late_start:, layer][late_valid]
        if late.size:
            late_mean = np.mean(late, axis=0)
            late_coherent_by_layer.append(float(np.dot(late_mean, late_mean)))
    return (
        float(np.mean(coherent_by_layer)) if coherent_by_layer else float("nan"),
        float(np.mean(late_coherent_by_layer)) if late_coherent_by_layer else float("nan"),
        float(np.mean(persistence_by_layer)) if persistence_by_layer else float("nan"),
    )


def _finite_stat(values: np.ndarray, mode: str) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan")
    if mode == "mean":
        return float(np.mean(finite))
    if mode == "max":
        return float(np.max(finite))
    raise ValueError(mode)


def _summarize_chain(
    scores: np.ndarray,
    residuals: dict[str, np.ndarray],
    cfg: FeasibleTangentConfig,
) -> np.ndarray:
    metric = {name: i for i, name in enumerate(TRANSITION_SCORE_NAMES)}
    output = np.full(len(CHAIN_SCORE_NAMES), np.nan, dtype=np.float32)
    position = {name: i for i, name in enumerate(CHAIN_SCORE_NAMES)}
    supported = scores[:, :, metric["rank_supported"]] > 0.5
    n_transition = int(scores.shape[0])
    late_start = max(0, n_transition - int(np.ceil(n_transition * cfg.late_fraction)))

    primary_raw = scores[:, :, metric["primary_escape_raw"]]
    primary = np.where(supported, scores[:, :, metric["primary_escape_supported"]], np.nan)
    output[position["primary_escape_raw_mean"]] = _finite_stat(primary_raw, "mean")
    output[position["primary_escape_mean"]] = _finite_stat(primary, "mean")
    output[position["primary_escape_late"]] = _finite_stat(primary[late_start:], "mean")
    output[position["primary_escape_max"]] = _finite_stat(primary, "max")

    primary_coherent, primary_late, primary_persistence = _residual_summary(
        residuals["primary"], supported, late_fraction=cfg.late_fraction
    )
    output[position["primary_coherent_escape"]] = primary_coherent
    output[position["primary_late_coherent_escape"]] = primary_late
    output[position["primary_normal_persistence"]] = primary_persistence

    for variant in ("phase", "shuffle", "wrong_problem"):
        score_name = f"{variant}_escape"
        chain_prefix = variant
        values = np.where(supported, scores[:, :, metric[score_name]], np.nan)
        output[position[f"{chain_prefix}_escape_mean"]] = _finite_stat(values, "mean")
        coherent, _, _ = _residual_summary(
            residuals[variant], supported, late_fraction=cfg.late_fraction
        )
        output[position[f"{chain_prefix}_coherent_escape"]] = coherent
    random_values = np.where(
        supported,
        scores[:, :, metric["random_escape"]],
        np.nan,
    )
    output[position["random_escape_mean"]] = _finite_stat(random_values, "mean")
    output[position["rank_support_rate"]] = float(np.mean(supported))
    output[position["mean_selected_rank"]] = _finite_stat(
        scores[:, :, metric["selected_rank"]], "mean"
    )
    output[position["state_match_cosine_mean"]] = _finite_stat(
        scores[:, :, metric["state_match_cosine"]], "mean"
    )
    output[position["shuffle_changed_rate_mean"]] = _finite_stat(
        scores[:, :, metric["shuffle_changed_rate"]], "mean"
    )
    return output


def run_feasible_tangent_gate(
    dataset: FlowTrajectoryDataset,
    cfg: FeasibleTangentConfig,
) -> FeasibleTangentResult:
    """Fit leave-one-response-out same-problem tangents and structural nulls."""

    if cfg.min_donors < cfg.max_rank + 2:
        raise ValueError("min_donors must be at least max_rank + 2")
    if cfg.max_donors < cfg.min_donors:
        raise ValueError("max_donors must be >= min_donors")
    if not 0.0 < cfg.rank_energy < 1.0:
        raise ValueError("rank_energy must be in (0,1)")
    if cfg.wrong_problem_draws < 1:
        raise ValueError("wrong_problem_draws must be positive")
    if cfg.causal_time_scale <= 0:
        raise ValueError("causal_time_scale must be positive")
    if cfg.layer_batch_size < 1:
        raise ValueError("layer_batch_size must be positive")

    device = _effective_device(cfg.device)
    supports = build_problem_supports(dataset)
    wrong_candidates = wrong_problem_candidates(supports, minimum_correct=cfg.min_donors)
    metric_index = {name: i for i, name in enumerate(TRANSITION_SCORE_NAMES)}
    transition_scores = [
        np.full(
            (trajectory.shape[0] - 1, trajectory.shape[1], len(TRANSITION_SCORE_NAMES)),
            np.nan,
            dtype=np.float32,
        )
        for trajectory in dataset.trajectories
    ]
    chain_scores = np.full(
        (dataset.n_samples, len(CHAIN_SCORE_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    skipped = {
        "insufficient_same_problem_donors": 0,
        "insufficient_wrong_problem_controls": 0,
        "zero_target_direction": 0,
    }
    scored_jobs = 0
    supported_jobs = 0

    transition_cache: dict[int, TransitionData] = {}

    def transition_for(index: int) -> TransitionData:
        if index not in transition_cache:
            transition_cache[index] = trajectory_transitions(dataset.trajectories[index])
        return transition_cache[index]

    for raw_problem in np.unique(dataset.problem_ids):
        problem = problem_key(raw_problem)
        support = supports[problem]
        target_indices = support.sample_indices
        problem_residuals: dict[int, dict[str, np.ndarray]] = {}
        jobs: list[_TangentJob] = []

        def flush() -> None:
            nonlocal jobs, scored_jobs, supported_jobs
            if not jobs:
                return
            target = np.stack([job.target for job in jobs]).astype(np.float32, copy=False)
            primary = _fit_primary(
                [job.primary for job in jobs], target, cfg, device=device
            )
            ranks = np.asarray(primary.selected_rank, dtype=np.int64)
            phase = _fit_matched_rank(
                [job.phase for job in jobs], target, ranks, device=device
            )
            shuffle = _fit_matched_rank(
                [job.shuffle for job in jobs], target, ranks, device=device
            )
            wrong_scores = []
            for draw in range(cfg.wrong_problem_draws):
                wrong_scores.append(
                    _fit_matched_rank(
                        [job.wrong[draw] for job in jobs],
                        target,
                        ranks,
                        device=device,
                    )
                )
            wrong_escape = np.mean(
                np.stack([value.escape for value in wrong_scores]), axis=0
            )
            wrong_residual = np.mean(
                np.stack([value.residual for value in wrong_scores]), axis=0
            )
            random = _score_random_subspace(
                target,
                ranks,
                np.asarray(
                    [
                        _stable_seed(job.sample, job.transition, job.layer)
                        for job in jobs
                    ],
                    dtype=np.int64,
                ),
                cfg,
                device=device,
            )
            for row, job in enumerate(jobs):
                score = transition_scores[job.sample][job.transition, job.layer]
                score[metric_index["primary_escape_raw"]] = primary.escape[row]
                if bool(primary.supported[row]):
                    score[metric_index["primary_escape_supported"]] = primary.escape[row]
                    supported_jobs += 1
                score[metric_index["phase_escape"]] = phase.escape[row]
                score[metric_index["shuffle_escape"]] = shuffle.escape[row]
                score[metric_index["wrong_problem_escape"]] = wrong_escape[row]
                score[metric_index["random_escape"]] = random.escape[row]
                score[metric_index["selected_rank"]] = primary.selected_rank[row]
                score[metric_index["captured_energy"]] = primary.captured_energy[row]
                score[metric_index["rank_supported"]] = float(primary.supported[row])
                score[metric_index["state_match_cosine"]] = job.state_match_cosine
                score[metric_index["shuffle_changed_rate"]] = job.shuffle_changed_rate
                buffers = problem_residuals[job.sample]
                buffers["primary"][job.transition, job.layer] = primary.residual[row]
                buffers["phase"][job.transition, job.layer] = phase.residual[row]
                buffers["shuffle"][job.transition, job.layer] = shuffle.residual[row]
                buffers["wrong_problem"][job.transition, job.layer] = wrong_residual[row]
            scored_jobs += len(jobs)
            jobs = []

        for target_index in target_indices:
            target_index = int(target_index)
            donor_pool = [
                index
                for index in support.correct_indices
                if int(index) != target_index
            ]
            if len(donor_pool) < cfg.min_donors:
                skipped["insufficient_same_problem_donors"] += 1
                continue
            donor_indices = select_donors(
                donor_pool,
                count=cfg.max_donors,
                seed=_stable_seed(cfg.random_seed, target_index, 11),
            )
            wrong_problems = _select_wrong_problems(
                problem,
                wrong_candidates,
                supports,
                donor_count=len(donor_indices),
                draws=cfg.wrong_problem_draws,
            )
            if len(wrong_problems) < cfg.wrong_problem_draws:
                skipped["insufficient_wrong_problem_controls"] += 1
                continue
            wrong_donors = []
            for draw, wrong_problem in enumerate(wrong_problems):
                wrong_donors.append(
                    select_donors(
                        supports[wrong_problem].correct_indices,
                        count=len(donor_indices),
                        seed=_stable_seed(cfg.random_seed, target_index, draw, 23),
                    )
                )
            for donor_index in donor_indices:
                transition_for(int(donor_index))
            for donor_group in wrong_donors:
                for donor_index in donor_group:
                    transition_for(int(donor_index))
            target_data = transition_for(target_index)
            n_transition, n_layer, hidden = target_data.direction.shape
            problem_residuals[target_index] = {
                name: np.full(
                    (n_transition, n_layer, hidden),
                    np.nan,
                    dtype=np.float32,
                )
                for name in ("primary", "phase", "shuffle", "wrong_problem")
            }
            for layer_start in range(0, n_layer, cfg.layer_batch_size):
                layer_positions = tuple(
                    range(
                        layer_start,
                        min(n_layer, layer_start + cfg.layer_batch_size),
                    )
                )
                (
                    primary_grid,
                    phase_grid,
                    shuffle_grid,
                    match_cosine_grid,
                    changed_rate_grid,
                ) = _align_reference_grid(
                    target_data,
                    donor_indices,
                    transition_cache,
                    cfg=cfg,
                    target_sample=target_index,
                    device=device,
                    include_nulls=True,
                    layer_positions=layer_positions,
                )
                wrong_grids = tuple(
                    _align_reference_grid(
                        target_data,
                        donor_group,
                        transition_cache,
                        cfg=cfg,
                        target_sample=target_index,
                        device=device,
                        include_nulls=False,
                        layer_positions=layer_positions,
                    )[0]
                    for donor_group in wrong_donors
                )
                for transition in range(n_transition):
                    for local_layer, layer in enumerate(layer_positions):
                        target_direction = target_data.direction[transition, layer]
                        if float(np.dot(target_direction, target_direction)) <= EPS:
                            skipped["zero_target_direction"] += 1
                            continue
                        jobs.append(
                            _TangentJob(
                                sample=target_index,
                                transition=transition,
                                layer=layer,
                                target=target_direction,
                                primary=primary_grid[transition, local_layer],
                                phase=phase_grid[transition, local_layer],
                                shuffle=shuffle_grid[transition, local_layer],
                                wrong=tuple(
                                    grid[transition, local_layer]
                                    for grid in wrong_grids
                                ),
                                state_match_cosine=float(
                                    match_cosine_grid[transition, local_layer]
                                ),
                                shuffle_changed_rate=float(
                                    changed_rate_grid[transition, local_layer]
                                ),
                            )
                        )
                        if len(jobs) >= cfg.batch_size:
                            flush()
        flush()
        for target_index, residuals in problem_residuals.items():
            chain_scores[target_index] = _summarize_chain(
                transition_scores[target_index], residuals, cfg
            )
        transition_cache.clear()

    metadata = {
        "method": "same_problem_feasible_tangent_gate",
        "geometry_only": True,
        "uses_labels_for_healthy_donors": True,
        "target_label_used_in_scoring": False,
        "prompt_transition_included": False,
        "alignment": "predecessor_state_cosine_plus_causal_step_kernel",
        "phase_uses_final_response_length": False,
        "causal_time_scale": cfg.causal_time_scale,
        "adaptive_rank": True,
        "rank_energy": cfg.rank_energy,
        "max_rank": cfg.max_rank,
        "min_donors": cfg.min_donors,
        "max_donors": cfg.max_donors,
        "layer_batch_size": cfg.layer_batch_size,
        "wrong_problem_draws": cfg.wrong_problem_draws,
        "effective_device": str(device),
        "scored_transition_layer_jobs": scored_jobs,
        "rank_supported_jobs": supported_jobs,
        "rank_supported_fraction": (
            supported_jobs / scored_jobs if scored_jobs else float("nan")
        ),
        "skipped": skipped,
    }
    return FeasibleTangentResult(
        dataset=dataset,
        transition_score_names=TRANSITION_SCORE_NAMES,
        transition_scores=transition_scores,
        chain_score_names=CHAIN_SCORE_NAMES,
        chain_scores=chain_scores,
        metadata=metadata,
    )
