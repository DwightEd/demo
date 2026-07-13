from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .evaluate import finite_json
from .flow_signature_audit import (
    FlowAuditConfig,
    crossfit_residualize_score,
    evaluate_score,
)
from .flow_signature_data import (
    FlowTrajectoryDataset,
    load_flow_trajectory_dataset,
    parse_layer_selection,
)


@dataclass
class DirectionalCloudDataset:
    base: FlowTrajectoryDataset
    clouds: list[np.ndarray]
    step_sizes: list[np.ndarray]
    cloud_layer_ids: np.ndarray
    cloud_hidden_dim: int
    response_tokens: np.ndarray
    skipped_clouds: dict[str, int]

    @property
    def n_samples(self) -> int:
        return len(self.clouds)


@dataclass(frozen=True)
class DirectionalConsensusConfig:
    late_fraction: float = 1.0 / 3.0
    fixed_window_tokens: int = 16
    batch_size: int = 64
    max_batch_tokens: int = 8192
    compute_device: str = "cuda"

    def validate(self) -> None:
        if not 0.0 < self.late_fraction <= 1.0:
            raise ValueError("late_fraction must lie in (0, 1]")
        if self.fixed_window_tokens < 2:
            raise ValueError("fixed_window_tokens must be at least 2")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")


@dataclass(frozen=True)
class DirectionalConsensusAuditConfig:
    folds: int = 5
    bootstrap: int = 1000
    permutations: int = 500
    length_match_ratio: float = 1.25
    seed: int = 13

    def validate(self) -> None:
        if self.folds < 2:
            raise ValueError("folds must be at least 2")
        if self.bootstrap < 0 or self.permutations < 0:
            raise ValueError("bootstrap and permutations must be non-negative")
        if self.length_match_ratio < 1.0:
            raise ValueError("length_match_ratio must be at least 1")


@dataclass
class DirectionalConsensusEncoding:
    step_resultant: list[np.ndarray]
    step_raw_spread: list[np.ndarray]
    step_pairwise_concentration: list[np.ndarray]
    step_debiased_dispersion: list[np.ndarray]
    chain_scores: dict[str, np.ndarray]
    runtime_seconds: float


def _cloud_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        array = np.asarray(value.tolist())
    if array.ndim == 2:
        array = array[:, None, :]
    if array.ndim != 3:
        raise ValueError(f"expected cloud [token, layer, hidden], got {array.shape}")
    return np.asarray(array)


def _resolve_cloud_layer_ids(z: np.lib.npyio.NpzFile, depth: int) -> tuple[str, np.ndarray]:
    candidates = ("cloud_layers", "cloud_store_layers", "hidden_layers")
    present: list[tuple[str, int]] = []
    for key in candidates:
        if key not in z.files:
            continue
        values = np.asarray(z[key]).reshape(-1)
        present.append((key, int(values.size)))
        if values.size == depth:
            return key, values.astype(np.int64)
    if present:
        raise ValueError(
            f"token clouds store {depth} layers, but cloud metadata sizes are {present}; "
            "refusing to guess the layer mapping"
        )
    return "implicit", np.arange(depth, dtype=np.int64)


def _subset_base(dataset: FlowTrajectoryDataset, positions: Sequence[int]) -> FlowTrajectoryDataset:
    index = np.asarray(positions, dtype=np.int64)
    return FlowTrajectoryDataset(
        source_path=dataset.source_path,
        vector_key=dataset.vector_key,
        trajectories=[dataset.trajectories[int(i)] for i in index],
        original_indices=dataset.original_indices[index],
        problem_ids=dataset.problem_ids[index],
        sample_idx=dataset.sample_idx[index],
        y_error=dataset.y_error[index],
        is_correct=dataset.is_correct[index],
        n_steps=dataset.n_steps[index],
        response_chars=dataset.response_chars[index],
        layer_ids=dataset.layer_ids.copy(),
        hidden_dim=dataset.hidden_dim,
        label_policy=dataset.label_policy,
        skipped=dict(dataset.skipped),
        metadata=dict(dataset.metadata),
    )


def inspect_directional_cloud_source(
    path: str | Path,
    *,
    vector_key: str = "auto",
    cloud_layers: str = "all",
    label_policy: str = "answer_format_ok",
    max_samples: int = 0,
) -> dict[str, Any]:
    dataset = load_directional_cloud_dataset(
        path,
        vector_key=vector_key,
        cloud_layers=cloud_layers,
        label_policy=label_policy,
        max_samples=max_samples,
    )
    contrastive = 0
    for problem in np.unique(dataset.base.problem_ids):
        labels = dataset.base.y_error[dataset.base.problem_ids == problem]
        contrastive += int(np.any(labels == 0) and np.any(labels == 1))
    return {
        "path": str(Path(path)),
        "vector_key": dataset.base.vector_key,
        "has_sv_clouds": True,
        "samples": dataset.n_samples,
        "errors": int(np.sum(dataset.base.y_error == 1)),
        "correct": int(np.sum(dataset.base.y_error == 0)),
        "problems": int(np.unique(dataset.base.problem_ids).size),
        "contrastive_problems": int(contrastive),
        "cloud_layers": dataset.cloud_layer_ids.tolist(),
        "cloud_hidden_dim": int(dataset.cloud_hidden_dim),
        "response_token_min": int(np.min(dataset.response_tokens)),
        "response_token_median": float(np.median(dataset.response_tokens)),
        "response_token_max": int(np.max(dataset.response_tokens)),
        "label_policy": dataset.base.label_policy,
        "skipped_base": dataset.base.skipped,
        "skipped_clouds": dataset.skipped_clouds,
        "ready": bool(contrastive > 0),
    }


def load_directional_cloud_dataset(
    path: str | Path,
    *,
    vector_key: str = "auto",
    cloud_layers: str = "all",
    label_policy: str = "answer_format_ok",
    max_samples: int = 0,
) -> DirectionalCloudDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    base = load_flow_trajectory_dataset(
        path,
        vector_key=vector_key,
        layers="all",
        label_policy=label_policy,
        max_samples=max_samples,
    )
    z = np.load(path, allow_pickle=True)
    if "sv_clouds" not in z.files or "cloud_sizes" not in z.files:
        raise FileNotFoundError(
            f"{path}: missing sv_clouds/cloud_sizes; use the existing multisample artifact "
            "extracted with --store_clouds"
        )
    raw_clouds = z["sv_clouds"]
    raw_sizes = z["cloud_sizes"]
    example = None
    for original in base.original_indices.tolist():
        try:
            candidate = _cloud_array(raw_clouds[int(original)])
        except (TypeError, ValueError):
            continue
        if candidate.shape[0] >= 2:
            example = candidate
            break
    if example is None:
        raise ValueError(f"{path}: no valid token cloud remains after label filtering")
    layer_key, all_layer_ids = _resolve_cloud_layer_ids(z, int(example.shape[1]))
    layer_positions, selected_layer_ids = parse_layer_selection(cloud_layers, all_layer_ids)

    clouds: list[np.ndarray] = []
    step_sizes: list[np.ndarray] = []
    positions: list[int] = []
    response_tokens: list[int] = []
    skipped = {
        "missing": 0,
        "invalid_shape": 0,
        "invalid_sizes": 0,
        "step_count_mismatch": 0,
        "nonfinite": 0,
        "zero_norm": 0,
    }
    for position, original in enumerate(base.original_indices.tolist()):
        value = raw_clouds[int(original)]
        size_value = raw_sizes[int(original)]
        if value is None or size_value is None:
            skipped["missing"] += 1
            continue
        try:
            cloud = _cloud_array(value)
        except (TypeError, ValueError):
            skipped["invalid_shape"] += 1
            continue
        sizes = np.asarray(size_value, dtype=np.int64).reshape(-1)
        if (
            sizes.size == 0
            or np.any(sizes <= 0)
            or int(np.sum(sizes)) != int(cloud.shape[0])
        ):
            skipped["invalid_sizes"] += 1
            continue
        if int(sizes.size) != int(base.n_steps[position]):
            skipped["step_count_mismatch"] += 1
            continue
        if cloud.shape[1] != all_layer_ids.size:
            skipped["invalid_shape"] += 1
            continue
        cloud = cloud[:, layer_positions, :]
        if not np.isfinite(cloud).all():
            skipped["nonfinite"] += 1
            continue
        if np.any(np.all(cloud == 0, axis=-1)):
            skipped["zero_norm"] += 1
            continue
        clouds.append(np.ascontiguousarray(cloud))
        step_sizes.append(np.ascontiguousarray(sizes))
        positions.append(position)
        response_tokens.append(int(cloud.shape[0]))
    if not clouds:
        raise ValueError(f"{path}: no valid aligned token clouds")
    hidden_dim = int(clouds[0].shape[2])
    expected = (selected_layer_ids.size, hidden_dim)
    if any(cloud.shape[1:] != expected for cloud in clouds):
        raise ValueError("selected token clouds do not share layer/hidden dimensions")
    subset = _subset_base(base, positions)
    subset.metadata.update(
        {
            "cloud_layer_metadata_key": layer_key,
            "available_cloud_layer_ids": all_layer_ids.tolist(),
            "selected_cloud_layer_ids": selected_layer_ids.tolist(),
        }
    )
    return DirectionalCloudDataset(
        base=subset,
        clouds=clouds,
        step_sizes=step_sizes,
        cloud_layer_ids=selected_layer_ids,
        cloud_hidden_dim=hidden_dim,
        response_tokens=np.asarray(response_tokens, dtype=np.int64),
        skipped_clouds=skipped,
    )


def directional_statistics(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return resultant length and off-diagonal mean cosine per leading group.

    ``vectors`` has shape ``[..., token, hidden]``. The second output is the
    exact U-statistic

        (n * R^2 - 1) / (n - 1)

    after unit normalization, so self-pairs cannot inflate concentration.
    """

    if vectors.ndim < 2:
        raise ValueError("vectors must have a token and hidden axis")
    n_tokens = int(vectors.shape[-2])
    if n_tokens < 2:
        shape = vectors.shape[:-2]
        nan = torch.full(shape, float("nan"), device=vectors.device, dtype=torch.float32)
        return nan, nan.clone()
    float_vectors = vectors.float()
    norms = torch.linalg.vector_norm(float_vectors, dim=-1, keepdim=True)
    if bool(torch.any(norms <= 1e-12)):
        raise ValueError("directional statistics require non-zero token vectors")
    unit = float_vectors / norms
    summed = unit.sum(dim=-2)
    norm2 = summed.square().sum(dim=-1)
    resultant = torch.sqrt(torch.clamp_min(norm2, 0.0)) / float(n_tokens)
    concentration = (norm2 - float(n_tokens)) / float(n_tokens * (n_tokens - 1))
    return resultant, torch.clamp(concentration, min=-1.0, max=1.0)


def _group_statistics(
    unit: torch.Tensor,
    group_ids: torch.Tensor,
    counts: torch.Tensor,
    n_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layers, hidden = int(unit.shape[1]), int(unit.shape[2])
    sums = torch.zeros((n_groups, layers, hidden), device=unit.device, dtype=torch.float32)
    sums.index_add_(0, group_ids, unit)
    norm2 = sums.square().sum(dim=-1)
    count = counts.to(device=unit.device, dtype=torch.float32)[:, None]
    resultant = torch.sqrt(torch.clamp_min(norm2, 0.0)) / count.clamp_min(1.0)
    denominator = count * (count - 1.0)
    concentration = (norm2 - count) / denominator.clamp_min(1.0)
    concentration = torch.where(
        count >= 2.0,
        torch.clamp(concentration, min=-1.0, max=1.0),
        torch.full_like(concentration, float("nan")),
    )
    return resultant, concentration


def _batch_positions(lengths: np.ndarray, batch_size: int, max_tokens: int) -> list[np.ndarray]:
    batches: list[np.ndarray] = []
    current: list[int] = []
    token_count = 0
    for index, length in enumerate(np.asarray(lengths, dtype=np.int64).tolist()):
        if current and (len(current) >= batch_size or token_count + length > max_tokens):
            batches.append(np.asarray(current, dtype=np.int64))
            current = []
            token_count = 0
        current.append(index)
        token_count += int(length)
    if current:
        batches.append(np.asarray(current, dtype=np.int64))
    return batches


def _nanmean(x: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    finite = torch.isfinite(x)
    numerator = torch.where(finite, x, torch.zeros_like(x)).sum(dim=dim)
    finite_count = finite.sum(dim=dim)
    result = numerator / finite_count.clamp_min(1)
    return torch.where(
        finite_count > 0,
        result,
        torch.full_like(result, float("nan")),
    )


def _nanmax(x: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(x)
    if not bool(torch.any(finite)):
        return torch.full((), float("nan"), device=x.device, dtype=x.dtype)
    return torch.where(
        finite,
        x,
        torch.full_like(x, -float("inf")),
    ).max()


def compute_directional_consensus(
    dataset: DirectionalCloudDataset,
    cfg: DirectionalConsensusConfig,
) -> DirectionalConsensusEncoding:
    cfg.validate()
    started = time.perf_counter()
    device = torch.device(cfg.compute_device)
    n_samples = dataset.n_samples
    chain_score_names = (
        "consensus.raw_spread.global",
        "consensus.raw_spread.step_mean",
        "consensus.raw_spread.late_mean",
        "consensus.raw_spread.step_max",
        "consensus.debiased_dispersion.global",
        "consensus.debiased_dispersion.step_mean",
        "consensus.debiased_dispersion.late_mean",
        "consensus.debiased_dispersion.step_max",
        "consensus.fixed_window_dispersion.mean",
    )
    chain_scores = {name: np.full(n_samples, np.nan, dtype=np.float32) for name in chain_score_names}
    step_resultant: list[np.ndarray | None] = [None] * n_samples
    step_raw_spread: list[np.ndarray | None] = [None] * n_samples
    step_concentration: list[np.ndarray | None] = [None] * n_samples
    step_debiased: list[np.ndarray | None] = [None] * n_samples

    batches = _batch_positions(
        dataset.response_tokens * int(dataset.cloud_layer_ids.size),
        batch_size=cfg.batch_size,
        max_tokens=cfg.max_batch_tokens,
    )
    with torch.inference_mode():
        for positions in batches:
            arrays = [dataset.clouds[int(i)] for i in positions]
            sizes_list = [dataset.step_sizes[int(i)] for i in positions]
            lengths = np.asarray([array.shape[0] for array in arrays], dtype=np.int64)
            cloud = torch.as_tensor(np.concatenate(arrays, axis=0), device=device)
            unit = F.normalize(cloud.float(), p=2.0, dim=-1, eps=1e-12)

            sample_ids_np = np.repeat(np.arange(len(arrays), dtype=np.int64), lengths)
            sample_ids = torch.as_tensor(sample_ids_np, device=device, dtype=torch.long)
            sample_counts = torch.as_tensor(lengths, device=device, dtype=torch.long)
            global_resultant, global_concentration = _group_statistics(
                unit,
                sample_ids,
                sample_counts,
                len(arrays),
            )

            step_ids_parts: list[np.ndarray] = []
            step_offset = 0
            for sizes in sizes_list:
                local = np.repeat(
                    np.arange(sizes.size, dtype=np.int64) + step_offset,
                    sizes,
                )
                step_ids_parts.append(local)
                step_offset += int(sizes.size)
            all_step_sizes = np.concatenate(sizes_list).astype(np.int64, copy=False)
            step_ids = torch.as_tensor(
                np.concatenate(step_ids_parts),
                device=device,
                dtype=torch.long,
            )
            step_counts = torch.as_tensor(all_step_sizes, device=device, dtype=torch.long)
            step_r, step_c = _group_statistics(unit, step_ids, step_counts, int(step_offset))
            step_raw = 1.0 - step_r
            step_deb = 1.0 - step_c

            window_values: list[torch.Tensor] = []
            window_sample_ids: list[int] = []
            token_offset = 0
            for local_index, length in enumerate(lengths.tolist()):
                windows = int(length // cfg.fixed_window_tokens)
                used = windows * cfg.fixed_window_tokens
                if windows:
                    segment = unit[token_offset : token_offset + used]
                    window_values.append(
                        segment.reshape(
                            windows,
                            cfg.fixed_window_tokens,
                            unit.shape[1],
                            unit.shape[2],
                        )
                    )
                    window_sample_ids.extend([local_index] * windows)
                token_offset += int(length)
            fixed_mean = torch.full(
                (len(arrays), unit.shape[1]),
                float("nan"),
                device=device,
                dtype=torch.float32,
            )
            if window_values:
                windows = torch.cat(window_values, dim=0).permute(0, 2, 1, 3)
                _, window_concentration = directional_statistics(windows)
                window_dispersion = 1.0 - window_concentration
                window_ids = torch.as_tensor(window_sample_ids, device=device, dtype=torch.long)
                window_sum = torch.zeros_like(fixed_mean)
                window_count = torch.zeros(len(arrays), device=device, dtype=torch.float32)
                window_sum.index_add_(0, window_ids, window_dispersion)
                window_count.index_add_(0, window_ids, torch.ones_like(window_ids, dtype=torch.float32))
                fixed_mean = window_sum / window_count[:, None].clamp_min(1.0)
                fixed_mean = torch.where(
                    window_count[:, None] > 0,
                    fixed_mean,
                    torch.full_like(fixed_mean, float("nan")),
                )

            step_cursor = 0
            for local_index, global_index in enumerate(positions.tolist()):
                n_steps = int(sizes_list[local_index].size)
                next_cursor = step_cursor + n_steps
                local_r = step_r[step_cursor:next_cursor]
                local_c = step_c[step_cursor:next_cursor]
                local_raw = step_raw[step_cursor:next_cursor]
                local_deb = step_deb[step_cursor:next_cursor]
                late_start = max(0, int(math.floor(n_steps * (1.0 - cfg.late_fraction))))

                chain_scores["consensus.raw_spread.global"][global_index] = float(
                    torch.mean(1.0 - global_resultant[local_index]).cpu()
                )
                chain_scores["consensus.raw_spread.step_mean"][global_index] = float(
                    _nanmean(local_raw, dim=(0, 1)).cpu()
                )
                chain_scores["consensus.raw_spread.late_mean"][global_index] = float(
                    _nanmean(local_raw[late_start:], dim=(0, 1)).cpu()
                )
                chain_scores["consensus.raw_spread.step_max"][global_index] = float(
                    _nanmax(local_raw).cpu()
                )
                chain_scores["consensus.debiased_dispersion.global"][global_index] = float(
                    torch.mean(1.0 - global_concentration[local_index]).cpu()
                )
                chain_scores["consensus.debiased_dispersion.step_mean"][global_index] = float(
                    _nanmean(local_deb, dim=(0, 1)).cpu()
                )
                chain_scores["consensus.debiased_dispersion.late_mean"][global_index] = float(
                    _nanmean(local_deb[late_start:], dim=(0, 1)).cpu()
                )
                chain_scores["consensus.debiased_dispersion.step_max"][global_index] = float(
                    _nanmax(local_deb).cpu()
                )
                chain_scores["consensus.fixed_window_dispersion.mean"][global_index] = float(
                    _nanmean(fixed_mean[local_index], dim=0).cpu()
                )

                step_resultant[global_index] = local_r.detach().cpu().numpy().astype(np.float32)
                step_raw_spread[global_index] = local_raw.detach().cpu().numpy().astype(np.float32)
                step_concentration[global_index] = local_c.detach().cpu().numpy().astype(np.float32)
                step_debiased[global_index] = local_deb.detach().cpu().numpy().astype(np.float32)
                step_cursor = next_cursor

    if any(value is None for value in step_resultant):
        raise RuntimeError("internal error: not all cloud samples were encoded")
    return DirectionalConsensusEncoding(
        step_resultant=[np.asarray(value) for value in step_resultant],
        step_raw_spread=[np.asarray(value) for value in step_raw_spread],
        step_pairwise_concentration=[np.asarray(value) for value in step_concentration],
        step_debiased_dispersion=[np.asarray(value) for value in step_debiased],
        chain_scores=chain_scores,
        runtime_seconds=time.perf_counter() - started,
    )


def _bh_qvalues(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    output = np.full(p.shape, np.nan, dtype=np.float64)
    finite = np.where(np.isfinite(p))[0]
    if finite.size == 0:
        return output
    order = finite[np.argsort(p[finite])]
    adjusted = np.empty(order.size, dtype=np.float64)
    running = 1.0
    for rank in range(order.size - 1, -1, -1):
        running = min(running, p[order[rank]] * order.size / (rank + 1))
        adjusted[rank] = min(1.0, running)
    output[order] = adjusted
    return output


def _length_matched_components(
    score: np.ndarray,
    y_error: np.ndarray,
    problem_ids: np.ndarray,
    lengths: np.ndarray,
    max_ratio: float,
) -> dict[Any, tuple[float, int]]:
    components: dict[Any, tuple[float, int]] = {}
    values = np.asarray(score, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    for problem in np.unique(problem_ids):
        indices = np.where(problem_ids == problem)[0]
        errors = indices[(y_error[indices] == 1) & np.isfinite(values[indices])]
        correct = indices[(y_error[indices] == 0) & np.isfinite(values[indices])]
        concordance = 0.0
        pairs = 0
        for error in errors:
            ratio = np.maximum(lengths[error], lengths[correct]) / np.maximum(
                np.minimum(lengths[error], lengths[correct]),
                1.0,
            )
            matched = correct[ratio <= max_ratio]
            if not matched.size:
                continue
            difference = values[error] - values[matched]
            concordance += float(np.sum(difference > 0) + 0.5 * np.sum(difference == 0))
            pairs += int(matched.size)
        if pairs:
            components[problem] = (concordance, pairs)
    return components


def _problem_pair_components(
    score: np.ndarray,
    y_error: np.ndarray,
    problem_ids: np.ndarray,
) -> dict[Any, tuple[float, int]]:
    components: dict[Any, tuple[float, int]] = {}
    values = np.asarray(score, dtype=np.float64)
    for problem in np.unique(problem_ids):
        indices = np.where(problem_ids == problem)[0]
        error = values[indices[(y_error[indices] == 1) & np.isfinite(values[indices])]]
        correct = values[indices[(y_error[indices] == 0) & np.isfinite(values[indices])]]
        if not error.size or not correct.size:
            continue
        difference = error[:, None] - correct[None, :]
        concordance = float(
            np.sum(difference > 0) + 0.5 * np.sum(difference == 0)
        )
        components[problem] = (concordance, int(difference.size))
    return components


def _component_auc(components: Mapping[Any, tuple[float, int]]) -> tuple[float, int]:
    pairs = int(sum(value[1] for value in components.values()))
    concordance = float(sum(value[0] for value in components.values()))
    return (concordance / pairs if pairs else float("nan")), pairs


def _component_bootstrap(
    components: Mapping[Any, tuple[float, int]],
    *,
    draws: int,
    seed: int,
    compute_device: str,
) -> list[float]:
    if draws <= 0 or len(components) < 2:
        return [float("nan"), float("nan")]
    values = list(components.values())
    device = torch.device(compute_device)
    concordance = torch.as_tensor(
        [value[0] for value in values],
        dtype=torch.float64,
        device=device,
    )
    pairs = torch.as_tensor(
        [value[1] for value in values],
        dtype=torch.float64,
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    selected = torch.randint(
        0,
        len(values),
        (int(draws), len(values)),
        device=device,
        generator=generator,
    )
    denominators = pairs[selected].sum(dim=1)
    valid = denominators > 0
    if not bool(torch.any(valid)):
        return [float("nan"), float("nan")]
    estimates = concordance[selected].sum(dim=1)[valid] / denominators[valid]
    quantiles = torch.quantile(
        estimates,
        torch.as_tensor([0.025, 0.975], device=device, dtype=estimates.dtype),
    )
    return [float(value) for value in quantiles.detach().cpu().tolist()]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _same_problem_permutation_p(
    score: np.ndarray,
    y_error: np.ndarray,
    problem_ids: np.ndarray,
    observed: float,
    *,
    permutations: int,
    seed: int,
    compute_device: str,
) -> float:
    if permutations <= 0 or not np.isfinite(observed):
        return float("nan")
    groups: list[tuple[np.ndarray, int]] = []
    total_pairs = 0
    values = np.asarray(score, dtype=np.float64)
    for problem in np.unique(problem_ids):
        indices = np.where(problem_ids == problem)[0]
        finite = indices[np.isfinite(values[indices])]
        labels = y_error[finite]
        n_error = int(np.sum(labels == 1))
        n_correct = int(np.sum(labels == 0))
        if not n_error or not n_correct:
            continue
        groups.append((_average_ranks(values[finite]), n_error))
        total_pairs += n_error * n_correct
    if not groups or not total_pairs:
        return float("nan")

    device = torch.device(compute_device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    concordance = torch.zeros(permutations, device=device, dtype=torch.float64)
    for ranks, n_error in groups:
        rank_tensor = torch.as_tensor(ranks, device=device, dtype=torch.float64)
        random_keys = torch.rand(
            (permutations, ranks.size),
            device=device,
            generator=generator,
        )
        selected = torch.topk(
            random_keys,
            k=n_error,
            dim=1,
            largest=False,
            sorted=False,
        ).indices
        concordance += rank_tensor[selected].sum(dim=1) - (
            n_error * (n_error + 1) / 2.0
        )
    permuted = concordance / float(total_pairs)
    exceed = int(torch.sum(permuted >= float(observed)).detach().cpu())
    return float((1 + exceed) / (permutations + 1))


def _bootstrap_auc_delta(
    first: np.ndarray,
    second: np.ndarray,
    y_error: np.ndarray,
    problem_ids: np.ndarray,
    *,
    draws: int,
    seed: int,
    compute_device: str,
) -> dict[str, Any]:
    first_components = _problem_pair_components(first, y_error, problem_ids)
    second_components = _problem_pair_components(second, y_error, problem_ids)
    keys = sorted(set(first_components) & set(second_components), key=str)
    first_auc, _ = _component_auc({key: first_components[key] for key in keys})
    second_auc, _ = _component_auc({key: second_components[key] for key in keys})
    point = first_auc - second_auc
    if draws <= 0 or len(keys) < 2:
        return {
            "point": point,
            "ci95": [float("nan"), float("nan")],
            "problems": len(keys),
        }
    device = torch.device(compute_device)
    first_concordance = torch.as_tensor(
        [first_components[key][0] for key in keys],
        device=device,
        dtype=torch.float64,
    )
    first_pairs = torch.as_tensor(
        [first_components[key][1] for key in keys],
        device=device,
        dtype=torch.float64,
    )
    second_concordance = torch.as_tensor(
        [second_components[key][0] for key in keys],
        device=device,
        dtype=torch.float64,
    )
    second_pairs = torch.as_tensor(
        [second_components[key][1] for key in keys],
        device=device,
        dtype=torch.float64,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    selected = torch.randint(
        0,
        len(keys),
        (draws, len(keys)),
        device=device,
        generator=generator,
    )
    first_draw = first_concordance[selected].sum(dim=1) / first_pairs[selected].sum(dim=1)
    second_draw = second_concordance[selected].sum(dim=1) / second_pairs[selected].sum(dim=1)
    delta = first_draw - second_draw
    quantiles = torch.quantile(
        delta,
        torch.as_tensor([0.025, 0.975], device=device, dtype=delta.dtype),
    )
    return {
        "point": point,
        "ci95": [float(value) for value in quantiles.detach().cpu().tolist()],
        "problems": len(keys),
    }


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    finite = np.isfinite(first) & np.isfinite(second)
    if int(np.sum(finite)) < 3:
        return float("nan")

    first_rank = _average_ranks(np.asarray(first[finite], dtype=np.float64))
    second_rank = _average_ranks(np.asarray(second[finite], dtype=np.float64))
    first_centered = first_rank - np.mean(first_rank)
    second_centered = second_rank - np.mean(second_rank)
    denominator = float(
        np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2))
    )
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(first_centered * second_centered) / denominator)


def _problem_effect_rows(
    scores: Mapping[str, np.ndarray],
    dataset: DirectionalCloudDataset,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y = dataset.base.y_error
    for name, score in scores.items():
        if name.startswith("control.") or name.endswith(".length_residual"):
            continue
        values = np.asarray(score, dtype=np.float64)
        for problem in np.unique(dataset.base.problem_ids):
            indices = np.where(dataset.base.problem_ids == problem)[0]
            error = values[indices[(y[indices] == 1) & np.isfinite(values[indices])]]
            correct = values[indices[(y[indices] == 0) & np.isfinite(values[indices])]]
            if not error.size or not correct.size:
                continue
            rows.append(
                {
                    "score": name,
                    "problem_id": problem,
                    "n_error": int(error.size),
                    "n_correct": int(correct.size),
                    "error_mean": float(np.mean(error)),
                    "correct_mean": float(np.mean(correct)),
                    "difference": float(np.mean(error) - np.mean(correct)),
                }
            )
    return rows


def _object_array(values: Sequence[np.ndarray]) -> np.ndarray:
    output = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        output[index] = np.asarray(value)
    return output


def run_directional_consensus_audit(
    dataset: DirectionalCloudDataset,
    compute_cfg: DirectionalConsensusConfig,
    audit_cfg: DirectionalConsensusAuditConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    audit_cfg.validate()
    encoding = compute_directional_consensus(dataset, compute_cfg)
    base = dataset.base
    statistical_cfg = FlowAuditConfig(
        folds=audit_cfg.folds,
        bootstrap=audit_cfg.bootstrap,
        permutations=audit_cfg.permutations,
        seed=audit_cfg.seed,
        compute_device=compute_cfg.compute_device,
    )
    controls = np.column_stack(
        [
            np.log1p(base.n_steps.astype(np.float64)),
            np.log1p(base.response_chars.astype(np.float64)),
            np.log1p(dataset.response_tokens.astype(np.float64)),
        ]
    )
    score_map: dict[str, np.ndarray] = {
        "control.log1p_n_steps": controls[:, 0],
        "control.log1p_response_chars": controls[:, 1],
        "control.log1p_response_tokens": controls[:, 2],
    }
    score_map.update(encoding.chain_scores)
    geometry_names = list(encoding.chain_scores)
    for name in geometry_names:
        score_map[f"{name}.length_residual"] = crossfit_residualize_score(
            score_map[name],
            controls,
            base.problem_ids,
            statistical_cfg,
        )

    confirmatory = [
        "consensus.debiased_dispersion.step_mean",
        "consensus.debiased_dispersion.global",
        "consensus.fixed_window_dispersion.mean",
        "consensus.debiased_dispersion.step_mean.length_residual",
        "consensus.debiased_dispersion.global.length_residual",
        "consensus.fixed_window_dispersion.mean.length_residual",
    ]
    point_cfg = FlowAuditConfig(
        folds=audit_cfg.folds,
        bootstrap=0,
        permutations=0,
        seed=audit_cfg.seed,
        compute_device=compute_cfg.compute_device,
    )
    rows: list[dict[str, Any]] = []
    for name, score in score_map.items():
        row = evaluate_score(name, score, base, point_cfg)
        components = _problem_pair_components(score, base.y_error, base.problem_ids)
        row["same_problem_ci95"] = _component_bootstrap(
            components,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + sum(ord(ch) for ch in name),
            compute_device=compute_cfg.compute_device,
        )
        row["same_problem_permutation_p"] = (
            _same_problem_permutation_p(
                score,
                base.y_error,
                base.problem_ids,
                row["same_problem_auroc"],
                permutations=audit_cfg.permutations,
                seed=audit_cfg.seed + 17 + sum(ord(ch) for ch in name),
                compute_device=compute_cfg.compute_device,
            )
            if name in confirmatory
            else float("nan")
        )
        rows.append(row)
    row_by_name = {row["name"]: row for row in rows}
    q_values = _bh_qvalues(
        [row_by_name[name]["same_problem_permutation_p"] for name in confirmatory]
    )
    for row in rows:
        row["confirmatory"] = bool(row["name"] in confirmatory)
        row["same_problem_bh_q"] = float("nan")
        row["spearman_response_tokens"] = _safe_spearman(
            np.asarray(score_map[row["name"]], dtype=np.float64),
            dataset.response_tokens.astype(np.float64),
        )
        row["spearman_response_chars"] = _safe_spearman(
            np.asarray(score_map[row["name"]], dtype=np.float64),
            base.response_chars.astype(np.float64),
        )
        components = _length_matched_components(
            score_map[row["name"]],
            base.y_error,
            base.problem_ids,
            dataset.response_tokens,
            audit_cfg.length_match_ratio,
        )
        matched_auc, matched_pairs = _component_auc(components)
        row["token_length_matched_auroc"] = matched_auc
        row["token_length_matched_pairs"] = matched_pairs
        row["token_length_matched_problems"] = int(len(components))
        row["token_length_matched_ci95"] = _component_bootstrap(
            components,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 1701 + sum(ord(ch) for ch in row["name"]),
            compute_device=compute_cfg.compute_device,
        )
    for name, q_value in zip(confirmatory, q_values):
        row_by_name[name]["same_problem_bh_q"] = float(q_value)

    raw = score_map["consensus.raw_spread.step_mean"]
    debiased = score_map["consensus.debiased_dispersion.step_mean"]
    raw_residual = score_map["consensus.raw_spread.step_mean.length_residual"]
    debiased_residual = score_map[
        "consensus.debiased_dispersion.step_mean.length_residual"
    ]
    deltas = {
        "debiased_minus_raw": _bootstrap_auc_delta(
            debiased,
            raw,
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 701,
            compute_device=compute_cfg.compute_device,
        ),
        "debiased_minus_raw_length_residual": _bootstrap_auc_delta(
            debiased_residual,
            raw_residual,
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 1701,
            compute_device=compute_cfg.compute_device,
        ),
        "fixed_window_minus_raw": _bootstrap_auc_delta(
            score_map["consensus.fixed_window_dispersion.mean"],
            raw,
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 2701,
            compute_device=compute_cfg.compute_device,
        ),
    }
    primary = row_by_name["consensus.debiased_dispersion.step_mean.length_residual"]
    gate = {
        "primary_score": primary["name"],
        "same_problem_auroc": primary["same_problem_auroc"],
        "same_problem_ci95": primary["same_problem_ci95"],
        "same_problem_bh_q": primary["same_problem_bh_q"],
        "token_length_matched_auroc": primary["token_length_matched_auroc"],
        "token_length_matched_ci95": primary["token_length_matched_ci95"],
        "passes": bool(
            np.isfinite(primary["same_problem_ci95"][0])
            and primary["same_problem_ci95"][0] > 0.5
            and np.isfinite(primary["same_problem_bh_q"])
            and primary["same_problem_bh_q"] < 0.05
            and np.isfinite(primary["token_length_matched_ci95"][0])
            and primary["token_length_matched_ci95"][0] > 0.5
        ),
        "rule": (
            "continue only if the length-residualized debiased step-mean score has "
            "problem-bootstrap CI above 0.5, BH q below 0.05, and its token-length-"
            "matched CI is also above 0.5"
        ),
    }
    report = {
        "meta": {
            "source": base.source_path,
            "vector_key": base.vector_key,
            "label_policy": base.label_policy,
            "samples": dataset.n_samples,
            "errors": int(np.sum(base.y_error == 1)),
            "correct": int(np.sum(base.y_error == 0)),
            "problems": int(np.unique(base.problem_ids).size),
            "contrastive_problems": int(
                sum(
                    np.any(base.y_error[base.problem_ids == problem] == 0)
                    and np.any(base.y_error[base.problem_ids == problem] == 1)
                    for problem in np.unique(base.problem_ids)
                )
            ),
            "cloud_layers": dataset.cloud_layer_ids.tolist(),
            "cloud_hidden_dim": int(dataset.cloud_hidden_dim),
            "fixed_window_tokens": int(compute_cfg.fixed_window_tokens),
            "length_match_ratio": float(audit_cfg.length_match_ratio),
            "runtime_seconds": float(encoding.runtime_seconds),
            "compute_device": compute_cfg.compute_device,
            "skipped_base": base.skipped,
            "skipped_clouds": dataset.skipped_clouds,
        },
        "hypothesis": (
            "Within the same problem, incorrect responses have lower token-direction "
            "consensus than correct responses after removing finite-token self-pair bias."
        ),
        "estimand": {
            "raw_resultant": "R = ||mean_i u_i||",
            "raw_spread": "1 - R",
            "pairwise_concentration": "(n * R^2 - 1) / (n - 1)",
            "debiased_dispersion": "1 - pairwise_concentration",
        },
        "claim_scope": {
            "supported_if_gate_passes": "response-level same-problem directional separability",
            "not_supported": (
                "first-error localization, temporal ordering, output-logit coupling, or causal mechanism"
            ),
        },
        "scores": rows,
        "auc_deltas": deltas,
        "decision_gate": gate,
        "problem_effects": _problem_effect_rows(score_map, dataset),
    }
    packed = {
        "original_indices": base.original_indices,
        "problem_ids": base.problem_ids,
        "sample_idx": base.sample_idx,
        "y_error": base.y_error,
        "is_correct": base.is_correct,
        "n_steps": base.n_steps,
        "response_chars": base.response_chars,
        "response_tokens": dataset.response_tokens,
        "cloud_layer_ids": dataset.cloud_layer_ids,
        "score_names": np.asarray(list(score_map), dtype=object),
        "scores": np.column_stack([score_map[name] for name in score_map]).astype(np.float32),
        "step_resultant": _object_array(encoding.step_resultant),
        "step_raw_spread": _object_array(encoding.step_raw_spread),
        "step_pairwise_concentration": _object_array(
            encoding.step_pairwise_concentration
        ),
        "step_debiased_dispersion": _object_array(
            encoding.step_debiased_dispersion
        ),
    }
    return report, packed


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in columns and not isinstance(value, (list, dict)):
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, report: Mapping[str, Any]) -> None:
    meta = report["meta"]
    gate = report["decision_gate"]
    lines = [
        "# Debiased Directional Consensus Audit",
        "",
        f"- Samples: `{meta['samples']}` (`{meta['errors']}` errors, `{meta['correct']}` correct)",
        f"- Problems: `{meta['problems']}`; contrastive: `{meta['contrastive_problems']}`",
        f"- Cloud layers: `{meta['cloud_layers']}`; hidden dim: `{meta['cloud_hidden_dim']}`",
        f"- Fixed token window: `{meta['fixed_window_tokens']}`",
        "",
        "## Fixed Hypothesis",
        "",
        report["hypothesis"],
        "",
        "The raw resultant contains self-pairs. The debiased concentration is the exact off-diagonal "
        "mean cosine, so it removes the deterministic finite-token term rather than merely regressing "
        "against length after the fact.",
        "",
        "## Scores",
        "",
        "| score | confirmatory | coverage | same-problem AUROC | CI95 | BH q | token-matched AUROC | token rho | cross AUROC |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in report["scores"]:
        ci = row["same_problem_ci95"]
        lines.append(
            f"| {row['name']} | {int(row['confirmatory'])} | {_fmt(row['coverage'], 3)} | "
            f"{_fmt(row['same_problem_auroc'])} | [{_fmt(ci[0])}, {_fmt(ci[1])}] | "
            f"{_fmt(row['same_problem_bh_q'])} | {_fmt(row['token_length_matched_auroc'])} | "
            f"{_fmt(row['spearman_response_tokens'])} | {_fmt(row['cross_problem_auroc'])} |"
        )
    lines.extend(
        [
            "",
            "## AUROC Deltas",
            "",
            "| comparison | delta | CI95 | problems |",
            "|---|---:|---|---:|",
        ]
    )
    for name, row in report["auc_deltas"].items():
        lines.append(
            f"| {name} | {_fmt(row['point'])} | [{_fmt(row['ci95'][0])}, "
            f"{_fmt(row['ci95'][1])}] | {row['problems']} |"
        )
    lines.extend(
        [
            "",
            "## Decision Gate",
            "",
            f"- Primary: `{gate['primary_score']}`",
            f"- Same-problem AUROC: `{_fmt(gate['same_problem_auroc'])}`; CI `{gate['same_problem_ci95']}`",
            f"- BH q: `{_fmt(gate['same_problem_bh_q'])}`",
            f"- Token-length-matched AUROC: `{_fmt(gate['token_length_matched_auroc'])}`; "
            f"CI `{gate['token_length_matched_ci95']}`",
            f"- **PASS: `{gate['passes']}`**",
            f"- Rule: {gate['rule']}",
            "",
            "## Claim Boundary",
            "",
            f"- A pass supports: {report['claim_scope']['supported_if_gate_passes']}.",
            f"- It does not support: {report['claim_scope']['not_supported']}.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_plots(output_dir: Path, report: Mapping[str, Any], packed: Mapping[str, np.ndarray]) -> list[Path]:
    import matplotlib.pyplot as plt

    names = [str(x) for x in packed["score_names"].tolist()]
    matrix = np.asarray(packed["scores"], dtype=np.float64)
    y = np.asarray(packed["y_error"], dtype=np.int64)
    tokens = np.asarray(packed["response_tokens"], dtype=np.float64)
    selected = [
        "consensus.raw_spread.step_mean",
        "consensus.debiased_dispersion.step_mean",
        "consensus.fixed_window_dispersion.mean",
    ]
    figure, axes = plt.subplots(1, len(selected), figsize=(14, 4))
    for axis, name in zip(axes, selected):
        values = matrix[:, names.index(name)]
        axis.hist(values[y == 0], bins=35, alpha=0.55, density=True, label="correct")
        axis.hist(values[y == 1], bins=35, alpha=0.55, density=True, label="error")
        axis.set_title(name.replace("consensus.", ""), fontsize=9)
        axis.set_xlabel("risk score")
    axes[0].set_ylabel("density")
    axes[-1].legend()
    figure.tight_layout()
    distribution_path = output_dir / "score_distributions.png"
    figure.savefig(distribution_path, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, name in zip(axes, selected[:2]):
        values = matrix[:, names.index(name)]
        for label, text, color in ((0, "correct", "tab:blue"), (1, "error", "tab:red")):
            keep = y == label
            axis.scatter(tokens[keep], values[keep], s=8, alpha=0.25, label=text, color=color)
        axis.set_xscale("log")
        axis.set_xlabel("response cloud tokens")
        axis.set_ylabel("risk score")
        axis.set_title(name.replace("consensus.", ""), fontsize=9)
    axes[-1].legend()
    figure.tight_layout()
    length_path = output_dir / "length_dependence.png"
    figure.savefig(length_path, dpi=180)
    plt.close(figure)
    return [distribution_path, length_path]


def write_directional_consensus_outputs(
    report: Mapping[str, Any],
    packed: Mapping[str, np.ndarray],
    *,
    output: str | Path,
    output_dir: str | Path,
    render_plots: bool = True,
) -> dict[str, str]:
    output_path = Path(output)
    directory = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **packed)
    summary_json = directory / "summary.json"
    summary_json.write_text(
        json.dumps(finite_json(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_md = directory / "summary.md"
    _write_summary(summary_md, report)
    scores_csv = directory / "score_table.csv"
    _write_csv(scores_csv, report["scores"])
    effects_csv = directory / "problem_effects.csv"
    _write_csv(effects_csv, report["problem_effects"])
    paths = {
        "scores_npz": str(output_path),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "score_table": str(scores_csv),
        "problem_effects": str(effects_csv),
    }
    if render_plots:
        for path in _render_plots(directory, report, packed):
            paths[path.stem] = str(path)
    return paths
