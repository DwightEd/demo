from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .first_error_geometry import (
    AxisGeometryDataset,
    EventMatch,
    StepGeometryDataset,
    crossfit_correct_nuisance_residuals,
    load_step_geometry_dataset,
    make_step_axis,
    map_matches_to_axis,
    match_correct_pseudo_events,
)


EPS = 1e-8
BASE_METRIC_NAMES = (
    "qpt_escape_ratio",
    "qpt_isotropic_distance",
    "qpt_nearest_direction_distance",
    "qpt_reference_topr_energy",
    "qpt_reference_effective_rank",
    "phase_only_escape_ratio",
    "shuffled_question_escape_ratio",
    "global_escape_ratio",
    "random_escape_ratio",
    "question_conditioning_excess",
    "window_normal_energy",
    "normal_persistence",
    "coherent_normal_drift",
)
OUTPUT_METRIC_NAMES = (
    "output_transverse_energy",
    "output_tangent_energy",
    "output_transverse_fraction",
    "output_normal_alignment",
    "output_tangent_alignment",
    "output_alignment_excess",
)
LEGACY_METRIC_NAMES = (
    "update_speed",
    "direction_spread",
    "direction_resultant_jl",
    "direction_spec_entropy_raw",
    "direction_spec_entropy_norm",
    "direction_effective_rank_norm",
)


@dataclass(frozen=True)
class ConditionalTangentConfig:
    device: str = "cuda"
    batch_size: int = 48
    folds: int = 5
    neighbors: int = 32
    search_multiplier: int = 4
    tangent_rank: int = 6
    q_temperature: float = 0.10
    phase_sigma: float = 0.20
    phase_mode: str = "causal_step"
    causal_time_scale: float = 4.0
    persistence_window: int = 3
    reference_policy: str = "correct_only"
    global_reference_cap: int = 512
    nuisance_ridge: float = 1e-3
    random_seed: int = 17


@dataclass
class ConditionalTangentDataset:
    source: StepGeometryDataset
    qvecs: list[np.ndarray]
    output_cotangents: list[np.ndarray] | None
    output_cotangent_key: str | None
    output_cotangent_kind: str | None
    stored_spread: list[np.ndarray]
    response_clouds: list[np.ndarray] | None
    response_cloud_layer_ids: np.ndarray
    metadata: dict[str, Any]


@dataclass
class ConditionalTangentResult:
    dataset: ConditionalTangentDataset
    axis: AxisGeometryDataset
    metric_names: tuple[str, ...]
    fields: list[np.ndarray]
    residual_fields: list[np.ndarray]
    normal_vectors: list[np.ndarray]
    legacy_fields: dict[str, list[np.ndarray]]
    legacy_residual_fields: dict[str, list[np.ndarray]]
    matches: list[EventMatch]
    metadata: dict[str, Any]


@dataclass
class _ReferenceJob:
    row: int
    step: int
    target: np.ndarray
    references: np.ndarray
    weights: np.ndarray


@dataclass
class _ReferenceScore:
    escape: np.ndarray
    isotropic: np.ndarray
    nearest: np.ndarray
    top_rank_energy: np.ndarray
    effective_rank: np.ndarray
    residuals: list[np.ndarray]


def _effective_device(value: str) -> torch.device:
    if str(value).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(str(value))


def _as_numeric(value: Any, *, dtype: np.dtype = np.float32) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        array = np.asarray(value.tolist())
    return np.asarray(array, dtype=dtype)


def _metadata_layers(
    z: np.lib.npyio.NpzFile,
    *,
    candidates: Sequence[str],
    depth: int,
) -> tuple[np.ndarray, str | None]:
    for key in candidates:
        if key not in z.files:
            continue
        values = np.asarray(z[key], dtype=np.int64).reshape(-1)
        if values.size == depth:
            return values, key
    return np.arange(depth, dtype=np.int64), None


def _select_layer_matrix(
    value: Any,
    *,
    stored_layers: np.ndarray,
    selected_layers: np.ndarray,
    hidden_dim: int,
) -> np.ndarray:
    array = _as_numeric(value)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != int(hidden_dim):
        raise ValueError(
            f"expected [layer,{hidden_dim}] question vector, got {array.shape}"
        )
    if array.shape[0] != stored_layers.size:
        raise ValueError(
            f"question vector stores {array.shape[0]} layers but metadata has "
            f"{stored_layers.size}"
        )
    missing = [int(layer) for layer in selected_layers if layer not in set(stored_layers.tolist())]
    if missing:
        raise ValueError(f"qvec is missing selected layers {missing}")
    positions = [int(np.where(stored_layers == layer)[0][0]) for layer in selected_layers]
    return np.ascontiguousarray(array[positions], dtype=np.float32)


def _record_value(raw: np.ndarray, index: int, n_records: int) -> Any:
    if raw.ndim >= 1 and raw.shape[0] == n_records:
        return raw[index]
    if raw.dtype == object and raw.size == n_records:
        return raw.reshape(-1)[index]
    raise ValueError(f"array with shape {raw.shape} is not record-aligned to {n_records} chains")


def _load_qvecs(
    z: np.lib.npyio.NpzFile,
    source: StepGeometryDataset,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if "qvec" not in z.files:
        raise FileNotFoundError(
            "question-conditioned tangent audit requires qvec; this artifact has none"
        )
    raw = z["qvec"]
    n_records = int(len(z[source.vector_key]))
    example = _as_numeric(_record_value(raw, int(source.original_indices[0]), n_records))
    if example.ndim == 1:
        depth = 1
    elif example.ndim == 2:
        depth = int(example.shape[0])
    else:
        raise ValueError(f"unsupported qvec record shape {example.shape}")
    stored_layers, metadata_key = _metadata_layers(
        z,
        candidates=("sv_layers", "layers_used", "qvec_layers", "layers"),
        depth=depth,
    )
    qvecs: list[np.ndarray] = []
    for original in source.original_indices:
        qvecs.append(
            _select_layer_matrix(
                _record_value(raw, int(original), n_records),
                stored_layers=stored_layers,
                selected_layers=source.layer_ids,
                hidden_dim=source.hidden_dim,
            )
        )
    if any(not np.isfinite(value).all() for value in qvecs):
        raise ValueError("qvec contains non-finite values on selected layers")
    return qvecs, {
        "qvec_layer_key": metadata_key,
        "qvec_layers": stored_layers.tolist(),
    }


def _resolve_output_cotangent_key(
    files: Sequence[str],
    requested: str,
) -> str | None:
    if requested not in {"", "auto", "none"}:
        if requested not in files:
            raise FileNotFoundError(f"requested output cotangent key {requested!r} is absent")
        return requested
    if requested == "none":
        return None
    for key in ("step_output_cotangent", "step_hidden_grad", "output_cotangent"):
        if key in files:
            return key
    return None


def _load_output_cotangents(
    z: np.lib.npyio.NpzFile,
    source: StepGeometryDataset,
    *,
    requested_key: str,
) -> tuple[list[np.ndarray] | None, str | None, str | None, dict[str, Any]]:
    key = _resolve_output_cotangent_key(z.files, requested_key)
    if key is None:
        return None, None, None, {"output_coupling_available": False}
    raw = z[key]
    n_records = int(len(z[source.vector_key]))
    example = _as_numeric(_record_value(raw, int(source.original_indices[0]), n_records))
    if example.ndim == 2:
        example = example[:, None, :]
    if example.ndim != 3:
        raise ValueError(f"{key} must store [step,layer,hidden], got {example.shape}")
    stored_layers, metadata_key = _metadata_layers(
        z,
        candidates=(
            f"{key}_layers",
            "output_cotangent_layers",
            "step_hidden_grad_layers",
            "sv_layers",
            "layers_used",
        ),
        depth=int(example.shape[1]),
    )
    missing = [int(layer) for layer in source.layer_ids if layer not in set(stored_layers.tolist())]
    if missing:
        raise ValueError(f"{key} is missing selected layers {missing}")
    positions = [int(np.where(stored_layers == layer)[0][0]) for layer in source.layer_ids]
    output: list[np.ndarray] = []
    for row, original in enumerate(source.original_indices):
        value = _as_numeric(_record_value(raw, int(original), n_records))
        if value.ndim == 2:
            value = value[:, None, :]
        expected_steps = source.trajectories[row].shape[0]
        if value.ndim != 3 or value.shape[0] < expected_steps or value.shape[2] != source.hidden_dim:
            raise ValueError(
                f"{key}[{int(original)}] has shape {value.shape}; expected at least "
                f"[{expected_steps},layer,{source.hidden_dim}]"
            )
        selected = np.asarray(value[:expected_steps, positions, :], dtype=np.float32)
        output.append(np.ascontiguousarray(selected))
    kind = "unspecified"
    kind_key = f"{key}_kind"
    if kind_key in z.files:
        values = np.asarray(z[kind_key], dtype=object).reshape(-1)
        if values.size:
            kind = str(values[0])
    return output, key, kind, {
        "output_coupling_available": True,
        "output_cotangent_key": key,
        "output_cotangent_kind": kind,
        "output_cotangent_layer_key": metadata_key,
        "output_cotangent_layers": stored_layers.tolist(),
    }


def _empty_step_layer(source: StepGeometryDataset) -> list[np.ndarray]:
    return [
        np.full((trajectory.shape[0], source.layer_ids.size), np.nan, dtype=np.float32)
        for trajectory in source.trajectories
    ]


def _load_stored_spread(
    z: np.lib.npyio.NpzFile,
    source: StepGeometryDataset,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    output = _empty_step_layer(source)
    if "stepcloud" not in z.files or "cloud_feature_names" not in z.files:
        return output, {"stored_spread_available": False}
    names = [str(value) for value in np.asarray(z["cloud_feature_names"], dtype=object)]
    if "resultant" not in names:
        return output, {"stored_spread_available": False}
    raw = z["stepcloud"]
    n_records = int(len(z[source.vector_key]))
    example = _as_numeric(_record_value(raw, int(source.original_indices[0]), n_records))
    if example.ndim != 3:
        return output, {"stored_spread_available": False, "stored_spread_shape": list(example.shape)}
    stored_layers, layer_key = _metadata_layers(
        z,
        candidates=("layers_used", "sv_layers", "layers"),
        depth=int(example.shape[1]),
    )
    feature = names.index("resultant")
    for row, original in enumerate(source.original_indices):
        value = _as_numeric(_record_value(raw, int(original), n_records))
        if value.ndim != 3:
            continue
        for layer_pos, layer in enumerate(source.layer_ids):
            if layer not in set(stored_layers.tolist()):
                continue
            stored_pos = int(np.where(stored_layers == layer)[0][0])
            n = min(output[row].shape[0], value.shape[0])
            output[row][:n, layer_pos] = 1.0 - value[:n, stored_pos, feature]
    return output, {
        "stored_spread_available": any(np.isfinite(value).any() for value in output),
        "stored_spread_layer_key": layer_key,
        "stored_spread_layers": stored_layers.tolist(),
    }


def _load_response_clouds(
    z: np.lib.npyio.NpzFile,
    source: StepGeometryDataset,
) -> tuple[list[np.ndarray] | None, np.ndarray, dict[str, Any]]:
    if "respcloud" not in z.files:
        return None, np.empty(0, dtype=np.int64), {"response_cloud_available": False}
    if "clouds_stored" in z.files and not bool(np.asarray(z["clouds_stored"]).reshape(-1)[0]):
        return None, np.empty(0, dtype=np.int64), {"response_cloud_available": False}
    raw = z["respcloud"]
    n_records = int(len(z[source.vector_key]))
    example = _as_numeric(_record_value(raw, int(source.original_indices[0]), n_records))
    if example.ndim == 2:
        example = example[:, None, :]
    if example.ndim != 3:
        return None, np.empty(0, dtype=np.int64), {
            "response_cloud_available": False,
            "response_cloud_shape": list(example.shape),
        }
    layers, layer_key = _metadata_layers(
        z,
        candidates=("cloud_store_layers", "hidden_layers", "layers_used", "sv_layers"),
        depth=int(example.shape[1]),
    )
    clouds: list[np.ndarray] = []
    for original in source.original_indices:
        value = _as_numeric(_record_value(raw, int(original), n_records))
        if value.ndim == 2:
            value = value[:, None, :]
        clouds.append(np.ascontiguousarray(value, dtype=np.float32))
    return clouds, layers, {
        "response_cloud_available": True,
        "response_cloud_layer_key": layer_key,
        "response_cloud_layers": layers.tolist(),
    }


def load_conditional_tangent_dataset(
    path: str | Path,
    *,
    vector_key: str = "auto",
    layers: str = "all",
    max_samples: int = 0,
    output_cotangent_key: str = "auto",
) -> ConditionalTangentDataset:
    source = load_step_geometry_dataset(
        path,
        vector_key=vector_key,
        layers=layers,
        max_samples=max_samples,
    )
    with np.load(source.source_path, allow_pickle=True) as z:
        qvecs, qmeta = _load_qvecs(z, source)
        cotangents, cotangent_key, cotangent_kind, ometa = _load_output_cotangents(
            z,
            source,
            requested_key=output_cotangent_key,
        )
        spread, smeta = _load_stored_spread(z, source)
        clouds, cloud_layers, cmeta = _load_response_clouds(z, source)
    return ConditionalTangentDataset(
        source=source,
        qvecs=qvecs,
        output_cotangents=cotangents,
        output_cotangent_key=cotangent_key,
        output_cotangent_kind=cotangent_kind,
        stored_spread=spread,
        response_clouds=clouds,
        response_cloud_layer_ids=cloud_layers,
        metadata={**qmeta, **ometa, **smeta, **cmeta},
    )


def _unit_rows(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, EPS)


def _updates(
    dataset: ConditionalTangentDataset,
    *,
    reference_policy: str,
    phase_mode: str,
    causal_time_scale: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    if reference_policy not in {"correct_only", "correct_plus_pre_error"}:
        raise ValueError(
            "reference_policy must be 'correct_only' or 'correct_plus_pre_error'"
        )
    if phase_mode not in {"causal_step", "normalized_chain"}:
        raise ValueError("phase_mode must be 'causal_step' or 'normalized_chain'")
    if float(causal_time_scale) <= 0:
        raise ValueError("causal_time_scale must be positive")
    directions: list[np.ndarray] = []
    speeds: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    healthy: list[np.ndarray] = []
    for row, states in enumerate(dataset.source.trajectories):
        delta = np.empty_like(states, dtype=np.float32)
        delta[0] = states[0] - dataset.qvecs[row]
        delta[1:] = states[1:] - states[:-1]
        speed = np.linalg.norm(delta, axis=-1)
        directions.append(_unit_rows(delta))
        speeds.append(np.asarray(speed, dtype=np.float32))
        count = states.shape[0]
        clock = np.arange(count, dtype=np.float32)
        if phase_mode == "causal_step":
            clock = clock / (clock + float(causal_time_scale))
        else:
            clock = clock / max(count - 1, 1)
        phases.append(clock)
        gold = int(dataset.source.gold_error_step[row])
        if gold < 0:
            healthy.append(np.ones(count, dtype=bool))
        elif reference_policy == "correct_plus_pre_error":
            healthy.append(np.arange(count, dtype=np.int64) < gold)
        else:
            healthy.append(np.zeros(count, dtype=bool))
    return directions, speeds, phases, healthy


def _causal_step_controls(
    source: StepGeometryDataset,
    *,
    time_scale: float,
) -> list[np.ndarray]:
    controls: list[np.ndarray] = []
    for lengths in source.step_lengths:
        count = int(lengths.size)
        step = np.arange(count, dtype=np.float64)
        clock = step / (step + float(time_scale))
        previous = np.concatenate([lengths[:1], lengths[:-1]])
        cumulative = np.cumsum(lengths)
        controls.append(
            np.column_stack(
                [
                    clock,
                    clock * clock,
                    np.log1p(lengths),
                    np.log1p(previous),
                    np.log1p(cumulative),
                ]
            )
        )
    return controls


def _fold_assignments(groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    unique = np.unique(groups)
    if unique.size < 2:
        return np.zeros(groups.size, dtype=np.int64)
    n_folds = max(2, min(int(folds), int(unique.size)))
    rng = np.random.default_rng(int(seed))
    shuffled = unique[rng.permutation(unique.size)]
    mapping = {value: i % n_folds for i, value in enumerate(shuffled)}
    return np.asarray([mapping[value] for value in groups], dtype=np.int64)


def _nearest_phase_reference(
    chain_phases: np.ndarray,
    chain_directions: np.ndarray,
    chain_healthy: np.ndarray,
    target_phase: float,
) -> tuple[np.ndarray, float] | None:
    valid = np.where(chain_healthy & np.isfinite(chain_directions).all(axis=1))[0]
    if valid.size == 0:
        return None
    local = int(valid[np.argmin(np.abs(chain_phases[valid] - float(target_phase)))])
    return chain_directions[local], float(abs(chain_phases[local] - float(target_phase)))


def _softmax_weights(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values)
    weights = np.exp(np.clip(values, -60.0, 60.0))
    weights /= max(float(np.sum(weights)), EPS)
    return weights.astype(np.float32)


def _build_jobs(
    *,
    test_rows: np.ndarray,
    train_rows: np.ndarray,
    layer: int,
    directions: Sequence[np.ndarray],
    phases: Sequence[np.ndarray],
    healthy: Sequence[np.ndarray],
    qvecs_unit: Sequence[np.ndarray],
    cfg: ConditionalTangentConfig,
    mode: str,
    seed: int,
) -> list[_ReferenceJob]:
    if train_rows.size < max(cfg.neighbors, cfg.tangent_rank + 2):
        return []
    target_device = _effective_device(cfg.device)
    train_q = torch.as_tensor(
        np.stack([qvecs_unit[int(row)][layer] for row in train_rows]),
        dtype=torch.float32,
        device=target_device,
    )
    test_q = torch.as_tensor(
        np.stack([qvecs_unit[int(row)][layer] for row in test_rows]),
        dtype=torch.float32,
        device=target_device,
    )
    similarities = (test_q @ train_q.T).detach().cpu().numpy()
    if mode == "shuffled_question":
        rng = np.random.default_rng(int(seed))
        similarities = similarities[:, rng.permutation(train_rows.size)]
    jobs: list[_ReferenceJob] = []
    search_k = min(
        train_rows.size,
        max(int(cfg.neighbors), int(cfg.neighbors) * int(cfg.search_multiplier)),
    )
    minimum = max(int(cfg.tangent_rank) + 2, 4)
    for test_pos, row_value in enumerate(test_rows):
        row = int(row_value)
        sim = similarities[test_pos]
        if mode in {"question_phase", "shuffled_question"}:
            order = np.argsort(-sim, kind="mergesort")[:search_k]
        elif mode == "phase_only":
            order = np.arange(train_rows.size, dtype=np.int64)
        else:
            raise ValueError(mode)
        for step, target_phase in enumerate(phases[row]):
            available: list[tuple[float, int, np.ndarray, float]] = []
            for position in order:
                train_row = int(train_rows[int(position)])
                item = _nearest_phase_reference(
                    phases[train_row],
                    directions[train_row][:, layer, :],
                    healthy[train_row],
                    float(target_phase),
                )
                if item is None:
                    continue
                ref, phase_distance = item
                if mode == "phase_only":
                    priority = -phase_distance
                    q_score = 0.0
                else:
                    priority = float(sim[int(position)])
                    q_score = priority / max(float(cfg.q_temperature), EPS)
                available.append((priority, train_row, ref, phase_distance + 0.0 * q_score))
            if mode == "phase_only":
                available.sort(key=lambda item: item[3])
            else:
                available.sort(key=lambda item: item[0], reverse=True)
            selected = available[: int(cfg.neighbors)]
            if len(selected) < minimum:
                continue
            references = np.stack([item[2] for item in selected]).astype(np.float16)
            phase_distance = np.asarray([item[3] for item in selected], dtype=np.float64)
            if mode == "phase_only":
                logits = -0.5 * (phase_distance / max(float(cfg.phase_sigma), EPS)) ** 2
            else:
                selected_positions = [int(np.where(train_rows == item[1])[0][0]) for item in selected]
                q_logits = sim[np.asarray(selected_positions, dtype=np.int64)] / max(
                    float(cfg.q_temperature), EPS
                )
                logits = q_logits - 0.5 * (
                    phase_distance / max(float(cfg.phase_sigma), EPS)
                ) ** 2
            jobs.append(
                _ReferenceJob(
                    row=row,
                    step=int(step),
                    target=np.asarray(directions[row][step, layer], dtype=np.float32),
                    references=references,
                    weights=_softmax_weights(logits),
                )
            )
    return jobs


@torch.inference_mode()
def _score_reference_jobs(
    jobs: Sequence[_ReferenceJob],
    *,
    rank: int,
    device: str,
    batch_size: int,
) -> _ReferenceScore:
    n = len(jobs)
    escape = np.full(n, np.nan, dtype=np.float32)
    isotropic = np.full(n, np.nan, dtype=np.float32)
    nearest = np.full(n, np.nan, dtype=np.float32)
    rank_energy = np.full(n, np.nan, dtype=np.float32)
    effective_rank = np.full(n, np.nan, dtype=np.float32)
    residuals = [np.empty(0, dtype=np.float32) for _ in range(n)]
    target_device = _effective_device(device)
    chunk_size = max(1, int(batch_size))
    for start in range(0, n, chunk_size):
        chunk = jobs[start : start + chunk_size]
        refs = torch.as_tensor(
            np.stack([job.references for job in chunk]),
            dtype=torch.float32,
            device=target_device,
        )
        weights = torch.as_tensor(
            np.stack([job.weights for job in chunk]),
            dtype=torch.float32,
            device=target_device,
        )
        target = torch.as_tensor(
            np.stack([job.target for job in chunk]),
            dtype=torch.float32,
            device=target_device,
        )
        weighted = torch.sqrt(weights.clamp_min(0.0))[:, :, None] * refs
        gram = weighted @ weighted.transpose(1, 2)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.clamp_min(0.0)
        use_rank = min(int(rank), int(refs.shape[1]), int(refs.shape[2]))
        values = eigenvalues[:, -use_rank:]
        vectors = eigenvectors[:, :, -use_rank:]
        basis = weighted.transpose(1, 2) @ vectors
        mode_floor = torch.maximum(
            eigenvalues[:, -1:] * 1e-6,
            torch.full_like(eigenvalues[:, -1:], EPS),
        )
        valid_modes = values > mode_floor
        basis = basis / torch.sqrt(values.clamp_min(EPS))[:, None, :]
        basis = basis * valid_modes[:, None, :]
        coefficients = torch.einsum("bdr,bd->br", basis, target)
        projection = torch.einsum("bdr,br->bd", basis, coefficients)
        residual = target - projection
        local_escape = torch.sum(residual * residual, dim=1).clamp(0.0, 1.25)
        cosine = torch.einsum("bkd,bd->bk", refs, target)
        local_isotropic = torch.sum(weights * (1.0 - cosine), dim=1)
        local_nearest = 1.0 - torch.max(cosine, dim=1).values
        total = torch.sum(eigenvalues, dim=1).clamp_min(EPS)
        local_rank_energy = torch.sum(values, dim=1) / total
        probability = eigenvalues / total[:, None]
        entropy = -torch.sum(
            torch.where(
                probability > EPS,
                probability * torch.log(probability.clamp_min(EPS)),
                torch.zeros_like(probability),
            ),
            dim=1,
        )
        local_effective_rank = torch.exp(entropy)
        sl = slice(start, start + len(chunk))
        escape[sl] = local_escape.detach().cpu().numpy()
        isotropic[sl] = local_isotropic.detach().cpu().numpy()
        nearest[sl] = local_nearest.detach().cpu().numpy()
        rank_energy[sl] = local_rank_energy.detach().cpu().numpy()
        effective_rank[sl] = local_effective_rank.detach().cpu().numpy()
        residual_cpu = residual.detach().cpu().numpy().astype(np.float32)
        for offset, value in enumerate(residual_cpu):
            residuals[start + offset] = value
        del refs, weights, target, weighted, gram, eigenvalues, eigenvectors, basis
    return _ReferenceScore(
        escape=escape,
        isotropic=isotropic,
        nearest=nearest,
        top_rank_energy=rank_energy,
        effective_rank=effective_rank,
        residuals=residuals,
    )


@torch.inference_mode()
def _fit_fixed_basis(
    references: np.ndarray,
    *,
    rank: int,
    device: str,
) -> np.ndarray:
    target_device = _effective_device(device)
    refs = torch.as_tensor(references, dtype=torch.float32, device=target_device)
    refs = refs / torch.linalg.vector_norm(refs, dim=1, keepdim=True).clamp_min(EPS)
    gram = refs @ refs.T
    values, vectors = torch.linalg.eigh(gram)
    use_rank = min(int(rank), refs.shape[0], refs.shape[1])
    values = values[-use_rank:]
    vectors = vectors[:, -use_rank:]
    mode_floor = torch.maximum(
        values[-1:] * 1e-6,
        torch.full_like(values[-1:], EPS),
    )
    valid_modes = values > mode_floor
    basis = refs.T @ vectors / torch.sqrt(values.clamp_min(EPS))[None, :]
    basis = basis * valid_modes[None, :]
    return basis.detach().cpu().numpy().astype(np.float32)


@torch.inference_mode()
def _score_fixed_basis(
    targets: np.ndarray,
    basis: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float32)
    output = np.full(values.shape[0], np.nan, dtype=np.float32)
    target_device = _effective_device(device)
    basis_tensor = torch.as_tensor(basis, dtype=torch.float32, device=target_device)
    chunk_size = max(1, int(batch_size))
    for start in range(0, values.shape[0], chunk_size):
        chunk = torch.as_tensor(
            values[start : start + chunk_size],
            dtype=torch.float32,
            device=target_device,
        )
        coefficients = chunk @ basis_tensor
        score = (1.0 - torch.sum(coefficients * coefficients, dim=1)).clamp(0.0, 1.25)
        output[start : start + chunk.shape[0]] = score.detach().cpu().numpy()
    return output


def _assign_jobs(
    fields: list[np.ndarray],
    metric_index: dict[str, int],
    normal_vectors: list[np.ndarray],
    jobs: Sequence[_ReferenceJob],
    scores: _ReferenceScore,
    *,
    layer: int,
    primary: bool,
) -> None:
    for index, job in enumerate(jobs):
        if primary:
            fields[job.row][job.step, layer, metric_index["qpt_escape_ratio"]] = scores.escape[index]
            fields[job.row][job.step, layer, metric_index["qpt_isotropic_distance"]] = scores.isotropic[index]
            fields[job.row][job.step, layer, metric_index["qpt_nearest_direction_distance"]] = scores.nearest[index]
            fields[job.row][job.step, layer, metric_index["qpt_reference_topr_energy"]] = scores.top_rank_energy[index]
            fields[job.row][job.step, layer, metric_index["qpt_reference_effective_rank"]] = scores.effective_rank[index]
            normal_vectors[job.row][job.step, layer] = scores.residuals[index]


def _fill_persistence_metrics(
    fields: list[np.ndarray],
    normal_vectors: Sequence[np.ndarray],
    metric_index: dict[str, int],
    window: int,
) -> None:
    width = max(1, int(window))
    for row, values in enumerate(normal_vectors):
        for layer in range(values.shape[1]):
            residual = np.asarray(values[:, layer, :], dtype=np.float32)
            valid = np.isfinite(residual).all(axis=1)
            for step in range(values.shape[0]):
                start = max(0, step - width + 1)
                mask = valid[start : step + 1]
                if not np.any(mask):
                    continue
                local = residual[start : step + 1][mask]
                squared = np.sum(local * local, axis=1)
                total = float(np.sum(squared))
                mean_residual = np.mean(local, axis=0)
                fields[row][step, layer, metric_index["window_normal_energy"]] = float(
                    np.mean(squared)
                )
                fields[row][step, layer, metric_index["coherent_normal_drift"]] = float(
                    np.dot(mean_residual, mean_residual)
                )
                fields[row][step, layer, metric_index["normal_persistence"]] = float(
                    np.dot(np.sum(local, axis=0), np.sum(local, axis=0))
                    / max(local.shape[0] * total, EPS)
                )


def _fill_output_coupling(
    dataset: ConditionalTangentDataset,
    fields: list[np.ndarray],
    normal_vectors: Sequence[np.ndarray],
    directions: Sequence[np.ndarray],
    metric_index: dict[str, int],
) -> None:
    if dataset.output_cotangents is None:
        return
    for row, gradients in enumerate(dataset.output_cotangents):
        for step in range(gradients.shape[0]):
            for layer in range(gradients.shape[1]):
                gradient = np.asarray(gradients[step, layer], dtype=np.float32)
                residual = np.asarray(normal_vectors[row][step, layer], dtype=np.float32)
                if not np.isfinite(gradient).all() or not np.isfinite(residual).all():
                    continue
                gnorm = float(np.linalg.norm(gradient))
                if gnorm <= EPS:
                    continue
                g = gradient / gnorm
                update = np.asarray(directions[row][step, layer], dtype=np.float32)
                tangent = update - residual
                transverse = float(np.dot(g, residual) ** 2)
                tangential = float(np.dot(g, tangent) ** 2)
                normal_energy = float(np.dot(residual, residual))
                tangent_energy = float(np.dot(tangent, tangent))
                normal_alignment = transverse / max(normal_energy, EPS)
                tangent_alignment = tangential / max(tangent_energy, EPS)
                fields[row][step, layer, metric_index["output_transverse_energy"]] = transverse
                fields[row][step, layer, metric_index["output_tangent_energy"]] = tangential
                fields[row][step, layer, metric_index["output_transverse_fraction"]] = (
                    transverse / max(transverse + tangential, EPS)
                )
                fields[row][step, layer, metric_index["output_normal_alignment"]] = (
                    normal_alignment
                )
                fields[row][step, layer, metric_index["output_tangent_alignment"]] = (
                    tangent_alignment
                )
                fields[row][step, layer, metric_index["output_alignment_excess"]] = (
                    normal_alignment - tangent_alignment
                )


def _compute_legacy_token_spectra(
    dataset: ConditionalTangentDataset,
    *,
    device: str,
    batch_size: int,
) -> dict[str, list[np.ndarray]]:
    source = dataset.source
    output = {name: _empty_step_layer(source) for name in LEGACY_METRIC_NAMES}
    directions, speeds, _, _ = _updates(
        dataset,
        reference_policy="correct_only",
        phase_mode="causal_step",
        causal_time_scale=4.0,
    )
    del directions
    for row, value in enumerate(speeds):
        output["update_speed"][row][:] = value
    for row in range(source.n_samples):
        output["direction_spread"][row][:] = dataset.stored_spread[row]
    if dataset.response_clouds is None:
        return output
    cloud_layer_to_pos = {
        int(layer): position for position, layer in enumerate(dataset.response_cloud_layer_ids)
    }
    jobs: list[tuple[int, int, int, np.ndarray]] = []
    for row, cloud in enumerate(dataset.response_clouds):
        response_start = int(source.step_ranges[row][0, 0])
        for step, (start, end) in enumerate(source.step_ranges[row]):
            lo = max(0, int(start) - response_start)
            hi = min(cloud.shape[0], int(end) - response_start + 1)
            if hi - lo < 2:
                continue
            for layer_pos, layer in enumerate(source.layer_ids):
                cloud_pos = cloud_layer_to_pos.get(int(layer))
                if cloud_pos is None:
                    continue
                token = np.asarray(cloud[lo:hi, cloud_pos, :], dtype=np.float32)
                if np.isfinite(token).all():
                    jobs.append((row, step, layer_pos, token))
    target_device = _effective_device(device)
    chunk_size = max(1, int(batch_size))
    for start in range(0, len(jobs), chunk_size):
        chunk = jobs[start : start + chunk_size]
        max_tokens = max(item[3].shape[0] for item in chunk)
        hidden = chunk[0][3].shape[1]
        values = torch.zeros(
            (len(chunk), max_tokens, hidden), dtype=torch.float32, device=target_device
        )
        weights = torch.zeros((len(chunk), max_tokens), dtype=torch.float32, device=target_device)
        counts = []
        for local, (_, _, _, token) in enumerate(chunk):
            count = token.shape[0]
            counts.append(count)
            values[local, :count] = torch.as_tensor(token, dtype=torch.float32, device=target_device)
            position = torch.arange(count, dtype=torch.float32, device=target_device)
            raw_weight = torch.exp(position / max(count - 1, 1))
            weights[local, :count] = raw_weight / torch.sum(raw_weight)
        unit = values / torch.linalg.vector_norm(values, dim=2, keepdim=True).clamp_min(EPS)
        resultant = torch.linalg.vector_norm(torch.sum(weights[:, :, None] * unit, dim=1), dim=1)
        weighted = torch.sqrt(weights)[:, :, None] * unit
        gram = weighted @ weighted.transpose(1, 2)
        eigen = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        total = torch.sum(eigen, dim=1).clamp_min(EPS)
        probability = eigen / total[:, None]
        entropy = -torch.sum(
            torch.where(
                probability > EPS,
                probability * torch.log(probability.clamp_min(EPS)),
                torch.zeros_like(probability),
            ),
            dim=1,
        )
        effective_rank = torch.exp(entropy)
        for local, (row, step, layer_pos, _) in enumerate(chunk):
            count = counts[local]
            h = float(entropy[local].item())
            output["direction_resultant_jl"][row][step, layer_pos] = float(
                resultant[local].item()
            )
            output["direction_spec_entropy_raw"][row][step, layer_pos] = h
            output["direction_spec_entropy_norm"][row][step, layer_pos] = h / max(
                float(np.log(max(count, 2))), EPS
            )
            output["direction_effective_rank_norm"][row][step, layer_pos] = float(
                effective_rank[local].item() / max(count, 1)
            )
        del values, weights, unit, weighted, gram, eigen
    return output


def run_conditional_tangent_audit(
    dataset: ConditionalTangentDataset,
    cfg: ConditionalTangentConfig,
    *,
    include_legacy_directional: bool = True,
) -> ConditionalTangentResult:
    source = dataset.source
    if int(cfg.tangent_rank) < 1 or int(cfg.tangent_rank) > source.hidden_dim:
        raise ValueError(
            f"tangent_rank must be in [1,{source.hidden_dim}], got {cfg.tangent_rank}"
        )
    metric_names = BASE_METRIC_NAMES + (
        OUTPUT_METRIC_NAMES if dataset.output_cotangents is not None else ()
    )
    metric_index = {name: index for index, name in enumerate(metric_names)}
    fields = [
        np.full(
            (trajectory.shape[0], source.layer_ids.size, len(metric_names)),
            np.nan,
            dtype=np.float32,
        )
        for trajectory in source.trajectories
    ]
    normal_dtype = (
        np.float32 if dataset.output_cotangents is not None else np.float16
    )
    normal_vectors = [
        np.full(
            (trajectory.shape[0], source.layer_ids.size, source.hidden_dim),
            np.nan,
            dtype=normal_dtype,
        )
        for trajectory in source.trajectories
    ]
    directions, speeds, phases, healthy = _updates(
        dataset,
        reference_policy=cfg.reference_policy,
        phase_mode=cfg.phase_mode,
        causal_time_scale=cfg.causal_time_scale,
    )
    qvecs_unit = [_unit_rows(value) for value in dataset.qvecs]
    assignments = _fold_assignments(source.problem_ids, cfg.folds, cfg.random_seed)
    rng = np.random.default_rng(int(cfg.random_seed))
    fold_diagnostics: list[dict[str, Any]] = []

    for fold in np.unique(assignments):
        train_rows = np.where(assignments != fold)[0]
        test_rows = np.where(assignments == fold)[0]
        diagnostic: dict[str, Any] = {
            "fold": int(fold),
            "train_chains": int(train_rows.size),
            "test_chains": int(test_rows.size),
            "layers": {},
        }
        for layer in range(source.layer_ids.size):
            primary_jobs = _build_jobs(
                test_rows=test_rows,
                train_rows=train_rows,
                layer=layer,
                directions=directions,
                phases=phases,
                healthy=healthy,
                qvecs_unit=qvecs_unit,
                cfg=cfg,
                mode="question_phase",
                seed=cfg.random_seed + int(fold) * 97 + layer,
            )
            primary = _score_reference_jobs(
                primary_jobs,
                rank=cfg.tangent_rank,
                device=cfg.device,
                batch_size=cfg.batch_size,
            )
            _assign_jobs(
                fields,
                metric_index,
                normal_vectors,
                primary_jobs,
                primary,
                layer=layer,
                primary=True,
            )
            primary_count = len(primary_jobs)
            del primary, primary_jobs

            shuffled_jobs = _build_jobs(
                test_rows=test_rows,
                train_rows=train_rows,
                layer=layer,
                directions=directions,
                phases=phases,
                healthy=healthy,
                qvecs_unit=qvecs_unit,
                cfg=cfg,
                mode="shuffled_question",
                seed=cfg.random_seed + int(fold) * 101 + layer,
            )
            shuffled = _score_reference_jobs(
                shuffled_jobs,
                rank=cfg.tangent_rank,
                device=cfg.device,
                batch_size=cfg.batch_size,
            )
            for index, job in enumerate(shuffled_jobs):
                fields[job.row][job.step, layer, metric_index["shuffled_question_escape_ratio"]] = shuffled.escape[index]
            shuffled_count = len(shuffled_jobs)
            del shuffled, shuffled_jobs

            phase_jobs = _build_jobs(
                test_rows=test_rows,
                train_rows=train_rows,
                layer=layer,
                directions=directions,
                phases=phases,
                healthy=healthy,
                qvecs_unit=qvecs_unit,
                cfg=cfg,
                mode="phase_only",
                seed=cfg.random_seed + int(fold) * 103 + layer,
            )
            phase_only = _score_reference_jobs(
                phase_jobs,
                rank=cfg.tangent_rank,
                device=cfg.device,
                batch_size=cfg.batch_size,
            )
            for index, job in enumerate(phase_jobs):
                fields[job.row][job.step, layer, metric_index["phase_only_escape_ratio"]] = phase_only.escape[index]
            phase_count = len(phase_jobs)
            del phase_only, phase_jobs

            train_refs = []
            for row in train_rows:
                valid = healthy[int(row)] & np.isfinite(directions[int(row)][:, layer]).all(axis=1)
                train_refs.extend(directions[int(row)][valid, layer].tolist())
            if train_refs:
                refs = np.asarray(train_refs, dtype=np.float32)
                if refs.shape[0] > int(cfg.global_reference_cap):
                    choose = rng.choice(
                        refs.shape[0], int(cfg.global_reference_cap), replace=False
                    )
                    refs = refs[choose]
                global_basis = _fit_fixed_basis(
                    refs,
                    rank=cfg.tangent_rank,
                    device=cfg.device,
                )
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    int(cfg.random_seed) + int(fold) * 1009 + layer * 37
                )
                random_matrix = torch.randn(
                    source.hidden_dim,
                    int(cfg.tangent_rank),
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                ).to(_effective_device(cfg.device))
                random_basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
                random_basis = random_basis.detach().cpu().numpy().astype(np.float32)
                for row in test_rows:
                    target = directions[int(row)][:, layer]
                    global_score = _score_fixed_basis(
                        target,
                        global_basis,
                        device=cfg.device,
                        batch_size=cfg.batch_size,
                    )
                    random_score = _score_fixed_basis(
                        target,
                        random_basis[:, : int(cfg.tangent_rank)],
                        device=cfg.device,
                        batch_size=cfg.batch_size,
                    )
                    fields[int(row)][:, layer, metric_index["global_escape_ratio"]] = global_score
                    fields[int(row)][:, layer, metric_index["random_escape_ratio"]] = random_score

            diagnostic["layers"][str(int(source.layer_ids[layer]))] = {
                "primary_jobs": int(primary_count),
                "shuffled_jobs": int(shuffled_count),
                "phase_only_jobs": int(phase_count),
                "healthy_references": int(len(train_refs)),
            }
        fold_diagnostics.append(diagnostic)

    for row in range(source.n_samples):
        primary = fields[row][:, :, metric_index["qpt_escape_ratio"]]
        shuffled = fields[row][:, :, metric_index["shuffled_question_escape_ratio"]]
        fields[row][:, :, metric_index["question_conditioning_excess"]] = primary - shuffled
    _fill_persistence_metrics(
        fields,
        normal_vectors,
        metric_index,
        cfg.persistence_window,
    )
    _fill_output_coupling(
        dataset,
        fields,
        normal_vectors,
        directions,
        metric_index,
    )

    source_matches = match_correct_pseudo_events(source)
    axis = make_step_axis(source)
    axis.controls = _causal_step_controls(
        source,
        time_scale=cfg.causal_time_scale,
    )
    axis.metadata = {
        **axis.metadata,
        "control_names": [
            "causal_step_clock",
            "causal_step_clock_sq",
            "log1p_current_step_length",
            "log1p_previous_step_length",
            "log1p_cumulative_seen_tokens",
        ],
        "control_protocol": (
            "causal nuisance controls only; no next-step length or final chain length"
        ),
    }
    matches = map_matches_to_axis(source_matches, source, axis)
    residual_fields = crossfit_correct_nuisance_residuals(
        fields,
        axis,
        matches=matches,
        device=cfg.device,
        folds=cfg.folds,
        ridge=cfg.nuisance_ridge,
        seed=cfg.random_seed,
    )
    legacy = (
        _compute_legacy_token_spectra(
            dataset,
            device=cfg.device,
            batch_size=cfg.batch_size,
        )
        if include_legacy_directional
        else {}
    )
    legacy_residual: dict[str, list[np.ndarray]] = {}
    if legacy:
        packed_legacy = [
            np.stack([legacy[name][row] for name in LEGACY_METRIC_NAMES], axis=2)
            for row in range(source.n_samples)
        ]
        packed_legacy_residual = crossfit_correct_nuisance_residuals(
            packed_legacy,
            axis,
            matches=matches,
            device=cfg.device,
            folds=cfg.folds,
            ridge=cfg.nuisance_ridge,
            seed=cfg.random_seed + 211,
        )
        for metric, name in enumerate(LEGACY_METRIC_NAMES):
            legacy_residual[name] = [
                value[:, :, metric] for value in packed_legacy_residual
            ]
    return ConditionalTangentResult(
        dataset=dataset,
        axis=axis,
        metric_names=metric_names,
        fields=fields,
        residual_fields=residual_fields,
        normal_vectors=normal_vectors,
        legacy_fields=legacy,
        legacy_residual_fields=legacy_residual,
        matches=matches,
        metadata={
            "requested_device": cfg.device,
            "effective_device": str(_effective_device(cfg.device)),
            "folds": int(cfg.folds),
            "neighbors": int(cfg.neighbors),
            "tangent_rank": int(cfg.tangent_rank),
            "q_temperature": float(cfg.q_temperature),
            "phase_sigma": float(cfg.phase_sigma),
            "phase_mode": cfg.phase_mode,
            "causal_time_scale": float(cfg.causal_time_scale),
            "persistence_window": int(cfg.persistence_window),
            "reference_policy": cfg.reference_policy,
            "healthy_reference_policy": (
                "all steps from fully correct training-fold chains only; target problems "
                "are group-held-out"
                if cfg.reference_policy == "correct_only"
                else "all steps from fully correct training-fold chains plus strictly "
                "pre-error prefixes; target problems are group-held-out"
            ),
            "first_step_anchor": "qvec; delta_0 = stepvec_0 - qvec",
            "causal_nuisance_controls": axis.metadata["control_names"],
            "fold_diagnostics": fold_diagnostics,
            "output_gate": (
                "available" if dataset.output_cotangents is not None else "not_tested_missing_cotangent"
            ),
            "output_cotangent_kind": dataset.output_cotangent_kind,
            "speed_summary": {
                "mean": float(np.mean(np.concatenate([value.reshape(-1) for value in speeds]))),
                "max": float(np.max(np.concatenate([value.reshape(-1) for value in speeds]))),
            },
        },
    )
