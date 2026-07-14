from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


STATE_GEOMETRY_NAMES = (
    "depth.path_length_mean",
    "depth.endpoint_displacement",
    "depth.tortuosity",
    "depth.turn_angle_mean",
    "depth.direction_dispersion",
    "depth.spectral_entropy",
    "depth.effective_rank",
    "depth.update_anisotropy",
    "temporal.velocity_mean",
    "temporal.velocity_cv",
    "temporal.direction_dispersion",
    "temporal.layer_transport_misalignment",
    "temporal.turn_angle_mean",
    "temporal.spectral_entropy",
    "coupling.depth_time_misalignment",
    "final_control.velocity",
    "final_control.turn_angle",
)


@dataclass
class GeometryCollection:
    chain_idx: np.ndarray
    problem_id: np.ndarray
    gold_error_step: np.ndarray
    is_correct: np.ndarray
    response_hash: np.ndarray
    matrices: list[np.ndarray]
    feature_names: tuple[str, ...]
    feature_groups: tuple[str, ...]
    layers: tuple[int, ...]
    preflight: dict[str, Any]

    def validate(self) -> None:
        n = len(self.chain_idx)
        if any(
            len(values) != n
            for values in (
                self.problem_id,
                self.gold_error_step,
                self.is_correct,
                self.response_hash,
                self.matrices,
            )
        ):
            raise ValueError("geometry collection chain metadata is inconsistent")
        if len(set(int(x) for x in self.chain_idx)) != n:
            raise ValueError("geometry chain_idx values must be unique")
        if len(self.feature_names) != len(self.feature_groups):
            raise ValueError("every geometry feature must have one group")
        for matrix in self.matrices:
            value = np.asarray(matrix)
            if value.ndim != 2 or value.shape[1] != len(self.feature_names):
                raise ValueError("geometry matrix disagrees with feature schema")


def _safe_unit(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(b, dim=-1)
    return (a * b).sum(dim=-1) / denom.clamp_min(eps)


def _spectral_statistics(
    delta: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, ...]:
    """Entropy, effective rank, and leading-energy ratio of row-vector clouds."""

    gram = delta @ delta.transpose(-1, -2)
    eigenvalues = torch.linalg.eigvalsh(gram.float()).clamp_min(0.0)
    probability = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy_nats = -(probability * torch.log(probability.clamp_min(eps))).sum(dim=-1)
    max_entropy = np.log(max(int(delta.shape[-2]), 2))
    entropy_norm = entropy_nats / max_entropy
    effective_rank = torch.exp(entropy_nats)
    anisotropy = eigenvalues[..., -1] / eigenvalues.sum(dim=-1).clamp_min(eps)
    return entropy_norm, effective_rank, anisotropy


@torch.inference_mode()
def compute_state_geometry(
    states: np.ndarray,
    *,
    compute_device: str = "cpu",
) -> np.ndarray:
    """Compute gauge-safe geometry from one or more layer-time state fields.

    ``states`` may be ``[step, layer, hidden]`` or
    ``[batch, step, layer, hidden]``. All coordinates are normalized before
    differences are taken. The resulting features are invariant to a global
    orthogonal basis change and to one global positive rescaling of the stored
    hidden states. No label enters this map.
    """

    array = np.asarray(states)
    squeeze_batch = array.ndim == 3
    if squeeze_batch:
        array = array[None, ...]
    if array.ndim != 4 or min(array.shape[:3]) < 1:
        raise ValueError(
            "states must have shape [step, layer, hidden] or "
            "[batch, step, layer, hidden]"
        )
    if array.shape[2] < 3:
        raise ValueError("at least three layer states are required for depth geometry")
    device = torch.device(compute_device)
    h = torch.as_tensor(array, dtype=torch.float32, device=device)
    u = _safe_unit(h)
    batch, steps, _, _ = u.shape

    depth_delta = u[:, :, 1:] - u[:, :, :-1]
    depth_norm = torch.linalg.vector_norm(depth_delta, dim=-1)
    path_mean = depth_norm.mean(dim=-1)
    path_sum = depth_norm.sum(dim=-1)
    endpoint = torch.linalg.vector_norm(u[:, :, -1] - u[:, :, 0], dim=-1)
    tortuosity = path_sum / endpoint.clamp_min(1e-8)
    depth_unit = _safe_unit(depth_delta)
    direction_dispersion = 1.0 - torch.linalg.vector_norm(
        depth_unit.mean(dim=2), dim=-1
    )
    depth_turn = torch.acos(
        _cosine(depth_delta[:, :, 1:], depth_delta[:, :, :-1]).clamp(-1.0, 1.0)
    ).mean(dim=-1)
    depth_entropy, depth_rank, depth_anisotropy = _spectral_statistics(depth_delta)

    nan = torch.full((batch, steps), float("nan"), device=device)
    temporal_velocity = nan.clone()
    temporal_cv = nan.clone()
    temporal_dispersion = nan.clone()
    transport_misalignment = nan.clone()
    temporal_turn = nan.clone()
    temporal_entropy = nan.clone()
    coupling = nan.clone()
    final_velocity = nan.clone()
    final_turn = nan.clone()
    if steps >= 2:
        time_delta = u[:, 1:] - u[:, :-1]
        time_norm = torch.linalg.vector_norm(time_delta, dim=-1)
        temporal_velocity[:, 1:] = time_norm.mean(dim=-1)
        temporal_cv[:, 1:] = time_norm.std(dim=-1, unbiased=False) / time_norm.mean(
            dim=-1
        ).clamp_min(1e-8)
        temporal_dispersion[:, 1:] = 1.0 - torch.linalg.vector_norm(
            _safe_unit(time_delta).mean(dim=2), dim=-1
        )
        transport_misalignment[:, 1:] = 1.0 - _cosine(
            time_delta[:, :, 1:], time_delta[:, :, :-1]
        ).mean(dim=-1)
        temporal_entropy[:, 1:] = _spectral_statistics(time_delta)[0]
        coupling[:, 1:] = 1.0 - _cosine(time_delta[:, :, :-1], depth_delta[:, 1:]).mean(
            dim=-1
        )
        final_velocity[:, 1:] = torch.linalg.vector_norm(time_delta[:, :, -1], dim=-1)
    if steps >= 3:
        time_delta = u[:, 1:] - u[:, :-1]
        temporal_turn[:, 2:] = torch.acos(
            _cosine(time_delta[:, 1:], time_delta[:, :-1]).clamp(-1.0, 1.0)
        ).mean(dim=-1)
        final_turn[:, 2:] = torch.acos(
            _cosine(time_delta[:, 1:, -1], time_delta[:, :-1, -1]).clamp(-1.0, 1.0)
        )

    result = torch.stack(
        (
            path_mean,
            endpoint,
            tortuosity,
            depth_turn,
            direction_dispersion,
            depth_entropy,
            depth_rank,
            depth_anisotropy,
            temporal_velocity,
            temporal_cv,
            temporal_dispersion,
            transport_misalignment,
            temporal_turn,
            temporal_entropy,
            coupling,
            final_velocity,
            final_turn,
        ),
        dim=-1,
    )
    output = result.detach().cpu().numpy().astype(np.float32, copy=False)
    return output[0] if squeeze_batch else output


def geometry_group(name: str) -> str:
    if name.startswith("depth."):
        return "depth"
    if name.startswith("temporal."):
        return "temporal"
    if name.startswith("coupling."):
        return "coupling"
    if name.startswith("icr.") or "icr" in name:
        return "icr"
    if name.startswith("final_control."):
        return "final_control"
    if name.startswith("legacy."):
        return "legacy_geometry"
    return "mechanism_geometry"


def _as_chain_metadata(z: np.lib.npyio.NpzFile, n: int) -> tuple[np.ndarray, ...]:
    chain_idx = (
        np.asarray(z["chain_idx"], dtype=np.int64)
        if "chain_idx" in z.files
        else np.arange(n)
    )
    if "problem_id" in z.files:
        problem_id = np.asarray(z["problem_id"], dtype=np.int64)
    elif "problem_ids" in z.files:
        problem_id = np.asarray(z["problem_ids"], dtype=np.int64)
    else:
        problem_id = chain_idx.copy()
    gold = (
        np.asarray(z["gold_error_step"], dtype=np.int64)
        if "gold_error_step" in z.files
        else np.full(n, -1, dtype=np.int64)
    )
    if "is_correct" in z.files:
        correct = np.asarray(z["is_correct"], dtype=np.int8)
    elif "is_correct_strict" in z.files:
        correct = np.asarray(z["is_correct_strict"], dtype=np.int8)
    else:
        correct = (gold < 0).astype(np.int8)
    return chain_idx, problem_id, gold, correct


def _response_hashes(z: np.lib.npyio.NpzFile, n: int) -> np.ndarray:
    values = None
    for key in ("responses", "response"):
        if key in z.files:
            values = z[key]
            break
    if values is None or len(values) != n:
        return np.asarray([""] * n, dtype=str)
    return np.asarray(
        [hashlib.sha256(str(values[i]).encode("utf-8")).hexdigest() for i in range(n)],
        dtype=str,
    )


def _scalar_string(z: np.lib.npyio.NpzFile, keys: Sequence[str]) -> str:
    for key in keys:
        if key not in z.files:
            continue
        try:
            value = np.asarray(z[key], dtype=object)
            if value.ndim == 0:
                return str(value.item())
            unique = sorted({str(item) for item in value.reshape(-1) if str(item)})
            if len(unique) == 1:
                return unique[0]
        except Exception:
            continue
    return ""


def _first_valid_object_shape(values: np.ndarray) -> tuple[int, ...]:
    for value in values:
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.size:
            return tuple(int(x) for x in arr.shape)
    return ()


def _load_state_sequences(
    z: np.lib.npyio.NpzFile,
    path: Path,
) -> tuple[list[np.ndarray | None], tuple[int, ...], str]:
    if "step_layer_state_vectors" in z.files:
        values = np.asarray(z["step_layer_state_vectors"])
        chain_ids = np.asarray(z["step_layer_state_vector_chain_idx"], dtype=np.int64)
        step_ids = np.asarray(z["step_layer_state_vector_step_idx"], dtype=np.int64)
        layers = tuple(int(x) for x in z["step_layer_state_vector_layers"])
        source = "packed_whole_layer_states"
    elif "step_layer_state_memmap_path" in z.files:
        memmap_path = path.parent / str(
            np.asarray(z["step_layer_state_memmap_path"]).item()
        )
        values = np.load(memmap_path, mmap_mode="r")
        chain_ids = np.asarray(z["step_layer_state_vector_chain_idx"], dtype=np.int64)
        step_ids = np.asarray(z["step_layer_state_vector_step_idx"], dtype=np.int64)
        layers = tuple(int(x) for x in z["step_layer_state_vector_layers"])
        source = "memmap_whole_layer_states"
    elif "stepvec" in z.files:
        objects = np.asarray(z["stepvec"], dtype=object)
        layers = tuple(int(x) for x in z["sv_layers"]) if "sv_layers" in z.files else ()
        sequences: list[np.ndarray | None] = []
        for value in objects:
            if value is None:
                sequences.append(None)
                continue
            arr = np.asarray(value)
            if arr.dtype == object:
                arr = np.asarray(arr.tolist())
            sequences.append(arr if arr.ndim == 3 and arr.size else None)
        return sequences, layers, "legacy_sparse_stepvec"
    else:
        return [], (), "none"

    unique_chain = np.unique(chain_ids)
    lookup: dict[int, np.ndarray] = {}
    for chain in unique_chain:
        take = np.where(chain_ids == chain)[0]
        order = take[np.argsort(step_ids[take], kind="stable")]
        lookup[int(chain)] = np.asarray(values[order])
    record_chain_ids = (
        np.asarray(z["chain_idx"], dtype=np.int64)
        if "chain_idx" in z.files
        else unique_chain
    )
    sequences = [lookup.get(int(chain)) for chain in record_chain_ids]
    return sequences, layers, source


def _mechanism_sequences(
    z: np.lib.npyio.NpzFile,
    chain_ids: np.ndarray,
    n_steps: Sequence[int],
) -> tuple[list[np.ndarray], list[str]]:
    if "step_scores" not in z.files or "step_score_names" not in z.files:
        return [np.empty((int(t), 0), dtype=np.float32) for t in n_steps], []
    names = [str(x) for x in z["step_score_names"]]
    keep = [
        i
        for i, name in enumerate(names)
        if "icr" in name.lower()
        or "transport" in name.lower()
        or "residual_mismatch" in name.lower()
    ]
    kept_names = [
        f"icr.{names[i]}" if "icr" in names[i].lower() else f"mechanism.{names[i]}"
        for i in keep
    ]
    packed_chain = (
        np.asarray(z["chain_idx"], dtype=np.int64)
        if "chain_idx" in z.files
        else np.arange(len(z["step_scores"]), dtype=np.int64)
    )
    row_by_chain = {int(chain): i for i, chain in enumerate(packed_chain)}
    scores = np.asarray(z["step_scores"], dtype=np.float32)
    output: list[np.ndarray] = []
    for chain, steps in zip(chain_ids, n_steps):
        row = row_by_chain.get(int(chain))
        if row is None or not keep:
            output.append(np.empty((int(steps), 0), dtype=np.float32))
        else:
            output.append(scores[row, : int(steps), keep])
    return output, kept_names


def _legacy_cloud_sequences(
    z: np.lib.npyio.NpzFile,
    n: int,
    n_steps: Sequence[int],
) -> tuple[list[np.ndarray], list[str]]:
    if "stepcloud" not in z.files or "cloud_feature_names" not in z.files:
        return [np.empty((int(t), 0), dtype=np.float32) for t in n_steps], []
    names = [str(x) for x in z["cloud_feature_names"]]
    requested = [
        name
        for name in ("spread", "resultant", "coherence", "cloud_D", "cloud_V")
        if name in names
    ]
    indices = [names.index(name) for name in requested]
    output: list[np.ndarray] = []
    objects = np.asarray(z["stepcloud"], dtype=object)
    for i in range(n):
        value = objects[i]
        if value is None or not indices:
            output.append(np.empty((int(n_steps[i]), 0), dtype=np.float32))
            continue
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 3:
            output.append(
                np.full((int(n_steps[i]), len(indices)), np.nan, dtype=np.float32)
            )
            continue
        # Layer-median avoids selecting a favorable depth after seeing labels.
        output.append(np.nanmedian(array[: int(n_steps[i]), :, indices], axis=1))
    return output, [f"legacy.{name}.layer_median" for name in requested]


def _batched_state_geometry(
    state_sequences: Sequence[np.ndarray | None],
    n_steps: Sequence[int],
    *,
    compute_device: str,
    batch_size: int,
) -> list[np.ndarray]:
    """Compute state geometry in exact-shape buckets for efficient GPU use."""

    output: list[np.ndarray | None] = [None] * len(n_steps)
    shape_buckets: dict[tuple[int, ...], list[tuple[int, np.ndarray]]] = {}
    for i, steps in enumerate(n_steps):
        state = state_sequences[i] if i < len(state_sequences) else None
        if state is None:
            output[i] = np.full(
                (int(steps), len(STATE_GEOMETRY_NAMES)), np.nan, dtype=np.float32
            )
            continue
        array = np.asarray(state)[: int(steps)]
        if array.ndim != 3:
            raise ValueError(
                f"chain {i}: state must have shape [step, layer, hidden], got "
                f"{array.shape}"
            )
        shape_buckets.setdefault(tuple(int(x) for x in array.shape), []).append(
            (i, array)
        )

    for bucket in shape_buckets.values():
        for start in range(0, len(bucket), batch_size):
            chunk = bucket[start : start + batch_size]
            stacked = np.stack([array for _, array in chunk], axis=0)
            features = compute_state_geometry(
                stacked,
                compute_device=compute_device,
            )
            for row, (chain_index, _) in enumerate(chunk):
                output[chain_index] = features[row]

    if any(value is None for value in output):
        raise RuntimeError("internal error: state geometry output was not populated")
    return [np.asarray(value, dtype=np.float32) for value in output]


def load_geometry_collection(
    path: str | Path,
    *,
    compute_device: str = "cpu",
    geometry_batch_size: int = 32,
    include_legacy_geometry: bool = True,
) -> GeometryCollection:
    if geometry_batch_size < 1:
        raise ValueError("geometry_batch_size must be positive")
    path = Path(path)
    z = np.load(path, allow_pickle=True)
    state_sequences, layers, state_source = _load_state_sequences(z, path)
    if state_sequences:
        n = len(state_sequences)
    elif "step_scores" in z.files:
        n = int(len(z["step_scores"]))
    elif "gold_error_step" in z.files:
        n = int(len(z["gold_error_step"]))
    else:
        raise ValueError(f"{path}: no step-layer states or mechanism step scores")
    chain_idx, problem_id, gold, correct = _as_chain_metadata(z, n)
    response_hash = _response_hashes(z, n)
    if len(chain_idx) != n:
        raise ValueError(f"{path}: state and chain metadata counts differ")

    n_steps: list[int] = []
    for i in range(n):
        if i < len(state_sequences) and state_sequences[i] is not None:
            n_steps.append(int(np.asarray(state_sequences[i]).shape[0]))
        elif "n_steps" in z.files:
            n_steps.append(int(np.asarray(z["n_steps"])[i]))
        elif "step_scores" in z.files:
            n_steps.append(
                int(np.isfinite(np.asarray(z["step_scores"])[i]).any(axis=1).sum())
            )
        else:
            raise ValueError(f"{path}: cannot determine n_steps for chain {i}")

    state_matrices = _batched_state_geometry(
        state_sequences,
        n_steps,
        compute_device=compute_device,
        batch_size=geometry_batch_size,
    )
    mechanism, mechanism_names = _mechanism_sequences(z, chain_idx, n_steps)
    if include_legacy_geometry:
        legacy, legacy_names = _legacy_cloud_sequences(z, n, n_steps)
    else:
        legacy = [np.empty((steps, 0), dtype=np.float32) for steps in n_steps]
        legacy_names = []

    feature_names = (
        tuple(STATE_GEOMETRY_NAMES) + tuple(mechanism_names) + tuple(legacy_names)
    )
    matrices = [
        np.concatenate([state_matrices[i], mechanism[i], legacy[i]], axis=1).astype(
            np.float32
        )
        for i in range(n)
    ]
    collection = GeometryCollection(
        chain_idx=chain_idx,
        problem_id=problem_id,
        gold_error_step=gold,
        is_correct=correct,
        response_hash=response_hash,
        matrices=matrices,
        feature_names=feature_names,
        feature_groups=tuple(geometry_group(name) for name in feature_names),
        layers=layers,
        preflight={
            "path": str(path),
            "state_source": state_source,
            "layers": list(layers),
            "contiguous_layers": bool(len(layers) < 2 or np.all(np.diff(layers) == 1)),
            "state_shape_example": (
                list(
                    _first_valid_object_shape(np.asarray(state_sequences, dtype=object))
                )
                if state_sequences
                else []
            ),
            "num_chains": int(n),
            "num_state_features": int(len(STATE_GEOMETRY_NAMES)),
            "geometry_batch_size": int(geometry_batch_size),
            "num_mechanism_features": int(len(mechanism_names)),
            "num_legacy_features": int(len(legacy_names)),
            "feature_groups": sorted(
                set(geometry_group(name) for name in feature_names)
            ),
            "model_name": _scalar_string(
                z, ("model_name", "observer_model", "source_model")
            ),
            "gold_error_step_available": bool("gold_error_step" in z.files),
            "final_answer_label_available": bool(
                "is_correct" in z.files or "is_correct_strict" in z.files
            ),
            "response_hash_coverage": float(np.mean(response_hash != "")),
            "mainline_geometry_ready": bool(
                state_source
                in {"packed_whole_layer_states", "memmap_whole_layer_states"}
            ),
            "tier": (
                "confirmatory_whole_layer"
                if state_source
                in {"packed_whole_layer_states", "memmap_whole_layer_states"}
                else "exploratory_legacy_or_partial"
            ),
        },
    )
    collection.validate()
    return collection


def feature_indices_for_groups(
    groups: Sequence[str],
    requested: Iterable[str],
) -> np.ndarray:
    wanted = set(str(x) for x in requested)
    return np.asarray(
        [i for i, group in enumerate(groups) if group in wanted], dtype=np.int64
    )
