from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .spectral_chain_dynamics import canonicalize_spectral_input


EPS = 1e-8

LAYER_TIME_FIELD_NAMES = (
    "lid",
    "depth_neighbor_rewire",
    "time_neighbor_rewire",
    "depth_tangent_drift",
    "time_tangent_drift",
    "plaquette_holonomy",
    "rank_singularity",
)

LAYER_TIME_STEP_METRICS = (
    "ltg_lid_median",
    "ltg_lid_depth_iqr",
    "ltg_depth_rewire_peak",
    "ltg_time_rewire_median",
    "ltg_depth_tangent_peak",
    "ltg_time_tangent_median",
    "ltg_holonomy_peak",
    "ltg_rank_singularity_rate",
)


@dataclass(frozen=True)
class LayerTimeGeometryConfig:
    """Settings for the label-free layer-time representation field.

    Geometry references, centering, projection, and adjacent-layer gauge maps
    are fitted on training problems only.  Error labels are deliberately absent
    from the field construction; they are used only by downstream validation.
    """

    n_folds: int = 5
    knn_k: int = 20
    tangent_k: int = 24
    tangent_rank: int = 6
    projection_dim: int = 64
    max_reference: int = 256
    random_seed: int = 13
    chunk_size: int = 128
    phase_grid_size: int = 11
    min_lid_coverage: float = 0.99
    min_connection_coverage: float = 0.95
    min_holonomy_coverage: float = 0.90
    require_contiguous_layers: bool = True
    require_mean_pooling: bool = True


@dataclass
class LayerStateBank:
    vectors: np.ndarray
    chain_idx: np.ndarray
    step_idx: np.ndarray
    layers: np.ndarray

    def take(self, mask: np.ndarray) -> "LayerStateBank":
        if np.all(mask):
            return self
        return LayerStateBank(
            vectors=self.vectors[mask],
            chain_idx=self.chain_idx[mask],
            step_idx=self.step_idx[mask],
            layers=self.layers,
        )


def append_layer_time_geometry(
    metrics: Mapping[str, Any],
    cfg: LayerTimeGeometryConfig = LayerTimeGeometryConfig(),
) -> dict[str, np.ndarray]:
    """Append a cross-fitted ``[chain, step, layer, observable]`` field.

    The observables are not stitched detectors.  They are measurements of
    one geometric object: local dimension, reference-neighborhood transport,
    and local tangent transport along the depth and reasoning-time axes.
    """

    packed = canonicalize_layer_time_input(metrics)
    try:
        return _append_layer_time_geometry_from_packed(packed, cfg)
    except Exception:
        _discard_temporary_layer_state(packed)
        raise


def _append_layer_time_geometry_from_packed(
    packed: dict[str, np.ndarray],
    cfg: LayerTimeGeometryConfig,
) -> dict[str, np.ndarray]:
    bank = load_layer_state_bank(packed)
    _validate_layers(bank.layers, cfg.require_contiguous_layers)
    pooling = str(np.asarray(packed.get("state_pooling_kind", "unknown")).reshape(-1)[0])
    representation = str(
        np.asarray(packed.get("state_representation_kind", "unknown")).reshape(-1)[0]
    )
    if representation != "hidden_state":
        raise ValueError(
            f"Expected hidden-state representations, found representation_kind={representation!r}"
        )
    if cfg.require_mean_pooling and pooling != "arithmetic_mean_over_step_tokens":
        raise ValueError(
            f"Expected arithmetic mean pooled hidden states, found pooling_kind={pooling!r}. "
            "Re-extract with --geometry_only, or explicitly allow legacy pooling for an ablation."
        )

    chain_idx = np.asarray(packed["chain_idx"], dtype=np.int64)
    n_steps = np.asarray(packed["n_steps"], dtype=np.int64)
    if chain_idx.size < 2:
        raise ValueError("layer-time geometry requires at least two chains")
    max_steps = int(np.max(n_steps)) if n_steps.size else 0
    chain_to_row = {int(c): i for i, c in enumerate(chain_idx.tolist())}
    point_rows = np.asarray([chain_to_row.get(int(c), -1) for c in bank.chain_idx], dtype=np.int64)
    valid = (
        (point_rows >= 0)
        & (bank.step_idx >= 0)
        & (bank.step_idx < n_steps[np.maximum(point_rows, 0)])
        & np.isfinite(bank.vectors).all(axis=(1, 2))
    )
    if not np.all(valid):
        bank = bank.take(valid)
        point_rows = point_rows[valid]
    if bank.vectors.shape[0] < max(8, cfg.knn_k + 1):
        raise ValueError("too few valid layer-state points for the requested neighborhood size")

    groups = np.asarray(packed.get("problem_id", chain_idx))
    if groups.shape[0] != chain_idx.shape[0]:
        raise ValueError("problem_id must have one value per chain")
    folds = make_group_folds(groups, cfg.n_folds, cfg.random_seed)
    row_fold = np.full(chain_idx.shape[0], -1, dtype=np.int32)
    field = np.full(
        (chain_idx.shape[0], max_steps, bank.vectors.shape[1], len(LAYER_TIME_FIELD_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    reference_sizes = np.full(len(folds), -1, dtype=np.int64)
    global_projection = shared_jl_projection(
        bank.vectors.shape[2],
        cfg.projection_dim,
        np.random.default_rng(int(cfg.random_seed) + 7919),
    )

    for fold_id, test_chain_rows in enumerate(folds):
        test_chain_rows = np.asarray(test_chain_rows, dtype=np.int64)
        row_fold[test_chain_rows] = int(fold_id)
        test_mask = np.isin(point_rows, test_chain_rows)
        train_mask = ~test_mask
        if not np.any(test_mask):
            raise ValueError(f"fold {fold_id} contains no layer-state points")
        rng = np.random.default_rng(int(cfg.random_seed) + 1009 * int(fold_id))
        train_chain_rows = np.setdiff1d(
            np.arange(chain_idx.size, dtype=np.int64),
            test_chain_rows,
            assume_unique=False,
        )
        ref_chain_rows = balanced_chain_reference_rows(
            train_chain_rows,
            groups,
            cfg.max_reference,
            rng,
        )
        if ref_chain_rows.size < max(4, cfg.knn_k, cfg.tangent_k):
            raise ValueError(
                f"fold {fold_id} has only {ref_chain_rows.size} reference chains; "
                "reduce k or collect more problem groups"
            )
        reference, reference_lengths = pack_reference_trajectories(
            bank.vectors,
            point_rows,
            bank.step_idx,
            ref_chain_rows,
            n_steps,
        )
        reference_sizes[fold_id] = int(ref_chain_rows.size)
        test_ids = np.where(test_mask)[0]
        query_phase = np.asarray(
            [
                0.0
                if n_steps[point_rows[point_i]] <= 1
                else bank.step_idx[point_i] / max(int(n_steps[point_rows[point_i]]) - 1, 1)
                for point_i in test_ids
            ],
            dtype=np.float64,
        )
        fold_field = score_layer_time_fold(
            reference,
            bank.vectors[test_ids],
            bank.chain_idx[test_ids],
            bank.step_idx[test_ids],
            cfg,
            rng,
            reference_lengths=reference_lengths,
            query_phase=query_phase,
            projection_matrix=global_projection,
        )
        for local_i, point_i in enumerate(test_ids):
            row = int(point_rows[point_i])
            step = int(bank.step_idx[point_i])
            field[row, step] = fold_field[local_i]

    if np.any(row_fold < 0):
        raise RuntimeError("group folds did not assign every chain exactly once")

    lid = field[..., LAYER_TIME_FIELD_NAMES.index("lid")]
    expected = np.zeros(lid.shape, dtype=bool)
    for row, count in enumerate(n_steps):
        expected[row, : int(count), :] = True
    lid_coverage = float(np.mean(np.isfinite(lid[expected]))) if np.any(expected) else 0.0
    if lid_coverage < float(cfg.min_lid_coverage):
        raise RuntimeError(
            f"OOF LID coverage {lid_coverage:.3f} is below required {cfg.min_lid_coverage:.3f}"
        )
    depth_tangent = field[..., LAYER_TIME_FIELD_NAMES.index("depth_tangent_drift")]
    time_tangent = field[..., LAYER_TIME_FIELD_NAMES.index("time_tangent_drift")]
    holonomy = field[..., LAYER_TIME_FIELD_NAMES.index("plaquette_holonomy")]
    expected_depth = expected.copy()
    expected_depth[:, :, 0] = False
    expected_time = expected.copy()
    expected_holonomy = expected.copy()
    expected_holonomy[:, :, 0] = False
    for row, count in enumerate(n_steps):
        expected_time[row, 0, :] = False
        expected_holonomy[row, 0, :] = False
        if count <= 1:
            expected_time[row] = False
            expected_holonomy[row] = False
    depth_coverage = float(np.mean(np.isfinite(depth_tangent[expected_depth]))) if np.any(expected_depth) else 1.0
    time_coverage = float(np.mean(np.isfinite(time_tangent[expected_time]))) if np.any(expected_time) else 1.0
    connection_coverage = min(depth_coverage, time_coverage)
    holonomy_coverage = float(np.mean(np.isfinite(holonomy[expected_holonomy]))) if np.any(expected_holonomy) else 1.0
    if connection_coverage < float(cfg.min_connection_coverage):
        raise RuntimeError(
            f"OOF connection coverage {connection_coverage:.3f} is below required "
            f"{cfg.min_connection_coverage:.3f}"
        )
    if holonomy_coverage < float(cfg.min_holonomy_coverage):
        raise RuntimeError(
            f"OOF holonomy coverage {holonomy_coverage:.3f} is below required "
            f"{cfg.min_holonomy_coverage:.3f}"
        )

    step_metric_map = reduce_layer_time_field(field)
    packed = _append_step_metrics(packed, step_metric_map)
    packed = _append_chain_metrics(packed, step_metric_map, n_steps)
    packed["layer_time_geometry_field"] = field.astype(np.float32)
    packed["layer_time_geometry_field_names"] = np.asarray(LAYER_TIME_FIELD_NAMES, dtype="<U40")
    packed["layer_time_geometry_layers"] = bank.layers.astype(np.int64)
    packed["layer_time_geometry_fold"] = row_fold
    packed["layer_time_geometry_reference_sizes"] = reference_sizes
    packed["layer_time_geometry_projection_digest"] = np.asarray(
        hashlib.sha256(np.ascontiguousarray(global_projection).tobytes()).hexdigest(),
        dtype=object,
    )
    packed["layer_time_geometry_lid_coverage"] = np.asarray(lid_coverage, dtype=np.float64)
    packed["layer_time_geometry_connection_coverage"] = np.asarray(connection_coverage, dtype=np.float64)
    packed["layer_time_geometry_holonomy_coverage"] = np.asarray(holonomy_coverage, dtype=np.float64)
    packed["layer_time_geometry_config"] = np.asarray(str(cfg), dtype=object)
    packed["layer_time_geometry_reference_policy"] = np.asarray(
        "problem-grouped out-of-fold; problem-balanced train-chain universe; phase-matched interpolation; labels unused",
        dtype=object,
    )
    packed["layer_time_geometry_pooling_kind"] = np.asarray(pooling, dtype=object)
    packed["layer_time_geometry_representation_kind"] = np.asarray(representation, dtype=object)
    return packed


def _discard_temporary_layer_state(packed: Mapping[str, Any]) -> None:
    """Release a derived memmap before deleting it on every failure path."""

    temporary = packed.get("step_layer_state_temporary_memmap_path")
    if temporary is None:
        return
    raw_state = packed.get("step_layer_state_vectors")
    mmap_obj = getattr(raw_state, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()
    try:
        Path(str(np.asarray(temporary).item())).unlink(missing_ok=True)
    except OSError:
        # Cleanup must never replace the scientific validation exception.
        pass


def canonicalize_layer_time_input(metrics: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Preserve or recover the explicit layer axis from supported schemas."""

    source = {str(k): np.asarray(v) for k, v in metrics.items()}
    if "step_layer_state_memmap_path" in source and "step_layer_state_vectors" not in source:
        state_path = Path(str(np.asarray(source["step_layer_state_memmap_path"]).item()))
        if not state_path.is_absolute():
            manifest = Path(str(np.asarray(source.get("layer_time_input_path", ".")).item()))
            state_path = manifest.resolve().parent / state_path
        if not state_path.exists():
            raise FileNotFoundError(f"whole-layer state memmap does not exist: {state_path}")
        state_store = np.load(state_path, mmap_mode="r")
        count = int(np.asarray(source["step_layer_state_memmap_count"]).item())
        if state_store.ndim != 3 or not 0 < count <= state_store.shape[0]:
            raise ValueError(f"invalid whole-layer state memmap shape/count: {state_store.shape}, {count}")
        source["step_layer_state_vectors"] = state_store[:count]
    if "state_representation_kind" not in source and bool(
        np.asarray(source.get("reasoning_subspace_used", False)).item()
    ):
        source["state_representation_kind"] = np.asarray(
            "reasoning_subspace_projection", dtype=object
        )
    if "step_layer_state_vectors" in source:
        source.setdefault("state_pooling_kind", np.asarray("unknown", dtype=object))
        return source
    if "sv_vec_mean" in source and "stepvec" not in source:
        return canonicalize_sv_vec_mean_input(source)

    if "step_state_vectors" in source and "stepvec" not in source:
        packed = source
    else:
        packed = canonicalize_spectral_input(source, "step_state_vectors")
        packed = {str(k): np.asarray(v) for k, v in packed.items()}
    for key in (
        "layer_time_source_format",
        "layer_time_embedding_depth_dropped",
        "model_sampling_metadata_json",
        "model_name",
        "sampling_seed",
        "sampling_temperature",
        "sampling_top_p",
        "sampling_max_new_tokens",
    ):
        if key in source:
            packed[key] = np.array(source[key], copy=True)
    if "step_layer_state_vectors" in packed:
        packed.setdefault("state_pooling_kind", np.asarray("unknown", dtype=object))
        return packed
    if "step_state_vectors" not in packed:
        raise ValueError(
            "No whole-layer state tensor found. Extract with --geometry_only, "
            "or provide canonical stepvec shaped [chain][step, layer, hidden]."
        )

    flat = np.asarray(packed["step_state_vectors"], dtype=np.float32)
    layers = np.asarray(
        packed.get("step_state_vector_layers", packed.get("layers", [])),
        dtype=np.int64,
    )
    if flat.ndim == 3:
        tensor = flat
    elif flat.ndim == 2 and layers.size and flat.shape[1] % layers.size == 0:
        tensor = flat.reshape(flat.shape[0], layers.size, flat.shape[1] // layers.size)
    else:
        raise ValueError(
            f"Cannot recover [point, layer, hidden] from step_state_vectors {flat.shape} "
            f"and {layers.size} layer labels"
        )
    packed["step_layer_state_vectors"] = tensor.astype(np.float16)
    packed["step_layer_state_vector_chain_idx"] = np.asarray(
        packed["step_state_vector_chain_idx"], dtype=np.int64
    )
    packed["step_layer_state_vector_step_idx"] = np.asarray(
        packed["step_state_vector_step_idx"], dtype=np.int64
    )
    packed["step_layer_state_vector_layers"] = layers
    packed.setdefault(
        "state_pooling_kind",
        np.asarray("arithmetic_mean_over_step_tokens", dtype=object),
    )
    return packed


def canonicalize_sv_vec_mean_input(source: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Convert exact per-chain mean vectors without constructing a flat LD copy."""

    representation = str(
        np.asarray(source.get("state_representation_kind", "hidden_state")).reshape(-1)[0]
    )
    projected = bool(np.asarray(source.get("reasoning_subspace_used", False)).item())
    if projected or representation != "hidden_state":
        found = "reasoning_subspace_projection" if projected else representation
        raise ValueError(
            f"Expected hidden-state representations, found representation_kind={found!r}"
        )

    raw = np.asarray(source["sv_vec_mean"], dtype=object)
    n = int(raw.shape[0])
    trajectories = [_layer_trajectory(raw[i]) for i in range(n)]
    if not trajectories or any(arr.shape[0] == 0 for arr in trajectories):
        raise ValueError("sv_vec_mean contains an empty trajectory")
    layers = np.asarray(
        source.get("layers_used", source.get("sv_layers", [])),
        dtype=np.int64,
    )
    if layers.size != trajectories[0].shape[1]:
        raise ValueError("sv_vec_mean depth does not match layers_used/sv_layers")
    if any(arr.shape[1:] != trajectories[0].shape[1:] for arr in trajectories):
        raise ValueError("sv_vec_mean trajectories have inconsistent layer/hidden shapes")
    dropped_embedding = bool(
        layers.size >= 2
        and int(layers[0]) == 0
        and np.array_equal(layers, np.arange(layers.size, dtype=np.int64))
    )
    layer_slice = slice(1, None) if dropped_embedding else slice(None)
    layers = layers[layer_slice]
    n_steps = np.asarray([arr.shape[0] for arr in trajectories], dtype=np.int64)
    total_steps = int(np.sum(n_steps))
    hidden_dim = int(trajectories[0].shape[2])

    cache_path: Path | None = None
    input_marker = str(np.asarray(source.get("layer_time_input_path", "")).item())
    cache_marker = str(np.asarray(source.get("layer_time_cache_dir", "")).item())
    if input_marker:
        input_path = Path(input_marker).resolve()
        cache_dir = Path(cache_marker).resolve() if cache_marker else input_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_id = str(
            np.asarray(source.get("layer_time_cache_id", uuid.uuid4().hex[:12])).item()
        )
        cache_path = cache_dir / f".{input_path.stem}.ltg-mean-states.{cache_id}.npy"
        partial_path = cache_dir / f".{input_path.stem}.ltg-mean-states.{cache_id}.partial.npy"
        state = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float16,
            shape=(total_steps, layers.size, hidden_dim),
        )
    else:
        partial_path = None
        state = np.empty((total_steps, layers.size, hidden_dim), dtype=np.float16)

    point_chain = np.empty(total_steps, dtype=np.int64)
    point_step = np.empty(total_steps, dtype=np.int64)
    cursor = 0
    for chain, arr in enumerate(trajectories):
        count = arr.shape[0]
        state[cursor : cursor + count] = np.asarray(arr[:, layer_slice, :], dtype=np.float16)
        point_chain[cursor : cursor + count] = chain
        point_step[cursor : cursor + count] = np.arange(count, dtype=np.int64)
        cursor += count
    if cache_path is not None and partial_path is not None:
        state.flush()
        state._mmap.close()
        partial_path.replace(cache_path)
        state = np.load(cache_path, mmap_mode="r")

    max_steps = int(np.max(n_steps))
    step_ranges = np.full((n, max_steps, 2), -1, dtype=np.int32)
    step_len = np.full((n, max_steps), np.nan, dtype=np.float32)
    raw_ranges = source.get("step_token_ranges")
    for chain in range(n):
        count = int(n_steps[chain])
        if raw_ranges is not None:
            ranges = np.asarray(raw_ranges[chain])
            if ranges.dtype == object:
                ranges = np.asarray(ranges.tolist())
            ranges = np.asarray(ranges, dtype=np.int64)
            take = min(count, ranges.shape[0]) if ranges.ndim == 2 else 0
            if take:
                step_ranges[chain, :take] = ranges[:take, :2]
                step_len[chain, :take] = ranges[:take, 1] - ranges[:take, 0] + 1
        step_len[chain, :count] = np.where(
            np.isfinite(step_len[chain, :count]),
            step_len[chain, :count],
            1.0,
        )
    rel_pos = np.full((n, max_steps), np.nan, dtype=np.float32)
    for chain, count in enumerate(n_steps):
        rel_pos[chain, : int(count)] = (
            0.0
            if count <= 1
            else np.arange(int(count), dtype=np.float32) / float(int(count) - 1)
        )

    chain_idx = np.arange(n, dtype=np.int64)
    problem_id = np.asarray(source.get("problem_ids", chain_idx), dtype=np.int64)
    gold = np.asarray(
        source.get("gold_error_step", source.get("labels", np.full(n, -1))),
        dtype=np.int64,
    )
    correct = np.asarray(
        source.get("is_correct_strict", source.get("is_correct", (gold < 0).astype(np.int64))),
        dtype=np.int64,
    )
    sample_idx = np.asarray(source.get("sample_idx", np.full(n, -1)), dtype=np.int64)
    packed: dict[str, np.ndarray] = {
        "chain_idx": chain_idx,
        "problem_id": problem_id,
        "gold_error_step": gold,
        "is_correct": correct,
        "sample_idx": sample_idx,
        "generator": _broadcast_metadata(source, ("generator", "generators", "model_name"), n),
        "dataset": _broadcast_metadata(source, ("dataset", "datasets", "subset"), n),
        "n_steps": n_steps,
        "step_token_ranges": step_ranges,
        "step_scores": np.stack([step_len, rel_pos], axis=2),
        "step_score_names": np.asarray(["step_len", "rel_pos"], dtype="<U96"),
        "chain_scores": np.stack([np.nanmean(step_len, axis=1), np.nanmean(rel_pos, axis=1)], axis=1),
        "chain_score_names": np.asarray(["mean_step_len", "mean_rel_pos"], dtype="<U96"),
        "layers": layers,
        "step_layer_state_vectors": state,
        "step_layer_state_vector_chain_idx": point_chain,
        "step_layer_state_vector_step_idx": point_step,
        "step_layer_state_vector_layers": layers,
        "state_representation_kind": np.asarray("hidden_state", dtype=object),
        "state_pooling_kind": np.asarray("arithmetic_mean_over_step_tokens", dtype=object),
        "layer_time_source_format": np.asarray("exact_sv_vec_mean", dtype=object),
        "layer_time_embedding_depth_dropped": np.asarray(dropped_embedding),
        "state_storage_kind": np.asarray(
            "derived_npy_memmap" if cache_path is not None else "in_memory_tensor",
            dtype=object,
        ),
    }
    if cache_path is not None:
        packed["step_layer_state_temporary_memmap_path"] = np.asarray(str(cache_path), dtype=object)
    for key in (
        "model_sampling_metadata_json",
        "model_name",
        "sampling_seed",
        "sampling_temperature",
        "sampling_top_p",
        "sampling_max_new_tokens",
    ):
        if key in source:
            packed[key] = np.asarray(source[key])
    return packed


def _layer_trajectory(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist())
    if arr.ndim == 2:
        arr = arr[:, None, :]
    if arr.ndim != 3:
        raise ValueError(f"expected [step,layer,hidden] trajectory, got {arr.shape}")
    return arr


def _broadcast_metadata(source: Mapping[str, Any], keys: tuple[str, ...], n: int) -> np.ndarray:
    value: Any = None
    for key in keys:
        if key in source:
            value = source[key]
            break
    if value is None:
        return np.asarray([""] * n, dtype=object)
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        return np.asarray([arr.item()] * n, dtype=object)
    if arr.shape[0] != n:
        raise ValueError(f"metadata {keys} has length {arr.shape[0]}, expected {n}")
    return arr


def load_layer_state_bank(metrics: Mapping[str, Any]) -> LayerStateBank:
    required = (
        "step_layer_state_vectors",
        "step_layer_state_vector_chain_idx",
        "step_layer_state_vector_step_idx",
        "step_layer_state_vector_layers",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError(f"layer-state schema is missing: {missing}")
    vectors = np.asarray(metrics[required[0]])
    if not np.issubdtype(vectors.dtype, np.floating):
        vectors = vectors.astype(np.float32)
    chains = np.asarray(metrics[required[1]], dtype=np.int64)
    steps = np.asarray(metrics[required[2]], dtype=np.int64)
    layers = np.asarray(metrics[required[3]], dtype=np.int64)
    if vectors.ndim != 3:
        raise ValueError(f"step_layer_state_vectors must be rank 3, got {vectors.shape}")
    if vectors.shape[0] != chains.size or chains.size != steps.size:
        raise ValueError("layer-state vectors and point metadata have different lengths")
    if vectors.shape[1] != layers.size:
        raise ValueError("layer-state tensor depth does not match layer labels")
    return LayerStateBank(vectors=vectors, chain_idx=chains, step_idx=steps, layers=layers)


def make_group_folds(groups: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Return chain-row folds with all samples of a problem kept together."""

    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    if unique.size < 2:
        raise ValueError("at least two distinct problem_id groups are required")
    rng = np.random.default_rng(int(seed))
    order = np.arange(unique.size)
    rng.shuffle(order)
    order = order[np.argsort(-np.bincount(inverse, minlength=unique.size)[order], kind="stable")]
    n_folds = max(2, min(int(n_folds), int(unique.size)))
    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    loads = np.zeros(n_folds, dtype=np.int64)
    counts = np.bincount(inverse, minlength=unique.size)
    for group_pos in order:
        target = int(np.argmin(loads))
        buckets[target].append(int(group_pos))
        loads[target] += int(counts[group_pos])
    folds = []
    for bucket in buckets:
        mask = np.isin(inverse, np.asarray(bucket, dtype=np.int64))
        folds.append(np.where(mask)[0].astype(np.int64))
    return folds


def balanced_reference_sample(
    train_ids: np.ndarray,
    chain_idx: np.ndarray,
    max_reference: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one shared reference universe without favoring long chains."""

    train_ids = np.asarray(train_ids, dtype=np.int64)
    limit = int(max_reference)
    if limit <= 0 or train_ids.size <= limit:
        return train_ids
    groups: list[np.ndarray] = []
    for chain in np.unique(chain_idx[train_ids]):
        ids = train_ids[chain_idx[train_ids] == chain].copy()
        rng.shuffle(ids)
        groups.append(ids)
    rng.shuffle(groups)
    selected: list[int] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for ids in groups:
            if depth < ids.size:
                selected.append(int(ids[depth]))
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return np.asarray(selected, dtype=np.int64)


def balanced_chain_reference_rows(
    train_rows: np.ndarray,
    problem_id: np.ndarray,
    max_reference: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose reference chains round-robin by problem, then by sample.

    This keeps the reference identity universe fixed across depth and phase
    without letting a problem with many sampled responses dominate the cloud.
    """

    train_rows = np.asarray(train_rows, dtype=np.int64)
    limit = int(max_reference)
    buckets: list[np.ndarray] = []
    for problem in np.unique(problem_id[train_rows]):
        rows = train_rows[problem_id[train_rows] == problem].copy()
        rng.shuffle(rows)
        buckets.append(rows)
    rng.shuffle(buckets)
    if not buckets:
        return np.empty(0, dtype=np.int64)
    n_problems = len(buckets)
    if limit <= 0:
        per_problem = min(rows.size for rows in buckets)
        chosen_buckets = buckets
    elif limit < n_problems:
        per_problem = 1
        chosen_buckets = buckets[:limit]
    else:
        per_problem = min(min(rows.size for rows in buckets), max(1, limit // n_problems))
        chosen_buckets = buckets
    selected = [int(row) for rows in chosen_buckets for row in rows[:per_problem]]
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def pack_reference_trajectories(
    vectors: np.ndarray,
    point_rows: np.ndarray,
    step_idx: np.ndarray,
    reference_rows: np.ndarray,
    n_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack complete train-chain trajectories for phase interpolation."""

    reference_rows = np.asarray(reference_rows, dtype=np.int64)
    lengths = np.asarray([int(n_steps[row]) for row in reference_rows], dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("reference chains must have at least one step")
    out = np.full(
        (reference_rows.size, int(np.max(lengths)), vectors.shape[1], vectors.shape[2]),
        np.nan,
        dtype=vectors.dtype,
    )
    for ref_pos, row in enumerate(reference_rows):
        ids = np.where(point_rows == int(row))[0]
        for point_i in ids:
            step = int(step_idx[point_i])
            if 0 <= step < lengths[ref_pos]:
                out[ref_pos, step] = vectors[point_i]
        if not np.isfinite(out[ref_pos, : lengths[ref_pos]]).all():
            raise ValueError(f"reference chain row {row} is missing one or more layer-state steps")
    return out, lengths


def interpolate_reference_trajectories(
    trajectories: np.ndarray,
    lengths: np.ndarray,
    phase: float,
) -> np.ndarray:
    """Interpolate one state per reference chain at normalized phase."""

    trajectories = np.asarray(trajectories, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.int64)
    phase = float(np.clip(phase, 0.0, 1.0))
    out = np.empty((trajectories.shape[0], trajectories.shape[2], trajectories.shape[3]), dtype=np.float32)
    for i, count in enumerate(lengths):
        if count <= 1:
            out[i] = trajectories[i, 0]
            continue
        position = phase * (int(count) - 1)
        lo = int(np.floor(position))
        hi = min(lo + 1, int(count) - 1)
        weight = float(position - lo)
        out[i] = (1.0 - weight) * trajectories[i, lo] + weight * trajectories[i, hi]
    return out


def fit_streaming_layer_normalizer(
    trajectories: np.ndarray,
    lengths: np.ndarray,
    phase_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-layer center and scalar RMS without materializing phase×chain."""

    n_layers = int(trajectories.shape[2])
    hidden_dim = int(trajectories.shape[3])
    total = np.zeros((n_layers, hidden_dim), dtype=np.float64)
    total_sq = np.zeros(n_layers, dtype=np.float64)
    count = 0
    for phase in np.asarray(phase_grid, dtype=np.float64):
        raw = interpolate_reference_trajectories(trajectories, lengths, float(phase))
        total += np.sum(raw, axis=0, dtype=np.float64)
        total_sq += np.einsum("nld,nld->l", raw, raw, dtype=np.float64, optimize=True)
        count += int(raw.shape[0])
    if count <= 0:
        raise ValueError("cannot fit a layer normalizer from an empty reference grid")
    mu = total / float(count)
    mean_square = total_sq / float(count * hidden_dim)
    variance = mean_square - np.mean(mu * mu, axis=1)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return mu[None, :, :].astype(np.float32), scale[None, :, None].astype(np.float32)


def project_layer_tensor(
    tensor: np.ndarray,
    mu: np.ndarray,
    scale: np.ndarray,
    projection: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Center, scalar-normalize, and project a layer tensor in point batches."""

    tensor = np.asarray(tensor)
    out = np.empty((tensor.shape[0], tensor.shape[1], projection.shape[1]), dtype=np.float32)
    batch_size = max(1, int(batch_size))
    for start in range(0, tensor.shape[0], batch_size):
        raw = np.asarray(tensor[start : start + batch_size], dtype=np.float32)
        out[start : start + raw.shape[0]] = np.einsum(
            "nld,dp->nlp",
            (raw - mu) / scale,
            projection,
            optimize=True,
        ).astype(np.float32)
    return out


def score_layer_time_fold(
    reference: np.ndarray,
    query: np.ndarray,
    query_chain_idx: np.ndarray,
    query_step_idx: np.ndarray,
    cfg: LayerTimeGeometryConfig,
    rng: np.random.Generator,
    *,
    reference_lengths: np.ndarray | None = None,
    query_phase: np.ndarray | None = None,
    projection_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Measure one held-out fold against phase-matched train trajectories.

    ``reference`` may be ``[reference, layer, hidden]`` for a static synthetic
    check, or ``[reference_chain, step, layer, hidden]`` for the real method.
    In the latter case each query step sees one interpolated state from every
    train reference chain at the same normalized reasoning phase.  Neighbor
    identities therefore remain comparable across both depth and time.
    """

    reference = np.asarray(reference)
    query = np.asarray(query)
    if reference.ndim == 3:
        trajectories = reference[:, None, :, :]
        lengths = np.ones(reference.shape[0], dtype=np.int64)
    elif reference.ndim == 4:
        trajectories = reference
        if reference_lengths is None:
            raise ValueError("reference_lengths is required for trajectory references")
        lengths = np.asarray(reference_lengths, dtype=np.int64)
    else:
        raise ValueError("reference must be [reference,layer,hidden] or [reference,step,layer,hidden]")
    if query.ndim != 3 or trajectories.shape[2:] != query.shape[1:]:
        raise ValueError("reference/query must share [layer, hidden] shape")
    if lengths.shape != (trajectories.shape[0],):
        raise ValueError("reference_lengths must have one entry per reference chain")
    if np.any(lengths <= 0) or np.any(lengths > trajectories.shape[1]):
        raise ValueError("reference_lengths falls outside the packed trajectory tensor")
    if query_phase is None:
        query_phase = np.zeros(query.shape[0], dtype=np.float64)
    query_phase = np.asarray(query_phase, dtype=np.float64)
    if query_phase.shape != (query.shape[0],) or not np.isfinite(query_phase).all():
        raise ValueError("query_phase must be finite with one value per query point")

    n_ref, n_layers, hidden_dim = trajectories.shape[0], trajectories.shape[2], trajectories.shape[3]
    if n_ref < 3 or n_layers < 2:
        raise ValueError("layer-time geometry needs at least three references and two layers")

    # Fit the gauge and normalization on an equal-phase grid, so long chains do
    # not contribute more mass merely because they contain more textual steps.
    phase_grid = np.linspace(0.0, 1.0, max(2, int(cfg.phase_grid_size)))
    projection = (
        shared_jl_projection(hidden_dim, cfg.projection_dim, rng)
        if projection_matrix is None
        else np.asarray(projection_matrix, dtype=np.float32)
    )
    if projection.ndim != 2 or projection.shape[0] != hidden_dim:
        raise ValueError(
            f"projection matrix must have shape [{hidden_dim}, p], got {projection.shape}"
        )
    mu, scale = fit_streaming_layer_normalizer(trajectories, lengths, phase_grid)
    qry = project_layer_tensor(
        query,
        mu,
        scale,
        projection,
        batch_size=max(1, int(cfg.chunk_size)),
    )

    phase_keys = np.asarray([round(float(np.clip(x, 0.0, 1.0)), 12) for x in query_phase])
    phase_references: dict[float, np.ndarray] = {}
    for phase in np.unique(phase_keys):
        raw = interpolate_reference_trajectories(trajectories, lengths, float(phase))
        phase_references[float(phase)] = project_layer_tensor(
            raw,
            mu,
            scale,
            projection,
            batch_size=max(1, int(cfg.chunk_size)),
        )

    k_topology = max(2, min(int(cfg.knn_k), n_ref))
    k_tangent = max(2, min(int(cfg.tangent_k), n_ref))
    k_all = max(k_topology, k_tangent)
    neighbor_ids = np.empty((query.shape[0], n_layers, k_all), dtype=np.int64)
    neighbor_d2 = np.empty((query.shape[0], n_layers, k_all), dtype=np.float32)
    for phase in np.unique(phase_keys):
        query_ids = np.where(phase_keys == phase)[0]
        phase_ref = phase_references[float(phase)]
        for layer_pos in range(n_layers):
            d2, nn = topk_sqdist(
                qry[query_ids, layer_pos],
                phase_ref[:, layer_pos],
                k=k_all,
                chunk_size=cfg.chunk_size,
            )
            neighbor_d2[query_ids, layer_pos] = d2
            neighbor_ids[query_ids, layer_pos] = nn

    field = np.full((query.shape[0], n_layers, len(LAYER_TIME_FIELD_NAMES)), np.nan, dtype=np.float32)
    lid_pos = LAYER_TIME_FIELD_NAMES.index("lid")
    for layer_pos in range(n_layers):
        field[:, layer_pos, lid_pos] = lid_mle(neighbor_d2[:, layer_pos, :k_topology]).astype(np.float32)

    depth_rewire_pos = LAYER_TIME_FIELD_NAMES.index("depth_neighbor_rewire")
    for layer_pos in range(1, n_layers):
        field[:, layer_pos, depth_rewire_pos] = neighbor_rewire(
            neighbor_ids[:, layer_pos - 1, :k_topology],
            neighbor_ids[:, layer_pos, :k_topology],
        )

    time_rewire_pos = LAYER_TIME_FIELD_NAMES.index("time_neighbor_rewire")
    point_lookup = {
        (int(chain), int(step)): i
        for i, (chain, step) in enumerate(zip(query_chain_idx, query_step_idx))
    }
    for i, (chain, step) in enumerate(zip(query_chain_idx, query_step_idx)):
        prev = point_lookup.get((int(chain), int(step) - 1))
        if prev is None:
            continue
        for layer_pos in range(n_layers):
            field[i, layer_pos, time_rewire_pos] = neighbor_rewire(
                neighbor_ids[prev : prev + 1, layer_pos, :k_topology],
                neighbor_ids[i : i + 1, layer_pos, :k_topology],
            )[0]

    projected_dim = int(projection.shape[1])
    tangent_rank = max(1, min(int(cfg.tangent_rank), k_tangent - 1, projected_dim))
    bases = np.empty(
        (query.shape[0], n_layers, projected_dim, tangent_rank),
        dtype=np.float32,
    )
    effective_ranks = np.zeros((query.shape[0], n_layers), dtype=np.int32)
    for i in range(query.shape[0]):
        phase_ref = phase_references[float(phase_keys[i])]
        for layer_pos in range(n_layers):
            local = phase_ref[neighbor_ids[i, layer_pos, :k_tangent], layer_pos]
            basis, effective_rank = local_tangent_frame(local, bases.shape[-1])
            bases[i, layer_pos] = basis
            effective_ranks[i, layer_pos] = int(effective_rank)

    # LID determines the fiber rank, capped only for numerical tractability.
    # A plaquette with unequal corner ranks is still evaluated on their common
    # subspace, while the rank singularity is reported explicitly.
    lid_values = field[..., lid_pos]
    local_ranks = np.ones((query.shape[0], n_layers), dtype=np.int32)
    finite_lid = np.isfinite(lid_values)
    local_ranks[finite_lid] = np.rint(lid_values[finite_lid]).astype(np.int32)
    local_ranks = np.clip(local_ranks, 1, tangent_rank)
    local_ranks = np.minimum(local_ranks, effective_ranks)

    depth_tangent_pos = LAYER_TIME_FIELD_NAMES.index("depth_tangent_drift")
    for layer_pos in range(1, n_layers):
        for i in range(query.shape[0]):
            phase_ref = phase_references[float(phase_keys[i])]
            local_ids = np.union1d(
                neighbor_ids[i, layer_pos - 1, :k_tangent],
                neighbor_ids[i, layer_pos, :k_tangent],
            )
            rank = int(min(local_ranks[i, layer_pos - 1], local_ranks[i, layer_pos]))
            if rank < 1:
                continue
            _, residual = fit_local_tangent_connection(
                phase_ref[local_ids, layer_pos - 1],
                phase_ref[local_ids, layer_pos],
                bases[i, layer_pos - 1],
                bases[i, layer_pos],
                rank,
            )
            field[i, layer_pos, depth_tangent_pos] = residual

    time_tangent_pos = LAYER_TIME_FIELD_NAMES.index("time_tangent_drift")
    for i, (chain, step) in enumerate(zip(query_chain_idx, query_step_idx)):
        prev = point_lookup.get((int(chain), int(step) - 1))
        if prev is None:
            continue
        prev_ref = phase_references[float(phase_keys[prev])]
        curr_ref = phase_references[float(phase_keys[i])]
        for layer_pos in range(n_layers):
            local_ids = np.union1d(
                neighbor_ids[prev, layer_pos, :k_tangent],
                neighbor_ids[i, layer_pos, :k_tangent],
            )
            rank = int(min(local_ranks[prev, layer_pos], local_ranks[i, layer_pos]))
            if rank < 1:
                continue
            _, residual = fit_local_tangent_connection(
                prev_ref[local_ids, layer_pos],
                curr_ref[local_ids, layer_pos],
                bases[prev, layer_pos],
                bases[i, layer_pos],
                rank,
            )
            field[i, layer_pos, time_tangent_pos] = residual

    # A plaquette is the smallest genuinely two-dimensional object.  The two
    # products below both map tangent coordinates from (t-1,l-1) to (t,l),
    # once depth-then-time and once time-then-depth.  Their discrepancy is
    # invariant to arbitrary orthogonal changes of local tangent frame.
    holonomy_pos = LAYER_TIME_FIELD_NAMES.index("plaquette_holonomy")
    singularity_pos = LAYER_TIME_FIELD_NAMES.index("rank_singularity")
    for i, (chain, step) in enumerate(zip(query_chain_idx, query_step_idx)):
        prev = point_lookup.get((int(chain), int(step) - 1))
        if prev is None:
            continue
        prev_ref = phase_references[float(phase_keys[prev])]
        curr_ref = phase_references[float(phase_keys[i])]
        for layer_pos in range(1, n_layers):
            corner_ranks = np.asarray(
                [
                    local_ranks[prev, layer_pos - 1],
                    local_ranks[prev, layer_pos],
                    local_ranks[i, layer_pos - 1],
                    local_ranks[i, layer_pos],
                ],
                dtype=np.int32,
            )
            rank = int(np.min(corner_ranks))
            field[i, layer_pos, singularity_pos] = float(np.unique(corner_ranks).size > 1)
            if rank < 1:
                field[i, layer_pos, singularity_pos] = 1.0
                continue

            ids_depth_prev = np.union1d(
                neighbor_ids[prev, layer_pos - 1, :k_tangent],
                neighbor_ids[prev, layer_pos, :k_tangent],
            )
            depth_prev, depth_prev_residual = fit_local_tangent_connection(
                prev_ref[ids_depth_prev, layer_pos - 1],
                prev_ref[ids_depth_prev, layer_pos],
                bases[prev, layer_pos - 1],
                bases[prev, layer_pos],
                rank,
            )
            ids_time_right = np.union1d(
                neighbor_ids[prev, layer_pos, :k_tangent],
                neighbor_ids[i, layer_pos, :k_tangent],
            )
            time_right, time_right_residual = fit_local_tangent_connection(
                prev_ref[ids_time_right, layer_pos],
                curr_ref[ids_time_right, layer_pos],
                bases[prev, layer_pos],
                bases[i, layer_pos],
                rank,
            )
            ids_time_left = np.union1d(
                neighbor_ids[prev, layer_pos - 1, :k_tangent],
                neighbor_ids[i, layer_pos - 1, :k_tangent],
            )
            time_left, time_left_residual = fit_local_tangent_connection(
                prev_ref[ids_time_left, layer_pos - 1],
                curr_ref[ids_time_left, layer_pos - 1],
                bases[prev, layer_pos - 1],
                bases[i, layer_pos - 1],
                rank,
            )
            ids_depth_curr = np.union1d(
                neighbor_ids[i, layer_pos - 1, :k_tangent],
                neighbor_ids[i, layer_pos, :k_tangent],
            )
            depth_curr, depth_curr_residual = fit_local_tangent_connection(
                curr_ref[ids_depth_curr, layer_pos - 1],
                curr_ref[ids_depth_curr, layer_pos],
                bases[i, layer_pos - 1],
                bases[i, layer_pos],
                rank,
            )
            if not np.isfinite(
                [
                    depth_prev_residual,
                    time_right_residual,
                    time_left_residual,
                    depth_curr_residual,
                ]
            ).all():
                continue
            path_depth_then_time = time_right @ depth_prev
            path_time_then_depth = depth_curr @ time_left
            field[i, layer_pos, holonomy_pos] = connection_path_discrepancy(
                path_depth_then_time,
                path_time_then_depth,
            )
    return field


def shared_jl_projection(hidden_dim: int, projection_dim: int, rng: np.random.Generator) -> np.ndarray:
    out_dim = int(hidden_dim) if int(projection_dim) <= 0 else min(int(hidden_dim), int(projection_dim))
    if out_dim == int(hidden_dim):
        return np.eye(hidden_dim, dtype=np.float32)
    mat = rng.normal(0.0, 1.0 / np.sqrt(out_dim), size=(hidden_dim, out_dim))
    return mat.astype(np.float32)


def topk_sqdist(query: np.ndarray, ref: np.ndarray, *, k: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    k = max(1, min(int(k), ref.shape[0]))
    chunk_size = max(1, int(chunk_size))
    out_d = np.empty((query.shape[0], k), dtype=np.float32)
    out_i = np.empty((query.shape[0], k), dtype=np.int64)
    ref_norm = np.sum(ref * ref, axis=1)[None, :]
    for start in range(0, query.shape[0], chunk_size):
        q = query[start : start + chunk_size]
        d2 = np.sum(q * q, axis=1, keepdims=True) + ref_norm - 2.0 * (q @ ref.T)
        d2 = np.maximum(d2, 0.0)
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        part = np.take_along_axis(d2, idx, axis=1)
        order = np.argsort(part, axis=1)
        out_i[start : start + q.shape[0]] = np.take_along_axis(idx, order, axis=1)
        out_d[start : start + q.shape[0]] = np.take_along_axis(part, order, axis=1)
    return out_d, out_i


def lid_mle(d2: np.ndarray) -> np.ndarray:
    """Levina-Bickel-style local intrinsic dimension using kNN radii."""

    d = np.sqrt(np.maximum(np.asarray(d2, dtype=np.float64), 0.0))
    out = np.full(d.shape[0], np.nan, dtype=np.float64)
    if d.shape[1] < 3:
        return out
    rk = d[:, -1]
    logs = np.log(np.maximum(rk[:, None], EPS) / np.maximum(d[:, :-1], EPS))
    den = np.mean(logs, axis=1)
    valid = np.isfinite(den) & (den > EPS) & (rk > EPS)
    out[valid] = 1.0 / den[valid]
    return out


def neighbor_rewire(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError("neighbor arrays must have identical shapes")
    out = np.empty(a.shape[0], dtype=np.float32)
    for i in range(a.shape[0]):
        overlap = np.intersect1d(a[i], b[i], assume_unique=False).size
        out[i] = 1.0 - float(overlap) / max(a.shape[1], 1)
    return out


def local_tangent_basis(points: np.ndarray, rank: int) -> np.ndarray:
    basis, _ = local_tangent_frame(points, rank)
    return basis


def local_tangent_frame(points: np.ndarray, rank: int) -> tuple[np.ndarray, int]:
    """Return a fixed-rank frame plus the neighborhood's numerical rank."""

    points = np.asarray(points, dtype=np.float64)
    rank = max(1, min(int(rank), points.shape[0] - 1, points.shape[1]))
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    threshold = max(float(singular[0]) * 1e-5, EPS) if singular.size else EPS
    effective_rank = int(np.sum(singular > threshold))
    return vt[:rank].T.astype(np.float32), effective_rank


def orthogonal_row_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Train-only Procrustes map Q with ``source @ Q ~= target``."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = source - np.mean(source, axis=0, keepdims=True)
    target = target - np.mean(target, axis=0, keepdims=True)
    cross = source.T @ target
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    return (u @ vt).astype(np.float32)


def subspace_sine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square sine of principal angles, in ``[0, 1]``."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rank = min(a.shape[1], b.shape[1])
    if rank == 0:
        return float("nan")
    singular = np.linalg.svd(a[:, :rank].T @ b[:, :rank], compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    return float(np.sqrt(np.mean(np.maximum(1.0 - singular * singular, 0.0))))


def fit_local_tangent_connection(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_basis: np.ndarray,
    target_basis: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, float]:
    """Fit a paired, local orthogonal transport in tangent coordinates.

    Rows correspond to the same reference-chain identities at the source and
    target cells.  The returned matrix maps source column coordinates to target
    column coordinates; the residual is a scale-free transport-confidence
    diagnostic in ``[0, 1]``.
    """

    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    rank = max(1, min(int(rank), source_basis.shape[1], target_basis.shape[1]))
    source = source_points - np.mean(source_points, axis=0, keepdims=True)
    target = target_points - np.mean(target_points, axis=0, keepdims=True)
    source_coord = source @ np.asarray(source_basis[:, :rank], dtype=np.float64)
    target_coord = target @ np.asarray(target_basis[:, :rank], dtype=np.float64)
    source_norm = float(np.linalg.norm(source_coord, ord="fro"))
    target_norm = float(np.linalg.norm(target_coord, ord="fro"))
    if source_norm <= EPS or target_norm <= EPS:
        return np.eye(rank, dtype=np.float32), float("nan")
    source_coord /= source_norm
    target_coord /= target_norm
    row_map = polar_connection(source_coord.T @ target_coord)
    residual = float(
        np.clip(
            np.linalg.norm(source_coord @ row_map - target_coord, ord="fro") / 2.0,
            0.0,
            1.0,
        )
    )
    return row_map.T.astype(np.float32), residual


def polar_connection(overlap: np.ndarray) -> np.ndarray:
    """Closest orthogonal coordinate transport for two local tangent frames."""

    overlap = np.asarray(overlap, dtype=np.float64)
    u, _, vt = np.linalg.svd(overlap, full_matrices=False)
    return (u @ vt).astype(np.float32)


def connection_path_discrepancy(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """Normalized Frobenius distance between two orthogonal path transports."""

    path_a = np.asarray(path_a, dtype=np.float64)
    path_b = np.asarray(path_b, dtype=np.float64)
    if path_a.shape != path_b.shape or path_a.ndim != 2 or path_a.shape[0] != path_a.shape[1]:
        raise ValueError("path transports must be same-shaped square matrices")
    rank = path_a.shape[0]
    if rank == 0 or not np.isfinite(path_a).all() or not np.isfinite(path_b).all():
        return float("nan")
    return float(np.linalg.norm(path_a - path_b, ord="fro") / (2.0 * np.sqrt(rank)))


def reduce_layer_time_field(field: np.ndarray) -> dict[str, np.ndarray]:
    names = {name: i for i, name in enumerate(LAYER_TIME_FIELD_NAMES)}
    lid = field[..., names["lid"]]
    depth_rewire = field[..., names["depth_neighbor_rewire"]]
    time_rewire = field[..., names["time_neighbor_rewire"]]
    depth_tangent = field[..., names["depth_tangent_drift"]]
    time_tangent = field[..., names["time_tangent_drift"]]
    holonomy = field[..., names["plaquette_holonomy"]]
    rank_singularity = field[..., names["rank_singularity"]]
    return {
        "ltg_lid_median": _nanquantile(lid, 0.50),
        "ltg_lid_depth_iqr": _nanquantile(lid, 0.75) - _nanquantile(lid, 0.25),
        "ltg_depth_rewire_peak": _nanmax(depth_rewire),
        "ltg_time_rewire_median": _nanquantile(time_rewire, 0.50),
        "ltg_depth_tangent_peak": _nanmax(depth_tangent),
        "ltg_time_tangent_median": _nanquantile(time_tangent, 0.50),
        "ltg_holonomy_peak": _nanmax(holonomy),
        "ltg_rank_singularity_rate": _nanmean(rank_singularity),
    }


def _nanquantile(x: np.ndarray, q: float) -> np.ndarray:
    out = np.full(x.shape[:2], np.nan, dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            vals = x[i, j]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                out[i, j] = float(np.quantile(vals, q))
    return out


def _nanmax(x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape[:2], np.nan, dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            vals = x[i, j]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                out[i, j] = float(np.max(vals))
    return out


def _nanmean(x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape[:2], np.nan, dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            vals = x[i, j]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                out[i, j] = float(np.mean(vals))
    return out


def _append_step_metrics(packed: dict[str, np.ndarray], metric_map: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    old = np.asarray(packed["step_scores"], dtype=np.float32)
    names = [str(x) for x in packed["step_score_names"].tolist()]
    additions = [name for name in LAYER_TIME_STEP_METRICS if name not in names]
    if additions:
        old = np.concatenate(
            [old, np.full((*old.shape[:2], len(additions)), np.nan, dtype=np.float32)],
            axis=2,
        )
        names.extend(additions)
    for name, values in metric_map.items():
        old[:, :, names.index(name)] = np.asarray(values, dtype=np.float32)
    packed["step_scores"] = old
    packed["step_score_names"] = np.asarray(names, dtype="<U96")
    return packed


def _append_chain_metrics(
    packed: dict[str, np.ndarray],
    metric_map: Mapping[str, np.ndarray],
    n_steps: np.ndarray,
) -> dict[str, np.ndarray]:
    old = np.asarray(packed["chain_scores"], dtype=np.float32)
    names = [str(x) for x in packed["chain_score_names"].tolist()]
    values: dict[str, np.ndarray] = {}
    for metric_name, matrix in metric_map.items():
        mean = np.full(n_steps.shape[0], np.nan, dtype=np.float32)
        peak = np.full(n_steps.shape[0], np.nan, dtype=np.float32)
        for row, count in enumerate(n_steps):
            vals = np.asarray(matrix[row, : int(count)], dtype=np.float32)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                mean[row] = float(np.mean(vals))
                peak[row] = float(np.max(vals))
        values[f"mean_{metric_name}"] = mean
        values[f"max_{metric_name}"] = peak
    additions = [name for name in values if name not in names]
    if additions:
        old = np.concatenate([old, np.full((old.shape[0], len(additions)), np.nan, dtype=np.float32)], axis=1)
        names.extend(additions)
    for name, value in values.items():
        old[:, names.index(name)] = value
    packed["chain_scores"] = old
    packed["chain_score_names"] = np.asarray(names, dtype="<U96")
    return packed


def _validate_layers(layers: np.ndarray, require_contiguous: bool) -> None:
    layers = np.asarray(layers, dtype=np.int64)
    if layers.size < 2:
        raise ValueError("whole-layer evolution requires at least two layer depths")
    if np.unique(layers).size != layers.size or np.any(np.diff(layers) <= 0):
        raise ValueError("layer labels must be unique and strictly increasing")
    if require_contiguous and np.any(np.diff(layers) != 1):
        raise ValueError(
            "layer-time geometry requires contiguous depths by default; re-extract with "
            "--geometry_only (all post-block layers), or explicitly allow sparse layers for a pilot"
        )
