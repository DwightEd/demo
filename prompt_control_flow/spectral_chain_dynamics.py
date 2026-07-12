from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .evaluate import auprc, auroc
from .metrics import summarize_step_metrics

EPS = 1e-8

SPECTRAL_STEP_METRICS = (
    "sd_tube_dist",
    "sd_spectral_leak",
    "sd_tangent_off",
    "sd_committor",
    "sd_step_speed",
)

SPECTRAL_CHAIN_METRICS = (
    "sd_curve_efficiency",
    "sd_path_length_per_phase",
    "sd_tube_auc",
    "sd_committor_auc",
    "sd_leak_auc",
)


@dataclass(frozen=True)
class SpectralChainConfig:
    """Configuration for whole-chain spectral-manifold validation.

    The audit is deliberately response/trajectory-centric.  It consumes an
    existing vector bank such as ``step_state_vectors`` and treats every chain
    as a phase-normalized curve on a cross-fitted spectral chart.  No test-chain
    label is used to construct the chart, healthy tube, or committor reference.
    """

    vector_key: str = "step_state_vectors"
    n_folds: int = 5
    n_modes: int = 12
    low_modes: int = 4
    max_landmarks: int = 1800
    kernel_k: int = 20
    committor_k: int = 30
    tube_k: int = 20
    phase_bandwidth: float = 0.15
    tangent_k: int = 32
    tangent_rank: int = 8
    curve_grid: int = 25
    diffusion_time: float = 1.0
    random_seed: int = 13


@dataclass
class VectorBank:
    vectors: np.ndarray
    row_idx: np.ndarray
    step_idx: np.ndarray
    chain_idx: np.ndarray
    layers: np.ndarray

    def take(self, mask: np.ndarray) -> "VectorBank":
        return VectorBank(
            vectors=self.vectors[mask],
            row_idx=self.row_idx[mask],
            step_idx=self.step_idx[mask],
            chain_idx=self.chain_idx[mask],
            layers=self.layers,
        )


def append_spectral_chain_dynamics(
    metrics: Mapping[str, Any],
    cfg: SpectralChainConfig = SpectralChainConfig(),
) -> dict[str, np.ndarray]:
    """Append cross-fitted spectral-chain dynamics scores to packed metrics.

    This validator can run on current extraction outputs as long as they contain
    a flat vector bank.  With ``step_state_vectors`` the curves are hidden-state
    trajectories; with ``step_vectors`` they are residual-write trajectories.
    """

    packed = canonicalize_spectral_input(metrics, cfg.vector_key)
    bank = load_vector_bank(packed, cfg.vector_key)

    n_steps = np.asarray(packed["n_steps"], dtype=np.int64)
    valid = (bank.row_idx >= 0) & (bank.step_idx >= 0) & (bank.step_idx < n_steps[np.maximum(bank.row_idx, 0)])
    bank = bank.take(valid)
    if bank.vectors.shape[0] < 8:
        raise ValueError(f"{cfg.vector_key} contains too few valid points for spectral-chain audit")

    chain_idx = np.asarray(packed["chain_idx"], dtype=np.int64)
    gold = np.asarray(packed["gold_error_step"], dtype=np.int64)
    response_y = response_error_labels(packed)
    point_labels = _point_error_labels(response_y, gold, bank.row_idx, bank.step_idx)
    phases = _point_phases(n_steps, bank.row_idx, bank.step_idx)

    max_steps = int(np.max(n_steps))
    step_metric_map = {name: np.full((len(n_steps), max_steps), np.nan, dtype=np.float64) for name in SPECTRAL_STEP_METRICS}
    chain_metric_map = {name: np.full(len(n_steps), np.nan, dtype=np.float64) for name in SPECTRAL_CHAIN_METRICS}

    rng = np.random.default_rng(int(cfg.random_seed))
    folds = _make_chain_folds(np.arange(len(chain_idx), dtype=np.int64), cfg.n_folds, rng)
    for fold in folds:
        test_rows = np.isin(bank.row_idx, fold)
        train_rows = ~test_rows
        if np.sum(test_rows) == 0 or np.sum(train_rows) < 8:
            continue
        train_healthy = train_rows & (point_labels == 0)
        if np.sum(train_healthy) < 4:
            train_healthy = train_rows
        model = _fit_spectral_chart(
            bank.vectors[train_rows],
            point_labels[train_rows],
            phases[train_rows],
            cfg,
            rng,
        )
        scores = _score_fold(
            model,
            bank.vectors,
            bank.row_idx,
            bank.step_idx,
            train_rows=train_rows,
            train_healthy=train_healthy,
            test_rows=test_rows,
            point_labels=point_labels,
            phases=phases,
            n_steps=n_steps,
            cfg=cfg,
        )
        for name, vals in scores.step_scores.items():
            _scatter_step_values(step_metric_map[name], bank.row_idx[test_rows], bank.step_idx[test_rows], vals)
        for local_row, vals in scores.chain_scores.items():
            for name, val in vals.items():
                chain_metric_map[name][int(local_row)] = float(val)

    packed = _append_step_metric_map(packed, step_metric_map)
    # Include standard response summaries of the new step curves.
    for i in range(len(n_steps)):
        series = {name: vals[i, : int(n_steps[i])] for name, vals in step_metric_map.items()}
        for name, val in summarize_step_metrics(series).items():
            chain_metric_map.setdefault(name, np.full(len(n_steps), np.nan, dtype=np.float64))[i] = val
    packed = _append_chain_metric_map(packed, chain_metric_map)
    packed["spectral_chain_vector_key"] = np.asarray(str(cfg.vector_key), dtype=object)
    packed["spectral_chain_config"] = np.asarray(str(cfg), dtype=object)
    packed["spectral_chain_validation"] = np.asarray(_validation_summary(packed), dtype=object)
    return packed


@dataclass
class FoldScores:
    step_scores: dict[str, np.ndarray]
    chain_scores: dict[int, dict[str, float]]


@dataclass
class SpectralChart:
    mu: np.ndarray
    scale: np.ndarray
    landmarks: np.ndarray
    landmark_labels: np.ndarray
    landmark_phases: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    train_phase: np.ndarray
    train_coords: np.ndarray
    healthy_x: np.ndarray
    healthy_phase: np.ndarray
    healthy_coords: np.ndarray
    eigvecs: np.ndarray
    eigvals: np.ndarray
    degree: np.ndarray
    bandwidth: np.ndarray
    diffusion_time: float


def load_vector_bank(metrics: Mapping[str, Any], vector_key: str) -> VectorBank:
    if vector_key not in metrics:
        fallback = "step_vectors" if vector_key == "step_state_vectors" else "step_state_vectors"
        if fallback in metrics:
            raise ValueError(f"{vector_key} not found; available fallback is {fallback}. Pass --vector_key {fallback} explicitly.")
        raise ValueError(f"{vector_key} not found. Re-extract with --store_step_state_vectors or --store_step_vectors.")
    vectors = np.asarray(metrics[vector_key], dtype=np.float32)
    prefix = "step_state_vector" if vector_key == "step_state_vectors" else "step_vector"
    chain_ids = np.asarray(metrics[f"{prefix}_chain_idx"], dtype=np.int64)
    step_idx = np.asarray(metrics[f"{prefix}_step_idx"], dtype=np.int64)
    all_chain_ids = np.asarray(metrics["chain_idx"], dtype=np.int64)
    row_lookup = {int(c): i for i, c in enumerate(all_chain_ids.tolist())}
    row_idx = np.asarray([row_lookup.get(int(c), -1) for c in chain_ids], dtype=np.int64)
    layers = np.asarray(metrics.get(f"{prefix}_layers", metrics.get("layers", [])), dtype=np.int64)
    return VectorBank(vectors=vectors, row_idx=row_idx, step_idx=step_idx, chain_idx=chain_ids, layers=layers)


def canonicalize_spectral_input(metrics: Mapping[str, Any], vector_key: str = "step_state_vectors") -> dict[str, np.ndarray]:
    """Return a packed metric-like object for spectral-chain validation.

    Two data regimes are supported:

    1. Mechanism extraction outputs with ``step_state_vectors`` or
       ``step_vectors``.
    2. Canonical ProcessBench full files documented in ``md/guides/DATA.md``
       with ``stepvec`` object arrays shaped per chain as ``(T, L, d)``.
    """

    if "chain_idx" in metrics and ("step_state_vectors" in metrics or "step_vectors" in metrics):
        return {k: np.array(v, copy=True) for k, v in metrics.items()}
    if "stepvec" not in metrics:
        return {k: np.array(v, copy=True) for k, v in metrics.items()}

    stepvec = np.asarray(metrics["stepvec"], dtype=object)
    n = int(len(stepvec))
    n_steps = np.asarray([_chain_step_count(stepvec[i]) for i in range(n)], dtype=np.int64)
    max_steps = int(np.max(n_steps)) if n else 0
    chain_idx = np.arange(n, dtype=np.int64)
    problem_id = np.asarray(metrics.get("problem_ids", chain_idx), dtype=np.int64)
    gold = np.asarray(metrics.get("gold_error_step", metrics.get("labels", np.full(n, -1))), dtype=np.int64)
    correct = np.asarray(metrics.get("is_correct_strict", metrics.get("is_correct", (gold < 0).astype(np.int64))), dtype=np.int64)
    sample_idx = np.asarray(metrics.get("sample_idx", np.full(n, -1)), dtype=np.int64)

    layers = np.asarray(metrics.get("sv_layers", metrics.get("layers_used", np.arange(_infer_num_layers(stepvec)))), dtype=np.int64)
    flat = []
    layered = []
    flat_chain = []
    flat_step = []
    step_len = np.full((n, max_steps), np.nan, dtype=np.float32)
    rel_pos = np.full((n, max_steps), np.nan, dtype=np.float32)
    step_ranges = _default_step_ranges(n, max_steps)
    if "step_token_ranges" in metrics:
        raw_ranges = np.asarray(metrics["step_token_ranges"], dtype=object)
        step_ranges = _pack_step_ranges(raw_ranges, n, max_steps)
        for i in range(n):
            for j in range(int(n_steps[i])):
                a, b = step_ranges[i, j]
                if a >= 0 and b >= a:
                    step_len[i, j] = float(b - a + 1)
    for i in range(n):
        arr = _chain_stepvec_array(stepvec[i])
        for j in range(min(arr.shape[0], int(n_steps[i]))):
            flat.append(arr[j].reshape(-1).astype(np.float32, copy=False))
            layered.append(arr[j].astype(np.float32, copy=False))
            flat_chain.append(i)
            flat_step.append(j)
            if not np.isfinite(step_len[i, j]):
                step_len[i, j] = 1.0
            rel_pos[i, j] = 0.0 if n_steps[i] <= 1 else float(j / max(int(n_steps[i]) - 1, 1))

    vectors = np.asarray(flat, dtype=np.float32)
    packed = {
        "chain_idx": chain_idx,
        "problem_id": problem_id,
        "gold_error_step": gold,
        "is_correct": correct,
        "sample_idx": sample_idx,
        "generator": _metadata_vector(metrics, ("generator", "generators", "model_name"), n),
        "dataset": _metadata_vector(metrics, ("dataset", "datasets", "subset"), n),
        "n_steps": n_steps,
        "step_token_ranges": step_ranges,
        "step_scores": np.stack([step_len, rel_pos], axis=2).astype(np.float32),
        "step_score_names": np.asarray(["step_len", "rel_pos"], dtype="<U96"),
        "chain_scores": np.stack([np.nanmean(step_len, axis=1), np.nanmean(rel_pos, axis=1)], axis=1).astype(np.float32),
        "chain_score_names": np.asarray(["mean_step_len", "mean_rel_pos"], dtype="<U96"),
        "layers": layers,
        vector_key: vectors.astype(np.float16),
        "step_layer_state_vectors": np.asarray(layered, dtype=np.float16),
        f"{_vector_prefix(vector_key)}_chain_idx": np.asarray(flat_chain, dtype=np.int64),
        f"{_vector_prefix(vector_key)}_step_idx": np.asarray(flat_step, dtype=np.int64),
        f"{_vector_prefix(vector_key)}_layers": layers,
        "step_layer_state_vector_chain_idx": np.asarray(flat_chain, dtype=np.int64),
        "step_layer_state_vector_step_idx": np.asarray(flat_step, dtype=np.int64),
        "step_layer_state_vector_layers": layers,
        "state_representation_kind": np.asarray(
            metrics.get("state_representation_kind", "hidden_state"), dtype=object
        ),
        "state_pooling_kind": np.asarray(
            metrics.get("state_pooling_kind", metrics.get("stepvec_mode", "legacy_step_exp")),
            dtype=object,
        ),
        "spectral_source_format": np.asarray("processbench_full_stepvec", dtype=object),
    }
    return packed


def _vector_prefix(vector_key: str) -> str:
    return "step_state_vector" if str(vector_key) == "step_state_vectors" else "step_vector"


def _metadata_vector(metrics: Mapping[str, Any], keys: tuple[str, ...], n: int) -> np.ndarray:
    value: Any = None
    for key in keys:
        if key in metrics:
            value = metrics[key]
            break
    if value is None:
        return np.asarray([""] * int(n), dtype=object)
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        return np.asarray([arr.item()] * int(n), dtype=object)
    if arr.shape[0] != int(n):
        raise ValueError(f"metadata {keys} has length {arr.shape[0]}, expected {n}")
    return arr


def _chain_stepvec_array(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist())
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        return arr[:, None, :]
    if arr.ndim == 1 and arr.size:
        return arr.reshape(1, 1, -1)
    return np.zeros((0, 1, 0), dtype=np.float32)


def _chain_step_count(x: Any) -> int:
    return int(_chain_stepvec_array(x).shape[0])


def _infer_num_layers(stepvec: np.ndarray) -> int:
    for x in stepvec:
        arr = _chain_stepvec_array(x)
        if arr.ndim == 3 and arr.shape[0] > 0:
            return int(arr.shape[1])
    return 0


def _default_step_ranges(n: int, max_steps: int) -> np.ndarray:
    return np.full((int(n), int(max_steps), 2), -1, dtype=np.int32)


def _pack_step_ranges(raw_ranges: np.ndarray, n: int, max_steps: int) -> np.ndarray:
    out = _default_step_ranges(n, max_steps)
    for i in range(int(n)):
        try:
            arr = np.asarray(raw_ranges[i])
            if arr.dtype == object:
                arr = np.asarray(arr.tolist())
            arr = np.asarray(arr, dtype=np.int64)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        take = min(int(max_steps), arr.shape[0])
        out[i, :take, 0] = arr[:take, 0]
        out[i, :take, 1] = arr[:take, 1]
    return out


def response_error_labels(metrics: Mapping[str, Any]) -> np.ndarray:
    if "is_correct" in metrics:
        correct = np.asarray(metrics["is_correct"], dtype=np.float64)
        if np.isfinite(correct).any() and np.any(correct >= 0):
            return (correct == 0).astype(np.int32)
    gold = np.asarray(metrics["gold_error_step"], dtype=np.int64)
    return (gold >= 0).astype(np.int32)


def _point_error_labels(response_y: np.ndarray, gold: np.ndarray, row_idx: np.ndarray, step_idx: np.ndarray) -> np.ndarray:
    out = np.zeros(row_idx.shape[0], dtype=np.int32)
    for i, (r, s) in enumerate(zip(row_idx, step_idx)):
        if r < 0:
            continue
        if response_y[int(r)] <= 0:
            out[i] = 0
            continue
        g = int(gold[int(r)])
        out[i] = 1 if (g < 0 or int(s) >= g) else 0
    return out


def _point_phases(n_steps: np.ndarray, row_idx: np.ndarray, step_idx: np.ndarray) -> np.ndarray:
    denom = np.maximum(n_steps[np.maximum(row_idx, 0)] - 1, 1)
    return np.clip(step_idx / denom, 0.0, 1.0).astype(np.float64)


def _make_chain_folds(chain_idx: np.ndarray, n_folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    chains = np.unique(chain_idx)
    rng.shuffle(chains)
    n_folds = max(2, min(int(n_folds), int(chains.size)))
    return [fold for fold in np.array_split(chains, n_folds) if fold.size]


def _fit_spectral_chart(
    train_vectors: np.ndarray,
    train_labels: np.ndarray,
    train_phases: np.ndarray,
    cfg: SpectralChainConfig,
    rng: np.random.Generator,
) -> SpectralChart:
    mu = np.nanmean(train_vectors, axis=0)
    scale = np.nanstd(train_vectors, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    train_x = ((train_vectors - mu) / scale).astype(np.float32)
    train_x = np.nan_to_num(train_x, nan=0.0, posinf=0.0, neginf=0.0)

    landmark_idx = _choose_landmarks(train_labels, cfg.max_landmarks, rng)
    landmarks = train_x[landmark_idx]
    landmark_labels = train_labels[landmark_idx]
    landmark_phases = train_phases[landmark_idx]

    d2 = _sqdist(landmarks, landmarks)
    bandwidth = _adaptive_bandwidth(d2, cfg.kernel_k)
    k = np.exp(-d2 / np.maximum(np.outer(bandwidth, bandwidth), EPS))
    np.fill_diagonal(k, 1.0)
    degree = np.maximum(np.sum(k, axis=1), EPS)
    s = k / np.sqrt(np.outer(degree, degree))
    eigvals, eigvecs = np.linalg.eigh(s.astype(np.float64))
    order = np.argsort(eigvals)[::-1]
    # Skip the first nearly-constant mode.
    take = order[1 : 1 + min(int(cfg.n_modes), max(1, landmarks.shape[0] - 1))]
    eigvals = np.maximum(eigvals[take], EPS)
    eigvecs = eigvecs[:, take]

    chart = SpectralChart(
        mu=mu.astype(np.float32),
        scale=scale.astype(np.float32),
        landmarks=landmarks.astype(np.float32),
        landmark_labels=landmark_labels.astype(np.int32),
        landmark_phases=landmark_phases.astype(np.float64),
        train_x=train_x,
        train_y=train_labels.astype(np.int32),
        train_phase=train_phases.astype(np.float64),
        train_coords=np.empty((train_x.shape[0], eigvals.size), dtype=np.float32),
        healthy_x=np.empty((0, train_x.shape[1]), dtype=np.float32),
        healthy_phase=np.empty(0, dtype=np.float64),
        healthy_coords=np.empty((0, eigvals.size), dtype=np.float32),
        eigvecs=eigvecs.astype(np.float32),
        eigvals=eigvals.astype(np.float32),
        degree=degree.astype(np.float32),
        bandwidth=bandwidth.astype(np.float32),
        diffusion_time=float(cfg.diffusion_time),
    )
    train_coords = _transform(chart, train_vectors)
    healthy = train_labels == 0
    chart.train_coords = train_coords.astype(np.float32)
    chart.healthy_x = train_x[healthy]
    chart.healthy_phase = train_phases[healthy].astype(np.float64)
    chart.healthy_coords = train_coords[healthy].astype(np.float32)
    if chart.healthy_x.shape[0] == 0:
        chart.healthy_x = train_x
        chart.healthy_phase = train_phases.astype(np.float64)
        chart.healthy_coords = train_coords.astype(np.float32)
    return chart


def _choose_landmarks(labels: np.ndarray, max_landmarks: int, rng: np.random.Generator) -> np.ndarray:
    n = labels.size
    max_landmarks = max(8, min(int(max_landmarks), int(n)))
    if n <= max_landmarks:
        return np.arange(n, dtype=np.int64)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    n_pos = min(pos.size, max(1, max_landmarks // 3))
    n_neg = max_landmarks - n_pos
    picks = []
    if pos.size:
        picks.append(rng.choice(pos, size=n_pos, replace=False))
    if neg.size:
        picks.append(rng.choice(neg, size=min(n_neg, neg.size), replace=False))
    out = np.concatenate(picks) if picks else rng.choice(np.arange(n), size=max_landmarks, replace=False)
    if out.size < max_landmarks:
        rest = np.setdiff1d(np.arange(n), out, assume_unique=False)
        extra = rng.choice(rest, size=min(max_landmarks - out.size, rest.size), replace=False)
        out = np.concatenate([out, extra])
    rng.shuffle(out)
    return out.astype(np.int64)


def _score_fold(
    model: SpectralChart,
    vectors: np.ndarray,
    row_idx: np.ndarray,
    step_idx: np.ndarray,
    *,
    train_rows: np.ndarray,
    train_healthy: np.ndarray,
    test_rows: np.ndarray,
    point_labels: np.ndarray,
    phases: np.ndarray,
    n_steps: np.ndarray,
    cfg: SpectralChainConfig,
) -> FoldScores:
    test_vec = vectors[test_rows]
    test_rows_idx = row_idx[test_rows]
    test_step_idx = step_idx[test_rows]
    test_phase = phases[test_rows]
    coords = _transform(model, test_vec)
    ambient = _standardize_with_model(model, test_vec)

    total_energy = np.sum(coords * coords, axis=1) + EPS
    low = max(1, min(int(cfg.low_modes), coords.shape[1]))
    high_energy = np.sum(coords[:, low:] * coords[:, low:], axis=1) if coords.shape[1] > low else np.zeros(coords.shape[0])
    spectral_leak = high_energy / total_energy

    tube_dist = _tube_distance(coords, test_phase, model.healthy_coords, model.healthy_phase, cfg)
    committor = _committor(coords, model.train_coords, model.train_y, cfg.committor_k)
    tangent_off = _tangent_off_scores(
        ambient,
        coords,
        test_rows_idx,
        test_step_idx,
        vectors,
        row_idx,
        step_idx,
        model,
        cfg,
    )
    step_speed = _step_speeds(coords, test_rows_idx, test_step_idx)

    step_scores = {
        "sd_tube_dist": tube_dist,
        "sd_spectral_leak": spectral_leak,
        "sd_tangent_off": tangent_off,
        "sd_committor": committor,
        "sd_step_speed": step_speed,
    }
    chain_scores = _chain_curve_scores(coords, test_rows_idx, test_step_idx, test_phase, step_scores, n_steps)
    return FoldScores(step_scores=step_scores, chain_scores=chain_scores)


def _transform(model: SpectralChart, vectors: np.ndarray) -> np.ndarray:
    x = _standardize_with_model(model, vectors)
    d2 = _sqdist(x, model.landmarks)
    # Use the landmark bandwidth only; query bandwidth is the median local
    # landmark scale to keep out-of-sample extension stable.
    q_bw = np.median(model.bandwidth)
    k = np.exp(-d2 / np.maximum(q_bw * model.bandwidth[None, :], EPS))
    degree_q = np.maximum(np.sum(k, axis=1), EPS)
    s_q = k / np.sqrt(degree_q[:, None] * model.degree[None, :])
    coords = (s_q @ model.eigvecs) / model.eigvals[None, :]
    coords = coords * (model.eigvals[None, :] ** float(model.diffusion_time))
    return np.nan_to_num(coords.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _standardize_with_model(model: SpectralChart, vectors: np.ndarray) -> np.ndarray:
    x = (np.asarray(vectors, dtype=np.float32) - model.mu) / model.scale
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _adaptive_bandwidth(d2: np.ndarray, k: int) -> np.ndarray:
    n = d2.shape[0]
    kk = max(1, min(int(k), max(1, n - 1)))
    part = np.partition(d2, kth=kk, axis=1)[:, kk]
    bw = np.sqrt(np.maximum(part, EPS))
    med = np.median(bw[np.isfinite(bw) & (bw > 0)])
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return np.where(bw <= 0, med, bw).astype(np.float64)


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    return np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0).astype(np.float64)


def _knn_indices(query: np.ndarray, ref: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if query.shape[0] == 0 or ref.shape[0] == 0:
        return np.empty((query.shape[0], 0), dtype=np.float64), np.empty((query.shape[0], 0), dtype=np.int64)
    k = max(1, min(int(k), ref.shape[0]))
    d2 = _sqdist(query, ref)
    idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(d2, idx, axis=1)
    order = np.argsort(part, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    part = np.take_along_axis(part, order, axis=1)
    return part, idx


def _tube_distance(coords: np.ndarray, phase: np.ndarray, healthy_coords: np.ndarray, healthy_phase: np.ndarray, cfg: SpectralChainConfig) -> np.ndarray:
    out = np.full(coords.shape[0], np.nan, dtype=np.float64)
    for i in range(coords.shape[0]):
        band = np.abs(healthy_phase - phase[i]) <= float(cfg.phase_bandwidth)
        ref = healthy_coords[band]
        if ref.shape[0] < 3:
            ref = healthy_coords
        d2, _ = _knn_indices(coords[i : i + 1], ref, cfg.tube_k)
        out[i] = float(np.sqrt(np.mean(d2[0]))) if d2.size else np.nan
    return out


def _committor(coords: np.ndarray, train_coords: np.ndarray, train_y: np.ndarray, k: int) -> np.ndarray:
    d2, idx = _knn_indices(coords, train_coords, k)
    if idx.shape[1] == 0:
        return np.full(coords.shape[0], np.nan, dtype=np.float64)
    weights = 1.0 / (np.sqrt(d2) + 1e-4)
    labels = train_y[idx]
    return np.sum(weights * labels, axis=1) / np.maximum(np.sum(weights, axis=1), EPS)


def _tangent_off_scores(
    ambient: np.ndarray,
    coords: np.ndarray,
    rows: np.ndarray,
    steps: np.ndarray,
    all_vectors: np.ndarray,
    all_rows: np.ndarray,
    all_steps: np.ndarray,
    model: SpectralChart,
    cfg: SpectralChainConfig,
) -> np.ndarray:
    out = np.full(coords.shape[0], np.nan, dtype=np.float64)
    row_step_to_vec = {(int(r), int(s)): i for i, (r, s) in enumerate(zip(all_rows, all_steps))}
    _, nn = _knn_indices(coords, model.healthy_coords, cfg.tangent_k)
    for i in range(coords.shape[0]):
        nxt = row_step_to_vec.get((int(rows[i]), int(steps[i]) + 1))
        if nxt is None or nn.shape[1] < 2:
            continue
        x_next = _standardize_with_model(model, all_vectors[nxt : nxt + 1])[0]
        delta = x_next - ambient[i]
        denom = float(np.dot(delta, delta))
        if denom <= EPS:
            out[i] = 0.0
            continue
        local = model.healthy_x[nn[i]]
        basis = _local_pca_basis(local, cfg.tangent_rank)
        if basis.size == 0:
            continue
        proj = basis @ (basis.T @ delta)
        out[i] = float(np.clip(1.0 - np.dot(proj, proj) / denom, 0.0, 1.0))
    return out


def _local_pca_basis(x: np.ndarray, rank: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return np.empty((x.shape[1], 0), dtype=np.float64)
    xc = x - np.mean(x, axis=0, keepdims=True)
    gram = xc @ xc.T
    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    keep = vals > (float(vals[0]) * 1e-6 if vals.size else 0.0)
    keep_idx = np.where(keep)[0][: max(1, min(int(rank), x.shape[0] - 1))]
    if keep_idx.size == 0:
        return np.empty((x.shape[1], 0), dtype=np.float64)
    basis = xc.T @ (vecs[:, keep_idx] / np.sqrt(np.maximum(vals[keep_idx], EPS))[None, :])
    q, _ = np.linalg.qr(basis)
    return q[:, : keep_idx.size]


def _step_speeds(coords: np.ndarray, rows: np.ndarray, steps: np.ndarray) -> np.ndarray:
    out = np.full(coords.shape[0], np.nan, dtype=np.float64)
    loc = {(int(r), int(s)): i for i, (r, s) in enumerate(zip(rows, steps))}
    for i, (r, s) in enumerate(zip(rows, steps)):
        nxt = loc.get((int(r), int(s) + 1))
        if nxt is not None:
            out[i] = float(np.linalg.norm(coords[nxt] - coords[i]))
    return out


def _chain_curve_scores(
    coords: np.ndarray,
    rows: np.ndarray,
    steps: np.ndarray,
    phases: np.ndarray,
    step_scores: Mapping[str, np.ndarray],
    n_steps: np.ndarray,
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for r in np.unique(rows):
        idx = np.where(rows == r)[0]
        order = idx[np.argsort(steps[idx])]
        if order.size == 0:
            continue
        curve = coords[order]
        if curve.shape[0] >= 2:
            seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
            path = float(np.sum(seg))
            chord = float(np.linalg.norm(curve[-1] - curve[0]))
            efficiency = chord / max(path, EPS)
            path_per_phase = path / max(float(phases[order[-1]] - phases[order[0]]), 1.0 / max(int(n_steps[int(r)]), 1))
        else:
            efficiency = np.nan
            path_per_phase = np.nan
        vals = {
            "sd_curve_efficiency": efficiency,
            "sd_path_length_per_phase": path_per_phase,
            "sd_tube_auc": _trapz_over_phase(phases[order], step_scores["sd_tube_dist"][order]),
            "sd_committor_auc": _trapz_over_phase(phases[order], step_scores["sd_committor"][order]),
            "sd_leak_auc": _trapz_over_phase(phases[order], step_scores["sd_spectral_leak"][order]),
        }
        out[int(r)] = vals
    return out


def _trapz_over_phase(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) == 0:
        return float("nan")
    if np.sum(ok) == 1:
        return float(y[ok][0])
    x = x[ok]
    y = y[ok]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = max(float(x[-1] - x[0]), EPS)
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(y, x)
    else:
        area = np.trapz(y, x)
    return float(area / span)


def _scatter_step_values(target: np.ndarray, rows: np.ndarray, steps: np.ndarray, vals: np.ndarray) -> None:
    for r, s, v in zip(rows, steps, vals):
        if 0 <= int(r) < target.shape[0] and 0 <= int(s) < target.shape[1]:
            target[int(r), int(s)] = float(v)


def _append_step_metric_map(packed: dict[str, np.ndarray], metric_map: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    old = np.asarray(packed["step_scores"], dtype=np.float32)
    names = [str(x) for x in packed["step_score_names"].tolist()]
    add_names = [name for name in metric_map if name not in names]
    if add_names:
        extra = np.full((old.shape[0], old.shape[1], len(add_names)), np.nan, dtype=np.float32)
        packed["step_scores"] = np.concatenate([old, extra], axis=2)
        names.extend(add_names)
        packed["step_score_names"] = np.asarray(names, dtype="<U96")
    for name, vals in metric_map.items():
        k = names.index(name)
        arr = np.asarray(vals, dtype=np.float32)
        packed["step_scores"][:, : arr.shape[1], k] = arr[:, : packed["step_scores"].shape[1]]
    return packed


def _append_chain_metric_map(packed: dict[str, np.ndarray], metric_map: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    old = np.asarray(packed["chain_scores"], dtype=np.float32)
    names = [str(x) for x in packed["chain_score_names"].tolist()]
    add_names = [name for name in metric_map if name not in names]
    if add_names:
        extra = np.full((old.shape[0], len(add_names)), np.nan, dtype=np.float32)
        packed["chain_scores"] = np.concatenate([old, extra], axis=1)
        names.extend(add_names)
        packed["chain_score_names"] = np.asarray(names, dtype="<U96")
    for name, vals in metric_map.items():
        k = names.index(name)
        packed["chain_scores"][:, k] = np.asarray(vals, dtype=np.float32)
    return packed


def _validation_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    y = response_error_labels(metrics)
    names = [str(x) for x in metrics["chain_score_names"].tolist()]
    scores = np.asarray(metrics["chain_scores"], dtype=np.float64)
    out = {}
    for name in names:
        if name.startswith("sd_") or name.startswith("mean_sd_") or name.startswith("max_sd_") or name.startswith("top20_mean_sd_"):
            k = names.index(name)
            out[name] = {"auroc": auroc(y, scores[:, k]), "auprc": auprc(y, scores[:, k])}
    return out
