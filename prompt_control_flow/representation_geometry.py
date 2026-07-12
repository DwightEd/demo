from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .config import MetricNames
from .metrics import summarize_step_metrics


EPS = 1e-12


@dataclass(frozen=True)
class GeometryAuditConfig:
    """Cross-fitted representation-geometry audit settings."""

    vector_key: str = "step_state_vectors"
    n_folds: int = 5
    knn_k: int = 20
    pca_var: float = 0.90
    max_pca_rank: int = 64
    random_projection_dim: int = 128
    layer_projection_dim: int = 64
    random_seed: int = 13
    chunk_size: int = 256


GEOMETRY_STEP_METRICS = (
    MetricNames.GEOM_BOUNDARY_PROJ,
    MetricNames.GEOM_HEALTHY_RESIDUAL,
    MetricNames.GEOM_LID,
    MetricNames.GEOM_KNN_ERROR_FRAC,
    MetricNames.GEOM_KNN_LABEL_ENTROPY,
    MetricNames.GEOM_LOCAL_SPEC_ENTROPY,
    MetricNames.GEOM_LAYER_NBR_INSTABILITY,
    MetricNames.GEOM_COMPARTMENT_SCORE,
)


def append_geometry_audit(metrics: Mapping[str, Any], cfg: GeometryAuditConfig = GeometryAuditConfig()) -> dict[str, np.ndarray]:
    """Append nonparametric geometry scores to a packed mechanism npz.

    The audit is chain-level cross-fitted.  For each fold, geometry references
    are built only from training chains: healthy step states, first-error step
    states, and their local neighborhoods.  Test-chain steps are then scored
    without using their labels to build centroids or neighborhoods.
    """

    packed = {str(k): np.asarray(v) for k, v in metrics.items()}
    bank = load_step_vector_bank(packed, cfg.vector_key)
    if bank.vectors.shape[0] == 0:
        raise ValueError(f"{cfg.vector_key} is empty")

    chain_idx = np.asarray(packed["chain_idx"], dtype=np.int64)
    n_steps = np.asarray(packed["n_steps"], dtype=np.int64)
    max_steps = int(np.nanmax(n_steps)) if n_steps.size else 0
    chain_to_row = {int(c): i for i, c in enumerate(chain_idx.tolist())}
    row_idx = np.asarray([chain_to_row.get(int(c), -1) for c in bank.chain_idx], dtype=np.int64)
    valid = (row_idx >= 0) & (bank.step_idx >= 0) & (bank.step_idx < n_steps[np.maximum(row_idx, 0)])
    bank = bank.take(valid)
    row_idx = row_idx[valid]
    if bank.vectors.shape[0] == 0:
        raise ValueError(f"{cfg.vector_key} has no rows matching chain_idx/n_steps")

    labels = _row_labels(packed, row_idx, bank.step_idx)
    folds = _make_chain_folds(bank.chain_idx, cfg.n_folds, cfg.random_seed)
    scores = {name: np.full(bank.vectors.shape[0], np.nan, dtype=np.float64) for name in GEOMETRY_STEP_METRICS}
    rng = np.random.default_rng(cfg.random_seed)
    layer_slices = _layer_slices(bank.vectors.shape[1], bank.layers)

    for fold in folds:
        test_mask = np.isin(bank.chain_idx, fold)
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            continue
        train_healthy = train_mask & labels.healthy
        train_error = train_mask & labels.first_error
        if np.sum(train_healthy) < max(5, cfg.knn_k // 2) or np.sum(train_error) < 2:
            continue

        fold_scores = _score_fold(
            bank.vectors,
            train_mask=train_mask,
            train_healthy=train_healthy,
            train_error=train_error,
            test_mask=test_mask,
            labels=labels,
            layer_slices=layer_slices,
            cfg=cfg,
            rng=rng,
        )
        test_indices = np.where(test_mask)[0]
        for name, vals in fold_scores.items():
            scores[name][test_indices] = vals

    geom_step = np.full((len(chain_idx), max_steps, len(GEOMETRY_STEP_METRICS)), np.nan, dtype=np.float32)
    for n, name in enumerate(GEOMETRY_STEP_METRICS):
        for src_i, val in enumerate(scores[name]):
            ri = int(row_idx[src_i])
            sj = int(bank.step_idx[src_i])
            if 0 <= ri < geom_step.shape[0] and 0 <= sj < geom_step.shape[1]:
                geom_step[ri, sj, n] = float(val)

    packed = _append_step_scores(packed, geom_step, GEOMETRY_STEP_METRICS)
    packed = _append_chain_geometry_scores(packed, geom_step, GEOMETRY_STEP_METRICS, n_steps)
    packed["geometry_audit_vector_key"] = np.asarray(str(cfg.vector_key), dtype=object)
    packed["geometry_audit_config"] = np.asarray(str(cfg), dtype=object)
    return packed


@dataclass
class StepVectorBank:
    vectors: np.ndarray
    chain_idx: np.ndarray
    step_idx: np.ndarray
    layers: np.ndarray

    def take(self, mask: np.ndarray) -> "StepVectorBank":
        return StepVectorBank(
            vectors=self.vectors[mask],
            chain_idx=self.chain_idx[mask],
            step_idx=self.step_idx[mask],
            layers=self.layers,
        )


@dataclass
class RowLabels:
    healthy: np.ndarray
    first_error: np.ndarray
    post_error: np.ndarray


def load_step_vector_bank(metrics: Mapping[str, Any], vector_key: str) -> StepVectorBank:
    if vector_key not in metrics:
        fallback = "step_vectors" if vector_key == "step_state_vectors" else "step_state_vectors"
        if fallback in metrics:
            raise ValueError(f"{vector_key} not found; available fallback is {fallback}. Use --vector_key {fallback}.")
        raise ValueError(f"{vector_key} not found. Re-extract with --store_step_state_vectors or --store_step_vectors.")
    prefix = vector_key[:-1] if vector_key.endswith("s") else vector_key
    chain_key = f"{prefix}_chain_idx"
    step_key = f"{prefix}_step_idx"
    layer_key = f"{prefix}_layers"
    for key in [chain_key, step_key, layer_key]:
        if key not in metrics:
            raise ValueError(f"{vector_key} exists but required metadata key {key!r} is missing")
    return StepVectorBank(
        vectors=np.asarray(metrics[vector_key], dtype=np.float32),
        chain_idx=np.asarray(metrics[chain_key], dtype=np.int64),
        step_idx=np.asarray(metrics[step_key], dtype=np.int64),
        layers=np.asarray(metrics[layer_key], dtype=np.int64),
    )


def _row_labels(metrics: Mapping[str, Any], row_idx: np.ndarray, step_idx: np.ndarray) -> RowLabels:
    gold = np.asarray(metrics["gold_error_step"], dtype=np.int64)
    g = gold[row_idx]
    healthy = (g < 0) | (step_idx < g)
    first_error = (g >= 0) & (step_idx == g)
    post_error = (g >= 0) & (step_idx > g)
    return RowLabels(healthy=healthy, first_error=first_error, post_error=post_error)


def _make_chain_folds(chain_idx: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    chains = np.unique(np.asarray(chain_idx, dtype=np.int64))
    rng = np.random.default_rng(seed)
    rng.shuffle(chains)
    n_folds = max(2, min(int(n_folds), chains.size))
    return [fold for fold in np.array_split(chains, n_folds) if fold.size]


def _score_fold(
    vectors: np.ndarray,
    *,
    train_mask: np.ndarray,
    train_healthy: np.ndarray,
    train_error: np.ndarray,
    test_mask: np.ndarray,
    labels: RowLabels,
    layer_slices: list[slice],
    cfg: GeometryAuditConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    mu, scale = _fit_standardizer(vectors[train_mask])
    x = _standardize(vectors, mu, scale)
    x_test = x[test_mask]
    x_healthy = x[train_healthy]
    x_error = x[train_error]
    x_train = x[train_mask]
    y_train = labels.first_error[train_mask].astype(np.int32)

    mu_h = np.mean(x_healthy, axis=0)
    mu_e = np.mean(x_error, axis=0)
    boundary = _unit(mu_e - mu_h)
    boundary_proj = (x_test - mu_h) @ boundary

    basis = _fit_pca_basis(x_healthy, cfg.pca_var, cfg.max_pca_rank)
    healthy_residual = _outside_basis_fraction(x_test - mu_h, basis)

    k = max(2, min(int(cfg.knn_k), x_healthy.shape[0]))
    healthy_d2, healthy_nn = _topk_sqdist(x_test, x_healthy, k=k, chunk_size=cfg.chunk_size)
    lid = _lid_from_sqdist(healthy_d2)

    all_k = max(2, min(int(cfg.knn_k), x_train.shape[0]))
    _, all_nn = _topk_sqdist(x_test, x_train, k=all_k, chunk_size=cfg.chunk_size)
    neighbor_y = y_train[all_nn]
    knn_error_frac = np.mean(neighbor_y, axis=1)
    knn_label_entropy = _binary_entropy(knn_error_frac)

    proj = _random_projection(x.shape[1], min(cfg.random_projection_dim, x.shape[1]), rng)
    x_healthy_proj = x_healthy @ proj
    local_spec_entropy = _local_spectral_entropy(x_healthy_proj, healthy_nn)

    layer_instability = _layer_neighbor_instability(
        x,
        train_mask=train_mask,
        test_mask=test_mask,
        layer_slices=layer_slices,
        cfg=cfg,
        rng=rng,
    )
    compartment = _compartment_score(
        boundary_proj=boundary_proj,
        healthy_residual=healthy_residual,
        lid=lid,
        knn_error_frac=knn_error_frac,
        knn_label_entropy=knn_label_entropy,
        local_spec_entropy=local_spec_entropy,
        layer_instability=layer_instability,
        k=k,
    )
    return {
        MetricNames.GEOM_BOUNDARY_PROJ: boundary_proj,
        MetricNames.GEOM_HEALTHY_RESIDUAL: healthy_residual,
        MetricNames.GEOM_LID: lid,
        MetricNames.GEOM_KNN_ERROR_FRAC: knn_error_frac,
        MetricNames.GEOM_KNN_LABEL_ENTROPY: knn_label_entropy,
        MetricNames.GEOM_LOCAL_SPEC_ENTROPY: local_spec_entropy,
        MetricNames.GEOM_LAYER_NBR_INSTABILITY: layer_instability,
        MetricNames.GEOM_COMPARTMENT_SCORE: compartment,
    }


def _fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def _standardize(x: np.ndarray, mu: np.ndarray, scale: np.ndarray) -> np.ndarray:
    y = (np.asarray(x, dtype=np.float32) - mu) / scale
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def _unit(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return np.zeros_like(x) if n <= EPS else x / n


def _fit_pca_basis(x: np.ndarray, var: float, max_rank: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] < 2:
        return np.zeros((x.shape[1], 0), dtype=np.float32)
    xc = x - np.mean(x, axis=0, keepdims=True)
    gram = xc @ xc.T
    vals, vecs = np.linalg.eigh(gram.astype(np.float64, copy=False))
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    keep = vals > 1e-8
    vals = vals[keep]
    vecs = vecs[:, keep]
    if vals.size == 0:
        return np.zeros((x.shape[1], 0), dtype=np.float32)
    csum = np.cumsum(vals) / max(float(np.sum(vals)), EPS)
    rank = int(np.searchsorted(csum, float(var), side="left") + 1)
    rank = max(1, min(rank, int(max_rank), vals.size))
    basis = xc.T @ (vecs[:, :rank] / np.sqrt(vals[:rank])[None, :])
    q, _ = np.linalg.qr(basis)
    return q[:, :rank].astype(np.float32, copy=False)


def _outside_basis_fraction(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    den = np.sum(x * x, axis=1)
    if basis.size == 0:
        inside = np.zeros(x.shape[0], dtype=np.float64)
    else:
        coeff = x @ basis
        inside = np.sum(coeff * coeff, axis=1)
    frac = 1.0 - inside / np.maximum(den, EPS)
    return np.clip(frac, 0.0, 1.0)


def _topk_sqdist(query: np.ndarray, ref: np.ndarray, *, k: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    k = max(1, min(int(k), ref.shape[0]))
    out_d = np.empty((query.shape[0], k), dtype=np.float32)
    out_i = np.empty((query.shape[0], k), dtype=np.int64)
    ref_norm = np.sum(ref * ref, axis=1)[None, :]
    for start in range(0, query.shape[0], int(chunk_size)):
        q = query[start : start + int(chunk_size)]
        d2 = np.sum(q * q, axis=1, keepdims=True) + ref_norm - 2.0 * (q @ ref.T)
        d2 = np.maximum(d2, 0.0)
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        part = np.take_along_axis(d2, idx, axis=1)
        order = np.argsort(part, axis=1)
        out_i[start : start + q.shape[0]] = np.take_along_axis(idx, order, axis=1)
        out_d[start : start + q.shape[0]] = np.take_along_axis(part, order, axis=1)
    return out_d, out_i


def _lid_from_sqdist(d2: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(np.asarray(d2, dtype=np.float64), EPS))
    rk = d[:, [-1]]
    logs = np.log(np.maximum(d, EPS) / np.maximum(rk, EPS))
    den = np.mean(logs, axis=1)
    vals = np.full(den.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(den) & (den < -EPS)
    vals[ok] = -1.0 / den[ok]
    return vals


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0)
    return h


def _random_projection(in_dim: int, out_dim: int, rng: np.random.Generator) -> np.ndarray:
    out_dim = max(1, min(int(out_dim), int(in_dim)))
    mat = rng.normal(0.0, 1.0 / np.sqrt(out_dim), size=(int(in_dim), out_dim))
    return mat.astype(np.float32)


def _local_spectral_entropy(ref_proj: np.ndarray, nn: np.ndarray) -> np.ndarray:
    out = np.full(nn.shape[0], np.nan, dtype=np.float64)
    for i, ids in enumerate(nn):
        local = ref_proj[ids]
        local = local - np.mean(local, axis=0, keepdims=True)
        gram = local @ local.T
        vals = np.linalg.eigvalsh(gram.astype(np.float64, copy=False))
        vals = vals[vals > 1e-10]
        if vals.size == 0:
            continue
        p = vals / np.sum(vals)
        out[i] = float(-np.sum(p * np.log(p)))
    return out


def _layer_slices(total_dim: int, layers: np.ndarray) -> list[slice]:
    n_layers = int(len(layers))
    if n_layers <= 1 or total_dim % n_layers != 0:
        return [slice(0, total_dim)]
    d = total_dim // n_layers
    return [slice(i * d, (i + 1) * d) for i in range(n_layers)]


def _layer_neighbor_instability(
    x: np.ndarray,
    *,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    layer_slices: list[slice],
    cfg: GeometryAuditConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(layer_slices) <= 1:
        return np.full(int(np.sum(test_mask)), np.nan, dtype=np.float64)
    x_train = x[train_mask]
    x_test = x[test_mask]
    k = max(2, min(int(cfg.knn_k), x_train.shape[0]))
    neighbor_sets = []
    for sl in layer_slices:
        dim = int(sl.stop - sl.start)
        proj = _random_projection(dim, min(cfg.layer_projection_dim, dim), rng)
        _, nn = _topk_sqdist(x_test[:, sl] @ proj, x_train[:, sl] @ proj, k=k, chunk_size=cfg.chunk_size)
        neighbor_sets.append(nn)
    vals = []
    for a, b in zip(neighbor_sets[:-1], neighbor_sets[1:]):
        overlap = np.empty(a.shape[0], dtype=np.float64)
        for i in range(a.shape[0]):
            overlap[i] = len(set(a[i].tolist()).intersection(set(b[i].tolist()))) / max(k, 1)
        vals.append(1.0 - overlap)
    return np.mean(np.vstack(vals), axis=0)


def _compartment_score(**kwargs: np.ndarray | int) -> np.ndarray:
    k = int(kwargs.pop("k"))
    boundary = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(kwargs["boundary_proj"], dtype=np.float64), -30, 30)))
    lid = np.log1p(np.asarray(kwargs["lid"], dtype=np.float64)) / np.log1p(max(k, 2))
    spec = np.asarray(kwargs["local_spec_entropy"], dtype=np.float64) / np.log(max(k, 2))
    pieces = [
        boundary,
        np.asarray(kwargs["healthy_residual"], dtype=np.float64),
        lid,
        np.asarray(kwargs["knn_error_frac"], dtype=np.float64),
        np.asarray(kwargs["knn_label_entropy"], dtype=np.float64),
        spec,
        np.asarray(kwargs["layer_instability"], dtype=np.float64),
    ]
    arr = np.vstack([np.clip(p, 0.0, 1.0) for p in pieces])
    return np.nanmean(arr, axis=0)


def _append_step_scores(packed: dict[str, np.ndarray], geom_step: np.ndarray, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    old = np.asarray(packed["step_scores"], dtype=np.float32)
    if old.shape[:2] != geom_step.shape[:2]:
        raise ValueError(f"step_scores shape {old.shape} incompatible with geometry shape {geom_step.shape}")
    old_names = [str(x) for x in packed["step_score_names"].tolist()]
    keep = [i for i, n in enumerate(names) if n not in old_names]
    if not keep:
        return packed
    packed["step_scores"] = np.concatenate([old, geom_step[:, :, keep]], axis=2)
    packed["step_score_names"] = np.asarray(old_names + [names[i] for i in keep], dtype="<U96")
    return packed


def _append_chain_geometry_scores(
    packed: dict[str, np.ndarray],
    geom_step: np.ndarray,
    names: tuple[str, ...],
    n_steps: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = []
    for i in range(geom_step.shape[0]):
        series = {name: geom_step[i, : int(n_steps[i]), k] for k, name in enumerate(names)}
        rows.append(summarize_step_metrics(series))
    new_names = sorted({name for row in rows for name in row.keys()})
    old_chain = np.asarray(packed["chain_scores"], dtype=np.float32)
    old_names = [str(x) for x in packed["chain_score_names"].tolist()]
    add_names = [name for name in new_names if name not in old_names]
    if not add_names:
        return packed
    add = np.full((geom_step.shape[0], len(add_names)), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        for j, name in enumerate(add_names):
            add[i, j] = float(row.get(name, np.nan))
    packed["chain_scores"] = np.concatenate([old_chain, add], axis=1)
    packed["chain_score_names"] = np.asarray(old_names + add_names, dtype="<U96")
    return packed
