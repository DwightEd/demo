#!/usr/bin/env python3
"""Healthy-Connection Residual Holonomy audit for ProcessBench step vectors.

This script validates the HCR-Holo idea on ProcessBench-style npz files.
It consumes raw per-step hidden vectors, learns a healthy low-rank connection
from correct training chains only, then scores each step boundary by residual
holonomy and closure error relative to matched healthy dynamics.

Supported input schemas:
  1. ProcessBench spectral-field extraction:
       labels            (-1 correct, >=0 first wrong step)
       layers_used
       sv_vec_<mode>     object array of (T, L, d)
       optional steps_text, problems, ids
  2. Older full_*.npz feature files:
       gold_error_step
       stepvec           object array of (T, L, d)
       optional step_token_ranges, steps_text, problem_ids

The local HCR score is assigned to the boundary t -> t+1 and localized at
to_step = t + 1. Therefore first-error localization uses to_step == gold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


EPS = 1e-8


@dataclass
class ChainRecord:
    index: int
    chain_id: str
    problem_id: str
    gold_error_step: int
    vectors: np.ndarray  # (T, L, d)
    steps_text: list[str]
    token_lengths: np.ndarray  # (T,)

    @property
    def n_steps(self) -> int:
        return int(self.vectors.shape[0])


@dataclass
class PCAState:
    mean: np.ndarray
    components: np.ndarray  # (r, d)
    scale: np.ndarray       # (r,)
    k: int

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        flat = arr.reshape(-1, arr.shape[-1])
        if self.components.shape[0] == 0:
            out = np.zeros((flat.shape[0], self.k), dtype=np.float64)
        else:
            score = (flat - self.mean) @ self.components.T
            score = score / self.scale
            if score.shape[1] < self.k:
                pad = np.zeros((score.shape[0], self.k - score.shape[1]), dtype=np.float64)
                out = np.concatenate([score, pad], axis=1)
            else:
                out = score[:, :self.k]
        return out.reshape(arr.shape[:-1] + (self.k,))


@dataclass
class RidgeMap:
    a_t: np.ndarray  # prediction is x @ a_t
    sigma_max: float
    n: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.a_t


@dataclass
class FoldModel:
    layer_positions: list[int]
    pcas: dict[int, PCAState]
    maps: dict[tuple[str, int, int, int, int], RidgeMap]
    stats: dict[tuple[str, int, int, int, int], tuple[float, float]]
    length_edges: np.ndarray
    cfg: argparse.Namespace


def to_py(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_py(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if not math.isfinite(val) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def safe_auc(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def text_len(s: str) -> int:
    return max(1, len(re.findall(r"\S+", str(s))))


def infer_op_id(text: str) -> int:
    """Coarse, label-free operation class.

    0 text/reasoning, 1 arithmetic/equation, 2 conclusion, 3 comparison/check.
    The point is not semantic parsing; it is a nuisance-control bin.
    """
    s = str(text).lower()
    if re.search(r"\b(therefore|thus|hence|answer|so the answer|final)\b|####", s):
        return 2
    if re.search(r"\b(check|verify|compare|must be|should be)\b", s):
        return 3
    if re.search(r"\d", s) and re.search(r"[=+\-*/×÷]|\b(add|subtract|multiply|divide|half|third|percent)\b", s):
        return 1
    return 0


def get_obj_array_value(arr: np.ndarray, i: int) -> Any:
    val = arr[i]
    if isinstance(val, np.ndarray) and val.shape == ():
        return val.item()
    return val


def pick_vector_key(z: np.lib.npyio.NpzFile, mode: str) -> str:
    if "stepvec" in z.files:
        return "stepvec"
    preferred = f"sv_vec_{mode}"
    if preferred in z.files:
        return preferred
    for key in z.files:
        if key.startswith("sv_vec_"):
            return key
    raise KeyError(
        "No raw step-vector field found. Expected `stepvec` or `sv_vec_<mode>`. "
        "Re-extract with `--step_vectors --sv_modes step_exp --store_vectors`."
    )


def load_records(path: str | Path, mode: str, max_chains: int | None = None) -> tuple[list[ChainRecord], list[int], str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. For ProcessBench, first extract with "
            "`01_extract_spectral_field.py --step_vectors --sv_modes step_exp --store_vectors`."
        )
    z = np.load(p, allow_pickle=True)
    vector_key = pick_vector_key(z, mode)
    raw_vectors = z[vector_key]

    if "gold_error_step" in z.files:
        gold = np.asarray(z["gold_error_step"], dtype=int)
    elif "labels" in z.files:
        gold = np.asarray(z["labels"], dtype=int)
    else:
        raise KeyError("Need `gold_error_step` or ProcessBench `labels`.")

    if "layers_used" in z.files:
        layers = [int(x) for x in np.asarray(z["layers_used"]).tolist()]
    elif "sv_layers" in z.files:
        layers = [int(x) for x in np.asarray(z["sv_layers"]).tolist()]
    else:
        first = next(np.asarray(v) for v in raw_vectors if v is not None)
        layers = list(range(first.shape[1]))

    ids = z["ids"] if "ids" in z.files else None
    problem_ids = z["problem_ids"] if "problem_ids" in z.files else ids
    steps_text_arr = z["steps_text"] if "steps_text" in z.files else None
    step_ranges = z["step_token_ranges"] if "step_token_ranges" in z.files else None

    records: list[ChainRecord] = []
    for i in range(len(raw_vectors)):
        if max_chains is not None and len(records) >= max_chains:
            break
        vec = get_obj_array_value(raw_vectors, i)
        if vec is None:
            continue
        vec = np.asarray(vec)
        if vec.ndim != 3 or vec.shape[0] < 2 or vec.shape[1] < 2:
            continue
        if not np.all(np.isfinite(vec)):
            continue
        g = int(gold[i])
        if g >= vec.shape[0]:
            continue

        if ids is not None:
            cid = str(get_obj_array_value(ids, i))
        else:
            cid = str(i)
        if problem_ids is not None:
            pid = str(get_obj_array_value(problem_ids, i))
        else:
            pid = cid

        texts: list[str]
        if steps_text_arr is not None:
            st = get_obj_array_value(steps_text_arr, i)
            if st is None:
                texts = [f"step {j}" for j in range(vec.shape[0])]
            else:
                texts = [str(x) for x in list(st)]
                if len(texts) < vec.shape[0]:
                    texts += [f"step {j}" for j in range(len(texts), vec.shape[0])]
                texts = texts[: vec.shape[0]]
        else:
            texts = [f"step {j}" for j in range(vec.shape[0])]

        if step_ranges is not None:
            rr = get_obj_array_value(step_ranges, i)
            try:
                lens = np.array([max(1, int(b) - int(a)) for a, b in rr], dtype=float)
            except Exception:
                lens = np.array([text_len(x) for x in texts], dtype=float)
        else:
            lens = np.array([text_len(x) for x in texts], dtype=float)
        if lens.shape[0] < vec.shape[0]:
            lens = np.pad(lens, (0, vec.shape[0] - lens.shape[0]), constant_values=float(np.median(lens) if lens.size else 1))
        lens = lens[: vec.shape[0]]

        records.append(
            ChainRecord(
                index=i,
                chain_id=cid,
                problem_id=pid,
                gold_error_step=g,
                vectors=vec.astype(np.float32, copy=False),
                steps_text=texts,
                token_lengths=lens.astype(float),
            )
        )
    if not records:
        raise RuntimeError(f"No valid chains loaded from {path}. vector_key={vector_key}")
    return records, layers, vector_key


def choose_layer_positions(n_layers: int, stride: int, max_layers: int | None) -> list[int]:
    stride = max(1, int(stride))
    pos = list(range(0, n_layers, stride))
    if pos[-1] != n_layers - 1:
        pos.append(n_layers - 1)
    if max_layers is not None and len(pos) > max_layers:
        raw = np.linspace(0, n_layers - 1, max_layers)
        pos = sorted(set(int(round(x)) for x in raw))
        if len(pos) < 2:
            pos = [0, n_layers - 1]
    return pos


def fit_length_edges(records: list[ChainRecord], train_idx: np.ndarray, cfg: argparse.Namespace) -> np.ndarray:
    if cfg.no_length_bins or cfg.length_bins <= 1:
        return np.array([], dtype=float)
    vals: list[float] = []
    for i in train_idx:
        r = records[int(i)]
        if r.gold_error_step < 0:
            vals.extend(np.log1p(r.token_lengths).tolist())
    if len(vals) < cfg.length_bins * 5:
        return np.array([], dtype=float)
    qs = np.linspace(0, 1, cfg.length_bins + 1)[1:-1]
    edges = np.quantile(np.asarray(vals, dtype=float), qs)
    return np.unique(edges)


def step_bin(record: ChainRecord, step: int, length_edges: np.ndarray, cfg: argparse.Namespace) -> tuple[int, int, int]:
    denom = max(1, record.n_steps - 1)
    phase = min(cfg.phase_bins - 1, max(0, int(math.floor(cfg.phase_bins * step / denom))))
    if length_edges.size:
        length_bin = int(np.searchsorted(length_edges, math.log1p(float(record.token_lengths[step])), side="right"))
    else:
        length_bin = 0
    if cfg.no_op_bins:
        op = 0
    else:
        op = infer_op_id(record.steps_text[step])
    return phase, length_bin, op


def fit_pcas(
    records: list[ChainRecord],
    train_idx: np.ndarray,
    layer_positions: list[int],
    cfg: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[int, PCAState]:
    pcas: dict[int, PCAState] = {}
    correct_idx = [int(i) for i in train_idx if records[int(i)].gold_error_step < 0]
    if not correct_idx:
        raise RuntimeError("No correct chains in train fold; cannot learn healthy bundle.")

    for lp in layer_positions:
        chunks = []
        for i in correct_idx:
            arr = records[i].vectors[:, lp, :].astype(np.float64, copy=False)
            chunks.append(arr)
        X = np.concatenate(chunks, axis=0)
        if X.shape[0] > cfg.max_pca_samples:
            take = rng.choice(X.shape[0], size=cfg.max_pca_samples, replace=False)
            X = X[take]
        mu = X.mean(axis=0)
        Xc = X - mu
        try:
            _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            Xc = Xc + rng.normal(0.0, 1e-6, size=Xc.shape)
            _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        r = int(min(cfg.bundle_dim, vt.shape[0]))
        comps = vt[:r].astype(np.float64, copy=False)
        if X.shape[0] > 1 and r > 0:
            scale = (s[:r] / math.sqrt(max(1, X.shape[0] - 1))).astype(np.float64)
            scale = np.maximum(scale, cfg.pca_eps)
        else:
            scale = np.ones((r,), dtype=np.float64)
        pcas[lp] = PCAState(mean=mu.astype(np.float64), components=comps, scale=scale, k=cfg.bundle_dim)
    return pcas


def transform_record(record: ChainRecord, layer_positions: list[int], pcas: dict[int, PCAState]) -> np.ndarray:
    T = record.n_steps
    k = next(iter(pcas.values())).k
    Z = np.zeros((T, len(layer_positions), k), dtype=np.float64)
    for m, lp in enumerate(layer_positions):
        Z[:, m, :] = pcas[lp].transform(record.vectors[:, lp, :])
    return Z


def add_pair(store: dict[tuple[str, int, int, int, int], list[tuple[np.ndarray, np.ndarray]]],
             kind: str, layer_m: int, b: tuple[int, int, int], x: np.ndarray, y: np.ndarray) -> None:
    phase, length_bin, op = b
    keys = [
        (kind, layer_m, phase, length_bin, op),
        (kind, layer_m, phase, -1, -1),
        (kind, layer_m, -1, -1, -1),
    ]
    for key in keys:
        store[key].append((x, y))


def ridge_fit(pairs: list[tuple[np.ndarray, np.ndarray]], ridge: float) -> RidgeMap | None:
    if not pairs:
        return None
    X = np.stack([p[0] for p in pairs], axis=0).astype(np.float64)
    Y = np.stack([p[1] for p in pairs], axis=0).astype(np.float64)
    k = X.shape[1]
    xtx = X.T @ X
    xty = X.T @ Y
    try:
        a_t = np.linalg.solve(xtx + ridge * np.eye(k), xty)
    except np.linalg.LinAlgError:
        a_t = np.linalg.pinv(xtx + ridge * np.eye(k)) @ xty
    try:
        sigma = float(np.linalg.svd(a_t, compute_uv=False)[0])
    except np.linalg.LinAlgError:
        sigma = float("nan")
    return RidgeMap(a_t=a_t, sigma_max=sigma, n=X.shape[0])


def get_map(model: FoldModel, kind: str, layer_m: int, b: tuple[int, int, int]) -> RidgeMap | None:
    phase, length_bin, op = b
    keys = [
        (kind, layer_m, phase, length_bin, op),
        (kind, layer_m, phase, -1, -1),
        (kind, layer_m, -1, -1, -1),
    ]
    for key in keys:
        if key in model.maps:
            return model.maps[key]
    return None


def get_stat(model: FoldModel, name: str, layer_m: int, b: tuple[int, int, int]) -> tuple[float, float]:
    phase, length_bin, op = b
    keys = [
        (name, layer_m, phase, length_bin, op),
        (name, layer_m, phase, -1, -1),
        (name, layer_m, -1, -1, -1),
    ]
    for key in keys:
        if key in model.stats:
            return model.stats[key]
    return (0.0, 1.0)


def raw_loop_scores(model: FoldModel, record: ChainRecord, Z: np.ndarray, boundary_t: int, layer_m: int) -> tuple[float, float, float] | None:
    """Return raw comm, close, and log sigma_max for boundary t -> t+1."""
    b0 = step_bin(record, boundary_t, model.length_edges, model.cfg)
    b1 = step_bin(record, boundary_t + 1, model.length_edges, model.cfg)
    Ad0 = get_map(model, "d", layer_m, b0)
    As0 = get_map(model, "s", layer_m, b0)
    As1 = get_map(model, "s", layer_m + 1, b0)
    Ad1 = get_map(model, "d", layer_m, b1)
    if Ad0 is None or As0 is None or As1 is None or Ad1 is None:
        return None
    z = Z[boundary_t, layer_m, :]
    ds = As1.predict(Ad0.predict(z))
    sd = Ad1.predict(As0.predict(z))
    actual = Z[boundary_t + 1, layer_m + 1, :]
    comm = float(np.mean((ds - sd) ** 2))
    pred = 0.5 * (ds + sd)
    close = float(np.mean((actual - pred) ** 2))
    lyap = float(math.log(max(As0.sigma_max, EPS))) if math.isfinite(As0.sigma_max) else 0.0
    return comm, close, lyap


def fit_fold_model(
    records: list[ChainRecord],
    train_idx: np.ndarray,
    layer_positions: list[int],
    cfg: argparse.Namespace,
    rng: np.random.Generator,
) -> FoldModel:
    pcas = fit_pcas(records, train_idx, layer_positions, cfg, rng)
    length_edges = fit_length_edges(records, train_idx, cfg)
    temp_model = FoldModel(layer_positions=layer_positions, pcas=pcas, maps={}, stats={}, length_edges=length_edges, cfg=cfg)

    coords: dict[int, np.ndarray] = {}
    pair_store: dict[tuple[str, int, int, int, int], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

    for i in train_idx:
        r = records[int(i)]
        if r.gold_error_step >= 0:
            continue
        Z = transform_record(r, layer_positions, pcas)
        coords[int(i)] = Z
        for t in range(r.n_steps):
            b = step_bin(r, t, length_edges, cfg)
            for m in range(len(layer_positions) - 1):
                add_pair(pair_store, "d", m, b, Z[t, m, :], Z[t, m + 1, :])
        for t in range(r.n_steps - 1):
            b = step_bin(r, t, length_edges, cfg)
            for m in range(len(layer_positions)):
                add_pair(pair_store, "s", m, b, Z[t, m, :], Z[t + 1, m, :])

    maps: dict[tuple[str, int, int, int, int], RidgeMap] = {}
    for key, pairs in pair_store.items():
        is_global = key[2:] == (-1, -1, -1)
        is_phase = key[3:] == (-1, -1)
        min_pairs = max(5, cfg.min_pairs // 2) if (is_global or is_phase) else cfg.min_pairs
        if len(pairs) >= min_pairs:
            rm = ridge_fit(pairs, cfg.ridge)
            if rm is not None:
                maps[key] = rm

    model = FoldModel(layer_positions=layer_positions, pcas=pcas, maps=maps, stats={}, length_edges=length_edges, cfg=cfg)
    stat_store: dict[tuple[str, int, int, int, int], list[float]] = defaultdict(list)
    for i, Z in coords.items():
        r = records[i]
        for t in range(r.n_steps - 1):
            b = step_bin(r, t, length_edges, cfg)
            phase, length_bin, op = b
            for m in range(len(layer_positions) - 1):
                vals = raw_loop_scores(model, r, Z, t, m)
                if vals is None:
                    continue
                comm, close, lyap = vals
                for name, val in [("comm", comm), ("close", close), ("lyap", lyap)]:
                    for key in [
                        (name, m, phase, length_bin, op),
                        (name, m, phase, -1, -1),
                        (name, m, -1, -1, -1),
                    ]:
                        stat_store[key].append(float(val))

    stats: dict[tuple[str, int, int, int, int], tuple[float, float]] = {}
    for key, vals in stat_store.items():
        if len(vals) >= 5:
            arr = np.asarray(vals, dtype=float)
            mu = float(np.mean(arr))
            sd = float(np.std(arr))
            stats[key] = (mu, max(sd, cfg.stat_eps))
    model.stats = stats
    return model


def score_record(model: FoldModel, record: ChainRecord) -> list[dict[str, Any]]:
    Z = transform_record(record, model.layer_positions, model.pcas)
    rows: list[dict[str, Any]] = []
    for t in range(record.n_steps - 1):
        to_step = t + 1
        comm_zs = []
        close_zs = []
        lyap_zs = []
        raw_comms = []
        raw_closes = []
        raw_lyaps = []
        for m in range(len(model.layer_positions) - 1):
            vals = raw_loop_scores(model, record, Z, t, m)
            if vals is None:
                continue
            comm, close, lyap = vals
            b = step_bin(record, t, model.length_edges, model.cfg)
            mu_c, sd_c = get_stat(model, "comm", m, b)
            mu_l, sd_l = get_stat(model, "close", m, b)
            mu_y, sd_y = get_stat(model, "lyap", m, b)
            comm_zs.append((comm - mu_c) / sd_c)
            close_zs.append((close - mu_l) / sd_l)
            lyap_zs.append((lyap - mu_y) / sd_y)
            raw_comms.append(comm)
            raw_closes.append(close)
            raw_lyaps.append(lyap)
        if not comm_zs:
            continue
        comm_z = np.asarray(comm_zs, dtype=float)
        close_z = np.asarray(close_zs, dtype=float)
        lyap_z = np.asarray(lyap_zs, dtype=float)
        hcr_layers = np.maximum(comm_z, 0.0) + np.maximum(close_z, 0.0)
        hcr_joint_max = float(np.max(hcr_layers))
        hcr_joint_mean = float(np.mean(hcr_layers))
        lyap_max = float(np.max(lyap_z))
        lyap_min = float(np.min(lyap_z))
        basin_capture = float(hcr_joint_max * max(0.0, -lyap_min))
        phase, length_bin, op = step_bin(record, to_step, model.length_edges, model.cfg)
        pos = float(to_step / max(1, record.n_steps - 1))
        rows.append(
            {
                "chain_index": record.index,
                "chain_id": record.chain_id,
                "problem_id": record.problem_id,
                "gold_error_step": record.gold_error_step,
                "from_step": t,
                "to_step": to_step,
                "n_steps": record.n_steps,
                "phase_bin": phase,
                "length_bin": length_bin,
                "op_id": op,
                "step_len": float(record.token_lengths[to_step]),
                "pos": pos,
                "is_error_chain": int(record.gold_error_step >= 0),
                "first_error_label": int(record.gold_error_step == to_step),
                "prefix_before_error_label": int(record.gold_error_step >= 0 and to_step < record.gold_error_step),
                "is_prefix_or_first": int(record.gold_error_step < 0 or to_step <= record.gold_error_step),
                "hcr_comm_max": float(np.max(comm_z)),
                "hcr_close_max": float(np.max(close_z)),
                "hcr_joint_max": hcr_joint_max,
                "hcr_joint_mean": hcr_joint_mean,
                "lyap_max": lyap_max,
                "lyap_min": lyap_min,
                "basin_capture": basin_capture,
                "raw_comm_mean": float(np.mean(raw_comms)),
                "raw_close_mean": float(np.mean(raw_closes)),
                "raw_lyap_mean": float(np.mean(raw_lyaps)),
            }
        )
    return rows


def row_features(row: dict[str, Any], feature_set: str) -> list[float]:
    controls = [
        float(row["pos"]),
        math.log1p(float(row["n_steps"])),
        math.log1p(float(row["step_len"])),
        float(row["op_id"]),
    ]
    hcr = [
        float(row["hcr_comm_max"]),
        float(row["hcr_close_max"]),
        float(row["hcr_joint_max"]),
        float(row["hcr_joint_mean"]),
        float(row["lyap_max"]),
        float(row["lyap_min"]),
        float(row["basin_capture"]),
    ]
    raw = [
        float(row["raw_comm_mean"]),
        float(row["raw_close_mean"]),
        float(row["raw_lyap_mean"]),
    ]
    if feature_set == "controls":
        return controls
    if feature_set == "hcr":
        return hcr
    if feature_set == "raw":
        return raw
    if feature_set == "controls+hcr":
        return controls + hcr
    if feature_set == "controls+raw":
        return controls + raw
    raise KeyError(feature_set)


def fit_predict_logreg(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], label_key: str, feature_set: str) -> list[float]:
    y_train = np.asarray([int(r[label_key]) for r in train_rows], dtype=int)
    if y_train.size == 0 or np.unique(y_train).size < 2 or not test_rows:
        return [float("nan")] * len(test_rows)
    X_train = np.asarray([row_features(r, feature_set) for r in train_rows], dtype=float)
    X_test = np.asarray([row_features(r, feature_set) for r in test_rows], dtype=float)
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0)
    sd = np.where(sd < EPS, 1.0, sd)
    X_train = np.nan_to_num((X_train - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num((X_test - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=0)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1].astype(float).tolist()


def make_folds(records: list[ChainRecord], n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(records)
    groups = np.array([r.problem_id for r in records], dtype=object)
    unique_groups = np.unique(groups)
    k = max(2, min(n_folds, len(unique_groups), n))
    if k < 2:
        idx = np.arange(n)
        return [(idx, idx)]
    gkf = GroupKFold(n_splits=k)
    dummy_y = np.array([int(r.gold_error_step >= 0) for r in records], dtype=int)
    return [(tr.astype(int), te.astype(int)) for tr, te in gkf.split(np.zeros(n), dummy_y, groups)]


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    first = [r for r in rows if int(r["is_prefix_or_first"]) == 1]
    out["first_error_rows"] = len(first)
    out["first_error_pos"] = int(sum(int(r["first_error_label"]) for r in first))
    out["first_error_auc"] = {
        "hcr_joint_max": safe_auc([r["first_error_label"] for r in first], [r["hcr_joint_max"] for r in first]),
        "hcr_comm_max": safe_auc([r["first_error_label"] for r in first], [r["hcr_comm_max"] for r in first]),
        "hcr_close_max": safe_auc([r["first_error_label"] for r in first], [r["hcr_close_max"] for r in first]),
        "basin_capture": safe_auc([r["first_error_label"] for r in first], [r["basin_capture"] for r in first]),
        "raw_comm_mean": safe_auc([r["first_error_label"] for r in first], [r["raw_comm_mean"] for r in first]),
        "raw_close_mean": safe_auc([r["first_error_label"] for r in first], [r["raw_close_mean"] for r in first]),
        "pos": safe_auc([r["first_error_label"] for r in first], [r["pos"] for r in first]),
        "step_len": safe_auc([r["first_error_label"] for r in first], [r["step_len"] for r in first]),
    }

    future = [r for r in rows if int(r["gold_error_step"]) < 0 or int(r["to_step"]) < int(r["gold_error_step"])]
    out["pre_error_future_rows"] = len(future)
    out["pre_error_future_pos"] = int(sum(int(r["is_error_chain"]) for r in future))
    out["pre_error_future_auc"] = {
        "hcr_joint_max": safe_auc([r["is_error_chain"] for r in future], [r["hcr_joint_max"] for r in future]),
        "hcr_joint_mean": safe_auc([r["is_error_chain"] for r in future], [r["hcr_joint_mean"] for r in future]),
        "basin_capture": safe_auc([r["is_error_chain"] for r in future], [r["basin_capture"] for r in future]),
        "pos": safe_auc([r["is_error_chain"] for r in future], [r["pos"] for r in future]),
        "step_len": safe_auc([r["is_error_chain"] for r in future], [r["step_len"] for r in future]),
    }
    return out


def rank_first_errors(records: list[ChainRecord], rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    by_chain: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_chain[int(r["chain_index"])].append(r)
    ranks = []
    percentiles = []
    skipped_gold0 = 0
    for rec in records:
        g = rec.gold_error_step
        if g < 0:
            continue
        if g == 0:
            skipped_gold0 += 1
            continue
        cand = [r for r in by_chain.get(rec.index, []) if int(r["to_step"]) <= g]
        gold_rows = [r for r in cand if int(r["to_step"]) == g]
        if not cand or not gold_rows:
            continue
        sorted_rows = sorted(cand, key=lambda x: float(x[score_key]), reverse=True)
        gold_to_step = int(gold_rows[0]["to_step"])
        rank = next(j + 1 for j, rr in enumerate(sorted_rows) if int(rr["to_step"]) == gold_to_step)
        ranks.append(rank)
        if len(sorted_rows) == 1:
            percentiles.append(1.0)
        else:
            percentiles.append(1.0 - (rank - 1) / (len(sorted_rows) - 1))
    if not ranks:
        return {"n": 0, "skipped_gold0": skipped_gold0, "top1": float("nan"), "mean_rank": float("nan"), "mean_percentile": float("nan")}
    arr = np.asarray(ranks, dtype=float)
    return {
        "n": int(arr.size),
        "skipped_gold0": int(skipped_gold0),
        "top1": float(np.mean(arr == 1)),
        "mean_rank": float(np.mean(arr)),
        "mean_percentile": float(np.mean(percentiles)),
    }


def response_metrics(records: list[ChainRecord], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_chain: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_chain[int(r["chain_index"])].append(r)
    y = []
    agg: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        rr = by_chain.get(rec.index, [])
        if not rr:
            continue
        y.append(int(rec.gold_error_step >= 0))
        for key in ["hcr_joint_max", "hcr_joint_mean", "basin_capture", "raw_comm_mean", "raw_close_mean"]:
            vals = np.asarray([float(x[key]) for x in rr], dtype=float)
            agg[f"{key}/max"].append(float(np.max(vals)))
            agg[f"{key}/mean"].append(float(np.mean(vals)))
            k = max(1, int(math.ceil(0.2 * vals.size)))
            agg[f"{key}/top20_mean"].append(float(np.mean(np.sort(vals)[-k:])))
    return {key: safe_auc(y, vals) for key, vals in sorted(agg.items())}


def run_cv(records: list[ChainRecord], layers: list[int], cfg: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    n_layers = records[0].vectors.shape[1]
    layer_positions = choose_layer_positions(n_layers, cfg.layer_stride, cfg.max_layers)
    folds = make_folds(records, cfg.n_folds)

    all_test_rows: list[dict[str, Any]] = []
    oof_preds: dict[str, dict[str, list[float]]] = {
        "first_error": defaultdict(list),
        "pre_error_future": defaultdict(list),
    }
    oof_labels: dict[str, list[int]] = {"first_error": [], "pre_error_future": []}

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        model = fit_fold_model(records, train_idx, layer_positions, cfg, rng)
        train_rows_all: list[dict[str, Any]] = []
        test_rows_all: list[dict[str, Any]] = []
        for i in train_idx:
            train_rows_all.extend(score_record(model, records[int(i)]))
        for i in test_idx:
            scored = score_record(model, records[int(i)])
            for r in scored:
                r["fold"] = fold_id
            test_rows_all.extend(scored)
        all_test_rows.extend(test_rows_all)

        train_first = [r for r in train_rows_all if int(r["is_prefix_or_first"]) == 1]
        test_first = [r for r in test_rows_all if int(r["is_prefix_or_first"]) == 1]
        train_future = [r for r in train_rows_all if int(r["gold_error_step"]) < 0 or int(r["to_step"]) < int(r["gold_error_step"])]
        test_future = [r for r in test_rows_all if int(r["gold_error_step"]) < 0 or int(r["to_step"]) < int(r["gold_error_step"])]

        for task, tr, te, label_key in [
            ("first_error", train_first, test_first, "first_error_label"),
            ("pre_error_future", train_future, test_future, "is_error_chain"),
        ]:
            oof_labels[task].extend([int(r[label_key]) for r in te])
            for fs in ["controls", "raw", "hcr", "controls+raw", "controls+hcr"]:
                oof_preds[task][fs].extend(fit_predict_logreg(tr, te, label_key, fs))

        print(
            f"fold {fold_id}: train={len(train_idx)} test={len(test_idx)} "
            f"maps={len(model.maps)} stats={len(model.stats)} "
            f"test_rows={len(test_rows_all)}"
        )

    eval_summary = evaluate_rows(all_test_rows)
    eval_summary["first_error_logreg_auc"] = {
        fs: safe_auc(oof_labels["first_error"], vals)
        for fs, vals in sorted(oof_preds["first_error"].items())
    }
    eval_summary["pre_error_future_logreg_auc"] = {
        fs: safe_auc(oof_labels["pre_error_future"], vals)
        for fs, vals in sorted(oof_preds["pre_error_future"].items())
    }
    eval_summary["first_error_rank"] = {
        "hcr_joint_max": rank_first_errors(records, all_test_rows, "hcr_joint_max"),
        "hcr_close_max": rank_first_errors(records, all_test_rows, "hcr_close_max"),
        "basin_capture": rank_first_errors(records, all_test_rows, "basin_capture"),
    }
    eval_summary["response_auc"] = response_metrics(records, all_test_rows)
    eval_summary["rows"] = all_test_rows
    eval_summary["layer_positions"] = layer_positions
    eval_summary["layer_ids"] = [layers[p] if p < len(layers) else p for p in layer_positions]
    return eval_summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n===== HCR-Holo summary =====")
    print("layer positions:", summary["layer_positions"])
    print("layer ids:", summary["layer_ids"])
    print("\nTask first_error:")
    print(f"  rows {summary['first_error_rows']} pos {summary['first_error_pos']}")
    for k, v in summary["first_error_auc"].items():
        print(f"  single {k:18s} AUROC {v:.3f}" if math.isfinite(v) else f"  single {k:18s} AUROC nan")
    for k, v in summary["first_error_logreg_auc"].items():
        print(f"  model  {k:18s} AUROC {v:.3f}" if math.isfinite(v) else f"  model  {k:18s} AUROC nan")
    print("  ranks:")
    for k, v in summary["first_error_rank"].items():
        print(
            f"    {k:16s} n={v['n']} skipped_gold0={v['skipped_gold0']} "
            f"top1={v['top1']:.3f} mean_rank={v['mean_rank']:.2f} "
            f"mean_pct={v['mean_percentile']:.3f}"
        )

    print("\nTask pre_error_future:")
    print(f"  rows {summary['pre_error_future_rows']} pos {summary['pre_error_future_pos']}")
    for k, v in summary["pre_error_future_auc"].items():
        print(f"  single {k:18s} AUROC {v:.3f}" if math.isfinite(v) else f"  single {k:18s} AUROC nan")
    for k, v in summary["pre_error_future_logreg_auc"].items():
        print(f"  model  {k:18s} AUROC {v:.3f}" if math.isfinite(v) else f"  model  {k:18s} AUROC nan")

    print("\nResponse AUC:")
    top = sorted(summary["response_auc"].items(), key=lambda kv: (-1 if not math.isfinite(kv[1]) else -kv[1], kv[0]))[:12]
    for k, v in top:
        print(f"  {k:28s} {v:.3f}" if math.isfinite(v) else f"  {k:28s} nan")


def write_outputs(summary: dict[str, Any], output_dir: str | Path, tag: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = summary.pop("rows")
    csv_path = out / f"{tag}_hcr_holo_step_scores.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    json_path = out / f"{tag}_hcr_holo_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_py(summary), f, ensure_ascii=False, indent=2)
    summary["rows"] = rows
    print(f"\nSaved step scores -> {csv_path}")
    print(f"Saved summary     -> {json_path}")


def make_synthetic_records(seed: int = 0) -> tuple[list[ChainRecord], list[int]]:
    rng = np.random.default_rng(seed)
    n_correct = 70
    n_error = 80
    k = 8
    d = 32
    L = 7
    records: list[ChainRecord] = []
    layer_bases = [rng.normal(size=(k, d)) / math.sqrt(k) for _ in range(L)]
    step_a = np.eye(k) + 0.04 * rng.normal(size=(k, k))
    depth_a = np.eye(k) + 0.03 * rng.normal(size=(k, k))
    wrong_vec = rng.normal(size=k)
    wrong_vec = wrong_vec / np.linalg.norm(wrong_vec)

    def build_chain(idx: int, error: bool) -> ChainRecord:
        T = int(rng.integers(5, 10))
        g = int(rng.integers(2, T)) if error else -1
        y = rng.normal(size=k)
        lat_steps = []
        offset = np.zeros(k)
        for t in range(T):
            if error and t >= g:
                offset = 2.5 * wrong_vec
            lat_steps.append(y + offset + 0.03 * rng.normal(size=k))
            y = step_a @ y
        arr = np.zeros((T, L, d), dtype=np.float32)
        for t in range(T):
            z = lat_steps[t]
            for l in range(L):
                z_l = np.linalg.matrix_power(depth_a, l) @ z
                arr[t, l, :] = z_l @ layer_bases[l] + 0.03 * rng.normal(size=d)
        texts = [f"step {t}: compute {t}+{idx}" for t in range(T)]
        return ChainRecord(
            index=idx,
            chain_id=f"syn-{idx}",
            problem_id=f"syn-{idx}",
            gold_error_step=g,
            vectors=arr,
            steps_text=texts,
            token_lengths=np.array([text_len(x) for x in texts], dtype=float),
        )

    for i in range(n_correct):
        records.append(build_chain(i, False))
    for j in range(n_error):
        records.append(build_chain(n_correct + j, True))
    return records, list(range(L))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", nargs="?", help="ProcessBench npz with raw step vectors.")
    ap.add_argument("--mode", default="step_exp", help="Use sv_vec_<mode> when stepvec is absent.")
    ap.add_argument("--output_dir", default="outputs/hcr_holo")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max_chains", type=int, default=None)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--bundle_dim", type=int, default=16)
    ap.add_argument("--max_pca_samples", type=int, default=6000)
    ap.add_argument("--pca_eps", type=float, default=1e-5)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--min_pairs", type=int, default=24)
    ap.add_argument("--stat_eps", type=float, default=1e-4)
    ap.add_argument("--phase_bins", type=int, default=4)
    ap.add_argument("--length_bins", type=int, default=3)
    ap.add_argument("--no_length_bins", action="store_true")
    ap.add_argument("--no_op_bins", action="store_true")
    ap.add_argument("--layer_stride", type=int, default=4)
    ap.add_argument("--max_layers", type=int, default=10)
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args()


def main() -> None:
    cfg = parse_args()
    if cfg.selftest:
        records, layers = make_synthetic_records(cfg.seed)
        cfg.n_folds = min(cfg.n_folds, 3)
        cfg.layer_stride = 1
        cfg.max_layers = None
        cfg.bundle_dim = min(cfg.bundle_dim, 8)
        cfg.min_pairs = min(cfg.min_pairs, 10)
        summary = run_cv(records, layers, cfg)
        print_summary(summary)
        auc = summary["first_error_auc"]["hcr_joint_max"]
        if not math.isfinite(auc) or auc < 0.75:
            raise SystemExit(f"SELFTEST FAILED: hcr_joint_max first-error AUC={auc}")
        print("\nSELFTEST PASSED")
        return

    if not cfg.npz:
        raise SystemExit("Provide an npz path or use --selftest.")
    records, layers, vector_key = load_records(cfg.npz, cfg.mode, cfg.max_chains)
    err = sum(r.gold_error_step >= 0 for r in records)
    corr = len(records) - err
    print(
        f"Loaded {len(records)} chains from {cfg.npz} | vector_key={vector_key} "
        f"| correct={corr} error={err} | layers={len(layers)}"
    )
    summary = run_cv(records, layers, cfg)
    summary["input_npz"] = str(cfg.npz)
    summary["vector_key"] = vector_key
    summary["n_chains"] = len(records)
    summary["n_correct"] = corr
    summary["n_error"] = err
    print_summary(summary)
    tag = cfg.tag or Path(cfg.npz).stem
    write_outputs(summary, cfg.output_dir, tag)


if __name__ == "__main__":
    main()
