#!/usr/bin/env python3
"""Learned Latent Separatrix audit for ProcessBench step vectors.

This script implements the LSRM proposal in
`IDEA_REBOOT_LEARNED_LATENT_SEPARABILITY.md`.

The core test is deliberately different from previous hand-built geometry
audits.  It consumes only raw per-step hidden vectors, learns an out-of-fold
latent chart, and asks whether first-error and response-level failures become
separable in that learned chart after nuisance variables are monitored.

Supported input schemas:
  1. Canonical full-trace feature files:
       stepvec           object array of (T, L, d)
       gold_error_step   -1 for fully correct, >=0 for first wrong step
       optional ids, problem_ids, steps_text, step_token_ranges, sv_layers
  2. 01_extract_spectral_field.py files:
       sv_vec_<mode>     object array of (T, L, d)
       labels            -1 for fully correct, >=0 first wrong step
       optional ids, problems/problem_ids, steps_text, layers_used/sv_layers

What is learned:
  h_t -> PCA/whitened x_t -> encoder f_theta(x_t) = z_t
  z_t -> per-step hazard lambda_t
  z_t -> energy E_t for basin/separatrix diagnostics
  z_t -> nuisance heads with gradient reversal

Main losses:
  * discrete survival loss for first-error timing
  * state energy loss for correct vs post/at-error states
  * supervised contrastive loss in the learned latent space
  * adversarial nuisance prediction loss for length/position/op/n_steps bins

The implementation is fold-specific: PCA, neural weights, and calibration
models are trained only on training folds grouped by problem id.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    class _MissingAutograd:
        Function = object

    class _NoGrad:
        def __call__(self, fn: Any) -> Any:
            return fn

        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: Any) -> bool:
            return False

    class _MissingTorch:
        autograd = _MissingAutograd()

        @staticmethod
        def is_tensor(_x: Any) -> bool:
            return False

        @staticmethod
        def no_grad() -> _NoGrad:
            return _NoGrad()

    torch = _MissingTorch()  # type: ignore[assignment]
    F = None
    DataLoader = None

    class Dataset:  # type: ignore[no-redef]
        pass

    class _MissingNN:
        Module = object

    nn = _MissingNN()  # type: ignore[assignment]
    HAVE_TORCH = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False


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
class PCAProjector:
    mean: np.ndarray
    components: np.ndarray  # (k, d)
    scale: np.ndarray       # (k,)

    @property
    def dim(self) -> int:
        return int(self.components.shape[0])

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        flat = arr.reshape(-1, arr.shape[-1])
        z = (flat - self.mean.astype(np.float32)) @ self.components.T.astype(np.float32)
        z = z / self.scale.astype(np.float32)
        return z.reshape(arr.shape[:-1] + (self.dim,)).astype(np.float32, copy=False)


def finite_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return finite_json(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def safe_auc(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    if HAVE_SKLEARN:
        return float(roc_auc_score(y, s))
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s, dtype=float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    p = int(np.sum(y == 1))
    n = int(np.sum(y == 0))
    return float((np.sum(ranks[y == 1]) - p * (p + 1) / 2.0) / (p * n))


def safe_auprc(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size == 0 or np.unique(y).size < 2 or not HAVE_SKLEARN:
        return float("nan")
    return float(average_precision_score(y, s))


def text_len(s: str) -> int:
    return max(1, len(re.findall(r"\S+", str(s))))


def infer_op_id(text: str) -> int:
    """Coarse nuisance operation bin, not a semantic parser."""
    s = str(text).lower()
    if re.search(r"\b(therefore|thus|hence|answer|final|so the answer)\b|####", s):
        return 2
    if re.search(r"\b(check|verify|compare|must be|should be)\b", s):
        return 3
    if re.search(r"\d", s) and re.search(r"[=+\-*/x]|\b(add|subtract|multiply|divide|half|third|percent)\b", s):
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
    raise KeyError("Need `stepvec` or `sv_vec_<mode>` raw step vectors.")


def infer_vector_layer_count(raw_vectors: np.ndarray) -> int:
    for v in raw_vectors:
        arr = np.asarray(v)
        if arr.ndim == 3:
            return int(arr.shape[1])
    raise ValueError("Could not infer layer count from vector array.")


def layer_metadata_for_vector_key(z: np.lib.npyio.NpzFile, vector_key: str, raw_vectors: np.ndarray) -> list[int]:
    n_vec_layers = infer_vector_layer_count(raw_vectors)

    candidates: list[tuple[str, list[int]]] = []
    if vector_key == "stepvec" and "sv_layers" in z.files:
        candidates.append(("sv_layers", [int(x) for x in np.asarray(z["sv_layers"]).tolist()]))
    if "layers_used" in z.files:
        candidates.append(("layers_used", [int(x) for x in np.asarray(z["layers_used"]).tolist()]))
    if "sv_layers" in z.files and vector_key != "stepvec":
        candidates.append(("sv_layers", [int(x) for x in np.asarray(z["sv_layers"]).tolist()]))

    for _name, vals in candidates:
        if len(vals) == n_vec_layers:
            return vals

    return list(range(n_vec_layers))


def load_records(path: str | Path, mode: str, max_chains: int | None = None) -> tuple[list[ChainRecord], list[int], str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. Existing ProcessBench files are usually "
            "`data/features/full_gsm8k.npz`, `data/features/full_math.npz`, or "
            "`data/features/full_omnimath.npz` on the GPU box."
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

    layers = layer_metadata_for_vector_key(z, vector_key, raw_vectors)

    ids = z["ids"] if "ids" in z.files else None
    problem_ids = None
    for key in ("problem_ids", "problems", "problem"):
        if key in z.files:
            problem_ids = z[key]
            break
    if problem_ids is None:
        problem_ids = ids
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
        if vec.ndim != 3 or vec.shape[0] < 2 or vec.shape[1] < 1:
            continue
        if not np.all(np.isfinite(vec)):
            continue
        g = int(gold[i])
        if g >= vec.shape[0]:
            continue

        cid = str(get_obj_array_value(ids, i)) if ids is not None else str(i)
        pid = str(get_obj_array_value(problem_ids, i)) if problem_ids is not None else cid

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
        if lens.size < vec.shape[0]:
            fill = float(np.median(lens)) if lens.size else 1.0
            lens = np.pad(lens, (0, vec.shape[0] - lens.size), constant_values=fill)
        lens = lens[: vec.shape[0]]

        records.append(
            ChainRecord(
                index=i,
                chain_id=cid,
                problem_id=pid,
                gold_error_step=g,
                vectors=vec.astype(np.float32, copy=False),
                steps_text=texts,
                token_lengths=lens.astype(np.float32, copy=False),
            )
        )
    if not records:
        raise RuntimeError(f"No valid chains loaded from {path}. vector_key={vector_key}")
    return records, layers, vector_key


def choose_layer_positions(n_layers: int, stride: int, max_layers: int | None) -> list[int]:
    stride = max(1, int(stride))
    pos = list(range(0, n_layers, stride))
    if not pos or pos[-1] != n_layers - 1:
        pos.append(n_layers - 1)
    if max_layers is not None and len(pos) > max_layers:
        raw = np.linspace(0, n_layers - 1, max_layers)
        pos = sorted(set(int(round(x)) for x in raw))
    return pos


def record_step_features(record: ChainRecord, layer_positions: Sequence[int], layer_pool: str) -> np.ndarray:
    x = record.vectors[:, list(layer_positions), :]
    if layer_pool == "mean":
        return np.mean(x, axis=1).astype(np.float32, copy=False)
    if layer_pool == "last":
        return x[:, -1, :].astype(np.float32, copy=False)
    if layer_pool == "first_last":
        return np.concatenate([x[:, 0, :], x[:, -1, :]], axis=1).astype(np.float32, copy=False)
    if layer_pool == "concat":
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2]).astype(np.float32, copy=False)
    raise ValueError(f"unknown layer_pool={layer_pool}")


def fit_pca_projector(
    records: Sequence[ChainRecord],
    train_idx: np.ndarray,
    layer_positions: Sequence[int],
    layer_pool: str,
    pca_dim: int,
    max_samples: int,
    seed: int,
) -> PCAProjector:
    rng = np.random.default_rng(seed)
    chunks = []
    for idx in train_idx:
        chunks.append(record_step_features(records[int(idx)], layer_positions, layer_pool))
    X = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if X.shape[0] > max_samples:
        take = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[take]
    mu = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    Xc = X - mu
    k = int(min(pca_dim, Xc.shape[0], Xc.shape[1]))
    if k <= 0:
        raise RuntimeError("PCA dimension collapsed to zero.")
    try:
        _, s, vt = np.linalg.svd(Xc.astype(np.float64, copy=False), full_matrices=False)
    except np.linalg.LinAlgError:
        jitter = rng.normal(0.0, 1e-6, size=Xc.shape)
        _, s, vt = np.linalg.svd((Xc + jitter).astype(np.float64), full_matrices=False)
    comps = vt[:k].astype(np.float32)
    scale = (s[:k] / math.sqrt(max(1, X.shape[0] - 1))).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-5))
    return PCAProjector(mean=mu, components=comps, scale=scale)


def transform_records(
    records: Sequence[ChainRecord],
    indices: Sequence[int],
    projector: PCAProjector,
    layer_positions: Sequence[int],
    layer_pool: str,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for idx in indices:
        rec = records[int(idx)]
        x = record_step_features(rec, layer_positions, layer_pool)
        out[int(idx)] = projector.transform(x)
    return out


def make_folds(records: Sequence[ChainRecord], n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(records)
    groups = np.array([r.problem_id for r in records], dtype=object)
    y = np.array([int(r.gold_error_step >= 0) for r in records], dtype=int)
    unique_groups = np.unique(groups)
    k = max(2, min(int(n_folds), len(unique_groups), n))
    if k < 2:
        idx = np.arange(n)
        return [(idx, idx)]
    if HAVE_SKLEARN:
        gkf = GroupKFold(n_splits=k)
        return [(tr.astype(int), te.astype(int)) for tr, te in gkf.split(np.zeros(n), y, groups)]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(tr.astype(int), te.astype(int)) for tr, te in skf.split(np.zeros(n), y)]


def bin_edges(values: Sequence[float], n_bins: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < max(5, n_bins):
        return np.array([], dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.quantile(x, qs))


def digitize_with_edges(value: float, edges: np.ndarray) -> int:
    if edges.size == 0 or not math.isfinite(float(value)):
        return 0
    return int(np.searchsorted(edges, float(value), side="right"))


def fit_nuisance_edges(records: Sequence[ChainRecord], train_idx: Sequence[int], args: argparse.Namespace) -> dict[str, np.ndarray]:
    step_lens = []
    n_steps = []
    for idx in train_idx:
        rec = records[int(idx)]
        step_lens.extend(np.log1p(rec.token_lengths).astype(float).tolist())
        n_steps.append(math.log1p(rec.n_steps))
    return {
        "length": bin_edges(step_lens, args.length_bins),
        "n_steps": bin_edges(n_steps, args.n_steps_bins),
    }


class ChainTensorDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ChainRecord],
        indices: Sequence[int],
        features: dict[int, np.ndarray],
        nuisance_edges: dict[str, np.ndarray],
        pos_bins: int,
    ) -> None:
        self.records = records
        self.indices = [int(i) for i in indices]
        self.features = features
        self.nuisance_edges = nuisance_edges
        self.pos_bins = int(pos_bins)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = self.indices[item]
        rec = self.records[idx]
        x = self.features[idx]
        T = rec.n_steps
        gold = int(rec.gold_error_step)

        state_y = np.zeros(T, dtype=np.float32)
        first_y = np.zeros(T, dtype=np.float32)
        pre_mask = np.ones(T, dtype=np.float32)
        if gold >= 0:
            state_y[gold:] = 1.0
            first_y[gold] = 1.0
            pre_mask[gold + 1 :] = 0.0

        pos = np.arange(T, dtype=np.float32) / max(1, T - 1)
        pos_bin = np.minimum(self.pos_bins - 1, np.floor(pos * self.pos_bins).astype(np.int64))
        len_bin = np.array(
            [digitize_with_edges(float(math.log1p(v)), self.nuisance_edges.get("length", np.array([]))) for v in rec.token_lengths],
            dtype=np.int64,
        )
        op_bin = np.array([infer_op_id(t) for t in rec.steps_text], dtype=np.int64)
        n_steps_bin = np.full(
            T,
            digitize_with_edges(float(math.log1p(T)), self.nuisance_edges.get("n_steps", np.array([]))),
            dtype=np.int64,
        )
        return {
            "chain_index": idx,
            "x": x.astype(np.float32, copy=False),
            "gold": gold,
            "state_y": state_y,
            "first_y": first_y,
            "pre_mask": pre_mask,
            "pos": pos,
            "pos_bin": pos_bin,
            "len_bin": len_bin,
            "op_bin": op_bin,
            "n_steps_bin": n_steps_bin,
            "step_len": rec.token_lengths.astype(np.float32, copy=False),
        }


def collate_chains(batch: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(batch)
    Tm = max(int(b["x"].shape[0]) for b in batch)
    D = int(batch[0]["x"].shape[1])
    x = np.zeros((B, Tm, D), dtype=np.float32)
    mask = np.zeros((B, Tm), dtype=np.float32)
    state_y = np.zeros((B, Tm), dtype=np.float32)
    first_y = np.zeros((B, Tm), dtype=np.float32)
    pre_mask = np.zeros((B, Tm), dtype=np.float32)
    pos = np.zeros((B, Tm), dtype=np.float32)
    step_len = np.ones((B, Tm), dtype=np.float32)
    pos_bin = np.zeros((B, Tm), dtype=np.int64)
    len_bin = np.zeros((B, Tm), dtype=np.int64)
    op_bin = np.zeros((B, Tm), dtype=np.int64)
    n_steps_bin = np.zeros((B, Tm), dtype=np.int64)
    gold = np.zeros((B,), dtype=np.int64)
    chain_index = np.zeros((B,), dtype=np.int64)
    for i, b in enumerate(batch):
        T = int(b["x"].shape[0])
        x[i, :T] = b["x"]
        mask[i, :T] = 1.0
        state_y[i, :T] = b["state_y"]
        first_y[i, :T] = b["first_y"]
        pre_mask[i, :T] = b["pre_mask"]
        pos[i, :T] = b["pos"]
        step_len[i, :T] = b["step_len"]
        pos_bin[i, :T] = b["pos_bin"]
        len_bin[i, :T] = b["len_bin"]
        op_bin[i, :T] = b["op_bin"]
        n_steps_bin[i, :T] = b["n_steps_bin"]
        gold[i] = int(b["gold"])
        chain_index[i] = int(b["chain_index"])
    return {
        "chain_index": torch.from_numpy(chain_index),
        "x": torch.from_numpy(x),
        "mask": torch.from_numpy(mask),
        "gold": torch.from_numpy(gold),
        "state_y": torch.from_numpy(state_y),
        "first_y": torch.from_numpy(first_y),
        "pre_mask": torch.from_numpy(pre_mask),
        "pos": torch.from_numpy(pos),
        "step_len": torch.from_numpy(step_len),
        "pos_bin": torch.from_numpy(pos_bin),
        "len_bin": torch.from_numpy(len_bin),
        "op_bin": torch.from_numpy(op_bin),
        "n_steps_bin": torch.from_numpy(n_steps_bin),
    }


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class LatentSeparatrixNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
        pos_bins: int,
        length_bins: int,
        n_steps_bins: int,
        op_bins: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.hazard = nn.Linear(latent_dim, 1)
        self.energy = nn.Sequential(
            nn.Linear(latent_dim, max(16, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )
        self.nuisance = nn.ModuleDict(
            {
                "pos": nn.Linear(latent_dim, pos_bins),
                "length": nn.Linear(latent_dim, length_bins),
                "op": nn.Linear(latent_dim, op_bins),
                "n_steps": nn.Linear(latent_dim, n_steps_bins),
            }
        )

    def forward(self, x: torch.Tensor, grl_lambda: float = 0.0) -> dict[str, torch.Tensor]:
        B, T, D = x.shape
        z = self.encoder(x.reshape(B * T, D)).reshape(B, T, -1)
        hazard_logit = self.hazard(z).squeeze(-1)
        energy = self.energy(z).squeeze(-1)
        zr = grad_reverse(z, grl_lambda) if grl_lambda > 0 else z
        nuis = {k: head(zr).reshape(B, T, -1) for k, head in self.nuisance.items()}
        return {"z": z, "hazard_logit": hazard_logit, "energy": energy, "nuisance": nuis}


def survival_loss(hazard_logit: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_h = F.logsigmoid(hazard_logit)
    log_not = F.logsigmoid(-hazard_logit)
    losses = []
    B, _T = hazard_logit.shape
    for i in range(B):
        m = mask[i] > 0.5
        Ti = int(m.sum().item())
        g = int(gold[i].item())
        if Ti <= 0:
            continue
        if g >= 0 and g < Ti:
            if g > 0:
                prefix = -log_not[i, :g].sum()
            else:
                prefix = torch.zeros((), device=hazard_logit.device)
            losses.append(prefix - log_h[i, g])
        else:
            losses.append(-log_not[i, :Ti].sum())
    if not losses:
        return torch.zeros((), device=hazard_logit.device)
    return torch.stack(losses).mean()


def energy_loss(energy: torch.Tensor, state_y: torch.Tensor, mask: torch.Tensor, margin: float) -> torch.Tensor:
    valid = mask > 0.5
    if not bool(valid.any()):
        return torch.zeros((), device=energy.device)
    e = energy[valid]
    y = state_y[valid]
    correct = y < 0.5
    wrong = y > 0.5
    parts = []
    if bool(correct.any()):
        parts.append(F.softplus(e[correct]).mean())
    if bool(wrong.any()):
        parts.append(F.softplus(margin - e[wrong]).mean())
    if not parts:
        return torch.zeros((), device=energy.device)
    return torch.stack(parts).mean()


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    max_points: int,
) -> torch.Tensor:
    valid = mask > 0.5
    zz = z[valid]
    yy = labels[valid].long()
    if zz.shape[0] < 4 or torch.unique(yy).numel() < 2:
        return torch.zeros((), device=z.device)
    if zz.shape[0] > max_points:
        perm = torch.randperm(zz.shape[0], device=z.device)[:max_points]
        zz = zz[perm]
        yy = yy[perm]
    zz = F.normalize(zz, dim=-1)
    sim = zz @ zz.T / max(float(temperature), EPS)
    eye = torch.eye(sim.shape[0], device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    same = (yy[:, None] == yy[None, :]) & (~eye)
    denom = torch.logsumexp(sim, dim=1)
    log_prob = sim - denom[:, None]
    pos_count = same.sum(dim=1)
    keep = pos_count > 0
    if not bool(keep.any()):
        return torch.zeros((), device=z.device)
    mean_log_prob_pos = (log_prob * same.float()).sum(dim=1) / pos_count.clamp_min(1).float()
    return -mean_log_prob_pos[keep].mean()


def nuisance_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    losses = []
    specs = [
        ("pos", "pos_bin"),
        ("length", "len_bin"),
        ("op", "op_bin"),
        ("n_steps", "n_steps_bin"),
    ]
    for name, target_key in specs:
        logits = out["nuisance"][name][valid]
        target = batch[target_key][valid].long()
        if logits.numel() == 0:
            continue
        n_classes = logits.shape[-1]
        target = torch.clamp(target, 0, n_classes - 1)
        losses.append(F.cross_entropy(logits, target))
    if not losses:
        return torch.zeros((), device=mask.device)
    return torch.stack(losses).mean()


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_autocast(device: torch.device, enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type=device.type, enabled=enabled)
        except TypeError:
            return torch.amp.autocast(device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def train_model(
    train_dataset: ChainTensorDataset,
    input_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> LatentSeparatrixNet:
    model = LatentSeparatrixNet(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        pos_bins=args.pos_bins,
        length_bins=args.length_bins,
        n_steps_bins=args.n_steps_bins,
        op_bins=4,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_chains,
        num_workers=0,
        drop_last=False,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(enabled=use_amp)

    for epoch in range(int(args.epochs)):
        model.train()
        total = defaultdict(float)
        n_batches = 0
        for batch in loader:
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with amp_autocast(device, enabled=use_amp):
                out = model(batch["x"], grl_lambda=args.grl_lambda if not args.no_nuisance else 0.0)
                mask = batch["mask"]
                surv = survival_loss(out["hazard_logit"], batch["gold"], mask)
                eng = energy_loss(out["energy"], batch["state_y"], mask, args.energy_margin)
                con = supervised_contrastive_loss(
                    out["z"],
                    batch["state_y"],
                    mask,
                    args.temperature,
                    args.max_contrast_points,
                )
                nuis = nuisance_loss(out, batch, mask)
                loss = args.survival_weight * surv
                if not args.no_energy:
                    loss = loss + args.energy_weight * eng
                if not args.no_contrastive:
                    loss = loss + args.contrastive_weight * con
                if not args.no_nuisance:
                    loss = loss + args.nuisance_weight * nuis
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            total["loss"] += float(loss.detach().cpu())
            total["surv"] += float(surv.detach().cpu())
            total["energy"] += float(eng.detach().cpu())
            total["contrastive"] += float(con.detach().cpu())
            total["nuisance"] += float(nuis.detach().cpu())
            n_batches += 1
        if args.verbose and (epoch == 0 or (epoch + 1) % max(1, args.print_every) == 0 or epoch + 1 == args.epochs):
            msg = " ".join(f"{k}={v / max(1, n_batches):.4f}" for k, v in total.items())
            print(f"epoch {epoch + 1:03d}/{args.epochs} {msg}")
    return model


@torch.no_grad()
def score_dataset(
    model: LatentSeparatrixNet,
    dataset: ChainTensorDataset,
    records: Sequence[ChainRecord],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_chains, num_workers=0)
    model.eval()
    rows: list[dict[str, Any]] = []
    chain_scores: dict[int, dict[str, Any]] = {}
    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(batch["x"], grl_lambda=0.0)
        hazard = torch.sigmoid(out["hazard_logit"]).detach().cpu().numpy()
        energy = out["energy"].detach().cpu().numpy()
        z = out["z"].detach().cpu().numpy()
        mask = batch["mask"].detach().cpu().numpy()
        chain_index = batch["chain_index"].detach().cpu().numpy()
        for bi, idx in enumerate(chain_index.tolist()):
            rec = records[int(idx)]
            T = int(mask[bi].sum())
            hz = hazard[bi, :T].astype(float)
            en = energy[bi, :T].astype(float)
            zz = z[bi, :T].astype(np.float32)
            survival = float(1.0 - np.prod(np.clip(1.0 - hz, 1e-6, 1.0)))
            chain_scores[int(idx)] = {
                "chain_index": int(idx),
                "chain_id": rec.chain_id,
                "problem_id": rec.problem_id,
                "gold_error_step": rec.gold_error_step,
                "n_steps": rec.n_steps,
                "y_chain_error": int(rec.gold_error_step >= 0),
                "survival_score": survival,
                "max_hazard": float(np.max(hz)),
                "mean_hazard": float(np.mean(hz)),
                "max_energy": float(np.max(en)),
                "mean_energy": float(np.mean(en)),
            }
            for t in range(T):
                g = rec.gold_error_step
                if g < 0:
                    phase = "correct_chain"
                elif t < g:
                    phase = "pre_error"
                elif t == g:
                    phase = "first_error"
                else:
                    phase = "post_error"
                rows.append(
                    {
                        "chain_index": int(idx),
                        "chain_id": rec.chain_id,
                        "problem_id": rec.problem_id,
                        "step_idx": int(t),
                        "gold_error_step": int(g),
                        "phase": phase,
                        "n_steps": int(rec.n_steps),
                        "step_len": float(rec.token_lengths[t]),
                        "log_step_len": float(math.log1p(float(rec.token_lengths[t]))),
                        "pos": float(t / max(1, rec.n_steps - 1)),
                        "op_id": int(infer_op_id(rec.steps_text[t])),
                        "y_chain_error": int(g >= 0),
                        "y_first_error": int(g == t),
                        "y_pre_error_future": int(g >= 0 and t < g),
                        "is_prefix_or_first": int(g < 0 or t <= g),
                        "is_pre_or_correct": int(g < 0 or t < g),
                        "hazard": float(hz[t]),
                        "energy": float(en[t]),
                        "latent_norm": float(np.linalg.norm(zz[t])),
                        "latent": zz[t],
                    }
                )
    return rows, chain_scores


def row_controls(row: dict[str, Any]) -> list[float]:
    return [
        float(row["pos"]),
        math.log1p(float(row["n_steps"])),
        float(row["log_step_len"]),
        float(row["op_id"]),
        float(row["pos"]) ** 2,
        float(row["log_step_len"]) * float(row["pos"]),
    ]


def row_latent_features(row: dict[str, Any]) -> list[float]:
    z = np.asarray(row["latent"], dtype=np.float32)
    return [
        float(row["hazard"]),
        float(row["energy"]),
        float(row["latent_norm"]),
        *z.astype(float).tolist(),
    ]


def fit_predict_logreg(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    label_key: str,
    feature_set: str,
) -> list[float]:
    if not HAVE_SKLEARN or not test_rows:
        return [float("nan")] * len(test_rows)
    y_train = np.asarray([int(r[label_key]) for r in train_rows], dtype=int)
    if y_train.size == 0 or np.unique(y_train).size < 2:
        return [float("nan")] * len(test_rows)
    if feature_set == "controls":
        X_train = np.asarray([row_controls(r) for r in train_rows], dtype=float)
        X_test = np.asarray([row_controls(r) for r in test_rows], dtype=float)
    elif feature_set == "latent":
        X_train = np.asarray([row_latent_features(r) for r in train_rows], dtype=float)
        X_test = np.asarray([row_latent_features(r) for r in test_rows], dtype=float)
    elif feature_set == "controls+latent":
        X_train = np.asarray([row_controls(r) + row_latent_features(r) for r in train_rows], dtype=float)
        X_test = np.asarray([row_controls(r) + row_latent_features(r) for r in test_rows], dtype=float)
    else:
        raise KeyError(feature_set)
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0)
    sd = np.where(sd < EPS, 1.0, sd)
    X_train = np.nan_to_num((X_train - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num((X_test - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=0)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1].astype(float).tolist()


def weighted_bin_auc(rows: Sequence[dict[str, Any]], score_key: str, label_key: str, bin_key: str) -> dict[str, Any]:
    by_bin: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bin[r[bin_key]].append(r)
    vals = []
    weights = []
    details = {}
    for b, rr in sorted(by_bin.items(), key=lambda kv: str(kv[0])):
        y = [int(r[label_key]) for r in rr]
        if len(set(y)) < 2:
            continue
        auc = safe_auc(y, [float(r[score_key]) for r in rr])
        if math.isfinite(auc):
            vals.append(auc)
            weights.append(len(rr))
            details[str(b)] = {"n": len(rr), "auc": auc}
    if not vals:
        return {"macro": float("nan"), "weighted": float("nan"), "bins": details}
    vals_arr = np.asarray(vals, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    return {
        "macro": float(np.mean(vals_arr)),
        "weighted": float(np.sum(vals_arr * w_arr) / np.sum(w_arr)),
        "bins": details,
    }


def rank_first_errors(records: Sequence[ChainRecord], rows: Sequence[dict[str, Any]], score_key: str) -> dict[str, Any]:
    by_chain: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_chain[int(r["chain_index"])].append(r)
    ranks = []
    percentiles = []
    skipped = 0
    for rec in records:
        g = int(rec.gold_error_step)
        if g < 0:
            continue
        cand = [r for r in by_chain.get(rec.index, []) if int(r["step_idx"]) <= g]
        gold_rows = [r for r in cand if int(r["step_idx"]) == g]
        if not cand or not gold_rows:
            skipped += 1
            continue
        sorted_rows = sorted(cand, key=lambda r: float(r[score_key]), reverse=True)
        rank = next(j + 1 for j, rr in enumerate(sorted_rows) if int(rr["step_idx"]) == g)
        ranks.append(rank)
        percentiles.append(1.0 if len(sorted_rows) == 1 else 1.0 - (rank - 1) / (len(sorted_rows) - 1))
    if not ranks:
        return {"n": 0, "skipped": skipped, "top1": float("nan"), "mean_rank": float("nan"), "mean_percentile": float("nan")}
    arr = np.asarray(ranks, dtype=float)
    return {
        "n": int(arr.size),
        "skipped": int(skipped),
        "top1": float(np.mean(arr == 1)),
        "mean_rank": float(np.mean(arr)),
        "mean_percentile": float(np.mean(percentiles)),
    }


def intrinsic_dim_pr(Z: np.ndarray) -> float:
    Z = np.asarray(Z, dtype=float)
    Z = Z[np.all(np.isfinite(Z), axis=1)]
    if Z.shape[0] < 3:
        return float("nan")
    X = Z - np.mean(Z, axis=0, keepdims=True)
    try:
        s = np.linalg.svd(X, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    eig = s**2
    den = float(np.sum(eig**2))
    if den <= EPS:
        return float("nan")
    return float((np.sum(eig) ** 2) / den)


def latent_separability(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if int(r["is_prefix_or_first"]) == 1]
    y = np.asarray([int(r["y_first_error"]) for r in usable], dtype=int)
    Z = np.asarray([np.asarray(r["latent"], dtype=float) for r in usable], dtype=float)
    if Z.size == 0:
        return {}
    out = {
        "n": int(len(usable)),
        "pos": int(np.sum(y == 1)),
        "id_first_error": intrinsic_dim_pr(Z[y == 1]),
        "id_non_error": intrinsic_dim_pr(Z[y == 0]),
    }
    if np.unique(y).size == 2:
        c0 = np.mean(Z[y == 0], axis=0)
        c1 = np.mean(Z[y == 1], axis=0)
        w = c1 - c0
        norm = np.linalg.norm(w)
        if norm > EPS:
            score = Z @ (w / norm)
            out["centroid_auc"] = safe_auc(y, score)
            out["centroid_margin"] = float(np.mean(score[y == 1]) - np.mean(score[y == 0]))
    return out


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def evaluate_all(
    records: Sequence[ChainRecord],
    rows: list[dict[str, Any]],
    chain_scores: dict[int, dict[str, Any]],
    oof_preds: dict[str, dict[str, list[float]]],
    oof_labels: dict[str, list[int]],
) -> dict[str, Any]:
    first_rows = [r for r in rows if int(r["is_prefix_or_first"]) == 1]
    future_rows = [r for r in rows if int(r["is_pre_or_correct"]) == 1]
    first_y = [int(r["y_first_error"]) for r in first_rows]
    future_y = [int(r["y_chain_error"]) for r in future_rows]

    summary: dict[str, Any] = {
        "n_chains": len(records),
        "n_rows": len(rows),
        "tasks": {},
        "rank": {},
        "response": {},
        "latent_separability": latent_separability(rows),
    }
    summary["tasks"]["first_error"] = {
        "rows": len(first_rows),
        "pos": int(sum(first_y)),
        "single": {
            "hazard": safe_auc(first_y, [r["hazard"] for r in first_rows]),
            "energy": safe_auc(first_y, [r["energy"] for r in first_rows]),
            "latent_norm": safe_auc(first_y, [r["latent_norm"] for r in first_rows]),
            "step_len": safe_auc(first_y, [r["step_len"] for r in first_rows]),
            "pos": safe_auc(first_y, [r["pos"] for r in first_rows]),
        },
        "single_auprc": {
            "hazard": safe_auprc(first_y, [r["hazard"] for r in first_rows]),
            "energy": safe_auprc(first_y, [r["energy"] for r in first_rows]),
        },
        "matched_auc": {
            "hazard_by_pos_bin": weighted_bin_auc(first_rows, "hazard", "y_first_error", "pos_bin_eval"),
            "hazard_by_len_bin": weighted_bin_auc(first_rows, "hazard", "y_first_error", "len_bin_eval"),
        },
    }
    summary["tasks"]["pre_error_future"] = {
        "rows": len(future_rows),
        "pos": int(sum(future_y)),
        "single": {
            "hazard": safe_auc(future_y, [r["hazard"] for r in future_rows]),
            "energy": safe_auc(future_y, [r["energy"] for r in future_rows]),
            "latent_norm": safe_auc(future_y, [r["latent_norm"] for r in future_rows]),
            "step_len": safe_auc(future_y, [r["step_len"] for r in future_rows]),
            "pos": safe_auc(future_y, [r["pos"] for r in future_rows]),
        },
    }

    for task in ["first_error", "pre_error_future"]:
        summary["tasks"][task]["models"] = {}
        for fs, pred in oof_preds[task].items():
            summary["tasks"][task]["models"][fs] = safe_auc(oof_labels[task], pred)
            summary["tasks"][task].setdefault("models_auprc", {})[fs] = safe_auprc(oof_labels[task], pred)

    for key in ["hazard", "energy", "latent_norm"]:
        summary["rank"][key] = rank_first_errors(records, rows, key)

    chains = [chain_scores[k] for k in sorted(chain_scores)]
    y_chain = [int(c["y_chain_error"]) for c in chains]
    for key in ["survival_score", "max_hazard", "mean_hazard", "max_energy", "mean_energy"]:
        summary["response"][key] = {
            "auc": safe_auc(y_chain, [c[key] for c in chains]),
            "auprc": safe_auprc(y_chain, [c[key] for c in chains]),
        }
    return summary


def add_eval_bins(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    len_edges = bin_edges([r["log_step_len"] for r in rows], args.length_bins)
    for r in rows:
        r["pos_bin_eval"] = int(min(args.pos_bins - 1, max(0, math.floor(float(r["pos"]) * args.pos_bins))))
        r["len_bin_eval"] = digitize_with_edges(float(r["log_step_len"]), len_edges)


def run_cv(records: list[ChainRecord], layers: list[int], args: argparse.Namespace) -> dict[str, Any]:
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch is required for latent_separatrix_audit.py")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    n_layers = records[0].vectors.shape[1]
    layer_positions = choose_layer_positions(n_layers, args.layer_stride, args.max_layers)
    folds = make_folds(records, args.n_folds, args.seed)

    all_rows: list[dict[str, Any]] = []
    all_chain_scores: dict[int, dict[str, Any]] = {}
    oof_preds: dict[str, dict[str, list[float]]] = {
        "first_error": defaultdict(list),
        "pre_error_future": defaultdict(list),
    }
    oof_labels: dict[str, list[int]] = {"first_error": [], "pre_error_future": []}
    fold_meta = []

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        projector = fit_pca_projector(
            records,
            train_idx,
            layer_positions,
            args.layer_pool,
            args.pca_dim,
            args.max_pca_samples,
            args.seed + fold_id,
        )
        needed = np.concatenate([train_idx, test_idx])
        features = transform_records(records, needed, projector, layer_positions, args.layer_pool)
        nuisance_edges = fit_nuisance_edges(records, train_idx, args)

        train_ds = ChainTensorDataset(records, train_idx, features, nuisance_edges, args.pos_bins)
        test_ds = ChainTensorDataset(records, test_idx, features, nuisance_edges, args.pos_bins)
        model = train_model(train_ds, projector.dim, args, device)

        train_rows, _ = score_dataset(model, train_ds, records, device, args.eval_batch_size)
        test_rows, test_chains = score_dataset(model, test_ds, records, device, args.eval_batch_size)
        for r in test_rows:
            r["fold"] = fold_id
        all_rows.extend(test_rows)
        all_chain_scores.update(test_chains)

        train_first = [r for r in train_rows if int(r["is_prefix_or_first"]) == 1]
        test_first = [r for r in test_rows if int(r["is_prefix_or_first"]) == 1]
        train_future = [r for r in train_rows if int(r["is_pre_or_correct"]) == 1]
        test_future = [r for r in test_rows if int(r["is_pre_or_correct"]) == 1]

        oof_labels["first_error"].extend([int(r["y_first_error"]) for r in test_first])
        oof_labels["pre_error_future"].extend([int(r["y_chain_error"]) for r in test_future])
        for fs in ["controls", "latent", "controls+latent"]:
            oof_preds["first_error"][fs].extend(fit_predict_logreg(train_first, test_first, "y_first_error", fs))
            oof_preds["pre_error_future"][fs].extend(fit_predict_logreg(train_future, test_future, "y_chain_error", fs))

        fold_meta.append(
            {
                "fold": fold_id,
                "train_chains": int(len(train_idx)),
                "test_chains": int(len(test_idx)),
                "pca_dim": int(projector.dim),
                "train_rows": int(sum(records[int(i)].n_steps for i in train_idx)),
                "test_rows": int(sum(records[int(i)].n_steps for i in test_idx)),
            }
        )
        print(
            f"fold {fold_id}: train={len(train_idx)} test={len(test_idx)} "
            f"pca_dim={projector.dim} test_rows={fold_meta[-1]['test_rows']}"
        )

    add_eval_bins(all_rows, args)
    summary = evaluate_all(records, all_rows, all_chain_scores, oof_preds, oof_labels)
    summary["folds"] = fold_meta
    summary["layers"] = {
        "layer_positions": [int(x) for x in layer_positions],
        "layer_ids": [int(layers[p]) if p < len(layers) else int(p) for p in layer_positions],
        "layer_pool": args.layer_pool,
    }
    return {
        "summary": summary,
        "rows": all_rows,
        "chain_scores": list(all_chain_scores.values()),
    }


def save_outputs(result: dict[str, Any], args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag

    rows = result["rows"]
    chain_scores = result["chain_scores"]
    summary = result["summary"]

    row_fields = [
        "fold",
        "chain_index",
        "chain_id",
        "problem_id",
        "step_idx",
        "gold_error_step",
        "phase",
        "n_steps",
        "step_len",
        "pos",
        "op_id",
        "pos_bin_eval",
        "len_bin_eval",
        "y_chain_error",
        "y_first_error",
        "y_pre_error_future",
        "hazard",
        "energy",
        "latent_norm",
    ]
    chain_fields = [
        "chain_index",
        "chain_id",
        "problem_id",
        "gold_error_step",
        "n_steps",
        "y_chain_error",
        "survival_score",
        "max_hazard",
        "mean_hazard",
        "max_energy",
        "mean_energy",
    ]
    write_csv(out_dir / f"{tag}_latent_separatrix_rows.csv", rows, row_fields)
    write_csv(out_dir / f"{tag}_latent_separatrix_chains.csv", chain_scores, chain_fields)

    latent = np.asarray([np.asarray(r["latent"], dtype=np.float32) for r in rows], dtype=np.float32)
    row_index = np.asarray([[int(r["chain_index"]), int(r["step_idx"]), int(r["y_first_error"]), int(r["y_chain_error"])] for r in rows], dtype=np.int32)
    np.savez_compressed(out_dir / f"{tag}_latent_separatrix_latents.npz", latent=latent, row_index=row_index)

    json_path = out_dir / f"{tag}_latent_separatrix_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(finite_json(summary), f, ensure_ascii=False, indent=2)

    md_path = out_dir / f"{tag}_latent_separatrix_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(render_markdown(summary, args))

    print(render_console(summary))
    print(f"\nSaved: {json_path}")


def render_console(summary: dict[str, Any]) -> str:
    lines = ["===== Latent Separatrix summary ====="]
    lines.append(f"chains {summary['n_chains']} | rows {summary['n_rows']}")
    if "layers" in summary:
        lines.append(f"layer positions {summary['layers'].get('layer_positions')}")
        lines.append(f"layer ids {summary['layers'].get('layer_ids')}")
    for task in ["first_error", "pre_error_future"]:
        t = summary["tasks"][task]
        lines.append(f"\nTask {task}: rows {t['rows']} pos {t['pos']}")
        for k, v in sorted(t["single"].items(), key=lambda kv: (-(kv[1] if math.isfinite(kv[1]) else -1), kv[0])):
            lines.append(f"  single {k:<14} AUROC {v:.3f}")
        for k, v in sorted(t.get("models", {}).items(), key=lambda kv: (-(kv[1] if math.isfinite(kv[1]) else -1), kv[0])):
            lines.append(f"  model  {k:<16} AUROC {v:.3f}")
    lines.append("\nResponse:")
    for k, d in sorted(summary["response"].items(), key=lambda kv: (-(kv[1]["auc"] if math.isfinite(kv[1]["auc"]) else -1), kv[0])):
        lines.append(f"  {k:<16} AUROC {d['auc']:.3f} AUPRC {d['auprc']:.3f}")
    lines.append("\nRanks:")
    for k, d in summary["rank"].items():
        lines.append(f"  {k:<12} n={d['n']} top1={d['top1']:.3f} mean_rank={d['mean_rank']:.2f}")
    sep = summary.get("latent_separability", {})
    if sep:
        lines.append("\nLatent separability:")
        for k in ["centroid_auc", "centroid_margin", "id_first_error", "id_non_error"]:
            if k in sep:
                lines.append(f"  {k}: {sep[k]:.4f}")
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any], args: argparse.Namespace) -> str:
    lines = [
        "# Latent Separatrix Audit Summary",
        "",
        f"- Input: `{args.input}`",
        f"- Chains: `{summary['n_chains']}`",
        f"- Step rows: `{summary['n_rows']}`",
        f"- Layer positions: `{summary['layers']['layer_positions']}`",
        f"- Layer ids: `{summary['layers']['layer_ids']}`",
        f"- Layer pool: `{summary['layers']['layer_pool']}`",
        "",
        "## Core Test",
        "",
        "The method tests whether raw step hidden states become more separable after a learned latent chart:",
        "",
        "$$",
        "z_t=f_\\theta(h_t),\\qquad \\lambda_t=\\sigma(w^\\top z_t+b)",
        "$$",
        "",
        "Response risk is computed by survival aggregation:",
        "",
        "$$",
        "P(\\mathrm{error\\ by\\ }T)=1-\\prod_{t=1}^{T}(1-\\lambda_t).",
        "$$",
        "",
        "## Results",
        "",
    ]
    for task in ["first_error", "pre_error_future"]:
        t = summary["tasks"][task]
        lines.extend([f"### {task}", "", f"- rows: `{t['rows']}`", f"- positives: `{t['pos']}`", "", "| score | AUROC | AUPRC |", "|---|---:|---:|"])
        for k, v in sorted(t["single"].items()):
            auprc = t.get("single_auprc", {}).get(k, float("nan"))
            lines.append(f"| single/{k} | {v:.4f} | {auprc:.4f} |")
        for k, v in sorted(t.get("models", {}).items()):
            auprc = t.get("models_auprc", {}).get(k, float("nan"))
            lines.append(f"| model/{k} | {v:.4f} | {auprc:.4f} |")
        lines.append("")
    lines.extend(["## Response", "", "| score | AUROC | AUPRC |", "|---|---:|---:|"])
    for k, d in sorted(summary["response"].items()):
        lines.append(f"| {k} | {d['auc']:.4f} | {d['auprc']:.4f} |")
    lines.extend(["", "## First-Error Rank", "", "| score | n | top1 | mean rank | mean percentile |", "|---|---:|---:|---:|---:|"])
    for k, d in summary["rank"].items():
        lines.append(f"| {k} | {d['n']} | {d['top1']:.4f} | {d['mean_rank']:.4f} | {d['mean_percentile']:.4f} |")
    lines.extend(["", "## Latent Separability", "", "```json", json.dumps(finite_json(summary.get("latent_separability", {})), ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def make_synthetic_records(seed: int = 0, n: int = 80, d: int = 96, layers: int = 4) -> list[ChainRecord]:
    rng = np.random.default_rng(seed)
    records = []
    v_err = rng.normal(size=d)
    v_err /= np.linalg.norm(v_err)
    v_len = rng.normal(size=d)
    v_len /= np.linalg.norm(v_len)
    for i in range(n):
        T = int(rng.integers(5, 10))
        is_err = i % 2 == 0
        gold = int(rng.integers(2, T - 1)) if is_err else -1
        vec = rng.normal(0, 0.4, size=(T, layers, d)).astype(np.float32)
        base = rng.normal(0, 0.7, size=d)
        step_lens = rng.integers(4, 30, size=T).astype(np.float32)
        for t in range(T):
            phase = t / max(1, T - 1)
            signal = 0.0
            if is_err and t >= gold:
                signal = 2.8 + 0.2 * (t - gold)
            for l in range(layers):
                vec[t, l] += base + phase * 0.4 * rng.normal(size=d)
                vec[t, l] += 0.04 * float(step_lens[t]) * v_len
                vec[t, l] += signal * v_err
        texts = [f"synthetic step {t} value {int(step_lens[t])}" for t in range(T)]
        records.append(
            ChainRecord(
                index=i,
                chain_id=f"syn-{i}",
                problem_id=f"prob-{i // 2}",
                gold_error_step=gold,
                vectors=vec,
                steps_text=texts,
                token_lengths=step_lens,
            )
        )
    return records


def default_selftest_args() -> argparse.Namespace:
    return argparse.Namespace(
        input="<synthetic>",
        output_dir="outputs/latent_separatrix_selftest",
        tag="latent_separatrix_selftest",
        mode="step_exp",
        max_chains=None,
        n_folds=4,
        seed=7,
        layer_stride=1,
        max_layers=4,
        layer_pool="mean",
        pca_dim=32,
        max_pca_samples=5000,
        hidden_dim=64,
        latent_dim=12,
        dropout=0.05,
        epochs=10,
        batch_size=16,
        eval_batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        amp=False,
        device="auto",
        survival_weight=1.0,
        energy_weight=0.2,
        contrastive_weight=0.05,
        nuisance_weight=0.05,
        grl_lambda=0.2,
        energy_margin=1.0,
        temperature=0.2,
        max_contrast_points=256,
        no_energy=False,
        no_contrastive=False,
        no_nuisance=False,
        length_bins=4,
        n_steps_bins=4,
        pos_bins=5,
        grad_clip=1.0,
        verbose=False,
        print_every=5,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="?", help="ProcessBench npz file with step vectors.")
    ap.add_argument("--output_dir", default="outputs/latent_separatrix")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--max_chains", type=int, default=None)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer_stride", type=int, default=1)
    ap.add_argument("--max_layers", type=int, default=8)
    ap.add_argument("--layer_pool", choices=["mean", "last", "first_last", "concat"], default="mean")
    ap.add_argument("--pca_dim", type=int, default=256)
    ap.add_argument("--max_pca_samples", type=int, default=20000)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--survival_weight", type=float, default=1.0)
    ap.add_argument("--energy_weight", type=float, default=0.25)
    ap.add_argument("--contrastive_weight", type=float, default=0.05)
    ap.add_argument("--nuisance_weight", type=float, default=0.05)
    ap.add_argument("--grl_lambda", type=float, default=0.2)
    ap.add_argument("--energy_margin", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_contrast_points", type=int, default=512)
    ap.add_argument("--no_energy", action="store_true")
    ap.add_argument("--no_contrastive", action="store_true")
    ap.add_argument("--no_nuisance", action="store_true")
    ap.add_argument("--length_bins", type=int, default=5)
    ap.add_argument("--n_steps_bins", type=int, default=5)
    ap.add_argument("--pos_bins", type=int, default=6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--print_every", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if not args.selftest and not args.input:
        ap.error("input is required unless --selftest is set")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.selftest:
        st_args = default_selftest_args()
        st_args.output_dir = args.output_dir
        st_args.tag = args.tag if args.tag != "run" else st_args.tag
        records = make_synthetic_records(seed=st_args.seed)
        layers = list(range(records[0].vectors.shape[1]))
        result = run_cv(records, layers, st_args)
        save_outputs(result, st_args)
        auc = result["summary"]["tasks"]["first_error"]["single"]["hazard"]
        if not math.isfinite(auc) or auc < 0.80:
            raise SystemExit(f"selftest failed: first_error hazard AUROC={auc}")
        return

    records, layers, vector_key = load_records(args.input, args.mode, args.max_chains)
    print(f"Loaded {len(records)} chains from {args.input}; vector_key={vector_key}; layers={layers}")
    result = run_cv(records, layers, args)
    result["summary"]["input"] = str(args.input)
    result["summary"]["vector_key"] = vector_key
    save_outputs(result, args)


if __name__ == "__main__":
    main()
