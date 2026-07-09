#!/usr/bin/env python3
"""Train a GeoFaith-style beta-VAE latent chart on ProcessBench step vectors.

This script is deliberately separate from `latent_separatrix_audit.py`.

Default training is unsupervised:
  step hidden state -> fold-local PCA/standardization -> beta-VAE -> latent z

Labels are used only after VAE training, to evaluate whether the learned latent
space separates correct and erroneous reasoning states.  This is closer to the
GeoFaith VAE path than the discriminative latent separatrix monitor.

Numerical policy:
  * no mixed precision by default;
  * reconstruction/KL losses are computed in float32;
  * encoder and decoder log variances are clamped;
  * gradients are clipped.

Those choices are intentional because VAE Gaussian NLL/KL is easy to overflow
in fp16 when hidden-state magnitudes or variance predictions are large.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass

    class _MissingNN:
        Module = object

    nn = _MissingNN()  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    HAVE_TORCH = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

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
    token_lengths: np.ndarray

    @property
    def n_steps(self) -> int:
        return int(self.vectors.shape[0])


@dataclass
class PCAStandardizer:
    mean: np.ndarray
    components: np.ndarray
    pca_scale: np.ndarray
    feat_mean: np.ndarray
    feat_std: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.components.shape[0])

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        flat = arr.reshape(-1, arr.shape[-1]).astype(np.float32, copy=False)
        z = (flat - self.mean.astype(np.float32)) @ self.components.T.astype(np.float32)
        z = z / self.pca_scale.astype(np.float32)
        z = (z - self.feat_mean.astype(np.float32)) / self.feat_std.astype(np.float32)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
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
    if not HAVE_SKLEARN:
        return float("nan")
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def text_len(s: str) -> int:
    return max(1, len(re.findall(r"\S+", str(s))))


def get_obj_array_value(arr: np.ndarray | None, i: int) -> Any:
    if arr is None:
        return None
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
    raise KeyError("Need `stepvec` or `sv_vec_<mode>` raw vectors.")


def infer_vector_layer_count(raw_vectors: np.ndarray) -> int:
    for v in raw_vectors:
        arr = np.asarray(v)
        if arr.ndim == 3:
            return int(arr.shape[1])
    raise ValueError("Could not infer layer count from vector array.")


def layer_metadata_for_vector_key(z: np.lib.npyio.NpzFile, vector_key: str, raw_vectors: np.ndarray) -> list[int]:
    n_vec_layers = infer_vector_layer_count(raw_vectors)
    candidates: list[list[int]] = []
    if vector_key == "stepvec" and "sv_layers" in z.files:
        candidates.append([int(x) for x in np.asarray(z["sv_layers"]).tolist()])
    if "layers_used" in z.files:
        candidates.append([int(x) for x in np.asarray(z["layers_used"]).tolist()])
    if vector_key != "stepvec" and "sv_layers" in z.files:
        candidates.append([int(x) for x in np.asarray(z["sv_layers"]).tolist()])
    for vals in candidates:
        if len(vals) == n_vec_layers:
            return vals
    return list(range(n_vec_layers))


def load_records(path: str | Path, mode: str, max_chains: int | None = None) -> tuple[list[ChainRecord], list[int], str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. Use canonical files such as "
            "`data/features/full_gsm8k.npz`, `data/features/full_math.npz`, "
            "or `data/features/full_omnimath.npz`."
        )
    z = np.load(p, allow_pickle=True)
    vector_key = pick_vector_key(z, mode)
    raw_vectors = z[vector_key]
    layers = layer_metadata_for_vector_key(z, vector_key, raw_vectors)

    if "gold_error_step" in z.files:
        gold = np.asarray(z["gold_error_step"], dtype=int)
    elif "labels" in z.files:
        gold = np.asarray(z["labels"], dtype=int)
    else:
        raise KeyError("Need `gold_error_step` or `labels`.")

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
        vec = np.asarray(get_obj_array_value(raw_vectors, i))
        if vec.ndim != 3 or vec.shape[0] < 2 or vec.shape[1] < 1:
            continue
        if not np.all(np.isfinite(vec)):
            continue
        g = int(gold[i])
        if g >= vec.shape[0]:
            continue

        cid = str(get_obj_array_value(ids, i)) if ids is not None else str(i)
        pid = str(get_obj_array_value(problem_ids, i)) if problem_ids is not None else cid

        st = get_obj_array_value(steps_text_arr, i) if steps_text_arr is not None else None
        if st is None:
            texts = [f"step {j}" for j in range(vec.shape[0])]
        else:
            texts = [str(x) for x in list(st)]
            if len(texts) < vec.shape[0]:
                texts += [f"step {j}" for j in range(len(texts), vec.shape[0])]
            texts = texts[: vec.shape[0]]

        token_lengths = np.array([text_len(s) for s in texts], dtype=np.float32)
        if step_ranges is not None:
            rng = get_obj_array_value(step_ranges, i)
            try:
                rr = np.asarray(rng)
                if rr.shape[0] >= vec.shape[0] and rr.shape[-1] >= 2:
                    token_lengths = np.maximum(1, rr[: vec.shape[0], 1] - rr[: vec.shape[0], 0]).astype(np.float32)
            except Exception:
                pass

        records.append(
            ChainRecord(
                index=int(i),
                chain_id=cid,
                problem_id=pid,
                gold_error_step=g,
                vectors=vec.astype(np.float32, copy=False),
                steps_text=texts,
                token_lengths=token_lengths,
            )
        )
    if not records:
        raise RuntimeError(f"No valid chains loaded from {path}; vector_key={vector_key}")
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
        return np.concatenate([x[:, 0, :], x[:, -1, :]], axis=-1).astype(np.float32, copy=False)
    if layer_pool == "concat":
        return x.reshape(x.shape[0], -1).astype(np.float32, copy=False)
    raise ValueError(f"Unknown layer_pool={layer_pool}")


def fit_pca_standardizer(
    records: Sequence[ChainRecord],
    train_idx: Sequence[int],
    layer_positions: Sequence[int],
    layer_pool: str,
    pca_dim: int,
    max_samples: int,
    seed: int,
) -> PCAStandardizer:
    rng = np.random.default_rng(seed)
    chunks = [record_step_features(records[int(i)], layer_positions, layer_pool) for i in train_idx]
    X = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if X.shape[0] > max_samples:
        take = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[take]
    mean = X.mean(axis=0, keepdims=False).astype(np.float32)
    Xc = X - mean
    k = int(max(1, min(pca_dim, Xc.shape[0] - 1, Xc.shape[1])))
    _u, s, vt = np.linalg.svd(Xc.astype(np.float32), full_matrices=False)
    components = vt[:k].astype(np.float32)
    pca_scale = (s[:k] / math.sqrt(max(1, Xc.shape[0] - 1))).astype(np.float32)
    pca_scale = np.maximum(pca_scale, 1e-6)
    Z = (Xc @ components.T) / pca_scale
    feat_mean = Z.mean(axis=0).astype(np.float32)
    feat_std = np.maximum(Z.std(axis=0).astype(np.float32), 1e-4)
    return PCAStandardizer(mean=mean, components=components, pca_scale=pca_scale, feat_mean=feat_mean, feat_std=feat_std)


def transform_records(
    records: Sequence[ChainRecord],
    indices: Sequence[int],
    projector: PCAStandardizer,
    layer_positions: Sequence[int],
    layer_pool: str,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for idx in indices:
        rec = records[int(idx)]
        out[int(idx)] = projector.transform(record_step_features(rec, layer_positions, layer_pool))
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


class StepDataset(Dataset):
    def __init__(self, records: Sequence[ChainRecord], indices: Sequence[int], features: dict[int, np.ndarray]) -> None:
        self.items: list[tuple[int, int]] = []
        self.records = records
        self.features = features
        for idx in indices:
            rec = records[int(idx)]
            for t in range(rec.n_steps):
                self.items.append((int(idx), int(t)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> dict[str, Any]:
        chain_idx, step_idx = self.items[item]
        rec = self.records[chain_idx]
        x = self.features[chain_idx][step_idx]
        g = int(rec.gold_error_step)
        return {
            "x": torch.from_numpy(x.astype(np.float32, copy=False)),
            "chain_index": chain_idx,
            "step_idx": step_idx,
            "gold": g,
            "n_steps": rec.n_steps,
            "pos": float(step_idx / max(1, rec.n_steps - 1)),
            "step_len": float(math.log1p(float(rec.token_lengths[step_idx]))),
            "y_chain_error": int(g >= 0),
            "y_first_error": int(g >= 0 and step_idx == g),
            "y_pre_error_future": int(g >= 0 and step_idx < g),
            "y_state_error_or_after": int(g >= 0 and step_idx >= g),
        }


class BetaVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        enc_logvar_clip: float,
        dec_logvar_min: float,
        dec_logvar_max: float,
    ) -> None:
        super().__init__()
        self.enc_logvar_clip = float(enc_logvar_clip)
        self.dec_logvar_min = float(dec_logvar_min)
        self.dec_logvar_max = float(dec_logvar_max)
        enc_layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        prev = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(prev, int(h)), nn.GELU(), nn.Dropout(dropout)])
            prev = int(h)
        self.encoder = nn.Sequential(*enc_layers)
        self.z_mu = nn.Linear(prev, latent_dim)
        self.z_logvar = nn.Linear(prev, latent_dim)

        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(list(hidden_dims)):
            dec_layers.extend([nn.Linear(prev, int(h)), nn.GELU(), nn.Dropout(dropout)])
            prev = int(h)
        self.decoder = nn.Sequential(*dec_layers)
        self.x_mu = nn.Linear(prev, input_dim)
        self.x_logvar = nn.Linear(prev, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x.float())
        mu = self.z_mu(h)
        logvar = torch.clamp(self.z_logvar(h), -self.enc_logvar_clip, self.enc_logvar_clip)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps = torch.randn_like(mu)
            return mu + torch.exp(0.5 * logvar) * eps
        return mu

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder(z.float())
        mu = self.x_mu(h)
        logvar = torch.clamp(self.x_logvar(h), self.dec_logvar_min, self.dec_logvar_max)
        return mu, logvar

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = self.decode(z)
        return {"z": z, "mu_z": mu_z, "logvar_z": logvar_z, "mu_x": mu_x, "logvar_x": logvar_x}


def vae_loss(out: dict[str, torch.Tensor], x: torch.Tensor, beta: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = x.float()
    mu_x = out["mu_x"].float()
    logvar_x = out["logvar_x"].float()
    inv_var = torch.exp(-logvar_x)
    rec_per_dim = 0.5 * (logvar_x + (target - mu_x).pow(2) * inv_var)
    rec = rec_per_dim.sum(dim=-1).mean()

    mu_z = out["mu_z"].float()
    logvar_z = out["logvar_z"].float()
    kl_per_dim = -0.5 * (1.0 + logvar_z - mu_z.pow(2) - torch.exp(logvar_z))
    kl = kl_per_dim.sum(dim=-1).mean()
    total = rec + float(beta) * kl
    total = torch.nan_to_num(total, nan=1e6, posinf=1e6, neginf=1e6)
    return total, {"rec": rec.detach(), "kl": kl.detach(), "beta": torch.tensor(float(beta), device=target.device)}


def collate_steps(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["x"] = torch.stack([b["x"] for b in batch], dim=0)
    for key in [
        "chain_index",
        "step_idx",
        "gold",
        "n_steps",
        "y_chain_error",
        "y_first_error",
        "y_pre_error_future",
        "y_state_error_or_after",
    ]:
        out[key] = torch.tensor([int(b[key]) for b in batch], dtype=torch.long)
    for key in ["pos", "step_len"]:
        out[key] = torch.tensor([float(b[key]) for b in batch], dtype=torch.float32)
    return out


def train_vae(dataset: StepDataset, input_dim: int, args: argparse.Namespace, device: torch.device) -> BetaVAE:
    hidden_dims = [int(x) for x in args.hidden_dims.split(",") if str(x).strip()]
    model = BetaVAE(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        enc_logvar_clip=args.enc_logvar_clip,
        dec_logvar_min=args.dec_logvar_min,
        dec_logvar_max=args.dec_logvar_max,
    ).to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_steps, num_workers=0)

    best_state = None
    best_loss = float("inf")
    patience_left = int(args.patience)
    for epoch in range(int(args.epochs)):
        model.train()
        beta = float(args.beta_max) * min(1.0, float(epoch + 1) / max(1.0, float(args.warmup_epochs)))
        totals = defaultdict(float)
        n = 0
        for batch in loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss, parts = vae_loss(out, x, beta)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            bs = int(x.shape[0])
            n += bs
            totals["loss"] += float(loss.detach().cpu()) * bs
            totals["rec"] += float(parts["rec"].cpu()) * bs
            totals["kl"] += float(parts["kl"].cpu()) * bs
        mean_loss = totals["loss"] / max(1, n)
        if args.verbose and (epoch + 1 == 1 or (epoch + 1) % args.print_every == 0):
            print(
                f"  epoch {epoch+1:03d} beta={beta:.3f} "
                f"loss={mean_loss:.4f} rec={totals['rec']/max(1,n):.4f} kl={totals['kl']/max(1,n):.4f}",
                flush=True,
            )
        if mean_loss < best_loss - 1e-4:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(args.patience)
        else:
            patience_left -= 1
            if epoch + 1 > args.warmup_epochs and patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad() if HAVE_TORCH else (lambda f: f)
def encode_dataset(model: BetaVAE, dataset: StepDataset, records: Sequence[ChainRecord], device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_steps, num_workers=0)
    rows: list[dict[str, Any]] = []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        out = model(x)
        loss_vec = 0.5 * (out["logvar_x"].float() + (x - out["mu_x"].float()).pow(2) * torch.exp(-out["logvar_x"].float()))
        rec_nll = loss_vec.sum(dim=-1)
        kl_vec = -0.5 * (1.0 + out["logvar_z"].float() - out["mu_z"].float().pow(2) - torch.exp(out["logvar_z"].float()))
        kl = kl_vec.sum(dim=-1)
        z_mu = out["mu_z"].detach().cpu().numpy().astype(np.float32)
        z_logvar = out["logvar_z"].detach().cpu().numpy().astype(np.float32)
        rec_np = rec_nll.detach().cpu().numpy()
        kl_np = kl.detach().cpu().numpy()
        dec_unc = out["logvar_x"].float().mean(dim=-1).detach().cpu().numpy()
        post_unc = out["logvar_z"].float().mean(dim=-1).detach().cpu().numpy()
        for j in range(z_mu.shape[0]):
            cidx = int(batch["chain_index"][j].item())
            tidx = int(batch["step_idx"][j].item())
            rec = records[cidx]
            rows.append(
                {
                    "chain_index": cidx,
                    "chain_id": rec.chain_id,
                    "problem_id": rec.problem_id,
                    "step_idx": tidx,
                    "gold_error_step": int(rec.gold_error_step),
                    "n_steps": int(rec.n_steps),
                    "pos": float(batch["pos"][j].item()),
                    "step_len": float(batch["step_len"][j].item()),
                    "y_chain_error": int(batch["y_chain_error"][j].item()),
                    "y_first_error": int(batch["y_first_error"][j].item()),
                    "y_pre_error_future": int(batch["y_pre_error_future"][j].item()),
                    "y_state_error_or_after": int(batch["y_state_error_or_after"][j].item()),
                    "rec_nll": float(rec_np[j]),
                    "kl": float(kl_np[j]),
                    "posterior_unc": float(post_unc[j]),
                    "decoder_unc": float(dec_unc[j]),
                    "latent_norm": float(np.linalg.norm(z_mu[j])),
                    "latent": z_mu[j],
                    "latent_logvar": z_logvar[j],
                }
            )
    return rows


def row_features(rows: Sequence[dict[str, Any]], mode: str) -> np.ndarray:
    mats = []
    for r in rows:
        controls = np.array(
            [
                float(r["pos"]),
                float(r["step_len"]),
                math.log1p(float(r["n_steps"])),
                float(r["pos"]) ** 2,
            ],
            dtype=np.float32,
        )
        scores = np.array(
            [
                float(r["rec_nll"]),
                float(r["kl"]),
                float(r["posterior_unc"]),
                float(r["decoder_unc"]),
                float(r["latent_norm"]),
            ],
            dtype=np.float32,
        )
        latent = np.asarray(r["latent"], dtype=np.float32)
        if mode == "controls":
            x = controls
        elif mode == "vae_scores":
            x = scores
        elif mode == "latent":
            x = latent
        elif mode == "controls+vae_scores":
            x = np.concatenate([controls, scores])
        elif mode == "controls+latent":
            x = np.concatenate([controls, latent])
        elif mode == "all":
            x = np.concatenate([controls, scores, latent])
        else:
            raise ValueError(mode)
        mats.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
    if not mats:
        return np.zeros((0, 1), dtype=np.float32)
    return np.stack(mats, axis=0).astype(np.float32)


def task_rows(rows: Sequence[dict[str, Any]], task: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected = []
    labels = []
    for r in rows:
        g = int(r["gold_error_step"])
        t = int(r["step_idx"])
        if task == "first_error":
            if g >= 0 and t > g:
                continue
            selected.append(r)
            labels.append(int(g >= 0 and t == g))
        elif task == "pre_error_future":
            if g >= 0 and t >= g:
                continue
            selected.append(r)
            labels.append(int(g >= 0 and t < g))
        elif task == "state_error_or_after":
            selected.append(r)
            labels.append(int(g >= 0 and t >= g))
        else:
            raise ValueError(task)
    return selected, np.asarray(labels, dtype=int)


def fit_predict_probe(train_rows: Sequence[dict[str, Any]], test_rows: Sequence[dict[str, Any]], task: str, mode: str) -> np.ndarray:
    if not HAVE_SKLEARN:
        return np.full(len(task_rows(test_rows, task)[0]), np.nan, dtype=float)
    tr, ytr = task_rows(train_rows, task)
    te, _yte = task_rows(test_rows, task)
    if len(tr) == 0 or len(te) == 0 or np.unique(ytr).size < 2:
        return np.full(len(te), np.nan, dtype=float)
    Xtr = row_features(tr, mode)
    Xte = row_features(te, mode)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1].astype(float)


def aggregate_response(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_chain: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_chain[int(r["chain_index"])].append(r)
    out = []
    for cidx, rs in by_chain.items():
        first = rs[0]
        d = {
            "chain_index": cidx,
            "chain_id": first["chain_id"],
            "problem_id": first["problem_id"],
            "gold_error_step": int(first["gold_error_step"]),
            "n_steps": int(first["n_steps"]),
            "y_chain_error": int(first["y_chain_error"]),
        }
        for key in ["rec_nll", "kl", "posterior_unc", "decoder_unc", "latent_norm"]:
            vals = np.asarray([float(r[key]) for r in rs], dtype=float)
            d[f"mean_{key}"] = float(np.mean(vals))
            d[f"max_{key}"] = float(np.max(vals))
            d[f"top20_mean_{key}"] = float(np.mean(np.sort(vals)[-max(1, int(math.ceil(0.2 * len(vals)))) :]))
        for pred_key in ["probe_controls", "probe_vae_scores", "probe_latent", "probe_controls_latent", "probe_all"]:
            vals = np.asarray([float(r.get(pred_key, np.nan)) for r in rs], dtype=float)
            if np.isfinite(vals).any():
                d[f"max_{pred_key}"] = float(np.nanmax(vals))
                d[f"mean_{pred_key}"] = float(np.nanmean(vals))
        out.append(d)
    return out


def evaluate_rows(rows: list[dict[str, Any]], chain_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"tasks": {}, "response": {}}
    single_scores = ["rec_nll", "kl", "posterior_unc", "decoder_unc", "latent_norm"]
    probe_scores = ["probe_controls", "probe_vae_scores", "probe_latent", "probe_controls_latent", "probe_all"]
    for task in ["first_error", "pre_error_future", "state_error_or_after"]:
        selected, y = task_rows(rows, task)
        task_summary: dict[str, Any] = {"rows": len(selected), "pos": int(y.sum()), "single": {}, "single_auprc": {}}
        for key in single_scores + probe_scores:
            s = [float(r.get(key, np.nan)) for r in selected]
            task_summary["single"][key] = safe_auc(y, s)
            task_summary["single_auprc"][key] = safe_auprc(y, s)
        summary["tasks"][task] = task_summary

    y_chain = [int(r["y_chain_error"]) for r in chain_rows]
    for key in chain_rows[0].keys() if chain_rows else []:
        if key.startswith(("mean_", "max_", "top20_mean_")):
            scores = [float(r.get(key, np.nan)) for r in chain_rows]
            summary["response"][key] = {"auc": safe_auc(y_chain, scores), "auprc": safe_auprc(y_chain, scores)}
    return summary


def latent_separability(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    correct = [np.asarray(r["latent"], dtype=np.float32) for r in rows if int(r["y_state_error_or_after"]) == 0]
    wrong = [np.asarray(r["latent"], dtype=np.float32) for r in rows if int(r["y_state_error_or_after"]) == 1]
    if len(correct) < 2 or len(wrong) < 2:
        return {}
    C = np.stack(correct)
    W = np.stack(wrong)
    cc = C.mean(axis=0)
    ww = W.mean(axis=0)
    d = ww - cc
    d = d / max(float(np.linalg.norm(d)), EPS)
    scores = np.concatenate([(C @ d), (W @ d)])
    labels = np.concatenate([np.zeros(len(C), dtype=int), np.ones(len(W), dtype=int)])
    margin = float((W @ d).mean() - (C @ d).mean())
    return {
        "centroid_auc": safe_auc(labels, scores),
        "centroid_margin": margin,
        "correct_latent_var": float(np.mean(np.var(C, axis=0))),
        "wrong_latent_var": float(np.mean(np.var(W, axis=0))),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def render_console(summary: dict[str, Any]) -> str:
    lines = ["===== Latent VAE audit summary ====="]
    lines.append(f"chains {summary['n_chains']} | rows {summary['n_rows']}")
    lines.append(f"layer positions {summary['layers']['layer_positions']}")
    lines.append(f"layer ids {summary['layers']['layer_ids']}")
    for task, t in summary["tasks"].items():
        lines.append(f"\nTask {task}: rows {t['rows']} pos {t['pos']}")
        for k, v in sorted(t["single"].items(), key=lambda kv: (-(kv[1] if math.isfinite(kv[1]) else -1), kv[0]))[:10]:
            lines.append(f"  {k:<24} AUROC {v:.3f}")
    lines.append("\nResponse:")
    for k, d in sorted(summary["response"].items(), key=lambda kv: (-(kv[1]["auc"] if math.isfinite(kv[1]["auc"]) else -1), kv[0]))[:12]:
        lines.append(f"  {k:<32} AUROC {d['auc']:.3f} AUPRC {d['auprc']:.3f}")
    sep = summary.get("latent_separability", {})
    if sep:
        lines.append("\nLatent separability:")
        for k, v in sep.items():
            lines.append(f"  {k}: {v:.4f}")
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any], args: argparse.Namespace) -> str:
    lines = [
        "# Latent VAE Audit Summary",
        "",
        f"- Input: `{args.input}`",
        f"- Chains: `{summary['n_chains']}`",
        f"- Rows: `{summary['n_rows']}`",
        f"- Layer positions: `{summary['layers']['layer_positions']}`",
        f"- Layer ids: `{summary['layers']['layer_ids']}`",
        f"- Layer pool: `{summary['layers']['layer_pool']}`",
        "",
        "## Objective",
        "",
        "$$",
        "\\mathcal{L}_{\\mathrm{VAE}}=\\mathcal{L}_{\\mathrm{rec}}+\\beta\\mathcal{L}_{\\mathrm{KL}}",
        "$$",
        "",
        "Labels are not used to train the VAE; they are used for held-out latent separability evaluation.",
        "",
    ]
    for task, t in summary["tasks"].items():
        lines.extend([f"## {task}", "", "| score | AUROC | AUPRC |", "|---|---:|---:|"])
        for k, v in sorted(t["single"].items()):
            lines.append(f"| {k} | {v:.4f} | {t['single_auprc'].get(k, float('nan')):.4f} |")
        lines.append("")
    lines.extend(["## Response", "", "| score | AUROC | AUPRC |", "|---|---:|---:|"])
    for k, d in sorted(summary["response"].items()):
        lines.append(f"| {k} | {d['auc']:.4f} | {d['auprc']:.4f} |")
    lines.extend(["", "## Latent Separability", "", "```json", json.dumps(finite_json(summary.get("latent_separability", {})), ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def save_outputs(result: dict[str, Any], args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    rows = result["rows"]
    chain_rows = result["chain_rows"]
    summary = result["summary"]

    row_fields = [
        "fold",
        "chain_index",
        "chain_id",
        "problem_id",
        "step_idx",
        "gold_error_step",
        "n_steps",
        "pos",
        "step_len",
        "y_chain_error",
        "y_first_error",
        "y_pre_error_future",
        "y_state_error_or_after",
        "rec_nll",
        "kl",
        "posterior_unc",
        "decoder_unc",
        "latent_norm",
        "probe_controls",
        "probe_vae_scores",
        "probe_latent",
        "probe_controls_latent",
        "probe_all",
    ]
    chain_fields = [
        "chain_index",
        "chain_id",
        "problem_id",
        "gold_error_step",
        "n_steps",
        "y_chain_error",
        "mean_rec_nll",
        "max_rec_nll",
        "mean_kl",
        "max_kl",
        "mean_posterior_unc",
        "max_posterior_unc",
        "max_probe_latent",
        "max_probe_all",
    ]
    write_csv(out_dir / f"{tag}_latent_vae_rows.csv", rows, row_fields)
    write_csv(out_dir / f"{tag}_latent_vae_chains.csv", chain_rows, chain_fields)

    latent = np.asarray([np.asarray(r["latent"], dtype=np.float32) for r in rows], dtype=np.float32)
    latent_logvar = np.asarray([np.asarray(r["latent_logvar"], dtype=np.float32) for r in rows], dtype=np.float32)
    row_index = np.asarray([[int(r["chain_index"]), int(r["step_idx"]), int(r["y_first_error"]), int(r["y_chain_error"])] for r in rows], dtype=np.int32)
    np.savez_compressed(out_dir / f"{tag}_latent_vae_latents.npz", latent=latent, latent_logvar=latent_logvar, row_index=row_index)

    with (out_dir / f"{tag}_latent_vae_summary.json").open("w", encoding="utf-8") as f:
        json.dump(finite_json(summary), f, ensure_ascii=False, indent=2)
    with (out_dir / f"{tag}_latent_vae_summary.md").open("w", encoding="utf-8") as f:
        f.write(render_markdown(summary, args))
    print(render_console(summary))
    print(f"\nSaved: {out_dir / f'{tag}_latent_vae_summary.json'}")


def run_cv(records: list[ChainRecord], layers: list[int], args: argparse.Namespace) -> dict[str, Any]:
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch is required for latent_vae_audit.py")
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
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        projector = fit_pca_standardizer(
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
        train_ds = StepDataset(records, train_idx, features)
        test_ds = StepDataset(records, test_idx, features)
        model = train_vae(train_ds, projector.dim, args, device)
        train_rows = encode_dataset(model, train_ds, records, device, args.eval_batch_size)
        test_rows = encode_dataset(model, test_ds, records, device, args.eval_batch_size)

        for task in ["first_error", "pre_error_future", "state_error_or_after"]:
            probe_map = {
                "probe_controls": "controls",
                "probe_vae_scores": "vae_scores",
                "probe_latent": "latent",
                "probe_controls_latent": "controls+latent",
                "probe_all": "all",
            }
            selected_test, _ = task_rows(test_rows, task)
            for out_key, mode in probe_map.items():
                pred = fit_predict_probe(train_rows, test_rows, task, mode)
                for r, s in zip(selected_test, pred):
                    if task == "first_error":
                        r[out_key] = float(s)
                    else:
                        r[f"{out_key}_{task}"] = float(s)
        for r in test_rows:
            r["fold"] = int(fold_id)
        all_rows.extend(test_rows)
        if args.verbose:
            print(f"fold {fold_id}: train={len(train_idx)} test={len(test_idx)} pca_dim={projector.dim} test_rows={len(test_rows)}", flush=True)

    chain_rows = aggregate_response(all_rows)
    summary = evaluate_rows(all_rows, chain_rows)
    summary["latent_separability"] = latent_separability(all_rows)
    summary["n_chains"] = len(records)
    summary["n_rows"] = len(all_rows)
    summary["layers"] = {
        "layer_positions": [int(x) for x in layer_positions],
        "layer_ids": [int(layers[p]) if p < len(layers) else int(p) for p in layer_positions],
        "layer_pool": args.layer_pool,
    }
    return {"summary": summary, "rows": all_rows, "chain_rows": chain_rows}


def make_synthetic_records(seed: int = 0, n: int = 60, d: int = 96, layers: int = 4) -> list[ChainRecord]:
    rng = np.random.default_rng(seed)
    records = []
    v = rng.normal(size=d)
    v /= np.linalg.norm(v)
    for i in range(n):
        T = int(rng.integers(5, 9))
        is_err = i % 2 == 0
        gold = int(rng.integers(2, T - 1)) if is_err else -1
        vec = rng.normal(0, 0.35, size=(T, layers, d)).astype(np.float32)
        base = rng.normal(0, 0.6, size=d)
        lengths = rng.integers(4, 25, size=T).astype(np.float32)
        for t in range(T):
            signal = (2.0 + 0.1 * (t - gold)) if is_err and t >= gold else 0.0
            for l in range(layers):
                vec[t, l] += base + signal * v
        records.append(
            ChainRecord(
                index=i,
                chain_id=f"syn-{i}",
                problem_id=f"prob-{i//2}",
                gold_error_step=gold,
                vectors=vec,
                steps_text=[f"synthetic step {t}" for t in range(T)],
                token_lengths=lengths,
            )
        )
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="?", help="ProcessBench npz file with step vectors.")
    ap.add_argument("--output_dir", default="outputs/latent_vae")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--max_chains", type=int, default=None)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer_stride", type=int, default=1)
    ap.add_argument("--max_layers", type=int, default=8)
    ap.add_argument("--layer_pool", choices=["mean", "last", "first_last", "concat"], default="mean")
    ap.add_argument("--pca_dim", type=int, default=256)
    ap.add_argument("--max_pca_samples", type=int, default=30000)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--hidden_dims", default="256,128,64")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--warmup_epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--eval_batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--beta_max", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--enc_logvar_clip", type=float, default=8.0)
    ap.add_argument("--dec_logvar_min", type=float, default=-4.0)
    ap.add_argument("--dec_logvar_max", type=float, default=4.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if not args.selftest and not args.input:
        ap.error("input is required unless --selftest is set")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.selftest:
        records = make_synthetic_records(seed=args.seed)
        layers = list(range(records[0].vectors.shape[1]))
        args.tag = "latent_vae_selftest" if args.tag == "run" else args.tag
        args.output_dir = args.output_dir or "outputs/latent_vae_selftest"
        args.epochs = min(args.epochs, 30)
        args.batch_size = min(args.batch_size, 128)
        result = run_cv(records, layers, args)
        result["summary"]["input"] = "<synthetic>"
        result["summary"]["vector_key"] = "synthetic"
        save_outputs(result, args)
        return

    records, layers, vector_key = load_records(args.input, args.mode, args.max_chains)
    print(f"Loaded {len(records)} chains from {args.input}; vector_key={vector_key}; layers={layers}")
    result = run_cv(records, layers, args)
    result["summary"]["input"] = str(args.input)
    result["summary"]["vector_key"] = vector_key
    save_outputs(result, args)


if __name__ == "__main__":
    main()
