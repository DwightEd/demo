#!/usr/bin/env python3
"""Same-problem audit for token-cloud Gram second-moment dynamics.

This is a stricter successor to the print-only Gram/effective-rank scripts.
It is intentionally not a generic trajectory-kernel experiment.  The main
object is the token matrix itself, not a pooled step vector:

    H_t in R^{n_t x d}
    G_t = H_t H_t^T / n_t, and centered G_t^c from H_t - mean(H_t)

Unit-row Gram features are reported only as an ablation because they discard
token-norm/radial information.  The audit asks whether these direct token-matrix
spectra add signal beyond the strong static first-moment controls that already
work in this project:

    baseline = exp-weighted cloud spread + step length + available uncertainty

The main acceptance criterion is out-of-fold same-problem paired AUROC
increment for baseline+Gram over baseline, with problem-cluster bootstrap CIs.
If the increment is absent, the script says so directly instead of wrapping the
weak signal in a larger model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from multisample_temporal_rupture_audit import (
    descriptive,
    finite_json,
    label_policy,
    paired_delta,
    problem_groups,
    within_pair_auroc,
)


EPS = 1e-12


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - progress is optional
    def tqdm(iterable, **_kwargs):
        return iterable


try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    HAVE_SKLEARN = True
except Exception:  # pragma: no cover - exercised only on minimal boxes
    HAVE_SKLEARN = False


def progress_iter(iterable, *, enabled: bool, **kwargs):
    if not enabled:
        return iterable
    return tqdm(iterable, **kwargs)


@dataclass
class ChainRow:
    idx: int
    problem_id: int
    y_err: int
    features: Dict[str, float]
    positions: Dict[str, float]


@dataclass
class AuditData:
    rows: List[ChainRow]
    y: np.ndarray
    problem_ids: np.ndarray
    groups: List[np.ndarray]
    feature_names: List[str]
    baseline_names: List[str]
    gram_groups: Dict[str, List[str]]
    policy_desc: str
    source: str
    layer_used: int
    spectral_backend: str
    spectral_device: str
    coverage: Dict[str, float]


def robust_center_scale(x: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    if mad > EPS:
        return med, 1.4826 * mad
    sd = float(np.std(a))
    return med, sd if sd > EPS else 1.0


def auroc(score: Iterable[float], y: Iterable[int]) -> float:
    s = np.asarray(score, dtype=np.float64)
    yy = np.asarray(y, dtype=int)
    m = np.isfinite(s)
    s, yy = s[m], yy[m]
    p = int((yy == 1).sum())
    n = int((yy == 0).sum())
    if p == 0 or n == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[yy == 1].sum() - p * (p + 1) / 2.0) / (p * n))


def bdir(a: float) -> float:
    return float(max(a, 1.0 - a)) if np.isfinite(a) else float("nan")


def local_groups(problem_ids: np.ndarray, y: np.ndarray, min_per_class: int) -> List[np.ndarray]:
    groups: List[np.ndarray] = []
    for p in np.unique(problem_ids):
        idx = np.where(problem_ids == p)[0]
        if np.sum(y[idx] == 1) >= min_per_class and np.sum(y[idx] == 0) >= min_per_class:
            groups.append(idx)
    return groups


def groups_to_mask(groups: Sequence[np.ndarray], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    for g in groups:
        out[np.asarray(g, dtype=int)] = True
    return out


def group_folds(groups: np.ndarray, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    uniq = np.unique(groups)
    if uniq.size < 2:
        return []
    rng = np.random.default_rng(seed)
    uniq = np.array(uniq, copy=True)
    rng.shuffle(uniq)
    k = int(min(max(2, k), uniq.size))
    fold_of = {int(g): i % k for i, g in enumerate(uniq)}
    f = np.array([fold_of[int(g)] for g in groups], dtype=int)
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def exp_weights(n: int, beta: float) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    if n == 1 or abs(beta) <= EPS:
        return np.ones(n, dtype=np.float64) / n
    pos = np.linspace(0.0, 1.0, n)
    z = beta * pos
    z -= z.max()
    w = np.exp(z)
    w /= max(float(w.sum()), EPS)
    return w


def token_rows(H: np.ndarray, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(H, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return (
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    norms = np.linalg.norm(X, axis=1)
    ok = norms > EPS
    X = X[ok]
    U = X / np.maximum(norms[ok, None], EPS)
    return X, U, exp_weights(X.shape[0], beta)


def spectrum_stats(lam: np.ndarray, prefix: str, top_k: int, alpha_k: int) -> Dict[str, float]:
    x = np.asarray(lam, dtype=np.float64)
    x = np.clip(x[np.isfinite(x) & (x > EPS)], 0.0, None)
    out: Dict[str, float] = {
        f"{prefix}_eff_rank": float("nan"),
        f"{prefix}_entropy": float("nan"),
        f"{prefix}_lam1": float("nan"),
        f"{prefix}_gap12": float("nan"),
        f"{prefix}_stable_rank": float("nan"),
        f"{prefix}_tail_auc": float("nan"),
        f"{prefix}_alpha": float("nan"),
        f"{prefix}_log_energy": float("nan"),
        f"{prefix}_logdet_mean": float("nan"),
        f"{prefix}_k50": float("nan"),
        f"{prefix}_k75": float("nan"),
        f"{prefix}_k90": float("nan"),
    }
    for k in (1, 2, 4, 8, 16):
        if k <= top_k:
            out[f"{prefix}_resid{k}"] = float("nan")
    if x.size == 0 or float(x.sum()) <= EPS:
        return out
    out[f"{prefix}_log_energy"] = float(np.log(float(x.sum()) + EPS))
    out[f"{prefix}_logdet_mean"] = float(np.mean(np.log(x + EPS)))
    p = x / float(x.sum())
    ent = float(-np.sum(p * np.log(p + EPS)))
    out[f"{prefix}_entropy"] = ent
    out[f"{prefix}_eff_rank"] = float(np.exp(ent))
    out[f"{prefix}_lam1"] = float(p[0])
    out[f"{prefix}_gap12"] = float(p[0] - (p[1] if p.size > 1 else 0.0))
    out[f"{prefix}_stable_rank"] = float(1.0 / max(p[0], EPS))
    csum = np.cumsum(p)
    tails = []
    for k in range(1, min(top_k, p.size) + 1):
        tails.append(float(max(0.0, 1.0 - csum[k - 1])))
    if tails:
        out[f"{prefix}_tail_auc"] = float(np.mean(tails))
    for k in (1, 2, 4, 8, 16):
        if k <= top_k:
            kk = min(k, p.size)
            out[f"{prefix}_resid{k}"] = float(max(0.0, 1.0 - csum[kk - 1]))
    for thr in (0.50, 0.75, 0.90):
        out[f"{prefix}_k{int(thr * 100)}"] = float(np.searchsorted(csum, thr) + 1)
    kk = min(int(alpha_k), p.size)
    if kk >= 3:
        xs = np.log(np.arange(1, kk + 1, dtype=np.float64))
        ys = np.log(p[:kk] + EPS)
        slope, _ = np.polyfit(xs, ys, 1)
        out[f"{prefix}_alpha"] = float(-slope)
    return out


def small_gram_eigvals(A: np.ndarray) -> np.ndarray:
    """Eigenvalues of A A^T, descending.

    For step token clouds n_tokens << hidden_dim, this is exactly the squared
    singular spectrum of A but avoids an expensive SVD of the tall hidden
    matrix.  The decomposition is over n_tokens x n_tokens.
    """
    X = np.asarray(A, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    G = X @ X.T
    G = 0.5 * (G + G.T)
    vals = np.linalg.eigvalsh(G)[::-1]
    return np.clip(vals, 0.0, None)


def empty_step_features(n: int, *, top_k: int, alpha_k: int) -> Dict[str, float]:
    out: Dict[str, float] = {"n_tok": float(n), "logN": math.log1p(max(n, 0))}
    out["kappa"] = float("nan")
    out["spread"] = float("nan")
    out["tok_norm_mean"] = float("nan")
    out["tok_norm_std"] = float("nan")
    out["tok_cen_trace"] = float("nan")
    out["unit_cen_trace"] = float("nan")
    for prefix in ("tok_raw", "tok_cen", "unit_raw", "unit_cen"):
        out.update(spectrum_stats(np.array([]), prefix, top_k, alpha_k))
    out["kappa_x_tok_cen_eff_rank"] = float("nan")
    out["spread_x_tok_cen_eff_rank"] = float("nan")
    return out


def assemble_step_features(
    *,
    n: int,
    norm_mean: float,
    norm_std: float,
    kappa: float,
    lam_tok_raw: np.ndarray,
    lam_tok_cen: np.ndarray,
    lam_unit_raw: np.ndarray,
    lam_unit_cen: np.ndarray,
    top_k: int,
    alpha_k: int,
) -> Dict[str, float]:
    out: Dict[str, float] = {"n_tok": float(n), "logN": math.log1p(max(n, 0))}
    out["tok_norm_mean"] = float(norm_mean)
    out["tok_norm_std"] = float(norm_std)
    out["kappa"] = float(kappa)
    out["spread"] = float(1.0 - kappa)
    out.update(spectrum_stats(lam_tok_raw, "tok_raw", top_k, alpha_k))
    out["tok_cen_trace"] = float(np.sum(lam_tok_cen))
    out.update(spectrum_stats(lam_tok_cen, "tok_cen", top_k, alpha_k))
    out.update(spectrum_stats(lam_unit_raw, "unit_raw", top_k, alpha_k))
    out["unit_cen_trace"] = float(np.sum(lam_unit_cen))
    out.update(spectrum_stats(lam_unit_cen, "unit_cen", top_k, alpha_k))
    if np.isfinite(out["tok_cen_eff_rank"]) and np.isfinite(kappa):
        out["kappa_x_tok_cen_eff_rank"] = float(kappa * out["tok_cen_eff_rank"])
        out["spread_x_tok_cen_eff_rank"] = float((1.0 - kappa) * out["tok_cen_eff_rank"])
    else:
        out["kappa_x_tok_cen_eff_rank"] = float("nan")
        out["spread_x_tok_cen_eff_rank"] = float("nan")
    return out


def step_gram_features(
    H: np.ndarray,
    *,
    beta: float,
    top_k: int,
    alpha_k: int,
    min_tokens: int,
) -> Dict[str, float]:
    X, U, w = token_rows(H, beta)
    n = int(X.shape[0])
    if n < min_tokens:
        return empty_step_features(n, top_k=top_k, alpha_k=alpha_k)

    norms = np.linalg.norm(X, axis=1)
    norm_mean = float(np.mean(norms))
    norm_std = float(np.std(norms))

    unit_mu = w @ U
    kappa = float(np.linalg.norm(unit_mu))

    # Main Geometry-of-Reason-style object: direct token hidden matrix.  The
    # sqrt(w) rows implement exp-weighted Gram H^T W H without pooling tokens.
    h_raw = X * np.sqrt(w[:, None])
    lam_tok_raw = small_gram_eigvals(h_raw)

    h_mu = w @ X
    h_cen = (X - h_mu[None, :]) * np.sqrt(w[:, None])
    lam_tok_cen = small_gram_eigvals(h_cen)

    # Direction-only ablation: useful for separating angular shape from radial
    # activation magnitude, but not the primary object.
    u_raw = U * np.sqrt(w[:, None])
    lam_unit_raw = small_gram_eigvals(u_raw)

    u_cen = (U - unit_mu[None, :]) * np.sqrt(w[:, None])
    lam_unit_cen = small_gram_eigvals(u_cen)

    return assemble_step_features(
        n=n,
        norm_mean=norm_mean,
        norm_std=norm_std,
        kappa=kappa,
        lam_tok_raw=lam_tok_raw,
        lam_tok_cen=lam_tok_cen,
        lam_unit_raw=lam_unit_raw,
        lam_unit_cen=lam_unit_cen,
        top_k=top_k,
        alpha_k=alpha_k,
    )


def resolve_spectral_backend(args: argparse.Namespace):
    requested = str(getattr(args, "spectral_backend", "auto")).lower()
    if requested == "cpu":
        return "cpu", None, None
    try:
        import torch
    except Exception as exc:
        if requested == "auto":
            return "cpu", None, None
        raise SystemExit(f"--spectral_backend={requested} requires torch: {exc}") from exc

    device_arg = str(getattr(args, "spectral_device", "") or "")
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device(device_arg or "cuda")
            return "torch", torch, device
        return "cpu", None, None
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--spectral_backend=cuda requested, but torch.cuda.is_available() is false")
        device = torch.device(device_arg or "cuda")
        return "torch", torch, device
    if requested == "torch":
        device = torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))
        return "torch", torch, device
    raise SystemExit(f"unknown --spectral_backend={requested!r}")


def torch_batched_step_features(
    steps: Sequence[np.ndarray],
    *,
    torch_mod,
    device,
    beta: float,
    top_k: int,
    alpha_k: int,
    min_tokens: int,
    batch_size: int,
) -> List[Dict[str, float]]:
    outs: List[Dict[str, float]] = []
    batch_size = max(1, int(batch_size))
    for start in range(0, len(steps), batch_size):
        batch = steps[start : start + batch_size]
        cleaned: List[np.ndarray] = []
        weights: List[np.ndarray] = []
        norm_stats: List[Tuple[float, float]] = []
        for H in batch:
            X = np.asarray(H, dtype=np.float32)
            if X.ndim != 2 or X.shape[0] == 0:
                cleaned.append(np.empty((0, 0), dtype=np.float32))
                weights.append(np.empty(0, dtype=np.float32))
                norm_stats.append((float("nan"), float("nan")))
                continue
            norms = np.linalg.norm(X, axis=1)
            ok = norms > EPS
            X = X[ok]
            if X.size:
                norms = np.linalg.norm(X, axis=1)
                norm_stats.append((float(np.mean(norms)), float(np.std(norms))))
            else:
                norm_stats.append((float("nan"), float("nan")))
            cleaned.append(X.astype(np.float32, copy=False))
            weights.append(exp_weights(X.shape[0], beta).astype(np.float32))

        max_n = max((x.shape[0] for x in cleaned), default=0)
        d = next((x.shape[1] for x in cleaned if x.ndim == 2 and x.shape[1] > 0), 0)
        if max_n < min_tokens or d == 0:
            outs.extend(empty_step_features(x.shape[0], top_k=top_k, alpha_k=alpha_k) for x in cleaned)
            continue

        Xpad = np.zeros((len(cleaned), max_n, d), dtype=np.float32)
        Wpad = np.zeros((len(cleaned), max_n), dtype=np.float32)
        for bi, X in enumerate(cleaned):
            n = X.shape[0]
            if n:
                Xpad[bi, :n, :] = X
                Wpad[bi, :n] = weights[bi]

        with torch_mod.no_grad():
            X_t = torch_mod.as_tensor(Xpad, device=device)
            W_t = torch_mod.as_tensor(Wpad, device=device)
            sqrtw = torch_mod.sqrt(torch_mod.clamp(W_t, min=0.0)).unsqueeze(-1)
            token_norm = torch_mod.linalg.norm(X_t, dim=2, keepdim=True).clamp_min(EPS)
            U_t = X_t / token_norm
            U_t = torch_mod.where(W_t.unsqueeze(-1) > 0.0, U_t, torch_mod.zeros_like(U_t))

            unit_mu = (W_t.unsqueeze(-1) * U_t).sum(dim=1)
            kappa = torch_mod.linalg.norm(unit_mu, dim=1).detach().cpu().numpy()

            h_raw = X_t * sqrtw
            G = torch_mod.bmm(h_raw, h_raw.transpose(1, 2))
            lam_tok_raw = torch_mod.linalg.eigvalsh(0.5 * (G + G.transpose(1, 2)))

            h_mu = (W_t.unsqueeze(-1) * X_t).sum(dim=1, keepdim=True)
            h_cen = (X_t - h_mu) * sqrtw
            G = torch_mod.bmm(h_cen, h_cen.transpose(1, 2))
            lam_tok_cen = torch_mod.linalg.eigvalsh(0.5 * (G + G.transpose(1, 2)))

            u_raw = U_t * sqrtw
            G = torch_mod.bmm(u_raw, u_raw.transpose(1, 2))
            lam_unit_raw = torch_mod.linalg.eigvalsh(0.5 * (G + G.transpose(1, 2)))

            u_cen = (U_t - unit_mu.unsqueeze(1)) * sqrtw
            G = torch_mod.bmm(u_cen, u_cen.transpose(1, 2))
            lam_unit_cen = torch_mod.linalg.eigvalsh(0.5 * (G + G.transpose(1, 2)))

            eig_arrays = [
                torch_mod.flip(torch_mod.clamp(v, min=0.0), dims=(1,)).detach().cpu().numpy()
                for v in (lam_tok_raw, lam_tok_cen, lam_unit_raw, lam_unit_cen)
            ]

        for bi, X in enumerate(cleaned):
            n = int(X.shape[0])
            if n < min_tokens:
                outs.append(empty_step_features(n, top_k=top_k, alpha_k=alpha_k))
                continue
            norm_mean, norm_std = norm_stats[bi]
            outs.append(
                assemble_step_features(
                    n=n,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                    kappa=float(kappa[bi]),
                    lam_tok_raw=eig_arrays[0][bi],
                    lam_tok_cen=eig_arrays[1][bi],
                    lam_unit_raw=eig_arrays[2][bi],
                    lam_unit_cen=eig_arrays[3][bi],
                    top_k=top_k,
                    alpha_k=alpha_k,
                )
            )
    return outs


def chain_gram_feature_rows(
    steps: Sequence[np.ndarray],
    *,
    backend_kind: str,
    torch_mod,
    device,
    beta: float,
    top_k: int,
    alpha_k: int,
    min_tokens: int,
    batch_size: int,
) -> List[Dict[str, float]]:
    if backend_kind == "torch":
        return torch_batched_step_features(
            steps,
            torch_mod=torch_mod,
            device=device,
            beta=beta,
            top_k=top_k,
            alpha_k=alpha_k,
            min_tokens=min_tokens,
            batch_size=batch_size,
        )
    return [
        step_gram_features(
            H,
            beta=beta,
            top_k=top_k,
            alpha_k=alpha_k,
            min_tokens=min_tokens,
        )
        for H in steps
    ]


def finite_mean(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def finite_std(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1)) if a.size > 1 else (0.0 if a.size else float("nan"))


def chain_z(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    med, sc = robust_center_scale(v)
    out = (v - med) / max(sc, EPS)
    out[~np.isfinite(out)] = np.nan
    return out


def prefix_z_max(x: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(x, dtype=np.float64)
    best = float("nan")
    best_pos = float("nan")
    for t in range(2, len(v)):
        hist = v[:t]
        hist = hist[np.isfinite(hist)]
        if hist.size < 2 or not np.isfinite(v[t]):
            continue
        z = float((v[t] - hist.mean()) / (hist.std(ddof=1) + EPS))
        if not np.isfinite(best) or z > best:
            best = z
            best_pos = float(t / max(1, len(v) - 1))
    return best, best_pos


def positive_jump_max(x: np.ndarray) -> Tuple[float, float]:
    z = chain_z(x)
    if z.size < 2:
        return float("nan"), float("nan")
    d = z[1:] - z[:-1]
    if not np.isfinite(d).any():
        return float("nan"), float("nan")
    k = int(np.nanargmax(d))
    return float(d[k]), float((k + 1) / max(1, z.size - 1))


def summarize_sequence(name: str, values: Sequence[float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    v = np.asarray(values, dtype=np.float64)
    feats = {
        f"{name}_mean": finite_mean(v),
        f"{name}_max": float(np.nanmax(v)) if np.isfinite(v).any() else float("nan"),
        f"{name}_late": finite_mean(v[int(math.floor(0.6 * len(v))) :]) if len(v) else float("nan"),
        f"{name}_std": finite_std(v),
    }
    dz = np.abs(np.diff(chain_z(v))) if len(v) >= 2 else np.array([], dtype=np.float64)
    feats[f"{name}_volatility"] = finite_mean(dz)
    jv, jp = positive_jump_max(v)
    pv, pp = prefix_z_max(v)
    feats[f"{name}_zjump_max"] = jv
    feats[f"{name}_prefix_zmax"] = pv
    pos = {
        f"{name}_zjump_max": jp,
        f"{name}_prefix_zmax": pp,
    }
    return feats, pos


def add_sequence_summaries(
    target: Dict[str, float],
    positions: Dict[str, float],
    seqs: Mapping[str, Sequence[float]],
) -> None:
    for name, vals in seqs.items():
        f, p = summarize_sequence(name, vals)
        target.update(f)
        positions.update(p)


def cloud_slices_from_sv(obj: Any, size_obj: Any, layer_i: int) -> Optional[List[np.ndarray]]:
    if obj is None or size_obj is None:
        return None
    C = np.asarray(obj, dtype=np.float64)
    sizes = np.asarray(size_obj, dtype=int).reshape(-1)
    if C.ndim != 3 or sizes.size == 0 or layer_i >= C.shape[1]:
        return None
    X = C[:, layer_i, :]
    out: List[np.ndarray] = []
    cur = 0
    for s in sizes:
        ss = int(s)
        if ss <= 0:
            continue
        out.append(X[cur : cur + ss])
        cur += ss
    return out if out else None


def cloud_slices_from_resp(obj: Any, ranges_obj: Any, layer_i: int) -> Optional[List[np.ndarray]]:
    if obj is None or ranges_obj is None:
        return None
    C = np.asarray(obj, dtype=np.float64)
    R = np.asarray(ranges_obj, dtype=int)
    if C.ndim != 3 or R.ndim != 2 or R.shape[1] < 2 or layer_i >= C.shape[1]:
        return None
    X = C[:, layer_i, :]
    a0 = int(R[0, 0])
    out: List[np.ndarray] = []
    for lo0, hi0 in R:
        lo = max(0, int(lo0) - a0)
        hi = min(len(X), int(hi0) - a0 + 1)
        if hi > lo:
            out.append(X[lo:hi])
    return out if out else None


def add_optional_step_sequences(data: np.lib.npyio.NpzFile, idx: int, T: int, seqs: Dict[str, np.ndarray]) -> None:
    for key, name in (
        ("sv_out_entropy", "out_entropy"),
        ("sv_out_committal", "out_committal"),
        ("sv_tok_entropy", "tok_entropy"),
        ("sv_tok_committal", "tok_committal"),
    ):
        if key not in data.files:
            continue
        v = np.asarray(data[key][idx], dtype=np.float64).reshape(-1)
        if v.size:
            seqs[name] = v[:T]
    for key, name in (
        ("tok_U_D", "U_D"),
        ("tok_U_C", "U_C"),
    ):
        if key not in data.files:
            continue
        arr = data[key][idx]
        if arr is None:
            continue
        v = np.asarray(arr, dtype=np.float64).reshape(-1)
        if v.size == T:
            seqs[name] = v


def select_layer(layers: Sequence[int], requested: int, nearest: bool) -> Tuple[int, int]:
    vals = [int(x) for x in layers]
    if not vals:
        raise SystemExit("no cloud layers recorded in npz")
    if requested in vals:
        i = vals.index(requested)
        return i, vals[i]
    if nearest:
        i = int(np.argmin([abs(v - requested) for v in vals]))
        return i, vals[i]
    raise SystemExit(f"requested layer {requested} not in stored cloud layers {vals}; pass --nearest_layer")


def load_audit_data(path: str, args: argparse.Namespace) -> AuditData:
    show_progress = not bool(getattr(args, "no_progress", False))
    backend_kind, torch_mod, spectral_device = resolve_spectral_backend(args)
    data = np.load(path, allow_pickle=True)
    y_err_all, mask_all, desc = label_policy(data, args.policy)
    problem_ids_all = data["problem_ids"].astype(int)

    groups0 = problem_groups(problem_ids_all, y_err_all, mask_all, args.min_per_class)
    if not groups0:
        raise SystemExit(f"policy {args.policy!r} has no contrastive same-problem groups")
    if args.max_problems:
        keep_p = np.array([int(problem_ids_all[g[0]]) for g in groups0[: int(args.max_problems)]], dtype=int)
        mask_all = mask_all & np.isin(problem_ids_all, keep_p)
        groups0 = problem_groups(problem_ids_all, y_err_all, mask_all, args.min_per_class)

    if "sv_clouds" in data.files and "cloud_sizes" in data.files:
        layers = [int(x) for x in data["cloud_layers"]] if "cloud_layers" in data.files else [args.layer]
        layer_i, layer_used = select_layer(layers, args.layer, args.nearest_layer)
        source = "sv_clouds"

        def get_steps(i: int) -> Optional[List[np.ndarray]]:
            return cloud_slices_from_sv(data["sv_clouds"][i], data["cloud_sizes"][i], layer_i)

        n_total = len(data["sv_clouds"])
    elif "respcloud" in data.files and "step_token_ranges" in data.files:
        layers = [int(x) for x in data["cloud_store_layers"]] if "cloud_store_layers" in data.files else [args.layer]
        layer_i, layer_used = select_layer(layers, args.layer, args.nearest_layer)
        source = "respcloud"

        def get_steps(i: int) -> Optional[List[np.ndarray]]:
            return cloud_slices_from_resp(data["respcloud"][i], data["step_token_ranges"][i], layer_i)

        n_total = len(data["respcloud"])
    else:
        raise SystemExit("need sv_clouds+cloud_sizes or respcloud+step_token_ranges")

    rows: List[ChainRow] = []
    contrast_mask = groups_to_mask(groups0, len(problem_ids_all))
    for i in progress_iter(
        range(n_total),
        enabled=show_progress,
        desc="token spectra",
        unit="chain",
        dynamic_ncols=True,
    ):
        if not contrast_mask[i]:
            continue
        steps = get_steps(i)
        if steps is None or len(steps) < args.min_steps:
            continue
        per_step: Dict[str, List[float]] = {}
        step_rows = chain_gram_feature_rows(
            steps,
            backend_kind=backend_kind,
            torch_mod=torch_mod,
            device=spectral_device,
            beta=args.kappa_beta,
            top_k=args.top_k,
            alpha_k=args.alpha_k,
            min_tokens=args.min_tokens,
            batch_size=getattr(args, "spectral_batch_size", 64),
        )
        for sf in step_rows:
            for k, v in sf.items():
                per_step.setdefault(k, []).append(v)
        add_optional_step_sequences(data, i, len(steps), per_step)

        feats: Dict[str, float] = {}
        pos: Dict[str, float] = {}
        add_sequence_summaries(feats, pos, per_step)
        rows.append(
            ChainRow(
                idx=int(i),
                problem_id=int(problem_ids_all[i]),
                y_err=int(y_err_all[i]),
                features=feats,
                positions=pos,
            )
        )
    if len(rows) < 20:
        raise SystemExit("not enough rows with usable token clouds")

    y = np.asarray([r.y_err for r in rows], dtype=int)
    pids = np.asarray([r.problem_id for r in rows], dtype=int)
    groups = local_groups(pids, y, args.min_per_class)
    keep = groups_to_mask(groups, len(rows))
    if not keep.all():
        rows = [r for j, r in enumerate(rows) if keep[j]]
        y = np.asarray([r.y_err for r in rows], dtype=int)
        pids = np.asarray([r.problem_id for r in rows], dtype=int)
        groups = local_groups(pids, y, args.min_per_class)

    all_feature_keys = sorted({k for r in rows for k in r.features})
    coverage = {
        k: float(np.mean([np.isfinite(r.features.get(k, float("nan"))) for r in rows]))
        for k in all_feature_keys
    }
    feature_names = [k for k, cov in coverage.items() if cov >= args.min_feature_coverage]
    if not feature_names:
        raise SystemExit("no Gram features pass coverage threshold")

    base_prefixes = (
        "spread_",
        "logN_",
        "out_entropy_",
        "out_committal_",
        "tok_entropy_",
        "tok_committal_",
        "tok_norm_",
        "U_D_",
        "U_C_",
    )
    baseline_names = [
        k
        for k in feature_names
        if k.startswith(base_prefixes)
        and any(s in k for s in ("_mean", "_max", "_late", "_std"))
        and "zjump" not in k
        and "prefix" not in k
    ]
    # Keep baseline compact so Gram increments are not hidden by many variants.
    preferred = [
        "spread_mean",
        "spread_max",
        "spread_late",
        "spread_std",
        "logN_mean",
        "logN_max",
        "out_entropy_mean",
        "out_entropy_max",
        "out_committal_mean",
        "out_committal_max",
        "tok_norm_mean",
        "tok_norm_std",
        "U_D_mean",
        "U_C_mean",
    ]
    baseline_names = [k for k in preferred if k in feature_names] + [
        k for k in baseline_names if k not in preferred
    ][: max(0, args.max_baseline_features - len([k for k in preferred if k in feature_names]))]
    if "spread_mean" not in baseline_names and "spread_max" not in baseline_names:
        raise SystemExit("cloud spread baseline unavailable; token clouds may be malformed")

    tok_raw = [k for k in feature_names if k.startswith("tok_raw_")]
    tok_cen = [k for k in feature_names if k.startswith("tok_cen_")]
    tok_scale = [k for k in feature_names if k.startswith("tok_cen_trace")]
    unit_ablation = [k for k in feature_names if k.startswith("unit_raw_") or k.startswith("unit_cen_") or k.startswith("unit_cen_trace")]
    gram_interact = [k for k in feature_names if k.startswith("kappa_x_tok_") or k.startswith("spread_x_tok_")]
    gram_tail = [
        k
        for k in tok_raw + tok_cen
        if any(x in k for x in ("resid", "tail_auc", "k50", "k75", "k90"))
    ]
    gram_level = [
        k
        for k in tok_raw + tok_cen + tok_scale + gram_interact
        if any(k.endswith(s) for s in ("_mean", "_max", "_late", "_std"))
    ]
    gram_dyn = [
        k
        for k in tok_raw + tok_cen + tok_scale + gram_interact
        if "zjump" in k or "prefix_zmax" in k or "volatility" in k
    ]
    gram_groups = {
        "token_raw_matrix": sorted(set(tok_raw)),
        "token_centered_matrix": sorted(set(tok_cen + tok_scale + gram_interact)),
        "token_spectral_tail": sorted(set(gram_tail)),
        "token_matrix_level": sorted(set(gram_level)),
        "token_matrix_dynamics": sorted(set(gram_dyn)),
        "token_matrix_all": sorted(set(tok_raw + tok_cen + tok_scale + gram_interact)),
        "unit_direction_ablation": sorted(set(unit_ablation)),
    }
    gram_groups = {k: [v for v in vals if v in feature_names] for k, vals in gram_groups.items()}
    gram_groups = {k: vals for k, vals in gram_groups.items() if vals}

    out = AuditData(
        rows=rows,
        y=y,
        problem_ids=pids,
        groups=groups,
        feature_names=feature_names,
        baseline_names=baseline_names,
        gram_groups=gram_groups,
        policy_desc=desc,
        source=source,
        layer_used=int(layer_used),
        spectral_backend=backend_kind,
        spectral_device=str(spectral_device) if spectral_device is not None else "cpu",
        coverage=coverage,
    )
    data.close()
    return out


def feature_matrix(rows: Sequence[ChainRow], names: Sequence[str]) -> np.ndarray:
    X = np.full((len(rows), len(names)), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        for j, name in enumerate(names):
            X[i, j] = r.features.get(name, float("nan"))
    return X


def fit_linear_witness(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.nanmedian(X, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    X0 = np.where(np.isfinite(X), X, center)
    med = np.median(X0, axis=0)
    mad = np.median(np.abs(X0 - med), axis=0) * 1.4826
    sd = np.std(X0, axis=0)
    scale = np.where(mad > EPS, mad, sd)
    scale = np.where(scale > EPS, scale, 1.0)
    Z = (X0 - med) / scale
    err = Z[y == 1]
    cor = Z[y == 0]
    if len(err) == 0 or len(cor) == 0:
        w = np.zeros(Z.shape[1], dtype=np.float64)
    else:
        w = err.mean(axis=0) - cor.mean(axis=0)
    return med, scale, w


def apply_linear_witness(X: np.ndarray, med: np.ndarray, scale: np.ndarray, w: np.ndarray) -> np.ndarray:
    X0 = np.where(np.isfinite(X), X, med)
    Z = (X0 - med) / np.maximum(scale, EPS)
    return Z @ w


def oof_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    scores = np.full(len(y), np.nan, dtype=np.float64)
    for fold, (tr, te) in enumerate(group_folds(groups, folds, seed)):
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr = X[tr]
        Xte = X[te]
        if HAVE_SKLEARN:
            clf = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0),
            )
            clf.fit(Xtr, y[tr])
            scores[te] = clf.predict_proba(Xte)[:, 1]
        else:
            med, scale, w = fit_linear_witness(Xtr, y[tr])
            scores[te] = apply_linear_witness(Xte, med, scale, w)
    return scores


def oof_residual_feature(
    x: np.ndarray,
    B: np.ndarray,
    y: np.ndarray,
    problem_ids: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    out = np.full(len(y), np.nan, dtype=np.float64)
    Xb = np.asarray(B, dtype=np.float64)
    xv = np.asarray(x, dtype=np.float64)
    for tr, te in group_folds(problem_ids, folds, seed):
        mtr = np.isfinite(xv[tr])
        if mtr.sum() < max(5, Xb.shape[1] + 2):
            continue
        med = np.nanmedian(Xb[tr], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Btr0 = np.where(np.isfinite(Xb[tr]), Xb[tr], med)
        Bte0 = np.where(np.isfinite(Xb[te]), Xb[te], med)
        mu = np.mean(Btr0, axis=0)
        sd = np.std(Btr0, axis=0)
        sd = np.where(sd > EPS, sd, 1.0)
        Ztr = (Btr0 - mu) / sd
        Zte = (Bte0 - mu) / sd
        A = np.column_stack([np.ones(len(tr)), Ztr])
        beta, *_ = np.linalg.lstsq(A[mtr], xv[tr][mtr], rcond=None)
        pred_tr = A @ beta
        pred_te = np.column_stack([np.ones(len(te)), Zte]) @ beta
        res_tr = xv[tr] - pred_tr
        res_te = xv[te] - pred_te
        auc_tr = auroc(res_tr, y[tr])
        sign = 1.0 if (not np.isfinite(auc_tr) or auc_tr >= 0.5) else -1.0
        out[te] = sign * res_te
    return out


def same_problem_increment_ci(
    sf: np.ndarray,
    sb: np.ndarray,
    y: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    n_boot: int,
    seed: int,
    progress: bool = False,
    desc: str = "bootstrap",
) -> Dict[str, Any]:
    point_f, pairs = within_pair_auroc(groups, sf, y)
    point_b, _ = within_pair_auroc(groups, sb, y)
    point = float(point_f - point_b) if np.isfinite(point_f) and np.isfinite(point_b) else float("nan")
    if n_boot <= 0 or not groups:
        return {"point": point, "lo": None, "hi": None, "sig": False, "pairs": int(pairs)}
    rng = np.random.default_rng(seed)
    vals = []
    boot_iter = progress_iter(
        range(int(n_boot)),
        enabled=progress and int(n_boot) >= 50,
        desc=desc,
        unit="boot",
        leave=False,
        dynamic_ncols=True,
    )
    for _ in boot_iter:
        chosen = [groups[int(j)] for j in rng.integers(0, len(groups), size=len(groups))]
        af, _ = within_pair_auroc(chosen, sf, y)
        ab, _ = within_pair_auroc(chosen, sb, y)
        if np.isfinite(af) and np.isfinite(ab):
            vals.append(float(af - ab))
    if not vals:
        return {"point": point, "lo": None, "hi": None, "sig": False, "pairs": int(pairs)}
    arr = np.asarray(vals, dtype=np.float64)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "sig": bool(lo > 0.0 or hi < 0.0),
        "pairs": int(pairs),
    }


def evaluate_score(
    name: str,
    score: np.ndarray,
    y: np.ndarray,
    groups: Sequence[np.ndarray],
    positions: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    au, pairs = within_pair_auroc(groups, score, y)
    row = {
        "name": name,
        "same_problem_paired_auroc": au,
        "same_problem_best_direction": bdir(au),
        "n_pairs": int(pairs),
        "cross_auroc": auroc(score, y),
        "cross_best_direction": bdir(auroc(score, y)),
        "paired_delta": paired_delta(groups, score, y),
        "score_error": descriptive(np.asarray(score)[y == 1]),
        "score_correct": descriptive(np.asarray(score)[y == 0]),
    }
    if positions is not None:
        p = np.asarray(positions, dtype=np.float64)
        row["argpos_error"] = descriptive(p[y == 1])
        row["argpos_correct"] = descriptive(p[y == 0])
    return row


def subset_groups(
    mask: np.ndarray,
    problem_ids: np.ndarray,
    y: np.ndarray,
    min_per_class: int,
) -> List[np.ndarray]:
    return local_groups(problem_ids[mask], y[mask], min_per_class)


def subset_eval(
    scores: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    y: np.ndarray,
    problem_ids: np.ndarray,
    min_per_class: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for sname, mask in masks.items():
        idx = np.where(mask)[0]
        if idx.size < 10:
            continue
        groups = subset_groups(mask, problem_ids, y, min_per_class)
        if not groups:
            continue
        rows = {}
        for name, score in scores.items():
            rows[name] = evaluate_score(name, np.asarray(score)[idx], y[idx], groups)
        out[sname] = {
            "n": int(idx.size),
            "n_error": int(y[idx].sum()),
            "n_correct": int(idx.size - y[idx].sum()),
            "n_problems": int(len(groups)),
            "scores": rows,
        }
    return out


def build_subset_masks(ad: AuditData) -> Dict[str, np.ndarray]:
    Xspread = feature_matrix(ad.rows, ["spread_mean"])[:, 0]
    masks: Dict[str, np.ndarray] = {"all": np.ones(len(ad.rows), dtype=bool)}
    if np.isfinite(Xspread).sum() >= 10:
        med = np.nanmedian(Xspread)
        q75 = np.nanquantile(Xspread, 0.75)
        masks["ambiguous_high_spread_q50"] = Xspread >= med
        masks["ambiguous_high_spread_q75"] = Xspread >= q75
    if "out_entropy_mean" in ad.feature_names:
        ent = feature_matrix(ad.rows, ["out_entropy_mean"])[:, 0]
        if np.isfinite(ent).sum() >= 10:
            masks["confident_low_entropy_q50"] = ent <= np.nanmedian(ent)
    rates = {}
    for p in np.unique(ad.problem_ids):
        idx = np.where(ad.problem_ids == p)[0]
        rates[int(p)] = float(ad.y[idx].mean())
    vals = np.asarray(list(rates.values()), dtype=np.float64)
    if vals.size:
        med_rate = float(np.median(vals))
        masks["hard_problem_high_error_rate"] = np.array([rates[int(p)] >= med_rate for p in ad.problem_ids], dtype=bool)
    return masks


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    show_progress = not bool(getattr(args, "no_progress", False))
    ad = load_audit_data(path, args)
    B = feature_matrix(ad.rows, ad.baseline_names)
    base_score = oof_scores(B, ad.y, ad.problem_ids, folds=args.folds, seed=args.seed)
    score_rows: Dict[str, Any] = {
        "OOF:baseline": evaluate_score("OOF:baseline", base_score, ad.y, ad.groups),
    }

    single_scores: Dict[str, np.ndarray] = {}
    single_rows: Dict[str, Any] = {}
    for name in ad.feature_names:
        x = feature_matrix(ad.rows, [name])[:, 0]
        pos = None
        if name in ad.rows[0].positions:
            pos = np.asarray([r.positions.get(name, float("nan")) for r in ad.rows], dtype=np.float64)
        single_rows[name] = evaluate_score(name, x, ad.y, ad.groups, pos)
        if name not in ad.baseline_names and (
            name.startswith("tok_raw_")
            or name.startswith("tok_cen_")
            or name.startswith("tok_cen_trace")
            or name.startswith("unit_raw_")
            or name.startswith("unit_cen_")
            or name.startswith("unit_cen_trace")
            or name.startswith("kappa_x_tok_")
            or name.startswith("spread_x_tok_")
        ):
            single_scores[name] = x

    residual_rows: Dict[str, Any] = {}
    best_residual_score: Optional[np.ndarray] = None
    best_residual_name = ""
    residual_items = list(single_scores.items())
    for j, (name, single_score) in enumerate(
        progress_iter(
            residual_items,
            enabled=show_progress,
            desc="residual Gram features",
            unit="feature",
            dynamic_ncols=True,
        )
    ):
        rscore = oof_residual_feature(
            single_score,
            B,
            ad.y,
            ad.problem_ids,
            ad.groups,
            folds=args.folds,
            seed=args.seed + 101 + j,
        )
        row = evaluate_score(f"resid_over_base:{name}", rscore, ad.y, ad.groups)
        residual_rows[name] = row
        if best_residual_score is None or row["same_problem_paired_auroc"] > residual_rows[best_residual_name]["same_problem_paired_auroc"]:
            best_residual_score = rscore
            best_residual_name = name

    group_rows: Dict[str, Any] = {}
    group_scores: Dict[str, np.ndarray] = {}
    group_items = list(ad.gram_groups.items())
    for gi, (gname, names) in enumerate(
        progress_iter(
            group_items,
            enabled=show_progress,
            desc="OOF Gram groups",
            unit="group",
            dynamic_ncols=True,
        )
    ):
        Xg = feature_matrix(ad.rows, names)
        full = oof_scores(np.column_stack([B, Xg]), ad.y, ad.problem_ids, folds=args.folds, seed=args.seed + 17 + gi)
        row = evaluate_score(f"OOF:baseline+{gname}", full, ad.y, ad.groups)
        row["n_features"] = int(len(names))
        row["increment_over_baseline"] = same_problem_increment_ci(
            full,
            base_score,
            ad.y,
            ad.groups,
            n_boot=args.bootstrap,
            seed=args.seed + 1000 + gi,
            progress=show_progress,
            desc=f"bootstrap {gname}",
        )
        group_rows[gname] = row
        group_scores[gname] = full

    primary_group_items = [
        (name, row) for name, row in group_rows.items() if not name.startswith("unit_")
    ] or list(group_rows.items())
    best_group_name, best_group_row = max(
        primary_group_items,
        key=lambda kv: np.nan_to_num(kv[1]["increment_over_baseline"]["point"], nan=-1e9),
    )
    best_static = max(
        single_rows.items(),
        key=lambda kv: np.nan_to_num(kv[1]["same_problem_best_direction"], nan=-1.0),
    )
    best_gram_single = max(
        ((n, r) for n, r in single_rows.items() if n in single_scores),
        key=lambda kv: np.nan_to_num(kv[1]["same_problem_best_direction"], nan=-1.0),
    )
    if best_residual_score is None:
        best_residual_score = np.full(len(ad.y), np.nan)
    subset_scores = {
        "OOF:baseline": base_score,
        f"OOF:baseline+{best_group_name}": group_scores[best_group_name],
        f"resid:{best_residual_name}": best_residual_score,
    }
    subset_rows = subset_eval(
        subset_scores,
        build_subset_masks(ad),
        ad.y,
        ad.problem_ids,
        args.min_per_class,
    )
    inc = best_group_row["increment_over_baseline"]
    decision = {
        "passes_increment_gate": bool(
            np.isfinite(inc["point"])
            and inc["point"] >= args.min_increment
            and (inc.get("lo") is None or inc["lo"] > 0.0)
        ),
        "min_increment": float(args.min_increment),
        "interpretation": (
            "second-moment Gram group adds over the static baseline"
            if np.isfinite(inc["point"]) and inc["point"] >= args.min_increment and (inc.get("lo") is None or inc["lo"] > 0.0)
            else "no robust OOF same-problem increment over the static baseline"
        ),
    }
    res = {
        "meta": {
            "input": os.path.abspath(path),
            "basename": os.path.basename(path),
            "policy": args.policy,
            "policy_description": ad.policy_desc,
            "source": ad.source,
            "layer": int(ad.layer_used),
            "spectral_backend": ad.spectral_backend,
            "spectral_device": ad.spectral_device,
            "kappa_pooling": f"exp(beta={args.kappa_beta:g})",
            "n_samples": int(len(ad.rows)),
            "n_error": int(ad.y.sum()),
            "n_correct": int(len(ad.y) - ad.y.sum()),
            "n_contrastive_problems": int(len(ad.groups)),
            "n_pairs": int(sum(int(ad.y[g].sum()) * int((ad.y[g] == 0).sum()) for g in ad.groups)),
            "baseline_features": ad.baseline_names,
            "gram_groups": {k: {"n": len(v), "features": v} for k, v in ad.gram_groups.items()},
            "coverage": ad.coverage,
            "notes": {
                "primary_metric": "OOF same-problem paired AUROC increment over spread/length/uncertainty baseline.",
                "kappa": "Only kappa uses L2-normalized token rows, matching the existing exp-weighted concentration baseline.",
                "token_matrix_gram": "Primary Gram features are direct raw/centered token-matrix spectra, not pooled step vectors and not row-unit spectra.",
                "unit_ablation": "Unit-row Gram features are reported as an ablation because they discard token-norm/radial information.",
                "closed_branch_rule": "If the increment gate fails, do not promote the branch to a larger latent model.",
            },
        },
        "headline": {
            "baseline_oof_same_problem_auroc": score_rows["OOF:baseline"]["same_problem_paired_auroc"],
            "best_group": best_group_name,
            "best_group_oof_same_problem_auroc": best_group_row["same_problem_paired_auroc"],
            "best_group_increment_over_baseline": best_group_row["increment_over_baseline"],
            "best_static_scalar": best_static[0],
            "best_static_scalar_same_problem_best_direction": best_static[1]["same_problem_best_direction"],
            "best_gram_scalar": best_gram_single[0],
            "best_gram_scalar_same_problem_best_direction": best_gram_single[1]["same_problem_best_direction"],
            "best_residual_feature": best_residual_name,
            "best_residual_same_problem_auroc": residual_rows[best_residual_name]["same_problem_paired_auroc"] if best_residual_name else None,
            "decision": decision,
        },
        "scores": {
            **score_rows,
            **{f"OOF:baseline+{k}": v for k, v in group_rows.items()},
        },
        "single_features": single_rows,
        "residual_features_over_baseline": residual_rows,
        "subsets": subset_rows,
    }
    return res


def write_markdown(path: str, res: Mapping[str, Any]) -> None:
    h = res["headline"]
    inc = h["best_group_increment_over_baseline"]
    ci = ""
    if inc.get("lo") is not None and inc.get("hi") is not None:
        ci = f" CI [{inc['lo']:+.3f}, {inc['hi']:+.3f}]"
    lines = [
        f"# Second-Moment Dynamics Audit: `{res['meta']['basename']}`",
        "",
        "## Headline",
        "",
        f"- Baseline OOF same-problem AUROC: `{h['baseline_oof_same_problem_auroc']:.3f}`",
        f"- Best Gram group: `{h['best_group']}` = `{h['best_group_oof_same_problem_auroc']:.3f}`",
        f"- Increment over baseline: `{inc['point']:+.3f}`{ci}",
        f"- Decision: **{h['decision']['interpretation']}**",
        f"- Best static scalar: `{h['best_static_scalar']}` = `{h['best_static_scalar_same_problem_best_direction']:.3f}`",
        f"- Best Gram scalar: `{h['best_gram_scalar']}` = `{h['best_gram_scalar_same_problem_best_direction']:.3f}`",
        f"- Best residual Gram feature: `{h['best_residual_feature']}` = `{h['best_residual_same_problem_auroc']:.3f}`",
        "",
        "## OOF Groups",
        "",
        "| score | same-problem AUROC | increment | CI | features |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = sorted(
        res["scores"].items(),
        key=lambda kv: np.nan_to_num(
            kv[1].get("increment_over_baseline", {}).get("point", kv[1]["same_problem_paired_auroc"]),
            nan=-1e9,
        ),
        reverse=True,
    )
    for name, row in rows:
        inc2 = row.get("increment_over_baseline")
        if inc2:
            inc_txt = f"{inc2['point']:+.3f}"
            ci_txt = "" if inc2.get("lo") is None else f"[{inc2['lo']:+.3f}, {inc2['hi']:+.3f}]"
        else:
            inc_txt = ""
            ci_txt = ""
        lines.append(
            f"| `{name}` | {row['same_problem_paired_auroc']:.3f} | {inc_txt} | {ci_txt} | {row.get('n_features', '')} |"
        )
    lines += [
        "",
        "## Top Single Gram Features",
        "",
        "| feature | best-dir AUROC | raw AUROC | pairs |",
        "|---|---:|---:|---:|",
    ]
    gram_rows = [
        (n, r)
        for n, r in res["single_features"].items()
        if n.startswith(("tok_raw_", "tok_cen_", "tok_cen_trace", "kappa_x_tok_", "spread_x_tok_"))
    ]
    gram_rows.sort(key=lambda kv: np.nan_to_num(kv[1]["same_problem_best_direction"], nan=-1), reverse=True)
    for name, row in gram_rows[:24]:
        lines.append(
            f"| `{name}` | {row['same_problem_best_direction']:.3f} | {row['same_problem_paired_auroc']:.3f} | {row['n_pairs']} |"
        )
    lines += [
        "",
        "## Conditional Subsets",
        "",
        "| subset | n | problems | baseline | best group | best residual |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sname, row in res["subsets"].items():
        scores = row["scores"]
        names = list(scores)
        base = scores.get("OOF:baseline", {})
        group = scores.get(f"OOF:baseline+{h['best_group']}", {})
        resid = scores.get(f"resid:{h['best_residual_feature']}", {})
        lines.append(
            f"| `{sname}` | {row['n']} | {row['n_problems']} | "
            f"{base.get('same_problem_paired_auroc', float('nan')):.3f} | "
            f"{group.get('same_problem_paired_auroc', float('nan')):.3f} | "
            f"{resid.get('same_problem_paired_auroc', float('nan')):.3f} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_outputs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    jpath = os.path.join(output_dir, f"{stem}.json")
    mpath = os.path.join(output_dir, f"{stem}.md")
    clean = finite_json(res)
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    write_markdown(mpath, clean)
    return jpath, mpath


def print_result(res: Mapping[str, Any]) -> None:
    h = res["headline"]
    meta = res["meta"]
    inc = h["best_group_increment_over_baseline"]
    ci = ""
    if inc.get("lo") is not None:
        ci = f" CI [{inc['lo']:+.3f}, {inc['hi']:+.3f}]"
    print(f"\n===== second-moment Gram dynamics | {meta['basename']} =====")
    print(
        f"samples {meta['n_samples']} | err {meta['n_error']} | "
        f"problems {meta['n_contrastive_problems']} | source {meta['source']} L{meta['layer']}"
    )
    print(
        f"baseline {h['baseline_oof_same_problem_auroc']:.3f} | "
        f"best group {h['best_group']}={h['best_group_oof_same_problem_auroc']:.3f} | "
        f"increment {inc['point']:+.3f}{ci}"
    )
    print(f"decision: {h['decision']['interpretation']}")
    print(
        f"best scalar {h['best_static_scalar']}={h['best_static_scalar_same_problem_best_direction']:.3f} | "
        f"best Gram scalar {h['best_gram_scalar']}={h['best_gram_scalar_same_problem_best_direction']:.3f} | "
        f"best residual {h['best_residual_feature']}={h['best_residual_same_problem_auroc']:.3f}"
    )
    print("\nOOF group increments:")
    for name, row in sorted(
        res["scores"].items(),
        key=lambda kv: np.nan_to_num(kv[1].get("increment_over_baseline", {}).get("point", -1e9), nan=-1e9),
        reverse=True,
    ):
        inc2 = row.get("increment_over_baseline")
        if inc2:
            ci2 = "" if inc2.get("lo") is None else f" [{inc2['lo']:+.3f},{inc2['hi']:+.3f}]"
            print(
                f"  {name:30s} AUROC {row['same_problem_paired_auroc']:.3f} "
                f"inc {inc2['point']:+.3f}{ci2}"
            )
        else:
            print(f"  {name:30s} AUROC {row['same_problem_paired_auroc']:.3f}")
    print("\nTop Gram scalars:")
    gram_rows = [
        (n, r)
        for n, r in res["single_features"].items()
        if n.startswith(("tok_raw_", "tok_cen_", "tok_cen_trace", "kappa_x_tok_", "spread_x_tok_"))
    ]
    gram_rows.sort(key=lambda kv: np.nan_to_num(kv[1]["same_problem_best_direction"], nan=-1), reverse=True)
    for name, row in gram_rows[:12]:
        print(
            f"  {name:36s} best-dir {row['same_problem_best_direction']:.3f} "
            f"raw {row['same_problem_paired_auroc']:.3f}"
        )
    if res["subsets"]:
        print("\nSubsets:")
        for sname, row in res["subsets"].items():
            base = row["scores"].get("OOF:baseline", {}).get("same_problem_paired_auroc", float("nan"))
            group = row["scores"].get(f"OOF:baseline+{h['best_group']}", {}).get("same_problem_paired_auroc", float("nan"))
            resid = row["scores"].get(f"resid:{h['best_residual_feature']}", {}).get("same_problem_paired_auroc", float("nan"))
            print(f"  {sname:32s} n={row['n']:4d} problems={row['n_problems']:3d} base {base:.3f} group {group:.3f} resid {resid:.3f}")


def _make_cloud(
    rng: np.random.Generator,
    *,
    n_tok: int,
    dim: int,
    kappa: float,
    residual_rank: int,
    noise: float,
) -> np.ndarray:
    e0 = np.zeros(dim, dtype=np.float64)
    e0[0] = 1.0
    rank = max(1, min(residual_rank, dim - 1))
    H = np.zeros((n_tok, 1, dim), dtype=np.float32)
    for i in range(n_tok):
        coeff = rng.normal(size=rank)
        coeff /= max(float(np.linalg.norm(coeff)), EPS)
        res = np.zeros(dim, dtype=np.float64)
        res[1 : 1 + rank] = coeff
        x = kappa * e0 + math.sqrt(max(0.0, 1.0 - kappa * kappa)) * res
        x += noise * rng.normal(size=dim)
        x /= max(float(np.linalg.norm(x)), EPS)
        H[i, 0, :] = x.astype(np.float32)
    return H


def make_selftest(
    path: str,
    *,
    seed: int = 0,
    n_problems: int = 18,
    samples_per_problem: int = 6,
    dim: int = 48,
) -> None:
    rng = np.random.default_rng(seed)
    sv_clouds: List[np.ndarray] = []
    sizes_all: List[np.ndarray] = []
    problem_ids: List[int] = []
    is_correct: List[int] = []
    fmt: List[bool] = []
    ent_all: List[np.ndarray] = []
    for p in range(n_problems):
        base_k = float(rng.uniform(0.47, 0.62))
        for s in range(samples_per_problem):
            err = s % 3 == 0
            T = int(rng.integers(5, 8))
            sizes = rng.integers(7, 13, size=T)
            chunks = []
            for t in range(T):
                late = t >= int(0.55 * T)
                kappa = base_k + 0.015 * rng.normal()
                if err and late:
                    # Same first moment, different residual shape: high-rank
                    # centered Gram should help; spread should not.
                    rr = 8
                else:
                    rr = 1
                chunks.append(
                    _make_cloud(
                        rng,
                        n_tok=int(sizes[t]),
                        dim=dim,
                        kappa=float(np.clip(kappa, 0.25, 0.85)),
                        residual_rank=rr,
                        noise=0.015,
                    )
                )
            sv_clouds.append(np.concatenate(chunks, axis=0))
            sizes_all.append(sizes.astype(np.int32))
            problem_ids.append(p)
            is_correct.append(0 if err else 1)
            fmt.append(True)
            ent_all.append(0.35 + 0.02 * rng.normal(size=T))
    arr_obj = lambda xs: np.asarray(xs, dtype=object)
    np.savez_compressed(
        path,
        problem_ids=np.asarray(problem_ids, dtype=np.int32),
        is_correct=np.asarray(is_correct, dtype=np.int8),
        is_correct_strict=np.asarray(is_correct, dtype=np.int8),
        format_ok=np.asarray(fmt, dtype=bool),
        sv_clouds=arr_obj(sv_clouds),
        cloud_sizes=arr_obj(sizes_all),
        cloud_layers=np.asarray([16], dtype=np.int32),
        sv_out_entropy=arr_obj(ent_all),
    )


def assert_selftest(res: Mapping[str, Any]) -> None:
    h = res["headline"]
    inc = h["best_group_increment_over_baseline"]
    if h["baseline_oof_same_problem_auroc"] >= 0.75:
        raise AssertionError("selftest baseline unexpectedly strong")
    if inc["point"] < 0.12:
        raise AssertionError("selftest Gram group did not add enough over baseline")
    if h["best_gram_scalar_same_problem_best_direction"] < 0.80:
        raise AssertionError("selftest did not recover centered Gram rank signal")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="")
    ap.add_argument("--policy", default="answer_format_ok", choices=["answer", "strict", "answer_format_ok"])
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--nearest_layer", action="store_true")
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--min_steps", type=int, default=2)
    ap.add_argument("--min_tokens", type=int, default=4)
    ap.add_argument("--min_feature_coverage", type=float, default=0.70)
    ap.add_argument("--max_baseline_features", type=int, default=12)
    ap.add_argument("--max_problems", type=int, default=0)
    ap.add_argument("--kappa_beta", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--alpha_k", type=int, default=12)
    ap.add_argument("--spectral_backend", default="auto", choices=["auto", "cpu", "torch", "cuda"])
    ap.add_argument("--spectral_device", default="", help="torch device, e.g. cuda, cuda:0, or cpu")
    ap.add_argument("--spectral_batch_size", type=int, default=64, help="steps per torch spectral batch")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--min_increment", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs/second_moment_dynamics")
    ap.add_argument("--no_progress", action="store_true", help="disable tqdm progress bars")
    ap.add_argument("--selftest", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "second_moment_selftest.npz")
            make_selftest(path, seed=args.seed)
            res = run(path, args)
            assert_selftest(res)
            print_result(res)
            print("selftest passed")
        return
    if not args.input:
        raise SystemExit("--input is required unless --selftest is passed")
    res = run(args.input, args)
    stem = os.path.splitext(os.path.basename(args.input))[0] + "_second_moment_dynamics"
    write_outputs(res, args.output_dir, stem)
    print_result(res)


if __name__ == "__main__":
    main()
