#!/usr/bin/env python3
"""Boundary-free token-stream geometry audit.

This script tests the deployable version of the project's directional
concentration hypothesis.  It does not require parsed reasoning steps to build
the signal.  For every generated token it computes causal sliding-window
geometry:

    u_t = h_t / ||h_t||
    R_t(W) = ||sum_i exp(-decay * (t-i)) u_i|| / sum_i exp(-decay * (t-i))
    spread_t(W) = 1 - R_t(W)

Optional step boundaries are used only for evaluation controls:

  - static step-spread baselines;
  - first-error alignment / alarm delay when gold_error_step exists.

The acceptance criterion is deliberately stricter than "does a scalar have
AUROC above chance": token-stream features must add out-of-fold signal beyond
length, entropy, and available static spread baselines under same-problem paired
ranking whenever contrastive problems are present.
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
    problem_groups,
    within_pair_auroc,
)
from premise_constraint_audit import bootstrap_within_increment, pair_rescue_report
from second_moment_dynamics_audit import group_folds, oof_scores, small_gram_eigvals


EPS = 1e-12


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - progress bars are optional
    def tqdm(iterable, **_kwargs):
        return iterable


@dataclass
class StreamRow:
    idx: int
    problem_id: int
    y_err: int
    n_tokens: int
    n_steps: int
    error_token: float
    features: Dict[str, float]
    risk_traces: Dict[str, np.ndarray]
    profile_traces: Dict[str, np.ndarray]


@dataclass
class StreamData:
    rows: List[StreamRow]
    y: np.ndarray
    problem_ids: np.ndarray
    groups: List[np.ndarray]
    feature_names: List[str]
    baseline_groups: Dict[str, List[str]]
    stream_groups: Dict[str, List[str]]
    source: str
    layer_used: int
    stream_backend: str
    stream_device: str
    policy_desc: str
    coverage: Dict[str, float]


def parse_ints(text: str) -> List[int]:
    return [int(x) for x in str(text).replace(";", ",").split(",") if x.strip()]


def scalar_str(x: Any) -> str:
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(x)


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


def bdir(x: float) -> float:
    return float(max(x, 1.0 - x)) if np.isfinite(x) else float("nan")


def safe_mean(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def safe_std(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1)) if a.size > 1 else (0.0 if a.size else float("nan"))


def finite_slope(v: np.ndarray, pos: Optional[np.ndarray] = None) -> float:
    y = np.asarray(v, dtype=np.float64)
    if pos is None:
        x = np.arange(y.size, dtype=np.float64) / max(1, y.size - 1)
    else:
        x = np.asarray(pos, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    xx = x[m] - float(x[m].mean())
    yy = y[m] - float(y[m].mean())
    den = float(np.dot(xx, xx))
    return float(np.dot(xx, yy) / den) if den > EPS else float("nan")


def hump_metrics(v: np.ndarray) -> Dict[str, float]:
    """Shape summary for "rise then fall" trajectory tests.

    A hump is deliberately a descriptive shape, not a classifier.  It is present
    when the trace has an interior peak and both early->peak and peak->late
    changes are positive after a small robust-scale threshold.
    """
    x = np.asarray(v, dtype=np.float64)
    m = np.isfinite(x)
    if m.sum() < 5:
        return {
            "hump_score": float("nan"),
            "hump_present": float("nan"),
            "hump_peak_pos": float("nan"),
            "hump_rise": float("nan"),
            "hump_fall": float("nan"),
        }
    idx = np.where(m)[0]
    vals = x[m]
    peak_local = int(np.argmax(vals))
    peak_i = int(idx[peak_local])
    peak_pos = float(peak_i / max(1, len(x) - 1))
    pos = np.arange(len(x), dtype=np.float64) / max(1, len(x) - 1)
    early = safe_mean(x[pos <= 0.25])
    late = safe_mean(x[pos >= 0.75])
    peak = float(vals[peak_local])
    rise = peak - early
    fall = peak - late
    finite = x[np.isfinite(x)]
    scale = float(np.nanstd(finite)) if finite.size > 1 else 0.0
    thresh = max(1e-6, 0.10 * scale)
    interior = 0.15 <= peak_pos <= 0.85
    present = bool(interior and rise > thresh and fall > thresh)
    return {
        "hump_score": float(min(rise, fall)),
        "hump_present": float(present),
        "hump_peak_pos": peak_pos,
        "hump_rise": float(rise),
        "hump_fall": float(fall),
    }


def robust_center_scale(v: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(v, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    if mad > EPS:
        return med, 1.4826 * mad
    sd = float(np.std(a))
    return med, sd if sd > EPS else 1.0


def causal_z(v: np.ndarray, *, warmup: int) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for t in range(max(1, warmup), len(x)):
        hist = x[:t]
        hist = hist[np.isfinite(hist)]
        if hist.size >= max(2, warmup // 2) and np.isfinite(x[t]):
            out[t] = (x[t] - float(hist.mean())) / (float(hist.std(ddof=1)) + EPS)
    return out


def positive_delta(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    if x.size >= 2:
        d = x[1:] - x[:-1]
        out[1:] = np.where(np.isfinite(d), np.maximum(d, 0.0), np.nan)
    return out


def abs_delta(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    if x.size >= 2:
        d = np.abs(x[1:] - x[:-1])
        out[1:] = np.where(np.isfinite(d), d, np.nan)
    return out


def exp_denominator(n: int, decay: float) -> float:
    if n <= 0:
        return 0.0
    if abs(decay) <= EPS:
        return float(n)
    q = math.exp(-float(decay))
    return float((1.0 - q**n) / max(1.0 - q, EPS))


def normalize_rows(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(H, dtype=np.float64)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    norms = np.linalg.norm(X, axis=1)
    ok = np.isfinite(norms) & (norms > EPS)
    U = np.full_like(X, np.nan, dtype=np.float64)
    U[ok] = X[ok] / np.maximum(norms[ok, None], EPS)
    return X, U


def sliding_resultant(U: np.ndarray, *, window: int, decay: float, min_window: int) -> np.ndarray:
    X = np.asarray(U, dtype=np.float64)
    T, D = X.shape
    out = np.full(T, np.nan, dtype=np.float64)
    if T == 0:
        return out
    q = math.exp(-float(decay))
    qW = q ** int(window)
    S = np.zeros(D, dtype=np.float64)
    for t in range(T):
        row = X[t]
        if np.isfinite(row).all():
            S = q * S + row
        else:
            S = q * S
        if t >= window and np.isfinite(X[t - window]).all():
            S -= qW * X[t - window]
        n = min(t + 1, int(window))
        if n >= min_window:
            out[t] = float(np.linalg.norm(S) / max(exp_denominator(n, decay), EPS))
    return np.clip(out, 0.0, 1.0)


def sliding_resultants_multi_cpu(
    U: np.ndarray,
    *,
    windows: Sequence[int],
    decay: float,
    min_window: int,
) -> Dict[int, np.ndarray]:
    """Compute several causal resultants in one token pass.

    The first implementation called `sliding_resultant` once per window.  That
    repeats the response scan and repeatedly casts 4096-dim hidden rows to
    float64.  This version keeps all window accumulators together and uses
    float32, which is enough for a diagnostic score and much closer to the
    stored hidden precision.
    """
    X = np.asarray(U, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return {int(W): np.full(0, np.nan, dtype=np.float64) for W in windows}
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Wv = np.asarray([int(W) for W in windows], dtype=np.int64)
    Wv = Wv[Wv > 0]
    if Wv.size == 0:
        return {}
    T, D = X.shape
    q = float(math.exp(-float(decay)))
    qW = np.asarray([q ** int(W) for W in Wv], dtype=np.float32)
    denoms = {
        int(W): np.asarray([exp_denominator(min(t + 1, int(W)), decay) for t in range(T)], dtype=np.float32)
        for W in Wv
    }
    S = np.zeros((len(Wv), D), dtype=np.float32)
    outs = {int(W): np.full(T, np.nan, dtype=np.float64) for W in Wv}
    min_by_w = np.asarray([min(int(min_window), int(W)) for W in Wv], dtype=np.int64)
    for t in range(T):
        row = X[t]
        S *= q
        S += row[None, :]
        for wi, W in enumerate(Wv):
            if t >= int(W):
                S[wi] -= qW[wi] * X[t - int(W)]
            n = min(t + 1, int(W))
            if n >= int(min_by_w[wi]):
                outs[int(W)][t] = float(np.linalg.norm(S[wi]) / max(float(denoms[int(W)][t]), EPS))
    for W in list(outs):
        outs[W] = np.clip(outs[W], 0.0, 1.0)
    return outs


def resolve_stream_backend(args: argparse.Namespace):
    requested = str(getattr(args, "stream_backend", "auto")).lower()
    if requested == "cpu":
        return "cpu", None, None
    try:
        import torch
    except Exception as exc:
        if requested == "auto":
            return "cpu", None, None
        raise SystemExit(f"--stream_backend={requested} requires torch: {exc}") from exc
    device_arg = str(getattr(args, "stream_device", "") or "")
    if requested == "auto":
        if torch.cuda.is_available():
            return "torch", torch, torch.device(device_arg or "cuda")
        return "cpu", None, None
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--stream_backend=cuda requested, but torch.cuda.is_available() is false")
        return "torch", torch, torch.device(device_arg or "cuda")
    if requested == "torch":
        return "torch", torch, torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))
    raise SystemExit(f"unknown --stream_backend={requested!r}")


def sliding_resultants_multi_torch(
    U: np.ndarray,
    *,
    windows: Sequence[int],
    decay: float,
    min_window: int,
    torch_mod,
    device,
) -> Dict[int, np.ndarray]:
    """GPU/torch grouped-convolution implementation of causal resultants.

    Input is one chain `(T, D)`.  We use grouped `conv1d` over the token axis:
    each hidden dimension is convolved independently with the same exponential
    kernel, then the channel vector norm gives the weighted resultant numerator.
    """
    import torch.nn.functional as F

    X = np.asarray(U, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return {int(W): np.full(0, np.nan, dtype=np.float64) for W in windows}
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    T, D = X.shape
    if T == 0 or D == 0:
        return {int(W): np.full(T, np.nan, dtype=np.float64) for W in windows}
    x = torch_mod.as_tensor(X.T[None, :, :], device=device)
    outs: Dict[int, np.ndarray] = {}
    q = float(math.exp(-float(decay)))
    with torch_mod.no_grad():
        for W0 in windows:
            W = int(W0)
            if W <= 0:
                continue
            # conv1d is cross-correlation.  With left padding, output t sees
            # original tokens t-W+1..t, so the earliest token gets q^(W-1).
            kernel = (q ** np.arange(W - 1, -1, -1, dtype=np.float32)).astype(np.float32)
            weight = torch_mod.as_tensor(kernel[None, None, :], device=device).repeat(D, 1, 1)
            padded = F.pad(x, (W - 1, 0))
            conv = F.conv1d(padded, weight, groups=D)
            num = torch_mod.linalg.norm(conv[0].transpose(0, 1), dim=1).detach().cpu().numpy()
            den = np.asarray([exp_denominator(min(t + 1, W), decay) for t in range(T)], dtype=np.float64)
            out = num.astype(np.float64) / np.maximum(den, EPS)
            out[np.arange(T) + 1 < min(int(min_window), W)] = np.nan
            outs[W] = np.clip(out, 0.0, 1.0)
    return outs


def sliding_resultants_multi(
    U: np.ndarray,
    *,
    windows: Sequence[int],
    decay: float,
    min_window: int,
    backend_kind: str,
    torch_mod,
    device,
) -> Dict[int, np.ndarray]:
    if backend_kind == "torch":
        return sliding_resultants_multi_torch(
            U,
            windows=windows,
            decay=decay,
            min_window=min_window,
            torch_mod=torch_mod,
            device=device,
        )
    return sliding_resultants_multi_cpu(U, windows=windows, decay=decay, min_window=min_window)


def alpha_from_eigvals(vals: np.ndarray, *, alpha_k: int) -> float:
    x = np.asarray(vals, dtype=np.float64)
    x = np.clip(x[np.isfinite(x) & (x > EPS)], 0.0, None)
    if x.size < 3 or float(x.sum()) <= EPS:
        return float("nan")
    p = x / float(x.sum())
    k = min(int(alpha_k), p.size)
    if k < 3:
        return float("nan")
    xs = np.log(np.arange(1, k + 1, dtype=np.float64))
    ys = np.log(p[:k] + EPS)
    slope, _ = np.polyfit(xs, ys, 1)
    return float(-slope)


def spectrum_metrics_from_eigvals(vals: np.ndarray, *, alpha_k: int) -> Dict[str, float]:
    x = np.asarray(vals, dtype=np.float64)
    x = np.clip(x[np.isfinite(x) & (x > EPS)], 0.0, None)
    out = {
        "alpha": float("nan"),
        "eff_rank": float("nan"),
        "entropy": float("nan"),
        "lam1": float("nan"),
        "stable_rank": float("nan"),
    }
    if x.size == 0 or float(x.sum()) <= EPS:
        return out
    p = x / float(x.sum())
    ent = float(-np.sum(p * np.log(p + EPS)))
    out["entropy"] = ent
    out["eff_rank"] = float(np.exp(ent))
    out["lam1"] = float(p[0])
    out["stable_rank"] = float(1.0 / max(p[0], EPS))
    out["alpha"] = alpha_from_eigvals(x, alpha_k=alpha_k)
    return out


def sliding_spectral_metrics(
    X: np.ndarray,
    U: np.ndarray,
    *,
    window: int,
    decay: float,
    min_window: int,
    alpha_k: int,
    stride: int,
    use_unit: bool,
) -> Dict[str, np.ndarray]:
    base = U if use_unit else X
    T = base.shape[0]
    out = {
        "alpha": np.full(T, np.nan, dtype=np.float64),
        "eff_rank": np.full(T, np.nan, dtype=np.float64),
        "entropy": np.full(T, np.nan, dtype=np.float64),
        "lam1": np.full(T, np.nan, dtype=np.float64),
        "stable_rank": np.full(T, np.nan, dtype=np.float64),
    }
    stride = max(1, int(stride))
    for t in range(max(0, min_window - 1), T, stride):
        lo = max(0, t - int(window) + 1)
        H = np.asarray(base[lo : t + 1], dtype=np.float32)
        ok = np.isfinite(H).all(axis=1)
        H = H[ok]
        if H.shape[0] < min_window:
            continue
        ages = np.arange(H.shape[0] - 1, -1, -1, dtype=np.float32)
        w = np.exp(-float(decay) * ages)
        w /= max(float(w.sum()), EPS)
        Y = H * np.sqrt(w[:, None])
        metrics = spectrum_metrics_from_eigvals(small_gram_eigvals(Y), alpha_k=alpha_k)
        for k, v in metrics.items():
            out[k][t] = v
    return out


def summarize_trace(prefix: str, v: np.ndarray, out: Dict[str, float]) -> None:
    x = np.asarray(v, dtype=np.float64)
    finite = x[np.isfinite(x)]
    out[f"{prefix}_mean"] = float(finite.mean()) if finite.size else float("nan")
    out[f"{prefix}_std"] = safe_std(finite)
    out[f"{prefix}_max"] = float(finite.max()) if finite.size else float("nan")
    out[f"{prefix}_min"] = float(finite.min()) if finite.size else float("nan")
    out[f"{prefix}_last"] = float(finite[-1]) if finite.size else float("nan")
    pos = np.arange(x.size, dtype=np.float64) / max(1, x.size - 1)
    early = x[pos <= 0.4]
    late = x[pos >= 0.6]
    out[f"{prefix}_early"] = safe_mean(early)
    out[f"{prefix}_late"] = safe_mean(late)
    out[f"{prefix}_amplitude"] = out[f"{prefix}_late"] - out[f"{prefix}_early"]
    out[f"{prefix}_slope"] = finite_slope(x, pos)
    out[f"{prefix}_late_slope"] = finite_slope(x[pos >= 0.6], pos[pos >= 0.6])
    dz = abs_delta(causal_z(x, warmup=4))
    out[f"{prefix}_volatility"] = safe_mean(dz)
    z = causal_z(x, warmup=4)
    if np.isfinite(z).any():
        out[f"{prefix}_czmax"] = float(np.nanmax(z))
        out[f"{prefix}_czmin"] = float(np.nanmin(z))
        out[f"{prefix}_czabsmax"] = float(np.nanmax(np.abs(z)))
    else:
        out[f"{prefix}_czmax"] = float("nan")
        out[f"{prefix}_czmin"] = float("nan")
        out[f"{prefix}_czabsmax"] = float("nan")
    if np.isfinite(x).any():
        imax = int(np.nanargmax(x))
        imin = int(np.nanargmin(x))
        out[f"{prefix}_argmax_pos"] = float(imax / max(1, x.size - 1))
        out[f"{prefix}_argmin_pos"] = float(imin / max(1, x.size - 1))
    else:
        out[f"{prefix}_argmax_pos"] = float("nan")
        out[f"{prefix}_argmin_pos"] = float("nan")
    for k, v in hump_metrics(x).items():
        out[f"{prefix}_{k}"] = v


def summarize_baseline_trace(prefix: str, v: Optional[np.ndarray], out: Dict[str, float]) -> None:
    if v is None:
        return
    x = np.asarray(v, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return
    summarize_trace(prefix, x, out)


def step_lengths_from_ranges(ranges: Any, n_tokens: int) -> Tuple[np.ndarray, float]:
    if ranges is None:
        return np.array([n_tokens], dtype=int), float("nan")
    R = np.asarray(ranges, dtype=int)
    if R.ndim != 2 or R.shape[1] < 2 or R.shape[0] == 0:
        return np.array([n_tokens], dtype=int), float("nan")
    base = int(R[0, 0])
    lengths: List[int] = []
    for lo0, hi0 in R:
        lo = max(0, int(lo0) - base)
        hi = min(n_tokens - 1, int(hi0) - base)
        if hi >= lo:
            lengths.append(int(hi - lo + 1))
    return np.asarray(lengths or [n_tokens], dtype=int), float(base)


def error_token_from_gold(gold: int, ranges: Any, n_tokens: int) -> float:
    if gold < 0 or ranges is None:
        return float("nan")
    R = np.asarray(ranges, dtype=int)
    if R.ndim != 2 or R.shape[0] == 0 or gold >= R.shape[0]:
        return float("nan")
    base = int(R[0, 0])
    return float(np.clip(int(R[gold, 0]) - base, 0, max(0, n_tokens - 1)))


def static_step_spread(H: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    X, U = normalize_rows(H)
    vals: List[float] = []
    cur = 0
    for s in np.asarray(lengths, dtype=int):
        ss = int(max(0, s))
        if ss <= 0:
            continue
        chunk = U[cur : cur + ss]
        cur += ss
        ok = np.isfinite(chunk).all(axis=1)
        if ok.sum() == 0:
            vals.append(float("nan"))
        else:
            vals.append(float(1.0 - np.linalg.norm(np.mean(chunk[ok], axis=0))))
    return np.asarray(vals, dtype=np.float64)


def per_step_trace_from_token_trace(token_trace: Optional[np.ndarray], lengths: np.ndarray) -> Optional[np.ndarray]:
    if token_trace is None:
        return None
    x = np.asarray(token_trace, dtype=np.float64).reshape(-1)
    vals: List[float] = []
    cur = 0
    for s in np.asarray(lengths, dtype=int):
        ss = int(max(0, s))
        if ss <= 0:
            continue
        vals.append(safe_mean(x[cur : cur + ss]))
        cur += ss
    return np.asarray(vals, dtype=np.float64) if vals else None


def resolve_labels(data: np.lib.npyio.NpzFile, policy: str) -> Tuple[np.ndarray, np.ndarray, str]:
    n = len(data["problem_ids"]) if "problem_ids" in data.files else len(data["gold_error_step"])
    if policy == "gold_error_step":
        if "gold_error_step" not in data.files:
            raise SystemExit("--policy=gold_error_step requires gold_error_step")
        return (data["gold_error_step"].astype(int) >= 0).astype(int), np.ones(n, dtype=bool), "gold_error_step >= 0"
    if "is_correct" not in data.files:
        if "gold_error_step" in data.files:
            return (data["gold_error_step"].astype(int) >= 0).astype(int), np.ones(n, dtype=bool), "gold_error_step >= 0"
        raise SystemExit("npz needs is_correct or gold_error_step labels")
    if policy == "answer":
        return (data["is_correct"].astype(int) == 0).astype(int), np.ones(n, dtype=bool), "answer incorrect"
    if policy == "strict":
        key = "is_correct_strict" if "is_correct_strict" in data.files else "is_correct"
        return (data[key].astype(int) == 0).astype(int), np.ones(n, dtype=bool), "strict incorrect"
    if policy == "answer_format_ok":
        if "format_ok" in data.files:
            mask = data["format_ok"].astype(bool)
            return (data["is_correct"].astype(int) == 0).astype(int), mask, "answer incorrect among format-ok samples"
        return (data["is_correct"].astype(int) == 0).astype(int), np.ones(n, dtype=bool), "answer incorrect (format_ok unavailable)"
    raise ValueError(policy)


def select_layer(layers: Sequence[int], requested: int, nearest: bool) -> Tuple[int, int]:
    vals = [int(x) for x in layers]
    if not vals:
        return 0, int(requested)
    if requested in vals:
        i = vals.index(requested)
        return i, vals[i]
    if nearest:
        i = int(np.argmin([abs(v - requested) for v in vals]))
        return i, vals[i]
    raise SystemExit(f"requested layer {requested} not in stored layers {vals}; pass --nearest_layer")


def hidden_dir_candidates(npz_path: str, hdir: str) -> List[str]:
    if not hdir:
        return []
    out = []
    if os.path.isabs(hdir):
        out.append(hdir)
    else:
        out.append(os.path.abspath(hdir))
        out.append(os.path.abspath(os.path.join(os.path.dirname(npz_path), hdir)))
        out.append(os.path.abspath(os.path.join(os.getcwd(), hdir)))
    seen = set()
    clean = []
    for p in out:
        if p not in seen:
            clean.append(p)
            seen.add(p)
    return clean


def resolve_hidden_file(data: np.lib.npyio.NpzFile, npz_path: str, args: argparse.Namespace, idx: int) -> Optional[str]:
    hdir = str(getattr(args, "hidden_dir", "") or "")
    if not hdir and "hidden_dir" in data.files:
        hdir = scalar_str(data["hidden_dir"])
    dirs = hidden_dir_candidates(npz_path, hdir)
    if not dirs:
        return None
    names: List[str] = []
    if "hidden_files" in data.files:
        names.append(str(data["hidden_files"][idx]))
    if "hidden_ids" in data.files:
        names.append(str(data["hidden_ids"][idx]) + ".npy")
    names.extend([f"{idx}.npy", f"gsm8k-{idx}.npy", f"chain-{idx}.npy"])
    for d in dirs:
        for name in names:
            if os.path.isabs(name) and os.path.exists(name):
                return name
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


def source_info(data: np.lib.npyio.NpzFile, path: str, args: argparse.Namespace) -> Tuple[str, int, int]:
    if (getattr(args, "hidden_dir", "") or "hidden_dir" in data.files or "hidden_files" in data.files) and "hidden_layers" in data.files:
        layers = [int(x) for x in data["hidden_layers"]]
        li, layer = select_layer(layers, args.layer, args.nearest_layer)
        return "full_hidden", li, layer
    if "sv_clouds" in data.files and "cloud_sizes" in data.files:
        layers = [int(x) for x in data["cloud_layers"]] if "cloud_layers" in data.files else [args.layer]
        li, layer = select_layer(layers, args.layer, args.nearest_layer)
        return "sv_clouds", li, layer
    if "respcloud" in data.files:
        layers = [int(x) for x in data["cloud_store_layers"]] if "cloud_store_layers" in data.files else [args.layer]
        li, layer = select_layer(layers, args.layer, args.nearest_layer)
        return "respcloud", li, layer
    raise SystemExit("need full hidden shards, sv_clouds+cloud_sizes, or respcloud")


def load_token_matrix(
    data: np.lib.npyio.NpzFile,
    npz_path: str,
    args: argparse.Namespace,
    *,
    idx: int,
    source: str,
    layer_i: int,
) -> Optional[np.ndarray]:
    if source == "full_hidden":
        fn = resolve_hidden_file(data, npz_path, args, idx)
        if fn is None:
            return None
        H = np.load(fn, mmap_mode="r" if not args.no_mmap else None)
        A = np.asarray(H)
        if A.ndim == 3:
            if layer_i >= A.shape[1]:
                return None
            return np.asarray(A[:, layer_i, :], dtype=np.float64)
        if A.ndim == 2:
            return np.asarray(A, dtype=np.float64)
        return None
    if source == "sv_clouds":
        C = np.asarray(data["sv_clouds"][idx], dtype=np.float64)
        if C.ndim != 3 or layer_i >= C.shape[1]:
            return None
        return np.asarray(C[:, layer_i, :], dtype=np.float64)
    if source == "respcloud":
        C = np.asarray(data["respcloud"][idx], dtype=np.float64)
        if C.ndim != 3 or layer_i >= C.shape[1]:
            return None
        return np.asarray(C[:, layer_i, :], dtype=np.float64)
    return None


def chain_lengths(
    data: np.lib.npyio.NpzFile,
    idx: int,
    n_tokens: int,
    source: str,
) -> Tuple[np.ndarray, Any]:
    if "step_token_ranges" in data.files:
        R = data["step_token_ranges"][idx]
        lens, _base = step_lengths_from_ranges(R, n_tokens)
        return lens, R
    if source == "sv_clouds" and "cloud_sizes" in data.files:
        lens = np.asarray(data["cloud_sizes"][idx], dtype=int).reshape(-1)
        return lens, None
    return np.asarray([n_tokens], dtype=int), None


def optional_token_entropy(data: np.lib.npyio.NpzFile, idx: int, lengths: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    token_entropy = None
    token_commit = None
    if "tok_U_D" in data.files and data["tok_U_D"][idx] is not None:
        token_entropy = np.asarray(data["tok_U_D"][idx], dtype=np.float64).reshape(-1)
    elif "sv_out_entropy" in data.files and data["sv_out_entropy"][idx] is not None:
        step_v = np.asarray(data["sv_out_entropy"][idx], dtype=np.float64).reshape(-1)
        if step_v.size:
            token_entropy = np.repeat(step_v[: len(lengths)], np.maximum(1, lengths[: step_v.size]))
    if "tok_U_C" in data.files and data["tok_U_C"][idx] is not None:
        token_commit = np.asarray(data["tok_U_C"][idx], dtype=np.float64).reshape(-1)
    elif "sv_out_committal" in data.files and data["sv_out_committal"][idx] is not None:
        step_v = np.asarray(data["sv_out_committal"][idx], dtype=np.float64).reshape(-1)
        if step_v.size:
            token_commit = np.repeat(step_v[: len(lengths)], np.maximum(1, lengths[: step_v.size]))
    return token_entropy, token_commit


def build_token_stream_features(
    H: np.ndarray,
    *,
    windows: Sequence[int],
    alpha_windows: Sequence[int],
    decay: float,
    min_window: int,
    alpha_k: int,
    alpha_stride: int,
    no_alpha: bool,
    stream_backend_kind: str,
    torch_mod,
    stream_device,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    X, U = normalize_rows(H)
    feats: Dict[str, float] = {}
    risk: Dict[str, np.ndarray] = {}
    profiles: Dict[str, np.ndarray] = {}
    resultants = sliding_resultants_multi(
        U,
        windows=windows,
        decay=decay,
        min_window=min_window,
        backend_kind=stream_backend_kind,
        torch_mod=torch_mod,
        device=stream_device,
    )
    for W in windows:
        if int(W) not in resultants:
            continue
        R = resultants[int(W)]
        spread = 1.0 - R
        summarize_trace(f"stream_resultant_w{W}", R, feats)
        summarize_trace(f"stream_spread_w{W}", spread, feats)
        d_spread = positive_delta(spread)
        z_spread = causal_z(spread, warmup=max(4, min_window // 2))
        summarize_trace(f"stream_dspread_w{W}", d_spread, feats)
        summarize_trace(f"stream_zspread_w{W}", z_spread, feats)
        risk[f"spread_w{W}"] = spread
        risk[f"zspread_w{W}"] = z_spread
        risk[f"dspread_w{W}"] = d_spread
        profiles[f"resultant_w{W}"] = R
        profiles[f"spread_w{W}"] = spread
    if not no_alpha:
        for W in alpha_windows:
            if int(W) < 3:
                continue
            raw_metrics = sliding_spectral_metrics(
                X,
                U,
                window=int(W),
                decay=decay,
                min_window=min(min_window, int(W)),
                alpha_k=alpha_k,
                stride=alpha_stride,
                use_unit=False,
            )
            unit_metrics = sliding_spectral_metrics(
                X,
                U,
                window=int(W),
                decay=decay,
                min_window=min(min_window, int(W)),
                alpha_k=alpha_k,
                stride=alpha_stride,
                use_unit=True,
            )
            raw = raw_metrics["alpha"]
            unit = unit_metrics["alpha"]
            summarize_trace(f"stream_alpha_raw_w{W}", raw, feats)
            summarize_trace(f"stream_alpha_unit_w{W}", unit, feats)
            for metric in ("eff_rank", "entropy", "lam1", "stable_rank"):
                summarize_trace(f"stream_{metric}_raw_w{W}", raw_metrics[metric], feats)
                summarize_trace(f"stream_{metric}_unit_w{W}", unit_metrics[metric], feats)
            draw = abs_delta(raw)
            dunit = abs_delta(unit)
            summarize_trace(f"stream_dalpha_raw_w{W}", draw, feats)
            summarize_trace(f"stream_dalpha_unit_w{W}", dunit, feats)
            risk[f"alpha_raw_phase_w{W}"] = draw
            risk[f"alpha_unit_phase_w{W}"] = dunit
            risk[f"alpha_raw_low_w{W}"] = -raw
            risk[f"alpha_unit_low_w{W}"] = -unit
            profiles[f"alpha_raw_w{W}"] = raw
            profiles[f"alpha_unit_w{W}"] = unit
            profiles[f"eff_rank_raw_w{W}"] = raw_metrics["eff_rank"]
            profiles[f"eff_rank_unit_w{W}"] = unit_metrics["eff_rank"]
            profiles[f"lam1_raw_w{W}"] = raw_metrics["lam1"]
            profiles[f"lam1_unit_w{W}"] = unit_metrics["lam1"]
    return feats, risk, profiles


def load_stream_data(path: str, args: argparse.Namespace) -> StreamData:
    data = np.load(path, allow_pickle=True)
    if "problem_ids" in data.files:
        pids_all = data["problem_ids"].astype(int)
    else:
        n = len(data["gold_error_step"])
        pids_all = np.arange(n, dtype=int)
    y_err_all, mask_all, desc = resolve_labels(data, args.policy)
    source, layer_i, layer_used = source_info(data, path, args)
    stream_backend_kind, torch_mod, stream_device = resolve_stream_backend(args)

    if args.max_problems:
        groups0 = problem_groups(pids_all, y_err_all, mask_all, args.min_per_class)
        keep_p = [int(pids_all[g[0]]) for g in groups0[: int(args.max_problems)]]
        mask_all = mask_all & np.isin(pids_all, np.asarray(keep_p, dtype=int))

    rows: List[StreamRow] = []
    total = len(pids_all)
    iterator = range(total)
    if not args.no_progress:
        iterator = tqdm(iterator, desc="token-stream", unit="chain", dynamic_ncols=True)
    windows = parse_ints(args.windows)
    alpha_windows = parse_ints(args.alpha_windows)
    for idx in iterator:
        if not mask_all[idx]:
            continue
        H = load_token_matrix(data, path, args, idx=idx, source=source, layer_i=layer_i)
        if H is None or H.ndim != 2 or H.shape[0] < args.min_tokens:
            continue
        if args.max_tokens and H.shape[0] > args.max_tokens:
            H = H[: int(args.max_tokens)]
        n_tokens = int(H.shape[0])
        lengths, ranges = chain_lengths(data, idx, n_tokens, source)
        lengths = lengths[lengths > 0]
        if lengths.size == 0:
            lengths = np.asarray([n_tokens], dtype=int)
        gold = int(data["gold_error_step"][idx]) if "gold_error_step" in data.files else -1
        err_tok = error_token_from_gold(gold, ranges, n_tokens) if ranges is not None else float("nan")

        feats: Dict[str, float] = {
            "base_n_tokens": float(n_tokens),
            "base_log_tokens": math.log1p(n_tokens),
            "base_n_steps": float(len(lengths)),
            "base_mean_step_tokens": float(np.mean(lengths)),
            "base_max_step_tokens": float(np.max(lengths)),
            "base_log_mean_step_tokens": math.log1p(float(np.mean(lengths))),
        }
        stream_feats, risk, profiles = build_token_stream_features(
            H,
            windows=windows,
            alpha_windows=alpha_windows,
            decay=args.decay,
            min_window=args.min_window,
            alpha_k=args.alpha_k,
            alpha_stride=args.alpha_stride,
            no_alpha=args.no_alpha,
            stream_backend_kind=stream_backend_kind,
            torch_mod=torch_mod,
            stream_device=stream_device,
        )
        feats.update(stream_feats)

        static_spread = static_step_spread(H, lengths)
        summarize_baseline_trace("base_static_spread", static_spread, feats)
        tok_ent, tok_com = optional_token_entropy(data, idx, lengths)
        summarize_baseline_trace("base_entropy", tok_ent, feats)
        summarize_baseline_trace("base_committal", tok_com, feats)
        summarize_baseline_trace("base_step_entropy", per_step_trace_from_token_trace(tok_ent, lengths), feats)
        summarize_baseline_trace("base_step_committal", per_step_trace_from_token_trace(tok_com, lengths), feats)

        rows.append(
            StreamRow(
                idx=int(idx),
                problem_id=int(pids_all[idx]),
                y_err=int(y_err_all[idx]),
                n_tokens=n_tokens,
                n_steps=int(len(lengths)),
                error_token=err_tok,
                features=feats,
                risk_traces=risk,
                profile_traces=profiles,
            )
        )
    data.close()
    if len(rows) < 20:
        raise SystemExit("not enough usable token streams")
    y = np.asarray([r.y_err for r in rows], dtype=int)
    pids = np.asarray([r.problem_id for r in rows], dtype=int)
    all_names = sorted({k for r in rows for k in r.features})
    coverage = {k: float(np.mean([np.isfinite(r.features.get(k, float("nan"))) for r in rows])) for k in all_names}
    feature_names = [k for k in all_names if coverage[k] >= args.min_feature_coverage]

    groups = problem_groups(pids, y, np.ones(len(y), dtype=bool), args.min_per_class)
    length_base = [k for k in feature_names if k.startswith("base_") and any(x in k for x in ("tokens", "steps"))]
    entropy_base = [k for k in feature_names if k.startswith(("base_entropy", "base_committal", "base_step_entropy", "base_step_committal"))]
    static_base = [k for k in feature_names if k.startswith("base_static_spread")]
    baseline_groups = {
        "length": sorted(set(length_base)),
        "length_entropy": sorted(set(length_base + entropy_base)),
        "length_entropy_static": sorted(set(length_base + entropy_base + static_base)),
    }
    baseline_groups = {k: v for k, v in baseline_groups.items() if v}
    kappa_names = [k for k in feature_names if k.startswith(("stream_spread", "stream_resultant", "stream_zspread"))]
    alpha_names = [k for k in feature_names if "stream_alpha" in k]
    spectrum_names = [
        k
        for k in feature_names
        if any(tok in k for tok in ("stream_alpha", "stream_eff_rank", "stream_entropy", "stream_lam1", "stream_stable_rank"))
    ]
    dyn_names = [
        k
        for k in feature_names
        if k.startswith("stream_")
        and any(s in k for s in ("dspread", "dalpha", "slope", "amplitude", "volatility", "czmax", "czmin", "czabsmax"))
    ]
    stream_groups = {
        "token_stream_kappa": sorted(set(kappa_names)),
        "token_stream_alpha": sorted(set(alpha_names)),
        "token_stream_spectrum": sorted(set(spectrum_names)),
        "token_stream_dynamics": sorted(set(dyn_names)),
        "token_stream_all": sorted(set(kappa_names + spectrum_names + dyn_names)),
    }
    stream_groups = {k: v for k, v in stream_groups.items() if v}
    return StreamData(
        rows=rows,
        y=y,
        problem_ids=pids,
        groups=groups,
        feature_names=feature_names,
        baseline_groups=baseline_groups,
        stream_groups=stream_groups,
        source=source,
        layer_used=int(layer_used),
        stream_backend=stream_backend_kind,
        stream_device=str(stream_device) if stream_device is not None else "cpu",
        policy_desc=desc,
        coverage=coverage,
    )


def feature_matrix(rows: Sequence[StreamRow], names: Sequence[str]) -> np.ndarray:
    X = np.full((len(rows), len(names)), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        for j, name in enumerate(names):
            X[i, j] = r.features.get(name, float("nan"))
    return X


def eval_values(name: str, vals: np.ndarray, sd: StreamData) -> Dict[str, Any]:
    w, pairs = within_pair_auroc(sd.groups, vals, sd.y) if sd.groups else (float("nan"), 0)
    c = auroc(vals, sd.y)
    return {
        "score": name,
        "n": int(np.isfinite(vals).sum()),
        "cross_auroc_error_high": c,
        "cross_best_direction": bdir(c),
        "within_pair_auroc_error_high": w,
        "within_best_direction": bdir(w),
        "within_pairs": int(pairs),
        "error": descriptive(vals[(sd.y == 1) & np.isfinite(vals)]),
        "correct": descriptive(vals[(sd.y == 0) & np.isfinite(vals)]),
    }


def eval_oof_group(sd: StreamData, name: str, names: Sequence[str], *, folds: int, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    X = feature_matrix(sd.rows, names)
    vals = oof_scores(X, sd.y, sd.problem_ids, folds=folds, seed=seed)
    row = eval_values(name, vals, sd)
    row["n_features"] = int(len(names))
    row["features"] = list(names)
    return vals, row


def best_single_features(sd: StreamData, *, limit: int = 40) -> Dict[str, Dict[str, Any]]:
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for name in sd.feature_names:
        vals = feature_matrix(sd.rows, [name]).reshape(-1)
        row = eval_values(name, vals, sd)
        if row["n"] > 0:
            rows.append((name, row))
    rows.sort(
        key=lambda kv: (
            np.nan_to_num(kv[1]["within_best_direction"], nan=-1.0),
            np.nan_to_num(kv[1]["cross_best_direction"], nan=-1.0),
        ),
        reverse=True,
    )
    return {k: v for k, v in rows[:limit]}


def trajectory_shape_report(sd: StreamData, *, max_traces: int = 24) -> Dict[str, Any]:
    trace_names = sorted({name for r in sd.rows for name in r.profile_traces})
    preferred = [
        n
        for n in trace_names
        if n.startswith(("resultant_w", "spread_w", "eff_rank_raw_w", "eff_rank_unit_w", "alpha_raw_w", "alpha_unit_w"))
    ]
    names = preferred[:max_traces]
    out: Dict[str, Any] = {}
    for name in names:
        rows: List[Tuple[int, Dict[str, float]]] = []
        for r in sd.rows:
            if name not in r.profile_traces:
                continue
            hm = hump_metrics(r.profile_traces[name])
            if np.isfinite(hm["hump_score"]):
                rows.append((r.y_err, hm))
        if not rows:
            continue
        y = np.asarray([yy for yy, _hm in rows], dtype=int)
        present = np.asarray([hm["hump_present"] for _yy, hm in rows], dtype=np.float64)
        score = np.asarray([hm["hump_score"] for _yy, hm in rows], dtype=np.float64)
        peak = np.asarray([hm["hump_peak_pos"] for _yy, hm in rows], dtype=np.float64)
        row = {
            "n": int(len(rows)),
            "hump_rate": float(np.nanmean(present)),
            "hump_score": descriptive(score),
            "peak_pos": descriptive(peak),
        }
        if y.size == score.size and y.size:
            row["error"] = {
                "hump_rate": float(np.nanmean(present[y == 1])) if np.any(y == 1) else float("nan"),
                "hump_score": descriptive(score[y == 1]),
                "peak_pos": descriptive(peak[y == 1]),
            }
            row["correct"] = {
                "hump_rate": float(np.nanmean(present[y == 0])) if np.any(y == 0) else float("nan"),
                "hump_score": descriptive(score[y == 0]),
                "peak_pos": descriptive(peak[y == 0]),
            }
            row["hump_score_cross_auroc_error_high"] = auroc(score, y)
        out[name] = row
    return out


def compact_trace(v: np.ndarray, *, max_points: int) -> List[Optional[float]]:
    x = np.asarray(v, dtype=np.float64).reshape(-1)
    if max_points and x.size > max_points:
        idx = np.linspace(0, x.size - 1, int(max_points)).round().astype(int)
        x = x[idx]
    out: List[Optional[float]] = []
    for val in x:
        out.append(float(val) if np.isfinite(val) else None)
    return out


def build_profile_rows(sd: StreamData, *, max_points: int, max_traces: int) -> List[Dict[str, Any]]:
    preferred = [
        n
        for n in sorted({name for r in sd.rows for name in r.profile_traces})
        if n.startswith(("resultant_w", "spread_w", "eff_rank_raw_w", "eff_rank_unit_w", "alpha_raw_w", "alpha_unit_w"))
    ]
    names = preferred[: int(max_traces)] if max_traces else preferred
    rows: List[Dict[str, Any]] = []
    for r in sd.rows:
        traces = {
            name: compact_trace(r.profile_traces[name], max_points=max_points)
            for name in names
            if name in r.profile_traces
        }
        shapes = {name: hump_metrics(r.profile_traces[name]) for name in traces if name in r.profile_traces}
        rows.append(
            {
                "idx": int(r.idx),
                "problem_id": int(r.problem_id),
                "y_err": int(r.y_err),
                "n_tokens": int(r.n_tokens),
                "n_steps": int(r.n_steps),
                "error_token": float(r.error_token) if np.isfinite(r.error_token) else None,
                "traces": traces,
                "hump": finite_json(shapes),
            }
        )
    return rows


def oof_alarm_metrics(
    sd: StreamData,
    *,
    trace_name: str,
    target_fpr: float,
    folds: int,
    seed: int,
    warmup_tokens: int,
) -> Dict[str, Any]:
    alarm = np.full(len(sd.rows), np.nan, dtype=np.float64)
    threshold_used = np.full(len(sd.rows), np.nan, dtype=np.float64)
    for tr, te in group_folds(sd.problem_ids, folds, seed):
        correct_train = [i for i in tr if sd.y[i] == 0 and trace_name in sd.rows[i].risk_traces]
        if not correct_train:
            continue
        maxima = []
        for i in correct_train:
            v = np.asarray(sd.rows[i].risk_traces[trace_name], dtype=np.float64)
            if v.size <= warmup_tokens:
                continue
            tail = v[warmup_tokens:]
            if np.isfinite(tail).any():
                maxima.append(float(np.nanmax(tail)))
        if not maxima:
            continue
        thr = float(np.quantile(np.asarray(maxima), 1.0 - target_fpr))
        for i in te:
            if trace_name not in sd.rows[i].risk_traces:
                continue
            v = np.asarray(sd.rows[i].risk_traces[trace_name], dtype=np.float64)
            if v.size <= warmup_tokens:
                continue
            hit = np.where(np.isfinite(v) & (np.arange(v.size) >= warmup_tokens) & (v > thr))[0]
            threshold_used[i] = thr
            if hit.size:
                alarm[i] = float(hit[0])
    correct = sd.y == 0
    error = sd.y == 1
    fp = correct & np.isfinite(alarm)
    hit = error & np.isfinite(alarm)
    err_tok = np.asarray([r.error_token for r in sd.rows], dtype=np.float64)
    has_error_time = error & np.isfinite(err_tok)
    aligned = hit & np.isfinite(err_tok)
    delay = alarm[aligned] - err_tok[aligned]
    pre = aligned & (alarm <= err_tok)
    endpoint = np.asarray(
        [alarm[i] / max(1, sd.rows[i].n_tokens - 1) if np.isfinite(alarm[i]) else np.nan for i in range(len(sd.rows))],
        dtype=np.float64,
    )
    return {
        "trace": trace_name,
        "target_chain_fpr": float(target_fpr),
        "observed_correct_fpr": float(fp.sum() / max(1, correct.sum())),
        "error_recall": float(hit.sum() / max(1, error.sum())),
        "n_error_with_gold_time": int(has_error_time.sum()),
        "gold_time_recall": float(aligned.sum() / max(1, has_error_time.sum())),
        "pre_error_or_onset_recall": float(pre.sum() / max(1, has_error_time.sum())),
        "delay_tokens": descriptive(delay),
        "alarm_endpoint_fraction": descriptive(endpoint[np.isfinite(endpoint)]),
        "threshold": descriptive(threshold_used[np.isfinite(threshold_used)]),
    }


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    sd = load_stream_data(path, args)
    baseline_scores: Dict[str, np.ndarray] = {}
    baseline_rows: Dict[str, Dict[str, Any]] = {}
    for bname, names in sd.baseline_groups.items():
        vals, row = eval_oof_group(sd, f"OOF:baseline_{bname}", names, folds=args.folds, seed=args.seed)
        baseline_scores[bname] = vals
        baseline_rows[bname] = row
    if "length_entropy_static" in baseline_scores:
        primary_base = "length_entropy_static"
    elif "length_entropy" in baseline_scores:
        primary_base = "length_entropy"
    else:
        primary_base = "length"
    base_score = baseline_scores[primary_base]

    stream_rows: Dict[str, Dict[str, Any]] = {}
    stream_scores: Dict[str, np.ndarray] = {}
    for gname, names in sd.stream_groups.items():
        all_names = sorted(set(sd.baseline_groups[primary_base] + names))
        vals, row = eval_oof_group(sd, f"OOF:baseline+{gname}", all_names, folds=args.folds, seed=args.seed)
        if sd.groups:
            row["increment_over_primary_baseline"] = bootstrap_within_increment(
                vals,
                base_score,
                groups=sd.groups,
                y_err=sd.y,
                n_boot=args.bootstrap,
                seed=args.seed,
            )
            row["baseline_miss_rescue"] = pair_rescue_report(vals, base_score, groups=sd.groups, y_err=sd.y)
        else:
            row["increment_over_primary_baseline"] = {
                "point": row["cross_auroc_error_high"] - baseline_rows[primary_base]["cross_auroc_error_high"],
                "lo": None,
                "hi": None,
                "sig": False,
            }
            row["baseline_miss_rescue"] = {}
        stream_rows[gname] = row
        stream_scores[gname] = vals

    single_rows = best_single_features(sd)
    best_group_name = max(
        stream_rows,
        key=lambda k: np.nan_to_num(
            stream_rows[k].get("increment_over_primary_baseline", {}).get("point", -1e9),
            nan=-1e9,
        ),
    )
    best_group = stream_rows[best_group_name]
    inc = best_group["increment_over_primary_baseline"]
    alarm_rows: Dict[str, Any] = {}
    trace_names = sorted({name for r in sd.rows for name in r.risk_traces})
    for name in trace_names:
        if any(key in name for key in ("spread", "alpha")):
            alarm_rows[name] = oof_alarm_metrics(
                sd,
                trace_name=name,
                target_fpr=args.alarm_fpr,
                folds=args.folds,
                seed=args.seed,
                warmup_tokens=args.alarm_warmup_tokens,
            )
    ranked_alarm = sorted(
        alarm_rows.items(),
        key=lambda kv: (kv[1]["gold_time_recall"], kv[1]["error_recall"], -kv[1]["observed_correct_fpr"]),
        reverse=True,
    )
    return {
        "meta": {
            "input": os.path.abspath(path),
            "basename": os.path.basename(path),
            "policy": args.policy,
            "policy_description": sd.policy_desc,
            "source": sd.source,
            "layer": int(sd.layer_used),
            "stream_backend": sd.stream_backend,
            "stream_device": sd.stream_device,
            "n_samples": int(len(sd.rows)),
            "n_error": int(sd.y.sum()),
            "n_correct": int(len(sd.y) - sd.y.sum()),
            "n_contrastive_problems": int(len(sd.groups)),
            "n_pairs": int(sum(int(sd.y[g].sum()) * int((sd.y[g] == 0).sum()) for g in sd.groups)),
            "windows": parse_ints(args.windows),
            "alpha_windows": [] if args.no_alpha else parse_ints(args.alpha_windows),
            "decay": float(args.decay),
            "alpha_stride": int(args.alpha_stride),
            "min_window": int(args.min_window),
            "coverage": sd.coverage,
            "baseline_groups": sd.baseline_groups,
            "stream_groups": {k: {"n": len(v), "features": v} for k, v in sd.stream_groups.items()},
            "method_notes": {
                "runtime_assumption": "Signals are computed from causal token windows; step boundaries are not needed at runtime.",
                "novelty_boundary": "The primitive resultant is not claimed as novel. The tested contribution is the boundary-free online protocol with length/difficulty controls and fixed-FPR delay metrics.",
                "confound_control": "Primary increment is over length, entropy, and available static step-spread baselines.",
                "anti_degradation": "If token-stream features cannot beat the baseline, this branch should be treated as a weak physiological marker, not renamed as a new detector.",
            },
        },
        "headline": {
            "primary_baseline": primary_base,
            "primary_baseline_row": baseline_rows[primary_base],
            "best_stream_group": best_group_name,
            "best_stream_group_row": best_group,
            "decision": {
                "passes_increment_gate": bool(
                    inc.get("point") is not None
                    and np.isfinite(inc.get("point", float("nan")))
                    and inc["point"] >= args.min_increment
                    and (inc.get("lo") is None or inc["lo"] > 0)
                ),
                "min_increment": float(args.min_increment),
                "interpretation": (
                    "token-stream geometry adds over length/entropy/static controls"
                    if inc.get("point") is not None
                    and np.isfinite(inc.get("point", float("nan")))
                    and inc["point"] >= args.min_increment
                    and (inc.get("lo") is None or inc["lo"] > 0)
                    else "no robust token-stream increment over length/entropy/static controls"
                ),
            },
            "best_alarm": ranked_alarm[0][1] if ranked_alarm else None,
        },
        "baseline_scores": baseline_rows,
        "stream_scores": stream_rows,
        "single_features": single_rows,
        "alarm_scores": {k: v for k, v in ranked_alarm},
        "trajectory_shape": trajectory_shape_report(sd, max_traces=args.shape_max_traces),
        "profiles": build_profile_rows(
            sd,
            max_points=args.profile_max_points,
            max_traces=args.profile_max_traces,
        )
        if args.save_profiles
        else [],
    }


def write_markdown(path: str, res: Mapping[str, Any]) -> None:
    def fmt3(x: Any, signed: bool = False) -> str:
        try:
            if x is None:
                return ""
            v = float(x)
            if not math.isfinite(v):
                return ""
            return f"{v:+.3f}" if signed else f"{v:.3f}"
        except Exception:
            return ""

    meta = res["meta"]
    head = res["headline"]
    inc = head["best_stream_group_row"].get("increment_over_primary_baseline", {})
    ci = ""
    if inc.get("lo") is not None:
        ci = f" CI [{inc.get('lo'):+.3f}, {inc.get('hi'):+.3f}]"
    lines = [
        f"# Token-Stream Geometry Audit: `{meta['basename']}`",
        "",
        "## Headline",
        "",
        f"- Source: `{meta['source']}` at layer `{meta['layer']}`; stream backend `{meta.get('stream_backend', 'cpu')}` on `{meta.get('stream_device', 'cpu')}`.",
        f"- Samples: `{meta['n_samples']}`; errors: `{meta['n_error']}`; contrastive problems: `{meta['n_contrastive_problems']}`.",
        f"- Primary baseline: `{head['primary_baseline']}` = within `{fmt3(head['primary_baseline_row']['within_pair_auroc_error_high'])}`, cross `{fmt3(head['primary_baseline_row']['cross_auroc_error_high'])}`.",
        f"- Best stream group: `{head['best_stream_group']}` = within `{fmt3(head['best_stream_group_row']['within_pair_auroc_error_high'])}`, cross `{fmt3(head['best_stream_group_row']['cross_auroc_error_high'])}`.",
        f"- Increment over primary baseline: `{fmt3(inc.get('point'), signed=True)}`{ci}.",
        f"- Decision: **{head['decision']['interpretation']}**.",
        "",
        "## Baselines",
        "",
        "| baseline | within AUROC | cross AUROC | features |",
        "|---|---:|---:|---:|",
    ]
    for name, row in res["baseline_scores"].items():
        lines.append(
            f"| `{name}` | {fmt3(row['within_pair_auroc_error_high'])} | {fmt3(row['cross_auroc_error_high'])} | {row['n_features']} |"
        )
    lines += [
        "",
        "## Stream Groups",
        "",
        "| group | within AUROC | cross AUROC | increment | CI | rescue | features |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = sorted(
        res["stream_scores"].items(),
        key=lambda kv: np.nan_to_num(kv[1].get("increment_over_primary_baseline", {}).get("point", -1e9), nan=-1e9),
        reverse=True,
    )
    for name, row in rows:
        inc2 = row.get("increment_over_primary_baseline", {})
        rescue = row.get("baseline_miss_rescue", {})
        ci2 = "" if inc2.get("lo") is None else f"[{inc2.get('lo'):+.3f}, {inc2.get('hi'):+.3f}]"
        rescue_rate = rescue.get("rescue_rate_among_baseline_misses", float("nan"))
        lines.append(
            f"| `{name}` | {fmt3(row['within_pair_auroc_error_high'])} | {fmt3(row['cross_auroc_error_high'])} | "
            f"{fmt3(inc2.get('point', float('nan')), signed=True)} | {ci2} | {fmt3(rescue_rate)} | {row['n_features']} |"
        )
    lines += [
        "",
        "## Top Single Features",
        "",
        "| feature | within best-dir | cross best-dir | raw within |",
        "|---|---:|---:|---:|",
    ]
    for name, row in list(res["single_features"].items())[:24]:
        lines.append(
            f"| `{name}` | {fmt3(row['within_best_direction'])} | {fmt3(row['cross_best_direction'])} | {fmt3(row['within_pair_auroc_error_high'])} |"
        )
    lines += [
        "",
        "## Online Alarm",
        "",
        "| trace | target FPR | observed FPR | error recall | gold-time recall | pre/onset recall | median delay | median endpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in list(res["alarm_scores"].items())[:16]:
        delay = row.get("delay_tokens", {}).get("median", float("nan"))
        endpoint = row.get("alarm_endpoint_fraction", {}).get("median", float("nan"))
        lines.append(
            f"| `{name}` | {fmt3(row['target_chain_fpr'])} | {fmt3(row['observed_correct_fpr'])} | "
            f"{fmt3(row['error_recall'])} | {fmt3(row['gold_time_recall'])} | {fmt3(row['pre_error_or_onset_recall'])} | "
            f"{fmt3(delay)} | {fmt3(endpoint)} |"
        )
    if res.get("trajectory_shape"):
        lines += [
            "",
            "## Trajectory Shape",
            "",
            "| trace | hump rate | error hump | correct hump | median peak | error AUROC by hump score |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, row in list(res["trajectory_shape"].items())[:24]:
            err = row.get("error", {})
            cor = row.get("correct", {})
            peak = row.get("peak_pos", {}).get("median", float("nan"))
            lines.append(
                f"| `{name}` | {fmt3(row.get('hump_rate'))} | {fmt3(err.get('hump_rate'))} | "
                f"{fmt3(cor.get('hump_rate'))} | {fmt3(peak)} | {fmt3(row.get('hump_score_cross_auroc_error_high'))} |"
            )
    if res.get("profile_file"):
        lines += [
            "",
            "## Profiles",
            "",
            f"- Per-chain trajectory profiles: `{res['profile_file']}`",
            f"- Profile rows: `{res.get('profile_rows', 0)}`",
        ]
    lines += [
        "",
        "## Interpretation Guardrails",
        "",
        "- The sliding resultant is not presented as a novel mathematical object.",
        "- The audited claim is deployability: causal token windows, no parsed-step runtime input, and explicit length/entropy/static controls.",
        "- If the increment gate fails, this result should close the token-stream geometry branch rather than motivate a renamed static scalar.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_outputs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    jpath = os.path.join(output_dir, stem + ".json")
    mpath = os.path.join(output_dir, stem + ".md")
    profiles = list(res.get("profiles") or [])
    clean_obj = dict(res)
    clean_obj.pop("profiles", None)
    if profiles:
        ppath = os.path.join(output_dir, stem + ".profiles.jsonl")
        with open(ppath, "w", encoding="utf-8") as f:
            for row in profiles:
                f.write(json.dumps(finite_json(row), ensure_ascii=False) + "\n")
        clean_obj["profile_file"] = ppath
        clean_obj["profile_rows"] = len(profiles)
    clean = finite_json(clean_obj)
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    write_markdown(mpath, clean)
    return jpath, mpath


def print_result(res: Mapping[str, Any]) -> None:
    meta = res["meta"]
    head = res["headline"]
    base = head["primary_baseline_row"]
    best = head["best_stream_group_row"]
    inc = best["increment_over_primary_baseline"]
    ci = "" if inc.get("lo") is None else f" CI [{inc['lo']:+.3f}, {inc['hi']:+.3f}]"
    print(f"\n===== token-stream geometry | {meta['basename']} =====")
    print(
        f"samples {meta['n_samples']} | err {meta['n_error']} | problems {meta['n_contrastive_problems']} | "
        f"source {meta['source']} L{meta['layer']} | backend {meta.get('stream_backend', 'cpu')}:{meta.get('stream_device', 'cpu')}"
    )
    print(
        f"baseline {head['primary_baseline']} within {base['within_pair_auroc_error_high']:.3f} "
        f"cross {base['cross_auroc_error_high']:.3f}"
    )
    print(
        f"best stream {head['best_stream_group']} within {best['within_pair_auroc_error_high']:.3f} "
        f"cross {best['cross_auroc_error_high']:.3f} | inc {inc.get('point'):+.3f}{ci}"
    )
    print(f"decision: {head['decision']['interpretation']}")
    if head.get("best_alarm"):
        a = head["best_alarm"]
        delay = a.get("delay_tokens", {}).get("median", float("nan"))
        endpoint = a.get("alarm_endpoint_fraction", {}).get("median", float("nan"))
        print(
            f"best alarm {a['trace']} | FPR {a['observed_correct_fpr']:.3f} | recall {a['error_recall']:.3f} | "
            f"gold-time {a['gold_time_recall']:.3f} | median delay {delay:.2f} | endpoint {endpoint:.3f}"
        )
    print("\nStream group increments:")
    for name, row in sorted(
        res["stream_scores"].items(),
        key=lambda kv: np.nan_to_num(kv[1].get("increment_over_primary_baseline", {}).get("point", -1e9), nan=-1e9),
        reverse=True,
    ):
        inc2 = row["increment_over_primary_baseline"]
        ci2 = "" if inc2.get("lo") is None else f" [{inc2['lo']:+.3f},{inc2['hi']:+.3f}]"
        print(
            f"  {name:24s} within {row['within_pair_auroc_error_high']:.3f} "
            f"cross {row['cross_auroc_error_high']:.3f} inc {inc2.get('point'):+.3f}{ci2}"
        )
    print("\nTop single features:")
    for name, row in list(res["single_features"].items())[:12]:
        print(
            f"  {name:42s} within-best {row['within_best_direction']:.3f} "
            f"cross-best {row['cross_best_direction']:.3f}"
        )
    if res.get("trajectory_shape"):
        print("\nTrajectory shape:")
        for name, row in list(res["trajectory_shape"].items())[:8]:
            err = row.get("error", {})
            cor = row.get("correct", {})
            peak = row.get("peak_pos", {}).get("median", float("nan"))
            print(
                f"  {name:28s} hump {row.get('hump_rate', float('nan')):.3f} "
                f"err {err.get('hump_rate', float('nan')):.3f} cor {cor.get('hump_rate', float('nan')):.3f} "
                f"peak {peak:.3f}"
            )
    if res.get("profile_file"):
        print(f"\nprofiles saved: {res['profile_file']}")


def _unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    return x / max(float(np.linalg.norm(x)), EPS)


def make_selftest(
    path: str,
    *,
    seed: int = 0,
    n_problems: int = 18,
    samples_per_problem: int = 6,
    dim: int = 48,
) -> None:
    rng = np.random.default_rng(seed)
    problem_ids: List[int] = []
    is_correct: List[int] = []
    fmt: List[bool] = []
    clouds: List[np.ndarray] = []
    sizes_all: List[np.ndarray] = []
    ranges_all: List[np.ndarray] = []
    gold_error: List[int] = []
    entropy_all: List[np.ndarray] = []
    for p in range(n_problems):
        base_len = int(rng.integers(6, 11))
        for s in range(samples_per_problem):
            err = s % 3 == 0
            T = base_len + int(rng.integers(-1, 2)) + (1 if err else 0)
            T = max(5, T)
            lens = rng.integers(5, 10, size=T)
            g = int(rng.integers(max(2, T // 2), T - 1)) if err else -1
            center = _unit(rng.normal(size=dim))
            H_steps: List[np.ndarray] = []
            step_entropy: List[float] = []
            for t in range(T):
                n = int(lens[t])
                local = _unit(0.92 * center + 0.08 * rng.normal(size=dim))
                if err and t >= g:
                    # Keep length and entropy only mildly shifted.  The
                    # discriminative event is a causal loss of directional
                    # agreement in the generated token stream.
                    mix = 0.42 if t == g else 0.55
                    rows = []
                    for _ in range(n):
                        off = _unit(rng.normal(size=dim))
                        rows.append(_unit(mix * local + (1.0 - mix) * off + 0.03 * rng.normal(size=dim)))
                    H = np.asarray(rows, dtype=np.float32)
                    step_entropy.append(float(0.35 + 0.03 * rng.normal()))
                else:
                    H = np.asarray([_unit(local + 0.08 * rng.normal(size=dim)) for _ in range(n)], dtype=np.float32)
                    step_entropy.append(float(0.34 + 0.03 * rng.normal()))
                H_steps.append(H)
            Hcat = np.concatenate(H_steps, axis=0)[:, None, :]
            lo = np.cumsum(np.r_[0, lens[:-1]])
            hi = lo + lens - 1
            ranges = np.stack([lo, hi], axis=1).astype(np.int32)
            problem_ids.append(p)
            is_correct.append(0 if err else 1)
            fmt.append(True)
            clouds.append(Hcat.astype(np.float32))
            sizes_all.append(lens.astype(np.int32))
            ranges_all.append(ranges)
            gold_error.append(g)
            entropy_all.append(np.asarray(step_entropy, dtype=np.float32))
    obj = lambda xs: np.asarray(xs, dtype=object)
    np.savez_compressed(
        path,
        problem_ids=np.asarray(problem_ids, dtype=np.int32),
        is_correct=np.asarray(is_correct, dtype=np.int8),
        is_correct_strict=np.asarray(is_correct, dtype=np.int8),
        format_ok=np.asarray(fmt, dtype=bool),
        gold_error_step=np.asarray(gold_error, dtype=np.int32),
        step_token_ranges=obj(ranges_all),
        sv_clouds=obj(clouds),
        cloud_sizes=obj(sizes_all),
        cloud_layers=np.asarray([16], dtype=np.int32),
        sv_out_entropy=obj(entropy_all),
    )


def assert_selftest(res: Mapping[str, Any]) -> None:
    h = res["headline"]
    best = h["best_stream_group_row"]
    base = h["primary_baseline_row"]
    inc = h["best_stream_group_row"]["increment_over_primary_baseline"]
    base_saturated = (
        np.isfinite(base.get("within_pair_auroc_error_high", float("nan")))
        and base["within_pair_auroc_error_high"] >= 0.98
    )
    stream_strong = (
        np.isfinite(best.get("within_pair_auroc_error_high", float("nan")))
        and best["within_pair_auroc_error_high"] >= 0.90
    )
    if not base_saturated and (inc.get("point") is None or inc["point"] < 0.08):
        raise AssertionError("selftest failed: token-stream signal did not add over controls")
    if not stream_strong:
        raise AssertionError("selftest failed: token-stream group did not recover the injected signal")
    alarm = h.get("best_alarm") or {}
    if alarm.get("gold_time_recall", 0.0) < 0.50:
        raise AssertionError("selftest failed: online alarm did not localize enough error onsets")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="")
    ap.add_argument("--policy", default="answer_format_ok", choices=["answer", "strict", "answer_format_ok", "gold_error_step"])
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--nearest_layer", action="store_true")
    ap.add_argument("--hidden_dir", default="", help="override hidden shard directory for full_hidden npz files")
    ap.add_argument("--no_mmap", action="store_true")
    ap.add_argument("--stream_backend", default="auto", choices=["auto", "cpu", "torch", "cuda"])
    ap.add_argument("--stream_device", default="", help="torch device for stream resultant, e.g. cuda, cuda:0, or cpu")
    ap.add_argument("--windows", default="8,16,32,64")
    ap.add_argument("--alpha_windows", default="16,32,64")
    ap.add_argument("--decay", type=float, default=0.08)
    ap.add_argument("--min_window", type=int, default=6)
    ap.add_argument("--min_tokens", type=int, default=12)
    ap.add_argument("--alpha_k", type=int, default=12)
    ap.add_argument("--alpha_stride", type=int, default=4)
    ap.add_argument("--no_alpha", action="store_true")
    ap.add_argument("--min_feature_coverage", type=float, default=0.70)
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--alarm_fpr", type=float, default=0.05)
    ap.add_argument("--alarm_warmup_tokens", type=int, default=8)
    ap.add_argument("--min_increment", type=float, default=0.02)
    ap.add_argument("--max_problems", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=0)
    ap.add_argument("--shape_max_traces", type=int, default=24)
    ap.add_argument("--save_profiles", action="store_true", help="write per-chain trajectory traces to *.profiles.jsonl")
    ap.add_argument("--profile_max_points", type=int, default=256, help="downsample each saved trace to this many points; 0 keeps all")
    ap.add_argument("--profile_max_traces", type=int, default=24, help="maximum trace names saved per chain; 0 keeps all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs/token_stream_geometry")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "token_stream_selftest.npz")
            make_selftest(path, seed=args.seed)
            args.no_progress = True
            args.bootstrap = min(args.bootstrap, 50)
            res = run(path, args)
            assert_selftest(res)
            print_result(res)
            print("selftest passed")
        return
    if not args.input:
        raise SystemExit("--input is required unless --selftest is passed")
    res = run(args.input, args)
    stem = os.path.splitext(os.path.basename(args.input))[0] + "_token_stream_geometry"
    write_outputs(res, args.output_dir, stem)
    print_result(res)


if __name__ == "__main__":
    main()
