#!/usr/bin/env python3
"""Constraint Transport Geometry audit.

This script tests a stronger hidden-state geometry hypothesis than the earlier
single-scalar spread/kappa audits:

    A faithful reasoning step should be a supported transport of the prefix
    constraint state.  A reasoning error is likely when the current hidden-state
    transition introduces energy outside the subspace spanned by the question
    anchor and the already-established prefix transitions, or when the transition
    cannot be explained by the locally transported previous reasoning direction.

The unit of analysis is the triple

    (prefix constraint subspace C_{t-1}, transition v_t, updated subspace C_t)

rather than a per-step cloud mean.  The script evaluates:

1. first-error localization;
2. pre-error future-error awareness;
3. response-level hazard aggregation without washing out local step evidence.

It also reports negative controls:

* random subspace;
* permuted step order;
* wrong question anchor.

The implementation intentionally requires raw hidden states for the CTG path.
It does not silently fall back to stepcloud statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


try:
    import torch

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    HAVE_TORCH = False


try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from anchorflow.data import Trace, load_traces  # noqa: E402


@dataclass
class StepRow:
    chain_idx: int
    chain_id: str
    problem_id: int
    step_idx: int
    n_steps: int
    gold_error_step: int
    phase: str
    y_first_error: int
    y_future_error: int
    y_chain_error: int
    features: Dict[str, float] = field(default_factory=dict)


def finite_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return finite_json(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def safe_name(text: Any, max_len: int = 120) -> str:
    s = str(text)
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() or ch in {"_", "-", "."} else "_")
    v = "".join(out).strip("_")
    return (v or "x")[:max_len]


def auroc(score: Iterable[float], y: Iterable[int]) -> float:
    s = np.asarray(list(score), dtype=np.float64)
    yy = np.asarray(list(y), dtype=int)
    m = np.isfinite(s)
    s, yy = s[m], yy[m]
    p = int(np.sum(yy == 1))
    n = int(np.sum(yy == 0))
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
    return float((np.sum(ranks[yy == 1]) - p * (p + 1) / 2.0) / (p * n))


def descriptive(vals: Iterable[float]) -> Dict[str, Any]:
    x = np.asarray([float(v) for v in vals], dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
    }


def parse_layers(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("--layers must contain at least one layer")
    return vals


def phase_for(gold: int, step_idx: int) -> str:
    if gold < 0:
        return "correct_chain"
    if step_idx < gold:
        return "pre_error"
    if step_idx == gold:
        return "first_error"
    return "post_error"


def first_error_label(phase: str, control_pool: str) -> int:
    if phase == "first_error":
        return 1
    if phase == "post_error":
        return -1
    if control_pool == "pre_and_correct":
        return 0
    if control_pool == "pre_error":
        return 0 if phase == "pre_error" else -1
    if control_pool == "correct_chain":
        return 0 if phase == "correct_chain" else -1
    raise ValueError(control_pool)


def future_error_label(phase: str) -> int:
    if phase == "pre_error":
        return 1
    if phase == "correct_chain":
        return 0
    return -1


def unit_np(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / max(n, EPS)


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit_np(a), unit_np(b)))


def choose_device(name: str) -> str:
    if name == "auto":
        return "cuda" if HAVE_TORCH and torch.cuda.is_available() else "cpu"
    if name == "cuda" and (not HAVE_TORCH or not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but torch.cuda is not available.")
    return name


def torch_tensor(x: np.ndarray, device: str):
    if not HAVE_TORCH:
        raise RuntimeError("CTG computation requires torch. Install PyTorch or run in the research env.")
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def orth_basis_t(vectors, *, rank_max: int, eps: float = 1e-7):
    """Return a column-orthonormal basis for the row span of `vectors`."""
    if vectors is None:
        raise ValueError("vectors cannot be None")
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")
    d = int(vectors.shape[1])
    if vectors.shape[0] == 0:
        return torch.zeros((d, 0), dtype=vectors.dtype, device=vectors.device)
    keep = torch.linalg.norm(vectors, dim=1) > eps
    A = vectors[keep]
    if A.numel() == 0:
        return torch.zeros((d, 0), dtype=vectors.dtype, device=vectors.device)
    try:
        _u, s, vh = torch.linalg.svd(A, full_matrices=False)
    except RuntimeError:
        _u, s, vh = torch.linalg.svd(A.cpu(), full_matrices=False)
        s = s.to(vectors.device)
        vh = vh.to(vectors.device)
    if s.numel() == 0:
        return torch.zeros((d, 0), dtype=vectors.dtype, device=vectors.device)
    rel = s / torch.clamp(s[0], min=eps)
    r = int(torch.sum(rel > eps).item())
    r = max(0, min(r, int(rank_max), int(vh.shape[0])))
    if r == 0:
        return torch.zeros((d, 0), dtype=vectors.dtype, device=vectors.device)
    return vh[:r].T.contiguous()


def projection_energy_t(v, Q) -> float:
    den = torch.sum(v * v)
    if Q.numel() == 0 or float(den.detach().cpu()) <= EPS:
        return 0.0
    coeff = Q.T @ v
    num = torch.sum(coeff * coeff)
    return float((num / torch.clamp(den, min=EPS)).detach().cpu())


def projection_vec_t(v, Q):
    if Q.numel() == 0:
        return torch.zeros_like(v)
    return Q @ (Q.T @ v)


def cos_t(a, b) -> float:
    den = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(den.detach().cpu()) <= EPS:
        return float("nan")
    return float((torch.dot(a, b) / torch.clamp(den, min=EPS)).detach().cpu())


def orth_basis_np(vectors: np.ndarray, *, rank_max: int, eps: float = 1e-7) -> np.ndarray:
    A = np.asarray(vectors, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("vectors must be 2D")
    d = int(A.shape[1])
    if A.shape[0] == 0:
        return np.zeros((d, 0), dtype=np.float64)
    keep = np.linalg.norm(A, axis=1) > eps
    A = A[keep]
    if A.size == 0:
        return np.zeros((d, 0), dtype=np.float64)
    _u, s, vh = np.linalg.svd(A, full_matrices=False)
    if s.size == 0:
        return np.zeros((d, 0), dtype=np.float64)
    rel = s / max(float(s[0]), eps)
    r = int(np.sum(rel > eps))
    r = max(0, min(r, int(rank_max), int(vh.shape[0])))
    return vh[:r].T.copy() if r else np.zeros((d, 0), dtype=np.float64)


def projection_energy_np(v: np.ndarray, Q: np.ndarray) -> float:
    den = float(np.dot(v, v))
    if Q.size == 0 or den <= EPS:
        return 0.0
    coeff = Q.T @ v
    return float(np.dot(coeff, coeff) / max(den, EPS))


def projection_vec_np(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if Q.size == 0:
        return np.zeros_like(v)
    return Q @ (Q.T @ v)


def prefix_spectrum_np(vectors: np.ndarray, rank_max: int) -> Dict[str, float]:
    A = np.asarray(vectors, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] < 1:
        return {
            "prefix_rank": 0.0,
            "prefix_effrank": float("nan"),
            "prefix_top_ratio": float("nan"),
            "prefix_log_volume": float("nan"),
        }
    keep = np.linalg.norm(A, axis=1) > 1e-7
    A = A[keep]
    if A.size == 0:
        return {
            "prefix_rank": 0.0,
            "prefix_effrank": float("nan"),
            "prefix_top_ratio": float("nan"),
            "prefix_log_volume": float("nan"),
        }
    _u, s, _vh = np.linalg.svd(A, full_matrices=False)
    s = s[: max(1, min(rank_max, int(s.size)))]
    e = s * s
    total = float(np.sum(e))
    if total <= EPS:
        return {
            "prefix_rank": 0.0,
            "prefix_effrank": float("nan"),
            "prefix_top_ratio": float("nan"),
            "prefix_log_volume": float("nan"),
        }
    p = e / total
    eff = float(np.exp(-np.sum(p * np.log(np.maximum(p, EPS)))))
    return {
        "prefix_rank": float(np.sum(s > 1e-7)),
        "prefix_effrank": eff,
        "prefix_top_ratio": float(np.max(e) / total),
        "prefix_log_volume": float(0.5 * np.sum(np.log1p(e))),
    }


def procrustes_transport_np(v_prev: np.ndarray, Q_old: np.ndarray, Q_new: np.ndarray) -> np.ndarray:
    if Q_old.size == 0 or Q_new.size == 0:
        return np.zeros_like(v_prev)
    k = min(int(Q_old.shape[1]), int(Q_new.shape[1]))
    if k == 0:
        return np.zeros_like(v_prev)
    A = Q_old[:, :k]
    B = Q_new[:, :k]
    U, _s, Vh = np.linalg.svd(B.T @ A, full_matrices=False)
    R = U @ Vh
    return B @ (R @ (A.T @ v_prev))


def compute_ctg_one_layer_np(
    states: np.ndarray,
    qvec: np.ndarray,
    *,
    basis: Optional[np.ndarray],
    rank_max: int,
    order: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    T, d = states.shape
    if order is None:
        order = np.arange(T, dtype=int)
        inv_order = order
    else:
        order = np.asarray(order, dtype=int)
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))
    H = np.asarray(states[order], dtype=np.float64)
    q = np.asarray(qvec, dtype=np.float64)
    if basis is not None:
        B = np.asarray(basis, dtype=np.float64)
        H = H @ B
        q = q @ B
    q = unit_np(q)
    d_r = int(H.shape[1])
    deltas = np.zeros_like(H)
    if T > 1:
        deltas[1:] = H[1:] - H[:-1]

    vals: Dict[str, List[float]] = {
        "unsupported": [],
        "support": [],
        "state_outside_prefix": [],
        "q_coupling": [],
        "q_alignment_state": [],
        "prev_cos": [],
        "transport_resid": [],
        "transport_cos": [],
        "transport_angle": [],
        "curvature": [],
        "transition_norm": [],
        "state_norm": [],
        "prefix_rank": [],
        "prefix_effrank": [],
        "prefix_top_ratio": [],
        "prefix_log_volume": [],
        "prefix_load": [],
        "subspace_angle_mean": [],
        "subspace_angle_max": [],
    }
    prev_basis = None
    prevprev_basis = None
    for t in range(T):
        if t == 0:
            for key in vals:
                if key in {"state_norm", "q_alignment_state"}:
                    continue
                vals[key].append(float("nan"))
            vals["state_norm"].append(float(np.linalg.norm(H[t])))
            vals["q_alignment_state"].append(cosine_np(H[t], q))
            prev_basis = orth_basis_np(q.reshape(1, -1), rank_max=rank_max)
            prevprev_basis = None
            continue

        prefix_parts = [q.reshape(1, -1)]
        if t > 1:
            prefix_parts.append(deltas[1:t])
        if t > 0:
            prefix_parts.append(H[:t] - H[0:1])
        prefix_vectors = np.concatenate(prefix_parts, axis=0)
        C_prev = orth_basis_np(prefix_vectors, rank_max=rank_max)
        C_cur = orth_basis_np(
            np.concatenate([prefix_vectors, deltas[t : t + 1], H[t : t + 1] - H[0:1]], axis=0),
            rank_max=rank_max,
        )
        v = deltas[t]
        v_norm = float(np.linalg.norm(v))
        support = projection_energy_np(v, C_prev)
        unsupported = 1.0 - support if np.isfinite(support) else float("nan")
        state_out = 1.0 - projection_energy_np(H[t], C_prev)
        q_coupling = projection_energy_np(v, q.reshape(-1, 1))
        prev_cos = cosine_np(v, deltas[t - 1]) if t >= 2 else float("nan")

        if t >= 2 and prevprev_basis is not None and prev_basis is not None:
            pred = procrustes_transport_np(deltas[t - 1], prevprev_basis, C_prev)
        elif t >= 2:
            pred = projection_vec_np(deltas[t - 1], C_prev)
        else:
            pred = np.zeros_like(v)
        pred_norm = float(np.linalg.norm(pred))
        transport_resid = float(np.linalg.norm(v - pred) / max(v_norm, EPS))
        transport_cos = cosine_np(v, pred) if pred_norm > EPS else float("nan")
        transport_angle = float("nan") if not np.isfinite(transport_cos) else 1.0 - transport_cos
        curvature = (
            float(np.linalg.norm(v - deltas[t - 1]) / max(np.linalg.norm(v) + np.linalg.norm(deltas[t - 1]), EPS))
            if t >= 2
            else float("nan")
        )
        spec = prefix_spectrum_np(prefix_vectors, rank_max)
        if C_prev.size and C_cur.size:
            k = min(int(C_prev.shape[1]), int(C_cur.shape[1]))
            if k > 0:
                sv = np.linalg.svd(C_prev[:, :k].T @ C_cur[:, :k], full_matrices=False, compute_uv=False)
                angles = np.arccos(np.clip(sv, 0.0, 1.0))
                angle_mean = float(np.mean(angles))
                angle_max = float(np.max(angles))
            else:
                angle_mean = float("nan")
                angle_max = float("nan")
        else:
            angle_mean = float("nan")
            angle_max = float("nan")

        vals["unsupported"].append(float(unsupported))
        vals["support"].append(float(support))
        vals["state_outside_prefix"].append(float(state_out))
        vals["q_coupling"].append(float(q_coupling))
        vals["q_alignment_state"].append(cosine_np(H[t], q))
        vals["prev_cos"].append(float(prev_cos))
        vals["transport_resid"].append(float(transport_resid))
        vals["transport_cos"].append(float(transport_cos))
        vals["transport_angle"].append(float(transport_angle))
        vals["curvature"].append(float(curvature))
        vals["transition_norm"].append(float(v_norm))
        vals["state_norm"].append(float(np.linalg.norm(H[t])))
        vals["prefix_rank"].append(spec["prefix_rank"])
        vals["prefix_effrank"].append(spec["prefix_effrank"])
        vals["prefix_top_ratio"].append(spec["prefix_top_ratio"])
        vals["prefix_log_volume"].append(spec["prefix_log_volume"])
        vals["prefix_load"].append(float(C_prev.shape[1]) / max(1.0, float(d_r)))
        vals["subspace_angle_mean"].append(float(angle_mean))
        vals["subspace_angle_max"].append(float(angle_max))
        prevprev_basis = prev_basis
        prev_basis = C_prev

    out = {k: np.asarray(v, dtype=np.float64) for k, v in vals.items()}
    return {k: arr[inv_order] for k, arr in out.items()}


def prefix_spectrum_t(vectors, rank_max: int) -> Dict[str, float]:
    if vectors.ndim != 2 or vectors.shape[0] < 1:
        return {
            "prefix_rank": 0.0,
            "prefix_effrank": float("nan"),
            "prefix_top_ratio": float("nan"),
            "prefix_log_volume": float("nan"),
        }
    keep = torch.linalg.norm(vectors, dim=1) > 1e-7
    A = vectors[keep]
    if A.numel() == 0:
        return {
            "prefix_rank": 0.0,
            "prefix_effrank": float("nan"),
            "prefix_top_ratio": float("nan"),
            "prefix_log_volume": float("nan"),
        }
    try:
        _u, s, _vh = torch.linalg.svd(A, full_matrices=False)
    except RuntimeError:
        _u, s, _vh = torch.linalg.svd(A.cpu(), full_matrices=False)
        s = s.to(vectors.device)
    s = s[: max(1, min(rank_max, int(s.numel())))]
    e = s * s
    total = torch.sum(e)
    if float(total.detach().cpu()) <= EPS:
        return {
            "prefix_rank": 0.0,
            "prefix_effrank": float("nan"),
            "prefix_top_ratio": float("nan"),
            "prefix_log_volume": float("nan"),
        }
    p = e / total
    eff = torch.exp(-torch.sum(p * torch.log(torch.clamp(p, min=EPS))))
    return {
        "prefix_rank": float(torch.sum(s > 1e-7).detach().cpu()),
        "prefix_effrank": float(eff.detach().cpu()),
        "prefix_top_ratio": float((torch.max(e) / total).detach().cpu()),
        "prefix_log_volume": float((0.5 * torch.sum(torch.log1p(e))).detach().cpu()),
    }


def procrustes_transport_t(v_prev, Q_old, Q_new):
    """Approximate parallel transport from C_old to C_new by Procrustes."""
    if Q_old.numel() == 0 or Q_new.numel() == 0:
        return torch.zeros_like(v_prev)
    k = min(int(Q_old.shape[1]), int(Q_new.shape[1]))
    if k == 0:
        return torch.zeros_like(v_prev)
    A = Q_old[:, :k]
    B = Q_new[:, :k]
    M = B.T @ A
    try:
        U, _s, Vh = torch.linalg.svd(M, full_matrices=False)
    except RuntimeError:
        U, _s, Vh = torch.linalg.svd(M.cpu(), full_matrices=False)
        U = U.to(v_prev.device)
        Vh = Vh.to(v_prev.device)
    R = U @ Vh
    coords = A.T @ v_prev
    return B @ (R @ coords)


def summarize_layers(per_layer: List[Dict[str, np.ndarray]], layers: Sequence[int], prefix: str) -> Dict[str, np.ndarray]:
    if not per_layer:
        return {}
    keys = sorted(set().union(*(d.keys() for d in per_layer)))
    T = len(next(iter(per_layer[0].values())))
    out: Dict[str, np.ndarray] = {}
    layer_x = np.asarray(layers, dtype=np.float64)
    if len(layer_x) > 1:
        layer_x = (layer_x - layer_x.min()) / max(float(layer_x.max() - layer_x.min()), EPS)
    else:
        layer_x = np.zeros_like(layer_x)
    for key in keys:
        mat = np.full((T, len(per_layer)), np.nan, dtype=np.float64)
        for j, d in enumerate(per_layer):
            if key in d:
                vals = np.asarray(d[key], dtype=np.float64)
                mat[: min(T, len(vals)), j] = vals[:T]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[f"{prefix}{key}_mean"] = np.nanmean(mat, axis=1)
            out[f"{prefix}{key}_max"] = np.nanmax(mat, axis=1)
            out[f"{prefix}{key}_min"] = np.nanmin(mat, axis=1)
            out[f"{prefix}{key}_std"] = np.nanstd(mat, axis=1)
        if len(per_layer) > 1:
            slope = np.full(T, np.nan, dtype=np.float64)
            for t in range(T):
                y = mat[t]
                m = np.isfinite(y)
                if int(m.sum()) >= 2:
                    xx = layer_x[m] - float(np.mean(layer_x[m]))
                    yy = y[m] - float(np.mean(y[m]))
                    den = float(np.dot(xx, xx))
                    if den > EPS:
                        slope[t] = float(np.dot(xx, yy) / den)
            out[f"{prefix}{key}_layer_slope"] = slope
            out[f"{prefix}{key}_late_minus_early"] = mat[:, -1] - mat[:, 0]
    return out


def compute_ctg_one_layer(
    states: np.ndarray,
    qvec: np.ndarray,
    *,
    basis: Optional[np.ndarray],
    rank_max: int,
    device: str,
    order: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute CTG features for one layer and one chain."""
    if states.ndim != 2:
        raise ValueError("states must be (steps, dim)")
    T, d = states.shape
    if qvec.shape[-1] != d:
        raise ValueError(f"qvec dim {qvec.shape[-1]} does not match hidden dim {d}")
    if basis is not None and basis.shape[0] != d:
        raise ValueError(f"basis dim {basis.shape[0]} does not match hidden dim {d}")

    if not HAVE_TORCH:
        return compute_ctg_one_layer_np(
            states,
            qvec,
            basis=basis,
            rank_max=rank_max,
            order=order,
        )

    if order is None:
        order = np.arange(T, dtype=int)
        inv_order = order
    else:
        order = np.asarray(order, dtype=int)
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

    H_np = np.asarray(states[order], dtype=np.float32)
    q_np = np.asarray(qvec, dtype=np.float32)
    if basis is not None:
        B = np.asarray(basis, dtype=np.float32)
        H_np = H_np @ B
        q_np = q_np @ B

    H = torch_tensor(H_np, device)
    q = torch_tensor(q_np, device)
    q = q / torch.clamp(torch.linalg.norm(q), min=EPS)
    d_r = int(H.shape[1])

    deltas = torch.zeros_like(H)
    if T > 1:
        deltas[1:] = H[1:] - H[:-1]

    vals: Dict[str, List[float]] = {
        "unsupported": [],
        "support": [],
        "state_outside_prefix": [],
        "q_coupling": [],
        "q_alignment_state": [],
        "prev_cos": [],
        "transport_resid": [],
        "transport_cos": [],
        "transport_angle": [],
        "curvature": [],
        "transition_norm": [],
        "state_norm": [],
        "prefix_rank": [],
        "prefix_effrank": [],
        "prefix_top_ratio": [],
        "prefix_log_volume": [],
        "prefix_load": [],
        "subspace_angle_mean": [],
        "subspace_angle_max": [],
    }

    prev_basis = None
    prevprev_basis = None
    for t in range(T):
        if t == 0:
            for key in vals:
                if key in {"state_norm", "q_alignment_state"}:
                    continue
                vals[key].append(float("nan"))
            vals["state_norm"].append(float(torch.linalg.norm(H[t]).detach().cpu()))
            vals["q_alignment_state"].append(cos_t(H[t], q))
            prev_basis = orth_basis_t(q.reshape(1, -1), rank_max=rank_max)
            prevprev_basis = None
            continue

        prefix_parts = [q.reshape(1, -1)]
        if t > 1:
            prefix_parts.append(deltas[1:t])
        if t > 0:
            prefix_parts.append(H[:t] - H[0:1])
        prefix_vectors = torch.cat(prefix_parts, dim=0)
        C_prev = orth_basis_t(prefix_vectors, rank_max=rank_max)

        current_parts = [prefix_vectors, deltas[t : t + 1], H[t : t + 1] - H[0:1]]
        C_cur = orth_basis_t(torch.cat(current_parts, dim=0), rank_max=rank_max)

        v = deltas[t]
        v_norm = torch.linalg.norm(v)
        support = projection_energy_t(v, C_prev)
        unsupported = 1.0 - support if np.isfinite(support) else float("nan")
        state_out = 1.0 - projection_energy_t(H[t], C_prev)
        q_coupling = projection_energy_t(v, q.reshape(-1, 1))
        prev_cos = cos_t(v, deltas[t - 1]) if t >= 2 else float("nan")

        if t >= 2 and prevprev_basis is not None and prev_basis is not None:
            pred = procrustes_transport_t(deltas[t - 1], prevprev_basis, C_prev)
        elif t >= 2:
            pred = projection_vec_t(deltas[t - 1], C_prev)
        else:
            pred = torch.zeros_like(v)
        pred_norm = torch.linalg.norm(pred)
        transport_resid = float(
            (torch.linalg.norm(v - pred) / torch.clamp(v_norm, min=EPS)).detach().cpu()
        )
        transport_cos = cos_t(v, pred) if float(pred_norm.detach().cpu()) > EPS else float("nan")
        transport_angle = float("nan") if not np.isfinite(transport_cos) else 1.0 - transport_cos

        if t >= 2:
            denom = torch.clamp(torch.linalg.norm(v) + torch.linalg.norm(deltas[t - 1]), min=EPS)
            curvature = float((torch.linalg.norm(v - deltas[t - 1]) / denom).detach().cpu())
        else:
            curvature = float("nan")

        spec = prefix_spectrum_t(prefix_vectors, rank_max)
        if C_prev.numel() and C_cur.numel():
            k = min(int(C_prev.shape[1]), int(C_cur.shape[1]))
            if k > 0:
                try:
                    sv = torch.linalg.svdvals(C_prev[:, :k].T @ C_cur[:, :k])
                except RuntimeError:
                    sv = torch.linalg.svdvals((C_prev[:, :k].T @ C_cur[:, :k]).cpu()).to(device)
                sv = torch.clamp(sv, 0.0, 1.0)
                angles = torch.arccos(sv)
                angle_mean = float(torch.mean(angles).detach().cpu())
                angle_max = float(torch.max(angles).detach().cpu())
            else:
                angle_mean = float("nan")
                angle_max = float("nan")
        else:
            angle_mean = float("nan")
            angle_max = float("nan")

        vals["unsupported"].append(float(unsupported))
        vals["support"].append(float(support))
        vals["state_outside_prefix"].append(float(state_out))
        vals["q_coupling"].append(float(q_coupling))
        vals["q_alignment_state"].append(cos_t(H[t], q))
        vals["prev_cos"].append(float(prev_cos))
        vals["transport_resid"].append(float(transport_resid))
        vals["transport_cos"].append(float(transport_cos))
        vals["transport_angle"].append(float(transport_angle))
        vals["curvature"].append(float(curvature))
        vals["transition_norm"].append(float(v_norm.detach().cpu()))
        vals["state_norm"].append(float(torch.linalg.norm(H[t]).detach().cpu()))
        vals["prefix_rank"].append(spec["prefix_rank"])
        vals["prefix_effrank"].append(spec["prefix_effrank"])
        vals["prefix_top_ratio"].append(spec["prefix_top_ratio"])
        vals["prefix_log_volume"].append(spec["prefix_log_volume"])
        vals["prefix_load"].append(float(C_prev.shape[1]) / max(1.0, float(d_r)))
        vals["subspace_angle_mean"].append(float(angle_mean))
        vals["subspace_angle_max"].append(float(angle_max))

        prevprev_basis = prev_basis
        prev_basis = C_prev

    out = {k: np.asarray(v, dtype=np.float64) for k, v in vals.items()}
    if order is not None:
        out = {k: arr[inv_order] for k, arr in out.items()}
    return out


def infer_layer_axis(arr: np.ndarray) -> str:
    if arr.ndim != 3:
        raise ValueError("layer axis exists only for 3D arrays")
    a, b, _d = arr.shape
    if a <= 256 and b > a:
        return "layer_first"
    if b <= 256:
        return "token_first"
    return "layer_first"


def load_hidden_array(path: str) -> np.ndarray:
    if not path:
        raise FileNotFoundError("empty hidden path")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False)
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        for key in ("hidden", "hiddens", "hidden_states", "arr_0"):
            if key in z.files:
                return z[key]
        if len(z.files) == 1:
            return z[z.files[0]]
        raise KeyError(f"Cannot infer hidden array key in {p}; keys={z.files}")
    raise ValueError(f"Unsupported hidden file suffix: {p.suffix}")


def layer_index_from_labels(labels: Optional[Sequence[int]], layer: int, n_layers: int, nearest: bool) -> int:
    if labels:
        vals = [int(x) for x in labels]
        if int(layer) in vals:
            return vals.index(int(layer))
        if nearest:
            return int(np.argmin([abs(x - int(layer)) for x in vals]))
    if 0 <= int(layer) < n_layers:
        return int(layer)
    if nearest:
        return int(min(max(int(layer), 0), n_layers - 1))
    raise ValueError(f"Layer {layer} not found in labels={labels} and cannot index {n_layers} layers")


def select_hidden_layer(arr: np.ndarray, layer: int, labels: Optional[Sequence[int]], nearest: bool) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        return np.asarray(a, dtype=np.float32)
    if a.ndim != 3:
        raise ValueError(f"Expected hidden array with ndim 2 or 3, got shape {a.shape}")
    axis = infer_layer_axis(a)
    if axis == "layer_first":
        li = layer_index_from_labels(labels, layer, int(a.shape[0]), nearest)
        return np.asarray(a[li], dtype=np.float32)
    li = layer_index_from_labels(labels, layer, int(a.shape[1]), nearest)
    return np.asarray(a[:, li, :], dtype=np.float32)


def pool_step_states(H: np.ndarray, ranges: np.ndarray, pool: str) -> Optional[np.ndarray]:
    rr = np.asarray(ranges, dtype=int)
    if rr.ndim != 2 or rr.shape[1] != 2 or len(rr) == 0:
        return None
    a0 = int(rr[0, 0])
    out = []
    for lo0, hi0 in rr:
        lo = max(0, int(lo0) - a0)
        hi = min(len(H), int(hi0) - a0 + 1)
        if hi <= lo:
            return None
        if pool == "last":
            out.append(H[hi - 1])
        elif pool == "mean":
            out.append(np.mean(H[lo:hi], axis=0))
        else:
            raise ValueError(pool)
    return np.asarray(out, dtype=np.float32)


def get_npz_layers(z: Mapping[str, Any], keys: Sequence[str]) -> Optional[List[int]]:
    for key in keys:
        if key in z:
            try:
                return [int(x) for x in np.asarray(z[key]).reshape(-1)]
            except Exception:
                continue
    return None


def get_qvec_for_layer(
    z: Mapping[str, Any],
    trace: Trace,
    chain_idx: int,
    layer: int,
    q_layers: Optional[Sequence[int]],
    nearest: bool,
) -> Optional[np.ndarray]:
    if "qvec" not in z:
        return trace.qvec
    qraw = z["qvec"]
    try:
        qi = np.asarray(qraw[chain_idx], dtype=np.float32)
    except Exception:
        qi = np.asarray(qraw, dtype=np.float32)
    if qi.ndim == 1:
        return qi
    if qi.ndim == 2:
        li = layer_index_from_labels(q_layers, layer, int(qi.shape[0]), nearest)
        return np.asarray(qi[li], dtype=np.float32)
    if qi.ndim == 3:
        li = layer_index_from_labels(q_layers, layer, int(qi.shape[1]), nearest)
        return np.asarray(qi[chain_idx, li], dtype=np.float32)
    return trace.qvec


def text_controls(step_text: str) -> Dict[str, float]:
    txt = step_text or ""
    return {
        "ctrl_chars": float(len(txt)),
        "ctrl_words": float(len(re.findall(r"\S+", txt))),
        "ctrl_digits": float(len(re.findall(r"\d", txt))),
        "ctrl_numbers": float(len(re.findall(r"[-+]?\d+(?:\.\d+)?", txt))),
        "ctrl_ops": float(sum(txt.count(op) for op in ["+", "-", "*", "/", "=", "%"])),
        "ctrl_latex": float(txt.count("\\") + txt.count("$")),
        "ctrl_equation": float(1 if ("=" in txt or "\\[" in txt or "$" in txt) else 0),
    }


def control_features(trace: Trace, t: int) -> Dict[str, float]:
    T = trace.n_steps
    ranges = np.asarray(trace.step_token_ranges, dtype=int)
    n_tok = float(max(1, ranges[t, 1] - ranges[t, 0] + 1))
    prefix_tok = float(np.sum(np.maximum(1, ranges[: t + 1, 1] - ranges[: t + 1, 0] + 1)))
    pos = float(t / max(1, T - 1))
    prev_tok = float(max(1, ranges[t - 1, 1] - ranges[t - 1, 0] + 1)) if t > 0 else 0.0
    feats = {
        "ctrl_step_idx": float(t),
        "ctrl_pos": pos,
        "ctrl_pos2": pos * pos,
        "ctrl_n_steps": float(T),
        "ctrl_remaining_steps": float(max(0, T - t - 1)),
        "ctrl_n_tok": n_tok,
        "ctrl_log_n_tok": float(np.log1p(n_tok)),
        "ctrl_prev_n_tok": prev_tok,
        "ctrl_prefix_tok": prefix_tok,
        "ctrl_log_prefix_tok": float(np.log1p(prefix_tok)),
    }
    if trace.steps_text and t < len(trace.steps_text):
        feats.update(text_controls(trace.steps_text[t]))
    else:
        feats.update(text_controls(""))
    return feats


def make_basis_from_unembedding(args: argparse.Namespace, device: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    if args.unembedding_path:
        p = Path(args.unembedding_path)
        if p.suffix == ".npy":
            W = np.load(p)
        elif p.suffix == ".npz":
            z = np.load(p)
            key = args.unembedding_key or ("W_U" if "W_U" in z.files else z.files[0])
            W = z[key]
        else:
            raise ValueError(f"Unsupported unembedding path suffix: {p.suffix}")
    elif args.model_name_or_path:
        try:
            from transformers import AutoModelForCausalLM
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("transformers is required for --model_name_or_path") from exc
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype="auto")
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            W = model.lm_head.weight.detach().float().cpu().numpy()
        elif hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
            W = model.get_output_embeddings().weight.detach().float().cpu().numpy()
        else:
            raise RuntimeError("Could not locate output embedding / lm_head.weight")
        del model
    else:
        raise ValueError("--basis unembedding requires --unembedding_path or --model_name_or_path")

    W = np.asarray(W, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(f"Unembedding matrix must be 2D, got {W.shape}")
    if not HAVE_TORCH:
        raise RuntimeError("Unembedding SVD requires torch.")
    Wt = torch_tensor(W, device)
    try:
        _U, S, Vh = torch.linalg.svd(Wt, full_matrices=False)
    except RuntimeError:
        _U, S, Vh = torch.linalg.svd(Wt.cpu(), full_matrices=False)
        S = S.to(device)
        Vh = Vh.to(device)
    d = int(Vh.shape[1])
    k = int(args.reasoning_dim or min(256, max(8, d // 4)))
    k = max(1, min(k, int(Vh.shape[0])))
    if args.basis_side == "bottom":
        B = Vh[-k:].T
    elif args.basis_side == "top":
        B = Vh[:k].T
    else:
        raise ValueError(args.basis_side)
    meta = {
        "basis": "unembedding",
        "side": args.basis_side,
        "reasoning_dim": int(k),
        "hidden_dim": int(d),
        "singular_top": float(S[0].detach().cpu()),
        "singular_cut": float(S[-k].detach().cpu()) if args.basis_side == "bottom" else float(S[k - 1].detach().cpu()),
    }
    return B.detach().cpu().numpy().astype(np.float32), meta


def load_basis(args: argparse.Namespace, hidden_dim_hint: Optional[int], device: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if args.basis == "identity":
        return None, {"basis": "identity_raw_hidden", "reasoning_dim": hidden_dim_hint}
    if args.basis == "npy":
        if not args.basis_path:
            raise ValueError("--basis npy requires --basis_path")
        p = Path(args.basis_path)
        if p.suffix == ".npz":
            z = np.load(p)
            key = args.basis_key or z.files[0]
            B = z[key]
        else:
            B = np.load(p)
        B = np.asarray(B, dtype=np.float32)
        if B.ndim != 2:
            raise ValueError(f"Basis matrix must be 2D, got {B.shape}")
        return B, {"basis": "npy", "path": str(p), "reasoning_dim": int(B.shape[1])}
    if args.basis == "unembedding":
        return make_basis_from_unembedding(args, device)
    raise ValueError(args.basis)


def random_basis(dim: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, rank)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return np.asarray(Q[:, :rank], dtype=np.float32)


def collect_hidden_dim(traces: Sequence[Trace], hidden_layers: Optional[Sequence[int]], args: argparse.Namespace) -> Optional[int]:
    for tr in traces:
        if not tr.hidden_path:
            continue
        try:
            arr = load_hidden_array(tr.hidden_path)
            H = select_hidden_layer(arr, args.layers[0], hidden_layers, args.nearest_layer)
            return int(H.shape[-1])
        except Exception:
            continue
    return None


def build_rows(args: argparse.Namespace) -> Tuple[List[StepRow], Dict[str, Any], Dict[str, List[str]]]:
    layers = args.layers
    traces, meta = load_traces(
        args.npz_path,
        dataset=args.dataset,
        layer=layers[0],
        max_chains=args.max_chains,
        hidden_dir=args.hidden_dir,
    )
    if not traces:
        raise RuntimeError(f"No traces loaded from {args.npz_path}")

    zfile = np.load(args.npz_path, allow_pickle=True)
    z = {k: zfile[k] for k in zfile.files}
    zfile.close()
    hidden_layers = get_npz_layers(z, ("hidden_layers", "layers_used", "sv_layers"))
    q_layers = get_npz_layers(z, ("sv_layers", "hidden_layers", "layers_used"))
    hidden_dim = collect_hidden_dim(traces, hidden_layers, args)
    if hidden_dim is None:
        raise RuntimeError(
            "CTG requires raw hidden files. Provide --hidden_dir or include hidden_dir/hidden_files in the npz."
        )

    device = choose_device(args.device)
    basis, basis_meta = load_basis(args, hidden_dim, device)
    analysis_dim = int(basis.shape[1]) if basis is not None else int(hidden_dim)
    rand_dim = int(args.random_dim or max(4, min(32, analysis_dim // 4 if analysis_dim >= 8 else analysis_dim)))
    rand_dim = max(1, min(rand_dim, hidden_dim, analysis_dim))
    rand_basis = random_basis(hidden_dim, rand_dim, args.seed + 991)

    rows: List[StepRow] = []
    missing_hidden = 0
    used_hidden = 0
    skipped_shape = 0
    rng = np.random.default_rng(args.seed)

    q_bank: Dict[Tuple[int, int], np.ndarray] = {}
    for tr in traces:
        for layer in layers:
            q = get_qvec_for_layer(z, tr, tr.idx, layer, q_layers, args.nearest_layer)
            if q is not None:
                q_bank[(tr.idx, layer)] = np.asarray(q, dtype=np.float32)

    for trace_pos, tr in enumerate(tqdm(traces, desc="CTG chains", disable=args.no_progress)):
        if not tr.hidden_path:
            missing_hidden += 1
            continue
        try:
            harr = load_hidden_array(tr.hidden_path)
        except Exception:
            missing_hidden += 1
            continue

        per_layer_ctg: List[Dict[str, np.ndarray]] = []
        per_layer_rand: List[Dict[str, np.ndarray]] = []
        per_layer_perm: List[Dict[str, np.ndarray]] = []
        per_layer_wrongq: List[Dict[str, np.ndarray]] = []
        try:
            for layer in layers:
                Htok = select_hidden_layer(harr, layer, hidden_layers, args.nearest_layer)
                states = pool_step_states(Htok, tr.step_token_ranges, args.step_pool)
                if states is None or states.shape[0] != tr.n_steps:
                    raise ValueError("invalid step pooled hidden states")
                q = q_bank.get((tr.idx, layer))
                if q is None:
                    raise ValueError("missing qvec")
                if q.shape[-1] != states.shape[-1]:
                    raise ValueError(f"qvec dim {q.shape[-1]} != hidden dim {states.shape[-1]}")

                per_layer_ctg.append(
                    compute_ctg_one_layer(
                        states,
                        q,
                        basis=basis,
                        rank_max=args.rank_max,
                        device=device,
                    )
                )
                per_layer_rand.append(
                    compute_ctg_one_layer(
                        states,
                        q,
                        basis=rand_basis,
                        rank_max=args.rank_max,
                        device=device,
                    )
                )
                if tr.n_steps >= 3:
                    order = rng.permutation(tr.n_steps)
                else:
                    order = np.arange(tr.n_steps)
                per_layer_perm.append(
                    compute_ctg_one_layer(
                        states,
                        q,
                        basis=basis,
                        rank_max=args.rank_max,
                        device=device,
                        order=order,
                    )
                )
                wrong_trace = traces[(trace_pos + 1) % len(traces)]
                wrong_q = q_bank.get((wrong_trace.idx, layer), q)
                per_layer_wrongq.append(
                    compute_ctg_one_layer(
                        states,
                        wrong_q,
                        basis=basis,
                        rank_max=args.rank_max,
                        device=device,
                    )
                )
            used_hidden += 1
        except Exception:
            skipped_shape += 1
            continue

        feats_by_step: Dict[str, np.ndarray] = {}
        feats_by_step.update(summarize_layers(per_layer_ctg, layers, "ctg_"))
        feats_by_step.update(summarize_layers(per_layer_rand, layers, "rand_"))
        feats_by_step.update(summarize_layers(per_layer_perm, layers, "perm_"))
        feats_by_step.update(summarize_layers(per_layer_wrongq, layers, "wrongq_"))

        for t in range(tr.n_steps):
            phase = phase_for(tr.gold_error_step, t)
            y_first = first_error_label(phase, args.control_pool)
            y_future = future_error_label(phase)
            row = StepRow(
                chain_idx=int(tr.idx),
                chain_id=str(tr.chain_id),
                problem_id=int(tr.problem_id),
                step_idx=int(t),
                n_steps=int(tr.n_steps),
                gold_error_step=int(tr.gold_error_step),
                phase=phase,
                y_first_error=int(y_first),
                y_future_error=int(y_future),
                y_chain_error=int(0 if tr.correct else 1),
                features=control_features(tr, t),
            )
            for name, arr in feats_by_step.items():
                if t < len(arr):
                    row.features[name] = float(arr[t])
            rows.append(row)

    if used_hidden == 0:
        raise RuntimeError(
            "No chains with usable raw hidden states were found. This script does not run a stepcloud fallback."
        )

    names = sorted(set().union(*(r.features.keys() for r in rows)))
    norm_markers = ("transition_norm", "state_norm")

    def geometry_names(prefix: str) -> List[str]:
        return [
            n
            for n in names
            if n.startswith(prefix) and not any(marker in n for marker in norm_markers)
        ]

    feature_groups = {
        "controls": [n for n in names if n.startswith("ctrl_")],
        "ctg": geometry_names("ctg_"),
        "ctg_norm": [n for n in names if n.startswith("ctg_") and any(marker in n for marker in norm_markers)],
        "rand": geometry_names("rand_"),
        "perm": geometry_names("perm_"),
        "wrongq": geometry_names("wrongq_"),
    }
    run_meta = {
        **meta,
        "npz_path": args.npz_path,
        "hidden_dir": args.hidden_dir,
        "layers": list(layers),
        "hidden_layers": hidden_layers,
        "q_layers": q_layers,
        "hidden_dim": int(hidden_dim),
        "analysis_dim": int(analysis_dim),
        "random_control_dim": int(rand_dim),
        "device": device,
        "basis": basis_meta,
        "rows": int(len(rows)),
        "chains_loaded": int(len(traces)),
        "chains_with_hidden": int(used_hidden),
        "missing_hidden": int(missing_hidden),
        "skipped_shape": int(skipped_shape),
        "feature_group_sizes": {k: len(v) for k, v in feature_groups.items()},
    }
    return rows, run_meta, feature_groups


def matrix_from_rows(rows: Sequence[StepRow], feature_names: Sequence[str], label_name: str):
    selected = []
    y = []
    groups = []
    chains = []
    for i, row in enumerate(rows):
        lab = getattr(row, label_name)
        if int(lab) < 0:
            continue
        selected.append(i)
        y.append(int(lab))
        groups.append(int(row.problem_id))
        chains.append(int(row.chain_idx))
    X = np.full((len(selected), len(feature_names)), np.nan, dtype=np.float64)
    for ii, ridx in enumerate(selected):
        feats = rows[ridx].features
        for j, name in enumerate(feature_names):
            X[ii, j] = float(feats.get(name, np.nan))
    return np.asarray(selected, dtype=int), X, np.asarray(y, dtype=int), np.asarray(groups), np.asarray(chains)


def group_folds(groups: np.ndarray, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups)
    ug = np.unique(groups)
    if len(ug) < 2:
        idx = np.arange(len(groups))
        return [(idx, idx)]
    rng = np.random.default_rng(seed)
    ug = ug.copy()
    rng.shuffle(ug)
    buckets = [[] for _ in range(max(2, min(k, len(ug))))]
    for i, g in enumerate(ug):
        buckets[i % len(buckets)].append(g)
    folds = []
    all_idx = np.arange(len(groups))
    for vals in buckets:
        test_mask = np.isin(groups, vals)
        train = all_idx[~test_mask]
        test = all_idx[test_mask]
        if len(train) and len(test):
            folds.append((train, test))
    return folds


def oof_logistic(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    folds: int,
) -> np.ndarray:
    if not HAVE_SKLEARN:
        raise RuntimeError("sklearn is required for grouped OOF evaluation.")
    scores = np.full(len(y), np.nan, dtype=np.float64)
    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return scores
    for train, test in group_folds(groups, folds, seed):
        if len(np.unique(y[train])) < 2:
            continue
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            ),
        )
        pipe.fit(X[train], y[train])
        scores[test] = pipe.predict_proba(X[test])[:, 1]
    return scores


def oof_logistic_residualized(
    X_ctrl: np.ndarray,
    X_geo: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    include_controls: bool,
    seed: int,
    folds: int,
) -> np.ndarray:
    if not HAVE_SKLEARN:
        raise RuntimeError("sklearn is required for grouped OOF evaluation.")
    scores = np.full(len(y), np.nan, dtype=np.float64)
    if X_geo.shape[1] == 0 or len(np.unique(y)) < 2:
        return scores
    for train, test in group_folds(groups, folds, seed):
        if len(np.unique(y[train])) < 2:
            continue
        imp_c = SimpleImputer(strategy="median")
        sc_c = StandardScaler()
        Ctr = sc_c.fit_transform(imp_c.fit_transform(X_ctrl[train]))
        Cte = sc_c.transform(imp_c.transform(X_ctrl[test]))

        imp_g = SimpleImputer(strategy="median")
        Gtr = imp_g.fit_transform(X_geo[train])
        Gte = imp_g.transform(X_geo[test])

        ridge = Ridge(alpha=1.0, random_state=seed)
        ridge.fit(Ctr, Gtr)
        Rtr = Gtr - ridge.predict(Ctr)
        Rte = Gte - ridge.predict(Cte)

        Xtr = np.concatenate([Ctr, Rtr], axis=1) if include_controls else Rtr
        Xte = np.concatenate([Cte, Rte], axis=1) if include_controls else Rte
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            ),
        )
        pipe.fit(Xtr, y[train])
        scores[test] = pipe.predict_proba(Xte)[:, 1]
    return scores


def within_chain_auc(scores: np.ndarray, y: np.ndarray, chains: np.ndarray) -> float:
    vals = []
    for c in np.unique(chains):
        m = chains == c
        if len(np.unique(y[m])) < 2:
            continue
        a = auroc(scores[m], y[m])
        if np.isfinite(a):
            vals.append(a)
    return float(np.mean(vals)) if vals else float("nan")


def cluster_boot_increment(
    sf: np.ndarray,
    sb: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    sf = np.asarray(sf, dtype=np.float64)
    sb = np.asarray(sb, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    m = np.isfinite(sf) & np.isfinite(sb)
    if int(m.sum()) < 30 or len(np.unique(y[m])) < 2:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "sig": False}
    point = auroc(sf[m], y[m]) - auroc(sb[m], y[m])
    if n_boot <= 0:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"), "sig": False}
    rng = np.random.default_rng(seed)
    ug = np.unique(groups[m])
    by = {g: np.where(m & (groups == g))[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        chosen = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([by[g] for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auroc(sf[idx], y[idx]) - auroc(sb[idx], y[idx]))
    if not vals:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"), "sig": False}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0 or hi < 0)}


def response_scores(
    rows_subset: Sequence[StepRow],
    scores: np.ndarray,
    *,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_chain: Dict[int, List[float]] = {}
    labels: Dict[int, int] = {}
    groups: Dict[int, int] = {}
    for row, sc in zip(rows_subset, scores):
        if not np.isfinite(sc):
            continue
        by_chain.setdefault(row.chain_idx, []).append(float(sc))
        labels[row.chain_idx] = int(row.y_chain_error)
        groups[row.chain_idx] = int(row.problem_id)
    out_s, out_y, out_g = [], [], []
    for c, vals in by_chain.items():
        v = np.asarray(vals, dtype=np.float64)
        if method == "max":
            agg = float(np.max(v))
        elif method == "topk_mean":
            k = max(1, min(3, int(math.ceil(math.sqrt(len(v))))))
            agg = float(np.mean(np.sort(v)[-k:]))
        elif method == "noisy_or":
            vv = np.clip(v, 1e-6, 1.0 - 1e-6)
            agg = float(1.0 - np.prod(1.0 - vv))
        elif method == "mean":
            agg = float(np.mean(v))
        else:
            raise ValueError(method)
        out_s.append(agg)
        out_y.append(labels[c])
        out_g.append(groups[c])
    return np.asarray(out_s), np.asarray(out_y, dtype=int), np.asarray(out_g)


def evaluate_task(
    rows: Sequence[StepRow],
    feature_groups: Mapping[str, List[str]],
    *,
    label_name: str,
    seed: int,
    folds: int,
    bootstrap: int,
) -> Dict[str, Any]:
    controls = feature_groups["controls"]
    ctg = feature_groups["ctg"]
    rand = feature_groups["rand"]
    perm = feature_groups["perm"]
    wrongq = feature_groups["wrongq"]

    model_specs = {
        "controls": controls,
        "ctg": ctg,
        "controls+ctg": controls + ctg,
        "rand_subspace": rand,
        "controls+rand_subspace": controls + rand,
        "permuted_steps": perm,
        "wrong_anchor": wrongq,
    }

    selected_idx, _X0, y, groups, chains = matrix_from_rows(rows, [], label_name)
    selected_rows = [rows[int(i)] for i in selected_idx]
    result: Dict[str, Any] = {
        "rows": int(len(y)),
        "pos": int(np.sum(y == 1)),
        "neg": int(np.sum(y == 0)),
        "models": {},
        "increments": {},
        "response": {},
    }
    if len(y) == 0 or len(np.unique(y)) < 2:
        return result

    score_bank: Dict[str, np.ndarray] = {}
    for name, feats in model_specs.items():
        _idx, X, yy, gg, cc = matrix_from_rows(rows, feats, label_name)
        scores = oof_logistic(X, yy, gg, seed=seed, folds=folds)
        score_bank[name] = scores
        result["models"][name] = {
            "n_features": int(len(feats)),
            "pooled": auroc(scores, yy),
            "within_chain": within_chain_auc(scores, yy, cc),
        }

    _idx, Xc, yy, gg, cc = matrix_from_rows(rows, controls, label_name)
    _idx, Xg, _yy, _gg, _cc = matrix_from_rows(rows, ctg, label_name)
    scores_resid = oof_logistic_residualized(
        Xc, Xg, yy, gg, include_controls=False, seed=seed, folds=folds
    )
    scores_ctrl_resid = oof_logistic_residualized(
        Xc, Xg, yy, gg, include_controls=True, seed=seed, folds=folds
    )
    score_bank["ctg_residualized"] = scores_resid
    score_bank["controls+ctg_residualized"] = scores_ctrl_resid
    result["models"]["ctg_residualized"] = {
        "n_features": int(len(ctg)),
        "pooled": auroc(scores_resid, yy),
        "within_chain": within_chain_auc(scores_resid, yy, cc),
    }
    result["models"]["controls+ctg_residualized"] = {
        "n_features": int(len(controls) + len(ctg)),
        "pooled": auroc(scores_ctrl_resid, yy),
        "within_chain": within_chain_auc(scores_ctrl_resid, yy, cc),
    }

    for lhs, rhs in [
        ("controls+ctg", "controls"),
        ("controls+ctg_residualized", "controls"),
        ("ctg", "rand_subspace"),
        ("ctg", "permuted_steps"),
        ("ctg", "wrong_anchor"),
    ]:
        result["increments"][f"{lhs}_vs_{rhs}"] = cluster_boot_increment(
            score_bank[lhs], score_bank[rhs], yy, gg, n_boot=bootstrap, seed=seed + 17
        )

    for model_name in ["controls", "ctg", "controls+ctg", "controls+ctg_residualized"]:
        result["response"][model_name] = {}
        for method in ["max", "topk_mean", "noisy_or", "mean"]:
            rs, ry, rg = response_scores(selected_rows, score_bank[model_name], method=method)
            result["response"][model_name][method] = {
                "auc": auroc(rs, ry),
                "n_chains": int(len(ry)),
                "pos": int(np.sum(ry == 1)),
                "neg": int(np.sum(ry == 0)),
            }
    return result


def write_outputs(
    rows: Sequence[StepRow],
    summary: Mapping[str, Any],
    feature_groups: Mapping[str, List[str]],
    output_dir: str,
    stem: str,
) -> Tuple[str, str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{stem}_constraint_transport_geometry.json"
    md_path = out / f"{stem}_constraint_transport_geometry.md"
    csv_path = out / f"{stem}_constraint_transport_geometry_step_rows.csv"

    all_features = sorted(set().union(*(r.features.keys() for r in rows)))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "chain_idx",
                "chain_id",
                "problem_id",
                "step_idx",
                "n_steps",
                "gold_error_step",
                "phase",
                "y_first_error",
                "y_future_error",
                "y_chain_error",
            ]
            + all_features
        )
        for r in rows:
            w.writerow(
                [
                    r.chain_idx,
                    r.chain_id,
                    r.problem_id,
                    r.step_idx,
                    r.n_steps,
                    r.gold_error_step,
                    r.phase,
                    r.y_first_error,
                    r.y_future_error,
                    r.y_chain_error,
                ]
                + [r.features.get(k, float("nan")) for k in all_features]
            )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(finite_json(summary), f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Constraint Transport Geometry Audit")
    lines.append("")
    meta = summary.get("meta", {})
    lines.append(f"- rows: {meta.get('rows')}")
    lines.append(f"- chains with hidden: {meta.get('chains_with_hidden')}")
    lines.append(f"- layers: {meta.get('layers')}")
    lines.append(f"- basis: {meta.get('basis')}")
    lines.append(f"- feature groups: {meta.get('feature_group_sizes')}")
    lines.append("")
    for task_name in ["first_error", "pre_error_future"]:
        task = summary.get("tasks", {}).get(task_name, {})
        lines.append(f"## {task_name}")
        lines.append("")
        lines.append(f"rows={task.get('rows')} pos={task.get('pos')} neg={task.get('neg')}")
        lines.append("")
        lines.append("| model | features | pooled AUROC | within-chain AUROC |")
        lines.append("|---|---:|---:|---:|")
        for name, vals in sorted(
            task.get("models", {}).items(),
            key=lambda kv: -float(kv[1].get("pooled") if kv[1].get("pooled") is not None else -999),
        ):
            pooled = vals.get("pooled")
            within = vals.get("within_chain")
            lines.append(
                f"| {name} | {vals.get('n_features')} | "
                f"{pooled:.3f} | {within:.3f} |"
                if pooled is not None and np.isfinite(pooled)
                else f"| {name} | {vals.get('n_features')} | nan | nan |"
            )
        lines.append("")
        lines.append("| increment | point | 95% low | 95% high |")
        lines.append("|---|---:|---:|---:|")
        for name, vals in task.get("increments", {}).items():
            lines.append(
                f"| {name} | {vals.get('point', float('nan')):.3f} | "
                f"{vals.get('lo', float('nan')):.3f} | {vals.get('hi', float('nan')):.3f} |"
            )
        lines.append("")
        lines.append("| response model | agg | AUROC |")
        lines.append("|---|---|---:|")
        for model, aggs in task.get("response", {}).items():
            for agg, vals in aggs.items():
                lines.append(f"| {model} | {agg} | {vals.get('auc', float('nan')):.3f} |")
        lines.append("")
    lines.append("## Feature Groups")
    lines.append("")
    for group, names in feature_groups.items():
        lines.append(f"- {group}: {len(names)}")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return str(json_path), str(md_path), str(csv_path)


def create_selftest_dataset(tmp: str, seed: int) -> Tuple[str, str]:
    rng = np.random.default_rng(seed)
    root = Path(tmp)
    hidden_dir = root / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    n = 140
    layers = np.array([10, 14, 18], dtype=int)
    d = 64
    ids = []
    problem_ids = []
    gold = []
    ranges_obj = []
    steps_obj = []
    responses = []
    questions = []
    hidden_files = []
    qvec = np.zeros((n, len(layers), d), dtype=np.float32)

    for i in range(n):
        T = int(rng.integers(5, 10))
        is_wrong = bool(i % 2 == 0)
        ge = int(rng.integers(2, T - 1)) if is_wrong else -1
        gold.append(ge)
        ids.append(f"chain_{i:04d}")
        problem_ids.append(i // 2)
        questions.append(f"Synthetic question {i}")
        step_ranges = []
        texts = []
        token_counts = [int(rng.integers(3, 8)) for _ in range(T)]
        offset = 0
        for t, ntok in enumerate(token_counts):
            step_ranges.append([offset, offset + ntok - 1])
            texts.append(f"Step {t}: synthetic calculation with {ntok} tokens")
            offset += ntok
        layer_tokens = []
        for li, layer in enumerate(layers):
            q = unit_np(rng.standard_normal(d))
            qvec[i, li] = q.astype(np.float32)
            support_a = unit_np(q + 0.20 * rng.standard_normal(d))
            support_b = unit_np(rng.standard_normal(d))
            support_b = unit_np(support_b - np.dot(support_b, support_a) * support_a)
            rogue = unit_np(rng.standard_normal(d))
            rogue = rogue - np.dot(rogue, support_a) * support_a - np.dot(rogue, support_b) * support_b
            rogue = unit_np(rogue)
            state = 0.2 * q + 0.05 * rng.standard_normal(d)
            toks_all = []
            for t, ntok in enumerate(token_counts):
                if t == 0:
                    delta = 0.15 * support_a + 0.02 * rng.standard_normal(d)
                elif is_wrong and t == ge:
                    delta = 0.10 * support_a + 1.20 * rogue + 0.05 * rng.standard_normal(d)
                elif is_wrong and t > ge:
                    delta = 0.25 * rogue + 0.08 * rng.standard_normal(d)
                else:
                    delta = 0.25 * support_a + 0.12 * support_b + 0.04 * rng.standard_normal(d)
                state = state + delta
                toks = state[None, :] + 0.03 * rng.standard_normal((ntok, d))
                toks_all.append(toks.astype(np.float32))
            layer_tokens.append(np.concatenate(toks_all, axis=0))
        hidden = np.stack(layer_tokens, axis=0)
        hfile = f"chain_{i:04d}.npy"
        np.save(hidden_dir / hfile, hidden)
        hidden_files.append(hfile)
        ranges_obj.append(np.asarray(step_ranges, dtype=int))
        steps_obj.append(np.asarray(texts, dtype=object))
        responses.append("\n".join(texts))

    npz_path = root / "synthetic_ctg.npz"
    np.savez(
        npz_path,
        ids=np.asarray(ids, dtype=object),
        problem_ids=np.asarray(problem_ids, dtype=int),
        gold_error_step=np.asarray(gold, dtype=int),
        step_token_ranges=np.asarray(ranges_obj, dtype=object),
        steps_text=np.asarray(steps_obj, dtype=object),
        responses=np.asarray(responses, dtype=object),
        questions=np.asarray(questions, dtype=object),
        hidden_files=np.asarray(hidden_files, dtype=object),
        hidden_dir=str(hidden_dir),
        hidden_layers=layers,
        sv_layers=layers,
        qvec=qvec,
    )
    return str(npz_path), str(hidden_dir)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    args.layers = parse_layers(args.layers) if isinstance(args.layers, str) else args.layers
    rows, meta, feature_groups = build_rows(args)
    tasks = {
        "first_error": evaluate_task(
            rows,
            feature_groups,
            label_name="y_first_error",
            seed=args.seed,
            folds=args.folds,
            bootstrap=args.bootstrap,
        ),
        "pre_error_future": evaluate_task(
            rows,
            feature_groups,
            label_name="y_future_error",
            seed=args.seed + 101,
            folds=args.folds,
            bootstrap=args.bootstrap,
        ),
    }
    summary = {
        "meta": meta,
        "tasks": tasks,
        "distributions": {},
    }
    for group_name in ["ctg", "rand", "perm", "wrongq"]:
        summary["distributions"][group_name] = {}
        for name in feature_groups[group_name]:
            if name.endswith("_mean") or name.endswith("_max"):
                vals = [r.features.get(name, float("nan")) for r in rows]
                summary["distributions"][group_name][name] = descriptive(vals)

    stem = safe_name(Path(args.npz_path).stem)
    output_dir = args.output_dir or str(Path("outputs") / f"constraint_transport_geometry_{stem}")
    jpath, mpath, cpath = write_outputs(rows, summary, feature_groups, output_dir, stem)
    summary["outputs"] = {"json": jpath, "markdown": mpath, "csv": cpath}

    print(f"===== constraint transport geometry | {Path(args.npz_path).name} =====")
    print(
        f"rows {meta['rows']} | chains {meta['chains_with_hidden']} | "
        f"features ctrl {len(feature_groups['controls'])} ctg {len(feature_groups['ctg'])} "
        f"rand {len(feature_groups['rand'])} perm {len(feature_groups['perm'])}"
    )
    for task_name, task in tasks.items():
        print(f"\nTask {task_name}:")
        print(f"  rows {task['rows']} pos {task['pos']} neg {task['neg']}")
        for name, vals in sorted(
            task.get("models", {}).items(),
            key=lambda kv: -float(kv[1].get("pooled") if np.isfinite(kv[1].get("pooled", np.nan)) else -999),
        ):
            print(
                f"  {name:<32} pooled {vals['pooled']:.3f} "
                f"within-chain {vals['within_chain']:.3f}"
            )
        for name, vals in task.get("increments", {}).items():
            print(
                f"  inc {name:<42} {vals['point']:+.3f} "
                f"[{vals['lo']:+.3f},{vals['hi']:+.3f}]"
            )
        print("  response:")
        best = []
        for model, aggs in task.get("response", {}).items():
            for agg, vals in aggs.items():
                best.append((vals["auc"], model, agg))
        for auc, model, agg in sorted(best, reverse=True)[:6]:
            print(f"    {model}/{agg:<10} AUC {auc:.3f}")
    print(f"\noutputs:\n  {jpath}\n  {mpath}\n  {cpath}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz_path", nargs="?", help="Canonical trace npz, e.g. data/full_gsm8k.npz")
    p.add_argument("--hidden_dir", default=None, help="Directory containing per-chain raw hidden .npy/.npz files")
    p.add_argument("--dataset", default="")
    p.add_argument("--layers", default="14", help="Comma-separated layers to analyze")
    p.add_argument("--nearest_layer", action="store_true")
    p.add_argument("--step_pool", choices=["mean", "last"], default="mean")
    p.add_argument("--basis", choices=["identity", "npy", "unembedding"], default="identity")
    p.add_argument("--basis_path", default=None)
    p.add_argument("--basis_key", default=None)
    p.add_argument("--unembedding_path", default=None)
    p.add_argument("--unembedding_key", default=None)
    p.add_argument("--model_name_or_path", default=None)
    p.add_argument("--basis_side", choices=["bottom", "top"], default="bottom")
    p.add_argument("--reasoning_dim", type=int, default=0)
    p.add_argument("--random_dim", type=int, default=0, help="Random-subspace negative-control dimension")
    p.add_argument("--rank_max", type=int, default=16)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max_chains", type=int, default=0)
    p.add_argument("--control_pool", choices=["pre_and_correct", "pre_error", "correct_chain"], default="pre_and_correct")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--no_progress", action="store_true")
    p.add_argument("--selftest", action="store_true", help="Run on a generated synthetic dataset")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.selftest:
        with tempfile.TemporaryDirectory(prefix="ctg_selftest_") as tmp:
            npz_path, hidden_dir = create_selftest_dataset(tmp, args.seed)
            args.npz_path = npz_path
            args.hidden_dir = hidden_dir
            args.layers = "10,14,18"
            args.basis = "identity"
            args.output_dir = args.output_dir or str(Path("outputs") / "constraint_transport_geometry_selftest")
            run(args)
        return
    if not args.npz_path:
        raise SystemExit("npz_path is required unless --selftest is used")
    run(args)


if __name__ == "__main__":
    main()
