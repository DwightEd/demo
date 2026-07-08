#!/usr/bin/env python3
"""Strict whole-chain Gram metric audit.

This script reproduces the hidden-state part of:

    What do Geometric Hallucination Detection Metrics Actually Measure?

at the correct object level: the Gram matrix over **all generated tokens in a
response / reasoning chain**, not a within-step token cloud.

For a selected layer l and whole-chain hidden matrix H_l in R^{m x d}:

    G_l = H_l H_l^T
    HS(H_l) = (1 / m) log det(G_l)
    ME(H_l) = - sum_i q_i log q_i, q_i = lambda_i / trace(G_l)

HS is reported only when G_l is full rank.  If the stored representation is a
low-dimensional projection and m exceeds that dimension, the strict HS value is
undefined; the script reports coverage instead of silently using a pseudo-logdet.

The paper's attention score is also supported, but only when an exact saved
self-attention diagonal is present.  Existing prompt-attention summaries such as
q_frac/sink_frac/attn_entropy are deliberately not used as substitutes.
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
    problem_groups,
    within_pair_auroc,
)
from premise_constraint_audit import bootstrap_within_increment, pair_rescue_report
from second_moment_dynamics_audit import auroc, bdir, oof_scores
from token_stream_geometry_audit import chain_lengths, load_token_matrix, source_info


EPS = 1e-12


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - progress bars are optional
    def tqdm(iterable, **_kwargs):
        return iterable


@dataclass
class ChainRow:
    idx: int
    problem_id: int
    y_err: int
    keep: bool
    n_tokens: int
    n_steps: int
    gold_error_step: int
    features: Dict[str, float]


@dataclass
class StepRow:
    chain_idx: int
    problem_id: int
    step_idx: int
    gold_error_step: int
    y_first_error: int
    features: Dict[str, float]


def scalar_str(x: Any) -> str:
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(x)


def sort_value(x: Any, default: float = -1.0) -> float:
    """Finite numeric sort key that survives JSON-cleaned None values."""
    try:
        if x is None:
            return float(default)
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def select_layer(layers: Sequence[int], target: int, nearest: bool) -> Tuple[int, int]:
    vals = [int(x) for x in layers]
    if int(target) in vals:
        i = vals.index(int(target))
        return i, vals[i]
    if not nearest:
        raise SystemExit(f"layer {target} not found; available layers: {vals}")
    i = int(np.argmin(np.abs(np.asarray(vals, dtype=int) - int(target))))
    return i, vals[i]


def chain_policy(data: np.lib.npyio.NpzFile, policy: str) -> Tuple[np.ndarray, np.ndarray, str]:
    if policy == "gold_error_step":
        if "gold_error_step" not in data.files:
            raise SystemExit("--policy gold_error_step requires gold_error_step")
        n = len(data["gold_error_step"])
        return (data["gold_error_step"].astype(int) >= 0).astype(int), np.ones(n, dtype=bool), "gold_error_step >= 0"
    return label_policy(data, policy)


def positive_eigvals(vals: np.ndarray, *, rel_tol: float) -> np.ndarray:
    x = np.asarray(vals, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    x = np.sort(x)[::-1]
    x = np.clip(x, 0.0, None)
    tol = max(EPS, float(rel_tol) * float(x[0] if x.size else 0.0))
    return x[x > tol]


def gram_metrics_from_eigvals(vals: np.ndarray, m: int, *, rel_tol: float) -> Dict[str, float]:
    """Paper HS/ME from eigenvalues of the whole-chain Gram matrix."""
    pos = positive_eigvals(vals, rel_tol=rel_tol)
    out = {
        "hs": float("nan"),
        "me": float("nan"),
        "eff_rank": float("nan"),
        "lam1": float("nan"),
        "log_energy": float("nan"),
        "rank": float(pos.size),
        "rank_frac": float(pos.size / max(1, int(m))),
        "hs_full_rank": float(pos.size == int(m)),
    }
    if pos.size == 0:
        return out
    total = float(np.sum(pos))
    q = pos / max(total, EPS)
    ent = float(-np.sum(q * np.log(q + EPS)))
    out["me"] = ent
    out["eff_rank"] = float(np.exp(ent))
    out["lam1"] = float(q[0])
    out["log_energy"] = float(np.log(total + EPS))
    if pos.size == int(m):
        out["hs"] = float(np.mean(np.log(pos + EPS)))
    return out


def gram_metrics_from_matrix(H: np.ndarray, *, rel_tol: float) -> Dict[str, float]:
    """Strict paper metrics for H H^T using singular values of H."""
    X = np.asarray(H, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return gram_metrics_from_eigvals(np.array([], dtype=np.float64), 0, rel_tol=rel_tol)
    s = np.linalg.svd(X, compute_uv=False)
    vals = s * s
    # If m > min(m, d), the missing eigenvalues of G are zeros.
    if vals.size < X.shape[0]:
        vals = np.concatenate([vals, np.zeros(X.shape[0] - vals.size, dtype=np.float64)])
    return gram_metrics_from_eigvals(vals, X.shape[0], rel_tol=rel_tol)


def whole_chain_kappa(H: np.ndarray) -> Tuple[float, float]:
    X = np.asarray(H, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return float("nan"), float("nan")
    norms = np.linalg.norm(X, axis=1)
    ok = np.isfinite(norms) & (norms > EPS)
    if ok.sum() == 0:
        return float("nan"), float("nan")
    U = X[ok] / np.maximum(norms[ok, None], EPS)
    kappa = float(np.linalg.norm(np.mean(U, axis=0)))
    return kappa, float(1.0 - kappa)


def step_endpoints(lengths: np.ndarray, n_tokens: int) -> np.ndarray:
    lens = np.asarray(lengths, dtype=int).reshape(-1)
    lens = lens[lens > 0]
    if lens.size == 0:
        return np.asarray([max(0, n_tokens - 1)], dtype=int)
    ends = np.cumsum(lens) - 1
    ends = ends[(ends >= 0) & (ends < int(n_tokens))]
    if ends.size == 0 or ends[-1] != int(n_tokens) - 1:
        ends = np.r_[ends, int(n_tokens) - 1]
    return np.unique(ends.astype(int))


def prefix_gram_traces(H: np.ndarray, endpoints: np.ndarray, *, rel_tol: float) -> Dict[str, np.ndarray]:
    """Paper HS/ME on H[:t] H[:t]^T at step endpoints."""
    X = np.asarray(H, dtype=np.float64)
    n = X.shape[0]
    G = X @ X.T
    G = 0.5 * (G + G.T)
    hs = np.full(len(endpoints), np.nan, dtype=np.float64)
    me = np.full(len(endpoints), np.nan, dtype=np.float64)
    lam1 = np.full(len(endpoints), np.nan, dtype=np.float64)
    eff = np.full(len(endpoints), np.nan, dtype=np.float64)
    rank_frac = np.full(len(endpoints), np.nan, dtype=np.float64)
    for j, end in enumerate(endpoints):
        m = int(np.clip(end + 1, 1, n))
        vals = np.linalg.eigvalsh(G[:m, :m])
        out = gram_metrics_from_eigvals(vals, m, rel_tol=rel_tol)
        hs[j] = out["hs"]
        me[j] = out["me"]
        lam1[j] = out["lam1"]
        eff[j] = out["eff_rank"]
        rank_frac[j] = out["rank_frac"]
    return {
        "prefix_hs": hs,
        "prefix_me": me,
        "prefix_lam1": lam1,
        "prefix_eff_rank": eff,
        "prefix_rank_frac": rank_frac,
    }


def delta_trace(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    out = np.full_like(v, np.nan, dtype=np.float64)
    if v.size:
        out[0] = v[0]
    if v.size > 1:
        out[1:] = v[1:] - v[:-1]
    return out


def array_obj_get(arr: Any, idx: int) -> Any:
    try:
        return arr[idx]
    except Exception:
        return None


def attention_layer_index(data: np.lib.npyio.NpzFile, args: argparse.Namespace) -> int:
    for key in ("attn_layers", "attention_layers", "layers_used", "hidden_layers", "cloud_layers"):
        if key in data.files:
            layers = [int(x) for x in data[key]]
            li, _layer = select_layer(layers, args.layer, args.nearest_layer)
            return li
    return 0


def load_attention_diag(
    data: np.lib.npyio.NpzFile,
    idx: int,
    args: argparse.Namespace,
    n_tokens: int,
) -> Optional[np.ndarray]:
    """Return exact self-attention diagonal as (tokens, heads), if stored.

    The paper's AS needs diag(A_l^head) for each generated token and head.  We
    accept only arrays whose key explicitly names an attention diagonal; summary
    features such as q_frac/sink_frac/attn_entropy are not valid substitutes.
    """
    key = None
    for cand in (
        "attn_self_diag",
        "self_attn_diag",
        "attention_self_diag",
        "attention_diag",
        "as_attention_diag",
    ):
        if cand in data.files:
            key = cand
            break
    if key is None:
        return None
    obj = array_obj_get(data[key], idx)
    if obj is None:
        return None
    A = np.asarray(obj, dtype=np.float64)
    li = attention_layer_index(data, args)
    if A.ndim == 1:
        out = A.reshape(-1, 1)
    elif A.ndim == 2:
        # Already selected layer: (tokens, heads) or (tokens, 1).
        out = A
    elif A.ndim == 3:
        # Prefer (tokens, layers, heads); fall back to (tokens, heads, layers).
        if A.shape[0] == n_tokens and li < A.shape[1]:
            out = A[:, li, :]
        elif A.shape[0] == n_tokens and li < A.shape[2]:
            out = A[:, :, li]
        elif li < A.shape[0]:
            B = A[li]
            out = B.T if B.shape[-1] == n_tokens else B
        else:
            return None
    else:
        return None
    out = np.asarray(out, dtype=np.float64)
    if out.ndim != 2 or out.shape[0] == 0:
        return None
    if out.shape[0] > n_tokens:
        out = out[:n_tokens]
    elif out.shape[0] < n_tokens:
        pad = np.full((n_tokens - out.shape[0], out.shape[1]), np.nan, dtype=np.float64)
        out = np.vstack([out, pad])
    return out


def attention_score(diag: Optional[np.ndarray], *, eps: float) -> float:
    if diag is None:
        return float("nan")
    A = np.asarray(diag, dtype=np.float64)
    vals = A[np.isfinite(A) & (A > 0.0)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(np.log(np.clip(vals, eps, 1.0))))


def prefix_attention_traces(diag: Optional[np.ndarray], endpoints: np.ndarray, *, eps: float) -> Dict[str, np.ndarray]:
    out = np.full(len(endpoints), np.nan, dtype=np.float64)
    if diag is None:
        return {"prefix_as": out}
    A = np.asarray(diag, dtype=np.float64)
    loga = np.where(np.isfinite(A) & (A > 0.0), np.log(np.clip(A, eps, 1.0)), np.nan)
    for j, end in enumerate(endpoints):
        vals = loga[: int(end) + 1]
        out[j] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return {"prefix_as": out}


def feature_matrix(rows: Sequence[ChainRow], names: Sequence[str]) -> np.ndarray:
    return np.asarray([[r.features.get(name, float("nan")) for name in names] for r in rows], dtype=np.float64)


def evaluate_score(
    name: str,
    score: np.ndarray,
    *,
    y: np.ndarray,
    mask: np.ndarray,
    groups: Sequence[np.ndarray],
) -> Dict[str, Any]:
    vals = np.asarray(score, dtype=np.float64)
    m = mask & np.isfinite(vals)
    cross = auroc(vals[m], y[m]) if m.any() else float("nan")
    within, pairs = within_pair_auroc(groups, vals, y)
    within_rev, _ = within_pair_auroc(groups, -vals, y)
    return {
        "name": name,
        "n": int(m.sum()),
        "coverage": float(m.mean()) if mask.any() else 0.0,
        "cross_auroc_error_high": float(cross),
        "cross_best_direction": bdir(cross),
        "within_pair_auroc_error_high": float(within),
        "within_best_direction": float(max(within, within_rev)) if np.isfinite(within) or np.isfinite(within_rev) else float("nan"),
        "within_pairs": int(pairs),
        "direction": "error_high" if np.nan_to_num(within, nan=cross) >= np.nan_to_num(within_rev, nan=1.0 - cross) else "error_low",
        "descriptive_error": descriptive(vals[mask & (y == 1)]),
        "descriptive_correct": descriptive(vals[mask & (y == 0)]),
    }


def usable_feature_names(rows: Sequence[ChainRow], prefix: str, min_coverage: float, mask: np.ndarray) -> List[str]:
    if not rows:
        return []
    keys = sorted({k for r in rows for k in r.features if k.startswith(prefix)})
    out = []
    for k in keys:
        vals = np.asarray([r.features.get(k, float("nan")) for r in rows], dtype=np.float64)
        cov = float(np.mean(np.isfinite(vals[mask]))) if mask.any() else 0.0
        if cov >= min_coverage:
            out.append(k)
    return out


def build_chain_rows(path: str, args: argparse.Namespace) -> Tuple[List[ChainRow], List[StepRow], Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    y_err, keep, policy_desc = chain_policy(data, args.policy)
    if "problem_ids" in data.files:
        problem_ids = data["problem_ids"].astype(int)
    else:
        problem_ids = np.arange(len(y_err), dtype=int)
    gold = data["gold_error_step"].astype(int) if "gold_error_step" in data.files else np.full(len(y_err), -1, dtype=int)
    source, layer_i, layer_used = source_info(data, path, args)

    n_total = len(y_err)
    if args.max_chains:
        n_total = min(n_total, int(args.max_chains))
    rows: List[ChainRow] = []
    step_rows: List[StepRow] = []
    skipped = {"policy": 0, "missing_hidden": 0, "too_short": 0}
    iterator = range(n_total)
    if not args.no_progress:
        iterator = tqdm(iterator, desc="whole-chain Gram", unit="chain", dynamic_ncols=True)
    for i in iterator:
        if not keep[i]:
            skipped["policy"] += 1
            continue
        H = load_token_matrix(data, path, args, idx=i, source=source, layer_i=layer_i)
        if H is None:
            skipped["missing_hidden"] += 1
            continue
        H = np.asarray(H, dtype=np.float64)
        if args.max_tokens and H.shape[0] > args.max_tokens:
            H = H[: int(args.max_tokens)]
        if H.ndim != 2 or H.shape[0] < args.min_tokens:
            skipped["too_short"] += 1
            continue
        lengths, ranges_obj = chain_lengths(data, i, int(H.shape[0]), source)
        endpoints = step_endpoints(lengths, int(H.shape[0]))
        n_steps = int(len(endpoints))
        kappa, spread = whole_chain_kappa(H)
        raw = gram_metrics_from_matrix(H, rel_tol=args.rank_rel_tol)
        cen = gram_metrics_from_matrix(H - np.mean(H, axis=0, keepdims=True), rel_tol=args.rank_rel_tol)
        diag = load_attention_diag(data, i, args, int(H.shape[0]))
        feats = {
            "n_tokens": float(H.shape[0]),
            "log_tokens": float(math.log1p(H.shape[0])),
            "n_steps": float(n_steps),
            "chain_kappa": kappa,
            "chain_spread": spread,
            "paper_hs_raw": raw["hs"],
            "paper_me_raw": raw["me"],
            "paper_eff_rank_raw": raw["eff_rank"],
            "paper_lam1_raw": raw["lam1"],
            "paper_log_energy_raw": raw["log_energy"],
            "paper_rank_frac_raw": raw["rank_frac"],
            "ablation_hs_centered": cen["hs"],
            "ablation_me_centered": cen["me"],
            "ablation_eff_rank_centered": cen["eff_rank"],
            "ablation_lam1_centered": cen["lam1"],
            "paper_as": attention_score(diag, eps=args.attn_eps),
        }
        rows.append(
            ChainRow(
                idx=int(i),
                problem_id=int(problem_ids[i]),
                y_err=int(y_err[i]),
                keep=bool(keep[i]),
                n_tokens=int(H.shape[0]),
                n_steps=n_steps,
                gold_error_step=int(gold[i]),
                features=feats,
            )
        )
        if args.no_prefix or "gold_error_step" not in data.files:
            continue
        traces = prefix_gram_traces(H, endpoints, rel_tol=args.rank_rel_tol)
        traces.update(prefix_attention_traces(diag, endpoints, eps=args.attn_eps))
        deltas = {f"delta_{k.replace('prefix_', '')}": delta_trace(v) for k, v in traces.items()}
        g = int(gold[i])
        for t in range(n_steps):
            if g < 0:
                y = 0
            elif t == g:
                y = 1
            elif t < g:
                y = 0
            else:
                continue
            sf: Dict[str, float] = {}
            for k, v in traces.items():
                sf[k] = float(v[t]) if t < len(v) else float("nan")
            for k, v in deltas.items():
                sf[k] = float(v[t]) if t < len(v) else float("nan")
            sf["step_pos"] = float(t / max(1, n_steps - 1))
            sf["step_len"] = float(lengths[t]) if t < len(lengths) else float("nan")
            step_rows.append(
                StepRow(
                    chain_idx=int(i),
                    problem_id=int(problem_ids[i]),
                    step_idx=int(t),
                    gold_error_step=g,
                    y_first_error=int(y),
                    features=sf,
                )
            )
    meta = {
        "input": os.path.abspath(path),
        "basename": os.path.basename(path),
        "policy": args.policy,
        "policy_description": policy_desc,
        "source": source,
        "layer": int(layer_used),
        "skipped": skipped,
    }
    data.close()
    return rows, step_rows, meta


def oof_group_score(
    rows: Sequence[ChainRow],
    names: Sequence[str],
    *,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> np.ndarray:
    X = feature_matrix(rows, names)
    if X.shape[1] == 0:
        return np.full(len(rows), np.nan, dtype=np.float64)
    return oof_scores(X, y, groups, folds=folds, seed=seed)


def evaluate_group_scores(
    rows: Sequence[ChainRow],
    names_by_group: Mapping[str, Sequence[str]],
    *,
    y: np.ndarray,
    mask: np.ndarray,
    problem_ids: np.ndarray,
    groups: Sequence[np.ndarray],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    out: Dict[str, Any] = {}
    scores: Dict[str, np.ndarray] = {}
    for gi, (name, names) in enumerate(names_by_group.items()):
        valid_names = [n for n in names if n]
        s = oof_group_score(rows, valid_names, y=y, groups=problem_ids, folds=args.folds, seed=args.seed + gi)
        scores[name] = s
        row = evaluate_score(name, s, y=y, mask=mask, groups=groups)
        row["features"] = list(valid_names)
        row["n_features"] = len(valid_names)
        out[name] = row
    return out, scores


def step_feature_array(rows: Sequence[StepRow], name: str) -> np.ndarray:
    return np.asarray([r.features.get(name, float("nan")) for r in rows], dtype=np.float64)


def step_labels(rows: Sequence[StepRow]) -> np.ndarray:
    return np.asarray([r.y_first_error for r in rows], dtype=int)


def step_loc_auc(rows: Sequence[StepRow], vals: np.ndarray) -> Tuple[float, int]:
    by_chain: Dict[int, List[int]] = {}
    for i, row in enumerate(rows):
        by_chain.setdefault(row.chain_idx, []).append(i)
    conc = 0.0
    pairs = 0
    for idx in by_chain.values():
        pos = [i for i in idx if rows[i].y_first_error == 1 and np.isfinite(vals[i])]
        neg = [i for i in idx if rows[i].y_first_error == 0 and rows[i].gold_error_step >= 0 and rows[i].step_idx < rows[i].gold_error_step and np.isfinite(vals[i])]
        for p in pos:
            for n in neg:
                conc += 1.0 if vals[p] > vals[n] else (0.5 if vals[p] == vals[n] else 0.0)
        pairs += len(pos) * len(neg)
    return (float(conc / pairs) if pairs else float("nan")), int(pairs)


def evaluate_step_scores(rows: Sequence[StepRow]) -> Dict[str, Any]:
    if not rows:
        return {}
    y = step_labels(rows)
    names = sorted({k for r in rows for k in r.features if k.startswith("prefix_") or k.startswith("delta_")})
    out: Dict[str, Any] = {}
    for name in names:
        v = step_feature_array(rows, name)
        m = np.isfinite(v)
        cross = auroc(v[m], y[m]) if m.any() else float("nan")
        loc, pairs = step_loc_auc(rows, v)
        loc_rev, _ = step_loc_auc(rows, -v)
        out[name] = {
            "n": int(m.sum()),
            "coverage": float(m.mean()) if len(m) else 0.0,
            "cross_auroc_first_error_high": float(cross),
            "cross_best_direction": bdir(cross),
            "within_error_chain_loc_error_high": float(loc),
            "within_error_chain_loc_best": float(max(loc, loc_rev)) if np.isfinite(loc) or np.isfinite(loc_rev) else float("nan"),
            "within_error_chain_pairs": int(pairs),
            "direction": "error_high" if np.nan_to_num(loc, nan=cross) >= np.nan_to_num(loc_rev, nan=1.0 - cross) else "error_low",
        }
    return dict(sorted(out.items(), key=lambda kv: sort_value(kv[1].get("within_error_chain_loc_best")), reverse=True))


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    rows, step_rows, meta = build_chain_rows(path, args)
    if len(rows) < 4 or len({r.y_err for r in rows}) < 2:
        raise SystemExit("not enough chain rows with both classes")
    y = np.asarray([r.y_err for r in rows], dtype=int)
    mask = np.asarray([r.keep for r in rows], dtype=bool)
    problem_ids = np.asarray([r.problem_id for r in rows], dtype=int)
    groups = problem_groups(problem_ids, y, mask, args.min_per_class)

    single_names = sorted({k for r in rows for k in r.features})
    single_scores = {
        name: evaluate_score(name, feature_matrix(rows, [name])[:, 0], y=y, mask=mask, groups=groups)
        for name in single_names
    }
    single_scores = dict(sorted(single_scores.items(), key=lambda kv: sort_value(kv[1].get("within_best_direction")), reverse=True))

    paper_all = usable_feature_names(rows, "paper_", args.min_feature_coverage, mask)
    exact_hsme = [n for n in ("paper_hs_raw", "paper_me_raw") if n in paper_all]
    exact_hsme_lam1 = [n for n in ("paper_hs_raw", "paper_me_raw", "paper_lam1_raw") if n in paper_all]
    as_names = [n for n in ("paper_as",) if n in paper_all]
    paper_aux = [n for n in paper_all if n not in set(exact_hsme_lam1 + as_names)]
    centered = usable_feature_names(rows, "ablation_", args.min_feature_coverage, mask)
    baseline_length = ["log_tokens", "n_steps"]
    baseline_geom = ["log_tokens", "n_steps", "chain_spread"]
    group_defs: Dict[str, Sequence[str]] = {
        "baseline_length": baseline_length,
        "baseline_length_spread": baseline_geom,
        "paper_exact_hs_me": exact_hsme,
        "paper_exact_hs_me_lam1": exact_hsme_lam1,
    }
    if as_names:
        group_defs["paper_exact_hs_me_as"] = exact_hsme + as_names
    if paper_aux:
        group_defs["paper_aux_ablation"] = exact_hsme_lam1 + paper_aux
    if centered:
        group_defs["centered_ablation"] = centered
    group_defs["baseline_plus_exact_paper"] = baseline_geom + exact_hsme + as_names
    group_scores, score_arrays = evaluate_group_scores(
        rows,
        group_defs,
        y=y,
        mask=mask,
        problem_ids=problem_ids,
        groups=groups,
        args=args,
    )
    primary_base = "baseline_length_spread" if "baseline_length_spread" in group_scores else "baseline_length"
    base_score = score_arrays.get(primary_base, np.full(len(rows), np.nan))
    for name, s in score_arrays.items():
        if name == primary_base:
            group_scores[name]["increment_over_primary_baseline"] = {"point": 0.0, "lo": 0.0, "hi": 0.0, "sig": False}
            continue
        group_scores[name]["increment_over_primary_baseline"] = bootstrap_within_increment(
            s,
            base_score,
            groups=groups,
            y_err=y,
            n_boot=args.bootstrap,
            seed=args.seed + 300,
        )
        group_scores[name]["baseline_miss_rescue"] = pair_rescue_report(s, base_score, groups=groups, y_err=y)

    ranked_groups = sorted(
        group_scores.items(),
        key=lambda kv: sort_value(kv[1].get("within_best_direction", kv[1].get("within_pair_auroc_error_high"))),
        reverse=True,
    )
    step_scores = evaluate_step_scores(step_rows)
    hs_cov = single_scores.get("paper_hs_raw", {}).get("coverage", 0.0)
    as_cov = single_scores.get("paper_as", {}).get("coverage", 0.0)
    res = {
        "meta": {
            **meta,
            "n_chains": int(len(rows)),
            "n_error": int(y.sum()),
            "n_correct": int((y == 0).sum()),
            "n_contrastive_problems": int(len(groups)),
            "within_pairs": int(sum(int((y[g] == 1).sum()) * int((y[g] == 0).sum()) for g in groups)),
            "n_step_rows": int(len(step_rows)),
            "rank_rel_tol": float(args.rank_rel_tol),
            "metric_protocol": {
                "hs_me": "strict whole-chain H_l H_l^T over all generated tokens",
                "hs": "reported only when G_l is full rank; no pseudo-logdet is used",
                "as": "reported only from exact saved self-attention diagonals",
                "prefix": "prefix/delta metrics are derived from whole-chain Gram prefixes for first-error localization",
            },
        },
        "headline": {
            "primary_baseline": primary_base,
            "primary_baseline_row": group_scores.get(primary_base),
            "best_group": ranked_groups[0][0] if ranked_groups else None,
            "best_group_row": ranked_groups[0][1] if ranked_groups else None,
            "paper_hs_coverage": float(hs_cov),
            "paper_as_coverage": float(as_cov),
            "attention_score_status": "available" if as_cov >= args.min_feature_coverage else "missing_exact_attention_diagonal",
            "best_step_prefix_metric": next(iter(step_scores.items())) if step_scores else None,
        },
        "single_scores": single_scores,
        "group_scores": group_scores,
        "step_prefix_scores": step_scores,
    }
    return res


def write_markdown(path: str, res: Mapping[str, Any]) -> None:
    def fmt(x: Any, signed: bool = False) -> str:
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
    lines = [
        f"# Whole-Chain Gram Metrics Audit: `{meta['basename']}`",
        "",
        "## Headline",
        "",
        f"- Source: `{meta['source']}` at layer `{meta['layer']}`.",
        f"- Chains: `{meta['n_chains']}`; errors: `{meta['n_error']}`; contrastive problems: `{meta['n_contrastive_problems']}`.",
        f"- Paper HS coverage: `{fmt(head['paper_hs_coverage'])}`; AS status: `{head['attention_score_status']}`.",
        f"- Primary baseline: `{head['primary_baseline']}`.",
        f"- Best group: `{head['best_group']}`.",
        "",
        "## Group Scores",
        "",
        "| group | within best | within err-high | cross best | increment | CI | features |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(
        res["group_scores"].items(),
        key=lambda kv: sort_value(kv[1].get("within_best_direction")),
        reverse=True,
    ):
        inc = row.get("increment_over_primary_baseline", {})
        ci = "" if inc.get("lo") is None else f"[{fmt(inc.get('lo'), True)}, {fmt(inc.get('hi'), True)}]"
        lines.append(
            f"| `{name}` | {fmt(row.get('within_best_direction'))} | {fmt(row.get('within_pair_auroc_error_high'))} | "
            f"{fmt(row.get('cross_best_direction'))} | {fmt(inc.get('point'), True)} | {ci} | {row.get('n_features', 0)} |"
        )
    lines += [
        "",
        "## Single Whole-Chain Features",
        "",
        "| feature | within best | within err-high | cross best | coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in list(res["single_scores"].items())[:24]:
        lines.append(
            f"| `{name}` | {fmt(row.get('within_best_direction'))} | {fmt(row.get('within_pair_auroc_error_high'))} | "
            f"{fmt(row.get('cross_best_direction'))} | {fmt(row.get('coverage'))} |"
        )
    if res.get("step_prefix_scores"):
        lines += [
            "",
            "## Prefix / First-Error Localization",
            "",
            "| prefix metric | within-chain loc best | loc err-high | cross best | coverage | pairs |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, row in list(res["step_prefix_scores"].items())[:24]:
            lines.append(
                f"| `{name}` | {fmt(row.get('within_error_chain_loc_best'))} | "
                f"{fmt(row.get('within_error_chain_loc_error_high'))} | {fmt(row.get('cross_best_direction'))} | "
                f"{fmt(row.get('coverage'))} | {row.get('within_error_chain_pairs', 0)} |"
            )
    lines += [
        "",
        "## Guardrails",
        "",
        "- HS/ME are computed on whole-chain token Gram matrices, not within-step clouds.",
        "- Strict HS is undefined when the Gram is rank-deficient; coverage is part of the result.",
        "- AS is not approximated by attention entropy or prompt-attention summaries.",
        "- Prefix deltas are a localization adaptation, not the original paper protocol.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_outputs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    jpath = os.path.join(output_dir, stem + ".json")
    mpath = os.path.join(output_dir, stem + ".md")
    clean = finite_json(res)
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    write_markdown(mpath, clean)
    return jpath, mpath


def print_result(res: Mapping[str, Any]) -> None:
    meta = res["meta"]
    head = res["headline"]
    print(f"\n===== whole-chain Gram metrics | {meta['basename']} =====")
    print(
        f"chains {meta['n_chains']} | err {meta['n_error']} | problems {meta['n_contrastive_problems']} | "
        f"source {meta['source']} L{meta['layer']}"
    )
    print(
        f"HS coverage {head['paper_hs_coverage']:.3f} | AS {head['attention_score_status']} | "
        f"baseline {head['primary_baseline']} | best group {head['best_group']}"
    )
    print("\nGroup scores:")
    for name, row in sorted(
        res["group_scores"].items(),
        key=lambda kv: sort_value(kv[1].get("within_best_direction")),
        reverse=True,
    ):
        inc = row.get("increment_over_primary_baseline", {})
        ci = "" if inc.get("lo") is None else f" [{inc.get('lo'):+.3f},{inc.get('hi'):+.3f}]"
        print(
            f"  {name:24s} within-best {row['within_best_direction']:.3f} "
            f"cross-best {row['cross_best_direction']:.3f} inc {inc.get('point', float('nan')):+.3f}{ci}"
        )
    print("\nTop single whole-chain features:")
    for name, row in list(res["single_scores"].items())[:12]:
        print(
            f"  {name:28s} within-best {row['within_best_direction']:.3f} "
            f"cross-best {row['cross_best_direction']:.3f} coverage {row['coverage']:.3f}"
        )
    if res.get("step_prefix_scores"):
        print("\nTop prefix localization metrics:")
        for name, row in list(res["step_prefix_scores"].items())[:10]:
            print(
                f"  {name:28s} loc-best {row['within_error_chain_loc_best']:.3f} "
                f"cross-best {row['cross_best_direction']:.3f} pairs {row['within_error_chain_pairs']}"
            )


def _unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    return x / max(float(np.linalg.norm(x)), EPS)


def make_selftest(path: str, *, seed: int = 0, n_problems: int = 12, samples_per_problem: int = 6, dim: int = 64) -> None:
    rng = np.random.default_rng(seed)
    problem_ids: List[int] = []
    is_correct: List[int] = []
    gold: List[int] = []
    clouds: List[np.ndarray] = []
    sizes_all: List[np.ndarray] = []
    attn_all: List[np.ndarray] = []
    for p in range(n_problems):
        anchor = _unit(rng.normal(size=dim))
        for s in range(samples_per_problem):
            err = s % 3 == 0
            T = 5
            lens = np.asarray([5, 5, 5, 5, 5], dtype=np.int32)
            g = 2 if err else -1
            rows: List[np.ndarray] = []
            diag_rows: List[np.ndarray] = []
            for t in range(T):
                scale = 0.05
                attn_level = 0.08
                if err and t >= g:
                    scale = 0.45
                    attn_level = 0.22
                for _ in range(int(lens[t])):
                    rows.append(anchor + scale * rng.normal(size=dim))
                    diag_rows.append(np.full(3, attn_level + 0.01 * rng.normal(), dtype=np.float64))
            H = np.asarray(rows, dtype=np.float32)[:, None, :]
            A = np.asarray(diag_rows, dtype=np.float32)[:, None, :]
            problem_ids.append(p)
            is_correct.append(0 if err else 1)
            gold.append(g)
            clouds.append(H)
            sizes_all.append(lens)
            attn_all.append(np.clip(A, 1e-4, 0.99))
    obj = lambda xs: np.asarray(xs, dtype=object)
    np.savez_compressed(
        path,
        problem_ids=np.asarray(problem_ids, dtype=np.int32),
        is_correct=np.asarray(is_correct, dtype=np.int8),
        is_correct_strict=np.asarray(is_correct, dtype=np.int8),
        format_ok=np.ones(len(is_correct), dtype=bool),
        gold_error_step=np.asarray(gold, dtype=np.int32),
        sv_clouds=obj(clouds),
        cloud_sizes=obj(sizes_all),
        cloud_layers=np.asarray([16], dtype=np.int32),
        attn_self_diag=obj(attn_all),
        attn_layers=np.asarray([16], dtype=np.int32),
    )


def assert_selftest(res: Mapping[str, Any]) -> None:
    hs = res["single_scores"].get("paper_hs_raw", {})
    me = res["single_scores"].get("paper_me_raw", {})
    best = res["headline"].get("best_group_row") or {}
    step_best = res["headline"].get("best_step_prefix_metric")
    if hs.get("within_best_direction", 0.0) < 0.85:
        raise AssertionError("selftest failed: whole-chain HS did not recover the injected volume signal")
    if me.get("within_best_direction", 0.0) < 0.85:
        raise AssertionError("selftest failed: whole-chain ME did not recover the injected spectral signal")
    if best.get("within_best_direction", 0.0) < 0.90:
        raise AssertionError("selftest failed: paper group score is too weak")
    if not step_best or step_best[1].get("within_error_chain_loc_best", 0.0) < 0.80:
        raise AssertionError("selftest failed: prefix/delta localization is too weak")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="")
    ap.add_argument("--policy", default="answer_format_ok", choices=["answer", "strict", "answer_format_ok", "gold_error_step"])
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--nearest_layer", action="store_true")
    ap.add_argument("--hidden_dir", default="", help="override hidden shard directory for full_hidden npz files")
    ap.add_argument("--no_mmap", action="store_true")
    ap.add_argument("--min_tokens", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=0)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--rank_rel_tol", type=float, default=1e-10)
    ap.add_argument("--attn_eps", type=float, default=1e-12)
    ap.add_argument("--min_feature_coverage", type=float, default=0.50)
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_prefix", action="store_true", help="skip prefix/delta first-error localization")
    ap.add_argument("--output_dir", default="outputs/whole_chain_gram_metrics")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "whole_chain_gram_selftest.npz")
            make_selftest(path, seed=args.seed)
            args.input = path
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
    stem = os.path.splitext(os.path.basename(args.input))[0] + "_whole_chain_gram_metrics"
    jpath, mpath = write_outputs(res, args.output_dir, stem)
    print_result(res)
    print(f"\nsaved: {jpath}\nsaved: {mpath}")


if __name__ == "__main__":
    main()
