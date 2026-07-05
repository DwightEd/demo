#!/usr/bin/env python3
"""Temporal rupture audit for same-problem multi-sampling data.

Unlike `multisample_feature_distribution.py`, this script does not collapse a
chain to a single static level first.  It builds per-step signal sequences and
asks whether incorrect samples show sharper local transitions than correct
samples for the same problem.

The audit is deliberately diagnostic:
  - same-problem paired AUROC is the headline metric;
  - local jump/window contrast are reported beside level summaries;
  - no CUSUM or monotone accumulation is used;
  - without first-error labels this tests whether a rupture signature exists,
    not whether the rupture aligns with the true first wrong step.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12


def finite_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return finite_json(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def descriptive(x: Iterable[float]) -> Dict[str, Any]:
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    q10, q25, q50, q75, q90 = np.percentile(a, [10, 25, 50, 75, 90])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "q10": float(q10),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "q90": float(q90),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _avg_ranks(sorted_vals: np.ndarray) -> np.ndarray:
    ranks = np.arange(1, sorted_vals.size + 1, dtype=np.float64)
    out = ranks.copy()
    i = 0
    while i < sorted_vals.size:
        j = i + 1
        while j < sorted_vals.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            out[i:j] = ranks[i:j].mean()
        i = j
    return out


def auroc_signed(err: Iterable[float], cor: Iterable[float]) -> float:
    err = np.asarray([v for v in err if np.isfinite(v)], dtype=np.float64)
    cor = np.asarray([v for v in cor if np.isfinite(v)], dtype=np.float64)
    if err.size == 0 or cor.size == 0:
        return float("nan")
    combined = np.concatenate([err, cor])
    labels = np.concatenate([np.ones(err.size), np.zeros(cor.size)])
    order = np.argsort(combined, kind="mergesort")
    ranks = _avg_ranks(combined[order])
    full = np.empty_like(ranks)
    full[order] = ranks
    sum_pos = full[labels == 1].sum()
    U = sum_pos - err.size * (err.size + 1) / 2.0
    return float(U / (err.size * cor.size))


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


def safe_mean(v: np.ndarray) -> float:
    a = np.asarray(v, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def band_cols(n_layers: int, band: str) -> np.ndarray:
    if band == "all":
        return np.arange(n_layers)
    if band == "deep":
        return np.arange(int(n_layers * 0.6), n_layers)
    if band == "mid":
        return np.arange(int(n_layers * 0.3), int(n_layers * 0.7))
    return np.array([int(x) for x in band.split(",") if x.strip()], dtype=int)


def window_mask(T: int, window: str) -> np.ndarray:
    if T <= 0:
        return np.zeros(0, dtype=bool)
    frac = np.arange(T) / max(1, T - 1)
    if window == "late":
        m = frac >= 0.6
        return m if m.any() else frac >= frac.max()
    if window == "early":
        m = frac < 0.4
        return m if m.any() else frac <= frac.min()
    return np.ones(T, dtype=bool)


def chain_z(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    med, sc = robust_center_scale(x)
    z = (x - med) / max(sc, EPS)
    z[~np.isfinite(z)] = np.nan
    return z


def positive_jump_max(v: np.ndarray, *, normalize: bool) -> Tuple[float, float]:
    x = chain_z(v) if normalize else np.asarray(v, dtype=np.float64)
    if x.size < 2:
        return float("nan"), float("nan")
    d = x[1:] - x[:-1]
    if not np.isfinite(d).any():
        return float("nan"), float("nan")
    k = int(np.nanargmax(d))
    return float(d[k]), float((k + 1) / max(1, x.size - 1))


def abs_jump_max(v: np.ndarray, *, normalize: bool) -> Tuple[float, float]:
    x = chain_z(v) if normalize else np.asarray(v, dtype=np.float64)
    if x.size < 2:
        return float("nan"), float("nan")
    d = np.abs(x[1:] - x[:-1])
    if not np.isfinite(d).any():
        return float("nan"), float("nan")
    k = int(np.nanargmax(d))
    return float(d[k]), float((k + 1) / max(1, x.size - 1))


def local_contrast_max(v: np.ndarray, *, width: int, normalize: bool) -> Tuple[float, float]:
    x = chain_z(v) if normalize else np.asarray(v, dtype=np.float64)
    T = x.size
    if T < 2:
        return float("nan"), float("nan")
    vals: List[float] = []
    pos: List[float] = []
    for t in range(1, T):
        pre = x[max(0, t - width) : t]
        post = x[t : min(T, t + width)]
        if pre.size == 0 or post.size == 0:
            continue
        vals.append(safe_mean(post) - safe_mean(pre))
        pos.append(t / max(1, T - 1))
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).any():
        return float("nan"), float("nan")
    k = int(np.nanargmax(a))
    return float(a[k]), float(pos[k])


def sequence_summaries(v: np.ndarray, *, sign: float, width: int) -> Dict[str, float]:
    x = sign * np.asarray(v, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {
            "level_mean": float("nan"),
            "level_late": float("nan"),
            "level_max": float("nan"),
            "jump_max": float("nan"),
            "jump_max_pos": float("nan"),
            "zjump_max": float("nan"),
            "zjump_max_pos": float("nan"),
            "abs_zjump_max": float("nan"),
            "abs_zjump_max_pos": float("nan"),
            "contrast_max": float("nan"),
            "contrast_max_pos": float("nan"),
            "zcontrast_max": float("nan"),
            "zcontrast_max_pos": float("nan"),
            "volatility": float("nan"),
        }
    jm, jp = positive_jump_max(x, normalize=False)
    zjm, zjp = positive_jump_max(x, normalize=True)
    ajm, ajp = abs_jump_max(x, normalize=True)
    cm, cp = local_contrast_max(x, width=width, normalize=False)
    zcm, zcp = local_contrast_max(x, width=width, normalize=True)
    d = np.diff(chain_z(x))
    return {
        "level_mean": safe_mean(x),
        "level_late": safe_mean(x[window_mask(x.size, "late")]),
        "level_max": float(np.nanmax(x)) if np.isfinite(x).any() else float("nan"),
        "jump_max": jm,
        "jump_max_pos": jp,
        "zjump_max": zjm,
        "zjump_max_pos": zjp,
        "abs_zjump_max": ajm,
        "abs_zjump_max_pos": ajp,
        "contrast_max": cm,
        "contrast_max_pos": cp,
        "zcontrast_max": zcm,
        "zcontrast_max_pos": zcp,
        "volatility": safe_mean(np.abs(d)),
    }


def step_resultant(H: np.ndarray) -> float:
    X = np.asarray(H, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return float("nan")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    U = X / np.maximum(norms, EPS)
    return float(np.linalg.norm(np.nanmean(U, axis=0)))


def add_cloud_sequences(data: np.lib.npyio.NpzFile, seqs: List[Dict[str, np.ndarray]]) -> None:
    if "sv_clouds" not in data.files or "cloud_sizes" not in data.files:
        return
    for i, (obj, size_obj) in enumerate(zip(data["sv_clouds"], data["cloud_sizes"])):
        if obj is None or size_obj is None:
            continue
        C = np.asarray(obj, dtype=np.float64)
        sizes = np.asarray(size_obj, dtype=int).reshape(-1)
        if C.ndim != 3 or C.shape[0] == 0 or sizes.size == 0:
            continue
        # Multisample files currently store one full-dim cloud layer.  If there
        # are more layers later, use the first as the stable backward-compatible
        # default for this audit.
        X = C[:, 0, :]
        cursor = 0
        resultants: List[float] = []
        norms: List[float] = []
        tok_counts: List[float] = []
        for s in sizes:
            s = int(s)
            if s <= 0:
                continue
            H = X[cursor : cursor + s]
            cursor += s
            resultants.append(step_resultant(H))
            norms.append(float(np.nanmean(np.linalg.norm(H, axis=1))) if H.size else float("nan"))
            tok_counts.append(float(s))
        if resultants:
            r = np.asarray(resultants, dtype=np.float64)
            seqs[i]["cloud_resultant"] = r
            seqs[i]["cloud_spread"] = 1.0 - r
            seqs[i]["cloud_norm"] = np.asarray(norms, dtype=np.float64)
            seqs[i]["step_token_count"] = np.asarray(tok_counts, dtype=np.float64)


def add_matrix_sequences(data: np.lib.npyio.NpzFile, seqs: List[Dict[str, np.ndarray]], *, bands: Sequence[str]) -> None:
    for metric in ("pr", "ae"):
        key = f"sv_{metric}_step_exp"
        if key not in data.files:
            continue
        raw = data[key]
        first = np.asarray(raw[0], dtype=np.float64)
        if first.ndim != 2:
            continue
        for i, obj in enumerate(raw):
            M = np.asarray(obj, dtype=np.float64)
            if M.ndim != 2:
                continue
            for band in bands:
                cols = band_cols(M.shape[1], band)
                valid = cols[cols < M.shape[1]]
                if valid.size == 0:
                    valid = np.arange(M.shape[1])
                seqs[i][f"{metric}_{band}"] = np.nanmean(M[:, valid], axis=1)


def add_logit_sequences(data: np.lib.npyio.NpzFile, seqs: List[Dict[str, np.ndarray]]) -> None:
    for key, name in (
        ("sv_out_entropy", "out_entropy"),
        ("sv_out_committal", "out_committal"),
        ("sv_tok_entropy", "tok_entropy"),
        ("sv_tok_committal", "tok_committal"),
    ):
        if key not in data.files:
            continue
        for i, obj in enumerate(data[key]):
            v = np.asarray(obj, dtype=np.float64).reshape(-1)
            if v.size:
                seqs[i][name] = v


def fit_diag_step_mahal(
    data: np.lib.npyio.NpzFile,
    *,
    mask: np.ndarray,
    y_err: np.ndarray,
    band: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if "sv_vec_step_exp" not in data.files or not bool(data.get("sv_vectors_stored", np.array(False))):
        return None
    raw = data["sv_vec_step_exp"]
    first = np.asarray(raw[0], dtype=np.float64)
    if first.ndim != 3:
        return None
    cols = band_cols(first.shape[1], band)
    rows: List[np.ndarray] = []
    pos: List[float] = []
    for i, obj in enumerate(raw):
        if not mask[i] or y_err[i] != 0:
            continue
        V = np.asarray(obj, dtype=np.float64)
        valid = cols[cols < V.shape[1]]
        if valid.size == 0:
            valid = np.arange(V.shape[1])
        S = np.nanmean(V[:, valid, :], axis=1)
        T = S.shape[0]
        for t in range(T):
            if np.isfinite(S[t]).all():
                rows.append(S[t])
                pos.append(t / max(1, T - 1))
    if len(rows) < 20:
        return None
    X = np.vstack(rows)
    mu = np.nanmean(X, axis=0)
    var = np.nanvar(X, axis=0) + 1e-6
    return mu, var, np.asarray(pos, dtype=np.float64)


def add_mahal_sequences(
    data: np.lib.npyio.NpzFile,
    seqs: List[Dict[str, np.ndarray]],
    *,
    mask: np.ndarray,
    y_err: np.ndarray,
    bands: Sequence[str],
) -> None:
    if "sv_vec_step_exp" not in data.files or not bool(data.get("sv_vectors_stored", np.array(False))):
        return
    raw = data["sv_vec_step_exp"]
    for band in bands:
        fit = fit_diag_step_mahal(data, mask=mask, y_err=y_err, band=band)
        if fit is None:
            continue
        mu, var, _ = fit
        for i, obj in enumerate(raw):
            V = np.asarray(obj, dtype=np.float64)
            if V.ndim != 3:
                continue
            cols = band_cols(V.shape[1], band)
            valid = cols[cols < V.shape[1]]
            if valid.size == 0:
                valid = np.arange(V.shape[1])
            S = np.nanmean(V[:, valid, :], axis=1)
            seqs[i][f"mahal_{band}"] = np.nansum(((S - mu) ** 2) / var, axis=1)
            seqs[i][f"vec_norm_{band}"] = np.linalg.norm(S, axis=1)


def label_policy(data: np.lib.npyio.NpzFile, policy: str) -> Tuple[np.ndarray, np.ndarray, str]:
    n = len(data["problem_ids"])
    if policy == "answer":
        return (data["is_correct"].astype(int) == 0).astype(int), np.ones(n, bool), "answer incorrect"
    if policy == "strict":
        return (data["is_correct_strict"].astype(int) == 0).astype(int), np.ones(n, bool), "strict incorrect"
    if policy == "answer_format_ok":
        return (
            (data["is_correct"].astype(int) == 0).astype(int),
            data["format_ok"].astype(bool),
            "answer incorrect among format-ok samples",
        )
    raise ValueError(policy)


def signal_sign(name: str) -> float:
    # Risk-high orientation.  The same signal is often also represented by its
    # complement, e.g. cloud_spread = 1 - cloud_resultant.
    if "resultant" in name:
        return -1.0
    return 1.0


def build_base_sequences(data: np.lib.npyio.NpzFile, *, bands: Sequence[str]) -> List[Dict[str, np.ndarray]]:
    n = len(data["problem_ids"])
    seqs: List[Dict[str, np.ndarray]] = [dict() for _ in range(n)]
    add_matrix_sequences(data, seqs, bands=bands)
    add_logit_sequences(data, seqs)
    add_cloud_sequences(data, seqs)
    if "n_steps" in data.files:
        for i, T in enumerate(data["n_steps"].astype(int)):
            seqs[i]["step_pos"] = np.arange(int(T), dtype=np.float64) / max(1, int(T) - 1)
    return seqs


def problem_groups(problem_ids: np.ndarray, y_err: np.ndarray, mask: np.ndarray, min_per_class: int) -> List[np.ndarray]:
    groups: List[np.ndarray] = []
    for p in np.unique(problem_ids[mask]):
        idx = np.where(mask & (problem_ids == p))[0]
        if np.sum(y_err[idx] == 1) >= min_per_class and np.sum(y_err[idx] == 0) >= min_per_class:
            groups.append(idx)
    return groups


def within_pair_auroc(groups: Sequence[np.ndarray], vals: np.ndarray, y_err: np.ndarray) -> Tuple[float, int]:
    conc = 0.0
    pairs = 0
    for idx in groups:
        err = [float(vals[i]) for i in idx if y_err[i] == 1 and np.isfinite(vals[i])]
        cor = [float(vals[i]) for i in idx if y_err[i] == 0 and np.isfinite(vals[i])]
        for a in err:
            for b in cor:
                conc += 1.0 if a > b else (0.5 if a == b else 0.0)
        pairs += len(err) * len(cor)
    return (conc / pairs if pairs else float("nan")), int(pairs)


def paired_delta(groups: Sequence[np.ndarray], vals: np.ndarray, y_err: np.ndarray) -> Dict[str, Any]:
    ds: List[float] = []
    for idx in groups:
        err = [float(vals[i]) for i in idx if y_err[i] == 1 and np.isfinite(vals[i])]
        cor = [float(vals[i]) for i in idx if y_err[i] == 0 and np.isfinite(vals[i])]
        if err and cor:
            ds.append(float(np.mean(err) - np.mean(cor)))
    a = np.asarray(ds, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    return {
        "n": int(a.size),
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "fraction_positive": float((a > 0).mean()),
    }


def evaluate_score(
    name: str,
    vals: np.ndarray,
    pos: np.ndarray,
    *,
    y_err: np.ndarray,
    mask: np.ndarray,
    groups: Sequence[np.ndarray],
) -> Dict[str, Any]:
    m = mask & np.isfinite(vals)
    err = vals[m & (y_err == 1)]
    cor = vals[m & (y_err == 0)]
    w, pairs = within_pair_auroc(groups, vals, y_err)
    return {
        "score": name,
        "n": int(m.sum()),
        "n_error": int((m & (y_err == 1)).sum()),
        "n_correct": int((m & (y_err == 0)).sum()),
        "error": descriptive(err),
        "correct": descriptive(cor),
        "cross_auroc_error_high": auroc_signed(err, cor),
        "within_pair_auroc_error_high": w,
        "best_direction_within": max(w, 1.0 - w) if np.isfinite(w) else float("nan"),
        "within_pairs": pairs,
        "paired_delta_error_minus_correct": paired_delta(groups, vals, y_err),
        "argpos_error": descriptive(pos[m & (y_err == 1)]),
        "argpos_correct": descriptive(pos[m & (y_err == 0)]),
    }


def profile_bins(values: List[np.ndarray], labels: np.ndarray, mask: np.ndarray, *, nbins: int) -> Dict[str, Any]:
    bins = np.linspace(0.0, 1.0, nbins + 1)
    out: Dict[str, Any] = {"bins": bins.tolist(), "error": [], "correct": []}
    for cls_name, cls_val in (("error", 1), ("correct", 0)):
        cls_rows: List[List[float]] = [[] for _ in range(nbins)]
        for i, v in enumerate(values):
            if not mask[i] or labels[i] != cls_val:
                continue
            x = np.asarray(v, dtype=np.float64)
            T = x.size
            if T == 0:
                continue
            pos = np.arange(T) / max(1, T - 1)
            for b in range(nbins):
                if b == nbins - 1:
                    mm = (pos >= bins[b]) & (pos <= bins[b + 1])
                else:
                    mm = (pos >= bins[b]) & (pos < bins[b + 1])
                if mm.any():
                    cls_rows[b].append(safe_mean(x[mm]))
        out[cls_name] = [descriptive(row) for row in cls_rows]
    return out


def run_policy(
    data: np.lib.npyio.NpzFile,
    base_seqs: List[Dict[str, np.ndarray]],
    *,
    policy: str,
    bands: Sequence[str],
    width: int,
    min_per_class: int,
    nbins: int,
) -> Dict[str, Any]:
    problem_ids = data["problem_ids"].astype(int)
    y_err, mask, desc = label_policy(data, policy)
    groups = problem_groups(problem_ids, y_err, mask, min_per_class)

    # Copy shallow dicts so policy-specific mahal channels do not leak across
    # policies.
    seqs = [dict(x) for x in base_seqs]
    add_mahal_sequences(data, seqs, mask=mask, y_err=y_err, bands=bands)

    channel_names = sorted({k for s in seqs for k in s.keys() if k != "step_pos"})
    rows: List[Dict[str, Any]] = []
    profiles: Dict[str, Any] = {}
    score_names = [
        "level_mean",
        "level_late",
        "level_max",
        "jump_max",
        "zjump_max",
        "abs_zjump_max",
        "contrast_max",
        "zcontrast_max",
        "volatility",
    ]
    pos_names = {
        "jump_max": "jump_max_pos",
        "zjump_max": "zjump_max_pos",
        "abs_zjump_max": "abs_zjump_max_pos",
        "contrast_max": "contrast_max_pos",
        "zcontrast_max": "zcontrast_max_pos",
    }

    for ch in channel_names:
        vals_by_score = {nm: np.full(len(seqs), np.nan, dtype=np.float64) for nm in score_names}
        pos_by_score = {nm: np.full(len(seqs), np.nan, dtype=np.float64) for nm in score_names}
        raw_profile_values: List[np.ndarray] = []
        sign = signal_sign(ch)
        for i, s in enumerate(seqs):
            if ch not in s:
                raw_profile_values.append(np.asarray([], dtype=np.float64))
                continue
            v = np.asarray(s[ch], dtype=np.float64).reshape(-1)
            raw_profile_values.append(sign * v)
            summ = sequence_summaries(v, sign=sign, width=width)
            for nm in score_names:
                vals_by_score[nm][i] = summ[nm]
                pos_by_score[nm][i] = summ.get(pos_names.get(nm, ""), float("nan"))
        profiles[ch] = profile_bins(raw_profile_values, y_err, mask, nbins=nbins)
        for nm in score_names:
            rows.append(
                evaluate_score(
                    f"{ch}.{nm}",
                    vals_by_score[nm],
                    pos_by_score[nm],
                    y_err=y_err,
                    mask=mask,
                    groups=groups,
                )
            )

    multi = build_multichannel_scores(seqs, channel_names, width=width)
    for nm, (vals, pos) in multi.items():
        rows.append(evaluate_score(nm, vals, pos, y_err=y_err, mask=mask, groups=groups))

    rows.sort(
        key=lambda r: (
            np.nan_to_num(r["best_direction_within"], nan=-1.0),
            np.nan_to_num(r["cross_auroc_error_high"], nan=-1.0),
        ),
        reverse=True,
    )
    return {
        "description": desc,
        "n_samples": int(mask.sum()),
        "n_error": int(y_err[mask].sum()),
        "n_correct": int(mask.sum() - y_err[mask].sum()),
        "n_contrastive_problems": int(len(groups)),
        "channels": channel_names,
        "results": rows,
        "profiles": profiles,
    }


def build_multichannel_scores(
    seqs: Sequence[Dict[str, np.ndarray]],
    channels: Sequence[str],
    *,
    width: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out = {
        "multi.zjump_l2_max": (np.full(len(seqs), np.nan), np.full(len(seqs), np.nan)),
        "multi.zjump_pos_sum_max": (np.full(len(seqs), np.nan), np.full(len(seqs), np.nan)),
        "multi.zcontrast_l2_max": (np.full(len(seqs), np.nan), np.full(len(seqs), np.nan)),
    }
    risk_channels = [c for c in channels if c != "step_token_count"]
    for i, s in enumerate(seqs):
        per: List[np.ndarray] = []
        minT = None
        for ch in risk_channels:
            if ch not in s:
                continue
            v = signal_sign(ch) * np.asarray(s[ch], dtype=np.float64).reshape(-1)
            if v.size < 2:
                continue
            z = chain_z(v)
            if not np.isfinite(z).any():
                continue
            per.append(z)
            minT = z.size if minT is None else min(minT, z.size)
        if not per or minT is None or minT < 2:
            continue
        Z = np.vstack([z[:minT] for z in per])
        d = Z[:, 1:] - Z[:, :-1]
        if d.size == 0:
            continue
        l2 = np.sqrt(np.nansum(d ** 2, axis=0))
        pos_sum = np.nansum(np.maximum(d, 0.0), axis=0)
        k = int(np.nanargmax(l2))
        out["multi.zjump_l2_max"][0][i] = float(l2[k])
        out["multi.zjump_l2_max"][1][i] = float((k + 1) / max(1, minT - 1))
        k2 = int(np.nanargmax(pos_sum))
        out["multi.zjump_pos_sum_max"][0][i] = float(pos_sum[k2])
        out["multi.zjump_pos_sum_max"][1][i] = float((k2 + 1) / max(1, minT - 1))

        contrasts: List[float] = []
        cpos: List[float] = []
        for t in range(1, minT):
            pre = Z[:, max(0, t - width) : t]
            post = Z[:, t : min(minT, t + width)]
            if pre.size == 0 or post.size == 0:
                continue
            diff = np.nanmean(post, axis=1) - np.nanmean(pre, axis=1)
            contrasts.append(float(np.sqrt(np.nansum(diff ** 2))))
            cpos.append(t / max(1, minT - 1))
        if contrasts:
            a = np.asarray(contrasts, dtype=np.float64)
            k3 = int(np.nanargmax(a))
            out["multi.zcontrast_l2_max"][0][i] = float(a[k3])
            out["multi.zcontrast_l2_max"][1][i] = float(cpos[k3])
    return out


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    bands = [x.strip() for x in args.bands.split(",") if x.strip()]
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    base = build_base_sequences(data, bands=bands)
    return {
        "meta": {
            "input": os.path.abspath(path),
            "basename": os.path.basename(path),
            "prompt_style": str(data["prompt_style"]) if "prompt_style" in data.files else "unknown",
            "step_split": str(data["step_split"]) if "step_split" in data.files else "unknown",
            "model": str(data["model_name"]) if "model_name" in data.files else "unknown",
            "bands": bands,
            "width": int(args.width),
            "notes": {
                "headline": "same-problem paired AUROC for local non-cumulative temporal scores",
                "limitation": "same-problem data has final correctness labels but no first-error step labels",
                "orientation": "scores are risk-high; cloud_resultant is multiplied by -1",
            },
        },
        "policies": {
            pol: run_policy(
                data,
                base,
                policy=pol,
                bands=bands,
                width=args.width,
                min_per_class=args.min_per_class,
                nbins=args.profile_bins,
            )
            for pol in policies
        },
    }


def write_outputs(res: Mapping[str, Any], output_dir: str, top: int) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stem = f"multisample_temporal_rupture_{os.path.splitext(str(res['meta']['basename']))[0]}"
    jp = os.path.join(output_dir, stem + ".json")
    mp = os.path.join(output_dir, stem + ".md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(finite_json(res), f, indent=2, ensure_ascii=False)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(f"# Multisample Temporal Rupture Audit: {res['meta']['basename']}\n\n")
        f.write("## Result Analysis\n\n")
        f.write("- This audit reads each sampled answer as a full per-step signal trajectory.\n")
        f.write("- Scores ending in `jump`, `contrast`, or `volatility` are local/non-cumulative dynamic scores.\n")
        f.write("- Scores ending in `level` are static summaries and are included as baselines.\n")
        f.write("- Same-problem paired AUROC is the main metric; cross-problem AUROC is only context.\n\n")
        for pol, sec in res["policies"].items():
            f.write(f"### {pol}\n\n")
            f.write(
                f"{sec['n_error']} error / {sec['n_correct']} correct samples; "
                f"{sec['n_contrastive_problems']} contrastive problems.\n\n"
            )
            f.write("| score | within | cross | err med | cor med | delta med | err argpos | cor argpos |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in sec["results"][:top]:
                dlt = r["paired_delta_error_minus_correct"]
                f.write(
                    f"| {r['score']} | {r['within_pair_auroc_error_high']:.3f} | "
                    f"{r['cross_auroc_error_high']:.3f} | "
                    f"{r['error'].get('median', float('nan')):.3f} | "
                    f"{r['correct'].get('median', float('nan')):.3f} | "
                    f"{dlt.get('median', float('nan')):.3f} | "
                    f"{r['argpos_error'].get('median', float('nan')):.3f} | "
                    f"{r['argpos_correct'].get('median', float('nan')):.3f} |\n"
                )
            f.write("\n")
        f.write("## Follow-Up Research Direction\n\n")
        f.write("- If local dynamic scores do not beat level summaries under `answer_format_ok`, current data does not yet support a rupture detector.\n")
        f.write("- If a jump/contrast score survives, inspect its profile and argpos before using it for online intervention.\n")
        f.write("- The next stronger test needs first-error labels or an automatic step verifier to align predicted rupture points with actual wrong steps.\n\n")
        f.write("## Optimization Suggestions\n\n")
        f.write("- Keep level, jump, contrast, and volatility separated; do not use CUSUM as an onset detector.\n")
        f.write("- Always compare against `cloud_spread.level_*`, because spread is the current static baseline.\n")
        f.write("- Treat large cross-problem but weak within-problem scores as difficulty/format proxies.\n")
    return jp, mp


def print_report(res: Mapping[str, Any], top: int) -> None:
    meta = res["meta"]
    print(f"\n===== multisample temporal rupture audit | {meta['basename']} =====")
    print(f"prompt={meta['prompt_style']} step_split={meta['step_split']} model={meta['model']}")
    for pol, sec in res["policies"].items():
        print(f"\n[{pol}] err={sec['n_error']} cor={sec['n_correct']} contrastive={sec['n_contrastive_problems']}")
        for r in sec["results"][:top]:
            dlt = r["paired_delta_error_minus_correct"]
            print(
                f"  {r['score']:34s} within {r['within_pair_auroc_error_high']:.3f} "
                f"cross {r['cross_auroc_error_high']:.3f} "
                f"err_med {r['error'].get('median', float('nan')):.3f} "
                f"cor_med {r['correct'].get('median', float('nan')):.3f} "
                f"delta {dlt.get('median', float('nan')):+.3f} "
                f"argpos_e {r['argpos_error'].get('median', float('nan')):.2f} "
                f"argpos_c {r['argpos_correct'].get('median', float('nan')):.2f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", default="outputs/multisample_temporal_rupture")
    ap.add_argument("--policies", default="answer_format_ok")
    ap.add_argument("--bands", default="mid,deep,all")
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--profile_bins", type=int, default=5)
    ap.add_argument("--top", type=int, default=24)
    args = ap.parse_args()

    res = run(args.input, args)
    jp, mp = write_outputs(res, args.output_dir, args.top)
    print_report(res, args.top)
    print(f"\nwrote {jp} and {mp}")


if __name__ == "__main__":
    main()
