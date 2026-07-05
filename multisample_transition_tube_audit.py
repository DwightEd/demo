#!/usr/bin/env python3
"""Low-rank transition-tube audit for same-problem multi-sampling data.

Hypothesis:
  Correct reasoning chains have step-to-step transitions that live in a
  constrained low-rank tube.  Incorrect chains either leave that tube
  (off-tube residual) or need more transition directions (higher effective
  rank / rank-to-explain energy).

This is a diagnostic over existing `sv_vec_step_exp` payloads.  It reports:
  1. global_crossfit: fit on correct chains from training problems only.
  2. problem_oracle: fit on correct samples from the same problem only.

The second setting is not deployable, but it is the cleanest check of whether
same-question correct samples define a low-rank transition manifold at all.
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


def descriptive(xs: Iterable[float]) -> Dict[str, Any]:
    a = np.asarray([x for x in xs if np.isfinite(x)], dtype=np.float64)
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
    err = np.asarray([x for x in err if np.isfinite(x)], dtype=np.float64)
    cor = np.asarray([x for x in cor if np.isfinite(x)], dtype=np.float64)
    if err.size == 0 or cor.size == 0:
        return float("nan")
    vals = np.concatenate([err, cor])
    labels = np.concatenate([np.ones(err.size), np.zeros(cor.size)])
    order = np.argsort(vals, kind="mergesort")
    ranks = _avg_ranks(vals[order])
    full = np.empty_like(ranks)
    full[order] = ranks
    sum_pos = full[labels == 1].sum()
    U = sum_pos - err.size * (err.size + 1) / 2.0
    return float(U / (err.size * cor.size))


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


def safe_mean(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def robust_scale(x: np.ndarray) -> Tuple[float, float]:
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


def normalize_steps(X: np.ndarray, mode: str) -> np.ndarray:
    Y = np.asarray(X, dtype=np.float64)
    if mode == "none":
        return Y
    if mode == "l2":
        n = np.linalg.norm(Y, axis=1, keepdims=True)
        return Y / np.maximum(n, EPS)
    if mode == "center_chain":
        return Y - np.nanmean(Y, axis=0, keepdims=True)
    raise ValueError(mode)


def step_sequences(data: np.lib.npyio.NpzFile, *, band: str, normalize: str) -> List[np.ndarray]:
    if "sv_vec_step_exp" not in data.files or not bool(data.get("sv_vectors_stored", np.array(False))):
        raise SystemExit("need sv_vectors_stored=True and sv_vec_step_exp in the npz")
    raw = data["sv_vec_step_exp"]
    out: List[np.ndarray] = []
    for obj in raw:
        V = np.asarray(obj, dtype=np.float64)
        if V.ndim != 3:
            out.append(np.empty((0, 0), dtype=np.float64))
            continue
        cols = band_cols(V.shape[1], band)
        cols = cols[cols < V.shape[1]]
        if cols.size == 0:
            cols = np.arange(V.shape[1])
        X = np.nanmean(V[:, cols, :], axis=1)
        good = np.isfinite(X).all(axis=1)
        X = X[good]
        out.append(normalize_steps(X, normalize) if X.size else X)
    return out


def transitions(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2 or X.shape[0] < 2:
        return np.empty((0, X.shape[1] if X.ndim == 2 else 0), dtype=np.float64)
    return X[1:] - X[:-1]


def collect_transitions(seqs: Sequence[np.ndarray], idxs: Sequence[int]) -> np.ndarray:
    rows: List[np.ndarray] = []
    for i in idxs:
        D = transitions(seqs[int(i)])
        if D.size:
            rows.append(D)
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(rows)


@dataclass
class TransitionTube:
    mu: np.ndarray
    comps: np.ndarray
    comps_max: np.ndarray
    rank: int
    residual_center: float
    residual_scale: float
    train_residual_mean: float


def fit_tube(
    D: np.ndarray,
    *,
    rank: int,
    max_rank: int,
    energy: float,
    min_transitions: int,
) -> Optional[TransitionTube]:
    if D.ndim != 2 or D.shape[0] < min_transitions or D.shape[1] == 0:
        return None
    D = D[np.isfinite(D).all(axis=1)]
    if D.shape[0] < min_transitions:
        return None
    mu = D.mean(axis=0)
    X = D - mu
    try:
        _, s, vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if vt.size == 0:
        return None
    max_r = int(min(max_rank, vt.shape[0], vt.shape[1]))
    if rank > 0:
        r = int(min(rank, max_r))
    else:
        e = s ** 2
        total = float(e.sum())
        if total <= EPS:
            r = 1
        else:
            r = int(np.searchsorted(np.cumsum(e) / total, energy) + 1)
            r = int(min(max(1, r), max_r))
    comps = vt[:r]
    comps_max = vt[:max_r]
    R = transition_residuals(D, mu, comps)
    med, scale = robust_scale(R["off_norm"])
    return TransitionTube(
        mu=mu,
        comps=comps,
        comps_max=comps_max,
        rank=r,
        residual_center=med,
        residual_scale=scale,
        train_residual_mean=float(np.nanmean(R["off_norm"])),
    )


def transition_residuals(D: np.ndarray, mu: np.ndarray, comps: np.ndarray) -> Dict[str, np.ndarray]:
    X = D - mu
    if comps.size:
        P = (X @ comps.T) @ comps
    else:
        P = np.zeros_like(X)
    off = X - P
    total = np.linalg.norm(X, axis=1)
    off_norm = np.linalg.norm(off, axis=1)
    return {
        "off_norm": off_norm,
        "off_ratio": off_norm / np.maximum(total, EPS),
        "total_norm": total,
    }


def rank_stats(D: np.ndarray, tube: TransitionTube, *, energy_q: float) -> Dict[str, float]:
    if D.size == 0:
        return {"rank_energy": float("nan"), "transition_eff_rank": float("nan"), "off_energy_ratio": float("nan")}
    X = D - tube.mu
    U = tube.comps_max
    if U.size == 0:
        return {"rank_energy": float("nan"), "transition_eff_rank": float("nan"), "off_energy_ratio": float("nan")}
    C = X @ U.T
    in_e = np.sum(C ** 2, axis=0)
    P = C @ U
    off_e = float(np.sum((X - P) ** 2))
    e = np.concatenate([in_e, np.array([off_e])])
    total = float(e.sum())
    if total <= EPS:
        return {"rank_energy": 0.0, "transition_eff_rank": 0.0, "off_energy_ratio": 0.0}
    cum = np.cumsum(in_e) / total
    if np.any(cum >= energy_q):
        rank_energy = float(np.searchsorted(cum, energy_q) + 1)
    else:
        rank_energy = float(U.shape[0] + 1)
    p = e[e > EPS] / total
    eff = float(np.exp(-np.sum(p * np.log(p)))) if p.size else 0.0
    return {
        "rank_energy": rank_energy,
        "transition_eff_rank": eff,
        "off_energy_ratio": off_e / total,
    }


def score_sequence(X: np.ndarray, tube: TransitionTube, *, rank_energy: float) -> Dict[str, float]:
    D = transitions(X)
    if D.size == 0:
        return {}
    R = transition_residuals(D, tube.mu, tube.comps)
    z = (R["off_norm"] - tube.residual_center) / max(tube.residual_scale, EPS)
    m_late = window_mask(z.size, "late")
    max_idx = int(np.nanargmax(z)) if np.isfinite(z).any() else -1
    out = {
        "off_mean": safe_mean(R["off_norm"]),
        "off_late": safe_mean(R["off_norm"][m_late]),
        "off_max": float(np.nanmax(R["off_norm"])) if np.isfinite(R["off_norm"]).any() else float("nan"),
        "off_ratio_mean": safe_mean(R["off_ratio"]),
        "off_ratio_late": safe_mean(R["off_ratio"][m_late]),
        "off_ratio_max": float(np.nanmax(R["off_ratio"])) if np.isfinite(R["off_ratio"]).any() else float("nan"),
        "off_z_mean": safe_mean(z),
        "off_z_late": safe_mean(z[m_late]),
        "off_z_max": float(np.nanmax(z)) if np.isfinite(z).any() else float("nan"),
        "off_z_max_pos": float((max_idx + 1) / max(1, X.shape[0] - 1)) if max_idx >= 0 else float("nan"),
        "tube_rank": float(tube.rank),
    }
    out.update(rank_stats(D, tube, energy_q=rank_energy))
    return out


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


def group_folds(groups: np.ndarray, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    k = min(k, len(uniq))
    fold_of = {int(g): i % k for i, g in enumerate(uniq)}
    f = np.array([fold_of[int(g)] for g in groups])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def same_problem_groups(problem_ids: np.ndarray, y_err: np.ndarray, mask: np.ndarray, min_per_class: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for p in np.unique(problem_ids[mask]):
        idx = np.where(mask & (problem_ids == p))[0]
        if np.sum(y_err[idx] == 1) >= min_per_class and np.sum(y_err[idx] == 0) >= min_per_class:
            out.append(idx)
    return out


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
    return {"n": int(a.size), "median": float(q50), "q25": float(q25), "q75": float(q75), "fraction_positive": float((a > 0).mean())}


def collect_score_rows(
    prefix: str,
    scores: Dict[str, np.ndarray],
    positions: Dict[str, np.ndarray],
    *,
    y_err: np.ndarray,
    mask: np.ndarray,
    groups: Sequence[np.ndarray],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, vals in scores.items():
        m = mask & np.isfinite(vals)
        err = vals[m & (y_err == 1)]
        cor = vals[m & (y_err == 0)]
        w, pairs = within_pair_auroc(groups, vals, y_err)
        row = {
            "score": f"{prefix}.{name}",
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
        }
        if name in positions:
            pos = positions[name]
            row["argpos_error"] = descriptive(pos[m & (y_err == 1)])
            row["argpos_correct"] = descriptive(pos[m & (y_err == 0)])
        rows.append(row)
    return rows


def global_crossfit_scores(
    seqs: Sequence[np.ndarray],
    problem_ids: np.ndarray,
    y_err: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    names = [
        "off_mean",
        "off_late",
        "off_max",
        "off_ratio_mean",
        "off_ratio_late",
        "off_ratio_max",
        "off_z_mean",
        "off_z_late",
        "off_z_max",
        "rank_energy",
        "transition_eff_rank",
        "off_energy_ratio",
        "tube_rank",
    ]
    scores = {n: np.full(len(seqs), np.nan, dtype=np.float64) for n in names}
    pos = {"off_z_max": np.full(len(seqs), np.nan, dtype=np.float64)}
    ranks: List[int] = []
    idx_all = np.where(mask)[0]
    folds = group_folds(problem_ids[mask], args.folds, args.seed)
    for tr_rel, te_rel in folds:
        tr_idx = idx_all[tr_rel]
        te_idx = idx_all[te_rel]
        correct = tr_idx[y_err[tr_idx] == 0]
        D = collect_transitions(seqs, correct)
        tube = fit_tube(D, rank=args.rank, max_rank=args.max_rank, energy=args.energy, min_transitions=args.min_transitions)
        if tube is None:
            continue
        ranks.append(tube.rank)
        for i in te_idx:
            s = score_sequence(seqs[int(i)], tube, rank_energy=args.rank_energy)
            for n in names:
                if n in s:
                    scores[n][i] = s[n]
            if "off_z_max_pos" in s:
                pos["off_z_max"][i] = s["off_z_max_pos"]
    meta = {"fold_ranks": ranks, "mean_rank": float(np.mean(ranks)) if ranks else None}
    return scores, pos, meta


def problem_oracle_scores(
    seqs: Sequence[np.ndarray],
    problem_ids: np.ndarray,
    y_err: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    names = [
        "off_mean",
        "off_late",
        "off_max",
        "off_ratio_mean",
        "off_ratio_late",
        "off_ratio_max",
        "off_z_mean",
        "off_z_late",
        "off_z_max",
        "rank_energy",
        "transition_eff_rank",
        "off_energy_ratio",
        "tube_rank",
    ]
    scores = {n: np.full(len(seqs), np.nan, dtype=np.float64) for n in names}
    pos = {"off_z_max": np.full(len(seqs), np.nan, dtype=np.float64)}
    used = 0
    ranks: List[int] = []
    for p in np.unique(problem_ids[mask]):
        idx = np.where(mask & (problem_ids == p))[0]
        correct = idx[y_err[idx] == 0]
        if len(correct) < args.oracle_min_correct:
            continue
        D = collect_transitions(seqs, correct)
        tube = fit_tube(D, rank=args.rank, max_rank=args.max_rank, energy=args.energy, min_transitions=args.oracle_min_transitions)
        if tube is None:
            continue
        used += 1
        ranks.append(tube.rank)
        for i in idx:
            s = score_sequence(seqs[int(i)], tube, rank_energy=args.rank_energy)
            for n in names:
                if n in s:
                    scores[n][i] = s[n]
            if "off_z_max_pos" in s:
                pos["off_z_max"][i] = s["off_z_max_pos"]
    meta = {"problem_tubes_used": used, "mean_rank": float(np.mean(ranks)) if ranks else None}
    return scores, pos, meta


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    problem_ids = data["problem_ids"].astype(int)
    seqs = step_sequences(data, band=args.band, normalize=args.normalize)
    policies: Dict[str, Any] = {}
    for pol in [x.strip() for x in args.policies.split(",") if x.strip()]:
        y_err, mask, desc = label_policy(data, pol)
        groups = same_problem_groups(problem_ids, y_err, mask, args.min_per_class)
        rows: List[Dict[str, Any]] = []
        diagnostics: Dict[str, Any] = {}
        gs, gp, gm = global_crossfit_scores(seqs, problem_ids, y_err, mask, args)
        diagnostics["global_crossfit"] = gm
        rows.extend(collect_score_rows("global", gs, gp, y_err=y_err, mask=mask, groups=groups))
        if args.problem_oracle:
            oscores, opos, ometa = problem_oracle_scores(seqs, problem_ids, y_err, mask, args)
            diagnostics["problem_oracle"] = ometa
            rows.extend(collect_score_rows("oracle", oscores, opos, y_err=y_err, mask=mask, groups=groups))
        rows.sort(
            key=lambda r: (
                np.nan_to_num(r["best_direction_within"], nan=-1.0),
                np.nan_to_num(r["cross_auroc_error_high"], nan=-1.0),
            ),
            reverse=True,
        )
        policies[pol] = {
            "description": desc,
            "n_samples": int(mask.sum()),
            "n_error": int(y_err[mask].sum()),
            "n_correct": int(mask.sum() - y_err[mask].sum()),
            "n_contrastive_problems": int(len(groups)),
            "diagnostics": diagnostics,
            "results": rows,
        }
    return {
        "meta": {
            "input": os.path.abspath(path),
            "basename": os.path.basename(path),
            "prompt_style": str(data["prompt_style"]) if "prompt_style" in data.files else "unknown",
            "step_split": str(data["step_split"]) if "step_split" in data.files else "unknown",
            "model": str(data["model_name"]) if "model_name" in data.files else "unknown",
            "band": args.band,
            "normalize": args.normalize,
            "rank": int(args.rank),
            "max_rank": int(args.max_rank),
            "energy": float(args.energy),
            "rank_energy": float(args.rank_energy),
            "notes": {
                "global": "correct-chain transition tube fitted on training problems only",
                "oracle": "same-problem correct samples define a diagnostic tube; not deployable",
                "rank_energy": "number of correct-basis directions needed to explain this chain's transition energy; max_rank+1 means off-basis energy remains too large",
            },
        },
        "policies": policies,
    }


def write_outputs(res: Mapping[str, Any], output_dir: str, top: int) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stem = f"multisample_transition_tube_{os.path.splitext(str(res['meta']['basename']))[0]}_{res['meta']['band']}_{res['meta']['normalize']}"
    jp = os.path.join(output_dir, stem + ".json")
    mp = os.path.join(output_dir, stem + ".md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(finite_json(res), f, indent=2, ensure_ascii=False)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(f"# Multisample Transition Tube Audit: {res['meta']['basename']}\n\n")
        f.write("## Result Analysis\n\n")
        f.write("- `global.*` scores are deployable-style OOF scores: train correct transition tube on other problems, score held-out problems.\n")
        f.write("- `oracle.*` scores are diagnostic: same-problem correct samples define the tube, so they test whether a question-specific correct manifold exists.\n")
        f.write("- Signed AUROC > 0.5 means error samples have larger off-tube/rank values.\n\n")
        for pol, sec in res["policies"].items():
            f.write(f"### {pol}\n\n")
            f.write(
                f"{sec['n_error']} error / {sec['n_correct']} correct samples; "
                f"{sec['n_contrastive_problems']} contrastive problems.\n\n"
            )
            f.write(f"Diagnostics: `{json.dumps(finite_json(sec['diagnostics']), ensure_ascii=False)}`\n\n")
            f.write("| score | within | cross | err med | cor med | delta med | err argpos | cor argpos |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in sec["results"][:top]:
                dlt = r["paired_delta_error_minus_correct"]
                ae = r.get("argpos_error", {})
                ac = r.get("argpos_correct", {})
                f.write(
                    f"| {r['score']} | {r['within_pair_auroc_error_high']:.3f} | "
                    f"{r['cross_auroc_error_high']:.3f} | "
                    f"{r['error'].get('median', float('nan')):.3f} | "
                    f"{r['correct'].get('median', float('nan')):.3f} | "
                    f"{dlt.get('median', float('nan')):.3f} | "
                    f"{ae.get('median', float('nan')):.3f} | {ac.get('median', float('nan')):.3f} |\n"
                )
            f.write("\n")
        f.write("## Follow-Up Research Direction\n\n")
        f.write("- If oracle scores are strong but global scores are weak, the manifold is question-conditioned and needs prompt/anchor conditioning.\n")
        f.write("- If rank/off-energy scores beat off-residual level scores, the useful signal is extra transition directions rather than larger residual magnitude.\n")
        f.write("- If both global and oracle scores are weak, current step vectors do not support the low-rank transition-tube hypothesis.\n\n")
        f.write("## Optimization Suggestions\n\n")
        f.write("- Compare `none` vs `l2` normalization; norm effects can masquerade as tube residuals.\n")
        f.write("- Inspect `off_z_max` argpos before treating a score as online-triggerable.\n")
        f.write("- Do not claim first-error localization from this dataset, which has only final-answer labels.\n")
    return jp, mp


def print_report(res: Mapping[str, Any], top: int) -> None:
    meta = res["meta"]
    print(f"\n===== transition tube audit | {meta['basename']} | {meta['band']} {meta['normalize']} =====")
    print(f"prompt={meta['prompt_style']} step_split={meta['step_split']} model={meta['model']}")
    for pol, sec in res["policies"].items():
        print(f"\n[{pol}] err={sec['n_error']} cor={sec['n_correct']} contrastive={sec['n_contrastive_problems']}")
        print(f"diagnostics {finite_json(sec['diagnostics'])}")
        for r in sec["results"][:top]:
            dlt = r["paired_delta_error_minus_correct"]
            ae = r.get("argpos_error", {})
            ac = r.get("argpos_correct", {})
            print(
                f"  {r['score']:32s} within {r['within_pair_auroc_error_high']:.3f} "
                f"cross {r['cross_auroc_error_high']:.3f} "
                f"err_med {r['error'].get('median', float('nan')):.3f} "
                f"cor_med {r['correct'].get('median', float('nan')):.3f} "
                f"delta {dlt.get('median', float('nan')):+.3f} "
                f"argpos_e {ae.get('median', float('nan')):.2f} "
                f"argpos_c {ac.get('median', float('nan')):.2f}"
            )


def make_selftest(path: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n_problems = 24
    k = 5
    dim = 12
    layers = 4
    good_basis = rng.normal(size=(3, dim))
    good_basis, _ = np.linalg.qr(good_basis.T)
    good_basis = good_basis.T
    bad_dir = rng.normal(size=dim)
    bad_dir = bad_dir - (bad_dir @ good_basis.T) @ good_basis
    bad_dir = bad_dir / np.linalg.norm(bad_dir)
    vecs = []
    pids = []
    sample_idx = []
    is_correct = []
    fmt = []
    steps = []
    for p in range(n_problems):
        for s in range(k):
            T = int(rng.integers(5, 9))
            correct = s < 3
            x = rng.normal(scale=0.05, size=dim)
            chain = []
            for t in range(T):
                delta = rng.normal(size=3) @ good_basis * 0.3
                if (not correct) and t >= T // 2:
                    delta = delta + bad_dir * 0.8
                x = x + delta
                chain.append(np.tile(x, (layers, 1)) + rng.normal(scale=0.01, size=(layers, dim)))
            vecs.append(np.asarray(chain, dtype=np.float32))
            pids.append(p)
            sample_idx.append(s)
            is_correct.append(1 if correct else 0)
            fmt.append(1)
            steps.append(T)
    np.savez(
        path,
        problem_ids=np.asarray(pids, int),
        sample_idx=np.asarray(sample_idx, int),
        is_correct=np.asarray(is_correct, int),
        is_correct_strict=np.asarray(is_correct, int),
        format_ok=np.asarray(fmt, int),
        n_steps=np.asarray(steps, int),
        sv_vectors_stored=np.asarray(True),
        sv_vec_step_exp=np.asarray(vecs, dtype=object),
        prompt_style=np.asarray("selftest"),
        step_split=np.asarray("synthetic"),
        model_name=np.asarray("synthetic"),
    )


def assert_selftest(res: Mapping[str, Any]) -> None:
    rows = res["policies"]["answer_format_ok"]["results"]
    best = max(float(r["within_pair_auroc_error_high"]) for r in rows if r["score"].startswith("oracle.off"))
    if best < 0.75:
        raise SystemExit(f"selftest failed: oracle off-tube signal too weak ({best:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output_dir", default="outputs/multisample_transition_tube")
    ap.add_argument("--policies", default="answer_format_ok")
    ap.add_argument("--band", default="mid")
    ap.add_argument("--normalize", default="l2", choices=["none", "l2", "center_chain"])
    ap.add_argument("--rank", type=int, default=0, help="fixed tube rank; 0 selects by --energy")
    ap.add_argument("--energy", type=float, default=0.90)
    ap.add_argument("--rank_energy", type=float, default=0.90)
    ap.add_argument("--max_rank", type=int, default=32)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--min_transitions", type=int, default=40)
    ap.add_argument("--problem_oracle", action="store_true", default=True)
    ap.add_argument("--no_problem_oracle", action="store_false", dest="problem_oracle")
    ap.add_argument("--oracle_min_correct", type=int, default=2)
    ap.add_argument("--oracle_min_transitions", type=int, default=8)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "transition_tube_selftest.npz")
            make_selftest(path, seed=args.seed)
            res = run(path, args)
            assert_selftest(res)
    else:
        if not args.input:
            raise SystemExit("pass --input or --selftest")
        res = run(args.input, args)
    jp, mp = write_outputs(res, args.output_dir, args.top)
    print_report(res, args.top)
    print(f"\nwrote {jp} and {mp}")


if __name__ == "__main__":
    main()
