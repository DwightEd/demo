#!/usr/bin/env python3
"""Same-problem path-kernel audit for trajectory shape differences.

This script tests a narrower question than the regime HSMM:

    Holding the problem fixed, are correct and incorrect trajectory *paths*
    drawn from different distributions after removing problem-level difficulty
    and static chain-level levels?

It uses two complementary tools:

1. Conditional MMD: a within-problem two-sample test over whole paths with
   label permutations performed inside each problem.
2. Cross-fitted kernel witnesses: train on held-out problems only, then score
   samples by similarity to error vs correct path distributions.

The audit reports level and shape-only representations.  Shape-only subtracts
each chain's own per-channel mean, so a positive result there cannot be reduced
to static spread/entropy levels.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from multisample_temporal_rupture_audit import (
    descriptive,
    finite_json,
    paired_delta,
    within_pair_auroc,
)
from trajectory_difference_audit import prepare_policy_data, path_signature_features


EPS = 1e-12
DEFAULT_CHANNELS = "cloud_spread,out_entropy,pr_mid,ae_mid"


@dataclass
class PathData:
    problem_ids: np.ndarray
    y_err: np.ndarray
    groups: List[np.ndarray]
    channels: List[str]
    grid: np.ndarray
    tensor: np.ndarray
    channel_coverage: Dict[str, float]


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


def local_groups(problem_ids: np.ndarray, y: np.ndarray, min_per_class: int) -> List[np.ndarray]:
    out = []
    for p in np.unique(problem_ids):
        idx = np.where(problem_ids == p)[0]
        if np.sum(y[idx] == 1) >= min_per_class and np.sum(y[idx] == 0) >= min_per_class:
            out.append(idx)
    return out


def load_path_data(path: str, args: argparse.Namespace) -> PathData:
    data = np.load(path, allow_pickle=True)
    requested = [x.strip() for x in args.channels.split(",") if x.strip()]
    bands = [x.strip() for x in args.bands.split(",") if x.strip()]
    pd = prepare_policy_data(
        data,
        policy=args.policy,
        bands=bands,
        requested_channels=requested,
        min_per_class=args.min_per_class,
        min_channel_coverage=args.min_channel_coverage,
        require_channels=args.require_channels,
        grid_size=args.grid,
        include_mahal=False,
    )
    idx = np.where(pd.contrast_mask)[0]
    if args.max_problems:
        keep = np.unique(pd.problem_ids[idx])[: int(args.max_problems)]
        idx = idx[np.isin(pd.problem_ids[idx], keep)]
    if idx.size < 20:
        raise SystemExit("not enough contrastive same-problem samples")
    pids = pd.problem_ids[idx]
    y = pd.y_err[idx].astype(int)
    groups = local_groups(pids, y, args.min_per_class)
    keep_mask = np.zeros(len(idx), dtype=bool)
    for g in groups:
        keep_mask[g] = True
    idx2 = np.where(keep_mask)[0]
    return PathData(
        problem_ids=pids[idx2],
        y_err=y[idx2],
        groups=local_groups(pids[idx2], y[idx2], args.min_per_class),
        channels=pd.channels,
        grid=pd.grid,
        tensor=pd.tensor[idx][idx2],
        channel_coverage=pd.channel_coverage,
    )


def problem_center(X: np.ndarray, problem_ids: np.ndarray) -> np.ndarray:
    Z = np.full_like(X, np.nan, dtype=np.float64)
    for p in np.unique(problem_ids):
        idx = np.where(problem_ids == p)[0]
        block = X[idx]
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(block, axis=0)
            mad = np.nanmedian(np.abs(block - med), axis=0) * 1.4826
            sd = np.nanstd(block, axis=0)
        scale = np.where(np.isfinite(mad) & (mad > EPS), mad, sd)
        scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
        med = np.where(np.isfinite(med), med, 0.0)
        Z[idx] = (block - med) / scale
    return Z


def detrend_chain(path: np.ndarray, mode: str) -> np.ndarray:
    X = np.asarray(path, dtype=np.float64)
    out = np.array(X, copy=True)
    if mode == "none":
        return out
    t = np.linspace(-1.0, 1.0, X.shape[1])
    for c in range(X.shape[0]):
        y = X[c]
        m = np.isfinite(y)
        if not m.any():
            continue
        if mode == "mean":
            out[c, m] = y[m] - np.mean(y[m])
        elif mode == "linear":
            if m.sum() >= 2:
                A = np.column_stack([t[m], np.ones(m.sum())])
                beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
                out[c, m] = y[m] - A @ beta
            else:
                out[c, m] = y[m] - np.mean(y[m])
        else:
            raise ValueError(mode)
    return out


def representation_tensor(rd: PathData, *, mode: str, censor_frac: float) -> np.ndarray:
    X = problem_center(rd.tensor, rd.problem_ids)
    L = max(2, int(math.floor(censor_frac * X.shape[2])))
    X = X[:, :, :L]
    if mode in ("shape_mean", "shape_linear"):
        detrend = "mean" if mode == "shape_mean" else "linear"
        X = np.stack([detrend_chain(x, detrend) for x in X], axis=0)
    elif mode != "level":
        raise ValueError(mode)
    return np.where(np.isfinite(X), X, 0.0)


def flatten_paths(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64).reshape(X.shape[0], -1)


def dct_features(X: np.ndarray, n_components: int) -> np.ndarray:
    N, C, T = X.shape
    K = int(min(max(1, n_components), T))
    n = np.arange(T, dtype=np.float64)
    basis = []
    for k in range(K):
        b = np.cos(np.pi * (n + 0.5) * k / T)
        b /= max(np.linalg.norm(b), EPS)
        basis.append(b)
    B = np.stack(basis, axis=0)  # K x T
    coeff = np.einsum("nct,kt->nck", X, B)
    return coeff.reshape(N, C * K)


def signature_features(X: np.ndarray, grid: np.ndarray, order: int) -> np.ndarray:
    feats = []
    g = np.linspace(0.0, 1.0, X.shape[2])
    for i in range(X.shape[0]):
        feats.append(path_signature_features(X[i], g, order=order))
    return np.asarray(feats, dtype=np.float64)


def robust_scale_fit(X: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = np.zeros(X.shape[1], dtype=np.float64)
    scale = np.ones(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        center[j], scale[j] = robust_center_scale(X[idx, j])
    return center, scale


def robust_scale_apply(X: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    Z = (np.asarray(X, dtype=np.float64) - center) / np.maximum(scale, EPS)
    return np.where(np.isfinite(Z), Z, 0.0)


def median_bandwidth(Z: np.ndarray, max_points: int, seed: int) -> float:
    X = np.asarray(Z, dtype=np.float64)
    if len(X) > max_points:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), size=max_points, replace=False)]
    if len(X) < 2:
        return 1.0
    diff = X[:, None, :] - X[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    vals = d2[np.triu_indices(len(X), k=1)]
    vals = vals[np.isfinite(vals) & (vals > EPS)]
    if vals.size == 0:
        return 1.0
    return float(np.sqrt(0.5 * np.median(vals)))


def rbf_kernel(A: np.ndarray, B: np.ndarray, bandwidth: float) -> np.ndarray:
    bw2 = max(float(bandwidth) ** 2, EPS)
    d2 = (
        np.sum(A * A, axis=1)[:, None]
        + np.sum(B * B, axis=1)[None, :]
        - 2.0 * (A @ B.T)
    )
    return np.exp(-0.5 * np.maximum(d2, 0.0) / bw2)


def kernel_witness_scores(
    Xfeat: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    kernel: str,
    seed: int,
    bandwidth_points: int,
) -> np.ndarray:
    scores = np.full(len(y), np.nan, dtype=np.float64)
    for fold, (tr, te) in enumerate(group_folds(groups, folds, seed)):
        if len(np.unique(y[tr])) < 2:
            continue
        center, scale = robust_scale_fit(Xfeat, tr)
        Ztr = robust_scale_apply(Xfeat[tr], center, scale)
        Zte = robust_scale_apply(Xfeat[te], center, scale)
        err = Ztr[y[tr] == 1]
        cor = Ztr[y[tr] == 0]
        if len(err) == 0 or len(cor) == 0:
            continue
        if kernel == "linear":
            w = np.mean(err, axis=0) - np.mean(cor, axis=0)
            scores[te] = Zte @ w
        elif kernel == "rbf":
            bw = median_bandwidth(Ztr, bandwidth_points, seed + fold)
            scores[te] = rbf_kernel(Zte, err, bw).mean(axis=1) - rbf_kernel(Zte, cor, bw).mean(axis=1)
        else:
            raise ValueError(kernel)
    return scores


def mmd_biased(K: np.ndarray, y: np.ndarray) -> float:
    e = np.where(y == 1)[0]
    c = np.where(y == 0)[0]
    if len(e) == 0 or len(c) == 0:
        return float("nan")
    return float(K[np.ix_(e, e)].mean() + K[np.ix_(c, c)].mean() - 2.0 * K[np.ix_(e, c)].mean())


def conditional_mmd_test(
    Xfeat: np.ndarray,
    y: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    permutations: int,
    seed: int,
    bandwidth_points: int,
) -> Dict[str, Any]:
    center, scale = robust_scale_fit(Xfeat, np.arange(len(y)))
    Z = robust_scale_apply(Xfeat, center, scale)
    bw = median_bandwidth(Z, bandwidth_points, seed)
    K = rbf_kernel(Z, Z, bw)
    weights = []
    vals = []
    for idx in groups:
        yy = y[idx]
        ne = int((yy == 1).sum())
        nc = int((yy == 0).sum())
        if ne and nc:
            vals.append(mmd_biased(K[np.ix_(idx, idx)], yy))
            weights.append(ne * nc)
    if not vals:
        return {"mmd": None, "p_ge": None, "bandwidth": bw}
    obs = float(np.average(vals, weights=weights))
    if permutations <= 0:
        return {"mmd": obs, "p_ge": None, "bandwidth": bw}
    rng = np.random.default_rng(seed)
    null = []
    yperm = np.array(y, copy=True)
    for _ in range(int(permutations)):
        pvals = []
        for idx in groups:
            yperm[idx] = rng.permutation(y[idx])
            pvals.append(mmd_biased(K[np.ix_(idx, idx)], yperm[idx]))
        null.append(float(np.average(pvals, weights=weights)))
    null_arr = np.asarray(null, dtype=np.float64)
    return {
        "mmd": obs,
        "p_ge": float((1.0 + np.sum(null_arr >= obs)) / (len(null_arr) + 1.0)),
        "bandwidth": bw,
        "null_mean": float(null_arr.mean()),
        "null_q95": float(np.quantile(null_arr, 0.95)),
    }


def score_permutation_pvalue(
    scores: np.ndarray,
    y: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    permutations: int,
    seed: int,
) -> Dict[str, Any]:
    obs, pairs = within_pair_auroc(groups, scores, y)
    if not np.isfinite(obs) or permutations <= 0:
        return {"observed": obs, "pairs": pairs, "p_ge": None}
    rng = np.random.default_rng(seed)
    vals = []
    yp = np.array(y, copy=True)
    for _ in range(int(permutations)):
        for idx in groups:
            yp[idx] = rng.permutation(y[idx])
        val, _ = within_pair_auroc(groups, scores, yp)
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return {"observed": obs, "pairs": pairs, "p_ge": None}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "observed": obs,
        "pairs": pairs,
        "p_ge": float((1.0 + np.sum(arr >= obs)) / (len(arr) + 1.0)),
        "null_mean": float(arr.mean()),
        "null_q95": float(np.quantile(arr, 0.95)),
    }


def evaluate_scores(
    scores: Mapping[str, np.ndarray],
    y: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    permutations: int,
    seed: int,
) -> Dict[str, Any]:
    rows = {}
    for name, s in scores.items():
        s = np.asarray(s, dtype=np.float64)
        au, pairs = within_pair_auroc(groups, s, y)
        rows[name] = {
            "same_problem_paired_auroc": au,
            "n_pairs": pairs,
            "paired_delta": paired_delta(groups, s, y),
            "score_error": descriptive(s[y == 1]),
            "score_correct": descriptive(s[y == 0]),
            "permutation": score_permutation_pvalue(
                s,
                y,
                groups,
                permutations=permutations,
                seed=seed + len(rows),
            ),
        }
    return rows


def static_scores(rd: PathData) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for ci, ch in enumerate(rd.channels):
        vals = rd.tensor[:, ci, :]
        out[f"mean:{ch}"] = np.nanmean(vals, axis=1)
        out[f"max:{ch}"] = np.nanmax(vals, axis=1)
        L = max(2, int(math.floor(0.8 * vals.shape[1])))
        out[f"mean80:{ch}"] = np.nanmean(vals[:, :L], axis=1)
    out["mean:all_channels"] = np.nanmean(rd.tensor, axis=(1, 2))
    return out


def build_features_for_rep(
    X: np.ndarray,
    rd: PathData,
    *,
    rep: str,
    dct_components: int,
    signature_order: int,
) -> Dict[str, np.ndarray]:
    feats = {
        f"{rep}:flat": flatten_paths(X),
        f"{rep}:dct": dct_features(X, dct_components),
    }
    if signature_order > 0:
        feats[f"{rep}:signature"] = signature_features(X, rd.grid[: X.shape[2]], signature_order)
    return feats


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    rd = load_path_data(path, args)
    reps: Dict[str, np.ndarray] = {}
    rep_specs = []
    for mode in ("level", "shape_mean", "shape_linear"):
        rep_specs.append((mode, 1.0, mode))
        rep_specs.append((mode, args.censor_frac, f"{mode}_censor{int(round(100 * args.censor_frac))}"))
    for mode, frac, name in rep_specs:
        reps[name] = representation_tensor(rd, mode=mode, censor_frac=frac)

    feature_sets: Dict[str, np.ndarray] = {}
    for name, X in reps.items():
        feature_sets.update(
            build_features_for_rep(
                X,
                rd,
                rep=name,
                dct_components=args.dct_components,
                signature_order=args.signature_order,
            )
        )

    scores: Dict[str, np.ndarray] = {}
    for fname, F in feature_sets.items():
        scores[f"witness_linear:{fname}"] = kernel_witness_scores(
            F,
            rd.y_err,
            rd.problem_ids,
            folds=args.folds,
            kernel="linear",
            seed=args.seed,
            bandwidth_points=args.bandwidth_points,
        )
        if fname.endswith(":flat"):
            scores[f"witness_rbf:{fname}"] = kernel_witness_scores(
                F,
                rd.y_err,
                rd.problem_ids,
                folds=args.folds,
                kernel="rbf",
                seed=args.seed + 17,
                bandwidth_points=args.bandwidth_points,
            )

    scores.update(static_scores(rd))
    score_table = evaluate_scores(scores, rd.y_err, rd.groups, permutations=args.score_permutations, seed=args.seed)
    mmd_rows = {}
    for fname, F in feature_sets.items():
        if fname.endswith(":flat") or fname.endswith(":dct"):
            mmd_rows[fname] = conditional_mmd_test(
                F,
                rd.y_err,
                rd.groups,
                permutations=args.mmd_permutations,
                seed=args.seed + len(mmd_rows) * 101,
                bandwidth_points=args.bandwidth_points,
            )

    best_static = max(
        ((name, row) for name, row in score_table.items() if not name.startswith("witness_")),
        key=lambda kv: np.nan_to_num(kv[1]["same_problem_paired_auroc"], nan=-1.0),
    )
    best_witness = max(
        ((name, row) for name, row in score_table.items() if name.startswith("witness_")),
        key=lambda kv: np.nan_to_num(kv[1]["same_problem_paired_auroc"], nan=-1.0),
    )
    best_shape = max(
        ((name, row) for name, row in score_table.items() if "shape_" in name),
        key=lambda kv: np.nan_to_num(kv[1]["same_problem_paired_auroc"], nan=-1.0),
    )
    return {
        "meta": {
            "input": os.path.abspath(path),
            "basename": os.path.basename(path),
            "policy": args.policy,
            "n_samples": int(len(rd.y_err)),
            "n_error": int(rd.y_err.sum()),
            "n_correct": int(len(rd.y_err) - rd.y_err.sum()),
            "n_contrastive_problems": int(len(rd.groups)),
            "channels": rd.channels,
            "channel_coverage": rd.channel_coverage,
            "grid_size": int(args.grid),
            "notes": {
                "same_problem_control": "All headline metrics are same-problem paired AUROC or within-problem label permutations.",
                "shape_only": "shape_mean/shape_linear subtract each chain's own level/trend before path comparison.",
                "no_pos_feature": "Position is only used as the path parameter, not as a predictive feature.",
            },
        },
        "headline": {
            "best_static_name": best_static[0],
            "best_static_same_problem_auroc": best_static[1]["same_problem_paired_auroc"],
            "best_witness_name": best_witness[0],
            "best_witness_same_problem_auroc": best_witness[1]["same_problem_paired_auroc"],
            "best_witness_minus_static": best_witness[1]["same_problem_paired_auroc"] - best_static[1]["same_problem_paired_auroc"],
            "best_shape_name": best_shape[0],
            "best_shape_same_problem_auroc": best_shape[1]["same_problem_paired_auroc"],
            "best_shape_minus_static": best_shape[1]["same_problem_paired_auroc"] - best_static[1]["same_problem_paired_auroc"],
        },
        "scores": score_table,
        "conditional_mmd": mmd_rows,
    }


def write_markdown(path: str, res: Mapping[str, Any]) -> None:
    h = res["headline"]
    lines = [
        f"# Within-Problem Path Kernel Audit: `{res['meta']['basename']}`",
        "",
        "## Headline",
        "",
        f"- Best static: `{h['best_static_name']}` = `{h['best_static_same_problem_auroc']:.3f}`",
        f"- Best witness: `{h['best_witness_name']}` = `{h['best_witness_same_problem_auroc']:.3f}`",
        f"- Witness minus static: `{h['best_witness_minus_static']:+.3f}`",
        f"- Best shape-only witness: `{h['best_shape_name']}` = `{h['best_shape_same_problem_auroc']:.3f}`",
        f"- Shape-only minus static: `{h['best_shape_minus_static']:+.3f}`",
        "",
        "## Top Scores",
        "",
        "| score | same-problem AUROC | pairs | permutation p_ge |",
        "|---|---:|---:|---:|",
    ]
    rows = sorted(
        res["scores"].items(),
        key=lambda kv: np.nan_to_num(kv[1]["same_problem_paired_auroc"], nan=-1.0),
        reverse=True,
    )
    for name, row in rows[:24]:
        p = row.get("permutation", {}).get("p_ge")
        ptxt = "" if p is None else f"{p:.4f}"
        lines.append(f"| `{name}` | {row['same_problem_paired_auroc']:.3f} | {row['n_pairs']} | {ptxt} |")
    lines += [
        "",
        "## Conditional MMD",
        "",
        "| representation | MMD | p_ge | null mean |",
        "|---|---:|---:|---:|",
    ]
    for name, row in sorted(res["conditional_mmd"].items()):
        p = row.get("p_ge")
        ptxt = "" if p is None else f"{p:.4f}"
        mmd = row.get("mmd")
        nmean = row.get("null_mean")
        lines.append(
            f"| `{name}` | {mmd if mmd is not None else float('nan'):.4f} | {ptxt} | "
            f"{nmean if nmean is not None else float('nan'):.4f} |"
        )
    lines.append("")
    lines.append("Shape-only rows are the key test for non-static trajectory information.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_outputs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    jpath = os.path.join(output_dir, f"{stem}.json")
    mpath = os.path.join(output_dir, f"{stem}.md")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(finite_json(res), f, ensure_ascii=False, indent=2)
    write_markdown(mpath, finite_json(res))
    return jpath, mpath


def print_result(res: Mapping[str, Any]) -> None:
    h = res["headline"]
    meta = res["meta"]
    print(f"\n===== within-problem path kernel | {meta['basename']} =====")
    print(
        f"samples {meta['n_samples']} | err {meta['n_error']} | "
        f"problems {meta['n_contrastive_problems']} | channels {meta['channels']}"
    )
    print(
        f"best static {h['best_static_name']}={h['best_static_same_problem_auroc']:.3f} | "
        f"best witness {h['best_witness_name']}={h['best_witness_same_problem_auroc']:.3f} | "
        f"shape {h['best_shape_name']}={h['best_shape_same_problem_auroc']:.3f}"
    )
    print("\nTop scores:")
    rows = sorted(
        res["scores"].items(),
        key=lambda kv: np.nan_to_num(kv[1]["same_problem_paired_auroc"], nan=-1.0),
        reverse=True,
    )
    for name, row in rows[:14]:
        print(f"  {name:42s} AUROC {row['same_problem_paired_auroc']:.3f} pairs={row['n_pairs']}")
    print("\nConditional MMD:")
    for name, row in sorted(res["conditional_mmd"].items()):
        p = row.get("p_ge")
        ptxt = "NA" if p is None else f"{p:.4f}"
        print(f"  {name:30s} MMD {row.get('mmd'):.4f} p_ge={ptxt}")


def _cloud_for_spread(spread: float, *, n_tok: int, dim: int) -> np.ndarray:
    r = float(np.clip(1.0 - spread, 0.02, 0.98))
    orth = math.sqrt(max(0.0, 1.0 - r * r))
    H = np.zeros((n_tok, 1, dim), dtype=np.float32)
    for i in range(n_tok):
        sign = 1.0 if i % 2 == 0 else -1.0
        ax = 1 + ((i // 2) % max(1, dim - 1))
        H[i, 0, 0] = r
        H[i, 0, ax] = sign * orth
    return H


def _object_array(xs: Sequence[object]) -> np.ndarray:
    out = np.empty(len(xs), dtype=object)
    out[:] = list(xs)
    return out


def make_selftest(path: str, *, seed: int = 0, n_problems: int = 18, samples_per_problem: int = 4) -> None:
    rng = np.random.default_rng(seed)
    correct = np.array([-0.8, -0.4, 0.0, 0.4, 0.8, 0.4, 0.0, -0.4], dtype=float)
    error = np.array([-0.8, 0.4, 0.8, 0.0, -0.4, 0.0, 0.4, -0.4], dtype=float)
    layers = 33
    dim = 12
    n_tok = 6
    ids = []
    pids = []
    sample_idx = []
    is_correct = []
    n_steps = []
    clouds = []
    sizes_all = []
    entropy_rows = []
    pr_rows = []
    ae_rows = []
    for p in range(n_problems):
        offset = rng.normal(scale=0.025)
        for s in range(samples_per_problem):
            err = int(s >= samples_per_problem // 2)
            proto = error if err else correct
            # tiny jitter keeps the shape signal but avoids exact duplicates.
            latent = proto + rng.normal(scale=0.015, size=len(proto))
            latent -= latent.mean()
            spread = np.clip(0.40 + 0.12 * latent + offset, 0.08, 0.84)
            entropy = 0.55 + 0.18 * latent + offset
            pr = 3.0 + 0.60 * latent
            ae = 0.35 + 0.18 * latent
            ids.append(f"p{p}_s{s}")
            pids.append(p)
            sample_idx.append(s)
            is_correct.append(0 if err else 1)
            n_steps.append(len(proto))
            sizes = np.full(len(proto), n_tok, dtype=np.int32)
            sizes_all.append(sizes)
            clouds.append(np.concatenate([_cloud_for_spread(v, n_tok=n_tok, dim=dim) for v in spread], axis=0))
            entropy_rows.append(entropy.astype(np.float32))
            pr_rows.append(np.tile(pr[:, None], (1, layers)).astype(np.float32))
            ae_rows.append(np.tile(ae[:, None], (1, layers)).astype(np.float32))
    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=object),
        problem_ids=np.asarray(pids, dtype=np.int32),
        sample_idx=np.asarray(sample_idx, dtype=np.int32),
        is_correct=np.asarray(is_correct, dtype=np.int32),
        is_correct_strict=np.asarray(is_correct, dtype=np.int32),
        format_ok=np.ones(len(ids), dtype=np.int32),
        n_steps=np.asarray(n_steps, dtype=np.int32),
        cloud_sizes=_object_array(sizes_all),
        sv_clouds=_object_array(clouds),
        sv_out_entropy=_object_array(entropy_rows),
        sv_pr_step_exp=_object_array(pr_rows),
        sv_ae_step_exp=_object_array(ae_rows),
        model_name=np.asarray("path-kernel-selftest"),
        prompt_style=np.asarray("path-kernel-selftest"),
        step_split=np.asarray("path-kernel-selftest"),
    )


def assert_selftest(res: Mapping[str, Any]) -> None:
    h = res["headline"]
    if h["best_shape_same_problem_auroc"] < 0.85:
        raise SystemExit("selftest failed: shape-only witness did not recover path-order signal")
    best_mmd = min((r.get("p_ge", 1.0) for k, r in res["conditional_mmd"].items() if "shape_" in k), default=1.0)
    if best_mmd > 0.10:
        raise SystemExit("selftest failed: shape-only MMD was not significant")


def main() -> None:
    ap = argparse.ArgumentParser(description="Same-problem path-kernel trajectory shape audit")
    ap.add_argument("--input", default=None)
    ap.add_argument("--policy", default="answer_format_ok", choices=["answer", "strict", "answer_format_ok"])
    ap.add_argument("--channels", default=DEFAULT_CHANNELS)
    ap.add_argument("--bands", default="mid")
    ap.add_argument("--require_channels", action="store_true")
    ap.add_argument("--min_channel_coverage", type=float, default=0.80)
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--max_problems", type=int, default=0)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--censor_frac", type=float, default=0.80)
    ap.add_argument("--dct_components", type=int, default=8)
    ap.add_argument("--signature_order", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bandwidth_points", type=int, default=600)
    ap.add_argument("--score_permutations", type=int, default=200)
    ap.add_argument("--mmd_permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs/within_problem_path_kernel")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            npz = os.path.join(td, "path_kernel_selftest.npz")
            make_selftest(npz, seed=args.seed)
            args.input = npz
            res = run(npz, args)
            assert_selftest(res)
            print_result(res)
            jpath, mpath = write_outputs(res, args.output_dir, "selftest")
            print(f"\nselftest passed; saved: {jpath} | {mpath}")
        return
    if not args.input:
        raise SystemExit("provide --input or --selftest")
    res = run(args.input, args)
    print_result(res)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    jpath, mpath = write_outputs(res, args.output_dir, stem)
    print(f"\nsaved: {jpath} | {mpath}")


if __name__ == "__main__":
    main()

