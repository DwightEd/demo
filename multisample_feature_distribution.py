#!/usr/bin/env python3
"""Distribution audit for previously useful signals on same-problem samples.

This is a companion to `multisample_data_audit.py`.  It focuses on the feature
families that looked useful in earlier experiments and reports whether their
error/correct separation survives same-problem paired evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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


def descriptive(x: Sequence[float]) -> Dict[str, Any]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
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


def auroc_signed(err: Sequence[float], cor: Sequence[float]) -> float:
    err = np.asarray(err, dtype=np.float64)
    cor = np.asarray(cor, dtype=np.float64)
    err = err[np.isfinite(err)]
    cor = cor[np.isfinite(cor)]
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


def cohen_d(err: Sequence[float], cor: Sequence[float]) -> float:
    err = np.asarray(err, dtype=np.float64)
    cor = np.asarray(cor, dtype=np.float64)
    err = err[np.isfinite(err)]
    cor = cor[np.isfinite(cor)]
    if err.size < 2 or cor.size < 2:
        return float("nan")
    pooled = ((err.size - 1) * err.var(ddof=1) + (cor.size - 1) * cor.var(ddof=1)) / (err.size + cor.size - 2)
    if pooled <= 0:
        return 0.0
    return float((err.mean() - cor.mean()) / math.sqrt(pooled))


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


def matrix_window_feature(obj: Any, cols: np.ndarray, window: str) -> float:
    M = np.asarray(obj, dtype=np.float64)
    if M.ndim != 2:
        return float("nan")
    valid = cols[cols < M.shape[1]]
    if valid.size == 0:
        valid = np.arange(M.shape[1])
    m = window_mask(M.shape[0], window)
    return safe_mean(M[m][:, valid])


def seq_window_feature(obj: Any, window: str) -> float:
    v = np.asarray(obj, dtype=np.float64).reshape(-1)
    m = window_mask(v.size, window)
    return safe_mean(v[m])


def seq_max(obj: Any) -> float:
    v = np.asarray(obj, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else float("nan")


def compute_vector_mahal(data: np.lib.npyio.NpzFile, band: str) -> np.ndarray:
    if not bool(data.get("sv_vectors_stored", np.array(False))):
        return np.full(len(data["problem_ids"]), np.nan)
    key = "sv_vec_step_exp"
    if key not in data.files:
        return np.full(len(data["problem_ids"]), np.nan)
    raw = data[key]
    first = np.asarray(raw[0], dtype=np.float64)
    if first.ndim != 3:
        return np.full(len(raw), np.nan)
    cols = band_cols(first.shape[1], band)
    vec = np.full((len(raw), first.shape[2]), np.nan, dtype=np.float64)
    for i, obj in enumerate(raw):
        V = np.asarray(obj, dtype=np.float64)
        valid = cols[cols < V.shape[1]]
        if valid.size == 0:
            valid = np.arange(V.shape[1])
        step_mean = np.nanmean(V[:, valid, :], axis=1)
        m = window_mask(step_mean.shape[0], "late")
        vec[i] = np.nanmean(step_mean[m], axis=0)
    ok = np.isfinite(vec).all(axis=1)
    out = np.full(len(raw), np.nan, dtype=np.float64)
    if ok.sum() < 3:
        return out
    mu = vec[ok].mean(axis=0)
    vr = vec[ok].var(axis=0) + 1e-6
    out[ok] = ((vec[ok] - mu) ** 2 / vr).sum(axis=1)
    return out


def cloud_step_resultant(H: np.ndarray) -> float:
    X = np.asarray(H, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return float("nan")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    U = X / np.maximum(norms, EPS)
    return float(np.linalg.norm(np.nanmean(U, axis=0)))


def compute_cloud_features(data: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    if "sv_clouds" not in data.files or "cloud_sizes" not in data.files:
        return {}
    clouds = data["sv_clouds"]
    sizes = data["cloud_sizes"]
    n = len(clouds)
    features = {
        "cloud_resultant_full": np.full(n, np.nan),
        "cloud_resultant_late": np.full(n, np.nan),
        "cloud_resultant_min": np.full(n, np.nan),
        "cloud_spread_full": np.full(n, np.nan),
        "cloud_spread_late": np.full(n, np.nan),
        "cloud_spread_max": np.full(n, np.nan),
        "cloud_mean_norm_late": np.full(n, np.nan),
    }
    for i, (obj, sz_obj) in enumerate(zip(clouds, sizes)):
        if obj is None or sz_obj is None:
            continue
        C = np.asarray(obj, dtype=np.float64)
        sz = np.asarray(sz_obj, dtype=int).reshape(-1)
        if C.ndim != 3 or C.shape[0] == 0 or sz.size == 0:
            continue
        # Use the first stored cloud layer by default. Current DATA.md says the
        # multisample files have one full-dim cloud layer.
        X = C[:, 0, :]
        vals: List[float] = []
        norms: List[float] = []
        cursor = 0
        for s in sz:
            if s <= 0:
                continue
            H = X[cursor : cursor + int(s)]
            cursor += int(s)
            if H.size == 0:
                continue
            vals.append(cloud_step_resultant(H))
            norms.append(float(np.nanmean(np.linalg.norm(H, axis=1))))
        r = np.asarray(vals, dtype=np.float64)
        if r.size == 0:
            continue
        m = window_mask(r.size, "late")
        spread = 1.0 - r
        features["cloud_resultant_full"][i] = safe_mean(r)
        features["cloud_resultant_late"][i] = safe_mean(r[m])
        features["cloud_resultant_min"][i] = float(np.nanmin(r))
        features["cloud_spread_full"][i] = safe_mean(spread)
        features["cloud_spread_late"][i] = safe_mean(spread[m])
        features["cloud_spread_max"][i] = float(np.nanmax(spread))
        nn = np.asarray(norms, dtype=np.float64)
        features["cloud_mean_norm_late"][i] = safe_mean(nn[m]) if nn.size == r.size else float("nan")
    return features


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
    ds = []
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


def build_features(data: np.lib.npyio.NpzFile, bands: Sequence[str]) -> Dict[str, np.ndarray]:
    N = len(data["problem_ids"])
    feats: Dict[str, np.ndarray] = {}
    if "n_steps" in data.files:
        feats["n_steps"] = data["n_steps"].astype(np.float64)
    for key, prefix in (("sv_out_entropy", "out_entropy"), ("sv_out_committal", "out_committal")):
        if key in data.files:
            raw = data[key]
            feats[f"{prefix}_mean"] = np.array([seq_window_feature(x, "full") for x in raw], dtype=np.float64)
            feats[f"{prefix}_late"] = np.array([seq_window_feature(x, "late") for x in raw], dtype=np.float64)
            feats[f"{prefix}_max"] = np.array([seq_max(x) for x in raw], dtype=np.float64)
    for metric in ("pr", "ae"):
        key = f"sv_{metric}_step_exp"
        if key not in data.files:
            continue
        raw = data[key]
        first = np.asarray(raw[0], dtype=np.float64)
        if first.ndim != 2:
            continue
        for band in bands:
            cols = band_cols(first.shape[1], band)
            for window in ("early", "late", "full"):
                feats[f"{metric}_{band}_{window}"] = np.array(
                    [matrix_window_feature(x, cols, window) for x in raw], dtype=np.float64
                )
    for band in bands:
        v = compute_vector_mahal(data, band)
        if np.isfinite(v).any():
            feats[f"mahal_{band}"] = v
    feats.update(compute_cloud_features(data))
    return feats


def evaluate_feature(name: str, vals: np.ndarray, y_err: np.ndarray, mask: np.ndarray, groups: Sequence[np.ndarray]) -> Dict[str, Any]:
    m = mask & np.isfinite(vals)
    err = vals[m & (y_err == 1)]
    cor = vals[m & (y_err == 0)]
    wauc, pairs = within_pair_auroc(groups, vals, y_err)
    dlt = paired_delta(groups, vals, y_err)
    return {
        "feature": name,
        "n": int(m.sum()),
        "n_error": int((m & (y_err == 1)).sum()),
        "n_correct": int((m & (y_err == 0)).sum()),
        "error": descriptive(err),
        "correct": descriptive(cor),
        "cohen_d_error_minus_correct": cohen_d(err, cor),
        "cross_auroc_error_high": auroc_signed(err, cor),
        "within_pair_auroc_error_high": wauc,
        "within_pairs": pairs,
        "paired_delta_error_minus_correct": dlt,
        "best_direction_within": max(wauc, 1.0 - wauc) if np.isfinite(wauc) else float("nan"),
    }


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    problem_ids = data["problem_ids"].astype(int)
    feats = build_features(data, bands=args.bands.split(","))
    policies: Dict[str, Any] = {}
    for pol in args.policies.split(","):
        y_err, mask, desc = label_policy(data, pol)
        groups = problem_groups(problem_ids, y_err, mask, min_per_class=args.min_per_class)
        rows = [evaluate_feature(name, vals, y_err, mask, groups) for name, vals in feats.items()]
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
            "results": rows,
        }
    return {
        "meta": {
            "input": os.path.abspath(path),
            "basename": os.path.basename(path),
            "model": str(data["model_name"]) if "model_name" in data.files else "unknown",
            "prompt_style": str(data["prompt_style"]) if "prompt_style" in data.files else "unknown",
            "step_split": str(data["step_split"]) if "step_split" in data.files else "unknown",
            "bands": args.bands.split(","),
            "features_computed": sorted(feats.keys()),
        },
        "policies": policies,
    }


def write_outputs(res: Mapping[str, Any], output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stem = f"multisample_feature_distribution_{os.path.splitext(str(res['meta']['basename']))[0]}"
    jp = os.path.join(output_dir, stem + ".json")
    mp = os.path.join(output_dir, stem + ".md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(finite_json(res), f, indent=2, ensure_ascii=False)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(f"# Multisample Feature Distribution: {res['meta']['basename']}\n\n")
        f.write("## Result Analysis\n\n")
        f.write("- Rows are signed as error-high: AUROC > 0.5 means error samples have larger feature values.\n")
        f.write("- `within_pair_auroc_error_high` is the main difficulty-controlled metric.\n")
        f.write("- `best_direction_within` is shown only to reveal separability when the direction flips; mechanism claims must keep the signed direction.\n\n")
        for pol, sec in res["policies"].items():
            f.write(f"### {pol}\n\n")
            f.write(
                f"{sec['n_error']} error / {sec['n_correct']} correct samples; "
                f"{sec['n_contrastive_problems']} contrastive problems.\n\n"
            )
            f.write("| feature | within | cross | d | err median | cor median | paired median delta | frac positive |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in sec["results"][:20]:
                dlt = r["paired_delta_error_minus_correct"]
                f.write(
                    f"| {r['feature']} | {r['within_pair_auroc_error_high']:.3f} | "
                    f"{r['cross_auroc_error_high']:.3f} | {r['cohen_d_error_minus_correct']:.3f} | "
                    f"{r['error'].get('median', float('nan')):.3f} | "
                    f"{r['correct'].get('median', float('nan')):.3f} | "
                    f"{dlt.get('median', float('nan')):.3f} | {dlt.get('fraction_positive', float('nan')):.3f} |\n"
                )
            f.write("\n")
        f.write("## Follow-Up Research Direction\n\n")
        f.write("- Treat features that are strong cross-problem but weak within-problem as difficulty or format proxies.\n")
        f.write("- Prioritize features that survive `answer_format_ok` with stable signed direction across 5shot/custom prompts.\n")
        f.write("- If `cloud_spread_*` collapses, the old spread/resultant signal is not a same-question trajectory discriminator by itself.\n\n")
        f.write("## Optimization Suggestions\n\n")
        f.write("- Use these distributions before adding new models; the target is same-problem separability, not only cross-problem AUROC.\n")
        f.write("- Compare every future detector against `n_steps`, `out_entropy_*`, `mahal_*`, and `cloud_spread_*` when available.\n")
    return jp, mp


def print_report(res: Mapping[str, Any], top: int) -> None:
    meta = res["meta"]
    print(f"\n===== multisample feature distribution | {meta['basename']} =====")
    print(f"prompt={meta['prompt_style']} step_split={meta['step_split']} model={meta['model']}")
    for pol, sec in res["policies"].items():
        print(f"\n[{pol}] err={sec['n_error']} cor={sec['n_correct']} contrastive={sec['n_contrastive_problems']}")
        for r in sec["results"][:top]:
            dlt = r["paired_delta_error_minus_correct"]
            print(
                f"  {r['feature']:24s} within {r['within_pair_auroc_error_high']:.3f} "
                f"cross {r['cross_auroc_error_high']:.3f} d {r['cohen_d_error_minus_correct']:+.3f} "
                f"err_med {r['error'].get('median', float('nan')):.3f} "
                f"cor_med {r['correct'].get('median', float('nan')):.3f} "
                f"delta {dlt.get('median', float('nan')):+.3f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", default="outputs/multisample_feature_distribution")
    ap.add_argument("--policies", default="answer,strict,answer_format_ok")
    ap.add_argument("--bands", default="mid,deep,all")
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--top", type=int, default=16)
    args = ap.parse_args()

    res = run(args.input, args)
    jp, mp = write_outputs(res, args.output_dir)
    print_report(res, args.top)
    print(f"\nwrote {jp} and {mp}")


if __name__ == "__main__":
    main()
