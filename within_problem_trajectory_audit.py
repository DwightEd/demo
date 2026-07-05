#!/usr/bin/env python3
"""Within-problem trajectory audit for same-question multi-sampling data.

The full ProcessBench data has one chain per problem, so error can be entangled
with problem difficulty.  The v2 same-problem multi-sampling files instead give
multiple sampled solutions for the same question.  This script asks:

  Holding the problem fixed, can trajectory-shape scores distinguish failing
  samples from successful samples?

It compares two learning policies:

  correct_only_distance  : fit a healthy trajectory baseline on correct chains.
  error_contrast         : use both correct and incorrect chains to learn a
                           held-out contrast direction.

The headline metric is within-problem paired AUROC over same-problem
(incorrect, correct) sample pairs.  All learned scores are cross-fit by problem
id, so held-out problems are never used to fit their own healthy/error baselines.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from audit_utils import finite_json, safe_mean, safe_std


EPS = 1e-9


def auroc(scores, labels) -> float:
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def bestdir(a: float) -> float:
    return float(max(a, 1.0 - a)) if np.isfinite(a) else float("nan")


def band_cols(n_layers: int, band: str) -> np.ndarray:
    if band == "all":
        return np.arange(n_layers)
    if band == "deep":
        return np.arange(int(n_layers * 0.6), n_layers)
    if band == "mid":
        return np.arange(int(n_layers * 0.3), max(int(n_layers * 0.7), int(n_layers * 0.3) + 1))
    if band == "early":
        return np.arange(0, max(1, int(n_layers * 0.3)))
    return np.array([int(x.strip()) for x in band.split(",") if x.strip()], dtype=int)


def finite_mean(x) -> float:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def finite_std(x) -> float:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    return float(v.std()) if len(v) else float("nan")


def causal_ema_prev(x: np.ndarray, *, alpha: float) -> np.ndarray:
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    state = np.nan
    for t, val in enumerate(v):
        out[t] = state
        if not np.isfinite(val):
            continue
        state = float(val) if not np.isfinite(state) else alpha * state + (1.0 - alpha) * float(val)
    return out


def leaky_cusum(x: np.ndarray, *, lam: float, kref: float) -> np.ndarray:
    out = np.zeros(len(x), float)
    c = 0.0
    for t, val in enumerate(np.asarray(x, float)):
        z = 0.0 if not np.isfinite(val) else float(val)
        c = max(0.0, lam * c + z - kref)
        out[t] = c
    return out


def robust_center_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, float)
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0) * 1.4826
    sd = np.nanstd(X, axis=0)
    scale = np.where(np.isfinite(mad) & (mad > EPS), mad, sd)
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    med = np.where(np.isfinite(med), med, 0.0)
    return med, scale


def group_folds(groups: np.ndarray, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return []
    rng = np.random.default_rng(seed)
    uniq = np.array(uniq)
    rng.shuffle(uniq)
    fold_of = {int(g): i % min(k, len(uniq)) for i, g in enumerate(uniq)}
    f = np.array([fold_of[int(g)] for g in groups])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in sorted(set(f))]


def within_pair_auroc(problem_ids: np.ndarray, score: np.ndarray, y_inc: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    conc = 0.0
    npair = 0
    for p in np.unique(problem_ids[mask]):
        idx = np.where(mask & (problem_ids == p))[0]
        inc = [float(score[i]) for i in idx if y_inc[i] == 1 and np.isfinite(score[i])]
        cor = [float(score[i]) for i in idx if y_inc[i] == 0 and np.isfinite(score[i])]
        if not inc or not cor:
            continue
        for a in inc:
            for b in cor:
                conc += 1.0 if a > b else (0.5 if a == b else 0.0)
        npair += len(inc) * len(cor)
    return (conc / npair if npair else float("nan")), int(npair)


def contrastive_problem_mask(problem_ids: np.ndarray, y_inc: np.ndarray, mask: np.ndarray, *, min_per_class: int) -> np.ndarray:
    keep = np.zeros(len(problem_ids), bool)
    for p in np.unique(problem_ids[mask]):
        idx = np.where(mask & (problem_ids == p))[0]
        if int((y_inc[idx] == 1).sum()) >= min_per_class and int((y_inc[idx] == 0).sum()) >= min_per_class:
            keep[idx] = True
    return keep


def sequence_from_metric(obj, cols: np.ndarray) -> np.ndarray:
    M = np.asarray(obj, float)
    if M.ndim == 1:
        return M.astype(float)
    if M.ndim == 2:
        valid_cols = cols[cols < M.shape[1]]
        if len(valid_cols) == 0:
            valid_cols = np.arange(M.shape[1])
        return np.nanmean(M[:, valid_cols], axis=1)
    raise ValueError(f"unsupported metric shape {M.shape}")


def summarize_sequence(x: np.ndarray, *, alpha: float, lam: float, kref: float) -> Dict[str, float]:
    v = np.asarray(x, float)
    finite = v[np.isfinite(v)]
    if len(finite) == 0:
        return {
            "mean": float("nan"),
            "early": float("nan"),
            "late": float("nan"),
            "max": float("nan"),
            "slope": float("nan"),
            "vol": float("nan"),
            "innov_max": float("nan"),
            "innov_late": float("nan"),
            "cusum_max": float("nan"),
            "n_steps": float(len(v)),
        }
    T = len(v)
    frac = np.arange(T) / max(1, T - 1)
    early = v[frac <= 0.33]
    late = v[frac >= 0.67]
    slow = causal_ema_prev(v, alpha=alpha)
    innov = v - slow
    med, scale = robust_center_scale(innov[:, None])
    z = (innov - med[0]) / scale[0]
    cs = leaky_cusum(np.maximum(z, 0.0), lam=lam, kref=kref)
    return {
        "mean": finite_mean(v),
        "early": finite_mean(early),
        "late": finite_mean(late),
        "max": float(np.nanmax(v)),
        "slope": finite_mean(late) - finite_mean(early),
        "vol": finite_std(v),
        "innov_max": float(np.nanmax(innov)) if np.isfinite(innov).any() else float("nan"),
        "innov_late": finite_mean(innov[frac >= 0.67]),
        "cusum_max": float(np.nanmax(cs)) if np.isfinite(cs).any() else float("nan"),
        "n_steps": float(T),
    }


def build_feature_matrix(data: np.lib.npyio.NpzFile, *, metric: str, mode: str, band: str, alpha: float, lam: float, kref: float) -> Tuple[np.ndarray, List[str]]:
    key = "sv_out_entropy" if metric == "entropy" else f"sv_{metric}_{mode}"
    if key not in data.files:
        raise SystemExit(f"missing {key}; available keys include: {data.files[:20]}")
    raw = data[key]
    if metric == "entropy":
        cols = np.array([0], dtype=int)
    else:
        first = np.asarray(raw[0], float)
        if first.ndim != 2:
            raise SystemExit(f"{key} expected per-step x layer arrays, got {first.shape}")
        cols = band_cols(first.shape[1], band)
    rows = []
    names = None
    for obj in raw:
        seq = sequence_from_metric(obj, cols)
        feats = summarize_sequence(seq, alpha=alpha, lam=lam, kref=kref)
        if names is None:
            names = list(feats.keys())
        rows.append([feats[n] for n in names])
    return np.asarray(rows, float), names or []


def crossfit_scores(
    X: np.ndarray,
    y_inc: np.ndarray,
    problem_ids: np.ndarray,
    mask: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    N, D = X.shape
    out = {
        "correct_only_distance": np.full(N, np.nan),
        "error_contrast": np.full(N, np.nan),
        "hybrid_distance_plus_contrast": np.full(N, np.nan),
    }
    for tr, te in group_folds(problem_ids[mask], folds, seed):
        idx_all = np.where(mask)[0]
        tr_idx = idx_all[tr]
        te_idx = idx_all[te]
        finite_tr = np.isfinite(X[tr_idx]).all(axis=1)
        tr_idx = tr_idx[finite_tr]
        if len(tr_idx) < max(20, D + 3):
            continue
        correct = tr_idx[y_inc[tr_idx] == 0]
        error = tr_idx[y_inc[tr_idx] == 1]
        if len(correct) < max(10, D // 2):
            continue
        mu_c, sc_c = robust_center_scale(X[correct])
        Zte = (X[te_idx] - mu_c) / sc_c
        dist = np.nansum(Zte ** 2, axis=1)
        out["correct_only_distance"][te_idx] = dist

        if len(error) >= max(10, D // 2):
            mu_e, _ = robust_center_scale(X[error])
            pooled = np.vstack([X[correct], X[error]])
            _, sc = robust_center_scale(pooled)
            direction = (mu_e - mu_c) / sc
            norm = float(np.linalg.norm(direction))
            if np.isfinite(norm) and norm > EPS:
                direction = direction / norm
                z = (X[te_idx] - mu_c) / sc
                contrast = z @ direction
                out["error_contrast"][te_idx] = contrast
                out["hybrid_distance_plus_contrast"][te_idx] = 0.5 * (dist / max(np.nanstd(dist), EPS)) + contrast
    return out


def row_for_score(name: str, score: np.ndarray, y_inc: np.ndarray, problem_ids: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
    wp, npair = within_pair_auroc(problem_ids, score, y_inc, mask)
    cp = auroc(score[mask], y_inc[mask])
    return {
        "score": name,
        "within_pair_auroc_high_error": wp,
        "within_pair_auroc_bestdir": bestdir(wp),
        "cross_problem_auroc_high_error": cp,
        "cross_problem_auroc_bestdir": bestdir(cp),
        "n_pairs": int(npair),
        "n": int(np.isfinite(score[mask]).sum()),
        "mean_error": safe_mean(score[mask & (y_inc == 1)]),
        "mean_correct": safe_mean(score[mask & (y_inc == 0)]),
    }


def label_policy(data: np.lib.npyio.NpzFile, policy: str) -> Tuple[np.ndarray, np.ndarray, str]:
    if policy == "answer":
        if "is_correct" not in data.files:
            raise SystemExit("missing is_correct")
        return (data["is_correct"].astype(int) == 0).astype(int), np.ones(len(data["problem_ids"]), bool), "answer incorrect"
    if policy == "strict":
        if "is_correct_strict" not in data.files:
            raise SystemExit("missing is_correct_strict")
        return (data["is_correct_strict"].astype(int) == 0).astype(int), np.ones(len(data["problem_ids"]), bool), "strict incorrect"
    if policy == "answer_format_ok":
        if "format_ok" not in data.files:
            raise SystemExit("missing format_ok")
        return (data["is_correct"].astype(int) == 0).astype(int), data["format_ok"].astype(bool), "answer incorrect among format_ok samples"
    raise SystemExit(f"unknown label policy {policy}")


def run(path: str, args: argparse.Namespace) -> Dict[str, object]:
    data = np.load(path, allow_pickle=True)
    problem_ids = data["problem_ids"].astype(int)
    y_inc, base_mask, label_desc = label_policy(data, args.label_policy)
    contrast_mask = contrastive_problem_mask(problem_ids, y_inc, base_mask, min_per_class=args.min_per_class)
    X, feature_names = build_feature_matrix(
        data,
        metric=args.metric,
        mode=args.mode,
        band=args.layer_band,
        alpha=args.slow_alpha,
        lam=args.lam,
        kref=args.kref,
    )
    scores = {f"summary_{name}": X[:, j] for j, name in enumerate(feature_names)}
    scores.update(crossfit_scores(X, y_inc, problem_ids, contrast_mask, folds=args.folds, seed=args.seed))
    rows = [row_for_score(name, score, y_inc, problem_ids, contrast_mask) for name, score in scores.items()]
    rows.sort(key=lambda r: np.nan_to_num(r["within_pair_auroc_bestdir"], nan=-1.0), reverse=True)
    return {
        "meta": {
            "input": path,
            "metric": args.metric,
            "mode": args.mode,
            "layer_band": args.layer_band,
            "label_policy": args.label_policy,
            "label_desc": label_desc,
            "features": feature_names,
        },
        "n_samples": int(len(problem_ids)),
        "n_problems": int(len(np.unique(problem_ids))),
        "n_error": int(y_inc[base_mask].sum()),
        "n_correct": int((base_mask & (y_inc == 0)).sum()),
        "n_contrastive_samples": int(contrast_mask.sum()),
        "n_contrastive_problems": int(len(np.unique(problem_ids[contrast_mask]))),
        "results": rows,
        "notes": {
            "correct_only_distance": "held-out distance to healthy summaries fitted on correct training samples only",
            "error_contrast": "held-out contrast direction learned from incorrect-vs-correct training samples",
            "within_pair_auroc": "P(score(incorrect sample) > score(correct sample)) for same-problem pairs",
            "why_same_problem": "problem difficulty is fixed within each pair; remaining separation is sample trajectory/failure signal",
        },
    }


def print_report(res: Dict[str, object], *, top: int) -> None:
    meta = res["meta"]
    print(
        f"\n===== within-problem trajectory audit | {os.path.basename(meta['input'])} | "
        f"{meta['metric']} {meta['mode']} {meta['layer_band']} | {meta['label_policy']} ====="
    )
    print(
        f"samples {res['n_samples']} | problems {res['n_problems']} | "
        f"contrastive problems {res['n_contrastive_problems']} | "
        f"contrastive samples {res['n_contrastive_samples']}"
    )
    print(f"errors {res['n_error']} | correct {res['n_correct']}")
    print("\nScores:")
    for r in res["results"][:top]:
        print(
            f"  {r['score']:32s} within {float(r['within_pair_auroc_high_error']):.3f} "
            f"best {float(r['within_pair_auroc_bestdir']):.3f} "
            f"cross {float(r['cross_problem_auroc_high_error']):.3f} "
            f"n={r['n']} pairs={r['n_pairs']} "
            f"err {float(r['mean_error']):+.3f} cor {float(r['mean_correct']):+.3f}"
        )


def write_outputs(res: Dict[str, object], output_dir: str, stem: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    clean = finite_json(res)
    with open(os.path.join(output_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, stem + ".md"), "w", encoding="utf-8") as f:
        f.write(f"# Within-Problem Trajectory Audit: {stem}\n\n")
        f.write("## Result Analysis\n\n")
        f.write("- The headline metric is same-problem paired AUROC, so problem difficulty is controlled by design.\n")
        f.write("- `correct_only_distance` tests the healthy-null view; `error_contrast` tests whether error chains add useful structure.\n")
        f.write("- Compare these learned scores against raw summary features such as late level, innovation, and CUSUM.\n\n")
        f.write("## Follow-Up Research Direction\n\n")
        f.write("- If `error_contrast` beats `correct_only_distance`, upgrade the main detector to a two-state healthy/error model.\n")
        f.write("- If dynamic summary features beat level features, bring trajectory-relative scoring back to ProcessBench onset detection.\n")
        f.write("- If all within-problem numbers collapse, current signals are mostly difficulty or format artifacts.\n\n")
        f.write("## Optimization Suggestions\n\n")
        f.write("- Run `answer_format_ok` to separate reasoning errors from format failures.\n")
        f.write("- Repeat over `ae`, `pr`, and `entropy`, and over `mid/deep/all` bands.\n")
        f.write("- Use this script before designing heavier HSMM/SLDS models; same-problem pairs are the cleanest first gate.\n")


def make_selftest(path: str, *, seed: int = 0, n_problems: int = 80, k: int = 8) -> None:
    rng = np.random.default_rng(seed)
    problem_ids, sample_idx, y, fmt, steps, arrs = [], [], [], [], [], []
    for p in range(n_problems):
        difficulty = rng.normal()
        for s in range(k):
            err = rng.random() < 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * difficulty)))
            T = int(rng.integers(4, 9))
            L = 4
            base = 0.4 + 0.08 * difficulty + 0.02 * rng.normal(size=(T, L))
            drift = np.linspace(0.0, 0.15 if err else 0.02, T)[:, None]
            noise = 0.03 * rng.normal(size=(T, L))
            M = base + drift + noise
            if err:
                M[-max(1, T // 3) :] += 0.18
            problem_ids.append(p)
            sample_idx.append(s)
            y.append(0 if err else 1)
            fmt.append(1)
            steps.append(T)
            arrs.append(M.astype(np.float32))
    obj = np.empty(len(arrs), dtype=object)
    obj[:] = arrs
    np.savez_compressed(
        path,
        problem_ids=np.asarray(problem_ids, int),
        sample_idx=np.asarray(sample_idx, int),
        is_correct=np.asarray(y, int),
        is_correct_strict=np.asarray(y, int),
        format_ok=np.asarray(fmt, int),
        n_steps=np.asarray(steps, int),
        sv_stored=np.array(True),
        sv_modes=np.asarray(["step_exp"], dtype=object),
        sv_ae_step_exp=obj,
        sv_pr_step_exp=obj,
        sv_out_entropy=obj,
    )


def assert_selftest(res: Dict[str, object]) -> None:
    rows = {r["score"]: r for r in res["results"]}
    best = max(float(r["within_pair_auroc_bestdir"]) for r in res["results"] if np.isfinite(r["within_pair_auroc_bestdir"]))
    if best < 0.75:
        raise SystemExit(f"selftest failed: best within AUROC too low ({best:.3f})")
    if rows.get("error_contrast", {}).get("within_pair_auroc_bestdir", 0.0) < 0.70:
        raise SystemExit("selftest failed: error_contrast did not recover injected error direction")


def main() -> None:
    ap = argparse.ArgumentParser(description="Same-problem multi-sample trajectory audit")
    ap.add_argument("--input", default="")
    ap.add_argument("--metric", default="ae", choices=["ae", "pr", "entropy"])
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--layer_band", default="mid")
    ap.add_argument("--label_policy", default="answer", choices=["answer", "strict", "answer_format_ok"])
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--slow_alpha", type=float, default=0.65)
    ap.add_argument("--lam", type=float, default=0.80)
    ap.add_argument("--kref", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--output_dir", default="outputs/within_problem_trajectory")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "within_problem_trajectory_selftest.npz")
            make_selftest(path, seed=args.seed)
            res = run(path, args)
            stem = "within_problem_trajectory_selftest"
            assert_selftest(res)
    else:
        if not args.input:
            raise SystemExit("pass --input or --selftest")
        res = run(args.input, args)
        stem = (
            f"within_problem_trajectory_{os.path.splitext(os.path.basename(args.input))[0]}_"
            f"{args.metric}_{args.mode}_{args.layer_band}_{args.label_policy}"
        )

    print_report(res, top=args.top)
    write_outputs(res, args.output_dir, stem)
    print(f"\nwrote {os.path.join(args.output_dir, stem + '.json')} and .md")


if __name__ == "__main__":
    main()
