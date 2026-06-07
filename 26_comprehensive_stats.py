"""Step 26: comprehensive statistical evaluation of correct vs error chains.

What this does (one stop):
    For every chain-level feature we compute:
      Descriptive (per group)        : n, mean, std, median, q25, q75, IQR, min, max
      Effect size (signed: err-cor)  : Cohen's d (+ bootstrap CI), Glass's delta,
                                        Cliff's delta
      Two-sample tests (cross-prob)  : Mann-Whitney U + p, Kolmogorov-Smirnov D + p
      Cross-problem AUROC            : signed (>0.5 -> error higher), + bootstrap CI
      Within-problem paired AUROC    : signed, over same-problem (inc, cor) pairs
      Within-problem paired delta    : median + IQR of (inc_mean - cor_mean) per
                                        problem, fraction-positive, n_problems
      Wilcoxon signed-rank           : z + two-sided p on the per-problem deltas
      Direction verdict              : "error_higher" / "correct_higher" / "n.s."

    Runs the whole battery under TWO label policies (if both are present):
      lenient = is_correct           (v1 compat: last-number fallback counts)
      strict  = is_correct_strict    (marker + match)

    And TWO layer bands (configurable):
      mid, deep                       (all also computed unless --skip_all_band)

    Features (per chain, late-window = frac >= 0.6 of steps; default window):
      pr_<band>_<window>     participation ratio (step_exp aggregated)
      ae_<band>_<window>     activation entropy (step_exp aggregated)
      mean_out_entropy       mean output token entropy
      n_steps                length baseline (the confound to beat)
      mahal_<band>           late-window Mahalanobis distance from healthy mean
                              (in-sample fit; flagged as INDICATIVE, not held-out)

Why this script exists:
    Existing analysis scripts (12 / 19 / 20 / 24 / 25) save AUROC values with
    max(a, 1-a) applied -> direction information is lost. 11 has Wilcoxon but
    only prints to stdout, never saves. The cross-problem v3 evaluation script
    (the one that produced step_effective_rank d=-0.870) is not in this repo.
    This script consolidates the missing pieces into ONE JSON output.

Usage:
    python 26_comprehensive_stats.py \
        --input data/gsm8k_multisample_sv.npz \
        --output data/comprehensive_stats.json \
        --bootstrap_n 1000

    Result is JSON; inspect with `cat` / `jq` / git diff. The script does NOT
    use scipy -- all tests are implemented from primary references so the
    output is reproducible across environments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Sequence

import numpy as np


# ============================================================================
# Statistics primitives (no scipy dependency)
# ============================================================================

def _safe_mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def _safe_std(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1)) if x.size > 1 else float("nan")


def descriptive(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None,
                "q25": None, "q75": None, "iqr": None, "min": None, "max": None}
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def cohen_d_signed(err: np.ndarray, cor: np.ndarray) -> float:
    """Signed Cohen's d using pooled SD; positive => error has higher mean."""
    err = np.asarray(err, dtype=np.float64); err = err[np.isfinite(err)]
    cor = np.asarray(cor, dtype=np.float64); cor = cor[np.isfinite(cor)]
    if err.size < 2 or cor.size < 2:
        return float("nan")
    me, mc = err.mean(), cor.mean()
    ve, vc = err.var(ddof=1), cor.var(ddof=1)
    pooled = ((err.size - 1) * ve + (cor.size - 1) * vc) / (err.size + cor.size - 2)
    if pooled <= 0:
        return 0.0
    return float((me - mc) / math.sqrt(pooled))


def glass_delta(err: np.ndarray, cor: np.ndarray) -> float:
    """Glass's delta: (mean_err - mean_cor) / std_cor.  Useful when groups
    have very different variances."""
    err = np.asarray(err, dtype=np.float64); err = err[np.isfinite(err)]
    cor = np.asarray(cor, dtype=np.float64); cor = cor[np.isfinite(cor)]
    if err.size < 1 or cor.size < 2:
        return float("nan")
    sc = cor.std(ddof=1)
    return float((err.mean() - cor.mean()) / sc) if sc > 0 else float("nan")


def cliff_delta(err: np.ndarray, cor: np.ndarray) -> float:
    """Cliff's delta (rank-based, distribution-free effect size).
    = P(err > cor) - P(err < cor).  In [-1, 1].  Sign matches Cohen's d."""
    err = np.asarray(err, dtype=np.float64); err = err[np.isfinite(err)]
    cor = np.asarray(cor, dtype=np.float64); cor = cor[np.isfinite(cor)]
    if err.size == 0 or cor.size == 0:
        return float("nan")
    # O((n_err + n_cor) log) via sorting + searchsorted
    cs = np.sort(cor)
    # for each err value, count #cor < err and #cor > err
    less = np.searchsorted(cs, err, side="left")
    leq = np.searchsorted(cs, err, side="right")
    gt = cs.size - leq                           # cor > err
    return float((less.sum() - gt.sum()) / (err.size * cs.size))


def auroc_signed(err: np.ndarray, cor: np.ndarray) -> float:
    """Signed AUROC: P(err > cor) + 0.5 * P(err == cor).  Equivalent to the
    Mann-Whitney U effect size.  > 0.5 => error has higher feature values."""
    err = np.asarray(err, dtype=np.float64); err = err[np.isfinite(err)]
    cor = np.asarray(cor, dtype=np.float64); cor = cor[np.isfinite(cor)]
    if err.size == 0 or cor.size == 0:
        return float("nan")
    combined = np.concatenate([err, cor])
    labels = np.concatenate([np.ones(err.size), np.zeros(cor.size)])
    order = np.argsort(combined, kind="mergesort")
    ranks = _avg_ranks(combined[order])
    rank_full = np.empty_like(ranks)
    rank_full[order] = ranks
    sum_pos = rank_full[labels == 1].sum()
    U = sum_pos - err.size * (err.size + 1) / 2.0
    return float(U / (err.size * cor.size))


def mann_whitney_u(err: np.ndarray, cor: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test (normal approximation with tie correction).
    Returns (U_smaller, two_sided_p)."""
    err = np.asarray(err, dtype=np.float64); err = err[np.isfinite(err)]
    cor = np.asarray(cor, dtype=np.float64); cor = cor[np.isfinite(cor)]
    n1, n2 = err.size, cor.size
    if n1 < 1 or n2 < 1:
        return float("nan"), float("nan")
    combined = np.concatenate([err, cor])
    ranks = _avg_ranks(np.sort(combined))
    # rerank: assign by sorted-order back to original positions
    order = np.argsort(combined, kind="mergesort")
    rank_full = np.empty_like(ranks)
    rank_full[order] = ranks
    R1 = rank_full[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)
    # normal approx with tie correction
    n = n1 + n2
    mu = n1 * n2 / 2.0
    _, counts = np.unique(combined, return_counts=True)
    tie_term = ((counts ** 3 - counts).sum()) / (n * (n - 1)) if n > 1 else 0.0
    sigma2 = n1 * n2 * (n + 1 - tie_term) / 12.0
    if sigma2 <= 0:
        return float(U), float("nan")
    z = (U1 - mu) / math.sqrt(sigma2)
    p = math.erfc(abs(z) / math.sqrt(2.0))                  # two-sided
    return float(U), float(p)


def ks_two_sample(err: np.ndarray, cor: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov.  Returns (D, two-sided p) via the
    Smirnov asymptotic series."""
    err = np.asarray(err, dtype=np.float64); err = err[np.isfinite(err)]
    cor = np.asarray(cor, dtype=np.float64); cor = cor[np.isfinite(cor)]
    n1, n2 = err.size, cor.size
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    # build empirical CDFs on the combined sort
    combined = np.sort(np.concatenate([err, cor]))
    cdf1 = np.searchsorted(np.sort(err), combined, side="right") / n1
    cdf2 = np.searchsorted(np.sort(cor), combined, side="right") / n2
    D = float(np.max(np.abs(cdf1 - cdf2)))
    en = math.sqrt(n1 * n2 / (n1 + n2))
    # Smirnov series approx p = 2 sum_{k=1}^{inf} (-1)^{k-1} exp(-2 k^2 lam^2)
    lam = (en + 0.12 + 0.11 / en) * D
    s = 0.0
    for k in range(1, 101):
        term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * (k * lam) ** 2)
        s += term
        if abs(term) < 1e-9:
            break
    p = max(0.0, min(1.0, s))
    return D, float(p)


def wilcoxon_signed_rank(diffs: np.ndarray):
    """Two-sided Wilcoxon signed-rank, normal approx with tie correction.
    diffs = paired differences (e.g., incorrect_mean - correct_mean per problem).
    Returns (z, two_sided_p, n_nonzero, fraction_positive)."""
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d) & (d != 0)]
    n = d.size
    if n < 6:
        return float("nan"), float("nan"), n, float("nan")
    ranks = _avg_ranks(np.sort(np.abs(d)))
    rank_full = np.empty_like(ranks)
    rank_full[np.argsort(np.abs(d), kind="mergesort")] = ranks
    W = float(rank_full[d > 0].sum())
    mu = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie = (counts ** 3 - counts).sum()
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0)
    if sigma == 0:
        return float("nan"), float("nan"), n, float("nan")
    z = (W - mu) / sigma
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return float(z), float(p), int(n), float((d > 0).mean())


def _avg_ranks(sorted_vals: np.ndarray) -> np.ndarray:
    """Mid-rank (average of tied ranks) for an array assumed pre-sorted."""
    n = sorted_vals.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[i:j] = avg
        i = j
    return ranks


def bootstrap_ci(estimator, *groups, n_boot=1000, alpha=0.05, seed=0):
    """Percentile CI via case resampling within each group.  estimator(*resampled)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        resampled = []
        for g in groups:
            g = np.asarray(g, dtype=np.float64)
            g = g[np.isfinite(g)]
            if g.size == 0:
                resampled = None
                break
            idx = rng.integers(0, g.size, size=g.size)
            resampled.append(g[idx])
        if resampled is None:
            continue
        try:
            v = estimator(*resampled)
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def within_paired_auroc_signed(idx_groups, feats, y_inc):
    """Signed within-problem paired AUROC over all same-problem (inc, cor) pairs.
    >0.5 => error has higher feature values."""
    conc = 0.0; npair = 0
    for idx in idx_groups:
        inc = [feats[i] for i in idx if y_inc[i] == 1 and np.isfinite(feats[i])]
        cor = [feats[i] for i in idx if y_inc[i] == 0 and np.isfinite(feats[i])]
        if not inc or not cor:
            continue
        for a in inc:
            for b in cor:
                conc += 1.0 if a > b else (0.5 if a == b else 0.0)
        npair += len(inc) * len(cor)
    return (conc / npair if npair else float("nan")), npair


def paired_problem_deltas(idx_groups, problem_ids_list, feats, y_inc):
    """Per-problem delta = mean(inc feats) - mean(cor feats).  Returns
    (deltas: np.ndarray, n_problems_used: int)."""
    deltas = []
    for idx in idx_groups:
        vinc = [feats[i] for i in idx if y_inc[i] == 1 and np.isfinite(feats[i])]
        vcor = [feats[i] for i in idx if y_inc[i] == 0 and np.isfinite(feats[i])]
        if vinc and vcor:
            deltas.append(float(np.mean(vinc) - np.mean(vcor)))
    return np.array(deltas, dtype=np.float64), len(deltas)


# ============================================================================
# Feature extraction
# ============================================================================

def band_cols(L_sub, band):
    if band == "all":
        return np.arange(L_sub)
    if band == "deep":
        return np.arange(int(L_sub * 0.6), L_sub)
    if band == "mid":
        return np.arange(int(L_sub * 0.3), int(L_sub * 0.7))
    return np.array([int(x) for x in band.split(",") if x.strip()])


def window_mask(T: int, kind: str, lo: float = 0.6, hi: float = 0.4):
    if T <= 0:
        return np.zeros(0, dtype=bool)
    fr = (np.arange(T) / max(1, T - 1))
    if kind == "late":
        m = fr >= lo
        return m if m.any() else (fr >= fr.max())
    if kind == "early":
        m = fr < hi
        return m if m.any() else (fr <= fr.min())
    return np.ones(T, dtype=bool)                   # "full"


def chain_window_feat(matrix_2d: np.ndarray, cols: np.ndarray, T: int,
                       window: str) -> float:
    """matrix_2d is (T, L_sub).  Returns mean over (selected cols, window)."""
    M = matrix_2d[:, cols]
    m = window_mask(T, window)
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(np.nanmean(M[m], axis=1)))


def extract_features(data, bands=("mid", "deep", "all"), include_mahal=True):
    """Compute all per-chain features.  Returns dict[name -> (N,) array]."""
    problem_ids = data["problem_ids"].astype(int)
    n_steps = data["n_steps"].astype(int)
    N = problem_ids.size

    # both PR and AE arrays are object-arrays of (T_i, L_sub) matrices
    PR = data["sv_pr_step_exp"]
    AE = data["sv_ae_step_exp"]
    L_sub = np.asarray(PR[0]).shape[1]

    out_ent = data["sv_out_entropy"] if "sv_out_entropy" in data.files else None

    feats = {}
    feats["n_steps"] = n_steps.astype(np.float64)
    if out_ent is not None:
        feats["mean_out_entropy"] = np.array(
            [_safe_mean(np.asarray(e)) for e in out_ent], dtype=np.float64)

    for band in bands:
        cols = band_cols(L_sub, band)
        for window in ("late", "early", "full"):
            pr_v = np.full(N, np.nan); ae_v = np.full(N, np.nan)
            for i in range(N):
                pr_v[i] = chain_window_feat(np.asarray(PR[i]), cols,
                                             int(n_steps[i]), window)
                ae_v[i] = chain_window_feat(np.asarray(AE[i]), cols,
                                             int(n_steps[i]), window)
            feats[f"pr_{band}_{window}"] = pr_v
            feats[f"ae_{band}_{window}"] = ae_v

    # in-sample Mahalanobis: simple, indicative only.  late-window step_exp
    # vectors (if --store_vectors was on); else skip.
    if include_mahal and bool(data.get("sv_vectors_stored", np.array(False))):
        VEC = data["sv_vec_step_exp"]
        d_dim = np.asarray(VEC[0]).shape[2]
        for band in bands:
            cols = band_cols(L_sub, band)
            late_vec = np.full((N, d_dim), np.nan)
            for i in range(N):
                V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
                with np.errstate(invalid="ignore"):
                    P = np.nanmean(V, axis=1)
                m = window_mask(P.shape[0], "late")
                with np.errstate(invalid="ignore"):
                    late_vec[i] = np.nanmean(P[m], axis=0)
            # use ALL chain-late vectors to fit healthy mean/var (in-sample,
            # documented as indicative -- not a held-out test)
            ok = np.isfinite(late_vec).all(1)
            mu = late_vec[ok].mean(0); vr = late_vec[ok].var(0) + 1e-6
            mahal = np.where(ok, ((late_vec - mu) ** 2 / vr).sum(1), np.nan)
            feats[f"mahal_{band}"] = mahal

    return feats


# ============================================================================
# Per-feature full statistical battery
# ============================================================================

def evaluate_feature(name: str, vals: np.ndarray, y_inc: np.ndarray,
                      problem_ids: np.ndarray, idx_groups, *,
                      n_boot: int, seed: int) -> dict:
    err = vals[y_inc == 1]; cor = vals[y_inc == 0]
    desc_cor = descriptive(cor); desc_err = descriptive(err)

    # effect sizes (signed: err - cor)
    d = cohen_d_signed(err, cor)
    d_lo, d_hi = bootstrap_ci(cohen_d_signed, err, cor,
                              n_boot=n_boot, seed=seed)
    glass = glass_delta(err, cor)
    cliff = cliff_delta(err, cor)

    # two-sample tests
    U, U_p = mann_whitney_u(err, cor)
    KS_D, KS_p = ks_two_sample(err, cor)

    # cross-problem signed AUROC
    auroc_cp = auroc_signed(err, cor)
    auroc_lo, auroc_hi = bootstrap_ci(auroc_signed, err, cor,
                                       n_boot=n_boot, seed=seed + 1)

    # within-problem paired AUROC (signed)
    auroc_wp, npair = within_paired_auroc_signed(idx_groups, vals, y_inc)

    # within-problem paired delta + Wilcoxon
    deltas, n_prob = paired_problem_deltas(idx_groups, problem_ids, vals, y_inc)
    if deltas.size:
        med_d = float(np.median(deltas))
        q25_d, q75_d = np.percentile(deltas, [25, 75])
        frac_pos = float((deltas > 0).mean())
    else:
        med_d = q25_d = q75_d = frac_pos = float("nan")
    z, w_p, n_nonzero, w_fpos = wilcoxon_signed_rank(deltas)

    # direction verdict.  Significant if cross-prob MW p<0.05 OR
    # within-prob Wilcoxon p<0.05.  Direction follows Cohen's d sign.
    sig = ((np.isfinite(U_p) and U_p < 0.05) or
           (np.isfinite(w_p) and w_p < 0.05))
    if sig and np.isfinite(d):
        direction = "error_higher" if d > 0 else "correct_higher"
    else:
        direction = "n.s."

    return {
        "feature_name": name,
        "descriptive": {"correct": desc_cor, "error": desc_err},
        "effect_size_signed_error_minus_correct": {
            "cohen_d": _jsonify(d),
            "cohen_d_ci95": [_jsonify(d_lo), _jsonify(d_hi)],
            "glass_delta": _jsonify(glass),
            "cliff_delta": _jsonify(cliff),
        },
        "two_sample_tests_cross_problem": {
            "mann_whitney_U": _jsonify(U),
            "mann_whitney_p_two_sided": _jsonify(U_p),
            "ks_D": _jsonify(KS_D),
            "ks_p_two_sided": _jsonify(KS_p),
        },
        "auroc_cross_problem_signed": {
            "value": _jsonify(auroc_cp),
            "ci95": [_jsonify(auroc_lo), _jsonify(auroc_hi)],
        },
        "auroc_within_problem_paired_signed": {
            "value": _jsonify(auroc_wp),
            "n_pairs": int(npair),
            "n_problems_contrastive": int(n_prob),
        },
        "wilcoxon_within_problem_paired": {
            "z": _jsonify(z), "p_two_sided": _jsonify(w_p),
            "n_nonzero_problems": int(n_nonzero),
            "fraction_positive": _jsonify(w_fpos),
            "median_paired_delta": _jsonify(med_d),
            "iqr_paired_delta": [_jsonify(q25_d), _jsonify(q75_d)],
            "fraction_positive_from_deltas": _jsonify(frac_pos),
        },
        "direction_verdict": direction,
    }


def _jsonify(v):
    if v is None:
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    fv = float(v)
    if math.isnan(fv):
        return "NaN"
    if math.isinf(fv):
        return "Inf" if fv > 0 else "-Inf"
    return fv


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/comprehensive_stats.json")
    ap.add_argument("--bands", default="mid,deep,all")
    ap.add_argument("--bootstrap_n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_mahal", action="store_true",
                    help="skip Mahalanobis (saves time on big data; PR/AE only)")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    bands = tuple(b.strip() for b in args.bands.split(",") if b.strip())

    problem_ids = data["problem_ids"].astype(int)
    N = problem_ids.size

    # build problem groups
    prob_idx = {}
    for i, p in enumerate(problem_ids):
        prob_idx.setdefault(int(p), []).append(i)

    # extract features once
    print(f"Loaded {N} chains over {len(prob_idx)} problems.  Extracting features...")
    feats = extract_features(data, bands=bands,
                              include_mahal=not args.no_mahal)
    print(f"  {len(feats)} features computed: {sorted(feats.keys())}")

    # label policies available
    has_strict = "is_correct_strict" in data.files
    policies = [("lenient", data["is_correct"].astype(int))]
    if has_strict:
        policies.append(("strict", data["is_correct_strict"].astype(int)))
        print(f"  strict labels found -- running both policies.")
    else:
        print(f"  strict labels NOT in npz -- running lenient (v1) only.  "
              f"Regenerate with new 10 to enable strict.")

    out = {
        "_meta": {
            "input_file": os.path.basename(args.input),
            "n_chains": int(N),
            "n_problems_total": len(prob_idx),
            "bands": list(bands),
            "bootstrap_n": int(args.bootstrap_n),
            "prompt_style": str(data["prompt_style"]) if "prompt_style" in data.files else "unknown",
            "step_split": str(data["step_split"]) if "step_split" in data.files else "unknown",
            "model": str(data["model_name"]) if "model_name" in data.files else "unknown",
            "label_policies_evaluated": [p[0] for p in policies],
            "feature_list": sorted(feats.keys()),
            "stats_definitions": {
                "cohen_d": "signed (error - correct) / pooled_sd; >0 means error higher",
                "auroc_cross_problem_signed": "P(error > correct); >0.5 means error higher; treats each chain as one obs",
                "auroc_within_problem_paired_signed": "P over same-problem (err, cor) pairs; difficulty controlled",
                "wilcoxon_within_problem_paired": "per-problem delta = mean(err) - mean(cor), Wilcoxon signed-rank on those",
                "direction_verdict": "error_higher / correct_higher / n.s. (sig if either MW-p OR Wilcoxon-p < 0.05)",
            },
        },
        "results": {},
    }

    for pol_name, is_correct in policies:
        y_inc = (is_correct == 0).astype(int)
        contrastive = [p for p, idx in prob_idx.items()
                       if any(y_inc[idx] == 1) and any(y_inc[idx] == 0)]
        idx_groups = [np.array(prob_idx[p]) for p in contrastive]
        n_err = int((y_inc == 1).sum()); n_cor = int((y_inc == 0).sum())
        print(f"\n=== Policy '{pol_name}': {n_err} error / {n_cor} correct; "
              f"{len(contrastive)} contrastive problems ===")
        section: dict = {
            "_section_meta": {
                "policy": pol_name,
                "n_correct_chains": n_cor,
                "n_error_chains": n_err,
                "n_contrastive_problems": len(contrastive),
            },
        }
        for fname in sorted(feats.keys()):
            print(f"  {fname:>30s} ...", end="", flush=True)
            section[fname] = evaluate_feature(
                fname, feats[fname], y_inc, problem_ids, idx_groups,
                n_boot=args.bootstrap_n, seed=args.seed)
            v = section[fname]
            d = v["effect_size_signed_error_minus_correct"]["cohen_d"]
            au = v["auroc_cross_problem_signed"]["value"]
            wp = v["auroc_within_problem_paired_signed"]["value"]
            print(f"  d={d if isinstance(d, str) else f'{d:+.3f}':>7}  "
                  f"AUROC_cp={au if isinstance(au, str) else f'{au:.3f}':>5}  "
                  f"AUROC_wp={wp if isinstance(wp, str) else f'{wp:.3f}':>5}  "
                  f"verdict={v['direction_verdict']}")
        out["results"][pol_name] = section

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, sort_keys=False)
    print(f"\nSaved -> {args.output}  ({os.path.getsize(args.output) // 1024} KB)")


if __name__ == "__main__":
    main()
