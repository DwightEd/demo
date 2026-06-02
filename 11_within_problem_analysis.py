"""Step 11 (Phase 2): within-problem analysis -- error vs difficulty, decided.

Consumes data/<tag>_multisample_sv.npz from 10_sample_and_extract.py, where for
the SAME problem we have several sampled solutions, each labelled correct/incorrect
by final-answer match.

The gate question: holding the PROBLEM fixed, do the FAILING samples have higher
activation participation than the SUCCEEDING ones?
  - YES (within-problem AUROC > 0.5, paired diff > 0, significant)
        => difficulty is controlled out; participation is a genuine
           failure-prediction signal. (b) wins.
  - NO  (within-problem AUROC ~ 0.5)
        => the cross-problem signal was just difficulty. (a) wins.

Reported:
  (A) Within-problem AUROC per weighting mode (the gate; difficulty-controlled).
  (B) Paired test across problems: median diff, %positive, Wilcoxon signed-rank p.
  (C) Early-window vs full chain (does the EARLY prefix predict failure?  -> the
      08(f) / Streaming-HD Obs-2 'early is more separable' claim, now length- and
      difficulty-controlled because we fix the window and the problem).
  (D) Per-step-position separation curve (where along the chain it splits).
  (E) Pooled (NOT difficulty-controlled) AUROC + bootstrap CI, for context only.
"""

from __future__ import annotations

import argparse
import math

import numpy as np


# --- metrics ---------------------------------------------------------------

def auroc(scores, labels):
    """AUROC with FIXED direction: higher score => label==1."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    m = ~np.isnan(scores)
    scores, labels = scores[m], labels[m]
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); start = csum - counts
    ranks = ((start + csum + 1) / 2.0)[inv]
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def _ranks(x):
    """Average ranks (1-based), ties averaged."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, dtype=np.float64)
    r[order] = np.arange(1, x.size + 1)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size); np.add.at(sums, inv, r)
    return (sums / counts)[inv]


def _norm_sf(z):  # one-sided survival of |z|
    return 0.5 * math.erfc(abs(z) / math.sqrt(2.0))


def wilcoxon_signed_rank(d):
    """Two-sided Wilcoxon signed-rank (normal approx, tie/zero handled).
    Returns (z, p, n_nonzero, frac_positive)."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d) & (d != 0)]
    n = d.size
    if n < 6:
        return float("nan"), float("nan"), n, float("nan")
    r = _ranks(np.abs(d))
    W = float(r[d > 0].sum())
    mu = n * (n + 1) / 4.0
    # tie correction
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie = (counts ** 3 - counts).sum()
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0)
    if sigma == 0:
        return float("nan"), float("nan"), n, float("nan")
    z = (W - mu) / sigma
    p = 2.0 * _norm_sf(z)
    return z, p, n, float((d > 0).mean())


def bootstrap_auroc_ci(scores, labels, n_boot=2000, seed=0, alpha=0.05):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = scores.size
    vals = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        a = auroc(scores[b], labels[b])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def band_indices(L_sub, band):
    if band == "all":
        return np.arange(L_sub)
    if band == "deep":
        return np.arange(int(L_sub * 0.6), L_sub)
    if band == "mid":
        return np.arange(int(L_sub * 0.3), int(L_sub * 0.7))
    if band == "early":
        return np.arange(0, max(1, int(L_sub * 0.3)))
    return np.array([int(x) for x in band.split(",") if x.strip()])


def per_step_band_avg(M, cols):
    M = np.asarray(M, dtype=np.float64)[:, cols]
    with np.errstate(invalid="ignore"):
        return np.nanmean(M, axis=1)        # (T,)


def _finite_mean(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="multisample npz from 10_sample_and_extract.py")
    ap.add_argument("--metric", default="ae", choices=["pr", "ae"])
    ap.add_argument("--layer_band", default="deep")
    ap.add_argument("--mode", default="step_exp",
                    help="weighting mode for the detailed test (b)-(e)")
    ap.add_argument("--early_window", type=int, default=3,
                    help="first-N-steps window for the early-prefix test (C)")
    ap.add_argument("--min_per_class", type=int, default=1,
                    help="min correct AND incorrect samples for a problem to "
                         "count as contrastive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/within_problem_analysis.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if not bool(data.get("sv_stored", np.array(False))):
        raise SystemExit("npz has no step vectors (sv_stored False).")

    modes = [str(m) for m in data["sv_modes"]]
    problem_ids = data["problem_ids"].astype(int)
    is_correct = data["is_correct"].astype(int)
    n_steps = data["n_steps"].astype(int)
    N = problem_ids.size
    y_inc = (is_correct == 0).astype(int)            # 1 = incorrect (failing)

    PR = {m: data[f"sv_{args.metric}_{m}"] for m in modes}
    L_sub = PR[modes[0]][0].shape[1]
    cols = band_indices(L_sub, args.layer_band)
    metric_name = "participation_ratio" if args.metric == "pr" else "activation_entropy"

    # group samples by problem
    prob_to_idx = {}
    for i, p in enumerate(problem_ids):
        prob_to_idx.setdefault(int(p), []).append(i)
    contrastive = [p for p, idx in prob_to_idx.items()
                   if (y_inc[idx] == 1).sum() >= args.min_per_class
                   and (y_inc[idx] == 0).sum() >= args.min_per_class]

    print(f"Loaded {N} samples over {len(prob_to_idx)} problems "
          f"(metric={metric_name}, band={args.layer_band} cols={[int(c) for c in cols]})")
    print(f"  overall: {int((is_correct == 1).sum())} correct / "
          f"{int((is_correct == 0).sum())} incorrect samples")
    print(f"  contrastive problems (>= {args.min_per_class} of each class): "
          f"{len(contrastive)}")
    if not contrastive:
        raise SystemExit("No contrastive problems; raise K / temperature / n_problems.")

    def feat_full(i, m):
        return _finite_mean(per_step_band_avg(PR[m][i], cols))

    def feat_early(i, m, w):
        return _finite_mean(per_step_band_avg(PR[m][i], cols)[:w])

    # ---- (A) within-problem AUROC per mode (the gate) ----
    # For each contrastive problem, AUROC(feat ; incorrect=1) using only that
    # problem's samples; then average across problems (difficulty-controlled).
    print("\n=== (A) WITHIN-PROBLEM AUROC (difficulty-controlled gate; "
          ">0.5 => failing samples have higher participation) ===")
    print(f"{'mode':10s}  {'wAUROC_full':>11s}  {'wAUROC_early':>12s}  {'#probs':>6s}")
    A = {}
    for m in modes:
        au_full, au_early = [], []
        for p in contrastive:
            idx = np.array(prob_to_idx[p])
            lab = y_inc[idx]
            ff = np.array([feat_full(i, m) for i in idx])
            fe = np.array([feat_early(i, m, args.early_window) for i in idx])
            af = auroc(ff, lab); ae_ = auroc(fe, lab)
            if not np.isnan(af):
                au_full.append(af)
            if not np.isnan(ae_):
                au_early.append(ae_)
        A[m] = dict(full=float(np.mean(au_full)) if au_full else float("nan"),
                    early=float(np.mean(au_early)) if au_early else float("nan"),
                    n=len(au_full))
        print(f"{m:10s}  {A[m]['full']:11.4f}  {A[m]['early']:12.4f}  {A[m]['n']:6d}")

    # ---- (B) paired test across problems for the chosen mode ----
    m = args.mode if args.mode in modes else modes[0]
    diffs_full, diffs_early = [], []
    for p in contrastive:
        idx = np.array(prob_to_idx[p])
        inc = idx[y_inc[idx] == 1]; cor = idx[y_inc[idx] == 0]
        df = (_finite_mean([feat_full(i, m) for i in inc]) -
              _finite_mean([feat_full(i, m) for i in cor]))
        de = (_finite_mean([feat_early(i, m, args.early_window) for i in inc]) -
              _finite_mean([feat_early(i, m, args.early_window) for i in cor]))
        diffs_full.append(df); diffs_early.append(de)
    diffs_full = np.array(diffs_full); diffs_early = np.array(diffs_early)

    print(f"\n=== (B) Paired within-problem diff (incorrect - correct), "
          f"mode={m} ===")
    for tag, d in [("full ", diffs_full), ("early", diffs_early)]:
        z, pval, n, fpos = wilcoxon_signed_rank(d)
        med = float(np.nanmedian(d))
        print(f"  {tag}: median delta={med:+.4f}  %positive={fpos*100 if fpos==fpos else float('nan'):5.1f}%"
              f"  Wilcoxon z={z:+.2f} p={pval:.2e}  (n={n})")

    # ---- (C) early vs full headline ----
    print(f"\n=== (C) Early-prefix vs full (mode={m}) ===")
    print(f"  within-problem AUROC: full={A[m]['full']:.4f}  "
          f"early(first {args.early_window})={A[m]['early']:.4f}")
    if A[m]['early'] >= A[m]['full'] - 0.01:
        print("  -> early window is at least as predictive as the full chain "
              "(supports early-warning, length- & difficulty-controlled).")
    else:
        print("  -> full chain more predictive than the early window here.")

    # ---- (D) per-step-position separation curve ----
    print(f"\n=== (D) Per-position within-problem AUROC (mode={m}; where it splits) ===")
    max_t = int(min(12, n_steps.max()))
    curve = []
    for t in range(max_t):
        aus = []
        for p in contrastive:
            idx = np.array(prob_to_idx[p])
            lab = y_inc[idx]
            vals = np.array([per_step_band_avg(PR[m][i], cols)[t]
                             if PR[m][i].shape[0] > t else np.nan for i in idx])
            if np.isfinite(vals).sum() >= 2 and len(np.unique(lab[np.isfinite(vals)])) == 2:
                a = auroc(vals, lab)
                if not np.isnan(a):
                    aus.append(a)
        mean_a = float(np.mean(aus)) if aus else float("nan")
        curve.append(mean_a)
        bar = "#" * int(max(0, (mean_a - 0.5) * 100)) if mean_a == mean_a else ""
        print(f"  step {t:2d}: wAUROC={mean_a:.4f}  (n={len(aus):3d})  {bar}")

    # ---- (E) pooled context (NOT difficulty-controlled) ----
    feat_all = np.array([feat_full(i, m) for i in range(N)])
    a_pool = auroc(feat_all, y_inc)
    lo, hi = bootstrap_auroc_ci(feat_all, y_inc, seed=args.seed)
    a_len = auroc(n_steps.astype(float), y_inc)
    print(f"\n=== (E) Pooled AUROC (context only; difficulty NOT controlled) ===")
    print(f"  AUROC(chain-mean {metric_name}, {m}) = {a_pool:.4f}  "
          f"[95% CI {lo:.4f}-{hi:.4f}]")
    print(f"  AUROC(n_steps)                        = {a_len:.4f}  (length proxy)")

    np.savez(args.output,
             modes=np.array(modes, dtype=object), mode=np.array(m),
             metric=np.array(args.metric), layer_band=np.array(args.layer_band),
             within_auroc_full=np.array([A[mm]["full"] for mm in modes]),
             within_auroc_early=np.array([A[mm]["early"] for mm in modes]),
             diffs_full=diffs_full, diffs_early=diffs_early,
             position_curve=np.array(curve),
             pooled_auroc=np.array(a_pool),
             pooled_ci=np.array([lo, hi]),
             n_contrastive=np.array(len(contrastive)))
    print(f"\nSaved -> {args.output}")

    # ---- verdict ----
    g = A[m]["full"]
    print("\n=== VERDICT (difficulty-controlled) ===")
    _, pval, _, _ = wilcoxon_signed_rank(diffs_full)
    if np.isfinite(g) and g > 0.55 and np.isfinite(pval) and pval < 0.05:
        print(f"  within-problem AUROC={g:.4f} (p={pval:.2e}) -> participation "
              "PREDICTS failure with difficulty controlled. (b) supported.")
    elif np.isfinite(g) and g < 0.53:
        print(f"  within-problem AUROC={g:.4f} ~ chance -> the cross-problem "
              "signal was DIFFICULTY, not error. (a). Pivot needed.")
    else:
        print(f"  within-problem AUROC={g:.4f} (p={pval:.2e}) -> inconclusive; "
              "need more contrastive problems / samples.")


if __name__ == "__main__":
    main()
