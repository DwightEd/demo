"""Quick discriminability report for an extract_features.py npz.

For every chain-level feature, computes:
  within  : within-PROBLEM paired AUROC = P(feat_error > feat_correct) over all
            (error, correct) chain pairs that share a problem_id  -- difficulty
            controlled by construction (the metric that matters).
  cross   : pooled AUROC over all chains (difficulty NOT controlled; the gap
            cross-within is the difficulty inflation).
  d       : Cohen's d (error - correct) / pooled_std.

Chain-level features:
  - paper trace-profile columns (UD_*, UC_*, UE_* : early/mid/late/slope/r2)
  - geometry: per (layer, feature) mean and late-window mean of the per-STEP
    series (norm, pr, ae, ed_half, e50, e90, ae_robust, anom_k5, anom_k10).

Labels: is_correct_strict (primary) and is_correct (lenient / answer-only).
Run:  python analyze_features.py data/features/sampled_v2_5shot_features.npz
"""

from __future__ import annotations

import argparse
import numpy as np


def auroc_cross(score, y):
    """AUROC with positive class y==1 (=error). Mann-Whitney rank statistic."""
    m = np.isfinite(score)
    s, yy = score[m], y[m]
    npos, nneg = int((yy == 1).sum()), int((yy == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sr = s[order]
    i = 0
    while i < len(sr):                       # average ranks for ties
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rsum = ranks[yy == 1].sum()
    return (rsum - npos * (npos + 1) / 2.0) / (npos * nneg)


def auroc_within(score, y, pid):
    """Pooled within-problem paired AUROC over (error, correct) pairs."""
    conc = tie = npair = 0.0
    for p in np.unique(pid):
        m = (pid == p) & np.isfinite(score)
        se, sc = score[m & (y == 1)], score[m & (y == 0)]
        if se.size == 0 or sc.size == 0:
            continue
        diff = se[:, None] - sc[None, :]
        conc += (diff > 0).sum()
        tie += (diff == 0).sum()
        npair += diff.size
    if npair == 0:
        return float("nan"), 0
    return (conc + 0.5 * tie) / npair, int(npair)


def cohen_d(score, y):
    m = np.isfinite(score)
    e, c = score[m & (y == 1)], score[m & (y == 0)]
    if e.size < 2 or c.size < 2:
        return float("nan")
    sp = np.sqrt(((e.size - 1) * e.var(ddof=1) + (c.size - 1) * c.var(ddof=1))
                 / (e.size + c.size - 2))
    return (e.mean() - c.mean()) / sp if sp > 0 else float("nan")


def late_window(series, frac=0.25):
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    k = max(1, int(round(x.size * frac)))
    return float(x[-k:].mean())


def build_features(z):
    """Return dict name -> (N,) chain-level array."""
    feats = {}
    # length baseline (the signal that geometry must beat)
    if "n_steps" in z.files:
        feats["n_steps"] = z["n_steps"].astype(float)
    if "n_resp_tokens" in z.files:
        feats["n_resp_tokens"] = z["n_resp_tokens"].astype(float)
    # paper trace-profile columns (already chain-level)
    cols = [str(c) for c in z["profile_cols"]]
    P = z["profile_paper"]
    for j, c in enumerate(cols):
        feats[c] = P[:, j]
    # geometry per (layer, feature): mean + late-window of the per-step series
    names = [str(x) for x in z["geom_feature_names"]]
    layers = [int(x) for x in z["layers_used"]]
    SG = z["stepgeom"]            # object: each (T, L, F)
    N = len(SG)
    for li, lyr in enumerate(layers):
        for fi, fn in enumerate(names):
            mean_arr = np.full(N, np.nan)
            late_arr = np.full(N, np.nan)
            for i in range(N):
                g = np.asarray(SG[i], float)
                if g.ndim == 3 and g.shape[1] > li and g.shape[2] > fi:
                    s = g[:, li, fi]
                    mean_arr[i] = np.nanmean(s) if np.isfinite(s).any() else np.nan
                    late_arr[i] = late_window(s)
            feats[f"{fn}_L{lyr}_mean"] = mean_arr
            feats[f"{fn}_L{lyr}_late"] = late_arr
    # optional point-cloud effective rank D / energy V / concentration C
    if "stepcloud" in z.files and bool(z.get("cloud_stored", np.array(False))):
        cnames = [str(x) for x in z["cloud_feature_names"]]
        SC = z["stepcloud"]
        for li, lyr in enumerate(layers):
            for fi, fn in enumerate(cnames):
                mean_arr = np.full(N, np.nan)
                late_arr = np.full(N, np.nan)
                for i in range(N):
                    g = np.asarray(SC[i], float)
                    if g.ndim == 3 and g.shape[1] > li and g.shape[2] > fi:
                        s = g[:, li, fi]
                        mean_arr[i] = np.nanmean(s) if np.isfinite(s).any() else np.nan
                        late_arr[i] = late_window(s)
                feats[f"{fn}_L{lyr}_mean"] = mean_arr
                feats[f"{fn}_L{lyr}_late"] = late_arr
    # whole-chain nonlinear intrinsic dimension (already chain-level, one per layer)
    if "chain_intrinsic" in z.files and bool(z.get("intrinsic_stored", np.array(False))):
        inames = [str(x) for x in z["intrinsic_names"]]
        CI = np.asarray(z["chain_intrinsic"], float)          # (N, L, n_est)
        for li, lyr in enumerate(layers):
            for ei, en in enumerate(inames):
                feats[f"{en}_L{lyr}"] = CI[:, li, ei]
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--sort", default="within_strict",
                    choices=["within_strict", "cross_strict", "within_lenient"])
    ap.add_argument("--format_ok_only", action="store_true",
                    help="restrict to format_ok==1 chains (drops '#### missing' "
                         "format failures, so strict==answer-only -> isolates the "
                         "reasoning signal from the format confound).")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    pid = z["problem_ids"].astype(int)
    y_strict = (z["is_correct_strict"].astype(int) == 0).astype(int)   # error=1
    y_lenient = (z["is_correct"].astype(int) == 0).astype(int)
    feats = build_features(z)

    keep = np.ones(len(pid), bool)
    if args.format_ok_only:
        if "format_ok" in z.files:
            fok = z["format_ok"].astype(int)
        elif "responses" in z.files:
            import re
            fok = np.array([1 if re.search(r"####", str(r)) else 0
                            for r in z["responses"]], int)
            print("  (format_ok not stored; recomputed from responses via '####')")
        else:
            fok = None
            print("  WARN: --format_ok_only requested but no format_ok/responses "
                  "in npz -> filter IGNORED (showing full data).")
        if fok is not None:
            keep = fok == 1
            print(f"[format_ok=1 subset: {int(keep.sum())}/{len(keep)} chains]")
    pid, y_strict, y_lenient = pid[keep], y_strict[keep], y_lenient[keep]
    feats = {k: v[keep] for k, v in feats.items()}
    N = len(pid)
    n_contrastive = sum(
        1 for p in np.unique(pid)
        if ((y_strict[pid == p] == 1).any() and (y_strict[pid == p] == 0).any()))

    print(f"file: {args.npz}")
    print(f"chains: {N} | problems: {len(np.unique(pid))} | "
          f"contrastive(strict): {n_contrastive} | "
          f"error rate strict: {y_strict.mean():.3f} lenient: {y_lenient.mean():.3f}")
    if "ue_on" in z.files:
        print(f"U_E stored: {bool(z['ue_on'])} | "
              f"layers: {[int(x) for x in z['layers_used']]}")

    rows = []
    for name, s in feats.items():
        wS, npair = auroc_within(s, y_strict, pid)
        wL, _ = auroc_within(s, y_lenient, pid)
        rows.append({
            "name": name,
            "within_strict": wS,
            "cross_strict": auroc_cross(s, y_strict),
            "d_strict": cohen_d(s, y_strict),
            "within_lenient": wL,
            "npair": npair,
        })

    def key(r):
        v = r[args.sort]
        return -abs((v if np.isfinite(v) else 0.5) - 0.5)
    rows.sort(key=key)

    print(f"\n{'feature':28s} {'within_str':>10s} {'cross_str':>9s} "
          f"{'d_str':>7s} {'within_len':>10s}")
    print("-" * 70)
    for r in rows[:args.top]:
        print(f"{r['name']:28s} {r['within_strict']:10.3f} {r['cross_strict']:9.3f} "
              f"{r['d_strict']:7.2f} {r['within_lenient']:10.3f}")
    print("\n(within>0.5 => error has HIGHER feature; <0.5 => correct higher. "
          "within vs cross gap = difficulty inflation.)")


if __name__ == "__main__":
    main()
