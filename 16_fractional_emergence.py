"""Step 16: does the solution-FAILURE signal emerge over the chain? (rigorous)

14 showed within-problem failure AUROC rises over ABSOLUTE step position, but
late absolute positions only include LONG solutions (subset bias). Here we bin by
FRACTIONAL position (each solution's steps mapped to [0,1], then into n_bins), so
EVERY solution contributes to every bin -> no length-subset bias. We report the
within-problem (difficulty-controlled) and cross-problem failure AUROC per
fractional bin. If within rises from ~chance (early) to ~0.7 (late), the
solution-specific failure signal genuinely emerges as the reasoning unfolds,
while the cross (difficulty) signal stays flat.
"""

from __future__ import annotations

import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


def band_indices(L_sub, band):
    if band == "all":
        return np.arange(L_sub)
    if band == "deep":
        return np.arange(int(L_sub * 0.6), L_sub)
    if band == "mid":
        return np.arange(int(L_sub * 0.3), int(L_sub * 0.7))
    return np.array([int(x) for x in band.split(",") if x.strip()])


def within_pair_auroc(idx_groups, feats, y_inc):
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


def probe_oof(X, y, groups, k, seed, kind):
    N = X.shape[0]; rng = np.random.default_rng(seed)
    if kind == "group":
        uniq = np.unique(groups); rng.shuffle(uniq)
        fo = {int(g): i % k for i, g in enumerate(uniq)}
        fold = np.array([fo[int(g)] for g in groups])
    else:
        idx = np.arange(N); rng.shuffle(idx); fold = np.empty(N, int); fold[idx] = np.arange(N) % k
    oof = np.full(N, np.nan)
    for f in range(k):
        tr = fold != f; te = fold == f
        if tr.sum() < 20 or len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--layer_band", default="mid")
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/fractional_emergence.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    VEC = data[f"sv_vec_{args.mode}"]
    problem_ids = data["problem_ids"].astype(int)
    y = (data["is_correct"].astype(int) == 0).astype(int)
    N = len(VEC)
    L_sub = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    cols = band_indices(L_sub, args.layer_band)

    perstep = []
    for i in range(N):
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
        with np.errstate(invalid="ignore"):
            perstep.append(np.nanmean(V, axis=1))            # (T_i, d)

    prob_to_idx = {}
    for i, p in enumerate(problem_ids):
        prob_to_idx.setdefault(int(p), []).append(i)

    def eval_feature(X):
        idxs = np.where(np.isfinite(X).all(axis=1))[0]
        if idxs.size < 30:
            return np.nan, np.nan, idxs.size
        Xm, ym, gm = X[idxs], y[idxs], problem_ids[idxs]
        sset = {int(p): set(v) for p, v in prob_to_idx.items()}
        ig = []
        for p, v in prob_to_idx.items():
            loc = [j for j, ii in enumerate(idxs) if ii in sset[int(p)]]
            loc = np.array(loc)
            if loc.size and any(ym[loc] == 1) and any(ym[loc] == 0):
                ig.append(loc)
        win, cro = [], []
        for s in range(args.n_seeds):
            og = probe_oof(Xm, ym, gm, args.kfold, args.seed + s, "group")
            orr = probe_oof(Xm, ym, gm, args.kfold, args.seed + s, "random")
            win.append(within_pair_auroc(ig, og, ym)[0])
            cro.append(roc_auc_score(ym, orr))
        return max(np.nanmean(win), 1 - np.nanmean(win)), float(np.nanmean(cro)), idxs.size

    print(f"Loaded {N} chains, d={d}, band={args.layer_band}, mode={args.mode}; "
          f"fractional position, {args.n_bins} bins")
    print(f"\n=== Failure AUROC vs FRACTIONAL position (every solution contributes) ===")
    print(f"{'frac':>10s}  {'within':>7s}  {'cross':>7s}  {'gap':>7s}  {'n':>5s}")
    curve = []
    for b in range(args.n_bins):
        lo, hi = b / args.n_bins, (b + 1) / args.n_bins
        X = np.full((N, d), np.nan)
        for i in range(N):
            T = perstep[i].shape[0]
            fr = (np.arange(T) / (T - 1)) if T > 1 else np.array([0.0])
            sel = (fr >= lo) & (fr < hi if b < args.n_bins - 1 else fr <= hi)
            if sel.any():
                with np.errstate(invalid="ignore"):
                    X[i] = np.nanmean(perstep[i][sel], axis=0)
        w, c, n = eval_feature(X)
        curve.append((w, c))
        gap = (c - w) if np.isfinite(c) and np.isfinite(w) else np.nan
        print(f"{lo:.1f}-{hi:.1f}  {w:7.4f}  {c:7.4f}  {gap:+7.4f}  {n:5d}")

    np.savez(args.output, curve=np.array(curve, dtype=object),
             band=np.array(args.layer_band), n_bins=np.array(args.n_bins))
    print(f"\nSaved -> {args.output}")
    print("\nRead: within rising 0.5->0.7 across fractions = solution-failure EMERGES "
          "as reasoning unfolds; cross flat = difficulty present throughout.")


if __name__ == "__main__":
    main()
