"""Step 22: train on the within-problem CONTRASTS directly (the right objective).

Every probe so far (12/18/20) is trained as a POOLED classifier -- input x, label y
(correct/incorrect) over all solutions -- but EVALUATED by within-problem PAIRED AUROC
(does the failing sibling score above the succeeding one). Objective != evaluation: the
pooled classifier spends capacity on the cross-problem difficulty axis, which the paired
metric does not reward.

The correct way to USE and AMPLIFY the favorable within-problem signal is to train on the
within-problem contrasts themselves. For each problem, form difference vectors
    d = x_incorrect - x_correct
over its (incorrect, correct) sibling pairs, and learn the direction w that ranks them:
    w . x_inc > w . x_cor   <=>   w . d > 0.
A logistic on {(d, +1)} u {(-d, -1)} (no intercept) optimizes EXACTLY the within-pair
AUROC objective. The subtraction also cancels the problem-level component a_p
(difficulty/content) for free, so w is forced onto the pure within-problem failure axis --
the fixed-effects transform done at the pair level, but supervised on the right target.

Rigor: pairs are built ONLY from TRAIN problems (GroupKFold); w is scored on held-out
problems' solutions; within-pair AUROC on the held-out predictions. Compared head-to-head
with the pooled probe on the same folds. Swept over band and (optionally) PCA denoising.
"""

from __future__ import annotations

import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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


def group_folds(groups, k, seed):
    uniq = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(uniq)
    fo = {int(g): i % k for i, g in enumerate(uniq)}
    f = np.array([fo[int(g)] for g in groups])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def build_pairs(X, y, problem_ids, idxs, max_pairs_per_problem=60, seed=0):
    """Within-problem difference vectors d = x_inc - x_cor over TRAIN problems."""
    rng = np.random.default_rng(seed)
    prob = {}
    for i in idxs:
        prob.setdefault(int(problem_ids[i]), []).append(i)
    diffs = []
    for p, v in prob.items():
        inc = [i for i in v if y[i] == 1]
        cor = [i for i in v if y[i] == 0]
        if not inc or not cor:
            continue
        pairs = [(a, b) for a in inc for b in cor]
        if len(pairs) > max_pairs_per_problem:
            sel = rng.choice(len(pairs), max_pairs_per_problem, replace=False)
            pairs = [pairs[s] for s in sel]
        for a, b in pairs:
            diffs.append(X[a] - X[b])
    return np.asarray(diffs)


def fit_pairwise(D, C, max_iter=3000):
    """Logistic on {(d,+1),(-d,-1)} with no intercept -> direction ranking the pairs."""
    Xtr = np.vstack([D, -D])
    ytr = np.concatenate([np.ones(len(D)), np.zeros(len(D))])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=max_iter, fit_intercept=False).fit(sc.transform(Xtr), ytr)
    return sc, clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--bands", default="mid,all,deep")
    ap.add_argument("--late_lo", type=float, default=0.6,
                    help="use the late-window band-mean vector (frac >= late_lo)")
    ap.add_argument("--pca_k", type=int, default=0, help="PCA denoise dim (0 = off)")
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--max_pairs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/pairwise.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    VEC = data[f"sv_vec_{args.mode}"]
    problem_ids = data["problem_ids"].astype(int)
    y = (data["is_correct"].astype(int) == 0).astype(int)
    N = len(VEC)
    L_sub = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]

    results = {}
    for band in args.bands.split(","):
        band = band.strip()
        cols = band_indices(L_sub, band)
        # late-window band-mean vector per solution
        X = np.full((N, d), np.nan)
        for i in range(N):
            V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
            with np.errstate(invalid="ignore"):
                P = np.nanmean(V, axis=1)
            T = P.shape[0]
            fr = (np.arange(T) / (T - 1)) if T > 1 else np.array([0.0])
            m = fr >= args.late_lo
            if not m.any():
                m = fr >= fr.max()
            with np.errstate(invalid="ignore"):
                X[i] = np.nanmean(P[m], axis=0)
        ok = np.isfinite(X).all(1)
        Xb, yb, pb = X[ok], y[ok], problem_ids[ok]
        prob = {}
        for i, p in enumerate(pb):
            prob.setdefault(int(p), []).append(i)
        idx_groups = [np.array(v) for v in prob.values() if any(yb[v] == 1) and any(yb[v] == 0)]

        pooled_s, pair_s = [], []
        for s in range(args.n_seeds):
            oof_pool = np.full(len(yb), np.nan); oof_pair = np.full(len(yb), np.nan)
            for tr, te in group_folds(pb, args.kfold, args.seed + s):
                if len(np.unique(yb[tr])) < 2:
                    continue
                Xtr, Xte = Xb[tr], Xb[te]
                if args.pca_k:
                    pca = PCA(n_components=min(args.pca_k, Xtr.shape[1]), random_state=0).fit(Xtr)
                    Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
                # pooled classifier (the old way)
                scp = StandardScaler().fit(Xtr)
                clp = LogisticRegression(C=args.C, max_iter=3000,
                                         class_weight="balanced").fit(scp.transform(Xtr), yb[tr])
                oof_pool[te] = clp.decision_function(scp.transform(Xte))
                # pairwise within-problem contrast (the right objective)
                tr_local = np.arange(len(tr))
                D = build_pairs(Xtr, yb[tr], pb[tr], tr_local,
                                max_pairs_per_problem=args.max_pairs, seed=args.seed + s)
                if len(D) < 10:
                    continue
                sc2, cl2 = fit_pairwise(D, args.C)
                oof_pair[te] = cl2.decision_function(sc2.transform(Xte))
            a0 = within_pair_auroc(idx_groups, oof_pool, yb)[0]
            a1 = within_pair_auroc(idx_groups, oof_pair, yb)[0]
            pooled_s.append(max(a0, 1 - a0)); pair_s.append(max(a1, 1 - a1))
        pm, ps = float(np.mean(pooled_s)), float(np.std(pooled_s))
        qm, qs = float(np.mean(pair_s)), float(np.std(pair_s))
        results[band] = (pm, ps, qm, qs)

    print(f"=== pooled-classifier vs within-problem PAIRWISE training "
          f"(PCA={args.pca_k or 'off'}, {args.n_seeds} seeds) ===")
    print(f"{'band':6s}  {'pooled (old)':>16s}   {'pairwise (new)':>16s}   {'delta':>7s}")
    for band, (pm, psd, qm, qsd) in results.items():
        print(f"{band:6s}  {pm:.4f} +/- {psd:.4f}   {qm:.4f} +/- {qsd:.4f}   {qm - pm:+.4f}")
    best = max(results.items(), key=lambda kv: kv[1][2])
    print(f"\n  best pairwise: band={best[0]}  within={best[1][2]:.4f}  "
          f"(pooled {best[1][0]:.4f}, delta {best[1][2]-best[1][0]:+.4f})")
    if best[1][2] > best[1][0] + 0.01:
        print("  -> training on the within-problem contrasts AMPLIFIES the signal "
              "(optimizing the evaluated objective beats pooled classification).")
    else:
        print("  -> pairwise ~ pooled: difficulty is already near-orthogonal, so pooled "
              "was already close to the within-problem axis.")

    np.savez(args.output,
             bands=np.array(list(results.keys()), dtype=object),
             pooled=np.array([results[b][0] for b in results]),
             pairwise=np.array([results[b][2] for b in results]),
             pca_k=np.array(args.pca_k))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
