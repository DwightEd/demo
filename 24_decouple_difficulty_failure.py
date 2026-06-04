"""Step 24: is difficulty-vs-failure separability a GEOMETRIC structure, not just a number?

We have two learned linear directions (all cross-fit over problems):
  w_fail : logistic predicting WITHIN-problem failure (sample-level)
  w_diff : ridge predicting the problem's fail-rate from activation (problem-level difficulty)

The worry: reasoning errors may be CAUSED by hard problems, and "hard" and "failed" might
both look "diffuse" -> the two could be the same direction. A single cosine is weak
(and in d~4096, cos=0.24 is ~15 sigma above the ~0 of random directions, i.e. they DO
share a real component). So we give a stronger, functional test: the 2x2 TRANSFER MATRIX.

Score every held-out solution by each direction, then ask each direction to do BOTH jobs:
                       within-problem FAILURE        problem DIFFICULTY (per-problem)
  w_fail               ~0.71 (its job)               ? -> if ~chance, fail-dir is difficulty-blind
  w_diff               ? -> if ~chance, diff-dir      ~0.60-0.65 (its job)
                            is within-failure-blind
If BOTH off-diagonals collapse to chance, the two signals occupy functionally distinct
directions (each predicts only its own target) -> geometric decoupling, far stronger than
a cosine. We also report cos(w_fail,w_diff) WITH the random-direction baseline (1/sqrt(d)),
and the variance of w_fail that lies along w_diff (cos^2).
"""

from __future__ import annotations

import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
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


def group_folds(groups, k, seed):
    uniq = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(uniq)
    fo = {int(g): i % k for i, g in enumerate(uniq)}
    f = np.array([fo[int(g)] for g in groups])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def bd(a):
    return max(a, 1 - a) if np.isfinite(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--layer_band", default="mid")
    ap.add_argument("--late_lo", type=float, default=0.6)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/decouple.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    VEC = data[f"sv_vec_{args.mode}"]
    problem_ids = data["problem_ids"].astype(int)
    y = (data["is_correct"].astype(int) == 0).astype(int)             # 1 = incorrect
    N = len(VEC)
    L_sub = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    cols = band_indices(L_sub, args.layer_band)

    # late-window band-mean vector per solution
    X = np.full((N, d), np.nan)
    for i in range(N):
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
        with np.errstate(invalid="ignore"):
            P = np.nanmean(V, axis=1)
        T = P.shape[0]
        fr = (np.arange(T) / (T - 1)) if T > 1 else np.array([0.0])
        m = fr >= args.late_lo
        if not m.any(): m = fr >= fr.max()
        with np.errstate(invalid="ignore"):
            X[i] = np.nanmean(P[m], axis=0)
    ok = np.isfinite(X).all(1)
    X, y, problem_ids = X[ok], y[ok], problem_ids[ok]
    N = len(y)

    prob = {}
    for i, p in enumerate(problem_ids):
        prob.setdefault(int(p), []).append(i)
    diff = np.zeros(N)                                                # per-problem fail-rate
    for p, v in prob.items():
        diff[v] = y[np.array(v)].mean()
    hard = (diff > np.median(diff)).astype(int)
    idx_groups = [np.array(v) for v in prob.values() if any(y[v] == 1) and any(y[v] == 0)]

    print(f"Loaded {N} solutions over {len(prob)} problems ({len(idx_groups)} contrastive); "
          f"d={d}, band={args.layer_band}; random-direction cos baseline ~ {1/np.sqrt(d):.3f}")

    def build_diffs(Xs, ytr, gtr, seed, max_pairs=60):
        rng = np.random.default_rng(seed); pr = {}
        for i in range(len(ytr)):
            pr.setdefault(int(gtr[i]), []).append(i)
        D = []
        for p, v in pr.items():
            inc = [i for i in v if ytr[i] == 1]; cor = [i for i in v if ytr[i] == 0]
            pairs = [(a, b) for a in inc for b in cor]
            if len(pairs) > max_pairs:
                pairs = [pairs[s] for s in rng.choice(len(pairs), max_pairs, replace=False)]
            for a, b in pairs:
                D.append(Xs[a] - Xs[b])
        return np.asarray(D)

    cosP, cosC = [], []                   # cos(pooled fail, pooled diff) ; cos(clean fail, clean diff)
    ff, df_fail, dd, fd_corr = [], [], [], []
    for s in range(args.n_seeds):
        sfc = np.full(N, np.nan); sdc = np.full(N, np.nan)
        for tr, te in group_folds(problem_ids, args.kfold, args.seed + s):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            ytr, gtr, difftr = y[tr], problem_ids[tr], diff[tr]
            # --- POOLED (contaminated) directions ---
            wfp = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced").fit(Xtr, ytr).coef_.ravel()
            wdp = Ridge(alpha=1.0).fit(Xtr, difftr).coef_.ravel()
            cosP.append(float(wfp @ wdp / (np.linalg.norm(wfp) * np.linalg.norm(wdp) + 1e-12)))
            # --- CLEAN directions ---
            # failure: within-problem differences cancel the problem-level (difficulty) component
            Dpairs = build_diffs(Xtr, ytr, gtr, args.seed + s)
            if len(Dpairs) < 10:
                continue
            Xd = np.vstack([Dpairs, -Dpairs]); yd = np.concatenate([np.ones(len(Dpairs)), np.zeros(len(Dpairs))])
            wfc = LogisticRegression(C=args.C, max_iter=3000, fit_intercept=False).fit(Xd, yd).coef_.ravel()
            # difficulty: estimated from CORRECT solutions only (no failure component present)
            cmask = ytr == 0
            wdc = Ridge(alpha=1.0).fit(Xtr[cmask], difftr[cmask]).coef_.ravel()
            cosC.append(float(wfc @ wdc / (np.linalg.norm(wfc) * np.linalg.norm(wdc) + 1e-12)))
            sfc[te] = Xte @ wfc; sdc[te] = Xte @ wdc

        ff.append(bd(within_pair_auroc(idx_groups, sfc, y)[0]))           # clean w_fail -> within-failure (job)
        df_fail.append(bd(within_pair_auroc(idx_groups, sdc, y)[0]))      # clean w_diff -> within-failure (KEY off)
        pids = list(prob.keys())
        sd_p = np.array([np.nanmean(sdc[np.array(prob[p])]) for p in pids])
        sf_p_corr = np.array([np.nanmean(sfc[np.array([i for i in prob[p] if y[i] == 0])])
                              if any(y[i] == 0 for i in prob[p]) else np.nan for p in pids])
        ph = np.array([hard[np.array(prob[p])][0] for p in pids])
        m1 = np.isfinite(sd_p)
        if len(np.unique(ph[m1])) == 2:
            dd.append(bd(roc_auc_score(ph[m1], sd_p[m1])))               # clean w_diff -> difficulty (job)
        m2 = np.isfinite(sf_p_corr)
        if len(np.unique(ph[m2])) == 2:
            fd_corr.append(bd(roc_auc_score(ph[m2], sf_p_corr[m2])))     # clean w_fail -> difficulty|correct (off)

    cp = float(np.mean(cosP)); cc = float(np.mean(cosC))
    A = float(np.mean(ff)); B = float(np.mean(df_fail))
    C = float(np.mean(dd)); D = float(np.mean(fd_corr)) if fd_corr else float("nan")

    print(f"\n=== cosine of failure vs difficulty directions (random ~ {1/np.sqrt(d):.3f}) ===")
    print(f"  POOLED estimation  : cos = {cp:+.3f}   <- contaminated (failures cluster in hard problems)")
    print(f"  CLEAN  estimation  : cos = {cc:+.3f}   <- failure from within-problem contrasts, difficulty from correct-only")
    print(f"  -> if POOLED high but CLEAN ~0: the apparent coupling is an estimation artifact, "
          f"the underlying directions are distinct.")
    print(f"\n=== transfer tests on CLEAN directions (chance = 0.50) ===")
    print(f"  [diag] clean w_fail -> within-failure        = {A:.3f}   (its job)")
    print(f"  [diag] clean w_diff -> per-problem difficulty = {C:.3f}  (its job)")
    print(f"  [KEY off] clean w_diff -> within-failure     = {B:.3f}   (expect ~chance)")
    print(f"  [off] clean w_fail -> difficulty|correct     = {D:.3f}   (expect ~chance)")
    if B < 0.58 and (not np.isfinite(D) or D < 0.60):
        print("\n  => GEOMETRIC DECOUPLING confirmed (clean directions): difficulty cannot pick "
              "the failing solution within a problem; failure does not track difficulty among "
              "correct solutions. Two functionally distinct directions.")
    else:
        print("\n  => residual coupling on a clean off-diagonal -> report honestly.")

    np.savez(args.output, cos_pooled=np.array(cp), cos_clean=np.array(cc),
             rand_cos=np.array(1/np.sqrt(d)),
             diag_fail=np.array(A), diag_diff=np.array(C),
             off_diff_to_fail=np.array(B), off_fail_to_diff_correct=np.array(D),
             band=np.array(args.layer_band))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
