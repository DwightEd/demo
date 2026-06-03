"""Step 15: are DIFFICULTY and FAILURE two different directions in activation space?

The temporal probe (14) showed: early steps encode problem DIFFICULTY (cross-
problem high, within-problem ~chance); the solution-specific FAILURE signal
emerges in the LATE steps (within-problem ~0.71). This script tests whether
these are geometrically SEPARABLE directions:

  - w_fail  : the within-problem failure direction (predict incorrect, late window)
  - w_diff  : the difficulty direction (predict the problem's fail-rate from the
              solution's activation -- a per-problem, difficulty/competence signal)
  - cosine(w_fail, w_diff): aligned (same axis) or orthogonal (two mechanisms)?
  - ORTHOGONALIZE: project the difficulty direction OUT of the features, re-run the
    failure probe. If failure AUROC survives -> failure is encoded in directions
    orthogonal to difficulty = a genuinely difficulty-independent mechanism.

All within-problem AUROCs use GroupKFold over problems (held-out, difficulty-
controlled). Feature = last-3-steps band-mean vector (where the failure signal is).
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


def failure_oof(Xs, y, groups, k, seed, C):
    oof = np.full(len(y), np.nan)
    for tr, te in group_folds(groups, k, seed):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(Xs[tr], y[tr]); oof[te] = clf.predict_proba(Xs[te])[:, 1]
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--layer_band", default="all")
    ap.add_argument("--window", type=int, default=3, help="last-N-steps window")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/direction_decomp.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    VEC = data[f"sv_vec_{args.mode}"]
    problem_ids = data["problem_ids"].astype(int)
    is_correct = data["is_correct"].astype(int)
    y = (is_correct == 0).astype(int)                         # 1 = incorrect
    N = len(VEC)
    L_sub = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    cols = band_indices(L_sub, args.layer_band)

    # last-W-steps band-mean vector per solution
    X = np.full((N, d), np.nan)
    for i in range(N):
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
        T = V.shape[0]
        with np.errstate(invalid="ignore"):
            X[i] = np.nanmean(V[max(0, T - args.window):].reshape(-1, d), axis=0)
    ok = np.isfinite(X).all(axis=1)
    X, y, problem_ids = X[ok], y[ok], problem_ids[ok]
    N = X.shape[0]

    # per-problem difficulty = fraction of the problem's solutions that FAIL
    prob_idx = {}
    for i, p in enumerate(problem_ids):
        prob_idx.setdefault(int(p), []).append(i)
    diff = np.zeros(N)
    for p, v in prob_idx.items():
        diff[v] = y[np.array(v)].mean()                       # fail-rate of problem p
    idx_groups = [np.array(v) for v in prob_idx.values()
                  if any(y[v] == 1) and any(y[v] == 0)]

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    hard = (diff > np.median(diff)).astype(int)

    print(f"Loaded {N} chains, {len(idx_groups)} contrastive problems, d={d}, "
          f"band={args.layer_band}, window=last{args.window}; "
          f"{int(y.sum())} incorrect / {int((1-y).sum())} correct")

    # ---- FULLY CROSS-FIT over problems (no in-sample leakage) ----
    # per fold: fit w_fail (logistic) and w_diff (ridge) on TRAIN problems; on TEST
    # problems collect (a) difficulty prediction, (b) failure prob, (c) failure prob
    # after projecting the difficulty axis OUT of the features (refit on residual).
    cosines = []
    raw_seeds, resid_seeds, diffauroc_seeds = [], [], []
    for s in range(args.n_seeds):
        diff_oof = np.full(N, np.nan); fail_oof = np.full(N, np.nan)
        fres_oof = np.full(N, np.nan)
        for tr, te in group_folds(problem_ids, args.kfold, args.seed + s):
            if len(np.unique(y[tr])) < 2:
                continue
            wf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced").fit(Xs[tr], y[tr])
            wd = Ridge(alpha=1.0).fit(Xs[tr], diff[tr])
            cf, cd = wf.coef_.ravel(), wd.coef_.ravel()
            cosines.append(float(np.dot(cf, cd) /
                                 (np.linalg.norm(cf) * np.linalg.norm(cd) + 1e-12)))
            diff_oof[te] = wd.predict(Xs[te])
            fail_oof[te] = wf.predict_proba(Xs[te])[:, 1]
            u = cd / (np.linalg.norm(cd) + 1e-12)             # difficulty axis (train-only)
            Xtr_r = Xs[tr] - np.outer(Xs[tr] @ u, u)
            Xte_r = Xs[te] - np.outer(Xs[te] @ u, u)
            wfr = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced").fit(Xtr_r, y[tr])
            fres_oof[te] = wfr.predict_proba(Xte_r)[:, 1]
        raw_seeds.append(max(within_pair_auroc(idx_groups, fail_oof, y)[0],
                             1 - within_pair_auroc(idx_groups, fail_oof, y)[0]))
        resid_seeds.append(max(within_pair_auroc(idx_groups, fres_oof, y)[0],
                               1 - within_pair_auroc(idx_groups, fres_oof, y)[0]))
        m = np.isfinite(diff_oof)
        diffauroc_seeds.append(roc_auc_score(hard[m], diff_oof[m])
                               if len(np.unique(hard[m])) == 2 else np.nan)

    cos = float(np.mean(cosines)); cos_s = float(np.std(cosines))
    diff_auroc = float(np.nanmean(diffauroc_seeds))
    a0, s0 = float(np.mean(raw_seeds)), float(np.std(raw_seeds))
    a1, s1 = float(np.mean(resid_seeds)), float(np.std(resid_seeds))

    print(f"\n=== Difficulty vs Failure directions (last-{args.window} window, "
          f"band={args.layer_band}; ALL cross-fit, held-out) ===")
    print(f"  cosine(w_fail, w_diff)                      = {cos:+.4f} +/- {cos_s:.4f}")
    print(f"  activation predicts difficulty (held-out AUROC) = {diff_auroc:.4f}")
    print(f"\n  within-problem FAILURE AUROC, raw           = {a0:.4f} +/- {s0:.4f}")
    print(f"  within-problem FAILURE AUROC, difficulty-OUT = {a1:.4f} +/- {s1:.4f}")
    print(f"  -> failure survives removing difficulty axis: delta {a1 - a0:+.4f}")
    if abs(cos) < 0.25 and a1 > a0 - 0.03:
        print("  => difficulty & failure largely SEPARABLE: failure is not just "
              "difficulty re-encoded (survives removing the difficulty axis).")
    elif a1 < a0 - 0.05:
        print("  => removing difficulty hurts failure: they share substantial axis.")
    else:
        print("  => partially separable; see cosine + delta.")

    np.savez(args.output, cosine=np.array(cos), diff_auroc=np.array(diff_auroc),
             fail_auroc_raw=np.array(a0), fail_auroc_diffout=np.array(a1),
             band=np.array(args.layer_band))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
