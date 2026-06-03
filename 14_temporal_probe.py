"""Step 14: temporal decomposition — per-step probe, cross-problem vs within-problem.

The chain-mean probe (12) averages over ALL steps, which (a) is cruder than the
Streaming-HD per-step design and (b) mixes discriminative early/mid steps with
anti-discriminative late steps. This script instead probes EACH STEP POSITION
separately and asks, at each position:
  - within-problem AUROC (difficulty-controlled: solution-specific failure)
  - cross-problem AUROC (difficulty leaks in)
The GAP at a position = how much that position's signal is problem-DIFFICULTY
(the model "knows the problem is hard") vs solution-specific FAILURE.

Hypothesis (from the 08(f) prefix result, cross-problem 0.8 but within-problem
~0.55): EARLY steps mostly encode difficulty / self-assessed competence (big
cross-vs-within gap, vanishes within-problem), while solution-specific failure
appears later. This script tests that, and compares early/mid/late/all-mean
window features to show all-mean is suboptimal.

Labels are SOLUTION-level final-answer correctness (the multisample has no step
labels); the per-step structure is in the stored vectors.
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


def probe_oof(X, y, groups, k, seed, kind):
    """OOF probe scores. kind='group' folds by problem; 'random' folds by sample."""
    N = X.shape[0]
    rng = np.random.default_rng(seed)
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
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--max_pos", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/temporal_probe.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    VEC = data[f"sv_vec_{args.mode}"]
    problem_ids = data["problem_ids"].astype(int)
    y = (data["is_correct"].astype(int) == 0).astype(int)
    N = len(VEC)
    L_sub = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    cols = band_indices(L_sub, args.layer_band)

    # per-step band-mean vector for each chain (list of (T_i, d))
    perstep = []
    for i in range(N):
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
        with np.errstate(invalid="ignore"):
            perstep.append(np.nanmean(V, axis=1))            # (T_i, d)

    prob_to_idx = {}
    for i, p in enumerate(problem_ids):
        prob_to_idx.setdefault(int(p), []).append(i)

    def eval_feature(X, mask):
        """within & cross AUROC on rows in mask (finite), averaged over seeds."""
        idxs = np.where(mask)[0]
        if idxs.size < 30:
            return np.nan, np.nan, idxs.size
        Xm, ym, gm = X[idxs], y[idxs], problem_ids[idxs]
        ig = [np.array([j for j, ii in enumerate(idxs) if ii in set(v)])
              for v in prob_to_idx.values()]
        ig = [g for g in ig if g.size and any(ym[g] == 1) and any(ym[g] == 0)]
        win, cro = [], []
        for s in range(args.n_seeds):
            og = probe_oof(Xm, ym, gm, args.kfold, args.seed + s, "group")
            orr = probe_oof(Xm, ym, gm, args.kfold, args.seed + s, "random")
            win.append(within_pair_auroc(ig, og, ym)[0])
            cro.append(roc_auc_score(ym, orr))
        def bd(a):
            return max(a, 1 - a)
        return bd(np.nanmean(win)), np.nanmean(cro), idxs.size

    print(f"Loaded {N} chains, d={d}, band={args.layer_band}, mode={args.mode}; "
          f"{int(y.sum())} incorrect / {int((1-y).sum())} correct")

    # ---- per-position probe ----
    print(f"\n=== Per-step-position probe: within (difficulty-controlled) vs cross "
          f"(difficulty leaks) ===")
    print(f"{'pos':>4s}  {'within':>7s}  {'cross':>7s}  {'gap=diff':>8s}  {'n':>5s}")
    curve = []
    for t in range(args.max_pos):
        Xt = np.full((N, d), np.nan)
        for i in range(N):
            if perstep[i].shape[0] > t:
                Xt[i] = perstep[i][t]
        w, c, n = eval_feature(Xt, np.isfinite(Xt).all(axis=1))
        curve.append((w, c))
        gap = (c - w) if (np.isfinite(c) and np.isfinite(w)) else np.nan
        print(f"{t:>4d}  {w:7.4f}  {c:7.4f}  {gap:+8.4f}  {n:5d}")

    # ---- window features (early / mid / late / all-mean) ----
    def window_feat(lo, hi):
        X = np.full((N, d), np.nan)
        for i in range(N):
            T = perstep[i].shape[0]
            a, b = lo, (T if hi is None else min(hi, T))
            if b > a:
                with np.errstate(invalid="ignore"):
                    X[i] = np.nanmean(perstep[i][a:b], axis=0)
        return X
    print(f"\n=== Window features (within-problem AUROC; shows all-mean is suboptimal) ===")
    print(f"{'window':>12s}  {'within':>7s}  {'cross':>7s}  {'n':>5s}")
    wins = {"early[0:3]": window_feat(0, 3), "mid[2:6]": window_feat(2, 6),
            "late[-3:]": None, "all-mean": window_feat(0, None)}
    # late: last 3 steps per chain
    Xl = np.full((N, d), np.nan)
    for i in range(N):
        T = perstep[i].shape[0]
        if T >= 1:
            with np.errstate(invalid="ignore"):
                Xl[i] = np.nanmean(perstep[i][max(0, T - 3):], axis=0)
    wins["late[-3:]"] = Xl
    wres = {}
    for name, X in wins.items():
        w, c, n = eval_feature(X, np.isfinite(X).all(axis=1))
        wres[name] = (w, c)
        print(f"{name:>12s}  {w:7.4f}  {c:7.4f}  {n:5d}")

    np.savez(args.output, position_curve=np.array(curve, dtype=object),
             window_results=np.array(wres, dtype=object), band=np.array(args.layer_band))
    print(f"\nSaved -> {args.output}")
    print("\nRead: large within-cross GAP at EARLY positions => that signal is "
          "problem-DIFFICULTY (model senses hard problem), not solution failure. "
          "Where WITHIN peaks => where solution-specific failure shows up.")


if __name__ == "__main__":
    main()
