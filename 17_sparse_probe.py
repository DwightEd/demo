"""Step 17: is reasoning failure SPARSE (a few neurons, like H-Neurons) or a
DISTRIBUTED low-rank direction? (the H-Neurons bridge; mechanism, not just detect.)

H-Neurons (2512.01797) claims hallucination lives in <0.1% of FFN neurons (axis-
aligned sparse). CAA/RepE say it's a dense residual-stream direction. Our PR
geometry says low effective dimensionality. These are different claims. We test
on OUR difficulty-controlled, within-problem failure probe (late window):

  - L1 path: L1-logistic at increasing strength -> within-problem AUROC vs #nonzero
    NEURONS. If a few tens of neurons recover ~0.71 -> axis-aligned sparse (supports
    H-Neurons). If it needs many -> not neuron-sparse.
  - PCA-rank path: probe on top-k principal components -> AUROC vs k (rank). If a few
    PCs recover ~0.71 -> low-RANK (could be dense across neurons -> distinct from
    H-Neurons' axis-aligned claim).

The contrast {few neurons?} vs {few PCs?} adjudicates sparse-vs-distributed.
All AUROCs are within-problem (GroupKFold, held-out, difficulty-controlled).
"""

from __future__ import annotations

import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


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


def oof_auroc(make_clf, X, y, groups, idx_groups, k, n_seeds, seed):
    vals = []; nnz = []
    for s in range(n_seeds):
        oof = np.full(len(y), np.nan); nz_fold = []
        for tr, te in group_folds(groups, k, seed + s):
            if len(np.unique(y[tr])) < 2:
                continue
            clf = make_clf()
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
            if hasattr(clf[-1] if hasattr(clf, "__getitem__") else clf, "coef_"):
                co = (clf[-1] if hasattr(clf, "__getitem__") else clf).coef_.ravel()
                nz_fold.append(int(np.sum(np.abs(co) > 1e-8)))
        a = within_pair_auroc(idx_groups, oof, y)[0]
        vals.append(max(a, 1 - a))
        if nz_fold:
            nnz.append(np.mean(nz_fold))
    return float(np.mean(vals)), (float(np.mean(nnz)) if nnz else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--layer_band", default="all")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/sparse_probe.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    VEC = data[f"sv_vec_{args.mode}"]
    problem_ids = data["problem_ids"].astype(int)
    y = (data["is_correct"].astype(int) == 0).astype(int)
    N = len(VEC)
    L_sub = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    cols = band_indices(L_sub, args.layer_band)

    X = np.full((N, d), np.nan)
    for i in range(N):
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]; T = V.shape[0]
        with np.errstate(invalid="ignore"):
            X[i] = np.nanmean(V[max(0, T - args.window):].reshape(-1, d), axis=0)
    ok = np.isfinite(X).all(axis=1)
    X, y, problem_ids = X[ok], y[ok], problem_ids[ok]
    N = X.shape[0]
    prob = {}
    for i, p in enumerate(problem_ids):
        prob.setdefault(int(p), []).append(i)
    idx_groups = [np.array(v) for v in prob.values() if any(y[v] == 1) and any(y[v] == 0)]

    Xs = StandardScaler().fit_transform(X)
    print(f"Loaded {N} chains, {len(idx_groups)} contrastive, d={d}, "
          f"band={args.layer_band}, window=last{args.window}")

    # ---- L1 path: AUROC vs #nonzero neurons ----
    print(f"\n=== L1 (sparse NEURONS): within-AUROC vs # nonzero dims ===")
    print(f"{'C':>8s}  {'wAUROC':>7s}  {'#nonzero':>9s}")
    l1 = {}
    for C in [0.0005, 0.001, 0.003, 0.01, 0.03, 0.1]:
        def mk(C=C):
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(C=C, penalty="l1", solver="liblinear",
                                                    max_iter=3000, class_weight="balanced"))
        a, nz = oof_auroc(mk, X, y, problem_ids, idx_groups, args.kfold, args.n_seeds, args.seed)
        l1[C] = (a, nz)
        print(f"{C:8.4f}  {a:7.4f}  {nz:9.1f}")

    # ---- PCA-rank path: AUROC vs # principal components ----
    print(f"\n=== PCA (low RANK): within-AUROC vs # top principal components ===")
    print(f"{'k':>6s}  {'wAUROC':>7s}")
    U, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    pca = {}
    for k in [2, 5, 10, 25, 50, 100, 300]:
        if k > Vt.shape[0]:
            continue
        Xk = (Xs - Xs.mean(0)) @ Vt[:k].T
        def mkp():
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
        a, _ = oof_auroc(mkp, Xk, y, problem_ids, idx_groups, args.kfold, args.n_seeds, args.seed)
        pca[k] = a
        print(f"{k:6d}  {a:7.4f}")

    np.savez(args.output, l1=np.array(l1, dtype=object), pca=np.array(pca, dtype=object),
             band=np.array(args.layer_band))
    print(f"\nSaved -> {args.output}")
    print("\nRead: if a few NEURONS (L1, ~tens) recover ~0.71 -> axis-aligned SPARSE "
          "(supports H-Neurons). If only a few PCs (low RANK) recover it but L1 needs "
          "many -> DISTRIBUTED low-rank direction (challenges neuron-sparsity).")


if __name__ == "__main__":
    main()
