"""Step 20: STRENGTHEN by ensembling the (de)correlated within-problem signals.

The phenomenology is settled and every single signal caps within-problem around
0.68-0.73. Earlier experiments showed several of these signals are roughly ORTHOGONAL
(participation perpendicular output-entropy; PR perpendicular effective-rank rho~-0.07;
failure-direction perpendicular difficulty cos~0.24; SPE is difficulty-INVARIANT unlike
the supervised probe). Combining decorrelated weak detectors reduces variance and can
beat the best single one -- but ONLY if they are genuinely decorrelated, which we
measure here rather than assume.

Under ONE GroupKFold-over-problems protocol we compute held-out (OOF) scores for each
component, then:
  1. each component's within-problem PAIRED AUROC (sanity / per-signal strength),
  2. the Pearson correlation MATRIX of the component scores (does ensembling have room?),
  3. ensemble A: sign-aligned z-score MEAN (unsupervised, no meta-training -> no leakage),
  4. ensemble B: meta-logistic stacked on the component scores (GroupKFold over problems),
  5. delta over the best single component -- reported honestly (a +0.01 is a +0.01).

Components (late-window band-mean vector unless noted):
  probe    L2-logistic on the full d-dim vector            (supervised, ~0.71)
  pca25    L2-logistic on top-25 PCs                        (supervised low-rank, ~0.73)
  spe      manifold-residual fraction, healthy subspace k   (unsupervised, difficulty-inv)
  delta    L2-logistic on (late - early) emergence vector   (trajectory)
  mahal    healthy diagonal Mahalanobis distance            (unsupervised)
  length   n_steps                                          (the confound to beat)
"""

from __future__ import annotations

import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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


def group_folds(groups, k, seed):
    uniq = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(uniq)
    fo = {int(g): i % k for i, g in enumerate(uniq)}
    f = np.array([fo[int(g)] for g in groups])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def healthy_subspace(Xtr_correct, kmax):
    """Top-kmax principal directions of the standardized healthy (correct) vectors,
    via the d x d covariance (n>>d). Returns (mu, sd, cm, B[kmax,d])."""
    H = Xtr_correct.astype(np.float32)
    mu = H.mean(0); sd = H.std(0) + 1e-6
    Hc = (H - mu) / sd
    cm = Hc.mean(0); Hc -= cm
    C = (Hc.T @ Hc) / max(1, Hc.shape[0] - 1)
    _, evecs = np.linalg.eigh(C)
    B = np.ascontiguousarray(evecs[:, ::-1][:, :kmax].T)
    return mu, sd, cm, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--layer_band", default="all")
    ap.add_argument("--pca_k", type=int, default=25)
    ap.add_argument("--spe_k", type=int, default=25)
    ap.add_argument("--late_lo", type=float, default=0.6)
    ap.add_argument("--early_hi", type=float, default=0.4)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/ensemble.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    VEC = data[f"sv_vec_{args.mode}"]
    problem_ids = data["problem_ids"].astype(int)
    y = (data["is_correct"].astype(int) == 0).astype(int)          # 1 = incorrect
    N = len(VEC)
    L_sub = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    cols = band_indices(L_sub, args.layer_band)
    n_steps = data["n_steps"].astype(float) if "n_steps" in data else np.full(N, np.nan)

    # per-solution windowed vectors + per-step cloud (for SPE)
    late = np.full((N, d), np.nan); early = np.full((N, d), np.nan)
    ps = [None] * N; latemask = [None] * N
    for i in range(N):
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
        with np.errstate(invalid="ignore"):
            P = np.nanmean(V, axis=1)                               # (T, d)
        ps[i] = P
        T = P.shape[0]
        fr = (np.arange(T) / (T - 1)) if T > 1 else np.array([0.0])
        lo = fr >= args.late_lo;  eo = fr < args.early_hi
        if not lo.any(): lo = fr >= fr.max()
        if not eo.any(): eo = fr <= fr.min()
        latemask[i] = lo
        with np.errstate(invalid="ignore"):
            late[i] = np.nanmean(P[lo], axis=0)
            early[i] = np.nanmean(P[eo], axis=0)
    delta = late - early
    valid = np.isfinite(late).all(1) & np.isfinite(delta).all(1)

    prob = {}
    for i, p in enumerate(problem_ids):
        prob.setdefault(int(p), []).append(i)
    idx_groups = [np.array(v) for v in prob.values() if any(y[v] == 1) and any(y[v] == 0)]

    print(f"Loaded {N} chains over {len(prob)} problems ({len(idx_groups)} contrastive); "
          f"d={d}, band={args.layer_band}; {int(y.sum())} incorrect / {int((1-y).sum())} correct")

    comp_names = ["probe", "pca25", "spe", "delta", "mahal", "length"]
    # accumulate OOF scores averaged over seeds
    acc = {c: np.zeros(N) for c in comp_names}
    cnt = {c: np.zeros(N) for c in comp_names}

    def fit_logistic(Xtr, ytr, Xte, pca_k=None):
        steps = [StandardScaler()]
        if pca_k: steps.append(PCA(n_components=pca_k, random_state=0))
        steps.append(LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced"))
        clf = make_pipeline(*steps).fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]

    for s in range(args.n_seeds):
        for tr, te in group_folds(problem_ids, args.kfold, args.seed + s):
            tr = tr[valid[tr]]; te = te[valid[te]]
            if len(np.unique(y[tr])) < 2 or len(te) == 0:
                continue
            # supervised components on the late-window vector
            acc["probe"][te] += fit_logistic(late[tr], y[tr], late[te]); cnt["probe"][te] += 1
            acc["pca25"][te] += fit_logistic(late[tr], y[tr], late[te], args.pca_k); cnt["pca25"][te] += 1
            acc["delta"][te] += fit_logistic(delta[tr], y[tr], delta[te]); cnt["delta"][te] += 1
            # unsupervised components from the healthy (correct-train) population
            heal_idx = tr[y[tr] == 0]
            if len(heal_idx) >= 5:
                Hc_steps = np.vstack([ps[i] for i in heal_idx])
                mu, sd, cm, B = healthy_subspace(Hc_steps, min(args.spe_k, d))
                muL = late[heal_idx].mean(0); vL = late[heal_idx].var(0) + 1e-6
                for i in te:
                    Z = (ps[i] - mu) / sd - cm
                    tot = (Z ** 2).sum(1) + 1e-12
                    ink = (np.cumsum((Z @ B.T) ** 2, axis=1))[:, min(args.spe_k, B.shape[0]) - 1]
                    spe = ((tot - ink) / tot)[latemask[i]].mean()
                    acc["spe"][i] += spe; cnt["spe"][i] += 1
                    acc["mahal"][i] += float(((late[i] - muL) ** 2 / vL).sum()); cnt["mahal"][i] += 1

    comps = {}
    for c in comp_names:
        if c == "length":
            comps[c] = n_steps.copy()
        else:
            v = np.full(N, np.nan); m = cnt[c] > 0
            v[m] = acc[c][m] / cnt[c][m]; comps[c] = v

    # 1) per-component within-problem paired AUROC (direction-free)
    print(f"\n=== component within-problem PAIRED AUROC ===")
    awith = {}
    for c in comp_names:
        a = within_pair_auroc(idx_groups, comps[c], y)[0]
        awith[c] = max(a, 1 - a) if np.isfinite(a) else float("nan")
        print(f"  {c:8s} {awith[c]:.4f}")

    # 2) correlation matrix of the component scores (room for ensembling?)
    keys = [c for c in comp_names if np.isfinite(comps[c]).sum() > 10]
    M = np.vstack([comps[c] for c in keys])
    ok = np.isfinite(M).all(0)
    Mok = M[:, ok]
    Cmat = np.corrcoef(Mok)
    print(f"\n=== component Pearson correlation (|rho| low -> ensembling can help) ===")
    print("          " + "  ".join(f"{k[:6]:>6s}" for k in keys))
    for i, k in enumerate(keys):
        print(f"  {k:7s} " + "  ".join(f"{Cmat[i, j]:+.2f}" for j in range(len(keys))))

    # sign-align each component so higher => more likely incorrect
    aligned = {}
    for c in keys:
        a = within_pair_auroc(idx_groups, comps[c], y)[0]
        sgn = 1.0 if (np.isfinite(a) and a >= 0.5) else -1.0
        z = comps[c].copy()
        mu_, sd_ = np.nanmean(z), np.nanstd(z) + 1e-9
        aligned[c] = sgn * (z - mu_) / sd_

    # 3) ensemble A: unsupervised sign-aligned z-score mean (no meta-training)
    A = np.nanmean(np.vstack([aligned[c] for c in keys]), axis=0)
    ensA = within_pair_auroc(idx_groups, A, y)[0]; ensA = max(ensA, 1 - ensA)

    # 4) ensemble B: meta-logistic stacked on component scores, GroupKFold over problems
    F = np.vstack([comps[c] for c in keys]).T                       # (N, n_comp)
    okF = np.isfinite(F).all(1)
    metaoof = np.full(N, np.nan)
    for s in range(args.n_seeds):
        for tr, te in group_folds(problem_ids, args.kfold, 100 + args.seed + s):
            tr = tr[okF[tr]]; te = te[okF[te]]
            if len(np.unique(y[tr])) < 2 or len(te) == 0:
                continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
            clf.fit(F[tr], y[tr])
            p = clf.predict_proba(F[te])[:, 1]
            metaoof[te] = p if np.all(np.isnan(metaoof[te])) else np.nanmean([metaoof[te], p], 0)
    ensB = within_pair_auroc(idx_groups, metaoof, y)[0]; ensB = max(ensB, 1 - ensB)

    best_c = max(keys, key=lambda c: awith[c])
    print(f"\n=== ENSEMBLE vs best single ===")
    print(f"  best single        : {best_c} = {awith[best_c]:.4f}")
    print(f"  ensemble A (z-mean): {ensA:.4f}   (delta {ensA - awith[best_c]:+.4f})")
    print(f"  ensemble B (meta)  : {ensB:.4f}   (delta {ensB - awith[best_c]:+.4f})")
    if max(ensA, ensB) > awith[best_c] + 0.01:
        print("  -> ensembling HELPS: the signals carry complementary failure information.")
    else:
        print("  -> ensembling does NOT beat the best single -> signals largely redundant; "
              "the within-problem ceiling is real (~best single).")

    np.savez(args.output,
             comp_names=np.array(keys, dtype=object),
             comp_within=np.array([awith[c] for c in keys]),
             corr=Cmat, ens_zmean=np.array(ensA), ens_meta=np.array(ensB),
             best_single=np.array(awith[best_c]), band=np.array(args.layer_band))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
