"""Step 19: Step-level Manifold Constraint Diagnostic (SMCD) -- the theory-backed core.

Hypothesis (the project's anchor, as a provable statement): correct reasoning steps
concentrate their activation energy in a LOW-DIMENSIONAL but non-trivial subspace (the
"healthy reasoning manifold"); erroneous steps DIFFUSE energy into the orthogonal
complement. We make this a derived quantity, not an ad-hoc trick:

  Healthy subspace.  Standardize step vectors by the CORRECT-step mean/std (healthy
  coordinate frame), then PCA the correct steps; U_k = top-k principal directions =
  the k-dim "non-trivial low-dim subset".

  Per-step violation (Squared Prediction Error / Q-statistic, Jackson-Mudholkar 1979):
      z = U_k U_k^T z  (in-manifold)  +  (I - U_k U_k^T) z  (out-of-manifold)
      SPE(z) = ||z_perp||^2 / ||z||^2   = fraction of energy OUTSIDE the healthy manifold
  Correct steps lock energy inside U_k (SPE low); erroneous steps leak out (SPE high).
  This is UNSUPERVISED (subspace fit on correct steps only -- no failure labels), so it
  is a genuine diagnostic, not a probe fit to the answer key. It is also natively
  STEP-LEVEL: SPE is computed per step, so argmax_t SPE(z_t) localizes the error step
  (-> ProcessBench gold-step evaluation).

  Distinct from the failed effective-rank M_D: M_D needs an SVD of ONE step's token
  cloud (n<<d, ill-posed). Here the subspace is estimated from THOUSANDS of correct
  steps; the query is a single vector projected onto it. No n<<d problem.

Rigor: strict cross-fit -- U_k and the healthy mean/std are fit ONLY on TRAIN problems'
correct steps; SPE is evaluated on HELD-OUT problems; chain score = late-window mean (or
max) of per-step SPE; reported as within-problem PAIRED AUROC (difficulty-controlled),
swept over k. Cross-problem pooled AUROC shown for reference only.
"""

from __future__ import annotations

import argparse
import numpy as np
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--layer_band", default="all")
    ap.add_argument("--ks", default="1,2,5,10,25,50,100,200",
                    help="healthy-subspace dims to sweep")
    ap.add_argument("--late_lo", type=float, default=0.6, help="late window = frac >= late_lo")
    ap.add_argument("--agg", default="latemean", choices=["latemean", "max", "mean"])
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/manifold_constraint.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    VEC = data[f"sv_vec_{args.mode}"]
    problem_ids = data["problem_ids"].astype(int)
    is_correct = data["is_correct"].astype(int)
    y = (is_correct == 0).astype(int)                      # 1 = incorrect
    N = len(VEC)
    L_sub = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    cols = band_indices(L_sub, args.layer_band)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    # per-solution per-step band-mean vectors + late-window mask
    ps = [None] * N; latemask = [None] * N
    for i in range(N):
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
        with np.errstate(invalid="ignore"):
            P = np.nanmean(V, axis=1)                       # (T, d)
        ps[i] = P
        T = P.shape[0]
        fr = (np.arange(T) / (T - 1)) if T > 1 else np.array([0.0])
        m = fr >= args.late_lo
        if not m.any():
            m = fr >= fr.max()
        latemask[i] = m
    valid = np.array([np.isfinite(ps[i]).all() for i in range(N)])

    prob = {}
    for i, p in enumerate(problem_ids):
        prob.setdefault(int(p), []).append(i)
    idx_groups = [np.array(v) for v in prob.values()
                  if any(y[v] == 1) and any(y[v] == 0)]

    print(f"Loaded {N} chains over {len(prob)} problems ({len(idx_groups)} contrastive); "
          f"d={d}, band={args.layer_band}, agg={args.agg}, late>={args.late_lo}; "
          f"{int(y.sum())} incorrect / {int((1 - y).sum())} correct")

    def chain_score(per_step_spe, mask):
        if args.agg == "max":
            return float(np.max(per_step_spe))
        if args.agg == "mean":
            return float(np.mean(per_step_spe))
        return float(np.mean(per_step_spe[mask]))

    # cross-fit: fit healthy frame + subspace on TRAIN-correct steps, score HELD-OUT
    within = {k: [] for k in ks}; cross = {k: [] for k in ks}
    norm_within = []                                        # ||z|| baseline (massive-act control)
    for s in range(args.n_seeds):
        spe_oof = {k: np.full(N, np.nan) for k in ks}
        norm_oof = np.full(N, np.nan)
        for tr, te in group_folds(problem_ids, args.kfold, args.seed + s):
            heal = [ps[i] for i in tr if valid[i] and y[i] == 0]
            if len(heal) < 5:
                continue
            H = np.vstack(heal).astype(np.float32)
            mu = H.mean(0); sd = H.std(0) + 1e-6
            Hc = (H - mu) / sd
            cm = Hc.mean(0); Hc -= cm
            n_h = Hc.shape[0]
            kmax = min(max(ks), min(Hc.shape))
            # principal directions = top eigenvectors of the d x d covariance. Since
            # n_steps >> d we use the covariance (one GEMM + eigh(d)) instead of a full
            # SVD of H -- the SVD would also compute the n x d left vectors we never use.
            C = (Hc.T @ Hc) / max(1, n_h - 1)
            evals, evecs = np.linalg.eigh(C)               # ascending eigenvalues
            B = np.ascontiguousarray(evecs[:, ::-1][:, :kmax].T)   # (kmax, d), desc
            for i in te:
                if not valid[i]:
                    continue
                Z = (ps[i] - mu) / sd - cm                  # (T, d)
                tot = (Z ** 2).sum(1) + 1e-12
                coords = Z @ B.T                            # (T, kmax)
                cum = np.cumsum(coords ** 2, axis=1)        # in-subspace energy up to each k
                m = latemask[i]
                norm_oof[i] = chain_score(np.sqrt(tot), m)
                for k in ks:
                    kk = min(k, kmax)
                    ink = cum[:, kk - 1]
                    spe = (tot - ink) / tot                 # out-of-manifold fraction
                    spe_oof[k][i] = chain_score(spe, m)
        for k in ks:
            a = within_pair_auroc(idx_groups, spe_oof[k], y)[0]
            within[k].append(max(a, 1 - a))
            m2 = np.isfinite(spe_oof[k])
            if len(np.unique(y[m2])) == 2:
                cr = roc_auc_score(y[m2], spe_oof[k][m2]); cross[k].append(max(cr, 1 - cr))
        a = within_pair_auroc(idx_groups, norm_oof, y)[0]
        norm_within.append(max(a, 1 - a))

    print(f"\n=== Manifold-constraint SPE: within-problem PAIRED AUROC vs healthy-subspace dim k ===")
    print(f"{'k':>6s}  {'within':>14s}   {'cross(ref)':>10s}")
    curve = []
    for k in ks:
        w = float(np.mean(within[k])); ws = float(np.std(within[k]))
        c = float(np.mean(cross[k])) if cross[k] else float("nan")
        curve.append((k, w, c))
        print(f"{k:6d}  {w:.4f} +/- {ws:.4f}   {c:10.4f}")
    nb = float(np.mean(norm_within))
    bk, bw, bc = max(curve, key=lambda t: t[1])
    print(f"\n  ||z|| norm baseline (within)         = {nb:.4f}  <- SPE must beat (not just magnitude)")
    print(f"  best: k={bk}  within={bw:.4f}  (cross-ref {bc:.4f})")
    print("  Read: a clear PEAK at low k = correct reasoning lives in a k-dim non-trivial "
          "subspace; error leaks outside it (SPE). If within >> ||z|| baseline, it is the "
          "GEOMETRY (out-of-manifold diffusion), not activation magnitude.")

    np.savez(args.output,
             ks=np.array([t[0] for t in curve]),
             within=np.array([t[1] for t in curve]),
             cross=np.array([t[2] for t in curve]),
             norm_within=np.array(nb), band=np.array(args.layer_band),
             agg=np.array(args.agg))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
