"""Step 24 (revised v2): geometric decoupling of difficulty vs failure (well-posed).

CRITICAL fix over v1: in the full d~4096 space with only ~200 problems, the difficulty
direction (Ridge over per-problem means) is UNDER-DETERMINED -> w_diff is essentially
regularization noise, and any cosine with it is ~0 trivially (an artifact, not orthogonality).
We therefore reduce to the top-k PCs (k << #problems, fit on train) FIRST, so both
directions are well-determined, then measure decoupling.

Standardization: mu/sigma from {train problems' CORRECT solutions}, applied to all.
PCA: top-k components fit on train solutions; all activations -> k-dim Z.

w_diff (problem-level): Ridge( per-problem CORRECT-mean(Z) -> diff_p=fail-rate ).
w_fail (sample-level) : one diff per problem d_p = mean_inc(Z) - mean_corr(Z);
                        logistic on {d_p->+1, -d_p->-1}, no intercept.

Decoupling (three together, all held-out, GroupKFold over problems):
  (1) cos(w_diff, w_fail)  (random baseline ~ 1/sqrt(k))
  (2) within-problem FAILURE paired AUROC: score with w_fail vs with w_fail RESIDUALIZED
      against w_diff (= activations with the w_diff axis removed). Stays ~baseline => the
      failure signal is orthogonal to difficulty (not reducible to it).  [KEY]
  (3) DIFFICULTY held-out corr^2: w_diff vs w_diff residualized against w_fail.  [reverse]
Sensitivity: if w_fail == w_diff (failure IS difficulty), (2) collapses to 0.5.
"""

from __future__ import annotations

import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA


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


def perp(w, axis):
    u = axis / (np.linalg.norm(axis) + 1e-12)
    return w - (w @ u) * u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--layer_band", default="mid")
    ap.add_argument("--late_lo", type=float, default=0.6)
    ap.add_argument("--pca_k", type=int, default=50, help="PCs (<< #problems) so directions are well-posed")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/decouple.npz")
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
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
        with np.errstate(invalid="ignore"):
            P = np.nanmean(V, axis=1)
        T = P.shape[0]
        fr = (np.arange(T) / (T - 1)) if T > 1 else np.array([0.0])
        m = fr >= args.late_lo
        if not m.any(): m = fr >= fr.max()
        with np.errstate(invalid="ignore"):
            X[i] = np.nanmean(P[m], axis=0)
    okm = np.isfinite(X).all(1)
    X, y, problem_ids = X[okm], y[okm], problem_ids[okm]
    N = len(y)

    prob = {}
    for i, p in enumerate(problem_ids):
        prob.setdefault(int(p), []).append(i)
    diffmap = {p: float(y[np.array(v)].mean()) for p, v in prob.items()}
    idx_groups = [np.array(v) for v in prob.values() if any(y[v] == 1) and any(y[v] == 0)]
    k = min(args.pca_k, d)
    print(f"Loaded {N} solutions over {len(prob)} problems ({len(idx_groups)} contrastive); "
          f"d={d} -> PCA k={k}; random cos ~ {1/np.sqrt(k):.3f}")

    def wdiff_of(Z, yy, gg, plist):
        Xb, yb = [], []
        for p in plist:
            ip = [i for i in range(len(yy)) if gg[i] == p and yy[i] == 0]
            if ip: Xb.append(Z[ip].mean(0)); yb.append(diffmap[p])
        if len(Xb) < 5: return None
        return Ridge(alpha=1.0).fit(np.asarray(Xb), np.asarray(yb)).coef_.ravel()

    def wfail_of(Z, yy, gg, plist):
        D = []
        for p in plist:
            ip = [i for i in range(len(yy)) if gg[i] == p]
            inc = [i for i in ip if yy[i] == 1]; cor = [i for i in ip if yy[i] == 0]
            if inc and cor: D.append(Z[inc].mean(0) - Z[cor].mean(0))
        if len(D) < 5: return None
        D = np.asarray(D); Xtr = np.vstack([D, -D]); yt = np.concatenate([np.ones(len(D)), np.zeros(len(D))])
        return LogisticRegression(C=0.5, max_iter=4000, fit_intercept=False).fit(Xtr, yt).coef_.ravel()

    cos_l = []
    fb, fr_, r2b, r2r = [], [], [], []
    for s in range(args.n_seeds):
        sfb = np.full(N, np.nan); sfr = np.full(N, np.nan)
        dt_t, dt_b, dt_r = [], [], []
        for tr, te in group_folds(problem_ids, args.kfold, args.seed + s):
            cm = (y[tr] == 0)
            if cm.sum() < 5: continue
            mu = X[tr][cm].mean(0); sd = X[tr][cm].std(0) + 1e-6
            Xs = (X - mu) / sd
            pca = PCA(n_components=k, random_state=0).fit(Xs[tr])
            Z = pca.transform(Xs)                                  # (N, k)
            g = problem_ids; tr_p = np.unique(g[tr]); te_p = np.unique(g[te])
            wd = wdiff_of(Z[tr], y[tr], g[tr], tr_p)
            wf = wfail_of(Z[tr], y[tr], g[tr], tr_p)
            if wd is None or wf is None: continue
            cos_l.append(float(wd @ wf / (np.linalg.norm(wd) * np.linalg.norm(wf) + 1e-12)))

            wf_perp = perp(wf, wd)                                  # failure with difficulty axis removed
            for i in te:
                sfb[i] = Z[i] @ wf; sfr[i] = Z[i] @ wf_perp
            wd_perp = perp(wd, wf)                                  # difficulty with failure axis removed
            for p in te_p:
                ipc = [i for i in prob[p] if y[i] == 0]
                if not ipc: continue
                m = Z[ipc].mean(0)
                dt_t.append(diffmap[p]); dt_b.append(float(m @ wd)); dt_r.append(float(m @ wd_perp))

        fb.append(bd(within_pair_auroc(idx_groups, sfb, y)[0]))
        fr_.append(bd(within_pair_auroc(idx_groups, sfr, y)[0]))

        def cr2(t, p):
            t, p = np.array(t), np.array(p)
            if len(t) < 5 or np.std(p) < 1e-9 or np.std(t) < 1e-9: return np.nan
            return float(np.corrcoef(t, p)[0, 1] ** 2)
        r2b.append(cr2(dt_t, dt_b)); r2r.append(cr2(dt_t, dt_r))

    cos = float(np.mean(cos_l)); cos_s = float(np.std(cos_l))
    FB, FR = float(np.mean(fb)), float(np.mean(fr_))
    RB, RR = float(np.nanmean(r2b)), float(np.nanmean(r2r))

    print(f"\n=== (1) cos(w_diff, w_fail) = {cos:+.3f} +/- {cos_s:.3f}  (random ~ {1/np.sqrt(k):.3f}) ===")
    print(f"\n=== (2) within-problem FAILURE paired AUROC ===")
    print(f"  with w_fail (baseline)            = {FB:.3f}")
    print(f"  w_fail with w_diff axis removed   = {FR:.3f}   (delta {FR-FB:+.3f})")
    print(f"\n=== (3) DIFFICULTY held-out corr^2 ===")
    print(f"  with w_diff (baseline)            = {RB:.3f}")
    print(f"  w_diff with w_fail axis removed   = {RR:.3f}   (delta {RR-RB:+.3f})")
    if FR > FB - 0.03 and RR > RB - 0.05:
        print("\n  => DECOUPLED: failure survives removing the difficulty axis AND difficulty "
              "survives removing the failure axis -> functionally orthogonal directions.")
    else:
        print("\n  => coupling: removing one axis hurts the other (report honestly).")

    np.savez(args.output, cos=np.array(cos), cos_std=np.array(cos_s), rand_cos=np.array(1/np.sqrt(k)),
             fail_base=np.array(FB), fail_resid=np.array(FR),
             diff_r2_base=np.array(RB), diff_r2_resid=np.array(RR),
             pca_k=np.array(k), band=np.array(args.layer_band))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
