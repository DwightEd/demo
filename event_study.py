"""Event study: is the fused geom+uncertainty error-score a PRECURSOR of the
first error, or only synchronous with it?

We have a within-chain first-error localizer (fuse_within: geom+unc). Align each
error chain at t=0 = first-error step and plot the mean error-score at offsets
Delta = j - k around it, AFTER residualizing the score on length+position
(cross-fit on correct chains) so a trivial 'later steps score higher' trend
cannot masquerade as a precursor.

Three shapes -> three claims:
  rises BEFORE 0 (Delta=-2,-1 already elevated)  -> PRECURSOR / early warning (high value)
  flat then jump AT 0                            -> synchronous detection
  elevated only AFTER 0                          -> lagged ripple (downstream of error)

Correct-chain steps give the residual baseline (~0). Reports mean +/- SE per offset
and tests whether Delta in {-1,-2} is already above the Delta<=-4 baseline.

Needs _coh.npz with resultant/coherence/norm/cloud_D + U_D/U_C.
"""

from __future__ import annotations

import argparse
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
except ImportError:
    raise SystemExit("needs scikit-learn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lo", type=int, default=-6)
    ap.add_argument("--hi", type=int, default=4)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    gnames = [str(x) for x in z["geom_feature_names"]]
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    SG, SC = z["stepgeom"], z["stepcloud"]; SR = z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None

    # features: geom+unc (the fused localizer) + length/pos for residualization
    F, DET, G, J, K, T_, NT, POS, CORR = [], [], [], [], [], [], [], [], []
    for i in range(len(SG)):
        sg = np.asarray(SG[i], float); sc = np.asarray(SC[i], float)
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0)
        a0 = int(rng[0, 0]); T = rng.shape[0]
        ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        for j in range(T):
            def cf(nm): return sc[j, li, cnames.index(nm)] if nm in cnames else np.nan
            lo = max(0, int(rng[j, 0]) - a0); hi = min((len(ud) if ud is not None else 0), int(rng[j, 1]) - a0 + 1)
            F.append([cf("resultant"), cf("coherence"), sg[j, li, gnames.index("norm")], cf("cloud_D"),
                      np.nanmean(ud[lo:hi]) if (ud is not None and hi > lo) else np.nan,
                      np.nanmean(uc[lo:hi]) if (uc is not None and hi > lo) else np.nan])
            ntok = int(rng[j, 1] - rng[j, 0] + 1)
            DET.append(1 if (not correct and j == k) else (0 if (correct or j < k) else np.nan))
            G.append(i); J.append(j); K.append(k); T_.append(T)
            NT.append(np.log(max(ntok, 1))); POS.append(j / max(1, T - 1)); CORR.append(correct)
    F = np.asarray(F, float); DET = np.asarray(DET, float); G = np.asarray(G, int)
    J = np.asarray(J, int); K = np.asarray(K, int)
    NT = np.asarray(NT, float); POS = np.asarray(POS, float); CORR = np.asarray(CORR, bool)
    for c in range(F.shape[1]):
        col = F[:, c]; col[~np.isfinite(col)] = np.nanmean(col[np.isfinite(col)])

    gkf = GroupKFold(args.folds)
    # cross-fit fused error-score (geom+unc), predict ALL steps
    score = np.full(len(F), np.nan)
    for tr, te in gkf.split(F, np.nan_to_num(DET), G):
        lab = np.isfinite(DET[tr])
        if len(np.unique(DET[tr][lab])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(F[tr][lab], DET[tr][lab].astype(int))
        score[te] = clf.predict_proba(F[te])[:, 1]

    # residualize score on length+position, cross-fit on CORRECT-chain steps
    resid = np.full(len(F), np.nan)
    X = np.c_[NT, POS]
    for tr, te in gkf.split(F, np.nan_to_num(DET), G):
        ctr = tr[CORR[tr]]
        if len(ctr) < 50:
            continue
        reg = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0)
        reg.fit(X[ctr], score[ctr]); resid[te] = score[te] - reg.predict(X[te])

    # baseline: correct-chain residual (should be ~0)
    base = np.nanmean(resid[CORR])
    print(f"file: {args.npz} | layer {args.layer} | error-chains {len(np.unique(G[~CORR]))}")
    print(f"correct-chain residual baseline: {base:+.4f}\n")
    print(f"{'Δ=j-k':>6s} {'n':>6s} {'mean_resid':>11s} {'SE':>7s}")
    err = ~CORR
    by = {}
    for d in range(args.lo, args.hi + 1):
        m = err & (J - K == d)
        v = resid[m]; v = v[np.isfinite(v)]
        if len(v) >= 5:
            by[d] = (len(v), v.mean(), v.std() / np.sqrt(len(v)))
            star = " <-- error step" if d == 0 else ""
            print(f"{d:>6d} {len(v):>6d} {v.mean():>+11.4f} {v.std()/np.sqrt(len(v)):>7.4f}{star}")

    # precursor test: Delta in {-1,-2} vs Delta <= -4 (pre baseline within error chains)
    pre = resid[err & (J - K <= -4)]; pre = pre[np.isfinite(pre)]
    near = resid[err & ((J - K == -1) | (J - K == -2))]; near = near[np.isfinite(near)]
    at0 = resid[err & (J - K == 0)]; at0 = at0[np.isfinite(at0)]
    def cmp(a, b, na, nb):
        d = a - b; se = np.sqrt(a.std()**2/len(a) + b.std()**2/len(b)) if (len(a) and len(b)) else np.nan
        return d, se
    if len(pre) and len(near):
        d, se = cmp(near.mean(), pre.mean(), 0, 0); d = near.mean() - pre.mean()
        se = np.sqrt(near.std()**2/len(near) + pre.std()**2/len(pre))
        print(f"\nprecursor: mean_resid(Δ∈-1,-2) − mean_resid(Δ≤-4) = {d:+.4f} "
              f"[{d-2*se:+.4f},{d+2*se:+.4f}] {'RISES before error' if d-2*se>0 else 'ns'}")
    if len(at0) and len(near):
        d = at0.mean() - near.mean(); se = np.sqrt(at0.std()**2/len(at0) + near.std()**2/len(near))
        print(f"jump:      mean_resid(Δ=0) − mean_resid(Δ∈-1,-2)   = {d:+.4f} "
              f"[{d-2*se:+.4f},{d+2*se:+.4f}]")
    print("\nread: rises before 0 (precursor>0) -> EARLY WARNING; jump at 0 with flat before "
          "-> synchronous detection; elevated only after 0 -> lagged ripple.")


if __name__ == "__main__":
    main()
