"""Orthogonality health-check: how much of resultant's step-level discrimination
is INDEPENDENT of norm and length? Run this BEFORE any 'improve it' work.

Step-level, ProcessBench labels (pos=gold first-error; neg=correct+pre-error).
Reports:
  (1) Spearman correlation matrix among resultant/coherence/norm/cloud_D/U_D/U_C/log n_tok
  (2) resultant AUROC: raw | residualized on norm | on norm+log(n_tok)
      (GBM cross-fit on correct steps, like confound_diag)
  (3) increment resultant OVER norm, and norm OVER resultant (GroupKFold logistic,
      chain-paired bootstrap) -- are they independent axes or one shadow?

Verdict: if resultant collapses after removing norm/length -> it is their shadow,
drop it and polish norm. If it survives with low corr -> independent axis.

(massive-removed resultant needs the resultant_bulk feature -> re-extract; this
script covers the norm/length half on existing _coh.npz.)
"""

from __future__ import annotations

import argparse
import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
except ImportError:
    raise SystemExit("needs scikit-learn")


def auroc(score, y):
    m = np.isfinite(score); s, yy = score[m], y[m]
    npos, nneg = int((yy == 1).sum()), int((yy == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sr = s[o]; i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def bdir(a):
    return max(a, 1 - a) if np.isfinite(a) else a


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    gnames = [str(x) for x in z["geom_feature_names"]]
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    SG, SC = z["stepgeom"], z["stepcloud"]; SR = z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None

    def gi(n):
        return gnames.index(n)
    def ci(n):
        return cnames.index(n)

    cols = {k: [] for k in ["resultant", "coherence", "norm", "cloud_D",
                            "U_D", "U_C", "logn"]}
    Y, G = [], []
    for i in range(len(SG)):
        sg = np.asarray(SG[i], float); sc = np.asarray(SC[i], float)
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0)
        a0 = int(rng[0, 0]); T = rng.shape[0]
        ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        for j in range(T):
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                continue
            cols["resultant"].append(sc[j, li, ci("resultant")] if "resultant" in cnames else np.nan)
            cols["coherence"].append(sc[j, li, ci("coherence")] if "coherence" in cnames else np.nan)
            cols["cloud_D"].append(sc[j, li, ci("cloud_D")] if "cloud_D" in cnames else np.nan)
            cols["norm"].append(sg[j, li, gi("norm")])
            ntok = int(rng[j, 1] - rng[j, 0] + 1)
            cols["logn"].append(np.log(max(ntok, 1)))
            lo = max(0, int(rng[j, 0]) - a0); hi = min((len(ud) if ud is not None else 0),
                                                       int(rng[j, 1]) - a0 + 1)
            cols["U_D"].append(np.nanmean(ud[lo:hi]) if (ud is not None and hi > lo) else np.nan)
            cols["U_C"].append(np.nanmean(uc[lo:hi]) if (uc is not None and hi > lo) else np.nan)
            Y.append(y); G.append(i)
    for k in cols:
        cols[k] = np.asarray(cols[k], float)
    Y = np.asarray(Y, int); G = np.asarray(G, int)

    keys = ["resultant", "coherence", "norm", "cloud_D", "U_D", "U_C", "logn"]
    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"\n=== (1) Spearman correlation matrix ===")
    print("            " + "".join(f"{k[:8]:>9s}" for k in keys))
    for a in keys:
        row = "".join(f"{spearman(cols[a], cols[b]):>9.2f}" for b in keys)
        print(f"{a:11s} {row}")
    print(f"\n  each metric's standalone AUROC (best dir):")
    for k in keys:
        print(f"    {k:11s} {bdir(auroc(cols[k], Y)):.3f}")

    # (2) residualize resultant on norm / norm+logn (GBM cross-fit on correct steps)
    R = cols["resultant"]
    gkf = GroupKFold(args.folds)
    def resid_on(X):
        out = np.full(len(R), np.nan)
        for tr, te in gkf.split(R, Y, G):
            mtr = tr[Y[tr] == 0]
            if len(mtr) < 30:
                continue
            reg = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0)
            reg.fit(X[mtr], R[mtr]); out[te] = R[te] - reg.predict(X[te])
        return out
    a_raw = bdir(auroc(R, Y))
    a_rn = bdir(auroc(resid_on(cols["norm"][:, None]), Y))
    a_rnl = bdir(auroc(resid_on(np.c_[cols["norm"], cols["logn"]]), Y))
    print(f"\n=== (2) resultant AUROC, removing norm / norm+length ===")
    print(f"  raw                         {a_raw:.3f}")
    print(f"  residualized on norm        {a_rn:.3f}  (drop {a_raw - a_rn:+.3f})")
    print(f"  residualized on norm+logn   {a_rnl:.3f}  (drop {a_raw - a_rnl:+.3f})")

    # (3) increments: resultant over norm, norm over resultant (logistic OOF + bootstrap)
    def oof(X):
        s = np.full(len(Y), np.nan)
        for tr, te in gkf.split(X, Y, G):
            if len(np.unique(Y[tr])) < 2:
                continue
            p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            p.fit(X[tr], Y[tr]); s[te] = p.predict_proba(X[te])[:, 1]
        return s
    sn = oof(cols["norm"][:, None]); sr = oof(R[:, None])
    snr = oof(np.c_[cols["norm"], R])
    a_n, a_r, a_nr = auroc(sn, Y), auroc(sr, Y), auroc(snr, Y)
    rng = np.random.default_rng(0); chains = np.unique(G)
    d_rn, d_nr = [], []
    for _ in range(500):
        cb = rng.choice(chains, size=len(chains), replace=True)
        m = np.concatenate([np.where(G == c)[0] for c in cb])
        d_rn.append(auroc(snr[m], Y[m]) - auroc(sn[m], Y[m]))   # resultant over norm
        d_nr.append(auroc(snr[m], Y[m]) - auroc(sr[m], Y[m]))   # norm over resultant
    def ci(d):
        d = np.asarray(d); lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return f"+{np.nanmean(d):.3f} [{lo:+.3f},{hi:+.3f}] {'SIG' if lo > 0 else 'ns'}"
    print(f"\n=== (3) are resultant and norm independent axes? ===")
    print(f"  norm alone {a_n:.3f} | resultant alone {a_r:.3f} | both {a_nr:.3f}")
    print(f"  resultant OVER norm:  {ci(d_rn)}")
    print(f"  norm OVER resultant:  {ci(d_nr)}")
    print("\nverdict: low corr + resultant survives removing norm/length + significant "
          "increment over norm => independent axis worth keeping/combining. Otherwise "
          "it is a norm/length shadow -> drop it, polish norm.")


if __name__ == "__main__":
    main()
