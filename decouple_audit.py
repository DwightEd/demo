"""(a) Signal self-decoupling: are `norm` and `resultant` ONE signal or TWO?

corr(norm, resultant) ~ 0.94 on full dims. Is that correlation massive-driven?
Compare to the BULK versions (global-fixed massive dims zeroed, from
extract_features --massive_global --cloud_eff_rank):
  - if corr(norm_bulk, resultant_bulk) COLLAPSES and both lose AUROC -> the shared
    signal lived in the massive subspace = ONE signal (massive energy);
  - if both SURVIVE and DECOUPLE (corr drops, each adds over the other) -> energy
    vs direction are TWO orthogonal axes;
  - if both survive but stay correlated -> ONE signal, but NOT massive (bulk).

Reports, at a layer: AUROC of each; Spearman corr matrix (full + bulk); and the
mutual increment (logistic, GroupKFold by chain) resultant_bulk over norm_bulk and
vice versa. Needs an npz with norm_bulk + resultant_bulk in stepcloud.
"""

from __future__ import annotations

import argparse
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
except ImportError:
    raise SystemExit("needs scikit-learn")


def auroc(s, y):
    m = np.isfinite(s); s, y = s[m], y[m]
    p, n = int((y == 1).sum()), int((y == 0).sum())
    if not p or not n:
        return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sr = s[o]; i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def bdir(a):
    return max(a, 1 - a) if np.isfinite(a) else a


def spear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m])))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    gnames = [str(x) for x in z["geom_feature_names"]]
    cnames = [str(x) for x in z["cloud_feature_names"]]
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    SG, SC, SR = z["stepgeom"], z["stepcloud"], z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)
    need = ["resultant", "resultant_bulk", "norm_bulk"]
    for nm in need:
        if nm not in cnames:
            raise SystemExit(f"{nm} not in stepcloud; re-extract with --cloud_eff_rank "
                             f"(+ --massive_global for clean bulk)")

    cols = {k: [] for k in ["norm", "resultant", "norm_bulk", "resultant_bulk"]}
    Y, G = [], []
    for i in range(len(SG)):
        sg = np.asarray(SG[i], float); sc = np.asarray(SC[i], float)
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0); T = rng.shape[0]
        for j in range(T):
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                continue
            cols["norm"].append(sg[j, li, gnames.index("norm")])
            cols["resultant"].append(sc[j, li, cnames.index("resultant")])
            cols["norm_bulk"].append(sc[j, li, cnames.index("norm_bulk")])
            cols["resultant_bulk"].append(sc[j, li, cnames.index("resultant_bulk")])
            Y.append(y); G.append(i)
    for k in cols:
        cols[k] = np.asarray(cols[k], float)
    Y = np.asarray(Y, int); G = np.asarray(G, int)
    keys = ["norm", "resultant", "norm_bulk", "resultant_bulk"]

    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"\nstandalone AUROC (best dir):")
    for k in keys:
        print(f"  {k:16s} {bdir(auroc(cols[k], Y)):.3f}")
    print(f"\nSpearman corr matrix:")
    print("              " + "".join(f"{k[:10]:>12s}" for k in keys))
    for a in keys:
        print(f"{a:14s}" + "".join(f"{spear(cols[a], cols[b]):>12.2f}" for b in keys))

    gkf = GroupKFold(args.folds)
    def oof(X):
        s = np.full(len(Y), np.nan)
        for tr, te in gkf.split(X, Y, G):
            if len(np.unique(Y[tr])) < 2:
                continue
            p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            p.fit(X[tr], Y[tr]); s[te] = p.predict_proba(X[te])[:, 1]
        return s
    def incr(a, b):  # AUROC(a+b) - AUROC(b), chain-paired bootstrap
        sa = oof(np.c_[cols[a], cols[b]]); sb = oof(cols[b][:, None])
        rng = np.random.default_rng(0); ch = np.unique(G); d = []
        for _ in range(1000):
            cb = rng.choice(ch, len(ch), replace=True)
            m = np.concatenate([np.where(G == c)[0] for c in cb])
            d.append(auroc(sa[m], Y[m]) - auroc(sb[m], Y[m]))
        d = np.asarray(d); lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return f"+{np.nanmean(d):.3f} [{lo:+.3f},{hi:+.3f}] {'SIG' if lo > 0 else 'ns'}"

    print(f"\nmutual increments (BULK, massive removed):")
    print(f"  resultant_bulk OVER norm_bulk: {incr('resultant_bulk', 'norm_bulk')}")
    print(f"  norm_bulk OVER resultant_bulk: {incr('norm_bulk', 'resultant_bulk')}")
    print("\nverdict: corr(norm_bulk,resultant_bulk) collapses + both AUROC ~0.5 => ONE signal "
          "(massive energy). Both survive + corr stays + mutual increments ns => ONE signal "
          "(bulk, not massive). Both survive + corr drops + an increment SIG => TWO axes.")


if __name__ == "__main__":
    main()
