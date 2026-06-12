"""Length-confound diagnostics for a per-step metric (default norm @ a layer).

Step-level, ProcessBench process labels (positive = gold first-error step;
negative = correct-chain steps + pre-error steps; post-error excluded).

(a) correlation of the metric with log(step length)        -- how much it tracks length
(b) AUROC after residualizing on length + position (GBM, cross-fit on correct)
(c) length-stratified within-bucket AUROC                  -- the gold test: at FIXED
    length, does the metric still separate first-error from good steps?

If (c) within-bucket AUROC stays away from 0.5, the metric is NOT just length.
"""

from __future__ import annotations

import argparse
import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--metric", default="norm")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--nbuckets", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    layers = [int(x) for x in z["layers_used"]]
    li = layers.index(args.layer)
    geom_names = [str(x) for x in z["geom_feature_names"]]
    cloud_names = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    if args.metric in geom_names:
        SRC, fi = z["stepgeom"], geom_names.index(args.metric)
    elif args.metric in cloud_names:
        SRC, fi = z["stepcloud"], cloud_names.index(args.metric)
    else:
        raise SystemExit(f"metric {args.metric} not found")
    ges = z["gold_error_step"].astype(int); SR = z["step_token_ranges"]

    M, NT, POS, Y, G, H = [], [], [], [], [], []
    for i in range(len(SRC)):
        g = np.asarray(SRC[i], float)
        if g.ndim == 3:
            g = g[:, li, fi]
        else:
            g = g[:, li]
        T = len(g); rng = np.asarray(SR[i], int); k = int(ges[i]); corr = (k < 0)
        for j in range(T):
            if not np.isfinite(g[j]):
                continue
            if corr or j < k:
                y, keep = 0, True
            elif j == k:
                y, keep = 1, True
            else:
                keep = False
            if keep:
                M.append(g[j]); NT.append(int(rng[j, 1] - rng[j, 0] + 1))
                POS.append(j / max(1, T - 1)); Y.append(y); G.append(i); H.append(corr)
    M = np.asarray(M, float); NT = np.asarray(NT, float); POS = np.asarray(POS, float)
    Y = np.asarray(Y, int); G = np.asarray(G, int); H = np.asarray(H, bool)
    logL = np.log(np.maximum(NT, 1))
    print(f"metric {args.metric}@L{args.layer} | steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"raw AUROC (best dir): {max(auroc(M, Y), 1 - auroc(M, Y)):.3f}")

    # (a) correlation with log(length)
    def pear(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    def spear(a, b):
        return pear(np.argsort(np.argsort(a)).astype(float),
                    np.argsort(np.argsort(b)).astype(float))
    print(f"\n(a) corr(metric, log step length):  Pearson {pear(M, logL):+.3f}  "
          f"Spearman {spear(M, logL):+.3f}")

    # (b) residualize on length + position
    resid = np.full(len(M), np.nan)
    X = np.c_[NT, POS]
    for tr, te in GroupKFold(args.folds).split(M, Y, G):
        Htr = H[tr]
        if Htr.sum() < 20:
            continue
        reg = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0)
        reg.fit(X[tr][Htr], M[tr][Htr])
        resid[te] = M[te] - reg.predict(X[te])
    a_raw = max(auroc(M, Y), 1 - auroc(M, Y))
    a_res = max(auroc(resid, Y), 1 - auroc(resid, Y))
    print(f"\n(b) residualize on length+position:  raw {a_raw:.3f} -> resid {a_res:.3f}"
          f"  (drop {a_raw - a_res:+.3f})")

    # (c) within length-bucket AUROC (length held ~constant inside a bucket)
    edges = np.quantile(NT, np.linspace(0, 1, args.nbuckets + 1))
    edges[-1] += 1
    b = np.clip(np.digitize(NT, edges[1:-1]), 0, args.nbuckets - 1)
    print(f"\n(c) within length-bucket AUROC ({args.nbuckets} quantile buckets):")
    print(f"  {'bucket(n_tok)':18s} {'n_err':>6s} {'n_good':>7s} {'AUROC(bestdir)':>15s}")
    num = den = 0.0
    for bb in range(args.nbuckets):
        mask = b == bb
        ne, ng = int(Y[mask].sum()), int((Y[mask] == 0).sum())
        a = auroc(M[mask], Y[mask]); a = max(a, 1 - a) if np.isfinite(a) else a
        lo, hi = NT[mask].min(), NT[mask].max()
        print(f"  [{int(lo):4d},{int(hi):4d}]{'':7s} {ne:6d} {ng:7d} {a:15.3f}")
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    print(f"  {'pooled (length-conditional)':18s} {'':6s} {'':7s} "
          f"{num/den if den else float('nan'):15.3f}")
    print("\n(c) pooled away from 0.5 => metric separates first-error from good steps "
          "AT FIXED LENGTH -> not just a length proxy.")


if __name__ == "__main__":
    main()
