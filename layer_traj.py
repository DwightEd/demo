"""Cross-layer TRAJECTORY of directional concentration as a feature.

Hypothesis (user): correct reasoning's token directions CONVERGE from shallow to
deep (tokens anchor more with depth); at a first-error step this convergence
BREAKS. So the SHAPE of resultant-across-layers (slope / deep-minus-shallow /
curvature) may beat the single best-layer scalar.

Per step we have resultant at L layers. Derived trajectory features:
  slope        : OLS slope of resultant vs layer index
  deep_minus_sh: mean(deep 3 layers) - mean(shallow 3 layers)  (the convergence increment)
  curvature    : mean 2nd difference
  rng          : max - min across layers
Tests each derived feature's AUROC (raw + within-length-bucket), and the KEY
question via GroupKFold logistic + chain-paired bootstrap:
  does the cross-layer SHAPE add over the single best-layer resultant?
  (also over [confound + U + best-layer resultant])

Caveat: adjacent layers are highly correlated (residual stream), so a 'trajectory'
can be trivial drift. First test: does the shape add AT ALL over one layer? If not,
the convergence idea is dead (consistent with the spectral-field falsification).

Runs on existing _coh.npz (resultant stored at all --layers).
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


def bucket_auroc(score, y, nt, nb=5):
    edges = np.quantile(nt, np.linspace(0, 1, nb + 1)); edges[-1] += 1
    b = np.clip(np.digitize(nt, edges[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        m = b == bb
        ne, ng = int(y[m].sum()), int((y[m] == 0).sum())
        a = bdir(auroc(score[m], y[m]))
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--metric", default="resultant")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    gnames = [str(x) for x in z["geom_feature_names"]]
    layers = np.array([int(x) for x in z["layers_used"]], float)
    if args.metric in cnames:
        SRC, fi = z["stepcloud"], cnames.index(args.metric)
    elif args.metric in gnames:
        SRC, fi = z["stepgeom"], gnames.index(args.metric)
    else:
        raise SystemExit(f"metric {args.metric} not found")
    SR = z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None
    L = len(layers)
    nd = 3 if L >= 6 else 1                              # deep/shallow window
    lz = (layers - layers.mean()) / (layers.std() + 1e-9)

    PROF, NT, POS, Y, G, UEX = [], [], [], [], [], []
    for i in range(len(SRC)):
        g = np.asarray(SRC[i], float)
        prof_all = g[:, :, fi] if g.ndim == 3 else g    # (T, L)
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0)
        a0 = int(rng[0, 0]); T = prof_all.shape[0]
        ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        for j in range(T):
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                continue
            p = prof_all[j]
            if not np.isfinite(p).all():
                continue
            PROF.append(p); NT.append(int(rng[j, 1] - rng[j, 0] + 1))
            POS.append(j / max(1, T - 1)); Y.append(y); G.append(i)
            if ud is not None:
                lo = max(0, int(rng[j, 0]) - a0); hi = min(len(ud), int(rng[j, 1]) - a0 + 1)
                UEX.append([np.nanmean(ud[lo:hi]) if hi > lo else np.nan,
                            np.nanmean(uc[lo:hi]) if hi > lo else np.nan])
    PROF = np.asarray(PROF, float); NT = np.asarray(NT, float); POS = np.asarray(POS, float)
    Y = np.asarray(Y, int); G = np.asarray(G, int)

    # derived trajectory features
    slope = ((PROF - PROF.mean(1, keepdims=True)) * lz).sum(1) / (lz @ lz)
    deep_sh = PROF[:, -nd:].mean(1) - PROF[:, :nd].mean(1)
    curv = np.diff(PROF, n=2, axis=1).mean(1) if L >= 3 else np.zeros(len(PROF))
    rngf = PROF.max(1) - PROF.min(1)
    best_li = int(np.argmax([bdir(auroc(PROF[:, l], Y)) for l in range(L)]))

    print(f"file: {args.npz} | metric {args.metric} | steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"layers {layers.astype(int).tolist()} | best single layer = {int(layers[best_li])}")
    print(f"\n{'feature':16s} {'raw':>7s} {'bucket':>8s}")
    feats = [("best-layer", PROF[:, best_li]), ("slope", slope),
             ("deep_minus_shallow", deep_sh), ("curvature", curv), ("range", rngf)]
    for nm, v in feats:
        print(f"{nm:16s} {bdir(auroc(v, Y)):7.3f} {bucket_auroc(v, Y, NT):8.3f}")

    # does the cross-layer SHAPE add over best single layer? (and over confound+U+best)
    gkf = GroupKFold(args.folds)
    def oof(X):
        s = np.full(len(Y), np.nan)
        for tr, te in gkf.split(X, Y, G):
            if len(np.unique(Y[tr])) < 2:
                continue
            p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            p.fit(X[tr], Y[tr]); s[te] = p.predict_proba(X[te])[:, 1]
        return s
    shape = np.c_[slope, deep_sh, curv, rngf]
    one = PROF[:, [best_li]]
    s_one = oof(one); s_shape = oof(shape); s_prof = oof(PROF); s_one_shape = oof(np.c_[one, shape])
    base = np.c_[np.log(np.maximum(NT, 1)), POS]
    if UEX:
        UEX = np.asarray(UEX, float)
        for c in range(UEX.shape[1]):
            col = UEX[:, c]; col[~np.isfinite(col)] = np.nanmean(col)
        base = np.c_[base, UEX]
    s_b = oof(base); s_b1 = oof(np.c_[base, one]); s_ball = oof(np.c_[base, PROF])

    def ci(sa, sb):
        rng = np.random.default_rng(0); chains = np.unique(G); d = []
        for _ in range(500):
            cb = rng.choice(chains, size=len(chains), replace=True)
            m = np.concatenate([np.where(G == c)[0] for c in cb])
            d.append(auroc(sa[m], Y[m]) - auroc(sb[m], Y[m]))
        d = np.asarray(d); lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return f"+{np.nanmean(d):.3f} [{lo:+.3f},{hi:+.3f}] {'SIG' if lo > 0 else 'ns'}"

    print(f"\n=== does cross-layer shape beat one layer? ===")
    print(f"  best single layer:          {auroc(s_one, Y):.3f}")
    print(f"  shape only (slope/ds/curv/rng): {auroc(s_shape, Y):.3f}")
    print(f"  full profile (all layers):  {auroc(s_prof, Y):.3f}")
    print(f"  one + shape:                {auroc(s_one_shape, Y):.3f}")
    print(f"  shape OVER one layer:       {ci(s_one_shape, s_one)}")
    print(f"\n=== over confound+U ===")
    print(f"  base(len+pos+U):            {auroc(s_b, Y):.3f}")
    print(f"  + best layer:               {auroc(s_b1, Y):.3f}")
    print(f"  + full profile:             {auroc(s_ball, Y):.3f}")
    print(f"  full-profile OVER base+1layer: {ci(s_ball, s_b1)}")
    print("\nread: if shape adds nothing over one layer (ns), cross-layer convergence is "
          "not signal (residual-stream drift). If it adds, the trajectory shape is real.")


if __name__ == "__main__":
    main()
