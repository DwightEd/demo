"""metric-then-pool vs pool-then-metric.

Current step features (stepgeom) are POOL-THEN-METRIC: exp-pool the token cloud
into one step vector z, then compute geometry(z) (norm/pr/ae/...).

This script tests the OPPOSITE order, METRIC-THEN-POOL: compute the geometry of
EACH token first (that is `tokgeom`, stored at extraction), then pool those
per-token scalars to a step value (mean / exp-weighted / max / std). Question:
does per-token-then-pool separate first-error from good steps -- and better than
pool-then-metric?

Step-level, ProcessBench process labels (positive=gold first-error; negative=
correct-chain + pre-error). Reports per (metric x pool): raw AUROC (best dir) +
within-length-bucket AUROC, side by side with the pool-then-metric stepgeom
value for the same metric. Then a fused logistic over ALL per-token-pooled
features: increment over (nuisance + U_D/U_C), to compare with fuse_detector's
pool-then-metric fused number.

Needs an npz with tokgeom (extract WITHOUT --no_token_geom).
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


def pool_vec(v, kind):
    """pool per-token values v (n,) -> scalar."""
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    if kind == "mean":
        return float(v.mean())
    if kind == "max":
        return float(v.max())
    if kind == "std":
        return float(v.std())
    if kind == "expw":                                  # later tokens heavier (Lu Eq.6)
        n = len(v)
        if n == 1:
            return float(v[0])
        w = np.exp(np.arange(n) / (n - 1)); w /= w.sum()
        return float((w * v).sum())
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--pools", default="mean,expw,max,std")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--with_ud", action="store_true")
    args = ap.parse_args()
    pools = args.pools.split(",")

    z = np.load(args.npz, allow_pickle=True)
    if "tokgeom" not in z.files or z["tokgeom"][0] is None:
        raise SystemExit("no tokgeom; extract WITHOUT --no_token_geom")
    names = [str(x) for x in z["geom_feature_names"]]
    layers = [int(x) for x in z["layers_used"]]
    li = layers.index(args.layer)
    TG = z["tokgeom"]; SG = z["stepgeom"]; SR = z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"] if (args.with_ud and "tok_U_D" in z.files) else None
    UC = z["tok_U_C"] if (args.with_ud and "tok_U_C" in z.files) else None

    F = len(names)
    # per-step: per-token-pooled features (F x pools), pool-then-metric stepgeom (F),
    # plus nuisances + U
    PT, PM, NT, POS, DENS, Y, G, UEX = [], [], [], [], [], [], [], []
    for i in range(len(TG)):
        if TG[i] is None:
            continue
        tg = np.asarray(TG[i], float)                  # (R, L, F)
        sg = np.asarray(SG[i], float)                  # (T, L, F)
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
            lo = max(0, int(rng[j, 0]) - a0); hi = min(tg.shape[0], int(rng[j, 1]) - a0 + 1)
            if hi - lo < 1:
                continue
            block = tg[lo:hi, li, :]                    # (n_j, F) per-token geometry
            row = []
            for fi in range(F):
                for p in pools:
                    row.append(pool_vec(block[:, fi], p))
            PT.append(row)
            PM.append(sg[j, li, :].tolist())            # pool-then-metric
            NT.append(hi - lo); POS.append(j / max(1, T - 1)); Y.append(y); G.append(i)
            if ud is not None:
                UEX.append([np.nanmean(ud[lo:hi]), np.nanmean(uc[lo:hi])])
    PT = np.asarray(PT, float); PM = np.asarray(PM, float)
    NT = np.asarray(NT, float); Y = np.asarray(Y, int); G = np.asarray(G, int)
    for A in (PT, PM):
        for c in range(A.shape[1]):
            col = A[:, c]; col[~np.isfinite(col)] = np.nanmean(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0

    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"pools: {pools}\n")
    print(f"{'metric':10s} | {'pool-then-metric':>16s} | metric-then-pool (raw / bucket)")
    print(f"{'':10s} | {'raw':>7s} {'bucket':>8s} | " + "  ".join(f"{p:>13s}" for p in pools))
    print("-" * (32 + 16 * len(pools)))
    col = 0
    for fi, nm in enumerate(names):
        pm_raw = bdir(auroc(PM[:, fi], Y)); pm_bk = bucket_auroc(PM[:, fi], Y, NT)
        cells = []
        for p in pools:
            v = PT[:, col]; col += 1
            cells.append(f"{bdir(auroc(v, Y)):.3f}/{bucket_auroc(v, Y, NT):.3f}")
        print(f"{nm:10s} | {pm_raw:7.3f} {pm_bk:8.3f} | " + "  ".join(f"{c:>13s}" for c in cells))

    # fused increment: per-token-pooled ALL features vs baseline (nuis+U)
    base = np.c_[NT, POS]
    base_lbl = "n_tok,pos"
    if UEX:
        UEX = np.asarray(UEX, float)
        for c in range(UEX.shape[1]):
            colu = UEX[:, c]; colu[~np.isfinite(colu)] = np.nanmean(colu)
        base = np.c_[base, UEX]; base_lbl += "+U_D,U_C"

    gkf = GroupKFold(args.folds)
    def oof(X):
        s = np.full(len(Y), np.nan)
        for tr, te in gkf.split(X, Y, G):
            if len(np.unique(Y[tr])) < 2:
                continue
            p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            p.fit(X[tr], Y[tr]); s[te] = p.predict_proba(X[te])[:, 1]
        return s
    a_base = auroc(oof(base), Y)
    a_pt = auroc(oof(np.c_[base, PT]), Y)
    a_pm = auroc(oof(np.c_[base, PM]), Y)
    print(f"\n=== fused detectors (GroupKFold logistic) ===")
    print(f"  baseline ({base_lbl}):                 {a_base:.3f}")
    print(f"  + pool-then-metric (stepgeom):        {a_pm:.3f}  (+{a_pm-a_base:.3f})")
    print(f"  + metric-then-pool (per-token pooled): {a_pt:.3f}  (+{a_pt-a_base:.3f})")
    print("\nread: if metric-then-pool raw/bucket or fused > pool-then-metric, computing "
          "geometry PER TOKEN first keeps signal that cloud-pooling-then-metric destroys.")


if __name__ == "__main__":
    main()
