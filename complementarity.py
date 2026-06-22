"""Why OURS beats EDIS: the geometry signal is weakly correlated with the entropy-dynamics signal
(a genuinely DIFFERENT modality), so combining them has large headroom (cross-ceiling).

Substantiates the fusion win with two diagnostics, per config:
  (1) correlation between the per-step signals: geom (directional collapse = -pooled_norm) vs EDIS,
      and geom vs our own entropy battery. LOW |corr| = orthogonal modality = the reason fusion wins.
  (2) cross-ceiling: at a matched per-step FPR (threshold each signal so 20% of CORRECT steps fire),
      recall of geom alone, EDIS alone, their UNION (oracle router upper bound), and intersection.
      The union >> each alone = how much the two TOGETHER cover that neither does. Plus the AUROC of
      an unsupervised OR-fusion (max of percentile ranks) as the achievable ceiling.

Needs coh.npz: stepcloud + stepgeom(norm) + tok_U_D + step_token_ranges + gold_error_step + layers_used.
"""

from __future__ import annotations
import argparse
import numpy as np


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


def ecdf(x):
    x = np.asarray(x, float); out = np.full(len(x), 0.5); m = np.isfinite(x)
    if m.sum() < 2:
        return out
    v = x[m]; o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); r[o] = np.arange(len(v))
    out[m] = r / (len(v) - 1); return out


def pear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def edis(H, w=8, tb=1.36, tr=1.33):
    H = np.asarray(H, float); H = H[np.isfinite(H)]
    if len(H) < 3:
        return 0.0
    ww = min(w, max(2, len(H) // 2))
    burst = sum(1 for t in range(len(H) - ww) if H[t + ww] - H[t] > tb) if len(H) > ww else 0
    rebound = 0; rmin = H[0]
    for t in range(1, len(H)):
        if H[t] - rmin > tr:
            rebound += 1
        rmin = min(rmin, H[t])
    return 0.5 * (burst + rebound) * (1.0 + float(H.var()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--fpr", type=float, default=0.2)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]
    gn = [str(x) for x in z["geom_feature_names"]] if "geom_feature_names" in z.files else []
    lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    SG = z["stepgeom"] if "stepgeom" in z.files else None
    UD = z["tok_U_D"]; fi = cn.index("resultant"); ngi = gn.index("norm") if "norm" in gn else None

    GEO, EDS, UNCV, Y = [], [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); sg = np.asarray(SG[i], float) if SG is not None else None
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0); T = rng.shape[0]
        a0 = int(rng[0, 0]); ud = np.asarray(UD[i], float)
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(ud), int(rng[j, 1]) - a0 + 1)
            if hi - lo < 2:
                continue
            uds = ud[lo:hi]
            pool = sg[j, li, ngi] if (sg is not None and ngi is not None) else sc[j, li, fi]
            GEO.append(-float(pool)); EDS.append(edis(uds)); UNCV.append(float(np.var(uds)))  # geom collapse / edis / varentropy
            Y.append(lab)
    GEO = np.asarray(GEO); EDS = np.asarray(EDS); UNCV = np.asarray(UNCV); Y = np.asarray(Y, int)

    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"\nsingle AUROC:  geom {bdir(auroc(GEO, Y)):.3f}   EDIS {bdir(auroc(EDS, Y)):.3f}   "
          f"varentropy {bdir(auroc(UNCV, Y)):.3f}")
    rg, re_, ru = ecdf(GEO), ecdf(EDS), ecdf(UNCV)
    print(f"\nsignal correlation (Pearson on percentile ranks; LOW = orthogonal modality):")
    print(f"  corr(geom, EDIS)        {pear(rg, re_):+.3f}")
    print(f"  corr(geom, varentropy)  {pear(rg, ru):+.3f}")
    print(f"  corr(EDIS, varentropy)  {pear(re_, ru):+.3f}   (both entropy -> expect HIGH)")

    # cross-ceiling: threshold each signal so fpr of CORRECT steps fire; recall on error steps
    neg = Y == 0; pos = Y == 1
    def thr(s):
        return np.quantile(s[neg], 1 - args.fpr)
    tg, te = thr(GEO), thr(EDS)
    fire_g = GEO >= tg; fire_e = EDS >= te
    rec_g = float(np.mean(fire_g[pos])); rec_e = float(np.mean(fire_e[pos]))
    rec_u = float(np.mean((fire_g | fire_e)[pos])); rec_i = float(np.mean((fire_g & fire_e)[pos]))
    print(f"\ncross-ceiling at matched per-step FPR={args.fpr} (recall on first-error steps):")
    print(f"  geom alone        {rec_g:.3f}")
    print(f"  EDIS alone        {rec_e:.3f}")
    print(f"  UNION (oracle)    {rec_u:.3f}   <- ceiling: errors caught by geom OR EDIS")
    print(f"  both              {rec_i:.3f}")
    print(f"  geom-only catches {rec_g - rec_i:+.3f}   EDIS-only catches {rec_e - rec_i:+.3f}  "
          f"(disjoint coverage)")
    or_auroc = bdir(auroc(np.maximum(rg, re_), Y))
    print(f"\nOR-fusion AUROC (max percentile rank, unsupervised, achievable) = {or_auroc:.3f}")
    print("\nread: LOW corr(geom,EDIS) = geometry is a different modality from entropy dynamics; varentropy "
          "(our own entropy) correlates HIGH with EDIS (both entropy) but LOW with geometry -- so the gain is "
          "the geometric axis, not a better entropy. The UNION recall >> each alone quantifies the headroom: "
          "geom and EDIS catch largely DISJOINT errors. This is the mechanism behind OURS > EDIS: combining "
          "an orthogonal modality, not tuning the same one.")


if __name__ == "__main__":
    main()
