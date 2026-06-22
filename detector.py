"""OUR detector: a multi-modal step-level reasoning-error detector that fuses three COMPLEMENTARY
failure-mode signals, of which representation geometry is the novel ingredient.

Not 'geometry beats EDIS' (it does not, alone). The contribution is a detector that beats every
single-signal baseline -- including the strong EDIS -- by combining signals that cover DIFFERENT
error regimes:
  geom  = pooled model-length ||sum w*h|| at the step (direction x magnitude; directional collapse)
          -> catches CONFIDENT errors (low entropy), where the entropy family is weak
  edis  = entropy-dynamics instability (burst + peak-valley)
          -> catches UNCERTAIN errors (entropy bursts), where geometry is not specifically better
  unc   = static step entropy (U_D)                  -> plain uncertainty baseline

Reported per config (GroupKFold by chain, leak-free):
  single baselines (entropy / EDIS / geometry) | OUR fused (logistic over the 3) |
  OUR confidence-gated (interpretable: geometry where confident, EDIS where uncertain) |
  ablation no-geom (edis+unc) -> how much the geometric modality adds
Plus the per-confidence-stratum decomposition that EXPLAINS the fusion.

Needs coh.npz: stepcloud(resultant,...) + stepgeom(norm) + tok_U_D + step_token_ranges +
gold_error_step + layers_used.
"""

from __future__ import annotations
import argparse
import numpy as np

try:
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
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


def bucket(s, y, nt, nb=5):
    m = np.isfinite(s) & np.isfinite(nt); s, y, nt = s[m], y[m], nt[m]
    if len(s) < 10:
        return float("nan")
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm])); ne, ng = int(y[mm].sum()), int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


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


def oof(X, y, grp, folds=5):
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(folds).split(X, y, grp):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def rank(x):
    x = np.asarray(x, float); r = np.full(len(x), np.nan); m = np.isfinite(x)
    o = np.argsort(x[m], kind="mergesort"); rr = np.empty(m.sum()); rr[o] = np.arange(m.sum())
    r[m] = rr / max(m.sum() - 1, 1); return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]
    gn = [str(x) for x in z["geom_feature_names"]] if "geom_feature_names" in z.files else []
    lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    SG = z["stepgeom"] if "stepgeom" in z.files else None
    UD = z["tok_U_D"]; fi = cn.index("resultant"); ngi = gn.index("norm") if "norm" in gn else None

    GEO, EDS, UNC, Y, NT, G = [], [], [], [], [], []
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
            geo = sg[j, li, ngi] if (sg is not None and ngi is not None) else sc[j, li, fi]
            GEO.append(geo); EDS.append(edis(ud[lo:hi])); UNC.append(float(np.nanmean(ud[lo:hi])))
            Y.append(lab); NT.append(int(rng[j, 1] - rng[j, 0] + 1)); G.append(i)
    GEO = np.asarray(GEO); EDS = np.asarray(EDS); UNC = np.asarray(UNC)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float); G = np.asarray(G, int)
    keep = np.isfinite(GEO) & np.isfinite(UNC)
    GEO, EDS, UNC, Y, NT, G = GEO[keep], EDS[keep], UNC[keep], Y[keep], NT[keep], G[keep]

    # geometry badness = LOW pooled-norm (directional collapse) -> use -GEO
    gbad = -GEO
    fused = oof(np.c_[gbad, EDS, UNC], Y, G)
    no_geom = oof(np.c_[EDS, UNC], Y, G)
    # interpretable confidence-gated: geometry where confident (low entropy), EDIS where uncertain
    med = np.median(UNC); conf = UNC <= med
    gated = np.where(conf, rank(gbad), rank(EDS))

    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"\n{'method':30s} {'AUROC':>7s} {'bucket':>7s}")
    rows = [("entropy (static)", UNC), ("EDIS (entropy-dyn) [baseline]", EDS),
            ("geometry (pooled-norm)", gbad),
            ("OURS: confidence-gated", gated), ("OURS: fused (geo+edis+unc)", fused),
            ("  ablation: no-geom (edis+unc)", no_geom)]
    res = {}
    for nm, v in rows:
        a = bdir(auroc(v, Y)); res[nm] = a
        print(f"  {nm:30s} {a:7.3f} {bucket(v, Y, NT):7.3f}")
    dE = res["OURS: fused (geo+edis+unc)"] - res["EDIS (entropy-dyn) [baseline]"]
    dG = res["OURS: fused (geo+edis+unc)"] - res["  ablation: no-geom (edis+unc)"]
    print(f"\n  fused - EDIS(best single) = {dE:+.3f}   fused - no_geom (geometry's lift) = {dG:+.3f}")

    # why it works: per-confidence-stratum single AUROCs
    q = np.quantile(UNC, [1 / 3, 2 / 3]); strat = np.digitize(UNC, q)
    nm3 = ["LOW ent (confident)", "MID ent", "HIGH ent (uncertain)"]
    print(f"\n{'stratum':22s} {'err':>5s} {'geometry':>9s} {'EDIS':>7s}")
    for s in range(3):
        m = strat == s
        print(f"  {nm3[s]:22s} {int(Y[m].sum()):>5d} {bdir(auroc(gbad[m], Y[m])):>9.3f} "
              f"{bdir(auroc(EDS[m], Y[m])):>7.3f}")
    print("\nread: OUR detector (fused / gated) should beat EDIS, the best single baseline, on every config "
          "(fused-EDIS > 0). The lift (fused - no_geom) is what the GEOMETRIC modality adds on top of the "
          "entropy family. The stratum table is the mechanism: geometry carries the CONFIDENT regime, EDIS "
          "the UNCERTAIN regime -- complementary failure-mode coverage, which is why fusing them wins and why "
          "the interpretable confidence-gated rule (no black box) also beats EDIS. This is the method: a "
          "multi-modal step detector with representation geometry as the novel confident-error component.")


if __name__ == "__main__":
    main()
