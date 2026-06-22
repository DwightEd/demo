"""Does the centered-covariance spectral SHAPE (PR/top1) add to the FULL detector (geometry+entropy),
not just over resultant alone? Joins respcloud (_cloud.npz -> shape) with tok_U_D (_coh.npz -> our
entropy battery) for the SAME responses, and compares OURS vs OURS+shape.

Motivated by spectral_shape.py: on the HARD config (omnimath) error steps disperse ISOTROPICALLY
(high PR on the centered unit-vector covariance) -> shape gives +0.055 over resultant. Here we check
the net increment on top of the full OURS detector (pooled_norm + our uncertainty battery), per config.

Pass BOTH files for one config: --cloud pb_X_cloud.npz --coh pb_X_coh.npz. Alignment is by response
index; gold_error_step is verified to match between files (responses skipped on mismatch).

Needs: _cloud.npz (respcloud + cloud_store_layers) and _coh.npz (stepgeom norm + tok_U_D), same run.
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


def unc_feats(e):
    e = np.asarray(e, float); e = e[np.isfinite(e)]
    if len(e) < 2:
        return [float(e.mean()) if len(e) else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    t = np.arange(len(e)); slope = float(np.polyfit(t, e, 1)[0]) if len(e) >= 3 else 0.0
    late = float(e[-max(1, len(e) // 3):].mean())
    return [float(e.mean()), float(e.var()), float(e.max()), float(e.max() - e.min()), slope, late]


def shape_feats(H):
    H = np.asarray(H, np.float64); nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
    if ok.sum() < 4:
        return None
    u = H[ok] / nrm[ok, None]; n = u.shape[0]
    Uc = u - u.mean(0); s = np.linalg.svd(Uc, compute_uv=False); lam = (s ** 2) / n; tot = float(lam.sum())
    if tot <= 1e-12:
        return None
    PR = float(tot * tot / (np.square(lam).sum() + 1e-18)); top1 = float(lam[0] / tot)
    return PR, top1


def oof(X, y, grp, folds=5):
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(folds).split(X, y, grp):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--coh", required=True)
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    zc = np.load(args.cloud, allow_pickle=True); zh = np.load(args.coh, allow_pickle=True)
    csl = [int(x) for x in zc["cloud_store_layers"]]; cli = csl.index(args.layer)
    RC = zc["respcloud"]; SRc = zc["step_token_ranges"]; gesc = zc["gold_error_step"].astype(int)
    cn = [str(x) for x in zh["cloud_feature_names"]]
    gn = [str(x) for x in zh["geom_feature_names"]] if "geom_feature_names" in zh.files else []
    lyu = [int(x) for x in zh["layers_used"]]; li = lyu.index(args.layer)
    SC = zh["stepcloud"]; SG = zh["stepgeom"] if "stepgeom" in zh.files else None
    SRh = zh["step_token_ranges"]; gesh = zh["gold_error_step"].astype(int)
    UD = zh["tok_U_D"]; fi = cn.index("resultant"); ngi = gn.index("norm") if "norm" in gn else None
    if len(RC) != len(SC):
        print(f"[warn] response count differs: cloud {len(RC)} vs coh {len(SC)}")
    nmatch = min(len(RC), len(SC))

    GEO, UNC, PR, T1, Y, NT, G = [], [], [], [], [], [], []
    skip = 0
    for i in range(nmatch):
        if RC[i] is None or int(gesc[i]) != int(gesh[i]):
            skip += 1; continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]
        sc = np.asarray(SC[i], float); sg = np.asarray(SG[i], float) if SG is not None else None
        rng = np.asarray(SRh[i], int); rngc = np.asarray(SRc[i], int)
        k = int(gesh[i]); correct = (k < 0); T = rng.shape[0]; a0 = int(rng[0, 0]); a0c = int(rngc[0, 0])
        ud = np.asarray(UD[i], float)
        if rng.shape[0] != rngc.shape[0]:
            skip += 1; continue
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(ud), int(rng[j, 1]) - a0 + 1)
            loc = max(0, int(rngc[j, 0]) - a0c); hic = min(rcl.shape[0], int(rngc[j, 1]) - a0c + 1)
            if hi - lo < 2 or hic - loc < 4:
                continue
            sf = shape_feats(rcl[loc:hic])
            if sf is None:
                continue
            geo = sg[j, li, ngi] if (sg is not None and ngi is not None) else sc[j, li, fi]
            GEO.append(geo); UNC.append(unc_feats(ud[lo:hi])); PR.append(sf[0]); T1.append(sf[1])
            Y.append(lab); NT.append(int(rng[j, 1] - rng[j, 0] + 1)); G.append(i)
    GEO = np.asarray(GEO); UNC = np.asarray(UNC, float); PR = np.asarray(PR); T1 = np.asarray(T1)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float); G = np.asarray(G, int)

    print(f"cloud={args.cloud} coh={args.coh} | layer {args.layer} | matched steps {len(Y)} "
          f"first-error {int(Y.sum())} | skipped responses {skip}")
    ours = oof(np.c_[-GEO, UNC], Y, G)
    ours_shape = oof(np.c_[-GEO, UNC, PR, T1], Y, G)
    geo_only = oof((-GEO).reshape(-1, 1), Y, G)
    geo_shape = oof(np.c_[-GEO, PR, T1], Y, G)
    print(f"\n{'method':30s} {'AUROC':>7s} {'bucket':>7s}")
    for nm, v in [("geometry (pooled-norm)", geo_only), ("geometry + shape", geo_shape),
                  ("OURS (geom + our entropy)", ours), ("OURS + shape", ours_shape)]:
        print(f"  {nm:30s} {bdir(auroc(v, Y)):7.3f} {bucket(v, Y, NT):7.3f}")
    print(f"\n  geo+shape - geo = {bdir(auroc(geo_shape, Y)) - bdir(auroc(geo_only, Y)):+.3f}   "
          f"OURS+shape - OURS = {bdir(auroc(ours_shape, Y)) - bdir(auroc(ours, Y)):+.3f}")
    print("\nread: the decisive number is OURS+shape - OURS. If > 0 (esp. on the hard config), the centered-"
          "covariance spectral shape adds a real increment ON TOP OF the full geometry+entropy detector -- a "
          "second within-step geometric axis (isotropy of the dispersion) that fires where total concentration "
          "saturates. If ~0, shape is already covered by geometry+entropy jointly. Run gsm8k and omnimath.")


if __name__ == "__main__":
    main()
