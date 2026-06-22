"""Rigorous ceiling probe: does centered-covariance spectral SHAPE (PR/top1) add a REAL increment
after controlling for EDIS (is PR an entropy-dynamics shadow?) and step length (is PR just token
count?). Joins respcloud (_cloud.npz shape) with tok_U_D (_coh.npz entropy/EDIS) for the same
responses.

Two controls that the earlier spectral_shape.py +0.055 did NOT do:
  (1) EDIS in the baseline -- PR measures isotropic dispersion, which may overlap entropy instability.
      The decisive test is the increment of shape over [resultant + EDIS], not over resultant alone.
  (2) step length in the baseline -- PR = (Sum l)^2/Sum l^2 is scale-invariant but NOT dimension-
      invariant; PR ~ 25-33 tracks token count. Put NT in the baseline (and report PR/n) so the
      increment is length-clean.

Reports |corr| of PR with resultant / EDIS / length (the shadow diagnostics) and a ladder of
increment tests: shape over {resultant} / {resultant+EDIS} / {resultant+EDIS+len} / OURS / OURS+EDIS+len.

Pass BOTH files for one config: --cloud pb_X_cloud.npz --coh pb_X_coh.npz.
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


def abscorr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return float("nan")
    return abs(float(np.corrcoef(a[m], b[m])[0, 1]))


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
    return PR, PR / n, top1                                        # PR, PR/n (length-normalized), top1


def oof(cols, y, grp, folds=5):
    X = np.column_stack(cols); s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(folds).split(X, y, grp):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return bdir(auroc(s, y))


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
    nmatch = min(len(RC), len(SC))

    GEO, RES, PR, PRN, T1, EDS, LEN, UNC, Y, G = [], [], [], [], [], [], [], [], [], []
    skip = 0
    for i in range(nmatch):
        if RC[i] is None or int(gesc[i]) != int(gesh[i]):
            skip += 1; continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]
        sc = np.asarray(SC[i], float); sg = np.asarray(SG[i], float) if SG is not None else None
        rng = np.asarray(SRh[i], int); rngc = np.asarray(SRc[i], int)
        if rng.shape[0] != rngc.shape[0]:
            skip += 1; continue
        k = int(gesh[i]); correct = (k < 0); T = rng.shape[0]; a0 = int(rng[0, 0]); a0c = int(rngc[0, 0])
        ud = np.asarray(UD[i], float)
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
            GEO.append(geo); RES.append(sc[j, li, fi]); PR.append(sf[0]); PRN.append(sf[1]); T1.append(sf[2])
            EDS.append(edis(ud[lo:hi])); LEN.append(float(hi - lo)); UNC.append(unc_feats(ud[lo:hi]))
            Y.append(lab); G.append(i)
    GEO = np.asarray(GEO); RES = np.asarray(RES); PR = np.asarray(PR); PRN = np.asarray(PRN)
    T1 = np.asarray(T1); EDS = np.asarray(EDS); LEN = np.asarray(LEN); UNC = np.asarray(UNC, float)
    Y = np.asarray(Y, int); G = np.asarray(G, int)
    gR, gG = -RES, -GEO; shape = [PR, T1]; ucols = [UNC[:, c] for c in range(UNC.shape[1])]

    print(f"cloud={args.cloud} | matched steps {len(Y)} first-error {int(Y.sum())} | skipped {skip}")
    print(f"\nshadow diagnostics |corr|:  PR-resultant {abscorr(PR, RES):.3f}  PR-EDIS {abscorr(PR, EDS):.3f}  "
          f"PR-length {abscorr(PR, LEN):.3f}   PR/n-length {abscorr(PRN, LEN):.3f}  PR/n-EDIS {abscorr(PRN, EDS):.3f}")
    print(f"\n{'baseline -> + shape(PR,top1)':44s} {'base':>6s} {'+shape':>7s} {'delta':>7s}")
    ladder = [
        ("resultant", [gR]),
        ("resultant + EDIS", [gR, EDS]),
        ("resultant + EDIS + length", [gR, EDS, LEN]),
        ("OURS (geom + entropy battery)", [gG] + ucols),
        ("OURS + EDIS + length", [gG] + ucols + [EDS, LEN]),
    ]
    for nm, base in ladder:
        b = oof(base, Y, G); bs = oof(base + shape, Y, G)
        print(f"  {nm:44s} {b:6.3f} {bs:7.3f} {bs - b:+7.3f}")
    # length-normalized shape variant (PR/n) over the strongest controlled baseline
    strong = [gG] + ucols + [EDS, LEN]
    print(f"\n  PR/n (length-normalized) over OURS+EDIS+len: "
          f"{oof(strong, Y, G):.3f} -> {oof(strong + [PRN, T1], Y, G):.3f}")

    # HEADLINE SELF-CHECK: does the GEOMETRY modality (pooled-norm) survive the SAME strict baseline
    # that just killed shape? (the shape lesson: an increment over a weak baseline can be leakage)
    print(f"\n  --- headline check: GEOMETRY (pooled-norm) under strict baselines ---")
    print(f"  {'baseline -> + geometry':44s} {'base':>6s} {'+geo':>7s} {'delta':>7s}")
    for nm, base in [("entropy battery (no geom)", ucols),
                     ("entropy battery + EDIS", ucols + [EDS]),
                     ("entropy battery + EDIS + length", ucols + [EDS, LEN])]:
        b = oof(base, Y, G); bg = oof(base + [gG], Y, G)
        print(f"  {nm:44s} {b:6.3f} {bg:7.3f} {bg - b:+7.3f}")
    print(f"  |corr| geom-EDIS {abscorr(GEO, EDS):.3f}  geom-length {abscorr(GEO, LEN):.3f}")
    print("\nread: DECISIVE = delta of shape over 'resultant + EDIS + length' and over 'OURS + EDIS + length'. "
          "If > 0 there, PR carries error signal that is NEITHER an EDIS shadow NOR token count -- a real "
          "second geometric axis. Check the shadow |corr|: high PR-EDIS or PR-length would warn; if PR/n "
          "(length-normalized) keeps the increment, length is excluded directly. If delta collapses to ~0 once "
          "EDIS+length are in the baseline, the earlier +0.055 was EDIS/length leakage, not shape.")


if __name__ == "__main__":
    main()
