"""Improve the geometric step-level signal to comprehensively beat step-EDIS.

Single static resultant beats step-EDIS only on gsm8k; on hard configs step-EDIS wins because it
uses WITHIN-STEP entropy dynamics while resultant is one pooled scalar. Here we fuse the full
GEOMETRIC step-feature family (cloud spectral + concentration + geom anomaly + attention, all
already extracted, no uncertainty) via a leak-free GroupKFold logistic, and compare to step-EDIS
and a geometry+EDIS fusion, for first-error localization. Goal: geometric ensemble > step-EDIS on
ALL configs (pooled + length-bucket), i.e. comprehensive superiority at the step level -- EDIS's
own stated future-work gap.

Needs coh.npz: stepcloud + stepgeom (+ stepattn) + tok_U_D (for step-EDIS) + step_token_ranges +
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
    m = np.isfinite(s); s, y, nt = s[m], y[m], nt[m]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    gn = [str(x) for x in z["geom_feature_names"]] if "geom_feature_names" in z.files else []
    an = [str(x) for x in z["attn_names"]] if "attn_names" in z.files else []
    lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    SC = z["stepcloud"]; SG = z["stepgeom"] if "stepgeom" in z.files else None
    SA = z["stepattn"] if "stepattn" in z.files else None
    SR = z["step_token_ranges"]; ges = z["gold_error_step"].astype(int); UD = z["tok_U_D"]
    # geometric family (NO uncertainty): all cloud + geom + attn features
    GEO = [("c", m) for m in cn] + [("g", m) for m in gn] + [("a", m) for m in an]
    ri = cn.index("resultant")

    FEAT, EDS, RES, Y, NT, G = [], [], [], [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); sg = np.asarray(SG[i], float) if SG is not None else None
        sa = np.asarray(SA[i], float) if SA is not None else None
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
            row = []
            for src, m in GEO:
                if src == "c" and m in cn:
                    row.append(sc[j, li, cn.index(m)])
                elif src == "g" and sg is not None and m in gn:
                    row.append(sg[j, li, gn.index(m)])
                elif src == "a" and sa is not None and m in an:
                    row.append(sa[j, li, an.index(m)] if sa.ndim == 3 else sa[j, an.index(m)])
                else:
                    row.append(np.nan)
            FEAT.append(row); EDS.append(edis(ud[lo:hi])); RES.append(sc[j, li, ri])
            Y.append(lab); NT.append(int(rng[j, 1] - rng[j, 0] + 1)); G.append(i)
    FEAT = np.asarray(FEAT, float); EDS = np.asarray(EDS); RES = np.asarray(RES)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float); G = np.asarray(G, int)
    for c in range(FEAT.shape[1]):
        col = FEAT[:, c]; col[~np.isfinite(col)] = np.nanmean(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0

    geo_oof = oof(FEAT, Y, G)
    fus_oof = oof(np.c_[FEAT, EDS], Y, G)
    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} first-error {int(Y.sum())} | geom feats {FEAT.shape[1]}")
    print(f"\n{'signal':28s} {'AUROC':>7s} {'bucket':>7s}")
    for nm, v in [("resultant (single)", -RES), ("step-EDIS (entropy-dyn)", EDS),
                  ("GEOM ENSEMBLE (ours)", geo_oof), ("GEOM + step-EDIS", fus_oof)]:
        print(f"  {nm:28s} {bdir(auroc(v, Y)):7.3f} {bucket(v, Y, NT):7.3f}")
    print("\nread: RESULT (observed): the logistic ensemble beats step-EDIS on gsm8k/math but LOSES on "
          "omnimath/olympiad (pooled AND bucket). A linear logistic stack washes out each feature's peak and "
          "does NOT comprehensively beat entropy dynamics -- see step_geom_instability.py for the principled "
          "MULTIPLICATIVE (EDIS-form) geometry score that preserves each component and targets the hard "
          "configs. GEOM+EDIS is the (logistic) fusion ceiling.")


if __name__ == "__main__":
    main()
