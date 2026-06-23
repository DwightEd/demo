"""Within the LOW-kappa subgroup only: does dkappa add over [kappa+pos+len]? (tests 'drop stratifies risk within low-kappa')."""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


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


def oof(cols, y, g):
    X = np.column_stack(cols); s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); ap.add_argument("--w", type=int, default=2); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]; lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    fi = cn.index("resultant"); SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    KA, DK, POS, PF, NT, Y, G = [], [], [], [], [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i]); correct = k < 0; T = rng.shape[0]
        kap = sc[:T, li, fi]
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            prior = kap[max(0, j - args.w):j]
            dk = kap[j] - np.nanmean(prior) if j >= 1 and np.isfinite(prior).any() else np.nan
            KA.append(kap[j]); DK.append(dk); POS.append(j); PF.append(j / (T - 1) if T > 1 else 0.0)
            NT.append(float(rng[j, 1] - rng[j, 0] + 1)); Y.append(lab); G.append(i)
    KA = np.asarray(KA); DK = np.asarray(DK); POS = np.asarray(POS, float); PF = np.asarray(PF)
    NT = np.asarray(NT, float); Y = np.asarray(Y, int); G = np.asarray(G, int)
    sub = np.isfinite(DK) & (KA <= np.nanquantile(KA, 1 / 3))
    ka, dk, po, pf, nt, y, g = KA[sub], DK[sub], POS[sub], PF[sub], NT[sub], Y[sub], G[sub]
    base = [-ka, po, pf, nt]
    sb = oof(base, y, g); sf = oof(base + [-dk], y, g)
    ab, af = auroc(sb, y), auroc(sf, y)
    ch = np.unique(g); rng = np.random.default_rng(0); ci = {c: np.where(g == c)[0] for c in ch}; ds = []
    for _ in range(200):
        idx = np.concatenate([ci[c] for c in rng.choice(ch, len(ch), replace=True)])
        ds.append(auroc(sf[idx], y[idx]) - auroc(sb[idx], y[idx]))
    cl, cu = np.percentile(ds, [2.5, 97.5])
    print(f"{args.npz} | low-kappa steps {len(y)} err {int(y.sum())} | dkappa over [kappa+pos+len]: {ab:.3f} -> {af:.3f}  (+{af-ab:.3f}, 95% CI [{cl:+.3f}, {cu:+.3f}])")


if __name__ == "__main__":
    main()
