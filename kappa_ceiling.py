"""All-layer kappa ceiling: concentration family x every layer, leak-free GroupKFold logistic/GBM vs single-L14 resultant."""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
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


def bdir(a):
    return max(a, 1 - a) if np.isfinite(a) else a


def bucket(s, y, nt, nb=5):
    m = np.isfinite(s) & np.isfinite(nt); s, y, nt = s[m], y[m], nt[m]
    if len(s) < 10:
        return float("nan")
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1); num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm])); ne = int(y[mm].sum()); ng = int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def oof(X, y, g, clf_fn):
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = clf_fn(); clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]; lyu = [int(x) for x in z["layers_used"]]
    SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    kfam = [c for c in ("resultant", "resultant_bulk", "resultant_unif", "coherence") if c in cn]
    idxs = [cn.index(c) for c in kfam]; ri = cn.index("resultant"); l14 = lyu.index(14)
    X, x14, NT, Y, G = [], [], [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i]); correct = k < 0; T = rng.shape[0]
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            X.append(sc[j][:, idxs].ravel()); x14.append(sc[j, l14, ri])
            NT.append(int(rng[j, 1] - rng[j, 0] + 1)); Y.append(lab); G.append(i)
    X = np.asarray(X, float); x14 = np.asarray(x14); NT = np.asarray(NT, float)
    Y = np.asarray(Y, int); G = np.asarray(G, int)
    col = np.nanmean(np.where(np.isfinite(X), X, np.nan), 0); Ximp = np.where(np.isfinite(X), X, col)
    logi = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    gbm = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05)
    print(f"{args.npz} | layers {lyu} | kappa-family {kfam} | feat {X.shape[1]} | steps {len(Y)} err {int(Y.sum())}")
    print(f"single L14 resultant   {bdir(auroc(-x14, Y)):.3f}")
    sl = oof(Ximp, Y, G, logi); print(f"all-layer logistic     {bdir(auroc(sl, Y)):.3f}  (bkt {bucket(sl, Y, NT):.3f})")
    sg = oof(X, Y, G, gbm); print(f"all-layer GBM          {bdir(auroc(sg, Y)):.3f}  (bkt {bucket(sg, Y, NT):.3f})")


if __name__ == "__main__":
    main()
