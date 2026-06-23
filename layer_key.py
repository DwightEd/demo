"""Per-layer kappa AUROC + leave-one-layer-out, to see which layers carry the multi-layer signal."""
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


def bdir(a):
    return max(a, 1 - a) if np.isfinite(a) else a


def oof(X, y, g):
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return bdir(auroc(s, y))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]; lyu = [int(x) for x in z["layers_used"]]
    SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int); ri = cn.index("resultant")
    R, Y, G = [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i]); correct = k < 0; T = rng.shape[0]
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            R.append(sc[j, :, ri]); Y.append(lab); G.append(i)
    R = np.asarray(R, float); Y = np.asarray(Y, int); G = np.asarray(G, int)
    col = np.nanmean(np.where(np.isfinite(R), R, np.nan), 0); R = np.where(np.isfinite(R), R, col)
    full = oof(R, Y, G)
    print(f"{args.npz} | layers {lyu} | full-stack {full:.3f}")
    print(f"{'layer':>6s} {'solo':>6s} {'LOO':>6s}")
    for li, L in enumerate(lyu):
        solo = bdir(auroc(-R[:, li], Y))
        loo = oof(np.delete(R, li, 1), Y, G)
        print(f"{L:6d} {solo:6.3f} {loo:6.3f}")


if __name__ == "__main__":
    main()
