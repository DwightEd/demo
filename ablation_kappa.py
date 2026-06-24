"""Ablation: single resultant vs best-single vs multilayer kappa-family; kappa vs kappa^2; exp vs uniform pooling.
Step-level (gold_error_step), GroupKFold OOF + length bucket. Answers: does multilayer earn its keep, is one metric enough, does squaring help, which pooling."""
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


def bucket(s, y, nt, nb=5):
    m = np.isfinite(s) & np.isfinite(nt); s, y, nt = s[m], y[m], nt[m]
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1); num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm])); ne = int(y[mm].sum()); ng = int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def oof(X, y, g):
    X = np.atleast_2d(X) if X.ndim > 1 else X.reshape(-1, 1)
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]; lyu = [int(x) for x in z["layers_used"]]
    feats = [f for f in ("resultant", "resultant_unif", "resultant_bulk", "coherence") if f in cn]
    fi = [cn.index(f) for f in feats]; li14 = lyu.index(args.layer)
    SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    M, NT, Y, G = [], [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i]); T = rng.shape[0]
        for j in range(T):
            if k < 0 or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            M.append(sc[j][:, fi]); NT.append(int(rng[j, 1] - rng[j, 0] + 1)); Y.append(lab); G.append(i)
    M = np.asarray(M); NT = np.asarray(NT, float); Y = np.asarray(Y, int); G = np.asarray(G, int)   # M: (n, L, nf)
    L, nf = M.shape[1], M.shape[2]
    res = feats.index("resultant"); unif = feats.index("resultant_unif") if "resultant_unif" in feats else res

    def rep(name, s):
        print(f"  {name:34s} AUROC {bdir(auroc(s, Y)):.3f}  bkt {bucket(s, Y, NT):.3f}")

    print(f"{args.npz} | steps {len(Y)} err {int(Y.sum())} | layers {L} feats {feats}")
    # single best (raw feature = monotone, no fitting needed)
    best = (-1, None)
    for li in range(L):
        for f in range(nf):
            a = bdir(auroc(M[:, li, f], Y))
            if a > best[0]:
                best = (a, (lyu[li], feats[f]))
    rep(f"L{args.layer} resultant (single)", -M[:, li14, res])
    print(f"  best single (raw) = {best[1]} : AUROC {best[0]:.3f}")
    rep("multilayer resultant only", oof(M[:, :, res], Y, G))
    rep("multilayer kappa-family", oof(M.reshape(len(Y), L * nf), Y, G))
    rep("multilayer family, R^2", oof((M ** 2).reshape(len(Y), L * nf), Y, G))
    rep(f"pool exp (resultant) L{args.layer}", -M[:, li14, res])
    rep(f"pool unif (resultant_unif) L{args.layer}", -M[:, li14, unif])
    rep("multilayer exp (resultant)", oof(M[:, :, res], Y, G))
    rep("multilayer unif (resultant_unif)", oof(M[:, :, unif], Y, G))


if __name__ == "__main__":
    main()
