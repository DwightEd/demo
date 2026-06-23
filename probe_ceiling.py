"""How much does a probe on the FULL step representation beat the kappa scalar?
Exp-pool per-token respcloud (JL-256) within each step (same w_n=(n-1)/(L-1) as 2601.02170),
train a leak-free linear probe on the 256-d step vector vs the kappa scalar. Step-level (gold_error_step),
+ length bucket. Quantifies the orientation/content kappa discards = the competitor's real method edge."""
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
    csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    V, KA, NT, Y, G = [], [], [], [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        H = np.asarray(RC[i], np.float64)[:, li, :]; rng = np.asarray(SR[i], int); a0 = int(rng[0, 0])
        k = int(ges[i]); correct = k < 0; T = rng.shape[0]
        nrm = np.linalg.norm(H, axis=1); U = np.zeros_like(H); ok = nrm > 1e-9; U[ok] = H[ok] / nrm[ok, None]
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(H), int(rng[j, 1]) - a0 + 1)
            if hi - lo < 2:
                continue
            L = hi - lo; w = np.exp(np.arange(L) / (L - 1)); w /= w.sum()
            V.append(w @ H[lo:hi]); KA.append(np.linalg.norm(w @ U[lo:hi]))
            NT.append(L); Y.append(lab); G.append(i)
    V = np.asarray(V); KA = np.asarray(KA); NT = np.asarray(NT, float); Y = np.asarray(Y, int); G = np.asarray(G, int)
    s_k = -KA
    s_v = oof(V, Y, G)
    s_kv = oof(np.column_stack([V, KA]), Y, G)
    print(f"{args.npz} | L{args.layer} | steps {len(Y)} err {int(Y.sum())} | repr dim {V.shape[1]}")
    for nm, s in [("kappa scalar", s_k), ("probe on repr (256d)", s_v), ("repr + kappa", s_kv)]:
        print(f"  {nm:22s} AUROC {bdir(auroc(s, Y)):.3f}  bkt(len) {bucket(s, Y, NT):.3f}")


if __name__ == "__main__":
    main()
