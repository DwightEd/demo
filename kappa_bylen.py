"""AUROC of geometry(kappa) vs entropy(EDIS) within step-length terciles: is geometry specifically weak on LONG (composite) steps?"""
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


def n_eff(n):
    if n < 2:
        return float(n)
    w = np.exp(np.arange(int(n)) / (n - 1))
    return float(w.sum() ** 2 / (w ** 2).sum())


def edis(H, w=8, tb=1.36, tr=1.33):
    H = np.asarray(H, float); H = H[np.isfinite(H)]
    if len(H) < 3:
        return 0.0
    ww = min(w, max(2, len(H) // 2))
    burst = sum(1 for t in range(len(H) - ww) if H[t + ww] - H[t] > tb) if len(H) > ww else 0
    reb = 0; rmin = H[0]
    for t in range(1, len(H)):
        if H[t] - rmin > tr:
            reb += 1
        rmin = min(rmin, H[t])
    return 0.5 * (burst + reb) * (1.0 + float(H.var()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]; lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    fi = cn.index("resultant"); SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int); UD = z["tok_U_D"]
    KA, ED, NT, Y = [], [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i]); correct = k < 0; T = rng.shape[0]; a0 = int(rng[0, 0]); ud = np.asarray(UD[i], float)
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
            KA.append(sc[j, li, fi]); ED.append(edis(ud[lo:hi])); NT.append(hi - lo); Y.append(lab)
    KA = np.asarray(KA); ED = np.asarray(ED); NT = np.asarray(NT, float); Y = np.asarray(Y, int)
    ne = np.array([n_eff(n) for n in NT]); KC = (ne * KA ** 2 - 1) / (ne - 1)   # bias-corrected kappa (removes #3)
    q = np.quantile(NT, [1 / 3, 2 / 3]); st = np.digitize(NT, q)
    print(f"{args.npz} | L{args.layer} | steps {len(Y)} err {int(Y.sum())}")
    print(f"  {'len stratum':16s} {'n':>5s} {'err':>4s} {'medlen':>7s} {'kappa':>6s} {'kappa_corr':>10s} {'EDIS':>6s}")
    for s, nm in enumerate(["short", "mid", "long"]):
        m = st == s
        print(f"  {nm:16s} {int(m.sum()):>5d} {int(Y[m].sum()):>4d} {int(np.median(NT[m])):>7d} "
              f"{bdir(auroc(-KA[m], Y[m])):>6.3f} {bdir(auroc(-KC[m], Y[m])):>10.3f} {bdir(auroc(ED[m], Y[m])):>6.3f}")


if __name__ == "__main__":
    main()
