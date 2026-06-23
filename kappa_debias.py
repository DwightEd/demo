"""Bias-corrected concentration kappa_corr=(n_eff*R^2-1)/(n_eff-1) removes finite-sample short-step inflation; AUROC raw vs bucket vs debiased."""
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


def bucket(s, y, nt, nb=6):
    m = np.isfinite(s) & np.isfinite(nt); s, y, nt = s[m], y[m], nt[m]
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1); num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm])); ne = int(y[mm].sum()); ng = int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def abscorr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return abs(float(np.corrcoef(a[m], b[m])[0, 1])) if m.sum() > 5 and a[m].std() > 0 and b[m].std() > 0 else float("nan")


def n_eff(n):
    if n < 2:
        return float(n)
    w = np.exp(np.arange(n) / (n - 1))
    return float(w.sum() ** 2 / (w ** 2).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]; lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    fi = cn.index("resultant"); SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    R, NT, Y = [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i]); correct = k < 0
        for j in range(rng.shape[0]):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            R.append(float(sc[j, li, fi])); NT.append(int(rng[j, 1] - rng[j, 0] + 1)); Y.append(lab)
    R = np.asarray(R); NT = np.asarray(NT, float); Y = np.asarray(Y, int)
    ne = np.array([n_eff(int(n)) for n in NT])
    kc = (ne * R ** 2 - 1) / (ne - 1)
    print(f"{args.npz} | L{args.layer} | steps {len(Y)} err {int(Y.sum())}")
    print(f"  raw resultant : AUROC {bdir(auroc(-R, Y)):.3f} | bucket {bucket(-R, Y, NT):.3f} | |corr|len {abscorr(R, NT):.3f}")
    print(f"  debiased kappa: AUROC {bdir(auroc(-kc, Y)):.3f} | bucket {bucket(-kc, Y, NT):.3f} | |corr|len {abscorr(kc, NT):.3f}")


if __name__ == "__main__":
    main()
