"""Bayes-optimal separability ceiling of the kappa (resultant) signal under the vMF view.
Estimates P(kappa|correct) vs P(kappa|error) and the max achievable AUROC from kappa alone,
(a) raw marginal and (b) within length-strata (controls chain/length confound, the honest ceiling).
The Bayes-optimal 1-D score is the likelihood ratio p1/p0 (handles non-monotone crossings);
also reports the monotone (-kappa) AUROC. Needs _coh.npz: stepcloud(resultant) + step_token_ranges
+ gold_error_step + layers_used + cloud_feature_names.
"""
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


def kde(x, grid, bw):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.full(len(grid), np.nan)
    d = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * d * d).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))


def lr_auroc(k, y):
    """Bayes-optimal AUROC via KDE likelihood ratio p(k|err)/p(k|cor) as the score."""
    k = np.asarray(k, float); y = np.asarray(y, int)
    m = np.isfinite(k); k, y = k[m], y[m]
    if y.sum() < 5 or (y == 0).sum() < 5:
        return float("nan")
    bw = 1.06 * k.std() * len(k) ** (-1 / 5) + 1e-9          # Silverman
    p0 = kde(k[y == 0], k, bw); p1 = kde(k[y == 1], k, bw)
    return auroc((p1 + 1e-12) / (p0 + 1e-12), y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--nbk", type=int, default=6)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]
    lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    fi = cn.index("resultant")
    SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)

    K, Y, NT = [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0)
        for j in range(rng.shape[0]):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            K.append(float(sc[j, li, fi])); Y.append(lab); NT.append(int(rng[j, 1] - rng[j, 0] + 1))
    K = np.asarray(K); Y = np.asarray(Y, int); NT = np.asarray(NT, float)

    a_mono = bdir(auroc(-K, Y)); a_lr = lr_auroc(K, Y)
    print(f"{args.npz} | L{args.layer} | steps {len(Y)} | err {int(Y.sum())} | "
          f"kappa cor {K[Y==0].mean():.3f} err {K[Y==1].mean():.3f}")
    print(f"  raw marginal:   monotone(-kappa) {a_mono:.3f} | Bayes-LR {a_lr:.3f}")

    e = np.quantile(NT, np.linspace(0, 1, args.nbk + 1)); e[-1] += 1
    b = np.clip(np.digitize(NT, e[1:-1]), 0, args.nbk - 1)
    num = den = mono_n = 0.0
    for bb in range(args.nbk):
        mm = b == bb; ne, ng = int(Y[mm].sum()), int((Y[mm] == 0).sum())
        if ne < 5 or ng < 5:
            continue
        al = lr_auroc(K[mm], Y[mm])
        if np.isfinite(al):
            num += al * ne * ng; den += ne * ng; mono_n += bdir(auroc(-K[mm], Y[mm])) * ne * ng
    if den:
        print(f"  length-controlled (honest ceiling): Bayes-LR {num/den:.3f} | monotone {mono_n/den:.3f}")
    else:
        print("  length-controlled: insufficient per-bucket counts")


if __name__ == "__main__":
    main()