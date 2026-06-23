"""Absolute kappa vs within-chain DROP (kappa - prior-step mean); and does chain baseline kappa predict error chains?"""
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]; lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    fi = cn.index("resultant"); SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    ABS, REL, NT, Y = [], [], [], []
    base, cerr = [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i]); correct = k < 0; T = rng.shape[0]
        kap = sc[:T, li, fi]
        prefix = kap[:k] if not correct else kap                 # healthy steps (before error)
        if np.isfinite(prefix).any():
            base.append(float(np.nanmean(prefix))); cerr.append(int(not correct))
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            rel = kap[j] - np.nanmean(kap[:j]) if j >= 1 and np.isfinite(kap[:j]).any() else np.nan
            ABS.append(kap[j]); REL.append(rel); NT.append(int(rng[j, 1] - rng[j, 0] + 1)); Y.append(lab)
    ABS = np.asarray(ABS); REL = np.asarray(REL); NT = np.asarray(NT, float); Y = np.asarray(Y, int)
    base = np.asarray(base); cerr = np.asarray(cerr, int)
    print(f"{args.npz} | L{args.layer} | steps {len(Y)} err {int(Y.sum())} | chains {len(base)} err-chains {int(cerr.sum())}")
    print(f"  step:  abs-kappa AUROC {bdir(auroc(-ABS, Y)):.3f} (bkt {bucket(-ABS, Y, NT):.3f}) | "
          f"drop(rel) AUROC {bdir(auroc(-REL, Y)):.3f} (bkt {bucket(-REL, Y, NT):.3f})")
    print(f"  chain: baseline-kappa -> error-chain AUROC {bdir(auroc(-base, cerr)):.3f}  "
          f"(>0.5 = low-baseline chains ARE more error-prone; ~0.5 = consistently-low NOT error-prone)")


if __name__ == "__main__":
    main()
