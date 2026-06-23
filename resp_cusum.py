"""Response-level: CUSUM of EMA-kappa LEVEL deficit (segmentation-free, online) vs min/mean EMA-kappa, + length bucket. respcloud."""
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


def bucket(s, y, nt, nb=5):
    m = np.isfinite(s) & np.isfinite(nt); s, y, nt = s[m], y[m], nt[m]
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1); num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm])); ne = int(y[mm].sum()); ng = int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def ema_kappa(U, ok, a):
    N, d = U.shape; out = np.full(N, np.nan); es = np.zeros(d); ew = 0.0
    for t in range(N):
        if ok[t]:
            es = a * es + U[t]; ew = a * ew + 1.0
        if ew > 0:
            out[t] = np.linalg.norm(es / ew)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--ema", type=float, default=0.9); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    seqs, NT, Y = [], [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        H = np.asarray(RC[i], np.float64)[:, li, :]; nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
        U = np.zeros_like(H); U[ok] = H[ok] / nrm[ok, None]
        ek = ema_kappa(U, ok, args.ema)
        if not np.isfinite(ek).any():
            continue
        seqs.append(ek); NT.append(int(np.isfinite(ek).sum())); Y.append(int(int(ges[i]) >= 0))
    b = float(np.nanmedian(np.concatenate(seqs)))
    CP, MN, ME = [], [], []
    for ek in seqs:
        x = ek[np.isfinite(ek)]; c = 0.0; peak = 0.0
        for v in x:
            c = max(0.0, c + (b - v)); peak = max(peak, c)
        CP.append(peak); MN.append(1 - float(x.min())); ME.append(1 - float(x.mean()))
    NT = np.asarray(NT, float); Y = np.asarray(Y, int)
    print(f"{args.npz} | L{args.layer} ema{args.ema} | responses {len(Y)} err {int(Y.sum())} | b={b:.3f}")
    for nm, v in [("CUSUM-peak", CP), ("min EMA-kappa", MN), ("mean EMA-kappa", ME)]:
        v = np.asarray(v, float)
        print(f"  {nm:16s} AUROC {bdir(auroc(v, Y)):.3f}  bkt(len) {bucket(v, Y, NT):.3f}")


if __name__ == "__main__":
    main()
