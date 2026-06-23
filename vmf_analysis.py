"""Segmentation-free kappa constructions vs semantic-step ceiling, on respcloud tokens (_cloud.npz)."""
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


def global_axis(TH, li):
    """top shared anisotropy axis via one-pass covariance of valid unit token vectors (O(d^2) mem)."""
    s = None; ss = None; n = 0
    for i in range(len(TH)):
        if TH[i] is None:
            continue
        H = np.asarray(TH[i], np.float64)[:, li, :]
        nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
        U = H[ok] / nrm[ok, None]
        if U.shape[0] == 0:
            continue
        s = U.sum(0) if s is None else s + U.sum(0)
        ss = U.T @ U if ss is None else ss + U.T @ U
        n += U.shape[0]
    mu = s / n; C = ss / n - np.outer(mu, mu)
    return np.linalg.eigh(C)[1][:, -1]


def per_token_kappa(U, ok, wins, ema_a, axis):
    """running-mean kappa for each construction at every token; returns dict name->(N,) array.
    kappa=||m||, kappa_noax=||m - (m.a)a||, m = running mean of unit vectors (O(Nd) total)."""
    N, d = U.shape
    out = {"causal": np.full(N, np.nan), "causal_nx": np.full(N, np.nan),
           "ema": np.full(N, np.nan), "ema_nx": np.full(N, np.nan)}
    for w in wins:
        out[f"w{w}"] = np.full(N, np.nan); out[f"w{w}_nx"] = np.full(N, np.nan)

    def km(m, cnt):
        if cnt < 2:
            return np.nan, np.nan
        mm = m / cnt
        return float(np.linalg.norm(mm)), float(np.linalg.norm(mm - (mm @ axis) * axis))

    cs = np.zeros(d); cc = 0                                   # causal running sum
    es = np.zeros(d); ew = 0.0                                 # ema running sum
    wbuf = {w: (np.zeros(d), 0) for w in wins}                 # fixed-window running sum + count
    for t in range(N):
        v = U[t] if ok[t] else None
        if v is not None:
            cs += v; cc += 1
            es = ema_a * es + v; ew = ema_a * ew + 1.0
        out["causal"][t], out["causal_nx"][t] = km(cs, cc)
        out["ema"][t], out["ema_nx"][t] = km(es, ew) if ew > 0 else (np.nan, np.nan)
        for w in wins:
            ws, wn = wbuf[w]
            if v is not None:
                ws += v; wn += 1
            if t - w >= 0 and ok[t - w]:                       # drop the token leaving the window
                ws -= U[t - w]; wn -= 1
            wbuf[w] = (ws, wn)
            out[f"w{w}"][t], out[f"w{w}_nx"][t] = km(ws, wn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--wins", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--ema", type=float, default=0.9)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)
    axis = global_axis(RC, li)

    methods = ["sem", "sem_nx", "causal", "causal_nx", "ema", "ema_nx"]
    for w in args.wins:
        methods += [f"w{w}", f"w{w}_nx"]
    S = {m: [] for m in methods}; Y = []

    for i in range(len(RC)):
        if RC[i] is None:
            continue
        H = np.asarray(RC[i], np.float64)[:, li, :]
        nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
        U = np.zeros_like(H); U[ok] = H[ok] / nrm[ok, None]
        rng = np.asarray(SR[i], int); a0 = int(rng[0, 0]); T = rng.shape[0]
        k = int(ges[i]); correct = (k < 0)
        ts = np.full(H.shape[0], -1, int)
        for j in range(T):
            lo = max(0, rng[j, 0] - a0); hi = min(H.shape[0] - 1, rng[j, 1] - a0)
            if lo <= hi:
                ts[lo:hi + 1] = j
        pt = per_token_kappa(U, ok, args.wins, args.ema, axis)

        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            idx = np.where(ts == j)[0]
            if idx.size < 2 or ok[idx].sum() < 2:
                continue
            Y.append(lab)
            mu = U[idx][ok[idx]].mean(0)                       # semantic-step pooled (validated)
            S["sem"].append(float(np.linalg.norm(mu)))
            S["sem_nx"].append(float(np.linalg.norm(mu - (mu @ axis) * axis)))
            for m in methods[2:]:                              # token-level: deepest dip in the step
                col = pt[m][idx]
                S[m].append(float(np.nanmin(col)) if np.isfinite(col).any() else np.nan)

    Y = np.asarray(Y, int)
    ceil = bdir(auroc(-np.asarray(S["sem"], float), Y))
    print(f"{args.npz} | L{args.layer} | steps {len(Y)} | err {int(Y.sum())} | ceiling(sem) {ceil:.3f}")
    print(f"{'method':12s} {'AUROC':>7s} {'vs_ceil':>8s}")
    for m in methods:
        a = bdir(auroc(-np.asarray(S[m], float), Y))
        print(f"{m:12s} {a:7.3f} {a - ceil:+8.3f}")


if __name__ == "__main__":
    main()