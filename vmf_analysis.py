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


def aggregate(col):
    c = col[np.isfinite(col)]
    if len(c) < 2:
        return dict(mean=np.nan, min=np.nan, slope=0.0, std=0.0, drop=0.0)
    t = np.arange(len(c))
    return dict(mean=float(c.mean()), min=float(c.min()), slope=float(np.polyfit(t, c, 1)[0]),
                std=float(c.std()), drop=float(c.ptp()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--wins", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--ema", type=float, default=0.9)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    axis = global_axis(RC, li)

    cons = ["causal", "causal_nx", "ema", "ema_nx"] + [f"w{w}{s}" for w in args.wins for s in ("", "_nx")]
    ops = ["mean", "min", "slope", "std", "drop"]; sgn = dict(mean=-1, min=-1, slope=-1, std=1, drop=1)
    Sstep = {(m, op): [] for m in cons for op in ops}; Stok = {m: [] for m in cons}
    sem, sem_nx, Y, Ytok = [], [], [], []

    for i in range(len(RC)):
        if RC[i] is None:
            continue
        H = np.asarray(RC[i], np.float64)[:, li, :]
        nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
        U = np.zeros_like(H); U[ok] = H[ok] / nrm[ok, None]
        rng = np.asarray(SR[i], int); a0 = int(rng[0, 0]); T = rng.shape[0]; k = int(ges[i]); correct = (k < 0)
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
            Y.append(lab); mu = U[idx][ok[idx]].mean(0)
            sem.append(float(np.linalg.norm(mu))); sem_nx.append(float(np.linalg.norm(mu - (mu @ axis) * axis)))
            for m in cons:
                ag = aggregate(pt[m][idx])
                for op in ops:
                    Sstep[(m, op)].append(ag[op])
                Stok[m].extend(pt[m][idx].tolist())
            Ytok.extend([lab] * idx.size)

    Y = np.asarray(Y, int); Ytok = np.asarray(Ytok, int)
    cl = bdir(auroc(-np.asarray(sem), Y)); clx = bdir(auroc(-np.asarray(sem_nx), Y))
    print(f"{args.npz} | L{args.layer} | steps {len(Y)} err {int(Y.sum())} | ceil sem {cl:.3f} sem_nx {clx:.3f}")
    print(f"{'construction':12s} " + " ".join(f"{op:>6s}" for op in ops) + "   tok")
    for m in cons:
        row = " ".join(f"{bdir(auroc(sgn[op] * np.asarray(Sstep[(m, op)], float), Y)):6.3f}" for op in ops)
        print(f"{m:12s} {row}  {bdir(auroc(-np.asarray(Stok[m], float), Ytok)):.3f}")


if __name__ == "__main__":
    main()