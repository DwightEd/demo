"""Within-chain first-error localization: rank steps by deepest-dip kappa vs pooled kappa vs length."""
from __future__ import annotations
import argparse
import numpy as np


def step_k(U, w):
    m = U.shape[0]
    if m < 2:
        return np.nan, np.nan
    pooled = float(np.linalg.norm(U.mean(0)))
    if m <= w:
        return pooled, pooled
    mins = min(float(np.linalg.norm(U[t - w + 1:t + 1].mean(0))) for t in range(w - 1, m))
    return pooled, min(mins, pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--w", type=int, default=16)
    args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)

    rr = {"pooled": [], "min": [], "length": []}; top1 = {"pooled": 0, "min": 0, "length": 0}; n = 0
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        k = int(ges[i])
        if k < 0:
            continue
        H = np.asarray(RC[i], np.float64)[:, li, :]
        nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
        U = np.zeros_like(H); U[ok] = H[ok] / nrm[ok, None]
        rng = np.asarray(SR[i], int); a0 = int(rng[0, 0]); T = rng.shape[0]
        pooled = np.full(T, np.nan); mink = np.full(T, np.nan); length = np.zeros(T)
        for j in range(T):
            lo = max(0, rng[j, 0] - a0); hi = min(H.shape[0], rng[j, 1] - a0 + 1)
            sel = U[lo:hi][ok[lo:hi]]
            length[j] = sel.shape[0]
            if sel.shape[0] >= 2:
                pooled[j], mink[j] = step_k(sel, args.w)
        if k >= T or not np.isfinite(mink[k]):
            continue
        n += 1
        for name, sc in (("pooled", pooled), ("min", mink), ("length", -length)):
            order = np.argsort(np.where(np.isfinite(sc), sc, np.inf))   # ascending: low kappa / long = suspicious
            rank = int(np.where(order == k)[0][0]) + 1
            rr[name].append(1.0 / rank); top1[name] += int(rank == 1)

    print(f"{args.npz} | L{args.layer} w{args.w} | error chains {n}")
    print(f"{'rank by':9s} {'MRR':>6s} {'top1':>6s}")
    for name in ("pooled", "min", "length"):
        print(f"{name:9s} {np.mean(rr[name]):6.3f} {top1[name]/n:6.3f}")


if __name__ == "__main__":
    main()
