"""Step-level (within-step cloud) HS / ME / lam1, strict to 2602.09158 (Eq 1-2), raw + centered.
At step level n (tokens) << d so the Gram is well-conditioned -> HS (log-det) is meaningful here
(unlike whole-response where it is tail-dominated). Pooled AUROC + length bucket + within-chain wc_loc(perp len)."""
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


def feats(H, center):
    m = len(H)
    if m < 4:
        return (np.nan, np.nan, np.nan)
    M = (H - H.mean(0)) if center else H
    s = np.linalg.svd(M, compute_uv=False); lam = s ** 2; nz = lam[lam > 1e-9]
    if len(nz) < 2:
        return (np.nan, np.nan, np.nan)
    q = nz / nz.sum()
    ME = float(-(q * np.log(q)).sum()); lam1 = float(q[0])
    HS = float(np.log(lam).sum() / m) if (len(lam) == m and lam.min() > 1e-9) else np.nan
    return (HS, ME, lam1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True); ges = z["gold_error_step"].astype(int); SR = z["step_token_ranges"]
    if "hidden_stored" in z.files and bool(z["hidden_stored"]):
        import hidden_io
        hd = str(z["hidden_dir"]); hl = [int(x) for x in z["hidden_layers"]]; lc = hl.index(args.layer); ids = z["ids"]
        def getH(i):
            return np.asarray(hidden_io.load_chain(hd, ids[i])[:, lc, :], np.float64)
        N = len(ids); src = "full-dim"
    else:
        csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer); RC = z["respcloud"]
        def getH(i):
            return None if RC[i] is None else np.asarray(RC[i], np.float64)[:, li, :]
        N = len(RC); src = "respcloud(JL)"
    cols = ["HS_raw", "ME_raw", "lam1_raw", "HS_cen", "ME_cen", "lam1_cen"]
    X, NT, Y, G = [], [], [], []; chains = []
    for i in range(N):
        H = getH(i)
        if H is None or len(H) < 4:
            continue
        rng = np.asarray(SR[i], int); a0 = int(rng[0, 0]); k = int(ges[i]); correct = k < 0; T = rng.shape[0]
        vec = np.full((T, 6), np.nan)
        for j in range(T):
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(H), int(rng[j, 1]) - a0 + 1)
            if hi - lo >= 4:
                vec[j, :3] = feats(H[lo:hi], False); vec[j, 3:] = feats(H[lo:hi], True)
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            X.append(vec[j]); NT.append(int(rng[j, 1] - rng[j, 0] + 1)); Y.append(lab); G.append(i)
        if not correct and T >= 4:
            chains.append({"vec": vec, "k": k, "nt": np.array([int(rng[j, 1] - rng[j, 0] + 1) for j in range(T)], float)})
    X = np.asarray(X); NT = np.asarray(NT, float); Y = np.asarray(Y, int)
    print(f"{args.npz} | L{args.layer} | {src} | steps {len(Y)} err {int(Y.sum())} | err-chains {len(chains)}")
    print(f"  {'metric':9s} {'pooled':>7s} {'bkt(len)':>9s} {'wc_loc(perp len)':>17s}")

    def wc(c):
        sign = 1.0 if auroc(X[:, c], Y) >= 0.5 else -1.0; locs, w = [], []
        for ch in chains:
            v = ch["vec"][:, c]; nt = ch["nt"]; k = ch["k"]; fin = np.isfinite(v) & np.isfinite(nt)
            if fin.sum() < 3 or not fin[k]:
                continue
            b = np.polyfit(nt[fin], v[fin], 1); res = sign * (v - (b[0] * nt + b[1]))
            others = np.array([j for j in range(len(v)) if j != k and fin[j]])
            if len(others) < 2:
                continue
            locs.append(np.mean(res[others] < res[k])); w.append(len(others))
        return np.average(locs, weights=np.asarray(w, float)) if locs else float("nan")
    for c, nm in enumerate(cols):
        print(f"  {nm:9s} {bdir(auroc(X[:, c], Y)):7.3f} {bucket(X[:, c], Y, NT):9.3f} {wc(c):17.3f}")


if __name__ == "__main__":
    main()
