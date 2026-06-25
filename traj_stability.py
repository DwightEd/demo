"""Within-problem (difficulty-controlled, self-sampled) test of TRAJECTORY-STABILITY objects across reasoning steps:
 P2 eff_rank temporal mean/std (is occupied-dim stationary or does it jump);
 P3 principal-subspace (top-k) Grassmann CHORDAL velocity mean/std (does the occupied subspace ORIENTATION evolve stably);
 P4 uncentered-Gram (H Hᵀ second-moment) Bures velocity mean (distinct object from centered-covariance velocity).
Reads <tag>_multisample_sv.npz (sv_clouds + cloud_sizes + problem_ids + is_correct). Chain-level features; within-problem
AUROC (incorrect vs correct, SAME problem) with one global direction, vs pooled cross-problem; n_tok = length baseline."""
from __future__ import annotations
import argparse
import numpy as np
from collections import defaultdict


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


def bures2_uncentered(Xa, Xb):
    """Squared Bures distance between UNCENTERED second moments (H Hᵀ / n), in the joint span."""
    na, nb = len(Xa), len(Xb); T = np.vstack([Xa, Xb])
    g, U = np.linalg.eigh(T @ T.T); keep = g > 1e-9 * max(g.max(), 1e-30)
    if keep.sum() < 2:
        return np.nan
    c = U[:, keep] * np.sqrt(g[keep]); a, b = c[:na], c[na:]
    A = a.T @ a / na; B = b.T @ b / nb
    wa, Ua = np.linalg.eigh(A); As = (Ua * np.sqrt(np.clip(wa, 0, None))) @ Ua.T
    wm = np.linalg.eigvalsh(As @ B @ As)
    return float(np.trace(A) + np.trace(B) - 2.0 * np.sqrt(np.clip(wm, 0, None)).sum())


def step_geom(H, k):
    """(eff_rank, top-k right singular vectors Vt[:k] of centered cloud) or (nan, None)."""
    n = len(H)
    if n < 4:
        return np.nan, None
    Hc = H - H.mean(0)
    _, s, Vt = np.linalg.svd(Hc, full_matrices=False)
    lam = (s ** 2); lam = lam[lam > 1e-9]
    if len(lam) < 2:
        return np.nan, None
    q = lam / lam.sum(); er = float(np.exp(-(q * np.log(q)).sum()))
    kk = min(k, Vt.shape[0])
    return er, (Vt[:kk] if kk >= 1 else None)


def chordal(V1, V2):
    """Normalized Grassmann chordal distance in [0,1] between two top-k subspaces (rows orthonormal)."""
    k = min(V1.shape[0], V2.shape[0]); M = V1[:k] @ V2[:k].T
    return float(np.sqrt(max(0.0, 1.0 - (M ** 2).sum() / k)))


def chain_feats(H, slices, k):
    ers, subs, gv = [], [], []
    prevV = None; prevH = None
    for a, b in slices:
        Hj = H[a:b]
        er, V = step_geom(Hj, k); ers.append(er)
        if V is not None and prevV is not None:
            subs.append(chordal(prevV, V))
        if prevH is not None and len(Hj) >= 4 and len(prevH) >= 4:
            d = bures2_uncentered(prevH, Hj)
            if np.isfinite(d):
                gv.append(max(d, 0.0))
        if V is not None:
            prevV = V
        prevH = Hj
    ers = np.array([e for e in ers if np.isfinite(e)], float)
    return {
        "effrank_mean": float(ers.mean()) if len(ers) else np.nan,
        "effrank_std": float(ers.std()) if len(ers) >= 2 else np.nan,
        "subspace_vel_mean": float(np.mean(subs)) if subs else np.nan,
        "subspace_vel_std": float(np.std(subs)) if len(subs) >= 2 else np.nan,
        "gram_bures_mean": float(np.sqrt(np.mean(gv))) if gv else np.nan,
        "n_tok": float(len(H)),
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=16); ap.add_argument("--k", type=int, default=5); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    pid = z["problem_ids"].astype(int)
    isc = (z["is_correct_strict"] if "is_correct_strict" in z.files else z["is_correct"]).astype(int)
    if "respcloud" in z.files:                                   # extract_features: respcloud + step_token_ranges
        csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer); RC = z["respcloud"]; SR = z["step_token_ranges"]
        def get(i):
            if RC[i] is None:
                return None, None
            H = np.asarray(RC[i], np.float64)[:, li, :]; rng = np.asarray(SR[i], int); a0 = int(rng[0, 0])
            sl = [(max(0, int(rng[j, 0]) - a0), min(len(H), int(rng[j, 1]) - a0 + 1)) for j in range(rng.shape[0])]
            return H, [(a, b) for a, b in sl if b - a >= 1]
    elif "sv_clouds" in z.files:                                 # 10_sample --store_clouds: sv_clouds + cloud_sizes
        cl = [int(x) for x in z["cloud_layers"]]; li = cl.index(args.layer); SVC, SZ = z["sv_clouds"], z["cloud_sizes"]
        def get(i):
            if SVC[i] is None:
                return None, None
            H = np.asarray(SVC[i], np.float64)[:, li, :]; sz = np.asarray(SZ[i], int)
            e = np.cumsum(sz); s = e - sz; return H, list(zip(s.tolist(), e.tolist()))
    else:
        raise SystemExit("npz has neither respcloud nor sv_clouds")
    yinc = (isc == 0).astype(int)
    rows, keep = [], []
    for i in range(len(RC) if "respcloud" in z.files else len(SVC)):
        H, sl = get(i)
        if H is None or sl is None or len(sl) < 2:
            continue
        rows.append(chain_feats(H, sl, args.k)); keep.append(i)
    keep = np.array(keep); yk = yinc[keep]; pk = pid[keep]
    groups = defaultdict(list)
    for j, p in enumerate(pk):
        groups[p].append(j)
    grp = [np.array(g) for g in groups.values() if len(g) >= 2 and 0 < int(yk[g].sum()) < len(g)]
    npairs = sum(int(yk[g].sum()) * int((yk[g] == 0).sum()) for g in grp)
    print(f"{args.npz} | L{args.layer} k{args.k} | chains {len(keep)} | usable-problems {len(grp)} | within-pairs {npairs}")
    print(f"  {'feature':18s} {'within(diff-ctrl)':>17s} {'pooled(cross)':>14s}   [ref: SPE~0.68 probe~0.71 scalar~0.55]")
    for nm in ["effrank_mean", "effrank_std", "subspace_vel_mean", "subspace_vel_std", "gram_bures_mean", "n_tok"]:
        f = np.array([r.get(nm, np.nan) for r in rows], float)
        ap_ = auroc(f, yk); sign = 1.0 if (np.isfinite(ap_) and ap_ >= 0.5) else -1.0
        num = den = 0.0
        for g in grp:
            a = auroc(sign * f[g], yk[g]); w = int(yk[g].sum()) * int((yk[g] == 0).sum())
            if np.isfinite(a):
                num += a * w; den += w
        wi = num / den if den else float("nan")
        print(f"  {nm:18s} {wi:17.3f} {(max(ap_, 1 - ap_) if np.isfinite(ap_) else float('nan')):14.3f}")


if __name__ == "__main__":
    main()
