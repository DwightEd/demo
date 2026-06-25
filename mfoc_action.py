"""Step-free MFOC action: sliding-window Bures velocity over the raw token stream (no gold steps in features).
v_w = Bures(cov(window_w), cov(window_{w+1})) aggregated over layers; gold error step -> error token region only for eval.
Tests H2 (post-error action collapses LOW) vs shock (onset HIGH), position-residualized, within-response. --shuffle = control."""
from __future__ import annotations
import argparse
import numpy as np
import hidden_io


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


def boot_ci(vals, fn=np.median, n=300):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < 5:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(0)
    bs = [fn(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return (float(fn(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def bures2(Xa, Xb):
    """Squared Bures distance between the covariances of two token blocks, computed in their joint span."""
    Ca = Xa - Xa.mean(0); Cb = Xb - Xb.mean(0); W = len(Xa)
    T = np.vstack([Ca, Cb])
    g, U = np.linalg.eigh(T @ T.T)
    keep = g > 1e-9 * max(g.max(), 1e-30)
    if keep.sum() < 2:
        return np.nan
    coords = U[:, keep] * np.sqrt(g[keep])
    a, b = coords[:W], coords[W:]
    A = a.T @ a / W; B = b.T @ b / W
    wa, Ua = np.linalg.eigh(A); As = (Ua * np.sqrt(np.clip(wa, 0, None))) @ Ua.T
    wm = np.linalg.eigvalsh(As @ B @ As)
    return float(np.trace(A) + np.trace(B) - 2.0 * np.sqrt(np.clip(wm, 0, None)).sum())


def velocity(Hs, W, s):
    """Per-window aggregated Bures velocity over layers; returns (midpoints, v, R)."""
    R = len(Hs[0]); starts = list(range(0, R - W + 1, s))
    if len(starts) < 3:
        return None
    mids, vel = [], []
    for wi in range(len(starts) - 1):
        a0, b0 = starts[wi], starts[wi + 1]
        v2 = [max(d, 0.0) for H in Hs for d in [bures2(H[a0:a0 + W], H[b0:b0 + W])] if np.isfinite(d)]
        if not v2:
            continue
        vel.append(np.sqrt(np.mean(v2))); mids.append((a0 + b0) / 2 + W / 2)
    return (np.array(mids), np.array(vel), R) if len(vel) >= 3 else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layers", default="10,14,18,22")
    ap.add_argument("--W", type=int, default=24); ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--npost", type=int, default=3); ap.add_argument("--shuffle", action="store_true"); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    assert "hidden_stored" in z.files and bool(z["hidden_stored"]), "need full-dim hidden shards"
    hd = str(z["hidden_dir"]); hl = [int(x) for x in z["hidden_layers"]]; ids = z["ids"]
    lcs = [hl.index(int(x)) for x in args.layers.split(",")]
    ges = z["gold_error_step"].astype(int); SR = z["step_token_ranges"]
    rng = np.random.default_rng(0)

    chains = []
    for i in range(len(ids)):
        a = hidden_io.load_chain(hd, ids[i])
        Hs = [np.asarray(a[:, lc, :], np.float64) for lc in lcs]
        if args.shuffle:
            perm = rng.permutation(len(Hs[0])); Hs = [H[perm] for H in Hs]
        out = velocity(Hs, args.W, args.stride)
        if out is None:
            continue
        mids, vel, R = out; rngc = np.asarray(SR[i], int); a0 = int(rngc[0, 0]); k = int(ges[i])
        ek = None if k < 0 else ((int(rngc[k, 0]) - a0) / R, (int(rngc[k, 1]) - a0) / R)
        chains.append({"p": mids / R, "v": vel, "correct": k < 0, "ek": ek})
    if not chains:
        print("no chains"); return

    pc = np.concatenate([c["p"] for c in chains if c["correct"]]) if any(c["correct"] for c in chains) else np.concatenate([c["p"] for c in chains])
    vc = np.concatenate([c["v"] for c in chains if c["correct"]]) if any(c["correct"] for c in chains) else np.concatenate([c["v"] for c in chains])
    coef = np.polyfit(pc, vc, 2)
    for c in chains:
        c["vr"] = c["v"] - np.polyval(coef, c["p"])

    dcol, shock, loc_low, loc_high, lw = [], [], [], [], []
    for c in chains:
        if c["correct"] or c["ek"] is None:
            continue
        s_k, e_k = c["ek"]; p, vr = c["p"], c["vr"]
        pre = vr[p < s_k]; postidx = np.where(p >= s_k)[0]
        if len(pre) < 1 or len(postidx) < 1:
            continue
        post = vr[postidx[:args.npost]]
        dcol.append(post.mean() - pre.mean()); shock.append(vr[postidx[0]] - pre.mean())
        em = (p >= s_k) & (p <= e_k); om = ~em
        if em.sum() >= 1 and om.sum() >= 2:
            ev = vr[em].mean()
            loc_low.append(np.mean(vr[om] > ev)); loc_high.append(np.mean(vr[om] < ev)); lw.append(int(om.sum()))

    tag = "SHUFFLED-CONTROL" if args.shuffle else "MFOC action"
    nerr = sum((not c["correct"]) and c["ek"] is not None for c in chains)
    print(f"{args.npz} | {tag} | W{args.W} s{args.stride} L[{args.layers}] | chains {len(chains)} err {nerr}")
    print(f"  Delta_collapse (post-pre v_res)   median {boot_ci(dcol)}   (<0 = H2 low-action collapse)")
    print(f"  onset shock    (post0-pre v_res)  median {boot_ci(shock)}   (>0 = shock at onset)")
    w = np.asarray(lw, float)
    if len(lw):
        print(f"  within-resp localization  low-action {np.average(loc_low, weights=w):.3f}   high-action {np.average(loc_high, weights=w):.3f}   (>0.5 = that direction localizes error)")


if __name__ == "__main__":
    main()
