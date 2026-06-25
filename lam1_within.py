"""Within-problem (difficulty-controlled, self-sampled) test of response-level spectral SCALARS.
Reads <tag>_multisample_sv.npz (problem_ids, is_correct, sv_clouds + cloud_layers; from 10_sample --store_clouds).
Whole-response token cloud per rollout -> kappa/lam1/effrank/HS (raw+centered); within-problem AUROC (incorrect vs
correct, SAME problem = difficulty held) with one global direction, vs pooled cross-problem; n_tok = length baseline."""
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


def feats(H):
    n = len(H); out = {"n_tok": float(n)}
    u = H / np.clip(np.linalg.norm(H, axis=1, keepdims=True), 1e-9, None)
    out["kappa"] = float(np.linalg.norm(u.mean(0)))
    for tg, M in (("raw", H), ("cen", H - H.mean(0))):
        s = np.linalg.svd(M, compute_uv=False); lam = s * s; nz = lam[lam > 1e-9]
        if len(nz) < 2:
            out[f"lam1_{tg}"] = out[f"effrank_{tg}"] = out[f"HS_{tg}"] = out[f"alpha_{tg}"] = np.nan; continue
        q = nz / nz.sum()
        out[f"lam1_{tg}"] = float(q[0])
        out[f"effrank_{tg}"] = float(np.exp(-(q * np.log(q)).sum()))
        out[f"HS_{tg}"] = float(np.log(lam).sum() / n) if (len(lam) == n and lam.min() > 1e-9) else np.nan
        sig = np.sqrt(nz); kk = np.arange(1, len(sig) + 1)         # power-law slope alpha (Yi Liu 2604.15350 Finding 7)
        out[f"alpha_{tg}"] = float(-np.polyfit(np.log(kk), np.log(sig), 1)[0])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    pid = z["problem_ids"].astype(int)
    isc = (z["is_correct_strict"] if "is_correct_strict" in z.files else z["is_correct"]).astype(int)
    if "respcloud" in z.files:                                   # extract_features --source sampled / processbench
        csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer); RC = z["respcloud"]
        def getH(i):
            return None if RC[i] is None else np.asarray(RC[i], np.float64)[:, li, :]
        N = len(RC); src = f"respcloud(JL) L{args.layer}"
    elif "sv_clouds" in z.files:                                 # 10_sample_and_extract --store_clouds
        cl = [int(x) for x in z["cloud_layers"]]; li = cl.index(args.layer); SVC = z["sv_clouds"]
        def getH(i):
            return None if SVC[i] is None else np.asarray(SVC[i], np.float64)[:, li, :]
        N = len(SVC); src = f"sv_clouds L{args.layer}"
    else:
        raise SystemExit("npz has neither respcloud nor sv_clouds (no token clouds stored)")
    yinc = (isc == 0).astype(int)
    rows, keep = [], []
    for i in range(N):
        H = getH(i)
        if H is not None and len(H) >= 4:
            rows.append(feats(H)); keep.append(i)
    keep = np.array(keep); yk = yinc[keep]; pk = pid[keep]
    print(f"# {src} | strict_labels={'is_correct_strict' in z.files}")
    groups = defaultdict(list)
    for j, p in enumerate(pk):
        groups[p].append(j)
    grp = [np.array(g) for g in groups.values() if len(g) >= 2 and 0 < int(yk[g].sum()) < len(g)]
    npairs = sum(int(yk[g].sum()) * int((yk[g] == 0).sum()) for g in grp)
    print(f"{args.npz} | L{args.layer} | rollouts {len(keep)} | usable-problems {len(grp)} | within-pairs {npairs} | incorrect {int(yk.sum())}/{len(yk)}")
    print(f"  {'feature':12s} {'within(diff-ctrl)':>17s} {'pooled(cross)':>14s}")
    for nm in ["kappa", "alpha_raw", "alpha_cen", "lam1_raw", "lam1_cen", "effrank_raw", "effrank_cen", "HS_raw", "HS_cen", "n_tok"]:
        f = np.array([r.get(nm, np.nan) for r in rows], float)
        ap_ = auroc(f, yk); sign = 1.0 if (np.isfinite(ap_) and ap_ >= 0.5) else -1.0
        num = den = 0.0
        for g in grp:
            a = auroc(sign * f[g], yk[g]); w = int(yk[g].sum()) * int((yk[g] == 0).sum())
            if np.isfinite(a):
                num += a * w; den += w
        wi = num / den if den else float("nan")
        print(f"  {nm:12s} {wi:17.3f} {(max(ap_, 1 - ap_) if np.isfinite(ap_) else float('nan')):14.3f}")


if __name__ == "__main__":
    main()
