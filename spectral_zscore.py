"""Perturbation/within-problem z-score of spectral features on SELF-SAMPLED rollouts.
Reference for a step = same-problem CORRECT-rollout steps with MATCHED token count n (cancels n-dependence,
finite-sample bias, and domain/difficulty baseline). Rollout anomaly = max |z| over its steps.
Task = within-problem: do z-scored features separate incorrect from correct rollouts (same problem)?
Compares per-problem n-matched z-score vs global-standardized raw. Needs sampled_*_cloud.npz (respcloud + is_correct + problem_ids)."""
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


def feats(H):
    Hc = H - H.mean(0); s = np.linalg.svd(Hc, compute_uv=False); lam = s ** 2; lam = lam[lam > 1e-9]
    if len(lam) < 2:
        return [np.nan, np.nan, np.nan]
    p = lam / lam.sum()
    return [float(np.log(lam).mean()), float(np.exp(-(p * np.log(p)).sum())), float(p[0])]  # HS, eff-rank, lam1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); ap.add_argument("--ntol", type=int, default=4); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    pid = z["problem_ids"].astype(int); isc = z["is_correct"].astype(int)
    names = ["HS", "effrank", "lam1"]
    rolls = []   # (problem_id, is_correct, [(f3, n), ...])
    if "sv_clouds" in z.files:                       # multisample format: per-rollout (n_tok, L, d) + per-step sizes
        cl_layers = [int(x) for x in z["cloud_layers"]]; li = cl_layers.index(args.layer)
        SVC, SZ = z["sv_clouds"], z["cloud_sizes"]
        for i in range(len(SVC)):
            if SVC[i] is None:
                continue
            cl = np.asarray(SVC[i], np.float64)[:, li, :]; sizes = np.asarray(SZ[i], int); pos = 0; steps = []
            for sz in sizes:
                if sz >= 4 and pos + sz <= len(cl):
                    steps.append((feats(cl[pos:pos + sz]), int(sz)))
                pos += sz
            if steps:
                rolls.append((int(pid[i]), int(isc[i]), steps))
    else:                                            # respcloud format (teacher-forcing extraction)
        csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer)
        RC, SR = z["respcloud"], z["step_token_ranges"]
        for i in range(len(RC)):
            if RC[i] is None:
                continue
            H = np.asarray(RC[i], np.float64)[:, li, :]; rng = np.asarray(SR[i], int); a0 = int(rng[0, 0]); steps = []
            for j in range(rng.shape[0]):
                lo = max(0, int(rng[j, 0]) - a0); hi = min(len(H), int(rng[j, 1]) - a0 + 1)
                if hi - lo >= 4:
                    steps.append((feats(H[lo:hi]), hi - lo))
            if steps:
                rolls.append((int(pid[i]), int(isc[i]), steps))
    # per-problem correct-rollout reference (feature, n) per channel
    from collections import defaultdict
    ref = defaultdict(list)   # problem_id -> list of (f3, n) from CORRECT rollouts
    for p, c, steps in rolls:
        if c == 1:
            ref[p].extend(steps)
    # global raw stats per channel (for the baseline)
    allf = np.array([f for _, _, steps in rolls for (f, n) in steps], float)
    gmu = np.nanmean(allf, 0); gsd = np.nanstd(allf, 0) + 1e-9

    def zscore_step(f, n, p, ch):
        rs = [rf[ch] for (rf, rn) in ref.get(p, []) if abs(rn - n) <= args.ntol and np.isfinite(rf[ch])]
        if len(rs) < 5:
            return np.nan
        m = np.mean(rs); s = np.std(rs) + 1e-9
        return abs(f[ch] - m) / s

    # within-problem AUROC (incorrect=1 vs correct=0), per channel, z-score vs global-raw
    byp = defaultdict(list)
    for ridx, (p, c, steps) in enumerate(rolls):
        byp[p].append((ridx, c))
    print(f"{args.npz} | L{args.layer} | rollouts {len(rolls)} | problems {len(byp)} | correct-rate {np.mean([c for _,c,_ in rolls]):.3f}")
    for ch, nm in enumerate(names):
        z_scores = []; raw_scores = []
        for p, c, steps in rolls:
            zs = [zscore_step(f, n, p, ch) for (f, n) in steps]; zs = [v for v in zs if np.isfinite(v)]
            rw = [abs((f[ch] - gmu[ch]) / gsd[ch]) for (f, n) in steps if np.isfinite(f[ch])]
            z_scores.append(max(zs) if zs else np.nan); raw_scores.append(max(rw) if rw else np.nan)
        z_scores = np.array(z_scores, float); raw_scores = np.array(raw_scores, float)
        # within-problem AUROC, weighted by pair count
        za = ra = den = 0.0
        for p, lst in byp.items():
            idx = [i for i, _ in lst]; yy = np.array([1 - rolls[i][1] for i in idx])  # incorrect=1
            if yy.sum() == 0 or yy.sum() == len(yy):
                continue
            w = int(yy.sum()) * int((yy == 0).sum())
            az = auroc(z_scores[idx], yy); ar = auroc(raw_scores[idx], yy)
            if np.isfinite(az):
                za += bdir(az) * w; ra += bdir(ar) * w; den += w
        print(f"  {nm:9s} within-problem AUROC:  z-score(per-problem,n-matched) {za/den:.3f}   raw(global) {ra/den:.3f}")


if __name__ == "__main__":
    main()
