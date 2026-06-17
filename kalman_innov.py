"""Kalman INNOVATION detector (discrete Zakai / sequential likelihood-ratio against a
healthy-reasoning model). NOT P_t -- in a linear KF the covariance is data-independent
(Riccati recursion), so P_t cannot separate correct/error. The data-dependent quantity is
the INNOVATION nu_t = y_t - E[y_t | healthy dynamics, past] : how much a step deviates
from what healthy reasoning would predict (the principled, model-based version of m_j/m_0).

State-space (scalar local-level + AR(1), c=1 for identifiability):
    x_t = a x_{t-1} + w_t,  w~N(0,Q)     # latent reasoning health
    y_t = x_t + v_t,        v~N(0,R)     # observation = per-step geometric signal (resultant)

Params (a,Q,R) estimated by METHOD OF MOMENTS on CORRECT chains only (healthy dynamics),
cross-fit by chain (GroupKFold) -> innovation on error chains has no leakage.

Tests (verdict = does innovation beat the raw observation, and earlier?):
  (c) AUROC: raw y_t  vs  |z_t| (normalized innovation)  vs  cumulative innovation
      -- pooled + within-length-bucket
  (b) event study: mean z_t by offset from first-error -- does it rise before/at the error
Deleted: (a) P_t (data-independent, AUROC==0.5 by construction).

Needs an npz with stepcloud(resultant) + gold_error_step + step_token_ranges.
"""

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
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        m = b == bb; ne, ng = int(y[m].sum()), int((y[m] == 0).sum())
        a = bdir(auroc(s[m], y[m]))
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def estimate_params(yseries):
    """method of moments on a list of correct-chain y-arrays -> (a, Q, R, mu)."""
    allv = np.concatenate(yseries); mu = allv.mean()
    g0 = np.mean((allv - mu) ** 2)
    p1n = p1d = p2n = p2d = 0.0
    for y in yseries:
        d = y - mu
        if len(d) >= 2:
            p1n += np.sum(d[1:] * d[:-1]); p1d += len(d) - 1
        if len(d) >= 3:
            p2n += np.sum(d[2:] * d[:-2]); p2d += len(d) - 2
    g1 = p1n / max(p1d, 1); g2 = p2n / max(p2d, 1)
    a = np.clip(g2 / g1 if abs(g1) > 1e-12 else 0.5, 0.05, 0.98)
    varx = g1 / a if abs(a) > 1e-9 else g0 * 0.5
    varx = min(max(varx, 1e-9), g0 * 0.999)
    Q = max(varx * (1 - a * a), 1e-9); R = max(g0 - varx, 1e-9)
    return a, Q, R, mu


def kalman_innov(y, a, Q, R, mu):
    """return per-step normalized innovation z_t (nan at t=0)."""
    T = len(y); z = np.full(T, np.nan)
    xh = y[0] - mu; P = Q + R
    for t in range(1, T):
        xp = a * xh; Pp = a * a * P + Q
        nu = (y[t] - mu) - xp; S = Pp + R
        z[t] = nu / np.sqrt(S)
        K = Pp / S; xh = xp + K * nu; P = (1 - K) * Pp
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--metric", default="resultant")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    gnames = [str(x) for x in z["geom_feature_names"]]
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    SC, SG, SR = z["stepcloud"], z["stepgeom"], z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)
    src, fi = (SC, cnames.index(args.metric)) if args.metric in cnames else (SG, gnames.index(args.metric))

    chains = []                       # (y_series, k, n_tok_series)
    for i in range(len(src)):
        g = np.asarray(src[i], float); seq = g[:, li, fi] if g.ndim == 3 else g[:, li]
        rng = np.asarray(SR[i], int); k = int(ges[i]); T = rng.shape[0]
        nt = np.array([int(rng[j, 1] - rng[j, 0] + 1) for j in range(T)], float)
        if np.isfinite(seq).all() and T >= 3:
            chains.append({"y": seq, "k": k, "nt": nt, "i": i, "correct": k < 0})

    from sklearn.model_selection import GroupKFold
    idx = np.arange(len(chains)); grp = idx
    Zall = {ci: None for ci in idx}
    gkf = GroupKFold(args.folds)
    for tr, te in gkf.split(idx, idx, grp):
        ytr = [chains[t]["y"] for t in tr if chains[t]["correct"]]
        if len(ytr) < 20:
            continue
        a, Q, R, mu = estimate_params(ytr)
        for t in te:
            Zall[t] = kalman_innov(chains[t]["y"], a, Q, R, mu)
    a, Q, R, mu = estimate_params([c["y"] for c in chains if c["correct"]])
    print(f"file: {args.npz} | layer {args.layer} | metric {args.metric} | chains {len(chains)}")
    print(f"healthy-dynamics params: a={a:.3f} Q={Q:.4f} R={R:.4f} mu={mu:.3f}")

    # AUROC arrays = LABELED steps only (j<k or correct -> 0; j==k -> 1; j>k skipped)
    # event arrays  = ALL j>=1 steps of ERROR chains (incl post-error) for the trajectory shape
    RAW, ABSZ, SZ, CUM, Y, NT = [], [], [], [], [], []
    EOFF, ESZ, ERAW = [], [], []
    for t in idx:
        zt = Zall[t]
        if zt is None:
            continue
        c = chains[t]; k = c["k"]; correct = c["correct"]; T = len(c["y"])
        cum = 0.0
        for j in range(T):
            if j >= 1 and np.isfinite(zt[j]):
                cum += zt[j] ** 2
            if j == 0:
                continue
            if not correct:                      # error chain: record full trajectory
                EOFF.append(j - k); ESZ.append(zt[j]); ERAW.append(c["y"][j])
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                continue                         # post-error: not a labeled detection target
            RAW.append(c["y"][j]); ABSZ.append(abs(zt[j])); SZ.append(zt[j]); CUM.append(cum)
            Y.append(y); NT.append(c["nt"][j])
    RAW = np.asarray(RAW); ABSZ = np.asarray(ABSZ); SZ = np.asarray(SZ); CUM = np.asarray(CUM)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float)
    EOFF = np.asarray(EOFF, int); ESZ = np.asarray(ESZ, float); ERAW = np.asarray(ERAW, float)

    print(f"\n=== (c) AUROC: raw observation vs Kalman innovation ===")
    print(f"  {'feature':22s} {'pooled':>8s} {'bucket':>8s}")
    for nm, v in [(f"raw {args.metric}", RAW), ("|z| (norm innovation)", ABSZ),
                  ("z (signed)", SZ), ("cumulative z^2", CUM)]:
        print(f"  {nm:22s} {bdir(auroc(v, Y)):8.3f} {bucket(v, Y, NT):8.3f}")

    print(f"\n=== (b) event study: signed-z trajectory by offset from first-error (Δ=0) ===")
    print(f"  {'Δ=j-k':>6s} {'n':>5s} {'mean z':>9s} {'SE':>7s}  {'mean raw':>9s}")
    for d in range(-4, 4):
        m = EOFF == d
        if m.sum() >= 5:
            se = ESZ[m].std() / np.sqrt(m.sum())
            star = " <-- error" if d == 0 else ""
            print(f"  {d:>6d} {int(m.sum()):>5d} {ESZ[m].mean():>+9.3f} {se:>7.3f}  {ERAW[m].mean():>+9.3f}{star}")
    pre = ESZ[(EOFF <= -3) & np.isfinite(ESZ)]; at0 = ESZ[(EOFF == 0) & np.isfinite(ESZ)]
    if len(pre) >= 5 and len(at0) >= 5:
        dd = at0.mean() - pre.mean(); se = np.sqrt(at0.std()**2/len(at0) + pre.std()**2/len(pre))
        print(f"  jump z(Δ=0) − z(Δ≤-3) = {dd:+.3f} [{dd-2*se:+.3f},{dd+2*se:+.3f}] "
              f"{'SIGNIFICANT' if abs(dd)-2*se > 0 else 'ns'}")

    print("\nVERDICT: if |z| / cumulative-z^2 AUROC > raw, or z rises before the error earlier "
          "than raw -> the Kalman innovation (deviation from healthy dynamics) beats the raw "
          "observation -> filtering framework helps. If not -> the temporal model adds nothing, "
          "but the detector gains a principled sequential-likelihood-ratio form.")


if __name__ == "__main__":
    main()
