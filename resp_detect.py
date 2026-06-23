"""Response-level detection (chain correct vs has-error): EDIS-style dynamics on the per-step kappa trajectory (GDIS) vs min/mean kappa vs EDIS-on-entropy."""
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


def edis_adp(seq, w=1):
    """EDIS-style on a sequence with adaptive threshold (tau = std): error = the quantity RISES."""
    s = np.asarray(seq, float); s = s[np.isfinite(s)]
    if len(s) < 3:
        return 0.0
    tau = s.std() + 1e-9; ww = min(w, max(1, len(s) // 2))
    burst = int(sum(1 for t in range(len(s) - ww) if s[t + ww] - s[t] > tau))
    pv = 0; rmin = s[0]
    for t in range(1, len(s)):
        if s[t] - rmin > tau:
            pv += 1
        rmin = min(rmin, s[t])
    return 0.5 * (burst + pv) * (1.0 + float(s.var()))


def trend(seq):
    """Linear-fit slope and R^2 (trajectory monotonicity)."""
    s = np.asarray(seq, float); s = s[np.isfinite(s)]; n = len(s)
    if n < 3:
        return 0.0, 0.0
    x = np.arange(n) - (n - 1) / 2; sm = s.mean()
    b = float((x * (s - sm)).sum() / (x ** 2).sum())
    ss = float(((s - sm) ** 2).sum())
    r2 = 1 - float(((s - (sm + b * x)) ** 2).sum()) / ss if ss > 0 else 0.0
    return b, r2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]; lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    fi = cn.index("resultant"); SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int); UD = z["tok_U_D"]
    isc = z["is_correct"].astype(int) if "is_correct" in z.files else None
    use_isc = isc is not None and not (ges >= 0).any()
    G_gdis, G_mink, G_meank, G_kvar, G_slope, G_r2, E_edis, E_mean, Y = [], [], [], [], [], [], [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i]); T = rng.shape[0]
        kseq = sc[:T, li, fi]; kseq = kseq[np.isfinite(kseq)]
        if len(kseq) < 2:
            continue
        ent = np.asarray(UD[i], float); b, r2 = trend(kseq)
        G_gdis.append(edis_adp(1 - kseq)); G_mink.append(1 - float(kseq.min())); G_meank.append(1 - float(kseq.mean()))
        G_kvar.append(float(kseq.var())); G_slope.append(-b); G_r2.append(r2)
        E_edis.append(edis_adp(ent, w=8)); E_mean.append(float(np.nanmean(ent)))
        Y.append(int(isc[i] == 0) if use_isc else int(k >= 0))
    Y = np.asarray(Y, int)
    print(f"{args.npz} | L{args.layer} | responses {len(Y)} err-chains {int(Y.sum())}")
    for nm, v in [("GDIS (kappa-dyn)", G_gdis), ("min-kappa", G_mink), ("mean-kappa", G_meank),
                  ("kappa-var", G_kvar), ("kappa-slope(down)", G_slope), ("kappa-R2(monot)", G_r2),
                  ("EDIS (entropy)", E_edis), ("mean-entropy", E_mean)]:
        print(f"  {nm:18s} AUROC {bdir(auroc(np.asarray(v, float), Y)):.3f}")


if __name__ == "__main__":
    main()
