"""Does the geometric signal catch CONFIDENT hallucinations -- the errors uncertainty is blind to?

Uncertainty (U_D entropy) dominates the fused detector, so the question is whether the geometric
increment is just redundant, or whether it specifically catches LOW-uncertainty (confident) errors
that an entropy detector structurally misses. Confident-but-wrong is the most dangerous failure
mode, so if geometry lives there, it is the valuable part -- not a small add-on.

Test: stratify labeled steps by per-step U_D into terciles. Within each stratum compute AUROC of
resultant (geometry) for first-error detection, and AUROC of U_D itself (should collapse toward
0.5 within a narrow stratum). If resultant's AUROC in the LOW-U_D stratum is ~0.7, geometry catches
confident hallucinations; if ~0.5, it only helps where uncertainty already does.

Needs coh.npz: stepcloud(resultant) + tok_U_D + step_token_ranges + gold_error_step.
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
    fin = np.isfinite(s) & np.isfinite(nt)
    s, y, nt = s[fin], y[fin], nt[fin]
    if len(s) < 10:
        return float("nan")
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        m = b == bb; a = bdir(auroc(s[m], y[m])); ne, ng = int(y[m].sum()), int((y[m] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]
    lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"]; fi = cn.index("resultant")

    RESULT, UDV, Y, NT = [], [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i])
        correct = (k < 0); T = rng.shape[0]; a0 = int(rng[0, 0]); ud = np.asarray(UD[i], float)
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(ud), int(rng[j, 1]) - a0 + 1)
            uv = np.nanmean(ud[lo:hi]) if hi > lo else np.nan
            RESULT.append(sc[j, li, fi]); UDV.append(uv); Y.append(lab)
            NT.append(int(rng[j, 1] - rng[j, 0] + 1))
    R = np.asarray(RESULT); U = np.asarray(UDV); Y = np.asarray(Y, int); NT = np.asarray(NT, float)
    keep = np.isfinite(R) & np.isfinite(U)
    R, U, Y, NT = R[keep], U[keep], Y[keep], NT[keep]

    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"overall: resultant AUROC {bdir(auroc(R,Y)):.3f} | U_D AUROC {bdir(auroc(U,Y)):.3f}")

    # tercile split by U_D
    q = np.quantile(U, [1/3, 2/3]); strat = np.digitize(U, q)
    names = ["LOW U_D (confident)", "MID U_D", "HIGH U_D (uncertain)"]
    print(f"\n{'stratum':22s} {'n':>6s} {'err':>5s} {'resultant AUROC':>16s} {'(bucket)':>9s} {'U_D AUROC':>10s}")
    for s in range(3):
        m = strat == s; ne = int(Y[m].sum())
        ar = bdir(auroc(R[m], Y[m])); bk = bucket(R[m], Y[m], NT[m]); au = bdir(auroc(U[m], Y[m]))
        print(f"  {names[s]:22s} {int(m.sum()):>6d} {ne:>5d} {ar:>16.3f} {bk:>9.3f} {au:>10.3f}")

    print("\nread: the decisive cell is resultant AUROC in 'LOW U_D (confident)'. If ~0.7, geometry "
          "catches CONFIDENT hallucinations -- exactly the low-entropy errors an uncertainty detector "
          "is blind to (its U_D AUROC there ~0.5). That reframes geometry from a small redundant add-on "
          "into the part that covers uncertainty's structural blind spot. If ~0.5, geometry only helps "
          "where uncertainty already does, and the increment is not about confident hallucinations.")


if __name__ == "__main__":
    main()
