"""Normalize each step's metric by the chain's FIRST step (the SMCD D_j/D_0 idea,
and exactly what the convergence figure plotted).

For each metric m and step j>=1: compare raw m_j vs ratio m_j/m_0 vs diff m_j-m_0,
where m_0 is the metric at step 0 of the same chain. Anchoring to the chain's own
first step removes the chain-level baseline (difficulty) -> a within-chain relative
measure. Reports detection AUROC (best dir) + within-length-bucket AUROC for each.

Honest expectation: normalizing by m_0 is a within-chain normalization -> it removes
the between-chain difficulty component, so the POOLED AUROC usually DROPS (like the
within-chain z-score did 0.82->0.66). It may be cleaner for within-chain localization.
Does it IMPROVE? -- this script answers directly.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    gnames = [str(x) for x in z["geom_feature_names"]]
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    SG, SC, SR = z["stepgeom"], z["stepcloud"], z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    METS = [("resultant", "c"), ("coherence", "c"), ("norm", "g"), ("cloud_D", "c"), ("U_D", "u")]

    out = {m[0]: {"raw": [], "ratio": [], "diff": [], "prev": []} for m in METS}
    NT, Y = [], []
    for i in range(len(SG)):
        sg = np.asarray(SG[i], float); sc = np.asarray(SC[i], float)
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0); T = rng.shape[0]
        a0 = int(rng[0, 0]); ud = np.asarray(UD[i], float) if UD is not None else None

        def val(nm, src, j):
            if src == "g":
                return sg[j, li, gnames.index(nm)]
            if src == "c":
                return sc[j, li, cnames.index(nm)] if nm in cnames else np.nan
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(ud), int(rng[j, 1]) - a0 + 1)
            return np.nanmean(ud[lo:hi]) if (ud is not None and hi > lo) else np.nan

        m0 = {nm: val(nm, src, 0) for nm, src in METS}
        for j in range(1, T):                       # j=0 is the reference, skip
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                continue
            for nm, src in METS:
                mj = val(nm, src, j); r0 = m0[nm]; mp = val(nm, src, j - 1)
                out[nm]["raw"].append(mj)
                out[nm]["ratio"].append(mj / r0 if abs(r0) > 1e-9 else np.nan)
                out[nm]["diff"].append(mj - r0)
                out[nm]["prev"].append(mj / mp if abs(mp) > 1e-9 else np.nan)   # m_j / m_{j-1}
            NT.append(int(rng[j, 1] - rng[j, 0] + 1)); Y.append(y)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float)
    for nm in out:
        for k2 in out[nm]:
            out[nm][k2] = np.asarray(out[nm][k2], float)

    print(f"file: {args.npz} | layer {args.layer} | steps(j>=1) {len(Y)} | first-error {int(Y.sum())}")
    print(f"\n{'metric':11s} | {'raw':>13s} | {'m_j/m_0':>13s} | {'m_j/m_(j-1)':>13s} | {'m_j-m_0':>13s}")
    for nm, _ in METS:
        r, rt, pv, df = out[nm]["raw"], out[nm]["ratio"], out[nm]["prev"], out[nm]["diff"]
        print(f"{nm:11s} | {bdir(auroc(r,Y)):.3f}/{bucket(r,Y,NT):.3f} | "
              f"{bdir(auroc(rt,Y)):.3f}/{bucket(rt,Y,NT):.3f} | "
              f"{bdir(auroc(pv,Y)):.3f}/{bucket(pv,Y,NT):.3f} | "
              f"{bdir(auroc(df,Y)):.3f}/{bucket(df,Y,NT):.3f}")
    print("\nformat raw/bucket = pooled AUROC / within-length-bucket AUROC. m_j/m_(j-1) = ratio to "
          "the IMMEDIATE previous step (most local dynamic view). If any relative beats raw -> "
          "normalizing helps; usually pooled DROPS (removes between-chain difficulty) while the "
          "length-controlled BUCKET may improve on hard configs -- that is the within-chain "
          "relative signal, mode-robust but weaker than raw which already encodes the drop.")


if __name__ == "__main__":
    main()
