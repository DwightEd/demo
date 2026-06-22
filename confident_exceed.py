"""The scoped EXCEED: geometry is the only signal that detects CONFIDENT reasoning errors -- the
low-entropy hallucinations that the entire entropy family (mean entropy AND its SOTA dynamic form
EDIS) is structurally blind to.

Confident hallucinations (model is wrong but low-entropy/sure) are the most dangerous failure mode
and exactly where entropy-based detection collapses to chance. The headline is not 'geometry beats
EDIS on average' (it does not) but: stratify labeled steps by confidence (step mean entropy); in the
CONFIDENT stratum, geometry (directional collapse) keeps ~0.72 while BOTH entropy and EDIS fall to
~0.5. That is a real exceed, scoped to the regime that matters and that entropy cannot enter.

Per step: resultant (geometry) | step mean entropy U_D (confidence / static entropy) |
step-EDIS (entropy dynamics, burst+peak-valley on the step's per-token entropy). Stratify by U_D
terciles; in each, AUROC of all three for first-error + danger share (errors living in the stratum).

Needs coh.npz: stepcloud(resultant) + tok_U_D + step_token_ranges + gold_error_step + layers_used.
Runs on all four configs.
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
    m = np.isfinite(s) & np.isfinite(nt); s, y, nt = s[m], y[m], nt[m]
    if len(s) < 10:
        return float("nan")
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm])); ne, ng = int(y[mm].sum()), int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def edis(H, w=8, tb=1.36, tr=1.33):
    H = np.asarray(H, float); H = H[np.isfinite(H)]
    if len(H) < 3:
        return 0.0
    ww = min(w, max(2, len(H) // 2))
    burst = sum(1 for t in range(len(H) - ww) if H[t + ww] - H[t] > tb) if len(H) > ww else 0
    rebound = 0; rmin = H[0]
    for t in range(1, len(H)):
        if H[t] - rmin > tr:
            rebound += 1
        rmin = min(rmin, H[t])
    return 0.5 * (burst + rebound) * (1.0 + float(H.var()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]
    gn = [str(x) for x in z["geom_feature_names"]] if "geom_feature_names" in z.files else []
    lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    SG = z["stepgeom"] if "stepgeom" in z.files else None
    UD = z["tok_U_D"]; fi = cn.index("resultant")
    mti = cn.index("mean_tok_norm") if "mean_tok_norm" in cn else None      # pure magnitude
    ngi = gn.index("norm") if "norm" in gn else None                        # pooled model-length

    RES, POOL, MAG, UDV, EDS, Y, NT = [], [], [], [], [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i])
        sg = np.asarray(SG[i], float) if SG is not None else None
        correct = (k < 0); T = rng.shape[0]; a0 = int(rng[0, 0]); ud = np.asarray(UD[i], float)
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(ud), int(rng[j, 1]) - a0 + 1)
            if hi - lo < 2:
                continue
            uds = ud[lo:hi]
            RES.append(sc[j, li, fi])
            POOL.append(sg[j, li, ngi] if (sg is not None and ngi is not None) else np.nan)  # pooled norm
            MAG.append(sc[j, li, mti] if mti is not None else np.nan)                          # mean tok norm
            UDV.append(float(np.nanmean(uds))); EDS.append(edis(uds))
            Y.append(lab); NT.append(int(rng[j, 1] - rng[j, 0] + 1))
    RES = np.asarray(RES); POOL = np.asarray(POOL); MAG = np.asarray(MAG)
    UDV = np.asarray(UDV); EDS = np.asarray(EDS); Y = np.asarray(Y, int); NT = np.asarray(NT, float)
    keep = np.isfinite(RES) & np.isfinite(UDV)
    RES, POOL, MAG, UDV, EDS, Y, NT = RES[keep], POOL[keep], MAG[keep], UDV[keep], EDS[keep], Y[keep], NT[keep]

    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"overall AUROC:  dir(resultant) {bdir(auroc(-RES, Y)):.3f}  pooled-norm {bdir(auroc(-POOL, Y)):.3f}  "
          f"mag(mean_tok_norm) {bdir(auroc(-MAG, Y)):.3f}  entropy {bdir(auroc(UDV, Y)):.3f}  "
          f"EDIS {bdir(auroc(EDS, Y)):.3f}")

    q = np.quantile(UDV, [1 / 3, 2 / 3]); strat = np.digitize(UDV, q)
    names = ["LOW entropy (CONFIDENT)", "MID entropy", "HIGH entropy (uncertain)"]
    print(f"\n{'confidence stratum':26s} {'n':>6s} {'err':>5s} {'dir':>6s} {'pool':>6s} {'mag':>6s} "
          f"{'entropy':>8s} {'EDIS':>7s}")
    for s in range(3):
        m = strat == s; ne = int(Y[m].sum())
        ad = bdir(auroc(-RES[m], Y[m])); ap = bdir(auroc(-POOL[m], Y[m])); am = bdir(auroc(-MAG[m], Y[m]))
        au = bdir(auroc(UDV[m], Y[m])); ae = bdir(auroc(EDS[m], Y[m]))
        print(f"  {names[s]:26s} {int(m.sum()):>6d} {ne:>5d} {ad:>6.3f} {ap:>6.3f} {am:>6.3f} "
              f"{au:>8.3f} {ae:>7.3f}")
    # length-clean check for the best geometry in the confident stratum
    cf = strat == 0
    print(f"\nCONFIDENT-stratum length-bucket:  dir {bucket(-RES[cf], Y[cf], NT[cf]):.3f}  "
          f"pooled-norm {bucket(-POOL[cf], Y[cf], NT[cf]):.3f}  mag {bucket(-MAG[cf], Y[cf], NT[cf]):.3f}")

    # SHARP confidence cut: in the most-confident X%, entropy is near-flat so EDIS (which needs entropy
    # dynamics) must mechanically collapse, while geometry does not depend on entropy variation.
    print(f"\n{'most-confident cut':18s} {'n':>6s} {'err':>5s} {'dir':>6s} {'pool':>6s} "
          f"{'entropy':>8s} {'EDIS':>7s}")
    for pct in [0.05, 0.10, 0.20]:
        thr = np.quantile(UDV, pct); m = UDV <= thr; ne = int(Y[m].sum())
        ad = bdir(auroc(-RES[m], Y[m])); ap = bdir(auroc(-POOL[m], Y[m]))
        au = bdir(auroc(UDV[m], Y[m])); ae = bdir(auroc(EDS[m], Y[m]))
        print(f"  bottom {int(pct*100):>3d}% entropy  {int(m.sum()):>6d} {ne:>5d} {ad:>6.3f} {ap:>6.3f} "
              f"{au:>8.3f} {ae:>7.3f}")

    conf = strat == 0
    share = int(Y[conf].sum()) / max(int(Y.sum()), 1)
    print(f"\nDANGER SHARE: {100*share:.0f}% of ALL first-errors live in the CONFIDENT stratum "
          f"(low-entropy, entropy-family blind).")
    print("read: CORRECTED CLAIM. At the tercile cut, EDIS does NOT collapse in the confident stratum "
          "(~0.67-0.73) -- it is entropy DYNAMICS, not level, so confident steps with entropy bursts still "
          "fire it; geometry is only marginally ahead there. The ONLY place the exceed can hold is the SHARP "
          "cut: in the bottom 5-10% entropy (near-flat) steps, EDIS has no entropy variation to compute "
          "burst/peak-valley and must collapse toward 0.5, while geometry (dir / pooled-norm) does not depend "
          "on entropy dynamics. DECISIVE = the bottom-5%/10% rows: if geom holds (~0.7) while EDIS drops to "
          "~0.5, geometry uniquely detects the genuinely-confident errors EDIS cannot; if EDIS still holds "
          "there, geometry has no exclusive territory over EDIS and the anchor must be reconsidered. Also "
          "watch dir vs pool: if pooled-norm wins, magnitude carries extra signal we should not strip.")


if __name__ == "__main__":
    main()
