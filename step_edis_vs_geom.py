"""Step-level head-to-head: geometric directional collapse (resultant) vs the STEP-LEVEL version of
EDIS (within-step entropy-dynamics instability), for first-error localization on ProcessBench.

Why this is our turf: EDIS operates at the trajectory/chain level and EXPLICITLY lists step-level
extension as future work ("segmenting at reasoning boundaries and computing local instability").
We already have a validated step-level geometric signal (~0.77). Here we build the step-level EDIS
baseline (entropy instability WITHIN each step's tokens) and show whether geometry beats it at
first-error localization -- the task EDIS does not do and where our transient signal is strongest
(unlike chain-level selection, where a transient single-step dip does not aggregate).

Per step (first-error label): resultant (stepcloud) | step mean-entropy (static U_D) |
step-EDIS (burst+peak-valley on the step's per-token entropy x (1+var)). AUROC pooled + length-bucket.

Needs coh.npz: stepcloud(resultant) + tok_U_D + step_token_ranges + gold_error_step + layers_used.
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
    m = np.isfinite(s); s, y, nt = s[m], y[m], nt[m]
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
    """EDIS on an entropy sequence: burst (cumulative growth over window w) + peak-valley (rise above
    running min) times (1+var). Higher = more unstable. For step-level we use a small window."""
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
    lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"]; fi = cn.index("resultant")

    RES, MENT, SEDIS, Y, NT = [], [], [], [], []
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
            uds = ud[lo:hi]
            if hi - lo < 2:
                continue
            RES.append(sc[j, li, fi]); MENT.append(float(np.nanmean(uds))); SEDIS.append(edis(uds))
            Y.append(lab); NT.append(int(rng[j, 1] - rng[j, 0] + 1))
    RES = np.asarray(RES); MENT = np.asarray(MENT); SEDIS = np.asarray(SEDIS)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float)

    print(f"file: {args.npz} | layer {args.layer} | labeled steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"\n{'step-level signal':28s} {'AUROC':>7s} {'bucket':>7s}")
    for nm, v in [("resultant (geometry, ours)", -RES),       # error = LOW R -> flip
                  ("step mean-entropy (static)", MENT),
                  ("step-EDIS (entropy-dyn)", SEDIS)]:
        print(f"  {nm:28s} {bdir(auroc(v, Y)):7.3f} {bucket(v, Y, NT):7.3f}")
    print("\nread: STEP-LEVEL first-error localization -- the task EDIS lists as future work and where our "
          "transient geometric dip is strongest (chain-level selection drowns it). WIN = resultant > step-EDIS "
          "> step mean-entropy: geometry localizes errors at the step level better than entropy dynamics, in "
          "EDIS's own stated gap. (Within a step, resultant is a single static value -- no GDIS trajectory -- "
          "so static resultant is the right geometric signal at this granularity.) Run all four configs.")


if __name__ == "__main__":
    main()
