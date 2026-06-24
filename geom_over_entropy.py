"""Difficulty + length controlled: does GEOMETRY add over ENTROPY?
Within each error chain (same problem), residualize kappa and entropy on step length,
then localize the first-error step. Compare: entropy alone vs geometry alone vs entropy+geometry
(equal z-weight, fixed/unsupervised -> no leak). If combined > entropy, geometry is a clean increment."""
from __future__ import annotations
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); ap.add_argument("--min_steps", type=int, default=4); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]; lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    fi = cn.index("resultant"); SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int); UD = z["tok_U_D"]

    def edis(H, w=8, tb=1.36, tr=1.33):
        H = np.asarray(H, float); H = H[np.isfinite(H)]
        if len(H) < 3:
            return np.nan
        ww = min(w, max(2, len(H) // 2))
        burst = sum(1 for t in range(len(H) - ww) if H[t + ww] - H[t] > tb) if len(H) > ww else 0
        reb = 0; rmin = H[0]
        for t in range(1, len(H)):
            if H[t] - rmin > tr:
                reb += 1
            rmin = min(rmin, H[t])
        return 0.5 * (burst + reb) * (1.0 + float(H.var()))

    chains = []   # per error chain: dict with resid_ka, resid_ed, k
    allka, alled = [], []
    for i in range(len(SC)):
        k = int(ges[i])
        if k < 1:
            continue
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); T = rng.shape[0]; a0 = int(rng[0, 0]); ud = np.asarray(UD[i], float)
        if T < args.min_steps:
            continue
        ka = sc[:T, li, fi]; nt = (rng[:T, 1] - rng[:T, 0] + 1).astype(float)
        ed = np.array([edis(ud[max(0, int(rng[j, 0]) - a0):min(len(ud), int(rng[j, 1]) - a0 + 1)]) for j in range(T)])
        fin = np.isfinite(ka) & np.isfinite(ed) & np.isfinite(nt)
        if fin.sum() < 3 or not fin[k]:
            continue
        bk = np.polyfit(nt[fin], ka[fin], 1); be = np.polyfit(nt[fin], ed[fin], 1)
        rka = -(ka - (bk[0] * nt + bk[1]))   # higher = more error (error has LOW kappa)
        red = (ed - (be[0] * nt + be[1]))    # higher = more error (error has HIGH entropy)
        chains.append({"rka": rka, "red": red, "k": k, "fin": fin})
        allka.append(rka[fin]); alled.append(red[fin])
    mk, sk = np.concatenate(allka).mean(), np.concatenate(allka).std() + 1e-9
    me, se = np.concatenate(alled).mean(), np.concatenate(alled).std() + 1e-9

    def wcloc(scorer):
        num = den = 0.0
        for ch in chains:
            k = ch["k"]; fin = ch["fin"]; s = scorer(ch)
            others = np.array([j for j in range(len(s)) if j != k and fin[j]])
            if len(others) < 2:
                continue
            num += np.sum(s[others] < s[k]) + 0.5 * np.sum(s[others] == s[k]); den += len(others)
        return num / den

    z_ed = lambda ch: (ch["red"] - me) / se
    z_ka = lambda ch: (ch["rka"] - mk) / sk
    print(f"{args.npz} | L{args.layer} | error-chains {len(chains)} | within-chain + length-controlled localization")
    print(f"  entropy only          wc_loc {wcloc(z_ed):.3f}")
    print(f"  geometry only         wc_loc {wcloc(z_ka):.3f}")
    print(f"  entropy + geometry    wc_loc {wcloc(lambda ch: z_ed(ch) + z_ka(ch)):.3f}")


if __name__ == "__main__":
    main()
