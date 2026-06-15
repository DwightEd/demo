"""WITHIN-CHAIN localization test: is the step-level signal usable for finding the
error step INSIDE a single chain, or is the pooled AUROC entirely a between-chain
(difficulty) effect?

Pooled AUROC mixes all chains' steps -- hard chains can be globally more diffuse
AND have the errors, inflating pooled AUROC while being useless for within-chain
monitoring (CUSUM / conformal / first-error localization are all within-chain).

For each metric we report:
  (1) pooled AUROC (best dir)                       -- what we have been reporting
  (2) within-chain z-scored pooled AUROC            -- subtract chain mean / std, then pool
                                                       (removes between-chain offset)
  (3) within-ERROR-chain localization AUROC          -- per error chain, first-error step vs
      (avg, weighted)                                  its own pre-error steps; THE test
  (4) first-error MRR within error chains            -- mean 1/rank of the first-error step

Verdict (user's bar): if (2)/(3) collapse toward 0.5 while (1) is ~0.7, the signal
is between-chain only -> cannot localize within a chain -> useless for the app.

Runs on existing _coh.npz (resultant/coherence/norm/cloud_D + U_D/U_C at a layer).
"""

from __future__ import annotations

import argparse
import numpy as np


def auroc(score, y):
    m = np.isfinite(score); s, yy = score[m], y[m]
    npos, nneg = int((yy == 1).sum()), int((yy == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sr = s[o]; i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def within_chain_z(score, G):
    z = np.full(len(score), np.nan)
    for c in np.unique(G):
        m = G == c; v = score[m]
        mu, sd = np.nanmean(v), np.nanstd(v)
        z[m] = (v - mu) / (sd + 1e-9)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--min_pre", type=int, default=3,
                    help="min pre-error steps for a chain to enter within-chain test")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    gnames = [str(x) for x in z["geom_feature_names"]]
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    SG, SC = z["stepgeom"], z["stepcloud"]; SR = z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None

    def cv(nm, sc): return sc[:, li, cnames.index(nm)] if nm in cnames else None
    def gv(nm, sg): return sg[:, li, gnames.index(nm)] if nm in gnames else None

    feats = {k: [] for k in ["resultant", "coherence", "norm", "cloud_D", "U_D", "U_C", "n_tok"]}
    Y, G, KIDX = [], [], []     # KIDX: per-step index j (for MRR), and error-chain grouping
    chain_of = []
    for i in range(len(SG)):
        sg = np.asarray(SG[i], float); sc = np.asarray(SC[i], float)
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0)
        a0 = int(rng[0, 0]); T = rng.shape[0]
        ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        for j in range(T):
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                continue
            feats["resultant"].append(sc[j, li, cnames.index("resultant")] if "resultant" in cnames else np.nan)
            feats["coherence"].append(sc[j, li, cnames.index("coherence")] if "coherence" in cnames else np.nan)
            feats["cloud_D"].append(sc[j, li, cnames.index("cloud_D")] if "cloud_D" in cnames else np.nan)
            feats["norm"].append(sg[j, li, gnames.index("norm")])
            feats["n_tok"].append(int(rng[j, 1] - rng[j, 0] + 1))
            lo = max(0, int(rng[j, 0]) - a0); hi = min((len(ud) if ud is not None else 0), int(rng[j, 1]) - a0 + 1)
            feats["U_D"].append(np.nanmean(ud[lo:hi]) if (ud is not None and hi > lo) else np.nan)
            feats["U_C"].append(np.nanmean(uc[lo:hi]) if (uc is not None and hi > lo) else np.nan)
            Y.append(y); G.append(i); KIDX.append(j); chain_of.append(not correct)
    for k in feats:
        feats[k] = np.asarray(feats[k], float)
    Y = np.asarray(Y, int); G = np.asarray(G, int); KIDX = np.asarray(KIDX, int)
    is_err_chain = np.asarray(chain_of, bool)

    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())} "
          f"| chains {len(np.unique(G))}")
    print(f"\n{'metric':11s} {'(1)pooled':>10s} {'(2)wc-z':>9s} {'(3)wc-loc':>10s} {'(4)MRR':>8s} {'nchains':>8s}")

    for nm in ["resultant", "coherence", "norm", "cloud_D", "U_D", "U_C", "n_tok"]:
        v = feats[nm]
        a_raw = auroc(v, Y); sign = 1.0 if a_raw >= 0.5 else -1.0
        a1 = max(a_raw, 1 - a_raw)
        a2 = auroc(sign * within_chain_z(v, G), Y); a2 = max(a2, 1 - a2)
        # (3)+(4): per error chain, first-error vs its own pre-error steps
        locs, mrrs, w = [], [], []
        for c in np.unique(G[is_err_chain]):
            m = G == c
            jj = KIDX[m]; yy = Y[m]; vv = sign * v[m]      # higher = more error-like
            kpos = np.where(yy == 1)[0]
            if len(kpos) != 1:
                continue
            pre = np.where(yy == 0)[0]
            if len(pre) < args.min_pre:
                continue
            pv = vv[kpos[0]]
            beat = np.mean(vv[pre] < pv)                    # fraction of pre-error beaten
            locs.append(beat); w.append(len(pre))
            rank = 1 + int(np.sum(vv[pre] >= pv))           # rank of first-error (1=best)
            mrrs.append(1.0 / rank)
        if locs:
            w = np.asarray(w, float)
            loc = np.average(locs, weights=w); mrr = np.mean(mrrs); nch = len(locs)
        else:
            loc = mrr = float("nan"); nch = 0
        print(f"{nm:11s} {a1:10.3f} {a2:9.3f} {loc:10.3f} {mrr:8.3f} {nch:8d}")

    print("\n(1) pooled = all steps mixed (between+within chain).")
    print("(2) within-chain z: subtract chain mean/std then pool -> removes between-chain.")
    print("(3) within-chain loc: per error chain, frac of its OWN pre-error steps the "
          "first-error beats (0.5=random WITHIN a chain). THE localization test.")
    print("(4) MRR: mean 1/rank of first-error among {pre-error+itself}; random ~ "
          "mean(1/(n+1)/2)-ish.")
    print("\nverdict: if (2)/(3) ~0.5 while (1) ~0.7, signal is between-chain (difficulty) "
          "only -> cannot localize the error WITHIN a chain -> useless for CUSUM/conformal.")


if __name__ == "__main__":
    main()
