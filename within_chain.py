"""WITHIN-CHAIN localization: can the signal find the error step INSIDE a chain,
or is the pooled AUROC purely a between-chain (difficulty) effect?

CRITICAL design point: in the detection labeling (pos=first-error, neg=pre-error,
post-error EXCLUDED) the error step is always the LAST kept step, so position
trivially "localizes" it and any position-trending metric looks good within-chain.
So the within-chain localization test uses the ProcessBench-native framing:
per error chain, positive = first-error step, negatives = ALL OTHER steps in the
chain (pre AND post) -> the error sits mid-chain, position is no longer trivial.
We report POSITION and n_tok as reference rows so a geometric metric must beat them.

Columns:
  pooled_det : detection AUROC (first-error vs correct+pre-error, post excluded) -- our standard
  wc_loc     : mean over error chains of AUROC(error vs all OTHER steps in same chain),
               weighted by #other-steps -- THE within-chain localization number
  MRR        : mean 1/rank of the first-error step among all its chain's steps
  rand_MRR   : MRR a random ranker would get (sum 1/r / n averaged) -- the floor

Verdict: if geometric wc_loc ~ position/n_tok wc_loc (or ~0.5), it does not localize
within a chain beyond position/length -> useless for CUSUM/conformal/localization.
Runs on existing _coh.npz.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--min_steps", type=int, default=4,
                    help="min steps in an error chain to enter the within-chain test")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    gnames = [str(x) for x in z["geom_feature_names"]]
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    SG, SC = z["stepgeom"], z["stepcloud"]; SR = z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None
    METRICS = ["resultant", "coherence", "norm", "cloud_D", "U_D", "U_C", "position", "n_tok"]

    # ---- per-chain store: for every step, all metric values + j + is_error ----
    chains = []           # list of dicts per chain
    det_v = {m: [] for m in METRICS}; det_y = []   # detection-framing pooled
    for i in range(len(SG)):
        sg = np.asarray(SG[i], float); sc = np.asarray(SC[i], float)
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0)
        a0 = int(rng[0, 0]); T = rng.shape[0]
        ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        vals = {m: np.full(T, np.nan) for m in METRICS}
        for j in range(T):
            def cf(nm):
                return sc[j, li, cnames.index(nm)] if nm in cnames else np.nan
            vals["resultant"][j] = cf("resultant"); vals["coherence"][j] = cf("coherence")
            vals["cloud_D"][j] = cf("cloud_D")
            vals["norm"][j] = sg[j, li, gnames.index("norm")]
            vals["position"][j] = j / max(1, T - 1)
            vals["n_tok"][j] = int(rng[j, 1] - rng[j, 0] + 1)
            lo = max(0, int(rng[j, 0]) - a0); hi = min((len(ud) if ud is not None else 0), int(rng[j, 1]) - a0 + 1)
            vals["U_D"][j] = np.nanmean(ud[lo:hi]) if (ud is not None and hi > lo) else np.nan
            vals["U_C"][j] = np.nanmean(uc[lo:hi]) if (uc is not None and hi > lo) else np.nan
            # detection-framing pooled labels
            if correct or j < k:
                for m in METRICS:
                    det_v[m].append(vals[m][j])
                det_y.append(0)
            elif j == k:
                for m in METRICS:
                    det_v[m].append(vals[m][j])
                det_y.append(1)
        if not correct:
            chains.append({"vals": vals, "k": k, "T": T})
    det_y = np.asarray(det_y, int)

    print(f"file: {args.npz} | layer {args.layer} | error-chains {len(chains)} | "
          f"pooled-det steps {len(det_y)} (pos {int(det_y.sum())})")
    print(f"\n{'metric':11s} {'pooled_det':>11s} {'wc_loc':>8s} {'MRR':>7s} {'rand_MRR':>9s} "
          f"{'wc_loc(⊥nt)':>11s} {'nchains':>8s}")

    for m in METRICS:
        dv = np.asarray(det_v[m], float)
        a_det = auroc(dv, det_y); sign = 1.0 if a_det >= 0.5 else -1.0
        a_det = max(a_det, 1 - a_det)
        locs, w, mrrs, rand = [], [], [], []
        for ch in chains:
            T = ch["T"]; k = ch["k"]
            if T < args.min_steps:
                continue
            v = sign * ch["vals"][m]
            if not np.isfinite(v[k]):
                continue
            others = np.array([jj for jj in range(T) if jj != k and np.isfinite(v[jj])])
            if len(others) < 2:
                continue
            beat = np.mean(v[others] < v[k])           # AUROC(error vs others) within chain
            locs.append(beat); w.append(len(others))
            rank = 1 + int(np.sum(v[others] >= v[k]))  # rank of error among finite steps
            mrrs.append(1.0 / rank)
            n = len(others) + 1
            rand.append(np.mean([1.0 / r for r in range(1, n + 1)]))
        # length-residualized within chain: is loc beyond "error step is longer"?
        rlocs, rw = [], []
        if m not in ("position", "n_tok"):
            for ch in chains:
                T = ch["T"]; k = ch["k"]
                if T < args.min_steps:
                    continue
                mv = ch["vals"][m]; nt = ch["vals"]["n_tok"]
                fin = np.isfinite(mv) & np.isfinite(nt)
                if fin.sum() < 3 or not fin[k]:
                    continue
                b = np.polyfit(nt[fin], mv[fin], 1)
                resid = sign * (mv - (b[0] * nt + b[1]))   # higher = more error-like
                others = np.array([jj for jj in range(T) if jj != k and fin[jj]])
                if len(others) < 2:
                    continue
                rlocs.append(np.mean(resid[others] < resid[k])); rw.append(len(others))
        if locs:
            w = np.asarray(w, float)
            rl = np.average(rlocs, weights=np.asarray(rw, float)) if rlocs else float("nan")
            print(f"{m:11s} {a_det:11.3f} {np.average(locs, weights=w):8.3f} "
                  f"{np.mean(mrrs):7.3f} {np.mean(rand):9.3f} {rl:11.3f} {len(locs):8d}")
        else:
            print(f"{m:11s} {a_det:11.3f} {'nan':>8s}")

    print("\npooled_det = first-error vs correct+pre-error (post excluded), all chains mixed.")
    print("wc_loc = per error chain, frac of OTHER steps (pre+post) the first-error beats")
    print("         (0.5 = random WITHIN a chain). Compare geometric rows to 'position'/'n_tok'.")
    print("MRR vs rand_MRR: localization rank quality vs the random-ranker floor.")
    print("wc_loc(⊥nt): wc_loc after residualizing the metric on n_tok WITHIN each chain "
          "-> localization BEYOND 'error step is longer'. >0.5 = geometric loc beyond length.")
    print("\nverdict: if resultant/norm wc_loc ~ position/n_tok wc_loc or ~0.5, the signal "
          "does NOT localize the error within a chain beyond position/length -> the pooled "
          "AUROC was between-chain (difficulty), useless for within-chain monitoring.")


if __name__ == "__main__":
    main()
