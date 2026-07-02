"""ZERO-COST PRE-CHECK (highest priority, no re-extraction, no Delta).

Question it answers, on the EXISTING cumulative-state _coh.npz:
  "After within-chain z-scoring, how much of the ~0.8 cumulative norm survives?"

The pooled detection AUROC mixes two things:
  (between-chain) error chains are globally harder/longer -> higher norm = DIFFICULTY
  (within-chain)  the first-error step is locally anomalous vs its own chain
Within-chain z-scoring each chain (subtract chain mean, divide chain std over the
kept detection steps) REMOVES the between-chain level. What survives is the purely
within-chain component. If norm drops from ~0.8 to ~0.6, the headline single-layer
signal was mostly difficulty -> the application-layer foundation moves, and that
outranks the cross-layer Delta verdict (you still have norm as a fallback only if
norm itself survives within-chain).

Framing matches within_chain.py / the standard detector:
  pooled detection set = first-error step (pos) + correct-chain steps & pre-error
  steps (neg); post-error steps excluded. Within-chain z-score is computed over
  exactly this kept set per chain (correct chain: all steps; error chain: steps 0..k).

Reports raw vs within-chain-z pooled AUROC with chain-paired bootstrap CIs on each
and on the DROP (raw - wcz). Runs on existing _coh.npz.

Usage:
  python precheck_wcz.py <coh.npz> --layer 14
"""
from __future__ import annotations

import argparse
import numpy as np


def auroc(score, y):
    m = np.isfinite(score)
    s, yy = score[m], y[m]
    npos, nneg = int((yy == 1).sum()), int((yy == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    sr = s[o]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def auroc_grouped(vals, ys, gids, order):
    """AUROC over the pooled steps belonging to chains listed in `order`
    (chain indices, possibly with repeats for bootstrap)."""
    vv, yy = [], []
    for g in order:
        vv.append(vals[g])
        yy.append(ys[g])
    return auroc(np.concatenate(vv), np.concatenate(yy))


def zscore_kept(v):
    """Within-chain z-score over finite entries; returns array same length with
    nan where the original was nan or the chain has <2 finite pts or std==0."""
    out = np.full_like(v, np.nan, dtype=float)
    fin = np.isfinite(v)
    if fin.sum() < 2:
        return out
    mu = v[fin].mean()
    sd = v[fin].std()
    if sd < 1e-12:
        return out
    out[fin] = (v[fin] - mu) / sd
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--metrics", default="norm,resultant,coherence,cloud_D,U_D",
                    help="comma list from geom/cloud/U_D")
    ap.add_argument("--boot", type=int, default=2000, help="chain-paired bootstrap reps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    gnames = [str(x) for x in z["geom_feature_names"]]
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    layers = [int(x) for x in z["layers_used"]]
    if args.layer not in layers:
        raise SystemExit(f"layer {args.layer} not in layers_used={layers}")
    li = layers.index(args.layer)
    SG, SC, SR = z["stepgeom"], z["stepcloud"], z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    def metric_step_values(i):
        """Return dict metric -> per-step array (T,) of cumulative values for chain i."""
        sg = np.asarray(SG[i], float)
        sc = np.asarray(SC[i], float)
        rng = np.asarray(SR[i], int)
        T = rng.shape[0]
        a0 = int(rng[0, 0])
        ud = np.asarray(UD[i], float) if UD is not None else None
        out = {}
        for m in metrics:
            v = np.full(T, np.nan)
            if m in gnames:
                v = sg[:, li, gnames.index(m)]
            elif m in cnames:
                v = sc[:, li, cnames.index(m)]
            elif m == "U_D" and ud is not None:
                for j in range(T):
                    lo = max(0, int(rng[j, 0]) - a0)
                    hi = min(len(ud), int(rng[j, 1]) - a0 + 1)
                    v[j] = np.nanmean(ud[lo:hi]) if hi > lo else np.nan
            out[m] = np.asarray(v, float)
        return out, T

    # per chain, per metric: kept-step raw values, wcz values, labels (detection framing)
    raw = {m: [] for m in metrics}
    wcz = {m: [] for m in metrics}
    ys = []                      # per-chain label arrays (kept steps)
    npos = nneg = 0
    nchains = 0
    for i in range(len(SG)):
        vals, T = metric_step_values(i)
        k = int(ges[i])
        correct = (k < 0)
        if correct:
            keep = np.arange(T)
            lab = np.zeros(T, int)
        else:
            if k < 0 or k >= T:
                continue
            keep = np.arange(0, k + 1)          # pre-error (neg) + first-error (pos), post excluded
            lab = np.zeros(len(keep), int)
            lab[-1] = 1                          # k is the last kept step
        if len(keep) == 0:
            continue
        nchains += 1
        ys.append(lab)
        npos += int(lab.sum())
        nneg += int((lab == 0).sum())
        for m in metrics:
            vk = vals[m][keep]
            raw[m].append(vk)
            wcz[m].append(zscore_kept(vk))

    ys = [np.asarray(a, int) for a in ys]
    rng = np.random.default_rng(args.seed)
    G = len(ys)
    print(f"file: {args.npz} | layer {args.layer} | chains {nchains} | "
          f"kept steps {npos + nneg} (pos {npos}, neg {nneg})")
    print(f"\n{'metric':11s} {'raw_AUROC':>10s} {'wcz_AUROC':>10s} {'drop':>8s} "
          f"{'drop 95% CI':>20s}")
    for m in metrics:
        rv, wv = raw[m], wcz[m]
        # sign from raw pooled AUROC; apply same sign to both so wcz is comparable
        a_raw = auroc(np.concatenate(rv), np.concatenate(ys))
        sign = 1.0 if (np.isnan(a_raw) or a_raw >= 0.5) else -1.0
        rv = [sign * a for a in rv]
        wv = [sign * a for a in wv]
        a_raw = auroc_grouped(rv, ys, None, range(G))
        a_wcz = auroc_grouped(wv, ys, None, range(G))
        drops = np.empty(args.boot)
        for b in range(args.boot):
            order = rng.integers(0, G, G)
            drops[b] = (auroc_grouped(rv, ys, None, order)
                        - auroc_grouped(wv, ys, None, order))
        lo, hi = np.nanpercentile(drops, [2.5, 97.5])
        print(f"{m:11s} {a_raw:10.3f} {a_wcz:10.3f} {a_raw - a_wcz:8.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]")

    print("\nraw_AUROC = pooled detection (first-error vs correct+pre-error, post excluded).")
    print("wcz_AUROC = same, but each chain z-scored over its kept steps first")
    print("            (removes between-chain level = removes difficulty).")
    print("drop = raw - wcz, chain-paired bootstrap CI. LARGE drop (CI well above 0)")
    print("       => the pooled signal was mostly BETWEEN-chain (difficulty); the")
    print("       within-chain component is what wcz_AUROC shows. If norm wcz ~0.5-0.6,")
    print("       the headline single-layer signal does not survive within-chain ->")
    print("       fix the foundation before investing in the cross-layer Delta verdict.")


if __name__ == "__main__":
    main()
