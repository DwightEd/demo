"""Step 7: CIM-faithful metrics on single chains, under the length gate.

This consumes a spectral_field .npz produced by
`01_extract_spectral_field.py --cim_metrics`, which stores per-(step,layer):
    M_Dtle  -- TLE nonlinear intrinsic dimension (CIM Eqn. 6)
    M_Vld   -- log-det information volume         (CIM Eqn. 14)
alongside the original M_D (linear effective rank) and M_V (spectral energy).

It answers: do the CIM-FAITHFUL quantities (and especially the JOINT
"low-dimension + high-information" anchor) beat the crude mean_D=0.62 baseline,
once length is controlled?

The CIM anchor is "low D_stim AND high V". We test several chain-level encodings
of that anchor:
    mean over the band of:
        Dtle, Vld                         (the two raw CIM quantities)
        Vld / Dtle                        (info per dimension -- ratio anchor)
        Vld - lam * Dtle                  (penalized anchor, a few lam values)
        Vld / exp(eps * Dtle)             (CIM H-style, eps=0.1)
plus the originals (D, V) for direct comparison.

Every feature passes the SAME length gate as 05/06:
    raw AUROC, length baseline, length-matched AUROC, partial rho | length.

Reads only the stored matrices; no re-extraction, no model.
"""

from __future__ import annotations

import argparse
import numpy as np


# --- metrics (standalone copies, identical to 05/06) -----------------------

def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    m = ~np.isnan(scores)
    scores, labels = scores[m], labels[m]
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def auroc_bestdir(scores, labels):
    a = auroc(scores, labels)
    if np.isnan(a):
        return a, "+"
    return (a, "+") if a >= 0.5 else (1.0 - a, "-")


def partial_spearman_given_length(feature, label, n_steps):
    feature = np.asarray(feature, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    n_steps = np.asarray(n_steps, dtype=np.float64)
    m = ~np.isnan(feature)
    feature, label, n_steps = feature[m], label[m], n_steps[m]
    if feature.size < 5:
        return float("nan")

    def rankz(x):
        r = np.argsort(np.argsort(x)).astype(np.float64)
        return (r - r.mean()) / (r.std() + 1e-12)

    rf, rl, rn = rankz(feature), rankz(label), rankz(n_steps)
    rf_res = rf - (np.dot(rf, rn) / np.dot(rn, rn)) * rn
    rl_res = rl - (np.dot(rl, rn) / np.dot(rn, rn)) * rn
    denom = np.linalg.norm(rf_res) * np.linalg.norm(rl_res)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(rf_res, rl_res) / denom)


def length_matched_subset(labels, n_steps, n_bins=8, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels); n_steps = np.asarray(n_steps)
    idx = np.arange(labels.size)
    if labels.size == 0:
        return idx
    edges = np.quantile(n_steps, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-6
    keep = []
    for b in range(n_bins):
        in_bin = idx[(n_steps >= edges[b]) & (n_steps < edges[b + 1])]
        pos = in_bin[labels[in_bin] == 1]; neg = in_bin[labels[in_bin] == 0]
        k = min(pos.size, neg.size)
        if k == 0:
            continue
        keep.append(rng.choice(pos, k, replace=False))
        keep.append(rng.choice(neg, k, replace=False))
    return np.concatenate(keep) if keep else idx


def band_indices(L_sub, band):
    if band == "all":
        return np.arange(L_sub)
    if band == "deep":
        return np.arange(int(L_sub * 0.6), L_sub)
    if band == "mid":
        return np.arange(int(L_sub * 0.3), int(L_sub * 0.7))
    if band == "early":
        return np.arange(0, max(1, int(L_sub * 0.3)))
    return np.array([int(x) for x in band.split(",") if x.strip()])


def chain_scalar(M, cols, agg="mean"):
    """Band-average a (T,L) matrix over cols, then aggregate over steps."""
    M = np.asarray(M, dtype=np.float64)[:, cols]
    with np.errstate(invalid="ignore"):
        per_step = np.nanmean(M, axis=1)          # (T,)
        if agg == "mean":
            return float(np.nanmean(per_step))
        if agg == "std":
            return float(np.nanstd(per_step))
        if agg == "min":
            return float(np.nanmin(per_step))
        if agg == "max":
            return float(np.nanmax(per_step))
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="spectral_field .npz produced WITH --cim_metrics")
    ap.add_argument("--layer_band", default="all")
    ap.add_argument("--n_match_bins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.1, help="eps in CIM H-style anchor")
    ap.add_argument("--lams", default="0.1,0.5,1.0",
                    help="lambda values for the penalized anchor Vld - lam*Dtle")
    ap.add_argument("--output", default="data/cim_metrics_analysis.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if not bool(data.get("cim_stored", np.array(False))):
        raise SystemExit("This npz has no CIM metrics. Re-run 01 with --cim_metrics.")

    labels_raw = data["labels"].astype(int)
    n_steps = data["n_steps"].astype(int)
    M_D = data["M_D"]; M_V = data["M_V"]
    M_Dtle = data["M_Dtle"]; M_Vld = data["M_Vld"]
    N = len(labels_raw)
    y = (labels_raw >= 0).astype(int)

    L_sub = M_D[0].shape[1]
    cols = band_indices(L_sub, args.layer_band)
    lams = [float(x) for x in args.lams.split(",") if x.strip()]
    print(f"Loaded {N} chains, L_sub={L_sub}, band={args.layer_band} -> cols {list(cols)}")
    print(f"  labels: {int(y.sum())} error / {int((1-y).sum())} correct")

    # Build per-chain feature dict.
    feats = {}  # name -> (N,) array
    def add(name, fn):
        feats[name] = np.array([fn(i) for i in range(N)], dtype=np.float64)

    # originals (for direct contrast)
    add("mean_D(linear)",  lambda i: chain_scalar(M_D[i], cols, "mean"))
    add("mean_V(energy)",  lambda i: chain_scalar(M_V[i], cols, "mean"))
    # CIM-faithful raw
    add("mean_Dtle",       lambda i: chain_scalar(M_Dtle[i], cols, "mean"))
    add("mean_Vld",        lambda i: chain_scalar(M_Vld[i], cols, "mean"))
    add("std_Dtle",        lambda i: chain_scalar(M_Dtle[i], cols, "std"))
    add("std_Vld",         lambda i: chain_scalar(M_Vld[i], cols, "std"))

    # JOINT anchors: "low D + high V"
    def ratio(i):
        d = chain_scalar(M_Dtle[i], cols, "mean")
        v = chain_scalar(M_Vld[i], cols, "mean")
        return v / d if (d and np.isfinite(d) and d > 1e-9) else np.nan
    add("Vld_over_Dtle", ratio)

    def hstyle(i):
        d = chain_scalar(M_Dtle[i], cols, "mean")
        v = chain_scalar(M_Vld[i], cols, "mean")
        if not (np.isfinite(d) and np.isfinite(v)):
            return np.nan
        return v / np.exp(args.eps * d)
    add(f"Vld/exp({args.eps}*Dtle)", hstyle)

    for lam in lams:
        def penalized(i, lam=lam):
            d = chain_scalar(M_Dtle[i], cols, "mean")
            v = chain_scalar(M_Vld[i], cols, "mean")
            if not (np.isfinite(d) and np.isfinite(v)):
                return np.nan
            return v - lam * d
        add(f"Vld-{lam}*Dtle", penalized)

    # length gate
    a_len, dir_len = auroc_bestdir(n_steps.astype(float), y)
    sub = length_matched_subset(y, n_steps, n_bins=args.n_match_bins, seed=args.seed)
    a_len_sub, _ = auroc_bestdir(n_steps[sub].astype(float), y[sub])
    print("\n=== Length baseline (the thing to beat) ===")
    print(f"  AUROC(n_steps)              = {a_len:.4f}  (dir {dir_len})")
    print(f"  matched: {sub.size} chains, AUROC(n_steps|matched) = {a_len_sub:.4f}  (~0.5)")

    print("\n=== CIM metrics & joint anchors (RAW vs MATCHED vs partial-rho) ===")
    print(f"{'feature':24s}  {'rawAUROC':>9s} {'dir':>3s}  {'matchAUROC':>10s}  {'partial_rho|len':>15s}")
    results = {}
    for name, col in feats.items():
        a_raw, d_raw = auroc_bestdir(col, y)
        a_sub, _ = auroc_bestdir(col[sub], y[sub])
        prho = partial_spearman_given_length(col, y, n_steps)
        results[name] = dict(raw=a_raw, dir=d_raw, matched=a_sub, partial_rho=prho)
        flag = "  <==" if (not np.isnan(a_raw)) and a_sub > 0.62 else ""
        print(f"{name:24s}  {a_raw:9.4f} {d_raw:>3s}  {a_sub:10.4f}  {prho:15.4f}{flag}")

    np.savez(args.output,
             feat_names=np.array(list(feats.keys()), dtype=object),
             feat_mat=np.array([feats[k] for k in feats], dtype=np.float64).T,
             y=y, n_steps=n_steps,
             length_auroc=np.array(a_len),
             length_auroc_matched=np.array(a_len_sub),
             results=np.array(results, dtype=object),
             layer_band=np.array(args.layer_band))
    print(f"\nSaved -> {args.output}")

    # verdict: did any CIM-faithful feature beat the mean_D(linear) baseline?
    base = results["mean_D(linear)"]["matched"]
    best = max(((n, r["matched"]) for n, r in results.items()
                if np.isfinite(r["matched"])), key=lambda t: t[1])
    print("\n=== VERDICT ===")
    print(f"  linear mean_D matched AUROC (the 0.62 baseline) = {base:.4f}")
    print(f"  best feature here: {best[0]} = {best[1]:.4f}")
    if best[1] > base + 0.02:
        print("  -> CIM-faithful metric BEATS linear mean_D. The 'low-dim + non-degenerate'")
        print(f"     anchor adds signal. Promote '{best[0]}' to the next stage.")
    else:
        print("  -> No CIM-faithful metric clearly beats linear mean_D on this band.")
        print("     The anchor's single-chain signal is capped near the mean_D level;")
        print("     try other bands, or move to a learned aggregator (trajectory model).")


if __name__ == "__main__":
    main()