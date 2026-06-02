"""Step 6: Does the *effective-rank divergence* you see in the aggregate plots
survive a length control?

Motivation
----------
The (step x layer) heatmaps and the comprehensive AUROC bar chart suggest error
chains have a more "divergent" effective rank: features like std(dD),
D_range = max-min, mean|dD| land around AUROC 0.63 -- clearly above the static
mean_D (~0.57) and above the geometry features (centroid / principal-angle /
W2, all ~0.5x in step 05).

But those divergence features grow *mechanically* with the number of steps: a
longer chain has a longer difference series, so its std / range / max are
larger for free. So an 0.63 AUROC could be a length proxy in disguise.

This script puts the D-divergence features through the SAME length gate used in
05_geometry_analysis.py:

    raw AUROC          uncontrolled
    AUROC(n_steps)     pure length baseline (the thing to beat)
    matched AUROC      on a length-matched subset (correct/error balanced per length bin)
    partial rho|len    association with the label after regressing out length

A feature is a *real* signal only if it beats the length baseline AND keeps its
AUROC on the matched subset AND has a non-trivial partial correlation. Otherwise
the "divergence" you see in the plots is length.

It reads only M_D from the spectral_field npz (no geometry, no re-extraction).
It also reports per-step-position results so you can see whether the divergence
is concentrated in late steps (which would be the length story).
"""

from __future__ import annotations

import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Metrics (same definitions as 05, copied so this script is standalone)
# ---------------------------------------------------------------------------

def auroc(scores, labels):
    """AUROC via the Mann-Whitney rank identity. labels in {0,1}."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    m = ~np.isnan(scores)
    scores, labels = scores[m], labels[m]
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
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
    """Rank partial correlation between feature and label, controlling n_steps."""
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
    """Within each length bin, keep equal numbers of correct and error chains."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n_steps = np.asarray(n_steps)
    idx = np.arange(labels.size)
    if labels.size == 0:
        return idx
    edges = np.quantile(n_steps, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6
    keep = []
    for b in range(n_bins):
        in_bin = idx[(n_steps >= edges[b]) & (n_steps < edges[b + 1])]
        pos = in_bin[labels[in_bin] == 1]
        neg = in_bin[labels[in_bin] == 0]
        k = min(pos.size, neg.size)
        if k == 0:
            continue
        keep.append(rng.choice(pos, k, replace=False))
        keep.append(rng.choice(neg, k, replace=False))
    if not keep:
        return idx
    return np.concatenate(keep)


# ---------------------------------------------------------------------------
# Layer band + D-divergence features
# ---------------------------------------------------------------------------

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


def d_dynamics_features(M_D, cols):
    """Compute chain-level D-divergence features over a layer band.

    M_D: (T, L) effective-rank field for one chain.
    cols: layer indices to average over.

    Returns dict of chain-level scalars plus the per-step |dD| series (band-avg)
    for the step-localization test.
    """
    M = np.asarray(M_D, dtype=np.float64)[:, cols]     # (T, |cols|)
    T = M.shape[0]
    # step-axis difference, per layer, then average across the band
    dD = np.diff(M, axis=0)                             # (T-1, |cols|)
    d_layeravg = np.nanmean(M, axis=1)                  # (T,)  D averaged over band, per step
    dD_layeravg = np.diff(d_layeravg)                   # (T-1,) step-diff of band-avg D
    abs_dD = np.abs(dD_layeravg)

    with np.errstate(invalid="ignore"):
        feats = {
            # static (for reference / contrast)
            "mean_D": float(np.nanmean(d_layeravg)),
            "std_D": float(np.nanstd(d_layeravg)),
            "D_last_over_first": float(d_layeravg[-1] / d_layeravg[0])
                if d_layeravg[0] not in (0.0, np.nan) else np.nan,
            # divergence / dynamics (the ~0.63 suspects)
            "std_dD": float(np.nanstd(dD_layeravg)) if dD_layeravg.size else np.nan,
            "D_range": float(np.nanmax(d_layeravg) - np.nanmin(d_layeravg)),
            "mean_abs_dD": float(np.nanmean(abs_dD)) if abs_dD.size else np.nan,
            "max_abs_dD": float(np.nanmax(abs_dD)) if abs_dD.size else np.nan,
            # cross-layer divergence at fixed step, averaged over steps
            "cross_layer_std_D": float(np.nanmean(np.nanstd(M, axis=1))),
        }
    series_abs_dD = abs_dD          # (T-1,)
    # fractional position of each diff (0..1) for the late-step diagnosis
    if abs_dD.size:
        frac = (np.arange(1, T) ) / max(T - 1, 1)
    else:
        frac = np.array([])
    return feats, series_abs_dD, frac


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="spectral_field .npz (M_D must be present)")
    ap.add_argument("--layer_band", default="all",
                    help='"all" | "deep" | "mid" | "early" | comma list')
    ap.add_argument("--n_match_bins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/d_dynamics_analysis.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if "M_D" not in data:
        raise SystemExit("This npz has no M_D field.")
    M_D_all = data["M_D"]                       # object array of (T, L)
    labels_raw = data["labels"].astype(int)     # -1 = correct, >=0 = first-error step
    n_steps = data["n_steps"].astype(int)
    N = len(labels_raw)
    y_chain = (labels_raw >= 0).astype(int)     # 1 = has error, 0 = correct

    L_sub = M_D_all[0].shape[1]
    cols = band_indices(L_sub, args.layer_band)
    print(f"Loaded {N} chains, L_sub={L_sub}, band={args.layer_band} -> cols {list(cols)}")
    print(f"  chain labels: {int(y_chain.sum())} error / {int((1-y_chain).sum())} correct")

    feat_names = None
    feat_mat = []
    step_recs = []   # (abs_dD value, is_first_error, frac_position)
    for i in range(N):
        feats, sabs, frac = d_dynamics_features(M_D_all[i], cols)
        if feat_names is None:
            feat_names = list(feats.keys())
        feat_mat.append([feats[k] for k in feat_names])
        if labels_raw[i] >= 0:
            tau = labels_raw[i]
            T = M_D_all[i].shape[0]
            for j in range(sabs.size):
                if np.isnan(sabs[j]):
                    continue
                # diff index j corresponds to step (j+1); first-error step = tau
                step_recs.append((sabs[j], int((j + 1) == tau), frac[j]))
    feat_mat = np.asarray(feat_mat, dtype=np.float64)

    # length baseline
    a_len, dir_len = auroc_bestdir(n_steps.astype(float), y_chain)
    sub = length_matched_subset(y_chain, n_steps, n_bins=args.n_match_bins, seed=args.seed)
    a_len_sub, _ = auroc_bestdir(n_steps[sub].astype(float), y_chain[sub])
    print("\n=== Length baseline (the thing to beat) ===")
    print(f"  AUROC(n_steps)              = {a_len:.4f}  (dir {dir_len})")
    print(f"  matched subset: {sub.size} chains, AUROC(n_steps|matched) = {a_len_sub:.4f}  (~0.5 expected)")

    print("\n=== D-divergence features (RAW vs LENGTH-MATCHED vs partial-rho) ===")
    print(f"{'feature':22s}  {'rawAUROC':>9s} {'dir':>3s}  {'matchAUROC':>10s}  {'partial_rho|len':>15s}")
    results = {}
    for c, name in enumerate(feat_names):
        col = feat_mat[:, c]
        a_raw, d_raw = auroc_bestdir(col, y_chain)
        a_sub, _ = auroc_bestdir(col[sub], y_chain[sub])
        prho = partial_spearman_given_length(col, y_chain, n_steps)
        results[name] = dict(raw=a_raw, dir=d_raw, matched=a_sub, partial_rho=prho)
        flag = ""
        if (not np.isnan(a_raw)) and a_raw > a_len + 0.03 and a_sub > 0.58:
            flag = "  <== survives length"
        print(f"{name:22s}  {a_raw:9.4f} {d_raw:>3s}  {a_sub:10.4f}  {prho:15.4f}{flag}")

    # step-level: is |dD| elevated at the first-error step?
    print("\n=== Step-level: |dD| at first-error step vs other steps ===")
    if step_recs:
        vals = np.array([r[0] for r in step_recs])
        labs = np.array([r[1] for r in step_recs])
        fracs = np.array([r[2] for r in step_recs])
        a, d = auroc_bestdir(vals, labs)
        print(f"  abs_dD            AUROC = {a:.4f} (dir {d})   "
              f"[{int(labs.sum())} first-error / {int((1-labs).sum())} other]")
        # late-step diagnosis: are first-error steps simply later in the chain?
        a_frac, d_frac = auroc_bestdir(fracs, labs)
        print(f"  frac_position     AUROC = {a_frac:.4f} (dir {d_frac})   "
              f"<- if high, 'divergence at error step' is really 'error steps are late'")
    else:
        print("  (no error chains with usable steps)")

    np.savez(
        args.output,
        feat_names=np.array(feat_names, dtype=object),
        feat_mat=feat_mat,
        y_chain=y_chain,
        n_steps=n_steps,
        length_auroc=np.array(a_len),
        length_auroc_matched=np.array(a_len_sub),
        results=np.array(results, dtype=object),
        layer_band=np.array(args.layer_band),
    )
    print(f"\nSaved -> {args.output}")

    survivors = [n for n, r in results.items()
                 if (not np.isnan(r["raw"])) and r["raw"] > a_len + 0.03
                 and r["matched"] > 0.58]
    print("\n=== VERDICT ===")
    if survivors:
        print("  D-divergence features that beat length AND hold up matched:")
        for s in survivors:
            print(f"    - {s}: matched AUROC = {results[s]['matched']:.3f}, "
                  f"partial_rho|len = {results[s]['partial_rho']:.3f}")
        print("  -> the 'error chains diverge more' signal is REAL, not length.")
    else:
        print("  No D-divergence feature survives the length gate.")
        print("  -> the divergence you see in the aggregate plots is LENGTH, not a")
        print("     length-independent error signal. effective-rank dynamics is out.")


if __name__ == "__main__":
    main()