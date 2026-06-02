"""Step 5: Orientation- and position-drift analysis, WITH length controls.

This consumes a spectral_field .npz produced by
`01_extract_spectral_field.py --store_geometry`. It tests whether the
geometry that effective rank throws away -- the cloud's POSITION (centroid)
and ORIENTATION (principal axes) -- carries a chain-level / step-level signal
for error detection that survives a length control.

The central discipline here is the LENGTH GATE. Error chains tend to be longer,
and any "step-to-step variability" feature mechanically grows with the number
of steps. So for every chain-level feature we report:

    AUROC(feature)             raw
    AUROC(n_steps)             pure length baseline
    AUROC(feature)             on a length-matched subset
    partial Spearman(feature, label | n_steps)   length-controlled association

A feature is only interesting if it beats the length baseline AND retains
its AUROC on the length-matched subset. Otherwise it is a length proxy.

Layer selection: by default we aggregate over a deep-layer band (the comprehensive
viz suggested ΔD discriminability lives in deep layers), but --layer_band lets you
change it.

Outputs a small npz + prints a decision table. No figures (keep it fast); pipe
into 03/04-style plotting later if a feature survives.
"""

from __future__ import annotations

import argparse
import numpy as np

from utils.geometry import (
    centroid_step_drift,
    centroid_curvature,
    principal_angle_drift,
    gaussian_w2_step_drift,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank (Mann-Whitney) identity. labels in {0,1}.

    Returns the *orientation-free* best of AUROC and 1-AUROC so a feature that
    is predictive in either direction is credited; we report which direction
    separately.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    m = ~np.isnan(scores)
    scores, labels = scores[m], labels[m]
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    sum_pos = ranks[labels == 1].sum()
    a = (sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(a)


def auroc_bestdir(scores, labels):
    a = auroc(scores, labels)
    if np.isnan(a):
        return a, "+"
    if a >= 0.5:
        return a, "+"
    return 1.0 - a, "-"


def partial_spearman_given_length(feature, label, n_steps):
    """Spearman correlation between feature and label after regressing out
    n_steps from both (rank-based partial correlation). One number; |rho| near
    0 means the feature's association with the label is explained by length.
    """
    feature = np.asarray(feature, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    n_steps = np.asarray(n_steps, dtype=np.float64)
    m = ~np.isnan(feature)
    feature, label, n_steps = feature[m], label[m], n_steps[m]
    if feature.size < 5:
        return float("nan")

    def rankz(x):
        r = np.argsort(np.argsort(x)).astype(np.float64)
        r = (r - r.mean()) / (r.std() + 1e-12)
        return r

    rf, rl, rn = rankz(feature), rankz(label), rankz(n_steps)
    # residualize feature and label on length
    rf_res = rf - (np.dot(rf, rn) / np.dot(rn, rn)) * rn
    rl_res = rl - (np.dot(rl, rn) / np.dot(rn, rn)) * rn
    denom = (np.linalg.norm(rf_res) * np.linalg.norm(rl_res))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(rf_res, rl_res) / denom)


def length_matched_subset(labels, n_steps, n_bins=8, seed=0):
    """Return indices of a length-matched subset: within each length bin, keep
    an equal number of correct and error chains (downsample the majority).
    """
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
# Feature extraction from one trajectory's geometry
# ---------------------------------------------------------------------------

def band_indices(L_sub, band):
    """Resolve a layer band spec into column indices."""
    if band == "all":
        return np.arange(L_sub)
    if band == "deep":
        return np.arange(int(L_sub * 0.6), L_sub)
    if band == "mid":
        return np.arange(int(L_sub * 0.3), int(L_sub * 0.7))
    if band == "early":
        return np.arange(0, max(1, int(L_sub * 0.3)))
    # explicit comma list
    return np.array([int(x) for x in band.split(",") if x.strip()])


def chain_drift_features(mu, eigvals, eigvecs, cols):
    """Compute per-step drift series for one trajectory over a layer band, then
    summarize each into chain-level scalars.

    mu:      (T, L, p)
    eigvals: (T, L, k)
    eigvecs: (T, L, p, k)
    cols:    layer indices to average over.

    Returns dict of chain-level features and the raw per-step series (averaged
    over the band) for the step-level test.
    """
    T = mu.shape[0]
    # per-layer drift series, then average across the band
    cen_drift_band = []
    cen_curv_band = []
    pang_drift_band = []
    w2_drift_band = []
    for l in cols:
        mus_l = mu[:, l, :]                       # (T, p)
        eva_l = [eigvals[j, l, :] for j in range(T)]
        eve_l = [eigvecs[j, l, :, :] for j in range(T)]
        cen_drift_band.append(centroid_step_drift(mus_l))
        cen_curv_band.append(centroid_curvature(mus_l))
        pang_drift_band.append(principal_angle_drift(eve_l, reduce="mean"))
        w2_drift_band.append(gaussian_w2_step_drift(mus_l, eva_l, eve_l))

    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cen_drift = np.nanmean(np.stack(cen_drift_band, 0), axis=0)   # (T,)
            cen_curv = np.nanmean(np.stack(cen_curv_band, 0), axis=0)
            pang_drift = np.nanmean(np.stack(pang_drift_band, 0), axis=0)
            w2_drift = np.nanmean(np.stack(w2_drift_band, 0), axis=0)

    feats = {
        # position dynamics
        "centroid_drift_mean": np.nanmean(cen_drift),
        "centroid_drift_std": np.nanstd(cen_drift),
        "centroid_drift_max": np.nanmax(cen_drift) if np.any(~np.isnan(cen_drift)) else np.nan,
        "centroid_curv_mean": np.nanmean(cen_curv),
        "centroid_curv_max": np.nanmax(cen_curv) if np.any(~np.isnan(cen_curv)) else np.nan,
        # orientation dynamics
        "pangle_drift_mean": np.nanmean(pang_drift),
        "pangle_drift_std": np.nanstd(pang_drift),
        "pangle_drift_max": np.nanmax(pang_drift) if np.any(~np.isnan(pang_drift)) else np.nan,
        # combined
        "w2_drift_mean": np.nanmean(w2_drift),
        "w2_drift_std": np.nanstd(w2_drift),
        "w2_drift_max": np.nanmax(w2_drift) if np.any(~np.isnan(w2_drift)) else np.nan,
    }
    series = {
        "centroid_drift": cen_drift,
        "pangle_drift": pang_drift,
        "w2_drift": w2_drift,
    }
    return feats, series


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="spectral_field .npz produced WITH --store_geometry")
    ap.add_argument("--layer_band", default="deep",
                    help='"all" | "deep" | "mid" | "early" | comma list of indices')
    ap.add_argument("--n_match_bins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/geometry_analysis.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if not bool(data.get("geom_stored", np.array(False))):
        raise SystemExit(
            "This npz has no geometry. Re-run 01 with --store_geometry first.")

    labels_raw = data["labels"].astype(int)          # -1 = correct, >=0 = first-error step
    n_steps = data["n_steps"].astype(int)
    geom_mu = data["geom_mu"]
    geom_eigvals = data["geom_eigvals"]
    geom_eigvecs = data["geom_eigvecs"]
    N = len(labels_raw)

    # chain-level binary label: 1 = has an error, 0 = fully correct
    y_chain = (labels_raw >= 0).astype(int)

    L_sub = geom_mu[0].shape[1]
    cols = band_indices(L_sub, args.layer_band)
    print(f"Loaded {N} trajectories, L_sub={L_sub}, band={args.layer_band} -> cols {list(cols)}")
    print(f"  chain labels: {int(y_chain.sum())} error / {int((1-y_chain).sum())} correct")

    # ---- per-chain features + collect step-level series for step test ----
    feat_names = None
    feat_mat = []
    step_records = []  # (drift_value, is_first_error) per step, for step-level AUROC
    for i in range(N):
        mu = np.asarray(geom_mu[i], dtype=np.float64)
        eva = np.asarray(geom_eigvals[i], dtype=np.float64)
        eve = np.asarray(geom_eigvecs[i], dtype=np.float64)
        feats, series = chain_drift_features(mu, eva, eve, cols)
        if feat_names is None:
            feat_names = list(feats.keys())
        feat_mat.append([feats[k] for k in feat_names])

        # step-level: only meaningful for error chains (which step is the bad one)
        if labels_raw[i] >= 0:
            tau = labels_raw[i]
            for sig_name, sig in series.items():
                for j in range(len(sig)):
                    if np.isnan(sig[j]):
                        continue
                    step_records.append((sig_name, sig[j], int(j == tau)))

    feat_mat = np.asarray(feat_mat, dtype=np.float64)

    # ---- length baseline ----
    a_len, dir_len = auroc_bestdir(n_steps.astype(float), y_chain)
    print("\n=== Length baseline (the thing to beat) ===")
    print(f"  AUROC(n_steps)              = {a_len:.4f}   (dir {dir_len})")

    # ---- length-matched subset ----
    sub = length_matched_subset(y_chain, n_steps,
                                n_bins=args.n_match_bins, seed=args.seed)
    a_len_sub, _ = auroc_bestdir(n_steps[sub].astype(float), y_chain[sub])
    print(f"  length-matched subset: {sub.size} chains, "
          f"AUROC(n_steps|matched) = {a_len_sub:.4f}  (should be ~0.5)")

    # ---- chain-level feature table with length controls ----
    print("\n=== Chain-level features (RAW vs LENGTH-MATCHED vs partial-rho) ===")
    print(f"{'feature':24s}  {'rawAUROC':>9s} {'dir':>3s}  "
          f"{'matchAUROC':>10s}  {'partial_rho|len':>15s}")
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
        print(f"{name:24s}  {a_raw:9.4f} {d_raw:>3s}  {a_sub:10.4f}  "
              f"{prho:15.4f}{flag}")

    # ---- step-level test (first-error step vs other steps), per signal ----
    print("\n=== Step-level: first-error step vs other steps ===")
    step_results = {}
    if step_records:
        for sig_name in sorted(set(r[0] for r in step_records)):
            vals = np.array([r[1] for r in step_records if r[0] == sig_name])
            labs = np.array([r[2] for r in step_records if r[0] == sig_name])
            a, d = auroc_bestdir(vals, labs)
            n_pos = int(labs.sum())
            n_neg = int((1 - labs).sum())
            step_results[sig_name] = dict(auroc=a, dir=d, n_pos=n_pos, n_neg=n_neg)
            print(f"  {sig_name:18s}  AUROC = {a:.4f} (dir {d})   "
                  f"[{n_pos} first-error / {n_neg} other]")
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
        matched_idx=np.array(sub),
        chain_results=np.array(results, dtype=object),
        step_results=np.array(step_results, dtype=object),
        layer_band=np.array(args.layer_band),
    )
    print(f"\nSaved -> {args.output}")

    # ---- one-line verdict ----
    survivors = [n for n, r in results.items()
                 if (not np.isnan(r["raw"])) and r["raw"] > a_len + 0.03
                 and r["matched"] > 0.58]
    print("\n=== VERDICT ===")
    if survivors:
        print("  Features that beat length AND hold up length-matched:")
        for s in survivors:
            print(f"    - {s}: matched AUROC = {results[s]['matched']:.3f}, "
                  f"partial_rho|len = {results[s]['partial_rho']:.3f}")
        print("  -> worth promoting to a forward-dynamics model (see discussion).")
    else:
        print("  No geometry feature cleanly beats the length baseline on this")
        print("  subset. Either the signal is length, or the band/dataset is wrong.")
        print("  Try --layer_band mid/all, or a different subset, before proceeding.")


if __name__ == "__main__":
    main()