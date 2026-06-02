"""Step 8: step-vector activation-participation analysis, under the length gate.

This consumes a spectral_field .npz produced by
`01_extract_spectral_field.py --step_vectors`, which stores, per weighting mode
m in {last, mean, linear, step_exp}:

    sv_pr_<m>       object array: one (T, L_sub) matrix per chain, the
                    participation ratio of each step's aggregated vector
    sv_ae_<m>       object array: one (T, L_sub) matrix per chain, the
                    activation entropy of each step's aggregated vector
    sv_out_entropy  object array: one (T,) per chain, the per-step output-token
                    entropy (model predictive uncertainty), layer-independent

The aggregation (token cloud -> one vector) and the PR/AE measurement already
happened at extraction time (see utils/step_vector.py). This script ONLY reads
the stored matrices; no re-extraction, no model.

It answers two questions:

  (a) DISCRIMINATION -- does activation participation, encoded as a chain-level
      scalar (band-average over layers, then average over steps), separate error
      chains from correct ones BEYOND what chain length alone explains? Reported
      as matchAUROC (length-matched AUROC, "the number that matters") for each of
      the four weighting modes. step_exp is the Streaming-HD paper-optimal mode.

  (b) MECHANISM -- does higher participation go with higher predictive
      uncertainty? Reported as rho(participation, output_entropy), a Spearman
      correlation pooled over all steps. A positive rho supports the
      "more active dims <-> more uncertain" hypothesis.

Every discrimination feature passes the SAME length gate as 05/06/07:
    raw AUROC, length baseline, length-matched AUROC, partial rho | length.
"""

from __future__ import annotations

import argparse
import numpy as np


# --- metrics (standalone copies, identical to 05/06/07) --------------------

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


def spearman(x, y):
    """Spearman rank correlation over the finite pairs of x and y."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 5:
        return float("nan"), int(x.size)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx = (rx - rx.mean()); ry = (ry - ry.mean())
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom < 1e-12:
        return float("nan"), int(x.size)
    return float(np.dot(rx, ry) / denom), int(x.size)


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


def bootstrap_auroc_ci(scores, labels, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for best-direction AUROC."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = scores.size
    vals = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        a, _ = auroc_bestdir(scores[b], labels[b])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


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


def per_step_band_avg(M, cols):
    """Band-average a (T, L) matrix over cols -> (T,) per-step values."""
    M = np.asarray(M, dtype=np.float64)[:, cols]
    with np.errstate(invalid="ignore"):
        return np.nanmean(M, axis=1)


def step_agg_scalar(ps, agg="mean"):
    """Aggregate a (T,) per-step series into one scalar (NaN-aware, finite steps in order).

    mean/std/max/min/range/last : summary statistics over steps
    slope                       : OLS trend over step index (does dimensionality diverge?)
    cusum                       : max one-sided upward CUSUM deviation from the chain's
                                  own mean -- a localizable spike, not a global level
    The point of the sweep: mean tests a GLOBAL level shift; max/cusum/slope test a
    LOCALIZED / DYNAMIC anomaly. Which one wins says whether step-level localization
    is even possible with this feature.
    """
    ps = np.asarray(ps, dtype=np.float64)
    v = ps[np.isfinite(ps)]
    if v.size == 0:
        return float("nan")
    if agg == "mean":  return float(v.mean())
    if agg == "std":   return float(v.std())
    if agg == "max":   return float(v.max())
    if agg == "min":   return float(v.min())
    if agg == "range": return float(v.max() - v.min())
    if agg == "last":  return float(v[-1])
    if agg == "slope":
        if v.size < 2:
            return float("nan")
        return float(np.polyfit(np.arange(v.size, dtype=np.float64), v, 1)[0])
    if agg == "cusum":
        mu = v.mean(); s = 0.0; smax = 0.0
        for val in v:
            s = max(0.0, s + (val - mu)); smax = max(smax, s)
        return float(smax)
    raise ValueError(f"unknown step agg {agg!r}")


def chain_scalar(M, cols, agg="mean"):
    """Band-average a (T, L) matrix over cols, then aggregate over steps."""
    return step_agg_scalar(per_step_band_avg(M, cols), agg)


def layer_feat(Mobj, li, N):
    """Per-chain step-mean of one single layer li -> (N,) feature."""
    out = np.full(N, np.nan)
    for i in range(N):
        col = np.asarray(Mobj[i], dtype=np.float64)[:, li]
        col = col[np.isfinite(col)]
        if col.size:
            out[i] = col.mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="spectral_field .npz produced WITH --step_vectors")
    ap.add_argument("--layer_band", default="all")
    ap.add_argument("--metric", default="pr", choices=["pr", "ae"],
                    help="pr = participation ratio, ae = activation entropy")
    ap.add_argument("--n_match_bins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step_aggs", default="mean,max,std,slope,cusum",
                    help="comma list of step-level aggregations to sweep (section c)")
    ap.add_argument("--per_layer", action="store_true",
                    help="also print the per-layer matchAUROC profile (section d)")
    ap.add_argument("--output", default="data/step_vector_analysis.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if not bool(data.get("sv_stored", np.array(False))):
        raise SystemExit("This npz has no step vectors. Re-run 01 with --step_vectors.")

    modes = [str(m) for m in data["sv_modes"]]
    labels_raw = data["labels"].astype(int)
    n_steps = data["n_steps"].astype(int)
    out_entropy = data["sv_out_entropy"]            # object array of (T,) per chain
    N = len(labels_raw)
    y = (labels_raw >= 0).astype(int)               # 1 = error chain, 0 = correct

    key = f"sv_{args.metric}_"
    PR = {m: data[f"{key}{m}"] for m in modes}       # m -> object array of (T,L)

    L_sub = PR[modes[0]][0].shape[1]
    cols = band_indices(L_sub, args.layer_band)
    metric_name = "participation_ratio" if args.metric == "pr" else "activation_entropy"
    print(f"Loaded {N} chains, L_sub={L_sub}, band={args.layer_band} -> cols {[int(c) for c in cols]}")
    print(f"  metric: {metric_name}   modes: {modes}")
    print(f"  labels: {int(y.sum())} error / {int((1 - y).sum())} correct")

    # --- length gate (the thing to beat) ---
    a_len, dir_len = auroc_bestdir(n_steps.astype(float), y)
    sub = length_matched_subset(y, n_steps, n_bins=args.n_match_bins, seed=args.seed)
    a_len_sub, _ = auroc_bestdir(n_steps[sub].astype(float), y[sub])
    print("\n=== Length baseline (the thing to beat) ===")
    print(f"  AUROC(n_steps)               = {a_len:.4f}  (dir {dir_len})")
    print(f"  matched: {sub.size} chains, AUROC(n_steps|matched) = {a_len_sub:.4f}  (~0.5)")

    # --- (a) DISCRIMINATION: chain-level activation participation per mode ---
    print("\n=== (a) Discrimination: chain-mean activation participation "
          "(RAW vs MATCHED vs partial-rho) ===")
    print(f"{'mode':10s}  {'rawAUROC':>9s} {'dir':>3s}  {'matchAUROC':>10s}  "
          f"{'partial_rho|len':>15s}")
    results = {}
    for m in modes:
        feat = np.array([chain_scalar(PR[m][i], cols, "mean") for i in range(N)],
                        dtype=np.float64)
        a_raw, d_raw = auroc_bestdir(feat, y)
        a_sub, _ = auroc_bestdir(feat[sub], y[sub])
        prho = partial_spearman_given_length(feat, y, n_steps)
        results[m] = dict(raw=a_raw, dir=d_raw, matched=a_sub, partial_rho=prho,
                          feat=feat)
        flag = "  <==" if (not np.isnan(a_sub)) and a_sub > a_len_sub + 0.05 else ""
        print(f"{m:10s}  {a_raw:9.4f} {d_raw:>3s}  {a_sub:10.4f}  {prho:15.4f}{flag}")

    # --- (b) MECHANISM: participation vs output-token entropy, pooled steps ---
    print("\n=== (b) Mechanism: rho(participation, output_entropy) pooled over "
          "steps ===")
    print("  (+ => more active dims go with more predictive uncertainty)")
    rhos = {}
    for m in modes:
        ps_pr, ps_ent = [], []
        for i in range(N):
            pr_i = per_step_band_avg(PR[m][i], cols)     # (T,)
            ent_i = np.asarray(out_entropy[i], dtype=np.float64)  # (T,)
            t = min(pr_i.size, ent_i.size)
            if t > 0:
                ps_pr.append(pr_i[:t]); ps_ent.append(ent_i[:t])
        if ps_pr:
            rho, n_pairs = spearman(np.concatenate(ps_pr), np.concatenate(ps_ent))
        else:
            rho, n_pairs = float("nan"), 0
        rhos[m] = dict(rho=rho, n=n_pairs)
        print(f"  rho({m:8s}, out_entropy) = {rho:7.4f}   (n={n_pairs} steps)")

    # --- (c) STEP-AGGREGATION sweep: global level shift vs localizable spike? ---
    step_aggs = [s.strip() for s in args.step_aggs.split(",") if s.strip()]
    print("\n=== (c) Step-aggregation sweep (matchAUROC; mean=global level, "
          "max/cusum=localized spike, slope=divergence) ===")
    print("mode      " + "".join(f"{a:>10s}" for a in step_aggs))
    agg_results = {}
    for m in modes:
        row = {}
        cells = []
        for agg in step_aggs:
            feat = np.array([step_agg_scalar(per_step_band_avg(PR[m][i], cols), agg)
                             for i in range(N)], dtype=np.float64)
            a_sub, _ = auroc_bestdir(feat[sub], y[sub])
            row[agg] = a_sub
            cells.append(f"{a_sub:10.4f}")
        agg_results[m] = row
        print(f"{m:10s}" + "".join(cells))

    # --- (d) PER-LAYER profile: which single layers carry the signal? ---
    per_layer = None
    layers_used = (data["layers_used"] if "layers_used" in data
                   else np.arange(L_sub)).astype(int)
    if args.per_layer:
        print("\n=== (d) Per-layer matchAUROC profile (step-mean; ALL layers; "
              "DIAGNOSTIC -- do not cherry-pick the best layer on small N) ===")
        print("layer       " + "".join(f"{m:>10s}" for m in modes))
        per_layer = np.full((L_sub, len(modes)), np.nan)
        for li in range(L_sub):
            cells = []
            for mi, m in enumerate(modes):
                feat = layer_feat(PR[m], li, N)
                a_sub, _ = auroc_bestdir(feat[sub], y[sub])
                per_layer[li, mi] = a_sub
                cells.append(f"{a_sub:10.4f}")
            net = int(layers_used[li]) if li < len(layers_used) else li
            print(f"L{li:02d}(net{net:>2d}) " + "".join(cells))
        print("  best single layer per mode:")
        for mi, m in enumerate(modes):
            bi = int(np.nanargmax(per_layer[:, mi]))
            net = int(layers_used[bi]) if bi < len(layers_used) else bi
            print(f"    {m:10s} layer idx {bi} (net {net}) "
                  f"matchAUROC={per_layer[bi, mi]:.4f}")

    # --- (e) IDENTITY check: is participation just effective rank D in disguise? ---
    rho_D = float("nan")
    if "M_D" in data:
        M_D = data["M_D"]
        best_overlap = max(
            modes, key=lambda m: results[m]["matched"]
            if np.isfinite(results[m]["matched"]) else -1.0)
        chainD = np.array([chain_scalar(M_D[i], cols, "mean") for i in range(N)],
                          dtype=np.float64)
        rho_D, nD = spearman(results[best_overlap]["feat"], chainD)
        print(f"\n=== (e) Identity check: rho(chain-mean {metric_name} [{best_overlap}], "
              f"chain-mean effective-rank M_D) ===")
        print(f"  rho = {rho_D:.4f}  (n={nD})   "
              f"[|rho|->1 means participation ~ effective-rank D, NOT a new signal]")

    # --- (f) ERROR vs DIFFICULTY: is the lift already in the PRE-error prefix? ---
    # If (c) says the signal is a global level shift (mean >> cusum/max), the danger
    # is that participation marks "hard / diffuse trajectory" rather than the error
    # event. Score error chains by their PRE-error steps only and compare to the
    # full-chain score on the same chains. prefix ~ full => global/difficulty signal
    # (cannot localize); prefix << full => lift concentrates at/after the error.
    best_mode = max(modes, key=lambda m: results[m]["matched"]
                    if np.isfinite(results[m]["matched"]) else -1.0)
    err_idx = np.where(labels_raw >= 1)[0]   # error chains with >=1 pre-error step
    cor_idx = np.where(labels_raw == -1)[0]  # all-correct chains
    prefix_auroc = full_auroc_raw = float("nan")
    prefix_matched = full_matched = prho_prefix = float("nan"); n_pref = 0
    if err_idx.size and cor_idx.size:
        full_feat = results[best_mode]["feat"]
        prefix_feat = np.full(N, np.nan)
        for i in err_idx:
            k = int(labels_raw[i])
            ps = per_step_band_avg(PR[best_mode][i], cols)[:k]   # steps before 1st error
            ps = ps[np.isfinite(ps)]
            if ps.size:
                prefix_feat[i] = ps.mean()
        n_pref = int(np.isfinite(prefix_feat[err_idx]).sum())
        lab = np.concatenate([np.ones(err_idx.size), np.zeros(cor_idx.size)])
        scF = np.concatenate([full_feat[err_idx],   full_feat[cor_idx]])
        scP = np.concatenate([prefix_feat[err_idx], full_feat[cor_idx]])
        sub_ns = n_steps[np.concatenate([err_idx, cor_idx])].astype(float)
        full_auroc_raw, _ = auroc_bestdir(scF, lab)
        prefix_auroc, _ = auroc_bestdir(scP, lab)
        # length-match on TOTAL chain length n_steps (same control as section a);
        # controls difficulty-via-length, NOT the prefix-vs-full averaging-support
        # asymmetry and NOT cross-problem difficulty (-> needs within-problem, 11).
        mm = length_matched_subset(lab, sub_ns, n_bins=args.n_match_bins, seed=args.seed)
        a_len_f, _ = auroc_bestdir(sub_ns[mm], lab[mm])
        full_matched, _ = auroc_bestdir(scF[mm], lab[mm])
        prefix_matched, _ = auroc_bestdir(scP[mm], lab[mm])
        prho_prefix = partial_spearman_given_length(scP, lab, sub_ns)
        ci_lo, ci_hi = bootstrap_auroc_ci(scP[mm], lab[mm], seed=args.seed)
        print(f"\n=== (f) Error vs difficulty ({best_mode}): is the lift already in "
              f"the PRE-error prefix? ===")
        print(f"  error chains with >=1 pre-error step: {n_pref}/{err_idx.size}")
        print(f"  RAW     : full={full_auroc_raw:.4f}   prefix={prefix_auroc:.4f}")
        print(f"  MATCHED : {mm.size} chains (len AUROC|matched={a_len_f:.4f} ~0.5)  "
              f"full={full_matched:.4f}  prefix={prefix_matched:.4f}  "
              f"[prefix 95% CI {ci_lo:.4f}-{ci_hi:.4f}]")
        print(f"  partial_rho(prefix, label | n_steps) = {prho_prefix:.4f}")
        if np.isfinite(prefix_matched) and np.isfinite(full_matched):
            if prefix_matched >= full_matched - 0.03:
                print("  -> length-matched prefix still >= full: the lift is present "
                      "BEFORE the error and is not a chain-length artifact.")
                print("     CAVEAT: ProcessBench correct/error are DIFFERENT problems, so "
                      "difficulty is still confounded -> within-problem test (11) decides.")
            else:
                print("  -> after length matching prefix < full: the lift concentrates "
                      "at/after the error; the raw prefix>full was a length artifact.")

    # --- save ---
    np.savez(
        args.output,
        modes=np.array(modes, dtype=object),
        metric=np.array(args.metric),
        layer_band=np.array(args.layer_band),
        y=y, n_steps=n_steps,
        feat_mat=np.array([results[m]["feat"] for m in modes], dtype=np.float64).T,
        length_auroc=np.array(a_len),
        length_auroc_matched=np.array(a_len_sub),
        results=np.array({m: {k: v for k, v in results[m].items() if k != "feat"}
                          for m in modes}, dtype=object),
        rhos=np.array(rhos, dtype=object),
        step_aggs=np.array(step_aggs, dtype=object),
        agg_results=np.array(agg_results, dtype=object),
        per_layer_auroc=(per_layer if per_layer is not None
                         else np.array(None, dtype=object)),
        rho_participation_vs_D=np.array(rho_D),
        prefix_auroc=np.array(prefix_auroc),
        full_auroc_raw=np.array(full_auroc_raw),
        prefix_auroc_matched=np.array(prefix_matched),
        full_auroc_matched=np.array(full_matched),
        prefix_partial_rho=np.array(prho_prefix),
    )
    print(f"\nSaved -> {args.output}")

    # --- verdict ---
    finite = [(m, results[m]["matched"]) for m in modes
              if np.isfinite(results[m]["matched"])]
    print("\n=== VERDICT ===")
    if not finite:
        print("  no finite matchAUROC (degenerate features). check the band/metric.")
        return
    best_mode, best_auroc = max(finite, key=lambda t: t[1])
    step_exp_auroc = results.get("step_exp", {}).get("matched", float("nan"))
    print(f"  length-matched baseline AUROC(n_steps|matched) = {a_len_sub:.4f}")
    print(f"  best weighting: {best_mode} matchAUROC = {best_auroc:.4f}")
    if np.isfinite(step_exp_auroc):
        print(f"  step_exp (paper-optimal) matchAUROC = {step_exp_auroc:.4f}")
    if best_auroc > a_len_sub + 0.05:
        print(f"  -> {best_mode} activation participation ({metric_name}) carries a "
              f"length-independent signal")
        print(f"     for error detection (matchAUROC {best_auroc:.4f} > baseline "
              f"{a_len_sub:.4f}). Promote it.")
    else:
        print(f"  -> activation participation does not clearly beat the length "
              f"baseline on this band")
        print(f"     (best matchAUROC {best_auroc:.4f} vs baseline {a_len_sub:.4f}); "
              f"try other bands/metric or a learned aggregator.")


if __name__ == "__main__":
    main()
