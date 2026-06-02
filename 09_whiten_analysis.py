"""Step 9: raw vs HEALTHY-STANDARDIZED participation, from stored step vectors.

Consumes a spectral_field npz produced by
`01_extract_spectral_field.py --step_vectors --store_vectors`, which keeps the
raw (un-normalized) step vectors per (step, layer, mode). Because the vectors
are stored, every normalization choice is done HERE (no model re-run):

  raw           : PR/AE of the raw step vector (current default participation)
  standardized  : PR/AE of (z - mu_l)/(sigma_l + eps), where mu_l/sigma_l are the
                  per-(layer,dim) mean/std of CORRECT chains' step vectors.
                  This is the anchor: "how many dims deviate from healthy" -- it
                  also neutralizes architecture-driven massive activations
                  (they are equal in correct/error -> ~0 after centering).

Leakage control: correct chains define the healthy baseline AND are scored, so
mu/sigma for a correct chain are computed with K-FOLD CROSS-FIT (excluding that
chain's fold); error chains use all correct chains. No held-out set needed.

For both versions we report the standard length gate: raw AUROC, length baseline,
length-matched AUROC, partial_rho|len, + bootstrap CI. The headline is whether
STANDARDIZED beats RAW (does referencing 'healthy' expose signal that raw,
massive-activation-dominated participation hides?).
"""

from __future__ import annotations

import argparse
import numpy as np

from utils.step_vector import participation_ratio, activation_entropy


# --- length-gate helpers (same as 08) --------------------------------------

def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64); labels = np.asarray(labels, dtype=np.int64)
    m = ~np.isnan(scores); scores, labels = scores[m], labels[m]
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); ranks = ((csum - counts) + csum + 1) / 2.0
    sum_pos = ranks[inv][labels == 1].sum()
    return float((sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def auroc_bestdir(scores, labels):
    a = auroc(scores, labels)
    if np.isnan(a):
        return a, "+"
    return (a, "+") if a >= 0.5 else (1.0 - a, "-")


def partial_spearman_given_length(feature, label, n_steps):
    feature = np.asarray(feature, float); label = np.asarray(label, float); n_steps = np.asarray(n_steps, float)
    m = ~np.isnan(feature); feature, label, n_steps = feature[m], label[m], n_steps[m]
    if feature.size < 5:
        return float("nan")
    rz = lambda x: (np.argsort(np.argsort(x)).astype(float) - (x.size - 1) / 2.0)
    rf, rl, rn = rz(feature), rz(label), rz(n_steps)
    rf -= (np.dot(rf, rn) / np.dot(rn, rn)) * rn
    rl -= (np.dot(rl, rn) / np.dot(rn, rn)) * rn
    den = np.linalg.norm(rf) * np.linalg.norm(rl)
    return float(np.dot(rf, rl) / den) if den > 1e-12 else float("nan")


def length_matched_subset(labels, n_steps, n_bins=8, seed=0):
    rng = np.random.default_rng(seed); labels = np.asarray(labels); n_steps = np.asarray(n_steps)
    idx = np.arange(labels.size)
    if labels.size == 0:
        return idx
    edges = np.quantile(n_steps, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-6
    keep = []
    for b in range(n_bins):
        ib = idx[(n_steps >= edges[b]) & (n_steps < edges[b + 1])]
        pos = ib[labels[ib] == 1]; neg = ib[labels[ib] == 0]; k = min(pos.size, neg.size)
        if k:
            keep.append(rng.choice(pos, k, replace=False)); keep.append(rng.choice(neg, k, replace=False))
    return np.concatenate(keep) if keep else idx


def bootstrap_auroc_ci(scores, labels, n_boot=2000, seed=0, alpha=0.05):
    scores = np.asarray(scores, float); labels = np.asarray(labels, np.int64)
    rng = np.random.default_rng(seed); n = scores.size; vals = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n); a, _ = auroc_bestdir(scores[b], labels[b])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan")
    return tuple(np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]).tolist())


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


def metric_fn(name):
    return participation_ratio if name == "pr" else activation_entropy


# --- healthy per-(layer,dim) stats with k-fold cross-fit over correct chains -

def healthy_stats(VEC, corr_idx, L_sub, d, kfold, seed):
    """Return (mu_all, sig_all, mu_excl, sig_excl, fold_of) where *_all are over
    ALL correct chains (L_sub,d) and *_excl[f] exclude fold f (for cross-fit)."""
    rng = np.random.default_rng(seed)
    order = corr_idx.copy(); rng.shuffle(order)
    fold_of = {int(ci): (j % kfold) for j, ci in enumerate(order)}
    tot_s = np.zeros((L_sub, d)); tot_q = np.zeros((L_sub, d)); tot_n = np.zeros(L_sub)
    f_s = np.zeros((kfold, L_sub, d)); f_q = np.zeros((kfold, L_sub, d)); f_n = np.zeros((kfold, L_sub))
    for ci in corr_idx:
        V = np.asarray(VEC[ci], dtype=np.float64)          # (T, L_sub, d)
        f = fold_of[int(ci)]
        for li in range(L_sub):
            X = V[:, li, :]
            X = X[np.isfinite(X).all(axis=1)]
            if X.size == 0:
                continue
            s = X.sum(0); q = (X ** 2).sum(0); nrows = X.shape[0]
            tot_s[li] += s; tot_q[li] += q; tot_n[li] += nrows
            f_s[f, li] += s; f_q[f, li] += q; f_n[f, li] += nrows
    def msig(s, qd, n):
        n = np.maximum(n, 1)[:, None]
        mu = s / n; var = qd / n - mu ** 2
        return mu, np.sqrt(np.clip(var, 0.0, None))
    mu_all, sig_all = msig(tot_s, tot_q, tot_n)
    mu_excl, sig_excl = [], []
    for f in range(kfold):
        mu_f, sig_f = msig(tot_s - f_s[f], tot_q - f_q[f], tot_n - f_n[f])
        mu_excl.append(mu_f); sig_excl.append(sig_f)
    return mu_all, sig_all, mu_excl, sig_excl, fold_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="npz from 01 --step_vectors --store_vectors")
    ap.add_argument("--metric", default="ae", choices=["pr", "ae"])
    ap.add_argument("--layer_band", default="deep")
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--n_match_bins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/whiten_analysis.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if not bool(data.get("sv_vectors_stored", np.array(False))):
        raise SystemExit("npz has no stored step vectors. Re-run 01 with "
                         "--step_vectors --store_vectors.")
    key = f"sv_vec_{args.mode}"
    if key not in data:
        raise SystemExit(f"{key} not in npz (modes stored: check --mode).")
    VEC = data[key]                          # object array, each (T, L_sub, d) fp16
    labels = data["labels"].astype(int)
    n_steps = data["n_steps"].astype(int)
    N = labels.size
    y = (labels >= 0).astype(int)            # 1 = error chain
    mfn = metric_fn(args.metric)

    L_sub = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    cols = band_indices(L_sub, args.layer_band)
    corr_idx = np.where(y == 0)[0]
    print(f"Loaded {N} chains ({int(y.sum())} error / {corr_idx.size} correct), "
          f"L_sub={L_sub}, d={d}, band={args.layer_band}, mode={args.mode}, "
          f"metric={args.metric}")

    print("Computing healthy per-(layer,dim) mean/std from correct chains "
          f"({args.kfold}-fold cross-fit) ...")
    mu_all, sig_all, mu_excl, sig_excl, fold_of = healthy_stats(
        VEC, corr_idx, L_sub, d, args.kfold, args.seed)

    def chain_feature(i, standardize):
        V = np.asarray(VEC[i], dtype=np.float64)             # (T, L_sub, d)
        if standardize:
            if y[i] == 0:                                    # correct -> cross-fit
                f = fold_of[int(i)]; mu, sig = mu_excl[f], sig_excl[f]
            else:                                            # error -> all correct
                mu, sig = mu_all, sig_all
        per_step = []
        for t in range(V.shape[0]):
            vals = []
            for li in cols:
                z = V[t, li, :]
                if not np.isfinite(z).all():
                    continue
                if standardize:
                    z = (z - mu[li]) / (sig[li] + args.eps)
                vals.append(mfn(z))
            if vals:
                per_step.append(np.nanmean(vals))
        return float(np.nanmean(per_step)) if per_step else float("nan")

    feats = {}
    for tag, std in [("raw", False), ("standardized", True)]:
        feats[tag] = np.array([chain_feature(i, std) for i in range(N)], float)

    a_len, dlen = auroc_bestdir(n_steps.astype(float), y)
    sub = length_matched_subset(y, n_steps, n_bins=args.n_match_bins, seed=args.seed)
    a_len_m, _ = auroc_bestdir(n_steps[sub].astype(float), y[sub])
    print(f"\n=== Length baseline ===\n  AUROC(n_steps)={a_len:.4f} (dir {dlen}); "
          f"matched {sub.size} chains, AUROC|matched={a_len_m:.4f} (~0.5)")

    print(f"\n=== raw vs HEALTHY-STANDARDIZED participation ({args.mode}, "
          f"{args.metric}, band={args.layer_band}) ===")
    print(f"{'version':14s}  {'rawAUROC':>9s} {'dir':>3s}  {'matchAUROC':>10s}  "
          f"{'partial_rho':>11s}  {'match 95% CI':>18s}")
    out = {}
    for tag in ("raw", "standardized"):
        f = feats[tag]
        a_raw, dr = auroc_bestdir(f, y)
        a_m, _ = auroc_bestdir(f[sub], y[sub])
        prho = partial_spearman_given_length(f, y, n_steps)
        lo, hi = bootstrap_auroc_ci(f[sub], y[sub], seed=args.seed)
        out[tag] = dict(raw=a_raw, matched=a_m, partial_rho=prho, ci=(lo, hi))
        print(f"{tag:14s}  {a_raw:9.4f} {dr:>3s}  {a_m:10.4f}  {prho:11.4f}  "
              f"[{lo:.4f}, {hi:.4f}]")

    np.savez(args.output,
             metric=np.array(args.metric), mode=np.array(args.mode),
             layer_band=np.array(args.layer_band),
             raw_feat=feats["raw"], std_feat=feats["standardized"],
             y=y, n_steps=n_steps,
             results=np.array(out, dtype=object),
             length_auroc_matched=np.array(a_len_m))
    print(f"\nSaved -> {args.output}")

    rm, sm = out["raw"]["matched"], out["standardized"]["matched"]
    print("\n=== VERDICT ===")
    if np.isfinite(sm) and np.isfinite(rm):
        if sm > rm + 0.03:
            print(f"  standardized matchAUROC {sm:.4f} > raw {rm:.4f}: referencing "
                  "'healthy' EXPOSES signal that raw (massive-activation-dominated) "
                  "participation hides. Anchor-faithful normalization helps.")
        elif sm < rm - 0.03:
            print(f"  standardized {sm:.4f} < raw {rm:.4f}: standardization HURTS "
                  "here (raw spread carried the signal).")
        else:
            print(f"  standardized {sm:.4f} ~ raw {rm:.4f}: normalization doesn't "
                  "change the chain-level discrimination materially.")
    print("  (ProcessBench: still difficulty-confounded -> within-problem (11) "
          "decides; this only controls length + massive activations.)")


if __name__ == "__main__":
    main()
