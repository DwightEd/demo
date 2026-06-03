"""Step 13: normalization ablation — is the participation signal just massive
activations? (the #1 reviewer rejection, per the literature.)

PR(z) = (sum z_i^2)^2 / sum z_i^4 is dominated by the largest coordinate. LLMs
have "massive activations" (Sun et al. 2402.17762; magnitudes orders above the
median), so RAW PR may mostly track "how big is the outlier dim", not effective
dimensionality. The established fixes (Mahalanobis++ 2505.18032; effective-rank
paper 2510.08389 whitens before computing) are: per-dim z-score (min), removing
outlier dims, or full ZCA whitening (gold standard).

This script computes the WITHIN-PROBLEM (difficulty-controlled) AUROC of the
chain-level participation under {raw, zscore, drop-top-k, ZCA}, plus distance
scores {diagonal Mahalanobis, full Mahalanobis (=||ZCA(z)||^2), L2-normalized
Mahalanobis (Mahalanobis++)}. Healthy stats from CORRECT chains only. If PR
survives ZCA, the signal is genuine geometry, not an outlier-dim artifact.
"""

from __future__ import annotations

import argparse
import numpy as np


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


def within_pair_auroc(idx_groups, feats, y_inc):
    conc = 0.0; npair = 0
    for idx in idx_groups:
        inc = [feats[i] for i in idx if y_inc[i] == 1 and np.isfinite(feats[i])]
        cor = [feats[i] for i in idx if y_inc[i] == 0 and np.isfinite(feats[i])]
        if not inc or not cor:
            continue
        for a in inc:
            for b in cor:
                conc += 1.0 if a > b else (0.5 if a == b else 0.0)
        npair += len(inc) * len(cor)
    return (conc / npair if npair else float("nan")), npair


def pr_vec(Z):                               # participation ratio over last axis
    with np.errstate(invalid="ignore", divide="ignore"):
        s2 = np.sum(Z ** 2, axis=-1); s4 = np.sum(Z ** 4, axis=-1)
        return np.where(s4 > 1e-12, s2 ** 2 / s4, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="multisample npz with stored vectors")
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--layer_band", default="mid")
    ap.add_argument("--shrink", type=float, default=0.1, help="ZCA covariance shrinkage")
    ap.add_argument("--topk", type=int, default=50, help="# outlier dims to drop")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--output", default="data/norm_ablation.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if not bool(data.get("sv_vectors_stored", np.array(False))):
        raise SystemExit("need stored vectors (10 --store_vectors).")
    VEC = data[f"sv_vec_{args.mode}"]
    problem_ids = data["problem_ids"].astype(int)
    y = (data["is_correct"].astype(int) == 0).astype(int)     # 1 = incorrect
    N = len(VEC)
    L_sub = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    cols = band_indices(L_sub, args.layer_band)

    # chain-level RAW vector: mean over (steps, band-layers)
    X = np.full((N, d), np.nan)
    for i in range(N):
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
        with np.errstate(invalid="ignore"):
            X[i] = np.nanmean(V.reshape(-1, d), axis=0)
    okm = np.isfinite(X).all(axis=1)
    X, y, problem_ids = X[okm], y[okm], problem_ids[okm]
    N = X.shape[0]

    prob_to_idx = {}
    for i, p in enumerate(problem_ids):
        prob_to_idx.setdefault(int(p), []).append(i)
    idx_groups = [np.array(v) for v in prob_to_idx.values()
                  if any(y[v] == 1) and any(y[v] == 0)]
    n_contrastive = len(idx_groups)

    # healthy stats from CORRECT chains
    Xc = X[y == 0]
    mu = Xc.mean(0)
    sig = Xc.std(0)
    Xc0 = Xc - mu
    Sigma = (Xc0.T @ Xc0) / max(1, Xc0.shape[0] - 1)
    Sigma = (1 - args.shrink) * Sigma + args.shrink * (np.trace(Sigma) / d) * np.eye(d)
    w, Q = np.linalg.eigh(Sigma)
    w = np.clip(w, 1e-8, None)
    W_zca = (Q * (1.0 / np.sqrt(w))) @ Q.T                    # Sigma^{-1/2}
    drop = np.argsort(sig)[::-1][:args.topk]                  # highest-variance dims
    keep_mask = np.ones(d, bool); keep_mask[drop] = False

    print(f"Loaded {N} chains, {n_contrastive} contrastive problems, d={d}, "
          f"band={args.layer_band}, mode={args.mode}")
    print(f"  healthy outlier dims (top sigma): max sigma={sig.max():.2f}, "
          f"median sigma={np.median(sig):.3f}, dropping top {args.topk}")

    # feature variants (all chain-level)
    Z_raw = X
    Z_zscore = (X - mu) / (sig + args.eps)
    Z_zca = (X - mu) @ W_zca
    Z_drop = X[:, keep_mask]

    feats = {
        "PR raw":              pr_vec(Z_raw),
        "PR zscore(diag)":     pr_vec(Z_zscore),
        "PR drop-top%d" % args.topk: pr_vec(Z_drop),
        "PR ZCA(full)":        pr_vec(Z_zca),
        "Mahal diag":          np.sum(Z_zscore ** 2, axis=1),
        "Mahal full(ZCA)":     np.sum(Z_zca ** 2, axis=1),
        "Mahal++ (L2norm)":    None,   # filled below
    }
    # Mahalanobis++ : L2-normalize x to unit sphere, then diagonal Mahalanobis
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + args.eps)
    mun = Xn[y == 0].mean(0); sign = Xn[y == 0].std(0)
    feats["Mahal++ (L2norm)"] = np.sum(((Xn - mun) / (sign + args.eps)) ** 2, axis=1)

    def bd(a):
        return (max(a, 1 - a), "+" if a >= 0.5 else "-") if np.isfinite(a) else (a, "?")

    print(f"\n=== Within-problem AUROC under normalization variants "
          f"(best-direction; healthy = correct chains) ===")
    print(f"{'variant':22s}  {'wAUROC':>8s}  {'dir':>3s}")
    out = {}
    for name, f in feats.items():
        a, npair = within_pair_auroc(idx_groups, f, y)
        a_bd, dr = bd(a)
        out[name] = a_bd
        print(f"{name:22s}  {a_bd:8.4f}  {dr:>3s}")

    np.savez(args.output, results=np.array(out, dtype=object),
             band=np.array(args.layer_band), n_contrastive=np.array(n_contrastive))
    print(f"\nSaved -> {args.output}")
    print("\nRead: if 'PR ZCA(full)' and 'PR drop-topK' stay well above 0.5, the "
          "participation signal is genuine geometry, NOT just massive activations.")


if __name__ == "__main__":
    main()
