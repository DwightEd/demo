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
    ap.add_argument("--pca_k", type=int, default=100,
                    help="PCA dims for well-conditioned full Mahalanobis (n>d so "
                         "raw d=4096 covariance is rank-deficient)")
    ap.add_argument("--kfold", type=int, default=5,
                    help="cross-fit folds over PROBLEMS (healthy stats fit on "
                         "OTHER problems -> no leakage)")
    ap.add_argument("--topk", type=int, default=50, help="# outlier dims to drop")
    ap.add_argument("--seed", type=int, default=0)
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

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + args.eps)   # L2-normalized
    print(f"Loaded {N} chains, {n_contrastive} contrastive problems, d={d}, "
          f"band={args.layer_band}, mode={args.mode}; "
          f"CROSS-FIT {args.kfold}-fold over problems (no leakage), PCA k={args.pca_k}")

    # ---- cross-fit over problems: healthy stats fit on OTHER problems ----
    uniq = np.unique(problem_ids); rng = np.random.default_rng(args.seed); rng.shuffle(uniq)
    fold_of = {int(g): i % args.kfold for i, g in enumerate(uniq)}
    fold = np.array([fold_of[int(p)] for p in problem_ids])

    names = ["PR raw", "PR zscore(diag)", f"PR drop-top{args.topk}", "PR PCA-whiten",
             "Mahal diag", f"Mahal PCA{args.pca_k}", "Mahal++ (L2norm)"]
    F = {n: np.full(N, np.nan) for n in names}
    F["PR raw"] = pr_vec(X)                                   # no fit needed

    for f in range(args.kfold):
        tr = (fold != f) & (y == 0)                          # healthy = correct, other folds
        te = fold == f
        if tr.sum() < args.pca_k + 5 or te.sum() == 0:
            continue
        Xc = X[tr]; mu = Xc.mean(0); sig = Xc.std(0)
        drop = np.argsort(sig)[::-1][:args.topk]; keep = np.ones(d, bool); keep[drop] = False
        # PCA on healthy (top-k directions + per-PC std)
        U, S, Vt = np.linalg.svd(Xc - mu, full_matrices=False)
        k = min(args.pca_k, Vt.shape[0])
        Vk = Vt[:k]; pcstd = (S[:k] / np.sqrt(max(1, Xc.shape[0] - 1))) + args.eps
        # L2-normalized healthy stats
        Xcn = Xn[tr]; mun = Xcn.mean(0); sign = Xcn.std(0)

        Xt = X[te]; Zz = (Xt - mu) / (sig + args.eps)
        PC = (Xt - mu) @ Vk.T / pcstd                        # PCA-whitened (k-dim)
        F["PR zscore(diag)"][te] = pr_vec(Zz)
        F[f"PR drop-top{args.topk}"][te] = pr_vec(Xt[:, keep])
        F["PR PCA-whiten"][te] = pr_vec(PC)
        F["Mahal diag"][te] = np.sum(Zz ** 2, axis=1)
        F[f"Mahal PCA{args.pca_k}"][te] = np.sum(PC ** 2, axis=1)
        F["Mahal++ (L2norm)"][te] = np.sum(((Xn[te] - mun) / (sign + args.eps)) ** 2, axis=1)

    feats = F

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
    print("\nRead: all stats are CROSS-FIT (healthy fit on OTHER problems) so no "
          "leakage. If 'PR PCA-whiten' / 'PR drop-top' stay >0.5 -> participation is "
          "genuine geometry, not a massive-activation artifact. 'Mahal PCA' is the "
          "leak-free 'distance from healthy manifold' (the earlier full-ZCA 1.0 was "
          "leakage: correct chains were in the covariance fit + d>n ill-conditioning).")


if __name__ == "__main__":
    main()
