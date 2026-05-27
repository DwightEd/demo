"""Step 2: Low-rank analysis of per-trajectory (step × layer) spectral fields.

This is the core empirical test of the working hypothesis:

    For a correct reasoning chain, the (T × L) effective-rank matrix M is
    approximately low-rank — a few principal components explain most of the
    cross-step / cross-layer variation. An error step manifests as a sparse
    departure from that low-rank structure, localised to a particular
    (j*, l*) cell.

For each trajectory we therefore compute three orthogonal signals:

    chain_lowrankness  = σ_1² / Σ σ_k²
        Trajectory-level scalar. Higher = more rank-1 dominant.
        Tested against the chain-level binary label "has any error step".

    step_residual[j]   = || (M - L_k)_{j,:} ||_2
        Per-step row norm of the rank-k SVD residual.
        Tested against the step-level binary label "is this the first error".

    layer_profile_corr[j] = corr(M[j,:], mean(M[:j,:], axis=0))
        Per-step Pearson correlation of the current layer profile with the
        accumulated mean — captures profile-shape consistency, complementary
        to total residual energy.

Outputs three AUROC numbers (chain-level low-rankness, step-level residual,
step-level profile-corr). These three numbers decide whether the hypothesis
holds and at what granularity.

Usage:
    python 02_lowrank_analysis.py \
        --input data/spectral_field.npz \
        --channel D \
        --rank_k 1 \
        --output data/analysis.npz
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

from utils import (
    lowrank_decompose,
    chain_lowrankness,
    step_residual_norms,
    layer_residual_norms,
    layer_profile_corr_with_prefix,
)


CHANNEL_KEYS = {"D": "M_D", "V": "M_V", "C": "M_C"}


def safe_auroc(y_true, y_score):
    """ROC-AUC that returns NaN on degenerate inputs."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    mask = ~np.isnan(y_score)
    if mask.sum() < 4:
        return float("nan")
    yt = y_true[mask]
    ys = y_score[mask]
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, ys))


def analyze_one(M, rank_k=1):
    """Run the full per-trajectory low-rank decomposition.

    Returns:
        dict with keys: lowrank, step_resid (T,), step_corr (T,), layer_resid (L,),
        sigmas (min(T,L),), Vt (k, L).
    """
    M = np.asarray(M, dtype=np.float64)
    T, L = M.shape

    out = {
        "lowrank_k1": float("nan"),
        "lowrank_k2": float("nan"),
        "step_resid": np.full(T, np.nan),
        "step_corr": np.full(T, np.nan),
        "layer_resid": np.full(L, np.nan),
        "sigmas": np.array([]),
    }
    if T < 3 or L < 2:
        return out

    L_k, R, sigmas, Vt = lowrank_decompose(M, k=rank_k, center=True)
    out["sigmas"] = sigmas
    out["lowrank_k1"] = chain_lowrankness(sigmas, k=1)
    out["lowrank_k2"] = chain_lowrankness(sigmas, k=2)
    out["step_resid"] = step_residual_norms(R)
    out["layer_resid"] = layer_residual_norms(R)
    out["step_corr"] = layer_profile_corr_with_prefix(M)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/spectral_field.npz")
    parser.add_argument("--channel", default="D", choices=list(CHANNEL_KEYS),
                        help="Which spectral channel to analyse: "
                             "D (effective rank), V (energy), C (top conc.)")
    parser.add_argument("--rank_k", type=int, default=1,
                        help="Rank of the low-rank approximation. "
                             "k=1 corresponds to the rank-1 hypothesis.")
    parser.add_argument("--output", default="data/analysis.npz")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    data = np.load(args.input, allow_pickle=True)
    ids = data["ids"]
    labels = data["labels"]           # first-error step index, -1 if correct
    M_field = data[CHANNEL_KEYS[args.channel]]   # object array of (T, L) matrices

    n = len(M_field)
    print(f"  {n} trajectories, channel = M_{args.channel}, rank_k = {args.rank_k}")

    # ---- Per-trajectory low-rank analysis ----
    print("Running per-trajectory low-rank decomposition ...")
    lowrank_k1 = np.full(n, np.nan, dtype=np.float64)
    lowrank_k2 = np.full(n, np.nan, dtype=np.float64)
    step_resid_arr = []
    step_corr_arr = []
    layer_resid_arr = []
    sigmas_arr = []

    for i, M in enumerate(tqdm(M_field)):
        result = analyze_one(M, rank_k=args.rank_k)
        lowrank_k1[i] = result["lowrank_k1"]
        lowrank_k2[i] = result["lowrank_k2"]
        step_resid_arr.append(result["step_resid"].astype(np.float32))
        step_corr_arr.append(result["step_corr"].astype(np.float32))
        layer_resid_arr.append(result["layer_resid"].astype(np.float32))
        sigmas_arr.append(result["sigmas"].astype(np.float32))

    # ---- Build chain-level and step-level labels ----
    chain_labels = (labels >= 0).astype(np.int32)

    step_labels_flat = []
    step_scores_resid_flat = []
    step_scores_neg_corr_flat = []

    for i, lab in enumerate(labels):
        n_steps = len(step_resid_arr[i])
        if lab < 0:
            y = np.zeros(n_steps, dtype=np.int32)
        else:
            y = np.zeros(n_steps, dtype=np.int32)
            if 0 <= lab < n_steps:
                y[lab] = 1
        for j in range(n_steps):
            step_labels_flat.append(y[j])
            step_scores_resid_flat.append(step_resid_arr[i][j])
            # high anomaly ↔ low correlation, so we negate for AUROC orientation
            step_scores_neg_corr_flat.append(
                -step_corr_arr[i][j] if not np.isnan(step_corr_arr[i][j]) else np.nan
            )

    step_labels_flat = np.asarray(step_labels_flat, dtype=np.int32)
    step_scores_resid_flat = np.asarray(step_scores_resid_flat, dtype=np.float64)
    step_scores_neg_corr_flat = np.asarray(step_scores_neg_corr_flat, dtype=np.float64)

    # ---- AUROCs ----
    # chain-level: low lowrankness → likely has error (sign reversed)
    auroc_chain = safe_auroc(chain_labels, -lowrank_k1)
    auroc_chain_k2 = safe_auroc(chain_labels, -lowrank_k2)
    auroc_step_resid = safe_auroc(step_labels_flat, step_scores_resid_flat)
    auroc_step_neg_corr = safe_auroc(step_labels_flat, step_scores_neg_corr_flat)

    print("\n=== Hypothesis-test AUROC ===")
    print(f"  channel = M_{args.channel},  rank_k = {args.rank_k}")
    n_chain_pos = int(chain_labels.sum())
    n_chain_neg = int((chain_labels == 0).sum())
    n_step_pos = int(step_labels_flat.sum())
    n_step_neg = int((step_labels_flat == 0).sum())
    print(f"  chain-level     ({n_chain_pos}+ / {n_chain_neg}-)")
    print(f"    AUROC(chain_lowrank_k=1) = {auroc_chain:.4f}")
    print(f"    AUROC(chain_lowrank_k=2) = {auroc_chain_k2:.4f}")
    print(f"  step-level      ({n_step_pos}+ / {n_step_neg}-)")
    print(f"    AUROC(step_residual_norm) = {auroc_step_resid:.4f}")
    print(f"    AUROC(neg_layer_profile_corr) = {auroc_step_neg_corr:.4f}")

    # Per-correct vs per-error chain summary of lowrankness for inspection
    mask_c = chain_labels == 0
    mask_e = chain_labels == 1
    if mask_c.any() and mask_e.any():
        print(f"\n  lowrank_k=1  correct: {np.nanmean(lowrank_k1[mask_c]):.4f} "
              f"± {np.nanstd(lowrank_k1[mask_c]):.4f}    "
              f"error: {np.nanmean(lowrank_k1[mask_e]):.4f} "
              f"± {np.nanstd(lowrank_k1[mask_e]):.4f}")
        print(f"  lowrank_k=2  correct: {np.nanmean(lowrank_k2[mask_c]):.4f} "
              f"± {np.nanstd(lowrank_k2[mask_c]):.4f}    "
              f"error: {np.nanmean(lowrank_k2[mask_e]):.4f} "
              f"± {np.nanstd(lowrank_k2[mask_e]):.4f}")

    # ---- Save ----
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        ids=ids,
        labels=labels,
        chain_labels=chain_labels,
        lowrank_k1=lowrank_k1,
        lowrank_k2=lowrank_k2,
        step_resid=np.array(step_resid_arr, dtype=object),
        step_corr=np.array(step_corr_arr, dtype=object),
        layer_resid=np.array(layer_resid_arr, dtype=object),
        sigmas=np.array(sigmas_arr, dtype=object),
        channel=np.array(args.channel),
        rank_k=np.array(args.rank_k),
        auroc_chain_k1=np.array(auroc_chain),
        auroc_chain_k2=np.array(auroc_chain_k2),
        auroc_step_residual=np.array(auroc_step_resid),
        auroc_step_neg_corr=np.array(auroc_step_neg_corr),
    )
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
