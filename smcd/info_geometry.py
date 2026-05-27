"""Information-geometric features from multi-layer spectral data.

Two levels of features:

1. Per-(step, layer) node features:
   Singular values → probability simplex → information-geometric quantities.

2. Trajectory manifold features (the core of our hypothesis):
   A WINDOW of consecutive steps' spectra, stacked into a matrix.
   The effective rank of this matrix = dimensionality of the trajectory manifold.

   Our hypothesis: correct reasoning → trajectory stays on low-dim manifold
   (stable, constrained evolution). Error → trajectory deviates, manifold dim increases.

   This is what window_rank measured (AUROC 0.694) — now we do it per-layer
   and with proper information-geometric tools.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


# ──────────────────────────────────────────────
# Basic information-geometric primitives
# ──────────────────────────────────────────────

def spectrum_to_distribution(sigma: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Normalize squared singular values to probability distribution on simplex."""
    s2 = sigma ** 2 + eps
    return s2 / s2.sum(dim=-1, keepdim=True)


def spectral_entropy(p: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of spectral distribution."""
    return -(p * torch.log(p + 1e-12)).sum(dim=-1)


def effective_rank(p: torch.Tensor) -> torch.Tensor:
    """Effective rank = exp(spectral entropy)."""
    return torch.exp(spectral_entropy(p))


def hellinger_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Hellinger distance on the simplex (Fisher-Rao derived)."""
    bc = (torch.sqrt(p) * torch.sqrt(q)).sum(dim=-1)
    return (1.0 - bc).clamp(min=0.0)


# ──────────────────────────────────────────────
# Trajectory manifold dimension (core feature)
# ──────────────────────────────────────────────

def window_manifold_rank(spectra_seq: torch.Tensor, window: int = 3) -> torch.Tensor:
    """Compute trajectory manifold dimension in a sliding window.

    For each step j, take the window of spectra [j-W+1, ..., j],
    stack as a (W, k) matrix, compute its effective rank.
    This measures: how many spectral dimensions does the trajectory
    span in this local window?

    Low → trajectory constrained to few directions (on low-dim manifold)
    High → trajectory wandering freely (off manifold)

    Args:
        spectra_seq: (T, k) sequence of spectral distributions
        window: window size W
    Returns:
        ranks: (T,) manifold dimension at each step
    """
    T, k = spectra_seq.shape
    ranks = torch.zeros(T)

    for j in range(T):
        start = max(0, j - window + 1)
        W = spectra_seq[start:j + 1]  # (w, k) where w <= window
        if W.shape[0] < 2:
            ranks[j] = 1.0
            continue
        # Effective rank of the trajectory window
        p = spectrum_to_distribution(W.T)  # treat columns as "singular values" of W
        # Actually: compute SVD of the window matrix, then effective rank of its spectrum
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        s_dist = S ** 2 / (S ** 2).sum().clamp(min=1e-10)
        ranks[j] = torch.exp(-(s_dist * torch.log(s_dist + 1e-12)).sum())

    return ranks


def trajectory_stability(spectra_seq: torch.Tensor, window: int = 3) -> torch.Tensor:
    """Measure trajectory stability: variance of step-to-step distances in a window.

    Low variance → stable evolution (consistent step sizes)
    High variance → erratic evolution (unstable)

    Args:
        spectra_seq: (T, k) spectral distributions
        window: window size
    Returns:
        stability: (T,) stability score (lower = more stable)
    """
    T, k = spectra_seq.shape
    stability = torch.zeros(T)

    # Step-to-step Hellinger distances
    dists = torch.zeros(T)
    for j in range(1, T):
        dists[j] = hellinger_distance(spectra_seq[j - 1:j], spectra_seq[j:j + 1])

    for j in range(T):
        start = max(1, j - window + 1)
        w_dists = dists[start:j + 1]
        if len(w_dists) < 2:
            stability[j] = 0.0
        else:
            stability[j] = w_dists.std()

    return stability


# ──────────────────────────────────────────────
# Full feature computation
# ──────────────────────────────────────────────

def compute_trajectory_features(
    data: List[Dict],
    meta: Dict,
    window: int = 3,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Compute trajectory manifold features from multi-layer spectral data.

    Per step j, per layer l:
        - manifold_rank(j, l): trajectory dimension in window (THE core feature)
        - stability(j, l): trajectory stability in window
        - hellinger(j, l): step-to-step spectral distance
        - step_eff_rank(j, l): per-step effective rank (information density, secondary)

    Total: 4L dims per step.

    Also computes per-step summary features:
        - mean/std of manifold_rank across layers
        - mean/std of stability across layers

    Total: 4L + 4 dims per step.

    Returns:
        features_list: list of (T, 4L+4) arrays
        labels_list: list of (T,) arrays
        example_labels: list of int
    """
    L = len(meta["layer_indices"])
    k = meta["k"]
    feat_dim = 4 * L + 4

    features_list = []
    labels_list = []
    example_labels = []

    for ex in data:
        steps = ex["steps"]
        T = len(steps)
        if T < 2:
            continue

        sigma_all = [s["sigma_ml"] for s in steps]  # list of (L, k)
        p_all = [spectrum_to_distribution(s) for s in sigma_all]

        # Per-layer trajectory features
        manifold_ranks = torch.zeros(T, L)
        stabilities = torch.zeros(T, L)
        hellingers = torch.zeros(T, L)
        step_ranks = torch.zeros(T, L)

        for l in range(L):
            # Extract layer l's spectral trajectory: (T, k)
            p_layer = torch.stack([p_all[j][l] for j in range(T)])
            sigma_layer = torch.stack([sigma_all[j][l] for j in range(T)])

            # Core: trajectory manifold dimension
            manifold_ranks[:, l] = window_manifold_rank(p_layer, window=window)

            # Trajectory stability
            stabilities[:, l] = trajectory_stability(p_layer, window=window)

            # Step-to-step distance
            for j in range(1, T):
                hellingers[j, l] = hellinger_distance(p_layer[j - 1:j], p_layer[j:j + 1])

            # Per-step effective rank (secondary)
            step_ranks[:, l] = effective_rank(p_layer)

        # Aggregate: per-layer features + cross-layer summaries
        feats = torch.zeros(T, feat_dim)
        feats[:, 0:L] = manifold_ranks
        feats[:, L:2*L] = stabilities
        feats[:, 2*L:3*L] = hellingers
        feats[:, 3*L:4*L] = step_ranks

        # Cross-layer summaries
        feats[:, 4*L] = manifold_ranks.mean(dim=1)
        feats[:, 4*L+1] = manifold_ranks.std(dim=1)
        feats[:, 4*L+2] = stabilities.mean(dim=1)
        feats[:, 4*L+3] = stabilities.std(dim=1)

        labels = np.array([s["is_error"] for s in steps], dtype=np.float32)
        features_list.append(feats.numpy())
        labels_list.append(labels)
        example_labels.append(ex["label"])

    return features_list, labels_list, example_labels
