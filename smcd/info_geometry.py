"""Information-geometric features from multi-layer spectral data.

Core idea: singular values σ at each step-layer define a point on the probability
simplex via normalization p_i = σ_i² / Σσ_i². The E×C×N constraints are natural
geometric quantities on this simplex:

    E (Expressiveness) = effective rank = exp(spectral entropy)
    C (Compression)    = bounded Hellinger distance between consecutive steps
    N (Non-degeneracy) = Hellinger distance > 0 and entropy not collapsed

All features are deterministic functions of the spectrum with clear
information-geometric meaning. No learned parameters.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


def spectrum_to_distribution(sigma: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Normalize squared singular values to probability distribution.

    Args:
        sigma: (..., k) singular values
    Returns:
        p: (..., k) probability distribution on simplex
    """
    s2 = sigma ** 2 + eps
    return s2 / s2.sum(dim=-1, keepdim=True)


def spectral_entropy(p: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of spectral distribution.

    Args:
        p: (..., k) probability distribution
    Returns:
        H: (...) entropy values
    """
    return -(p * torch.log(p + 1e-12)).sum(dim=-1)


def effective_rank(p: torch.Tensor) -> torch.Tensor:
    """Effective rank = exp(spectral entropy).

    Args:
        p: (..., k) probability distribution
    Returns:
        r_eff: (...) effective rank values
    """
    return torch.exp(spectral_entropy(p))


def hellinger_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Hellinger distance between two distributions on the simplex.

    d_H²(p, q) = 1 - Σ√(p_i · q_i)

    This is a natural metric derived from the Fisher-Rao geometry.

    Args:
        p, q: (..., k) probability distributions
    Returns:
        d: (...) Hellinger distances (in [0, 1])
    """
    bc = (torch.sqrt(p) * torch.sqrt(q)).sum(dim=-1)  # Bhattacharyya coefficient
    return (1.0 - bc).clamp(min=0.0)


def spectral_gap(sigma: torch.Tensor) -> torch.Tensor:
    """Ratio of top two singular values: σ_1 / σ_2.

    Large gap → energy concentrated in one direction.

    Args:
        sigma: (..., k) singular values (sorted descending)
    Returns:
        gap: (...)
    """
    return sigma[..., 0] / (sigma[..., 1] + 1e-10)


def compute_ecn_features(
    data: List[Dict],
    meta: Dict,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Compute E×C×N information-geometric features from multi-layer spectral data.

    Per step j, computes:
        E-features: effective rank per layer r_eff(l)                → L dims
        C-features: Hellinger distance to previous step per layer    → L dims
        N-features: delta effective rank per layer                   → L dims

    Total: 3L dims per step, every dimension has clear meaning.

    Returns:
        features_list: list of (T, 3L) arrays
        labels_list: list of (T,) arrays
        example_labels: list of int (-1 = correct, else first error step)
    """
    L = len(meta["layer_indices"])
    k = meta["k"]

    features_list = []
    labels_list = []
    example_labels = []

    for ex in data:
        steps = ex["steps"]
        T = len(steps)
        if T < 2:
            continue

        # Compute distributions for all steps and layers
        # sigma_ml[j] has shape (L, k)
        sigma_all = [s["sigma_ml"] for s in steps]  # list of (L, k)
        p_all = [spectrum_to_distribution(s) for s in sigma_all]  # list of (L, k)

        step_features = []
        for j in range(T):
            p_j = p_all[j]  # (L, k)

            # E: effective rank per layer
            e_j = effective_rank(p_j)  # (L,)

            # C: Hellinger distance to previous step (per layer)
            if j == 0:
                c_j = torch.zeros(L)
            else:
                c_j = hellinger_distance(p_all[j - 1], p_j)  # (L,)

            # N: delta effective rank (per layer)
            if j == 0:
                n_j = torch.zeros(L)
            else:
                e_prev = effective_rank(p_all[j - 1])
                n_j = e_j - e_prev  # (L,)

            f_j = torch.cat([e_j, c_j, n_j])  # (3L,)
            step_features.append(f_j)

        features = torch.stack(step_features).numpy()  # (T, 3L)
        labels = np.array([s["is_error"] for s in steps], dtype=np.float32)

        features_list.append(features)
        labels_list.append(labels)
        example_labels.append(ex["label"])

    return features_list, labels_list, example_labels


def compute_constraint_scores(
    features: np.ndarray,
    L: int,
    e_bounds: Tuple[float, float] = (1.5, None),
    c_bound: float = 0.3,
) -> np.ndarray:
    """Compute instantaneous S_j = E_j × C_j × N_j from feature array.

    This is a direct, interpretable diagnostic — no learning needed.

    Args:
        features: (T, 3L) feature array
        L: number of layers
        e_bounds: (min_eff_rank, max_eff_rank) — None = no upper bound
        c_bound: max acceptable Hellinger distance per step

    Returns:
        S: (T,) constraint scores in [0, 1]
    """
    T = features.shape[0]
    e_all = features[:, :L]       # (T, L) effective ranks
    c_all = features[:, L:2*L]    # (T, L) Hellinger distances
    n_all = features[:, 2*L:]     # (T, L) delta effective ranks

    # E_j: mean effective rank, normalized to [0,1] via sigmoid
    e_mean = e_all.mean(axis=1)  # (T,)
    e_min = e_bounds[0]
    E_j = 1.0 / (1.0 + np.exp(-(e_mean - e_min)))  # high when rank > threshold

    # C_j: mean Hellinger distance should be bounded
    c_mean = c_all.mean(axis=1)  # (T,)
    C_j = 1.0 / (1.0 + np.exp(5.0 * (c_mean - c_bound)))  # high when distance < bound

    # N_j: evolution should continue (distance > 0, rank not collapsed)
    # For j=0, set N_j = 1 (no prior step)
    n_activity = np.abs(n_all).mean(axis=1)  # (T,) mean absolute rank change
    d_activity = c_mean  # reuse Hellinger as activity measure
    N_j = np.where(
        c_mean < 1e-6,  # stagnation
        0.1,  # penalize
        np.tanh(d_activity * 10)  # reward non-zero evolution
    )
    N_j[0] = 1.0  # first step has no prior

    S_j = E_j * C_j * N_j
    return S_j
