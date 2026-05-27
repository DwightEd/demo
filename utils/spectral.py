"""Spectral utilities for the (step × layer) low-rank analysis.

Core operations:
  - per-(step, layer) effective rank (entropy of normalized squared singular values)
  - per-(step, layer) spectral energy (sum of squared singular values)
  - per-(step, layer) top concentration (largest singular value squared / total)
  - full SVD of the spectral field matrix and its rank-k residual

All routines are pure numpy; no model dependency. Designed so that the spectral
field extraction (01_*.py) and the low-rank analysis (02_*.py) share the same
primitives.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Per-(step, layer) spectral indicators on token cloud H_j^(l) in R^{n_j x d}
# ---------------------------------------------------------------------------

def token_cloud_singular_values(H: np.ndarray) -> np.ndarray:
    """Singular values of a token cloud matrix H ∈ R^{n × d}.

    H is centered before SVD so the leading direction captures variation,
    not the mean offset.

    Args:
        H: (n, d) token hidden-state matrix at one (step, layer).

    Returns:
        singular values in descending order, length = min(n, d).
        Returns an empty array if n < 2.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] < 2:
        return np.array([], dtype=np.float64)
    Hc = H - H.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Hc, full_matrices=False, compute_uv=False)
    return s


def effective_rank(sigmas: np.ndarray, eps: float = 1e-15) -> float:
    """Effective rank = exp(spectral entropy on energy-normalised σ²).

    D = exp(-Σ p_i log p_i),  p_i = σ_i² / Σ σ_k².

    Bounded in [1, len(sigmas)]; equals k for a uniform rank-k spectrum.
    Returns NaN on degenerate input.
    """
    sigmas = np.asarray(sigmas, dtype=np.float64)
    sigmas = sigmas[sigmas > eps]
    if sigmas.size == 0:
        return float("nan")
    p = (sigmas ** 2) / (sigmas ** 2).sum()
    return float(np.exp(-np.sum(p * np.log(p + eps))))


def spectral_energy(sigmas: np.ndarray) -> float:
    """Total energy V = Σ σ²."""
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if sigmas.size == 0:
        return float("nan")
    return float((sigmas ** 2).sum())


def top_concentration(sigmas: np.ndarray) -> float:
    """C = σ_1² / Σ σ_k². NaN on degenerate input."""
    sigmas = np.asarray(sigmas, dtype=np.float64)
    sigmas = sigmas[sigmas > 0]
    if sigmas.size == 0:
        return float("nan")
    total = (sigmas ** 2).sum()
    if total <= 0:
        return float("nan")
    return float((sigmas[0] ** 2) / total)


def step_layer_spectral_summary(H: np.ndarray) -> tuple[float, float, float]:
    """One-shot (D, V, C) at a single (step, layer)."""
    s = token_cloud_singular_values(H)
    if s.size == 0:
        return float("nan"), float("nan"), float("nan")
    return effective_rank(s), spectral_energy(s), top_concentration(s)


# ---------------------------------------------------------------------------
# Low-rank decomposition of the (step × layer) spectral field matrix M
# ---------------------------------------------------------------------------

def lowrank_decompose(M: np.ndarray, k: int = 1, center: bool = True
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Truncated-SVD decomposition M ≈ L_k + R.

    The matrix M ∈ R^{T × L} is centered along rows (subtract per-column mean
    across steps) when center=True; this isolates the across-step variation
    that the low-rank hypothesis predicts to be small for correct chains.

    Args:
        M: (T, L) spectral field matrix (rows = steps, cols = layers).
        k: rank of the low-rank approximation.
        center: subtract per-column mean before SVD.

    Returns:
        L_k: (T, L) rank-k approximation.
        R:   (T, L) residual = M_centered - L_k.
        sigmas: (min(T,L),) full singular value spectrum.
        Vt: (k, L) top-k right singular vectors (rows).
    """
    M = np.asarray(M, dtype=np.float64)
    if center:
        Mc = M - M.mean(axis=0, keepdims=True)
    else:
        Mc = M

    U, s, Vt = np.linalg.svd(Mc, full_matrices=False)
    k = max(1, min(k, s.size))
    L_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    R = Mc - L_k
    return L_k, R, s, Vt[:k, :]


def chain_lowrankness(sigmas: np.ndarray, k: int = 1) -> float:
    """Energy fraction captured by the top-k singular values.

        LowRankness_k = (Σ_{i≤k} σ_i²) / (Σ σ_j²).

    Approaches 1 when the matrix is rank-k dominant; smaller when high-rank
    structure dominates. Recommended k=1 for the rank-1 hypothesis.
    """
    sigmas = np.asarray(sigmas, dtype=np.float64)
    sigmas = sigmas[sigmas > 0]
    if sigmas.size == 0:
        return float("nan")
    k = max(1, min(k, sigmas.size))
    return float((sigmas[:k] ** 2).sum() / (sigmas ** 2).sum())


def step_residual_norms(R: np.ndarray) -> np.ndarray:
    """e_j = ||R_{j,:}||_2 — per-step deviation from the low-rank structure."""
    R = np.asarray(R, dtype=np.float64)
    return np.linalg.norm(R, axis=1)


def layer_residual_norms(R: np.ndarray) -> np.ndarray:
    """g_l = ||R_{:,l}||_2 — per-layer disruption energy.

    Useful for the layer-functional attribution discussed in the extension
    (which layer's pattern got broken at the anomalous step).
    """
    R = np.asarray(R, dtype=np.float64)
    return np.linalg.norm(R, axis=0)


# ---------------------------------------------------------------------------
# Auxiliary signal: layer-profile correlation with prefix mean
# ---------------------------------------------------------------------------

def layer_profile_corr_with_prefix(M: np.ndarray) -> np.ndarray:
    """ρ_j = corr(M[j,:], mean(M[:j,:], axis=0)) for j ≥ 2.

    Captures "is the current step's layer profile shape consistent with the
    accumulated shape so far". A complementary signal to the SVD residual:
    measures angular alignment of layer profiles rather than total residual
    energy.

    Returns:
        rho: (T,) array, NaN for j < 2.
    """
    M = np.asarray(M, dtype=np.float64)
    T, _ = M.shape
    rho = np.full(T, np.nan, dtype=np.float64)
    for j in range(2, T):
        v_j = M[j]
        v_mean = M[:j].mean(axis=0)
        # Pearson correlation (centered, unit-norm cosine)
        a = v_j - v_j.mean()
        b = v_mean - v_mean.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-15:
            continue
        rho[j] = float(np.dot(a, b) / denom)
    return rho
