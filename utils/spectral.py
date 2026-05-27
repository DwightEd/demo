"""Spectral utilities for the (step × layer) low-rank analysis.

Core operations:
  - per-(step, layer) effective rank (entropy of normalized squared singular values)
  - per-(step, layer) spectral energy (sum of squared singular values)
  - per-(step, layer) top concentration (largest singular value squared / total)
  - full SVD of the spectral field matrix and its rank-k residual

Reasoning-subspace projection (HARP, Hu et al. ICLR 2026):
  - SVD the unembedding matrix W_U ∈ R^{V × d}.
  - The right singular vectors associated with the top-σ directions span the
    *semantic subspace* — they carry the components of hidden states that
    directly drive next-token logits.
  - The right singular vectors associated with the bottom-σ directions span the
    *reasoning subspace* — they encode internal computation that is invisible
    at the current decoding step.
  - For analyzing the structure of intermediate reasoning, we project each
    token-cloud onto the reasoning subspace before computing spectral
    indicators; this isolates the components of variation that are not
    dominated by the semantic prediction signal.

All routines are pure numpy except `compute_unembedding_svd`, which accepts
torch tensors so the (V × d) SVD can be done on GPU. The downstream analysis
(01_*.py / 02_*.py) consumes only numpy arrays.
"""

from __future__ import annotations

import os
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

# ---------------------------------------------------------------------------
# Reasoning subspace via unembedding SVD (HARP-style)
# ---------------------------------------------------------------------------

def compute_unembedding_svd(W_U, cache_path: str | None = None):
    """SVD of the unembedding matrix W_U ∈ R^{V × d}.

    The right singular vectors form an orthonormal basis of the hidden-state
    space; the singular values rank these basis directions by their
    contribution to the output logits. Specifically, W_U h = U Σ V^T h, so the
    component of h along the i-th right singular vector contributes σ_i to the
    output, scaled into a vocabulary direction by u_i.

    Caching: SVD of (V, d) is expensive but only needs to be done once per
    model. Pass `cache_path` to persist the result.

    Args:
        W_U: numpy ndarray or torch.Tensor of shape (V, d).
        cache_path: optional .npz path to load/save (Vt, S).

    Returns:
        Vt: (d, d) numpy array, right singular vectors as rows, ordered by
            descending σ.
        S:  (d,)  numpy array of singular values, descending.
    """
    if cache_path is not None and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["Vt"], data["S"]

    # Use torch for the heavy SVD when available, fall back to numpy otherwise.
    try:
        import torch
        is_torch = isinstance(W_U, torch.Tensor)
    except ImportError:
        torch = None
        is_torch = False

    if torch is not None and (is_torch or torch.cuda.is_available()):
        if not is_torch:
            W_U_t = torch.from_numpy(np.asarray(W_U, dtype=np.float32))
        else:
            W_U_t = W_U
        if torch.cuda.is_available() and not W_U_t.is_cuda:
            W_U_t = W_U_t.cuda()
        W_U_t = W_U_t.float()
        with torch.no_grad():
            _, S_t, Vh_t = torch.linalg.svd(W_U_t, full_matrices=False)
        S = S_t.detach().cpu().numpy().astype(np.float64)
        Vt = Vh_t.detach().cpu().numpy().astype(np.float64)
    else:
        W_U_np = np.asarray(W_U, dtype=np.float64)
        _, S, Vt = np.linalg.svd(W_U_np, full_matrices=False)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, Vt=Vt, S=S)

    return Vt, S


def select_reasoning_subspace(Vt: np.ndarray, S: np.ndarray,
                              mode: str = "energy",
                              threshold: float = 0.95,
                              ) -> tuple[np.ndarray, dict]:
    """Pick the reasoning subspace from the unembedding SVD.

    Two equivalent ways to define the cutoff are supported. The "energy" mode
    follows HARP's original choice and the typical convention in spectral
    analysis. The "dim_ratio" mode mirrors HARP's empirical "~5% of d" remark.

    Args:
        Vt: (d, d) right singular vectors of W_U, rows in descending-σ order.
        S:  (d,)   singular values, descending.
        mode:
            "energy": split so that the first dimensions capture `threshold`
                fraction of total energy (Σσ²); the *remainder* is reasoning.
                With threshold=0.95, semantic gets top-95% energy and the
                reasoning subspace gets the residual 5%.
            "dim_ratio": the reasoning subspace is the bottom `threshold`
                fraction of the d singular directions. With threshold=0.05,
                reasoning gets 5% × d dimensions.
        threshold: meaning depends on mode (see above).

    Returns:
        V_R: (d, d_R) reasoning subspace basis with columns as basis vectors.
        meta: dict recording (mode, threshold, d_total, d_semantic, d_reasoning,
            energy_in_reasoning) for traceability and downstream logging.
    """
    Vt = np.asarray(Vt, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    d = Vt.shape[0]
    assert S.shape[0] == d, "Vt and S must agree on d"

    if mode == "energy":
        energy = S ** 2
        cumsum = np.cumsum(energy)
        total = cumsum[-1]
        # Smallest index k_sem such that cumsum[k_sem-1] >= threshold * total.
        cutoff_idx = int(np.searchsorted(cumsum, threshold * total) + 1)
        cutoff_idx = max(1, min(cutoff_idx, d - 1))
    elif mode == "dim_ratio":
        d_R = max(1, int(round(threshold * d)))
        cutoff_idx = d - d_R
    else:
        raise ValueError(f"Unknown mode={mode}")

    d_R = d - cutoff_idx
    V_R = Vt[cutoff_idx:, :].T  # (d, d_R), columns are reasoning basis vectors

    meta = {
        "mode": mode,
        "threshold": float(threshold),
        "d_total": int(d),
        "d_semantic": int(cutoff_idx),
        "d_reasoning": int(d_R),
        "energy_in_reasoning": float(np.sum(S[cutoff_idx:] ** 2) /
                                     max(np.sum(S ** 2), 1e-30)),
    }
    return V_R, meta


def project_to_reasoning(H: np.ndarray, V_R: np.ndarray) -> np.ndarray:
    """Project hidden-state matrix H onto the reasoning subspace.

    Args:
        H:   (n, d) token hidden states at one (step, layer).
        V_R: (d, d_R) reasoning subspace basis with columns as basis vectors.

    Returns:
        H_R: (n, d_R) projection of H onto the reasoning subspace.

    Semantics: each row of H is a token's d-dim hidden state. The projection
    keeps only the components orthogonal to the top-σ directions of the
    unembedding map, i.e., the components that do not directly drive the
    current-step logits. Token-cloud structure in this projected space
    reflects the *internal* computation rather than the surface prediction.
    """
    H = np.asarray(H, dtype=np.float64)
    V_R = np.asarray(V_R, dtype=np.float64)
    return H @ V_R


# ---------------------------------------------------------------------------
# Layer-profile auxiliary
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
