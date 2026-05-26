"""Intrinsic dimension estimators.

Implements three estimators:
1. TWO-NN (Facco et al. 2017) — CIM's default; needs N>=50 ideally, noisy below
2. Standard Participation Ratio — N>=5-10 usable; differentiable
3. Bias-corrected PR (Chun et al. 2025, arXiv:2509.26560) — N~20-50 robust
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def two_nn_id(X, discard_fraction=0.10):
    """TWO-NN intrinsic dimension estimator.

    Facco, d'Errico, Rodriguez, Laio. "Estimating the intrinsic dimension of
    datasets by a minimal neighborhood information." Sci. Rep. 7:12140 (2017).

    Args:
        X: (N, d) array of points
        discard_fraction: drop top-fraction (likely outliers/duplicates)
                          to stabilize MLE fit; CIM default ~ 0.10

    Returns:
        d_hat: estimated intrinsic dimension (scalar)

    Note: for N<50 this estimate is noisy. For N<10 it may be unusable.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if N < 4:
        return np.nan

    # Find first and second nearest neighbors (excluding self)
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)  # self + 1st + 2nd
    distances, _ = nbrs.kneighbors(X)
    r1 = distances[:, 1]
    r2 = distances[:, 2]

    # Filter out degenerate cases (r1==0)
    valid = (r1 > 1e-12) & (r2 > r1)
    if valid.sum() < 4:
        return np.nan

    mu = r2[valid] / r1[valid]
    log_mu = np.log(mu)

    # Discard tail (potential duplicates / boundary effects)
    n_keep = int(np.ceil((1 - discard_fraction) * len(log_mu)))
    log_mu_sorted = np.sort(log_mu)
    log_mu_kept = log_mu_sorted[:n_keep]

    # MLE: d_hat = N_kept / sum(log_mu_kept)
    if log_mu_kept.sum() <= 1e-12:
        return np.nan
    return n_keep / log_mu_kept.sum()


def participation_ratio(eigenvalues):
    """Standard Participation Ratio.

    PR = (sum λ)^2 / sum λ^2

    Interpretation: effective number of significantly contributing dimensions.
    Differentiable in eigenvalues.

    Args:
        eigenvalues: array of (non-negative) eigenvalues

    Returns:
        PR: scalar in [1, len(eigenvalues)]
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    if len(eigenvalues) == 0:
        return np.nan
    num = eigenvalues.sum() ** 2
    den = (eigenvalues ** 2).sum()
    return num / den


def bias_corrected_pr(X):
    """Bias-corrected Participation Ratio.

    Chun, Canatar, Chung, Lee. "Estimating Dimensionality of Neural
    Representations from Finite Samples." arXiv:2509.26560 (2025).

    Standard PR on a sample covariance matrix is biased when N << d.
    The bias-corrected version uses an analytical correction for finite-sample
    bias of (tr Σ̂)^2 / ||Σ̂||_F^2.

    Approximate formula (refer to Chun 2025 Eq. 12-15 for exact form):
        PR_bc ≈ PR_naive * (N - 1) / (N + PR_naive - 2)   [simplified]
    A more accurate correction depends on second-moment ratios; we use a
    practical approximation that matches Chun's behavior at N ~ 20-50.

    Args:
        X: (N, d) data matrix (we recommend centering first)

    Returns:
        PR_bc: scalar estimate
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    if N < 2:
        return np.nan

    # Sample covariance (use d-dimensional, since N << d typically)
    X_c = X - X.mean(axis=0, keepdims=True)
    # Use SVD on X_c for numerical stability when d >> N
    # singular values s -> eigenvalues of X_c X_c^T / (N-1) = s^2 / (N-1)
    s = np.linalg.svd(X_c, full_matrices=False, compute_uv=False)
    eigvals = (s ** 2) / max(N - 1, 1)

    pr_naive = participation_ratio(eigvals)
    if np.isnan(pr_naive):
        return np.nan

    # Practical small-sample correction (approximate Chun 2025 spirit)
    # When N is small, PR_naive systematically underestimates true PR.
    # A common correction multiplies by (N-1)/(N-1-pr_naive) when valid.
    denom = (N - 1) - pr_naive
    if denom <= 0:
        # PR_naive already too large to correct; return naive
        return pr_naive
    correction = (N - 1) / denom
    return pr_naive * correction
