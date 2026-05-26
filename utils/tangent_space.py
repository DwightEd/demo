"""Local tangent space estimation and manifold self-consistency.

Tools for estimating the local tangent space of a manifold from a sliding
window of points, and measuring how well a step transition aligns with
this estimated tangent space (manifold self-consistency).
"""

import numpy as np


def local_pca(X, k=None, return_eigvals=True):
    """Local PCA: top-k principal directions of the centered point cloud.

    Args:
        X: (N, d) matrix of points (e.g., a sliding window of trajectory)
        k: number of top components to keep; if None, keep all
        return_eigvals: if True, also return (sorted) eigenvalues

    Returns:
        T_basis: (k, d) matrix whose rows are top-k principal directions
        eigvals: (k,) array of corresponding eigenvalues (if requested)
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    X_c = X - X.mean(axis=0, keepdims=True)

    # SVD on X_c: X_c = U S V^T, principal directions are rows of V^T
    _, S, Vt = np.linalg.svd(X_c, full_matrices=False)

    if k is None:
        k = min(N - 1, d)
    k = max(1, min(k, len(S)))

    T_basis = Vt[:k]  # (k, d)
    eigvals = (S[:k] ** 2) / max(N - 1, 1)  # convert singular values to eigenvalues

    if return_eigvals:
        return T_basis, eigvals
    return T_basis


def subspace_angle_principal(A, B):
    """Principal angle between two subspaces (in radians).

    Computes the largest principal angle (max canonical angle) between
    two subspaces represented as orthonormal-row matrices A (k_A, d) and
    B (k_B, d).

    Reference: Bjorck & Golub 1973 "Numerical methods for computing angles
    between linear subspaces."

    Args:
        A: (k_A, d) basis (rows orthonormal recommended; we re-orthonormalize)
        B: (k_B, d) basis

    Returns:
        max_angle: largest principal angle in [0, π/2], in radians
    """
    # Re-orthonormalize defensively (QR on transposed)
    QA, _ = np.linalg.qr(A.T)  # (d, k_A)
    QB, _ = np.linalg.qr(B.T)  # (d, k_B)

    # Singular values of QA^T @ QB are cosines of principal angles
    sigmas = np.linalg.svd(QA.T @ QB, compute_uv=False)
    # Clip due to floating point
    sigmas = np.clip(sigmas, -1.0, 1.0)
    angles = np.arccos(sigmas)
    return float(angles.max())


def manifold_self_consistency(step_vec, T_basis):
    """How much of step_vec lies in the tangent space spanned by T_basis.

    rho = || P_T step_vec ||^2 / || step_vec ||^2

    Interpretation:
        rho == 1.0 -> transition fully aligned with manifold tangent space
        rho << 1.0 -> transition has large off-manifold component (suspicious)

    Args:
        step_vec: (d,) vector of step transition (e.g., r_j - r_{j-1})
        T_basis: (k, d) orthonormal-row basis of local tangent space

    Returns:
        rho: scalar in [0, 1]
    """
    step_vec = np.asarray(step_vec, dtype=np.float64).reshape(-1)
    norm_sq = float(step_vec @ step_vec)
    if norm_sq < 1e-20:
        return np.nan

    # Re-orthonormalize T_basis defensively
    QT, _ = np.linalg.qr(T_basis.T)  # (d, k)
    # Projection coefficients
    coeffs = QT.T @ step_vec  # (k,)
    proj_norm_sq = float(coeffs @ coeffs)
    return proj_norm_sq / norm_sq
