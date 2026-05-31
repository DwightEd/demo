"""Geometric primitives for the *orientation-and-position* trajectory analysis.

Motivation
----------
The original pipeline collapses each (step, layer) token cloud
H_j^(l) in R^{n_j x d_R} to three scalars (effective rank D, energy V,
concentration C). Those scalars are functions of the covariance *spectrum*
only. They are blind to two things that a step-to-step trajectory cares about:

  - WHERE the cloud sits   -> its centroid  mu_j
  - WHICH WAY the cloud points -> its principal axes (top eigenvectors of cov)

Effective rank is invariant to both: translate the cloud anywhere, or rotate
its principal axes by 90 degrees, and D does not move. So the "centroid drift"
and "orientation drift" signals were never measured -- they were discarded at
extraction time.

This module provides:
  1. cloud_geometry()   -- reduce one token cloud to (mu, top-k eigvals, top-k eigvecs)
  2. centroid_step_drift() / centroid_curvature()  -- first-order (position) dynamics
  3. principal_angle_drift() / grassmann_distance() -- second-order (orientation) dynamics
  4. bures_distance() / gaussian_w2()  -- combined position+shape+orientation distance
  5. align_basis()      -- Procrustes/sign alignment so eigenvectors are comparable across steps

All routines are pure numpy. Everything operates *inside* the reasoning
subspace if H was already projected there by the caller.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Reduce one token cloud to a position + orientation descriptor
# ---------------------------------------------------------------------------

def cloud_geometry(H: np.ndarray, k: int = 4, eps: float = 1e-12):
    """Reduce a token cloud to (centroid, top-k eigenvalues, top-k eigenvectors).

    The eigen-decomposition is of the *centered* covariance, so the
    eigenvectors describe the cloud's orientation (its principal axes) and the
    eigenvalues its spread along those axes. The centroid is kept separately as
    the cloud's position.

    Args:
        H: (n, p) token cloud at one (step, layer). p = d_R if projected.
        k: number of leading principal axes to keep.
        eps: numerical floor.

    Returns:
        mu:      (p,)   centroid (position).
        eigvals: (k,)   leading eigenvalues of the covariance, descending,
                        zero-padded if rank < k.
        eigvecs: (p, k) leading eigenvectors as columns, zero-padded if
                        rank < k. Sign is arbitrary at this stage; use
                        align_basis() before comparing across steps.
        Returns (None, None, None) if n < 2.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] < 2:
        return None, None, None
    n, p = H.shape
    mu = H.mean(axis=0)
    Hc = H - mu
    # Economy SVD of the centered cloud: Hc = U S Vt.
    # Right singular vectors Vt are the principal axes; S^2/(n-1) are eigenvalues.
    try:
        _, s, Vt = np.linalg.svd(Hc, full_matrices=False)
    except np.linalg.LinAlgError:
        return mu, np.zeros(k), np.zeros((p, k))
    eigvals_full = (s ** 2) / max(n - 1, 1)
    kk = min(k, Vt.shape[0])
    eigvals = np.zeros(k, dtype=np.float64)
    eigvecs = np.zeros((p, k), dtype=np.float64)
    eigvals[:kk] = eigvals_full[:kk]
    eigvecs[:, :kk] = Vt[:kk, :].T  # columns = principal axes
    return mu, eigvals, eigvecs


# ---------------------------------------------------------------------------
# 2. Position (centroid) dynamics
# ---------------------------------------------------------------------------

def centroid_step_drift(mus: np.ndarray) -> np.ndarray:
    """Per-step centroid displacement ||mu_j - mu_{j-1}||_2.

    Args:
        mus: (T, p) centroids, one row per step (in step order).

    Returns:
        drift: (T,) with drift[0] = NaN (no previous step).
    """
    mus = np.asarray(mus, dtype=np.float64)
    T = mus.shape[0]
    out = np.full(T, np.nan)
    for j in range(1, T):
        out[j] = float(np.linalg.norm(mus[j] - mus[j - 1]))
    return out


def centroid_curvature(mus: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-step turning angle of the centroid path (radians).

    angle between v_j = mu_j - mu_{j-1} and v_{j-1} = mu_{j-1} - mu_{j-2}.
    Large angle = the position trajectory sharply changed direction.

    Args:
        mus: (T, p) centroids in step order.

    Returns:
        kappa: (T,) radians; kappa[0], kappa[1] = NaN.
    """
    mus = np.asarray(mus, dtype=np.float64)
    T = mus.shape[0]
    out = np.full(T, np.nan)
    for j in range(2, T):
        v1 = mus[j] - mus[j - 1]
        v0 = mus[j - 1] - mus[j - 2]
        d = float(np.linalg.norm(v1) * np.linalg.norm(v0))
        if d <= eps:
            continue
        c = float(np.clip(np.dot(v1, v0) / d, -1.0, 1.0))
        out[j] = float(np.arccos(c))
    return out


# ---------------------------------------------------------------------------
# 3. Orientation dynamics: principal angles / Grassmann distance
# ---------------------------------------------------------------------------

def principal_angles(U: np.ndarray, V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Principal angles (radians) between two subspaces spanned by columns.

    U: (p, k), V: (p, k). Columns need not be orthonormal; we orthonormalize
    via QR first. Returns min(k, rank) angles in ascending order.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    # Drop zero-padded columns (rank deficiency).
    U = U[:, np.linalg.norm(U, axis=0) > eps]
    V = V[:, np.linalg.norm(V, axis=0) > eps]
    if U.shape[1] == 0 or V.shape[1] == 0:
        return np.array([np.nan])
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    s = np.linalg.svd(Qu.T @ Qv, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def principal_angle_drift(eigvecs_seq: list[np.ndarray],
                          reduce: str = "mean") -> np.ndarray:
    """Per-step orientation drift = principal angle(s) between this step's
    principal subspace and the previous step's.

    Args:
        eigvecs_seq: list of (p, k) principal-axis matrices, one per step.
        reduce: how to collapse the k angles into one number per step.
            "mean"   -> mean angle
            "max"    -> largest angle (most-rotated axis)
            "first"  -> the leading principal angle only

    Returns:
        drift: (T,) radians; drift[0] = NaN.
    """
    T = len(eigvecs_seq)
    out = np.full(T, np.nan)
    for j in range(1, T):
        ang = principal_angles(eigvecs_seq[j - 1], eigvecs_seq[j])
        if ang.size == 0 or np.all(np.isnan(ang)):
            continue
        if reduce == "max":
            out[j] = float(np.nanmax(ang))
        elif reduce == "first":
            out[j] = float(ang[0])
        else:
            out[j] = float(np.nanmean(ang))
    return out


def grassmann_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Grassmann (geodesic) distance = L2 norm of the principal angles.

    A single scalar capturing total subspace rotation between two clouds.
    """
    ang = principal_angles(U, V)
    if ang.size == 0 or np.all(np.isnan(ang)):
        return float("nan")
    return float(np.sqrt(np.nansum(ang ** 2)))


# ---------------------------------------------------------------------------
# 4. Combined position + shape + orientation: Bures / Gaussian-W2
# ---------------------------------------------------------------------------

def _cov_from_eig(eigvals: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """Reconstruct a (low-rank) covariance from kept eigvals/eigvecs."""
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvecs = np.asarray(eigvecs, dtype=np.float64)
    return (eigvecs * eigvals[None, :]) @ eigvecs.T


def bures_distance(eigvals1, eigvecs1, eigvals2, eigvecs2,
                   eps: float = 1e-12) -> float:
    """Bures (Wasserstein-2 between zero-mean Gaussians) distance between two
    clouds described by their top-k eigendecompositions.

    Bures^2(A, B) = tr(A) + tr(B) - 2 tr( (A^{1/2} B A^{1/2})^{1/2} ).

    Captures shape + orientation change in one scalar (position handled
    separately by the centroid term; see gaussian_w2). Computed on the
    reconstructed low-rank covariances.
    """
    A = _cov_from_eig(eigvals1, eigvecs1)
    B = _cov_from_eig(eigvals2, eigvecs2)
    # Symmetric PSD sqrt of A via eigendecomposition.
    wa, Va = np.linalg.eigh((A + A.T) / 2)
    wa = np.clip(wa, 0, None)
    A_half = (Va * np.sqrt(wa)[None, :]) @ Va.T
    M = A_half @ B @ A_half
    wm, _ = np.linalg.eigh((M + M.T) / 2)
    wm = np.clip(wm, 0, None)
    cross = float(np.sum(np.sqrt(wm)))
    val = float(np.trace(A) + np.trace(B) - 2.0 * cross)
    return float(np.sqrt(max(val, 0.0)))


def gaussian_w2(mu1, eigvals1, eigvecs1, mu2, eigvals2, eigvecs2) -> float:
    """2-Wasserstein distance between two Gaussians N(mu, Sigma) approximating
    the clouds. One scalar combining position drift + shape/orientation drift:

        W2^2 = ||mu1 - mu2||^2 + Bures^2(Sigma1, Sigma2).
    """
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    pos = float(np.sum((mu1 - mu2) ** 2))
    bur = bures_distance(eigvals1, eigvecs1, eigvals2, eigvecs2)
    return float(np.sqrt(max(pos + bur ** 2, 0.0)))


def gaussian_w2_step_drift(mus, eigvals_seq, eigvecs_seq) -> np.ndarray:
    """Per-step Gaussian-W2 distance to the previous step.

    Args:
        mus:        (T, p) centroids in step order.
        eigvals_seq: list of (k,) eigenvalue vectors per step.
        eigvecs_seq: list of (p, k) eigenvector matrices per step.

    Returns:
        (T,) with [0] = NaN.
    """
    T = len(eigvals_seq)
    out = np.full(T, np.nan)
    for j in range(1, T):
        out[j] = gaussian_w2(
            mus[j - 1], eigvals_seq[j - 1], eigvecs_seq[j - 1],
            mus[j], eigvals_seq[j], eigvecs_seq[j],
        )
    return out


# ---------------------------------------------------------------------------
# 5. Basis alignment (so eigenvectors are comparable across steps)
# ---------------------------------------------------------------------------

def align_basis(V_ref: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Align principal-axis matrix V to reference V_ref via orthogonal
    Procrustes, so that sign flips and within-degenerate-subspace rotations do
    not create spurious "drift".

    NOTE: only needed if you compare *individual* eigenvectors. The principal-
    angle and Grassmann routines above are already invariant to basis choice
    within each subspace, so they do NOT require this. Provided for the
    step-embedding use case (Section: building z_j for the forward model).

    Args:
        V_ref: (p, k) reference basis (columns).
        V:     (p, k) basis to align.

    Returns:
        V_aligned: (p, k) = V @ R where R is the optimal orthogonal map.
    """
    V_ref = np.asarray(V_ref, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    M = V.T @ V_ref
    U, _, Wt = np.linalg.svd(M)
    R = U @ Wt
    return V @ R