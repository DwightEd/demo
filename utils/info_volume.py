"""Information volume (CIM-2 formula).

Implements the log-det information volume from:
Ma et al. "Reasoning Emerges from Constrained Inference Manifolds in
Large Language Models." arXiv:2605.08142 (2026), Eq. 14.

V_l(x) = 0.5 * log det( I + (d_l / T(x)) * Z_l(x) Z_l(x)^T )

where Z_l is the centered trajectory matrix at layer l.

This is the Gaussian channel capacity expression; it integrates the
log-spectrum of the trajectory covariance and is sensitive to degenerate
directions (V drops sharply when any eigenvalue collapses).
"""

import numpy as np


def info_volume_cim(X):
    """Compute CIM-2 information volume on a point cloud.

    V = 0.5 * log det( I + (d/T) * X_centered @ X_centered^T )

    Computed via the identity log det(I + A B^T) = log det(I + B^T A)
    to switch to the smaller of d-dim and T-dim matrices.

    Args:
        X: (T, d) trajectory matrix (T points in d-dim ambient space)

    Returns:
        V: scalar information volume (in nats)
    """
    X = np.asarray(X, dtype=np.float64)
    T, d = X.shape
    if T < 2:
        return np.nan

    X_c = X - X.mean(axis=0, keepdims=True)

    # When d >> T (typical for LLM hidden state with d=4096, T=5-30):
    # use I_T + (d/T) * X_c @ X_c^T  ->  switch to T x T matrix
    # since log det(I + (d/T) X X^T) on d x d = log det(I + (d/T) X^T X) on T x T
    # (matrix determinant lemma / Sylvester's identity)
    M_T = np.eye(T) + (d / T) * (X_c @ X_c.T)
    sign, logdet = np.linalg.slogdet(M_T)

    if sign <= 0:
        # Numerical issue; return nan rather than misleading value
        return np.nan
    return 0.5 * logdet
