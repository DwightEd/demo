"""Grassmannian geometry: Log map, geodesic distance, tangent space PCA.

The Grassmannian Gr(k, d) is the space of k-dimensional subspaces of R^d.
Each point is represented by V in R^{d x k} with V^T V = I_k.

Key operations:
- Log map: tangent vector at V1 pointing toward V2 (encodes subspace rotation)
- Geodesic distance: sqrt(sum(theta_i^2)) via principal angles
- TangentPCA: learn low-dimensional basis for tangent vectors
"""

import torch
import numpy as np


def grassmann_log(V1, V2):
    """Exact logarithmic map on Gr(k, d).

    Computes tangent vector Delta at V1 such that Exp_{V1}(Delta) = V2.
    Delta satisfies V1^T @ Delta = 0 (tangent space condition).

    Uses principal angle decomposition:
      V1^T V2 = P diag(cos theta) Q^T
      Delta = W diag(theta) P^T
    where W are unit tangent directions in each principal plane.

    Args:
        V1: (d, k) orthonormal — base point
        V2: (d, k) orthonormal — target point

    Returns:
        Delta: (d, k) tangent vector at V1
    """
    M = V1.T @ V2  # (k, k)

    # Principal angle decomposition: M = P diag(cos theta) Q^T
    P, cos_theta, QT = torch.linalg.svd(M)
    cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)  # principal angles

    # Align with principal directions
    V1a = V1 @ P      # (d, k) — V1 rotated to principal frame
    V2a = V2 @ QT.T   # (d, k) — V2 rotated to principal frame

    # Unit tangent directions: W_i = (V2a_i - cos(theta_i) V1a_i) / sin(theta_i)
    sin_theta = torch.sin(theta)

    # Handle near-zero angles (no rotation needed)
    safe = sin_theta > 1e-7
    W = torch.zeros_like(V1a)
    if safe.any():
        idx = safe
        W[:, idx] = (V2a[:, idx] - V1a[:, idx] * cos_theta[idx].unsqueeze(0)) \
                     / sin_theta[idx].unsqueeze(0)

    # Scale by angle, rotate back to original frame
    Delta = (W * theta.unsqueeze(0)) @ P.T  # (d, k)

    return Delta


def grassmann_distance(V1, V2):
    """Geodesic distance on Gr(k, d): sqrt(sum(theta_i^2))."""
    cos_theta = torch.linalg.svdvals(V1.T @ V2).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)
    return torch.sqrt((theta ** 2).sum())


def principal_angles(V1, V2):
    """Principal angles between subspaces span(V1) and span(V2)."""
    cos_theta = torch.linalg.svdvals(V1.T @ V2).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos_theta)


class TangentPCA:
    """PCA on flattened Grassmannian tangent vectors.

    Tangent vectors Delta in R^{d x k} have dk entries but only k(d-k) degrees
    of freedom. In practice, correct trajectory tangent vectors concentrate on
    a much lower-dimensional subspace. PCA finds this subspace.
    """

    def __init__(self, n_components=16):
        self.n_components = n_components
        self.basis = None     # (dk, r) projection matrix
        self.mean = None      # (dk,) mean
        self.explained_var = None

    def fit(self, tangent_vectors):
        """Fit PCA from list of (d, k) tangent vectors."""
        X = torch.stack([v.reshape(-1) for v in tangent_vectors]).float()
        N, dk = X.shape

        self.mean = X.mean(dim=0)
        Xc = X - self.mean

        # Use Gram matrix when N < dk (almost always true)
        if N < dk:
            G = Xc @ Xc.T  # (N, N)
            eigvals, eigvecs = torch.linalg.eigh(G)
            # Descending order
            idx = eigvals.argsort(descending=True)
            eigvals = eigvals[idx].clamp(min=0)
            eigvecs = eigvecs[:, idx]

            r = min(self.n_components, N, (eigvals > 1e-10).sum().item())
            # Map to feature space
            self.basis = Xc.T @ eigvecs[:, :r]  # (dk, r)
            norms = self.basis.norm(dim=0, keepdim=True).clamp(min=1e-10)
            self.basis = self.basis / norms
            self.explained_var = eigvals[:r] / (N - 1)
        else:
            C = Xc.T @ Xc / (N - 1)
            eigvals, eigvecs = torch.linalg.eigh(C)
            idx = eigvals.argsort(descending=True)
            r = min(self.n_components, len(idx))
            self.basis = eigvecs[:, idx[:r]]
            self.explained_var = eigvals[idx[:r]]

        total = self.explained_var.sum()
        cum = self.explained_var.cumsum(0) / total.clamp(min=1e-10)
        print(f"  TangentPCA: {r} components, cumulative explained variance = {cum[-1]:.4f}")

    def project(self, tangent_vector):
        """Project (d, k) tangent vector to R^r."""
        v = tangent_vector.reshape(-1).float() - self.mean
        return v @ self.basis

    def project_batch(self, tangent_vectors):
        """Project list of (d, k) tangent vectors to (N, r) matrix."""
        X = torch.stack([v.reshape(-1) for v in tangent_vectors]).float()
        return (X - self.mean) @ self.basis
