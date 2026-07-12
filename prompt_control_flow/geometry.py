from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


EPS = 1e-12


@dataclass
class BasisResult:
    basis: np.ndarray
    mean: np.ndarray
    singular_values: np.ndarray
    rank: int


def _finite_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {x.shape}")
    return x[np.all(np.isfinite(x), axis=1)]


def orthonormal_basis(x: np.ndarray, k: int, *, center: bool = True) -> BasisResult:
    """Return the top-k right singular vectors as a d x r orthonormal basis."""

    x = _finite_rows(x)
    if x.shape[0] == 0:
        raise ValueError("cannot build a basis from zero finite rows")
    mu = x.mean(axis=0) if center else np.zeros(x.shape[1], dtype=np.float64)
    xc = x - mu
    if xc.shape[0] == 1:
        v = xc[0]
        nrm = float(np.linalg.norm(v))
        if nrm <= EPS:
            basis = np.zeros((x.shape[1], 0), dtype=np.float64)
            s = np.zeros(0, dtype=np.float64)
            return BasisResult(basis=basis, mean=mu, singular_values=s, rank=0)
        basis = (v / nrm).reshape(-1, 1)
        return BasisResult(basis=basis, mean=mu, singular_values=np.asarray([nrm]), rank=1)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    r = min(int(k), int(vt.shape[0]), int(np.sum(s > EPS)))
    basis = vt[:r].T.copy() if r > 0 else np.zeros((x.shape[1], 0), dtype=np.float64)
    return BasisResult(basis=basis, mean=mu, singular_values=s[:r].copy(), rank=r)


def random_basis(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthonormal d x k basis."""

    k = min(max(int(k), 0), int(dim))
    if k == 0:
        return np.zeros((int(dim), 0), dtype=np.float64)
    q, _ = np.linalg.qr(rng.normal(size=(int(dim), k)))
    return q[:, :k].astype(np.float64, copy=False)


def projection_energy_fraction(x: np.ndarray, basis: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Per-row squared-energy fraction explained by an orthonormal basis."""

    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    basis = np.asarray(basis, dtype=np.float64)
    den = np.sum(x * x, axis=1)
    if basis.size == 0 or basis.shape[1] == 0:
        num = np.zeros(x.shape[0], dtype=np.float64)
    else:
        proj = x @ basis
        num = np.sum(proj * proj, axis=1)
    out = num / np.maximum(den, eps)
    out[~np.isfinite(out)] = np.nan
    return np.clip(out, 0.0, 1.0)


def project(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if basis.size == 0 or basis.shape[1] == 0:
        return np.zeros_like(x)
    return (x @ basis) @ basis.T


def principal_angle_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Grassmann sine-distance between two orthonormal bases."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0 or a.shape[1] == 0 or b.shape[1] == 0:
        return float("nan")
    s = np.linalg.svd(a.T @ b, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.sqrt(np.sum(1.0 - s * s)))

