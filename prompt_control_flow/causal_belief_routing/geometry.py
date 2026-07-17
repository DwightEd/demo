from __future__ import annotations

from typing import Sequence

import numpy as np

from .finite_field import validate_prime_modulus


def uniform_belief(support_mask: np.ndarray) -> np.ndarray:
    support = np.asarray(support_mask, dtype=bool)
    count = int(support.sum())
    if support.ndim != 1 or count < 1:
        raise ValueError("support mask must be one-dimensional and non-empty")
    result = support.astype(np.float64)
    result /= float(count)
    return result


def direct_query_distribution(
    assignments: np.ndarray,
    support_mask: np.ndarray,
    query: Sequence[int] | np.ndarray,
    modulus: int,
) -> np.ndarray:
    p = validate_prime_modulus(modulus)
    points = np.asarray(assignments, dtype=np.int64)
    support = np.asarray(support_mask, dtype=bool)
    vector = np.asarray(query, dtype=np.int64)
    if points.ndim != 2 or support.shape != (points.shape[0],):
        raise ValueError("support must align with assignments")
    if vector.shape != (points.shape[1],):
        raise ValueError("query width must match assignment dimension")
    residues = (points[support] @ vector) % p
    counts = np.bincount(residues, minlength=p).astype(np.float64)
    if counts.sum() <= 0:
        raise ValueError("query distribution requires non-empty support")
    return counts / counts.sum()


def fourier_coordinates(
    assignments: np.ndarray,
    support_mask: np.ndarray,
    frequencies: np.ndarray,
    modulus: int,
) -> np.ndarray:
    """Characteristic-function coordinates of a uniform finite-field belief."""

    p = validate_prime_modulus(modulus)
    points = np.asarray(assignments, dtype=np.int64)
    support = np.asarray(support_mask, dtype=bool)
    modes = np.asarray(frequencies, dtype=np.int64)
    if points.ndim != 2 or modes.ndim != 2:
        raise ValueError("assignments and frequencies must be matrices")
    if points.shape[1] != modes.shape[1]:
        raise ValueError("assignment and frequency dimensions differ")
    if support.shape != (points.shape[0],) or not np.any(support):
        raise ValueError("support must select at least one assignment")
    phase = (2.0 * np.pi / p) * ((points[support] @ modes.T) % p)
    return np.exp(1j * phase).mean(axis=0)


def split_fourier(phi: np.ndarray, *, drop_zero: bool = True) -> np.ndarray:
    values = np.asarray(phi, dtype=np.complex128)
    if values.ndim != 1:
        raise ValueError("Fourier coordinates must be one-dimensional")
    selected = values[1:] if drop_zero else values
    return np.concatenate([selected.real, selected.imag]).astype(np.float64)


def join_fourier(
    coordinates: np.ndarray,
    num_frequencies: int,
    *,
    zero_value: complex = 1.0 + 0.0j,
) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    retained = int(num_frequencies) - 1
    if values.shape != (2 * retained,):
        raise ValueError("split Fourier coordinate length is inconsistent")
    result = np.empty(int(num_frequencies), dtype=np.complex128)
    result[0] = zero_value
    result[1:] = values[:retained] + 1j * values[retained:]
    return result


def query_distribution_from_fourier(
    phi: np.ndarray,
    frequencies: np.ndarray,
    query: Sequence[int] | np.ndarray,
    modulus: int,
) -> np.ndarray:
    """Recover the distribution of a linear query by finite Fourier inversion."""

    p = validate_prime_modulus(modulus)
    values = np.asarray(phi, dtype=np.complex128)
    modes = np.asarray(frequencies, dtype=np.int64)
    vector = np.asarray(query, dtype=np.int64) % p
    if values.shape != (modes.shape[0],):
        raise ValueError("one Fourier value is required per frequency")
    lookup = {tuple(row.tolist()): index for index, row in enumerate(modes % p)}
    characteristic = np.asarray(
        [values[lookup[tuple(((multiple * vector) % p).tolist())]] for multiple in range(p)],
        dtype=np.complex128,
    )
    residues = np.arange(p, dtype=np.float64)
    multiples = np.arange(p, dtype=np.float64)
    inverse_phase = np.exp(-2j * np.pi * multiples[:, None] * residues[None, :] / p)
    distribution = np.real(characteristic @ inverse_phase) / float(p)
    distribution[np.abs(distribution) < 1e-12] = 0.0
    distribution = np.clip(distribution, 0.0, None)
    total = float(distribution.sum())
    if total <= 0.0:
        return np.full(p, 1.0 / p, dtype=np.float64)
    return distribution / total


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    left = np.asarray(p, dtype=np.float64)
    right = np.asarray(q, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("beliefs must be aligned vectors")
    left = left / left.sum()
    right = right / right.sum()
    affinity = float(np.sum(np.sqrt(np.clip(left * right, 0.0, None))))
    return float(2.0 * np.arccos(np.clip(affinity, -1.0, 1.0)))
