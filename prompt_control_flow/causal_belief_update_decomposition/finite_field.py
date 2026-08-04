from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np


def is_prime(value: int) -> bool:
    candidate = int(value)
    if candidate < 2:
        return False
    for divisor in range(2, int(candidate**0.5) + 1):
        if candidate % divisor == 0:
            return False
    return True


def validate_prime_modulus(modulus: int) -> int:
    value = int(modulus)
    if not is_prime(value):
        raise ValueError("modulus must be prime so row-space arithmetic is a field")
    return value


def enumerate_vectors(modulus: int, dimension: int) -> np.ndarray:
    p = validate_prime_modulus(modulus)
    if int(dimension) < 1:
        raise ValueError("dimension must be positive")
    return np.asarray(list(product(range(p), repeat=int(dimension))), dtype=np.int64)


def row_reduce_mod(matrix: Sequence[Sequence[int]] | np.ndarray, modulus: int) -> np.ndarray:
    """Return reduced row-echelon form over the prime field ``GF(modulus)``."""

    p = validate_prime_modulus(modulus)
    values = np.asarray(matrix, dtype=np.int64) % p
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    reduced = values.copy()
    pivot_row = 0
    for column in range(reduced.shape[1]):
        candidates = np.flatnonzero(reduced[pivot_row:, column] % p)
        if len(candidates) == 0:
            continue
        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            reduced[[pivot_row, selected]] = reduced[[selected, pivot_row]]
        inverse = pow(int(reduced[pivot_row, column]), -1, p)
        reduced[pivot_row] = (reduced[pivot_row] * inverse) % p
        for row in range(reduced.shape[0]):
            if row == pivot_row:
                continue
            coefficient = int(reduced[row, column]) % p
            if coefficient:
                reduced[row] = (
                    reduced[row] - coefficient * reduced[pivot_row]
                ) % p
        pivot_row += 1
        if pivot_row == reduced.shape[0]:
            break
    nonzero = np.any(reduced % p != 0, axis=1)
    return reduced[nonzero]


def matrix_rank_mod(matrix: Sequence[Sequence[int]] | np.ndarray, modulus: int) -> int:
    return int(row_reduce_mod(matrix, modulus).shape[0])


def in_row_span(
    vector: Sequence[int] | np.ndarray,
    matrix: Sequence[Sequence[int]] | np.ndarray,
    modulus: int,
) -> bool:
    values = np.asarray(matrix, dtype=np.int64)
    candidate = np.asarray(vector, dtype=np.int64)
    if values.ndim != 2 or candidate.shape != (values.shape[1],):
        raise ValueError("vector width must match matrix width")
    before = matrix_rank_mod(values, modulus)
    after = matrix_rank_mod(np.vstack([values, candidate]), modulus)
    return bool(before == after)


def affine_support_mask(
    assignments: np.ndarray,
    coefficients: np.ndarray,
    rhs: np.ndarray,
    modulus: int,
) -> np.ndarray:
    p = validate_prime_modulus(modulus)
    points = np.asarray(assignments, dtype=np.int64)
    matrix = np.asarray(coefficients, dtype=np.int64)
    targets = np.asarray(rhs, dtype=np.int64)
    if points.ndim != 2 or matrix.ndim != 2:
        raise ValueError("assignments and coefficients must be matrices")
    if matrix.shape[1] != points.shape[1]:
        raise ValueError("constraint width must match assignment dimension")
    if targets.shape != (matrix.shape[0],):
        raise ValueError("one right-hand side is required per constraint")
    if matrix.shape[0] == 0:
        return np.ones(points.shape[0], dtype=bool)
    return np.all((points @ matrix.T) % p == targets[None, :] % p, axis=1)


def linear_combination(
    rows: np.ndarray,
    coefficients: Sequence[int] | np.ndarray,
    modulus: int,
) -> np.ndarray:
    p = validate_prime_modulus(modulus)
    matrix = np.asarray(rows, dtype=np.int64)
    weights = np.asarray(coefficients, dtype=np.int64)
    if matrix.ndim != 2 or weights.shape != (matrix.shape[0],):
        raise ValueError("linear-combination weights must match the number of rows")
    return (weights @ matrix) % p
