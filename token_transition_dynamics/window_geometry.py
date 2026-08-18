from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "tle_intrinsic_dimension",
    "information_volume",
    "velocity_innovation_ratio",
    "path_efficiency",
)


def _pairwise_distances(gram: np.ndarray) -> np.ndarray:
    squared_norm = np.diagonal(gram, axis1=1, axis2=2)
    squared = squared_norm[:, :, None] + squared_norm[:, None, :] - 2.0 * gram
    return np.sqrt(np.maximum(squared, 0.0))


def _local_tle(neighbor_distances: np.ndarray, neighbor_pairwise: np.ndarray) -> float:
    """Tight Local Estimator for one centre and its ordered neighbours.

    This is a NumPy adaptation of scikit-dimension's BSD-3-Clause TLE
    implementation (Jonathan Bac, 2020), which implements Amsaleg et al.
    (KDD 2019). The surrounding estimator only changes how centres are sampled.
    """
    distances = np.asarray(neighbor_distances, dtype=np.float64)
    pairwise = np.asarray(neighbor_pairwise, dtype=np.float64)
    radius = float(distances[-1])
    if radius <= 0.0:
        return np.nan

    count = distances.size
    distance_i = np.tile(distances[:, None], (1, count))
    distance_j = distance_i.T
    z_squared = 2.0 * distance_i**2 + 2.0 * distance_j**2 - pairwise**2
    denominator = 2.0 * (radius**2 - distance_i**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        s_value = radius * (
            np.sqrt(
                np.maximum(
                    (distance_i**2 + pairwise**2 - distance_j**2) ** 2
                    + 4.0 * pairwise**2 * (radius**2 - distance_i**2),
                    0.0,
                )
            )
            - (distance_i**2 + pairwise**2 - distance_j**2)
        ) / denominator
        t_value = radius * (
            np.sqrt(
                np.maximum(
                    (distance_i**2 + z_squared - distance_j**2) ** 2
                    + 4.0 * z_squared * (radius**2 - distance_i**2),
                    0.0,
                )
            )
            - (distance_i**2 + z_squared - distance_j**2)
        ) / denominator

    radius_rows = np.isclose(distances, radius, rtol=1e-12, atol=1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_value[radius_rows, :] = (
            radius
            * pairwise[radius_rows, :] ** 2
            / (
                radius**2
                + pairwise[radius_rows, :] ** 2
                - distance_j[radius_rows, :] ** 2
            )
        )
        t_value[radius_rows, :] = (
            radius
            * z_squared[radius_rows, :]
            / (
                radius**2
                + z_squared[radius_rows, :]
                - distance_j[radius_rows, :] ** 2
            )
        )

    zero_i = distance_i <= 1e-12
    t_value[zero_i] = distance_j[zero_i]
    s_value[zero_i] = distance_j[zero_i]
    zero_j = distance_j <= 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        boundary = radius * pairwise / (radius + pairwise)
    t_value[zero_j] = boundary[zero_j]
    s_value[zero_j] = boundary[zero_j]

    repeated = pairwise <= 1e-12
    np.fill_diagonal(repeated, False)
    t_value[repeated] = radius
    s_value[repeated] = radius
    dropped_pairs = int(np.sum(repeated))

    invalid = (
        ~np.isfinite(t_value)
        | ~np.isfinite(s_value)
        | (t_value <= 1e-12)
        | (s_value <= 1e-12)
    )
    np.fill_diagonal(invalid, False)
    dropped_measurements = int(np.sum(invalid))
    t_value[invalid] = radius
    s_value[invalid] = radius
    np.fill_diagonal(t_value, radius)
    np.fill_diagonal(s_value, radius)

    positive = distances > 1e-12
    denominator_sum = (
        np.sum(np.log(t_value / radius))
        + np.sum(np.log(s_value / radius))
        + 2.0 * np.sum(np.log(distances[positive] / radius))
    )
    numerator = -2.0 * (
        count**2
        - dropped_measurements
        - int(np.sum(~positive))
        - dropped_pairs
    )
    if denominator_sum >= -1e-12:
        return np.nan
    return float(numerator / denominator_sum)


def _tle_by_layer(distances: np.ndarray, neighbors: int, centers: int) -> np.ndarray:
    token_count = distances.shape[1]
    centre_indices = np.unique(
        np.linspace(0, token_count - 1, min(centers, token_count), dtype=np.int64)
    )
    estimates = np.empty(distances.shape[0], dtype=np.float64)
    for layer in range(distances.shape[0]):
        local: list[float] = []
        for centre in centre_indices:
            order = np.argsort(distances[layer, centre], kind="stable")
            neighbor_indices = order[order != centre][:neighbors]
            estimate = _local_tle(
                distances[layer, centre, neighbor_indices],
                distances[layer][np.ix_(neighbor_indices, neighbor_indices)],
            )
            if np.isfinite(estimate) and estimate > 0.0:
                local.append(estimate)
        if not local:
            raise ValueError(
                f"TLE is undefined for stored-layer index {layer}; the token window is degenerate"
            )
        estimates[layer] = float(np.mean(local))
    return estimates


def window_features(
    states: np.ndarray,
    *,
    neighbors: int,
    tle_centers: int,
) -> np.ndarray:
    """Return raw-space geometry and ordered dynamics for one token window.

    Input is ``[token, stored_layer, hidden]``; output is
    ``[stored_layer, feature]``. No projection or step pooling is applied.
    """
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"states must have shape [token,layer,hidden], got {values.shape}")
    if neighbors < 2 or neighbors >= values.shape[0]:
        raise ValueError("neighbors must be at least 2 and smaller than the window size")
    if tle_centers < 1:
        raise ValueError("tle_centers must be positive")

    by_layer = np.transpose(values, (1, 0, 2))
    centered = by_layer - np.mean(by_layer, axis=1, keepdims=True)
    gram = centered @ np.transpose(centered, (0, 2, 1))
    distances = _pairwise_distances(gram)
    intrinsic_dimension = _tle_by_layer(distances, neighbors, tle_centers)

    token_count, hidden_size = values.shape[0], values.shape[2]
    identity = np.eye(token_count, dtype=np.float64)[None, :, :]
    sign, logdet = np.linalg.slogdet(identity + (hidden_size / token_count) * gram)
    if np.any(sign <= 0):
        raise ValueError("information-volume Gram matrix is not positive definite")
    information_volume = 0.5 * logdet

    velocity = np.diff(by_layer, axis=1)
    speed_squared = np.sum(velocity**2, axis=2)
    acceleration = np.diff(velocity, axis=1)
    innovation_ratio = np.mean(np.sum(acceleration**2, axis=2), axis=1) / np.maximum(
        np.mean(speed_squared, axis=1), 1e-12
    )
    path_length = np.sum(np.sqrt(speed_squared), axis=1)
    displacement = np.linalg.norm(by_layer[:, -1] - by_layer[:, 0], axis=1)
    path_efficiency = np.clip(displacement / np.maximum(path_length, 1e-12), 0.0, 1.0)

    return np.column_stack(
        [intrinsic_dimension, information_volume, innovation_ratio, path_efficiency]
    )
