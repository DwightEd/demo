from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd

from .core import DEFAULT_METRICS, EPS, _component_ids, connected_component_folds


@dataclass(frozen=True)
class LayerTimeDataset:
    """Matched dense event windows with explicit time, layer, and feature axes."""

    states: np.ndarray
    labels: np.ndarray
    pair_ids: np.ndarray
    component_ids: np.ndarray
    row_ids: np.ndarray
    time_offsets: np.ndarray
    layer_ids: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AffineMap:
    matrix: np.ndarray
    offset: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) @ self.matrix + self.offset


def operator_spectral_metrics(matrix: np.ndarray, *, tol: float = 1e-9) -> dict[str, float]:
    """Return scale, phase, polar-rotation, rank, and non-normality summaries."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square matrix")
    eigenvalues = np.linalg.eigvals(matrix)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    polar_factor = u @ vt
    orientation_reversing = float(np.linalg.det(polar_factor) < 0.0)
    correction = np.eye(matrix.shape[0])
    correction[-1, -1] = -1.0 if orientation_reversing else 1.0
    proper_rotation = u @ correction @ vt
    polar_eigenvalues = np.linalg.eigvals(proper_rotation)
    weights = singular_values / max(float(np.sum(singular_values)), EPS)
    positive = weights[weights > 0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    frobenius_sq = float(np.linalg.norm(matrix, ord="fro") ** 2)
    eigen_energy = float(np.sum(np.abs(eigenvalues) ** 2))
    henrici = np.sqrt(max(frobenius_sq - eigen_energy, 0.0)) / max(np.sqrt(frobenius_sq), EPS)
    condition = float(np.inf if singular_values[-1] <= EPS else singular_values[0] / singular_values[-1])
    return {
        "complex_eigen_fraction": float(np.mean(np.abs(np.imag(eigenvalues)) > tol)),
        "mean_abs_eigenphase_pi": float(np.mean(np.abs(np.angle(eigenvalues))) / np.pi),
        "polar_proper_rotation_rms_pi": float(
            np.sqrt(np.mean(np.angle(polar_eigenvalues) ** 2)) / np.pi
        ),
        "polar_orientation_reversing": orientation_reversing,
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "condition_number": condition,
        "singular_effective_rank": effective_rank,
        "normalized_henrici": float(henrici),
    }


def affine_plaquette_discrepancy(
    depth_top: np.ndarray,
    time_right: np.ndarray,
    time_left: np.ndarray,
    depth_bottom: np.ndarray,
) -> float:
    """Normalized linear disagreement between depth→time and time→depth paths."""
    first = np.asarray(depth_top) @ np.asarray(time_right)
    second = np.asarray(time_left) @ np.asarray(depth_bottom)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("plaquette operators must compose to equal-sized matrices")
    return float(np.linalg.norm(first - second, ord="fro") / max(np.sqrt(first.size), 1.0))


def project_jvp_operator(jvp_columns: np.ndarray, output_basis: np.ndarray) -> np.ndarray:
    """Project columns `J @ input_basis` into one shared output coordinate basis."""
    jvp_columns = np.asarray(jvp_columns, dtype=np.float64)
    output_basis = np.asarray(output_basis, dtype=np.float64)
    if jvp_columns.ndim != 2 or output_basis.ndim != 2:
        raise ValueError("JVP columns and output basis must be matrices")
    if jvp_columns.shape[0] != output_basis.shape[0]:
        raise ValueError("JVP output dimension must match output basis dimension")
    return output_basis.T @ jvp_columns


def _fit_shared_projection(
    states: np.ndarray, rank: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.asarray(states, dtype=np.float64).reshape(-1, states.shape[-1])
    center = np.mean(flat, axis=0)
    selected_rank = min(int(rank), int(min(flat.shape)))
    if selected_rank < 1:
        raise ValueError("projection rank must be positive")
    centered = flat - center
    if selected_rank < min(centered.shape):
        _, _, vt = randomized_svd(
            centered,
            n_components=selected_rank,
            n_iter=4,
            random_state=int(seed),
        )
    else:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:selected_rank].T
    coordinates = (flat - center) @ basis
    scale = np.std(coordinates, axis=0)
    scale[scale < 1e-8] = 1.0
    return center, basis, scale


def _project_states(states: np.ndarray, center: np.ndarray, basis: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.asarray(states) - center) @ basis) / scale


def _fit_affine(source: np.ndarray, target: np.ndarray, ridge_alpha: float) -> AffineMap:
    model = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    model.fit(source, target)
    return AffineMap(matrix=np.asarray(model.coef_).T, offset=np.asarray(model.intercept_))


def _fit_operator_field(states: np.ndarray, ridge_alpha: float) -> tuple[list[list[AffineMap]], list[list[AffineMap]]]:
    _, n_time, n_layer, _ = states.shape
    depth = [
        [_fit_affine(states[:, t, layer], states[:, t, layer + 1], ridge_alpha) for layer in range(n_layer - 1)]
        for t in range(n_time)
    ]
    time = [
        [_fit_affine(states[:, t, layer], states[:, t + 1, layer], ridge_alpha) for layer in range(n_layer)]
        for t in range(n_time - 1)
    ]
    return depth, time


def _mean_norm(values: list[np.ndarray], n_samples: int) -> np.ndarray:
    if not values:
        return np.zeros(n_samples, dtype=np.float64)
    return np.mean(np.stack(values, axis=1), axis=1)


def _score_operator_field(
    states: np.ndarray,
    depth: list[list[AffineMap]],
    time: list[list[AffineMap]],
) -> dict[str, np.ndarray]:
    n_samples, n_time, n_layer, _ = states.shape
    radial: list[np.ndarray] = []
    depth_residuals: list[np.ndarray] = []
    time_residuals: list[np.ndarray] = []
    plaquettes: list[np.ndarray] = []
    for t in range(n_time):
        for layer in range(n_layer - 1):
            radial.append(np.linalg.norm(states[:, t, layer + 1] - states[:, t, layer], axis=1))
            predicted = depth[t][layer].apply(states[:, t, layer])
            depth_residuals.append(np.linalg.norm(states[:, t, layer + 1] - predicted, axis=1))
    for t in range(n_time - 1):
        for layer in range(n_layer):
            radial.append(np.linalg.norm(states[:, t + 1, layer] - states[:, t, layer], axis=1))
            predicted = time[t][layer].apply(states[:, t, layer])
            time_residuals.append(np.linalg.norm(states[:, t + 1, layer] - predicted, axis=1))
        for layer in range(n_layer - 1):
            depth_then_time = time[t][layer + 1].apply(states[:, t, layer + 1])
            time_then_depth = depth[t + 1][layer].apply(states[:, t + 1, layer])
            plaquettes.append(np.linalg.norm(depth_then_time - time_then_depth, axis=1))
    return {
        "radial_edge_change": _mean_norm(radial, n_samples),
        "depth_operator_residual": _mean_norm(depth_residuals, n_samples),
        "time_operator_residual": _mean_norm(time_residuals, n_samples),
        "plaquette_observed_disagreement": _mean_norm(plaquettes, n_samples),
    }


def _field_diagnostics(
    depth: list[list[AffineMap]],
    time: list[list[AffineMap]],
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    operator_cells: list[dict[str, Any]] = []
    for t, row in enumerate(depth):
        for layer, item in enumerate(row):
            operator_cells.append(
                {"axis": "depth", "time_index": t, "layer_index": layer, **operator_spectral_metrics(item.matrix)}
            )
    for t, row in enumerate(time):
        for layer, item in enumerate(row):
            operator_cells.append(
                {"axis": "time", "time_index": t, "layer_index": layer, **operator_spectral_metrics(item.matrix)}
            )
    metric_names = tuple(operator_spectral_metrics(np.eye(depth[0][0].matrix.shape[0])))
    metrics = [{name: cell[name] for name in metric_names} for cell in operator_cells]
    means = {name: float(np.mean([entry[name] for entry in metrics])) for name in metrics[0]}
    plaquette_cells: list[dict[str, Any]] = []
    for t in range(len(time)):
        for layer in range(len(depth[t])):
            plaquette_cells.append(
                {
                    "time_index": t,
                    "layer_index": layer,
                    "linear_discrepancy": affine_plaquette_discrepancy(
                    depth[t][layer].matrix,
                    time[t][layer + 1].matrix,
                    time[t][layer].matrix,
                    depth[t + 1][layer].matrix,
                    ),
                }
            )
    return means, operator_cells, plaquette_cells


def crossfit_layer_time_scores(
    data: LayerTimeDataset,
    *,
    rank: int = 2,
    n_splits: int = 5,
    seed: int = 17,
    ridge_alpha: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Cross-fit control operator fields in one shared fold-specific coordinate gauge."""
    if data.states.ndim != 4 or data.states.shape[1] < 2 or data.states.shape[2] < 2:
        raise ValueError("states must have shape [sample,time>=2,layer>=2,feature]")
    n_samples = data.states.shape[0]
    output = {
        name: np.full(n_samples, np.nan)
        for name in (
            "radial_edge_change", "depth_operator_residual", "time_operator_residual",
            "plaquette_observed_disagreement",
        )
    }
    spectral_means: list[dict[str, float]] = []
    operator_cell_folds: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    plaquette_cell_folds: dict[tuple[int, int], list[float]] = {}
    fold_sizes: list[int] = []
    component_overlaps: list[int] = []
    selected_ranks: list[int] = []
    folds = connected_component_folds(data.component_ids, n_splits=n_splits, seed=seed)
    for fold_index, (train, test) in enumerate(folds):
        overlap = np.intersect1d(data.component_ids[train], data.component_ids[test]).size
        component_overlaps.append(int(overlap))
        train_control = train[data.labels[train] == 0]
        available_rank = min(data.states.shape[-1], train_control.size - 1)
        selected_rank = min(int(rank), int(available_rank))
        if selected_rank < 1:
            raise ValueError("each fold needs at least two independent control samples")
        center, basis, scale = _fit_shared_projection(
            data.states[train_control], selected_rank, seed=seed + fold_index
        )
        projected_control = _project_states(data.states[train_control], center, basis, scale)
        projected_test = _project_states(data.states[test], center, basis, scale)
        depth, time = _fit_operator_field(projected_control, ridge_alpha)
        fold_scores = _score_operator_field(projected_test, depth, time)
        for name, values in fold_scores.items():
            output[name][test] = values
        field_mean, fold_operator_cells, fold_plaquettes = _field_diagnostics(depth, time)
        spectral_means.append(field_mean)
        for cell in fold_operator_cells:
            key = (str(cell["axis"]), int(cell["time_index"]), int(cell["layer_index"]))
            operator_cell_folds.setdefault(key, []).append(cell)
        for cell in fold_plaquettes:
            key = (int(cell["time_index"]), int(cell["layer_index"]))
            plaquette_cell_folds.setdefault(key, []).append(float(cell["linear_discrepancy"]))
        selected_ranks.append(selected_rank)
        fold_sizes.append(int(test.size))
    if any(not np.isfinite(values).all() for values in output.values()):
        raise RuntimeError("cross-fitting did not assign every sample exactly once")
    metric_names = tuple(spectral_means[0])
    operator_cells = []
    for (axis, time_index, layer_index), cells in sorted(operator_cell_folds.items()):
        operator_cells.append(
            {
                "axis": axis,
                "time_index": time_index,
                "layer_index": layer_index,
                **{name: float(np.mean([cell[name] for cell in cells])) for name in metric_names},
            }
        )
    plaquette_cells = [
        {
            "time_index": key[0],
            "layer_index": key[1],
            "linear_discrepancy": float(np.mean(values)),
        }
        for key, values in sorted(plaquette_cell_folds.items())
    ]
    diagnostics: dict[str, Any] = {
        "n_splits": len(folds),
        "fold_sizes": fold_sizes,
        "projection_rank": int(min(selected_ranks)),
        "requested_rank": int(rank),
        "shared_basis_scope": "one basis fit across all train-control time/layer cells per fold",
        "n_plaquettes": int((data.states.shape[1] - 1) * (data.states.shape[2] - 1)),
        "max_component_overlap": int(max(component_overlaps)),
        "mean_linear_plaquette_discrepancy": float(
            np.mean([cell["linear_discrepancy"] for cell in plaquette_cells])
        ),
        "mean_operator_spectrum": {
            name: float(np.mean([entry[name] for entry in spectral_means])) for name in metric_names
        },
        "operator_cells": operator_cells,
        "plaquette_cells": plaquette_cells,
    }
    return output, diagnostics


def _event_window(
    trajectories: np.ndarray,
    row: int,
    event: int,
    offsets: np.ndarray,
    metric_indices: np.ndarray,
) -> np.ndarray | None:
    trajectory = np.asarray(trajectories[int(row)], dtype=np.float64)
    indices = event + offsets
    if trajectory.ndim != 3 or np.any(indices < 0) or np.any(indices >= trajectory.shape[0]):
        return None
    window = trajectory[indices][:, :, metric_indices]
    if not np.isfinite(window).all():
        return None
    return window


def load_matched_layer_time_geometry(
    path: str | Path,
    *,
    offsets: Iterable[int] = (-1, 0),
    metrics: Iterable[str] = DEFAULT_METRICS,
    variant: str = "raw",
) -> LayerTimeDataset:
    """Load complete matched event windows without flattening time or layer axes."""
    path = Path(path)
    requested_offsets = np.asarray(tuple(offsets), dtype=np.int64)
    if requested_offsets.ndim != 1 or requested_offsets.size < 2:
        raise ValueError("at least two time offsets are required")
    if np.any(np.diff(requested_offsets) != 1):
        raise ValueError("time offsets must be consecutive and increasing")
    requested_metrics = tuple(metrics)
    with np.load(path, allow_pickle=True) as archive:
        required = {
            "geometry_names", "layer_ids", "match_error_rows", "match_control_rows",
            "match_error_events", "match_control_events",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"missing required arrays: {sorted(missing)}")
        field = "geometry" if variant == "raw" else "residual_geometry"
        if field not in archive.files:
            raise ValueError(f"missing geometry variant: {field}")
        names = [str(value) for value in archive["geometry_names"]]
        unknown = sorted(set(requested_metrics).difference(names))
        if unknown:
            raise ValueError(f"unknown metrics: {unknown}")
        metric_indices = np.asarray([names.index(name) for name in requested_metrics], dtype=np.int64)
        trajectories = archive[field]
        layers = archive["layer_ids"].astype(np.int64)
        error_rows = archive["match_error_rows"].astype(np.int64)
        control_rows = archive["match_control_rows"].astype(np.int64)
        error_events = archive["match_error_events"].astype(np.int64)
        control_events = archive["match_control_events"].astype(np.int64)
        axis_kind = str(archive["axis_kind"].item()) if "axis_kind" in archive.files else "unknown"

        states: list[np.ndarray] = []
        labels: list[int] = []
        pair_ids: list[int] = []
        row_ids: list[int] = []
        retained_error: list[int] = []
        retained_control: list[int] = []
        for er, cr, ee, ce in zip(error_rows, control_rows, error_events, control_events):
            error_window = _event_window(trajectories, int(er), int(ee), requested_offsets, metric_indices)
            control_window = _event_window(trajectories, int(cr), int(ce), requested_offsets, metric_indices)
            if error_window is None or control_window is None:
                continue
            pair = len(retained_error)
            for label, row, window in ((1, er, error_window), (0, cr, control_window)):
                states.append(window)
                labels.append(label)
                pair_ids.append(pair)
                row_ids.append(int(row))
            retained_error.append(int(er))
            retained_control.append(int(cr))

    if len(retained_error) < 2:
        raise ValueError("fewer than two complete matched pairs remain after window filtering")
    pair_components = _component_ids(np.asarray(retained_error), np.asarray(retained_control))
    return LayerTimeDataset(
        states=np.asarray(states, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int8),
        pair_ids=np.asarray(pair_ids, dtype=np.int64),
        component_ids=np.repeat(pair_components, 2),
        row_ids=np.asarray(row_ids, dtype=np.int64),
        time_offsets=requested_offsets,
        layer_ids=layers,
        feature_names=requested_metrics,
        metadata={
            "source_path": str(path.resolve()),
            "axis_kind": axis_kind,
            "variant": variant,
            "representation_scope": "derived_geometry_proxy_not_hidden_state",
            "n_source_pairs": int(len(error_rows)),
            "n_retained_pairs": int(len(retained_error)),
            "n_dropped_pairs": int(len(error_rows) - len(retained_error)),
            "n_components": int(np.unique(pair_components).size),
        },
    )
