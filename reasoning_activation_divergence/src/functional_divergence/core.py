from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EPS = 1e-12
DEFAULT_METRICS = ("delta_norm", "relative_delta_norm")


@dataclass(frozen=True)
class MatchedDataset:
    previous: np.ndarray
    current: np.ndarray
    labels: np.ndarray
    pair_ids: np.ndarray
    component_ids: np.ndarray
    row_ids: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]


def _component_ids(error_rows: np.ndarray, control_rows: np.ndarray) -> np.ndarray:
    n = len(error_rows)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    owner: dict[int, int] = {}
    for pair, rows in enumerate(zip(error_rows, control_rows)):
        for row in {int(rows[0]), int(rows[1])}:
            if row in owner:
                union(pair, owner[row])
            else:
                owner[row] = pair
    roots = [find(i) for i in range(n)]
    remap = {root: j for j, root in enumerate(sorted(set(roots)))}
    return np.asarray([remap[root] for root in roots], dtype=np.int64)


def _event_vector(
    trajectories: np.ndarray,
    row: int,
    event: int,
    metric_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    trajectory = np.asarray(trajectories[int(row)], dtype=np.float64)
    if trajectory.ndim != 3 or event <= 0 or event >= trajectory.shape[0]:
        return None
    previous = trajectory[event - 1][:, metric_indices].reshape(-1)
    current = trajectory[event][:, metric_indices].reshape(-1)
    if not (np.isfinite(previous).all() and np.isfinite(current).all()):
        return None
    return previous, current


def load_matched_geometry(
    path: str | Path,
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
    variant: str = "raw",
) -> MatchedDataset:
    path = Path(path)
    with np.load(path, allow_pickle=True) as archive:
        required = {
            "geometry_names",
            "layer_ids",
            "match_error_rows",
            "match_control_rows",
            "match_error_events",
            "match_control_events",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"missing required arrays: {sorted(missing)}")
        field = "geometry" if variant == "raw" else "residual_geometry"
        if field not in archive.files:
            raise ValueError(f"missing geometry variant: {field}")
        names = [str(value) for value in archive["geometry_names"]]
        requested = tuple(metrics)
        unknown = sorted(set(requested).difference(names))
        if unknown:
            raise ValueError(f"unknown metrics: {unknown}")
        metric_indices = np.asarray([names.index(name) for name in requested], dtype=np.int64)
        layers = archive["layer_ids"].astype(int)
        trajectories = archive[field]
        error_rows = archive["match_error_rows"].astype(int)
        control_rows = archive["match_control_rows"].astype(int)
        error_events = archive["match_error_events"].astype(int)
        control_events = archive["match_control_events"].astype(int)
        axis_kind = str(archive["axis_kind"].item()) if "axis_kind" in archive.files else "unknown"

        previous_rows: list[np.ndarray] = []
        current_rows: list[np.ndarray] = []
        labels: list[int] = []
        pair_ids: list[int] = []
        original_rows: list[int] = []
        retained_error: list[int] = []
        retained_control: list[int] = []
        for er, cr, ee, ce in zip(error_rows, control_rows, error_events, control_events):
            error_vector = _event_vector(trajectories, er, ee, metric_indices)
            control_vector = _event_vector(trajectories, cr, ce, metric_indices)
            if error_vector is None or control_vector is None:
                continue
            local_pair = len(retained_error)
            for label, row, vectors in ((1, er, error_vector), (0, cr, control_vector)):
                previous_rows.append(vectors[0])
                current_rows.append(vectors[1])
                labels.append(label)
                pair_ids.append(local_pair)
                original_rows.append(int(row))
            retained_error.append(int(er))
            retained_control.append(int(cr))

    if len(retained_error) < 2:
        raise ValueError("fewer than two complete matched pairs remain after boundary filtering")
    pair_components = _component_ids(np.asarray(retained_error), np.asarray(retained_control))
    feature_names = tuple(f"layer{layer}.{metric}" for layer in layers for metric in requested)
    return MatchedDataset(
        previous=np.asarray(previous_rows, dtype=np.float64),
        current=np.asarray(current_rows, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int8),
        pair_ids=np.asarray(pair_ids, dtype=np.int64),
        component_ids=np.repeat(pair_components, 2),
        row_ids=np.asarray(original_rows, dtype=np.int64),
        feature_names=feature_names,
        metadata={
            "source_path": str(path.resolve()),
            "axis_kind": axis_kind,
            "variant": variant,
            "metrics": list(requested),
            "layers": layers.tolist(),
            "n_source_pairs": int(len(error_rows)),
            "n_retained_pairs": int(len(retained_error)),
            "n_dropped_pairs": int(len(error_rows) - len(retained_error)),
            "n_components": int(np.unique(pair_components).size),
        },
    )


def connected_component_folds(
    component_ids: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    component_ids = np.asarray(component_ids)
    components = np.unique(component_ids)
    if components.size < 2:
        raise ValueError("at least two independent connected components are required")
    n_splits = max(2, min(int(n_splits), int(components.size)))
    shuffled = np.random.default_rng(seed).permutation(components)
    chunks = np.array_split(shuffled, n_splits)
    folds = []
    for chunk in chunks:
        test = np.isin(component_ids, chunk)
        folds.append((np.where(~test)[0], np.where(test)[0]))
    return folds


def _scale_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-8] = 1.0
    return center, scale


def crossfit_transport_fisher(
    data: MatchedDataset,
    *,
    n_splits: int = 5,
    seed: int = 17,
    ridge_alpha: float = 10.0,
    probe_c: float = 0.25,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "the legacy task-probe Fisher analysis requires scikit-learn; "
            "the raw residual-stream CLI does not"
        ) from exc
    n = data.labels.size
    scores = {
        "state_change_l2": np.full(n, np.nan),
        "transport_residual_l2": np.full(n, np.nan),
        "probe_pullback_fisher": np.full(n, np.nan),
        "signed_functional_projection": np.full(n, np.nan),
    }
    control_sse = 0.0
    control_sst = 0.0
    fisher_coefficients: list[float] = []
    fold_sizes: list[int] = []
    for train, test in connected_component_folds(data.component_ids, n_splits=n_splits, seed=seed):
        train_control = train[data.labels[train] == 0]
        if train_control.size == 0 or np.unique(data.labels[train]).size != 2:
            raise ValueError("each training fold must contain control and error samples")
        x_center, x_scale = _scale_fit(data.previous[train_control])
        y_center, y_scale = _scale_fit(data.current[train_control])
        x_train_control = (data.previous[train_control] - x_center) / x_scale
        y_train_control = (data.current[train_control] - y_center) / y_scale
        transport = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
        transport.fit(x_train_control, y_train_control)

        def residual(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            x = (data.previous[indices] - x_center) / x_scale
            y = (data.current[indices] - y_center) / y_scale
            prediction = transport.predict(x)
            return y - prediction, x, y

        residual_train, _, _ = residual(train)
        residual_test, x_test, y_test = residual(test)
        train_control_mask = data.labels[train] == 0
        r_center, r_scale = _scale_fit(residual_train[train_control_mask])
        z_train = (residual_train - r_center) / r_scale
        z_test = (residual_test - r_center) / r_scale
        probe = LogisticRegression(
            C=float(probe_c),
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
        probe.fit(z_train, data.labels[train])
        weight = probe.coef_[0]
        train_probability = probe.predict_proba(z_train)[:, 1]
        fisher_coefficient = float(np.mean(train_probability * (1.0 - train_probability)))

        scores["state_change_l2"][test] = np.linalg.norm(y_test - x_test, axis=1)
        scores["transport_residual_l2"][test] = np.linalg.norm(residual_test, axis=1)
        projection = z_test @ weight
        scores["signed_functional_projection"][test] = projection
        scores["probe_pullback_fisher"][test] = fisher_coefficient * projection**2

        test_control_mask = data.labels[test] == 0
        control_residual = residual_test[test_control_mask]
        control_target = y_test[test_control_mask]
        control_sse += float(np.sum(control_residual**2))
        control_sst += float(np.sum(control_target**2))
        fisher_coefficients.append(fisher_coefficient)
        fold_sizes.append(int(test.size))

    if any(not np.isfinite(value).all() for value in scores.values()):
        raise RuntimeError("cross-fitting did not assign every sample exactly once")
    diagnostics = {
        "n_splits": int(len(fold_sizes)),
        "fold_sizes": fold_sizes,
        "mean_control_transport_r2": float(1.0 - control_sse / max(control_sst, EPS)),
        "mean_fisher_coefficient": float(np.mean(fisher_coefficients)),
        "fisher_rank_by_construction": 1,
        "fisher_scope": "binary task-error probe pullback; not model-native logits Fisher",
    }
    return scores, diagnostics


def paired_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    pair_ids: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 17,
) -> dict[str, float | int]:
    differences: list[float] = []
    wins: list[float] = []
    for pair in np.unique(pair_ids):
        idx = np.where(pair_ids == pair)[0]
        error = scores[idx[labels[idx] == 1]]
        control = scores[idx[labels[idx] == 0]]
        if error.size != 1 or control.size != 1 or not np.isfinite(error[0] + control[0]):
            continue
        difference = float(error[0] - control[0])
        differences.append(difference)
        wins.append(1.0 if difference > 0 else 0.5 if difference == 0 else 0.0)
    diff = np.asarray(differences)
    win = np.asarray(wins)
    if diff.size == 0:
        raise ValueError("no complete finite matched pairs")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diff.size, size=(int(n_boot), diff.size))
    boot_auc = np.mean(win[indices], axis=1)
    boot_diff = np.mean(diff[indices], axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(max(int(n_boot), 1000), diff.size))
    observed = abs(float(np.mean(diff)))
    p_value = (1.0 + np.sum(np.abs(np.mean(signs * diff, axis=1)) >= observed)) / (signs.shape[0] + 1.0)
    return {
        "n_pairs": int(diff.size),
        "paired_auroc": float(np.mean(win)),
        "paired_auroc_ci_low": float(np.quantile(boot_auc, 0.025)),
        "paired_auroc_ci_high": float(np.quantile(boot_auc, 0.975)),
        "mean_paired_difference": float(np.mean(diff)),
        "difference_ci_low": float(np.quantile(boot_diff, 0.025)),
        "difference_ci_high": float(np.quantile(boot_diff, 0.975)),
        "sign_flip_p": float(p_value),
    }


def _paired_wins(scores: np.ndarray, labels: np.ndarray, pair_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    retained_pairs: list[int] = []
    wins: list[float] = []
    for pair in np.unique(pair_ids):
        idx = np.where(pair_ids == pair)[0]
        error = scores[idx[labels[idx] == 1]]
        control = scores[idx[labels[idx] == 0]]
        if error.size != 1 or control.size != 1 or not np.isfinite(error[0] + control[0]):
            continue
        difference = float(error[0] - control[0])
        retained_pairs.append(int(pair))
        wins.append(1.0 if difference > 0 else 0.5 if difference == 0 else 0.0)
    return np.asarray(retained_pairs), np.asarray(wins)


def paired_auc_difference(
    candidate: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    pair_ids: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 17,
) -> dict[str, float | int]:
    candidate_pairs, candidate_wins = _paired_wins(candidate, labels, pair_ids)
    baseline_pairs, baseline_wins = _paired_wins(baseline, labels, pair_ids)
    common = np.intersect1d(candidate_pairs, baseline_pairs)
    if common.size == 0:
        raise ValueError("no common finite pairs for method comparison")
    candidate_lookup = dict(zip(candidate_pairs.tolist(), candidate_wins.tolist()))
    baseline_lookup = dict(zip(baseline_pairs.tolist(), baseline_wins.tolist()))
    differences = np.asarray([candidate_lookup[int(pair)] - baseline_lookup[int(pair)] for pair in common])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, common.size, size=(int(n_boot), common.size))
    bootstrap = np.mean(differences[indices], axis=1)
    return {
        "n_pairs": int(common.size),
        "delta_paired_auroc": float(np.mean(differences)),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
    }


def categorical_pullback_fisher_energy(
    output_jvp: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Compute v^T J^T(diag(p)-pp^T)Jv from JVPs, without forming J or F."""
    output_jvp = np.asarray(output_jvp, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if output_jvp.shape != probabilities.shape or output_jvp.ndim != 2:
        raise ValueError("output_jvp and probabilities must have the same [batch, classes] shape")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probabilities must be non-negative and sum to one")
    mean = np.sum(probabilities * output_jvp, axis=1)
    energy = np.sum(probabilities * output_jvp**2, axis=1) - mean**2
    return np.maximum(energy, 0.0)
