from __future__ import annotations

import json

import numpy as np

from functional_divergence.core import (
    MatchedDataset,
    categorical_pullback_fisher_energy,
    connected_component_folds,
    crossfit_transport_fisher,
    load_matched_geometry,
    paired_auc_difference,
    paired_summary,
)


def _write_geometry_fixture(path, *, reused_control: bool = False) -> None:
    names = np.asarray(["delta_norm", "relative_delta_norm"], dtype=object)
    trajectories = []
    for row in range(4):
        trajectory = np.zeros((3, 2, 2), dtype=np.float32)
        trajectory[1] = row + np.asarray([[0.1, 0.2], [0.3, 0.4]])
        trajectory[2] = trajectory[1] + (2.0 if row < 2 else 0.2)
        trajectories.append(trajectory)
    np.savez_compressed(
        path,
        axis_kind=np.asarray("step"),
        problem_ids=np.arange(4),
        layer_ids=np.asarray([8, 16]),
        geometry_names=names,
        geometry=np.asarray(trajectories, dtype=object),
        residual_geometry=np.asarray(trajectories, dtype=object),
        match_error_rows=np.asarray([0, 1]),
        match_control_rows=np.asarray([2, 2 if reused_control else 3]),
        match_error_events=np.asarray([2, 2]),
        match_control_events=np.asarray([2, 2]),
        metadata_json=np.asarray(json.dumps({"source": "fixture"})),
    )


def test_loader_builds_online_event_vectors_and_links_reused_controls(tmp_path):
    path = tmp_path / "geometry.npz"
    _write_geometry_fixture(path, reused_control=True)

    data = load_matched_geometry(path)

    assert data.previous.shape == (4, 4)
    assert data.current.shape == (4, 4)
    assert data.labels.tolist() == [1, 0, 1, 0]
    assert data.pair_ids.tolist() == [0, 0, 1, 1]
    assert np.unique(data.component_ids).size == 1
    assert data.feature_names[0] == "layer8.delta_norm"


def test_connected_component_folds_never_split_a_component():
    component_ids = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    folds = connected_component_folds(component_ids, n_splits=3, seed=7)

    seen = np.zeros(component_ids.size, dtype=bool)
    for train, test in folds:
        assert not set(component_ids[train]) & set(component_ids[test])
        seen[test] = True
    assert seen.all()


def test_probe_pullback_fisher_recovers_functional_direction():
    rng = np.random.default_rng(4)
    n_pairs, d = 120, 24
    previous = rng.normal(scale=0.2, size=(2 * n_pairs, d))
    current = previous + rng.normal(size=(2 * n_pairs, d))
    labels = np.tile([1, 0], n_pairs)
    current[labels == 1, 0] += 2.0
    data = MatchedDataset(
        previous=previous,
        current=current,
        labels=labels,
        pair_ids=np.repeat(np.arange(n_pairs), 2),
        component_ids=np.repeat(np.arange(n_pairs), 2),
        row_ids=np.arange(2 * n_pairs),
        feature_names=tuple(f"f{i}" for i in range(d)),
        metadata={},
    )

    scores, diagnostics = crossfit_transport_fisher(data, n_splits=5, seed=11)
    euclidean = paired_summary(scores["transport_residual_l2"], labels, data.pair_ids, n_boot=300, seed=3)
    fisher = paired_summary(scores["probe_pullback_fisher"], labels, data.pair_ids, n_boot=300, seed=3)

    assert np.isfinite(scores["probe_pullback_fisher"]).all()
    assert fisher["paired_auroc"] > 0.75
    assert fisher["paired_auroc"] > euclidean["paired_auroc"]
    assert diagnostics["mean_control_transport_r2"] < 1.0


def test_paired_summary_handles_ties_and_is_reproducible():
    score = np.asarray([2.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    labels = np.asarray([1, 0, 1, 0, 1, 0])
    pair_ids = np.repeat(np.arange(3), 2)

    first = paired_summary(score, labels, pair_ids, n_boot=200, seed=9)
    second = paired_summary(score, labels, pair_ids, n_boot=200, seed=9)

    assert first == second
    assert first["paired_auroc"] == 0.5
    assert first["n_pairs"] == 3


def test_categorical_pullback_fisher_uses_output_jvp_without_full_jacobian():
    probabilities = np.asarray([[0.5, 0.5], [0.8, 0.2]])
    output_jvp = np.asarray([[1.0, -1.0], [2.0, 2.0]])

    energy = categorical_pullback_fisher_energy(output_jvp, probabilities)

    np.testing.assert_allclose(energy, [1.0, 0.0], atol=1e-12)


def test_paired_auc_difference_uses_the_same_pairs_for_both_methods():
    labels = np.asarray([1, 0, 1, 0, 1, 0])
    pair_ids = np.repeat(np.arange(3), 2)
    better = np.asarray([2.0, 1.0, 3.0, 1.0, 4.0, 1.0])
    worse = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    comparison = paired_auc_difference(better, worse, labels, pair_ids, n_boot=100, seed=2)

    assert comparison["n_pairs"] == 3
    assert comparison["delta_paired_auroc"] == 1.0
    assert comparison["ci_low"] == 1.0
    assert comparison["ci_high"] == 1.0
