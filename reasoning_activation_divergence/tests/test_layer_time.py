from __future__ import annotations

import numpy as np

from functional_divergence.layer_time import (
    LayerTimeDataset,
    affine_plaquette_discrepancy,
    crossfit_layer_time_scores,
    load_matched_layer_time_geometry,
    operator_spectral_metrics,
    project_jvp_operator,
)
from functional_divergence.core import paired_summary


def test_spectral_metrics_separate_rotation_from_radial_stretch() -> None:
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    stretch = np.diag([2.0, 0.5])

    rotating = operator_spectral_metrics(rotation)
    radial = operator_spectral_metrics(stretch)

    assert np.isclose(rotating["complex_eigen_fraction"], 1.0)
    assert np.isclose(rotating["mean_abs_eigenphase_pi"], 0.5)
    assert np.isclose(rotating["polar_proper_rotation_rms_pi"], 0.5)
    assert np.isclose(rotating["condition_number"], 1.0)
    assert np.isclose(radial["complex_eigen_fraction"], 0.0)
    assert np.isclose(radial["mean_abs_eigenphase_pi"], 0.0)
    assert np.isclose(radial["polar_proper_rotation_rms_pi"], 0.0)
    assert np.isclose(radial["condition_number"], 4.0)


def test_polar_metrics_do_not_mislabel_reflection_as_rotation() -> None:
    reflection = np.diag([1.0, -1.0])

    metrics = operator_spectral_metrics(reflection)

    assert np.isclose(metrics["polar_orientation_reversing"], 1.0)
    assert np.isclose(metrics["polar_proper_rotation_rms_pi"], 0.0)


def test_plaquette_discrepancy_detects_non_commutation() -> None:
    identity = np.eye(2)
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    stretch = np.diag([2.0, 0.5])

    assert np.isclose(
        affine_plaquette_discrepancy(identity, identity, identity, identity),
        0.0,
    )
    assert affine_plaquette_discrepancy(rotation, stretch, stretch, rotation) > 0.1


def test_projected_jvp_operator_uses_shared_input_output_bases() -> None:
    jacobian = np.diag([1.0, 2.0, 4.0])
    input_basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    output_basis = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    jvp_columns = jacobian @ input_basis

    projected = project_jvp_operator(jvp_columns, output_basis)

    assert projected.shape == (2, 2)
    assert np.allclose(projected, output_basis.T @ jacobian @ input_basis)


def test_geometry_adapter_preserves_time_and_layer_and_groups_reused_rows(tmp_path) -> None:
    trajectories = np.empty(5, dtype=object)
    for row in range(5):
        values = np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2)
        trajectories[row] = values + 100.0 * row
    path = tmp_path / "geometry_audit.npz"
    np.savez(
        path,
        geometry=trajectories,
        residual_geometry=trajectories,
        geometry_names=np.asarray(["delta_norm", "relative_delta_norm"]),
        layer_ids=np.asarray([0, 2, 4]),
        match_error_rows=np.asarray([0, 1, 3]),
        match_control_rows=np.asarray([2, 2, 4]),
        match_error_events=np.asarray([2, 2, 2]),
        match_control_events=np.asarray([2, 2, 2]),
        axis_kind=np.asarray("step"),
    )

    data = load_matched_layer_time_geometry(path, offsets=(-1, 0))

    assert data.states.shape == (6, 2, 3, 2)
    assert data.time_offsets.tolist() == [-1, 0]
    assert data.layer_ids.tolist() == [0, 2, 4]
    assert data.feature_names == ("delta_norm", "relative_delta_norm")
    assert data.component_ids[0] == data.component_ids[2]
    assert data.component_ids[1] == data.component_ids[3]
    assert data.component_ids[4] != data.component_ids[0]


def test_crossfit_operator_field_detects_inconsistent_intermediate_cell() -> None:
    rng = np.random.default_rng(4)
    samples: list[np.ndarray] = []
    labels: list[int] = []
    pair_ids: list[int] = []
    for pair in range(30):
        base = rng.normal(size=2)
        control = np.empty((2, 3, 2))
        for time in range(2):
            for layer in range(3):
                control[time, layer] = base + time * np.asarray([0.4, -0.2]) + layer * np.asarray([0.1, 0.3])
        error = control.copy()
        error[0, 1] += np.asarray([3.0, -2.0])
        for label, state in ((1, error), (0, control)):
            samples.append(state)
            labels.append(label)
            pair_ids.append(pair)
    data = LayerTimeDataset(
        states=np.asarray(samples),
        labels=np.asarray(labels),
        pair_ids=np.asarray(pair_ids),
        component_ids=np.repeat(np.arange(30), 2),
        row_ids=np.arange(60),
        time_offsets=np.asarray([-1, 0]),
        layer_ids=np.asarray([0, 1, 2]),
        feature_names=("x", "y"),
        metadata={"representation_scope": "synthetic"},
    )

    scores, diagnostics = crossfit_layer_time_scores(data, rank=2, n_splits=5, ridge_alpha=1e-6)
    summary = paired_summary(
        scores["plaquette_observed_disagreement"], data.labels, data.pair_ids, n_boot=200, seed=5
    )

    assert summary["paired_auroc"] > 0.95
    assert diagnostics["projection_rank"] == 2
    assert diagnostics["n_plaquettes"] == 2
    assert diagnostics["max_component_overlap"] == 0
    assert len(diagnostics["operator_cells"]) == 7
    assert len(diagnostics["plaquette_cells"]) == 2
    assert {cell["axis"] for cell in diagnostics["operator_cells"]} == {"depth", "time"}
