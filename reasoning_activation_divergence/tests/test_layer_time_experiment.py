from __future__ import annotations

import csv
import json

import numpy as np

from functional_divergence.layer_time_experiment import run_layer_time_experiment


def _write_fixture(path) -> None:
    trajectories = np.empty(12, dtype=object)
    for row in range(12):
        base = np.asarray([float(row), -0.2 * row])
        values = np.empty((4, 3, 2))
        for time in range(4):
            for layer in range(3):
                values[time, layer] = base + time * np.asarray([0.4, -0.2]) + layer * np.asarray([0.1, 0.3])
        if row % 2 == 0:
            values[1, 1] += np.asarray([2.0, -1.5])
        trajectories[row] = values
    np.savez(
        path,
        geometry=trajectories,
        residual_geometry=trajectories,
        geometry_names=np.asarray(["delta_norm", "relative_delta_norm"]),
        layer_ids=np.asarray([0, 2, 4]),
        match_error_rows=np.arange(0, 12, 2),
        match_control_rows=np.arange(1, 12, 2),
        match_error_events=np.full(6, 2),
        match_control_events=np.full(6, 2),
        axis_kind=np.asarray("step"),
    )


def test_layer_time_experiment_writes_versioned_machine_readable_outputs(tmp_path) -> None:
    source = tmp_path / "geometry.npz"
    output = tmp_path / "joint"
    _write_fixture(source)

    result = run_layer_time_experiment(
        inputs=[source], output_dir=output, rank=2, n_splits=2, n_boot=50, seed=5
    )

    assert result["schema_version"] == "layer_time_operator_v1"
    assert result["evidence_boundary"] == "derived geometry proxy; not hidden-state Jacobian evidence"
    assert (output / "results.json").is_file()
    assert (output / "pair_scores.csv").is_file()
    assert (output / "metric_comparison.png").is_file()
    saved = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert saved["datasets"][0]["diagnostics"]["n_plaquettes"] == 2
    with (output / "pair_scores.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 6

    run_layer_time_experiment(
        inputs=[source], output_dir=output, rank=2, n_splits=2, n_boot=20, seed=7
    )
    assert len(list(output.glob("results_*.json"))) == 1
    assert len(list(output.glob("pair_scores_*.csv"))) == 1
