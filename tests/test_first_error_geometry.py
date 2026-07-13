from __future__ import annotations

from pathlib import Path

import numpy as np

from prompt_control_flow.first_error_geometry import (
    FirstErrorGeometryConfig,
    StepGeometryDataset,
    compute_geometry_fields,
    load_step_geometry_dataset,
    load_token_axis,
    make_step_axis,
    map_matches_to_axis,
    match_correct_pseudo_events,
    run_first_error_geometry_audit,
)
from prompt_control_flow.first_error_geometry_report import write_geometry_audit_report


def _object_array(values: list[np.ndarray]) -> np.ndarray:
    output = np.empty(len(values), dtype=object)
    output[:] = values
    return output


def test_geometry_fields_match_right_angle_menger_curvature() -> None:
    trajectory = np.asarray(
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0)],
        dtype=np.float32,
    )[:, None, :]
    field = compute_geometry_fields([trajectory], device="cpu", batch_size=4)[0]

    np.testing.assert_allclose(field[1:, 0, 0], np.ones(3), atol=1e-6)
    np.testing.assert_allclose(field[1, 0, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(field[2, 0, 2], np.pi / 2.0, atol=1e-6)
    np.testing.assert_allclose(field[2, 0, 3], np.sqrt(2.0), atol=1e-6)
    np.testing.assert_allclose(field[2, 0, 4], np.sqrt(2.0), atol=1e-6)


def _write_source_npz(path: Path, hidden_dir: Path | None = None) -> None:
    trajectories = [
        np.zeros((3, 2, 4), dtype=np.float32),
        np.ones((3, 2, 4), dtype=np.float32),
    ]
    ranges = [
        np.asarray([[5, 7], [8, 10], [11, 13]], dtype=np.int32),
        np.asarray([[4, 6], [7, 9], [10, 12]], dtype=np.int32),
    ]
    payload = {
        "ids": np.asarray(["error", "correct"], dtype=object),
        "problem_ids": np.asarray([11, 12]),
        "gold_error_step": np.asarray([1, -1], dtype=np.int32),
        "is_correct": np.asarray([0, 1], dtype=np.int32),
        "stepvec": _object_array(trajectories),
        "sv_layers": np.asarray([8, 12], dtype=np.int32),
        "step_token_ranges": _object_array(ranges),
    }
    if hidden_dir is not None:
        hidden_dir.mkdir(parents=True, exist_ok=True)
        hidden_files = np.asarray(["error.npy", "correct.npy"], dtype=object)
        for index, name in enumerate(hidden_files):
            values = np.arange(9 * 2 * 4, dtype=np.float32).reshape(9, 2, 4) + index
            np.save(hidden_dir / str(name), values)
        payload.update(
            {
                "hidden_files": hidden_files,
                "hidden_dir": np.asarray(str(hidden_dir), dtype=object),
                "hidden_layers": np.asarray([10, 14], dtype=np.int32),
            }
        )
    np.savez_compressed(path, **payload)


def test_loader_uses_authoritative_first_error_labels_and_layers(tmp_path: Path) -> None:
    path = tmp_path / "full_toy.npz"
    _write_source_npz(path)
    source = load_step_geometry_dataset(path, layers="12")

    assert source.n_samples == 2
    assert source.layer_ids.tolist() == [12]
    assert source.is_correct.tolist() == [0, 1]
    assert source.gold_error_step.tolist() == [1, -1]
    assert source.step_lengths[0].tolist() == [3.0, 3.0, 3.0]


def test_token_shards_align_event_to_first_token_of_error_step(tmp_path: Path) -> None:
    path = tmp_path / "full_toy.npz"
    hidden_dir = tmp_path / "hidden"
    _write_source_npz(path, hidden_dir)
    source = load_step_geometry_dataset(path)
    axis = load_token_axis(source, hidden_dir=hidden_dir, layers="14")

    assert axis.layer_ids.tolist() == [14]
    assert axis.event_indices.tolist() == [3, -1]
    assert axis.trajectories[0].shape == (9, 1, 4)
    np.testing.assert_allclose(axis.controls[0][3, 2], np.log1p(3.0), atol=1e-7)
    np.testing.assert_allclose(axis.controls[0][3, 3], 0.0, atol=1e-7)


def _synthetic_event_dataset(n_pairs: int = 12) -> StepGeometryDataset:
    trajectories: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    gold: list[int] = []
    correct: list[int] = []
    for is_error in ([1] * n_pairs + [0] * n_pairs):
        if is_error:
            points = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
            gold.append(2)
            correct.append(0)
        else:
            points = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
            gold.append(-1)
            correct.append(1)
        trajectories.append(np.asarray(points, dtype=np.float32)[:, None, :])
        item_ranges = np.asarray([[2 * i, 2 * i + 1] for i in range(5)], dtype=np.int64)
        ranges.append(item_ranges)
        lengths.append(np.full(5, 2.0, dtype=np.float64))
    count = len(trajectories)
    return StepGeometryDataset(
        source_path="synthetic",
        vector_key="stepvec",
        trajectories=trajectories,
        original_indices=np.arange(count, dtype=np.int64),
        ids=np.asarray([str(i) for i in range(count)], dtype=object),
        problem_ids=np.arange(count, dtype=np.int64),
        is_correct=np.asarray(correct, dtype=np.int64),
        gold_error_step=np.asarray(gold, dtype=np.int64),
        n_steps=np.full(count, 5, dtype=np.int64),
        step_ranges=ranges,
        step_lengths=lengths,
        layer_ids=np.asarray([8], dtype=np.int64),
        hidden_dim=2,
        skipped={},
        metadata={},
    )


def test_event_audit_recovers_injected_corner_after_nuisance_control(tmp_path: Path) -> None:
    source = _synthetic_event_dataset()
    source_matches = match_correct_pseudo_events(source)
    axis = make_step_axis(source)
    matches = map_matches_to_axis(source_matches, source, axis)
    result = run_first_error_geometry_audit(
        axis,
        matches,
        FirstErrorGeometryConfig(
            device="cpu",
            batch_size=32,
            bootstrap=40,
            permutations=99,
            nuisance_folds=4,
            step_offsets=(-1, 0, 1),
        ),
    )

    row = next(
        item
        for item in result.event_rows
        if item["variant"] == "nuisance_residual"
        and item["metric"] == "turn_angle_rad"
        and item["layer"] == 8
        and item["offset"] == 0
    )
    assert row["n_pairs"] == 12
    assert row["pair_coverage"] == 1.0
    assert row["paired_difference"] > 1.4
    assert row["matched_event_auroc"] == 1.0

    paths = write_geometry_audit_report(result, tmp_path / "report", render_plots=False)
    for value in paths.values():
        assert Path(value).exists()
