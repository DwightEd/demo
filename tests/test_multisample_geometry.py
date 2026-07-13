from __future__ import annotations

from pathlib import Path

import numpy as np

from prompt_control_flow.flow_signature_data import FlowTrajectoryDataset
from prompt_control_flow.multisample_geometry import (
    MultisampleGeometryConfig,
    _common_finite_scores,
    build_geometry_profiles,
    run_multisample_geometry_audit,
    write_multisample_geometry_outputs,
)


def _dataset(n_problems: int = 12) -> FlowTrajectoryDataset:
    rng = np.random.default_rng(7)
    trajectories: list[np.ndarray] = []
    problem_ids: list[int] = []
    sample_idx: list[int] = []
    y_error: list[int] = []
    for problem in range(n_problems):
        for sample in range(3):
            if sample < 2:
                points = np.asarray([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], dtype=np.float32)
                label = 0
            else:
                points = np.asarray([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)], dtype=np.float32)
                label = 1
            points += rng.normal(0.0, 0.01, size=points.shape).astype(np.float32)
            layers = np.stack([points, 1.7 * points], axis=1)
            trajectories.append(layers)
            problem_ids.append(problem)
            sample_idx.append(sample)
            y_error.append(label)
    y = np.asarray(y_error, dtype=np.int64)
    count = len(trajectories)
    return FlowTrajectoryDataset(
        source_path="synthetic",
        vector_key="synthetic",
        trajectories=trajectories,
        original_indices=np.arange(count, dtype=np.int64),
        problem_ids=np.asarray(problem_ids, dtype=np.int64),
        sample_idx=np.asarray(sample_idx, dtype=np.int64),
        y_error=y,
        is_correct=1 - y,
        n_steps=np.full(count, 5, dtype=np.int64),
        response_chars=np.full(count, 100, dtype=np.int64),
        layer_ids=np.asarray([8, 16], dtype=np.int64),
        hidden_dim=2,
        label_policy="answer_format_ok",
        skipped={},
        metadata={},
    )


def test_profile_builder_preserves_metric_shape_and_coverage() -> None:
    field = np.full((4, 2, 5), np.nan, dtype=np.float32)
    field[1:, :, 0] = 1.0
    field[1:, :, 1] = 0.5
    field[1:3, :, 2:] = 0.25
    profiles = build_geometry_profiles([field], phase_points=8)
    assert profiles.profiles.shape == (1, 8, 2, 5)
    assert profiles.static_features.shape == (1, 2, 5, 4)
    assert profiles.valid_features.all()
    np.testing.assert_allclose(profiles.profiles[0, :, 0, 0], 1.0)


def test_dynamic_static_comparison_uses_identical_finite_support() -> None:
    first, second, coverage = _common_finite_scores(
        np.asarray([1.0, np.nan, 3.0, 4.0]),
        np.asarray([1.5, 2.0, np.nan, 4.5]),
    )
    assert coverage == 0.5
    assert np.isfinite(first).tolist() == [True, False, False, True]
    assert np.isfinite(second).tolist() == [True, False, False, True]


def test_same_problem_dynamic_geometry_recovers_injected_corner(tmp_path: Path) -> None:
    dataset = _dataset()
    report, packed = run_multisample_geometry_audit(
        dataset,
        MultisampleGeometryConfig(
            phase_points=8,
            batch_size=64,
            compute_device="cpu",
            folds=4,
            bootstrap=40,
            permutations=39,
            seed=5,
        ),
    )
    scores = {row["name"]: row for row in report["headline_scores"]}
    assert scores["geometry.dynamic_support_mean"]["same_problem_auroc"] > 0.95
    assert report["meta"]["contrastive_problems"] == 12
    assert packed["scores"].shape[0] == dataset.n_samples

    paths = write_multisample_geometry_outputs(
        report,
        packed,
        output=tmp_path / "scores.npz",
        output_dir=tmp_path / "audit",
        keep_profiles=False,
        render_plots=False,
    )
    for value in paths.values():
        assert Path(value).exists()
    saved = np.load(paths["scores"], allow_pickle=True)
    assert "profiles" not in saved.files
