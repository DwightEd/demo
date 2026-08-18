from __future__ import annotations

from pathlib import Path

import numpy as np

from token_transition_dynamics.experiment import (
    ExperimentConfig,
    TokenTransitionExperiment,
)


def _write_manifest(root: Path, layers: list[int]) -> Path:
    domain_dir = root / "synthetic" / "selected"
    shard_dir = domain_dir / "raw_residual_stream"
    shard_dir.mkdir(parents=True)

    shard_paths: list[str] = []
    step_ranges: list[np.ndarray] = []
    for row in range(4):
        states = np.zeros((6, len(layers), 4), dtype=np.float32)
        shard = shard_dir / f"chain_{row}.npy"
        np.save(shard, states)
        shard_paths.append(str(shard.relative_to(domain_dir)))
        step_ranges.append(np.asarray([[0, 2], [3, 5]], dtype=np.int64))

    manifest = domain_dir / "trace.raw_residual_stream.npz"
    np.savez(
        manifest,
        response_token_state_files=np.asarray(shard_paths, dtype=object),
        response_token_state_layers=np.asarray(layers, dtype=np.int64),
        response_token_state_storage_kind=np.asarray("per_chain_npy_shards_v1"),
        response_token_state_snapshot_kind=np.asarray("raw_residual_stream"),
        response_token_state_counts=np.full(4, 6, dtype=np.int64),
        response_token_ranges=np.tile(np.asarray([0, 5]), (4, 1)),
        step_token_ranges=np.asarray(step_ranges, dtype=object),
        gold_error_step=np.asarray([-1, -1, 1, 1], dtype=np.int64),
        problem_group_id=np.asarray(["p0", "p1", "p2", "p3"], dtype=object),
    )
    return manifest


def test_preflight_accepts_sparse_layers_as_independent_token_trajectories(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path, layers=[8, 10, 12])
    config = ExperimentConfig(
        data_root=tmp_path,
        domains=("synthetic",),
        output_dir=tmp_path / "out",
    )

    report = TokenTransitionExperiment(config).inspect()

    assert report["synthetic"]["layers"] == [8, 10, 12]
    assert report["synthetic"]["analysis_axis"] == "token_time_within_each_stored_layer"
    assert report["synthetic"]["depth_semantics"] == "sparse_observation_layers"


def _write_dynamics_dataset(root: Path, n_groups: int = 48) -> None:
    domain_dir = root / "synthetic" / "selected"
    shard_dir = domain_dir / "raw_residual_stream"
    shard_dir.mkdir(parents=True)
    rng = np.random.default_rng(123)
    layers = np.asarray([8, 10, 12], dtype=np.int64)
    files: list[str] = []
    ranges: list[np.ndarray] = []
    errors: list[int] = []
    groups: list[str] = []

    response_start = 100
    for group in range(n_groups):
        for is_error in (False, True):
            token = np.linspace(0.0, 2.5, 40)
            states = np.zeros((40, 3, 8), dtype=np.float32)
            for layer in range(3):
                phase = 0.15 * layer
                states[:, layer, 0] = token
                states[:, layer, 1] = np.sin(token + phase)
                states[:, layer, 2] = np.cos(token + phase)
                states[:, layer, 3] = token**2 / 4.0
            states += rng.normal(scale=0.004, size=states.shape)
            if is_error:
                alternating = np.where(np.arange(16) % 2 == 0, 1.0, -1.0)
                states[16:32, :, 5] += 2.5 * alternating[:, None]
                states[16:32, :, 6] -= 1.8 * alternating[:, None]
            row = len(files)
            shard = shard_dir / f"chain_{row}.npy"
            np.save(shard, states)
            files.append(str(shard.relative_to(domain_dir)))
            ranges.append(
                np.asarray([[100, 115], [116, 131], [132, 139]], dtype=np.int64)
            )
            errors.append(1 if is_error else -1)
            groups.append(f"problem-{group:03d}")

    np.savez(
        domain_dir / "trace.raw_residual_stream.npz",
        response_token_state_files=np.asarray(files, dtype=object),
        response_token_state_layers=layers,
        response_token_state_storage_kind=np.asarray("per_chain_npy_shards_v1"),
        response_token_state_snapshot_kind=np.asarray("raw_residual_stream"),
        response_token_state_counts=np.full(len(files), 40, dtype=np.int64),
        response_token_ranges=np.tile(
            np.asarray([response_start, response_start + 39]), (len(files), 1)
        ),
        step_token_ranges=np.asarray(ranges, dtype=object),
        gold_error_step=np.asarray(errors, dtype=np.int64),
        problem_group_id=np.asarray(groups, dtype=object),
        chain_idx=np.arange(len(files), dtype=np.int64),
    )


def test_token_level_experiment_scores_raw_window_geometry_without_labels(tmp_path: Path) -> None:
    _write_dynamics_dataset(tmp_path)
    config = ExperimentConfig(
        data_root=tmp_path,
        domains=("synthetic",),
        output_dir=tmp_path / "out",
        window_size=12,
        neighbors=5,
        tle_centers=4,
        train_windows_per_chain=8,
        calibration_windows_per_chain=6,
        max_test_windows_per_chain=12,
        position_bins=2,
        min_baseline_samples=3,
        bootstrap_samples=30,
        seed=17,
    )

    report = TokenTransitionExperiment(config).run()

    assert report["protocol"]["uses_step_pooling"] is False
    assert report["protocol"]["uses_step_scores"] is False
    assert report["protocol"]["uses_pca"] is False
    assert report["protocol"]["uses_kmeans"] is False
    assert report["protocol"]["uses_chain_correctness_to_select_fit_cohort"] is True
    assert report["protocol"]["uses_first_error_locations_for_model_fit"] is False
    assert set(report["models"]) == {"geometry", "dynamics", "combined"}
    assert report["models"]["combined"]["test_token"]["auroc"] > 0.80
    assert report["data"]["split_counts"]["train_error"] > 0
    assert report["data"]["fit_counts"]["error_chains"] == 0
    assert report["data"]["domains"]["synthetic"]["layers"] == [8, 10, 12]
    test_error_chains = report["data"]["split_counts"]["test_error"]
    assert report["data"]["window_counts"]["test"] == 24 * test_error_chains
    assert report["data"]["window_counts"]["test_positive"] == 7 * test_error_chains
    assert (tmp_path / "out" / "results.json").is_file()
    assert (tmp_path / "out" / "summary.txt").is_file()
