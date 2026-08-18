from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from token_transition_dynamics.experiment import ExperimentConfig, TokenTransitionExperiment


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


def test_run_rejects_nonconsecutive_hidden_layers(tmp_path: Path) -> None:
    _write_manifest(tmp_path, layers=[0, 2, 4])
    config = ExperimentConfig(
        data_root=tmp_path,
        domains=("synthetic",),
        output_dir=tmp_path / "out",
        pca_dim=2,
        clusters=(1, 2),
        bootstrap_samples=10,
    )

    with pytest.raises(ValueError, match="consecutive hidden layers"):
        TokenTransitionExperiment(config).run()


def _write_dynamics_dataset(root: Path, n_groups: int = 48) -> None:
    domain_dir = root / "synthetic" / "selected"
    shard_dir = domain_dir / "raw_residual_stream"
    shard_dir.mkdir(parents=True)
    rng = np.random.default_rng(123)
    layers = np.asarray([0, 1, 2], dtype=np.int64)
    files: list[str] = []
    ranges: list[np.ndarray] = []
    errors: list[int] = []
    groups: list[str] = []

    response_start = 100
    for group in range(n_groups):
        for is_error in (False, True):
            states = np.empty((8, 3, 4), dtype=np.float32)
            initial = rng.normal(size=(8, 4))
            states[:, 0] = initial
            regime = np.where(initial[:, :1] >= 0.0, 1.0, -1.0)
            states[:, 1] = initial + regime * np.asarray([0.8, -0.4, 0.2, 0.0])
            states[:, 2] = states[:, 1] + regime * np.asarray([-0.2, 0.7, 0.0, 0.3])
            states += rng.normal(scale=0.025, size=states.shape)
            if is_error:
                states[4:, 2] += np.asarray([0.0, 0.0, 3.5, -2.5])
            row = len(files)
            shard = shard_dir / f"chain_{row}.npy"
            np.save(shard, states)
            files.append(str(shard.relative_to(domain_dir)))
            ranges.append(np.asarray([[100, 103], [104, 107]], dtype=np.int64))
            errors.append(1 if is_error else -1)
            groups.append(f"problem-{group:03d}")

    np.savez(
        domain_dir / "trace.raw_residual_stream.npz",
        response_token_state_files=np.asarray(files, dtype=object),
        response_token_state_layers=layers,
        response_token_state_storage_kind=np.asarray("per_chain_npy_shards_v1"),
        response_token_state_snapshot_kind=np.asarray("raw_residual_stream"),
        response_token_state_counts=np.full(len(files), 8, dtype=np.int64),
        response_token_ranges=np.tile(
            np.asarray([response_start, response_start + 7]), (len(files), 1)
        ),
        step_token_ranges=np.asarray(ranges, dtype=object),
        gold_error_step=np.asarray(errors, dtype=np.int64),
        problem_group_id=np.asarray(groups, dtype=object),
        chain_idx=np.arange(len(files), dtype=np.int64),
    )


def test_token_level_experiment_fits_four_arms_without_step_pooling(tmp_path: Path) -> None:
    _write_dynamics_dataset(tmp_path)
    config = ExperimentConfig(
        data_root=tmp_path,
        domains=("synthetic",),
        output_dir=tmp_path / "out",
        pca_dim=4,
        clusters=(1, 2),
        tokens_per_chain=8,
        max_test_tokens_per_chain=8,
        max_pca_rows=2048,
        bootstrap_samples=30,
        seed=17,
    )

    report = TokenTransitionExperiment(config).run()

    assert report["protocol"]["uses_step_pooling"] is False
    assert report["protocol"]["uses_step_scores"] is False
    assert report["protocol"]["uses_error_labels_for_model_fit"] is False
    assert set(report["models"]) == {
        "euclidean_k1",
        "euclidean_k2",
        "spherical_k1",
        "spherical_k2",
    }
    assert report["models"]["euclidean_k1"]["test_token"]["auroc"] > 0.80
    assert (
        report["models"]["euclidean_k2"]["calibration_correct_nll"]
        < report["models"]["euclidean_k1"]["calibration_correct_nll"]
    )
    assert report["data"]["split_counts"]["train_error"] > 0
    assert report["data"]["fit_counts"]["error_chains"] == 0
    assert (tmp_path / "out" / "results.json").is_file()
    assert (tmp_path / "out" / "summary.txt").is_file()
