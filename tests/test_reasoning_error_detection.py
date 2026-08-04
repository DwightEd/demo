from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reasoning_error_detection.data import (
    FeatureConfig,
    ProcessBenchFeatureLoader,
)


def _write_processbench_features(path: Path) -> None:
    chain_idx = np.asarray([10, 11, 12], dtype=np.int64)
    n_steps = np.asarray([2, 3, 2], dtype=np.int64)
    state_chain_idx = np.repeat(chain_idx, n_steps)
    state_step_idx = np.concatenate(
        [np.arange(count, dtype=np.int64) for count in n_steps]
    )
    layers = np.asarray([8, 16], dtype=np.int64)
    pre = np.arange(7 * 2 * 4, dtype=np.float32).reshape(7, 2, 4) / 10.0
    end = pre.copy()
    end[:, 0, 0] += np.asarray([0.1, 0.2, 0.1, 2.0, 0.3, 3.0, 0.2])
    end[:, 1, 1] += np.asarray([0.2, 0.1, 0.2, 1.5, 0.4, 2.5, 0.1])
    np.save(path.parent / "pre_states.npy", pre)
    np.save(path.parent / "end_states.npy", end)

    ranges = np.full((3, 3, 2), -1, dtype=np.int64)
    ranges[0, :2] = [[10, 11], [12, 15]]
    ranges[1, :3] = [[20, 21], [22, 24], [25, 29]]
    ranges[2, :2] = [[30, 33], [34, 35]]
    step_scores = np.full((3, 3, 3), np.nan, dtype=np.float32)
    step_scores[0, :2] = [[0.2, 2.0, 0.1], [0.3, 4.0, 0.2]]
    step_scores[1, :3] = [[0.2, 2.0, 0.1], [0.9, 3.0, 0.8], [0.7, 5.0, 0.7]]
    step_scores[2, :2] = [[0.8, 4.0, 0.9], [0.6, 2.0, 0.6]]

    np.savez_compressed(
        path,
        chain_idx=chain_idx,
        problem_group_ids=np.asarray([100, 101, 102], dtype=np.int64),
        gold_error_step=np.asarray([-1, 1, 0], dtype=np.int64),
        n_steps=n_steps,
        step_token_ranges=ranges,
        step_score_names=np.asarray(
            ["token_entropy", "step_len", "icr_mean"], dtype=object
        ),
        step_scores=step_scores,
        step_pre_state_memmap_path=np.asarray("pre_states.npy", dtype=object),
        step_pre_state_memmap_count=np.asarray(7, dtype=np.int64),
        step_pre_state_vector_chain_idx=state_chain_idx,
        step_pre_state_vector_step_idx=state_step_idx,
        step_end_state_memmap_path=np.asarray("end_states.npy", dtype=object),
        step_end_state_memmap_count=np.asarray(7, dtype=np.int64),
        step_end_state_vector_chain_idx=state_chain_idx,
        step_end_state_vector_step_idx=state_step_idx,
        step_layer_state_vector_layers=layers,
    )


def test_loader_builds_label_safe_first_error_rows_from_processbench_features(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.npz"
    _write_processbench_features(path)

    data = ProcessBenchFeatureLoader(
        FeatureConfig(layers=(8, 16), projection_dim=2, seed=7, device="cpu")
    ).load(path)

    assert data.chain_idx.tolist() == [10, 10, 11, 11, 11, 12, 12]
    assert data.step_idx.tolist() == [0, 1, 0, 1, 2, 0, 1]
    assert data.onset_label.tolist() == [0, 0, 0, 1, -1, 1, -1]
    assert data.onset_eligible.tolist() == [True, True, True, True, False, True, False]
    assert data.chain_error.tolist() == [0, 0, 1, 1, 1, 1, 1]
    assert data.control_names == (
        "control.log1p_step_index",
        "control.log1p_step_tokens",
        "control.log1p_previous_step_tokens",
        "control.log1p_cumulative_tokens",
    )
    assert data.output_names == ("output.token_entropy",)
    assert data.routing_names == ("routing.icr_mean",)
    assert data.residual_features.shape == (7, 12)
    assert np.isfinite(data.residual_features).all()


def test_loader_fails_when_processbench_residual_state_views_are_missing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.npz"
    np.savez_compressed(
        path,
        chain_idx=np.asarray([0]),
        problem_ids=np.asarray([0]),
        gold_error_step=np.asarray([-1]),
        n_steps=np.asarray([1]),
        step_token_ranges=np.asarray([[[1, 2]]]),
    )

    with pytest.raises(ValueError, match="step_pre_state"):
        ProcessBenchFeatureLoader(FeatureConfig(device="cpu")).load(path)
