from __future__ import annotations

import numpy as np

from functional_divergence.raw_residual import (
    inspect_raw_residual_source,
    load_matched_raw_residual,
)


def _states(row: int) -> np.ndarray:
    values = np.empty((8, 3, 4), dtype=np.float32)
    for token in range(8):
        for layer in range(3):
            values[token, layer] = row + token * 0.1 + layer * 0.01 + np.arange(4)
    return values


def _metadata_arrays() -> dict[str, np.ndarray]:
    ranges = np.empty(4, dtype=object)
    for row in range(4):
        ranges[row] = np.asarray([[10, 11], [12, 14], [15, 17]], dtype=np.int64)
    return {
        "gold_error_step": np.asarray([1, -1, 1, -1]),
        "problem_ids": np.asarray([10, 20, 30, 40]),
        "step_token_ranges": ranges,
    }


def test_canonical_full_loader_reads_raw_hidden_shards_and_event_windows(tmp_path) -> None:
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    files = []
    for row in range(4):
        name = f"gsm8k-{row}.npy"
        np.save(hidden / name, _states(row))
        files.append(name)
    manifest = tmp_path / "full_gsm8k.npz"
    np.savez(
        manifest,
        **_metadata_arrays(),
        hidden_files=np.asarray(files, dtype=object),
        hidden_layers=np.asarray([10, 14, 18]),
        hidden_stored=np.asarray(True),
    )

    info = inspect_raw_residual_source(manifest, hidden_dir=hidden)
    data = load_matched_raw_residual(
        manifest, hidden_dir=hidden, offsets=(-1, 0, 1), layers=(10, 18)
    )

    assert info["source_format"] == "canonical_full_hidden_shards_v1"
    assert info["n_records"] == 4
    assert info["first_shard_shape"] == [8, 3, 4]
    assert data.states.shape == (4, 3, 2, 4)
    assert data.layer_ids.tolist() == [10, 18]
    assert data.metadata["representation_scope"] == "raw_residual_stream"
    assert data.metadata["depth_semantics"] == "sparse_depth_interval"
    assert data.metadata["n_retained_pairs"] == 2
    # Error step 1 starts at absolute token 12; shards start at response token 10.
    assert np.allclose(data.states[0, 1], _states(0)[2, [0, 2]])


def test_raw_loader_groups_distinct_pairs_that_share_a_problem(tmp_path) -> None:
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    files = []
    for row in range(4):
        name = f"gsm8k-{row}.npy"
        np.save(hidden / name, _states(row))
        files.append(name)
    metadata = _metadata_arrays()
    manifest = tmp_path / "full_gsm8k.npz"
    np.savez(
        manifest,
        **{**metadata, "problem_ids": np.asarray([99, 20, 99, 40])},
        hidden_files=np.asarray(files, dtype=object),
        hidden_layers=np.asarray([10, 14, 18]),
        hidden_stored=np.asarray(True),
    )

    data = load_matched_raw_residual(manifest, hidden_dir=hidden, offsets=(-1, 0))

    assert data.component_ids[0] == data.component_ids[2]
    assert data.metadata["component_grouping"] == "matched_rows_plus_problem_ids"


def test_exact_manifest_loader_resolves_relative_response_state_files(tmp_path) -> None:
    shard_dir = tmp_path / "states"
    shard_dir.mkdir()
    files = []
    for row in range(4):
        path = shard_dir / f"trace-{row}.npy"
        np.save(path, _states(row))
        files.append(str(path.relative_to(tmp_path)))
    manifest = tmp_path / "trace.npz"
    metadata = _metadata_arrays()
    metadata.pop("problem_ids")
    np.savez(
        manifest,
        **metadata,
        problem_group_id=np.asarray([7, 20, 7, 40]),
        response_token_state_files=np.asarray(files, dtype=object),
        response_token_state_layers=np.asarray([8, 10, 12]),
        response_token_state_counts=np.full(4, 8),
        response_token_state_storage_kind=np.asarray("per_chain_npy_shards_v1"),
        response_token_state_snapshot_kind=np.asarray("raw_residual_stream"),
        response_token_ranges=np.asarray([[10, 18]] * 4),
    )

    info = inspect_raw_residual_source(manifest)
    data = load_matched_raw_residual(manifest, offsets=(-1, 0), max_pairs=1)

    assert info["source_format"] == "exact_response_state_manifest_v1"
    assert info["snapshot_kind"] == "raw_residual_stream"
    assert data.states.shape == (2, 2, 3, 4)
    assert data.metadata["n_retained_pairs"] == 1
    assert data.metadata["source_format"] == "exact_response_state_manifest_v1"
    assert data.metadata["problem_group_field"] == "problem_group_id"


def test_loader_rejects_exact_manifest_without_raw_residual_provenance(tmp_path) -> None:
    shard = tmp_path / "trace.npy"
    np.save(shard, _states(0))
    manifest = tmp_path / "trace.npz"
    ranges = np.empty(1, dtype=object)
    ranges[0] = np.asarray([[0, 2], [3, 5]])
    np.savez(
        manifest,
        gold_error_step=np.asarray([-1]),
        problem_ids=np.asarray([0]),
        step_token_ranges=ranges,
        response_token_state_files=np.asarray([shard.name], dtype=object),
        response_token_state_layers=np.asarray([8, 10, 12]),
        response_token_state_snapshot_kind=np.asarray("unverified"),
    )

    try:
        inspect_raw_residual_source(manifest)
    except ValueError as exc:
        assert "raw_residual_stream" in str(exc)
    else:
        raise AssertionError("unverified snapshots must fail closed")
