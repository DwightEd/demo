import numpy as np
import pytest
import json

from token_residual_dispersion.audit import analyze_trace
from token_residual_dispersion.data import TokenStateTrace, load_token_state_traces
from token_residual_dispersion.metrics import (
    DispersionConfig,
    block_writes_from_states,
    component_conflict,
    compute_dispersion_field,
    depth_deltas_from_states,
)
from token_residual_dispersion.selftest import run_selftest, synthetic_trace
from token_residual_dispersion.cli import main as dispersion_main


def test_sparse_layers_are_not_mislabeled_as_block_writes():
    states = np.zeros((5, 3, 7))
    with pytest.raises(ValueError, match="consecutive"):
        block_writes_from_states(states, layers=[0, 2, 4])


def test_sparse_depth_pilot_preserves_honest_intervals():
    states = np.zeros((5, 3, 7))
    states[:, 1, 0] = 2.0
    states[:, 2, 0] = 5.0
    deltas, source, target = depth_deltas_from_states(
        states, layers=[8, 10, 12], allow_sparse=True
    )
    np.testing.assert_array_equal(source, [8, 10])
    np.testing.assert_array_equal(target, [10, 12])
    np.testing.assert_allclose(deltas[:, :, 0], [[2.0, 3.0]] * 5)


def test_sparse_pilot_is_labeled_as_interval_delta():
    states = np.zeros((6, 3, 5))
    states[:, 1, 0] = np.arange(6)
    states[:, 2, 1] = np.arange(6)
    trace = TokenStateTrace("legacy", states, np.asarray([8, 10, 12]), "test")
    analysis = analyze_trace(
        trace,
        DispersionConfig(windows=(4,), min_tokens=2),
        allow_unverified_snapshots=True,
        allow_sparse_depth_deltas=True,
    )
    assert analysis["delta_kind"] == "sparse_multi_block_depth_interval_delta_pilot"
    np.testing.assert_array_equal(analysis["depth_spans"], [2, 2])


def test_weighted_scatter_identity_and_directional_contrast():
    rng = np.random.default_rng(2)
    coherent = np.tile(np.eye(1, 24, 0), (32, 2, 1))
    coherent += 0.01 * rng.normal(size=coherent.shape)
    diffuse = rng.normal(size=(32, 2, 24))
    writes = np.concatenate([coherent, diffuse], axis=0)
    field = compute_dispersion_field(
        writes,
        DispersionConfig(windows=(12,), min_tokens=4, decay=0.2),
    )
    assert np.nanmax(field["identity_error"]) < 1e-10
    assert np.nanmean(field["pair_dispersion"][48:]) > np.nanmean(
        field["pair_dispersion"][16:28]
    ) + 0.5


def test_causal_windows_do_not_read_future_tokens():
    trace = synthetic_trace()
    writes, _ = block_writes_from_states(trace.states, trace.layers)
    config = DispersionConfig(windows=(8, 16), min_tokens=3)
    reference = compute_dispersion_field(writes, config)
    changed = writes.copy()
    changed[50:] = np.random.default_rng(9).normal(size=changed[50:].shape)
    counterfactual = compute_dispersion_field(changed, config)
    np.testing.assert_allclose(
        reference["pair_dispersion"][:50],
        counterfactual["pair_dispersion"][:50],
        equal_nan=True,
    )


def test_min_tokens_counts_nonzero_directions():
    writes = np.zeros((6, 1, 4))
    writes[0, 0, 0] = 1.0
    writes[5, 0, 1] = 1.0
    field = compute_dispersion_field(
        writes,
        DispersionConfig(windows=(6,), min_tokens=4),
    )
    assert np.isnan(field["pair_dispersion"][-1, 0, 0])
    assert field["valid_tokens"][-1, 0, 0] == 2
    assert field["mean_write_norm"][-1, 0, 0] == pytest.approx(2 / 6)

    writes[1, 0, 0] = 1.0
    writes[2, 0, 0] = 1.0
    enough = compute_dispersion_field(
        writes,
        DispersionConfig(windows=(6,), min_tokens=4),
    )
    assert np.isfinite(enough["pair_dispersion"][-1, 0, 0])
    assert enough["mean_write_norm"][-1, 0, 0] == pytest.approx(4 / 6)


def test_component_conflict_detects_opposed_writes():
    attention = np.ones((4, 3, 8))
    mlp = -attention
    conflict = component_conflict(attention, mlp)
    np.testing.assert_allclose(conflict["antagonism"], 1.0)
    np.testing.assert_allclose(conflict["cancellation"], 1.0)


def test_synthetic_transition_selftest_passes():
    assert run_selftest()["passed"] is True


def test_extraction_manifest_loader_resolves_relative_shards(tmp_path):
    states = np.zeros((6, 3, 5), dtype=np.float32)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.save(shard_dir / "chain_7.npy", states)
    manifest = tmp_path / "features.npz"
    np.savez(
        manifest,
        response_token_state_files=np.asarray(["shards/chain_7.npy"], dtype=object),
        response_token_state_counts=np.asarray([6]),
        response_token_state_layers=np.asarray([4, 5, 6]),
        chain_idx=np.asarray([7]),
    )
    traces = load_token_state_traces(manifest)
    assert traces[0].trace_id == "7"
    np.testing.assert_array_equal(traces[0].layers, [4, 5, 6])


def test_direct_state_input_requires_layer_metadata(tmp_path):
    path = tmp_path / "states.npy"
    np.save(path, np.zeros((4, 3, 5)))
    with pytest.raises(ValueError, match="layer ids"):
        load_token_state_traces(path)


def test_unverified_snapshot_provenance_is_rejected():
    trace = synthetic_trace()
    unverified = type(trace)(trace.trace_id, trace.states, trace.layers, trace.source, "unverified")
    with pytest.raises(ValueError, match="provenance"):
        analyze_trace(unverified, DispersionConfig(windows=(4,), min_tokens=3))


def test_legacy_manifest_cli_streams_and_labels_pilot(tmp_path):
    states = np.random.default_rng(4).normal(size=(8, 3, 6)).astype(np.float16)
    shard_dir = tmp_path / "trace.response_states.run"
    shard_dir.mkdir()
    np.save(shard_dir / "row.npy", states)
    manifest = tmp_path / "trace.npz"
    np.savez(
        manifest,
        response_token_state_files=np.asarray(
            ["trace.response_states.run/row.npy"], dtype=object
        ),
        response_token_state_counts=np.asarray([8]),
        response_token_state_layers=np.asarray([8, 10, 12]),
        chain_idx=np.asarray([3]),
    )
    output_dir = tmp_path / "audit"
    assert dispersion_main([
        "--input", str(manifest),
        "--output-dir", str(output_dir),
        "--windows", "4",
        "--legacy-sparse-pilot",
    ]) == 0
    summary = json.loads((output_dir / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["delta_kinds"] == ["sparse_multi_block_depth_interval_delta_pilot"]
    assert summary["traces"][0]["depth_intervals"] == [[8, 10], [10, 12]]
