from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prompt_control_flow.cli.extract_mechanisms import parse_layers
from prompt_control_flow.layer_time_geometry import (
    LAYER_TIME_FIELD_NAMES,
    LayerTimeGeometryConfig,
    append_layer_time_geometry,
    canonicalize_layer_time_input,
    connection_path_discrepancy,
    make_group_folds,
    score_layer_time_fold,
)
from prompt_control_flow.metrics import (
    compute_step_layer_state_vectors,
    compute_step_state_vectors,
)
from prompt_control_flow.schema import inspect_npz_schema


def test_layer_state_pooling_preserves_depth_axis_and_flat_compatibility() -> None:
    hidden = []
    for layer in range(4):
        arr = np.arange(18, dtype=np.float32).reshape(6, 3) + 100.0 * layer
        hidden.append(arr)
    tensor = compute_step_layer_state_vectors(
        hidden,
        step_ranges=[(1, 2), (3, 5)],
        layers=[1, 2, 3],
    )
    flat = compute_step_state_vectors(
        hidden,
        step_ranges=[(1, 2), (3, 5)],
        layers=[1, 2, 3],
    )
    assert tensor.shape == (2, 3, 3)
    assert flat.shape == (2, 9)
    assert np.allclose(flat, tensor.reshape(2, -1))
    assert parse_layers("all") == ()


def test_problem_group_folds_never_split_same_problem() -> None:
    groups = np.asarray([10, 10, 11, 11, 12, 12, 13, 13])
    folds = make_group_folds(groups, n_folds=3, seed=7)
    seen = np.zeros(groups.size, dtype=np.int32)
    for fold in folds:
        seen[fold] += 1
        for group in np.unique(groups[fold]):
            assert set(np.where(groups == group)[0]).issubset(set(fold.tolist()))
    assert np.all(seen == 1)


def test_shared_layer_geometry_has_zero_depth_rewiring() -> None:
    rng = np.random.default_rng(2)
    base_ref = rng.normal(size=(30, 7)).astype(np.float32)
    base_query = rng.normal(size=(6, 7)).astype(np.float32)
    reference = np.repeat(base_ref[:, None, :], 3, axis=1)
    query = np.repeat(base_query[:, None, :], 3, axis=1)
    chain = np.repeat(np.arange(2), 3)
    step = np.tile(np.arange(3), 2)
    field = score_layer_time_fold(
        reference,
        query,
        chain,
        step,
        LayerTimeGeometryConfig(
            knn_k=6,
            tangent_k=8,
            tangent_rank=3,
            projection_dim=5,
        ),
        np.random.default_rng(4),
    )
    depth_rewire = field[..., LAYER_TIME_FIELD_NAMES.index("depth_neighbor_rewire")]
    depth_tangent = field[..., LAYER_TIME_FIELD_NAMES.index("depth_tangent_drift")]
    holonomy = field[..., LAYER_TIME_FIELD_NAMES.index("plaquette_holonomy")]
    assert np.nanmax(np.abs(depth_rewire)) == 0.0
    assert np.nanmax(np.abs(depth_tangent)) < 1e-3
    assert np.nanmax(np.abs(holonomy)) < 1e-3


def test_holonomy_path_discrepancy_is_gauge_invariant() -> None:
    rng = np.random.default_rng(3)
    qa, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    qb, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    left, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    right, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    original = connection_path_discrepancy(qa, qb)
    transformed = connection_path_discrepancy(left @ qa @ right, left @ qb @ right)
    assert np.isclose(original, transformed, atol=1e-7)
    assert connection_path_discrepancy(qa, qa) < 1e-12


def _toy_metrics() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    n_chains, n_steps, n_layers, hidden = 12, 3, 4, 6
    vectors = []
    vector_chain = []
    vector_step = []
    for chain in range(n_chains):
        problem = chain // 2
        for step in range(n_steps):
            base = rng.normal(size=(n_layers, hidden))
            base += 0.2 * problem + 0.1 * step + 0.05 * np.arange(n_layers)[:, None]
            vectors.append(base)
            vector_chain.append(chain)
            vector_step.append(step)
    return {
        "chain_idx": np.arange(n_chains, dtype=np.int64),
        "problem_id": np.repeat(np.arange(n_chains // 2), 2),
        "gold_error_step": np.asarray([1, -1] * (n_chains // 2), dtype=np.int64),
        "is_correct": np.asarray([0, 1] * (n_chains // 2), dtype=np.int64),
        "sample_idx": np.tile(np.arange(2), n_chains // 2),
        "generator": np.asarray(["toy"] * n_chains, dtype=object),
        "dataset": np.asarray(["toy"] * n_chains, dtype=object),
        "n_steps": np.full(n_chains, n_steps, dtype=np.int64),
        "step_token_ranges": np.zeros((n_chains, n_steps, 2), dtype=np.int32),
        "step_scores": np.zeros((n_chains, n_steps, 0), dtype=np.float32),
        "step_score_names": np.asarray([], dtype="<U96"),
        "chain_scores": np.zeros((n_chains, 0), dtype=np.float32),
        "chain_score_names": np.asarray([], dtype="<U96"),
        "layers": np.arange(1, n_layers + 1, dtype=np.int64),
        "step_layer_state_vectors": np.asarray(vectors, dtype=np.float32),
        "step_layer_state_vector_chain_idx": np.asarray(vector_chain, dtype=np.int64),
        "step_layer_state_vector_step_idx": np.asarray(vector_step, dtype=np.int64),
        "step_layer_state_vector_layers": np.arange(1, n_layers + 1, dtype=np.int64),
        "state_representation_kind": np.asarray("hidden_state", dtype=object),
        "state_pooling_kind": np.asarray("arithmetic_mean_over_step_tokens", dtype=object),
    }


def test_layer_time_field_is_grouped_oof_and_label_free() -> None:
    metrics = _toy_metrics()
    cfg = LayerTimeGeometryConfig(
        n_folds=3,
        knn_k=5,
        tangent_k=6,
        tangent_rank=2,
        projection_dim=5,
        max_reference=40,
        random_seed=11,
    )
    enriched = append_layer_time_geometry(metrics, cfg)
    field = np.asarray(enriched["layer_time_geometry_field"], dtype=np.float32)
    assert field.shape == (12, 3, 4, len(LAYER_TIME_FIELD_NAMES))
    folds = np.asarray(enriched["layer_time_geometry_fold"])
    for problem in np.unique(metrics["problem_id"]):
        assert np.unique(folds[metrics["problem_id"] == problem]).size == 1
    assert np.isfinite(field).any()
    assert "ltg_holonomy_peak" in enriched["step_score_names"].tolist()

    relabeled = dict(metrics)
    relabeled["gold_error_step"] = np.full(12, -1, dtype=np.int64)
    relabeled["is_correct"] = np.ones(12, dtype=np.int64)
    second = append_layer_time_geometry(relabeled, cfg)
    assert np.allclose(
        field,
        np.asarray(second["layer_time_geometry_field"], dtype=np.float32),
        equal_nan=True,
    )


def test_exact_sv_vec_mean_adapter_preserves_layer_tensor() -> None:
    step_vectors = np.empty(2, dtype=object)
    step_vectors[0] = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    step_vectors[1] = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    packed = canonicalize_layer_time_input(
        {
            "sv_vec_mean": step_vectors,
            "layers_used": np.arange(1, 5),
            "problem_ids": np.asarray([4, 5]),
            "is_correct": np.asarray([1, 0]),
            "reasoning_subspace_used": np.asarray(False),
        }
    )
    assert packed["step_layer_state_vectors"].shape == (5, 4, 5)
    assert str(np.asarray(packed["state_pooling_kind"]).item()) == "arithmetic_mean_over_step_tokens"
    assert str(np.asarray(packed["state_representation_kind"]).item()) == "hidden_state"

    with_embedding = canonicalize_layer_time_input(
        {
            "sv_vec_mean": step_vectors,
            "layers_used": np.arange(4),
            "problem_ids": np.asarray([4, 5]),
            "is_correct": np.asarray([1, 0]),
            "reasoning_subspace_used": np.asarray(False),
        }
    )
    assert with_embedding["step_layer_state_vectors"].shape == (5, 3, 5)
    assert with_embedding["step_layer_state_vector_layers"].tolist() == [1, 2, 3]
    assert bool(np.asarray(with_embedding["layer_time_embedding_depth_dropped"]).item())


def test_geometry_memmap_manifest_loads_without_flattening(tmp_path) -> None:
    state_path = tmp_path / "toy.states.npy"
    store = np.lib.format.open_memmap(state_path, mode="w+", dtype=np.float16, shape=(5, 3, 4))
    store[:] = np.arange(store.size, dtype=np.float16).reshape(store.shape)
    store.flush()
    packed = canonicalize_layer_time_input(
        {
            "layer_time_input_path": np.asarray(str(tmp_path / "manifest.npz"), dtype=object),
            "step_layer_state_memmap_path": np.asarray(state_path.name, dtype=object),
            "step_layer_state_memmap_count": np.asarray(5),
            "step_layer_state_vector_chain_idx": np.asarray([0, 0, 0, 1, 1]),
            "step_layer_state_vector_step_idx": np.asarray([0, 1, 2, 0, 1]),
            "step_layer_state_vector_layers": np.asarray([1, 2, 3]),
            "state_representation_kind": np.asarray("hidden_state", dtype=object),
            "state_pooling_kind": np.asarray("arithmetic_mean_over_step_tokens", dtype=object),
        }
    )
    assert packed["step_layer_state_vectors"].shape == (5, 3, 4)
    assert packed["step_layer_state_vectors"].dtype == np.float16
    assert np.allclose(packed["step_layer_state_vectors"][4, 2], store[4, 2])


def test_sv_vec_mean_uses_layered_conversion_memmap_without_ld_copy(tmp_path) -> None:
    values = np.empty(2, dtype=object)
    values[0] = np.ones((2, 4, 3), dtype=np.float16)
    values[1] = np.ones((3, 4, 3), dtype=np.float16) * 2
    packed = canonicalize_layer_time_input(
        {
            "layer_time_input_path": np.asarray(str(tmp_path / "source.npz"), dtype=object),
            "sv_vec_mean": values,
            "layers_used": np.arange(4),
            "problem_ids": np.asarray([1, 2]),
            "is_correct": np.asarray([1, 0]),
            "reasoning_subspace_used": np.asarray(False),
        }
    )
    assert "step_state_vectors" not in packed
    assert packed["step_layer_state_vectors"].shape == (5, 3, 3)
    assert packed["step_layer_state_vectors"].dtype == np.float16
    assert Path(str(np.asarray(packed["step_layer_state_temporary_memmap_path"]).item())).exists()


def test_projected_sv_vec_mean_fails_before_creating_conversion_memmap(tmp_path) -> None:
    values = np.empty(2, dtype=object)
    values[0] = np.ones((2, 4, 3), dtype=np.float16)
    values[1] = np.ones((2, 4, 3), dtype=np.float16)
    with pytest.raises(ValueError, match="Expected hidden-state representations"):
        canonicalize_layer_time_input(
            {
                "layer_time_input_path": np.asarray(str(tmp_path / "source.npz"), dtype=object),
                "layer_time_cache_dir": np.asarray(str(tmp_path), dtype=object),
                "layer_time_cache_id": np.asarray("projected"),
                "sv_vec_mean": values,
                "layers_used": np.arange(4),
                "problem_ids": np.asarray([1, 2]),
                "reasoning_subspace_used": np.asarray(True),
            }
        )
    assert not list(tmp_path.glob(".*.ltg-mean-states.*.npy"))


def test_post_conversion_validation_failure_releases_temporary_memmap(tmp_path) -> None:
    values = np.empty(2, dtype=object)
    values[0] = np.ones((2, 4, 3), dtype=np.float16)
    values[1] = np.ones((2, 4, 3), dtype=np.float16)
    with pytest.raises(ValueError, match="contiguous depths"):
        append_layer_time_geometry(
            {
                "layer_time_input_path": np.asarray(str(tmp_path / "source.npz"), dtype=object),
                "layer_time_cache_dir": np.asarray(str(tmp_path), dtype=object),
                "layer_time_cache_id": np.asarray("sparse"),
                "sv_vec_mean": values,
                "layers_used": np.asarray([1, 3, 5, 7]),
                "problem_ids": np.asarray([1, 2]),
                "reasoning_subspace_used": np.asarray(False),
            }
        )
    assert not list(tmp_path.glob(".*.ltg-mean-states.*.npy"))


def test_schema_preflight_reports_layer_time_mainline_readiness(tmp_path) -> None:
    path = tmp_path / "manifest.npz"
    np.savez(
        path,
        step_layer_state_memmap_path=np.asarray("states.npy", dtype=object),
        step_layer_state_vector_layers=np.asarray([1, 2, 3, 4]),
        state_pooling_kind=np.asarray("arithmetic_mean_over_step_tokens", dtype=object),
        state_representation_kind=np.asarray("hidden_state", dtype=object),
    )
    status = inspect_npz_schema(path)
    assert status["has_layer_state_memmap"]
    assert status["layer_time_mainline_ready"]
