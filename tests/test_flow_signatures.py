from __future__ import annotations

import numpy as np
import pytest
import torch

from prompt_control_flow.flow_signature_audit import (
    FlowAuditConfig,
    make_order_sensitive_synthetic_dataset,
    run_flow_signature_audit,
)
from prompt_control_flow.flow_signature_data import (
    inspect_flow_source,
    load_flow_trajectory_dataset,
)
from prompt_control_flow.flow_signatures import (
    FlowSignatureConfig,
    _prefix_logsignatures,
    encode_reasoning_flows,
)


def _path(*points: tuple[float, float]) -> np.ndarray:
    return np.asarray(points, dtype=np.float32)[:, None, :]


def test_order_two_distinguishes_order_when_order_one_and_endpoint_match() -> None:
    xy = _path((0, 0), (1, 0), (1, 1))
    yx = _path((0, 0), (0, 1), (1, 1))
    encoding, shuffled = encode_reasoning_flows(
        [xy, yx],
        FlowSignatureConfig(projection_dim=2, phase_points=4, seed=3),
        include_shuffled=True,
    )
    np.testing.assert_allclose(
        encoding.order1_prefix[0, -1],
        encoding.order1_prefix[1, -1],
        atol=1e-6,
    )
    difference = encoding.order2_prefix[0, -1] - encoding.order2_prefix[1, -1]
    assert np.sqrt(np.sum(difference * difference)) > 0.1
    np.testing.assert_allclose(
        encoding.order1_prefix[:, -1],
        shuffled.order1_prefix[:, -1],
        atol=1e-6,
    )


def test_signature_is_translation_and_piecewise_refinement_invariant() -> None:
    coarse = _path((0, 0), (1, 0), (1, 1))
    translated = coarse + np.asarray([[[7.0, -4.0]]], dtype=np.float32)
    refined = _path((0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1))
    encoding, _ = encode_reasoning_flows(
        [coarse, translated, refined],
        FlowSignatureConfig(projection_dim=2, phase_points=8, seed=5),
        include_shuffled=False,
    )
    np.testing.assert_allclose(encoding.order2_prefix[0], encoding.order2_prefix[1], atol=2e-6)
    np.testing.assert_allclose(encoding.order2_prefix[0], encoding.order2_prefix[2], atol=2e-6)


def test_pairwise_signature_distance_is_orthogonally_invariant() -> None:
    increments = torch.tensor(
        [
            [[[1.0, 0.0]], [[0.0, 1.0]]],
            [[[0.0, 1.0]], [[1.0, 0.0]]],
        ]
    )
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    _, original, _ = _prefix_logsignatures(
        increments,
        torch.tensor([2, 2]),
        phase_points=4,
        progress_weight=1.0,
        eps=1e-8,
    )
    _, rotated, _ = _prefix_logsignatures(
        increments @ rotation,
        torch.tensor([2, 2]),
        phase_points=4,
        progress_weight=1.0,
        eps=1e-8,
    )
    original_distance = torch.linalg.vector_norm(original[0, -1] - original[1, -1])
    rotated_distance = torch.linalg.vector_norm(rotated[0, -1] - rotated[1, -1])
    torch.testing.assert_close(original_distance, rotated_distance)


def test_loader_supports_multisample_vectors_and_policy(tmp_path) -> None:
    vectors = np.empty(4, dtype=object)
    for i in range(4):
        vectors[i] = np.zeros((3, 2, 5), dtype=np.float32) + i
    path = tmp_path / "toy_multisample.npz"
    np.savez_compressed(
        path,
        sv_vec_step_exp=vectors,
        layers_used=np.asarray([8, 16]),
        problem_ids=np.asarray([3, 3, 4, 4]),
        sample_idx=np.asarray([0, 1, 0, 1]),
        is_correct=np.asarray([1, 0, 1, 0]),
        is_correct_strict=np.asarray([1, 0, 1, 0]),
        format_ok=np.asarray([1, 1, 1, 0]),
        responses=np.asarray(["a", "bb", "ccc", "dddd"], dtype=object),
    )
    status = inspect_flow_source(path)
    assert status["vector_key"] == "sv_vec_step_exp"
    assert status["layer_ids"] == [8, 16]
    dataset = load_flow_trajectory_dataset(path, layers="16", label_policy="answer_format_ok")
    assert dataset.n_samples == 3
    assert dataset.layer_ids.tolist() == [16]
    assert dataset.y_error.tolist() == [0, 1, 0]
    assert dataset.skipped["label_policy"] == 1


def test_loader_refuses_mismatched_layer_metadata(tmp_path) -> None:
    vectors = np.empty(2, dtype=object)
    vectors[:] = [np.zeros((3, 2, 4), dtype=np.float32)] * 2
    path = tmp_path / "bad_layers.npz"
    np.savez_compressed(
        path,
        stepvec=vectors,
        sv_layers=np.asarray([8, 12, 16]),
        problem_ids=np.asarray([0, 1]),
        gold_error_step=np.asarray([-1, 1]),
    )
    with pytest.raises(ValueError, match="refusing to guess"):
        load_flow_trajectory_dataset(path, label_policy="processbench")


def test_batch_partition_does_not_change_encoding() -> None:
    paths = [
        _path((0, 0), (1, 0), (1, 1)),
        _path((2, -1), (2, 0), (3, 0)),
        _path((0, 0), (0.5, 0), (1, 0), (1, 1)),
    ]
    cfg = FlowSignatureConfig(projection_dim=2, phase_points=6, seed=17)
    one, _ = encode_reasoning_flows(paths, cfg, batch_size=1, include_shuffled=False)
    all_at_once, _ = encode_reasoning_flows(paths, cfg, batch_size=8, include_shuffled=False)
    np.testing.assert_allclose(one.order2_prefix, all_at_once.order2_prefix, atol=2e-6)


def test_end_to_end_order_control_recovers_only_second_order_signal() -> None:
    dataset = make_order_sensitive_synthetic_dataset(n_problems=16, seed=11)
    encoding, shuffled = encode_reasoning_flows(
        dataset.trajectories,
        FlowSignatureConfig(projection_dim=4, phase_points=8, seed=11),
        batch_size=32,
        include_shuffled=True,
    )
    report, packed = run_flow_signature_audit(
        dataset,
        encoding,
        shuffled,
        FlowAuditConfig(folds=4, bootstrap=50, permutations=20, seed=11),
    )
    assert report["scores"]["support_o2_endpoint"]["same_problem_auroc"] > 0.9
    assert report["headline"]["order2_minus_order1"]["point"] > 0.25
    assert report["headline"]["chronological_minus_shuffled"]["point"] > 0.15
    assert packed["profiles"].shape[:2] == (dataset.n_samples, 8)
