from __future__ import annotations

import numpy as np
import pytest

from crwh.ghost import (
    GhostMahalanobisEnsemble,
    GhostTraceEmbedding,
    LowRankShrunkMahalanobis,
    pool_response_hidden,
    select_mid_depths,
)


MID_DEPTHS = (8, 10, 11, 13, 14, 16)
ALL_DEPTHS = (*MID_DEPTHS, 32)


def _trace(
    *,
    trace_id: str,
    label: int,
    location: float,
) -> GhostTraceEmbedding:
    layer_offsets = np.arange(len(ALL_DEPTHS), dtype=np.float64)[:, None]
    features = np.asarray([0.0, 0.1, -0.1, 0.2], dtype=np.float64)[None, :]
    return GhostTraceEmbedding(
        trace_id=trace_id,
        problem_id=f"problem-{trace_id}",
        response_label=label,
        first_error=-1 if label == 0 else 0,
        generator_model="fixture",
        layer_depths=ALL_DEPTHS,
        response_mean=location + 0.01 * layer_offsets + features,
        response_last=location + 0.01 * layer_offsets + features,
        token_count=12,
        response_token_count=4,
        step_count=2,
    )


def test_select_mid_depths_maps_quarter_to_half_depth_without_off_by_one() -> None:
    assert select_mid_depths(num_hidden_layers=32) == MID_DEPTHS


def test_response_pooling_uses_only_tokens_owned_by_reasoning_steps() -> None:
    hidden = np.arange(14, dtype=np.float64).reshape(7, 2)
    step_ranges = np.asarray([[2, 4], [5, 7]], dtype=np.int64)

    mean, last, positions = pool_response_hidden(hidden, step_ranges)

    assert positions.tolist() == [2, 3, 5, 6]
    assert np.allclose(mean, hidden[[2, 3, 5, 6]].mean(axis=0))
    assert np.array_equal(last, hidden[6])


def test_low_rank_shrunk_mahalanobis_matches_direct_covariance_inverse() -> None:
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(9, 5))
    queries = rng.normal(size=(3, 5))
    shrinkage = 0.2
    regularization = 1e-6

    detector = LowRankShrunkMahalanobis(
        shrinkage=shrinkage,
        regularization=regularization,
    ).fit(reference)
    actual = detector.score_samples(queries)

    mean = reference.mean(axis=0)
    centered = reference - mean
    empirical = centered.T @ centered / (len(reference) - 1)
    isotropic = max(float(np.trace(empirical) / reference.shape[1]), regularization)
    covariance = (
        (1.0 - shrinkage) * empirical
        + (shrinkage * isotropic + regularization) * np.eye(reference.shape[1])
    )
    differences = queries - mean
    expected = np.einsum(
        "ni,ij,nj->n",
        differences,
        np.linalg.inv(covariance),
        differences,
    )

    assert np.allclose(actual, expected, rtol=1e-7, atol=1e-7)


def test_ensemble_fit_is_fail_closed_on_non_normal_reference() -> None:
    references = [
        _trace(trace_id=f"normal-{index}", label=0, location=0.01 * index)
        for index in range(5)
    ]
    references.append(_trace(trace_id="error", label=1, location=5.0))

    with pytest.raises(ValueError, match="normal"):
        GhostMahalanobisEnsemble(
            mid_depths=MID_DEPTHS,
            final_depth=32,
        ).fit(references, representation="response_mean")


def test_mid_layer_ensemble_scores_far_held_out_trace_as_more_anomalous() -> None:
    references = [
        _trace(trace_id=f"normal-{index}", label=0, location=0.01 * index)
        for index in range(8)
    ]
    calibration = [
        _trace(trace_id=f"calibration-{index}", label=0, location=0.005 + 0.01 * index)
        for index in range(4)
    ]
    ensemble = GhostMahalanobisEnsemble(
        mid_depths=MID_DEPTHS,
        final_depth=32,
        shrinkage=0.2,
    ).fit(references, representation="response_mean")
    ensemble.calibrate(calibration)

    scored = ensemble.score(
        (
            _trace(trace_id="near", label=0, location=0.03),
            _trace(trace_id="far", label=1, location=4.0),
        )
    )

    assert scored.layer_percentiles.shape == (2, len(ALL_DEPTHS))
    assert np.isfinite(scored.layer_distances).all()
    assert np.all((0.0 <= scored.mid_fused_percentile) & (scored.mid_fused_percentile <= 1.0))
    assert scored.mid_fused_percentile[1] > scored.mid_fused_percentile[0]
    assert scored.final_percentile[1] > scored.final_percentile[0]


def test_calibration_is_normal_only_and_disjoint_from_fit_references() -> None:
    references = [
        _trace(trace_id=f"normal-{index}", label=0, location=0.01 * index)
        for index in range(5)
    ]
    ensemble = GhostMahalanobisEnsemble(
        mid_depths=MID_DEPTHS,
        final_depth=32,
    ).fit(references, representation="response_last")

    with pytest.raises(RuntimeError, match="calibrate"):
        ensemble.score((_trace(trace_id="held", label=0, location=0.02),))
    with pytest.raises(ValueError, match="normal"):
        ensemble.calibrate(
            (_trace(trace_id="calibration-error", label=1, location=2.0),)
        )
    with pytest.raises(ValueError, match="overlap"):
        ensemble.calibrate((references[0], references[1]))


def test_saved_detector_reproduces_held_out_scores(tmp_path) -> None:
    references = [
        _trace(trace_id=f"fit-{index}", label=0, location=0.01 * index)
        for index in range(6)
    ]
    calibration = [
        _trace(trace_id=f"cal-{index}", label=0, location=0.02 * index)
        for index in range(3)
    ]
    held_out = (
        _trace(trace_id="held-normal", label=0, location=0.02),
        _trace(trace_id="held-error", label=1, location=2.0),
    )
    detector = GhostMahalanobisEnsemble(
        mid_depths=MID_DEPTHS,
        final_depth=32,
    ).fit(references, representation="response_mean")
    detector.calibrate(calibration)
    expected = detector.score(held_out)
    artifact = tmp_path / "model.npz"

    detector.save(artifact)
    actual = GhostMahalanobisEnsemble.load(artifact).score(held_out)

    assert np.array_equal(actual.layer_distances, expected.layer_distances)
    assert np.array_equal(
        actual.mid_fused_percentile,
        expected.mid_fused_percentile,
    )
