from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crwh.builder import ResidualWriteHypergraphBuilder
from crwh.contracts import CounterfactualExample
from crwh.features import paired_token_signatures, relational_token_signatures
from crwh.geometry import summarize_write_group
from crwh.one_class import ShrunkMahalanobis

from test_contracts_and_builder import make_trace


def test_write_geometry_has_expected_limiting_cases() -> None:
    aligned = summarize_write_group(np.asarray([[1.0, 0.0], [1.0, 0.0]]))
    opposed = summarize_write_group(np.asarray([[1.0, 0.0], [-1.0, 0.0]]))
    zero = summarize_write_group(np.zeros((2, 2)))

    assert aligned.cancellation == 0.0
    assert opposed.cancellation == 1.0
    assert aligned.effective_sources == 2.0
    assert aligned.dominance == 0.5
    assert np.isfinite(zero.as_array()).all()


def test_relational_signatures_are_query_aligned_and_fixed_width() -> None:
    graph = ResidualWriteHypergraphBuilder(
        attention_threshold=0.15,
        min_sources=2,
    ).build(make_trace())

    signatures = relational_token_signatures(graph)

    assert signatures.shape[0] == len(graph.response_token_ids)
    assert signatures.shape[1] > graph.edge_features.shape[1]
    assert np.isfinite(signatures).all()


def test_paired_signatures_capture_context_change_not_response_misalignment() -> None:
    builder = ResidualWriteHypergraphBuilder(
        attention_threshold=0.15,
        min_sources=2,
    )
    factual_trace = make_trace(view_name="factual")
    counterfactual_trace = replace(
        make_trace(view_name="counterfactual"),
        attention=0.5 * factual_trace.attention,
        source_writes=0.5 * factual_trace.source_writes,
        expected_updates=0.5 * factual_trace.expected_updates,
    )
    example = CounterfactualExample(
        sample_id="sample-1",
        factual=builder.build(factual_trace),
        counterfactual=builder.build(counterfactual_trace),
    )

    signatures = paired_token_signatures(example)

    assert (
        signatures.context_relation_delta.shape
        == signatures.factual_relation.shape
    )
    assert np.any(signatures.context_relation_delta > 0.0)
    assert np.all(signatures.context_state_delta == 0.0)
    assert np.all(signatures.paraphrase_relation_delta == 0.0)
    assert np.all(signatures.paraphrase_state_delta == 0.0)
    assert signatures.combined().shape[0] == 2


def test_paired_signatures_capture_contextual_influence_on_response_states() -> None:
    builder = ResidualWriteHypergraphBuilder(
        attention_threshold=0.15,
        min_sources=2,
    )
    factual_trace = make_trace(view_name="factual")
    counterfactual_trace = make_trace(view_name="counterfactual")
    shifted_nodes = counterfactual_trace.node_features.copy()
    shifted_nodes[counterfactual_trace.receiver_positions] += 3.0
    example = CounterfactualExample(
        sample_id="sample-1",
        factual=builder.build(factual_trace),
        counterfactual=builder.build(
            replace(counterfactual_trace, node_features=shifted_nodes)
        ),
    )

    signatures = paired_token_signatures(example)

    assert np.all(signatures.context_state_delta > 0.0)
    assert np.all(signatures.context_relation_delta == 0.0)


def test_shrunk_mahalanobis_scores_obvious_ood_points_higher() -> None:
    rng = np.random.default_rng(7)
    reference = rng.normal(0.0, 0.1, size=(64, 4))
    detector = ShrunkMahalanobis(shrinkage=0.2).fit(reference)

    inlier_scores = detector.score_samples(reference[:8])
    outlier_scores = detector.score_samples(np.full((8, 4), 5.0))

    assert np.isfinite(inlier_scores).all()
    assert np.isfinite(outlier_scores).all()
    assert float(outlier_scores.mean()) > 20.0 * float(inlier_scores.mean())


def test_shrunk_mahalanobis_handles_singular_reference_covariance() -> None:
    reference = np.ones((8, 3), dtype=np.float64)
    detector = ShrunkMahalanobis(shrinkage=0.1).fit(reference)

    scores = detector.score_samples(np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))

    assert np.isfinite(scores).all()
    assert scores[0] == 0.0
    assert scores[1] > scores[0]


def test_shrunk_mahalanobis_projects_wide_hidden_states_deterministically() -> None:
    rng = np.random.default_rng(13)
    reference = rng.normal(size=(32, 1024))
    values = rng.normal(size=(4, 1024))

    first = ShrunkMahalanobis(
        shrinkage=0.1,
        max_features=16,
        projection_seed=9,
    ).fit(reference)
    second = ShrunkMahalanobis(
        shrinkage=0.1,
        max_features=16,
        projection_seed=9,
    ).fit(reference)

    assert first.precision_.shape == (16, 16)
    assert np.allclose(first.score_samples(values), second.score_samples(values))


def test_shrunk_mahalanobis_rejects_nonfinite_hyperparameters() -> None:
    with pytest.raises(ValueError, match="finite"):
        ShrunkMahalanobis(shrinkage=float("nan"))
