from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from crwh.builder import ResidualWriteHypergraphBuilder
from crwh.contracts import CounterfactualExample
from crwh.detector import MultiViewOneClassDetector
from crwh.model import MultiViewHypergraphDetector
from crwh.training import TrainingConfig, train_semisupervised

from test_model_and_objective import make_example
from test_contracts_and_builder import make_trace


def shifted_example(shift: float) -> CounterfactualExample:
    builder = ResidualWriteHypergraphBuilder(
        attention_threshold=0.15,
        min_sources=2,
    )
    factual_trace = make_trace(view_name="factual")
    factual_trace = replace(
        factual_trace,
        node_features=factual_trace.node_features + shift,
        token_labels=np.asarray([0, 0]),
    )
    counterfactual_trace = make_trace(view_name="counterfactual")
    counterfactual_trace = replace(
        counterfactual_trace,
        node_features=counterfactual_trace.node_features + shift,
        token_labels=np.asarray([0, 0]),
    )
    return CounterfactualExample(
        sample_id="sample-1",
        factual=builder.build(factual_trace),
        counterfactual=builder.build(counterfactual_trace),
        normal_reference=True,
    )


def test_multiview_one_class_detector_returns_component_and_fused_scores() -> None:
    reference = [shifted_example(0.0), shifted_example(0.01)]
    detector = MultiViewOneClassDetector(shrinkage=0.2).fit(reference)

    inlier = detector.score(reference[0])
    outlier = detector.score(shifted_example(20.0))

    assert inlier.fused.shape == (2,)
    assert inlier.fused_percentile.shape == (2,)
    assert inlier.context_state.shape == (2,)
    assert inlier.context_relation.shape == (2,)
    assert np.isfinite(inlier.fused).all()
    assert np.all((inlier.fused_percentile > 0.0) & (inlier.fused_percentile < 1.0))
    assert np.isfinite(outlier.fused).all()
    assert float(outlier.state.mean()) > float(inlier.state.mean())


def test_one_class_fit_rejects_examples_not_declared_as_normal_reference() -> None:
    detector = MultiViewOneClassDetector(shrinkage=0.2)

    with np.testing.assert_raises_regex(ValueError, "normal_reference"):
        detector.fit([make_example()])


def test_semisupervised_training_accepts_mixed_labeled_and_unlabeled_examples() -> None:
    labeled = make_example()
    unlabeled_factual = replace(
        labeled.factual,
        token_labels=np.asarray([-1, -1]),
    )
    unlabeled_counterfactual = replace(
        labeled.counterfactual,
        token_labels=np.asarray([-1, -1]),
    )
    unlabeled_paraphrase = replace(
        labeled.paraphrase,
        token_labels=np.asarray([-1, -1]),
    )
    unlabeled = CounterfactualExample(
        sample_id=labeled.sample_id,
        factual=unlabeled_factual,
        counterfactual=unlabeled_counterfactual,
        paraphrase=unlabeled_paraphrase,
    )
    model = MultiViewHypergraphDetector(
        node_dim=3,
        edge_dim=len(labeled.factual.edge_feature_names),
        hidden_dim=8,
        num_layers=1,
    )

    history = train_semisupervised(
        model,
        [labeled, unlabeled],
        config=TrainingConfig(epochs=2, learning_rate=1e-3, seed=3),
        device=torch.device("cpu"),
    )

    assert len(history) == 2
    assert all(np.isfinite(epoch.total_loss) for epoch in history)
    assert history[-1].labeled_tokens == 2


def test_training_config_rejects_nonfinite_optimizer_parameters() -> None:
    with pytest.raises(ValueError, match="finite"):
        TrainingConfig(learning_rate=float("nan"))


def test_training_fails_closed_when_no_example_has_an_active_objective() -> None:
    example = make_example(include_paraphrase=False)
    no_labels = replace(
        example.factual,
        token_labels=np.asarray([-1, -1]),
    )
    no_counterfactual_labels = replace(
        example.counterfactual,
        token_labels=np.asarray([-1, -1]),
    )
    example = CounterfactualExample(
        sample_id=example.sample_id,
        factual=no_labels,
        counterfactual=no_counterfactual_labels,
    )
    model = MultiViewHypergraphDetector(
        node_dim=3,
        edge_dim=len(example.factual.edge_feature_names),
        hidden_dim=8,
        num_layers=1,
    )

    with pytest.raises(ValueError, match="active training objective"):
        train_semisupervised(
            model,
            [example],
            config=TrainingConfig(epochs=1),
            device=torch.device("cpu"),
        )


def test_training_reports_skipped_no_signal_examples_in_a_mixed_set() -> None:
    active = make_example(include_paraphrase=False)
    inactive = CounterfactualExample(
        sample_id=active.sample_id,
        factual=replace(
            active.factual,
            token_labels=np.asarray([-1, -1]),
        ),
        counterfactual=replace(
            active.counterfactual,
            token_labels=np.asarray([-1, -1]),
        ),
    )
    model = MultiViewHypergraphDetector(
        node_dim=3,
        edge_dim=len(active.factual.edge_feature_names),
        hidden_dim=8,
        num_layers=1,
    )

    history = train_semisupervised(
        model,
        [active, inactive],
        config=TrainingConfig(epochs=1),
        device=torch.device("cpu"),
    )

    assert history[0].active_examples == 1
    assert history[0].skipped_examples == 1
    assert history[0].labeled_tokens == 2
