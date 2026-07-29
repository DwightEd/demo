from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from crwh.builder import ResidualWriteHypergraphBuilder
from crwh.contracts import CounterfactualExample
from crwh.model import (
    DirectedResidualHypergraphLayer,
    ModelOutputs,
    MultiViewHypergraphDetector,
)
from crwh.objectives import SemiSupervisedObjective

from test_contracts_and_builder import make_trace


def make_example(*, include_paraphrase: bool = True) -> CounterfactualExample:
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
    paraphrase = None
    if include_paraphrase:
        paraphrase_trace = make_trace(view_name="paraphrase")
        perturbed_nodes = paraphrase_trace.node_features.copy()
        perturbed_nodes[0, 0] += 0.25
        paraphrase = builder.build(
            replace(paraphrase_trace, node_features=perturbed_nodes)
        )
    return CounterfactualExample(
        sample_id="sample-1",
        factual=builder.build(factual_trace),
        counterfactual=builder.build(counterfactual_trace),
        paraphrase=paraphrase,
    )


def test_multiview_detector_returns_one_logit_per_aligned_response_token() -> None:
    example = make_example()
    model = MultiViewHypergraphDetector(
        node_dim=3,
        edge_dim=len(example.factual.edge_feature_names),
        hidden_dim=8,
        num_layers=1,
    )

    outputs = model(example)

    assert outputs.logits.shape == (2,)
    assert outputs.factual_embedding.shape == (2, 8)
    assert outputs.counterfactual_embedding.shape == (2, 8)
    assert torch.isfinite(outputs.logits).all()


def test_detector_handles_graphs_with_no_retained_hyperedges() -> None:
    builder = ResidualWriteHypergraphBuilder(
        attention_threshold=1.1,
        min_sources=2,
    )
    factual = builder.build(make_trace(view_name="factual"))
    counterfactual = builder.build(make_trace(view_name="counterfactual"))
    example = CounterfactualExample(
        sample_id="sample-1",
        factual=factual,
        counterfactual=counterfactual,
    )
    model = MultiViewHypergraphDetector(
        node_dim=3,
        edge_dim=len(factual.edge_feature_names),
        hidden_dim=8,
        num_layers=1,
    )

    outputs = model(example)
    outputs.logits.sum().backward()

    assert outputs.logits.shape == (2,)
    assert torch.isfinite(outputs.logits).all()


def test_directed_layer_updates_only_receivers_and_is_membership_order_invariant() -> None:
    torch.manual_seed(17)
    layer = DirectedResidualHypergraphLayer(hidden_dim=3, edge_dim=2)
    layer.eval()
    states = torch.as_tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ]
    )
    incidence = torch.as_tensor(
        [[0, 1, 2, 1, 3], [0, 0, 0, 1, 1]],
        dtype=torch.long,
    )
    receivers = torch.as_tensor([2, 3], dtype=torch.long)
    edge_features = torch.as_tensor([[0.2, 0.4], [0.6, 0.8]])

    baseline = layer(states, incidence, receivers, edge_features)
    reordered = layer(
        states,
        incidence[:, torch.as_tensor([4, 2, 0, 3, 1])],
        receivers,
        edge_features,
    )
    changed_states = states.clone()
    changed_states[0, 0] += 2.0
    source_changed = layer(
        changed_states,
        incidence,
        receivers,
        edge_features,
    )
    changed_edges = edge_features.clone()
    changed_edges[0, 0] += 2.0
    edge_changed = layer(
        states,
        incidence,
        receivers,
        changed_edges,
    )

    assert torch.equal(baseline[:2], states[:2])
    assert torch.allclose(baseline, reordered)
    assert not torch.allclose(baseline[2], source_changed[2])
    assert not torch.allclose(baseline[2], edge_changed[2])


def test_unlabeled_tokens_skip_bce_but_keep_unsupervised_losses() -> None:
    example = make_example()
    model = MultiViewHypergraphDetector(
        node_dim=3,
        edge_dim=len(example.factual.edge_feature_names),
        hidden_dim=8,
        num_layers=1,
    )
    outputs = model(example)
    losses = SemiSupervisedObjective(
        consistency_weight=0.5,
        contrastive_weight=0.5,
        contrastive_margin=0.2,
    )(outputs, labels=torch.full((2,), -1))

    losses.total.backward()

    assert losses.supervised.item() == 0.0
    assert losses.consistency.item() > 0.0
    assert losses.contrastive.item() >= 0.0
    assert torch.isfinite(losses.total)
    gradient_mass = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert gradient_mass > 0.0


def test_labeled_tokens_contribute_supervised_bce() -> None:
    example = make_example()
    model = MultiViewHypergraphDetector(
        node_dim=3,
        edge_dim=len(example.factual.edge_feature_names),
        hidden_dim=8,
        num_layers=1,
    )
    outputs = model(example)
    labels = torch.as_tensor(np.asarray([0, 1]), dtype=torch.long)

    losses = SemiSupervisedObjective()(outputs, labels=labels)

    assert losses.supervised.item() > 0.0
    assert torch.isfinite(losses.total)


def test_counterfactual_contrast_is_not_forced_on_hallucinated_tokens() -> None:
    zeros = torch.zeros((2, 3), requires_grad=True)
    outputs = ModelOutputs(
        logits=torch.zeros(2, requires_grad=True),
        factual_embedding=zeros,
        counterfactual_embedding=zeros,
        paraphrase_embedding=zeros,
    )
    objective = SemiSupervisedObjective(contrastive_margin=0.5)

    normal = objective(
        outputs,
        labels=torch.zeros(2, dtype=torch.long),
        context_reference_mask=torch.ones(2, dtype=torch.bool),
    )
    hallucinated = objective(outputs, labels=torch.ones(2, dtype=torch.long))

    assert normal.contrastive.item() == 0.5
    assert hallucinated.contrastive.item() == 0.0


def test_withheld_correct_labels_do_not_change_unsupervised_losses() -> None:
    zeros = torch.zeros((2, 3), requires_grad=True)
    outputs = ModelOutputs(
        logits=torch.zeros(2, requires_grad=True),
        factual_embedding=zeros,
        counterfactual_embedding=zeros,
        paraphrase_embedding=zeros,
    )
    objective = SemiSupervisedObjective(supervised_weight=0.0)
    ineligible = torch.zeros(2, dtype=torch.bool)

    visible = objective(
        outputs,
        labels=torch.zeros(2, dtype=torch.long),
        context_reference_mask=ineligible,
    )
    withheld = objective(
        outputs,
        labels=torch.full((2,), -1, dtype=torch.long),
        context_reference_mask=ineligible,
    )

    assert visible.consistency.item() == withheld.consistency.item()
    assert visible.contrastive.item() == withheld.contrastive.item()
    assert visible.total.item() == withheld.total.item()
