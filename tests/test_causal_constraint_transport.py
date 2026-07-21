from __future__ import annotations

import numpy as np
import pytest

from hypergraph.attention.cct import (
    AssemblyInputs,
    CausalHypergraphBuilder,
    CausalTraceAssembler,
    ConstraintBundleAnalyzer,
    FirstErrorLabels,
    FirstErrorSurvival,
    InterventionEffect,
    OutputEffectiveTransport,
    TransportInputs,
    TraceIdentity,
    TraceRepository,
)
from hypergraph.attention.evaluation import (
    BinaryReport,
    GroupedBootstrapReport,
    LocalizationReport,
    PredictionRow,
)
from hypergraph.attention.splitting import (
    FixedHoldoutConfig,
    FixedHoldoutSplitter,
    TraceMeta,
)
from hypergraph.attention.cct.processbench import (
    PlainReasoningRenderer,
    ProcessBenchRecord,
    ShardSpec,
    TokenizerAligner,
)


def _transport_inputs() -> TransportInputs:
    attention = np.asarray(
        [
            [
                [0.6, 0.4, 0.0, 0.0],
                [0.2, 0.3, 0.5, 0.0],
            ],
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.1, 0.2, 0.7, 0.0],
            ],
        ],
        dtype=np.float64,
    )
    ov_writes = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    output_directions = np.asarray([[1.0, 0.0], [1.0, 1.0]])
    source_writes = np.einsum("hqs,hsd->qsd", attention, ov_writes)
    return TransportInputs(
        attention=attention,
        content_effect=np.einsum("hsd,qd->hqs", ov_writes, output_directions),
        source_writes=source_writes,
        output_directions=output_directions,
        residual_updates=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        prompt_end=2,
        receiver_positions=np.asarray([2, 3]),
    )


def test_output_effective_transport_separates_routing_content_and_output_direction():
    inputs = _transport_inputs()

    result = OutputEffectiveTransport().compute(inputs)

    expected_per_head = inputs.attention * inputs.content_effect
    np.testing.assert_allclose(result.per_head, expected_per_head)
    np.testing.assert_allclose(result.signed, expected_per_head.sum(axis=0))
    assert result.prompt_mask.tolist() == [True, True, False, False]
    assert np.all((result.prompt_fraction >= 0.0) & (result.prompt_fraction <= 1.0))


def test_transport_inputs_reject_future_attention_mass():
    attention = _transport_inputs().attention.copy()
    attention[:, 0, 3] = 0.2

    with pytest.raises(ValueError, match="future source"):
        TransportInputs(
            attention=attention,
            content_effect=_transport_inputs().content_effect,
            source_writes=_transport_inputs().source_writes,
            output_directions=_transport_inputs().output_directions,
            residual_updates=_transport_inputs().residual_updates,
            prompt_end=2,
            receiver_positions=np.asarray([2, 3]),
        )


def test_constraint_bundle_escape_is_zero_for_bundle_parallel_update():
    inputs = _transport_inputs()
    result = OutputEffectiveTransport().compute(inputs)

    geometry = ConstraintBundleAnalyzer(energy=0.99).analyze(inputs, result)

    assert geometry.tangent_rank.tolist() == [1, 2]
    assert geometry.transverse_escape[0] == pytest.approx(0.0, abs=1e-8)
    assert np.all(
        (geometry.transverse_escape >= 0.0) & (geometry.transverse_escape <= 1.0)
    )
    assert np.isfinite(geometry.as_features()).all()
    assert geometry.feature_names == (
        "prompt_support",
        "response_support",
        "effective_update",
        "transverse_fraction",
        "transverse_escape",
        "tangent_rank",
    )


def test_directional_escape_is_bounded_when_parallel_and_transverse_effects_cancel():
    inputs = TransportInputs(
        attention=np.asarray([[[1.0, 0.0]]]),
        content_effect=np.asarray([[[-1.0, 0.0]]]),
        source_writes=np.asarray([[[1.0, 0.0], [0.0, 0.0]]]),
        output_directions=np.asarray([[-1.0, 1.0]]),
        residual_updates=np.asarray([[1.0, 1.0]]),
        prompt_end=1,
        receiver_positions=np.asarray([1]),
    )
    contribution = OutputEffectiveTransport().compute(inputs)

    geometry = ConstraintBundleAnalyzer().analyze(inputs, contribution)

    assert geometry.effective_update[0] == pytest.approx(0.0)
    assert geometry.transverse_escape[0] == pytest.approx(0.5)


def test_hyperedge_requires_non_additive_intervention_effect():
    inputs = _transport_inputs()
    contribution = OutputEffectiveTransport().compute(inputs)
    node_features = np.arange(12, dtype=np.float64).reshape(4, 3)
    effects = (
        InterventionEffect(
            query_index=0,
            sources=(0, 1),
            singleton_effects=np.asarray([0.2, 0.1]),
            joint_effect=0.8,
        ),
        InterventionEffect(
            query_index=1,
            sources=(1, 2),
            singleton_effects=np.asarray([0.25, 0.25]),
            joint_effect=0.5,
        ),
    )

    graph = CausalHypergraphBuilder(min_effect=0.05, min_synergy=0.1).build(
        node_features=node_features,
        contribution=contribution,
        interventions=effects,
        response_nodes=np.asarray([2, 3]),
    )

    assert graph.num_edges == 3
    assert graph.edge_kind.tolist() == ["hyper", "pair", "pair"]
    assert graph.receivers.tolist() == [2, 3, 3]
    assert graph.edge_feature_names == (
        "signed_effect",
        "absolute_effect",
        "synergy",
        "prompt_fraction",
    )
    assert graph.edge_features[0, 2] == pytest.approx(0.5)


def test_first_error_survival_ignores_post_error_steps():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([-2.0, 1.5, 9.0], requires_grad=True)
    labels = FirstErrorLabels(num_steps=3, first_error=1)

    loss = FirstErrorSurvival.loss(logits, labels)
    loss.backward()

    assert float(loss) > 0.0
    assert logits.grad is not None
    assert logits.grad[0] > 0.0
    assert logits.grad[1] < 0.0
    assert logits.grad[2] == pytest.approx(0.0)


def test_response_error_probability_uses_noisy_or_not_mean_pooling():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([-4.0, 2.0, -4.0])

    probability = FirstErrorSurvival.response_error_probability(logits)
    expected = 1.0 - torch.sigmoid(-logits).prod()

    assert probability == pytest.approx(float(expected))
    assert probability > float(torch.sigmoid(logits.mean()))


def test_binary_report_exposes_failure_of_all_positive_threshold_predictions():
    report = BinaryReport.from_scores(
        labels=np.asarray([0, 0, 1, 1, 1]),
        scores=np.asarray([0.51, 0.52, 0.60, 0.70, 0.80]),
    )

    assert report.accuracy == pytest.approx(3 / 5)
    assert report.balanced_accuracy == pytest.approx(0.5)
    assert report.specificity == pytest.approx(0.0)
    assert report.mcc == pytest.approx(0.0)
    assert 0.0 <= report.ece <= 1.0


def test_localization_report_uses_tie_aware_full_step_ranks():
    report = LocalizationReport.from_traces(
        first_errors=[1, -1, 0],
        step_scores=[
            np.asarray([0.1, 0.8, 0.8]),
            np.asarray([0.9, 0.1]),
            np.asarray([0.7, 0.2]),
        ],
    )

    assert report.error_traces == 2
    assert report.top1 == pytest.approx(0.5)
    assert report.mean_rank == pytest.approx(1.25)
    assert report.mean_reciprocal_rank == pytest.approx((2 / 3 + 1) / 2)


def test_processbench_record_and_token_alignment_preserve_step_identity():
    record = ProcessBenchRecord.from_mapping(
        {
            "id": "sample",
            "problem_id": "problem",
            "problem": "Compute 1 + 1.",
            "steps": ["We add.", "The answer is 2."],
            "label": -1,
            "generator": "Llama",
        },
        index=0,
    )
    rendered = PlainReasoningRenderer().render(record)
    offsets = np.asarray(
        [
            [0, 0],
            [0, rendered.prompt_end_char - 2],
            [rendered.step_char_spans[0, 0], rendered.step_char_spans[0, 1]],
            [rendered.step_char_spans[1, 0], rendered.step_char_spans[1, 1]],
        ]
    )

    ranges = TokenizerAligner.align_offsets(offsets, rendered.step_char_spans)

    assert record.labels.first_error == -1
    assert ranges.tolist() == [[2, 3], [3, 4]]


def test_fixed_holdout_is_problem_disjoint_and_deterministic():
    traces = [
        TraceMeta(
            trace_id=f"trace-{index}",
            group_id=f"problem-{index // 2}",
            group_is_fallback=False,
            split=None,
            response_label=(index // 2) % 2,
            gold_step=-1 if (index // 2) % 2 == 0 else index % 3,
            num_steps=3,
            num_response_tokens=10 + index % 5,
            generator_model="model",
        )
        for index in range(60)
    ]
    splitter = FixedHoldoutSplitter(
        FixedHoldoutConfig(seed=19, validation_ratio=0.1, test_ratio=0.2)
    )

    first = splitter.split(traces)
    second = splitter.split(traces)

    assert first == second
    groups = [
        set(partition.group_ids)
        for partition in (first.train, first.validation, first.test)
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert sorted(
        first.train.indices + first.validation.indices + first.test.indices
    ) == list(range(len(traces)))


def test_extraction_shards_are_disjoint_and_cover_the_global_order():
    shards = [ShardSpec(num_shards=2, shard_index=index) for index in range(2)]
    assignments = [
        [position for position in range(9) if shard.includes(position)]
        for shard in shards
    ]

    assert assignments == [[0, 2, 4, 6, 8], [1, 3, 5, 7]]
    assert not set(assignments[0]) & set(assignments[1])
    assert sorted(assignments[0] + assignments[1]) == list(range(9))


def _assembled_trace():
    inputs = _transport_inputs()
    return CausalTraceAssembler(
        bundle_energy=0.99, min_effect=0.05, min_synergy=0.1
    ).assemble(
        AssemblyInputs(
            identity=TraceIdentity(
                trace_id="trace-1",
                problem_id="problem-1",
                generator_model="generator",
                observer_model="observer",
                layer_id=14,
                response_tokens=8,
            ),
            node_features=np.arange(12, dtype=np.float64).reshape(4, 3),
            transport=inputs,
            interventions=(
                InterventionEffect(
                    query_index=0,
                    sources=(0, 1),
                    singleton_effects=np.asarray([0.2, 0.1]),
                    joint_effect=0.8,
                ),
                InterventionEffect(
                    query_index=1,
                    sources=(1, 2),
                    singleton_effects=np.asarray([0.25, 0.25]),
                    joint_effect=0.5,
                ),
            ),
            labels=FirstErrorLabels(num_steps=2, first_error=1),
        )
    )


def test_trace_assembler_places_geometry_only_on_step_receivers():
    trace = _assembled_trace()

    assert trace.graph.node_features.shape == (4, 10)
    assert np.all(trace.graph.node_features[[0, 1], 3:] == 0.0)
    assert trace.graph.node_features[2, -1] == pytest.approx(1.0)
    assert trace.graph.node_features[3, -1] == pytest.approx(1.0)
    assert trace.response_label == 1


def test_trace_repository_round_trip_has_no_pickle_or_free_metadata(tmp_path):
    repository = TraceRepository(tmp_path)
    original = _assembled_trace()

    path = repository.save(original)
    restored = repository.load(path)

    assert restored.trace_id == original.trace_id
    assert restored.problem_id == original.problem_id
    assert restored.labels == original.labels
    np.testing.assert_allclose(
        restored.graph.node_features, original.graph.node_features
    )
    assert not hasattr(restored, "metadata")


def test_graph_controls_isolate_topology_and_geometry():
    from hypergraph.attention.cct.controls import (
        CausalCardinalityRewire,
        HiddenOnlyControl,
        NoEdgeControl,
        NoGeometryControl,
        PairwiseControl,
    )

    trace = _assembled_trace()
    no_edge = NoEdgeControl().apply(trace)
    pairwise = PairwiseControl().apply(trace)
    rewired = CausalCardinalityRewire(seed=3).apply(trace)
    no_geometry = NoGeometryControl().apply(trace)
    hidden_only = HiddenOnlyControl().apply(trace)

    assert no_edge.graph.num_edges == 0
    assert pairwise.graph.num_edges == 4
    assert pairwise.graph.edge_kind.tolist() == ["pair"] * 4
    assert rewired.graph.num_edges == trace.graph.num_edges
    for edge, receiver in enumerate(rewired.graph.receivers):
        members = rewired.graph.incidence[0, rewired.graph.incidence[1] == edge]
        assert np.all(members[members != receiver] < receiver)
    assert np.all(no_geometry.graph.node_features[:, -7:-1] == 0.0)
    np.testing.assert_allclose(
        no_geometry.graph.node_features[:, -1], trace.graph.node_features[:, -1]
    )
    assert hidden_only.graph.num_edges == 0
    assert np.all(hidden_only.graph.node_features[:, -7:-1] == 0.0)
    np.testing.assert_allclose(
        hidden_only.graph.node_features[:, -1], trace.graph.node_features[:, -1]
    )
    np.testing.assert_allclose(
        no_geometry.graph.node_features[:, :-7], trace.graph.node_features[:, :-7]
    )


def test_feature_normalizer_preserves_structural_zero_geometry():
    from hypergraph.attention.cct.training import FeatureNormalizer

    trace = _assembled_trace()
    normalizer = FeatureNormalizer.fit((trace,))
    normalized = normalizer.transform(trace)

    assert np.all(normalized.graph.node_features[[0, 1], -7:-1] == 0.0)
    assert normalized.graph.node_features[:, -1].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert np.isfinite(normalized.graph.node_features).all()


def test_disjoint_trace_batch_shifts_nodes_edges_and_step_ranges():
    from hypergraph.attention.cct.training import TraceBatch

    trace = _assembled_trace()
    batch = TraceBatch.from_traces((trace, trace))

    assert batch.graph.num_nodes == 2 * trace.graph.num_nodes
    assert batch.graph.num_edges == 2 * trace.graph.num_edges
    assert batch.step_ranges == ((0, 2), (2, 4))
    second_memberships = batch.graph.incidence[
        :, batch.graph.incidence[1] >= trace.graph.num_edges
    ]
    assert np.all(second_memberships[0] >= trace.graph.num_nodes)
    assert batch.graph.response_nodes.tolist() == [2, 3, 6, 7]


def test_directed_layer_updates_receivers_only():
    torch = pytest.importorskip("torch")
    from hypergraph.attention.cct.model import DirectedCausalLayer, TensorHypergraph

    inputs = _transport_inputs()
    contribution = OutputEffectiveTransport().compute(inputs)
    graph = CausalHypergraphBuilder(min_effect=0.05, min_synergy=0.1).build(
        node_features=np.arange(12, dtype=np.float32).reshape(4, 3),
        contribution=contribution,
        interventions=(
            InterventionEffect(
                query_index=0,
                sources=(0, 1),
                singleton_effects=np.asarray([0.2, 0.1]),
                joint_effect=0.8,
            ),
        ),
        response_nodes=np.asarray([2, 3]),
    )
    tensors = TensorHypergraph.from_graph(graph)
    states = torch.randn(4, 5)
    layer = DirectedCausalLayer(hidden_dim=5, edge_dim=4)

    updated = layer(states, tensors)

    assert updated.shape == states.shape
    torch.testing.assert_close(updated[[0, 1, 3]], states[[0, 1, 3]])
    assert not torch.equal(updated[2], states[2])


def test_constraint_transport_detector_emits_one_hazard_per_response_node():
    torch = pytest.importorskip("torch")
    from hypergraph.attention.cct.model import ConstraintTransportDetector

    inputs = _transport_inputs()
    contribution = OutputEffectiveTransport().compute(inputs)
    graph = CausalHypergraphBuilder(min_effect=0.05, min_synergy=0.1).build(
        node_features=np.arange(12, dtype=np.float32).reshape(4, 3),
        contribution=contribution,
        interventions=(
            InterventionEffect(
                query_index=0,
                sources=(0, 1),
                singleton_effects=np.asarray([0.2, 0.1]),
                joint_effect=0.8,
            ),
        ),
        response_nodes=np.asarray([2, 3]),
    )
    detector = ConstraintTransportDetector(
        node_dim=3, edge_dim=4, hidden_dim=8, num_layers=2
    )

    hazards = detector(graph)

    assert hazards.shape == (2,)
    assert torch.isfinite(hazards).all()


def test_selected_attention_reconstruction_is_causal_and_normalized():
    torch = pytest.importorskip("torch")
    from hypergraph.attention.cct.hf_backend import HuggingFaceTransportBackend

    queries = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])

    attention = HuggingFaceTransportBackend._causal_attention_from_qk(
        queries, keys, [0, 1]
    )

    torch.testing.assert_close(attention.sum(-1), torch.ones((1, 2)))
    assert attention[0, 0, 1:].tolist() == [0.0, 0.0]
    assert attention[0, 1, 2].item() == 0.0


def test_prediction_row_requires_aligned_step_decision():
    row = PredictionRow(
        trace_id="trace-1",
        problem_id="problem-1",
        label=1,
        probability=0.8,
        first_error=1,
        predicted_step=1,
        step_probabilities=(0.2, 0.9),
    )

    assert row.predicted_step == 1
    with pytest.raises(ValueError, match="predicted_step"):
        PredictionRow(
            trace_id="trace-2",
            problem_id="problem-2",
            label=0,
            probability=0.1,
            first_error=-1,
            predicted_step=2,
            step_probabilities=(0.1, 0.2),
        )


def test_grouped_bootstrap_resamples_problems_not_individual_traces():
    rows = tuple(
        PredictionRow(
            trace_id=f"trace-{group}-{label}",
            problem_id=f"problem-{group}",
            label=label,
            probability=0.9 if label else 0.1,
            first_error=0 if label else -1,
            predicted_step=0,
            step_probabilities=(0.9 if label else 0.1,),
        )
        for group in range(4)
        for label in (0, 1)
    )

    report = GroupedBootstrapReport.from_predictions(
        rows, replicates=50, confidence=0.95, seed=7
    )

    assert report.groups == 4
    assert report.auroc.defined_replicates == 50
    assert report.auroc.lower == pytest.approx(1.0)
    assert report.auroc.upper == pytest.approx(1.0)
    assert report.aupr.lower == pytest.approx(1.0)


def test_held_out_prediction_table_is_explicit_and_auditable(tmp_path):
    from hypergraph.attention.cct.cli import _write_predictions

    path = tmp_path / "predictions_test.csv"
    row = PredictionRow(
        trace_id="trace-1",
        problem_id="problem-1",
        label=1,
        probability=0.75,
        first_error=1,
        predicted_step=1,
        step_probabilities=(0.1, 0.8),
    )

    _write_predictions(path, (row,))

    content = path.read_text(encoding="utf-8")
    assert "trace_id,problem_id,label,probability" in content
    assert "trace-1,problem-1,1,0.75,1,1" in content
    assert "[0.1, 0.8]" in content
