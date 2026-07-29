from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crwh.builder import ResidualWriteHypergraphBuilder
from crwh.contracts import (
    CounterfactualExample,
    ResidualWriteTrace,
    TraceProvenance,
)


PROVENANCE = TraceProvenance(
    model_id="toy-model",
    model_revision="toy-revision",
    tokenizer_id="toy-tokenizer",
    tokenizer_revision="toy-tokenizer-revision",
    prompt_format_id="toy-chat-template-v1",
    layer_id=4,
    extractor_id="attention_key_token_ov_v1",
    projection_id="identity-r2",
    output_anchor_id="fixed-token-margin",
    teacher_forcing_offset=0,
)


def make_trace(
    *,
    view_name: str = "factual",
    response_token_ids: tuple[int, ...] = (101, 102),
    corrupt_reconstruction: bool = False,
) -> ResidualWriteTrace:
    # Two heads, two response queries, four source/receiver positions, rank two.
    attention = np.asarray(
        [
            [[0.6, 0.3, 0.1, 0.0], [0.2, 0.4, 0.3, 0.1]],
            [[0.5, 0.4, 0.1, 0.0], [0.1, 0.3, 0.4, 0.2]],
        ],
        dtype=np.float64,
    )
    source_writes = np.zeros((2, 2, 4, 2), dtype=np.float64)
    source_writes[..., 0] = attention
    source_writes[..., 1] = -0.5 * attention
    expected_updates = source_writes.sum(axis=2)
    if corrupt_reconstruction:
        expected_updates = expected_updates.copy()
        expected_updates[0, 0, 0] += 1.0

    return ResidualWriteTrace(
        sample_id="sample-1",
        view_name=view_name,
        perturbation_kind={
            "factual": "factual",
            "paraphrase": "paraphrase",
            "counterfactual": "evidence_replacement",
        }[view_name],
        perturbation_id=f"toy-{view_name}",
        perturbation_seed=7,
        perturbation_validated=True,
        provenance=PROVENANCE,
        node_features=np.arange(12, dtype=np.float64).reshape(4, 3),
        attention=attention,
        source_writes=source_writes,
        expected_updates=expected_updates,
        output_directions=np.asarray([[1.0, 0.0], [1.0, 0.0]]),
        receiver_positions=np.asarray([2, 3]),
        response_token_ids=np.asarray(response_token_ids),
        source_roles=np.asarray([0, 0, 2, 2]),
        token_labels=np.asarray([0, 1]),
    )


def test_trace_rejects_future_attention() -> None:
    trace = make_trace()
    attention = trace.attention.copy()
    attention[0, 0, 3] = 0.2

    with pytest.raises(ValueError, match="future"):
        ResidualWriteTrace(
            sample_id=trace.sample_id,
            view_name=trace.view_name,
            perturbation_kind=trace.perturbation_kind,
            perturbation_id=trace.perturbation_id,
            perturbation_seed=trace.perturbation_seed,
            perturbation_validated=trace.perturbation_validated,
            provenance=trace.provenance,
            node_features=trace.node_features,
            attention=attention,
            source_writes=trace.source_writes,
            expected_updates=trace.expected_updates,
            output_directions=trace.output_directions,
            receiver_positions=trace.receiver_positions,
            response_token_ids=trace.response_token_ids,
            source_roles=trace.source_roles,
            token_labels=trace.token_labels,
        )


def test_trace_rejects_fractional_token_ids_before_integer_cast() -> None:
    trace = make_trace()

    with pytest.raises(ValueError, match="integer"):
        replace(trace, response_token_ids=np.asarray([101.5, 102.0]))


def test_trace_and_builder_reject_nonfinite_gating_parameters() -> None:
    with pytest.raises(ValueError, match="causal_tolerance"):
        replace(make_trace(), causal_tolerance=float("nan"))
    with pytest.raises(ValueError, match="attention_threshold"):
        ResidualWriteHypergraphBuilder(attention_threshold=float("nan"))
    with pytest.raises(ValueError, match="reconstruction_tolerance"):
        ResidualWriteHypergraphBuilder(
            reconstruction_tolerance=float("nan")
        )


def test_contract_rejects_string_boole_and_noninteger_provenance() -> None:
    with pytest.raises(ValueError, match="perturbation_validated"):
        replace(make_trace(), perturbation_validated="false")
    with pytest.raises(ValueError, match="teacher_forcing_offset"):
        replace(PROVENANCE, teacher_forcing_offset="0")


def test_builder_fails_closed_when_source_writes_do_not_reconstruct() -> None:
    builder = ResidualWriteHypergraphBuilder(attention_threshold=0.15)

    with pytest.raises(ValueError, match="reconstruction"):
        builder.build(make_trace(corrupt_reconstruction=True))


def test_builder_creates_directed_response_hyperedges_with_finite_features() -> None:
    graph = ResidualWriteHypergraphBuilder(
        attention_threshold=0.15,
        min_sources=2,
    ).build(make_trace())

    assert graph.num_edges == 4
    assert graph.num_heads == 2
    assert graph.edge_features.shape == (4, len(graph.edge_feature_names))
    assert np.isfinite(graph.edge_features).all()
    assert set(graph.receivers.tolist()) == {2, 3}
    assert np.array_equal(graph.response_nodes, np.asarray([2, 3]))
    assert set(graph.edge_kind.tolist()) == {"write"}

    # Every edge contains its explicitly stored receiver and at least two sources.
    for edge_id, receiver in enumerate(graph.receivers):
        members = graph.incidence[0, graph.incidence[1] == edge_id]
        assert receiver in members
        assert len(members[members != receiver]) >= 2

    self_mass = graph.edge_features[
        :, graph.edge_feature_names.index("self_write_mass")
    ]
    assert np.all(self_mass > 0.0)


def test_counterfactual_example_requires_identical_response_tokens() -> None:
    factual = ResidualWriteHypergraphBuilder(attention_threshold=0.15).build(
        make_trace(view_name="factual")
    )
    counterfactual = ResidualWriteHypergraphBuilder(
        attention_threshold=0.15
    ).build(
        make_trace(
            view_name="counterfactual",
            response_token_ids=(101, 999),
        )
    )

    with pytest.raises(ValueError, match="response token"):
        CounterfactualExample(
            sample_id="sample-1",
            factual=factual,
            counterfactual=counterfactual,
        )


def test_counterfactual_example_requires_compatible_node_feature_widths() -> None:
    builder = ResidualWriteHypergraphBuilder(attention_threshold=0.15)
    factual = builder.build(make_trace(view_name="factual"))
    counterfactual = builder.build(make_trace(view_name="counterfactual"))
    counterfactual = replace(
        counterfactual,
        node_features=np.column_stack(
            (counterfactual.node_features, np.ones(counterfactual.num_nodes))
        ),
    )

    with pytest.raises(ValueError, match="node feature width"):
        CounterfactualExample(
            sample_id="sample-1",
            factual=factual,
            counterfactual=counterfactual,
        )


def test_counterfactual_example_requires_a_fixed_output_direction_anchor() -> None:
    builder = ResidualWriteHypergraphBuilder(attention_threshold=0.15)
    factual = builder.build(make_trace(view_name="factual"))
    counterfactual = builder.build(make_trace(view_name="counterfactual"))
    counterfactual = replace(
        counterfactual,
        output_directions=-counterfactual.output_directions,
    )

    with pytest.raises(ValueError, match="fixed output directions"):
        CounterfactualExample(
            sample_id="sample-1",
            factual=factual,
            counterfactual=counterfactual,
        )


def test_counterfactual_example_rejects_conflicting_visible_labels() -> None:
    builder = ResidualWriteHypergraphBuilder(attention_threshold=0.15)
    factual = builder.build(make_trace(view_name="factual"))
    counterfactual = builder.build(make_trace(view_name="counterfactual"))
    counterfactual = replace(
        counterfactual,
        token_labels=np.asarray([1, 0]),
    )

    with pytest.raises(ValueError, match="token labels"):
        CounterfactualExample(
            sample_id="sample-1",
            factual=factual,
            counterfactual=counterfactual,
        )


def test_normal_reference_rejects_positive_labels_in_any_view() -> None:
    builder = ResidualWriteHypergraphBuilder(attention_threshold=0.15)
    factual_trace = replace(
        make_trace(view_name="factual"),
        token_labels=np.asarray([-1, -1]),
    )
    counterfactual_trace = replace(
        make_trace(view_name="counterfactual"),
        token_labels=np.asarray([1, 1]),
    )

    with pytest.raises(ValueError, match="normal_reference"):
        CounterfactualExample(
            sample_id="sample-1",
            factual=builder.build(factual_trace),
            counterfactual=builder.build(counterfactual_trace),
            normal_reference=True,
        )


def test_contrastive_eligibility_rejects_known_hallucinated_tokens() -> None:
    builder = ResidualWriteHypergraphBuilder(attention_threshold=0.15)

    with pytest.raises(ValueError, match="contrastive_eligible"):
        CounterfactualExample(
            sample_id="sample-1",
            factual=builder.build(make_trace(view_name="factual")),
            counterfactual=builder.build(
                make_trace(view_name="counterfactual")
            ),
            contrastive_eligible=np.asarray([False, True]),
        )


def test_graph_contract_rejects_an_edge_without_its_declared_receiver() -> None:
    graph = ResidualWriteHypergraphBuilder(attention_threshold=0.15).build(
        make_trace()
    )
    keep = ~(
        (graph.incidence[1] == 0)
        & (graph.incidence[0] == graph.receivers[0])
    )

    with pytest.raises(ValueError, match="declared receiver"):
        replace(graph, incidence=graph.incidence[:, keep])
