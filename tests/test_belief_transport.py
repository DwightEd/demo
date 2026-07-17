from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from prompt_control_flow.belief_transport.artifact import (
    BeliefTraceArtifact,
    merge_belief_trace_artifacts,
)
from prompt_control_flow.belief_transport.belief import (
    fisher_rao_distance,
    mask_to_belief,
    masked_belief_update,
    support_log_odds,
    transition_diagnostics,
)
from prompt_control_flow.belief_transport.audit import (
    BeliefAuditConfig,
    belief_prediction_metrics,
    build_matched_wrong_masks,
    directional_transport_rows,
)
from prompt_control_flow.belief_transport.decoder import build_group_folds
from prompt_control_flow.belief_transport.decoder import (
    DecoderConfig,
    cross_fit_belief_decoder,
)
from prompt_control_flow.belief_transport.model_capture import (
    SelectiveBoundaryCapture,
    resolve_model_topology,
)
from prompt_control_flow.belief_transport.world import (
    WindTunnelConfig,
    build_hypothesis_grid,
    generate_worlds,
    load_worlds_jsonl,
    write_worlds_jsonl,
)
from prompt_control_flow.belief_transport.extraction import (
    BoundaryExtractionConfig,
    build_extraction_rows,
    length_bucket_batches,
    render_chat_observer_prompt,
)


def test_exact_mask_update_matches_uniform_conditioning() -> None:
    prior = np.full(4, 0.25, dtype=np.float64)
    condition = np.asarray([1, 0, 1, 0], dtype=bool)

    posterior = masked_belief_update(prior, condition)

    assert np.allclose(posterior, [0.5, 0.0, 0.5, 0.0])
    assert np.isclose(mask_to_belief(condition).sum(), 1.0)


def test_boundary_extraction_rejects_negative_projection_seed() -> None:
    with pytest.raises(ValueError, match="output_sketch_seed"):
        BoundaryExtractionConfig(output_sketch_seed=-1).validate()


def test_belief_audit_rejects_negative_null_match_tolerance() -> None:
    with pytest.raises(ValueError, match="max_null_information_gain_gap"):
        BeliefAuditConfig(max_null_information_gain_gap=-1.0).validate()


def test_fisher_rao_is_symmetric_and_zero_on_identity() -> None:
    p = np.asarray([0.7, 0.2, 0.1], dtype=np.float64)
    q = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)

    assert np.isclose(fisher_rao_distance(p, p), 0.0)
    assert np.isclose(fisher_rao_distance(p, q), fisher_rao_distance(q, p))
    assert fisher_rao_distance(p, q) > 0.0


def test_transition_diagnostics_reward_the_true_constraint_operator() -> None:
    before = np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    true_mask = np.asarray([1, 0, 1, 0], dtype=bool)
    wrong_mask = np.asarray([0, 1, 0, 1], dtype=bool)
    after = masked_belief_update(before, true_mask)

    true = transition_diagnostics(before, after, true_mask)
    wrong = transition_diagnostics(before, after, wrong_mask)

    assert true["transport_residual"] < wrong["transport_residual"]
    assert true["support_gain"] > 0.0
    assert wrong["unsupported_contraction"] > true["unsupported_contraction"]
    assert support_log_odds(after, true_mask) > support_log_odds(before, true_mask)


def test_generated_worlds_have_monotone_exact_posteriors_and_unique_endpoints() -> None:
    cfg = WindTunnelConfig(
        domain_size=6,
        min_steps=3,
        max_steps=6,
        template_families=3,
        seed=13,
    )
    worlds = generate_worlds(24, cfg)
    hypotheses = build_hypothesis_grid(cfg.domain_size)

    assert len(worlds) == 24
    for world in worlds:
        prefixes = world.prefix_states(hypotheses)
        counts = [int(state.feasible_mask.sum()) for state in prefixes]
        assert len(world.conditions) >= cfg.min_steps
        assert len(world.conditions) <= cfg.max_steps
        assert all(a > b for a, b in zip(counts, counts[1:]))
        assert counts[-1] == 1
        target_index = world.target[0] * cfg.domain_size + world.target[1]
        assert prefixes[-1].feasible_mask[target_index]


def test_wind_tunnel_jsonl_roundtrip_preserves_semantics(tmp_path) -> None:
    cfg = WindTunnelConfig(domain_size=5, min_steps=3, max_steps=6, seed=7)
    worlds = generate_worlds(4, cfg)
    path = tmp_path / "wind_tunnel.jsonl"

    write_worlds_jsonl(path, worlds, cfg)
    loaded, loaded_cfg = load_worlds_jsonl(path)

    assert loaded_cfg == cfg
    assert loaded == worlds
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first["schema_version"] == "constraint_belief_wind_tunnel_v1"
    assert "conditions" in first


def test_extraction_rows_preserve_problem_local_transition_alignment() -> None:
    cfg = WindTunnelConfig(domain_size=5, min_steps=3, max_steps=6, seed=19)
    worlds = generate_worlds(3, cfg)

    rows, hypotheses = build_extraction_rows(worlds)

    assert hypotheses.shape == (25, 2)
    for world in worlds:
        local = [row for row in rows if row.problem_id == world.problem_id]
        assert [row.prefix_index for row in local] == list(range(len(local)))
        assert local[0].previous_prefix_index == -1
        assert all(row.previous_prefix_index == row.prefix_index - 1 for row in local[1:])
        assert all(row.feasible_mask[row.target_hypothesis] for row in local)
        assert int(local[-1].feasible_mask.sum()) == 1


def test_belief_trace_artifact_roundtrip_rejects_misaligned_state_rows(tmp_path) -> None:
    cfg = WindTunnelConfig(domain_size=4, min_steps=3, max_steps=5, seed=23)
    rows, hypotheses = build_extraction_rows(generate_worlds(2, cfg))
    n_rows = len(rows)
    layers = np.asarray([0, 2], dtype=np.int64)
    states = np.arange(n_rows * 2 * 6, dtype=np.float32).reshape(n_rows, 2, 6)
    artifact = BeliefTraceArtifact.from_rows(
        rows,
        hypotheses=hypotheses,
        layers=layers,
        states=states,
        prompts=[f"prompt-{i}" for i in range(n_rows)],
        input_ids=[np.asarray([1, 2, i + 3]) for i in range(n_rows)],
        output_entropy=np.linspace(1.0, 0.5, n_rows),
        output_margin=np.linspace(0.1, 1.0, n_rows),
        output_topk_mass=np.linspace(0.2, 0.8, n_rows),
        metadata={"model": "dummy"},
    )
    path = tmp_path / "trace.npz"

    artifact.save(path)
    loaded = BeliefTraceArtifact.load(path)

    assert loaded.states.shape == (n_rows, 2, 6)
    assert np.array_equal(loaded.layers, layers)
    assert loaded.metadata["model"] == "dummy"
    assert loaded.state_semantics == "assistant_boundary_residual_state"

    with np.testing.assert_raises(ValueError):
        BeliefTraceArtifact.from_rows(
            rows,
            hypotheses=hypotheses,
            layers=layers,
            states=states[:-1],
            prompts=["x"] * n_rows,
            input_ids=[np.asarray([1])] * n_rows,
            output_entropy=np.zeros(n_rows),
            output_margin=np.zeros(n_rows),
            output_topk_mass=np.zeros(n_rows),
            metadata={},
        )


def test_length_bucket_batches_respect_row_and_token_limits() -> None:
    batches = list(
        length_bucket_batches(
            token_lengths=[5, 2, 8, 3, 4],
            batch_size=3,
            max_batch_tokens=10,
        )
    )

    assert sorted(index for batch in batches for index in batch) == list(range(5))
    for batch in batches:
        assert len(batch) <= 3
        assert max([5, 2, 8, 3, 4][index] for index in batch) * len(batch) <= 10


def test_chat_rendering_does_not_add_special_tokens_twice() -> None:
    class DummyTokenizer:
        chat_template = "available"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return f"<bos>{messages[1]['content']}<assistant>"

    rendered, add_special_tokens, protocol = render_chat_observer_prompt(
        DummyTokenizer(), "question"
    )

    assert rendered == "<bos>question<assistant>"
    assert add_special_tokens is False
    assert protocol == "tokenizer_chat_template_generation_boundary"


def test_trace_shards_merge_in_problem_prefix_order(tmp_path) -> None:
    cfg = WindTunnelConfig(domain_size=4, min_steps=3, max_steps=5, seed=31)
    worlds = generate_worlds(4, cfg)
    paths = []
    for shard_index, shard_worlds in enumerate((worlds[::2], worlds[1::2])):
        rows, hypotheses = build_extraction_rows(shard_worlds)
        n_rows = len(rows)
        artifact = BeliefTraceArtifact.from_rows(
            rows,
            hypotheses=hypotheses,
            layers=np.asarray([1], dtype=np.int64),
            states=np.full((n_rows, 1, 3), shard_index, dtype=np.float32),
            prompts=[row.prompt_text for row in rows],
            input_ids=[np.asarray([1, row.prefix_index + 2]) for row in rows],
            output_entropy=np.ones(n_rows),
            output_margin=np.ones(n_rows),
            output_topk_mass=np.ones(n_rows),
            metadata={"model": "dummy", "tokenizer": "dummy"},
        )
        path = tmp_path / f"shard-{shard_index}.npz"
        artifact.save(path)
        paths.append(path)

    merged = merge_belief_trace_artifacts(paths)

    ordering = list(zip(merged.problem_ids.tolist(), merged.prefix_index.tolist()))
    assert ordering == sorted(ordering)
    assert len(set(merged.problem_ids.tolist())) == 4


def test_group_folds_never_split_a_problem_and_cover_every_row_once() -> None:
    groups = np.repeat(np.arange(12), 3)
    folds = build_group_folds(groups, num_folds=4, seed=11)
    test_counts = np.zeros(len(groups), dtype=np.int64)

    for train_index, test_index in folds:
        assert set(groups[train_index]).isdisjoint(set(groups[test_index]))
        test_counts[test_index] += 1

    assert np.array_equal(test_counts, np.ones(len(groups), dtype=np.int64))


def test_belief_metrics_prefer_exact_predictions_to_uniform_predictions() -> None:
    support = np.asarray(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    target = np.asarray([mask_to_belief(row) for row in support])
    uniform = np.full_like(target, 0.25)

    exact_metrics = belief_prediction_metrics(target, target, support)
    uniform_metrics = belief_prediction_metrics(target, uniform, support)

    assert exact_metrics["kl_nats"] < uniform_metrics["kl_nats"]
    assert exact_metrics["fisher_rao"] < uniform_metrics["fisher_rao"]
    assert exact_metrics["support_auc"] > uniform_metrics["support_auc"]


def test_exact_beliefs_have_lower_transport_residual_for_true_than_wrong_masks(tmp_path) -> None:
    cfg = WindTunnelConfig(domain_size=5, min_steps=3, max_steps=6, seed=37)
    rows, hypotheses = build_extraction_rows(generate_worlds(30, cfg))
    n_rows = len(rows)
    artifact = BeliefTraceArtifact.from_rows(
        rows,
        hypotheses=hypotheses,
        layers=np.asarray([1]),
        states=np.zeros((n_rows, 1, 2), dtype=np.float32),
        prompts=[row.prompt_text for row in rows],
        input_ids=[np.asarray([1, 2]) for _ in rows],
        output_entropy=np.ones(n_rows),
        output_margin=np.ones(n_rows),
        output_topk_mass=np.ones(n_rows),
        metadata={"model": "dummy", "tokenizer": "dummy"},
    )
    predictions = np.asarray([mask_to_belief(mask) for mask in artifact.feasible_mask])
    wrong_masks = build_matched_wrong_masks(artifact, seed=5)

    transitions = directional_transport_rows(artifact, predictions, wrong_masks)

    assert len(transitions["problem_id"]) == int(np.sum(artifact.prefix_index > 0))
    for index in np.flatnonzero(artifact.prefix_index > 0):
        true_support = artifact.feasible_mask[index]
        wrong_support = wrong_masks[index]
        assert int(wrong_support.sum()) == int(true_support.sum())
        assert int(np.logical_xor(wrong_support, true_support).sum()) == 2
    assert np.mean(transitions["true_transport_residual"]) < 1e-6
    assert np.mean(transitions["wrong_transport_residual"]) > 0.1
    assert np.mean(transitions["operator_advantage"]) > 0.1
    assert np.max(transitions["information_gain_gap"]) < 1e-12


def test_selective_capture_matches_decoder_boundary_without_full_hidden_history() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, increment):
            super().__init__()
            self.increment = float(increment)

        def forward(self, hidden):
            return (hidden + self.increment,)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(16, 4)
            self.layers = nn.ModuleList([Block(1.0), Block(2.0)])
            self.norm = nn.Identity()

        def forward(self, input_ids, attention_mask, use_cache, return_dict):
            hidden = self.embed_tokens(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)[0]
            hidden = self.norm(hidden)
            return SimpleNamespace(last_hidden_state=hidden)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.lm_head = nn.Linear(4, 16, bias=False)

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

    model = Model()
    topology = resolve_model_topology(model)
    input_ids = torch.asarray([[1, 2, 3], [4, 5, 0]])
    attention_mask = torch.asarray([[1, 1, 1], [1, 1, 0]])
    last_indices = torch.asarray([2, 1])

    with SelectiveBoundaryCapture(topology, (0, 1, 2), last_indices) as capture:
        model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        captured = capture.states()

    embedded = model.model.embed_tokens(input_ids)[torch.asarray([0, 1]), last_indices]
    assert captured.shape == (2, 3, 4)
    assert torch.allclose(captured[:, 0], embedded)
    assert torch.allclose(captured[:, 1], embedded + 1.0)
    assert torch.allclose(captured[:, 2], embedded + 3.0)


def test_cross_fit_soft_decoder_recovers_held_out_beliefs_from_informative_states() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(41)
    groups = np.repeat(np.arange(24), 2)
    labels = np.repeat(np.arange(24) % 4, 2)
    features = np.eye(4, dtype=np.float32)[labels]
    features = np.concatenate(
        [features, rng.normal(scale=0.01, size=(len(labels), 4)).astype(np.float32)],
        axis=1,
    )
    target = np.eye(4, dtype=np.float32)[labels]
    cfg = DecoderConfig(
        num_folds=4,
        epochs=50,
        batch_size=32,
        learning_rate=5e-3,
        weight_decay=1e-3,
        patience=8,
        device="cpu",
        seed=43,
    )

    result = cross_fit_belief_decoder(features, target, groups, cfg)
    loss = -np.mean(np.log(np.clip(result.predictions[np.arange(len(labels)), labels], 1e-9, None)))

    assert result.predictions.shape == target.shape
    assert loss < 0.2
