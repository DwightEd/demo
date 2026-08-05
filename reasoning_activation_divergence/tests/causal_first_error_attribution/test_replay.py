from __future__ import annotations

import numpy as np
import pytest

from functional_divergence.causal_first_error_attribution.replay import (
    DecisionTraceExtractor,
    edge_margin_proxy,
    head_source_graph,
    head_source_residual_messages,
)


def test_head_source_messages_reconstruct_attention_branch() -> None:
    torch = pytest.importorskip("torch")
    attention = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    attention[0, 0, 2] = torch.tensor([0.2, 0.3, 0.5])
    attention[0, 1, 2] = torch.tensor([0.6, 0.1, 0.3])
    values = torch.tensor(
        [
            [
                [[1.0, 2.0], [10.0, 20.0]],
                [[3.0, 4.0], [30.0, 40.0]],
                [[5.0, 6.0], [50.0, 60.0]],
            ]
        ],
        dtype=torch.float32,
    )
    output_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    messages, mass, head_output = head_source_residual_messages(
        attention,
        values,
        output_weight,
        query_position=2,
    )

    contexts = attention[0, :, 2, :, None] * values[0].permute(1, 0, 2)
    expected = contexts.sum(dim=1).reshape(-1)
    assert messages.shape == (2, 3, 4)
    assert mass.shape == (2, 3)
    assert head_output.shape == (2, 2)
    assert torch.allclose(messages.sum(dim=(0, 1)), expected)
    assert torch.allclose(head_output, contexts.sum(dim=1))


def test_graph_edge_uses_residual_message_not_attention_weight() -> None:
    torch = pytest.importorskip("torch")
    attention = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    output_weight = torch.eye(2, dtype=torch.float32)
    values_a = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    values_b = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32)
    output_direction = torch.tensor([2.0, -3.0], dtype=torch.float32)

    messages_a, mass_a, _ = head_source_residual_messages(
        attention, values_a, output_weight, query_position=0
    )
    messages_b, mass_b, _ = head_source_residual_messages(
        attention, values_b, output_weight, query_position=0
    )
    score_a = edge_margin_proxy(messages_a, output_direction)
    score_b = edge_margin_proxy(messages_b, output_direction)

    assert torch.equal(mass_a, mass_b)
    assert score_a.item() == pytest.approx(2.0)
    assert score_b.item() == pytest.approx(-3.0)
    assert score_a.item() != score_b.item()


def test_memory_bounded_graph_matches_dense_residual_messages() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(7)
    attention = torch.rand((1, 2, 4, 4), generator=generator)
    attention = attention / attention.sum(dim=-1, keepdim=True)
    values = torch.randn((1, 4, 2, 3), generator=generator)
    output_weight = torch.randn((6, 6), generator=generator)
    output_direction = torch.randn((6,), generator=generator)

    messages, dense_mass, dense_heads = head_source_residual_messages(
        attention, values, output_weight, query_position=3
    )
    proxy, mass, heads, reconstructed = head_source_graph(
        attention,
        values,
        output_weight,
        output_direction,
        query_position=3,
    )

    assert torch.allclose(proxy, edge_margin_proxy(messages, output_direction))
    assert torch.equal(mass, dense_mass)
    assert torch.allclose(heads, dense_heads)
    assert torch.allclose(reconstructed, messages.sum(dim=(0, 1)), atol=1e-5)


def test_graph_supports_a_cached_single_decision_query() -> None:
    torch = pytest.importorskip("torch")
    attention = torch.tensor([[[[0.2, 0.3, 0.5]]]], dtype=torch.float32)
    values = torch.tensor(
        [[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]],
        dtype=torch.float32,
    )
    output_weight = torch.eye(2, dtype=torch.float32)
    direction = torch.tensor([1.0, -1.0], dtype=torch.float32)

    proxy, mass, heads, reconstructed = head_source_graph(
        attention,
        values,
        output_weight,
        direction,
        query_position=0,
    )

    assert proxy.shape == (1, 3)
    assert mass.shape == (1, 3)
    assert heads.shape == (1, 2)
    assert reconstructed.shape == (2,)


def test_replay_rejects_future_query_or_key_positions() -> None:
    torch = pytest.importorskip("torch")
    attention = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    values = torch.ones((1, 2, 1, 2), dtype=torch.float32)
    output_weight = torch.eye(2, dtype=torch.float32)

    with pytest.raises(ValueError, match="final observable token"):
        head_source_residual_messages(
            attention, values, output_weight, query_position=0
        )


def test_decision_trace_extractor_keeps_heads_and_source_tokens() -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    class FakeAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = 1
            self.num_key_value_heads = 1
            self.head_dim = 2
            self.v_proj = torch.nn.Linear(2, 2, bias=False)
            self.o_proj = torch.nn.Linear(2, 2, bias=False)
            torch.nn.init.eye_(self.v_proj.weight)
            torch.nn.init.eye_(self.o_proj.weight)

        def forward(self, hidden_states, **_kwargs):
            length = hidden_states.shape[1]
            values = self.v_proj(hidden_states).reshape(1, length, 1, 2)
            causal = torch.tril(torch.ones(length, length, device=hidden_states.device))
            weights = causal / causal.sum(dim=-1, keepdim=True)
            weights = weights[None, None]
            context = torch.einsum("bhqk,bkhd->bqhd", weights, values)
            output = self.o_proj(context.reshape(1, length, 2))
            return output, weights

    class FakeBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = FakeAttention()
            self.mlp = torch.nn.Linear(2, 2, bias=False)
            torch.nn.init.zeros_(self.mlp.weight)

        def forward(self, hidden_states, **kwargs):
            attention, weights = self.self_attn(hidden_states, **kwargs)
            after_attention = hidden_states + attention
            return after_attention + self.mlp(after_attention), weights

    class FakeBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(16, 2)
            self.layers = torch.nn.ModuleList([FakeBlock()])

        def forward(self, input_ids, **kwargs):
            hidden = self.embed_tokens(input_ids)
            for block in self.layers:
                hidden = block(hidden, **kwargs)[0]
            return hidden

    class FakeLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = FakeBackbone()
            self.lm_head = torch.nn.Linear(2, 16, bias=False)
            self.config = SimpleNamespace(
                _attn_implementation="eager",
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=2,
            )

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids, **kwargs):
            hidden = self.model(input_ids, **kwargs)
            return SimpleNamespace(logits=self.lm_head(hidden))

    model = FakeLM()
    artifact = DecisionTraceExtractor(layers=(1,), topk=3).extract(
        model=model,
        input_ids=np.asarray([1, 2, 3], dtype=np.int64),
        source_step_ids=np.asarray([-1, 0, 0], dtype=np.int16),
        correct_token_id=4,
        wrong_token_id=5,
        metadata={
            "case_id": "synthetic",
            "model_name": "fake",
            "model_revision_or_unknown": "test",
            "tokenizer_name": "fake",
            "tokenizer_revision_or_unknown": "test",
            "source_trace_fingerprint": "trace",
            "pair_file_fingerprint": "pairs",
        },
    )

    assert artifact.decision_position == 2
    assert artifact.attn_head_output.shape == (1, 1, 2)
    assert artifact.attn_edge_mass.shape == (1, 1, 3)
    assert artifact.attn_edge_mass.dtype == np.float32
    assert artifact.attn_edge_margin_proxy.shape == (1, 1, 3)
    assert artifact.metadata["attention_reconstruction_max_relative_error"] < 1e-6


def test_decision_trace_extractor_uses_real_llama_cache_contract() -> None:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(config).eval()

    artifact = DecisionTraceExtractor(layers=(1, 2), topk=4).extract(
        model=model,
        input_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        source_step_ids=np.asarray([-1, -1, 0, 0], dtype=np.int16),
        correct_token_id=5,
        wrong_token_id=6,
        metadata={
            "case_id": "tiny-llama",
            "model_name": "tiny-random-llama",
            "model_revision_or_unknown": "test",
            "tokenizer_name": "none",
            "tokenizer_revision_or_unknown": "test",
            "source_trace_fingerprint": "trace",
            "pair_file_fingerprint": "pairs",
        },
    )

    assert artifact.attn_edge_mass.shape == (2, 2, 4)
    np.testing.assert_allclose(
        artifact.attn_edge_mass.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-5
    )
