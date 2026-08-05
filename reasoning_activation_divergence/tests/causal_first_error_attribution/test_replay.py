from __future__ import annotations

import numpy as np
import pytest

from functional_divergence.causal_first_error_attribution.replay import (
    DecisionTraceExtractor,
    edge_margin_proxy,
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
    assert artifact.attn_edge_margin_proxy.shape == (1, 1, 3)
    assert artifact.metadata["attention_reconstruction_max_relative_error"] < 1e-6
