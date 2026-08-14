from __future__ import annotations

import numpy as np
import pytest

from functional_divergence.causal_first_error_attribution.message_fisher import (
    FFN_DIRECTION_ID,
    SourceMessageFisherRunner,
    categorical_fisher_gram,
    fisher_diagnostics,
    source_binned_residual_messages,
)


def test_source_binned_messages_reconstruct_attention_output() -> None:
    torch = pytest.importorskip("torch")
    attention = torch.tensor(
        [[[[0.1, 0.2, 0.3, 0.4]], [[0.4, 0.3, 0.2, 0.1]]]],
        dtype=torch.float32,
    )
    values = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0]],
                [[3.0, 0.0], [0.0, 3.0]],
                [[4.0, 0.0], [0.0, 4.0]],
            ]
        ],
        dtype=torch.float32,
    )
    output_weight = torch.eye(4, dtype=torch.float32)

    source_ids, messages, reconstructed = source_binned_residual_messages(
        attention,
        values,
        output_weight,
        source_step_ids=np.asarray([-1, -1, 0, 1]),
        query_position=0,
    )

    assert source_ids.tolist() == [-1, 0, 1]
    assert messages.shape == (3, 4)
    assert torch.allclose(messages.sum(dim=0), reconstructed, atol=1e-6)


def test_categorical_fisher_gram_matches_explicit_covariance() -> None:
    logits = np.asarray([0.3, -0.2, 0.7], dtype=np.float64)
    jacobian = np.asarray(
        [[1.0, 2.0, -1.0], [0.5, -0.5, 1.5]], dtype=np.float64
    )
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    covariance = np.diag(probabilities) - np.outer(probabilities, probabilities)

    gram = categorical_fisher_gram(probabilities, jacobian)

    np.testing.assert_allclose(gram, jacobian @ covariance @ jacobian.T)


def test_categorical_fisher_ignores_constant_logit_shift_directions() -> None:
    probabilities = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    jacobian = np.asarray(
        [[1.0, 1.0, 1.0], [2.0, -1.0, 0.0]], dtype=np.float64
    )

    gram = categorical_fisher_gram(probabilities, jacobian)

    assert gram[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert gram[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert gram[1, 1] > 0.0


def test_fisher_diagnostics_report_anisotropy_and_typed_loadings() -> None:
    gram = np.diag([9.0, 1.0, 0.0])

    diagnostics = fisher_diagnostics(
        gram,
        direction_source_ids=np.asarray([-1, 0, -32768]),
    )

    assert diagnostics["largest_eigenvalue"] == pytest.approx(9.0)
    assert diagnostics["effective_rank"] == pytest.approx(1.384145488, rel=1e-6)
    assert diagnostics["anisotropy"] == pytest.approx(0.9)
    assert diagnostics["top_prompt_loading"] == pytest.approx(1.0)
    assert diagnostics["top_ffn_loading"] == pytest.approx(0.0)


def test_fisher_math_rejects_invalid_probability_or_direction_shapes() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        categorical_fisher_gram(
            np.asarray([0.2, 0.2]), np.asarray([[1.0, -1.0]])
        )
    with pytest.raises(ValueError, match="vocabulary"):
        categorical_fisher_gram(
            np.asarray([0.5, 0.5]), np.asarray([[1.0, 0.0, -1.0]])
        )


def test_source_message_runner_reconstructs_block_and_measures_output_geometry() -> None:
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
            batch, length, _hidden = hidden_states.shape
            values = self.v_proj(hidden_states).reshape(batch, length, 1, 2)
            causal = torch.tril(
                torch.ones(length, length, device=hidden_states.device)
            )
            weights = causal / causal.sum(dim=-1, keepdim=True)
            weights = weights[None, None].expand(batch, 1, length, length)
            context = torch.einsum("bhqk,bkhd->bqhd", weights, values)
            output = self.o_proj(context.reshape(batch, length, 2))
            return output, weights

    class FakeBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = FakeAttention()
            self.mlp = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.mlp.weight.copy_(torch.tensor([[0.2, 0.1], [0.0, 0.3]]))

        def forward(self, hidden_states, **kwargs):
            attention, weights = self.self_attn(hidden_states, **kwargs)
            after_attention = hidden_states + attention
            return after_attention + self.mlp(after_attention), weights

    class FakeBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 2)
            self.layers = torch.nn.ModuleList([FakeBlock()])
            with torch.no_grad():
                self.embed_tokens.weight.copy_(
                    torch.tensor(
                        [
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [0.0, 1.0],
                            [1.0, 1.0],
                            [2.0, -1.0],
                            [-1.0, 2.0],
                            [0.5, -0.5],
                            [-0.5, 0.5],
                        ]
                    )
                )

        def forward(self, input_ids, **kwargs):
            hidden = self.embed_tokens(input_ids)
            for block in self.layers:
                hidden = block(hidden, **kwargs)[0]
            return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)

    class FakeLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = FakeBackbone()
            self.lm_head = torch.nn.Linear(2, 8, bias=False)
            self.config = SimpleNamespace(
                _attn_implementation="eager",
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=2,
            )
            with torch.no_grad():
                self.lm_head.weight.copy_(self.model.embed_tokens.weight)

        def get_output_embeddings(self):
            return self.lm_head

    result = SourceMessageFisherRunner(
        layers=(1,), epsilon=1e-3, perturbation_batch_size=2
    ).run(
        model=FakeLM(),
        input_ids=np.asarray([1, 2, 3], dtype=np.int64),
        source_step_ids=np.asarray([-1, 0, 0], dtype=np.int16),
    )

    assert result.source_ids.tolist() == [-1, 0]
    assert result.source_messages.shape == (1, 2, 2)
    np.testing.assert_allclose(
        result.source_messages.sum(axis=1), result.attention_output, atol=1e-6
    )
    np.testing.assert_allclose(
        result.residual_pre + result.attention_output + result.mlp_output,
        result.residual_post,
        atol=1e-6,
    )
    assert result.direction_source_ids.tolist() == [-1, 0, FFN_DIRECTION_ID]
    assert result.fisher_gram.shape == (1, 3, 3)
    assert np.linalg.eigvalsh(result.fisher_gram[0]).min() >= -1e-8
    assert result.attention_reconstruction_error.max() < 1e-6
    assert result.block_reconstruction_error.max() < 1e-6
    assert np.isfinite(result.quadratic_relative_error).all()
    assert result.summary_rows()[0]["largest_eigenvalue"] > 0.0


def test_source_message_runner_uses_real_llama_cache_and_gqa_contract() -> None:
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

    result = SourceMessageFisherRunner(
        layers=(1, 2), epsilon=0.05, perturbation_batch_size=2
    ).run(
        model=model,
        input_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        source_step_ids=np.asarray([-1, -1, 0, 0], dtype=np.int16),
    )

    assert result.source_messages.shape == (2, 2, 8)
    assert result.fisher_gram.shape == (2, 3, 3)
    assert result.attention_reconstruction_error.max() < 1e-4
    assert result.block_reconstruction_error.max() < 1e-4
