from __future__ import annotations

import numpy as np
import pytest

from functional_divergence.causal_first_error_attribution.message_fisher import (
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
