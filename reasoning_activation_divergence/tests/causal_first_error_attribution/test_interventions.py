from __future__ import annotations

import numpy as np
import pytest

from functional_divergence.causal_first_error_attribution.interventions import (
    CausalInterventionRunner,
    FactorialOutcomes,
    classify_intervention_scope,
    replace_source_message,
)


def test_source_specific_patch_changes_only_selected_attention_message() -> None:
    messages = np.asarray(
        [
            [[1.0, 0.0], [2.0, 0.0]],
            [[0.0, 3.0], [0.0, 4.0]],
        ],
        dtype=np.float64,
    )
    branch = messages.sum(axis=(0, 1))
    replacement = np.asarray([10.0, 20.0])

    patched_branch, patched_messages = replace_source_message(
        branch,
        messages,
        head=0,
        source=1,
        replacement=replacement,
    )

    expected_messages = messages.copy()
    expected_messages[0, 1] = replacement
    np.testing.assert_array_equal(patched_messages, expected_messages)
    np.testing.assert_array_equal(patched_branch, expected_messages.sum(axis=(0, 1)))
    np.testing.assert_array_equal(patched_messages[1], messages[1])
    np.testing.assert_array_equal(patched_messages[0, 0], messages[0, 0])


def test_factorial_patch_recovers_known_attention_ffn_and_interaction_effects() -> None:
    outcomes = FactorialOutcomes(m00=-2.0, m10=1.0, m01=0.0, m11=6.0)

    effects = outcomes.effects()

    assert effects.attention == pytest.approx(3.0)
    assert effects.ffn == pytest.approx(2.0)
    assert effects.interaction == pytest.approx(3.0)


def test_natural_post_divergence_patch_is_labelled_rescue_not_root_cause() -> None:
    assert (
        classify_intervention_scope("target_correction", patch_changed_margin=True)
        == "mediation_or_rescue"
    )
    assert (
        classify_intervention_scope("controlled_root", patch_changed_margin=True)
        == "controlled_path_attribution_candidate"
    )
    assert (
        classify_intervention_scope("target_correction", patch_changed_margin=False)
        == "unsupported"
    )


def test_intervention_runner_composes_attention_and_ffn_branches_exactly() -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    class IdentityBranch(torch.nn.Module):
        def forward(self, hidden_states, **_kwargs):
            return hidden_states

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = IdentityBranch()
            self.mlp = IdentityBranch()

        def forward(self, hidden_states, **kwargs):
            attention = self.self_attn(hidden_states, **kwargs)
            after_attention = hidden_states + attention
            return after_attention + self.mlp(after_attention)

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(3, 2)
            self.layers = torch.nn.ModuleList([Block()])
            with torch.no_grad():
                self.embed_tokens.weight.zero_()
                self.embed_tokens.weight[1, 0] = 1.0
                self.embed_tokens.weight[2, 0] = 2.0

        def forward(self, input_ids, **kwargs):
            hidden = self.embed_tokens(input_ids)
            for block in self.layers:
                hidden = block(hidden, **kwargs)
            return hidden

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Backbone()
            self.lm_head = torch.nn.Linear(2, 3, bias=False)
            with torch.no_grad():
                self.lm_head.weight.zero_()
                self.lm_head.weight[1, 0] = 1.0
            self.config = SimpleNamespace()

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.lm_head(self.model(input_ids, **kwargs)))

    run = CausalInterventionRunner(layer=1).run(
        model=Model(),
        recipient_input_ids=np.asarray([1], dtype=np.int64),
        donor_input_ids=np.asarray([2], dtype=np.int64),
        correct_token_id=1,
        wrong_token_id=0,
        pair_kind="controlled_root",
    )

    assert run.factorial.m00 == pytest.approx(4.0)
    assert run.factorial.m10 == pytest.approx(5.0)
    assert run.factorial.m01 == pytest.approx(6.0)
    assert run.factorial.m11 == pytest.approx(7.0)
    assert run.pre_state_margin == pytest.approx(8.0)
    assert run.effects.attention == pytest.approx(1.0)
    assert run.effects.ffn == pytest.approx(2.0)
    assert run.effects.interaction == pytest.approx(0.0)
    assert run.claim_scope == "controlled_path_attribution_candidate"
