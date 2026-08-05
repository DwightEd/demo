from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hidden_state_geometry.component_replay import (
    _as_hidden,
    _attention_module,
    _mlp_module,
    _resolve_model_topology,
)


@dataclass(frozen=True)
class FactorialEffects:
    attention: float
    ffn: float
    interaction: float


@dataclass(frozen=True)
class FactorialOutcomes:
    """Correct-vs-wrong log-probability margins under a 2x2 branch patch."""

    m00: float
    m10: float
    m01: float
    m11: float

    def __post_init__(self) -> None:
        values = np.asarray([self.m00, self.m10, self.m01, self.m11], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("factorial outcomes must be finite")

    def effects(self) -> FactorialEffects:
        return FactorialEffects(
            attention=float(self.m10 - self.m00),
            ffn=float(self.m01 - self.m00),
            interaction=float(self.m11 - self.m10 - self.m01 + self.m00),
        )


@dataclass(frozen=True)
class InterventionRun:
    factorial: FactorialOutcomes
    pre_state_margin: float
    claim_scope: str

    @property
    def effects(self) -> FactorialEffects:
        return self.factorial.effects()


@dataclass(frozen=True)
class _BranchSnapshot:
    pre_state: object
    attention: object
    ffn: object
    margin: float


def replace_source_message(
    attention_branch: np.ndarray,
    source_messages: np.ndarray,
    *,
    head: int,
    source: int,
    replacement: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace exactly one [head, source] residual message."""
    branch = np.asarray(attention_branch)
    messages = np.asarray(source_messages)
    value = np.asarray(replacement)
    if messages.ndim != 3:
        raise ValueError("source_messages must have shape [H,K,D]")
    if branch.shape != (messages.shape[2],) or value.shape != branch.shape:
        raise ValueError("branch and replacement must have hidden shape [D]")
    if not np.allclose(branch, messages.sum(axis=(0, 1)), rtol=1e-5, atol=1e-7):
        raise ValueError("source_messages do not reconstruct attention_branch")
    if not (0 <= int(head) < messages.shape[0]):
        raise IndexError("head lies outside source_messages")
    if not (0 <= int(source) < messages.shape[1]):
        raise IndexError("source lies outside source_messages")
    patched = messages.copy()
    patched[int(head), int(source)] = value
    return patched.sum(axis=(0, 1)), patched


def classify_intervention_scope(
    pair_kind: str, *, patch_changed_margin: bool
) -> str:
    """Keep rescue/mediation evidence separate from root-cause candidates."""
    if pair_kind not in {"target_correction", "controlled_root"}:
        raise ValueError("unsupported pair_kind")
    if not patch_changed_margin:
        return "unsupported"
    if pair_kind == "controlled_root":
        return "controlled_path_attribution_candidate"
    return "mediation_or_rescue"


def _margin(logits, correct_token_id: int, wrong_token_id: int) -> float:
    return float(
        (
            logits[0, -1, int(correct_token_id)]
            - logits[0, -1, int(wrong_token_id)]
        )
        .float()
        .detach()
        .cpu()
        .item()
    )


def _replace_output(output, value):
    hidden = _as_hidden(output).clone()
    hidden[0, -1] = value.to(device=hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


class CausalInterventionRunner:
    """Compose recipient and donor branches at one pre-registered block."""

    def __init__(self, *, layer: int) -> None:
        self.layer = int(layer)
        if self.layer < 1:
            raise ValueError("layer must be a one-based positive block index")

    def run(
        self,
        *,
        model: object,
        recipient_input_ids: np.ndarray,
        donor_input_ids: np.ndarray,
        correct_token_id: int,
        wrong_token_id: int,
        pair_kind: str,
    ) -> InterventionRun:
        recipient = self._snapshot(
            model,
            recipient_input_ids,
            correct_token_id=correct_token_id,
            wrong_token_id=wrong_token_id,
        )
        donor = self._snapshot(
            model,
            donor_input_ids,
            correct_token_id=correct_token_id,
            wrong_token_id=wrong_token_id,
        )
        m10 = self._patched_margin(
            model,
            recipient_input_ids,
            correct_token_id,
            wrong_token_id,
            attention=donor.attention,
            ffn=recipient.ffn,
        )
        m01 = self._patched_margin(
            model,
            recipient_input_ids,
            correct_token_id,
            wrong_token_id,
            ffn=donor.ffn,
        )
        m11 = self._patched_margin(
            model,
            recipient_input_ids,
            correct_token_id,
            wrong_token_id,
            attention=donor.attention,
            ffn=donor.ffn,
        )
        pre_state_margin = self._patched_margin(
            model,
            recipient_input_ids,
            correct_token_id,
            wrong_token_id,
            pre_state=donor.pre_state,
        )
        factorial = FactorialOutcomes(recipient.margin, m10, m01, m11)
        effects = factorial.effects()
        changed = max(
            abs(effects.attention),
            abs(effects.ffn),
            abs(effects.interaction),
            abs(pre_state_margin - recipient.margin),
        ) > 1e-9
        return InterventionRun(
            factorial=factorial,
            pre_state_margin=pre_state_margin,
            claim_scope=classify_intervention_scope(
                pair_kind, patch_changed_margin=changed
            ),
        )

    def _block(self, model: object):
        topology = _resolve_model_topology(model)
        if self.layer > len(topology.blocks):
            raise ValueError("layer exceeds the model block count")
        return topology.blocks[self.layer - 1]

    @staticmethod
    def _tokens(model: object, token_ids: np.ndarray):
        import torch

        values = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if values.size < 1:
            raise ValueError("intervention input must contain at least one token")
        device = next(model.parameters()).device
        return torch.as_tensor(values[None, :], device=device, dtype=torch.long)

    @staticmethod
    def _forward(model: object, token_tensor):
        return model(
            input_ids=token_tensor,
            attention_mask=token_tensor.new_ones(token_tensor.shape),
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )

    def _snapshot(
        self,
        model: object,
        token_ids: np.ndarray,
        *,
        correct_token_id: int,
        wrong_token_id: int,
    ) -> _BranchSnapshot:
        import torch

        block = self._block(model)
        attention_module = _attention_module(block)
        ffn_module = _mlp_module(block)
        captured: dict[str, object] = {}

        def pre_hook(_module, inputs):
            captured["pre_state"] = inputs[0][0, -1].detach().clone()

        def attention_hook(_module, _inputs, output):
            captured["attention"] = _as_hidden(output)[0, -1].detach().clone()

        def ffn_hook(_module, _inputs, output):
            captured["ffn"] = _as_hidden(output)[0, -1].detach().clone()

        handles = [
            block.register_forward_pre_hook(pre_hook),
            attention_module.register_forward_hook(attention_hook),
            ffn_module.register_forward_hook(ffn_hook),
        ]
        model.eval()
        try:
            with torch.inference_mode():
                result = self._forward(model, self._tokens(model, token_ids))
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != {"pre_state", "attention", "ffn"}:
            raise RuntimeError("intervention snapshot missed a transformer branch")
        return _BranchSnapshot(
            pre_state=captured["pre_state"],
            attention=captured["attention"],
            ffn=captured["ffn"],
            margin=_margin(result.logits, correct_token_id, wrong_token_id),
        )

    def _patched_margin(
        self,
        model: object,
        token_ids: np.ndarray,
        correct_token_id: int,
        wrong_token_id: int,
        *,
        pre_state=None,
        attention=None,
        ffn=None,
    ) -> float:
        import torch

        block = self._block(model)
        handles = []
        if pre_state is not None:

            def patch_pre(_module, inputs):
                hidden = inputs[0].clone()
                hidden[0, -1] = pre_state.to(
                    device=hidden.device, dtype=hidden.dtype
                )
                return (hidden, *inputs[1:])

            handles.append(block.register_forward_pre_hook(patch_pre))
        if attention is not None:

            def patch_attention(_module, _inputs, output):
                return _replace_output(output, attention)

            handles.append(
                _attention_module(block).register_forward_hook(patch_attention)
            )
        if ffn is not None:

            def patch_ffn(_module, _inputs, output):
                return _replace_output(output, ffn)

            handles.append(_mlp_module(block).register_forward_hook(patch_ffn))
        model.eval()
        try:
            with torch.inference_mode():
                result = self._forward(model, self._tokens(model, token_ids))
        finally:
            for handle in handles:
                handle.remove()
        return _margin(result.logits, correct_token_id, wrong_token_id)


__all__ = [
    "CausalInterventionRunner",
    "FactorialEffects",
    "FactorialOutcomes",
    "InterventionRun",
    "classify_intervention_scope",
    "replace_source_message",
]
