from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np

from ..hidden_state_geometry.component_replay import (
    _as_hidden,
    _attention_module,
    _expand_grouped_query_values,
    _head_topology,
    _mlp_module,
    _reshape_value_projection,
    _resolve_model_topology,
)
from .contracts import ONSET_TRACE_SCHEMA, OnsetTraceArtifact


def head_source_residual_messages(
    attention,
    expanded_values,
    output_weight,
    *,
    query_position: int,
):
    """Resolve one decision token's head/source messages in residual space.

    The caller must crop the replay input at the decision boundary. Enforcing the
    decision query as the final token makes future-token leakage impossible at
    this boundary rather than relying only on the model's causal mask.
    """
    import torch

    weights = torch.as_tensor(attention)
    values = torch.as_tensor(expanded_values)
    projection = torch.as_tensor(output_weight)
    if weights.ndim != 4 or weights.shape[0] != 1:
        raise ValueError("attention must have shape [1,H,Q,K]")
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError("expanded_values must have shape [1,K,H,Dh]")
    if int(query_position) != int(weights.shape[2]) - 1:
        raise ValueError("decision query must be the final observable token")
    if weights.shape[2] != weights.shape[3]:
        raise ValueError("decision replay requires a square causal prefix")
    if values.shape[1] != weights.shape[3] or values.shape[2] != weights.shape[1]:
        raise ValueError("attention and expanded value topology disagree")

    heads = int(weights.shape[1])
    head_dim = int(values.shape[3])
    hidden = heads * head_dim
    if projection.shape != (hidden, hidden):
        raise ValueError("output_weight must have shape [H*Dh,H*Dh]")

    mass = weights[0, :, int(query_position), :].float()
    value_by_head = values[0].permute(1, 0, 2).float()
    contexts = mass[:, :, None] * value_by_head
    head_projection = projection.float().reshape(hidden, heads, head_dim)
    messages = torch.einsum("hkd,ohd->hko", contexts, head_projection)
    return messages, mass, contexts.sum(dim=1)


def edge_margin_proxy(messages, output_direction):
    """Project residual messages onto a pre-registered correct-vs-wrong axis."""
    import torch

    values = torch.as_tensor(messages)
    direction = torch.as_tensor(output_direction, device=values.device)
    if values.ndim != 3:
        raise ValueError("messages must have shape [H,K,D]")
    if direction.ndim != 1 or direction.shape[0] != values.shape[2]:
        raise ValueError("output_direction must have shape [D]")
    return torch.einsum("hkd,d->hk", values.float(), direction.float())


class _DecisionCapture:
    def __init__(
        self,
        model: object,
        blocks: Sequence[object],
        layers: tuple[int, ...],
        output_direction,
        reconstruction_rtol: float,
    ) -> None:
        self.model = model
        self.blocks = blocks
        self.layers = layers
        self.output_direction = output_direction
        self.reconstruction_rtol = float(reconstruction_rtol)
        self.pre: dict[int, object] = {}
        self.raw_values: dict[int, object] = {}
        self.head_output: dict[int, object] = {}
        self.attn_output: dict[int, object] = {}
        self.mlp_output: dict[int, object] = {}
        self.edge_proxy: dict[int, object] = {}
        self.edge_mass: dict[int, object] = {}
        self.relative_error: dict[int, float] = {}
        self.handles: list[object] = []

    def __enter__(self) -> Self:
        for position, depth in enumerate(self.layers):
            block = self.blocks[depth - 1]
            attention = _attention_module(block)
            mlp = _mlp_module(block)
            value_projection = getattr(attention, "v_proj", None)
            if value_projection is None:
                raise TypeError("attention module does not expose v_proj")

            def capture_pre(_module, inputs, *, index=position):
                hidden = inputs[0]
                self.pre[index] = hidden[0, -1].detach()

            def capture_values(_module, _inputs, output, *, index=position):
                values = output[0] if isinstance(output, (tuple, list)) else output
                self.raw_values[index] = values.detach()

            def capture_attention(_module, _inputs, output, *, index=position):
                self._capture_attention(index, _module, output)

            def capture_mlp(_module, _inputs, output, *, index=position):
                self.mlp_output[index] = _as_hidden(output)[0, -1].detach()

            self.handles.append(block.register_forward_pre_hook(capture_pre))
            self.handles.append(
                value_projection.register_forward_hook(capture_values)
            )
            self.handles.append(attention.register_forward_hook(capture_attention))
            self.handles.append(mlp.register_forward_hook(capture_mlp))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _capture_attention(self, position: int, module: object, output: object) -> None:
        import torch

        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError("attention module did not return attention weights")
        attention = output[1]
        if attention is None:
            raise RuntimeError(
                "attention weights are unavailable; use eager attention and "
                "output_attentions=True"
            )
        raw_values = self.raw_values.pop(position, None)
        if raw_values is None:
            raise RuntimeError("v_proj capture did not run before attention output")
        heads, kv_heads, head_dim = _head_topology(self.model, module)
        values = _reshape_value_projection(
            raw_values, kv_heads=kv_heads, head_dim=head_dim
        )
        expanded = _expand_grouped_query_values(values, heads)
        output_projection = getattr(module, "o_proj", None)
        if output_projection is None or not hasattr(output_projection, "weight"):
            raise TypeError("attention module does not expose o_proj.weight")
        messages, mass, head_output = head_source_residual_messages(
            attention,
            expanded,
            output_projection.weight,
            query_position=int(attention.shape[2]) - 1,
        )
        actual = _as_hidden(output)[0, -1].float()
        reconstructed = messages.sum(dim=(0, 1)).float()
        relative = torch.linalg.vector_norm(reconstructed - actual) / torch.clamp(
            torch.linalg.vector_norm(actual), min=1e-12
        )
        value = float(relative.detach().cpu().item())
        if value > self.reconstruction_rtol:
            raise ValueError(
                "head/source messages do not reconstruct the attention branch: "
                f"relative_error={value:.6g}"
            )
        self.head_output[position] = head_output.detach()
        self.attn_output[position] = actual.detach()
        self.edge_proxy[position] = edge_margin_proxy(
            messages, self.output_direction
        ).detach()
        self.edge_mass[position] = mass.detach()
        self.relative_error[position] = value

    def arrays(self) -> tuple[np.ndarray, ...]:
        stores = (
            self.pre,
            self.head_output,
            self.attn_output,
            self.mlp_output,
            self.edge_proxy,
            self.edge_mass,
        )
        if any(set(store) != set(range(len(self.layers))) for store in stores):
            raise RuntimeError("decision replay did not capture every selected layer")

        def stack(store: dict[int, object]) -> np.ndarray:
            return np.stack(
                [store[index].float().cpu().numpy() for index in range(len(self.layers))]
            )

        return tuple(stack(store) for store in stores)


class DecisionTraceExtractor:
    """Replay one cropped prefix and extract its decision-token message graph."""

    def __init__(
        self,
        *,
        layers: Sequence[int],
        topk: int = 20,
        reconstruction_rtol: float = 3e-2,
    ) -> None:
        self.layers = tuple(int(layer) for layer in layers)
        self.topk = int(topk)
        self.reconstruction_rtol = float(reconstruction_rtol)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must be a non-empty unique sequence")
        if min(self.layers) < 1 or self.topk < 1:
            raise ValueError("layers and topk must be positive")
        if not np.isfinite(self.reconstruction_rtol) or self.reconstruction_rtol < 0:
            raise ValueError("reconstruction_rtol must be finite and nonnegative")

    def extract(
        self,
        *,
        model: object,
        input_ids: np.ndarray,
        source_step_ids: np.ndarray,
        correct_token_id: int,
        wrong_token_id: int,
        metadata: dict[str, Any],
    ) -> OnsetTraceArtifact:
        import torch

        tokens = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        source_steps = np.asarray(source_step_ids, dtype=np.int16).reshape(-1)
        if tokens.size < 1 or source_steps.shape != tokens.shape:
            raise ValueError("input_ids and source_step_ids must be aligned")
        topology = _resolve_model_topology(model)
        if max(self.layers) > len(topology.blocks):
            raise ValueError("selected layer exceeds the model block count")
        implementation = str(
            getattr(getattr(model, "config", None), "_attn_implementation", "")
        )
        if implementation and implementation != "eager":
            raise ValueError("decision replay requires eager attention")
        output_embeddings = model.get_output_embeddings()
        weights = output_embeddings.weight
        vocabulary = int(weights.shape[0])
        if not (0 <= int(correct_token_id) < vocabulary):
            raise ValueError("correct_token_id lies outside the vocabulary")
        if not (0 <= int(wrong_token_id) < vocabulary):
            raise ValueError("wrong_token_id lies outside the vocabulary")
        output_direction = (
            weights[int(correct_token_id)] - weights[int(wrong_token_id)]
        ).detach()
        device = next(model.parameters()).device
        token_tensor = torch.as_tensor(tokens[None, :], device=device, dtype=torch.long)
        attention_mask = torch.ones_like(token_tensor)
        model.eval()
        with torch.inference_mode():
            with _DecisionCapture(
                model,
                topology.blocks,
                self.layers,
                output_direction,
                self.reconstruction_rtol,
            ) as capture:
                result = model(
                    input_ids=token_tensor,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=True,
                    return_dict=True,
                )
            (
                resid,
                head_output,
                attn_output,
                mlp_output,
                edge_proxy,
                edge_mass,
            ) = capture.arrays()
        logits = result.logits[0, -1].float()
        count = min(self.topk, int(logits.shape[0]))
        top_values, top_ids = torch.topk(logits, k=count)
        artifact_metadata = {
            **metadata,
            "schema": ONSET_TRACE_SCHEMA,
            "extraction_config": {
                "layers": list(self.layers),
                "topk": self.topk,
                "reconstruction_rtol": self.reconstruction_rtol,
            },
            "attention_reconstruction_max_relative_error": max(
                capture.relative_error.values(), default=0.0
            ),
        }
        artifact = OnsetTraceArtifact(
            input_ids=tokens.astype(np.int32),
            decision_position=len(tokens) - 1,
            source_token_positions=np.arange(len(tokens), dtype=np.int32),
            source_step_ids=source_steps,
            selected_layers=np.asarray(self.layers, dtype=np.int16),
            wrong_token_id=int(wrong_token_id),
            correct_token_id=int(correct_token_id),
            logits_topk_ids=top_ids.cpu().numpy().astype(np.int32),
            logits_topk_values=top_values.cpu().numpy().astype(np.float32),
            resid_pre_block=resid.astype(np.float16),
            attn_head_output=head_output.astype(np.float16),
            attn_branch_output=attn_output.astype(np.float16),
            mlp_output=mlp_output.astype(np.float16),
            attn_edge_margin_proxy=edge_proxy.astype(np.float16),
            attn_edge_mass=edge_mass.astype(np.float16),
            metadata=artifact_metadata,
        )
        artifact.validate()
        return artifact


__all__ = [
    "DecisionTraceExtractor",
    "edge_margin_proxy",
    "head_source_residual_messages",
]
