from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from .component_contract import ATTN_SUM_ATOL, ATTN_SUM_RTOL, COMPONENT_STEP_SCHEMA


@dataclass(frozen=True)
class ComponentExtractionConfig:
    layers: tuple[int, ...]
    model_name: str
    model_revision: str
    tokenizer_name: str
    tokenizer_revision: str
    extractor_commit: str
    attention_reconstruction_atol: float = ATTN_SUM_ATOL
    attention_reconstruction_rtol: float = ATTN_SUM_RTOL
    replay_fidelity_rtol: float = ATTN_SUM_RTOL
    overwrite: bool = False

    def __post_init__(self) -> None:
        layers = tuple(int(value) for value in self.layers)
        if not layers:
            raise ValueError("at least one component layer is required")
        if len(set(layers)) != len(layers):
            raise ValueError("component layers must be unique")
        if any(layer < 1 for layer in layers):
            raise ValueError("component layers are one-based decoder depths")
        object.__setattr__(self, "layers", layers)
        for name in (
            "model_name",
            "model_revision",
            "tokenizer_name",
            "tokenizer_revision",
            "extractor_commit",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "attention_reconstruction_atol",
            "attention_reconstruction_rtol",
            "replay_fidelity_rtol",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)

    def replay_config_sha256(self) -> str:
        payload = {
            "schema": COMPONENT_STEP_SCHEMA,
            "layers": list(self.layers),
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "attention_reconstruction_atol": self.attention_reconstruction_atol,
            "attention_reconstruction_rtol": self.attention_reconstruction_rtol,
            "replay_fidelity_rtol": self.replay_fidelity_rtol,
            "replay_mode": "teacher_forced_backbone_attention_components_v1",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _ModelTopology:
    backbone: object
    blocks: Sequence[object]


@dataclass(frozen=True)
class SourceSpan:
    source_step_id: int
    start: int
    end: int


def _first_attribute(owner: object, names: Sequence[str]) -> object | None:
    for name in names:
        value = getattr(owner, name, None)
        if value is not None:
            return value
    return None


def _resolve_model_topology(model: object) -> _ModelTopology:
    backbone = _first_attribute(model, ("model", "transformer", "base_model"))
    if backbone is None or backbone is model:
        backbone = getattr(model, "base_model", None)
    if backbone is None or backbone is model:
        raise TypeError("could not resolve a decoder-only backbone")

    blocks = _first_attribute(backbone, ("layers", "h", "blocks"))
    if blocks is None:
        decoder = getattr(backbone, "decoder", None)
        blocks = getattr(decoder, "layers", None) if decoder is not None else None
    if blocks is None or len(blocks) < 1:
        raise TypeError("could not resolve transformer blocks")
    return _ModelTopology(backbone=backbone, blocks=blocks)


def _source_spans_for_step(
    *,
    prompt_end: int,
    boundary_token_end: np.ndarray,
    current_step: int,
) -> list[SourceSpan]:
    boundaries = np.asarray(boundary_token_end, dtype=np.int64)
    step = int(current_step)
    if boundaries.ndim != 1 or boundaries.shape[0] < step + 2:
        raise ValueError("boundary_token_end does not cover the requested step")
    if int(boundaries[0]) != int(prompt_end):
        raise ValueError("prompt_end must equal boundary_token_end[0]")
    spans = [SourceSpan(-1, 0, int(prompt_end))]
    for source_step in range(step + 1):
        spans.append(
            SourceSpan(
                source_step,
                int(boundaries[source_step]),
                int(boundaries[source_step + 1]),
            )
        )
    target_exclusive = int(boundaries[step + 1])
    covered = [token for span in spans for token in range(span.start, span.end)]
    if covered != list(range(target_exclusive)):
        raise ValueError("source spans must cover every prefix token exactly once")
    return spans


def _source_spans_for_layout(
    boundary_token_end: np.ndarray, step_count: int
) -> list[list[SourceSpan]]:
    return [
        _source_spans_for_step(
            prompt_end=int(boundary_token_end[0]),
            boundary_token_end=boundary_token_end,
            current_step=step,
        )
        for step in range(int(step_count))
    ]


def _expand_grouped_query_values(values, num_heads: int):
    kv_heads = int(values.shape[2])
    heads = int(num_heads)
    if heads < 1 or kv_heads < 1 or heads % kv_heads != 0:
        raise ValueError("invalid grouped-query attention head topology")
    repeats = heads // kv_heads
    return values.repeat_interleave(repeats, dim=2)


def _span_parts(span: SourceSpan | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(span, SourceSpan):
        return span.source_step_id, span.start, span.end
    source_step_id, start, end = span
    return int(source_step_id), int(start), int(end)


def _source_contexts_for_boundary_queries(
    attention,
    values,
    *,
    target_indices,
    source_spans: Sequence[Sequence[SourceSpan | tuple[int, int, int]]],
):
    import torch

    if attention.ndim != 4:
        raise ValueError("attention weights must have shape [batch,heads,query,key]")
    if values.ndim != 4:
        raise ValueError("values must have shape [batch,key,heads,head_dim]")
    if attention.shape[0] != 1 or values.shape[0] != 1:
        raise ValueError("component extraction uses batch size one")
    if attention.shape[1] != values.shape[2] or attention.shape[3] != values.shape[1]:
        raise ValueError("attention/value topology mismatch")
    targets = torch.as_tensor(target_indices, device=attention.device, dtype=torch.long)
    if targets.numel() != len(source_spans):
        raise ValueError("target indices and source spans are not aligned")
    heads = int(attention.shape[1])
    head_dim = int(values.shape[3])
    max_blocks = max(len(spans) for spans in source_spans)
    contexts = torch.zeros(
        (len(source_spans), max_blocks, heads, head_dim),
        device=attention.device,
        dtype=torch.float32,
    )
    weights = attention.float()
    expanded_values = values.float()
    for step, spans in enumerate(source_spans):
        target = int(targets[step].item())
        if target < 0 or target >= attention.shape[2]:
            raise ValueError("target index lies outside attention query axis")
        for block, span in enumerate(spans):
            _source_id, start, end = _span_parts(span)
            if not (0 <= start < end <= target + 1):
                raise ValueError("source span lies outside the target causal prefix")
            block_weights = weights[0, :, target, start:end]
            block_values = expanded_values[0, start:end, :, :]
            contexts[step, block] = torch.einsum(
                "hk,khd->hd", block_weights, block_values
            )
    return contexts


def _residual_writes_from_source_contexts(contexts, weight, bias=None):
    import torch

    if contexts.ndim != 4:
        raise ValueError("source contexts must have shape [step,source,head,head_dim]")
    step_count, block_count, heads, head_dim = contexts.shape
    hidden = int(heads * head_dim)
    projection_weight = weight.float()
    if projection_weight.shape != (hidden, hidden):
        raise ValueError("o_proj weight shape disagrees with attention heads")
    if bias is not None:
        bias_value = bias.detach() if hasattr(bias, "detach") else bias
        if torch.as_tensor(bias_value).float().abs().max().item() > 0.0:
            raise ValueError("nonzero o_proj bias cannot be assigned to source blocks")
    flat = contexts.float().reshape(step_count, block_count, hidden)
    return torch.matmul(flat, projection_weight.T)


def _attention_module(block: object) -> object:
    for name in ("self_attn", "attn", "attention"):
        module = getattr(block, name, None)
        if module is not None:
            return module
    raise TypeError("decoder block does not expose a self-attention module")


def _mlp_module(block: object) -> object:
    for name in ("mlp", "feed_forward", "ffn"):
        module = getattr(block, name, None)
        if module is not None:
            return module
    raise TypeError("decoder block does not expose an MLP/feed-forward module")


def _head_topology(model: object, attention_module: object) -> tuple[int, int, int]:
    heads = int(
        getattr(attention_module, "num_heads", 0)
        or getattr(model.config, "num_attention_heads", 0)
    )
    kv_heads = int(
        getattr(attention_module, "num_key_value_heads", 0)
        or getattr(model.config, "num_key_value_heads", 0)
        or heads
    )
    head_dim = int(
        getattr(attention_module, "head_dim", 0)
        or int(getattr(model.config, "hidden_size", 0)) // max(heads, 1)
    )
    if min(heads, kv_heads, head_dim) < 1 or heads % kv_heads != 0:
        raise TypeError("model exposes an invalid grouped-query attention topology")
    return heads, kv_heads, head_dim


def _as_hidden(output):
    import torch

    hidden = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(hidden) or hidden.ndim != 3:
        raise TypeError("captured component must have shape [batch,seq,hidden]")
    return hidden


def _boundary_values(hidden, token_indices):
    import torch

    if hidden.shape[0] != 1:
        raise ValueError("component extraction uses batch size one")
    indices = torch.as_tensor(token_indices, device=hidden.device, dtype=torch.long)
    return hidden[0, indices].detach()


def _attention_weights_from_output(output):
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        return None
    return output[1]


def _without_attention_weights(output):
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        return output
    if isinstance(output, tuple):
        return (output[0], None, *output[2:])
    without_weights = list(output)
    without_weights[1] = None
    return without_weights


def _reshape_value_projection(raw_values, *, kv_heads: int, head_dim: int):
    if raw_values.ndim != 3 or raw_values.shape[-1] != kv_heads * head_dim:
        raise ValueError("captured v_proj output has an unexpected shape")
    return raw_values.reshape(
        raw_values.shape[0], raw_values.shape[1], kv_heads, head_dim
    )


class _ReplayComponentCapture:
    def __init__(
        self,
        *,
        model: object,
        blocks: Sequence[object],
        selected_depths: Sequence[int],
        boundary_indices,
        step_indices,
        source_spans: Sequence[Sequence[SourceSpan | tuple[int, int, int]]],
        config: ComponentExtractionConfig,
    ) -> None:
        self.model = model
        self.blocks = tuple(blocks)
        self.selected_depths = tuple(int(depth) for depth in selected_depths)
        self.boundary_indices = boundary_indices
        self.step_indices = step_indices
        self.source_spans = source_spans
        self.config = config
        self.depth_to_position = {
            depth: position for position, depth in enumerate(self.selected_depths)
        }
        self.value_outputs: dict[int, object] = {}
        self.attention_messages: dict[int, object] = {}
        self.attention_outputs: dict[int, object] = {}
        self.mlp_outputs: dict[int, object] = {}
        self.block_outputs: dict[int, object] = {}
        self.max_attention_reconstruction_error = 0.0
        self.handles: list[object] = []

    def __enter__(self) -> Self:
        for depth, block in enumerate(self.blocks, start=1):
            attention = _attention_module(block)
            selected_position = self.depth_to_position.get(depth)
            if selected_position is not None:
                projection = getattr(attention, "v_proj", None)
                if projection is None:
                    raise TypeError("attention module does not expose v_proj")

                def capture_value(
                    _module, _inputs, output, *, position=selected_position
                ):
                    value = output[0] if isinstance(output, (tuple, list)) else output
                    self.value_outputs[int(position)] = value.detach()

                self.handles.append(projection.register_forward_hook(capture_value))

            def capture_attention(_module, _inputs, output, *, depth_value=depth):
                position = self.depth_to_position.get(int(depth_value))
                if position is not None:
                    self._process_attention(int(position), _module, output)
                return _without_attention_weights(output)

            self.handles.append(attention.register_forward_hook(capture_attention))

            if selected_position is not None:
                mlp = _mlp_module(block)

                def capture_mlp(
                    _module, _inputs, output, *, position=selected_position
                ):
                    hidden = _as_hidden(output)
                    self.mlp_outputs[int(position)] = _boundary_values(
                        hidden, self.step_indices
                    )

                def capture_block(
                    _module, _inputs, output, *, position=selected_position
                ):
                    hidden = _as_hidden(output)
                    self.block_outputs[int(position)] = _boundary_values(
                        hidden, self.boundary_indices
                    )

                self.handles.append(mlp.register_forward_hook(capture_mlp))
                self.handles.append(block.register_forward_hook(capture_block))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.value_outputs.clear()

    def _process_attention(self, position: int, module: object, output: object) -> None:
        import torch

        raw_values = self.value_outputs.pop(position, None)
        attention = _attention_weights_from_output(output)
        if attention is None:
            raise RuntimeError(
                "attention weights were not returned; load the model with "
                "attn_implementation='eager' and call output_attentions=True"
            )
        if raw_values is None:
            raise RuntimeError("v_proj capture did not run before attention hook")
        heads, kv_heads, head_dim = _head_topology(self.model, module)
        values = _reshape_value_projection(
            raw_values,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        expanded_values = _expand_grouped_query_values(values, heads)
        contexts = _source_contexts_for_boundary_queries(
            attention,
            expanded_values,
            target_indices=self.step_indices,
            source_spans=self.source_spans,
        )
        output_projection = getattr(module, "o_proj", None)
        if output_projection is None or not hasattr(output_projection, "weight"):
            raise TypeError("attention module does not expose o_proj.weight")
        messages = _residual_writes_from_source_contexts(
            contexts,
            output_projection.weight,
            getattr(output_projection, "bias", None),
        )
        hidden = _as_hidden(output)
        actual = _boundary_values(hidden, self.step_indices).float()
        reconstructed = messages.sum(dim=1).float()
        absolute = torch.linalg.vector_norm(reconstructed - actual, dim=-1)
        relative = absolute / torch.clamp(
            torch.linalg.vector_norm(actual, dim=-1), min=1e-12
        )
        failed = torch.logical_and(
            absolute > float(self.config.attention_reconstruction_atol),
            relative > float(self.config.attention_reconstruction_rtol),
        )
        if bool(torch.any(failed).item()):
            maximum = float(torch.max(relative).detach().cpu().item())
            raise ValueError(
                "attention source-message reconstruction exceeded tolerance: "
                f"max_relative_error={maximum:.6g}"
            )
        self.max_attention_reconstruction_error = max(
            self.max_attention_reconstruction_error,
            float(torch.max(relative).detach().cpu().item()),
        )
        self.attention_messages[position] = messages.detach()
        self.attention_outputs[position] = actual.detach()

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        missing = [
            depth
            for depth, position in self.depth_to_position.items()
            if position not in self.attention_messages
            or position not in self.attention_outputs
            or position not in self.mlp_outputs
            or position not in self.block_outputs
        ]
        if missing:
            raise RuntimeError(f"component replay missed selected depths {missing}")
        attn_msg = []
        attn_out = []
        mlp_out = []
        resid = []
        for position in range(len(self.selected_depths)):
            attn_msg.append(self.attention_messages[position].float().cpu().numpy())
            attn_out.append(self.attention_outputs[position].float().cpu().numpy())
            mlp_out.append(self.mlp_outputs[position].float().cpu().numpy())
            resid.append(self.block_outputs[position].float().cpu().numpy())
        return (
            np.stack(resid, axis=1),
            np.stack(attn_msg, axis=1),
            np.stack(attn_out, axis=1),
            np.stack(mlp_out, axis=1),
        )


def _attn_implementation(model: object) -> str:
    config = getattr(model, "config", None)
    for name in ("_attn_implementation", "attn_implementation"):
        value = getattr(config, name, None)
        if value is not None:
            return str(value)
    return ""


def _validate_model_for_replay(model: object, layers: tuple[int, ...]) -> None:
    implementation = _attn_implementation(model)
    if implementation and implementation != "eager":
        raise ValueError(
            "component extraction requires eager attention; "
            f"model reports attn_implementation={implementation!r}"
        )
    topology = _resolve_model_topology(model)
    if max(layers) >= len(topology.blocks):
        raise ValueError(
            f"component layer {max(layers)} is not a stored raw block-output depth; "
            f"valid component depths are 1..{len(topology.blocks) - 1}, while "
            f"stored depth {len(topology.blocks)} is the final normalized state"
        )


def _replay_components(
    model: object,
    *,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    boundary_token_end: np.ndarray,
    step_token_end: np.ndarray,
    source_spans: Sequence[Sequence[SourceSpan | tuple[int, int, int]]],
    config: ComponentExtractionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    import torch

    _validate_model_for_replay(model, config.layers)
    topology = _resolve_model_topology(model)
    device = next(model.parameters()).device
    replay_input_ids = torch.as_tensor(
        input_ids[None, :], dtype=torch.long, device=device
    )
    replay_attention_mask = torch.as_tensor(
        attention_mask[None, :], dtype=torch.long, device=device
    )
    boundary_indices = torch.as_tensor(
        boundary_token_end.astype(np.int64) - 1,
        dtype=torch.long,
        device=device,
    )
    step_indices = torch.as_tensor(
        step_token_end.astype(np.int64) - 1,
        dtype=torch.long,
        device=device,
    )
    model.eval()
    with torch.inference_mode():
        with _ReplayComponentCapture(
            model=model,
            blocks=topology.blocks,
            selected_depths=config.layers,
            boundary_indices=boundary_indices,
            step_indices=step_indices,
            source_spans=source_spans,
            config=config,
        ) as capture:
            topology.backbone(
                input_ids=replay_input_ids,
                attention_mask=replay_attention_mask,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
        resid, attn_msg, attn_out, mlp_out = capture.arrays()
    return (
        resid,
        attn_msg,
        attn_out,
        mlp_out,
        float(capture.max_attention_reconstruction_error),
    )
