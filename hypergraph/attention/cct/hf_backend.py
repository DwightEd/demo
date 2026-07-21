from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import InterventionEffect, TransportInputs
from .pipeline import AssemblyInputs, TraceIdentity
from .processbench import (
    PlainReasoningRenderer,
    ProcessBenchRecord,
    TokenizerAligner,
)


@dataclass(frozen=True)
class HuggingFaceExtractionConfig:
    layer_id: int
    top_sources: int = 3
    node_dim: int = 64
    projection_seed: int = 17
    device: str = "cuda"
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"

    def validate(self) -> None:
        if self.layer_id < 0 or self.top_sources <= 0 or self.node_dim <= 0:
            raise ValueError("layer_id, top_sources, and node_dim are invalid")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if self.attention_implementation not in {"sdpa", "eager"}:
            raise ValueError("attention_implementation must be sdpa or eager")


class HuggingFaceTransportBackend:
    """Llama-family adapter for compact output-effective transport traces."""

    def __init__(
        self, model, tokenizer, *, model_name: str, config: HuggingFaceExtractionConfig
    ) -> None:
        config.validate()
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.config = config
        self.renderer = PlainReasoningRenderer()
        self.aligner = TokenizerAligner()
        self.layers = self._layers(model)
        self._projection = None
        if config.layer_id >= len(self.layers):
            raise ValueError(
                f"layer {config.layer_id} does not exist in a {len(self.layers)}-layer model"
            )

    @classmethod
    def from_pretrained(
        cls, model_name: str, config: HuggingFaceExtractionConfig
    ) -> "HuggingFaceTransportBackend":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, config.dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation=config.attention_implementation,
            low_cpu_mem_usage=True,
        ).to(config.device)
        model.eval()
        return cls(model, tokenizer, model_name=model_name, config=config)

    def extract(self, record: ProcessBenchRecord) -> AssemblyInputs:
        import torch

        rendered = self.renderer.render(record)
        tokenized = self.aligner.tokenize(self.tokenizer, rendered)
        input_ids = torch.as_tensor(
            tokenized.input_ids, dtype=torch.long, device=self.config.device
        )[None, :]
        model_limit = int(getattr(self.model.config, "max_position_embeddings", 0))
        if model_limit and input_ids.shape[1] > model_limit:
            raise ValueError(
                f"trace has {input_ids.shape[1]} tokens but the model supports {model_limit}; "
                "truncation is forbidden"
            )

        self.model.zero_grad(set_to_none=True)
        block = self.layers[self.config.layer_id]
        captured: dict[str, object] = {}

        def capture_layer_input(_module, arguments):
            captured["layer_input"] = arguments[0]

        def capture_attention_output(_module, _arguments, output):
            captured["attention_output"] = (
                output[0] if isinstance(output, tuple) else output
            )

        def capture_layer_output(_module, _arguments, output):
            captured["layer_output"] = (
                output[0] if isinstance(output, tuple) else output
            )

        handles = (
            block.register_forward_pre_hook(capture_layer_input),
            block.self_attn.register_forward_hook(capture_attention_output),
            block.register_forward_hook(capture_layer_output),
        )
        with torch.enable_grad():
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
            finally:
                for handle in handles:
                    handle.remove()
            attention_output = captured.get("attention_output")
            layer_input = captured.get("layer_input")
            layer_output = captured.get("layer_output")
            if attention_output is None or layer_input is None or layer_output is None:
                raise RuntimeError("failed to capture the selected transformer block")
            target_positions, query_positions = self._risk_tokens(
                outputs.logits[0], input_ids[0], tokenized.step_ranges
            )
            output_directions = self._output_directions(
                outputs.logits[0],
                attention_output,
                input_ids[0],
                target_positions,
                query_positions,
            )
            with torch.no_grad():
                residual_updates = attention_output[0, query_positions].float()
                basis = self._analysis_basis(output_directions, residual_updates)
                projected_directions = output_directions @ basis
                projected_updates = residual_updates @ basis
                attention = self._selected_attention(
                    layer_input.detach(), query_positions
                )
                values, output_weights = self._value_and_output_weights(
                    layer_input.detach()
                )
                projected_ov = torch.einsum(
                    "hsk,hkr->hsr",
                    values,
                    torch.einsum("dhk,dr->hkr", output_weights, basis),
                )
                source_writes = torch.einsum("hqs,hsr->qsr", attention, projected_ov)
                content_effect = torch.einsum(
                    "hsr,qr->hqs", projected_ov, projected_directions
                )
                baseline_log_probabilities = self._observed_log_probabilities(
                    outputs.logits[0], input_ids[0], target_positions, query_positions
                )
                node_features = (
                    torch.nn.functional.normalize(
                        layer_output[0].float() @ self._node_projection(layer_output),
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )

        transport = TransportInputs(
            attention=attention.detach().cpu().numpy(),
            content_effect=content_effect.detach().cpu().numpy(),
            source_writes=source_writes.detach().cpu().numpy(),
            output_directions=projected_directions.detach().cpu().numpy(),
            residual_updates=projected_updates.detach().cpu().numpy(),
            prompt_end=tokenized.prompt_end,
            receiver_positions=np.asarray(query_positions, dtype=np.int64),
        )
        intervention_attention = attention.detach()
        intervention_content = content_effect.detach()
        intervention_values = values.detach()
        intervention_output = output_weights.detach()
        del outputs, attention_output, layer_input, layer_output
        interventions = self._interventions(
            input_ids=input_ids,
            query_positions=query_positions,
            target_positions=target_positions,
            baseline_log_probabilities=baseline_log_probabilities,
            attention=intervention_attention,
            content_effect=intervention_content,
            values=intervention_values,
            output_weights=intervention_output,
        )
        response_tokens = int(
            np.sum(tokenized.step_ranges[:, 1] - tokenized.step_ranges[:, 0])
        )
        return AssemblyInputs(
            identity=TraceIdentity(
                trace_id=record.trace_id,
                problem_id=record.problem_id,
                generator_model=record.generator_model,
                observer_model=self.model_name,
                layer_id=self.config.layer_id,
                response_tokens=response_tokens,
            ),
            node_features=node_features,
            transport=transport,
            interventions=interventions,
            labels=record.labels,
        )

    @staticmethod
    def _layers(model):
        core = getattr(model, "model", None)
        layers = getattr(core, "layers", None)
        if layers is None:
            raise TypeError(
                "CCT extraction currently supports models exposing model.layers"
            )
        return layers

    @staticmethod
    def _risk_tokens(
        logits, input_ids, step_ranges: np.ndarray
    ) -> tuple[list[int], list[int]]:
        import torch

        targets: list[int] = []
        queries: list[int] = []
        log_probabilities = torch.log_softmax(logits[:-1].float(), dim=-1)
        for start, stop in step_ranges:
            candidates = torch.arange(
                max(int(start), 1), int(stop), device=logits.device
            )
            if not len(candidates):
                raise ValueError("a reasoning step has no predictable token")
            observed = log_probabilities[candidates - 1, input_ids[candidates]]
            target = int(candidates[torch.argmin(observed)])
            targets.append(target)
            queries.append(target - 1)
        if any(right <= left for left, right in zip(queries, queries[1:])):
            raise ValueError("risk-token queries are not strictly ordered")
        return targets, queries

    @staticmethod
    def _output_directions(logits, layer_output, input_ids, targets, queries):
        import torch

        directions = []
        for index, (target, query) in enumerate(zip(targets, queries)):
            log_probability = torch.log_softmax(logits[query].float(), dim=-1)[
                input_ids[target]
            ]
            gradient = torch.autograd.grad(
                log_probability,
                layer_output,
                retain_graph=index + 1 < len(targets),
                create_graph=False,
            )[0]
            directions.append(gradient[0, query].float())
        return torch.stack(directions)

    @staticmethod
    def _analysis_basis(output_directions, residual_updates):
        import torch

        coordinates = torch.cat((output_directions, residual_updates), dim=0).float()
        _, singular_values, right = torch.linalg.svd(coordinates, full_matrices=False)
        tolerance = (
            torch.finfo(coordinates.dtype).eps
            * max(coordinates.shape)
            * singular_values[0]
        )
        rank = max(int((singular_values > tolerance).sum()), 1)
        return right[:rank].T.contiguous()

    def _node_projection(self, layer_output):
        import torch

        expected = (layer_output.shape[-1], self.config.node_dim)
        if self._projection is None:
            generator = torch.Generator(device=layer_output.device)
            generator.manual_seed(self.config.projection_seed)
            self._projection = torch.randn(
                *expected,
                generator=generator,
                device=layer_output.device,
                dtype=torch.float32,
            ) / np.sqrt(self.config.node_dim)
        if tuple(self._projection.shape) != expected:
            raise ValueError("cached node projection does not match the model width")
        return self._projection

    def _selected_attention(self, layer_input, query_positions):
        import torch

        attention = self.layers[self.config.layer_id].self_attn
        normalized = self.layers[self.config.layer_id].input_layernorm(layer_input)
        batch, sequence, _ = normalized.shape
        heads = int(attention.config.num_attention_heads)
        key_value_heads = int(attention.config.num_key_value_heads)
        head_dim = int(attention.head_dim)
        queries = attention.q_proj(normalized).view(batch, sequence, heads, head_dim)
        keys = attention.k_proj(normalized).view(
            batch, sequence, key_value_heads, head_dim
        )
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        if hasattr(attention, "q_norm"):
            queries = attention.q_norm(queries)
        if hasattr(attention, "k_norm"):
            keys = attention.k_norm(keys)
        position_ids = torch.arange(sequence, device=normalized.device)[None, :]
        rotary = getattr(attention, "rotary_emb", None)
        if rotary is None:
            rotary = getattr(getattr(self.model, "model", None), "rotary_emb", None)
        if rotary is None:
            raise TypeError("selected attention reconstruction requires rotary_emb")
        cos, sin = rotary(normalized, position_ids)
        if cos.ndim == 2:
            cos, sin = cos[None, None], sin[None, None]
        elif cos.ndim == 3:
            cos, sin = cos[:, None], sin[:, None]
        queries = queries * cos + self._rotate_half(queries) * sin
        keys = keys * cos + self._rotate_half(keys) * sin
        keys = keys.repeat_interleave(heads // key_value_heads, dim=1)
        selected = queries[0, :, query_positions].float()
        keys = keys[0].float()
        return self._causal_attention_from_qk(selected, keys, query_positions)

    @staticmethod
    def _rotate_half(states):
        import torch

        left, right = states.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    @staticmethod
    def _causal_attention_from_qk(queries, keys, query_positions):
        import torch

        logits = torch.einsum("hqk,hsk->hqs", queries, keys) / np.sqrt(
            queries.shape[-1]
        )
        sources = torch.arange(keys.shape[1], device=keys.device)
        positions = torch.as_tensor(query_positions, device=keys.device)
        logits = logits.masked_fill(
            sources[None, None, :] > positions[None, :, None], -torch.inf
        )
        return torch.softmax(logits, dim=-1)

    def _value_and_output_weights(self, layer_input):
        import torch

        block = self.layers[self.config.layer_id]
        attention = block.self_attn
        normalized = block.input_layernorm(layer_input)
        values = attention.v_proj(normalized)[0]
        heads = int(attention.config.num_attention_heads)
        key_value_heads = int(attention.config.num_key_value_heads)
        head_dim = int(attention.head_dim)
        values = values.reshape(values.shape[0], key_value_heads, head_dim).transpose(
            0, 1
        )
        values = values.repeat_interleave(heads // key_value_heads, dim=0).float()
        output_weights = attention.o_proj.weight.float().reshape(-1, heads, head_dim)
        return values, output_weights

    @staticmethod
    def _observed_log_probabilities(logits, input_ids, targets, queries):
        import torch

        return torch.stack(
            [
                torch.log_softmax(logits[query].float(), dim=-1)[input_ids[target]]
                for target, query in zip(targets, queries)
            ]
        ).detach()

    def _interventions(
        self,
        *,
        input_ids,
        query_positions,
        target_positions,
        baseline_log_probabilities,
        attention,
        content_effect,
        values,
        output_weights,
    ) -> tuple[InterventionEffect, ...]:
        import torch

        effects: list[InterventionEffect] = []
        block = self.layers[self.config.layer_id]
        for query_index, (query, target) in enumerate(
            zip(query_positions, target_positions)
        ):
            score = (
                (attention[:, query_index] * content_effect[:, query_index])
                .sum(0)
                .abs()
            )
            score[query:] = -1.0
            candidate_count = min(self.config.top_sources, query)
            if candidate_count <= 0:
                continue
            sources = torch.topk(score, candidate_count).indices
            deltas = torch.stack(
                [
                    torch.einsum(
                        "h,hk,dhk->d",
                        attention[:, query_index, source],
                        values[:, source],
                        output_weights,
                    )
                    for source in sources
                ]
            )
            variants = torch.cat((deltas, deltas.sum(0, keepdim=True)), dim=0)

            def subtract_transport(_module, _arguments, output):
                hidden = output[0] if isinstance(output, tuple) else output
                changed = hidden.clone()
                changed[:, query] = changed[:, query] - variants.to(changed.dtype)
                if isinstance(output, tuple):
                    return (changed, *output[1:])
                return changed

            handle = block.self_attn.register_forward_hook(subtract_transport)
            try:
                batch_ids = input_ids.repeat(len(variants), 1)
                with torch.no_grad():
                    ablated = self.model(
                        input_ids=batch_ids,
                        attention_mask=torch.ones_like(batch_ids),
                        use_cache=False,
                        return_dict=True,
                    ).logits
                log_probabilities = torch.log_softmax(
                    ablated[:, query].float(), dim=-1
                )[:, input_ids[0, target]]
            finally:
                handle.remove()
            changes = baseline_log_probabilities[query_index] - log_probabilities
            effects.append(
                InterventionEffect(
                    query_index=query_index,
                    sources=tuple(int(source) for source in sources.cpu()),
                    singleton_effects=changes[:-1].cpu().numpy(),
                    joint_effect=float(changes[-1].cpu()),
                )
            )
        return tuple(effects)
