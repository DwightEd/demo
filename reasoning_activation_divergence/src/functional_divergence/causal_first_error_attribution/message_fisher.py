from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np

from ..hidden_state_geometry.component_replay import (
    _as_hidden,
    _attention_module,
    _attention_weights_from_output,
    _expand_grouped_query_values,
    _head_topology,
    _mlp_module,
    _reshape_value_projection,
    _resolve_model_topology,
)
from .replay import _decision_contexts, _PrefixValueCapture

FFN_DIRECTION_ID = -32768


@dataclass(frozen=True)
class SourceMessageFisherResult:
    """One decision boundary's exact messages and functional Fisher Gram."""

    layers: np.ndarray
    source_ids: np.ndarray
    direction_source_ids: np.ndarray
    source_messages: np.ndarray
    attention_mass: np.ndarray
    attention_output: np.ndarray
    mlp_output: np.ndarray
    residual_pre: np.ndarray
    residual_post: np.ndarray
    fisher_gram: np.ndarray
    euclidean_gram: np.ndarray
    observed_symmetric_kl: np.ndarray
    predicted_quadratic_kl: np.ndarray
    quadratic_relative_error: np.ndarray
    attention_reconstruction_error: np.ndarray
    block_reconstruction_error: np.ndarray
    baseline_entropy: float
    epsilon: float

    def validate(self) -> None:
        layer_count = len(self.layers)
        source_count = len(self.source_ids)
        direction_count = source_count + 1
        if self.direction_source_ids.shape != (direction_count,):
            raise ValueError("direction ids must contain every source and one FFN")
        if (
            not np.array_equal(self.direction_source_ids[:-1], self.source_ids)
            or int(self.direction_source_ids[-1]) != FFN_DIRECTION_ID
        ):
            raise ValueError("direction ids must end with the typed FFN direction")
        if self.source_messages.ndim != 3:
            raise ValueError("source_messages must have shape [layer,source,hidden]")
        hidden = self.source_messages.shape[2]
        expected_vector = (layer_count, hidden)
        if self.source_messages.shape != (layer_count, source_count, hidden):
            raise ValueError("source_messages do not align with layers and sources")
        if (
            self.attention_mass.ndim != 3
            or self.attention_mass.shape[0] != layer_count
            or self.attention_mass.shape[2] != source_count
        ):
            raise ValueError("attention_mass must have shape [layer,head,source]")
        if np.any(self.attention_mass < 0.0) or not np.allclose(
            self.attention_mass.sum(axis=2), 1.0, rtol=1e-5, atol=1e-6
        ):
            raise ValueError("attention_mass must be normalized over source nodes")
        for name in (
            "attention_output",
            "mlp_output",
            "residual_pre",
            "residual_post",
        ):
            if np.asarray(getattr(self, name)).shape != expected_vector:
                raise ValueError(f"{name} must have shape [layer,hidden]")
        if self.fisher_gram.shape != (
            layer_count,
            direction_count,
            direction_count,
        ):
            raise ValueError("fisher_gram must have shape [layer,direction,direction]")
        if self.euclidean_gram.shape != self.fisher_gram.shape:
            raise ValueError("euclidean_gram must align with fisher_gram")
        for name in (
            "observed_symmetric_kl",
            "predicted_quadratic_kl",
            "quadratic_relative_error",
        ):
            if np.asarray(getattr(self, name)).shape != (
                layer_count,
                direction_count,
            ):
                raise ValueError(f"{name} must have shape [layer,direction]")
        for name in (
            "attention_reconstruction_error",
            "block_reconstruction_error",
        ):
            if np.asarray(getattr(self, name)).shape != (layer_count,):
                raise ValueError(f"{name} must have shape [layer]")
        numeric = (
            self.source_messages,
            self.attention_mass,
            self.attention_output,
            self.mlp_output,
            self.residual_pre,
            self.residual_post,
            self.fisher_gram,
            self.euclidean_gram,
            self.observed_symmetric_kl,
            self.predicted_quadratic_kl,
            self.quadratic_relative_error,
            self.attention_reconstruction_error,
            self.block_reconstruction_error,
        )
        if any(not np.isfinite(np.asarray(values)).all() for values in numeric):
            raise ValueError("Fisher result contains non-finite values")
        if not np.isfinite(self.baseline_entropy) or self.baseline_entropy < 0.0:
            raise ValueError("baseline_entropy must be finite and nonnegative")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        for gram in self.fisher_gram:
            fisher_diagnostics(gram, direction_source_ids=self.direction_source_ids)
        for gram in self.euclidean_gram:
            fisher_diagnostics(gram, direction_source_ids=self.direction_source_ids)

    def summary_rows(self) -> list[dict[str, float | int]]:
        self.validate()
        rows = []
        for index, layer in enumerate(self.layers):
            diagnostics = dict(
                fisher_diagnostics(
                    self.fisher_gram[index],
                    direction_source_ids=self.direction_source_ids,
                )
            )
            euclidean = fisher_diagnostics(
                self.euclidean_gram[index],
                direction_source_ids=self.direction_source_ids,
            )
            direction_count = len(self.direction_source_ids)
            diagnostics.update(
                {
                    "layer": int(layer),
                    "source_count": len(self.source_ids),
                    "baseline_entropy": float(self.baseline_entropy),
                    "median_quadratic_relative_error": float(
                        np.median(self.quadratic_relative_error[index])
                    ),
                    "max_attention_reconstruction_error": float(
                        self.attention_reconstruction_error[index]
                    ),
                    "max_block_reconstruction_error": float(
                        self.block_reconstruction_error[index]
                    ),
                    "mean_fisher_energy": float(
                        diagnostics["fisher_trace"] / direction_count
                    ),
                    "euclidean_largest_eigenvalue": float(
                        euclidean["largest_eigenvalue"]
                    ),
                    "euclidean_trace": float(euclidean["fisher_trace"]),
                    "mean_euclidean_energy": float(
                        euclidean["fisher_trace"] / direction_count
                    ),
                    "fisher_to_euclidean_trace_ratio": float(
                        diagnostics["fisher_trace"]
                        / max(float(euclidean["fisher_trace"]), 1e-12)
                    ),
                }
            )
            rows.append(diagnostics)
        return rows


def source_binned_residual_messages(
    attention,
    expanded_values,
    output_weight,
    *,
    source_step_ids: np.ndarray,
    query_position: int,
):
    """Aggregate exact decision-token attention writes by source step.

    Token contexts are grouped before the output projection, avoiding the
    prohibitively large dense ``[head, source_token, hidden]`` tensor.
    """
    source_ids, messages, _source_mass, reconstructed = (
        source_binned_attention_components(
            attention,
            expanded_values,
            output_weight,
            source_step_ids=source_step_ids,
            query_position=query_position,
        )
    )
    return source_ids, messages, reconstructed


def source_binned_attention_components(
    attention,
    expanded_values,
    output_weight,
    *,
    source_step_ids: np.ndarray,
    query_position: int,
):
    """Return residual writes and normalized head-wise mass per source node."""
    import torch

    contexts, mass, head_projection = _decision_contexts(
        attention,
        expanded_values,
        output_weight,
        query_position=query_position,
    )
    source_ids = np.asarray(source_step_ids)
    if source_ids.ndim != 1 or source_ids.shape[0] != contexts.shape[1]:
        raise ValueError("source_step_ids must align with the attention key axis")
    if not np.issubdtype(source_ids.dtype, np.integer):
        raise ValueError("source_step_ids must be integers")

    unique_ids = np.unique(source_ids.astype(np.int64, copy=False))
    messages = []
    source_mass = []
    for source_id in unique_ids:
        mask = torch.as_tensor(
            source_ids == source_id, device=contexts.device, dtype=torch.bool
        )
        grouped_context = contexts[:, mask].sum(dim=1)
        source_mass.append(mass[:, mask].sum(dim=1))
        messages.append(torch.einsum("hd,ohd->o", grouped_context, head_projection))
    stacked = torch.stack(messages, dim=0)
    stacked_mass = torch.stack(source_mass, dim=1)
    return unique_ids, stacked, stacked_mass, stacked.sum(dim=0)


def _backbone_hidden(output):
    hidden = getattr(output, "last_hidden_state", None)
    return _as_hidden(output) if hidden is None else hidden


def _replace_output_with_delta(output, delta):
    hidden = _as_hidden(output).clone()
    if delta.shape != (hidden.shape[0], hidden.shape[2]):
        raise ValueError("patch delta must have shape [batch,hidden]")
    hidden[:, -1] = hidden[:, -1] + delta.to(device=hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


class _MessageCapture:
    def __init__(
        self,
        *,
        model: object,
        blocks,
        layers: tuple[int, ...],
        source_step_ids: np.ndarray,
        prefix_values: dict[int, object] | None = None,
    ) -> None:
        self.model = model
        self.blocks = blocks
        self.layers = layers
        self.source_step_ids = np.asarray(source_step_ids, dtype=np.int64)
        self.prefix_values = prefix_values or {}
        self.pre: dict[int, object] = {}
        self.raw_values: dict[int, object] = {}
        self.messages: dict[int, object] = {}
        self.mass: dict[int, object] = {}
        self.attention: dict[int, object] = {}
        self.mlp: dict[int, object] = {}
        self.post: dict[int, object] = {}
        self.source_ids: np.ndarray | None = None
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
                self.pre[index] = inputs[0][0, -1].detach()

            def capture_values(_module, _inputs, output, *, index=position):
                values = output[0] if isinstance(output, (tuple, list)) else output
                self.raw_values[index] = values.detach()

            def capture_attention(_module, _inputs, output, *, index=position):
                self._capture_attention(index, _module, output)

            def capture_mlp(_module, _inputs, output, *, index=position):
                self.mlp[index] = _as_hidden(output)[0, -1].detach()

            def capture_post(_module, _inputs, output, *, index=position):
                self.post[index] = _as_hidden(output)[0, -1].detach()

            self.handles.extend(
                (
                    block.register_forward_pre_hook(capture_pre),
                    value_projection.register_forward_hook(capture_values),
                    attention.register_forward_hook(capture_attention),
                    mlp.register_forward_hook(capture_mlp),
                    block.register_forward_hook(capture_post),
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _capture_attention(self, position: int, module: object, output: object) -> None:
        attention = _attention_weights_from_output(output)
        if attention is None:
            raise RuntimeError(
                "attention weights are unavailable; use eager attention and "
                "output_attentions=True"
            )
        raw_values = self.raw_values.pop(position, None)
        if raw_values is None:
            raise RuntimeError("v_proj capture did not run before attention output")
        previous = self.prefix_values.get(position)
        if previous is not None:
            import torch

            raw_values = torch.cat((previous.to(raw_values.device), raw_values), dim=1)
        heads, kv_heads, head_dim = _head_topology(self.model, module)
        values = _reshape_value_projection(
            raw_values, kv_heads=kv_heads, head_dim=head_dim
        )
        expanded = _expand_grouped_query_values(values, heads)
        output_projection = getattr(module, "o_proj", None)
        if output_projection is None or not hasattr(output_projection, "weight"):
            raise TypeError("attention module does not expose o_proj.weight")
        source_ids, messages, source_mass, reconstructed = (
            source_binned_attention_components(
                attention,
                expanded,
                output_projection.weight,
                source_step_ids=self.source_step_ids,
                query_position=int(attention.shape[2]) - 1,
            )
        )
        if self.source_ids is None:
            self.source_ids = source_ids
        elif not np.array_equal(self.source_ids, source_ids):
            raise RuntimeError("source grouping changed across selected layers")
        self.messages[position] = messages.detach()
        self.mass[position] = source_mass.detach()
        self.attention[position] = _as_hidden(output)[0, -1].detach()
        if reconstructed.shape != self.attention[position].shape:
            raise RuntimeError("reconstructed attention has the wrong hidden shape")

    def arrays(self) -> tuple[np.ndarray, ...]:
        stores = (
            self.pre,
            self.messages,
            self.mass,
            self.attention,
            self.mlp,
            self.post,
        )
        expected = set(range(len(self.layers)))
        if self.source_ids is None or any(set(store) != expected for store in stores):
            raise RuntimeError("message replay did not capture every selected layer")

        def stack(store: dict[int, object]) -> np.ndarray:
            return np.stack(
                [
                    store[index].float().cpu().numpy()
                    for index in range(len(self.layers))
                ]
            )

        return (
            self.source_ids.copy(),
            stack(self.messages),
            stack(self.mass),
            stack(self.attention),
            stack(self.mlp),
            stack(self.pre),
            stack(self.post),
        )


def categorical_fisher_gram(
    probabilities: np.ndarray, logit_jacobian: np.ndarray
) -> np.ndarray:
    """Project the categorical Fisher metric onto intervention directions."""
    probability = np.asarray(probabilities, dtype=np.float64)
    jacobian = np.asarray(logit_jacobian, dtype=np.float64)
    if probability.ndim != 1 or probability.size < 2:
        raise ValueError("probabilities must be a vocabulary vector")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.isclose(probability.sum(), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("probabilities must sum to one")
    if jacobian.ndim != 2 or jacobian.shape[1] != probability.size:
        raise ValueError("logit_jacobian must have shape [direction,vocabulary]")
    if jacobian.shape[0] < 1 or not np.isfinite(jacobian).all():
        raise ValueError("logit_jacobian must contain finite directions")

    centered = jacobian - (jacobian @ probability)[:, None]
    gram = (centered * probability[None, :]) @ centered.T
    return 0.5 * (gram + gram.T)


def fisher_diagnostics(
    gram: np.ndarray, *, direction_source_ids: np.ndarray
) -> Mapping[str, float | int]:
    """Return coordinate-free spectrum and typed top-direction loadings."""
    matrix = np.asarray(gram, dtype=np.float64)
    source_ids = np.asarray(direction_source_ids)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be a square matrix")
    if source_ids.shape != (matrix.shape[0],):
        raise ValueError("direction_source_ids must align with gram")
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix, matrix.T, rtol=1e-7, atol=1e-9
    ):
        raise ValueError("gram must be finite and symmetric")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    tolerance = max(float(np.max(np.abs(eigenvalues))), 1.0) * 1e-7
    if float(eigenvalues.min(initial=0.0)) < -tolerance:
        raise ValueError("gram must be positive semidefinite")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    positive = eigenvalues[eigenvalues > tolerance]
    if total > 0.0:
        normalized = eigenvalues[eigenvalues > 0.0] / total
        effective_rank = float(np.exp(-np.sum(normalized * np.log(normalized))))
        anisotropy = largest / total
        top_loading = np.square(eigenvectors[:, -1])
    else:
        effective_rank = 0.0
        anisotropy = 0.0
        top_loading = np.zeros(matrix.shape[0], dtype=np.float64)
    condition_number = float(largest / positive[0]) if positive.size > 0 else 0.0
    return {
        "largest_eigenvalue": largest,
        "fisher_trace": total,
        "effective_rank": effective_rank,
        "anisotropy": float(anisotropy),
        "condition_number": condition_number,
        "numerical_rank": int(positive.size),
        "top_prompt_loading": float(top_loading[source_ids == -1].sum()),
        "top_ffn_loading": float(top_loading[source_ids == FFN_DIRECTION_ID].sum()),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    exponential = np.exp(shifted)
    return exponential / exponential.sum()


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    maximum = np.max(values)
    return values - (maximum + np.log(np.exp(values - maximum).sum()))


class SourceMessageFisherRunner:
    """Measure output Fisher geometry of attention-source and FFN updates."""

    def __init__(
        self,
        *,
        layers: tuple[int, ...],
        epsilon: float = 0.05,
        perturbation_batch_size: int = 4,
    ) -> None:
        self.layers = tuple(int(layer) for layer in layers)
        self.epsilon = float(epsilon)
        self.perturbation_batch_size = int(perturbation_batch_size)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must be a non-empty unique sequence")
        if min(self.layers) < 1:
            raise ValueError("layers must use positive one-based indices")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if self.perturbation_batch_size < 1:
            raise ValueError("perturbation_batch_size must be positive")

    def run(
        self,
        *,
        model: object,
        input_ids: np.ndarray,
        source_step_ids: np.ndarray,
    ) -> SourceMessageFisherResult:
        import torch

        tokens = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        source_steps = np.asarray(source_step_ids, dtype=np.int64).reshape(-1)
        if tokens.size < 1 or source_steps.shape != tokens.shape:
            raise ValueError(
                "input_ids and source_step_ids must be non-empty and aligned"
            )
        topology = _resolve_model_topology(model)
        if max(self.layers) > len(topology.blocks):
            raise ValueError("selected layer exceeds the model block count")
        implementation = str(
            getattr(getattr(model, "config", None), "_attn_implementation", "")
        )
        if implementation and implementation != "eager":
            raise ValueError("source-message replay requires eager attention")
        device = next(model.parameters()).device
        token_tensor = torch.as_tensor(tokens[None, :], device=device, dtype=torch.long)
        model.eval()
        (
            source_ids,
            source_messages,
            attention_mass,
            attention_output,
            mlp_output,
            residual_pre,
            residual_post,
        ) = self._capture_messages(
            model=model,
            topology=topology,
            token_tensor=token_tensor,
            source_step_ids=source_steps,
        )
        baseline_logits = self._plain_logits(
            model=model, topology=topology, token_tensor=token_tensor
        )
        probability = _softmax(baseline_logits)
        baseline_entropy = float(
            -np.sum(probability * np.log(np.clip(probability, 1e-300, None)))
        )

        direction_ids = np.concatenate(
            (source_ids.astype(np.int64), np.asarray([FFN_DIRECTION_ID]))
        )
        gram_rows = []
        euclidean_rows = []
        observed_rows = []
        predicted_rows = []
        relative_rows = []
        for layer_index, layer in enumerate(self.layers):
            directions = np.concatenate(
                (source_messages[layer_index], mlp_output[layer_index][None, :]), axis=0
            )
            euclidean_rows.append(directions @ directions.T)
            plus, minus = self._directional_logits(
                model=model,
                topology=topology,
                token_tensor=token_tensor,
                layer=layer,
                directions=directions,
                direction_ids=direction_ids,
            )
            jacobian = (plus - minus) / (2.0 * self.epsilon)
            gram = categorical_fisher_gram(probability, jacobian)
            observed = self._symmetric_kl(baseline_logits, plus, minus)
            predicted = 0.5 * self.epsilon**2 * np.diag(gram)
            relative = np.abs(observed - predicted) / np.maximum.reduce(
                (observed, predicted, np.full_like(observed, 1e-12))
            )
            gram_rows.append(gram)
            observed_rows.append(observed)
            predicted_rows.append(predicted)
            relative_rows.append(relative)

        attention_reconstruction = np.linalg.norm(
            source_messages.sum(axis=1) - attention_output, axis=1
        ) / np.maximum(np.linalg.norm(attention_output, axis=1), 1e-12)
        block_reconstruction = np.linalg.norm(
            residual_pre + attention_output + mlp_output - residual_post, axis=1
        ) / np.maximum(np.linalg.norm(residual_post, axis=1), 1e-12)
        result = SourceMessageFisherResult(
            layers=np.asarray(self.layers, dtype=np.int16),
            source_ids=source_ids.astype(np.int32),
            direction_source_ids=direction_ids.astype(np.int32),
            source_messages=source_messages.astype(np.float32),
            attention_mass=attention_mass.astype(np.float32),
            attention_output=attention_output.astype(np.float32),
            mlp_output=mlp_output.astype(np.float32),
            residual_pre=residual_pre.astype(np.float32),
            residual_post=residual_post.astype(np.float32),
            fisher_gram=np.asarray(gram_rows, dtype=np.float64),
            euclidean_gram=np.asarray(euclidean_rows, dtype=np.float64),
            observed_symmetric_kl=np.asarray(observed_rows, dtype=np.float64),
            predicted_quadratic_kl=np.asarray(predicted_rows, dtype=np.float64),
            quadratic_relative_error=np.asarray(relative_rows, dtype=np.float64),
            attention_reconstruction_error=attention_reconstruction.astype(np.float64),
            block_reconstruction_error=block_reconstruction.astype(np.float64),
            baseline_entropy=baseline_entropy,
            epsilon=self.epsilon,
        )
        result.validate()
        return result

    def _capture_messages(
        self, *, model: object, topology, token_tensor, source_step_ids: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        import torch

        attention_mask = torch.ones_like(token_tensor)
        past_key_values = None
        prefix_values: dict[int, object] = {}
        with torch.inference_mode():
            if token_tensor.shape[1] > 1:
                with _PrefixValueCapture(topology.blocks, self.layers) as capture:
                    prefix_result = topology.backbone(
                        input_ids=token_tensor[:, :-1],
                        attention_mask=attention_mask[:, :-1],
                        use_cache=True,
                        output_attentions=False,
                        return_dict=True,
                    )
                past_key_values = getattr(prefix_result, "past_key_values", None)
                if past_key_values is not None:
                    prefix_values = capture.values
            cached = past_key_values is not None
            replay_tokens = token_tensor[:, -1:] if cached else token_tensor
            arguments = {
                "input_ids": replay_tokens,
                "attention_mask": attention_mask,
                "use_cache": cached,
                "output_attentions": True,
                "return_dict": True,
            }
            if cached:
                arguments["past_key_values"] = past_key_values
                arguments["position_ids"] = torch.full_like(
                    replay_tokens, token_tensor.shape[1] - 1
                )
            with _MessageCapture(
                model=model,
                blocks=topology.blocks,
                layers=self.layers,
                source_step_ids=source_step_ids,
                prefix_values=prefix_values,
            ) as capture:
                topology.backbone(**arguments)
        return capture.arrays()

    @staticmethod
    def _plain_logits(*, model: object, topology, token_tensor) -> np.ndarray:
        import torch

        with torch.inference_mode():
            output = topology.backbone(
                input_ids=token_tensor,
                attention_mask=torch.ones_like(token_tensor),
                use_cache=False,
                output_attentions=False,
                return_dict=True,
            )
            logits = model.get_output_embeddings()(_backbone_hidden(output)[:, -1])
        return logits[0].float().cpu().numpy().astype(np.float64)

    def _directional_logits(
        self,
        *,
        model: object,
        topology,
        token_tensor,
        layer: int,
        directions: np.ndarray,
        direction_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        block = topology.blocks[int(layer) - 1]
        attention_module = _attention_module(block)
        mlp_module = _mlp_module(block)
        plus_rows = []
        minus_rows = []
        direction_tensor = torch.as_tensor(
            directions, device=token_tensor.device, dtype=next(model.parameters()).dtype
        )
        source_mask = torch.as_tensor(
            direction_ids != FFN_DIRECTION_ID,
            device=token_tensor.device,
            dtype=torch.bool,
        )
        for start in range(0, len(directions), self.perturbation_batch_size):
            stop = min(start + self.perturbation_batch_size, len(directions))
            chunk = direction_tensor[start:stop]
            chunk_source = source_mask[start:stop]
            signed = torch.cat((chunk, -chunk), dim=0) * self.epsilon
            signed_source = torch.cat((chunk_source, chunk_source), dim=0)
            attention_delta = torch.where(signed_source[:, None], signed, 0.0)
            mlp_delta = torch.where(signed_source[:, None], 0.0, signed)

            def patch_attention(_module, _inputs, output, *, delta=attention_delta):
                return _replace_output_with_delta(output, delta)

            def patch_mlp(_module, _inputs, output, *, delta=mlp_delta):
                return _replace_output_with_delta(output, delta)

            handles = (
                attention_module.register_forward_hook(patch_attention),
                mlp_module.register_forward_hook(patch_mlp),
            )
            repeated = token_tensor.expand(2 * len(chunk), -1)
            try:
                with torch.inference_mode():
                    output = topology.backbone(
                        input_ids=repeated,
                        attention_mask=torch.ones_like(repeated),
                        use_cache=False,
                        output_attentions=False,
                        return_dict=True,
                    )
                    logits = model.get_output_embeddings()(
                        _backbone_hidden(output)[:, -1]
                    ).float()
            finally:
                for handle in handles:
                    handle.remove()
            size = len(chunk)
            plus_rows.append(logits[:size].cpu().numpy())
            minus_rows.append(logits[size:].cpu().numpy())
        return (
            np.concatenate(plus_rows, axis=0).astype(np.float64),
            np.concatenate(minus_rows, axis=0).astype(np.float64),
        )

    @staticmethod
    def _symmetric_kl(
        baseline_logits: np.ndarray, plus: np.ndarray, minus: np.ndarray
    ) -> np.ndarray:
        probability = _softmax(baseline_logits)
        baseline_log_probability = _log_softmax(baseline_logits)
        observed = []
        for positive, negative in zip(plus, minus):
            positive_kl = np.sum(
                probability * (baseline_log_probability - _log_softmax(positive))
            )
            negative_kl = np.sum(
                probability * (baseline_log_probability - _log_softmax(negative))
            )
            observed.append(max(0.0, 0.5 * float(positive_kl + negative_kl)))
        return np.asarray(observed, dtype=np.float64)


__all__ = [
    "FFN_DIRECTION_ID",
    "SourceMessageFisherResult",
    "SourceMessageFisherRunner",
    "categorical_fisher_gram",
    "fisher_diagnostics",
    "source_binned_attention_components",
    "source_binned_residual_messages",
]
