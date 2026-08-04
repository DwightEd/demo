from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from prompt_control_flow.belief_transport.extraction import length_bucket_batches
from prompt_control_flow.belief_transport.model_capture import resolve_model_topology

from .charts import LayerChartBundle
from .routing_extraction import (
    _attention_module,
    _last_visible_indices,
    _opposite_updates,
    _pad_saved_rows,
    _project_total_write,
)
from .schema import CausalBeliefTrace
from .update_metrics import component_reconstruction_error, component_update_metrics
from .update_schema import BELIEF_UPDATE_SCHEMA, BeliefUpdateTrace


@dataclass(frozen=True)
class BeliefUpdateExtractionConfig:
    batch_size: int = 8
    max_batch_tokens: int = 4096
    max_seq_len: int = 1024
    show_progress: bool = True
    allow_failed_representation_gate: bool = False

    def validate(self) -> None:
        if min(self.batch_size, self.max_batch_tokens, self.max_seq_len) < 1:
            raise ValueError("belief-update batch and sequence limits must be positive")


def representation_gate_failure_message(
    charts: LayerChartBundle,
) -> str | None:
    gate = charts.metadata.get("decision_gate", {})
    if bool(gate.get("ready_for_routing_analysis", False)):
        return None
    conditions = gate.get("conditions", {})
    failed_conditions = sorted(
        str(name) for name, passed in conditions.items() if not bool(passed)
    )
    if failed_conditions:
        return (
            "representation gate did not authorize belief-update decomposition; "
            f"failed conditions: {', '.join(failed_conditions)}"
        )
    return (
        "representation gate did not authorize belief-update decomposition; "
        "the chart contains no failed-condition details"
    )


def _mlp_module(block):
    for name in ("mlp", "feed_forward", "ffn"):
        module = getattr(block, name, None)
        if module is not None:
            return module
    raise TypeError("decoder block does not expose an MLP/feed-forward module")


class _BlockComponentCapture:
    """Capture only the target-token writes needed for block decomposition."""

    def __init__(self, blocks: Sequence[object], target_indices) -> None:
        self.blocks = list(blocks)
        self.target_indices = target_indices
        self.block_inputs: dict[int, object] = {}
        self.attention_outputs: dict[int, object] = {}
        self.mlp_outputs: dict[int, object] = {}
        self.block_outputs: dict[int, object] = {}
        self.handles: list[object] = []

    def _boundary(self, output):
        import torch

        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise TypeError("captured component must have shape [batch, seq, hidden]")
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_index, self.target_indices].detach()

    def __enter__(self) -> "_BlockComponentCapture":
        for position, block in enumerate(self.blocks):
            attention = _attention_module(block)
            mlp = _mlp_module(block)

            def capture_input(_module, inputs, *, index=position):
                if not inputs:
                    raise RuntimeError("decoder block pre-hook received no residual input")
                self.block_inputs[int(index)] = self._boundary(inputs[0])

            def capture_attention(_module, _inputs, output, *, index=position):
                self.attention_outputs[int(index)] = self._boundary(output)

            def capture_mlp(_module, _inputs, output, *, index=position):
                self.mlp_outputs[int(index)] = self._boundary(output)

            def capture_block(_module, _inputs, output, *, index=position):
                self.block_outputs[int(index)] = self._boundary(output)

            self.handles.extend(
                [
                    block.register_forward_pre_hook(capture_input),
                    attention.register_forward_hook(capture_attention),
                    mlp.register_forward_hook(capture_mlp),
                    block.register_forward_hook(capture_block),
                ]
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def components(self, position: int):
        index = int(position)
        stores = (
            self.block_inputs,
            self.attention_outputs,
            self.mlp_outputs,
            self.block_outputs,
        )
        if any(index not in store for store in stores):
            raise RuntimeError(f"block component capture missed selected position {index}")
        block_delta = self.block_outputs[index] - self.block_inputs[index]
        return self.attention_outputs[index], self.mlp_outputs[index], block_delta

    def output_state(self, position: int):
        index = int(position)
        if index not in self.block_outputs:
            raise RuntimeError(f"block output capture missed selected position {index}")
        return self.block_outputs[index]


def _metric_outputs(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    names = (
        "attention_alignment_true",
        "attention_alignment_opposite",
        "mlp_alignment_true",
        "mlp_alignment_opposite",
        "block_alignment_true",
        "block_alignment_opposite",
        "attention_target_progress",
        "mlp_target_progress",
        "block_target_progress",
        "attention_target_error",
        "mlp_target_error",
        "block_target_error",
        "attention_write_norm",
        "mlp_write_norm",
        "block_write_norm",
        "reconstruction_relative_error",
        "state_replay_relative_error",
    )
    return {name: np.empty(shape, dtype=np.float32) for name in names}


def _store_component_metrics(
    outputs: dict[str, np.ndarray],
    prefix: str,
    positions: np.ndarray,
    layer_position: int,
    coordinates: np.ndarray,
    true_update: np.ndarray,
    opposite_update: np.ndarray,
) -> None:
    metrics = component_update_metrics(coordinates, true_update, opposite_update)
    outputs[f"{prefix}_alignment_true"][positions, layer_position] = metrics[
        "alignment_true"
    ]
    outputs[f"{prefix}_alignment_opposite"][positions, layer_position] = metrics[
        "alignment_opposite"
    ]
    outputs[f"{prefix}_target_progress"][positions, layer_position] = metrics[
        "target_progress"
    ]
    outputs[f"{prefix}_target_error"][positions, layer_position] = metrics[
        "target_error"
    ]


def extract_belief_update_decomposition(
    model,
    tokenizer,
    trace: CausalBeliefTrace,
    charts: LayerChartBundle,
    cfg: BeliefUpdateExtractionConfig,
    *,
    metadata: dict[str, Any] | None = None,
) -> BeliefUpdateTrace:
    """Measure attention, MLP, and actual block writes in belief coordinates."""

    import torch

    cfg.validate()
    gate_failure = representation_gate_failure_message(charts)
    if gate_failure is not None and not cfg.allow_failed_representation_gate:
        raise RuntimeError(
            f"{gate_failure}; use the exploratory override only if this failure "
            "is intentional"
        )
    topology = resolve_model_topology(model)
    final_depth = topology.num_depths - 1
    chart_lookup = {int(layer): index for index, layer in enumerate(charts.layers)}
    selected_depths = [
        int(layer) for layer in charts.layers if 1 <= int(layer) < int(final_depth)
    ]
    if not selected_depths:
        raise ValueError("charts contain no raw decoder-block depths")
    chart_positions = [chart_lookup[depth] for depth in selected_depths]
    blocks = [topology.blocks[depth - 1] for depth in selected_depths]
    row_indices = np.flatnonzero(trace.current_mask).astype(np.int64)
    if len(row_indices) < 2:
        raise ValueError("belief-update extraction requires current-query observations")
    if max(len(trace.input_ids[index]) for index in row_indices) > cfg.max_seq_len:
        raise ValueError("trace exceeds belief-update max_seq_len")
    outputs = _metric_outputs((len(row_indices), len(selected_depths)))
    device = next(model.parameters()).device
    batches = length_bucket_batches(
        [len(trace.input_ids[index]) for index in row_indices],
        batch_size=cfg.batch_size,
        max_batch_tokens=cfg.max_batch_tokens,
    )
    iterator = batches
    if cfg.show_progress:
        from tqdm import tqdm

        iterator = tqdm(batches, desc="belief-update decomposition batches")
    model.eval()
    with torch.inference_mode():
        for batch_positions in iterator:
            positions = np.asarray(batch_positions, dtype=np.int64)
            global_rows = row_indices[positions]
            model_inputs = _pad_saved_rows(tokenizer, trace, global_rows, device)
            last_indices = _last_visible_indices(model_inputs["attention_mask"])
            with _BlockComponentCapture(blocks, last_indices) as capture:
                topology.backbone(
                    **model_inputs,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
            true_update = trace.update_fourier[global_rows].astype(np.float32)
            opposite_update = _opposite_updates(trace, global_rows)
            batch_pairs = trace.pair_ids[global_rows]
            for local_layer, chart_position in enumerate(chart_positions):
                attention, mlp, block = capture.components(local_layer)
                attention_np = attention.float().cpu().numpy()
                mlp_np = mlp.float().cpu().numpy()
                block_np = block.float().cpu().numpy()
                replayed_state = capture.output_state(local_layer).float().cpu().numpy()
                stored_state = trace.states[global_rows, chart_position].astype(
                    np.float32
                )
                projected = {
                    "attention": _project_total_write(
                        attention_np, batch_pairs, charts, chart_position
                    ),
                    "mlp": _project_total_write(
                        mlp_np, batch_pairs, charts, chart_position
                    ),
                    "block": _project_total_write(
                        block_np, batch_pairs, charts, chart_position
                    ),
                }
                for prefix, coordinates in projected.items():
                    _store_component_metrics(
                        outputs,
                        prefix,
                        positions,
                        local_layer,
                        coordinates,
                        true_update,
                        opposite_update,
                    )
                outputs["attention_write_norm"][positions, local_layer] = np.linalg.norm(
                    attention_np, axis=-1
                )
                outputs["mlp_write_norm"][positions, local_layer] = np.linalg.norm(
                    mlp_np, axis=-1
                )
                outputs["block_write_norm"][positions, local_layer] = np.linalg.norm(
                    block_np, axis=-1
                )
                outputs["reconstruction_relative_error"][positions, local_layer] = (
                    component_reconstruction_error(attention_np, mlp_np, block_np)
                )
                outputs["state_replay_relative_error"][positions, local_layer] = (
                    np.linalg.norm(replayed_state - stored_state, axis=-1)
                    / np.maximum(np.linalg.norm(stored_state, axis=-1), 1e-12)
                )
    artifact_metadata = dict(metadata or {})
    artifact_metadata.update(
        schema=BELIEF_UPDATE_SCHEMA,
        source_trace_schema=trace.metadata.get("schema"),
        source_model=trace.metadata.get("model"),
        selected_depths=selected_depths,
        skipped_chart_depths=[
            int(layer) for layer in charts.layers if int(layer) not in selected_depths
        ],
        extraction_mode="target_token_block_component_write_v1",
        representation_gate=gate,
        component_identity="block_delta = attention_output + mlp_output",
        replay_identity="captured_block_output = stored_boundary_state",
    )
    result = BeliefUpdateTrace(
        row_indices=row_indices,
        pair_ids=trace.pair_ids[row_indices].astype(np.int64),
        branches=trace.branches[row_indices].astype(np.int8),
        layers=np.asarray(selected_depths, dtype=np.int64),
        metadata=artifact_metadata,
        **outputs,
    )
    result.validate()
    return result
