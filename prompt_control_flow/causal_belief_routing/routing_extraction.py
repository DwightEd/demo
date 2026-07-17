from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from prompt_control_flow.belief_transport.extraction import length_bucket_batches
from prompt_control_flow.belief_transport.model_capture import resolve_model_topology

from .charts import LayerChartBundle
from .routing import (
    cosine_alignment,
    head_residual_writes,
    length_matched_control_mask,
    source_head_pre_output,
)
from .routing_schema import EvidenceRoutingTrace, ROUTING_SCHEMA
from .schema import CausalBeliefTrace


@dataclass(frozen=True)
class RoutingExtractionConfig:
    batch_size: int = 4
    max_batch_tokens: int = 2048
    max_seq_len: int = 1024
    show_progress: bool = True
    allow_failed_representation_gate: bool = False

    def validate(self) -> None:
        if min(self.batch_size, self.max_batch_tokens, self.max_seq_len) < 1:
            raise ValueError("routing batch and sequence limits must be positive")


def _attention_module(block):
    for name in ("self_attn", "attn", "attention"):
        module = getattr(block, name, None)
        if module is not None:
            return module
    raise TypeError("decoder block does not expose a self-attention module")


def _head_topology(model, attention_module) -> tuple[int, int, int]:
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


class _ValueCapture:
    def __init__(self, attention_modules: Sequence[object]) -> None:
        self.attention_modules = list(attention_modules)
        self.values: dict[int, object] = {}
        self.handles: list[object] = []

    def __enter__(self) -> "_ValueCapture":
        for position, module in enumerate(self.attention_modules):
            projection = getattr(module, "v_proj", None)
            if projection is None:
                raise TypeError("attention module does not expose v_proj")

            def capture(_module, _inputs, output, *, index=position):
                value = output[0] if isinstance(output, (tuple, list)) else output
                self.values[int(index)] = value.detach()

            self.handles.append(projection.register_forward_hook(capture))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _pad_saved_rows(tokenizer, trace: CausalBeliefTrace, row_indices: Sequence[int], device):
    features = [
        {
            "input_ids": trace.input_ids[int(index)].tolist(),
            "attention_mask": [1] * len(trace.input_ids[int(index)]),
        }
        for index in row_indices
    ]
    padded = tokenizer.pad(
        features,
        padding=True,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
        return_tensors="pt",
    )
    return {
        key: value.to(device, non_blocking=True)
        for key, value in padded.items()
        if key in {"input_ids", "attention_mask"}
    }


def _last_visible_indices(attention_mask):
    import torch

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions[None, :].expand_as(attention_mask)
    return torch.where(
        attention_mask.bool(), positions, torch.full_like(positions, -1)
    ).max(dim=1).values


def _source_masks(
    trace: CausalBeliefTrace,
    row_indices: np.ndarray,
    attention_mask,
):
    import torch

    batch, padded_length = attention_mask.shape
    evidence = np.zeros((batch, padded_length), dtype=bool)
    ranges = trace.evidence_token_ranges[row_indices]
    lengths = attention_mask.sum(dim=1).detach().cpu().numpy().astype(np.int64)
    for local, (start, stop) in enumerate(ranges):
        if not (0 <= int(start) < int(stop) <= int(lengths[local])):
            raise ValueError("stored evidence range is outside replayed input IDs")
        evidence[local, int(start) : int(stop)] = True
    control = length_matched_control_mask(lengths, ranges, padded_length)
    return (
        torch.as_tensor(evidence, device=attention_mask.device),
        torch.as_tensor(control, device=attention_mask.device),
    )


def _opposite_updates(
    trace: CausalBeliefTrace,
    row_indices: np.ndarray,
) -> np.ndarray:
    lookup = {
        (int(trace.pair_ids[index]), int(trace.branches[index])): int(index)
        for index in np.flatnonzero(trace.current_mask)
    }
    return np.stack(
        [
            trace.update_fourier[
                lookup[(int(trace.pair_ids[index]), 1 - int(trace.branches[index]))]
            ]
            for index in row_indices
        ]
    ).astype(np.float32)


def _project_head_writes(
    writes: np.ndarray,
    pair_ids: np.ndarray,
    bundle: LayerChartBundle,
    layer_position: int,
) -> np.ndarray:
    values = np.asarray(writes, dtype=np.float32)
    rows, heads, hidden = values.shape
    projection = bundle.projection(layer_position)
    projected = projection.transform(values.reshape(rows * heads, hidden)).reshape(
        rows, heads, -1
    )
    result = np.empty(
        (rows, heads, bundle.weights.shape[-1]), dtype=np.float32
    )
    folds = np.asarray([bundle.fold_for_pair(int(pair)) for pair in pair_ids])
    for fold in np.unique(folds):
        selected = folds == int(fold)
        chart = bundle.chart(int(fold), int(layer_position))
        local = projected[selected]
        result[selected] = chart.project_direction(
            local.reshape(-1, local.shape[-1])
        ).reshape(local.shape[0], heads, -1)
    return result


def _project_total_write(
    writes: np.ndarray,
    pair_ids: np.ndarray,
    bundle: LayerChartBundle,
    layer_position: int,
) -> np.ndarray:
    return _project_head_writes(
        np.asarray(writes)[:, None, :], pair_ids, bundle, layer_position
    )[:, 0]


def _reshape_values(raw_values, *, kv_heads: int, head_dim: int):
    if raw_values.ndim != 3 or raw_values.shape[-1] != kv_heads * head_dim:
        raise ValueError("captured v_proj output has an unexpected shape")
    return raw_values.reshape(raw_values.shape[0], raw_values.shape[1], kv_heads, head_dim)


def extract_evidence_routing(
    model,
    tokenizer,
    trace: CausalBeliefTrace,
    charts: LayerChartBundle,
    cfg: RoutingExtractionConfig,
    *,
    metadata: dict[str, Any] | None = None,
) -> EvidenceRoutingTrace:
    """Measure source-specific attention writes in held-out belief coordinates."""

    import torch

    cfg.validate()
    gate = charts.metadata.get("decision_gate", {})
    if not cfg.allow_failed_representation_gate and not bool(
        gate.get("ready_for_routing_analysis", False)
    ):
        raise RuntimeError(
            "representation gate did not authorize routing analysis; pass the "
            "exploratory override only if this failure is intentional"
        )
    topology = resolve_model_topology(model)
    final_depth = topology.num_depths - 1
    chart_layer_lookup = {int(layer): index for index, layer in enumerate(charts.layers)}
    selected_depths = [
        int(layer)
        for layer in charts.layers
        if 1 <= int(layer) < int(final_depth)
    ]
    if not selected_depths:
        raise ValueError("charts contain no pre-final-norm decoder depths")
    layer_positions = [chart_layer_lookup[depth] for depth in selected_depths]
    blocks = [topology.blocks[depth - 1] for depth in selected_depths]
    attention_modules = [_attention_module(block) for block in blocks]
    head_topologies = [_head_topology(model, module) for module in attention_modules]
    if len({item[0] for item in head_topologies}) != 1:
        raise ValueError("selected layers have inconsistent attention head counts")
    num_heads = int(head_topologies[0][0])

    row_indices = np.flatnonzero(trace.current_mask).astype(np.int64)
    if len(row_indices) < 2:
        raise ValueError("routing extraction requires current-query observations")
    if max(len(trace.input_ids[index]) for index in row_indices) > cfg.max_seq_len:
        raise ValueError("trace exceeds routing max_seq_len")
    n_rows = len(row_indices)
    shape = (n_rows, len(selected_depths), num_heads)
    outputs = {
        name: np.empty(shape, dtype=np.float32)
        for name in (
            "evidence_mass",
            "control_mass",
            "evidence_alignment_true",
            "evidence_alignment_opposite",
            "control_alignment_true",
            "control_alignment_opposite",
            "evidence_write_norm",
            "control_write_norm",
        )
    }
    layer_true = np.empty((n_rows, len(selected_depths)), dtype=np.float32)
    layer_opposite = np.empty_like(layer_true)
    device = next(model.parameters()).device
    batches = length_bucket_batches(
        [len(trace.input_ids[index]) for index in row_indices],
        batch_size=cfg.batch_size,
        max_batch_tokens=cfg.max_batch_tokens,
    )
    iterator = batches
    if cfg.show_progress:
        from tqdm import tqdm

        iterator = tqdm(batches, desc="evidence routing batches")
    model.eval()
    with torch.inference_mode():
        for batch_positions in iterator:
            global_rows = row_indices[np.asarray(batch_positions, dtype=np.int64)]
            model_inputs = _pad_saved_rows(tokenizer, trace, global_rows, device)
            last_indices = _last_visible_indices(model_inputs["attention_mask"])
            evidence_mask, control_mask = _source_masks(
                trace, global_rows, model_inputs["attention_mask"]
            )
            with _ValueCapture(attention_modules) as capture:
                model_output = topology.backbone(
                    **model_inputs,
                    use_cache=False,
                    output_attentions=True,
                    return_dict=True,
                )
            attentions = getattr(model_output, "attentions", None)
            if attentions is None or len(attentions) != len(topology.blocks):
                raise RuntimeError(
                    "observer did not return one attention tensor per decoder block; "
                    "load it with attn_implementation='eager'"
                )
            true_update = trace.update_fourier[global_rows].astype(np.float32)
            opposite_update = _opposite_updates(trace, global_rows)
            batch_pairs = trace.pair_ids[global_rows]
            for local_layer, (depth, chart_position, module, topology_values) in enumerate(
                zip(
                    selected_depths,
                    layer_positions,
                    attention_modules,
                    head_topologies,
                    strict=True,
                )
            ):
                heads, kv_heads, head_dim = topology_values
                attention = attentions[depth - 1].float()
                values = _reshape_values(
                    capture.values[local_layer],
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                )
                evidence_pre, evidence_mass = source_head_pre_output(
                    attention,
                    values,
                    target_indices=last_indices,
                    source_mask=evidence_mask,
                )
                control_pre, control_mass = source_head_pre_output(
                    attention,
                    values,
                    target_indices=last_indices,
                    source_mask=control_mask,
                )
                output_projection = getattr(module, "o_proj", None)
                if output_projection is None or not hasattr(output_projection, "weight"):
                    raise TypeError("attention module does not expose o_proj.weight")
                evidence_writes = head_residual_writes(
                    evidence_pre, output_projection.weight
                ).float()
                control_writes = head_residual_writes(
                    control_pre, output_projection.weight
                ).float()
                evidence_np = evidence_writes.detach().cpu().numpy()
                control_np = control_writes.detach().cpu().numpy()
                evidence_coordinates = _project_head_writes(
                    evidence_np, batch_pairs, charts, chart_position
                )
                control_coordinates = _project_head_writes(
                    control_np, batch_pairs, charts, chart_position
                )
                positions = np.asarray(batch_positions, dtype=np.int64)
                outputs["evidence_mass"][positions, local_layer] = (
                    evidence_mass.detach().cpu().numpy()
                )
                outputs["control_mass"][positions, local_layer] = (
                    control_mass.detach().cpu().numpy()
                )
                outputs["evidence_alignment_true"][positions, local_layer] = (
                    cosine_alignment(evidence_coordinates, true_update)
                )
                outputs["evidence_alignment_opposite"][positions, local_layer] = (
                    cosine_alignment(evidence_coordinates, opposite_update)
                )
                outputs["control_alignment_true"][positions, local_layer] = (
                    cosine_alignment(control_coordinates, true_update)
                )
                outputs["control_alignment_opposite"][positions, local_layer] = (
                    cosine_alignment(control_coordinates, opposite_update)
                )
                outputs["evidence_write_norm"][positions, local_layer] = np.linalg.norm(
                    evidence_np, axis=-1
                )
                outputs["control_write_norm"][positions, local_layer] = np.linalg.norm(
                    control_np, axis=-1
                )
                total_coordinates = _project_total_write(
                    evidence_np.sum(axis=1), batch_pairs, charts, chart_position
                )
                layer_true[positions, local_layer] = cosine_alignment(
                    total_coordinates, true_update
                )
                layer_opposite[positions, local_layer] = cosine_alignment(
                    total_coordinates, opposite_update
                )

    artifact_metadata = dict(metadata or {})
    artifact_metadata.update(
        schema=ROUTING_SCHEMA,
        source_trace_schema=trace.metadata.get("schema"),
        source_model=trace.metadata.get("model"),
        selected_depths=selected_depths,
        skipped_chart_depths=[
            int(layer) for layer in charts.layers if int(layer) not in selected_depths
        ],
        num_heads=num_heads,
        extraction_mode="evidence_source_ov_write_v1",
        representation_gate=gate,
    )
    result = EvidenceRoutingTrace(
        row_indices=row_indices,
        pair_ids=trace.pair_ids[row_indices].astype(np.int64),
        branches=trace.branches[row_indices].astype(np.int8),
        layers=np.asarray(selected_depths, dtype=np.int64),
        evidence_mass=outputs["evidence_mass"],
        control_mass=outputs["control_mass"],
        evidence_alignment_true=outputs["evidence_alignment_true"],
        evidence_alignment_opposite=outputs["evidence_alignment_opposite"],
        control_alignment_true=outputs["control_alignment_true"],
        control_alignment_opposite=outputs["control_alignment_opposite"],
        evidence_write_norm=outputs["evidence_write_norm"],
        control_write_norm=outputs["control_write_norm"],
        layer_alignment_true=layer_true,
        layer_alignment_opposite=layer_opposite,
        metadata=artifact_metadata,
    )
    result.validate()
    return result
