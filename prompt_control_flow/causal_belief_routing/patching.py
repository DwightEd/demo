from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from prompt_control_flow.belief_transport.model_capture import resolve_model_topology

from .charts import build_group_fold_ids
from .metrics import jensen_shannon_divergence, softmax
from .patch_schema import PATCH_SCHEMA, SourcePatchTrace
from .routing import source_head_pre_output
from .routing_extraction import (
    _ValueCapture,
    _attention_module,
    _head_topology,
    _last_visible_indices,
    _pad_saved_rows,
    _reshape_values,
    _source_masks,
)
from .schema import CausalBeliefTrace
from .extraction import aggregate_residue_logits


@dataclass(frozen=True)
class SourcePatchConfig:
    max_pairs: int = 0
    max_seq_len: int = 1024
    max_replay_js: float = 0.01
    show_progress: bool = True
    allow_failed_routing_gate: bool = False

    def validate(self) -> None:
        if int(self.max_pairs) < 0 or int(self.max_seq_len) < 1:
            raise ValueError("patch pair and sequence limits are invalid")
        if not 0.0 <= float(self.max_replay_js) <= np.log(2.0):
            raise ValueError("max_replay_js is outside the JS range")


def apply_source_component_patch(
    head_input,
    *,
    target_indices,
    component_delta,
    selected_heads: Sequence[int],
):
    """Add donor-minus-recipient source components at selected head slices."""

    import torch

    values = head_input
    delta = component_delta
    if values.ndim != 3 or delta.ndim != 3:
        raise ValueError("attention head input and source delta must be rank three")
    batch, _sequence, width = values.shape
    if target_indices.shape != (batch,) or delta.shape[0] != batch:
        raise ValueError("patch batch dimensions are misaligned")
    heads = int(delta.shape[1])
    head_dim = int(delta.shape[2])
    if width != heads * head_dim:
        raise ValueError("attention output width does not match source head components")
    result = values.clone()
    batch_index = torch.arange(batch, device=values.device)
    reshaped = result[batch_index, target_indices].reshape(batch, heads, head_dim)
    for head in selected_heads:
        if not 0 <= int(head) < heads:
            raise ValueError("selected head is outside the attention topology")
        reshaped[:, int(head)] += delta[:, int(head)].to(
            device=values.device, dtype=values.dtype
        )
    result[batch_index, target_indices] = reshaped.reshape(batch, width)
    return result


def _load_routing_selections(path: str | Path) -> tuple[dict[int, list[tuple[int, int]]], dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    selections: dict[int, list[tuple[int, int]]] = {}
    for row in report.get("head_selections", []):
        fold = int(row["fold"])
        selections[fold] = [
            (int(item["layer"]), int(item["head"]))
            for item in row.get("selected", [])
        ]
    if not selections:
        raise ValueError("routing audit contains no cross-fitted head selections")
    return selections, report


def _future_rows(trace: CausalBeliefTrace) -> dict[int, dict[int, int]]:
    result: dict[int, dict[int, int]] = {}
    for index in np.flatnonzero(trace.query_roles == "future"):
        pair = int(trace.pair_ids[index])
        branch = int(trace.branches[index])
        result.setdefault(pair, {})[branch] = int(index)
    invalid = [pair for pair, rows in result.items() if set(rows) != {0, 1}]
    if invalid:
        raise ValueError(f"future trace is incomplete for pairs {invalid[:5]}")
    return result


def frozen_pair_folds(
    all_pair_ids: Sequence[int] | np.ndarray,
    selected_pair_ids: Sequence[int] | np.ndarray,
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    all_pairs = np.asarray(all_pair_ids, dtype=np.int64)
    selected = np.asarray(selected_pair_ids, dtype=np.int64)
    if len(np.unique(all_pairs)) != len(all_pairs):
        raise ValueError("all_pair_ids must contain each pair exactly once")
    all_fold_ids = build_group_fold_ids(all_pairs, num_folds=folds, seed=seed)
    mapping = {
        int(pair): int(fold)
        for pair, fold in zip(all_pairs, all_fold_ids, strict=True)
    }
    missing = [int(pair) for pair in selected if int(pair) not in mapping]
    if missing:
        raise KeyError(f"selected pairs are absent from the frozen fold map: {missing[:5]}")
    return np.asarray([mapping[int(pair)] for pair in selected], dtype=np.int16)


def _residue_token_groups(trace: CausalBeliefTrace) -> list[list[int]]:
    values = trace.metadata.get("residue_token_id_groups")
    if values is None:
        legacy = trace.metadata.get("residue_token_ids")
        values = [[value] for value in legacy] if isinstance(legacy, list) else None
    if not isinstance(values, list) or len(values) != trace.residue_logits.shape[1]:
        raise ValueError("trace does not contain exact residue token groups")
    groups = [[int(token) for token in group] for group in values]
    if any(not group for group in groups):
        raise ValueError("residue token groups must be non-empty")
    return groups


def _baseline_source_components(
    model,
    tokenizer,
    trace: CausalBeliefTrace,
    row_indices: np.ndarray,
    depths: Sequence[int],
):
    import torch

    topology = resolve_model_topology(model)
    modules = [_attention_module(topology.blocks[int(depth) - 1]) for depth in depths]
    head_topologies = [_head_topology(model, module) for module in modules]
    device = next(model.parameters()).device
    inputs = _pad_saved_rows(tokenizer, trace, row_indices, device)
    last = _last_visible_indices(inputs["attention_mask"])
    evidence_mask, control_mask = _source_masks(
        trace, row_indices, inputs["attention_mask"]
    )
    with _ValueCapture(modules) as capture:
        output = topology.backbone(
            **inputs,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    if output.attentions is None:
        raise RuntimeError("source patch baseline requires eager attention tensors")
    batch_index = torch.arange(len(row_indices), device=device)
    final_state = output.last_hidden_state[batch_index, last]
    logits = topology.output_head(final_state).float()
    evidence: dict[int, object] = {}
    control: dict[int, object] = {}
    for local, (depth, head_values) in enumerate(zip(depths, head_topologies, strict=True)):
        _heads, kv_heads, head_dim = head_values
        values = _reshape_values(
            capture.values[local], kv_heads=kv_heads, head_dim=head_dim
        )
        attention = output.attentions[int(depth) - 1].float()
        evidence[int(depth)] = source_head_pre_output(
            attention,
            values,
            target_indices=last,
            source_mask=evidence_mask,
        )[0]
        control[int(depth)] = source_head_pre_output(
            attention,
            values,
            target_indices=last,
            source_mask=control_mask,
        )[0]
    return logits, evidence, control


def _patched_logits(
    model,
    tokenizer,
    trace: CausalBeliefTrace,
    recipient_row: int,
    selected: dict[int, list[int]],
    deltas: dict[int, object],
):
    topology = resolve_model_topology(model)
    device = next(model.parameters()).device
    inputs = _pad_saved_rows(tokenizer, trace, [recipient_row], device)
    last = _last_visible_indices(inputs["attention_mask"])
    handles = []
    for depth, heads in selected.items():
        module = _attention_module(topology.blocks[int(depth) - 1])
        projection = getattr(module, "o_proj", None)
        if projection is None:
            raise TypeError("attention module does not expose o_proj")

        def patch(_module, inputs_tuple, *, layer=int(depth), chosen=tuple(heads)):
            if not inputs_tuple:
                raise RuntimeError("o_proj pre-hook received no head input")
            modified = apply_source_component_patch(
                inputs_tuple[0],
                target_indices=last,
                component_delta=deltas[layer],
                selected_heads=chosen,
            )
            return (modified,) + tuple(inputs_tuple[1:])

        handles.append(projection.register_forward_pre_hook(patch))
    try:
        output = topology.backbone(
            **inputs,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    final_state = output.last_hidden_state[0, last[0]]
    return topology.output_head(final_state).float()


def _logodds(logits: np.ndarray, donor: int, recipient: int) -> float:
    return float(logits[int(donor)] - logits[int(recipient)])


def extract_source_patches(
    model,
    tokenizer,
    trace: CausalBeliefTrace,
    routing_summary_path: str | Path,
    cfg: SourcePatchConfig,
    *,
    metadata: dict[str, Any] | None = None,
) -> SourcePatchTrace:
    """Patch donor evidence components into recipient attention-head paths."""

    import torch

    cfg.validate()
    selections, routing_report = _load_routing_selections(routing_summary_path)
    routing_gate = routing_report.get("decision_gate", {})
    if not cfg.allow_failed_routing_gate and not bool(
        routing_gate.get("ready_for_causal_patching", False)
    ):
        raise RuntimeError(
            "routing gate did not authorize causal patching; use the exploratory "
            "override only if this is intentional"
        )
    folds = int(routing_report["config"]["folds"])
    split_seed = int(routing_report["config"]["seed"])
    future = _future_rows(trace)
    all_pairs = np.asarray(sorted(future), dtype=np.int64)
    pairs = all_pairs
    if cfg.max_pairs > 0:
        pairs = pairs[: int(cfg.max_pairs)]
    pair_fold_rows = frozen_pair_folds(
        all_pairs,
        pairs,
        folds=folds,
        seed=split_seed,
    )
    residue_groups = _residue_token_groups(trace)
    records: dict[str, list[Any]] = {
        name: []
        for name in (
            "pair_ids",
            "recipient_branches",
            "donor_branches",
            "fold_ids",
            "selected_head_counts",
            "replay_js",
            "evidence_logodds_shift",
            "control_logodds_shift",
            "random_head_logodds_shift",
            "evidence_donor_probability_shift",
            "control_donor_probability_shift",
            "random_head_donor_probability_shift",
        )
    }
    iterator = list(zip(pairs.tolist(), pair_fold_rows.tolist(), strict=True))
    if cfg.show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="source-specific path patches")
    skipped: dict[str, int] = {}
    skip_examples: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for pair, fold in iterator:
            try:
                selected_items = selections[int(fold)]
                selected: dict[int, list[int]] = {}
                for depth, head in selected_items:
                    selected.setdefault(int(depth), []).append(int(head))
                depths = sorted(selected)
                rows = np.asarray([future[int(pair)][0], future[int(pair)][1]])
                if max(len(trace.input_ids[index]) for index in rows) > cfg.max_seq_len:
                    raise ValueError("future prompt exceeds patch max_seq_len")
                baseline_logits, evidence, control = _baseline_source_components(
                    model, tokenizer, trace, rows, depths
                )
                residue_baseline = (
                    aggregate_residue_logits(baseline_logits, residue_groups)
                    .detach()
                    .cpu()
                    .numpy()
                )
                replay_js = max(
                    jensen_shannon_divergence(
                        softmax(trace.residue_logits[row]), softmax(residue_baseline[local])
                    )
                    for local, row in enumerate(rows)
                )
                if replay_js > cfg.max_replay_js:
                    raise RuntimeError(
                        f"baseline replay JS {replay_js:.6f} exceeds {cfg.max_replay_js:.6f}"
                    )
                future_labels = np.argmax(trace.exact_query_distributions[rows], axis=1)
                num_heads = int(next(iter(evidence.values())).shape[1])
                rng = np.random.default_rng(split_seed + 104729 * int(pair))
                random_selected: dict[int, list[int]] = {}
                for depth, heads in selected.items():
                    candidates = np.asarray(
                        [head for head in range(num_heads) if head not in set(heads)],
                        dtype=np.int64,
                    )
                    if len(candidates) < len(heads):
                        raise ValueError("too few unselected heads for a matched random null")
                    random_selected[depth] = rng.choice(
                        candidates, size=len(heads), replace=False
                    ).tolist()
                for recipient in (0, 1):
                    donor = 1 - recipient
                    evidence_delta = {
                        depth: evidence[depth][donor : donor + 1]
                        - evidence[depth][recipient : recipient + 1]
                        for depth in depths
                    }
                    control_delta = {
                        depth: control[depth][donor : donor + 1]
                        - control[depth][recipient : recipient + 1]
                        for depth in depths
                    }
                    evidence_logits = _patched_logits(
                        model,
                        tokenizer,
                        trace,
                        int(rows[recipient]),
                        selected,
                        evidence_delta,
                    )
                    evidence_logits = (
                        aggregate_residue_logits(
                            evidence_logits[None, :], residue_groups
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    control_logits = _patched_logits(
                        model,
                        tokenizer,
                        trace,
                        int(rows[recipient]),
                        selected,
                        control_delta,
                    )
                    control_logits = (
                        aggregate_residue_logits(
                            control_logits[None, :], residue_groups
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    random_head_logits = _patched_logits(
                        model,
                        tokenizer,
                        trace,
                        int(rows[recipient]),
                        random_selected,
                        evidence_delta,
                    )
                    random_head_logits = (
                        aggregate_residue_logits(
                            random_head_logits[None, :], residue_groups
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    baseline = residue_baseline[recipient]
                    donor_label = int(future_labels[donor])
                    recipient_label = int(future_labels[recipient])
                    baseline_probability = softmax(baseline)
                    evidence_probability = softmax(evidence_logits)
                    control_probability = softmax(control_logits)
                    random_head_probability = softmax(random_head_logits)
                    records["pair_ids"].append(int(pair))
                    records["recipient_branches"].append(recipient)
                    records["donor_branches"].append(donor)
                    records["fold_ids"].append(int(fold))
                    records["selected_head_counts"].append(
                        sum(len(heads) for heads in selected.values())
                    )
                    records["replay_js"].append(float(replay_js))
                    records["evidence_logodds_shift"].append(
                        _logodds(evidence_logits, donor_label, recipient_label)
                        - _logodds(baseline, donor_label, recipient_label)
                    )
                    records["control_logodds_shift"].append(
                        _logodds(control_logits, donor_label, recipient_label)
                        - _logodds(baseline, donor_label, recipient_label)
                    )
                    records["random_head_logodds_shift"].append(
                        _logodds(random_head_logits, donor_label, recipient_label)
                        - _logodds(baseline, donor_label, recipient_label)
                    )
                    records["evidence_donor_probability_shift"].append(
                        float(evidence_probability[donor_label] - baseline_probability[donor_label])
                    )
                    records["control_donor_probability_shift"].append(
                        float(control_probability[donor_label] - baseline_probability[donor_label])
                    )
                    records["random_head_donor_probability_shift"].append(
                        float(
                            random_head_probability[donor_label]
                            - baseline_probability[donor_label]
                        )
                    )
            except (RuntimeError, TypeError, ValueError, KeyError) as error:
                key = type(error).__name__
                skipped[key] = skipped.get(key, 0) + 1
                if len(skip_examples) < 20:
                    skip_examples.append(
                        {
                            "pair_id": int(pair),
                            "error_type": key,
                            "message": str(error),
                        }
                    )
    attempted_directions = 2 * len(pairs)
    coverage = len(records["pair_ids"]) / max(attempted_directions, 1)
    if not records["pair_ids"]:
        raise RuntimeError(f"all source patches failed: {skipped}")
    patch_metadata = dict(metadata or {})
    patch_metadata.update(
        schema=PATCH_SCHEMA,
        source_trace=trace.metadata.get("schema"),
        routing_summary=str(routing_summary_path),
        routing_gate=routing_gate,
        attempted_pairs=int(len(pairs)),
        attempted_directions=int(attempted_directions),
        coverage=float(coverage),
        skip_reasons=skipped,
        skip_examples=skip_examples,
        extraction_mode="source_specific_head_path_patch_v1",
    )
    result = SourcePatchTrace(
        pair_ids=np.asarray(records["pair_ids"], dtype=np.int64),
        recipient_branches=np.asarray(records["recipient_branches"], dtype=np.int8),
        donor_branches=np.asarray(records["donor_branches"], dtype=np.int8),
        fold_ids=np.asarray(records["fold_ids"], dtype=np.int16),
        selected_head_counts=np.asarray(records["selected_head_counts"], dtype=np.int16),
        replay_js=np.asarray(records["replay_js"], dtype=np.float32),
        evidence_logodds_shift=np.asarray(
            records["evidence_logodds_shift"], dtype=np.float32
        ),
        control_logodds_shift=np.asarray(
            records["control_logodds_shift"], dtype=np.float32
        ),
        random_head_logodds_shift=np.asarray(
            records["random_head_logodds_shift"], dtype=np.float32
        ),
        evidence_donor_probability_shift=np.asarray(
            records["evidence_donor_probability_shift"], dtype=np.float32
        ),
        control_donor_probability_shift=np.asarray(
            records["control_donor_probability_shift"], dtype=np.float32
        ),
        random_head_donor_probability_shift=np.asarray(
            records["random_head_donor_probability_shift"], dtype=np.float32
        ),
        metadata=patch_metadata,
    )
    result.validate()
    return result
