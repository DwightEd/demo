from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .config import ExtractionConfig, STEP_METRIC_NAMES
from .data_contract import DATA_CONTRACT_VERSION
from .data import ChainRecord, process_correct_from_gold
from .extractors import (
    MechanismExtractor,
    PromptResidualFlowExtractor,
    UncertaintyExtractor,
)
from .metrics import (
    compute_prompt_token_layer_states,
    compute_response_token_layer_states,
    compute_step_boundary_state_vectors,
    compute_step_layer_state_vectors,
    compute_step_residual_vectors,
    summarize_step_metrics,
)
from .replay_protocols import (
    EXACT_ARTIFACT_REPLAY,
    STORED_RENDERED_PROMPT_REPLAY,
    build_observer_response,
    render_processbench_observer_prompt,
)
from .teacher_forcing import ForwardCache, run_teacher_forcing
from .storage import cast_state_array
from utils.step_boundaries import (
    SPAN_RANGE_CONVENTION,
    STEP_TOKEN_RANGE_CONVENTION,
    TOKEN_OFFSET_CONVENTION,
    TRACE_SCHEMA_VERSION,
)


@dataclass
class MechanismExtraction:
    record: ChainRecord
    step_scores: Dict[str, np.ndarray]
    chain_scores: Dict[str, float]
    n_steps: int
    token_scores: Dict[str, np.ndarray] = field(default_factory=dict)
    step_token_ranges: list[tuple[int, int]] = field(default_factory=list)
    step_vectors: np.ndarray | None = None
    step_state_vectors: np.ndarray | None = None
    step_layer_state_vectors: np.ndarray | None = None
    step_pre_state_vectors: np.ndarray | None = None
    step_end_state_vectors: np.ndarray | None = None
    prompt_token_layer_states: np.ndarray | None = None
    response_token_layer_states: np.ndarray | None = None
    trace_input_ids: np.ndarray | None = None
    trace_attention_mask: np.ndarray | None = None
    trace_token_offsets: np.ndarray | None = None
    prompt_token_range: tuple[int, int] = (-1, -1)
    question_token_range: tuple[int, int] = (-1, -1)
    response_token_range: tuple[int, int] = (-1, -1)
    rendered_prompt: str = ""
    response_text: str = ""
    layers: tuple[int, ...] = ()
    metadata: dict = field(default_factory=dict)


# Backward-compatible name used by older tests and scripts.
ChainExtraction = MechanismExtraction


def extract_chain_mechanisms(
    model,
    tokenizer,
    record: ChainRecord,
    cfg: ExtractionConfig,
    extractors: Sequence[MechanismExtractor],
) -> MechanismExtraction | None:
    """Teacher-force one chain and compute requested mechanism scores."""

    if not extractors and not any(
        (
            cfg.store_step_vectors,
            cfg.store_step_state_vectors,
            cfg.store_prompt_token_states,
            cfg.store_response_token_states,
        )
    ):
        raise ValueError("at least one extractor or state-storage target is required")
    response = (
        record.response
        if record.response
        else build_observer_response(record.steps, separator=cfg.response_separator)
    )
    if record.exact_input_ids is not None:
        if not record.rendered_prompt:
            raise ValueError("exact token replay requires the stored rendered prompt")
        prompt = str(record.rendered_prompt)
        replay_protocol = EXACT_ARTIFACT_REPLAY
        prompt_provenance = "stored_exact_generation_artifact"
        messages_json = ""
    elif record.rendered_prompt:
        prompt = str(record.rendered_prompt)
        replay_protocol = STORED_RENDERED_PROMPT_REPLAY
        prompt_provenance = "stored_prompt_retokenized_on_current_tokenizer"
        messages_json = ""
    else:
        rendered = render_processbench_observer_prompt(
            tokenizer,
            record.problem,
            protocol=cfg.replay_protocol,
        )
        prompt = rendered.rendered_prompt
        replay_protocol = rendered.protocol
        prompt_provenance = rendered.provenance
        messages_json = rendered.messages_json
    if not prompt.strip() or not response.strip():
        return None

    need_hidden = (
        any(e.requires_hidden for e in extractors)
        or bool(cfg.store_step_vectors)
        or bool(cfg.store_step_state_vectors)
        or bool(cfg.store_prompt_token_states)
        or bool(cfg.store_response_token_states)
    )
    need_attention = any(e.requires_attention for e in extractors)
    need_logits = any(e.requires_logits for e in extractors)
    requested_depths = tuple(int(x) for x in cfg.layers)
    hidden_depths = None
    attention_blocks = None
    if requested_depths:
        hidden_depths = tuple(
            sorted({depth for selected in requested_depths for depth in (selected - 1, selected)})
        )
        attention_blocks = tuple(sorted({depth - 1 for depth in requested_depths}))

    cache = run_teacher_forcing(
        model,
        tokenizer,
        prompt,
        response,
        output_hidden_states=need_hidden,
        output_attentions=need_attention,
        output_logits=need_logits,
        max_seq_len=cfg.max_seq_len,
        steps=record.steps,
        exact_input_ids=record.exact_input_ids,
        exact_attention_mask=record.exact_attention_mask,
        exact_token_offsets=record.exact_token_offsets,
        exact_step_token_ranges=record.exact_step_token_ranges,
        exact_response_start_token=record.exact_response_start_token,
        exact_question_token_range=record.exact_question_token_range,
        question_text=record.problem,
        replay_protocol=replay_protocol,
        prompt_provenance=prompt_provenance,
        messages_json=messages_json,
        hidden_depths=hidden_depths,
        attention_blocks=attention_blocks,
        max_attention_tokens=(
            int(cfg.full_attention_token_threshold) if need_attention else None
        ),
    )
    if len(cache.step_token_ranges) == 0:
        return None

    # An empty layer tuple is the explicit ``all post-block depths`` sentinel.
    # Hugging Face hidden_states[0] is the embedding output; indices 1..N are
    # the outputs after transformer blocks 1..N.
    resolved_layers = tuple(int(x) for x in cfg.layers)
    if not resolved_layers and cache.hidden_states is not None:
        resolved_layers = tuple(range(1, len(cache.hidden_states)))
    if cache.hidden_states is not None:
        invalid_depths = [
            depth
            for depth in resolved_layers
            if depth <= 0 or depth >= len(cache.hidden_states)
        ]
        if invalid_depths:
            raise ValueError(
                f"requested post-block hidden-state depths {invalid_depths} are outside "
                f"[1, {len(cache.hidden_states) - 1}]"
            )
    active_cfg = replace(cfg, layers=resolved_layers)

    step_scores: Dict[str, np.ndarray] = {}
    response_start, response_stop = cache.response_token_range
    token_scores: Dict[str, np.ndarray] = {
        f"output_{name}": np.asarray(values, dtype=np.float32)[
            int(response_start) : int(response_stop)
        ]
        for name, values in cache.token_output_summaries.items()
    }
    for extractor in extractors:
        scores = extractor.compute(cache, record, active_cfg)
        for name, values in scores.items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim != 1 or arr.shape[0] != len(cache.step_token_ranges):
                raise ValueError(
                    f"extractor {extractor.name!r} returned {name!r} with shape "
                    f"{arr.shape}; step scores must be [n_steps]"
                )
            step_scores[name] = arr

    step_vectors = None
    if cfg.store_step_vectors and cache.hidden_states is not None:
        step_vectors = compute_step_residual_vectors(
            cache.hidden_states,
            step_ranges=cache.step_token_ranges,
            layers=active_cfg.layers,
        )
    step_state_vectors = None
    step_layer_state_vectors = None
    step_pre_state_vectors = None
    step_end_state_vectors = None
    if cfg.store_step_state_vectors and cache.hidden_states is not None:
        step_layer_state_vectors = compute_step_layer_state_vectors(
            cache.hidden_states,
            step_ranges=cache.step_token_ranges,
            layers=active_cfg.layers,
        )
        step_pre_state_vectors, step_end_state_vectors = compute_step_boundary_state_vectors(
            cache.hidden_states,
            step_ranges=cache.step_token_ranges,
            layers=active_cfg.layers,
        )
        if cfg.store_flat_step_state_vectors:
            step_state_vectors = step_layer_state_vectors.reshape(step_layer_state_vectors.shape[0], -1)

    prompt_token_layer_states = None
    if cfg.store_prompt_token_states and cache.hidden_states is not None:
        prompt_token_layer_states = compute_prompt_token_layer_states(
            cache.hidden_states,
            prompt_token_range=cache.prompt_token_range,
            layers=active_cfg.layers,
        )

    response_token_layer_states = None
    if cfg.store_response_token_states and cache.hidden_states is not None:
        response_token_layer_states = compute_response_token_layer_states(
            cache.hidden_states,
            response_token_range=cache.response_token_range,
            layers=active_cfg.layers,
        )

    chain_scores = summarize_step_metrics(step_scores)
    return MechanismExtraction(
        record=record,
        step_scores=step_scores,
        chain_scores=chain_scores,
        token_scores=token_scores,
        n_steps=len(cache.step_token_ranges),
        step_token_ranges=list(cache.step_token_ranges),
        step_vectors=step_vectors,
        step_state_vectors=step_state_vectors,
        step_layer_state_vectors=step_layer_state_vectors,
        step_pre_state_vectors=step_pre_state_vectors,
        step_end_state_vectors=step_end_state_vectors,
        prompt_token_layer_states=prompt_token_layer_states,
        response_token_layer_states=response_token_layer_states,
        trace_input_ids=np.asarray(cache.input_ids, dtype=np.int64),
        trace_attention_mask=np.asarray(cache.attention_mask, dtype=np.int8),
        trace_token_offsets=np.asarray(cache.offset_mapping, dtype=np.int32),
        prompt_token_range=tuple(cache.prompt_token_range),
        question_token_range=tuple(cache.question_token_range),
        response_token_range=tuple(cache.response_token_range),
        rendered_prompt=prompt,
        response_text=response,
        layers=active_cfg.layers,
        metadata={
            "seq_len": int(cache.seq_len),
            "prompt_len_tokens": int(cache.prompt_len_tokens),
            "response_start_token": int(cache.response_start_token),
            "extractors": [e.name for e in extractors],
            "token_replay_kind": str(cache.replay_kind),
            "replay_protocol": str(cache.replay_protocol),
            "prompt_provenance": str(cache.prompt_provenance),
            "rendered_prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
            "messages_json": str(cache.messages_json),
            "retained_hidden_depths": list(cache.retained_hidden_depths),
            "retained_attention_blocks": list(cache.retained_attention_blocks),
            "hidden_state_depth_semantics": "depth_d_is_hidden_states_d_post_block",
            "block_update_semantics": "delta_depth_d_equals_h_d_minus_h_d_minus_1",
            "step_mean_state_temporal_semantics": "post_token_retrospective",
            "step_pre_state_temporal_semantics": "causal_before_first_step_token",
            "logit_prediction_semantics": "logits_i_predict_token_i_plus_1",
            "icr_semantics": (
                "attention_source_state_vs_total_block_update_proxy_not_exact_ov_attribution"
                if any(e.name == "icr" for e in extractors)
                else "not_extracted"
            ),
            "response_generator": record.generator or "",
            "source_model": record.source_model or "",
            "source_tokenizer": record.source_tokenizer or "",
            "source_model_revision": record.source_model_revision or "",
            "source_tokenizer_revision": record.source_tokenizer_revision or "",
            "loaded_model": str(getattr(getattr(model, "config", None), "_name_or_path", "")),
            "loaded_model_revision": str(getattr(getattr(model, "config", None), "_commit_hash", "") or ""),
            "loaded_tokenizer": str(getattr(tokenizer, "name_or_path", "")),
            "loaded_tokenizer_revision": str(getattr(tokenizer, "init_kwargs", {}).get("revision", "") or ""),
        },
    )


def extract_chain_prompt_flow(model, tokenizer, record: ChainRecord, cfg: ExtractionConfig) -> ChainExtraction | None:
    """Backward-compatible prompt-flow wrapper with optional uncertainty."""

    extractors: list[MechanismExtractor] = [PromptResidualFlowExtractor()]
    if cfg.include_entropy:
        extractors.append(UncertaintyExtractor())

    return extract_chain_mechanisms(
        model,
        tokenizer,
        record,
        cfg,
        extractors=extractors,
    )


def pack_extractions(extractions: Sequence[MechanismExtraction]) -> Dict[str, np.ndarray]:
    if not extractions:
        raise ValueError("no extractions to pack")

    expected_layers = tuple(int(x) for x in extractions[0].layers)
    for extraction in extractions:
        if tuple(int(x) for x in extraction.layers) != expected_layers:
            raise ValueError("all extractions must use the same declared layer axis")
        if int(extraction.n_steps) != len(extraction.step_token_ranges):
            raise ValueError(
                f"chain {extraction.record.chain_idx} has n_steps={extraction.n_steps} "
                f"but {len(extraction.step_token_ranges)} token ranges"
            )
        process_correct = extraction.record.process_correct
        if process_correct is not None and int(process_correct) >= 0:
            expected_process_correct = process_correct_from_gold(
                extraction.record.gold_error_step
            )
            if expected_process_correct < 0:
                raise ValueError(
                    f"chain {extraction.record.chain_idx} declares process_correct="
                    f"{process_correct} while gold_error_step is unavailable"
                )
            if int(process_correct) != expected_process_correct:
                raise ValueError(
                    f"chain {extraction.record.chain_idx} has inconsistent process "
                    f"label: gold_error_step={extraction.record.gold_error_step}, "
                    f"process_correct={process_correct}"
                )
    state_dtype = _state_storage_dtype(extractions)

    max_steps = max(e.n_steps for e in extractions)
    step_names = _ordered_union([e.step_scores.keys() for e in extractions], preferred=STEP_METRIC_NAMES)
    chain_names = _ordered_union([e.chain_scores.keys() for e in extractions])
    token_names = _ordered_union([e.token_scores.keys() for e in extractions])

    step_scores = np.full((len(extractions), max_steps, len(step_names)), np.nan, dtype=np.float32)
    chain_scores = np.full((len(extractions), len(chain_names)), np.nan, dtype=np.float32)
    step_ranges = np.full((len(extractions), max_steps, 2), -1, dtype=np.int32)

    flat_vectors = []
    flat_vector_chain_idx = []
    flat_vector_step_idx = []
    flat_state_vectors = []
    flat_layer_state_vectors = []
    flat_pre_state_vectors = []
    flat_end_state_vectors = []
    flat_state_vector_chain_idx = []
    flat_state_vector_step_idx = []
    flat_layer_state_vector_chain_idx = []
    flat_layer_state_vector_step_idx = []
    flat_pre_state_vector_chain_idx = []
    flat_pre_state_vector_step_idx = []
    flat_end_state_vector_chain_idx = []
    flat_end_state_vector_step_idx = []
    flat_token_scores = {name: [] for name in token_names}
    flat_token_chain_idx = []

    for i, e in enumerate(extractions):
        for k, name in enumerate(step_names):
            vals = np.asarray(e.step_scores.get(name, []), dtype=np.float32)
            step_scores[i, : min(vals.size, max_steps), k] = vals[:max_steps]
        for k, name in enumerate(chain_names):
            chain_scores[i, k] = float(e.chain_scores.get(name, np.nan))
        for j, (a, b) in enumerate(e.step_token_ranges[:max_steps]):
            step_ranges[i, j] = (int(a), int(b))
        if e.step_vectors is not None and e.step_vectors.size:
            v = np.asarray(e.step_vectors, dtype=np.float32)
            take = min(v.shape[0], e.n_steps)
            flat_vectors.append(v[:take])
            flat_vector_chain_idx.extend([e.record.chain_idx] * take)
            flat_vector_step_idx.extend(list(range(take)))
        if e.step_state_vectors is not None and e.step_state_vectors.size:
            v = np.asarray(e.step_state_vectors, dtype=np.float32)
            take = min(v.shape[0], e.n_steps)
            flat_state_vectors.append(v[:take])
            flat_state_vector_chain_idx.extend([e.record.chain_idx] * take)
            flat_state_vector_step_idx.extend(list(range(take)))
        if e.step_layer_state_vectors is not None and e.step_layer_state_vectors.size:
            v = np.asarray(e.step_layer_state_vectors, dtype=np.float32)
            take = min(v.shape[0], e.n_steps)
            flat_layer_state_vectors.append(v[:take])
            flat_layer_state_vector_chain_idx.extend([e.record.chain_idx] * take)
            flat_layer_state_vector_step_idx.extend(list(range(take)))
        if e.step_pre_state_vectors is not None and e.step_pre_state_vectors.size:
            v = np.asarray(e.step_pre_state_vectors, dtype=np.float32)
            take = min(v.shape[0], e.n_steps)
            flat_pre_state_vectors.append(v[:take])
            flat_pre_state_vector_chain_idx.extend([e.record.chain_idx] * take)
            flat_pre_state_vector_step_idx.extend(list(range(take)))
        if e.step_end_state_vectors is not None and e.step_end_state_vectors.size:
            v = np.asarray(e.step_end_state_vectors, dtype=np.float32)
            take = min(v.shape[0], e.n_steps)
            flat_end_state_vectors.append(v[:take])
            flat_end_state_vector_chain_idx.extend([e.record.chain_idx] * take)
            flat_end_state_vector_step_idx.extend(list(range(take)))
        if token_names:
            expected_tokens = max(
                0, int(e.response_token_range[1]) - int(e.response_token_range[0])
            )
            for name in token_names:
                vals = np.asarray(
                    e.token_scores.get(name, []), dtype=np.float32
                ).reshape(-1)
                if vals.size == 0:
                    vals = np.full(expected_tokens, np.nan, dtype=np.float32)
                elif vals.size != expected_tokens:
                    raise ValueError(
                        f"chain {e.record.chain_idx} token score {name!r} has "
                        f"{vals.size} values, expected {expected_tokens}"
                    )
                flat_token_scores[name].append(vals)
            flat_token_chain_idx.extend([e.record.chain_idx] * expected_tokens)

    trace_ids = [
        np.asarray([] if e.trace_input_ids is None else e.trace_input_ids, dtype=np.int64)
        for e in extractions
    ]
    trace_masks = [
        np.asarray(
            [] if e.trace_attention_mask is None else e.trace_attention_mask,
            dtype=np.int8,
        )
        for e in extractions
    ]
    trace_offsets = [
        np.asarray(
            [] if e.trace_token_offsets is None else e.trace_token_offsets,
            dtype=np.int32,
        ).reshape(-1, 2)
        for e in extractions
    ]
    prompt_ids = [ids[a:b] for ids, (a, b) in zip(trace_ids, [e.prompt_token_range for e in extractions])]
    prompt_masks = [mask[a:b] for mask, (a, b) in zip(trace_masks, [e.prompt_token_range for e in extractions])]
    response_ids = [ids[a:b] for ids, (a, b) in zip(trace_ids, [e.response_token_range for e in extractions])]
    response_masks = [mask[a:b] for mask, (a, b) in zip(trace_masks, [e.response_token_range for e in extractions])]

    packed = {
        "chain_idx": np.asarray([e.record.chain_idx for e in extractions], dtype=np.int64),
        "problem_id": np.asarray([e.record.problem_id for e in extractions], dtype=np.int64),
        "problem_ids": np.asarray([e.record.problem_id for e in extractions], dtype=np.int64),
        "problem_group_id": np.asarray(
            [e.record.problem_group_id for e in extractions], dtype=object
        ),
        "problem_group_ids": np.asarray(
            [e.record.problem_group_id for e in extractions], dtype=object
        ),
        "problems": _object_vector([e.record.problem for e in extractions]),
        "gold_error_step": np.asarray([e.record.gold_error_step for e in extractions], dtype=np.int64),
        "process_correct": np.asarray(
            [
                e.record.process_correct
                if e.record.process_correct is not None
                else -1
                for e in extractions
            ],
            dtype=np.int64,
        ),
        "final_answer_correct": np.asarray(
            [
                e.record.final_answer_correct
                if e.record.final_answer_correct is not None
                else -1
                for e in extractions
            ],
            dtype=np.int64,
        ),
        "is_correct": np.asarray([e.record.is_correct if e.record.is_correct is not None else -1 for e in extractions], dtype=np.int64),
        "sample_idx": np.asarray([e.record.sample_idx if e.record.sample_idx is not None else -1 for e in extractions], dtype=np.int64),
        "generator": np.asarray([e.record.generator or "" for e in extractions], dtype=object),
        "source_model": np.asarray(
            [
                str(
                    e.metadata.get("loaded_model")
                    or e.metadata.get("source_model")
                    or e.record.source_model
                    or ""
                )
                for e in extractions
            ],
            dtype=object,
        ),
        "source_tokenizer": np.asarray(
            [
                str(
                    e.metadata.get("loaded_tokenizer")
                    or e.metadata.get("source_tokenizer")
                    or e.record.source_tokenizer
                    or ""
                )
                for e in extractions
            ],
            dtype=object,
        ),
        "source_model_revision": np.asarray(
            [
                str(
                    e.metadata.get("loaded_model_revision")
                    or e.record.source_model_revision
                    or ""
                )
                for e in extractions
            ],
            dtype=object,
        ),
        "source_tokenizer_revision": np.asarray(
            [
                str(
                    e.metadata.get("loaded_tokenizer_revision")
                    or e.record.source_tokenizer_revision
                    or ""
                )
                for e in extractions
            ],
            dtype=object,
        ),
        "dataset": np.asarray([e.record.dataset or "" for e in extractions], dtype=object),
        "n_steps": np.asarray([e.n_steps for e in extractions], dtype=np.int64),
        "step_token_ranges": step_ranges,
        "prompt_token_ranges": np.asarray(
            [e.prompt_token_range for e in extractions], dtype=np.int32
        ),
        "question_token_ranges": np.asarray(
            [e.question_token_range for e in extractions], dtype=np.int32
        ),
        "response_token_ranges": np.asarray(
            [e.response_token_range for e in extractions], dtype=np.int32
        ),
        "rendered_prompts": _object_vector([e.rendered_prompt for e in extractions]),
        "responses": _object_vector(
            [e.response_text or e.record.response for e in extractions]
        ),
        "steps_text": _object_vector([list(e.record.steps) for e in extractions]),
        "prompt_token_ids": _object_vector(prompt_ids),
        "prompt_attention_mask": _object_vector(prompt_masks),
        "response_token_ids": _object_vector(response_ids),
        "response_attention_mask": _object_vector(response_masks),
        "input_ids": _object_vector(trace_ids),
        "attention_mask": _object_vector(trace_masks),
        "token_offsets": _object_vector(trace_offsets),
        "full_input_ids": _object_vector(trace_ids),
        "full_attention_mask": _object_vector(trace_masks),
        "full_token_offsets": _object_vector(trace_offsets),
        "prompt_token_counts": np.asarray(
            [max(0, int(b) - int(a)) for a, b in [e.prompt_token_range for e in extractions]],
            dtype=np.int64,
        ),
        "replay_protocol": np.asarray(
            [str(e.metadata.get("replay_protocol", "")) for e in extractions],
            dtype=object,
        ),
        "prompt_provenance": np.asarray(
            [str(e.metadata.get("prompt_provenance", "")) for e in extractions],
            dtype=object,
        ),
        "rendered_prompt_sha256": np.asarray(
            [
                str(e.metadata.get("rendered_prompt_sha256", ""))
                for e in extractions
            ],
            dtype=object,
        ),
        "messages_json": np.asarray(
            [str(e.metadata.get("messages_json", "")) for e in extractions],
            dtype=object,
        ),
        "token_replay_kind": np.asarray(
            [str(e.metadata.get("token_replay_kind", "")) for e in extractions],
            dtype=object,
        ),
        "trace_schema_version": np.asarray(TRACE_SCHEMA_VERSION),
        "trace_token_add_special_tokens": np.asarray(False),
        "token_offset_convention": np.asarray(TOKEN_OFFSET_CONVENTION),
        "step_token_range_convention": np.asarray(STEP_TOKEN_RANGE_CONVENTION),
        "span_range_convention": np.asarray(SPAN_RANGE_CONVENTION),
        "hidden_state_token_semantics": np.asarray("h_i_after_reading_token_i"),
        "logit_prediction_semantics": np.asarray("logits_i_predict_token_i_plus_1"),
        "step_prediction_position_shift": np.asarray(-1, dtype=np.int8),
        "data_contract_version": np.asarray(DATA_CONTRACT_VERSION),
        "label_semantics": np.asarray(
            "process_correct_and_final_answer_correct_are_distinct"
        ),
        "is_correct_semantics": np.asarray(
            "compatibility_alias_read_explicit_label_fields_first"
        ),
        "time_axis_kind": np.asarray("annotated_reasoning_step_index"),
        "step_scores": step_scores,
        "step_score_names": np.asarray(step_names, dtype="<U96"),
        "chain_scores": chain_scores,
        "chain_score_names": np.asarray(chain_names, dtype="<U96"),
        "layers": np.asarray(extractions[0].layers, dtype=np.int64),
        "state_representation_kind": np.asarray("hidden_state", dtype=object),
        "state_pooling_kind": np.asarray("arithmetic_mean_over_step_tokens", dtype=object),
        "state_storage_dtype": np.asarray(state_dtype, dtype=object),
        "metadata_json": np.asarray(
            [json.dumps(e.metadata, sort_keys=True, ensure_ascii=False) for e in extractions],
            dtype=object,
        ),
    }
    if flat_vectors:
        packed["step_vectors"] = cast_state_array(
            np.concatenate(flat_vectors, axis=0), state_dtype
        )
        packed["step_vector_chain_idx"] = np.asarray(flat_vector_chain_idx, dtype=np.int64)
        packed["step_vector_step_idx"] = np.asarray(flat_vector_step_idx, dtype=np.int64)
        packed["step_vector_layers"] = np.asarray(extractions[0].layers, dtype=np.int64)
    if flat_state_vectors:
        packed["step_state_vectors"] = cast_state_array(
            np.concatenate(flat_state_vectors, axis=0), state_dtype
        )
        packed["step_state_vector_chain_idx"] = np.asarray(flat_state_vector_chain_idx, dtype=np.int64)
        packed["step_state_vector_step_idx"] = np.asarray(flat_state_vector_step_idx, dtype=np.int64)
        packed["step_state_vector_layers"] = np.asarray(extractions[0].layers, dtype=np.int64)
    if flat_layer_state_vectors:
        packed["step_layer_state_vectors"] = cast_state_array(
            np.concatenate(flat_layer_state_vectors, axis=0), state_dtype
        )
        packed["step_layer_state_vector_chain_idx"] = np.asarray(flat_layer_state_vector_chain_idx, dtype=np.int64)
        packed["step_layer_state_vector_step_idx"] = np.asarray(flat_layer_state_vector_step_idx, dtype=np.int64)
        packed["step_layer_state_vector_layers"] = np.asarray(extractions[0].layers, dtype=np.int64)
    if flat_pre_state_vectors:
        packed["step_pre_state_vectors"] = cast_state_array(
            np.concatenate(flat_pre_state_vectors, axis=0), state_dtype
        )
        packed["step_pre_state_vector_chain_idx"] = np.asarray(flat_pre_state_vector_chain_idx, dtype=np.int64)
        packed["step_pre_state_vector_step_idx"] = np.asarray(flat_pre_state_vector_step_idx, dtype=np.int64)
        packed["step_pre_state_vector_layers"] = np.asarray(extractions[0].layers, dtype=np.int64)
    if flat_end_state_vectors:
        packed["step_end_state_vectors"] = cast_state_array(
            np.concatenate(flat_end_state_vectors, axis=0), state_dtype
        )
        packed["step_end_state_vector_chain_idx"] = np.asarray(flat_end_state_vector_chain_idx, dtype=np.int64)
        packed["step_end_state_vector_step_idx"] = np.asarray(flat_end_state_vector_step_idx, dtype=np.int64)
        packed["step_end_state_vector_layers"] = np.asarray(extractions[0].layers, dtype=np.int64)
    if token_names:
        packed["token_score_names"] = np.asarray(token_names, dtype="<U96")
        for name in token_names:
            packed[f"token_score_{name}"] = np.concatenate(flat_token_scores[name]).astype(np.float32)
        packed["token_score_chain_idx"] = np.asarray(flat_token_chain_idx, dtype=np.int64)
    if any("response_token_state_file" in e.metadata for e in extractions):
        packed["response_token_state_files"] = np.asarray(
            [str(e.metadata.get("response_token_state_file", "")) for e in extractions],
            dtype=object,
        )
        packed["response_token_state_counts"] = np.asarray(
            [int(e.metadata.get("response_token_state_count", 0)) for e in extractions],
            dtype=np.int64,
        )
        packed["response_token_state_layers"] = np.asarray(
            extractions[0].layers, dtype=np.int64
        )
        packed["response_token_state_storage_kind"] = np.asarray(
            "per_chain_npy_shards_v1", dtype=object
        )
    if any("prompt_token_state_file" in e.metadata for e in extractions):
        packed["prompt_token_state_files"] = np.asarray(
            [str(e.metadata.get("prompt_token_state_file", "")) for e in extractions],
            dtype=object,
        )
        packed["prompt_token_state_counts"] = np.asarray(
            [int(e.metadata.get("prompt_token_state_count", 0)) for e in extractions],
            dtype=np.int64,
        )
        packed["prompt_token_state_layers"] = np.asarray(
            extractions[0].layers, dtype=np.int64
        )
        packed["prompt_token_state_storage_kind"] = np.asarray(
            "per_chain_npy_shards_v1", dtype=object
        )
    return packed


def save_extractions(extractions: Sequence[MechanismExtraction], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("wb") as stream:
            np.savez_compressed(stream, **pack_extractions(extractions))
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _ordered_union(groups: Sequence[object], preferred: Sequence[str] = ()) -> list[str]:
    seen = set()
    out: list[str] = []
    for name in preferred:
        for group in groups:
            if name in group and name not in seen:
                seen.add(name)
                out.append(str(name))
                break
    for group in groups:
        for name in group:
            if name not in seen:
                seen.add(name)
                out.append(str(name))
    return out


def _object_vector(values: Sequence[object]) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    out[:] = list(values)
    return out


def _state_storage_dtype(extractions: Sequence[MechanismExtraction]) -> str:
    declared = {
        str(e.metadata.get("state_storage_dtype", "float16"))
        for e in extractions
    }
    if len(declared) != 1:
        raise ValueError(f"mixed state storage dtypes are not allowed: {sorted(declared)}")
    return declared.pop()
