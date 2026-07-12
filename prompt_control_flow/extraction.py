from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .config import ExtractionConfig, STEP_METRIC_NAMES
from .data import ChainRecord
from .extractors import MechanismExtractor, PromptResidualFlowExtractor
from .metrics import (
    compute_step_layer_state_vectors,
    compute_step_residual_vectors,
    summarize_step_metrics,
)
from .teacher_forcing import ForwardCache, build_prompt_response, run_teacher_forcing


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

    if not extractors and not cfg.store_step_state_vectors:
        raise ValueError("at least one extractor or store_step_state_vectors=True is required")
    default_prompt, default_response = build_prompt_response(record.problem, record.steps)
    prompt = record.rendered_prompt if record.rendered_prompt is not None else default_prompt
    response = record.response if record.response else default_response
    if not prompt.strip() or not response.strip():
        return None

    need_hidden = any(e.requires_hidden for e in extractors) or bool(cfg.store_step_vectors) or bool(cfg.store_step_state_vectors)
    need_attention = any(e.requires_attention for e in extractors)
    need_logits = any(e.requires_logits for e in extractors) or bool(cfg.include_entropy)

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
    )
    if len(cache.step_token_ranges) == 0:
        return None

    # An empty layer tuple is the explicit ``all post-block depths`` sentinel.
    # Hugging Face hidden_states[0] is the embedding output; indices 1..N are
    # the outputs after transformer blocks 1..N.
    resolved_layers = tuple(int(x) for x in cfg.layers)
    if not resolved_layers and cache.hidden_states is not None:
        resolved_layers = tuple(range(1, len(cache.hidden_states)))
    active_cfg = replace(cfg, layers=resolved_layers)

    step_scores: Dict[str, np.ndarray] = {}
    token_scores: Dict[str, np.ndarray] = {}
    for extractor in extractors:
        scores = extractor.compute(cache, record, active_cfg)
        for name, values in scores.items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim == 1 and arr.shape[0] == len(cache.step_token_ranges):
                step_scores[name] = arr
            else:
                token_scores[name] = arr

    step_vectors = None
    if cfg.store_step_vectors and cache.hidden_states is not None:
        hidden = [np.asarray(h, dtype=np.float32) for h in cache.hidden_states]
        step_vectors = compute_step_residual_vectors(
            hidden,
            step_ranges=cache.step_token_ranges,
            layers=active_cfg.layers,
        )
    step_state_vectors = None
    step_layer_state_vectors = None
    if cfg.store_step_state_vectors and cache.hidden_states is not None:
        hidden = [np.asarray(h, dtype=np.float32) for h in cache.hidden_states]
        step_layer_state_vectors = compute_step_layer_state_vectors(
            hidden,
            step_ranges=cache.step_token_ranges,
            layers=active_cfg.layers,
        ).astype(np.float16)
        if cfg.store_flat_step_state_vectors:
            step_state_vectors = step_layer_state_vectors.reshape(step_layer_state_vectors.shape[0], -1)

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
        layers=active_cfg.layers,
        metadata={
            "seq_len": int(cache.seq_len),
            "prompt_len_tokens": int(cache.prompt_len_tokens),
            "response_start_token": int(cache.response_start_token),
            "extractors": [e.name for e in extractors],
            "token_replay_kind": str(cache.replay_kind),
            "source_model": record.generator or "",
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
    """Backward-compatible hidden-only prompt-flow extraction wrapper."""

    return extract_chain_mechanisms(
        model,
        tokenizer,
        record,
        cfg,
        extractors=[PromptResidualFlowExtractor()],
    )


def pack_extractions(extractions: Sequence[MechanismExtraction]) -> Dict[str, np.ndarray]:
    if not extractions:
        raise ValueError("no extractions to pack")

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
    flat_state_vector_chain_idx = []
    flat_state_vector_step_idx = []
    flat_layer_state_vector_chain_idx = []
    flat_layer_state_vector_step_idx = []
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
            v = np.asarray(e.step_layer_state_vectors, dtype=np.float16)
            take = min(v.shape[0], e.n_steps)
            flat_layer_state_vectors.append(v[:take])
            flat_layer_state_vector_chain_idx.extend([e.record.chain_idx] * take)
            flat_layer_state_vector_step_idx.extend(list(range(take)))
        for name in token_names:
            vals = np.asarray(e.token_scores.get(name, []), dtype=np.float32).reshape(-1)
            flat_token_scores[name].append(vals)
        if token_names:
            max_tok = max((np.asarray(e.token_scores.get(name, []), dtype=np.float32).size for name in token_names), default=0)
            flat_token_chain_idx.extend([e.record.chain_idx] * max_tok)

    packed = {
        "chain_idx": np.asarray([e.record.chain_idx for e in extractions], dtype=np.int64),
        "problem_id": np.asarray([e.record.problem_id for e in extractions], dtype=np.int64),
        "gold_error_step": np.asarray([e.record.gold_error_step for e in extractions], dtype=np.int64),
        "is_correct": np.asarray([e.record.is_correct if e.record.is_correct is not None else -1 for e in extractions], dtype=np.int64),
        "sample_idx": np.asarray([e.record.sample_idx if e.record.sample_idx is not None else -1 for e in extractions], dtype=np.int64),
        "generator": np.asarray([e.record.generator or "" for e in extractions], dtype=object),
        "dataset": np.asarray([e.record.dataset or "" for e in extractions], dtype=object),
        "n_steps": np.asarray([e.n_steps for e in extractions], dtype=np.int64),
        "step_token_ranges": step_ranges,
        "step_scores": step_scores,
        "step_score_names": np.asarray(step_names, dtype="<U96"),
        "chain_scores": chain_scores,
        "chain_score_names": np.asarray(chain_names, dtype="<U96"),
        "layers": np.asarray(extractions[0].layers, dtype=np.int64),
        "state_representation_kind": np.asarray("hidden_state", dtype=object),
        "state_pooling_kind": np.asarray("arithmetic_mean_over_step_tokens", dtype=object),
        "metadata_json": np.asarray(
            [json.dumps(e.metadata, sort_keys=True, ensure_ascii=False) for e in extractions],
            dtype=object,
        ),
    }
    if flat_vectors:
        packed["step_vectors"] = np.concatenate(flat_vectors, axis=0).astype(np.float16)
        packed["step_vector_chain_idx"] = np.asarray(flat_vector_chain_idx, dtype=np.int64)
        packed["step_vector_step_idx"] = np.asarray(flat_vector_step_idx, dtype=np.int64)
        packed["step_vector_layers"] = np.asarray(extractions[0].layers, dtype=np.int64)
    if flat_state_vectors:
        packed["step_state_vectors"] = np.concatenate(flat_state_vectors, axis=0).astype(np.float16)
        packed["step_state_vector_chain_idx"] = np.asarray(flat_state_vector_chain_idx, dtype=np.int64)
        packed["step_state_vector_step_idx"] = np.asarray(flat_state_vector_step_idx, dtype=np.int64)
        packed["step_state_vector_layers"] = np.asarray(extractions[0].layers, dtype=np.int64)
    if flat_layer_state_vectors:
        packed["step_layer_state_vectors"] = np.concatenate(flat_layer_state_vectors, axis=0)
        packed["step_layer_state_vector_chain_idx"] = np.asarray(flat_layer_state_vector_chain_idx, dtype=np.int64)
        packed["step_layer_state_vector_step_idx"] = np.asarray(flat_layer_state_vector_step_idx, dtype=np.int64)
        packed["step_layer_state_vector_layers"] = np.asarray(extractions[0].layers, dtype=np.int64)
    if token_names:
        packed["token_score_names"] = np.asarray(token_names, dtype="<U96")
        for name in token_names:
            packed[f"token_score_{name}"] = np.concatenate(flat_token_scores[name]).astype(np.float32)
        packed["token_score_chain_idx"] = np.asarray(flat_token_chain_idx, dtype=np.int64)
    return packed


def save_extractions(extractions: Sequence[MechanismExtraction], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pack_extractions(extractions))


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
