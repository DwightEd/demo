from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Sequence

import numpy as np
import torch

from prompt_control_flow.data import ChainRecord
from prompt_control_flow.teacher_forcing import (
    build_prompt_response,
    prepare_teacher_forcing_trace,
)

from .logit_trace import (
    LogitTraceConfig,
    aggregate_token_features_to_steps,
    compact_features_from_logits,
    step_feature_names,
    token_feature_names,
)
from .schema import CompactTraceItem, TraceArtifact


@dataclass
class CompactTraceAccumulator:
    cfg: LogitTraceConfig
    items: list[CompactTraceItem] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def add(self, item: CompactTraceItem) -> None:
        item.validate(
            len(token_feature_names(self.cfg.sketch_dim)),
            len(step_feature_names(self.cfg.sketch_dim)),
        )
        self.items.append(item)

    def add_skip(self, record: ChainRecord, exc: BaseException) -> None:
        self.skipped.append(
            {
                "chain_idx": int(record.chain_idx),
                "problem_id": int(record.problem_id),
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )

    def pack(self, metadata: dict[str, Any] | None = None) -> TraceArtifact:
        return TraceArtifact.from_items(
            self.items,
            token_feature_names=token_feature_names(self.cfg.sketch_dim),
            step_feature_names=step_feature_names(self.cfg.sketch_dim),
            metadata={
                **dict(metadata or {}),
                "top_k": int(self.cfg.top_k),
                "sketch_dim": int(self.cfg.sketch_dim),
                "token_chunk_size": int(self.cfg.token_chunk_size),
                "num_skipped": int(len(self.skipped)),
            },
        )


def _causal_backbone(model):
    prefix = str(getattr(model, "base_model_prefix", "") or "")
    backbone = getattr(model, prefix, None) if prefix else None
    if backbone is None or backbone is model:
        backbone = getattr(model, "model", None)
    if backbone is None or backbone is model:
        backbone = getattr(model, "transformer", None)
    if backbone is None or backbone is model:
        raise TypeError(
            "cannot locate the causal-LM backbone; expected base_model_prefix, model, or transformer"
        )
    output_head = model.get_output_embeddings()
    if output_head is None:
        raise TypeError("model.get_output_embeddings() returned None")
    return backbone, output_head


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:  # pragma: no cover - invalid model object
        raise ValueError("model has no parameters") from exc


def _autocast_context(device: torch.device, model) -> torch.autocast:
    dtype = next(model.parameters()).dtype
    enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


@torch.inference_mode()
def extract_compact_trace_item(
    model,
    tokenizer,
    record: ChainRecord,
    *,
    cfg: LogitTraceConfig,
    max_seq_len: int,
) -> CompactTraceItem:
    """Replay one response and retain only compact causal output features.

    The backbone is evaluated once.  The vocabulary projection is then applied
    only to response prediction positions in small GPU chunks, avoiding a
    persistent ``[sequence, vocabulary]`` tensor.
    """

    cfg.validate()
    if record.rendered_prompt is not None:
        prompt = str(record.rendered_prompt)
        response = str(record.response)
    else:
        if not record.problem and record.exact_input_ids is None:
            raise ValueError(
                "record has neither problem text nor an exact token trace; replay would omit the prompt"
            )
        prompt, fallback_response = build_prompt_response(record.problem, record.steps)
        response = str(record.response or fallback_response)

    trace = prepare_teacher_forcing_trace(
        tokenizer,
        prompt,
        response,
        steps=record.steps,
        max_seq_len=int(max_seq_len),
        exact_input_ids=record.exact_input_ids,
        exact_attention_mask=record.exact_attention_mask,
        exact_token_offsets=record.exact_token_offsets,
        exact_step_token_ranges=record.exact_step_token_ranges,
        exact_response_start_token=record.exact_response_start_token,
    )
    response_start = int(trace["response_start_token"])
    step_ranges = [(int(a), int(b)) for a, b in trace["step_token_ranges"]]
    response_stop = max(b for _, b in step_ranges)
    if response_start <= 0:
        raise ValueError(
            "response_start_token must be positive for causal next-token scoring"
        )
    if response_stop < response_start:
        raise ValueError("response token span is empty")

    device = _model_device(model)
    input_ids = torch.as_tensor(trace["input_ids"], dtype=torch.long, device=device)
    attention_mask = torch.as_tensor(
        trace["attention_mask"], dtype=torch.long, device=device
    )
    backbone, output_head = _causal_backbone(model)
    with _autocast_context(device, model):
        base_output = backbone(
            input_ids=input_ids[None, :],
            attention_mask=attention_mask[None, :],
            use_cache=False,
            return_dict=True,
        )
        hidden = base_output.last_hidden_state[0]

    target_positions = torch.arange(
        response_start,
        response_stop + 1,
        device=device,
        dtype=torch.long,
    )
    target_ids = input_ids.index_select(0, target_positions)
    prediction_positions = target_positions - 1
    chunks: list[torch.Tensor] = []
    previous_logits: torch.Tensor | None = None
    final_logits_bias = getattr(model, "final_logits_bias", None)
    for start in range(0, len(target_positions), int(cfg.token_chunk_size)):
        stop = min(len(target_positions), start + int(cfg.token_chunk_size))
        with _autocast_context(device, model):
            logits = output_head(
                hidden.index_select(0, prediction_positions[start:stop])
            )
            if final_logits_bias is not None:
                logits = logits + final_logits_bias.to(
                    device=logits.device, dtype=logits.dtype
                )
        compact, previous_logits = compact_features_from_logits(
            logits,
            target_ids[start:stop],
            cfg,
            previous_logits=previous_logits,
        )
        chunks.append(compact.detach().cpu())
        del logits, compact

    token_features = torch.cat(chunks, dim=0).numpy().astype(np.float32, copy=False)
    step_features, relative_ranges = aggregate_token_features_to_steps(
        token_features,
        step_ranges,
        response_start_token=response_start,
    )
    item = CompactTraceItem(
        chain_idx=int(record.chain_idx),
        problem_id=int(record.problem_id),
        gold_error_step=int(record.gold_error_step),
        is_correct=int(record.is_correct) if record.is_correct is not None else -1,
        sample_idx=int(record.sample_idx) if record.sample_idx is not None else -1,
        dataset=str(record.dataset or ""),
        generator=str(record.generator or ""),
        response_hash=hashlib.sha256(response.encode("utf-8")).hexdigest(),
        token_ids=target_ids.detach().cpu().numpy().astype(np.int64, copy=False),
        token_features=token_features,
        step_features=step_features,
        step_token_ranges=relative_ranges,
        replay_kind=str(trace["replay_kind"]),
        metadata={
            "sequence_tokens": int(len(input_ids)),
            "response_tokens": int(len(target_ids)),
            "vocabulary_size": int(output_head.weight.shape[0]),
        },
    )
    item.validate(
        len(token_feature_names(cfg.sketch_dim)),
        len(step_feature_names(cfg.sketch_dim)),
    )
    del base_output, hidden, target_ids, input_ids, attention_mask
    return item
