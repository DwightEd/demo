from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Sequence

import numpy as np

from .artifact import BeliefTraceArtifact
from .model_capture import (
    SelectiveBoundaryCapture,
    parse_residual_depths,
    resolve_model_topology,
)

from .world import (
    ConstraintWorld,
    build_hypothesis_grid,
    constraint_mask,
    render_prefix_prompt,
)


@dataclass(frozen=True)
class ExtractionRow:
    row_id: int
    problem_id: int
    template_family: int
    prefix_index: int
    previous_prefix_index: int
    target_hypothesis: int
    feasible_mask: np.ndarray
    condition_mask: np.ndarray
    prompt_text: str

    @property
    def prompt_sha256(self) -> str:
        return sha256(self.prompt_text.encode("utf-8")).hexdigest()


def build_extraction_rows(
    worlds: Sequence[ConstraintWorld],
) -> tuple[list[ExtractionRow], np.ndarray]:
    if not worlds:
        raise ValueError("at least one constraint world is required")
    domain_sizes = {int(world.domain_size) for world in worlds}
    if len(domain_sizes) != 1:
        raise ValueError("all worlds in one trace must share a hypothesis universe")
    problem_ids = [int(world.problem_id) for world in worlds]
    if len(set(problem_ids)) != len(problem_ids):
        raise ValueError("problem_id must be unique before prefix expansion")
    domain_size = domain_sizes.pop()
    hypotheses = build_hypothesis_grid(domain_size)
    all_hypotheses = np.ones(len(hypotheses), dtype=bool)
    rows: list[ExtractionRow] = []
    for world in sorted(worlds, key=lambda item: item.problem_id):
        prefixes = world.prefix_states(hypotheses)
        target_hypothesis = world.target[0] * domain_size + world.target[1]
        for state in prefixes:
            condition = (
                all_hypotheses.copy()
                if state.condition is None
                else constraint_mask(state.condition, hypotheses)
            )
            rows.append(
                ExtractionRow(
                    row_id=len(rows),
                    problem_id=int(world.problem_id),
                    template_family=int(world.template_family),
                    prefix_index=int(state.prefix_index),
                    previous_prefix_index=int(state.prefix_index - 1),
                    target_hypothesis=int(target_hypothesis),
                    feasible_mask=np.asarray(state.feasible_mask, dtype=bool),
                    condition_mask=np.asarray(condition, dtype=bool),
                    prompt_text=render_prefix_prompt(world, state.prefix_index),
                )
            )
    return rows, hypotheses


def render_chat_observer_prompt(tokenizer, prompt_text: str) -> tuple[str, bool, str]:
    """Render an instruction-model readout boundary without token ambiguity."""

    messages = [
        {
            "role": "system",
            "content": (
                "Maintain the exact set of assignments consistent with the stated "
                "constraints. Do not discard a feasible assignment without evidence."
            ),
        },
        {"role": "user", "content": str(prompt_text)},
    ]
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    if callable(apply_template) and chat_template:
        rendered = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(rendered), False, "tokenizer_chat_template_generation_boundary"
    fallback = (
        "System: Maintain the exact feasible assignment set.\n"
        f"User: {prompt_text}\nAssistant:"
    )
    return fallback, True, "plain_text_generation_boundary"


def length_bucket_batches(
    token_lengths: Sequence[int],
    *,
    batch_size: int,
    max_batch_tokens: int,
) -> list[list[int]]:
    """Pack similarly sized prompts under both row and padded-token budgets."""

    if int(batch_size) < 1 or int(max_batch_tokens) < 1:
        raise ValueError("batch_size and max_batch_tokens must be positive")
    lengths = [int(value) for value in token_lengths]
    if any(value < 1 for value in lengths):
        raise ValueError("token lengths must be positive")
    ordered = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in ordered:
        candidate_max = max(current_max, lengths[index])
        candidate_size = len(current) + 1
        exceeds = (
            candidate_size > int(batch_size)
            or candidate_max * candidate_size > int(max_batch_tokens)
        )
        if current and exceeds:
            batches.append(current)
            current = []
            current_max = 0
        if lengths[index] > int(max_batch_tokens):
            raise ValueError(
                f"one prompt has {lengths[index]} tokens, above max_batch_tokens="
                f"{int(max_batch_tokens)}"
            )
        current.append(index)
        current_max = max(current_max, lengths[index])
    if current:
        batches.append(current)
    return batches


@dataclass(frozen=True)
class BoundaryExtractionConfig:
    layers: str = "8,12,16,20,24,28,32"
    batch_size: int = 16
    max_batch_tokens: int = 4096
    max_seq_len: int = 1024
    output_top_k: int = 64
    output_sketch_dim: int = 64
    output_sketch_seed: int = 991
    state_dtype: str = "float16"
    show_progress: bool = True

    def validate(self) -> None:
        if min(self.batch_size, self.max_batch_tokens, self.max_seq_len) < 1:
            raise ValueError("batch and sequence limits must be positive")
        if min(self.output_top_k, self.output_sketch_dim) < 1:
            raise ValueError("output_top_k and output_sketch_dim must be positive")
        if self.output_sketch_seed < 0:
            raise ValueError("output_sketch_seed must be non-negative")
        if self.state_dtype not in {"float16", "float32"}:
            raise ValueError("state_dtype must be float16 or float32")


def _encode_prompts(tokenizer, rows: Sequence[ExtractionRow], max_seq_len: int):
    prompts: list[str] = []
    input_ids: list[np.ndarray] = []
    protocols: set[str] = set()
    add_special_values: set[bool] = set()
    for row in rows:
        rendered, add_special_tokens, protocol = render_chat_observer_prompt(
            tokenizer, row.prompt_text
        )
        encoded = tokenizer(
            rendered,
            add_special_tokens=bool(add_special_tokens),
            truncation=False,
            return_attention_mask=False,
        )
        ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        if ids.ndim != 1 or len(ids) == 0:
            raise ValueError(f"problem {row.problem_id} produced invalid input_ids")
        if len(ids) > int(max_seq_len):
            raise ValueError(
                f"problem {row.problem_id} prefix {row.prefix_index} has {len(ids)} "
                f"tokens, above max_seq_len={int(max_seq_len)}"
            )
        prompts.append(rendered)
        input_ids.append(ids)
        protocols.add(protocol)
        add_special_values.add(bool(add_special_tokens))
    if len(protocols) != 1 or len(add_special_values) != 1:
        raise RuntimeError("observer rendering protocol changed within one extraction")
    return prompts, input_ids, protocols.pop(), add_special_values.pop()


def _last_visible_indices(attention_mask):
    import torch

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    masked = torch.where(attention_mask.bool(), positions, torch.full_like(positions, -1))
    last = masked.max(dim=1).values
    if torch.any(last < 0):
        raise ValueError("a padded batch contains an empty sequence")
    return last


def _build_logit_projection(logits, sketch_dim: int, seed: int):
    import torch

    generator = torch.Generator(device=logits.device)
    generator.manual_seed(int(seed))
    signs = torch.randint(
        0,
        2,
        (int(logits.shape[-1]), int(sketch_dim)),
        generator=generator,
        device=logits.device,
        dtype=torch.int8,
    )
    return (signs.float().mul_(2.0).sub_(1.0)) / np.sqrt(float(sketch_dim))


def _compact_output_statistics(logits, top_k: int, projection):
    import torch
    import torch.nn.functional as functional

    values = logits.float()
    log_z = torch.logsumexp(values, dim=-1)
    probability = torch.softmax(values, dim=-1)
    entropy = log_z - torch.sum(probability * values, dim=-1)
    k = min(max(int(top_k), 2), int(values.shape[-1]))
    top_values = torch.topk(values, k=k, dim=-1).values
    margin = top_values[:, 0] - top_values[:, 1]
    topk_mass = torch.exp(top_values - log_z[:, None]).sum(dim=-1)
    centered = values - torch.mean(values, dim=-1, keepdim=True)
    normalized = functional.normalize(centered, p=2.0, dim=-1)
    sketch = normalized @ projection
    return entropy, margin, topk_mass, sketch


def _cast_boundary_states(states, dtype: str) -> np.ndarray:
    values = states.detach().float().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("captured residual states contain non-finite values")
    if dtype == "float16":
        limit = np.finfo(np.float16).max
        if np.max(np.abs(values), initial=0.0) > limit:
            raise OverflowError("captured residual state cannot be represented as float16")
        return values.astype(np.float16)
    return values.astype(np.float32)


def extract_boundary_belief_trace(
    model,
    tokenizer,
    worlds: Sequence[ConstraintWorld],
    cfg: BoundaryExtractionConfig,
    *,
    metadata: dict[str, Any] | None = None,
) -> BeliefTraceArtifact:
    """Extract exact prefix-boundary states with bounded accelerator memory."""

    import torch

    cfg.validate()
    rows, hypotheses = build_extraction_rows(worlds)
    prompts, input_ids, protocol, add_special_tokens = _encode_prompts(
        tokenizer, rows, cfg.max_seq_len
    )
    topology = resolve_model_topology(model)
    depths = parse_residual_depths(cfg.layers, topology.num_depths)
    hidden_size = int(
        getattr(model.config, "hidden_size", 0)
        or getattr(model.config, "n_embd", 0)
    )
    if hidden_size < 1:
        raise TypeError("model config does not expose hidden_size")
    state_np_dtype = np.float16 if cfg.state_dtype == "float16" else np.float32
    states = np.empty((len(rows), len(depths), hidden_size), dtype=state_np_dtype)
    output_entropy = np.empty(len(rows), dtype=np.float32)
    output_margin = np.empty(len(rows), dtype=np.float32)
    output_topk_mass = np.empty(len(rows), dtype=np.float32)
    output_logit_sketch = np.empty(
        (len(rows), int(cfg.output_sketch_dim)), dtype=np.float32
    )
    device = next(model.parameters()).device
    batches = length_bucket_batches(
        [len(ids) for ids in input_ids],
        batch_size=cfg.batch_size,
        max_batch_tokens=cfg.max_batch_tokens,
    )
    iterator = batches
    if cfg.show_progress:
        from tqdm import tqdm

        iterator = tqdm(batches, desc="belief boundary batches")
    model.eval()
    logit_projection = None
    with torch.inference_mode():
        for batch_indices in iterator:
            features = [
                {
                    "input_ids": input_ids[index].tolist(),
                    "attention_mask": [1] * len(input_ids[index]),
                }
                for index in batch_indices
            ]
            padded = tokenizer.pad(
                features,
                padding=True,
                pad_to_multiple_of=8 if device.type == "cuda" else None,
                return_tensors="pt",
            )
            model_inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in padded.items()
                if key in {"input_ids", "attention_mask"}
            }
            last_indices = _last_visible_indices(model_inputs["attention_mask"])
            with SelectiveBoundaryCapture(topology, depths, last_indices) as capture:
                topology.backbone(
                    **model_inputs,
                    use_cache=False,
                    return_dict=True,
                )
                boundary = capture.states()
                final_state = capture.final_state()
            logits = topology.output_head(final_state)
            if logit_projection is None:
                logit_projection = _build_logit_projection(
                    logits, cfg.output_sketch_dim, cfg.output_sketch_seed
                )
            entropy, margin, topk_mass, logit_sketch = _compact_output_statistics(
                logits, cfg.output_top_k, logit_projection
            )
            batch_states = _cast_boundary_states(boundary, cfg.state_dtype)
            batch_rows = np.asarray(batch_indices, dtype=np.int64)
            states[batch_rows] = batch_states
            output_entropy[batch_rows] = entropy.detach().cpu().numpy()
            output_margin[batch_rows] = margin.detach().cpu().numpy()
            output_topk_mass[batch_rows] = topk_mass.detach().cpu().numpy()
            output_logit_sketch[batch_rows] = logit_sketch.detach().cpu().numpy()

    trace_metadata = dict(metadata or {})
    trace_metadata.update(
        observer_protocol=protocol,
        add_special_tokens=bool(add_special_tokens),
        num_residual_depths=int(topology.num_depths),
        selected_depths=list(depths),
        state_dtype=cfg.state_dtype,
        output_top_k=int(cfg.output_top_k),
        output_sketch_dim=int(cfg.output_sketch_dim),
        output_sketch_seed=int(cfg.output_sketch_seed),
        max_seq_len=int(cfg.max_seq_len),
        extraction_mode="selective_boundary_hooks_no_full_hidden_history",
    )
    return BeliefTraceArtifact.from_rows(
        rows,
        hypotheses=hypotheses,
        layers=np.asarray(depths, dtype=np.int64),
        states=states,
        prompts=prompts,
        input_ids=input_ids,
        output_entropy=output_entropy,
        output_margin=output_margin,
        output_topk_mass=output_topk_mass,
        output_logit_sketch=output_logit_sketch,
        metadata=trace_metadata,
    )
