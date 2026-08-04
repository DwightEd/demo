from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Sequence

import numpy as np

from prompt_control_flow.belief_transport.extraction import length_bucket_batches
from prompt_control_flow.belief_transport.model_capture import (
    SelectiveBoundaryCapture,
    parse_residual_depths,
    resolve_model_topology,
)

from .data import AliasObservation, build_alias_observations
from .schema import CausalBeliefTrace
from .world import PredictiveAliasWorld


SYSTEM_MESSAGE = (
    "Maintain the exact set of assignments consistent with every modular "
    "constraint. Treat the listed constraints as authoritative and answer the "
    "requested residue only."
)


@dataclass(frozen=True)
class StateExtractionConfig:
    layers: str = "0,4,8,12,16,20,24,28,32"
    batch_size: int = 16
    max_batch_tokens: int = 4096
    max_seq_len: int = 1024
    logit_sketch_dim: int = 256
    logit_sketch_seed: int = 1701
    state_dtype: str = "float16"
    show_progress: bool = True

    def validate(self) -> None:
        if min(self.batch_size, self.max_batch_tokens, self.max_seq_len) < 1:
            raise ValueError("batch and sequence limits must be positive")
        if int(self.logit_sketch_dim) < 1:
            raise ValueError("logit_sketch_dim must be positive")
        if int(self.logit_sketch_seed) < 0:
            raise ValueError("logit_sketch_seed must be non-negative")
        if self.state_dtype not in {"float16", "float32"}:
            raise ValueError("state_dtype must be float16 or float32")


@dataclass(frozen=True)
class EncodedObservation:
    rendered_prompt: str
    input_ids: np.ndarray
    evidence_token_range: tuple[int, int]
    protocol: str
    add_special_tokens: bool


def render_chat_prompt(tokenizer, user_text: str) -> tuple[str, bool, str]:
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": str(user_text)},
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
    fallback = f"System: {SYSTEM_MESSAGE}\nUser: {user_text}\nAssistant:"
    return fallback, True, "plain_text_generation_boundary"


def _token_range_for_char_span(
    offsets: Sequence[tuple[int, int]],
    char_start: int,
    char_stop: int,
) -> tuple[int, int]:
    indices = [
        index
        for index, (start, stop) in enumerate(offsets)
        if stop > start and stop > int(char_start) and start < int(char_stop)
    ]
    if not indices:
        raise ValueError("evidence sentence did not align to any visible token")
    return int(indices[0]), int(indices[-1] + 1)


def encode_observations(
    tokenizer,
    observations: Sequence[AliasObservation],
    *,
    max_seq_len: int,
) -> list[EncodedObservation]:
    encoded_rows: list[EncodedObservation] = []
    protocols: set[str] = set()
    special_modes: set[bool] = set()
    for row in observations:
        rendered, add_special_tokens, protocol = render_chat_prompt(
            tokenizer, row.user_text
        )
        char_start = rendered.rfind(row.branch_evidence_text)
        if char_start < 0:
            raise ValueError(
                f"pair {row.pair_id} branch evidence is absent after chat rendering"
            )
        char_stop = char_start + len(row.branch_evidence_text)
        encoded = tokenizer(
            rendered,
            add_special_tokens=bool(add_special_tokens),
            truncation=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        if "offset_mapping" not in encoded:
            raise TypeError("exact evidence alignment requires a fast tokenizer")
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        offsets = [(int(start), int(stop)) for start, stop in encoded["offset_mapping"]]
        if input_ids.ndim != 1 or len(input_ids) < 1:
            raise ValueError("tokenizer produced an invalid one-dimensional input")
        if len(input_ids) > int(max_seq_len):
            raise ValueError(
                f"pair {row.pair_id} query {row.query_role} has {len(input_ids)} "
                f"tokens, above max_seq_len={int(max_seq_len)}"
            )
        evidence_range = _token_range_for_char_span(offsets, char_start, char_stop)
        encoded_rows.append(
            EncodedObservation(
                rendered_prompt=rendered,
                input_ids=input_ids,
                evidence_token_range=evidence_range,
                protocol=protocol,
                add_special_tokens=bool(add_special_tokens),
            )
        )
        protocols.add(protocol)
        special_modes.add(bool(add_special_tokens))
    if len(protocols) != 1 or len(special_modes) != 1:
        raise RuntimeError("observer rendering protocol changed within one trace")
    return encoded_rows


def resolve_residue_token_groups(
    tokenizer, modulus: int
) -> tuple[tuple[int, ...], ...]:
    token_groups: list[tuple[int, ...]] = []
    for residue in range(int(modulus)):
        selected: list[int] = []
        for candidate in (str(residue), f" {residue}", f"\n{residue}"):
            encoded = tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if len(encoded) == 1:
                selected.append(int(encoded[0]))
        group = tuple(sorted(set(selected)))
        if not group:
            raise ValueError(
                f"residue {residue} is not a single token; this model requires a "
                "sequence-likelihood extractor instead of boundary logits"
            )
        token_groups.append(group)
    flattened = [token for group in token_groups for token in group]
    if len(set(flattened)) != len(flattened):
        raise ValueError("residue surface forms map to overlapping token groups")
    return tuple(token_groups)


def aggregate_residue_logits(logits, token_groups: Sequence[Sequence[int]]):
    import torch

    if logits.ndim != 2:
        raise ValueError("boundary logits must have shape [batch, vocabulary]")
    return torch.stack(
        [
            torch.logsumexp(logits[:, [int(token) for token in group]].float(), dim=1)
            for group in token_groups
        ],
        dim=1,
    )


def _last_visible_indices(attention_mask):
    import torch

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    masked = torch.where(attention_mask.bool(), positions, torch.full_like(positions, -1))
    last = masked.max(dim=1).values
    if torch.any(last < 0):
        raise ValueError("a padded batch contains an empty sequence")
    return last


def _logit_projection(logits, dimension: int, seed: int):
    import torch

    generator = torch.Generator(device=logits.device)
    generator.manual_seed(int(seed))
    signs = torch.randint(
        0,
        2,
        (int(logits.shape[-1]), int(dimension)),
        generator=generator,
        device=logits.device,
        dtype=torch.int8,
    )
    return signs.float().mul_(2.0).sub_(1.0).div_(np.sqrt(float(dimension)))


def _sketch_logits(logits, projection):
    import torch.nn.functional as functional

    centered = logits.float() - logits.float().mean(dim=-1, keepdim=True)
    return functional.normalize(centered, p=2.0, dim=-1) @ projection


def _cast_states(states, dtype: str) -> np.ndarray:
    values = states.detach().float().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("captured residual states contain non-finite values")
    if dtype == "float16":
        if np.max(np.abs(values), initial=0.0) > np.finfo(np.float16).max:
            raise OverflowError("captured residual states overflow float16")
        return values.astype(np.float16)
    return values.astype(np.float32)


def extract_causal_belief_states(
    model,
    tokenizer,
    worlds: Sequence[PredictiveAliasWorld],
    cfg: StateExtractionConfig,
    *,
    metadata: dict[str, Any] | None = None,
) -> CausalBeliefTrace:
    """Extract compact all-layer boundary states without storing token histories."""

    import torch

    cfg.validate()
    observations, frequencies = build_alias_observations(worlds)
    encoded_rows = encode_observations(
        tokenizer, observations, max_seq_len=cfg.max_seq_len
    )
    topology = resolve_model_topology(model)
    depths = parse_residual_depths(cfg.layers, topology.num_depths)
    hidden_size = int(
        getattr(model.config, "hidden_size", 0)
        or getattr(model.config, "n_embd", 0)
    )
    if hidden_size < 1:
        raise TypeError("model config does not expose hidden_size")
    state_dtype = np.float16 if cfg.state_dtype == "float16" else np.float32
    n_rows = len(observations)
    modulus = int(observations[0].modulus)
    states = np.empty((n_rows, len(depths), hidden_size), dtype=state_dtype)
    residue_logits = np.empty((n_rows, modulus), dtype=np.float32)
    logit_sketch = np.empty((n_rows, cfg.logit_sketch_dim), dtype=np.float32)
    residue_token_groups = resolve_residue_token_groups(tokenizer, modulus)
    device = next(model.parameters()).device
    batches = length_bucket_batches(
        [len(row.input_ids) for row in encoded_rows],
        batch_size=cfg.batch_size,
        max_batch_tokens=cfg.max_batch_tokens,
    )
    iterator = batches
    if cfg.show_progress:
        from tqdm import tqdm

        iterator = tqdm(batches, desc="causal belief state batches")
    projection = None
    model.eval()
    with torch.inference_mode():
        for batch_indices in iterator:
            features = [
                {
                    "input_ids": encoded_rows[index].input_ids.tolist(),
                    "attention_mask": [1] * len(encoded_rows[index].input_ids),
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
                boundary_states = capture.states()
                final_state = capture.final_state()
            boundary_logits = topology.output_head(final_state).float()
            if projection is None:
                projection = _logit_projection(
                    boundary_logits, cfg.logit_sketch_dim, cfg.logit_sketch_seed
                )
            row_indices = np.asarray(batch_indices, dtype=np.int64)
            states[row_indices] = _cast_states(boundary_states, cfg.state_dtype)
            residue_logits[row_indices] = (
                aggregate_residue_logits(boundary_logits, residue_token_groups)
                .detach()
                .cpu()
                .numpy()
            )
            logit_sketch[row_indices] = (
                _sketch_logits(boundary_logits, projection).detach().cpu().numpy()
            )

    trace_metadata = dict(metadata or {})
    rendered_protocols = {row.protocol for row in encoded_rows}
    trace_metadata.update(
        observer_protocol=next(iter(rendered_protocols)),
        system_message_sha256=sha256(SYSTEM_MESSAGE.encode("utf-8")).hexdigest(),
        num_residual_depths=int(topology.num_depths),
        selected_depths=list(depths),
        residue_token_ids=[int(group[0]) for group in residue_token_groups],
        residue_token_id_groups=[list(group) for group in residue_token_groups],
        state_dtype=cfg.state_dtype,
        logit_sketch_dim=int(cfg.logit_sketch_dim),
        logit_sketch_seed=int(cfg.logit_sketch_seed),
        extraction_mode="selective_boundary_predictive_alias_v1",
    )
    return CausalBeliefTrace.from_observations(
        observations,
        frequencies=frequencies,
        layers=np.asarray(depths, dtype=np.int64),
        states=states,
        residue_logits=residue_logits,
        logit_sketch=logit_sketch,
        rendered_prompts=[row.rendered_prompt for row in encoded_rows],
        input_ids=[row.input_ids for row in encoded_rows],
        evidence_token_ranges=np.asarray(
            [row.evidence_token_range for row in encoded_rows], dtype=np.int64
        ),
        metadata=trace_metadata,
    )
