from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator
from typing import Any, Generic, Sequence, Tuple, TypeVar, overload

import numpy as np

from utils.step_boundaries import TokenAlignmentError, char_span_to_token_range


T = TypeVar("T")


class SparseLayerSequence(Sequence[T], Generic[T]):
    """Index-preserving sparse view over model layer/depth outputs."""

    def __init__(self, total: int, values: dict[int, T]) -> None:
        self._total = int(total)
        self._values = dict(values)

    def __len__(self) -> int:
        return self._total

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: int | slice) -> T | list[T]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._total))]
        normalized = int(index)
        if normalized < 0:
            normalized += self._total
        if not 0 <= normalized < self._total:
            raise IndexError(normalized)
        if normalized not in self._values:
            raise IndexError(
                f"layer/depth {normalized} was not retained; available={sorted(self._values)}"
            )
        return self._values[normalized]

    def __iter__(self) -> Iterator[T]:
        for index in sorted(self._values):
            yield self._values[index]

    @property
    def retained_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._values))


@dataclass
class ForwardCache:
    input_ids: "object"
    attention_mask: "object"
    offset_mapping: Sequence[Tuple[int, int]]
    prompt_len_tokens: int
    prompt_token_range: tuple[int, int]
    question_token_range: tuple[int, int]
    response_start_token: int
    response_token_range: tuple[int, int]
    step_token_ranges: list[tuple[int, int]]
    hidden_states: Sequence["object"] | None = None
    attentions: Sequence["object"] | None = None
    logits: "object" | None = None
    token_output_summaries: dict[str, "object"] = field(default_factory=dict)
    seq_len: int = 0
    replay_kind: str = ""
    replay_protocol: str = ""
    prompt_provenance: str = ""
    messages_json: str = ""
    retained_hidden_depths: tuple[int, ...] = ()
    retained_attention_blocks: tuple[int, ...] = ()

    @property
    def prompt_content_token_indices(self) -> np.ndarray:
        return visible_token_indices(self.offset_mapping, self.prompt_token_range)

    @property
    def question_token_indices(self) -> np.ndarray:
        return visible_token_indices(self.offset_mapping, self.question_token_range)


def build_prompt_response(problem: str, steps: Sequence[str]) -> tuple[str, str]:
    prompt = f"Problem: {problem}\n\nSolution:\n\n"
    response = "\n\n".join(str(s) for s in steps)
    return prompt, response


def prompt_token_indices(offsets: Sequence[Tuple[int, int]], prompt_len_chars: int) -> np.ndarray:
    idx = []
    for i, (a, b) in enumerate(offsets):
        if a == b == 0 and i > 0:
            continue
        if b <= prompt_len_chars and b > a:
            idx.append(i)
    return np.asarray(idx, dtype=np.int64)


def visible_token_indices(
    offsets: Sequence[Tuple[int, int]],
    token_range: tuple[int, int],
) -> np.ndarray:
    """Return non-special token indices inside a half-open token range."""

    start, stop = (int(token_range[0]), int(token_range[1]))
    if start < 0 or stop < start:
        return np.asarray([], dtype=np.int64)
    return np.asarray(
        [i for i in range(start, min(stop, len(offsets))) if offsets[i][1] > offsets[i][0]],
        dtype=np.int64,
    )


def response_start_token(offsets: Sequence[Tuple[int, int]], prompt_len_chars: int) -> int:
    for i, (a, b) in enumerate(offsets):
        if b > prompt_len_chars and b > a:
            return int(i)
    return int(len(offsets))


def run_teacher_forcing(
    model,
    tokenizer,
    prompt: str,
    response: str,
    *,
    output_hidden_states: bool,
    output_attentions: bool,
    output_logits: bool = True,
    max_seq_len: int = 4096,
    steps: Sequence[str] | None = None,
    exact_input_ids: Sequence[int] | None = None,
    exact_attention_mask: Sequence[int] | None = None,
    exact_token_offsets: Sequence[Tuple[int, int]] | None = None,
    exact_step_token_ranges: Sequence[Tuple[int, int]] | None = None,
    exact_response_start_token: int | None = None,
    exact_question_token_range: Tuple[int, int] | None = None,
    question_text: str | None = None,
    replay_protocol: str = "",
    prompt_provenance: str = "",
    messages_json: str = "",
    hidden_depths: Sequence[int] | None = None,
    attention_blocks: Sequence[int] | None = None,
    max_attention_tokens: int | None = None,
) -> ForwardCache:
    """Run one teacher-forced forward pass and align response steps.

    Selected hidden/attention tensors remain on the model device so batched
    linear algebra is performed there. Only compact scores and explicitly
    requested state artifacts are transferred to CPU; full attentions are
    never persisted.
    """

    import torch

    trace = prepare_teacher_forcing_trace(
        tokenizer,
        prompt,
        response,
        steps=steps,
        max_seq_len=max_seq_len,
        exact_input_ids=exact_input_ids,
        exact_attention_mask=exact_attention_mask,
        exact_token_offsets=exact_token_offsets,
        exact_step_token_ranges=exact_step_token_ranges,
        exact_response_start_token=exact_response_start_token,
        exact_question_token_range=exact_question_token_range,
        question_text=question_text,
    )
    offsets = trace["token_offsets"]
    resp_start = int(trace["response_start_token"])
    ranges = trace["step_token_ranges"]
    enc = {
        "input_ids": torch.tensor([trace["input_ids"]], dtype=torch.long),
        "attention_mask": torch.tensor([trace["attention_mask"]], dtype=torch.long),
    }
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    seq_len = int(enc["input_ids"].shape[1])
    if (
        output_attentions
        and max_attention_tokens is not None
        and seq_len > int(max_attention_tokens)
    ):
        raise RuntimeError(
            f"attention extraction requires {seq_len} tokens, above the explicit "
            f"short-sequence limit {int(max_attention_tokens)}; no exact generic "
            "layerwise-attention path is implemented, so extraction stops before "
            "allocating an O(sequence^2) tensor"
        )

    forward_model = model
    if not output_logits:
        base_model = getattr(model, "base_model", None)
        if base_model is not None and base_model is not model:
            forward_model = base_model

    with torch.inference_mode():
        out = forward_model(
            **enc,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            use_cache=False,
            return_dict=True,
        )

    input_ids = enc["input_ids"][0].detach().cpu()
    attention_mask = enc["attention_mask"][0].detach().cpu()
    seq_len = int(input_ids.shape[0])
    safe_ranges = [(int(a), int(b)) for (a, b) in ranges]

    hidden = None
    retained_hidden_depths: tuple[int, ...] = ()
    if output_hidden_states and getattr(out, "hidden_states", None) is not None:
        total_depths = len(out.hidden_states)
        requested = (
            tuple(range(total_depths))
            if hidden_depths is None
            else tuple(sorted({int(x) for x in hidden_depths}))
        )
        if any(depth < 0 or depth >= total_depths for depth in requested):
            raise ValueError(
                f"requested hidden depths {requested} outside [0, {total_depths - 1}]"
            )
        hidden = SparseLayerSequence(
            total_depths,
            {depth: out.hidden_states[depth][0].detach() for depth in requested},
        )
        retained_hidden_depths = requested

    attn = None
    retained_attention_blocks: tuple[int, ...] = ()
    if output_attentions and getattr(out, "attentions", None) is not None:
        total_blocks = len(out.attentions)
        requested = (
            tuple(range(total_blocks))
            if attention_blocks is None
            else tuple(sorted({int(x) for x in attention_blocks}))
        )
        if any(block < 0 or block >= total_blocks for block in requested):
            raise ValueError(
                f"requested attention blocks {requested} outside [0, {total_blocks - 1}]"
            )
        attn = SparseLayerSequence(
            total_blocks,
            {block: out.attentions[block][0].detach() for block in requested},
        )
        retained_attention_blocks = requested

    logits = None
    token_output_summaries: dict[str, object] = {}
    if output_logits and getattr(out, "logits", None) is not None:
        token_output_summaries = compute_compact_token_output_summaries(
            out.logits[0], enc["input_ids"][0]
        )

    del out

    return ForwardCache(
        input_ids=input_ids,
        attention_mask=attention_mask,
        offset_mapping=offsets,
        prompt_len_tokens=resp_start,
        prompt_token_range=(0, resp_start),
        question_token_range=tuple(int(x) for x in trace["question_token_range"]),
        response_start_token=resp_start,
        response_token_range=tuple(int(x) for x in trace["response_token_range"]),
        step_token_ranges=safe_ranges,
        hidden_states=hidden,
        attentions=attn,
        logits=logits,
        token_output_summaries=token_output_summaries,
        seq_len=seq_len,
        replay_kind=str(trace["replay_kind"]),
        replay_protocol=str(replay_protocol),
        prompt_provenance=str(prompt_provenance),
        messages_json=str(messages_json),
        retained_hidden_depths=retained_hidden_depths,
        retained_attention_blocks=retained_attention_blocks,
    )


def compute_compact_token_output_summaries(
    logits,
    input_ids,
    *,
    token_chunk_size: int = 128,
    top_k: int = 10,
) -> dict[str, "object"]:
    """Compute causal output summaries on GPU and transfer only O(sequence)."""

    import torch

    seq_len = int(input_ids.shape[0])
    names = ("entropy", "nll", "chosen_logprob", "top1_top2_margin", "topk_mass")
    outputs = {
        name: torch.full((seq_len,), float("nan"), dtype=torch.float32, device="cpu")
        for name in names
    }
    if seq_len <= 1:
        return outputs

    # logits[i] predicts input_ids[i + 1]. Chunking avoids another full
    # sequence-by-vocabulary softmax allocation on the accelerator.
    for target_start in range(1, seq_len, max(int(token_chunk_size), 1)):
        target_stop = min(seq_len, target_start + max(int(token_chunk_size), 1))
        pred = logits[target_start - 1 : target_stop - 1].float()
        target = input_ids[target_start:target_stop]
        log_z = torch.logsumexp(pred, dim=-1)
        prob = torch.softmax(pred, dim=-1)
        entropy = log_z - torch.sum(prob * pred, dim=-1)
        chosen = pred.gather(1, target[:, None]).squeeze(1)
        chosen_logprob = chosen - log_z
        nll = -chosen_logprob
        k = min(max(int(top_k), 2), int(pred.shape[-1]))
        top_values = torch.topk(pred, k=k, dim=-1).values
        margin = top_values[:, 0] - top_values[:, 1]
        top_mass = torch.sum(torch.exp(top_values - log_z[:, None]), dim=-1)
        values = {
            "entropy": entropy,
            "nll": nll,
            "chosen_logprob": chosen_logprob,
            "top1_top2_margin": margin,
            "topk_mass": top_mass,
        }
        for name, value in values.items():
            outputs[name][target_start:target_stop] = value.detach().cpu()
        del pred, prob, top_values
    if int(top_k) == 10:
        # Backward-compatible alias whose name is valid only for the default.
        outputs["top10_mass"] = outputs["topk_mass"]
    return outputs


# Compatibility alias for callers created before the helper became part of the
# extraction contract. New code should use the public name above.
_compact_token_output_summaries = compute_compact_token_output_summaries


def prepare_teacher_forcing_trace(
    tokenizer,
    prompt: str,
    response: str,
    *,
    steps: Sequence[str] | None,
    max_seq_len: int,
    exact_input_ids: Sequence[int] | None = None,
    exact_attention_mask: Sequence[int] | None = None,
    exact_token_offsets: Sequence[Tuple[int, int]] | None = None,
    exact_step_token_ranges: Sequence[Tuple[int, int]] | None = None,
    exact_response_start_token: int | None = None,
    exact_question_token_range: Tuple[int, int] | None = None,
    question_text: str | None = None,
) -> dict[str, Any]:
    """Prepare one token axis for both step alignment and model replay.

    If an extraction artifact supplies exact arrays, they are replayed directly
    and never decoded/re-tokenized.  Otherwise the shared no-special-token
    alignment builder creates the axis once.  In either route, step ranges and
    hidden states index the exact same sequence.
    """

    supplied = exact_input_ids is not None
    optional = [
        exact_attention_mask,
        exact_token_offsets,
        exact_step_token_ranges,
        exact_response_start_token,
    ]
    if supplied and any(value is None for value in optional):
        raise TokenAlignmentError(
            "exact_input_ids requires attention_mask, token_offsets, step_token_ranges, "
            "and response_start_token"
        )
    if supplied:
        ids = [int(x) for x in exact_input_ids] if exact_input_ids is not None else []
        mask = [int(x) for x in exact_attention_mask] if exact_attention_mask is not None else []
        offsets = (
            [(int(a), int(b)) for a, b in exact_token_offsets]
            if exact_token_offsets is not None
            else []
        )
        ranges = (
            [(int(a), int(b)) for a, b in exact_step_token_ranges]
            if exact_step_token_ranges is not None
            else []
        )
        response_start = int(exact_response_start_token)
        if len(ids) != len(mask) or len(ids) != len(offsets):
            raise TokenAlignmentError("exact input_ids, attention_mask, and token_offsets disagree")
        if any(value not in (0, 1) for value in mask):
            raise TokenAlignmentError("exact attention_mask must be binary")
        if any(a < 0 or b < a for a, b in offsets):
            raise TokenAlignmentError("exact token_offsets contains an invalid half-open span")
        visible_offsets = [(a, b) for a, b in offsets if b > a]
        if any(
            visible_offsets[i][0] < visible_offsets[i - 1][0]
            for i in range(1, len(visible_offsets))
        ):
            raise TokenAlignmentError("exact token_offsets is not monotone")
        if not 0 <= response_start <= len(ids):
            raise TokenAlignmentError("exact response_start_token is outside input_ids")
        if any(a < response_start or b < a or b >= len(ids) for a, b in ranges):
            raise TokenAlignmentError("an exact step token range is outside input_ids")
        if any(ranges[i][0] <= ranges[i - 1][1] for i in range(1, len(ranges))):
            raise TokenAlignmentError("exact step token ranges must be strictly ordered and non-overlapping")
        if steps is not None and len(ranges) != len(steps):
            raise TokenAlignmentError(
                f"exact step ranges ({len(ranges)}) do not match kept step strings ({len(steps)})"
            )
        if exact_question_token_range is None:
            if question_text:
                question = str(question_text)
                question_start = str(prompt).rfind(question)
                if question_start < 0:
                    raise TokenAlignmentError(
                        "target question is not present in the exact rendered prompt"
                    )
                question_range = char_span_to_token_range(
                    offsets,
                    question_start,
                    question_start + len(question),
                    name="question",
                )
                if question_range[1] > response_start:
                    raise TokenAlignmentError(
                        "derived question token range escapes the exact prompt"
                    )
            else:
                question_range = (-1, -1)
        else:
            question_range = tuple(int(x) for x in exact_question_token_range)
            if not (
                len(question_range) == 2
                and 0 <= question_range[0] < question_range[1] <= response_start
            ):
                raise TokenAlignmentError(
                    "exact question token range must be half-open inside the prompt"
                )
    else:
        try:
            from utils.step_boundaries import build_exact_trace_alignment
        except Exception:  # pragma: no cover - package import fallback
            from ..utils.step_boundaries import build_exact_trace_alignment  # type: ignore
        trace = build_exact_trace_alignment(
            tokenizer,
            prompt,
            response,
            list(steps) if steps is not None else None,
            question_text=question_text,
        )
        ids = [int(x) for x in trace["input_ids"]]
        mask = [int(x) for x in trace["attention_mask"]]
        offsets = [(int(a), int(b)) for a, b in trace["token_offsets"]]
        ranges = [(int(a), int(b)) for a, b in trace["all_step_token_ranges"]]
        response_start = int(trace["response_token_range"][0])
        question_range = tuple(int(x) for x in trace["question_token_range"])

    limit = len(ids) if int(max_seq_len) <= 0 else min(len(ids), int(max_seq_len))
    if limit < response_start:
        raise ValueError(
            f"max_seq_len={max_seq_len} truncates the prompt at token {response_start}"
        )
    ids = ids[:limit]
    mask = mask[:limit]
    offsets = offsets[:limit]
    truncated_ranges = [(a, b) for a, b in ranges if b >= limit]
    if truncated_ranges:
        raise TokenAlignmentError(
            f"max_seq_len={max_seq_len} truncates {len(truncated_ranges)} reasoning "
            "step range(s); refusing to change the chain/step axis"
        )
    ranges = [(a, b) for a, b in ranges if b < limit]
    if not ranges:
        raise ValueError("no complete reasoning step remains on the teacher-forcing axis")
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "token_offsets": offsets,
        "response_start_token": response_start,
        "response_token_range": (response_start, len(ids)),
        "question_token_range": question_range,
        "step_token_ranges": ranges,
        "replay_kind": "exact_artifact_ids" if supplied else "single_axis_no_special_fallback",
    }
