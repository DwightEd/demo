from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .ghost import (
    GhostTraceEmbedding,
    response_token_positions,
)


@dataclass(frozen=True)
class ChatTokenizedTrace:
    input_ids: np.ndarray
    step_ranges: np.ndarray
    prompt_tokens: int


def _align_offsets_to_steps(
    offsets: np.ndarray,
    spans: np.ndarray,
) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.int64)
    spans = np.asarray(spans, dtype=np.int64)
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError("offset_mapping must have shape [tokens, 2]")
    ranges = []
    occupied: set[int] = set()
    for step_index, (start, stop) in enumerate(spans):
        members = [
            index
            for index, (left, right) in enumerate(offsets)
            if right > left and left < stop and right > start
        ]
        if not members:
            raise ValueError(f"reasoning step {step_index} maps to no response token")
        if members != list(range(members[0], members[-1] + 1)):
            raise ValueError(f"reasoning step {step_index} is not token-contiguous")
        overlap = occupied.intersection(members)
        if overlap:
            raise ValueError(f"response tokens overlap multiple steps: {sorted(overlap)}")
        occupied.update(members)
        ranges.append((members[0], members[-1] + 1))
    return np.asarray(ranges, dtype=np.int64)


def tokenize_chat_record(
    tokenizer,
    record,
    *,
    separator: str = "\n\n",
) -> ChatTokenizedTrace:
    """Tokenize a fixed ProcessBench answer without generation or truncation."""

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("GHOST extraction requires a fast tokenizer")
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise ValueError("the observer tokenizer has no chat template")

    def template_ids(messages, *, add_generation_prompt: bool) -> np.ndarray:
        rendered = apply_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(rendered, Mapping):
            rendered = rendered.get("input_ids")
        values = np.asarray(rendered, dtype=np.int64)
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 1 or not len(values):
            raise ValueError("chat template produced no one-dimensional tokens")
        return values

    def template_text(messages, *, add_generation_prompt: bool) -> str:
        rendered = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if not isinstance(rendered, str) or not rendered:
            raise ValueError("chat template produced no serialized text")
        return rendered

    user_message = {"role": "user", "content": record.question}
    prefix_text = template_text(
        [user_message],
        add_generation_prompt=True,
    )
    clean_steps = tuple(str(step).strip() for step in record.steps)
    if not clean_steps or any(not step for step in clean_steps):
        raise ValueError("fixed response contains an empty normalized reasoning step")
    response = separator.join(clean_steps)
    full_messages = [user_message, {"role": "assistant", "content": response}]
    empty_messages = [user_message, {"role": "assistant", "content": ""}]
    full_text = template_text(
        full_messages,
        add_generation_prompt=False,
    )
    empty_assistant_text = template_text(
        empty_messages,
        add_generation_prompt=False,
    )
    if (
        len(empty_assistant_text) < len(prefix_text)
        or not empty_assistant_text.startswith(prefix_text)
    ):
        raise ValueError(
            "chat template assistant prefix is not stable between generation "
            "and completed-conversation rendering"
        )
    suffix_text = empty_assistant_text[len(prefix_text) :]
    if (
        len(full_text) < len(prefix_text) + len(suffix_text)
        or not full_text.startswith(prefix_text)
        or (suffix_text and not full_text.endswith(suffix_text))
    ):
        raise ValueError("completed chat text does not preserve prompt/suffix boundaries")
    response_stop = len(full_text) - len(suffix_text) if suffix_text else len(full_text)
    if full_text[len(prefix_text) : response_stop] != response:
        raise ValueError("chat template altered the normalized assistant response")

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    full_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    canonical_full_ids = template_ids(
        full_messages,
        add_generation_prompt=False,
    )
    if not np.array_equal(canonical_full_ids, full_ids):
        raise ValueError(
            "offset-tokenized full chat disagrees with canonical chat tokenization"
        )
    spans = []
    cursor = len(prefix_text)
    for index, step in enumerate(clean_steps):
        spans.append((cursor, cursor + len(step)))
        cursor += len(step)
        if index + 1 < len(clean_steps):
            cursor += len(separator)
    offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
    if full_ids.ndim != 1 or not len(full_ids):
        raise ValueError("fixed response maps to no tokens")
    step_ranges = _align_offsets_to_steps(
        offsets,
        np.asarray(spans, dtype=np.int64),
    )
    return ChatTokenizedTrace(
        input_ids=full_ids,
        step_ranges=step_ranges,
        prompt_tokens=int(step_ranges[0, 0]),
    )


def resolve_decoder_blocks(model) -> Sequence[object]:
    candidates = (
        ("layers",),
        ("model", "layers"),
        ("decoder", "layers"),
        ("transformer", "h"),
    )
    for path in candidates:
        value = model
        for component in path:
            value = getattr(value, component, None)
            if value is None:
                break
        if value is not None and hasattr(value, "__len__") and len(value):
            return value
    raise ValueError("cannot locate the observer model's decoder block sequence")


class HookedResponseStateExtractor:
    """Capture pooled mid-block outputs and the normalized final model output."""

    def __init__(
        self,
        model,
        *,
        mid_depths: Sequence[int],
        final_depth: int,
    ) -> None:
        blocks = resolve_decoder_blocks(model)
        middle = tuple(int(depth) for depth in mid_depths)
        if (
            not middle
            or tuple(sorted(set(middle))) != middle
            or any(depth < 1 or depth >= len(blocks) for depth in middle)
        ):
            raise ValueError("mid_depths must identify unique non-final decoder blocks")
        if int(final_depth) != len(blocks):
            raise ValueError("final_depth must equal the decoder block count")
        configured = getattr(getattr(model, "config", None), "num_hidden_layers", None)
        if configured is not None and int(configured) != len(blocks):
            raise ValueError("model config and resolved decoder block count disagree")
        self.model = model
        self.mid_depths = middle
        self.final_depth = int(final_depth)
        self._positions = None
        self._captured: dict[int, tuple[object, object]] = {}
        self._handles = []
        for depth in middle:
            self._handles.append(
                blocks[depth - 1].register_forward_hook(self._hook(depth))
            )

    def _hook(self, depth: int):
        def capture(_module, _inputs, output) -> None:
            if self._positions is None:
                raise RuntimeError("response positions were not armed before forward")
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if getattr(hidden, "ndim", None) != 3 or hidden.shape[0] != 1:
                raise ValueError("decoder block output must have shape [1, tokens, hidden]")
            selected = hidden[0].index_select(0, self._positions)
            self._captured[depth] = (
                selected.float().mean(dim=0).detach(),
                selected[-1].float().detach(),
            )

        return capture

    def extract(
        self,
        *,
        input_ids,
        response_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        if getattr(input_ids, "ndim", None) != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape [1, tokens]")
        positions = np.asarray(response_positions, dtype=np.int64)
        if (
            positions.ndim != 1
            or not len(positions)
            or np.any(positions < 0)
            or np.any(positions >= input_ids.shape[1])
            or np.any(positions[1:] <= positions[:-1])
        ):
            raise ValueError("response_positions must be increasing valid token indices")
        try:
            parameter = next(self.model.parameters())
            device = parameter.device
        except StopIteration:
            device = input_ids.device
        ids = input_ids.to(device)
        self._positions = torch.as_tensor(positions, dtype=torch.long, device=device)
        self._captured = {}
        with torch.inference_mode():
            output = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
        final = getattr(output, "last_hidden_state", None)
        if final is None and isinstance(output, Mapping):
            final = output.get("last_hidden_state")
        if final is None or final.ndim != 3 or final.shape[0] != 1:
            raise ValueError("observer model returned no final hidden state")
        selected_final = final[0].index_select(0, self._positions)
        self._captured[self.final_depth] = (
            selected_final.float().mean(dim=0).detach(),
            selected_final[-1].float().detach(),
        )
        missing = set(self.mid_depths).difference(self._captured)
        self._positions = None
        if missing:
            raise RuntimeError(f"decoder hooks did not capture depths: {sorted(missing)}")
        depths = (*self.mid_depths, self.final_depth)
        means = (
            torch.stack([self._captured[depth][0] for depth in depths])
            .cpu()
            .numpy()
        )
        lasts = (
            torch.stack([self._captured[depth][1] for depth in depths])
            .cpu()
            .numpy()
        )
        if not np.isfinite(means).all() or not np.isfinite(lasts).all():
            raise ValueError("observer produced non-finite hidden states")
        return means, lasts

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "HookedResponseStateExtractor":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def extract_processbench_embeddings(
    records: Iterable[object],
    *,
    model,
    tokenizer,
    mid_depths: Sequence[int],
    final_depth: int,
    max_tokens: int,
) -> tuple[GhostTraceEmbedding, ...]:
    import torch

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    traces = []
    with HookedResponseStateExtractor(
        model,
        mid_depths=mid_depths,
        final_depth=final_depth,
    ) as extractor:
        for index, record in enumerate(records, start=1):
            tokenized = tokenize_chat_record(tokenizer, record)
            if len(tokenized.input_ids) > max_tokens:
                raise ValueError(
                    f"trace {record.trace_id} has {len(tokenized.input_ids)} tokens, "
                    f"exceeding max_tokens={max_tokens}; truncation is forbidden"
                )
            positions = response_token_positions(
                tokenized.step_ranges,
                token_count=len(tokenized.input_ids),
            )
            mean, last = extractor.extract(
                input_ids=torch.as_tensor(tokenized.input_ids, dtype=torch.long)[None, :],
                response_positions=positions,
            )
            first_error = int(record.labels.first_error)
            traces.append(
                GhostTraceEmbedding(
                    trace_id=record.trace_id,
                    problem_id=record.problem_id,
                    response_label=int(first_error >= 0),
                    first_error=first_error,
                    generator_model=record.generator_model,
                    layer_depths=(*tuple(mid_depths), int(final_depth)),
                    response_mean=mean,
                    response_last=last,
                    token_count=len(tokenized.input_ids),
                    response_token_count=len(positions),
                    step_count=len(record.steps),
                )
            )
            print(
                f"extracted {index}: trace={record.trace_id} "
                f"tokens={len(tokenized.input_ids)} response_tokens={len(positions)}",
                flush=True,
            )
    return tuple(traces)


def save_embedding_artifact(
    path: str | Path,
    traces: Sequence[GhostTraceEmbedding],
    *,
    metadata: Mapping[str, object],
) -> None:
    rows = tuple(traces)
    if not rows:
        raise ValueError("cannot save an empty embedding artifact")
    if len({row.trace_id for row in rows}) != len(rows):
        raise ValueError("embedding trace IDs must be unique")
    depths = rows[0].layer_depths
    shape = rows[0].response_mean.shape
    if any(
        row.layer_depths != depths or row.response_mean.shape != shape for row in rows
    ):
        raise ValueError("all traces must share layer depths and hidden width")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        schema_version=np.asarray("ghost_embeddings_v1"),
        metadata_json=np.asarray(
            json.dumps(dict(metadata), sort_keys=True, ensure_ascii=False)
        ),
        trace_ids=np.asarray([row.trace_id for row in rows]),
        problem_ids=np.asarray([row.problem_id for row in rows]),
        response_labels=np.asarray(
            [row.response_label for row in rows], dtype=np.int8
        ),
        first_errors=np.asarray([row.first_error for row in rows], dtype=np.int64),
        generator_models=np.asarray([row.generator_model for row in rows]),
        layer_depths=np.asarray(depths, dtype=np.int64),
        response_means=np.stack([row.response_mean for row in rows]).astype(
            np.float32
        ),
        response_lasts=np.stack([row.response_last for row in rows]).astype(
            np.float32
        ),
        token_counts=np.asarray([row.token_count for row in rows], dtype=np.int64),
        response_token_counts=np.asarray(
            [row.response_token_count for row in rows], dtype=np.int64
        ),
        step_counts=np.asarray([row.step_count for row in rows], dtype=np.int64),
    )


def load_embedding_artifact(
    path: str | Path,
) -> tuple[tuple[GhostTraceEmbedding, ...], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        if archive["schema_version"].item() != "ghost_embeddings_v1":
            raise ValueError("unsupported GHOST embedding artifact schema")
        metadata = json.loads(str(archive["metadata_json"].item()))
        trace_ids = archive["trace_ids"]
        problem_ids = archive["problem_ids"]
        labels = archive["response_labels"]
        first_errors = archive["first_errors"]
        generators = archive["generator_models"]
        depths = tuple(int(value) for value in archive["layer_depths"])
        means = archive["response_means"]
        lasts = archive["response_lasts"]
        token_counts = archive["token_counts"]
        response_counts = archive["response_token_counts"]
        step_counts = archive["step_counts"]
        count = len(trace_ids)
        arrays = (
            problem_ids,
            labels,
            first_errors,
            generators,
            means,
            lasts,
            token_counts,
            response_counts,
            step_counts,
        )
        if any(len(array) != count for array in arrays):
            raise ValueError("embedding artifact arrays are not aligned")
        traces = tuple(
            GhostTraceEmbedding(
                trace_id=str(trace_ids[index]),
                problem_id=str(problem_ids[index]),
                response_label=int(labels[index]),
                first_error=int(first_errors[index]),
                generator_model=str(generators[index]),
                layer_depths=depths,
                response_mean=np.asarray(means[index], dtype=np.float64),
                response_last=np.asarray(lasts[index], dtype=np.float64),
                token_count=int(token_counts[index]),
                response_token_count=int(response_counts[index]),
                step_count=int(step_counts[index]),
            )
            for index in range(count)
        )
    return traces, metadata
