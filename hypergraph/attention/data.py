"""Strict adapters for per-trace attention artifacts.

The faithful attention model needs every array on the same *complete sequence*
axis (prompt followed by response).  This module deliberately performs the
alignment once, before graph construction, and refuses ambiguous shapes rather
than silently shifting response labels or hidden states around a BOS token.

Supported containers are a mapping (or a list of mappings) saved as ``.pt`` /
``.pth`` and a mapping saved as ``.npz``.  A dense batch ``[B,L,H,N,N]`` is
split into records, but the preferred and least ambiguous format is one dense
``[L,H,N,N]`` attention tensor per file.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np


TRACE_CONTRACT = "exact_prompt_response_attention_v1"
MODEL_COMMIT_SOURCES = frozenset(
    {
        "remote_resolved_model_commit",
        "remote_pinned_requested_commit",
        "remote_tokenizer_metadata_only",
        "local_model_metadata_commit",
        "local_tokenizer_metadata_only",
        "local_declared_commit",
        "unavailable",
    }
)
VERIFIED_MODEL_COMMIT_SOURCES = frozenset(
    {
        "remote_resolved_model_commit",
        "remote_pinned_requested_commit",
    }
)


_ATTENTION_KEYS = ("attention", "attentions", "self_attention", "self_attentions")
_ATTENTION_LAYER_ID_KEYS = ("attention_layer_ids", "attention_layers")
_ATTENTION_HEAD_ID_KEYS = ("attention_head_ids", "attention_heads")
_MODEL_LAYER_COUNT_KEYS = ("num_model_layers", "model_num_layers")
_MODEL_HEAD_COUNT_KEYS = ("num_model_heads", "model_num_heads")
_TOKEN_KEYS = ("token_ids", "input_ids", "tokens")
_RESPONSE_KEYS = ("response_idx", "response_start", "prompt_length", "prompt_len")
_ACTIVATION_KEYS = ("activation", "activations", "hidden", "hidden_states")
_TOKEN_LABEL_KEYS = ("token_y", "y_token", "token_labels", "hallucination_labels")
_TOKEN_LABEL_MASK_KEYS = ("token_label_mask", "exact_label_mask", "token_loss_mask")
_RESPONSE_LABEL_KEYS = (
    "response_y",
    "response_label",
    "hallucination_label",
    "is_hallucinated",
    "is_incorrect",
)
_STEP_RANGE_KEYS = (
    "step_ranges",
    "response_step_ranges",
    "token_ranges",
    "step_token_ranges",
)
_GOLD_STEP_KEYS = ("gold_step", "gold_error_step", "first_error_step")
_STEP_MASK_KEYS = ("step_loss_mask", "step_mask")
_GROUP_KEYS = ("problem_id", "question_id", "group_id", "source_id")
_ID_KEYS = ("trace_id", "chain_id", "sample_id", "id")
_SPLIT_KEYS = ("split", "split_name", "partition")
_PROVENANCE_KEYS = (
    "trace_contract",
    "model_name",
    "model_commit_hash",
    "model_commit_source",
    "tokenizer_name",
    "prompt_style",
    "replay_mode",
    "step_alignment_policy",
    "replay_fidelity",
    "unverified_generator_weights_explicitly_allowed",
    "prompt_provenance",
    "generator_model",
    "generator_model_commit",
    "prompt_add_special_tokens",
    "extraction_dtype",
    "attention_storage_dtype",
    "activation_layer",
    "extraction_fingerprint",
)
_AUDIT_PROVENANCE_KEYS = (
    "rendered_prompt_sha256",
    "response_text_sha256",
    "generation_terminal_token_count",
    "extraction_method_json",
    "extraction_scope_fingerprint",
    "extraction_scope_json",
    "source_input_sha256",
    "source_row_index",
    "extraction_forward_mode",
    "chunk_equivalence_status",
    "chunk_equivalence_json",
)


class TraceFormatError(ValueError):
    """Raised when a trace cannot be aligned without guessing."""


@dataclass(frozen=True)
class TraceLoadConfig:
    """Controls the only two conventions that cannot always be inferred.

    ``step_end`` describes the source artifact.  Canonical ranges are always
    half-open.  ``step_axis='auto'`` treats a range beginning before
    ``response_idx`` as response-relative and otherwise as full-sequence.

    Hidden states are optional.  In auto layout, ``[L,N,D]`` and ``[N,F]`` are
    recognized by matching ``N`` to the token count.  By default only the last
    hidden layer is retained; pass ``activation_layers=None`` to concatenate all
    available layers or an explicit tuple to select layers.
    """

    step_end: str = "exclusive"
    step_axis: str = "auto"
    activation_layout: str = "auto"
    activation_layers: Optional[Tuple[int, ...]] = (-1,)
    require_attention: bool = True
    require_causal: bool = False
    causal_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if self.step_end not in {"exclusive", "inclusive"}:
            raise ValueError("step_end must be 'exclusive' or 'inclusive'")
        if self.step_axis not in {"auto", "full", "response"}:
            raise ValueError("step_axis must be 'auto', 'full', or 'response'")
        if self.activation_layout not in {"auto", "node_first", "layer_first"}:
            raise ValueError(
                "activation_layout must be 'auto', 'node_first', or 'layer_first'"
            )
        if not np.isfinite(self.causal_tolerance) or self.causal_tolerance < 0:
            raise ValueError("causal_tolerance must be finite and non-negative")


@dataclass
class AttentionTrace:
    """Canonical, construction-ready representation of one generation trace."""

    attention: np.ndarray
    token_ids: np.ndarray
    response_idx: int
    attention_layer_ids: Optional[np.ndarray] = None
    attention_head_ids: Optional[np.ndarray] = None
    num_model_layers: Optional[int] = None
    num_model_heads: Optional[int] = None
    activation: Optional[np.ndarray] = None
    token_y: Optional[np.ndarray] = None
    token_label_mask: Optional[np.ndarray] = None
    response_y: Optional[float] = None
    step_ranges: Optional[np.ndarray] = None
    gold_step: Optional[int] = None
    step_loss_mask: Optional[np.ndarray] = None
    group_id: str = ""
    trace_id: str = ""
    split: Optional[str] = None
    source_path: str = ""
    group_is_fallback: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def num_response_tokens(self) -> int:
        return self.num_tokens - int(self.response_idx)

    def builder_kwargs(self, *, use_activation: bool = False) -> Dict[str, Any]:
        """Return exactly the arguments accepted by graph construction."""

        return {
            "attention": self.attention,
            "token_ids": self.token_ids,
            "response_idx": int(self.response_idx),
            "attention_layer_ids": self.attention_layer_ids,
            "attention_head_ids": self.attention_head_ids,
            "num_model_layers": self.num_model_layers,
            "num_model_heads": self.num_model_heads,
            "activation": self.activation if use_activation else None,
            "token_y": self.token_y,
            "token_label_mask": self.token_label_mask,
            "response_y": self.response_y,
            "step_ranges": self.step_ranges,
            "gold_step": self.gold_step,
            "step_loss_mask": self.step_loss_mask,
            "trace_id": self.trace_id,
            "group_id": self.group_id,
            "split": self.split,
        }


def trace_provenance_fingerprint(trace: AttentionTrace) -> Optional[str]:
    """Hash model/template/extraction provenance plus the actual attention axes."""

    provenance = trace_method_provenance(trace)
    if not provenance:
        return None
    provenance.update(
        {
            "attention_layer_ids": (
                None
                if trace.attention_layer_ids is None
                else trace.attention_layer_ids.tolist()
            ),
            "attention_head_ids": (
                None if trace.attention_head_ids is None else trace.attention_head_ids.tolist()
            ),
            "num_model_layers": trace.num_model_layers,
            "num_model_heads": trace.num_model_heads,
        }
    )
    encoded = json.dumps(provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def trace_method_provenance(trace: AttentionTrace) -> Dict[str, Any]:
    """Return homogeneous method-level provenance; sample hashes stay separate."""

    return {
        key: trace.metadata.get(key)
        for key in _PROVENANCE_KEYS
        if trace.metadata.get(key) not in (None, "")
    }


def model_identity_matches(source: str, requested: str) -> bool:
    """Match an exact model identity, allowing a local path to alias its leaf.

    Two different remote repository names with the same final component are not
    treated as identical.  The leaf exception exists only because an audited
    model is commonly replayed from a local snapshot directory.
    """

    source_key = str(source).strip().replace("\\", "/").rstrip("/").lower()
    requested_key = str(requested).strip().replace("\\", "/").rstrip("/").lower()
    if not source_key or not requested_key:
        return False
    if source_key == requested_key:
        return True
    source_local = source_key.startswith("/") or (
        len(source_key) >= 3 and source_key[1:3] == ":/"
    )
    requested_local = requested_key.startswith("/") or (
        len(requested_key) >= 3 and requested_key[1:3] == ":/"
    )
    return bool(
        (source_local or requested_local)
        and source_key.rsplit("/", 1)[-1] == requested_key.rsplit("/", 1)[-1]
    )


def is_immutable_commit_hash(value: Any) -> bool:
    """Return whether ``value`` is a usable abbreviated/full hexadecimal commit."""

    return bool(re.fullmatch(r"[0-9a-fA-F]{7,64}", str(value).strip()))


def commit_hashes_match(left: Any, right: Any) -> bool:
    """Compare full or abbreviated immutable commit hashes."""

    if not is_immutable_commit_hash(left) or not is_immutable_commit_hash(right):
        return False
    left_key, right_key = str(left).strip().lower(), str(right).strip().lower()
    return left_key.startswith(right_key) or right_key.startswith(left_key)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Tuple[Any, Optional[str]]:
    present = [key for key in keys if key in mapping and mapping[key] is not None]
    if len(present) > 1:
        raise TraceFormatError(
            f"ambiguous aliases {present}; provide exactly one canonical field"
        )
    if present:
        key = present[0]
        return mapping[key], key
    return None, None


def _scalar(value: Any, *, name: str) -> Any:
    array = _as_numpy(value)
    if array.size != 1:
        raise TraceFormatError(f"{name} must be scalar, got shape {array.shape}")
    return array.reshape(-1)[0].item()


def _integer_scalar(value: Any, *, name: str) -> int:
    scalar = _scalar(value, name=name)
    array = np.asarray(scalar)
    if not np.issubdtype(array.dtype, np.integer):
        if (
            not np.issubdtype(array.dtype, np.number)
            or not np.isfinite(scalar)
            or scalar != np.floor(scalar)
        ):
            raise TraceFormatError(f"{name} must be an integer scalar, got {scalar!r}")
    return int(scalar)


def _binary_scalar(value: Any, *, name: str) -> float:
    """Return a strict binary scalar without truthiness/string coercion."""

    scalar = _scalar(value, name=name)
    if isinstance(scalar, (bool, np.bool_)):
        return float(bool(scalar))
    array = np.asarray(scalar)
    if (
        not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(scalar)
        or float(scalar) not in (0.0, 1.0)
    ):
        raise TraceFormatError(f"{name} must be an actual bool or numeric 0/1, got {scalar!r}")
    return float(scalar)


def _canonical_attention(value: Any, n_tokens: int, *, source: str) -> np.ndarray:
    if isinstance(value, (list, tuple)) and value:
        layers = []
        for layer in value:
            item = _as_numpy(layer)
            if item.ndim == 4 and item.shape[0] == 1:
                item = item[0]
            if item.ndim != 3:
                raise TraceFormatError(
                    f"{source}: each attention list item must be [H,N,N], got {item.shape}"
                )
            layers.append(item)
        array = np.stack(layers, axis=0)
    else:
        array = _as_numpy(value)
        if array.dtype == object and array.ndim == 1 and len(array):
            try:
                array = np.stack([_as_numpy(item) for item in array], axis=0)
            except ValueError as exc:
                raise TraceFormatError(
                    f"{source}: ragged attention layers cannot form [L,H,N,N]"
                ) from exc
    while array.ndim > 4 and 1 in array.shape:
        array = np.squeeze(array, axis=next(i for i, size in enumerate(array.shape) if size == 1))
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise TraceFormatError(
            f"{source}: attention must be [L,H,N,N] (or [H,N,N]), got {array.shape}"
        )
    if tuple(array.shape[-2:]) != (n_tokens, n_tokens):
        raise TraceFormatError(
            f"{source}: attention token axes {array.shape[-2:]} do not match "
            f"token_ids length {n_tokens}; check BOS/EOS and response offsets"
        )
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise TraceFormatError(f"{source}: attention has an empty layer/head axis")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise TraceFormatError(f"{source}: attention contains NaN or infinity")
    if float(array.min(initial=0.0)) < -1e-6:
        raise TraceFormatError(f"{source}: attention contains negative values")
    return np.ascontiguousarray(array)


def _canonical_attention_axis(
    ids_value: Any,
    total_value: Any,
    size: int,
    *,
    name: str,
    source: str,
) -> Tuple[np.ndarray, int]:
    if size < 1:
        if ids_value is not None:
            raise TraceFormatError(f"{source}: {name} ids require a non-empty attention axis")
        total = (
            0
            if total_value is None
            else _integer_scalar(total_value, name=f"num_model_{name}")
        )
        return np.zeros((0,), dtype=np.int64), total
    if ids_value is None:
        ids = np.arange(size, dtype=np.int64)
    else:
        raw_ids = np.asarray(_as_numpy(ids_value)).reshape(-1)
        if not np.issubdtype(raw_ids.dtype, np.integer):
            if (
                not np.issubdtype(raw_ids.dtype, np.number)
                or not np.isfinite(raw_ids).all()
                or not np.equal(raw_ids, np.floor(raw_ids)).all()
            ):
                raise TraceFormatError(f"{source}: {name} must contain integer axis ids")
        ids = np.asarray(raw_ids, dtype=np.int64)
    if ids.shape != (size,):
        raise TraceFormatError(
            f"{source}: {name} has shape {ids.shape}; expected ({size},) to match attention"
        )
    if np.any(ids < 0) or len(np.unique(ids)) != size:
        raise TraceFormatError(
            f"{source}: {name} must contain unique non-negative model-axis ids"
        )
    inferred_total = int(ids.max()) + 1
    total = (
        inferred_total
        if total_value is None
        else _integer_scalar(total_value, name=f"num_model_{name}")
    )
    if total < inferred_total:
        raise TraceFormatError(
            f"{source}: model-axis total {total} cannot contain maximum {name} id "
            f"{int(ids.max())}"
        )
    return np.ascontiguousarray(ids), total


def _select_layers(array: np.ndarray, layers: Optional[Tuple[int, ...]], *, source: str) -> np.ndarray:
    if layers is None:
        return array
    indices = []
    for raw in layers:
        index = int(raw)
        if index < 0:
            index += int(array.shape[0])
        if not 0 <= index < int(array.shape[0]):
            raise TraceFormatError(
                f"{source}: activation layer {raw} outside [0,{array.shape[0]})"
            )
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise TraceFormatError(f"{source}: duplicate activation layers {layers}")
    return array[indices]


def _canonical_activation(
    value: Any,
    n_tokens: int,
    config: TraceLoadConfig,
    *,
    source: str,
) -> Optional[np.ndarray]:
    if value is None:
        return None

    if isinstance(value, (list, tuple)) and value:
        parts = []
        for layer in value:
            item = _as_numpy(layer)
            if item.ndim == 3 and item.shape[0] == 1:
                item = item[0]
            if item.ndim != 2 or item.shape[0] != n_tokens:
                raise TraceFormatError(
                    f"{source}: hidden-state list item must be [N,D], got {item.shape}"
                )
            parts.append(item)
        array = np.stack(parts, axis=0)
    else:
        array = _as_numpy(value)

    # Common transformers output layouts with an explicit batch dimension.
    if array.ndim == 4:
        if array.shape[1] == 1:  # [L,1,N,D]
            array = array[:, 0]
        elif array.shape[0] == 1:  # [1,L,N,D]
            array = array[0]
        else:
            raise TraceFormatError(
                f"{source}: activation has an unsupported non-singleton batch axis {array.shape}"
            )

    if array.ndim == 2:
        if array.shape[0] != n_tokens:
            raise TraceFormatError(
                f"{source}: activation [N,F] has N={array.shape[0]}, expected {n_tokens}"
            )
        if config.activation_layers not in {None, (-1,)}:
            raise TraceFormatError(
                f"{source}: cannot select layers from an already flattened [N,F] activation"
            )
        result = array
    elif array.ndim == 3:
        layout = config.activation_layout
        if layout == "auto":
            layer_match = array.shape[1] == n_tokens
            node_match = array.shape[0] == n_tokens
            if layer_match and node_match:
                raise TraceFormatError(
                    f"{source}: activation shape {array.shape} is axis-ambiguous; set "
                    "activation_layout explicitly"
                )
            if layer_match:
                layout = "layer_first"
            elif node_match:
                layout = "node_first"
            else:
                raise TraceFormatError(
                    f"{source}: no activation axis matches token count {n_tokens}: {array.shape}"
                )
        if layout == "layer_first":
            if array.shape[1] != n_tokens:
                raise TraceFormatError(
                    f"{source}: layer-first activation must be [L,N,D], got {array.shape}"
                )
            chosen = _select_layers(array, config.activation_layers, source=source)
            result = np.moveaxis(chosen, 0, 1).reshape(n_tokens, -1)
        else:
            if array.shape[0] != n_tokens:
                raise TraceFormatError(
                    f"{source}: node-first activation must be [N,L,D], got {array.shape}"
                )
            layer_first = np.moveaxis(array, 1, 0)
            chosen = _select_layers(layer_first, config.activation_layers, source=source)
            result = np.moveaxis(chosen, 0, 1).reshape(n_tokens, -1)
    else:
        raise TraceFormatError(
            f"{source}: activation must be [N,F], [L,N,D], or [N,L,D], got {array.shape}"
        )

    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise TraceFormatError(f"{source}: activation contains NaN or infinity")
    return np.ascontiguousarray(result)


def _canonical_token_labels(value: Any, n_tokens: int, response_idx: int, *, source: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = _as_numpy(value)
    if array.size == 1:
        return None
    array = np.asarray(array).reshape(-1)
    n_response = n_tokens - response_idx
    if array.size == n_response:
        full = np.full(n_tokens, -100.0, dtype=np.float32)
        full[response_idx:] = array.astype(np.float32)
        array = full
    elif array.size != n_tokens:
        raise TraceFormatError(
            f"{source}: token labels have length {array.size}; expected {n_tokens} "
            f"(full sequence) or {n_response} (response only)"
        )
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    allowed = np.isin(array[finite], (-100.0, 0.0, 1.0))
    if not bool(np.all(allowed)):
        bad = np.unique(array[finite][~allowed])[:5]
        raise TraceFormatError(f"{source}: exact token labels must be 0/1/-100, got {bad}")
    array[~finite] = -100.0
    return array


def _canonical_token_label_mask(
    value: Any,
    token_y: Optional[np.ndarray],
    n_tokens: int,
    response_idx: int,
    *,
    source: str,
) -> Optional[np.ndarray]:
    if token_y is None:
        if value is not None:
            raise TraceFormatError(
                f"{source}: token label mask is present but token labels are missing"
            )
        return None
    valid_labels = np.isin(token_y, (0.0, 1.0))
    # Prompt positions are nodes but never exact hallucination targets.  Some
    # legacy full-axis files store zeros there; zeros must not turn into labels.
    valid_labels[:response_idx] = False
    if value is None:
        return np.asarray(valid_labels, dtype=bool)
    mask = np.asarray(_as_numpy(value), dtype=bool).reshape(-1)
    n_response = n_tokens - response_idx
    if mask.size == n_response:
        full = np.zeros(n_tokens, dtype=bool)
        full[response_idx:] = mask
        mask = full
    elif mask.size != n_tokens:
        raise TraceFormatError(
            f"{source}: token label mask has length {mask.size}; expected {n_tokens} "
            f"(full sequence) or {n_response} (response only)"
        )
    if np.any(mask & ~valid_labels):
        raise TraceFormatError(
            f"{source}: token label mask selects a non-binary/ignored label"
        )
    mask[:response_idx] = False
    return np.asarray(mask & valid_labels, dtype=bool)


def _canonical_step_ranges(
    value: Any,
    n_tokens: int,
    response_idx: int,
    config: TraceLoadConfig,
    *,
    source: str,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    raw_ranges = np.asarray(_as_numpy(value))
    if not np.issubdtype(raw_ranges.dtype, np.integer):
        if not np.issubdtype(raw_ranges.dtype, np.number) or not np.all(
            np.isfinite(raw_ranges) & np.equal(raw_ranges, np.floor(raw_ranges))
        ):
            raise TraceFormatError(f"{source}: step_ranges must contain integer offsets")
    ranges = np.asarray(raw_ranges, dtype=np.int64)
    if ranges.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise TraceFormatError(f"{source}: step_ranges must be [S,2], got {ranges.shape}")
    ranges = ranges.copy()
    if config.step_end == "inclusive":
        ranges[:, 1] += 1
    axis = config.step_axis
    if axis == "auto":
        response_length = n_tokens - response_idx
        could_be_response = bool(
            np.all(ranges[:, 0] >= 0) and np.all(ranges[:, 1] <= response_length)
        )
        could_be_full = bool(
            np.all(ranges[:, 0] >= response_idx) and np.all(ranges[:, 1] <= n_tokens)
        )
        if could_be_response and could_be_full and response_idx > 0:
            raise TraceFormatError(
                f"{source}: step range axis is ambiguous; set step_axis='response' or 'full'"
            )
        if could_be_response:
            axis = "response"
        elif could_be_full:
            axis = "full"
        else:
            raise TraceFormatError(
                f"{source}: step ranges fit neither the response-relative nor full token axis"
            )
    if axis == "response":
        ranges += response_idx
    if np.any(ranges[:, 0] < response_idx) or np.any(ranges[:, 1] > n_tokens):
        raise TraceFormatError(
            f"{source}: canonical step ranges fall outside response [{response_idx},{n_tokens})"
        )
    if np.any(ranges[:, 0] >= ranges[:, 1]):
        raise TraceFormatError(f"{source}: every step range must be non-empty and half-open")
    if len(ranges) > 1 and np.any(ranges[1:, 0] < ranges[:-1, 1]):
        raise TraceFormatError(f"{source}: step ranges overlap or are out of order")
    return ranges


def _text_id(value: Any) -> str:
    scalar = _scalar(value, name="identifier")
    return str(scalar)


def canonicalize_trace(
    mapping: Mapping[str, Any],
    *,
    config: Optional[TraceLoadConfig] = None,
    source_path: str = "",
    record_index: int = 0,
) -> AttentionTrace:
    """Validate and canonicalize one source mapping."""

    config = config or TraceLoadConfig()
    source = f"{source_path or '<memory>'}[{record_index}]"
    token_value, _ = _first(mapping, _TOKEN_KEYS)
    if token_value is None:
        raise TraceFormatError(f"{source}: missing one of token-id fields {_TOKEN_KEYS}")
    raw_token_ids = np.asarray(_as_numpy(token_value)).squeeze()
    if not np.issubdtype(raw_token_ids.dtype, np.integer):
        if (
            not np.issubdtype(raw_token_ids.dtype, np.number)
            or not np.isfinite(raw_token_ids).all()
            or not np.equal(raw_token_ids, np.floor(raw_token_ids)).all()
        ):
            raise TraceFormatError(f"{source}: token_ids must contain integers")
    token_ids = np.asarray(raw_token_ids, dtype=np.int64)
    if token_ids.ndim != 1 or token_ids.size == 0:
        raise TraceFormatError(f"{source}: token_ids must be a non-empty vector, got {token_ids.shape}")
    n_tokens = int(token_ids.size)

    response_value, response_key = _first(mapping, _RESPONSE_KEYS)
    if response_value is None:
        raise TraceFormatError(f"{source}: missing one of response offset fields {_RESPONSE_KEYS}")
    response_idx = _integer_scalar(response_value, name=response_key or "response_idx")
    if not 0 <= response_idx < n_tokens:
        raise TraceFormatError(
            f"{source}: response_idx={response_idx} outside non-empty sequence [0,{n_tokens})"
        )

    attention_value, _ = _first(mapping, _ATTENTION_KEYS)
    if attention_value is None:
        if config.require_attention:
            raise TraceFormatError(f"{source}: missing dense attention field {_ATTENTION_KEYS}")
        attention = np.zeros((0, 0, n_tokens, n_tokens), dtype=np.float32)
    else:
        attention = _canonical_attention(attention_value, n_tokens, source=source)
        if config.require_causal:
            upper = np.triu(attention, k=1)
            if float(np.max(np.abs(upper), initial=0.0)) > config.causal_tolerance:
                raise TraceFormatError(
                    f"{source}: attention has future-token mass above causal tolerance "
                    f"{config.causal_tolerance}"
                )

    layer_ids_value, _ = _first(mapping, _ATTENTION_LAYER_ID_KEYS)
    head_ids_value, _ = _first(mapping, _ATTENTION_HEAD_ID_KEYS)
    model_layers_value, _ = _first(mapping, _MODEL_LAYER_COUNT_KEYS)
    model_heads_value, _ = _first(mapping, _MODEL_HEAD_COUNT_KEYS)
    attention_layer_ids, num_model_layers = _canonical_attention_axis(
        layer_ids_value,
        model_layers_value,
        int(attention.shape[0]),
        name="layers",
        source=source,
    )
    attention_head_ids, num_model_heads = _canonical_attention_axis(
        head_ids_value,
        model_heads_value,
        int(attention.shape[1]),
        name="heads",
        source=source,
    )

    activation_value, _ = _first(mapping, _ACTIVATION_KEYS)
    activation = _canonical_activation(
        activation_value, n_tokens, config, source=source
    )

    token_label_value, token_label_key = _first(mapping, _TOKEN_LABEL_KEYS)
    token_y = _canonical_token_labels(
        token_label_value, n_tokens, response_idx, source=source
    )
    token_mask_value, _ = _first(mapping, _TOKEN_LABEL_MASK_KEYS)
    token_label_mask = _canonical_token_label_mask(
        token_mask_value, token_y, n_tokens, response_idx, source=source
    )

    range_value, _ = _first(mapping, _STEP_RANGE_KEYS)
    step_ranges = _canonical_step_ranges(
        range_value, n_tokens, response_idx, config, source=source
    )

    gold_value, gold_key = _first(mapping, _GOLD_STEP_KEYS)
    if gold_value is not None and step_ranges is not None and mapping.get("label") is not None:
        raise TraceFormatError(
            f"{source}: ambiguous gold-step aliases {gold_key!r} and 'label'"
        )
    if gold_value is None and step_ranges is not None and "label" in mapping:
        candidate = _as_numpy(mapping["label"])
        if candidate.size == 1:
            gold_value, gold_key = mapping["label"], "label"
    gold_step = (
        None
        if gold_value is None
        else _integer_scalar(gold_value, name=gold_key or "gold_step")
    )
    if gold_step is not None and step_ranges is None:
        raise TraceFormatError(f"{source}: gold_step is present but step_ranges are missing")
    if step_ranges is not None and gold_step is not None:
        if gold_step < -1 or gold_step >= len(step_ranges):
            raise TraceFormatError(
                f"{source}: gold_step={gold_step} outside -1..{len(step_ranges) - 1}"
            )

    if mapping.get("risk_mask") is not None:
        raise TraceFormatError(
            f"{source}: risk_mask is not accepted as step validity because legacy risk "
            "masks are commonly derived from gold_step; provide a label-independent "
            "step_loss_mask explicitly if observations are genuinely missing"
        )
    step_mask_value, _ = _first(mapping, _STEP_MASK_KEYS)
    step_loss_mask = None
    if step_mask_value is not None:
        if step_ranges is None:
            raise TraceFormatError(f"{source}: step_loss_mask is present but step_ranges are missing")
        step_loss_mask = np.asarray(_as_numpy(step_mask_value), dtype=bool).reshape(-1)
        if len(step_loss_mask) != len(step_ranges):
            raise TraceFormatError(
                f"{source}: step_loss_mask length {len(step_loss_mask)} != {len(step_ranges)} steps"
            )
        if not bool(np.any(step_loss_mask)):
            raise TraceFormatError(f"{source}: step_loss_mask selects no step")
        if gold_step is not None and gold_step >= 0 and not step_loss_mask[gold_step]:
            raise TraceFormatError(
                f"{source}: step_loss_mask hides the gold first-error step"
            )

    response_value, response_label_key = _first(mapping, _RESPONSE_LABEL_KEYS)
    if response_value is not None and mapping.get("is_correct") is not None:
        raise TraceFormatError(
            f"{source}: ambiguous response labels {response_label_key!r} and 'is_correct'"
        )
    # A scalar hallucination_labels field is response-level in the legacy data.
    if response_value is None and token_label_value is not None and token_y is None:
        response_value, response_label_key = token_label_value, token_label_key
    elif response_value is not None and token_label_value is not None and token_y is None:
        raise TraceFormatError(
            f"{source}: ambiguous scalar response labels {response_label_key!r} and "
            f"{token_label_key!r}"
        )
    if response_value is not None:
        response_y = _binary_scalar(
            response_value, name=response_label_key or "response_y"
        )
    elif "is_correct" in mapping:
        response_y = 1.0 - _binary_scalar(mapping["is_correct"], name="is_correct")
    elif gold_step is not None:
        response_y = float(gold_step >= 0)
    elif token_y is not None:
        response_labels = token_y[response_idx:]
        response_mask = (
            np.asarray(token_label_mask[response_idx:], dtype=bool)
            if token_label_mask is not None
            else np.isin(response_labels, (0.0, 1.0))
        )
        if np.any(response_mask & (response_labels == 1.0)):
            # One genuinely annotated hallucinated token is sufficient to make
            # the response positive, even when the remaining tokens are
            # unlabelled.
            response_y = 1.0
        elif len(response_mask) and bool(np.all(response_mask)):
            # A negative response can only be inferred when every response
            # token has an exact label.  Treating a partially observed clean
            # prefix as a negative response would fabricate ground truth.
            response_y = 0.0
        else:
            response_y = None
    else:
        response_y = None
    if response_y is not None and token_y is not None:
        response_labels = token_y[response_idx:]
        response_mask = np.asarray(token_label_mask[response_idx:], dtype=bool)
        has_exact_positive = bool(
            np.any(response_mask & (response_labels == 1.0))
        )
        full_exact_coverage = bool(len(response_mask) and np.all(response_mask))
        if response_y == 0.0 and has_exact_positive:
            raise TraceFormatError(
                f"{source}: response_y=0 conflicts with an exact positive token label"
            )
        if response_y == 1.0 and full_exact_coverage and not has_exact_positive:
            raise TraceFormatError(
                f"{source}: response_y=1 conflicts with fully observed all-negative token labels"
            )
    if (
        response_y is not None
        and gold_step is not None
        and response_y != float(gold_step >= 0)
    ):
        raise TraceFormatError(f"{source}: response_y conflicts with gold_step")

    trace_value, _ = _first(mapping, _ID_KEYS)
    default_id = f"{Path(source_path).stem or 'trace'}-{record_index}"
    trace_id = default_id if trace_value is None else _text_id(trace_value)
    group_value, group_key = _first(mapping, _GROUP_KEYS)
    group_is_fallback = group_value is None
    group_id = trace_id if group_is_fallback else _text_id(group_value)
    split_value, _ = _first(mapping, _SPLIT_KEYS)
    split = None if split_value is None else _text_id(split_value).lower()

    metadata = {
        "source_path": source_path,
        "record_index": int(record_index),
        "source_response_key": response_key,
        "source_token_label_key": token_label_key,
        "source_response_label_key": response_label_key,
        "canonical_step_end": "exclusive",
        "canonical_step_axis": "full",
    }
    for key in (*_PROVENANCE_KEYS, *_AUDIT_PROVENANCE_KEYS):
        if key in mapping and mapping[key] is not None:
            metadata[key] = _scalar(mapping[key], name=key)
    return AttentionTrace(
        attention=attention,
        token_ids=np.ascontiguousarray(token_ids),
        response_idx=response_idx,
        attention_layer_ids=attention_layer_ids,
        attention_head_ids=attention_head_ids,
        num_model_layers=num_model_layers,
        num_model_heads=num_model_heads,
        activation=activation,
        token_y=token_y,
        token_label_mask=token_label_mask,
        response_y=response_y,
        step_ranges=step_ranges,
        gold_step=gold_step,
        step_loss_mask=step_loss_mask,
        group_id=group_id,
        trace_id=trace_id,
        split=split,
        source_path=source_path,
        group_is_fallback=group_is_fallback,
        metadata=metadata,
    )


def discover_trace_files(inputs: Sequence[str], *, recursive: bool = True) -> Tuple[Path, ...]:
    """Resolve files/directories/globs deterministically and remove duplicates."""

    found: Dict[str, Path] = {}
    for raw in inputs:
        path = Path(raw).expanduser()
        candidates: Iterable[Path]
        if any(char in raw for char in "*?[]"):
            candidates = path.parent.glob(path.name)
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            candidates = path.glob(pattern)
        elif path.is_file():
            candidates = (path,)
        else:
            raise FileNotFoundError(f"trace input does not exist: {raw}")
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in {".npz", ".pt", ".pth"}:
                resolved = candidate.resolve()
                found[str(resolved).lower()] = resolved
    return tuple(found[key] for key in sorted(found))


def _load_pt(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError(f"loading {path.suffix} requires PyTorch") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older torch without weights_only.
        return torch.load(path, map_location="cpu")


def _npz_mapping(path: Path) -> Dict[str, Any]:
    # Object arrays are common in local research artifacts.  Treat input files
    # as trusted; never load an untrusted NPZ with pickle enabled.
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def _split_dense_batch(mapping: Mapping[str, Any], *, source: str) -> Iterator[Mapping[str, Any]]:
    attention_value, _ = _first(mapping, _ATTENTION_KEYS)
    if attention_value is None:
        yield mapping
        return
    attention = _as_numpy(attention_value)
    if attention.dtype == object and attention.ndim == 1:
        batch_size = int(len(attention))
    elif attention.ndim == 5:
        batch_size = int(attention.shape[0])
    else:
        yield mapping
        return

    def is_batched_field(key: str, array: np.ndarray) -> bool:
        if array.ndim == 0 or len(array) != batch_size:
            return False
        if key in _ATTENTION_KEYS:
            return True
        if array.dtype == object:
            return True
        if key in _TOKEN_KEYS:
            return array.ndim >= 2
        if key in _ACTIVATION_KEYS:
            return array.ndim >= 3
        if key in _TOKEN_LABEL_KEYS:
            return array.ndim >= 2
        if key in _TOKEN_LABEL_MASK_KEYS:
            return array.ndim >= 2
        if key in _STEP_RANGE_KEYS:
            return array.ndim >= 3
        if key in (
            *_RESPONSE_KEYS,
            *_RESPONSE_LABEL_KEYS,
            *_GOLD_STEP_KEYS,
            *_STEP_MASK_KEYS,
            *_GROUP_KEYS,
            *_ID_KEYS,
            *_SPLIT_KEYS,
            "label",
            "is_correct",
        ):
            return True
        return False

    for index in range(batch_size):
        record: Dict[str, Any] = {}
        for key, value in mapping.items():
            try:
                array = _as_numpy(value)
            except Exception:
                record[key] = value
                continue
            if is_batched_field(key, array):
                record[key] = value[index]
            else:
                record[key] = value
        yield record


def load_trace_file(
    path: str | Path, *, config: Optional[TraceLoadConfig] = None
) -> Tuple[AttentionTrace, ...]:
    """Load every trace record contained in one supported artifact."""

    path = Path(path).expanduser().resolve()
    config = config or TraceLoadConfig()
    if path.suffix.lower() == ".npz":
        raw = _npz_mapping(path)
    elif path.suffix.lower() in {".pt", ".pth"}:
        raw = _load_pt(path)
    else:
        raise TraceFormatError(f"unsupported trace extension: {path.suffix}")

    if isinstance(raw, Mapping):
        records = list(_split_dense_batch(raw, source=str(path)))
    elif isinstance(raw, (list, tuple)):
        records = list(raw)
    else:
        raise TraceFormatError(
            f"{path}: root object must be a mapping or list of mappings, got {type(raw).__name__}"
        )
    traces = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TraceFormatError(
                f"{path}[{index}]: record must be a mapping, got {type(record).__name__}"
            )
        traces.append(
            canonicalize_trace(record, config=config, source_path=str(path), record_index=index)
        )
    return tuple(traces)


def iter_traces(
    inputs: Sequence[str],
    *,
    config: Optional[TraceLoadConfig] = None,
    recursive: bool = True,
) -> Iterator[AttentionTrace]:
    files = discover_trace_files(inputs, recursive=recursive)
    if not files:
        raise FileNotFoundError("no .npz/.pt/.pth attention traces found")
    for path in files:
        yield from load_trace_file(path, config=config)


def trace_summary(trace: AttentionTrace) -> Dict[str, Any]:
    """Small JSON-safe record used by the inspect/build manifests."""

    attention = trace.attention
    upper_mass = (
        float(np.max(np.abs(np.triu(attention, k=1)), initial=0.0))
        if attention.size
        else None
    )
    return {
        "trace_id": trace.trace_id,
        "group_id": trace.group_id,
        "group_is_fallback": trace.group_is_fallback,
        "split": trace.split,
        "source_path": trace.source_path,
        "num_tokens": trace.num_tokens,
        "num_prompt_tokens": int(trace.response_idx),
        "num_response_tokens": trace.num_response_tokens,
        "num_layers": int(attention.shape[0]),
        "num_heads": int(attention.shape[1]),
        "attention_layer_ids": (
            None
            if trace.attention_layer_ids is None
            else trace.attention_layer_ids.tolist()
        ),
        "attention_head_ids": (
            None if trace.attention_head_ids is None else trace.attention_head_ids.tolist()
        ),
        "num_model_layers": trace.num_model_layers,
        "num_model_heads": trace.num_model_heads,
        "max_future_attention": upper_mass,
        "activation_dim": None if trace.activation is None else int(trace.activation.shape[1]),
        "has_exact_token_labels": trace.token_y is not None,
        "num_exact_labeled_tokens": (
            None if trace.token_label_mask is None else int(np.sum(trace.token_label_mask))
        ),
        "num_positive_tokens": (
            None if trace.token_y is None else int(np.sum(trace.token_y == 1.0))
        ),
        "response_y": trace.response_y,
        "num_steps": None if trace.step_ranges is None else int(len(trace.step_ranges)),
        "gold_step": trace.gold_step,
        "provenance_fingerprint": trace_provenance_fingerprint(trace),
        "provenance": {
            key: trace.metadata[key]
            for key in _PROVENANCE_KEYS
            if key in trace.metadata
        },
        "audit_provenance": {
            key: trace.metadata[key]
            for key in _AUDIT_PROVENANCE_KEYS
            if key in trace.metadata
        },
    }


def safe_trace_stem(trace: AttentionTrace) -> str:
    """Filesystem-safe but readable id; caller may append a content hash."""

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(trace.trace_id)).strip("._")
    return stem[:96] or "trace"


def dump_trace_summary(trace: AttentionTrace) -> str:
    return json.dumps(trace_summary(trace), ensure_ascii=False, sort_keys=True)


__all__ = [
    "MODEL_COMMIT_SOURCES",
    "TRACE_CONTRACT",
    "VERIFIED_MODEL_COMMIT_SOURCES",
    "AttentionTrace",
    "TraceFormatError",
    "TraceLoadConfig",
    "canonicalize_trace",
    "commit_hashes_match",
    "discover_trace_files",
    "dump_trace_summary",
    "iter_traces",
    "is_immutable_commit_hash",
    "load_trace_file",
    "model_identity_matches",
    "safe_trace_stem",
    "trace_summary",
    "trace_method_provenance",
    "trace_provenance_fingerprint",
]
