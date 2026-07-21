"""Construct faithful attention-row hypergraphs with NumPy only.

For every selected layer/head and response receiver ``i``, the constructor
creates one hyperedge from past tokens whose attention exceeds a threshold:

``{i} union {j < i : attention[layer, head, i, j] > threshold}``.

This is the mechanism used by the local original implementation.  Optional
top-k pruning, source restrictions, attention-weighted incidences, and
receiver-only propagation are explicit ablations rather than silent defaults.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .schema import (
    EDGE_ATTR_NAMES,
    EDGE_MARK_NAMES,
    EXTENDED_EDGE_ATTR_NAMES,
    AttentionHypergraph,
    AttentionHypergraphConfig,
)


_CONFIG_ALIASES = {
    "tau": "threshold",
    "include_receiver": "include_center",
    "prompt_policy": "source_scope",
    "incidence_weighting": "incidence_weight_mode",
}


def coerce_attention_config(
    config: Optional[Union[AttentionHypergraphConfig, Mapping[str, Any]]],
) -> AttentionHypergraphConfig:
    """Return a validated config, accepting a small set of CLI aliases.

    ``min_members`` is accepted for compatibility with the local original.  It
    counts total incidences, so the optional centre/receiver is subtracted when
    converting it to the canonical ``min_sources`` field.
    """

    if config is None:
        return AttentionHypergraphConfig()
    if isinstance(config, AttentionHypergraphConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("config must be AttentionHypergraphConfig, a mapping, or None")

    allowed = {item.name for item in fields(AttentionHypergraphConfig)}
    raw_config = {str(key): value for key, value in config.items()}
    if "min_members" in raw_config:
        if "min_sources" in raw_config:
            raise ValueError("config cannot specify both min_members and min_sources")
        include_center = raw_config.get(
            "include_center", raw_config.get("include_receiver", True)
        )
        min_sources = _canonical_integer_scalar(
            raw_config.pop("min_members"), name="min_members"
        ) - int(bool(include_center))
        if min_sources < 1:
            raise ValueError(
                "legacy min_members must leave at least one historical source after "
                "accounting for the receiver centre"
            )
        raw_config["min_sources"] = min_sources
    canonical = {}
    for raw_key, value in raw_config.items():
        key = _CONFIG_ALIASES.get(str(raw_key), str(raw_key))
        if key not in allowed:
            raise ValueError(f"unknown attention hypergraph config field: {raw_key!r}")
        if key in canonical:
            raise ValueError(f"config specifies {key!r} more than once")
        canonical[key] = value
    return AttentionHypergraphConfig(**canonical)


def _canonical_attention(attention: np.ndarray) -> np.ndarray:
    value = np.asarray(attention)
    if value.ndim != 4:
        raise ValueError(
            "attention must have shape (layers, heads, tokens, tokens), "
            f"got {value.shape}"
        )
    n_layers, n_heads, n_queries, n_keys = value.shape
    if n_layers < 1 or n_heads < 1 or n_queries < 1 or n_queries != n_keys:
        raise ValueError(
            "attention must contain at least one layer/head/token and have square token axes"
        )
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError("attention must be numeric")
    value = np.asarray(value, np.float32)
    if not np.isfinite(value).all():
        raise ValueError("attention contains NaN or infinity")
    tolerance = 1e-6
    if float(value.min()) < -tolerance or float(value.max()) > 1.0 + tolerance:
        raise ValueError("attention values must lie in [0, 1]")
    # Avoid another full [L,H,N,N] allocation for already valid arrays.  Tiny
    # numerical excursions are clipped once; canonical float32 inputs are then
    # reused by all downstream builders.
    if float(value.min()) < 0.0 or float(value.max()) > 1.0:
        value = np.clip(value, 0.0, 1.0)
    return value


def _canonical_integer_scalar(value: Any, *, name: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be an integer scalar")
    scalar = array.reshape(-1)[0]
    if not np.issubdtype(array.dtype, np.integer):
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(scalar) or scalar != np.floor(scalar):
            raise ValueError(f"{name} must be an integer scalar")
    return int(scalar)


def _canonical_axis_ids(
    values: Optional[Sequence[int]],
    size: int,
    total_size: Optional[int],
    *,
    name: str,
) -> Tuple[np.ndarray, int]:
    if values is None:
        ids = np.arange(size, dtype=np.int64)
    else:
        raw_ids = np.asarray(values)
        if not np.issubdtype(raw_ids.dtype, np.integer):
            raise ValueError(f"{name} must contain integer model-axis ids")
        ids = np.asarray(raw_ids, dtype=np.int64)
    if ids.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {ids.shape}")
    if np.any(ids < 0) or len(np.unique(ids)) != size:
        raise ValueError(f"{name} must contain unique non-negative model-axis ids")
    inferred_total = int(ids.max()) + 1
    total = (
        inferred_total
        if total_size is None
        else _canonical_integer_scalar(total_size, name=f"num_model_{name}")
    )
    if total < inferred_total:
        raise ValueError(
            f"total size {total} cannot contain maximum {name} id {int(ids.max())}"
        )
    return np.ascontiguousarray(ids), total


def _selected_axis_positions(
    requested: Optional[Tuple[int, ...]], axis_ids: np.ndarray, *, name: str
) -> Tuple[int, ...]:
    if requested is None:
        return tuple(range(len(axis_ids)))
    lookup = {int(axis_id): position for position, axis_id in enumerate(axis_ids)}
    missing = [axis_id for axis_id in requested if axis_id not in lookup]
    if missing:
        raise ValueError(
            f"{name} requests model-axis ids {missing}, but this trace stores "
            f"{axis_ids.tolist()}"
        )
    return tuple(lookup[axis_id] for axis_id in requested)


def build_attention_node_features(
    attention: np.ndarray,
    activation: Optional[np.ndarray] = None,
    *,
    mode: str = "attention_diagonal",
) -> np.ndarray:
    """Return self-attention diagonals plus optional node-major activations.

    ``attention`` has shape ``(L,H,N,N)``.  Diagonal values are flattened to
    ``(N,L*H)`` in layer-major/head-minor order.  ``activation`` must be
    node-major with shape ``(N,F)`` (or ``(N,...)``, which is flattened).  The
    loader, not this function, must select activation layers and put the token
    axis first; this strictness prevents silent BOS/axis misalignment.
    """

    value = _canonical_attention(attention)
    n_layers, n_heads, n_tokens, _ = value.shape
    diagonal = np.diagonal(value, axis1=2, axis2=3)
    self_attention = diagonal.transpose(2, 0, 1).reshape(n_tokens, n_layers * n_heads)
    if mode not in {
        "attention_diagonal",
        "activation_only",
        "diagonal_plus_activation",
    }:
        raise ValueError(f"unknown node feature mode: {mode!r}")
    parts = (
        [np.asarray(self_attention, np.float32)]
        if mode in {"attention_diagonal", "diagonal_plus_activation"}
        else []
    )

    needs_activation = mode in {"activation_only", "diagonal_plus_activation"}
    if needs_activation and activation is None:
        raise ValueError(f"node_feature_mode={mode!r} requires aligned activation features")
    if activation is not None and not needs_activation:
        raise ValueError(
            "activation was provided but node_feature_mode='attention_diagonal'; "
            "select activation_only or diagonal_plus_activation explicitly"
        )
    if activation is not None and needs_activation:
        node_activation = np.asarray(activation)
        if node_activation.ndim < 2 or node_activation.shape[0] != n_tokens:
            raise ValueError(
                "activation must be node-major with shape (tokens, features...), "
                f"expected first dimension {n_tokens}, got {node_activation.shape}"
            )
        if not np.issubdtype(node_activation.dtype, np.number):
            raise TypeError("activation must be numeric")
        node_activation = np.asarray(node_activation, np.float32).reshape(n_tokens, -1)
        if node_activation.shape[1] < 1:
            raise ValueError("activation must have at least one feature column")
        if not np.isfinite(node_activation).all():
            raise ValueError("activation contains NaN or infinity")
        parts.append(node_activation)

    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)


def _candidate_sources(receiver: int, response_idx: int, source_scope: str) -> np.ndarray:
    if source_scope == "all_past":
        return np.arange(receiver, dtype=np.int64)
    if source_scope == "prompt_only":
        return np.arange(min(receiver, response_idx), dtype=np.int64)
    if source_scope == "response_only":
        return np.arange(response_idx, receiver, dtype=np.int64)
    raise ValueError(f"unknown source_scope: {source_scope!r}")


def _top_k_sources(sources: np.ndarray, scores: np.ndarray, top_k: Optional[int]) -> np.ndarray:
    if top_k is None or len(sources) <= top_k:
        return sources
    # Stable tie handling makes top-k deterministic; restore token order so the
    # output matches threshold-only construction's chronological membership.
    order = np.argsort(-scores, kind="stable")[:top_k]
    return np.sort(sources[order])


def _select_sources(
    candidates: np.ndarray,
    scores: np.ndarray,
    config: AttentionHypergraphConfig,
) -> np.ndarray:
    """Apply one explicit sparsifier to a receiver's eligible history."""

    if config.source_selection == "threshold":
        sources = candidates[scores > float(config.threshold)]
        if len(sources):
            sources = _top_k_sources(
                sources,
                scores[scores > float(config.threshold)],
                config.top_k,
            )
        return sources
    if config.source_selection == "threshold_fallback_topk":
        sources = candidates[scores > float(config.threshold)]
        if len(sources):
            return sources
        positive = scores > 0.0
        return _top_k_sources(candidates[positive], scores[positive], config.top_k)
    if config.source_selection == "top_k_only":
        return _top_k_sources(candidates, scores, config.top_k)
    if config.source_selection == "cumulative_mass":
        positive = scores > 0.0
        sources = candidates[positive]
        positive_scores = scores[positive]
        total = float(np.sum(positive_scores))
        if not len(sources) or total <= 0.0:
            return np.zeros((0,), dtype=np.int64)
        order = np.argsort(-positive_scores, kind="stable")
        cutoff = int(
            np.searchsorted(
                np.cumsum(positive_scores[order]),
                float(config.cumulative_mass) * total,
                side="left",
            )
        ) + 1
        return np.sort(sources[order[:cutoff]])
    raise ValueError(f"unknown source_selection: {config.source_selection!r}")


def _incidence_weights(raw_attention: np.ndarray, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones(len(raw_attention), dtype=np.float32)
    if mode == "attention":
        return np.asarray(raw_attention, np.float32)
    if mode == "normalized_attention":
        denominator = float(np.sum(raw_attention))
        if denominator <= 0.0:
            # This is only reachable for a malformed zero-attention edge; the
            # thresholded source contract normally guarantees a positive sum.
            return np.full(len(raw_attention), 1.0 / len(raw_attention), np.float32)
        return np.asarray(raw_attention / denominator, np.float32)
    raise ValueError(f"unknown incidence_weight_mode: {mode!r}")


def build_attention_hyperedges(
    attention: np.ndarray,
    response_idx: int,
    config: Optional[Union[AttentionHypergraphConfig, Mapping[str, Any]]] = None,
    *,
    attention_layer_ids: Optional[Sequence[int]] = None,
    attention_head_ids: Optional[Sequence[int]] = None,
    num_model_layers: Optional[int] = None,
    num_model_heads: Optional[int] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build attention-row hyperedges and return canonical NumPy arrays.

    Returns ``(he_index, he_attr, he_mark, he_count, he_weight,
    he_attention, he_receiver, he_layer, he_head)``.  Only response tokens are
    receivers, so prompt-to-prompt hyperedges are excluded by construction.
    Thresholding is strict (``attention > threshold``), matching the original.
    """

    value = _canonical_attention(attention)
    cfg = coerce_attention_config(config)
    n_layers, n_heads, n_tokens, _ = value.shape
    response_idx = _canonical_integer_scalar(response_idx, name="response_idx")
    if not 0 <= response_idx < n_tokens:
        raise ValueError(f"response_idx must lie in [0, {n_tokens}), got {response_idx}")
    layer_ids, total_layers = _canonical_axis_ids(
        attention_layer_ids,
        n_layers,
        num_model_layers,
        name="attention_layer_ids",
    )
    head_ids, total_heads = _canonical_axis_ids(
        attention_head_ids,
        n_heads,
        num_model_heads,
        name="attention_head_ids",
    )
    layers = _selected_axis_positions(
        cfg.selected_layers, layer_ids, name="selected_layers"
    )
    heads = _selected_axis_positions(cfg.selected_heads, head_ids, name="selected_heads")

    member_groups: List[np.ndarray] = []
    attention_groups: List[np.ndarray] = []
    weight_groups: List[np.ndarray] = []
    attrs: List[List[float]] = []
    marks: List[List[float]] = []
    receivers: List[int] = []
    edge_layers: List[int] = []
    edge_heads: List[int] = []

    for layer in layers:
        model_layer = int(layer_ids[layer])
        layer_normalized = float(model_layer) / float(max(1, total_layers - 1))
        for head in heads:
            model_head = int(head_ids[head])
            head_normalized = float(model_head) / float(max(1, total_heads - 1))
            # The local original flattens the stored [L,H,...] tensor before
            # constructing this three-column edge attribute. Preserve that
            # local channel index; global model ids remain in he_layer/he_head.
            flattened_head = layer * n_heads + head
            flattened_head_normalized = float(flattened_head) / float(
                max(1, n_layers * n_heads - 1)
            )
            matrix = value[layer, head]
            for receiver in range(response_idx, n_tokens):
                candidates = _candidate_sources(receiver, response_idx, cfg.source_scope)
                if not len(candidates):
                    continue
                candidate_scores = matrix[receiver, candidates]
                sources = _select_sources(candidates, candidate_scores, cfg)
                if len(sources) < cfg.min_sources:
                    continue

                if cfg.include_center:
                    members = np.concatenate(
                        [sources, np.asarray([receiver], dtype=np.int64)]
                    )
                else:
                    members = sources
                raw_attention = np.asarray(matrix[receiver, members], np.float32)
                member_groups.append(members)
                attention_groups.append(raw_attention)
                weight_groups.append(
                    _incidence_weights(raw_attention, cfg.incidence_weight_mode)
                )
                member_count = int(len(members))
                if cfg.edge_attr_mode == "faithful":
                    attrs.append(
                        [
                            float(np.mean(raw_attention)),
                            float(np.max(raw_attention)),
                            flattened_head_normalized,
                        ]
                    )
                else:
                    attrs.append(
                        [
                            float(np.mean(raw_attention)),
                            float(np.max(raw_attention)),
                            layer_normalized,
                            head_normalized,
                            float(member_count) / float(max(1, n_tokens)),
                            float(np.log1p(member_count)) / float(np.log1p(n_tokens)),
                        ]
                    )
                has_prompt_source = bool(np.any(sources < response_idx))
                marks.append([1.0, 0.0] if has_prompt_source else [0.0, 1.0])
                receivers.append(receiver)
                edge_layers.append(model_layer)
                edge_heads.append(model_head)

    if not member_groups:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros(
                (
                    0,
                    len(
                        EDGE_ATTR_NAMES
                        if cfg.edge_attr_mode == "faithful"
                        else EXTENDED_EDGE_ATTR_NAMES
                    ),
                ),
                dtype=np.float32,
            ),
            np.zeros((0, len(EDGE_MARK_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    nodes = np.concatenate(member_groups).astype(np.int64, copy=False)
    counts = np.asarray([len(members) for members in member_groups], dtype=np.int64)
    edge_ids = np.repeat(np.arange(len(member_groups), dtype=np.int64), counts)
    return (
        np.vstack([nodes, edge_ids]),
        np.asarray(attrs, dtype=np.float32),
        np.asarray(marks, dtype=np.float32),
        counts,
        np.concatenate(weight_groups).astype(np.float32, copy=False),
        np.concatenate(attention_groups).astype(np.float32, copy=False),
        np.asarray(receivers, dtype=np.int64),
        np.asarray(edge_layers, dtype=np.int64),
        np.asarray(edge_heads, dtype=np.int64),
    )


def _canonical_optional_labels(
    token_y: Optional[np.ndarray],
    token_label_mask: Optional[np.ndarray],
    response_y: Optional[float],
    step_ranges: Optional[np.ndarray],
    gold_step: Optional[int],
    step_loss_mask: Optional[np.ndarray],
    *,
    n_tokens: int,
    response_idx: int,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[float],
    Optional[np.ndarray],
    Optional[int],
    Optional[np.ndarray],
]:
    canonical_token_y: Optional[np.ndarray] = None
    canonical_token_mask: Optional[np.ndarray] = None
    if token_y is not None:
        canonical_token_y = np.asarray(token_y, np.float32).copy()
        if canonical_token_y.shape != (n_tokens,):
            raise ValueError(f"token_y must have shape ({n_tokens},), got {canonical_token_y.shape}")
        finite = np.isfinite(canonical_token_y)
        canonical_token_y[~finite] = -100.0
        if not np.all(np.isin(canonical_token_y, (-100.0, 0.0, 1.0))):
            raise ValueError("token_y must contain 0/1 labels or the -100 ignore sentinel")
        if np.any(canonical_token_y[:response_idx] == 1.0):
            raise ValueError("Prompt tokens cannot be marked as positive hallucination labels")
        derived_mask = np.isin(canonical_token_y, (0.0, 1.0))
        derived_mask[:response_idx] = False
        if token_label_mask is None:
            canonical_token_mask = derived_mask
        else:
            canonical_token_mask = np.asarray(token_label_mask, np.bool_)
            if canonical_token_mask.shape != (n_tokens,):
                raise ValueError(
                    f"token_label_mask must have shape ({n_tokens},), "
                    f"got {canonical_token_mask.shape}"
                )
            if np.any(canonical_token_mask[:response_idx]):
                raise ValueError("token_label_mask cannot supervise prompt tokens")
            if np.any(canonical_token_mask & ~np.isin(canonical_token_y, (0.0, 1.0))):
                raise ValueError("token_label_mask selects ignored/non-binary token labels")
    elif token_label_mask is not None:
        raise ValueError("token_label_mask requires token_y")

    canonical_response_y: Optional[float] = None
    if response_y is not None:
        canonical_response_y = float(response_y)
        if not np.isfinite(canonical_response_y) or canonical_response_y not in (0.0, 1.0):
            raise ValueError("response_y must be a finite binary scalar")

    canonical_ranges: Optional[np.ndarray] = None
    if step_ranges is not None:
        raw_ranges = np.asarray(step_ranges)
        if raw_ranges.ndim != 2 or raw_ranges.shape[1] != 2:
            raise ValueError(f"step_ranges must have shape (steps, 2), got {raw_ranges.shape}")
        if not np.issubdtype(raw_ranges.dtype, np.integer):
            if not np.issubdtype(raw_ranges.dtype, np.number) or not np.all(
                np.equal(raw_ranges, np.floor(raw_ranges))
            ):
                raise ValueError("step_ranges must contain integer offsets on the token axis")
        canonical_ranges = np.asarray(raw_ranges, np.int64)
        if len(canonical_ranges):
            starts, stops = canonical_ranges[:, 0], canonical_ranges[:, 1]
            if np.any(starts < response_idx) or np.any(stops > n_tokens) or np.any(starts >= stops):
                raise ValueError(
                    "step_ranges must be non-empty absolute half-open ranges within the response"
                )
            if np.any(starts[1:] < stops[:-1]):
                raise ValueError("step_ranges must be sorted and non-overlapping")

    canonical_gold_step: Optional[int] = None
    if gold_step is not None:
        if canonical_ranges is None:
            raise ValueError("gold_step requires step_ranges")
        canonical_gold_step = _canonical_integer_scalar(gold_step, name="gold_step")
        if canonical_gold_step != -1 and not 0 <= canonical_gold_step < len(canonical_ranges):
            raise ValueError("gold_step must be -1 (no erroneous step) or index step_ranges")

    canonical_step_mask: Optional[np.ndarray] = None
    if step_loss_mask is not None:
        if canonical_ranges is None:
            raise ValueError("step_loss_mask requires step_ranges")
        canonical_step_mask = np.asarray(step_loss_mask, np.bool_)
        if canonical_step_mask.shape != (len(canonical_ranges),):
            raise ValueError(
                "step_loss_mask must have exactly one entry per step, "
                f"got {canonical_step_mask.shape} for {len(canonical_ranges)} steps"
            )
        if not np.any(canonical_step_mask):
            raise ValueError("step_loss_mask must select at least one step")
        if canonical_gold_step is not None and canonical_gold_step >= 0 and not canonical_step_mask[
            canonical_gold_step
        ]:
            raise ValueError("step_loss_mask cannot hide the gold first-error step")

    if (
        canonical_response_y is not None
        and canonical_gold_step is not None
        and canonical_response_y != float(canonical_gold_step >= 0)
    ):
        raise ValueError("response_y conflicts with gold_step")
    if canonical_response_y is not None and canonical_token_y is not None:
        assert canonical_token_mask is not None
        response_labels = canonical_token_y[response_idx:]
        response_mask = canonical_token_mask[response_idx:]
        has_exact_positive = bool(
            np.any(response_mask & (response_labels == 1.0))
        )
        full_exact_coverage = bool(len(response_mask) and np.all(response_mask))
        if canonical_response_y == 0.0 and has_exact_positive:
            raise ValueError("response_y=0 conflicts with an exact positive token label")
        if (
            canonical_response_y == 1.0
            and full_exact_coverage
            and not has_exact_positive
        ):
            raise ValueError(
                "response_y=1 conflicts with fully observed all-negative token labels"
            )

    return (
        canonical_token_y,
        canonical_token_mask,
        canonical_response_y,
        canonical_ranges,
        canonical_gold_step,
        canonical_step_mask,
    )


def validate_attention_hypergraph(
    graph: AttentionHypergraph,
    config: Optional[Union[AttentionHypergraphConfig, Mapping[str, Any]]] = None,
) -> None:
    """Fail fast on shape, alignment, causality, and label-granularity errors."""

    cfg = graph.construction_config if config is None else coerce_attention_config(config)
    if not isinstance(graph.construction_config, AttentionHypergraphConfig):
        raise TypeError("graph.construction_config must be AttentionHypergraphConfig")
    if graph.construction_config != cfg:
        raise ValueError("graph construction_config does not match validation config")
    if graph.x.ndim != 2 or graph.x.shape[0] < 1 or graph.x.shape[1] < 1:
        raise ValueError("x must have shape (tokens, positive_feature_dim)")
    if not np.isfinite(graph.x).all():
        raise ValueError("x contains NaN or infinity")
    n_nodes = graph.num_nodes
    if graph.token_ids.shape != (n_nodes,):
        raise ValueError("token_ids must contain exactly one id per graph node")
    if graph.attention_layer_ids.ndim != 1 or not len(graph.attention_layer_ids):
        raise ValueError("attention_layer_ids must be a non-empty vector")
    if graph.attention_head_ids.ndim != 1 or not len(graph.attention_head_ids):
        raise ValueError("attention_head_ids must be a non-empty vector")
    if (
        np.any(graph.attention_layer_ids < 0)
        or len(np.unique(graph.attention_layer_ids)) != len(graph.attention_layer_ids)
        or np.any(graph.attention_head_ids < 0)
        or len(np.unique(graph.attention_head_ids)) != len(graph.attention_head_ids)
    ):
        raise ValueError("stored attention layer/head ids must be unique and non-negative")
    if (
        int(graph.num_model_layers) <= int(np.max(graph.attention_layer_ids))
        or int(graph.num_model_heads) <= int(np.max(graph.attention_head_ids))
    ):
        raise ValueError("model layer/head totals do not contain the stored attention axes")
    faithful_node_dim = len(graph.attention_layer_ids) * len(graph.attention_head_ids)
    if (
        cfg.node_feature_mode == "attention_diagonal"
        and graph.x.shape[1] != faithful_node_dim
    ):
        raise ValueError("attention_diagonal x must contain exactly one column per layer/head")
    if (
        cfg.node_feature_mode == "diagonal_plus_activation"
        and graph.x.shape[1] <= faithful_node_dim
    ):
        raise ValueError("diagonal_plus_activation x is missing activation feature columns")
    if not 0 <= int(graph.response_idx) < n_nodes:
        raise ValueError("response_idx lies outside the token sequence")
    if graph.propagation_mode != cfg.propagation_mode:
        raise ValueError("graph propagation_mode does not match construction config")
    if graph.incidence_weight_mode != cfg.incidence_weight_mode:
        raise ValueError("graph incidence_weight_mode does not match construction config")

    n_edges = graph.num_hyperedges
    n_incidence = graph.num_incidences
    if graph.he_index.shape != (2, n_incidence):
        raise ValueError("he_index must have shape (2, incidences)")
    expected_edge_names = (
        EDGE_ATTR_NAMES if cfg.edge_attr_mode == "faithful" else EXTENDED_EDGE_ATTR_NAMES
    )
    if tuple(graph.edge_attr_names) != tuple(expected_edge_names):
        raise ValueError("edge_attr_names do not match edge_attr_mode")
    if graph.he_attr.shape != (n_edges, len(expected_edge_names)):
        raise ValueError("he_attr has an invalid shape")
    if graph.he_mark.shape != (n_edges, len(EDGE_MARK_NAMES)):
        raise ValueError("he_mark has an invalid shape")
    for name, value in (
        ("he_count", graph.he_count),
        ("he_receiver", graph.he_receiver),
        ("he_layer", graph.he_layer),
        ("he_head", graph.he_head),
    ):
        if value.shape != (n_edges,):
            raise ValueError(f"{name} must have shape ({n_edges},)")
    for name, value in (("he_weight", graph.he_weight), ("he_attention", graph.he_attention)):
        if value.shape != (n_incidence,):
            raise ValueError(f"{name} must align one-to-one with incidences")
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
    if not np.isfinite(graph.he_attr).all() or not np.isfinite(graph.he_mark).all():
        raise ValueError("hyperedge attributes/marks must be finite")
    if n_edges and (
        np.any(graph.he_attr < 0.0)
        or np.any(graph.he_attr > 1.0)
        or np.any(graph.he_mark < 0.0)
        or np.any(graph.he_mark > 1.0)
    ):
        raise ValueError("normalized edge attributes and marks must lie in [0, 1]")
    if np.any(graph.he_attention > 1.0 + 1e-6):
        raise ValueError("he_attention values must lie in [0, 1]")
    if n_edges and (
        np.any(graph.he_receiver < graph.response_idx) or np.any(graph.he_receiver >= n_nodes)
    ):
        raise ValueError("every hyperedge receiver must be a response token")
    if n_edges:
        selected_layer_ids = (
            set(graph.attention_layer_ids.tolist())
            if cfg.selected_layers is None
            else set(cfg.selected_layers)
        )
        selected_head_ids = (
            set(graph.attention_head_ids.tolist())
            if cfg.selected_heads is None
            else set(cfg.selected_heads)
        )
        if not set(graph.he_layer.tolist()).issubset(selected_layer_ids):
            raise ValueError("he_layer contains an unselected model layer id")
        if not set(graph.he_head.tolist()).issubset(selected_head_ids):
            raise ValueError("he_head contains an unselected model head id")
        if cfg.edge_attr_mode == "faithful":
            layer_positions = {
                int(value): position
                for position, value in enumerate(graph.attention_layer_ids.tolist())
            }
            head_positions = {
                int(value): position
                for position, value in enumerate(graph.attention_head_ids.tolist())
            }
            expected_flattened = np.asarray(
                [
                    layer_positions[int(layer)] * len(head_positions)
                    + head_positions[int(head)]
                    for layer, head in zip(graph.he_layer, graph.he_head)
                ],
                dtype=np.float64,
            ) / float(max(1, len(layer_positions) * len(head_positions) - 1))
            if not np.allclose(graph.he_attr[:, 2], expected_flattened):
                raise ValueError(
                    "flattened local layer-head edge attribute disagrees with stored axes"
                )
        else:
            expected_layer = graph.he_layer.astype(np.float64) / float(
                max(1, int(graph.num_model_layers) - 1)
            )
            expected_head = graph.he_head.astype(np.float64) / float(
                max(1, int(graph.num_model_heads) - 1)
            )
            if not np.allclose(graph.he_attr[:, 2], expected_layer) or not np.allclose(
                graph.he_attr[:, 3], expected_head
            ):
                raise ValueError("layer/head edge attributes disagree with axis metadata")
    if n_edges and (np.any(graph.he_count < 1) or int(graph.he_count.sum()) != n_incidence):
        raise ValueError("he_count must contain positive counts summing to incidence count")
    if n_edges and not np.allclose(graph.he_mark.sum(axis=1), 1.0):
        raise ValueError("he_mark rows must be one-hot")

    if n_incidence:
        nodes, edge_ids = graph.he_index
        if np.any(nodes < 0) or np.any(nodes >= n_nodes):
            raise ValueError("he_index contains an out-of-range node")
        if np.any(edge_ids < 0) or np.any(edge_ids >= n_edges):
            raise ValueError("he_index contains an out-of-range edge")
        expected_edge_ids = np.repeat(np.arange(n_edges, dtype=np.int64), graph.he_count)
        if not np.array_equal(edge_ids, expected_edge_ids):
            raise ValueError("incidences must be grouped contiguously in hyperedge order")
        if cfg.edge_attr_mode == "extended":
            if not np.allclose(
                graph.he_attr[:, 4], graph.he_count.astype(np.float32) / float(n_nodes)
            ):
                raise ValueError("member_fraction edge attribute disagrees with he_count")
            if not np.allclose(
                graph.he_attr[:, 5],
                np.log1p(graph.he_count.astype(np.float32)) / float(np.log1p(n_nodes)),
            ):
                raise ValueError(
                    "normalized log1p member-count edge attribute disagrees with he_count"
                )
        incidence_receivers = graph.he_receiver[edge_ids]
        if np.any(nodes > incidence_receivers):
            raise ValueError("attention hyperedges cannot contain future tokens")
        for edge in range(n_edges):
            edge_nodes = nodes[edge_ids == edge]
            if len(np.unique(edge_nodes)) != len(edge_nodes):
                raise ValueError("a hyperedge cannot contain duplicate members")
            receiver_count = int(np.count_nonzero(edge_nodes == graph.he_receiver[edge]))
            expected_receiver_count = 1 if cfg.include_center else 0
            if receiver_count != expected_receiver_count:
                raise ValueError("receiver membership does not match include_center")
            source_nodes = edge_nodes[edge_nodes != graph.he_receiver[edge]]
            if len(source_nodes) < cfg.min_sources or np.any(source_nodes >= graph.he_receiver[edge]):
                raise ValueError("each edge must contain the configured number of past sources")
            expected_mark = np.asarray(
                [1.0, 0.0]
                if np.any(source_nodes < graph.response_idx)
                else [0.0, 1.0],
                np.float32,
            )
            if not np.array_equal(graph.he_mark[edge], expected_mark):
                raise ValueError("he_mark disagrees with prompt/response membership")
            edge_attention = graph.he_attention[edge_ids == edge]
            if not np.isclose(graph.he_attr[edge, 0], np.mean(edge_attention)):
                raise ValueError("attention_mean edge attribute disagrees with incidences")
            if not np.isclose(graph.he_attr[edge, 1], np.max(edge_attention)):
                raise ValueError("attention_max edge attribute disagrees with incidences")
            if cfg.source_scope == "prompt_only" and np.any(source_nodes >= graph.response_idx):
                raise ValueError("prompt_only edges contain a response source")
            if cfg.source_scope == "response_only" and np.any(source_nodes < graph.response_idx):
                raise ValueError("response_only edges contain a prompt source")
            if cfg.top_k is not None and len(source_nodes) > cfg.top_k:
                raise ValueError("hyperedge contains more sources than configured top_k")

        if cfg.incidence_weight_mode == "uniform" and not np.allclose(graph.he_weight, 1.0):
            raise ValueError("uniform incidence weighting requires every he_weight to be one")
        if cfg.incidence_weight_mode == "attention" and not np.allclose(
            graph.he_weight, graph.he_attention
        ):
            raise ValueError("attention incidence weighting must preserve raw attention")
        if cfg.incidence_weight_mode == "normalized_attention":
            edge_weight_sums = np.add.reduceat(
                graph.he_weight, np.r_[0, np.cumsum(graph.he_count)[:-1]]
            )
            if not np.allclose(edge_weight_sums, 1.0):
                raise ValueError("normalized attention weights must sum to one per edge")

    # Reuse the same strict label checks without deriving or mutating labels.
    canonical_labels = _canonical_optional_labels(
        graph.token_y,
        graph.token_label_mask,
        graph.response_y,
        graph.step_ranges,
        graph.gold_step,
        graph.step_loss_mask,
        n_tokens=n_nodes,
        response_idx=int(graph.response_idx),
    )
    if graph.token_y is not None and not np.array_equal(
        graph.token_y, canonical_labels[0], equal_nan=True
    ):
        raise ValueError("token_y is not canonical")
    if graph.token_label_mask is not None and not np.array_equal(
        graph.token_label_mask, canonical_labels[1]
    ):
        raise ValueError("token_label_mask is not canonical")


def build_attention_hypergraph(
    attention: np.ndarray,
    token_ids: Sequence[int],
    response_idx: int,
    *,
    attention_layer_ids: Optional[Sequence[int]] = None,
    attention_head_ids: Optional[Sequence[int]] = None,
    num_model_layers: Optional[int] = None,
    num_model_heads: Optional[int] = None,
    activation: Optional[np.ndarray] = None,
    token_y: Optional[np.ndarray] = None,
    token_label_mask: Optional[np.ndarray] = None,
    response_y: Optional[float] = None,
    step_ranges: Optional[np.ndarray] = None,
    gold_step: Optional[int] = None,
    step_loss_mask: Optional[np.ndarray] = None,
    trace_id: str = "",
    group_id: str = "",
    split: Optional[str] = None,
    config: Optional[Union[AttentionHypergraphConfig, Mapping[str, Any]]] = None,
) -> AttentionHypergraph:
    """Construct one faithful prompt+response attention hypergraph.

    ``token_y`` is accepted only when exact token labels exist.  Passing
    ``gold_step`` stores the step target as-is; it deliberately does *not*
    synthesize token labels for every token in the erroneous step.
    """

    value = _canonical_attention(attention)
    cfg = coerce_attention_config(config)
    n_tokens = value.shape[2]
    ids = np.asarray(token_ids)
    if ids.shape != (n_tokens,):
        raise ValueError(f"token_ids must have shape ({n_tokens},), got {ids.shape}")
    if not np.issubdtype(ids.dtype, np.integer):
        if not np.issubdtype(ids.dtype, np.number) or not np.all(np.equal(ids, np.floor(ids))):
            raise ValueError("token_ids must contain integers")
    ids = np.asarray(ids, np.int64)
    response_idx = _canonical_integer_scalar(response_idx, name="response_idx")
    if not 0 <= response_idx < n_tokens:
        raise ValueError(f"response_idx must lie in [0, {n_tokens}), got {response_idx}")

    layer_ids, total_layers = _canonical_axis_ids(
        attention_layer_ids,
        int(value.shape[0]),
        num_model_layers,
        name="attention_layer_ids",
    )
    head_ids, total_heads = _canonical_axis_ids(
        attention_head_ids,
        int(value.shape[1]),
        num_model_heads,
        name="attention_head_ids",
    )
    x = build_attention_node_features(value, activation, mode=cfg.node_feature_mode)
    (
        he_index,
        he_attr,
        he_mark,
        he_count,
        he_weight,
        he_attention,
        he_receiver,
        he_layer,
        he_head,
    ) = build_attention_hyperedges(
        value,
        response_idx,
        cfg,
        attention_layer_ids=layer_ids,
        attention_head_ids=head_ids,
        num_model_layers=total_layers,
        num_model_heads=total_heads,
    )
    canonical_labels = _canonical_optional_labels(
        token_y,
        token_label_mask,
        response_y,
        step_ranges,
        gold_step,
        step_loss_mask,
        n_tokens=n_tokens,
        response_idx=response_idx,
    )
    graph = AttentionHypergraph(
        x=x,
        token_ids=ids,
        response_idx=response_idx,
        he_index=he_index,
        he_attr=he_attr,
        he_mark=he_mark,
        he_count=he_count,
        he_weight=he_weight,
        he_attention=he_attention,
        he_receiver=he_receiver,
        he_layer=he_layer,
        he_head=he_head,
        propagation_mode=cfg.propagation_mode,
        incidence_weight_mode=cfg.incidence_weight_mode,
        construction_config=cfg,
        edge_attr_names=(
            EDGE_ATTR_NAMES if cfg.edge_attr_mode == "faithful" else EXTENDED_EDGE_ATTR_NAMES
        ),
        attention_layer_ids=layer_ids,
        attention_head_ids=head_ids,
        num_model_layers=total_layers,
        num_model_heads=total_heads,
        token_y=canonical_labels[0],
        token_label_mask=canonical_labels[1],
        response_y=canonical_labels[2],
        step_ranges=canonical_labels[3],
        gold_step=canonical_labels[4],
        step_loss_mask=canonical_labels[5],
        trace_id=str(trace_id),
        group_id=str(group_id),
        split=None if split is None else str(split),
    )
    validate_attention_hypergraph(graph, cfg)
    return graph


def build_attention_hypergraph_from_trace(
    trace: Mapping[str, Any],
    *,
    config: Optional[Union[AttentionHypergraphConfig, Mapping[str, Any]]] = None,
) -> AttentionHypergraph:
    """Build from a canonical trace mapping without importing PyTorch.

    Required keys are ``attention``, ``token_ids``, and ``response_idx``.
    Optional keys mirror :func:`build_attention_hypergraph`.  Legacy exact
    RAGTruth labels may use ``hallucination_labels`` or ``y_token``; aliases are
    rejected if more than one is present, preventing ambiguous supervision.
    """

    required = ("attention", "token_ids", "response_idx")
    missing = [key for key in required if key not in trace]
    if missing:
        raise KeyError(f"trace is missing required keys: {missing}")
    token_label_keys = [
        key for key in ("token_y", "hallucination_labels", "y_token") if key in trace
    ]
    if len(token_label_keys) > 1:
        raise ValueError(f"trace contains ambiguous token-label aliases: {token_label_keys}")
    activation_keys = [key for key in ("activation", "activations") if key in trace]
    if len(activation_keys) > 1:
        raise ValueError(f"trace contains ambiguous activation aliases: {activation_keys}")

    return build_attention_hypergraph(
        trace["attention"],
        trace["token_ids"],
        trace["response_idx"],
        attention_layer_ids=trace.get("attention_layer_ids", trace.get("attention_layers")),
        attention_head_ids=trace.get("attention_head_ids", trace.get("attention_heads")),
        num_model_layers=trace.get("num_model_layers"),
        num_model_heads=trace.get("num_model_heads"),
        activation=trace[activation_keys[0]] if activation_keys else None,
        token_y=trace[token_label_keys[0]] if token_label_keys else None,
        token_label_mask=trace.get("token_label_mask"),
        response_y=trace.get("response_y"),
        step_ranges=trace.get("step_ranges"),
        gold_step=trace.get("gold_step"),
        step_loss_mask=trace.get("step_loss_mask"),
        trace_id=trace.get("trace_id", trace.get("sample_id", trace.get("id", ""))),
        group_id=trace.get("group_id", trace.get("problem_id", trace.get("question_id", ""))),
        split=trace.get("split"),
        config=config,
    )
