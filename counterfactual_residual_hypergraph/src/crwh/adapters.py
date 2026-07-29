from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .contracts import (
    ResidualWriteHypergraph,
    TraceProvenance,
    _finite_array,
    _validated_int,
)


def _numpy(value: object, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _attention_baseline_graph(
    *,
    sample_id: str,
    view_name: str,
    node_features: object,
    incidence: object,
    edge_attributes: object,
    response_nodes: np.ndarray,
    response_token_ids: np.ndarray,
    token_labels: np.ndarray | None,
    num_heads: int,
    provenance: TraceProvenance,
) -> ResidualWriteHypergraph:
    nodes = _finite_array(
        _numpy(node_features),
        name="legacy node_features",
        ndim=2,
    )
    he_index = _finite_array(
        _numpy(incidence),
        name="legacy incidence",
        ndim=2,
        dtype=np.int64,
    )
    attributes = _finite_array(
        _numpy(edge_attributes),
        name="legacy edge attributes",
        ndim=2,
    )
    response_nodes = _finite_array(
        response_nodes,
        name="legacy response_nodes",
        ndim=1,
        dtype=np.int64,
    )
    response_token_ids = _finite_array(
        response_token_ids,
        name="legacy response_token_ids",
        ndim=1,
        dtype=np.int64,
    )
    if token_labels is not None:
        token_labels = _finite_array(
            token_labels,
            name="legacy token_labels",
            ndim=1,
            dtype=np.int64,
        )
    if he_index.ndim != 2 or he_index.shape[0] != 2:
        raise ValueError("legacy incidence must have shape [2, memberships]")
    if attributes.ndim != 2 or attributes.shape[1] < 3:
        raise ValueError("legacy edge attributes must contain mean/max/head")
    num_heads = _validated_int(num_heads, name="num_heads", minimum=1)

    edge_count = attributes.shape[0]
    features = np.zeros(
        (edge_count, len(ResidualWriteHypergraph.edge_feature_names)),
        dtype=np.float64,
    )
    features[:, :3] = attributes[:, :3]
    node_ids, edge_ids = he_index
    if he_index.size and (
        edge_ids.min() < 0
        or edge_ids.max() >= edge_count
        or node_ids.min() < 0
        or node_ids.max() >= nodes.shape[0]
    ):
        raise ValueError("legacy incidence contains an invalid node or edge")
    membership_count = np.bincount(edge_ids, minlength=edge_count)
    if np.any(membership_count == 0):
        raise ValueError("every legacy hyperedge must have at least one member")

    # Both source repositories add the causal query token, which is the
    # maximum member under their causal construction.
    receivers = np.full(edge_count, -1, dtype=np.int64)
    if edge_count:
        np.maximum.at(receivers, edge_ids, node_ids)
    response_lookup = np.full(nodes.shape[0], -1, dtype=np.int64)
    response_lookup[response_nodes] = np.arange(len(response_nodes))
    queries = response_lookup[receivers] if edge_count else receivers.copy()
    if np.any(queries < 0):
        raise ValueError("legacy hyperedge does not contain a response receiver")
    is_source = node_ids != receivers[edge_ids]
    source_edges = edge_ids[is_source]
    source_nodes = node_ids[is_source]
    source_count = np.bincount(source_edges, minlength=edge_count)
    if np.any(source_count == 0):
        raise ValueError("legacy hyperedge must contain a non-receiver source")
    evidence_count = np.bincount(
        source_edges,
        weights=(~np.isin(source_nodes, response_nodes)).astype(np.float64),
        minlength=edge_count,
    )
    features[:, 3] = source_count / max(nodes.shape[0], 1)
    features[:, 4] = evidence_count / source_count

    heads = np.rint(
        np.clip(attributes[:, 2], 0.0, 1.0) * max(num_heads - 1, 0)
    ).astype(np.int64)
    roles = np.zeros(nodes.shape[0], dtype=np.int64)
    roles[response_nodes] = 2
    return ResidualWriteHypergraph(
        sample_id=sample_id,
        view_name=view_name,
        perturbation_kind="legacy_attention",
        perturbation_id=f"{sample_id}/{view_name}",
        perturbation_seed=None,
        perturbation_validated=False,
        provenance=provenance,
        num_heads=num_heads,
        node_features=nodes,
        incidence=he_index,
        receivers=receivers,
        edge_features=features,
        edge_heads=heads,
        edge_queries=queries,
        edge_kind=np.repeat(
            np.asarray(["adapted_attention_ablation"], dtype=str), edge_count
        ),
        response_nodes=response_nodes,
        response_token_ids=response_token_ids,
        output_directions=np.zeros((len(response_nodes), 0), dtype=np.float64),
        source_roles=roles,
        token_labels=token_labels,
    )


def from_attnhyper(
    payload: Mapping[str, object],
    *,
    view_name: str = "attnhyper",
    num_heads: int = 1,
    provenance: TraceProvenance | None = None,
) -> ResidualWriteHypergraph:
    """Adapt the first repository's saved dictionary without upgrading its claims."""

    nodes = _numpy(payload["x"])
    response_start = _validated_int(
        payload["response_idx"],
        name="response_idx",
        minimum=0,
    )
    if response_start >= nodes.shape[0]:
        raise ValueError("response_idx must identify a node in x")
    response_nodes = np.arange(response_start, nodes.shape[0], dtype=np.int64)
    if "token_ids" not in payload:
        raise ValueError(
            "token_ids are required; positions cannot establish cross-view alignment"
        )
    token_ids = _finite_array(
        _numpy(payload["token_ids"]),
        name="token_ids",
        ndim=1,
        dtype=np.int64,
    )
    response_ids = (
        token_ids[response_nodes]
        if token_ids.shape == (nodes.shape[0],)
        else token_ids
    )
    raw_labels = payload.get("y_token")
    labels = None
    if raw_labels is not None:
        raw_labels = _finite_array(
            _numpy(raw_labels),
            name="y_token",
            ndim=1,
            dtype=np.int64,
        )
        labels = (
            raw_labels[response_nodes]
            if raw_labels.shape == (nodes.shape[0],)
            else raw_labels
        )
    provenance = provenance or TraceProvenance(
        model_id="legacy-unknown",
        model_revision="unknown",
        tokenizer_id="legacy-unknown",
        tokenizer_revision="unknown",
        prompt_format_id="unknown",
        layer_id=-1,
        extractor_id="attnhyper_attention_statistics",
        projection_id="attnhyper_self_attention",
        output_anchor_id="none",
        teacher_forcing_offset=0,
    )
    return _attention_baseline_graph(
        sample_id=str(payload.get("source_id", "legacy")),
        view_name=view_name,
        node_features=nodes,
        incidence=payload["he_incidence_index"],
        edge_attributes=payload["he_attr"],
        response_nodes=response_nodes,
        response_token_ids=response_ids,
        token_labels=labels,
        num_heads=num_heads,
        provenance=provenance,
    )


def from_mvht_view(
    payload: Mapping[str, object],
    *,
    sample_id: str,
    view: str,
    num_heads: int = 1,
    response_token_ids: object | None = None,
    provenance: TraceProvenance | None = None,
) -> ResidualWriteHypergraph:
    """Adapt one MVHTR subspace as an explicitly attention-only baseline view."""

    keys = {
        "semantic": ("x_S", "he_index_S", "he_attr_S"),
        "reasoning": ("x_R", "he_index_R", "he_attr_R"),
    }
    if view not in keys:
        raise ValueError("view must be 'semantic' or 'reasoning'")
    node_key, incidence_key, attributes_key = keys[view]
    response_mask = _numpy(payload["response_mask"])
    if response_mask.ndim != 1 or not np.issubdtype(
        response_mask.dtype,
        np.bool_,
    ):
        raise ValueError("response_mask must be a one-dimensional boolean array")
    response_nodes = np.flatnonzero(response_mask).astype(np.int64)
    token_ids = payload.get("token_ids", response_token_ids)
    if token_ids is None:
        raise ValueError(
            "token_ids or explicit response_token_ids are required for alignment"
        )
    token_ids = _finite_array(
        _numpy(token_ids),
        name="token_ids",
        ndim=1,
        dtype=np.int64,
    )
    response_ids = (
        token_ids[response_nodes]
        if token_ids.shape == response_mask.shape
        else token_ids
    )
    provenance = provenance or TraceProvenance(
        model_id="legacy-unknown",
        model_revision="unknown",
        tokenizer_id="legacy-unknown",
        tokenizer_revision="unknown",
        prompt_format_id="unknown",
        layer_id=-1,
        extractor_id="mvht_attention_cosine_statistics",
        projection_id=f"mvht_{view}",
        output_anchor_id="none",
        teacher_forcing_offset=0,
    )
    return _attention_baseline_graph(
        sample_id=sample_id,
        view_name=f"mvht_{view}",
        node_features=payload[node_key],
        incidence=payload[incidence_key],
        edge_attributes=payload[attributes_key],
        response_nodes=response_nodes,
        response_token_ids=response_ids,
        token_labels=None,
        num_heads=num_heads,
        provenance=provenance,
    )
