from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


def _array(value: np.ndarray, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite numeric array")
    return array


@dataclass(frozen=True)
class TransportInputs:
    attention: np.ndarray
    content_effect: np.ndarray
    source_writes: np.ndarray
    output_directions: np.ndarray
    residual_updates: np.ndarray
    prompt_end: int
    receiver_positions: np.ndarray
    causal_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        attention = _array(self.attention, name="attention", ndim=3)
        content_effect = _array(self.content_effect, name="content_effect", ndim=3)
        source_writes = _array(self.source_writes, name="source_writes", ndim=3)
        output_directions = _array(
            self.output_directions, name="output_directions", ndim=2
        )
        residual_updates = _array(
            self.residual_updates, name="residual_updates", ndim=2
        )
        receivers = _array(
            self.receiver_positions, name="receiver_positions", ndim=1
        ).astype(np.int64, copy=False)

        heads, queries, sources = attention.shape
        if content_effect.shape != attention.shape:
            raise ValueError("content_effect must have shape [heads, queries, sources]")
        if source_writes.shape[:2] != (queries, sources):
            raise ValueError("source_writes must have shape [queries, sources, rank]")
        rank = source_writes.shape[2]
        if output_directions.shape != (queries, rank):
            raise ValueError("output_directions must have shape [queries, rank]")
        if residual_updates.shape != (queries, rank):
            raise ValueError("residual_updates must have shape [queries, rank]")
        if receivers.shape != (queries,):
            raise ValueError("receiver_positions must contain one position per query")
        if not 0 < int(self.prompt_end) <= sources:
            raise ValueError("prompt_end must lie inside the source axis")
        if np.any(attention < -self.causal_tolerance):
            raise ValueError("attention cannot contain negative mass")
        if np.any((receivers < 0) | (receivers >= sources)):
            raise ValueError("receiver_positions must lie inside the source axis")
        if np.any(np.diff(receivers) <= 0):
            raise ValueError("receiver_positions must be strictly increasing")

        source_positions = np.arange(sources)[None, None, :]
        future = source_positions > receivers[None, :, None]
        if np.any(np.where(future, attention, 0.0) > self.causal_tolerance):
            raise ValueError("attention contains mass on a future source")

        object.__setattr__(self, "attention", attention)
        object.__setattr__(self, "content_effect", content_effect)
        object.__setattr__(self, "source_writes", source_writes)
        object.__setattr__(self, "output_directions", output_directions)
        object.__setattr__(self, "residual_updates", residual_updates)
        object.__setattr__(self, "receiver_positions", receivers)


@dataclass(frozen=True)
class ContributionMap:
    per_head: np.ndarray
    signed: np.ndarray
    prompt_mask: np.ndarray
    receiver_positions: np.ndarray

    def __post_init__(self) -> None:
        per_head = _array(self.per_head, name="per_head", ndim=3)
        signed = _array(self.signed, name="signed", ndim=2)
        prompt_mask = np.asarray(self.prompt_mask, dtype=bool)
        receivers = np.asarray(self.receiver_positions, dtype=np.int64)
        if signed.shape != per_head.shape[1:]:
            raise ValueError("signed must have shape [queries, sources]")
        if prompt_mask.shape != (per_head.shape[2],):
            raise ValueError("prompt_mask must have shape [sources]")
        if receivers.shape != (per_head.shape[1],):
            raise ValueError("receiver_positions must have shape [queries]")
        object.__setattr__(self, "per_head", per_head)
        object.__setattr__(self, "signed", signed)
        object.__setattr__(self, "prompt_mask", prompt_mask)
        object.__setattr__(self, "receiver_positions", receivers)

    @property
    def prompt_fraction(self) -> np.ndarray:
        magnitude = np.abs(self.per_head).sum(axis=0)
        total = magnitude.sum(axis=1)
        prompt = magnitude[:, self.prompt_mask].sum(axis=1)
        return np.divide(prompt, total, out=np.zeros_like(prompt), where=total > 0)


@dataclass(frozen=True)
class ConstraintGeometry:
    prompt_support: np.ndarray
    response_support: np.ndarray
    effective_update: np.ndarray
    transverse_fraction: np.ndarray
    transverse_escape: np.ndarray
    tangent_rank: np.ndarray

    feature_names: ClassVar[tuple[str, ...]] = (
        "prompt_support",
        "response_support",
        "effective_update",
        "transverse_fraction",
        "transverse_escape",
        "tangent_rank",
    )

    def as_features(self) -> np.ndarray:
        columns = [np.asarray(getattr(self, name)) for name in self.feature_names]
        if not columns or any(column.ndim != 1 for column in columns):
            raise ValueError("geometry features must be one-dimensional")
        features = np.column_stack(columns)
        if not np.isfinite(features).all():
            raise ValueError("geometry features must be finite")
        return features


@dataclass(frozen=True)
class InterventionEffect:
    query_index: int
    sources: tuple[int, ...]
    singleton_effects: np.ndarray
    joint_effect: float

    def __post_init__(self) -> None:
        effects = _array(
            self.singleton_effects, name="singleton_effects", ndim=1
        ).astype(np.float64, copy=False)
        sources = tuple(int(source) for source in self.sources)
        if not sources or len(sources) != len(set(sources)):
            raise ValueError("sources must be a non-empty unique tuple")
        if effects.shape != (len(sources),):
            raise ValueError("singleton_effects must align with sources")
        if self.query_index < 0 or not np.isfinite(self.joint_effect):
            raise ValueError("query_index and joint_effect must be valid")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "singleton_effects", effects)

    @property
    def synergy(self) -> float:
        return float(self.joint_effect - self.singleton_effects.sum())


@dataclass(frozen=True)
class CausalHypergraph:
    node_features: np.ndarray
    incidence: np.ndarray
    receivers: np.ndarray
    edge_features: np.ndarray
    edge_kind: np.ndarray
    response_nodes: np.ndarray

    edge_feature_names: ClassVar[tuple[str, ...]] = (
        "signed_effect",
        "absolute_effect",
        "synergy",
        "prompt_fraction",
    )

    def __post_init__(self) -> None:
        nodes = _array(self.node_features, name="node_features", ndim=2)
        incidence = np.asarray(self.incidence, dtype=np.int64)
        receivers = np.asarray(self.receivers, dtype=np.int64)
        edge_features = _array(self.edge_features, name="edge_features", ndim=2)
        edge_kind = np.asarray(self.edge_kind, dtype=str)
        response_nodes = np.asarray(self.response_nodes, dtype=np.int64)
        edges = edge_features.shape[0]
        if incidence.ndim != 2 or incidence.shape[0] != 2:
            raise ValueError("incidence must have shape [2, memberships]")
        if receivers.shape != (edges,) or edge_kind.shape != (edges,):
            raise ValueError("receivers and edge_kind must align with edges")
        if edge_features.shape[1] != len(self.edge_feature_names):
            raise ValueError("edge_features have an unexpected width")
        if incidence.size:
            if incidence[0].min() < 0 or incidence[0].max() >= len(nodes):
                raise ValueError("incidence contains an invalid node")
            if incidence[1].min() < 0 or incidence[1].max() >= edges:
                raise ValueError("incidence contains an invalid edge")
        if np.any((receivers < 0) | (receivers >= len(nodes))):
            raise ValueError("receivers contain an invalid node")
        if np.any((response_nodes < 0) | (response_nodes >= len(nodes))):
            raise ValueError("response_nodes contain an invalid node")
        object.__setattr__(self, "node_features", nodes)
        object.__setattr__(self, "incidence", incidence)
        object.__setattr__(self, "receivers", receivers)
        object.__setattr__(self, "edge_features", edge_features)
        object.__setattr__(self, "edge_kind", edge_kind)
        object.__setattr__(self, "response_nodes", response_nodes)

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_features.shape[0])


@dataclass(frozen=True)
class FirstErrorLabels:
    num_steps: int
    first_error: int = -1

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.first_error < -1 or self.first_error >= self.num_steps:
            raise ValueError("first_error must be -1 or a valid step index")
