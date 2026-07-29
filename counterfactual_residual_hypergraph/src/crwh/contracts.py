from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import ClassVar

import numpy as np


def _validated_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _validated_int(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validated_finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _require_nonempty_strings(
    values: tuple[object, ...],
    *,
    name: str,
) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must be non-empty strings")


def _finite_array(
    value: object,
    *,
    name: str,
    ndim: int,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    try:
        numeric = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} must contain only finite values")
    target = np.dtype(dtype)
    if np.issubdtype(target, np.integer):
        if not np.equal(numeric, np.round(numeric)).all():
            raise ValueError(f"{name} must contain integer values")
        return numeric.astype(target, copy=False)
    return np.asarray(value, dtype=target)


@dataclass(frozen=True)
class TraceProvenance:
    """Identity of the extraction coordinate system shared across paired views."""

    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    prompt_format_id: str
    layer_id: int
    extractor_id: str
    projection_id: str
    output_anchor_id: str
    teacher_forcing_offset: int

    def __post_init__(self) -> None:
        strings = (
            self.model_id,
            self.model_revision,
            self.tokenizer_id,
            self.tokenizer_revision,
            self.prompt_format_id,
            self.extractor_id,
            self.projection_id,
            self.output_anchor_id,
        )
        _require_nonempty_strings(strings, name="provenance identifiers")
        layer_id = _validated_int(self.layer_id, name="layer_id", minimum=-1)
        teacher_forcing_offset = _validated_int(
            self.teacher_forcing_offset,
            name="teacher_forcing_offset",
        )
        object.__setattr__(self, "layer_id", layer_id)
        object.__setattr__(
            self,
            "teacher_forcing_offset",
            teacher_forcing_offset,
        )


@dataclass(frozen=True)
class ResidualWriteTrace:
    """One teacher-forced context view with supplied source-resolved OV writes."""

    sample_id: str
    view_name: str
    perturbation_kind: str
    perturbation_id: str
    perturbation_seed: int | None
    perturbation_validated: bool
    provenance: TraceProvenance
    node_features: np.ndarray
    attention: np.ndarray
    source_writes: np.ndarray
    expected_updates: np.ndarray
    output_directions: np.ndarray
    receiver_positions: np.ndarray
    response_token_ids: np.ndarray
    source_roles: np.ndarray
    token_labels: np.ndarray | None = None
    causal_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        _require_nonempty_strings(
            (
                self.sample_id,
                self.view_name,
                self.perturbation_kind,
                self.perturbation_id,
            ),
            name="sample/view/perturbation identifiers",
        )
        perturbation_validated = _validated_bool(
            self.perturbation_validated,
            name="perturbation_validated",
        )
        perturbation_seed = None
        if self.perturbation_seed is not None:
            perturbation_seed = _validated_int(
                self.perturbation_seed,
                name="perturbation_seed",
            )
        causal_tolerance = _validated_finite_real(
            self.causal_tolerance,
            name="causal_tolerance",
            strictly_positive=True,
        )
        if not isinstance(self.provenance, TraceProvenance):
            raise ValueError("provenance must be a TraceProvenance")
        nodes = _finite_array(self.node_features, name="node_features", ndim=2)
        attention = _finite_array(self.attention, name="attention", ndim=3)
        writes = _finite_array(self.source_writes, name="source_writes", ndim=4)
        updates = _finite_array(
            self.expected_updates, name="expected_updates", ndim=3
        )
        directions = _finite_array(
            self.output_directions, name="output_directions", ndim=2
        )
        receivers = _finite_array(
            self.receiver_positions,
            name="receiver_positions",
            ndim=1,
            dtype=np.int64,
        )
        response_ids = _finite_array(
            self.response_token_ids,
            name="response_token_ids",
            ndim=1,
            dtype=np.int64,
        )
        roles = _finite_array(
            self.source_roles, name="source_roles", ndim=1, dtype=np.int64
        )

        heads, queries, sources = attention.shape
        if nodes.shape[0] != sources:
            raise ValueError("node_features must align with the attention source axis")
        if writes.shape[:3] != attention.shape:
            raise ValueError(
                "source_writes must have shape [heads, queries, sources, rank]"
            )
        rank = writes.shape[-1]
        if updates.shape != (heads, queries, rank):
            raise ValueError(
                "expected_updates must have shape [heads, queries, rank]"
            )
        if directions.shape != (queries, rank):
            raise ValueError(
                "output_directions must have shape [queries, rank]"
            )
        if receivers.shape != (queries,) or response_ids.shape != (queries,):
            raise ValueError(
                "receiver_positions and response_token_ids must align with queries"
            )
        if roles.shape != (sources,):
            raise ValueError("source_roles must contain one role per node")
        if np.any(attention < -causal_tolerance):
            raise ValueError("attention cannot contain negative mass")
        if np.any((receivers < 0) | (receivers >= sources)):
            raise ValueError("receiver_positions contain an invalid node")
        if np.any(roles[receivers] != 2):
            raise ValueError("receiver positions must use response role 2")
        if np.any(np.diff(receivers) <= 0):
            raise ValueError("receiver_positions must be strictly increasing")

        source_positions = np.arange(sources)[None, None, :]
        future = source_positions > receivers[None, :, None]
        if np.any(np.where(future, attention, 0.0) > causal_tolerance):
            raise ValueError("attention contains mass on a future source")
        future_write_norm = np.linalg.norm(
            np.where(future[..., None], writes, 0.0), axis=-1
        )
        if np.any(future_write_norm > causal_tolerance):
            raise ValueError("source_writes contain a future residual write")

        labels = None
        if self.token_labels is not None:
            labels = _finite_array(
                self.token_labels, name="token_labels", ndim=1, dtype=np.int64
            )
            if labels.shape != (queries,):
                raise ValueError("token_labels must align with response queries")
            if np.any(~np.isin(labels, (-1, 0, 1))):
                raise ValueError("token_labels must use -1/0/1")

        object.__setattr__(self, "node_features", nodes)
        object.__setattr__(self, "attention", attention)
        object.__setattr__(self, "source_writes", writes)
        object.__setattr__(self, "expected_updates", updates)
        object.__setattr__(self, "output_directions", directions)
        object.__setattr__(self, "receiver_positions", receivers)
        object.__setattr__(self, "response_token_ids", response_ids)
        object.__setattr__(self, "source_roles", roles)
        object.__setattr__(self, "token_labels", labels)
        object.__setattr__(
            self,
            "perturbation_validated",
            perturbation_validated,
        )
        object.__setattr__(self, "perturbation_seed", perturbation_seed)
        object.__setattr__(self, "causal_tolerance", causal_tolerance)


@dataclass(frozen=True)
class ResidualWriteHypergraph:
    """Directed hypergraph whose edges are identified head/query write groups."""

    edge_feature_names: ClassVar[tuple[str, ...]] = (
        "mean_attention",
        "max_attention",
        "head_fraction",
        "source_fraction",
        "evidence_fraction",
        "write_mass",
        "resultant_norm",
        "cancellation",
        "effective_sources",
        "dominance",
        "signed_support",
        "self_attention",
        "self_write_mass",
        "self_signed_support",
    )

    sample_id: str
    view_name: str
    perturbation_kind: str
    perturbation_id: str
    perturbation_seed: int | None
    perturbation_validated: bool
    provenance: TraceProvenance
    num_heads: int
    node_features: np.ndarray
    incidence: np.ndarray
    receivers: np.ndarray
    edge_features: np.ndarray
    edge_heads: np.ndarray
    edge_queries: np.ndarray
    edge_kind: np.ndarray
    response_nodes: np.ndarray
    response_token_ids: np.ndarray
    output_directions: np.ndarray
    source_roles: np.ndarray
    token_labels: np.ndarray | None = None

    def __post_init__(self) -> None:
        _require_nonempty_strings(
            (
                self.sample_id,
                self.view_name,
                self.perturbation_kind,
                self.perturbation_id,
            ),
            name="sample/view/perturbation identifiers",
        )
        perturbation_validated = _validated_bool(
            self.perturbation_validated,
            name="perturbation_validated",
        )
        perturbation_seed = None
        if self.perturbation_seed is not None:
            perturbation_seed = _validated_int(
                self.perturbation_seed,
                name="perturbation_seed",
            )
        num_heads = _validated_int(
            self.num_heads,
            name="num_heads",
            minimum=1,
        )
        nodes = _finite_array(self.node_features, name="node_features", ndim=2)
        incidence = _finite_array(
            self.incidence, name="incidence", ndim=2, dtype=np.int64
        )
        receivers = _finite_array(
            self.receivers, name="receivers", ndim=1, dtype=np.int64
        )
        features = _finite_array(self.edge_features, name="edge_features", ndim=2)
        heads = _finite_array(
            self.edge_heads, name="edge_heads", ndim=1, dtype=np.int64
        )
        queries = _finite_array(
            self.edge_queries, name="edge_queries", ndim=1, dtype=np.int64
        )
        raw_kinds = np.asarray(self.edge_kind)
        if raw_kinds.ndim != 1 or not all(
            isinstance(value, (str, np.str_)) and str(value).strip()
            for value in raw_kinds.tolist()
        ):
            raise ValueError("edge_kind must contain non-empty strings")
        kinds = raw_kinds.astype(str, copy=False)
        response_nodes = _finite_array(
            self.response_nodes,
            name="response_nodes",
            ndim=1,
            dtype=np.int64,
        )
        response_ids = _finite_array(
            self.response_token_ids,
            name="response_token_ids",
            ndim=1,
            dtype=np.int64,
        )
        output_directions = _finite_array(
            self.output_directions,
            name="output_directions",
            ndim=2,
        )
        roles = _finite_array(
            self.source_roles,
            name="source_roles",
            ndim=1,
            dtype=np.int64,
        )
        if not isinstance(self.provenance, TraceProvenance):
            raise ValueError("provenance must be a TraceProvenance")

        if incidence.ndim != 2 or incidence.shape[0] != 2:
            raise ValueError("incidence must have shape [2, memberships]")
        edge_count = features.shape[0]
        if features.shape[1] != len(self.edge_feature_names):
            raise ValueError("edge_features have an unexpected width")
        aligned = (receivers, heads, queries, kinds)
        if any(value.shape != (edge_count,) for value in aligned):
            raise ValueError("edge metadata must align with edge_features")
        if response_nodes.shape != response_ids.shape:
            raise ValueError("response nodes and token ids must align")
        if output_directions.shape[0] != len(response_nodes):
            raise ValueError("output_directions must align with response nodes")
        if roles.shape != (nodes.shape[0],):
            raise ValueError("source_roles must align with nodes")
        if incidence.size:
            if incidence[0].min() < 0 or incidence[0].max() >= nodes.shape[0]:
                raise ValueError("incidence contains an invalid node")
            if incidence[1].min() < 0 or incidence[1].max() >= edge_count:
                raise ValueError("incidence contains an invalid edge")
        if np.any((receivers < 0) | (receivers >= nodes.shape[0])):
            raise ValueError("receivers contain an invalid node")
        if np.any((heads < 0) | (heads >= num_heads)):
            raise ValueError("edge_heads contain an invalid head")
        if np.any((response_nodes < 0) | (response_nodes >= nodes.shape[0])):
            raise ValueError("response_nodes contain an invalid node")
        if np.any(roles[response_nodes] != 2):
            raise ValueError("response_nodes must use response role 2")
        if len(np.unique(response_nodes)) != len(response_nodes):
            raise ValueError("response_nodes must be unique")
        if np.any((queries < 0) | (queries >= len(response_nodes))):
            raise ValueError("edge_queries contain an invalid response query")
        if edge_count:
            edge_ids = incidence[1]
            membership_count = np.bincount(edge_ids, minlength=edge_count)
            if np.any(membership_count == 0):
                raise ValueError("every edge must contain at least one membership")
            receiver_member = incidence[0] == receivers[edge_ids]
            receiver_hits = np.bincount(
                edge_ids,
                weights=receiver_member.astype(np.int64),
                minlength=edge_count,
            )
            source_hits = membership_count - receiver_hits
            if np.any(receiver_hits == 0):
                raise ValueError("every edge must contain its declared receiver")
            if np.any(source_hits == 0):
                raise ValueError("every edge must contain a non-receiver source")
        labels = None
        if self.token_labels is not None:
            labels = _finite_array(
                self.token_labels,
                name="token_labels",
                ndim=1,
                dtype=np.int64,
            )
            if labels.shape != response_nodes.shape:
                raise ValueError("token_labels must align with response nodes")
            if np.any(~np.isin(labels, (-1, 0, 1))):
                raise ValueError("token_labels must use -1/0/1")

        object.__setattr__(self, "node_features", nodes)
        object.__setattr__(self, "incidence", incidence)
        object.__setattr__(self, "receivers", receivers)
        object.__setattr__(self, "edge_features", features)
        object.__setattr__(self, "edge_heads", heads)
        object.__setattr__(self, "edge_queries", queries)
        object.__setattr__(self, "edge_kind", kinds)
        object.__setattr__(self, "response_nodes", response_nodes)
        object.__setattr__(self, "response_token_ids", response_ids)
        object.__setattr__(self, "output_directions", output_directions)
        object.__setattr__(self, "source_roles", roles)
        object.__setattr__(self, "token_labels", labels)
        object.__setattr__(
            self,
            "perturbation_validated",
            perturbation_validated,
        )
        object.__setattr__(self, "perturbation_seed", perturbation_seed)
        object.__setattr__(self, "num_heads", num_heads)

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_features.shape[0])


@dataclass(frozen=True)
class CounterfactualExample:
    """Aligned fixed-response graphs under factual and perturbed contexts."""

    sample_id: str
    factual: ResidualWriteHypergraph
    counterfactual: ResidualWriteHypergraph
    paraphrase: ResidualWriteHypergraph | None = None
    normal_reference: bool = False
    contrastive_eligible: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        normal_reference = _validated_bool(
            self.normal_reference,
            name="normal_reference",
        )
        views = (self.factual, self.counterfactual)
        if self.paraphrase is not None:
            views += (self.paraphrase,)
        if any(view.sample_id != self.sample_id for view in views):
            raise ValueError("all views must share the example sample_id")
        if any(not view.perturbation_validated for view in views):
            raise ValueError("all paired views must have validated perturbations")
        if self.factual.perturbation_kind != "factual":
            raise ValueError("the factual view must use perturbation_kind='factual'")
        counterfactual_kinds = {
            "evidence_removal",
            "evidence_replacement",
            "contradiction",
            "random_context",
        }
        if self.counterfactual.perturbation_kind not in counterfactual_kinds:
            raise ValueError("counterfactual view has an unsupported perturbation_kind")
        if self.paraphrase is not None and self.paraphrase.perturbation_kind != "paraphrase":
            raise ValueError(
                "the paraphrase view must use perturbation_kind='paraphrase'"
            )
        if any(view.provenance != self.factual.provenance for view in views[1:]):
            raise ValueError("all views must share identical extraction provenance")
        expected = self.factual.response_token_ids
        if any(not np.array_equal(view.response_token_ids, expected) for view in views[1:]):
            raise ValueError("all views must have identical response token ids")
        node_width = self.factual.node_features.shape[1]
        if any(view.node_features.shape[1] != node_width for view in views[1:]):
            raise ValueError("all views must have the same node feature width")
        if any(
            view.output_directions.shape
            != self.factual.output_directions.shape
            or not np.allclose(
                view.output_directions,
                self.factual.output_directions,
                rtol=0.0,
                atol=1e-10,
            )
            for view in views[1:]
        ):
            raise ValueError("all views must share fixed output directions")
        for left_index, left in enumerate(views):
            if left.token_labels is None:
                continue
            for right in views[left_index + 1 :]:
                if right.token_labels is None:
                    continue
                visible = (left.token_labels >= 0) & (right.token_labels >= 0)
                if np.any(left.token_labels[visible] != right.token_labels[visible]):
                    raise ValueError("paired views contain conflicting token labels")
        if normal_reference:
            if any(
                view.token_labels is not None
                and np.any(view.token_labels == 1)
                for view in views
            ):
                raise ValueError(
                    "normal_reference cannot contain hallucination labels"
                )
        eligible = self.contrastive_eligible
        if eligible is None:
            eligible = np.zeros(len(expected), dtype=bool)
        else:
            eligible = np.asarray(eligible)
            if eligible.shape != expected.shape:
                raise ValueError(
                    "contrastive_eligible must align with response tokens"
                )
            if not np.issubdtype(eligible.dtype, np.bool_):
                raise ValueError(
                    "contrastive_eligible must contain boolean values"
                )
            eligible = eligible.astype(bool, copy=False)
        known_hallucinated = np.zeros(len(expected), dtype=bool)
        for view in views:
            if view.token_labels is not None:
                known_hallucinated |= view.token_labels == 1
        if np.any(eligible & known_hallucinated):
            raise ValueError(
                "contrastive_eligible cannot include a known hallucinated token"
            )
        object.__setattr__(self, "normal_reference", normal_reference)
        object.__setattr__(self, "contrastive_eligible", eligible)
