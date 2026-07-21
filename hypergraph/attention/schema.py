"""Framework-neutral schema for attention-row token hypergraphs.

The schema intentionally depends only on NumPy.  PyTorch/PyG conversion and
message-passing policy belong to the data/model layers, not construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


EDGE_ATTR_NAMES: Tuple[str, ...] = (
    "attention_mean",
    "attention_max",
    "flattened_head_normalized",
)
"""Faithful three-column edge attributes from the local original."""


EXTENDED_EDGE_ATTR_NAMES: Tuple[str, ...] = (
    "attention_mean",
    "attention_max",
    "layer_normalized",
    "head_normalized",
    "member_fraction",
    "log1p_member_count_normalized",
)
"""Length-normalized experimental edge attributes (explicit opt-in)."""


EDGE_MARK_NAMES: Tuple[str, ...] = ("prompt_cross", "response_only")
"""One-hot column names for :attr:`AttentionHypergraph.he_mark`."""

TARGET_ALIGNMENT = "same_index_post_emission"
"""Original row-i/label-i alignment: detection after token i has been observed."""


_SOURCE_SCOPES = frozenset({"all_past", "prompt_only", "response_only"})
_WEIGHT_MODES = frozenset({"uniform", "attention", "normalized_attention"})
_PROPAGATION_MODES = frozenset({"symmetric", "receiver"})
_EDGE_ATTR_MODES = frozenset({"faithful", "extended"})
_SOURCE_SELECTION_MODES = frozenset(
    {"threshold", "threshold_fallback_topk", "top_k_only", "cumulative_mass"}
)
_NODE_FEATURE_MODES = frozenset(
    {"attention_diagonal", "activation_only", "diagonal_plus_activation"}
)


@dataclass(frozen=True)
class AttentionHypergraphConfig:
    """Configuration for faithful attention-row hypergraph construction.

    Library defaults preserve the earlier pure-threshold compatibility path.
    The original-aligned response wrapper explicitly supplies the local
    ``processed_hypergraph.py`` settings: ``threshold=0.05``,
    ``source_selection='threshold_fallback_topk'``, ``top_k=16``, and
    ``min_sources=2``. Layer/head selection remains explicit because the local
    original contains an unconditional first-head ``break``.

    ``propagation_mode`` is construction metadata.  Both modes preserve the
    receiver index; a downstream model decides whether an edge updates all its
    members (``symmetric``) or only its receiver (``receiver``).
    """

    threshold: float = 0.01
    top_k: Optional[int] = None
    source_selection: str = "threshold"
    cumulative_mass: float = 0.8
    min_sources: int = 1
    include_center: bool = True
    source_scope: str = "all_past"
    incidence_weight_mode: str = "uniform"
    propagation_mode: str = "symmetric"
    edge_attr_mode: str = "faithful"
    node_feature_mode: str = "attention_diagonal"
    selected_layers: Optional[Tuple[int, ...]] = None
    selected_heads: Optional[Tuple[int, ...]] = None
    target_alignment: str = TARGET_ALIGNMENT

    @property
    def tau(self) -> float:
        """Alias used by the original implementation and experiment reports."""

        return float(self.threshold)

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold) or not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("threshold must be finite and lie in [0, 1]")
        if self.top_k is not None:
            top_k_value = np.asarray(self.top_k)
            if (
                top_k_value.size != 1
                or not np.issubdtype(top_k_value.dtype, np.integer)
            ):
                raise ValueError("top_k must be None or a positive integer")
            if int(top_k_value.reshape(-1)[0]) < 1:
                raise ValueError("top_k must be None or a positive integer")
        if self.source_selection not in _SOURCE_SELECTION_MODES:
            raise ValueError(
                "source_selection must be one of "
                f"{sorted(_SOURCE_SELECTION_MODES)}, got {self.source_selection!r}"
            )
        if not np.isfinite(self.cumulative_mass) or not 0.0 < float(self.cumulative_mass) <= 1.0:
            raise ValueError("cumulative_mass must be finite and lie in (0, 1]")
        if (
            self.source_selection in {"top_k_only", "threshold_fallback_topk"}
            and self.top_k is None
        ):
            raise ValueError(f"source_selection={self.source_selection!r} requires top_k")
        if self.source_selection == "cumulative_mass" and self.top_k is not None:
            raise ValueError(
                "cumulative_mass selection cannot also set top_k; vary one sparsifier at a time"
            )
        min_sources_value = np.asarray(self.min_sources)
        if min_sources_value.size != 1 or not np.issubdtype(
            min_sources_value.dtype, np.integer
        ):
            raise ValueError("min_sources must be an integer")
        if int(min_sources_value.reshape(-1)[0]) < 1:
            raise ValueError("min_sources must be at least one")
        if self.source_scope not in _SOURCE_SCOPES:
            raise ValueError(
                f"source_scope must be one of {sorted(_SOURCE_SCOPES)}, got {self.source_scope!r}"
            )
        if self.incidence_weight_mode not in _WEIGHT_MODES:
            raise ValueError(
                "incidence_weight_mode must be one of "
                f"{sorted(_WEIGHT_MODES)}, got {self.incidence_weight_mode!r}"
            )
        if self.propagation_mode not in _PROPAGATION_MODES:
            raise ValueError(
                "propagation_mode must be one of "
                f"{sorted(_PROPAGATION_MODES)}, got {self.propagation_mode!r}"
            )
        if self.edge_attr_mode not in _EDGE_ATTR_MODES:
            raise ValueError(
                f"edge_attr_mode must be one of {sorted(_EDGE_ATTR_MODES)}, "
                f"got {self.edge_attr_mode!r}"
            )
        if self.node_feature_mode not in _NODE_FEATURE_MODES:
            raise ValueError(
                "node_feature_mode must be one of "
                f"{sorted(_NODE_FEATURE_MODES)}, got {self.node_feature_mode!r}"
            )
        if self.target_alignment != TARGET_ALIGNMENT:
            raise ValueError(
                "only target_alignment='same_index_post_emission' is implemented; "
                "next-token alignment needs a separately specified graph contract"
            )
        object.__setattr__(self, "top_k", None if self.top_k is None else int(self.top_k))
        object.__setattr__(self, "min_sources", int(self.min_sources))
        object.__setattr__(self, "include_center", bool(self.include_center))
        object.__setattr__(
            self,
            "selected_layers",
            _canonical_indices(self.selected_layers, name="selected_layers"),
        )
        object.__setattr__(
            self,
            "selected_heads",
            _canonical_indices(self.selected_heads, name="selected_heads"),
        )


def _canonical_indices(
    values: Optional[Tuple[int, ...]], *, name: str
) -> Optional[Tuple[int, ...]]:
    if values is None:
        return None
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices")
    indices = tuple(int(value) for value in raw.tolist())
    if not indices:
        raise ValueError(f"{name} cannot be empty; use None to select every index")
    if any(value < 0 for value in indices):
        raise ValueError(f"{name} cannot contain negative indices")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} cannot contain duplicate indices")
    return indices


@dataclass(frozen=True)
class AttentionHypergraph:
    """Arrays for one prompt-and-response attention hypergraph.

    All original sequence tokens are nodes.  ``he_index[0]`` contains member
    node indices and ``he_index[1]`` the corresponding hyperedge indices.
    ``he_attention`` always stores raw attention values per incidence, whereas
    ``he_weight`` stores the configured downstream propagation weight.

    Label fields are optional and retain their source granularity.  In
    particular, ``gold_step`` is never expanded into token labels.  If supplied,
    ``step_ranges`` must use absolute, half-open token offsets in the complete
    prompt+response sequence.
    """

    x: np.ndarray
    token_ids: np.ndarray
    response_idx: int
    he_index: np.ndarray
    he_attr: np.ndarray
    he_mark: np.ndarray
    he_count: np.ndarray
    he_weight: np.ndarray
    he_attention: np.ndarray
    he_receiver: np.ndarray
    he_layer: np.ndarray
    he_head: np.ndarray
    propagation_mode: str
    incidence_weight_mode: str
    construction_config: AttentionHypergraphConfig
    edge_attr_names: Tuple[str, ...]
    attention_layer_ids: np.ndarray
    attention_head_ids: np.ndarray
    num_model_layers: int
    num_model_heads: int
    token_y: Optional[np.ndarray] = None
    token_label_mask: Optional[np.ndarray] = None
    response_y: Optional[float] = None
    step_ranges: Optional[np.ndarray] = None
    gold_step: Optional[int] = None
    step_loss_mask: Optional[np.ndarray] = None
    trace_id: str = ""
    group_id: str = ""
    split: Optional[str] = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_hyperedges(self) -> int:
        return int(self.he_attr.shape[0])

    @property
    def num_incidences(self) -> int:
        return int(self.he_index.shape[1])

    @property
    def response_start(self) -> int:
        """Compatibility alias for :attr:`response_idx`."""

        return int(self.response_idx)

    @property
    def y(self) -> Optional[np.ndarray]:
        """Compatibility alias for exact token labels, if available."""

        return self.token_y

    @property
    def edge_mark_names(self) -> Tuple[str, ...]:
        return EDGE_MARK_NAMES
