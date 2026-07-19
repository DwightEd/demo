"""Faithful attention-row hypergraph construction.

This package is deliberately NumPy-only.  Importing it never requires PyTorch;
framework conversion, objectives, and training live in neighbouring modules.
"""

from .construction import (
    build_attention_hyperedges,
    build_attention_hypergraph,
    build_attention_hypergraph_from_trace,
    build_attention_node_features,
    coerce_attention_config,
    validate_attention_hypergraph,
)
from .schema import (
    EDGE_ATTR_NAMES,
    EDGE_MARK_NAMES,
    EXTENDED_EDGE_ATTR_NAMES,
    TARGET_ALIGNMENT,
    AttentionHypergraph,
    AttentionHypergraphConfig,
)

__all__ = [
    "EDGE_ATTR_NAMES",
    "EDGE_MARK_NAMES",
    "EXTENDED_EDGE_ATTR_NAMES",
    "TARGET_ALIGNMENT",
    "AttentionHypergraph",
    "AttentionHypergraphConfig",
    "build_attention_hyperedges",
    "build_attention_hypergraph",
    "build_attention_hypergraph_from_trace",
    "build_attention_node_features",
    "coerce_attention_config",
    "validate_attention_hypergraph",
]
