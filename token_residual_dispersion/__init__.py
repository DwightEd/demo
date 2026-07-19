"""Step-free, token-causal analysis of residual-write directional dispersion."""

from .metrics import (
    DispersionConfig,
    block_writes_from_states,
    component_conflict,
    compute_dispersion_field,
    depth_deltas_from_states,
    residual_arc_length,
)

__all__ = [
    "DispersionConfig",
    "block_writes_from_states",
    "component_conflict",
    "compute_dispersion_field",
    "depth_deltas_from_states",
    "residual_arc_length",
]
