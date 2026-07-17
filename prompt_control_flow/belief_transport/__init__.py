"""Constraint-supported belief transport experiments.

This package deliberately separates exact finite-world semantics from model
extraction and statistical auditing. The finite world is the reference system;
hidden-state geometry is only interpreted after it predicts that reference on
held-out problem groups.
"""

from .belief import (
    fisher_rao_distance,
    mask_to_belief,
    masked_belief_update,
    transition_diagnostics,
)
from .world import WindTunnelConfig, generate_worlds

__all__ = [
    "WindTunnelConfig",
    "fisher_rao_distance",
    "generate_worlds",
    "mask_to_belief",
    "masked_belief_update",
    "transition_diagnostics",
]
