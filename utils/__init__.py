"""Utility primitives for the (step × layer) low-rank spectral analysis."""

from .spectral import (
    token_cloud_singular_values,
    effective_rank,
    spectral_energy,
    top_concentration,
    step_layer_spectral_summary,
    lowrank_decompose,
    chain_lowrankness,
    step_residual_norms,
    layer_residual_norms,
    layer_profile_corr_with_prefix,
)
from .step_boundaries import (
    find_step_token_ranges,
    split_response_into_steps,
)

__all__ = [
    "token_cloud_singular_values",
    "effective_rank",
    "spectral_energy",
    "top_concentration",
    "step_layer_spectral_summary",
    "lowrank_decompose",
    "chain_lowrankness",
    "step_residual_norms",
    "layer_residual_norms",
    "layer_profile_corr_with_prefix",
    "find_step_token_ranges",
    "split_response_into_steps",
]
