"""Functional divergence analysis for matched reasoning events."""

from .core import (
    MatchedDataset,
    categorical_pullback_fisher_energy,
    crossfit_transport_fisher,
    load_matched_geometry,
    paired_auc_difference,
    paired_summary,
)
from .layer_time import (
    LayerTimeDataset,
    crossfit_layer_time_scores,
    load_matched_layer_time_geometry,
    operator_spectral_metrics,
)
from .raw_residual import inspect_raw_residual_source, load_matched_raw_residual

__all__ = [
    "MatchedDataset",
    "categorical_pullback_fisher_energy",
    "crossfit_transport_fisher",
    "load_matched_geometry",
    "paired_auc_difference",
    "paired_summary",
    "LayerTimeDataset",
    "crossfit_layer_time_scores",
    "load_matched_layer_time_geometry",
    "operator_spectral_metrics",
    "inspect_raw_residual_source",
    "load_matched_raw_residual",
]
