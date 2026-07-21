"""Functional divergence analysis for matched reasoning events."""

from .core import (
    MatchedDataset,
    categorical_pullback_fisher_energy,
    crossfit_transport_fisher,
    load_matched_geometry,
)
from .statistics import paired_auc_difference, paired_summary
from .config import RunConfig, SourceConfig
from .domain import CohortSummary, DatasetMetadata, DatasetResult, ExperimentResult, SourceProvenance
from .layer_time import (
    LayerTimeDataset,
    crossfit_layer_time_scores,
    load_matched_layer_time_geometry,
    operator_spectral_metrics,
)
from .raw_residual import inspect_raw_residual_source, load_matched_raw_residual
from .runner import ExperimentRunner

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
    "RunConfig",
    "SourceConfig",
    "SourceProvenance",
    "CohortSummary",
    "DatasetMetadata",
    "DatasetResult",
    "ExperimentResult",
    "ExperimentRunner",
]
