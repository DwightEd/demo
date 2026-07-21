from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SourceProvenance:
    manifest_path: str
    source_format: str
    snapshot_kind: str
    representation_scope: str
    axis_kind: str
    problem_group_field: str
    generator_field: str
    generator_filter: str | None
    response_generators: tuple[str, ...]


@dataclass(frozen=True)
class CohortSummary:
    manifest_records: int
    selected_records: int
    error_records: int
    correct_records: int
    candidate_pairs: int
    retained_pairs: int
    dropped_boundary_pairs: int
    components: int


@dataclass(frozen=True)
class DatasetMetadata:
    provenance: SourceProvenance
    cohort: CohortSummary
    depth_semantics: str
    component_grouping: str

    def to_dict(self) -> dict[str, Any]:
        """Flatten the stable output schema only at the reporting boundary."""
        source = self.provenance
        cohort = self.cohort
        return {
            "source_path": source.manifest_path,
            "axis_kind": source.axis_kind,
            "source_format": source.source_format,
            "snapshot_kind": source.snapshot_kind,
            "representation_scope": source.representation_scope,
            "depth_semantics": self.depth_semantics,
            "n_manifest_records": cohort.manifest_records,
            "n_source_records": cohort.selected_records,
            "n_error_records": cohort.error_records,
            "n_correct_records": cohort.correct_records,
            "n_candidate_pairs": cohort.candidate_pairs,
            "n_retained_pairs": cohort.retained_pairs,
            "n_dropped_boundary_pairs": cohort.dropped_boundary_pairs,
            "n_components": cohort.components,
            "component_grouping": self.component_grouping,
            "problem_group_field": source.problem_group_field,
            "response_generator_filter": source.generator_filter,
            "generator_field": source.generator_field,
            "response_generators": list(source.response_generators),
        }


@dataclass(frozen=True)
class LayerTimeDataset:
    """Matched residual windows with explicit sample, time, layer, and hidden axes."""

    states: np.ndarray
    labels: np.ndarray
    pair_ids: np.ndarray
    component_ids: np.ndarray
    row_ids: np.ndarray
    time_offsets: np.ndarray
    layer_ids: np.ndarray
    feature_names: tuple[str, ...]
    metadata: DatasetMetadata


@dataclass(frozen=True)
class DatasetResult:
    metadata: DatasetMetadata
    pairs: int
    time_offsets: tuple[int, ...]
    layer_ids: tuple[int, ...]
    hidden_dim: int
    diagnostics: dict[str, Any]
    metrics: dict[str, Any]
    comparisons: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metadata.to_dict(),
            "n_pairs": self.pairs,
            "time_offsets": list(self.time_offsets),
            "layer_ids": list(self.layer_ids),
            "hidden_dim": self.hidden_dim,
            "diagnostics": self.diagnostics,
            "metrics": self.metrics,
            "comparisons": self.comparisons,
        }


@dataclass(frozen=True)
class ExperimentResult:
    dataset: DatasetResult
    seed: int
    bootstrap: int
    rank: int
    ridge_alpha: float
    schema_version: str = "raw_residual_layer_time_operator_v1"
    method: str = "component-grouped cross-fitted projected operator field on raw residual-stream shards"
    evidence_boundary: str = (
        "empirical local operators on stored residual states; not autograd model Jacobians"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "evidence_boundary": self.evidence_boundary,
            "seed": self.seed,
            "n_boot": self.bootstrap,
            "requested_rank": self.rank,
            "ridge_alpha": self.ridge_alpha,
            "dataset": self.dataset.to_dict(),
        }
