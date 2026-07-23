from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ACQUISITION_MODES = {
    "observer_teacher_forcing_replay": False,
    "generation_matched_online_trace": True,
}


@dataclass(frozen=True)
class TraceSource:
    """One dataset's verified residual manifest and aligned output artifact."""

    dataset: str
    manifest: Path
    acquisition_mode: str
    exact_trace: Path | None = None
    hidden_dir: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", Path(self.manifest).expanduser())
        if self.exact_trace is not None:
            object.__setattr__(self, "exact_trace", Path(self.exact_trace).expanduser())
        if self.hidden_dir is not None:
            object.__setattr__(self, "hidden_dir", Path(self.hidden_dir).expanduser())
        if not self.dataset.strip():
            raise ValueError("dataset name cannot be empty")
        if self.acquisition_mode not in ACQUISITION_MODES:
            raise ValueError(
                f"unsupported acquisition mode {self.acquisition_mode!r}; "
                f"available={sorted(ACQUISITION_MODES)}"
            )


@dataclass(frozen=True)
class OutputEvidence:
    acquisition_mode: str
    acquisition_mode_source: str
    hidden_evidence_kind: str
    output_evidence_kind: str
    output_feature_names: tuple[str, ...]
    full_vocab_logits_stored: bool
    full_vocab_logits_used: bool
    generation_matched_online_states: bool
    manifest_records: int
    selected_records: int


@dataclass(frozen=True)
class ChainSample:
    """Immutable lightweight descriptor; the large hidden shard stays on disk."""

    chain_id: int
    manifest_row: int
    problem_group: str
    dataset: str
    generator: str
    observer_model: str
    state_path: Path
    state_count: int
    response_start: int
    step_ranges: np.ndarray
    layer_ids: np.ndarray
    output_steps: np.ndarray
    output_feature_names: tuple[str, ...]
    first_error_step: int
    problem_hash: str | None = None

    @property
    def n_steps(self) -> int:
        return int(len(self.step_ranges))

    @property
    def cluster_group(self) -> str:
        """Dataset-local cluster key; numeric IDs may be reused across datasets."""
        return f"{self.dataset}::{self.problem_group}"


@dataclass(frozen=True)
class HiddenGeometryDataset:
    samples: tuple[ChainSample, ...]
    labels: np.ndarray
    evidence: OutputEvidence

    def __post_init__(self) -> None:
        if self.labels.shape != (len(self.samples),):
            raise ValueError("labels must have one entry per chain")
        if len(self.samples) == 0:
            raise ValueError("dataset contains no selected chains")
        if not np.isin(self.labels, (0, 1)).all():
            raise ValueError("labels must use 1=error and 0=fully-correct")

    @property
    def groups(self) -> np.ndarray:
        return np.asarray(
            [sample.cluster_group for sample in self.samples],
            dtype=object,
        )

    @property
    def domains(self) -> np.ndarray:
        return np.asarray([sample.dataset for sample in self.samples], dtype=object)
