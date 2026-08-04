from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


ROUTING_SCHEMA = "causal_belief_routing_evidence_writes_v1"


@dataclass
class EvidenceRoutingTrace:
    row_indices: np.ndarray
    pair_ids: np.ndarray
    branches: np.ndarray
    layers: np.ndarray
    evidence_mass: np.ndarray
    control_mass: np.ndarray
    evidence_alignment_true: np.ndarray
    evidence_alignment_opposite: np.ndarray
    control_alignment_true: np.ndarray
    control_alignment_opposite: np.ndarray
    evidence_write_norm: np.ndarray
    control_write_norm: np.ndarray
    layer_alignment_true: np.ndarray
    layer_alignment_opposite: np.ndarray
    metadata: dict[str, Any]

    @property
    def evidence_margin(self) -> np.ndarray:
        return self.evidence_alignment_true - self.evidence_alignment_opposite

    @property
    def control_margin(self) -> np.ndarray:
        return self.control_alignment_true - self.control_alignment_opposite

    def validate(self) -> None:
        n = len(self.row_indices)
        if self.pair_ids.shape != (n,) or self.branches.shape != (n,):
            raise ValueError("routing row metadata is misaligned")
        expected = (n, len(self.layers))
        for name in (
            "evidence_mass",
            "control_mass",
            "evidence_alignment_true",
            "evidence_alignment_opposite",
            "control_alignment_true",
            "control_alignment_opposite",
            "evidence_write_norm",
            "control_write_norm",
        ):
            values = np.asarray(getattr(self, name))
            if values.ndim != 3 or values.shape[:2] != expected:
                raise ValueError(f"{name} must have shape [row, layer, head]")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")
        for name in ("layer_alignment_true", "layer_alignment_opposite"):
            values = np.asarray(getattr(self, name))
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite [row, layer]")
        if self.metadata.get("schema") != ROUTING_SCHEMA:
            raise ValueError("unsupported routing artifact schema")

    def save(self, path: str | Path, *, compressed: bool = True) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "row_indices": self.row_indices,
            "pair_ids": self.pair_ids,
            "branches": self.branches,
            "layers": self.layers,
            "evidence_mass": self.evidence_mass,
            "control_mass": self.control_mass,
            "evidence_alignment_true": self.evidence_alignment_true,
            "evidence_alignment_opposite": self.evidence_alignment_opposite,
            "control_alignment_true": self.control_alignment_true,
            "control_alignment_opposite": self.control_alignment_opposite,
            "evidence_write_norm": self.evidence_write_norm,
            "control_write_norm": self.control_write_norm,
            "layer_alignment_true": self.layer_alignment_true,
            "layer_alignment_opposite": self.layer_alignment_opposite,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        saver = np.savez_compressed if compressed else np.savez
        with output.open("wb") as handle:
            saver(handle, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceRoutingTrace":
        with np.load(Path(path), allow_pickle=False) as data:
            result = cls(
                row_indices=data["row_indices"],
                pair_ids=data["pair_ids"],
                branches=data["branches"],
                layers=data["layers"],
                evidence_mass=data["evidence_mass"],
                control_mass=data["control_mass"],
                evidence_alignment_true=data["evidence_alignment_true"],
                evidence_alignment_opposite=data["evidence_alignment_opposite"],
                control_alignment_true=data["control_alignment_true"],
                control_alignment_opposite=data["control_alignment_opposite"],
                evidence_write_norm=data["evidence_write_norm"],
                control_write_norm=data["control_write_norm"],
                layer_alignment_true=data["layer_alignment_true"],
                layer_alignment_opposite=data["layer_alignment_opposite"],
                metadata=json.loads(str(data["metadata_json"].item())),
            )
        result.validate()
        return result
