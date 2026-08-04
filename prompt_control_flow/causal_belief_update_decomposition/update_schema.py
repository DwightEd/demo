from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


BELIEF_UPDATE_SCHEMA = "causal_belief_update_decomposition_v1"

_COMPONENT_FIELDS = (
    "attention_alignment_true",
    "attention_alignment_opposite",
    "mlp_alignment_true",
    "mlp_alignment_opposite",
    "block_alignment_true",
    "block_alignment_opposite",
    "attention_target_progress",
    "mlp_target_progress",
    "block_target_progress",
    "attention_target_error",
    "mlp_target_error",
    "block_target_error",
    "attention_write_norm",
    "mlp_write_norm",
    "block_write_norm",
    "reconstruction_relative_error",
    "state_replay_relative_error",
)

_NON_NEGATIVE_FIELDS = (
    "attention_target_error",
    "mlp_target_error",
    "block_target_error",
    "attention_write_norm",
    "mlp_write_norm",
    "block_write_norm",
    "reconstruction_relative_error",
    "state_replay_relative_error",
)


@dataclass
class BeliefUpdateTrace:
    row_indices: np.ndarray
    pair_ids: np.ndarray
    branches: np.ndarray
    layers: np.ndarray
    attention_alignment_true: np.ndarray
    attention_alignment_opposite: np.ndarray
    mlp_alignment_true: np.ndarray
    mlp_alignment_opposite: np.ndarray
    block_alignment_true: np.ndarray
    block_alignment_opposite: np.ndarray
    attention_target_progress: np.ndarray
    mlp_target_progress: np.ndarray
    block_target_progress: np.ndarray
    attention_target_error: np.ndarray
    mlp_target_error: np.ndarray
    block_target_error: np.ndarray
    attention_write_norm: np.ndarray
    mlp_write_norm: np.ndarray
    block_write_norm: np.ndarray
    reconstruction_relative_error: np.ndarray
    state_replay_relative_error: np.ndarray
    metadata: dict[str, Any]

    @property
    def attention_margin(self) -> np.ndarray:
        return self.attention_alignment_true - self.attention_alignment_opposite

    @property
    def mlp_margin(self) -> np.ndarray:
        return self.mlp_alignment_true - self.mlp_alignment_opposite

    @property
    def block_margin(self) -> np.ndarray:
        return self.block_alignment_true - self.block_alignment_opposite

    @property
    def mlp_margin_gain(self) -> np.ndarray:
        return self.block_margin - self.attention_margin

    @property
    def mlp_target_error_reduction(self) -> np.ndarray:
        return self.attention_target_error - self.block_target_error

    def validate(self) -> None:
        n = len(self.row_indices)
        if self.pair_ids.shape != (n,) or self.branches.shape != (n,):
            raise ValueError("belief-update row metadata is misaligned")
        if self.layers.ndim != 1 or len(self.layers) < 1:
            raise ValueError("belief-update trace requires at least one layer")
        if len(np.unique(self.layers)) != len(self.layers):
            raise ValueError("belief-update layers must be unique")
        expected = (n, len(self.layers))
        for name in _COMPONENT_FIELDS:
            values = np.asarray(getattr(self, name))
            if values.shape != expected:
                raise ValueError(f"{name} must have shape [row, layer]")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")
        for name in _NON_NEGATIVE_FIELDS:
            if np.any(np.asarray(getattr(self, name)) < 0.0):
                raise ValueError(f"{name} must be non-negative")
        if self.metadata.get("schema") != BELIEF_UPDATE_SCHEMA:
            raise ValueError("unsupported belief-update artifact schema")

    def save(self, path: str | Path, *, compressed: bool = True) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "row_indices": self.row_indices,
            "pair_ids": self.pair_ids,
            "branches": self.branches,
            "layers": self.layers,
            **{name: getattr(self, name) for name in _COMPONENT_FIELDS},
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        saver = np.savez_compressed if compressed else np.savez
        with output.open("wb") as handle:
            saver(handle, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "BeliefUpdateTrace":
        with np.load(Path(path), allow_pickle=False) as data:
            result = cls(
                row_indices=data["row_indices"],
                pair_ids=data["pair_ids"],
                branches=data["branches"],
                layers=data["layers"],
                **{name: data[name] for name in _COMPONENT_FIELDS},
                metadata=json.loads(str(data["metadata_json"].item())),
            )
        result.validate()
        return result
