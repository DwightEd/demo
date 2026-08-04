from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


PATCH_SCHEMA = "causal_belief_routing_source_patch_v1"


@dataclass
class SourcePatchTrace:
    pair_ids: np.ndarray
    recipient_branches: np.ndarray
    donor_branches: np.ndarray
    fold_ids: np.ndarray
    selected_head_counts: np.ndarray
    replay_js: np.ndarray
    evidence_logodds_shift: np.ndarray
    control_logodds_shift: np.ndarray
    random_head_logodds_shift: np.ndarray
    evidence_donor_probability_shift: np.ndarray
    control_donor_probability_shift: np.ndarray
    random_head_donor_probability_shift: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        n = len(self.pair_ids)
        for name in (
            "recipient_branches",
            "donor_branches",
            "fold_ids",
            "selected_head_counts",
            "replay_js",
            "evidence_logodds_shift",
            "control_logodds_shift",
            "random_head_logodds_shift",
            "evidence_donor_probability_shift",
            "control_donor_probability_shift",
            "random_head_donor_probability_shift",
        ):
            values = np.asarray(getattr(self, name))
            if values.shape != (n,):
                raise ValueError(f"{name} must have one value per patch direction")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")
        if np.any(self.recipient_branches == self.donor_branches):
            raise ValueError("donor and recipient branches must differ")
        if np.any(self.selected_head_counts < 1):
            raise ValueError("each patch requires at least one selected head")
        if self.metadata.get("schema") != PATCH_SCHEMA:
            raise ValueError("unsupported source-patch schema")

    def save(self, path: str | Path, *, compressed: bool = True) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pair_ids": self.pair_ids,
            "recipient_branches": self.recipient_branches,
            "donor_branches": self.donor_branches,
            "fold_ids": self.fold_ids,
            "selected_head_counts": self.selected_head_counts,
            "replay_js": self.replay_js,
            "evidence_logodds_shift": self.evidence_logodds_shift,
            "control_logodds_shift": self.control_logodds_shift,
            "random_head_logodds_shift": self.random_head_logodds_shift,
            "evidence_donor_probability_shift": self.evidence_donor_probability_shift,
            "control_donor_probability_shift": self.control_donor_probability_shift,
            "random_head_donor_probability_shift": self.random_head_donor_probability_shift,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        saver = np.savez_compressed if compressed else np.savez
        with output.open("wb") as handle:
            saver(handle, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "SourcePatchTrace":
        with np.load(Path(path), allow_pickle=False) as data:
            result = cls(
                pair_ids=data["pair_ids"],
                recipient_branches=data["recipient_branches"],
                donor_branches=data["donor_branches"],
                fold_ids=data["fold_ids"],
                selected_head_counts=data["selected_head_counts"],
                replay_js=data["replay_js"],
                evidence_logodds_shift=data["evidence_logodds_shift"],
                control_logodds_shift=data["control_logodds_shift"],
                random_head_logodds_shift=data["random_head_logodds_shift"],
                evidence_donor_probability_shift=data[
                    "evidence_donor_probability_shift"
                ],
                control_donor_probability_shift=data[
                    "control_donor_probability_shift"
                ],
                random_head_donor_probability_shift=data[
                    "random_head_donor_probability_shift"
                ],
                metadata=json.loads(str(data["metadata_json"].item())),
            )
        result.validate()
        return result
