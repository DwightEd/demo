from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


DIRECTION_NAMES = ("field", "shuffle", "random")
BASELINE_STEP_FEATURE_NAMES = (
    "entropy",
    "chosen_nll",
    "top1_top2_margin",
    "top1_probability",
)


@dataclass(frozen=True)
class CausalPullbackConfig:
    """Configuration for causal output-pullback extraction."""

    layer: int = 16
    min_donors: int = 6
    max_donors: int = 11
    epsilon_fraction: float = 0.02
    variant_batch_size: int = 8
    logit_token_chunk: int = 16
    calibration_floor: float = 2e-2
    replay_cosine_threshold: float = 0.98
    linearity_half_step: bool = True
    random_seed: int = 17

    def validate(self) -> None:
        if self.layer < 1:
            raise ValueError("layer must be a hidden-state index >= 1")
        if self.min_donors < 3:
            raise ValueError("min_donors must be at least three")
        if self.max_donors < self.min_donors:
            raise ValueError("max_donors must be >= min_donors")
        if self.epsilon_fraction <= 0.0:
            raise ValueError("epsilon_fraction must be positive")
        if self.variant_batch_size < 1 or self.logit_token_chunk < 1:
            raise ValueError("batch and token chunk sizes must be positive")
        if not -1.0 <= self.replay_cosine_threshold <= 1.0:
            raise ValueError("replay_cosine_threshold must lie in [-1, 1]")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")


@dataclass
class FieldWitnesses:
    """Geometry-derived perturbation directions for one reasoning chain."""

    field_direction: np.ndarray
    shuffle_direction: np.ndarray
    random_direction: np.ndarray
    field_witness_norm: np.ndarray
    shuffle_witness_norm: np.ndarray
    field_energy: np.ndarray
    field_calibrated_energy: np.ndarray
    donor_count: int

    @property
    def n_transitions(self) -> int:
        return int(self.field_direction.shape[0])

    def validate(self, hidden_dim: int) -> None:
        expected = (self.n_transitions, int(hidden_dim))
        for name in ("field_direction", "shuffle_direction", "random_direction"):
            value = np.asarray(getattr(self, name))
            if value.shape != expected:
                raise ValueError(f"{name} has shape {value.shape}, expected {expected}")
        for name in (
            "field_witness_norm",
            "shuffle_witness_norm",
            "field_energy",
            "field_calibrated_energy",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != (self.n_transitions,):
                raise ValueError(f"{name} must have one value per transition")


@dataclass
class CausalPullbackItem:
    chain_idx: int
    original_index: int
    problem_id: int
    sample_idx: int
    is_correct: int
    n_steps: int
    response_chars: int
    layer: int
    donor_count: int
    replay_kind: str
    replay_cosine: np.ndarray
    baseline_step_features: np.ndarray
    field_energy: np.ndarray
    field_calibrated_energy: np.ndarray
    witness_norms: np.ndarray
    fisher_transfer: np.ndarray
    chosen_logprob_transfer: np.ndarray
    entropy_transfer: np.ndarray
    primary_half_fisher_transfer: np.ndarray
    perturbation_scale: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        transitions = int(self.n_steps) - 1
        if transitions < 1:
            raise ValueError("a pullback item needs at least two steps")
        if np.asarray(self.replay_cosine).shape != (self.n_steps,):
            raise ValueError("replay_cosine shape does not match n_steps")
        if np.asarray(self.baseline_step_features).shape != (
            self.n_steps,
            len(BASELINE_STEP_FEATURE_NAMES),
        ):
            raise ValueError("baseline_step_features has an invalid shape")
        for name in ("field_energy", "field_calibrated_energy", "perturbation_scale"):
            if np.asarray(getattr(self, name)).shape != (transitions,):
                raise ValueError(f"{name} must have one value per transition")
        if np.asarray(self.witness_norms).shape != (
            len(DIRECTION_NAMES),
            transitions,
        ):
            raise ValueError("witness_norms has an invalid shape")
        transfer_shape = (len(DIRECTION_NAMES), transitions, self.n_steps)
        for name in (
            "fisher_transfer",
            "chosen_logprob_transfer",
            "entropy_transfer",
        ):
            if np.asarray(getattr(self, name)).shape != transfer_shape:
                raise ValueError(f"{name} has an invalid shape")
        if np.asarray(self.primary_half_fisher_transfer).shape != (
            transitions,
            self.n_steps,
        ):
            raise ValueError("primary_half_fisher_transfer has an invalid shape")


@dataclass
class CausalPullbackArtifact:
    items: list[CausalPullbackItem]
    metadata: dict[str, Any]
    skipped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for item in self.items:
            item.validate()

        def objects(name: str) -> np.ndarray:
            result = np.empty(self.n_items, dtype=object)
            for index, item in enumerate(self.items):
                result[index] = np.asarray(getattr(item, name))
            return result

        np.savez_compressed(
            path,
            chain_idx=np.asarray([item.chain_idx for item in self.items], dtype=np.int64),
            original_index=np.asarray(
                [item.original_index for item in self.items], dtype=np.int64
            ),
            problem_ids=np.asarray([item.problem_id for item in self.items], dtype=np.int64),
            sample_idx=np.asarray([item.sample_idx for item in self.items], dtype=np.int64),
            is_correct=np.asarray([item.is_correct for item in self.items], dtype=np.int8),
            n_steps=np.asarray([item.n_steps for item in self.items], dtype=np.int32),
            response_chars=np.asarray(
                [item.response_chars for item in self.items], dtype=np.int32
            ),
            layer=np.asarray([item.layer for item in self.items], dtype=np.int32),
            donor_count=np.asarray([item.donor_count for item in self.items], dtype=np.int16),
            replay_kind=np.asarray([item.replay_kind for item in self.items], dtype=object),
            replay_cosine=objects("replay_cosine"),
            baseline_step_features=objects("baseline_step_features"),
            field_energy=objects("field_energy"),
            field_calibrated_energy=objects("field_calibrated_energy"),
            witness_norms=objects("witness_norms"),
            fisher_transfer=objects("fisher_transfer"),
            chosen_logprob_transfer=objects("chosen_logprob_transfer"),
            entropy_transfer=objects("entropy_transfer"),
            primary_half_fisher_transfer=objects("primary_half_fisher_transfer"),
            perturbation_scale=objects("perturbation_scale"),
            item_metadata=np.asarray(
                [json.dumps(item.metadata, sort_keys=True) for item in self.items],
                dtype=object,
            ),
            direction_names=np.asarray(DIRECTION_NAMES, dtype=object),
            baseline_step_feature_names=np.asarray(
                BASELINE_STEP_FEATURE_NAMES, dtype=object
            ),
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
            skipped_json=np.asarray(json.dumps(self.skipped, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CausalPullbackArtifact":
        z = np.load(path, allow_pickle=True)
        names = tuple(str(x) for x in z["direction_names"].tolist())
        if names != DIRECTION_NAMES:
            raise ValueError(f"direction schema mismatch: {names}")
        feature_names = tuple(
            str(x) for x in z["baseline_step_feature_names"].tolist()
        )
        if feature_names != BASELINE_STEP_FEATURE_NAMES:
            raise ValueError(f"baseline feature schema mismatch: {feature_names}")
        items: list[CausalPullbackItem] = []
        for index in range(len(z["chain_idx"])):
            item = CausalPullbackItem(
                chain_idx=int(z["chain_idx"][index]),
                original_index=int(z["original_index"][index]),
                problem_id=int(z["problem_ids"][index]),
                sample_idx=int(z["sample_idx"][index]),
                is_correct=int(z["is_correct"][index]),
                n_steps=int(z["n_steps"][index]),
                response_chars=int(z["response_chars"][index]),
                layer=int(z["layer"][index]),
                donor_count=int(z["donor_count"][index]),
                replay_kind=str(z["replay_kind"][index]),
                replay_cosine=np.asarray(z["replay_cosine"][index], dtype=np.float32),
                baseline_step_features=np.asarray(
                    z["baseline_step_features"][index], dtype=np.float32
                ),
                field_energy=np.asarray(z["field_energy"][index], dtype=np.float32),
                field_calibrated_energy=np.asarray(
                    z["field_calibrated_energy"][index], dtype=np.float32
                ),
                witness_norms=np.asarray(z["witness_norms"][index], dtype=np.float32),
                fisher_transfer=np.asarray(z["fisher_transfer"][index], dtype=np.float32),
                chosen_logprob_transfer=np.asarray(
                    z["chosen_logprob_transfer"][index], dtype=np.float32
                ),
                entropy_transfer=np.asarray(
                    z["entropy_transfer"][index], dtype=np.float32
                ),
                primary_half_fisher_transfer=np.asarray(
                    z["primary_half_fisher_transfer"][index], dtype=np.float32
                ),
                perturbation_scale=np.asarray(
                    z["perturbation_scale"][index], dtype=np.float32
                ),
                metadata=json.loads(str(z["item_metadata"][index])),
            )
            item.validate()
            items.append(item)
        return cls(
            items=items,
            metadata=json.loads(str(np.asarray(z["metadata_json"]).item())),
            skipped=json.loads(str(np.asarray(z["skipped_json"]).item())),
        )
