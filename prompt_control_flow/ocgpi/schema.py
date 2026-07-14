from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


OCGPI_TRACE_SCHEMA_VERSION = "ocgpi_compact_output_trace_v1"


@dataclass
class CompactTraceItem:
    """Compact causal output trace for one reasoning response.

    ``token_features[j]`` is computed from the logits that predict response
    token ``j``.  Step ranges are inclusive and relative to this response-token
    axis, so they never depend on prompt length after extraction.
    """

    chain_idx: int
    problem_id: int
    gold_error_step: int
    is_correct: int
    sample_idx: int
    dataset: str
    generator: str
    response_hash: str
    token_ids: np.ndarray
    token_features: np.ndarray
    step_features: np.ndarray
    step_token_ranges: np.ndarray
    replay_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, token_dim: int, step_dim: int) -> None:
        token_ids = np.asarray(self.token_ids)
        token_features = np.asarray(self.token_features)
        step_features = np.asarray(self.step_features)
        ranges = np.asarray(self.step_token_ranges)
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be one-dimensional")
        if token_features.ndim != 2 or token_features.shape != (
            len(token_ids),
            token_dim,
        ):
            raise ValueError("token_features shape does not match token_ids/schema")
        if step_features.ndim != 2 or step_features.shape[1] != step_dim:
            raise ValueError("step_features shape does not match schema")
        if ranges.shape != (step_features.shape[0], 2):
            raise ValueError("step_token_ranges must be [n_steps, 2]")
        if len(ranges) == 0:
            raise ValueError("a trace must contain at least one complete step")
        if np.any(ranges[:, 0] < 0) or np.any(ranges[:, 1] < ranges[:, 0]):
            raise ValueError("invalid inclusive step token range")
        if np.any(ranges[:, 1] >= len(token_ids)):
            raise ValueError("a step token range exceeds the response-token axis")
        if len(ranges) > 1 and np.any(ranges[1:, 0] <= ranges[:-1, 1]):
            raise ValueError("step token ranges must be ordered and non-overlapping")
        if not np.isfinite(token_features).all():
            raise ValueError("token_features contains a non-finite value")
        if not np.isfinite(step_features).all():
            raise ValueError("step_features contains a non-finite value")


@dataclass
class TraceArtifact:
    """Packed, disk-friendly OC-GPI output trace collection."""

    chain_idx: np.ndarray
    problem_id: np.ndarray
    gold_error_step: np.ndarray
    is_correct: np.ndarray
    sample_idx: np.ndarray
    dataset: np.ndarray
    generator: np.ndarray
    response_hash: np.ndarray
    n_steps: np.ndarray
    token_offsets: np.ndarray
    token_ids: np.ndarray
    token_features: np.ndarray
    token_feature_names: tuple[str, ...]
    step_offsets: np.ndarray
    step_features: np.ndarray
    step_feature_names: tuple[str, ...]
    step_token_ranges: np.ndarray
    replay_kind: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_chains(self) -> int:
        return int(len(self.chain_idx))

    def validate(self) -> None:
        n = self.n_chains
        one_per_chain = (
            self.problem_id,
            self.gold_error_step,
            self.is_correct,
            self.sample_idx,
            self.dataset,
            self.generator,
            self.response_hash,
            self.n_steps,
            self.replay_kind,
        )
        if any(len(values) != n for values in one_per_chain):
            raise ValueError("trace metadata arrays have inconsistent chain counts")
        if not np.isin(self.is_correct, (-1, 0, 1)).all():
            raise ValueError("is_correct must use -1=unknown, 0=error, 1=correct")
        if self.token_offsets.shape != (n + 1,) or self.step_offsets.shape != (n + 1,):
            raise ValueError("packed offsets must have n_chains + 1 entries")
        if self.token_offsets[0] != 0 or self.step_offsets[0] != 0:
            raise ValueError("packed offsets must begin at zero")
        if np.any(np.diff(self.token_offsets) <= 0) or np.any(
            np.diff(self.step_offsets) <= 0
        ):
            raise ValueError("every chain must contain tokens and steps")
        if int(self.token_offsets[-1]) != len(self.token_ids):
            raise ValueError("token_offsets does not terminate at token_ids length")
        if int(self.step_offsets[-1]) != len(self.step_token_ranges):
            raise ValueError("step_offsets does not terminate at step range length")
        if self.token_features.shape != (
            len(self.token_ids),
            len(self.token_feature_names),
        ):
            raise ValueError("packed token feature matrix disagrees with schema")
        if self.step_features.shape != (
            len(self.step_token_ranges),
            len(self.step_feature_names),
        ):
            raise ValueError("packed step feature matrix disagrees with schema")
        if self.step_token_ranges.shape[1:] != (2,):
            raise ValueError("packed step ranges must be [total_steps, 2]")
        if len(np.unique(self.chain_idx.astype(np.int64))) != n:
            raise ValueError("chain_idx must uniquely identify each packed chain")
        if (
            not np.isfinite(self.token_features).all()
            or not np.isfinite(self.step_features).all()
        ):
            raise ValueError("trace artifact contains non-finite features")
        for i in range(n):
            sa, sb = int(self.step_offsets[i]), int(self.step_offsets[i + 1])
            ta, tb = int(self.token_offsets[i]), int(self.token_offsets[i + 1])
            if sb - sa != int(self.n_steps[i]):
                raise ValueError(
                    f"chain {int(self.chain_idx[i])}: n_steps disagrees with offsets"
                )
            gold = int(self.gold_error_step[i])
            if gold < -1 or gold >= int(self.n_steps[i]):
                raise ValueError(
                    f"chain {int(self.chain_idx[i])}: gold_error_step={gold} is outside "
                    f"[-1, {int(self.n_steps[i]) - 1}]"
                )
            local = self.step_token_ranges[sa:sb]
            if np.any(local[:, 0] < 0) or np.any(local[:, 1] >= tb - ta):
                raise ValueError(
                    f"chain {int(self.chain_idx[i])}: invalid local step ranges"
                )

    def token_matrix(self, i: int) -> np.ndarray:
        a, b = int(self.token_offsets[i]), int(self.token_offsets[i + 1])
        return self.token_features[a:b]

    def token_id_vector(self, i: int) -> np.ndarray:
        a, b = int(self.token_offsets[i]), int(self.token_offsets[i + 1])
        return self.token_ids[a:b]

    def step_matrix(self, i: int) -> np.ndarray:
        a, b = int(self.step_offsets[i]), int(self.step_offsets[i + 1])
        return self.step_features[a:b]

    def step_ranges(self, i: int) -> np.ndarray:
        a, b = int(self.step_offsets[i]), int(self.step_offsets[i + 1])
        return self.step_token_ranges[a:b]

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            trace_schema_version=np.asarray(OCGPI_TRACE_SCHEMA_VERSION),
            chain_idx=self.chain_idx.astype(np.int64),
            problem_id=self.problem_id.astype(np.int64),
            gold_error_step=self.gold_error_step.astype(np.int64),
            is_correct=self.is_correct.astype(np.int8),
            sample_idx=self.sample_idx.astype(np.int64),
            dataset=np.asarray(self.dataset, dtype=str),
            generator=np.asarray(self.generator, dtype=str),
            response_hash=np.asarray(self.response_hash, dtype=str),
            n_steps=self.n_steps.astype(np.int32),
            token_offsets=self.token_offsets.astype(np.int64),
            token_ids=self.token_ids.astype(np.int64),
            token_features=self.token_features.astype(np.float32),
            token_feature_names=np.asarray(self.token_feature_names, dtype=str),
            step_offsets=self.step_offsets.astype(np.int64),
            step_features=self.step_features.astype(np.float32),
            step_feature_names=np.asarray(self.step_feature_names, dtype=str),
            step_token_ranges=self.step_token_ranges.astype(np.int32),
            replay_kind=np.asarray(self.replay_kind, dtype=str),
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TraceArtifact":
        z = np.load(path, allow_pickle=False)
        version = str(np.asarray(z["trace_schema_version"]).item())
        if version != OCGPI_TRACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported OC-GPI trace schema {version!r}")
        artifact = cls(
            chain_idx=np.asarray(z["chain_idx"], dtype=np.int64),
            problem_id=np.asarray(z["problem_id"], dtype=np.int64),
            gold_error_step=np.asarray(z["gold_error_step"], dtype=np.int64),
            is_correct=np.asarray(z["is_correct"], dtype=np.int8),
            sample_idx=np.asarray(z["sample_idx"], dtype=np.int64),
            dataset=np.asarray(z["dataset"], dtype=str),
            generator=np.asarray(z["generator"], dtype=str),
            response_hash=np.asarray(z["response_hash"], dtype=str),
            n_steps=np.asarray(z["n_steps"], dtype=np.int32),
            token_offsets=np.asarray(z["token_offsets"], dtype=np.int64),
            token_ids=np.asarray(z["token_ids"], dtype=np.int64),
            token_features=np.asarray(z["token_features"], dtype=np.float32),
            token_feature_names=tuple(str(x) for x in z["token_feature_names"]),
            step_offsets=np.asarray(z["step_offsets"], dtype=np.int64),
            step_features=np.asarray(z["step_features"], dtype=np.float32),
            step_feature_names=tuple(str(x) for x in z["step_feature_names"]),
            step_token_ranges=np.asarray(z["step_token_ranges"], dtype=np.int32),
            replay_kind=np.asarray(z["replay_kind"], dtype=str),
            metadata=json.loads(str(np.asarray(z["metadata_json"]).item())),
        )
        artifact.validate()
        return artifact

    @classmethod
    def from_items(
        cls,
        items: Sequence[CompactTraceItem],
        *,
        token_feature_names: Sequence[str],
        step_feature_names: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> "TraceArtifact":
        if not items:
            raise ValueError("cannot pack an empty trace collection")
        token_names = tuple(str(x) for x in token_feature_names)
        step_names = tuple(str(x) for x in step_feature_names)
        if len(set(token_names)) != len(token_names) or len(set(step_names)) != len(
            step_names
        ):
            raise ValueError("feature names must be unique")
        for item in items:
            item.validate(len(token_names), len(step_names))

        token_lengths = np.asarray(
            [len(item.token_ids) for item in items], dtype=np.int64
        )
        step_lengths = np.asarray(
            [len(item.step_token_ranges) for item in items], dtype=np.int64
        )
        token_offsets = np.concatenate([[0], np.cumsum(token_lengths)]).astype(np.int64)
        step_offsets = np.concatenate([[0], np.cumsum(step_lengths)]).astype(np.int64)
        artifact = cls(
            chain_idx=np.asarray([item.chain_idx for item in items], dtype=np.int64),
            problem_id=np.asarray([item.problem_id for item in items], dtype=np.int64),
            gold_error_step=np.asarray(
                [item.gold_error_step for item in items], dtype=np.int64
            ),
            is_correct=np.asarray([item.is_correct for item in items], dtype=np.int8),
            sample_idx=np.asarray([item.sample_idx for item in items], dtype=np.int64),
            dataset=np.asarray([item.dataset for item in items], dtype=str),
            generator=np.asarray([item.generator for item in items], dtype=str),
            response_hash=np.asarray([item.response_hash for item in items], dtype=str),
            n_steps=step_lengths.astype(np.int32),
            token_offsets=token_offsets,
            token_ids=np.concatenate(
                [np.asarray(item.token_ids, dtype=np.int64) for item in items]
            ),
            token_features=np.concatenate(
                [np.asarray(item.token_features, dtype=np.float32) for item in items],
                axis=0,
            ),
            token_feature_names=token_names,
            step_offsets=step_offsets,
            step_features=np.concatenate(
                [np.asarray(item.step_features, dtype=np.float32) for item in items],
                axis=0,
            ),
            step_feature_names=step_names,
            step_token_ranges=np.concatenate(
                [np.asarray(item.step_token_ranges, dtype=np.int32) for item in items],
                axis=0,
            ),
            replay_kind=np.asarray([item.replay_kind for item in items], dtype=str),
            metadata={
                **dict(metadata or {}),
                "item_metadata": {
                    str(item.chain_idx): item.metadata
                    for item in items
                    if item.metadata
                },
            },
        )
        artifact.validate()
        return artifact
