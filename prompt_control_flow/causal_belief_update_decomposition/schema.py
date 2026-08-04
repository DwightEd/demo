from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import AliasObservation


TRACE_SCHEMA = "causal_belief_routing_trace_v1"


def _pack_ragged_int(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    chunks: list[np.ndarray] = []
    for index, row in enumerate(rows):
        values = np.asarray(row, dtype=np.int64).reshape(-1)
        chunks.append(values)
        offsets[index + 1] = offsets[index] + len(values)
    flat = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.int64)
    return flat, offsets


def _unpack_ragged_int(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(flat, dtype=np.int64)
    bounds = np.asarray(offsets, dtype=np.int64)
    if bounds.ndim != 1 or len(bounds) < 1 or bounds[0] != 0 or bounds[-1] != len(values):
        raise ValueError("invalid ragged integer offsets")
    return [values[bounds[i] : bounds[i + 1]].copy() for i in range(len(bounds) - 1)]


@dataclass
class CausalBeliefTrace:
    pair_ids: np.ndarray
    branches: np.ndarray
    query_roles: np.ndarray
    template_families: np.ndarray
    query_vectors: np.ndarray
    exact_query_distributions: np.ndarray
    support_masks: np.ndarray
    base_support_masks: np.ndarray
    exact_beliefs: np.ndarray
    fourier_targets: np.ndarray
    update_fourier: np.ndarray
    frequencies: np.ndarray
    layers: np.ndarray
    states: np.ndarray
    residue_logits: np.ndarray
    logit_sketch: np.ndarray
    rendered_prompts: np.ndarray
    input_ids: list[np.ndarray]
    evidence_token_ranges: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_rows(self) -> int:
        return int(len(self.pair_ids))

    @property
    def current_mask(self) -> np.ndarray:
        return np.asarray(self.query_roles == "current", dtype=bool)

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[AliasObservation],
        *,
        frequencies: np.ndarray,
        layers: np.ndarray,
        states: np.ndarray,
        residue_logits: np.ndarray,
        logit_sketch: np.ndarray,
        rendered_prompts: Sequence[str],
        input_ids: Sequence[np.ndarray],
        evidence_token_ranges: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> "CausalBeliefTrace":
        if not observations:
            raise ValueError("trace requires at least one observation")
        n_rows = len(observations)
        trace = cls(
            pair_ids=np.asarray([row.pair_id for row in observations], dtype=np.int64),
            branches=np.asarray([row.branch for row in observations], dtype=np.int8),
            query_roles=np.asarray([row.query_role for row in observations], dtype="U8"),
            template_families=np.asarray(
                [row.template_family for row in observations], dtype=np.int8
            ),
            query_vectors=np.stack([row.query_vector for row in observations]),
            exact_query_distributions=np.stack(
                [row.exact_query_distribution for row in observations]
            ).astype(np.float32),
            support_masks=np.stack([row.support_mask for row in observations]),
            base_support_masks=np.stack(
                [row.base_support_mask for row in observations]
            ),
            exact_beliefs=np.stack([row.exact_belief for row in observations]).astype(
                np.float32
            ),
            fourier_targets=np.stack(
                [row.fourier_coordinates for row in observations]
            ).astype(np.float32),
            update_fourier=np.stack([row.update_fourier for row in observations]).astype(
                np.float32
            ),
            frequencies=np.asarray(frequencies, dtype=np.int64),
            layers=np.asarray(layers, dtype=np.int64),
            states=np.asarray(states),
            residue_logits=np.asarray(residue_logits, dtype=np.float32),
            logit_sketch=np.asarray(logit_sketch, dtype=np.float32),
            rendered_prompts=np.asarray([str(value) for value in rendered_prompts]),
            input_ids=[np.asarray(value, dtype=np.int64) for value in input_ids],
            evidence_token_ranges=np.asarray(evidence_token_ranges, dtype=np.int64),
            metadata=dict(metadata or {}),
        )
        if trace.states.shape[0] != n_rows:
            raise ValueError("state rows do not align with observations")
        trace.metadata.setdefault("schema", TRACE_SCHEMA)
        trace.validate()
        return trace

    def validate(self) -> None:
        n = self.n_rows
        aligned = {
            "branches": self.branches,
            "query_roles": self.query_roles,
            "template_families": self.template_families,
            "query_vectors": self.query_vectors,
            "exact_query_distributions": self.exact_query_distributions,
            "support_masks": self.support_masks,
            "base_support_masks": self.base_support_masks,
            "exact_beliefs": self.exact_beliefs,
            "fourier_targets": self.fourier_targets,
            "update_fourier": self.update_fourier,
            "states": self.states,
            "residue_logits": self.residue_logits,
            "logit_sketch": self.logit_sketch,
            "rendered_prompts": self.rendered_prompts,
            "evidence_token_ranges": self.evidence_token_ranges,
        }
        for name, values in aligned.items():
            if len(values) != n:
                raise ValueError(f"{name} has {len(values)} rows, expected {n}")
        if len(self.input_ids) != n:
            raise ValueError("input_ids do not align with trace rows")
        if self.states.ndim != 3 or self.states.shape[1] != len(self.layers):
            raise ValueError("states must have shape [rows, layers, hidden]")
        if self.evidence_token_ranges.shape != (n, 2):
            raise ValueError("evidence token ranges must have shape [rows, 2]")
        if self.residue_logits.shape != self.exact_query_distributions.shape:
            raise ValueError("residue logits and exact query distributions differ")
        if self.support_masks.shape != self.exact_beliefs.shape:
            raise ValueError("belief and support shapes differ")
        if self.frequencies.ndim != 2:
            raise ValueError("frequencies must be a matrix")
        expected_fourier = 2 * (len(self.frequencies) - 1)
        if self.fourier_targets.shape[1] != expected_fourier:
            raise ValueError("Fourier target width does not match the frequency grid")
        if not np.allclose(self.exact_query_distributions.sum(axis=1), 1.0):
            raise ValueError("exact query distributions must sum to one")
        if not np.isfinite(self.states.astype(np.float32)).all():
            raise ValueError("states contain non-finite values")
        if self.metadata.get("schema") != TRACE_SCHEMA:
            raise ValueError("trace metadata has an unsupported schema")
        token_groups = self.metadata.get("residue_token_id_groups")
        if token_groups is not None:
            if not isinstance(token_groups, list) or len(token_groups) != self.residue_logits.shape[1]:
                raise ValueError("residue token groups do not match the class logits")
            flattened = [int(token) for group in token_groups for token in group]
            if any(not group for group in token_groups) or len(flattened) != len(
                set(flattened)
            ):
                raise ValueError("residue token groups must be non-empty and disjoint")

    def save(self, path: str | Path, *, compressed: bool = False) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        flat_ids, id_offsets = _pack_ragged_int(self.input_ids)
        payload = {
            "pair_ids": self.pair_ids,
            "branches": self.branches,
            "query_roles": self.query_roles,
            "template_families": self.template_families,
            "query_vectors": self.query_vectors,
            "exact_query_distributions": self.exact_query_distributions,
            "support_masks": self.support_masks,
            "base_support_masks": self.base_support_masks,
            "exact_beliefs": self.exact_beliefs,
            "fourier_targets": self.fourier_targets,
            "update_fourier": self.update_fourier,
            "frequencies": self.frequencies,
            "layers": self.layers,
            "states": self.states,
            "residue_logits": self.residue_logits,
            "logit_sketch": self.logit_sketch,
            "rendered_prompts": self.rendered_prompts,
            "input_ids_flat": flat_ids,
            "input_ids_offsets": id_offsets,
            "evidence_token_ranges": self.evidence_token_ranges,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        saver = np.savez_compressed if compressed else np.savez
        with output.open("wb") as handle:
            saver(handle, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "CausalBeliefTrace":
        with np.load(Path(path), allow_pickle=False) as data:
            trace = cls(
                pair_ids=data["pair_ids"],
                branches=data["branches"],
                query_roles=data["query_roles"],
                template_families=data["template_families"],
                query_vectors=data["query_vectors"],
                exact_query_distributions=data["exact_query_distributions"],
                support_masks=data["support_masks"],
                base_support_masks=data["base_support_masks"],
                exact_beliefs=data["exact_beliefs"],
                fourier_targets=data["fourier_targets"],
                update_fourier=data["update_fourier"],
                frequencies=data["frequencies"],
                layers=data["layers"],
                states=data["states"],
                residue_logits=data["residue_logits"],
                logit_sketch=data["logit_sketch"],
                rendered_prompts=data["rendered_prompts"],
                input_ids=_unpack_ragged_int(
                    data["input_ids_flat"], data["input_ids_offsets"]
                ),
                evidence_token_ranges=data["evidence_token_ranges"],
                metadata=json.loads(str(data["metadata_json"].item())),
            )
        trace.validate()
        return trace
