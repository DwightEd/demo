from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from .extraction import ExtractionRow


TRACE_SCHEMA_VERSION = "constraint_belief_trace_v1"
STATE_SEMANTICS = "assistant_boundary_residual_state"


def _object_array(values: Sequence[object]) -> np.ndarray:
    output = np.empty(len(values), dtype=object)
    output[:] = list(values)
    return output


@dataclass
class BeliefTraceArtifact:
    problem_ids: np.ndarray
    template_families: np.ndarray
    prefix_index: np.ndarray
    previous_prefix_index: np.ndarray
    target_hypothesis: np.ndarray
    feasible_mask: np.ndarray
    condition_mask: np.ndarray
    hypotheses: np.ndarray
    layers: np.ndarray
    states: np.ndarray
    prompts: np.ndarray
    prompt_sha256: np.ndarray
    input_ids: np.ndarray
    prompt_token_count: np.ndarray
    output_entropy: np.ndarray
    output_margin: np.ndarray
    output_topk_mass: np.ndarray
    output_logit_sketch: np.ndarray
    metadata: dict[str, Any]
    state_semantics: str = STATE_SEMANTICS
    schema_version: str = TRACE_SCHEMA_VERSION

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[ExtractionRow],
        *,
        hypotheses: np.ndarray,
        layers: np.ndarray,
        states: np.ndarray,
        prompts: Sequence[str],
        input_ids: Sequence[np.ndarray],
        output_entropy: np.ndarray,
        output_margin: np.ndarray,
        output_topk_mass: np.ndarray,
        output_logit_sketch: np.ndarray | None = None,
        metadata: Mapping[str, Any],
    ) -> "BeliefTraceArtifact":
        n_rows = len(rows)
        if len(prompts) != n_rows or len(input_ids) != n_rows:
            raise ValueError("prompt and input-id rows must align with extraction rows")
        artifact = cls(
            problem_ids=np.asarray([row.problem_id for row in rows], dtype=np.int64),
            template_families=np.asarray(
                [row.template_family for row in rows], dtype=np.int64
            ),
            prefix_index=np.asarray([row.prefix_index for row in rows], dtype=np.int64),
            previous_prefix_index=np.asarray(
                [row.previous_prefix_index for row in rows], dtype=np.int64
            ),
            target_hypothesis=np.asarray(
                [row.target_hypothesis for row in rows], dtype=np.int64
            ),
            feasible_mask=np.stack([row.feasible_mask for row in rows]).astype(bool),
            condition_mask=np.stack([row.condition_mask for row in rows]).astype(bool),
            hypotheses=np.asarray(hypotheses, dtype=np.int64),
            layers=np.asarray(layers, dtype=np.int64),
            states=np.asarray(states),
            prompts=_object_array([str(value) for value in prompts]),
            prompt_sha256=_object_array(
                [sha256(str(value).encode("utf-8")).hexdigest() for value in prompts]
            ),
            input_ids=_object_array(
                [np.asarray(value, dtype=np.int64) for value in input_ids]
            ),
            prompt_token_count=np.asarray(
                [len(np.asarray(value)) for value in input_ids], dtype=np.int64
            ),
            output_entropy=np.asarray(output_entropy, dtype=np.float32),
            output_margin=np.asarray(output_margin, dtype=np.float32),
            output_topk_mass=np.asarray(output_topk_mass, dtype=np.float32),
            output_logit_sketch=(
                np.zeros((n_rows, 0), dtype=np.float32)
                if output_logit_sketch is None
                else np.asarray(output_logit_sketch, dtype=np.float32)
            ),
            metadata=dict(metadata),
        )
        artifact.validate()
        return artifact
    @property
    def n_rows(self) -> int:
        return int(len(self.problem_ids))

    def validate(self) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema: {self.schema_version}")
        if self.state_semantics != STATE_SEMANTICS:
            raise ValueError(f"unsupported state semantics: {self.state_semantics}")
        n_rows = self.n_rows
        aligned = {
            "template_families": self.template_families,
            "prefix_index": self.prefix_index,
            "previous_prefix_index": self.previous_prefix_index,
            "target_hypothesis": self.target_hypothesis,
            "feasible_mask": self.feasible_mask,
            "condition_mask": self.condition_mask,
            "states": self.states,
            "prompts": self.prompts,
            "prompt_sha256": self.prompt_sha256,
            "input_ids": self.input_ids,
            "prompt_token_count": self.prompt_token_count,
            "output_entropy": self.output_entropy,
            "output_margin": self.output_margin,
            "output_topk_mass": self.output_topk_mass,
            "output_logit_sketch": self.output_logit_sketch,
        }
        for name, values in aligned.items():
            if len(values) != n_rows:
                raise ValueError(f"{name} has {len(values)} rows; expected {n_rows}")
        hypotheses = np.asarray(self.hypotheses)
        if hypotheses.ndim != 2 or hypotheses.shape[1] != 2:
            raise ValueError("hypotheses must have shape [num_hypotheses, 2]")
        n_hypotheses = len(hypotheses)
        if self.feasible_mask.shape != (n_rows, n_hypotheses):
            raise ValueError("feasible_mask shape does not match rows and hypotheses")
        if self.condition_mask.shape != (n_rows, n_hypotheses):
            raise ValueError("condition_mask shape does not match rows and hypotheses")
        if self.states.ndim != 3 or self.states.shape[1] != len(self.layers):
            raise ValueError("states must have shape [rows, layers, hidden_dim]")
        if not np.isfinite(self.states).all():
            raise ValueError("states contain non-finite values")
        for name in ("output_entropy", "output_margin", "output_topk_mass"):
            if not np.isfinite(np.asarray(getattr(self, name))).all():
                raise ValueError(f"{name} contains non-finite values")
        if self.output_logit_sketch.ndim != 2:
            raise ValueError("output_logit_sketch must have shape [rows, sketch_dim]")
        if not np.isfinite(self.output_logit_sketch).all():
            raise ValueError("output_logit_sketch contains non-finite values")
        if np.any(self.feasible_mask.sum(axis=1) == 0):
            raise ValueError("every row must have a non-empty feasible support")
        row_index = {
            (int(problem), int(prefix)): index
            for index, (problem, prefix) in enumerate(
                zip(self.problem_ids, self.prefix_index)
            )
        }
        if len(row_index) != n_rows:
            raise ValueError("problem and prefix pairs must be unique")
        for index in range(n_rows):
            target = int(self.target_hypothesis[index])
            if not 0 <= target < n_hypotheses:
                raise ValueError("target_hypothesis is outside the hypothesis universe")
            if not bool(self.feasible_mask[index, target]):
                raise ValueError("target hypothesis is absent from a feasible support")
            prefix = int(self.prefix_index[index])
            previous = int(self.previous_prefix_index[index])
            if prefix == 0:
                if previous != -1 or not bool(self.condition_mask[index].all()):
                    raise ValueError("initial rows must use the identity condition")
                continue
            if previous != prefix - 1:
                raise ValueError("previous_prefix_index must identify the causal predecessor")
            key = (int(self.problem_ids[index]), previous)
            if key not in row_index:
                raise ValueError("a transition predecessor is missing")
            prior = self.feasible_mask[row_index[key]]
            expected = prior & self.condition_mask[index]
            if not np.array_equal(expected, self.feasible_mask[index]):
                raise ValueError("feasible support does not equal prior intersect condition")
        if any(len(np.asarray(ids)) == 0 for ids in self.input_ids):
            raise ValueError("input_ids cannot contain empty prompts")

    def _payload(self) -> dict[str, np.ndarray]:
        return {
            "schema_version": np.asarray(self.schema_version),
            "state_semantics": np.asarray(self.state_semantics),
            "problem_ids": self.problem_ids,
            "template_families": self.template_families,
            "prefix_index": self.prefix_index,
            "previous_prefix_index": self.previous_prefix_index,
            "target_hypothesis": self.target_hypothesis,
            "feasible_mask": self.feasible_mask,
            "condition_mask": self.condition_mask,
            "hypotheses": self.hypotheses,
            "layers": self.layers,
            "states": self.states,
            "prompts": self.prompts,
            "prompt_sha256": self.prompt_sha256,
            "input_ids": self.input_ids,
            "prompt_token_count": self.prompt_token_count,
            "output_entropy": self.output_entropy,
            "output_margin": self.output_margin,
            "output_topk_mass": self.output_topk_mass,
            "output_logit_sketch": self.output_logit_sketch,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }

    def save(self, path: str | Path, *, compressed: bool = False) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f"{output.name}.partial")
        writer = np.savez_compressed if compressed else np.savez
        with partial.open("wb") as handle:
            writer(handle, **self._payload())
        partial.replace(output)

    @classmethod
    def load(cls, path: str | Path) -> "BeliefTraceArtifact":
        with np.load(Path(path), allow_pickle=True) as data:
            artifact = cls(
                problem_ids=np.asarray(data["problem_ids"], dtype=np.int64),
                template_families=np.asarray(data["template_families"], dtype=np.int64),
                prefix_index=np.asarray(data["prefix_index"], dtype=np.int64),
                previous_prefix_index=np.asarray(
                    data["previous_prefix_index"], dtype=np.int64
                ),
                target_hypothesis=np.asarray(data["target_hypothesis"], dtype=np.int64),
                feasible_mask=np.asarray(data["feasible_mask"], dtype=bool),
                condition_mask=np.asarray(data["condition_mask"], dtype=bool),
                hypotheses=np.asarray(data["hypotheses"], dtype=np.int64),
                layers=np.asarray(data["layers"], dtype=np.int64),
                states=np.asarray(data["states"]),
                prompts=np.asarray(data["prompts"], dtype=object),
                prompt_sha256=np.asarray(data["prompt_sha256"], dtype=object),
                input_ids=np.asarray(data["input_ids"], dtype=object),
                prompt_token_count=np.asarray(data["prompt_token_count"], dtype=np.int64),
                output_entropy=np.asarray(data["output_entropy"], dtype=np.float32),
                output_margin=np.asarray(data["output_margin"], dtype=np.float32),
                output_topk_mass=np.asarray(data["output_topk_mass"], dtype=np.float32),
                output_logit_sketch=np.asarray(
                    data["output_logit_sketch"], dtype=np.float32
                ),
                metadata=json.loads(str(np.asarray(data["metadata_json"]).item())),
                state_semantics=str(np.asarray(data["state_semantics"]).item()),
                schema_version=str(np.asarray(data["schema_version"]).item()),
            )
        artifact.validate()
        return artifact


def merge_belief_trace_artifacts(
    paths: Sequence[str | Path],
) -> BeliefTraceArtifact:
    """Merge disjoint problem shards and re-establish canonical row ordering."""

    if not paths:
        raise ValueError("at least one trace shard is required")
    artifacts = [BeliefTraceArtifact.load(path) for path in paths]
    reference = artifacts[0]
    for artifact in artifacts[1:]:
        if not np.array_equal(artifact.hypotheses, reference.hypotheses):
            raise ValueError("trace shards use different hypothesis universes")
        if not np.array_equal(artifact.layers, reference.layers):
            raise ValueError("trace shards use different residual depths")
        for key in ("model", "tokenizer"):
            if artifact.metadata.get(key) != reference.metadata.get(key):
                raise ValueError(f"trace shard metadata differs for {key}")
        if artifact.state_semantics != reference.state_semantics:
            raise ValueError("trace shards use different state semantics")

    concatenated: dict[str, np.ndarray] = {}
    row_fields = (
        "problem_ids",
        "template_families",
        "prefix_index",
        "previous_prefix_index",
        "target_hypothesis",
        "feasible_mask",
        "condition_mask",
        "states",
        "prompts",
        "prompt_sha256",
        "input_ids",
        "prompt_token_count",
        "output_entropy",
        "output_margin",
        "output_topk_mass",
        "output_logit_sketch",
    )
    for name in row_fields:
        concatenated[name] = np.concatenate(
            [np.asarray(getattr(artifact, name)) for artifact in artifacts], axis=0
        )
    order = np.lexsort(
        (concatenated["prefix_index"], concatenated["problem_ids"])
    )
    for name in row_fields:
        concatenated[name] = concatenated[name][order]
    metadata = dict(reference.metadata)
    metadata.update(
        merged_shards=[str(Path(path)) for path in paths],
        num_merged_shards=len(paths),
    )
    merged = BeliefTraceArtifact(
        **concatenated,
        hypotheses=reference.hypotheses.copy(),
        layers=reference.layers.copy(),
        metadata=metadata,
        state_semantics=reference.state_semantics,
        schema_version=reference.schema_version,
    )
    merged.validate()
    return merged
