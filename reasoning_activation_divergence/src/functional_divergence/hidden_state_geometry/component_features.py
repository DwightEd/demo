from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .component_contract import ComponentStepArtifact, load_component_artifact
from .contracts import ChainSample
from .tasks import TaskExample


@dataclass(frozen=True)
class ComponentStepSelection:
    artifact: ComponentStepArtifact
    current_step: int


def load_component_step_selection(example: TaskExample) -> ComponentStepSelection:
    """Load and validate the component artifact row aligned to one post_step example."""
    if example.task_name != "post_step":
        raise ValueError("component step data is only aligned for post_step examples")
    if example.boundary_step is None:
        raise ValueError("post_step examples require a boundary_step")
    current_step = int(example.boundary_step)
    if example.visible_steps != current_step + 1:
        raise ValueError("post_step visible_steps must equal boundary_step + 1")

    sample = example.sample
    if current_step < 0 or current_step >= sample.n_steps:
        raise ValueError(f"chain {sample.chain_id}: boundary_step exceeds n_steps")
    if sample.component_path is None:
        raise ValueError(f"chain {sample.chain_id}: sample.component_path is required")

    path = Path(sample.component_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = load_component_artifact(path)
    _validate_metadata(artifact, sample)
    _validate_step_boundaries(artifact, sample)
    _validate_step_label(artifact, sample, current_step)
    return ComponentStepSelection(artifact=artifact, current_step=current_step)


def _validate_metadata(artifact: ComponentStepArtifact, sample: ChainSample) -> None:
    dataset = str(artifact.metadata.get("dataset", ""))
    if dataset != sample.dataset:
        raise ValueError(
            f"chain {sample.chain_id}: metadata dataset disagrees: "
            f"{dataset!r} != {sample.dataset!r}"
        )
    _validate_metadata_chain_id(artifact.metadata, int(sample.chain_id))
    if int(artifact.first_error_step) != int(sample.first_error_step):
        raise ValueError(
            f"chain {sample.chain_id}: first_error_step disagrees between "
            "component artifact and ChainSample"
        )


def _validate_metadata_chain_id(metadata: dict, chain_id: int) -> None:
    value = metadata.get("sample_id")
    accepted = {str(chain_id), f"chain-{chain_id}", f"chain_{chain_id}"}
    if str(value) not in accepted or (
        "chain_id" in metadata and str(metadata["chain_id"]) not in accepted
    ):
        raise ValueError(f"chain {chain_id}: metadata chain id disagrees: {value!r}")


def _validate_step_boundaries(
    artifact: ComponentStepArtifact, sample: ChainSample
) -> None:
    ranges = np.asarray(sample.step_ranges, dtype=np.int64)
    if ranges.shape != (sample.n_steps, 2):
        raise ValueError(f"chain {sample.chain_id}: invalid ChainSample step_ranges")
    expected_step_start = ranges[:, 0]
    expected_step_end = ranges[:, 1] + 1
    expected_boundary = np.concatenate(
        [
            np.asarray([int(sample.response_start)], dtype=np.int64),
            expected_step_end,
        ]
    )
    if not np.array_equal(artifact.boundary_token_end, expected_boundary):
        raise ValueError(f"chain {sample.chain_id}: boundary_token_end disagrees")
    if not np.array_equal(artifact.step_token_start, expected_step_start):
        raise ValueError(f"chain {sample.chain_id}: step_token_start disagrees")
    if not np.array_equal(artifact.step_token_end, expected_step_end):
        raise ValueError(f"chain {sample.chain_id}: step_token_end disagrees")


def _validate_step_label(
    artifact: ComponentStepArtifact, sample: ChainSample, current_step: int
) -> None:
    labels = np.asarray(artifact.step_label, dtype=np.int64)
    if labels.shape != (sample.n_steps,):
        raise ValueError(f"chain {sample.chain_id}: step_label count disagrees")
    expected_label = int(sample.first_error_step == current_step)
    actual_label = int(labels[current_step])
    if actual_label == -1:
        raise ValueError(f"chain {sample.chain_id}: post-error component row requested")
    if actual_label != expected_label:
        raise ValueError(
            f"chain {sample.chain_id}: component step_label disagrees with "
            "post_step label semantics"
        )


__all__ = ["ComponentStepSelection", "load_component_step_selection"]
