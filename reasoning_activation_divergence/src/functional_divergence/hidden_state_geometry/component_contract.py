from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

COMPONENT_STEP_SCHEMA = "component_step_v1"
SOURCE_STEP_PADDING = -32768

# Float16 extraction accumulates quantized source messages before storing them.
# A 3% absolute/relative tolerance admits half-precision roundoff while still
# rejecting missing or materially inconsistent residual-space message sums.
ATTN_SUM_ATOL = 3e-2
ATTN_SUM_RTOL = 3e-2

REQUIRED_ARRAYS = (
    "input_ids",
    "step_token_start",
    "step_token_end",
    "boundary_token_end",
    "first_error_step",
    "step_label",
    "source_step_id",
    "source_mask",
    "resid_boundary",
    "attn_msg_resid_by_source",
    "attn_out_step",
    "mlp_out_step",
    "residual_layers",
    "attention_layers",
    "mlp_layers",
    "metadata_json",
)

REQUIRED_METADATA_KEYS = (
    "schema",
    "sample_id",
    "dataset",
    "model_name",
    "model_revision",
    "tokenizer_name",
    "tokenizer_revision",
    "extractor_commit",
    "source_trace_sha256",
    "generation_config_sha256",
)


@dataclass(frozen=True)
class ComponentStepArtifact:
    """One chain's step-aligned component tensors and provenance."""

    input_ids: np.ndarray
    step_token_start: np.ndarray
    step_token_end: np.ndarray
    boundary_token_end: np.ndarray
    first_error_step: int
    step_label: np.ndarray
    source_step_id: np.ndarray
    source_mask: np.ndarray
    resid_boundary: np.ndarray
    attn_msg_resid_by_source: np.ndarray
    attn_out_step: np.ndarray
    mlp_out_step: np.ndarray
    residual_layers: np.ndarray
    attention_layers: np.ndarray
    mlp_layers: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        """Validate shapes, labels, source causality, and component sums."""
        self._validate_metadata()

        input_ids = _integer_vector(self.input_ids, "input_ids")
        step_token_start = _integer_vector(self.step_token_start, "step_token_start")
        step_token_end = _integer_vector(self.step_token_end, "step_token_end")
        boundary_token_end = _integer_vector(
            self.boundary_token_end, "boundary_token_end"
        )
        step_label = _integer_vector(self.step_label, "step_label")
        residual_layers = _integer_vector(self.residual_layers, "residual_layers")
        attention_layers = _integer_vector(self.attention_layers, "attention_layers")
        mlp_layers = _integer_vector(self.mlp_layers, "mlp_layers")

        source_step_id = _integer_array(self.source_step_id, "source_step_id")
        source_mask = np.asarray(self.source_mask)
        if not np.issubdtype(source_mask.dtype, np.bool_):
            raise ValueError("source_mask must be boolean")

        resid_boundary = _finite_float_array(self.resid_boundary, "resid_boundary")
        attn_msg = _finite_float_array(
            self.attn_msg_resid_by_source, "attn_msg_resid_by_source"
        )
        attn_out = _finite_float_array(self.attn_out_step, "attn_out_step")
        mlp_out = _finite_float_array(self.mlp_out_step, "mlp_out_step")

        step_count = int(step_token_start.shape[0])
        token_count = int(input_ids.shape[0])
        if step_count == 0:
            raise ValueError("component artifacts must contain at least one step")
        if step_token_end.shape != (step_count,):
            raise ValueError("step_token_end must have shape [S]")
        if boundary_token_end.shape != (step_count + 1,):
            raise ValueError("boundary_token_end must have shape [S+1]")
        if step_label.shape != (step_count,):
            raise ValueError("step_label must have shape [S]")

        if source_step_id.ndim != 2:
            raise ValueError("source_step_id must have shape [S,B]")
        source_block_count = int(source_step_id.shape[1])
        if source_block_count == 0:
            raise ValueError("source_step_id must include at least one source block")
        if source_step_id.shape != (step_count, source_block_count):
            raise ValueError("source_step_id must have shape [S,B]")
        if source_mask.shape != (step_count, source_block_count):
            raise ValueError("source_mask must have shape [S,B]")

        if resid_boundary.ndim != 3:
            raise ValueError("resid_boundary must have shape [S+1,Lr,D]")
        residual_layer_count = int(resid_boundary.shape[1])
        hidden_size = int(resid_boundary.shape[2]) if resid_boundary.ndim == 3 else 0
        if resid_boundary.shape != (
            step_count + 1,
            residual_layer_count,
            hidden_size,
        ):
            raise ValueError("resid_boundary must have shape [S+1,Lr,D]")
        if residual_layer_count == 0 or hidden_size == 0:
            raise ValueError(
                "resid_boundary must include at least one layer and dimension"
            )
        if residual_layers.shape != (residual_layer_count,):
            raise ValueError("residual_layers must have shape [Lr]")

        if attn_msg.ndim != 4:
            raise ValueError("attn_msg_resid_by_source must have shape [S,La,B,D]")
        attention_layer_count = int(attn_msg.shape[1])
        if attn_msg.shape != (
            step_count,
            attention_layer_count,
            source_block_count,
            hidden_size,
        ):
            raise ValueError("attn_msg_resid_by_source must have shape [S,La,B,D]")
        if attention_layer_count == 0:
            raise ValueError("attn_msg_resid_by_source must include at least one layer")
        if attn_out.shape != (step_count, attention_layer_count, hidden_size):
            raise ValueError("attn_out_step must have shape [S,La,D]")
        if attention_layers.shape != (attention_layer_count,):
            raise ValueError("attention_layers must have shape [La]")

        if mlp_out.ndim != 3:
            raise ValueError("mlp_out_step must have shape [S,Lm,D]")
        mlp_layer_count = int(mlp_out.shape[1])
        if mlp_out.shape != (step_count, mlp_layer_count, hidden_size):
            raise ValueError("mlp_out_step must have shape [S,Lm,D]")
        if mlp_layer_count == 0:
            raise ValueError("mlp_out_step must include at least one layer")
        if mlp_layers.shape != (mlp_layer_count,):
            raise ValueError("mlp_layers must have shape [Lm]")

        _validate_unique_layers(residual_layers, "residual_layers")
        _validate_unique_layers(attention_layers, "attention_layers")
        _validate_unique_layers(mlp_layers, "mlp_layers")
        _validate_token_boundaries(
            token_count, step_token_start, step_token_end, boundary_token_end
        )
        _validate_labels(step_label, self.first_error_step)
        _validate_sources(source_step_id, source_mask)
        _validate_attention_sum(attn_msg, attn_out, source_mask)

    def save(self, path: str | Path) -> None:
        """Validate and write one compressed NPZ artifact with scalar metadata."""
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_ids": self.input_ids,
            "step_token_start": self.step_token_start,
            "step_token_end": self.step_token_end,
            "boundary_token_end": self.boundary_token_end,
            "first_error_step": np.asarray(self.first_error_step, dtype=np.int32),
            "step_label": self.step_label,
            "source_step_id": self.source_step_id,
            "source_mask": self.source_mask,
            "resid_boundary": self.resid_boundary,
            "attn_msg_resid_by_source": self.attn_msg_resid_by_source,
            "attn_out_step": self.attn_out_step,
            "mlp_out_step": self.mlp_out_step,
            "residual_layers": self.residual_layers,
            "attention_layers": self.attention_layers,
            "mlp_layers": self.mlp_layers,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        with output.open("wb") as handle:
            np.savez_compressed(handle, **payload)

    @classmethod
    def load(cls, path: str | Path) -> ComponentStepArtifact:
        """Load and validate a component_step_v1 NPZ artifact."""
        with np.load(Path(path), allow_pickle=False) as archive:
            missing = sorted(set(REQUIRED_ARRAYS).difference(archive.files))
            if missing:
                raise ValueError(
                    f"component artifact is missing required arrays: {missing}"
                )
            metadata = _metadata_from_archive(archive)
            result = cls(
                input_ids=archive["input_ids"],
                step_token_start=archive["step_token_start"],
                step_token_end=archive["step_token_end"],
                boundary_token_end=archive["boundary_token_end"],
                first_error_step=_integer_scalar(
                    archive["first_error_step"], "first_error_step"
                ),
                step_label=archive["step_label"],
                source_step_id=archive["source_step_id"],
                source_mask=archive["source_mask"],
                resid_boundary=archive["resid_boundary"],
                attn_msg_resid_by_source=archive["attn_msg_resid_by_source"],
                attn_out_step=archive["attn_out_step"],
                mlp_out_step=archive["mlp_out_step"],
                residual_layers=archive["residual_layers"],
                attention_layers=archive["attention_layers"],
                mlp_layers=archive["mlp_layers"],
                metadata=metadata,
            )
        result.validate()
        return result

    def _validate_metadata(self) -> None:
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        missing = [key for key in REQUIRED_METADATA_KEYS if key not in self.metadata]
        if missing:
            raise ValueError(f"metadata is missing required keys: {missing}")
        if self.metadata.get("schema") != COMPONENT_STEP_SCHEMA:
            raise ValueError("unsupported component artifact schema")
        for key in REQUIRED_METADATA_KEYS:
            value = self.metadata[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"metadata {key!r} must be a non-empty string")
        for key in ("source_trace_sha256", "generation_config_sha256"):
            value = self.metadata[key]
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(f"metadata {key!r} must be a lowercase SHA-256 digest")


def load_component_artifact(path: str | Path) -> ComponentStepArtifact:
    """Load a component_step_v1 artifact from a compressed NPZ file."""
    return ComponentStepArtifact.load(path)


def validate_component_artifact(artifact: ComponentStepArtifact) -> None:
    """Validate a component_step_v1 artifact without modifying it."""
    artifact.validate()


def _metadata_from_archive(archive: np.lib.npyio.NpzFile) -> dict[str, Any]:
    metadata_json = np.asarray(archive["metadata_json"])
    if metadata_json.shape != ():
        raise ValueError("metadata_json must be a scalar JSON string")
    try:
        metadata = json.loads(str(metadata_json.item()))
    except json.JSONDecodeError as exc:
        raise ValueError("metadata_json contains invalid JSON") from exc
    if not isinstance(metadata, dict):
        raise TypeError("metadata_json must decode to a JSON object")
    return metadata


def _integer_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = _integer_array(values, name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 integer array")
    return array


def _integer_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array")
    return array


def _integer_scalar(values: np.ndarray, name: str) -> int:
    array = _integer_array(values, name)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar integer")
    return int(array.item())


def _finite_float_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be a floating array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_unique_layers(layers: np.ndarray, name: str) -> None:
    if layers.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one layer")
    if len(np.unique(layers)) != layers.shape[0]:
        raise ValueError(f"{name} must contain unique layer ids")


def _validate_token_boundaries(
    token_count: int,
    step_token_start: np.ndarray,
    step_token_end: np.ndarray,
    boundary_token_end: np.ndarray,
) -> None:
    if np.any(boundary_token_end < 0) or np.any(boundary_token_end > token_count):
        raise ValueError("boundary_token_end must lie within input_ids")
    if np.any(np.diff(boundary_token_end) <= 0):
        raise ValueError("boundary_token_end must be strictly increasing")
    if not np.array_equal(step_token_end, boundary_token_end[1:]):
        raise ValueError("step_token_end must align exactly with boundary_token_end")
    if np.any(step_token_start < 0) or np.any(step_token_end > token_count):
        raise ValueError("step token ranges must lie within input_ids")
    if np.any(step_token_end <= step_token_start):
        raise ValueError("step token ranges must be non-empty half-open intervals")
    if np.any(step_token_start < boundary_token_end[:-1]):
        raise ValueError("step token ranges must not overlap earlier steps")
    if boundary_token_end[0] < 1 or boundary_token_end[-1] != token_count:
        raise ValueError(
            "boundary_token_end must start after a non-empty prompt and end at input_ids"
        )


def _validate_labels(step_label: np.ndarray, first_error_step: int) -> None:
    if not np.isin(step_label, (-1, 0, 1)).all():
        raise ValueError("step_label values must be in {-1,0,1}")
    step_count = len(step_label)
    if first_error_step < -1 or first_error_step >= step_count:
        raise ValueError("first_error_step must be -1 or a valid step index")
    if first_error_step == -1:
        if not np.all(step_label == 0):
            raise ValueError("first_error_step=-1 requires every step_label to be zero")
        return
    if not np.all(step_label[:first_error_step] == 0):
        raise ValueError("pre-error step_label values must be zero")
    if step_label[first_error_step] != 1:
        raise ValueError("first_error_step must identify the unique event label")
    if not np.all(step_label[first_error_step + 1 :] == -1):
        raise ValueError("post-error step_label values must be -1")


def _validate_sources(source_step_id: np.ndarray, source_mask: np.ndarray) -> None:
    if not np.all(source_step_id[~source_mask] == SOURCE_STEP_PADDING):
        raise ValueError("source_step_id padding must agree with source_mask")
    valid_ids = source_step_id[source_mask]
    if np.any(valid_ids == SOURCE_STEP_PADDING):
        raise ValueError("valid source_step_id entries cannot use padding")
    step_count = source_step_id.shape[0]
    if np.any((valid_ids < -1) | (valid_ids >= step_count)):
        raise ValueError("source_step_id values must be -1 prompt or valid step ids")

    for target_step in range(step_count):
        target_ids = source_step_id[target_step, source_mask[target_step]]
        if target_ids.size == 0:
            raise ValueError("each step must contain at least one valid source block")
        if len(np.unique(target_ids)) != len(target_ids):
            raise ValueError("source_step_id blocks must be unique within each step")
        if int(np.sum(target_ids == -1)) != 1:
            raise ValueError("each step must contain exactly one prompt source block")
        if np.any(target_ids > target_step):
            raise ValueError("source_step_id cannot reference a future step")


def _validate_attention_sum(
    attn_msg: np.ndarray, attn_out: np.ndarray, source_mask: np.ndarray
) -> None:
    padded = ~source_mask[:, None, :, None]
    if np.any(np.where(padded, attn_msg, 0.0) != 0.0):
        raise ValueError("padded attention message blocks must be exactly zero")
    valid_messages = attn_msg * source_mask[:, None, :, None]
    expected = valid_messages.astype(np.float32).sum(axis=2)
    actual = attn_out.astype(np.float32)
    if not np.allclose(actual, expected, rtol=ATTN_SUM_RTOL, atol=ATTN_SUM_ATOL):
        raise ValueError(
            "attn_out_step must approximately equal the sum of valid "
            "attn_msg_resid_by_source blocks"
        )


__all__ = [
    "ATTN_SUM_ATOL",
    "ATTN_SUM_RTOL",
    "COMPONENT_STEP_SCHEMA",
    "SOURCE_STEP_PADDING",
    "ComponentStepArtifact",
    "load_component_artifact",
    "validate_component_artifact",
]
