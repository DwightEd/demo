from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .component_contract import (
    COMPONENT_STEP_SCHEMA,
    SOURCE_STEP_PADDING,
    ComponentStepArtifact,
)
from .component_replay import (
    ComponentExtractionConfig,
    _replay_components,
    _source_spans_for_layout,
)
from .contracts import ChainSample, TraceSource
from .data import load_step_end_states


_SOURCE_REVISION_UNAVAILABLE = "unavailable_in_source_trace"


@dataclass(frozen=True)
class ComponentExtractionResult:
    written: tuple[Path, ...]
    skipped: tuple[Path, ...]


@dataclass(frozen=True)
class TraceReplayRecord:
    trace_path: Path
    record_index: int
    chain_id: int
    input_ids: np.ndarray
    attention_mask: np.ndarray
    prompt_token_count: int
    step_token_ranges: np.ndarray
    first_error_step: int
    dataset: str
    provenance: dict[str, str]


@dataclass(frozen=True)
class ComponentLayout:
    step_token_start: np.ndarray
    step_token_end: np.ndarray
    boundary_token_end: np.ndarray
    step_label: np.ndarray
    source_step_id: np.ndarray
    source_mask: np.ndarray


class _CachedTraceArchive:
    """In-memory view of only the small replay/alignment arrays in a large NPZ."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self._arrays = arrays
        self.files = tuple(arrays)

    def __getitem__(self, name: str) -> np.ndarray:
        return self._arrays[name]


def _cache_trace_archive(path: Path) -> _CachedTraceArchive:
    required = {
        "chain_idx",
        "full_input_ids",
        "full_attention_mask",
        "prompt_token_counts",
        "n_steps",
        "step_token_ranges",
        "gold_error_step",
        "dataset",
    }
    optional = {
        "metadata_json",
        "source_model",
        "model_name",
        "loaded_model",
        "observer_model",
        "source_model_revision",
        "model_revision",
        "revision",
        "source_tokenizer",
        "tokenizer_name",
        "tokenizer",
        "source_tokenizer_revision",
        "tokenizer_revision",
        "tokenizer_revision_id",
    }
    with np.load(path, allow_pickle=True) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"trace.npz is missing required arrays: {sorted(missing)}")
        names = required | optional.intersection(archive.files)
        arrays = {name: np.asarray(archive[name]) for name in names}
    return _CachedTraceArchive(arrays)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_row(archive: np.lib.npyio.NpzFile, name: str, row: int) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"trace.npz is missing required array {name!r}")
    values = np.asarray(archive[name])
    if values.dtype == object and values.ndim == 1:
        result = np.asarray(values[int(row)], dtype=np.int64)
    else:
        if values.shape[0] <= int(row):
            raise ValueError(f"{name} is not record-aligned")
        result = np.asarray(values[int(row)], dtype=np.int64)
    return result.reshape(-1)


def _record_vector(
    archive: np.lib.npyio.NpzFile,
    name: str,
    count: int,
    *,
    required: bool = True,
    default: object = "",
) -> np.ndarray:
    if name not in archive.files:
        if required:
            raise ValueError(f"trace.npz is missing required array {name!r}")
        return np.full(count, default, dtype=object)
    values = np.asarray(archive[name], dtype=object)
    if values.ndim == 0:
        return np.full(count, values.item(), dtype=object)
    values = values.reshape(-1)
    if values.shape != (count,):
        raise ValueError(f"{name} is not record-aligned")
    return values


def _ranges_row(archive: np.lib.npyio.NpzFile, row: int) -> np.ndarray:
    if "step_token_ranges" not in archive.files:
        raise ValueError("trace.npz is missing required array 'step_token_ranges'")
    values = np.asarray(archive["step_token_ranges"], dtype=object)
    if values.shape[0] <= int(row):
        raise ValueError("step_token_ranges is not record-aligned")
    ranges = np.asarray(values[int(row)], dtype=np.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("step_token_ranges row must have shape [step,2]")
    return ranges


def _trim_right_padding(
    input_ids: np.ndarray, attention_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    mask = np.asarray(attention_mask, dtype=np.int64).reshape(-1)
    if ids.shape != mask.shape:
        raise ValueError("full_input_ids and full_attention_mask shapes disagree")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("full_attention_mask must contain only 0/1 values")
    token_count = int(mask.sum())
    expected = np.concatenate(
        [
            np.ones(token_count, dtype=np.int64),
            np.zeros(mask.shape[0] - token_count, dtype=np.int64),
        ]
    )
    if not np.array_equal(mask, expected):
        raise ValueError("full_attention_mask contains interior padding")
    if token_count < 1:
        raise ValueError("replay input cannot be empty")
    return ids[:token_count], mask[:token_count]


def _metadata_json(
    archive: np.lib.npyio.NpzFile, row: int, count: int
) -> dict[str, Any]:
    if "metadata_json" not in archive.files:
        return {}
    raw = _record_vector(archive, "metadata_json", count, required=False)
    text = str(raw[int(row)])
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("metadata_json contains invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise TypeError("metadata_json row must decode to a JSON object")
    return parsed


def _provenance_field(
    archive: np.lib.npyio.NpzFile,
    metadata: dict[str, Any],
    row: int,
    count: int,
    names: Sequence[str],
) -> str:
    for name in names:
        if name in archive.files:
            value = _record_vector(archive, name, count, required=False)[int(row)]
            if str(value):
                return str(value)
    for name in names:
        value = metadata.get(name)
        if value is not None and str(value):
            return str(value)
    return ""


def _normalize_provenance(value: str) -> str:
    return "".join(char for char in str(value).lower() if char.isalnum())


def _model_family(value: object) -> str:
    normalized = _normalize_provenance(str(value))
    if "llama31" in normalized:
        for size in ("405b", "70b", "8b"):
            if size in normalized:
                return f"llama31-{size}"
    return normalized


def _dataset_id(value: object) -> str:
    normalized = _normalize_provenance(str(value))
    known = ("olympiadbench", "omnimath", "gsm8k", "math")
    if normalized in known:
        return normalized
    tokens = [
        _normalize_provenance(token) for token in re.split(r"[^a-zA-Z0-9]+", str(value))
    ]
    matches = [name for name in known if name in tokens]
    if len(matches) == 1:
        return matches[0]
    return normalized


def _provenance_matches(expected: str, actual: str) -> bool:
    if not actual:
        return False
    if expected == actual:
        return True
    expected_norm = _normalize_provenance(expected)
    actual_norm = _normalize_provenance(actual)
    return bool(
        expected_norm
        and actual_norm
        and (
            expected_norm == actual_norm
            or actual_norm in expected_norm
            or expected_norm in actual_norm
            or _model_family(expected) == _model_family(actual)
        )
    )


def _validate_provenance(
    provenance: dict[str, str],
    config: ComponentExtractionConfig,
    chain_id: int,
) -> None:
    expected = {
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "tokenizer_name": config.tokenizer_name,
        "tokenizer_revision": config.tokenizer_revision,
    }
    for key, value in expected.items():
        actual = provenance.get(key, "")
        if key.endswith("_revision") and value == "auto":
            if not actual:
                raise ValueError(f"chain {chain_id}: trace provenance is missing {key}")
            continue
        if not _provenance_matches(value, actual):
            raise ValueError(
                f"chain {chain_id}: replay provenance mismatch for {key}: "
                f"trace={actual!r}, replay={value!r}"
            )


def _load_trace_replay_record(
    archive: object,
    path: Path,
    sample: ChainSample,
    config: ComponentExtractionConfig | None = None,
) -> TraceReplayRecord:
    required = {
        "chain_idx",
        "full_input_ids",
        "full_attention_mask",
        "prompt_token_counts",
        "n_steps",
        "step_token_ranges",
        "gold_error_step",
        "dataset",
    }
    missing = required.difference(archive.files)
    if missing:
        raise ValueError(f"trace.npz is missing required arrays: {sorted(missing)}")
    chain_ids = np.asarray(archive["chain_idx"], dtype=np.int64).reshape(-1)
    matches = np.flatnonzero(chain_ids == int(sample.chain_id))
    if matches.size != 1:
        raise ValueError(
            f"chain {sample.chain_id}: trace.npz must contain exactly one row"
        )
    row = int(matches[0])
    input_ids, attention_mask = _trim_right_padding(
        _integer_row(archive, "full_input_ids", row),
        _integer_row(archive, "full_attention_mask", row),
    )
    count = len(chain_ids)
    prompt_counts = _record_vector(archive, "prompt_token_counts", count)
    step_counts = _record_vector(archive, "n_steps", count)
    gold = _record_vector(archive, "gold_error_step", count)
    datasets = _record_vector(archive, "dataset", count)
    padded_ranges = _ranges_row(archive, row)
    step_count = int(step_counts[row])
    if step_count < 1 or step_count > padded_ranges.shape[0]:
        raise ValueError(
            f"chain {sample.chain_id}: n_steps={step_count} is incompatible with "
            f"step_token_ranges rows={padded_ranges.shape[0]}"
        )
    ranges = padded_ranges[:step_count]
    metadata = _metadata_json(archive, row, count)
    provenance = {
        "model_name": _provenance_field(
            archive,
            metadata,
            row,
            count,
            ("source_model", "model_name", "loaded_model", "observer_model"),
        ),
        "model_revision": (
            _provenance_field(
                archive,
                metadata,
                row,
                count,
                ("source_model_revision", "model_revision", "revision"),
            )
            or _SOURCE_REVISION_UNAVAILABLE
        ),
        "tokenizer_name": _provenance_field(
            archive,
            metadata,
            row,
            count,
            ("source_tokenizer", "tokenizer_name", "tokenizer"),
        ),
        "tokenizer_revision": (
            _provenance_field(
                archive,
                metadata,
                row,
                count,
                (
                    "source_tokenizer_revision",
                    "tokenizer_revision",
                    "tokenizer_revision_id",
                ),
            )
            or _SOURCE_REVISION_UNAVAILABLE
        ),
    }

    prompt_count = int(prompt_counts[row])
    if prompt_count != int(sample.response_start):
        raise ValueError(
            f"chain {sample.chain_id}: prompt token count disagrees with ChainSample"
        )
    if int(gold[row]) != int(sample.first_error_step):
        raise ValueError(f"chain {sample.chain_id}: first-error label disagrees")
    if _dataset_id(datasets[row]) != _dataset_id(sample.dataset):
        raise ValueError(f"chain {sample.chain_id}: dataset provenance disagrees")
    expected_ranges = np.asarray(sample.step_ranges, dtype=np.int64)
    if step_count != expected_ranges.shape[0]:
        raise ValueError(
            f"chain {sample.chain_id}: n_steps disagrees with ChainSample: "
            f"trace={step_count}, sample={expected_ranges.shape[0]}"
        )
    if ranges.shape != expected_ranges.shape or not np.array_equal(
        ranges, expected_ranges
    ):
        raise ValueError(f"chain {sample.chain_id}: step_token_ranges disagree")
    expected_tokens = int(expected_ranges[-1, 1]) + 1
    if input_ids.shape[0] != expected_tokens:
        raise ValueError(
            f"chain {sample.chain_id}: replay token count disagrees with step ranges"
        )
    if sample.state_count >= 0 and input_ids.shape[0] - prompt_count != int(
        sample.state_count
    ):
        raise ValueError(
            f"chain {sample.chain_id}: replay response token count disagrees with shard"
        )
    if config is not None:
        _validate_provenance(provenance, config, int(sample.chain_id))
    return TraceReplayRecord(
        trace_path=path,
        record_index=row,
        chain_id=int(sample.chain_id),
        input_ids=input_ids.astype(np.int64, copy=False),
        attention_mask=attention_mask.astype(np.int64, copy=False),
        prompt_token_count=prompt_count,
        step_token_ranges=ranges,
        first_error_step=int(gold[row]),
        dataset=str(datasets[row]),
        provenance=provenance,
    )


def load_trace_replay_record(
    trace_path: str | Path,
    sample: ChainSample,
    config: ComponentExtractionConfig | None = None,
) -> TraceReplayRecord:
    """Load one selected chain's stored teacher-forcing replay tokens by chain_idx."""
    path = Path(trace_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    archive = _cache_trace_archive(path)
    return _load_trace_replay_record(archive, path, sample, config)


def _build_component_layout(sample: ChainSample) -> ComponentLayout:
    ranges = np.asarray(sample.step_ranges, dtype=np.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2 or ranges.shape[0] < 1:
        raise ValueError(f"chain {sample.chain_id}: invalid step ranges")
    if np.any(ranges < 0) or np.any(ranges[:, 1] < ranges[:, 0]):
        raise ValueError(f"chain {sample.chain_id}: invalid step token range values")
    step_token_start = ranges[:, 0].astype(np.int32)
    step_token_end = (ranges[:, 1] + 1).astype(np.int32)
    boundary_token_end = np.concatenate(
        [
            np.asarray([int(sample.response_start)], dtype=np.int32),
            step_token_end,
        ]
    )
    step_count = ranges.shape[0]
    if int(sample.first_error_step) == -1:
        step_label = np.zeros(step_count, dtype=np.int8)
    else:
        if sample.first_error_step < 0 or sample.first_error_step >= step_count:
            raise ValueError(f"chain {sample.chain_id}: first-error label is invalid")
        step_label = np.full(step_count, -1, dtype=np.int8)
        step_label[: int(sample.first_error_step)] = 0
        step_label[int(sample.first_error_step)] = 1
    source_step_id = np.full(
        (step_count, step_count + 1), SOURCE_STEP_PADDING, dtype=np.int16
    )
    source_mask = np.zeros((step_count, step_count + 1), dtype=bool)
    for step in range(step_count):
        ids = np.asarray([-1, *range(step + 1)], dtype=np.int16)
        source_step_id[step, : ids.shape[0]] = ids
        source_mask[step, : ids.shape[0]] = True
    return ComponentLayout(
        step_token_start=step_token_start,
        step_token_end=step_token_end,
        boundary_token_end=boundary_token_end,
        step_label=step_label,
        source_step_id=source_step_id,
        source_mask=source_mask,
    )


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    difference = np.asarray(actual, dtype=np.float32) - np.asarray(
        expected, dtype=np.float32
    )
    numerator = np.linalg.norm(difference, axis=-1)
    denominator = np.maximum(
        np.linalg.norm(np.asarray(expected, dtype=np.float32), axis=-1),
        1e-12,
    )
    return numerator / denominator


def validate_replay_fidelity(
    sample: ChainSample,
    *,
    layers: np.ndarray,
    replayed_step_states: np.ndarray,
    max_relative_error: float,
) -> float:
    requested = np.asarray(layers, dtype=np.int64).reshape(-1)
    available = np.asarray(sample.layer_ids, dtype=np.int64).reshape(-1)
    missing = sorted(set(requested.tolist()).difference(available.tolist()))
    if missing:
        raise ValueError(
            f"chain {sample.chain_id}: component layers are absent from hidden shard: "
            f"{missing}"
        )
    positions = np.asarray(
        [int(np.where(available == layer)[0][0]) for layer in requested],
        dtype=np.int64,
    )
    replayed = np.asarray(replayed_step_states, dtype=np.float32)
    expected_shape = (sample.n_steps, requested.shape[0])
    if replayed.ndim != 3 or replayed.shape[:2] != expected_shape:
        raise ValueError("replayed_step_states must have shape [step,layer,hidden]")
    stored = load_step_end_states(sample)[:, positions, :]
    if stored.shape != replayed.shape:
        raise ValueError(
            f"chain {sample.chain_id}: stored hidden shard shape disagrees with replay"
        )
    errors = _relative_error(replayed, stored)
    maximum = float(np.max(errors))
    if maximum > float(max_relative_error):
        raise ValueError(
            f"chain {sample.chain_id}: replay fidelity relative error {maximum:.6g} "
            f"exceeds {float(max_relative_error):.6g}"
        )
    return maximum


def _metadata_for_artifact(
    *,
    sample: ChainSample,
    record: TraceReplayRecord,
    config: ComponentExtractionConfig,
    source_trace_sha256: str,
    max_attention_reconstruction_error: float,
    max_replay_relative_error: float,
) -> dict[str, Any]:
    model_revision = (
        record.provenance["model_revision"]
        if config.model_revision == "auto"
        else config.model_revision
    )
    tokenizer_revision = (
        record.provenance["tokenizer_revision"]
        if config.tokenizer_revision == "auto"
        else config.tokenizer_revision
    )
    return {
        "schema": COMPONENT_STEP_SCHEMA,
        "sample_id": f"chain-{sample.chain_id}",
        "chain_id": f"chain-{sample.chain_id}",
        "dataset": sample.dataset,
        "model_name": config.model_name,
        "model_revision": model_revision,
        "tokenizer_name": config.tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "extractor_commit": config.extractor_commit,
        "source_trace_sha256": source_trace_sha256,
        "generation_config_sha256": config.replay_config_sha256(),
        "source_trace_path": str(record.trace_path),
        "source_trace_record_index": int(record.record_index),
        "prompt_token_count": int(record.prompt_token_count),
        "token_count": int(record.input_ids.shape[0]),
        "selected_layers": [int(layer) for layer in config.layers],
        "extraction_mode": "teacher_forced_backbone_attention_components_v1",
        "model_identity_verification": (
            "selected_layer_step_end_activation_replay_fidelity"
        ),
        "tokenizer_identity_verification": "stored_input_ids_without_retokenization",
        "attention_reconstruction_atol": float(config.attention_reconstruction_atol),
        "attention_reconstruction_rtol": float(config.attention_reconstruction_rtol),
        "max_attention_reconstruction_relative_error": float(
            max_attention_reconstruction_error
        ),
        "replay_fidelity_max_relative_error": float(max_replay_relative_error),
        "replay_fidelity_rtol": float(config.replay_fidelity_rtol),
    }


def _artifact_for_replay(
    *,
    sample: ChainSample,
    record: TraceReplayRecord,
    config: ComponentExtractionConfig,
    source_trace_sha256: str,
    resid_boundary: np.ndarray,
    attn_msg_resid_by_source: np.ndarray,
    attn_out_step: np.ndarray,
    mlp_out_step: np.ndarray,
    max_attention_reconstruction_error: float,
    max_replay_relative_error: float,
) -> ComponentStepArtifact:
    layout = _build_component_layout(sample)
    layers = np.asarray(config.layers, dtype=np.int16)
    return ComponentStepArtifact(
        input_ids=np.asarray(record.input_ids, dtype=np.int32),
        step_token_start=layout.step_token_start,
        step_token_end=layout.step_token_end,
        boundary_token_end=layout.boundary_token_end,
        first_error_step=int(sample.first_error_step),
        step_label=layout.step_label,
        source_step_id=layout.source_step_id,
        source_mask=layout.source_mask,
        resid_boundary=np.asarray(resid_boundary, dtype=np.float16),
        attn_msg_resid_by_source=np.asarray(attn_msg_resid_by_source, dtype=np.float16),
        attn_out_step=np.asarray(attn_out_step, dtype=np.float16),
        mlp_out_step=np.asarray(mlp_out_step, dtype=np.float16),
        residual_layers=layers,
        attention_layers=layers,
        mlp_layers=layers,
        metadata=_metadata_for_artifact(
            sample=sample,
            record=record,
            config=config,
            source_trace_sha256=source_trace_sha256,
            max_attention_reconstruction_error=max_attention_reconstruction_error,
            max_replay_relative_error=max_replay_relative_error,
        ),
    )


def _artifact_path(sample: ChainSample, source: TraceSource) -> Path:
    if sample.component_path is not None:
        return Path(sample.component_path)
    if source.component_dir is None:
        raise ValueError(
            f"chain {sample.chain_id}: TraceSource.component_dir is required"
        )
    return Path(source.component_dir) / f"chain_{sample.chain_id}.component_step_v1.npz"


def _validate_existing_artifact(
    *,
    path: Path,
    sample: ChainSample,
    record: TraceReplayRecord,
    config: ComponentExtractionConfig,
    source_trace_sha256: str,
) -> None:
    artifact = ComponentStepArtifact.load(path)
    layout = _build_component_layout(sample)
    expected_layers = np.asarray(config.layers, dtype=np.int16)
    checks = {
        "input_ids": np.array_equal(artifact.input_ids, record.input_ids),
        "step_token_start": np.array_equal(
            artifact.step_token_start, layout.step_token_start
        ),
        "step_token_end": np.array_equal(
            artifact.step_token_end, layout.step_token_end
        ),
        "boundary_token_end": np.array_equal(
            artifact.boundary_token_end, layout.boundary_token_end
        ),
        "step_label": np.array_equal(artifact.step_label, layout.step_label),
        "source_step_id": np.array_equal(
            artifact.source_step_id, layout.source_step_id
        ),
        "source_mask": np.array_equal(artifact.source_mask, layout.source_mask),
        "residual_layers": np.array_equal(artifact.residual_layers, expected_layers),
        "attention_layers": np.array_equal(artifact.attention_layers, expected_layers),
        "mlp_layers": np.array_equal(artifact.mlp_layers, expected_layers),
    }
    failed = [name for name, ok in checks.items() if not ok]
    metadata = artifact.metadata
    expected_metadata = {
        "dataset": sample.dataset,
        "model_name": config.model_name,
        "model_revision": (
            record.provenance["model_revision"]
            if config.model_revision == "auto"
            else config.model_revision
        ),
        "tokenizer_name": config.tokenizer_name,
        "tokenizer_revision": (
            record.provenance["tokenizer_revision"]
            if config.tokenizer_revision == "auto"
            else config.tokenizer_revision
        ),
        "extractor_commit": config.extractor_commit,
        "source_trace_sha256": source_trace_sha256,
        "generation_config_sha256": config.replay_config_sha256(),
    }
    for key, value in expected_metadata.items():
        if str(metadata.get(key, "")) != str(value):
            failed.append(f"metadata.{key}")
    if metadata.get("sample_id") not in {
        str(sample.chain_id),
        f"chain-{sample.chain_id}",
        f"chain_{sample.chain_id}",
    }:
        failed.append("metadata.sample_id")
    if int(artifact.first_error_step) != int(sample.first_error_step):
        failed.append("first_error_step")
    if failed:
        raise ValueError(
            f"{path}: existing component artifact is not aligned: {', '.join(failed)}"
        )


def _atomic_save_artifact(artifact: ComponentStepArtifact, path: Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp.npz")
    try:
        artifact.save(temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


class ComponentTraceExtractor:
    """Replay stored token IDs and save step-aligned component artifacts."""

    def __init__(self, config: ComponentExtractionConfig) -> None:
        if not isinstance(config, ComponentExtractionConfig):
            raise TypeError("config must be a ComponentExtractionConfig")
        self.config = config

    def source_trace_sha256(self, trace_path: str | Path) -> str:
        return _sha256(Path(trace_path).expanduser().resolve())

    def extract(
        self,
        *,
        model: object,
        samples: Iterable[ChainSample],
        sources: Iterable[TraceSource],
        progress: object | None = None,
    ) -> ComponentExtractionResult:
        source_by_dataset = {source.dataset: source for source in sources}
        sample_list = tuple(samples)
        selected_datasets = {sample.dataset for sample in sample_list}
        source_material: dict[
            str, tuple[TraceSource, Path, str, _CachedTraceArchive]
        ] = {}
        for dataset in sorted(selected_datasets):
            if dataset not in source_by_dataset:
                raise ValueError(f"{dataset}: no TraceSource was provided")
            source = source_by_dataset[dataset]
            trace_path = (
                Path(source.exact_trace or source.manifest.parent / "trace.npz")
                .expanduser()
                .resolve()
            )
            source_material[dataset] = (
                source,
                trace_path,
                self.source_trace_sha256(trace_path),
                _cache_trace_archive(trace_path),
            )
        written: list[Path] = []
        skipped: list[Path] = []
        iterable = sample_list
        if progress is not None:
            iterable = progress.track(
                sample_list,
                total=len(sample_list),
                description="component traces",
            )
        for sample in iterable:
            source, trace_path, source_hash, archive = source_material[sample.dataset]
            record = _load_trace_replay_record(archive, trace_path, sample, self.config)
            output_path = _artifact_path(sample, source)
            if output_path.exists() and not self.config.overwrite:
                _validate_existing_artifact(
                    path=output_path,
                    sample=sample,
                    record=record,
                    config=self.config,
                    source_trace_sha256=source_hash,
                )
                skipped.append(output_path)
                continue
            layout = _build_component_layout(sample)
            (
                resid_boundary,
                attn_msg_resid_by_source,
                attn_out_step,
                mlp_out_step,
                attention_error,
            ) = _replay_components(
                model,
                input_ids=record.input_ids,
                attention_mask=record.attention_mask,
                boundary_token_end=layout.boundary_token_end,
                step_token_end=layout.step_token_end,
                source_spans=_source_spans_for_layout(
                    layout.boundary_token_end, len(layout.step_token_start)
                ),
                config=self.config,
            )
            replay_error = validate_replay_fidelity(
                sample,
                layers=np.asarray(self.config.layers, dtype=np.int64),
                replayed_step_states=resid_boundary[1:],
                max_relative_error=self.config.replay_fidelity_rtol,
            )
            artifact = _artifact_for_replay(
                sample=sample,
                record=record,
                config=self.config,
                source_trace_sha256=source_hash,
                resid_boundary=resid_boundary,
                attn_msg_resid_by_source=attn_msg_resid_by_source,
                attn_out_step=attn_out_step,
                mlp_out_step=mlp_out_step,
                max_attention_reconstruction_error=attention_error,
                max_replay_relative_error=replay_error,
            )
            _atomic_save_artifact(artifact, output_path)
            written.append(output_path)
        return ComponentExtractionResult(written=tuple(written), skipped=tuple(skipped))


__all__ = [
    "ComponentExtractionConfig",
    "ComponentExtractionResult",
    "ComponentTraceExtractor",
    "TraceReplayRecord",
    "load_trace_replay_record",
    "validate_replay_fidelity",
]
