from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from ..raw_residual import _resolve_source
from .contracts import (
    ACQUISITION_MODES,
    ChainSample,
    HiddenGeometryDataset,
    OutputEvidence,
    TraceSource,
)


@dataclass(frozen=True)
class _OutputRow:
    values: np.ndarray
    n_steps: int
    gold_error_step: int
    step_ranges: np.ndarray
    generator: str
    dataset: str
    observer_model: str


def _normalized(value: object) -> str:
    return "".join(char for char in str(value).lower() if char.isalnum())


def _dataset_id(value: object) -> str:
    normalized = _normalized(value)
    known = ("olympiadbench", "omnimath", "gsm8k", "math")
    if normalized in known:
        return normalized
    tokens = [_normalized(token) for token in re.split(r"[^a-zA-Z0-9]+", str(value))]
    matches = [name for name in known if name in tokens]
    if len(matches) == 1:
        return matches[0]
    return normalized


def _model_family(value: object) -> str:
    normalized = _normalized(value)
    if "llama31" in normalized:
        for size in ("405b", "70b", "8b"):
            if size in normalized:
                return f"llama31-{size}"
    return normalized


def _model_id(value: object) -> str:
    family = _model_family(value)
    normalized = _normalized(value)
    return f"{family}-instruct" if "instruct" in normalized else family


def _record_vector(
    archive: np.lib.npyio.NpzFile, name: str, count: int, *, default: object = ""
) -> np.ndarray:
    if name not in archive.files:
        return np.full(count, default, dtype=object)
    values = np.asarray(archive[name], dtype=object)
    if values.ndim == 0:
        return np.full(count, values.item(), dtype=object)
    values = values.reshape(-1)
    if values.shape != (count,):
        raise ValueError(f"{name} is not record-aligned")
    return values


def _observer_models(archive: np.lib.npyio.NpzFile, count: int) -> np.ndarray:
    for name in ("loaded_model", "observer_model"):
        direct = _record_vector(archive, name, count)
        if np.any(direct != ""):
            return direct
    metadata = _record_vector(archive, "metadata_json", count)
    models = []
    for value in metadata:
        try:
            item = json.loads(str(value)) if str(value) else {}
        except json.JSONDecodeError as exc:
            raise ValueError("metadata_json contains invalid JSON") from exc
        models.append(item.get("loaded_model") or item.get("observer_model") or "")
    return np.asarray(models, dtype=object)


def _generators(archive: np.lib.npyio.NpzFile, count: int) -> np.ndarray:
    for name in ("response_generator", "generator"):
        direct = _record_vector(archive, name, count)
        if np.any(direct != ""):
            return direct
    metadata = _record_vector(archive, "metadata_json", count)
    values = []
    for value in metadata:
        try:
            item = json.loads(str(value)) if str(value) else {}
        except json.JSONDecodeError as exc:
            raise ValueError("metadata_json contains invalid JSON") from exc
        values.append(item.get("response_generator") or item.get("generator") or "")
    return np.asarray(values, dtype=object)


def _output_path(source: TraceSource, manifest_files: set[str]) -> Path:
    if "step_scores" in manifest_files:
        return source.manifest.resolve()
    path = source.exact_trace or source.manifest.parent / "trace.npz"
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"output summaries are absent from {source.manifest}; aligned trace not found: {path}"
        )
    return path


def _load_output_rows(
    path: Path, requested: tuple[str, ...]
) -> tuple[dict[int, _OutputRow], bool]:
    with np.load(path, allow_pickle=True) as archive:
        required = {
            "chain_idx",
            "step_scores",
            "step_score_names",
            "step_token_ranges",
            "dataset",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"output summaries are missing required arrays: {sorted(missing)}")
        chain_ids = np.asarray(archive["chain_idx"], dtype=np.int64).reshape(-1)
        if len(np.unique(chain_ids)) != len(chain_ids):
            raise ValueError("output trace chain_idx is not unique")
        scores = np.asarray(archive["step_scores"], dtype=np.float32)
        names = tuple(str(value) for value in archive["step_score_names"])
        unknown = sorted(set(requested).difference(names))
        if unknown:
            raise ValueError(
                f"requested output summaries are not stored: {unknown}; available={list(names)}"
            )
        if scores.ndim != 3 or scores.shape[0] != len(chain_ids):
            raise ValueError("step_scores must be [chain,step,feature]")
        indices = np.asarray([names.index(name) for name in requested], dtype=np.int64)
        n_steps = _record_vector(archive, "n_steps", len(chain_ids), default=scores.shape[1])
        gold = _record_vector(archive, "gold_error_step", len(chain_ids), default=-2)
        ranges = np.asarray(archive["step_token_ranges"], dtype=object)
        if len(ranges) != len(chain_ids):
            raise ValueError("output step_token_ranges is not record-aligned")
        generators = _generators(archive, len(chain_ids))
        datasets = _record_vector(archive, "dataset", len(chain_ids))
        observers = _observer_models(archive, len(chain_ids))
        if np.any(generators == "") or np.any(datasets == "") or np.any(observers == ""):
            raise ValueError("output summaries lack generator/dataset/observer provenance")
        if "is_correct" in archive.files:
            correct = np.asarray(archive["is_correct"], dtype=np.int8).reshape(-1)
            gold_values = np.asarray(gold, dtype=np.int64)
            if correct.shape != (len(chain_ids),):
                raise ValueError("output is_correct is not record-aligned")
            known = np.isin(correct, (0, 1)) & (gold_values != -2)
            if not np.array_equal(
                correct[known].astype(bool), gold_values[known] == -1
            ):
                raise ValueError("output is_correct disagrees with gold_error_step")
        rows = {
            int(chain_id): _OutputRow(
                values=np.asarray(
                    scores[row, : int(n_steps[row])][:, indices], dtype=np.float32
                ),
                n_steps=int(n_steps[row]),
                gold_error_step=int(gold[row]),
                step_ranges=np.asarray(ranges[row], dtype=np.int64)[: int(n_steps[row])],
                generator=str(generators[row]),
                dataset=str(datasets[row]),
                observer_model=str(observers[row]),
            )
            for row, chain_id in enumerate(chain_ids)
        }
        full_logits = bool(
            {"logits", "full_vocab_logits", "response_token_logits"} & set(archive.files)
        )
    return rows, full_logits


def load_hidden_geometry_dataset(
    sources: Iterable[TraceSource],
    *,
    response_generator: str,
    observer_model: str,
    output_features: Iterable[str] = ("token_entropy", "token_nll"),
) -> HiddenGeometryDataset:
    """Load verified Llama raw states and aligned logits-derived step summaries."""
    requested = tuple(str(name) for name in output_features)
    if not requested:
        raise ValueError("at least one output summary is required")
    samples: list[ChainSample] = []
    labels: list[int] = []
    manifest_total = 0
    any_full_logits = False
    wanted_observer = _model_family(observer_model)
    source_list = tuple(sources)
    datasets = [source.dataset for source in source_list]
    if len({_normalized(value) for value in datasets}) != len(datasets):
        raise ValueError("trace source dataset names must be unique")
    modes = {source.acquisition_mode for source in source_list}
    if len(modes) != 1:
        raise ValueError("all trace sources must use the same acquisition mode")
    acquisition_mode = next(iter(modes))
    wanted_generator = _model_family(response_generator)
    for specification in source_list:
        raw = _resolve_source(
            specification.manifest,
            specification.hidden_dir,
            response_generator,
        )
        manifest_total += raw.n_manifest_records
        with np.load(raw.manifest_path, allow_pickle=True) as archive:
            files = set(archive.files)
            if not {"problem_group_id", "problem_ids"}.intersection(files):
                raise ValueError("raw manifest requires an explicit problem-group field")
            chain_ids = _record_vector(archive, "chain_idx", raw.n_manifest_records)
            if np.all(chain_ids == ""):
                chain_ids = np.arange(raw.n_manifest_records)
            if len(np.unique(chain_ids)) != len(chain_ids):
                raise ValueError(f"{specification.dataset}: manifest chain_idx must be unique")
            n_steps = _record_vector(archive, "n_steps", raw.n_manifest_records, default=-1)
            observers = _observer_models(archive, raw.n_manifest_records)
            manifest_datasets = _record_vector(
                archive, "dataset", raw.n_manifest_records
            )
            if np.any(manifest_datasets == ""):
                raise ValueError("raw manifest lacks record-aligned dataset provenance")
            selected_groups = np.asarray(raw.problem_ids, dtype=object).astype(str)
            if np.any(np.char.strip(selected_groups) == ""):
                raise ValueError("problem-group identifiers cannot be empty")
            selected_hashes = np.asarray(
                [
                    value if value.startswith("problem_sha256:") else None
                    for value in selected_groups
                ],
                dtype=object,
            )
        output_rows, full_logits = _load_output_rows(
            _output_path(specification, files), requested
        )
        any_full_logits |= full_logits
        for local_row, manifest_row in enumerate(raw.manifest_rows):
            manifest_row = int(manifest_row)
            chain_id = int(chain_ids[manifest_row])
            observed_by = str(observers[manifest_row])
            if not wanted_observer or _model_family(observed_by) != wanted_observer:
                raise ValueError(
                    f"chain {chain_id}: observer {observed_by!r} does not match {observer_model!r}"
                )
            if chain_id not in output_rows:
                raise ValueError(f"chain {chain_id} is missing from aligned output summaries")
            output_row = output_rows[chain_id]
            output = output_row.values
            output_count = output_row.n_steps
            output_gold = output_row.gold_error_step
            count = int(n_steps[manifest_row])
            count = output_count if count < 0 else count
            ranges = np.asarray(raw.step_ranges[local_row][:count], dtype=np.int64)
            if ranges.shape != (count, 2) or np.any(ranges < 0):
                raise ValueError(f"chain {chain_id}: invalid unpadded step ranges")
            if output_row.step_ranges.shape != ranges.shape or not np.array_equal(
                output_row.step_ranges, ranges
            ):
                raise ValueError(f"chain {chain_id}: step ranges disagree across artifacts")
            gold = int(raw.gold_error_step[local_row])
            if gold == -2:
                continue
            if gold < -2:
                raise ValueError(f"chain {chain_id}: unsupported first-error label {gold}")
            if gold >= count:
                raise ValueError(f"chain {chain_id}: first-error label exceeds n_steps")
            if output_gold != -2 and output_gold != gold:
                raise ValueError(f"chain {chain_id}: first-error label disagrees across artifacts")
            if output.shape != (count, len(requested)):
                raise ValueError(f"chain {chain_id}: output summaries disagree with n_steps")
            raw_generator = str(raw.response_generators[local_row])
            generator_ids = {_model_id(raw_generator), _model_id(output_row.generator)}
            if (
                _model_family(raw_generator) != wanted_generator
                or _model_family(output_row.generator) != wanted_generator
                or len(generator_ids) != 1
            ):
                raise ValueError(
                    f"chain {chain_id}: generator provenance disagrees: "
                    f"raw={raw_generator!r}, output={output_row.generator!r}"
                )
            expected_dataset = _dataset_id(specification.dataset)
            if expected_dataset != _dataset_id(
                manifest_datasets[manifest_row]
            ) or expected_dataset != _dataset_id(output_row.dataset):
                raise ValueError(f"chain {chain_id}: dataset provenance disagrees")
            if (
                _model_family(output_row.observer_model) != wanted_observer
                or _model_id(output_row.observer_model) != _model_id(observed_by)
            ):
                raise ValueError(f"chain {chain_id}: output observer provenance disagrees")
            state_count = (
                int(raw.counts[local_row]) if raw.counts is not None else -1
            )
            samples.append(
                ChainSample(
                    chain_id=chain_id,
                    manifest_row=manifest_row,
                    problem_group=str(raw.problem_ids[local_row]),
                    dataset=specification.dataset,
                    generator=raw_generator,
                    observer_model=observed_by,
                    state_path=Path(str(raw.files[local_row])),
                    state_count=state_count,
                    response_start=int(raw.response_starts[local_row]),
                    step_ranges=ranges,
                    layer_ids=np.asarray(raw.layers, dtype=np.int64),
                    output_steps=output,
                    output_feature_names=requested,
                    first_error_step=gold,
                    problem_hash=selected_hashes[local_row],
                )
            )
            labels.append(int(gold >= 0))
    return HiddenGeometryDataset(
        samples=tuple(samples),
        labels=np.asarray(labels, dtype=np.int8),
        evidence=OutputEvidence(
            acquisition_mode=acquisition_mode,
            acquisition_mode_source="explicit_trace_source_configuration",
            hidden_evidence_kind=f"{acquisition_mode}_raw_residual_stream",
            output_evidence_kind=(
                "teacher_forced_step_summary"
                if acquisition_mode == "observer_teacher_forcing_replay"
                else "online_generation_step_summary"
            ),
            output_feature_names=requested,
            full_vocab_logits_stored=any_full_logits,
            full_vocab_logits_used=False,
            generation_matched_online_states=ACQUISITION_MODES[acquisition_mode],
            manifest_records=int(manifest_total),
            selected_records=len(samples),
        ),
    )


def load_step_end_states(sample: ChainSample, visible_steps: int | None = None) -> np.ndarray:
    """Read one real shard and select the hidden state after each completed step."""
    if not sample.state_path.is_file():
        raise FileNotFoundError(sample.state_path)
    shard = np.load(sample.state_path, mmap_mode="r", allow_pickle=False)
    if shard.ndim != 3 or shard.shape[1] != len(sample.layer_ids):
        raise ValueError(f"{sample.state_path}: expected [token,layer,hidden], got {shard.shape}")
    if sample.state_count >= 0 and shard.shape[0] != sample.state_count:
        raise ValueError(f"{sample.state_path}: token count disagrees with manifest")
    count = sample.n_steps if visible_steps is None else int(visible_steps)
    if count < 1 or count > sample.n_steps:
        raise ValueError(f"visible_steps must lie in [1, {sample.n_steps}]")
    indices = sample.step_ranges[:count, 1] - int(sample.response_start)
    if np.any(indices < 0) or np.any(indices >= shard.shape[0]):
        raise ValueError(f"chain {sample.chain_id}: step ends exceed response-state shard")
    states = np.asarray(shard[indices], dtype=np.float32)
    if not np.isfinite(states).all():
        raise ValueError(f"chain {sample.chain_id}: hidden states contain non-finite values")
    return states
