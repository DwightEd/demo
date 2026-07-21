from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .layer_time import LayerTimeDataset


@dataclass(frozen=True)
class _RawSource:
    manifest_path: Path
    source_format: str
    files: np.ndarray
    layers: np.ndarray
    gold_error_step: np.ndarray
    problem_ids: np.ndarray
    problem_group_field: str
    step_ranges: list[np.ndarray]
    response_starts: np.ndarray
    snapshot_kind: str
    counts: np.ndarray | None
    manifest_rows: np.ndarray
    n_manifest_records: int
    response_generators: np.ndarray | None
    generator_field: str
    generator_filter: str | None


@dataclass(frozen=True)
class _Match:
    error_row: int
    control_row: int
    error_step: int
    control_step: int


def _scalar(archive: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in archive.files:
        return default
    value = np.asarray(archive[key])
    return value.item() if value.ndim == 0 else value


def _ranges(value: Any) -> np.ndarray:
    array = np.asarray(value.tolist() if np.asarray(value).dtype == object else value, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 1:
        raise ValueError(f"step_token_ranges must contain [step,2], got {array.shape}")
    if np.any(array[:, 1] < array[:, 0]):
        raise ValueError("step_token_ranges contains an end before its start")
    return array


def _normalized_model_name(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _response_generators(
    archive: np.lib.npyio.NpzFile, n_records: int
) -> tuple[np.ndarray | None, str]:
    for field in ("response_generator", "generator"):
        if field in archive.files:
            values = np.asarray(archive[field], dtype=object).reshape(-1)
            if values.shape != (n_records,):
                raise ValueError(f"{field} is not record-aligned")
            return values, field
    if "metadata_json" in archive.files:
        metadata = np.asarray(archive["metadata_json"], dtype=object).reshape(-1)
        if metadata.shape != (n_records,):
            raise ValueError("metadata_json is not record-aligned")
        values = np.asarray(
            [str(json.loads(str(item)).get("response_generator", "")) for item in metadata],
            dtype=object,
        )
        if np.any(values != ""):
            return values, "metadata_json.response_generator"
    return None, "missing"


def _resolve_source(
    path: str | Path,
    hidden_dir: str | Path | None,
    response_generator: str | None = None,
) -> _RawSource:
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with np.load(manifest, allow_pickle=True) as archive:
        if "gold_error_step_kept" in archive.files:
            gold = np.asarray(archive["gold_error_step_kept"], dtype=np.int64)
        elif "gold_error_step" in archive.files:
            gold = np.asarray(archive["gold_error_step"], dtype=np.int64)
        else:
            raise ValueError("raw first-error analysis requires gold_error_step")
        if "step_token_ranges" not in archive.files:
            raise ValueError("raw first-error analysis requires step_token_ranges")
        step_ranges = [_ranges(value) for value in archive["step_token_ranges"]]
        n_records = len(step_ranges)
        if gold.shape != (n_records,):
            raise ValueError("gold_error_step and step_token_ranges lengths disagree")
        if "problem_group_id" in archive.files:
            problem_group_field = "problem_group_id"
            problem_ids = np.asarray(archive[problem_group_field])
        elif "problem_ids" in archive.files:
            problem_group_field = "problem_ids"
            problem_ids = np.asarray(archive[problem_group_field])
        else:
            problem_group_field = "row_index_fallback"
            problem_ids = np.arange(n_records)

        if "response_token_state_files" in archive.files:
            source_format = "exact_response_state_manifest_v1"
            files = np.asarray(archive["response_token_state_files"], dtype=object).reshape(-1)
            if "response_token_state_layers" not in archive.files:
                raise ValueError("exact manifest is missing response_token_state_layers")
            layers = np.asarray(archive["response_token_state_layers"], dtype=np.int64).reshape(-1)
            storage = str(_scalar(archive, "response_token_state_storage_kind", "per_chain_npy_shards_v1"))
            if storage != "per_chain_npy_shards_v1":
                raise ValueError(f"unsupported response state storage: {storage}")
            snapshot_kind = str(_scalar(archive, "response_token_state_snapshot_kind", "unverified"))
            if snapshot_kind != "raw_residual_stream":
                raise ValueError(
                    "exact response snapshots must declare raw_residual_stream; "
                    f"got {snapshot_kind!r}"
                )
            count_value = _scalar(archive, "response_token_state_counts", None)
            counts = None if count_value is None else np.asarray(count_value, dtype=np.int64).reshape(-1)
            base = manifest.parent
        elif "hidden_files" in archive.files:
            source_format = "canonical_full_hidden_shards_v1"
            files = np.asarray(archive["hidden_files"], dtype=object).reshape(-1)
            if "hidden_layers" not in archive.files:
                raise ValueError("canonical full manifest is missing hidden_layers")
            if not bool(_scalar(archive, "hidden_stored", True)):
                raise ValueError("canonical manifest declares hidden_stored=False")
            layers = np.asarray(archive["hidden_layers"], dtype=np.int64).reshape(-1)
            snapshot_kind = "raw_residual_stream"
            counts = None
            if hidden_dir is not None:
                base = Path(hidden_dir).expanduser().resolve()
            else:
                stored = _scalar(archive, "hidden_dir", None)
                if stored is None:
                    raise FileNotFoundError("canonical full input requires --hidden-dir")
                base = Path(str(np.asarray(stored).reshape(-1)[0])).expanduser().resolve()
        else:
            raise ValueError(
                "manifest needs canonical hidden_files or exact response_token_state_files"
            )
        if files.shape != (n_records,):
            raise ValueError("hidden-state file list is not record-aligned")
        if problem_ids.shape[0] != n_records:
            raise ValueError("problem_ids is not record-aligned")
        if counts is not None and counts.shape != (n_records,):
            raise ValueError("response_token_state_counts is not record-aligned")
        if "response_token_ranges" in archive.files:
            response_starts = np.asarray(
                [int(np.asarray(value).reshape(-1)[0]) for value in archive["response_token_ranges"]],
                dtype=np.int64,
            )
        elif "prompt_token_counts" in archive.files:
            response_starts = np.asarray(archive["prompt_token_counts"], dtype=np.int64)
        else:
            response_starts = np.asarray([value[0, 0] for value in step_ranges], dtype=np.int64)
        if response_starts.shape != (n_records,):
            raise ValueError("response token starts are not record-aligned")
        generators, generator_field = _response_generators(archive, n_records)
        manifest_rows = np.arange(n_records, dtype=np.int64)
        if response_generator is not None:
            requested = _normalized_model_name(response_generator)
            if not requested:
                raise ValueError("response_generator must contain letters or digits")
            if generators is None:
                raise ValueError(
                    "response-generator filtering was requested but the manifest has no "
                    "response_generator/generator provenance"
                )
            mask = np.asarray(
                [requested in _normalized_model_name(str(value)) for value in generators],
                dtype=bool,
            )
            if not np.any(mask):
                available = sorted({str(value) for value in generators})
                raise ValueError(
                    f"no records match response generator {response_generator!r}; "
                    f"available={available}"
                )
            manifest_rows = manifest_rows[mask]
            gold = gold[mask]
            problem_ids = problem_ids[mask]
            step_ranges = [value for value, keep in zip(step_ranges, mask) if keep]
            files = files[mask]
            response_starts = response_starts[mask]
            generators = generators[mask]
            if counts is not None:
                counts = counts[mask]
        resolved_files = np.asarray(
            [str((Path(item) if Path(str(item)).is_absolute() else base / str(item)).resolve()) for item in files],
            dtype=object,
        )
    return _RawSource(
        manifest_path=manifest,
        source_format=source_format,
        files=resolved_files,
        layers=layers,
        gold_error_step=gold,
        problem_ids=problem_ids,
        problem_group_field=problem_group_field,
        step_ranges=step_ranges,
        response_starts=response_starts,
        snapshot_kind=snapshot_kind,
        counts=counts,
        manifest_rows=manifest_rows,
        n_manifest_records=n_records,
        response_generators=generators,
        generator_field=generator_field,
        generator_filter=response_generator,
    )


def _load_shard(source: _RawSource, row: int) -> np.ndarray:
    path = Path(str(source.files[row]))
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 3 or values.shape[1] != source.layers.size:
        raise ValueError(
            f"{path}: expected [token,{source.layers.size},hidden], got {values.shape}"
        )
    if source.counts is not None and int(source.counts[row]) != values.shape[0]:
        raise ValueError(f"{path}: token count disagrees with manifest")
    return values


def inspect_raw_residual_source(
    path: str | Path,
    *,
    hidden_dir: str | Path | None = None,
    response_generator: str | None = None,
) -> dict[str, Any]:
    """Fail-closed preflight that touches only the first raw shard."""
    source = _resolve_source(path, hidden_dir, response_generator)
    first = _load_shard(source, 0)
    return {
        "manifest_path": str(source.manifest_path),
        "source_format": source.source_format,
        "snapshot_kind": source.snapshot_kind,
        "n_manifest_records": source.n_manifest_records,
        "n_records": int(source.gold_error_step.size),
        "n_error_records": int(np.sum(source.gold_error_step >= 0)),
        "n_correct_records": int(np.sum(source.gold_error_step < 0)),
        "layers": source.layers.tolist(),
        "depth_semantics": (
            "adjacent_block" if np.all(np.diff(source.layers) == 1) else "sparse_depth_interval"
        ),
        "first_shard": str(source.files[0]),
        "first_shard_shape": list(first.shape),
        "response_generator_filter": source.generator_filter,
        "generator_field": source.generator_field,
        "response_generators": (
            []
            if source.response_generators is None
            else sorted({str(value) for value in source.response_generators})
        ),
    }


def _match_events(source: _RawSource, same_problem_bonus: float) -> list[_Match]:
    errors = np.where(source.gold_error_step >= 0)[0]
    controls = np.where(source.gold_error_step < 0)[0]
    if errors.size == 0 or controls.size == 0:
        raise ValueError(
            "both first-error and fully-correct records are required after filtering; "
            f"errors={errors.size}, correct={controls.size}, "
            f"response_generator={source.generator_filter!r}"
        )
    cost = np.full((errors.size, controls.size), np.inf)
    chosen = np.zeros_like(cost, dtype=np.int64)
    for error_index, error_row in enumerate(errors):
        error_step = int(source.gold_error_step[error_row])
        error_ranges = source.step_ranges[error_row]
        if error_step >= len(error_ranges):
            raise ValueError(f"gold_error_step[{error_row}] is outside step_token_ranges")
        error_count = len(error_ranges)
        error_relative = error_step / max(error_count - 1, 1)
        error_length = error_ranges[error_step, 1] - error_ranges[error_step, 0] + 1
        have_same_problem = np.any(source.problem_ids[controls] == source.problem_ids[error_row])
        for control_index, control_row in enumerate(controls):
            ranges = source.step_ranges[control_row]
            count = len(ranges)
            relative = np.arange(count) / max(count - 1, 1)
            lengths = ranges[:, 1] - ranges[:, 0] + 1
            candidates = (
                ((relative - error_relative) / 0.20) ** 2
                + ((np.log1p(lengths) - np.log1p(error_length)) / 0.75) ** 2
                + ((np.log1p(count) - np.log1p(error_count)) / 0.45) ** 2
            )
            step = int(np.argmin(candidates))
            value = float(candidates[step])
            if have_same_problem and source.problem_ids[control_row] != source.problem_ids[error_row]:
                value += float(same_problem_bonus)
            cost[error_index, control_index] = value
            chosen[error_index, control_index] = step
    assigned_error, assigned_control = linear_sum_assignment(cost)
    matches = [
        _Match(int(errors[ei]), int(controls[ci]), int(source.gold_error_step[errors[ei]]), int(chosen[ei, ci]))
        for ei, ci in zip(assigned_error, assigned_control)
    ]
    assigned = set(assigned_error.tolist())
    for error_index in sorted(set(range(errors.size)) - assigned):
        control_index = int(np.argmin(cost[error_index]))
        matches.append(
            _Match(
                int(errors[error_index]),
                int(controls[control_index]),
                int(source.gold_error_step[errors[error_index]]),
                int(chosen[error_index, control_index]),
            )
        )
    return sorted(matches, key=lambda item: item.error_row)


def _layer_positions(requested: str | Iterable[int], available: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(requested, str):
        selected = available if requested == "all" else np.asarray(
            [int(value) for value in requested.split(",") if value.strip()], dtype=np.int64
        )
    else:
        selected = np.asarray(tuple(requested), dtype=np.int64)
    unknown = sorted(set(selected.tolist()) - set(available.tolist()))
    if selected.size < 2 or unknown:
        raise ValueError(f"select at least two available layers; unknown={unknown}")
    return np.asarray([int(np.where(available == value)[0][0]) for value in selected]), selected


def _pair_components(
    error_rows: list[int], control_rows: list[int], problem_ids: np.ndarray
) -> np.ndarray:
    """Group matched pairs connected by a reused row or any shared problem id."""
    n_pairs = len(error_rows)
    parent = np.arange(n_pairs)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owner: dict[tuple[str, str], int] = {}
    for pair, (error_row, control_row) in enumerate(zip(error_rows, control_rows)):
        for row in {int(error_row), int(control_row)}:
            problem = problem_ids[row]
            problem_value = problem.item() if isinstance(problem, np.generic) else problem
            keys = (("row", str(row)), ("problem", repr(problem_value)))
            for key in keys:
                if key in owner:
                    union(pair, owner[key])
                else:
                    owner[key] = pair
    roots = [find(index) for index in range(n_pairs)]
    remap = {root: index for index, root in enumerate(sorted(set(roots)))}
    return np.asarray([remap[root] for root in roots], dtype=np.int64)


def load_matched_raw_residual(
    path: str | Path,
    *,
    hidden_dir: str | Path | None = None,
    offsets: Iterable[int] = (-2, -1, 0, 1),
    layers: str | Iterable[int] = "all",
    max_pairs: int = 0,
    same_problem_bonus: float = 25.0,
    response_generator: str | None = None,
) -> LayerTimeDataset:
    """Load matched first-error windows directly from raw response-token residual shards."""
    source = _resolve_source(path, hidden_dir, response_generator)
    time_offsets = np.asarray(tuple(offsets), dtype=np.int64)
    if time_offsets.size < 2 or np.any(np.diff(time_offsets) != 1):
        raise ValueError("offsets must contain at least two consecutive increasing integers")
    positions, selected_layers = _layer_positions(layers, source.layers)
    matches = _match_events(source, same_problem_bonus)
    if max_pairs > 0:
        matches = matches[: int(max_pairs)]
    states: list[np.ndarray] = []
    labels: list[int] = []
    pair_ids: list[int] = []
    row_ids: list[int] = []
    retained_error: list[int] = []
    retained_control: list[int] = []
    dropped = 0
    for match in matches:
        windows = []
        for row, step in ((match.error_row, match.error_step), (match.control_row, match.control_step)):
            event = int(source.step_ranges[row][step, 0] - source.response_starts[row])
            indices = event + time_offsets
            shard = _load_shard(source, row)
            if np.any(indices < 0) or np.any(indices >= shard.shape[0]):
                windows = []
                break
            # Slice the tiny event-time window before selecting layers so the
            # mmap reader never materializes the full response trajectory.
            window = np.asarray(shard[indices][:, positions, :], dtype=np.float32)
            if not np.isfinite(window).all():
                windows = []
                break
            windows.append(window)
        if len(windows) != 2:
            dropped += 1
            continue
        pair = len(retained_error)
        for label, row, window in (
            (1, match.error_row, windows[0]), (0, match.control_row, windows[1])
        ):
            states.append(window)
            labels.append(label)
            pair_ids.append(pair)
            row_ids.append(int(source.manifest_rows[row]))
        retained_error.append(match.error_row)
        retained_control.append(match.control_row)
    if len(retained_error) < 2 and max_pairs != 1:
        raise ValueError("fewer than two complete raw matched pairs remain")
    if not retained_error:
        raise ValueError("no complete raw matched pairs remain")
    components = _pair_components(retained_error, retained_control, source.problem_ids)
    return LayerTimeDataset(
        states=np.asarray(states, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int8),
        pair_ids=np.asarray(pair_ids, dtype=np.int64),
        component_ids=np.repeat(components, 2),
        row_ids=np.asarray(row_ids, dtype=np.int64),
        time_offsets=time_offsets,
        layer_ids=selected_layers,
        feature_names=tuple(f"hidden_{index}" for index in range(states[0].shape[-1])),
        metadata={
            "source_path": str(source.manifest_path),
            "axis_kind": "token",
            "source_format": source.source_format,
            "snapshot_kind": source.snapshot_kind,
            "representation_scope": "raw_residual_stream",
            "depth_semantics": (
                "adjacent_block" if np.all(np.diff(selected_layers) == 1) else "sparse_depth_interval"
            ),
            "n_manifest_records": source.n_manifest_records,
            "n_source_records": int(source.gold_error_step.size),
            "n_candidate_pairs": int(len(matches)),
            "n_retained_pairs": int(len(retained_error)),
            "n_dropped_boundary_pairs": int(dropped),
            "n_components": int(np.unique(components).size),
            "component_grouping": "matched_rows_plus_problem_ids",
            "problem_group_field": source.problem_group_field,
            "response_generator_filter": source.generator_filter,
            "generator_field": source.generator_field,
            "response_generators": (
                []
                if source.response_generators is None
                else sorted({str(value) for value in source.response_generators})
            ),
        },
    )
