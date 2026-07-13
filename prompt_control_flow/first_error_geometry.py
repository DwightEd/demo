from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .flow_signature_data import (
    _resolve_layer_ids,
    _trajectory_array,
    parse_layer_selection,
    resolve_vector_key,
)


EPS = 1e-8
GEOMETRY_NAMES = (
    "delta_norm",
    "relative_delta_norm",
    "turn_angle_rad",
    "menger_curvature",
    "scale_free_curvature",
)


@dataclass(frozen=True)
class FirstErrorGeometryConfig:
    """Configuration for the first-error-aligned geometry audit."""

    device: str = "cuda"
    batch_size: int = 16
    bootstrap: int = 1000
    permutations: int = 1000
    nuisance_folds: int = 5
    nuisance_ridge: float = 1e-3
    random_seed: int = 13
    step_offsets: tuple[int, ...] = (-2, -1, 0, 1, 2)
    token_offsets: tuple[int, ...] = tuple(range(-16, 17))


@dataclass
class StepGeometryDataset:
    source_path: str
    vector_key: str
    trajectories: list[np.ndarray]
    original_indices: np.ndarray
    ids: np.ndarray
    problem_ids: np.ndarray
    is_correct: np.ndarray
    gold_error_step: np.ndarray
    n_steps: np.ndarray
    step_ranges: list[np.ndarray]
    step_lengths: list[np.ndarray]
    layer_ids: np.ndarray
    hidden_dim: int
    skipped: dict[str, int]
    metadata: dict[str, Any]

    @property
    def n_samples(self) -> int:
        return len(self.trajectories)


@dataclass
class AxisGeometryDataset:
    axis_kind: str
    trajectories: list[np.ndarray]
    controls: list[np.ndarray]
    event_indices: np.ndarray
    original_indices: np.ndarray
    problem_ids: np.ndarray
    is_correct: np.ndarray
    layer_ids: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_samples(self) -> int:
        return len(self.trajectories)


@dataclass(frozen=True)
class EventMatch:
    error_row: int
    control_row: int
    error_step: int
    control_step: int
    cost: float
    same_problem: bool
    reused_control: bool


@dataclass
class GeometryAuditResult:
    axis: AxisGeometryDataset
    geometry: list[np.ndarray]
    residual_geometry: list[np.ndarray]
    matches: list[EventMatch]
    event_rows: list[dict[str, Any]]
    discrimination_rows: list[dict[str, Any]]
    metadata: dict[str, Any]


def _as_step_ranges(value: Any, n_steps: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        array = np.asarray(value.tolist())
    array = np.asarray(array, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < n_steps:
        raise ValueError(f"invalid step_token_ranges shape={array.shape}, expected at least [{n_steps},2]")
    array = array[:n_steps]
    if np.any(array[:, 1] < array[:, 0]):
        raise ValueError("step_token_ranges contains an end before its start")
    return array


def load_step_geometry_dataset(
    path: str | Path,
    *,
    vector_key: str = "auto",
    layers: str = "all",
    max_samples: int = 0,
) -> StepGeometryDataset:
    """Load ProcessBench step trajectories without re-running the model."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    if "gold_error_step" not in z.files:
        raise ValueError("first-error geometry requires gold_error_step")
    if "step_token_ranges" not in z.files:
        raise ValueError("first-error geometry requires step_token_ranges for length controls")

    key = resolve_vector_key(z.files, vector_key)
    raw = z[key]
    n_records = int(len(raw))
    gold_all = np.asarray(z["gold_error_step"], dtype=np.int64)
    if gold_all.shape[0] != n_records:
        raise ValueError("gold_error_step length does not match trajectories")
    problem_all = (
        np.asarray(z["problem_ids"])
        if "problem_ids" in z.files
        else np.arange(n_records, dtype=np.int64)
    )
    ids_all = (
        np.asarray(z["ids"], dtype=object)
        if "ids" in z.files
        else np.asarray([str(i) for i in range(n_records)], dtype=object)
    )

    example = None
    for value in raw:
        try:
            candidate = _trajectory_array(value)
        except (TypeError, ValueError):
            continue
        if candidate.shape[0] >= 2:
            example = candidate
            break
    if example is None:
        raise ValueError(f"{path}: no valid step trajectory")
    all_layers = _resolve_layer_ids(z, key, int(example.shape[1]))
    layer_positions, selected_layers = parse_layer_selection(layers, all_layers)

    trajectories: list[np.ndarray] = []
    ranges_out: list[np.ndarray] = []
    lengths_out: list[np.ndarray] = []
    original: list[int] = []
    ids: list[Any] = []
    problems: list[Any] = []
    gold: list[int] = []
    correct: list[int] = []
    skipped = {
        "invalid_shape": 0,
        "too_short": 0,
        "nonfinite": 0,
        "invalid_ranges": 0,
        "invalid_gold": 0,
    }

    for i, value in enumerate(raw):
        if max_samples and len(trajectories) >= int(max_samples):
            break
        try:
            trajectory = _trajectory_array(value)
        except (TypeError, ValueError):
            skipped["invalid_shape"] += 1
            continue
        if trajectory.shape[0] < 2:
            skipped["too_short"] += 1
            continue
        if trajectory.shape[1] != all_layers.size:
            skipped["invalid_shape"] += 1
            continue
        trajectory = np.asarray(trajectory[:, layer_positions, :], dtype=np.float32)
        if not np.isfinite(trajectory).all():
            skipped["nonfinite"] += 1
            continue
        try:
            ranges = _as_step_ranges(z["step_token_ranges"][i], trajectory.shape[0])
        except (TypeError, ValueError):
            skipped["invalid_ranges"] += 1
            continue
        g = int(gold_all[i])
        if g >= trajectory.shape[0]:
            skipped["invalid_gold"] += 1
            continue
        trajectories.append(np.ascontiguousarray(trajectory))
        ranges_out.append(ranges)
        lengths_out.append((ranges[:, 1] - ranges[:, 0] + 1).astype(np.float64))
        original.append(i)
        ids.append(ids_all[i])
        problems.append(problem_all[i])
        gold.append(g)
        correct.append(int(g < 0))

    if not trajectories:
        raise ValueError(f"{path}: no valid trajectories remain")
    hidden_dim = int(trajectories[0].shape[2])
    expected = (selected_layers.size, hidden_dim)
    if any(x.shape[1:] != expected for x in trajectories):
        raise ValueError("trajectory layer/hidden shapes are inconsistent")

    stored_correct_mismatch = 0
    if "is_correct" in z.files:
        stored = np.asarray(z["is_correct"], dtype=np.int64)[np.asarray(original)]
        stored_correct_mismatch = int(np.sum(stored != np.asarray(correct, dtype=np.int64)))
    metadata = {
        "available_layers": all_layers.tolist(),
        "selected_layers": selected_layers.tolist(),
        "stored_correct_mismatch": stored_correct_mismatch,
        "has_hidden_shards": bool(
            "hidden_files" in z.files and "hidden_layers" in z.files and len(z["hidden_layers"]) > 0
        ),
        "hidden_layers": (
            np.asarray(z["hidden_layers"], dtype=np.int64).tolist()
            if "hidden_layers" in z.files
            else []
        ),
    }
    return StepGeometryDataset(
        source_path=str(path.resolve()),
        vector_key=key,
        trajectories=trajectories,
        original_indices=np.asarray(original, dtype=np.int64),
        ids=np.asarray(ids, dtype=object),
        problem_ids=np.asarray(problems),
        is_correct=np.asarray(correct, dtype=np.int64),
        gold_error_step=np.asarray(gold, dtype=np.int64),
        n_steps=np.asarray([x.shape[0] for x in trajectories], dtype=np.int64),
        step_ranges=ranges_out,
        step_lengths=lengths_out,
        layer_ids=selected_layers,
        hidden_dim=hidden_dim,
        skipped=skipped,
        metadata=metadata,
    )


def _step_controls(dataset: StepGeometryDataset) -> list[np.ndarray]:
    controls: list[np.ndarray] = []
    for lengths in dataset.step_lengths:
        t = int(lengths.size)
        rel = np.arange(t, dtype=np.float64) / max(t - 1, 1)
        prev_len = np.concatenate([lengths[:1], lengths[:-1]])
        next_len = np.concatenate([lengths[1:], lengths[-1:]])
        controls.append(
            np.column_stack(
                [
                    rel,
                    rel * rel,
                    np.log1p(lengths),
                    np.log1p(prev_len),
                    np.log1p(next_len),
                    np.full(t, np.log1p(t), dtype=np.float64),
                ]
            )
        )
    return controls


def make_step_axis(dataset: StepGeometryDataset) -> AxisGeometryDataset:
    return AxisGeometryDataset(
        axis_kind="step",
        trajectories=dataset.trajectories,
        controls=_step_controls(dataset),
        event_indices=dataset.gold_error_step.copy(),
        original_indices=dataset.original_indices.copy(),
        problem_ids=dataset.problem_ids.copy(),
        is_correct=dataset.is_correct.copy(),
        layer_ids=dataset.layer_ids.copy(),
        metadata={
            "event_definition": "step index; delta_norm at offset 0 is the edge entering the first-error step",
            "control_names": [
                "relative_step_position",
                "relative_step_position_sq",
                "log1p_step_length",
                "log1p_previous_step_length",
                "log1p_next_step_length",
                "log1p_num_steps",
            ],
        },
    )


def _resolve_hidden_root(z: np.lib.npyio.NpzFile, override: str | Path | None) -> Path:
    if override is not None:
        root = Path(override)
        if not root.exists():
            raise FileNotFoundError(f"hidden shard directory does not exist: {root}")
        return root
    if "hidden_dir" not in z.files:
        raise FileNotFoundError("NPZ has no hidden_dir; pass --hidden_dir")
    value = np.asarray(z["hidden_dir"], dtype=object).reshape(-1)
    root = Path(str(value[0])) if value.size else Path("")
    if not root.exists():
        raise FileNotFoundError(
            f"stored hidden_dir is unavailable on this machine: {root}. Pass --hidden_dir explicitly."
        )
    return root


def _token_controls(ranges: np.ndarray, n_tokens: int) -> np.ndarray:
    starts = ranges[:, 0]
    ends = ranges[:, 1]
    response_start = int(starts[0])
    absolute = response_start + np.arange(n_tokens, dtype=np.int64)
    step_idx = np.searchsorted(starts, absolute, side="right") - 1
    step_idx = np.clip(step_idx, 0, len(starts) - 1)
    step_len = ends[step_idx] - starts[step_idx] + 1
    within = (absolute - starts[step_idx]) / np.maximum(step_len, 1)
    within = np.clip(within, 0.0, 1.0)
    rel = np.arange(n_tokens, dtype=np.float64) / max(n_tokens - 1, 1)
    return np.column_stack(
        [
            rel,
            rel * rel,
            np.log1p(step_len.astype(np.float64)),
            within.astype(np.float64),
            np.full(n_tokens, np.log1p(n_tokens), dtype=np.float64),
        ]
    )


def load_token_axis(
    source: StepGeometryDataset,
    *,
    hidden_dir: str | Path | None = None,
    layers: str = "all",
) -> AxisGeometryDataset:
    """Load existing per-token hidden shards and align events to error-step starts."""

    z = np.load(source.source_path, allow_pickle=True)
    if "hidden_files" not in z.files or "hidden_layers" not in z.files:
        raise FileNotFoundError("NPZ does not declare per-token hidden shards")
    hidden_files = np.asarray(z["hidden_files"], dtype=object)
    if hidden_files.shape[0] != len(z[source.vector_key]):
        raise ValueError("hidden_files is not record-aligned with the source NPZ")
    all_layers = np.asarray(z["hidden_layers"], dtype=np.int64).reshape(-1)
    layer_positions, selected_layers = parse_layer_selection(layers, all_layers)
    root = _resolve_hidden_root(z, hidden_dir)

    trajectories: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    events: list[int] = []
    original: list[int] = []
    problems: list[Any] = []
    correct: list[int] = []
    missing: list[str] = []
    for row, original_idx in enumerate(source.original_indices):
        file_name = str(hidden_files[int(original_idx)])
        shard = root / file_name
        if not shard.exists():
            missing.append(str(shard))
            continue
        values = np.load(shard, mmap_mode="r")
        if values.ndim != 3 or values.shape[1] != all_layers.size:
            raise ValueError(
                f"{shard}: expected [token,{all_layers.size},hidden], got {values.shape}"
            )
        trajectory = np.asarray(values[:, layer_positions, :], dtype=np.float32)
        if trajectory.shape[0] < 3 or not np.isfinite(trajectory).all():
            continue
        ranges = source.step_ranges[row]
        response_start = int(ranges[0, 0])
        g = int(source.gold_error_step[row])
        event = int(ranges[g, 0] - response_start) if g >= 0 else -1
        if event >= trajectory.shape[0]:
            raise ValueError(f"{shard}: first-error token boundary {event} is out of range")
        trajectories.append(np.ascontiguousarray(trajectory))
        controls.append(_token_controls(ranges, trajectory.shape[0]))
        events.append(event)
        original.append(int(original_idx))
        problems.append(source.problem_ids[row])
        correct.append(int(source.is_correct[row]))
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} hidden shards are missing under {root}; examples: {preview}"
        )
    if not trajectories:
        raise ValueError("no valid token hidden trajectories were loaded")
    return AxisGeometryDataset(
        axis_kind="token",
        trajectories=trajectories,
        controls=controls,
        event_indices=np.asarray(events, dtype=np.int64),
        original_indices=np.asarray(original, dtype=np.int64),
        problem_ids=np.asarray(problems),
        is_correct=np.asarray(correct, dtype=np.int64),
        layer_ids=selected_layers,
        metadata={
            "hidden_dir": str(root.resolve()),
            "event_definition": "first token of the first-error step; angle/curvature require one future token",
            "control_names": [
                "relative_token_position",
                "relative_token_position_sq",
                "log1p_containing_step_length",
                "within_step_fraction",
                "log1p_response_tokens",
            ],
        },
    )


def _device(value: str) -> torch.device:
    requested = str(value)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


@torch.inference_mode()
def compute_geometry_fields(
    trajectories: Sequence[np.ndarray],
    *,
    device: str = "cuda",
    batch_size: int = 16,
    eps: float = EPS,
) -> list[np.ndarray]:
    """Compute velocity, turning angle, and Menger curvature in GPU batches.

    Output item ``i`` has shape ``[axis, layer, metric]``. ``delta_norm[t]``
    is the incoming edge ``z[t]-z[t-1]``. Turning and curvature at ``t`` use
    the triplet ``(z[t-1], z[t], z[t+1])``.
    """

    if not trajectories:
        return []
    layer_hidden = trajectories[0].shape[1:]
    if len(layer_hidden) != 2:
        raise ValueError("trajectories must have shape [axis,layer,hidden]")
    if any(x.ndim != 3 or x.shape[1:] != layer_hidden for x in trajectories):
        raise ValueError("trajectory layer/hidden shapes are inconsistent")
    target = _device(device)
    out: list[np.ndarray] = []
    for start in range(0, len(trajectories), max(1, int(batch_size))):
        chunk = trajectories[start : start + max(1, int(batch_size))]
        lengths = [int(x.shape[0]) for x in chunk]
        max_len = max(lengths)
        states = torch.zeros(
            (len(chunk), max_len, layer_hidden[0], layer_hidden[1]),
            dtype=torch.float32,
            device=target,
        )
        valid = torch.zeros((len(chunk), max_len), dtype=torch.bool, device=target)
        for j, values in enumerate(chunk):
            n = int(values.shape[0])
            states[j, :n] = torch.as_tensor(values, dtype=torch.float32, device=target)
            valid[j, :n] = True

        field = torch.full(
            (len(chunk), max_len, layer_hidden[0], len(GEOMETRY_NAMES)),
            float("nan"),
            dtype=torch.float32,
            device=target,
        )
        delta = states[:, 1:] - states[:, :-1]
        delta_valid = valid[:, 1:] & valid[:, :-1]
        speed = torch.linalg.vector_norm(delta, dim=-1)
        state_scale = 0.5 * (
            torch.linalg.vector_norm(states[:, 1:], dim=-1)
            + torch.linalg.vector_norm(states[:, :-1], dim=-1)
        )
        relative_speed = speed / state_scale.clamp_min(float(eps))
        field[:, 1:, :, 0] = torch.where(
            delta_valid[:, :, None], speed, torch.full_like(speed, float("nan"))
        )
        field[:, 1:, :, 1] = torch.where(
            delta_valid[:, :, None], relative_speed, torch.full_like(relative_speed, float("nan"))
        )

        if max_len >= 3:
            incoming = delta[:, :-1]
            outgoing = delta[:, 1:]
            interior_valid = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
            in_norm = torch.linalg.vector_norm(incoming, dim=-1)
            out_norm = torch.linalg.vector_norm(outgoing, dim=-1)
            denom = (in_norm * out_norm).clamp_min(float(eps))
            cosine = torch.sum(incoming * outgoing, dim=-1) / denom
            cosine = cosine.clamp(-1.0, 1.0)
            angle = torch.acos(cosine)
            sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0, max=1.0))
            chord = torch.linalg.vector_norm(states[:, 2:] - states[:, :-2], dim=-1)
            kappa = 2.0 * sine / chord.clamp_min(float(eps))
            scale_free = kappa * 0.5 * (in_norm + out_norm)
            nondegenerate = (
                interior_valid[:, :, None]
                & (in_norm > float(eps))
                & (out_norm > float(eps))
                & (chord > float(eps))
            )
            nan = torch.full_like(angle, float("nan"))
            field[:, 1:-1, :, 2] = torch.where(nondegenerate, angle, nan)
            field[:, 1:-1, :, 3] = torch.where(nondegenerate, kappa, nan)
            field[:, 1:-1, :, 4] = torch.where(nondegenerate, scale_free, nan)

        field_cpu = field.detach().cpu().numpy()
        out.extend([field_cpu[j, : lengths[j]].copy() for j in range(len(chunk))])
        del states, valid, field
    return out


def match_correct_pseudo_events(
    dataset: StepGeometryDataset,
    *,
    same_problem_bonus: float = 25.0,
) -> list[EventMatch]:
    """One-to-one match each error event to a correct-chain pseudo event.

    Matching uses only nuisance variables: number of steps, relative event
    position, and event-step token length. Correctness geometry is never used.
    """

    errors = np.where((dataset.is_correct == 0) & (dataset.gold_error_step >= 0))[0]
    controls = np.where(dataset.is_correct == 1)[0]
    if errors.size == 0 or controls.size == 0:
        return []
    cost = np.full((errors.size, controls.size), np.inf, dtype=np.float64)
    chosen_step = np.zeros((errors.size, controls.size), dtype=np.int64)
    for ei, error_row in enumerate(errors):
        g = int(dataset.gold_error_step[error_row])
        te = int(dataset.n_steps[error_row])
        rel_e = g / max(te - 1, 1)
        len_e = float(dataset.step_lengths[error_row][g])
        same_candidates = np.any(dataset.problem_ids[controls] == dataset.problem_ids[error_row])
        for ci, control_row in enumerate(controls):
            tc = int(dataset.n_steps[control_row])
            lengths = dataset.step_lengths[control_row]
            rel_c = np.arange(tc, dtype=np.float64) / max(tc - 1, 1)
            candidate_cost = (
                ((rel_c - rel_e) / 0.20) ** 2
                + ((np.log1p(lengths) - np.log1p(len_e)) / 0.75) ** 2
                + ((np.log1p(tc) - np.log1p(te)) / 0.45) ** 2
            )
            best = int(np.argmin(candidate_cost))
            value = float(candidate_cost[best])
            same = dataset.problem_ids[control_row] == dataset.problem_ids[error_row]
            if same_candidates and not same:
                value += float(same_problem_bonus)
            cost[ei, ci] = value
            chosen_step[ei, ci] = best

    row_idx, col_idx = linear_sum_assignment(cost)
    matches: list[EventMatch] = []
    matched_errors = set()
    for ei, ci in zip(row_idx.tolist(), col_idx.tolist()):
        error_row = int(errors[ei])
        control_row = int(controls[ci])
        matched_errors.add(ei)
        matches.append(
            EventMatch(
                error_row=error_row,
                control_row=control_row,
                error_step=int(dataset.gold_error_step[error_row]),
                control_step=int(chosen_step[ei, ci]),
                cost=float(cost[ei, ci]),
                same_problem=bool(dataset.problem_ids[error_row] == dataset.problem_ids[control_row]),
                reused_control=False,
            )
        )
    # If there are slightly more error than correct responses, retain coverage
    # with nearest-neighbour reuse and expose that fact in the match table.
    for ei in sorted(set(range(errors.size)) - matched_errors):
        ci = int(np.argmin(cost[ei]))
        error_row = int(errors[ei])
        control_row = int(controls[ci])
        matches.append(
            EventMatch(
                error_row=error_row,
                control_row=control_row,
                error_step=int(dataset.gold_error_step[error_row]),
                control_step=int(chosen_step[ei, ci]),
                cost=float(cost[ei, ci]),
                same_problem=bool(dataset.problem_ids[error_row] == dataset.problem_ids[control_row]),
                reused_control=True,
            )
        )
    return sorted(matches, key=lambda x: x.error_row)


def map_matches_to_axis(
    matches: Sequence[EventMatch],
    step_source: StepGeometryDataset,
    axis: AxisGeometryDataset,
) -> list[EventMatch]:
    """Map step-row matches onto a step or token geometry axis."""

    original_to_axis = {int(value): i for i, value in enumerate(axis.original_indices)}
    out: list[EventMatch] = []
    for match in matches:
        error_original = int(step_source.original_indices[match.error_row])
        control_original = int(step_source.original_indices[match.control_row])
        if error_original not in original_to_axis or control_original not in original_to_axis:
            continue
        erow = original_to_axis[error_original]
        crow = original_to_axis[control_original]
        if axis.axis_kind == "step":
            error_event = int(match.error_step)
            control_event = int(match.control_step)
        elif axis.axis_kind == "token":
            source_error = match.error_row
            source_control = match.control_row
            error_ranges = step_source.step_ranges[source_error]
            control_ranges = step_source.step_ranges[source_control]
            error_event = int(error_ranges[match.error_step, 0] - error_ranges[0, 0])
            control_event = int(control_ranges[match.control_step, 0] - control_ranges[0, 0])
        else:
            raise ValueError(f"unknown axis kind {axis.axis_kind!r}")
        out.append(
            EventMatch(
                error_row=erow,
                control_row=crow,
                error_step=error_event,
                control_step=control_event,
                cost=match.cost,
                same_problem=match.same_problem,
                reused_control=match.reused_control,
            )
        )
    return out


def _group_folds(groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    unique = np.unique(groups)
    rng = np.random.default_rng(int(seed))
    shuffled = unique[rng.permutation(unique.size)]
    mapping = {value: i % max(2, min(int(folds), unique.size)) for i, value in enumerate(shuffled)}
    return np.asarray([mapping[value] for value in groups], dtype=np.int64)


def _paired_group_folds(
    axis: AxisGeometryDataset,
    matches: Sequence[EventMatch],
    folds: int,
    seed: int,
) -> np.ndarray:
    """Keep same-problem rows and matched event/control pairs in one fold."""

    parent = np.arange(axis.n_samples, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    problem_owner: dict[Any, int] = {}
    for row, problem in enumerate(axis.problem_ids.tolist()):
        if problem in problem_owner:
            union(row, problem_owner[problem])
        else:
            problem_owner[problem] = row
    for match in matches:
        union(int(match.error_row), int(match.control_row))
    components = np.asarray([find(row) for row in range(axis.n_samples)], dtype=np.int64)
    return _group_folds(components, folds, seed)


def crossfit_correct_nuisance_residuals(
    fields: Sequence[np.ndarray],
    axis: AxisGeometryDataset,
    *,
    matches: Sequence[EventMatch] = (),
    device: str = "cuda",
    folds: int = 5,
    ridge: float = 1e-3,
    seed: int = 13,
) -> list[np.ndarray]:
    """Residualize geometry on position/length using correct training chains only."""

    if len(fields) != axis.n_samples or len(axis.controls) != axis.n_samples:
        raise ValueError("field/control records do not match the geometry axis")
    assignments = _paired_group_folds(axis, matches, folds, seed)
    output = [np.full_like(field, np.nan, dtype=np.float32) for field in fields]
    n_layers = fields[0].shape[1]
    n_metrics = fields[0].shape[2]
    target = _device(device)

    for fold in np.unique(assignments):
        train_rows = np.where((assignments != fold) & (axis.is_correct == 1))[0]
        test_rows = np.where(assignments == fold)[0]
        if train_rows.size < 3:
            continue
        train_x = np.concatenate([axis.controls[i] for i in train_rows], axis=0)
        mean = np.nanmean(train_x, axis=0)
        scale = np.nanstd(train_x, axis=0)
        scale = np.where(scale > EPS, scale, 1.0)

        for layer in range(n_layers):
            for metric in range(n_metrics):
                train_y = np.concatenate([fields[i][:, layer, metric] for i in train_rows])
                valid = np.isfinite(train_y) & np.isfinite(train_x).all(axis=1)
                if np.sum(valid) <= train_x.shape[1] + 2:
                    continue
                x = (train_x[valid] - mean) / scale
                x = np.column_stack([np.ones(x.shape[0]), x])
                y = train_y[valid].astype(np.float64)
                x_tensor = torch.as_tensor(x, dtype=torch.float32, device=target)
                y_tensor = torch.as_tensor(y, dtype=torch.float32, device=target)
                penalty = torch.eye(x_tensor.shape[1], dtype=torch.float32, device=target)
                penalty *= float(ridge)
                penalty[0, 0] = 0.0
                beta = torch.linalg.solve(
                    x_tensor.T @ x_tensor + penalty,
                    x_tensor.T @ y_tensor,
                )
                for row in test_rows:
                    test_x = axis.controls[row]
                    test_y = fields[row][:, layer, metric]
                    ok = np.isfinite(test_y) & np.isfinite(test_x).all(axis=1)
                    if not np.any(ok):
                        continue
                    design = (test_x[ok] - mean) / scale
                    design = np.column_stack([np.ones(design.shape[0]), design])
                    design_tensor = torch.as_tensor(
                        design, dtype=torch.float32, device=target
                    )
                    prediction = (design_tensor @ beta).detach().cpu().numpy()
                    output[row][ok, layer, metric] = (
                        test_y[ok] - prediction
                    ).astype(np.float32)
    return output


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(score) & ((y == 0) | (y == 1))
    y = y[valid]
    score = score[valid]
    pos = score[y == 1]
    neg = score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    comparisons = pos[:, None] - neg[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def _bootstrap_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if draws <= 0 or values.size == 1:
        value = float(np.mean(values))
        return value, value
    indices = rng.integers(0, values.size, size=(int(draws), values.size))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _sign_flip_p(values: np.ndarray, permutations: int, rng: np.random.Generator) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or permutations <= 0:
        return float("nan")
    observed = abs(float(np.mean(values)))
    exceed = 0
    remaining = int(permutations)
    while remaining > 0:
        take = min(remaining, 512)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(take, values.size))
        permuted = np.abs(np.mean(signs * values[None, :], axis=1))
        exceed += int(np.sum(permuted >= observed))
        remaining -= take
    return float((exceed + 1) / (int(permutations) + 1))


def _bh_qvalues(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full(p.shape, np.nan, dtype=np.float64)
    valid = np.where(np.isfinite(p))[0]
    if valid.size == 0:
        return out
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * valid.size / np.arange(1, valid.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def summarize_matched_events(
    fields: Sequence[np.ndarray],
    residual_fields: Sequence[np.ndarray],
    axis: AxisGeometryDataset,
    matches: Sequence[EventMatch],
    *,
    offsets: Sequence[int],
    bootstrap: int,
    permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for variant, values in (("raw", fields), ("nuisance_residual", residual_fields)):
        for metric_idx, metric in enumerate(GEOMETRY_NAMES):
            for layer_pos, layer in enumerate(axis.layer_ids):
                for offset in offsets:
                    error_values: list[float] = []
                    control_values: list[float] = []
                    for match in matches:
                        ei = int(match.error_step + int(offset))
                        ci = int(match.control_step + int(offset))
                        if ei < 0 or ci < 0:
                            continue
                        if ei >= values[match.error_row].shape[0] or ci >= values[match.control_row].shape[0]:
                            continue
                        ev = float(values[match.error_row][ei, layer_pos, metric_idx])
                        cv = float(values[match.control_row][ci, layer_pos, metric_idx])
                        if np.isfinite(ev) and np.isfinite(cv):
                            error_values.append(ev)
                            control_values.append(cv)
                    error_arr = np.asarray(error_values, dtype=np.float64)
                    control_arr = np.asarray(control_values, dtype=np.float64)
                    difference = error_arr - control_arr
                    err_low, err_high = _bootstrap_ci(error_arr, bootstrap, rng)
                    ctl_low, ctl_high = _bootstrap_ci(control_arr, bootstrap, rng)
                    diff_low, diff_high = _bootstrap_ci(difference, bootstrap, rng)
                    dz = (
                        float(np.mean(difference) / (np.std(difference, ddof=1) + EPS))
                        if difference.size > 1
                        else float("nan")
                    )
                    p_value = (
                        _sign_flip_p(difference, permutations, rng)
                        if int(offset) == 0
                        else float("nan")
                    )
                    rows.append(
                        {
                            "axis": axis.axis_kind,
                            "variant": variant,
                            "metric": metric,
                            "layer": int(layer),
                            "offset": int(offset),
                            "n_pairs": int(difference.size),
                            "pair_coverage": float(difference.size / max(len(matches), 1)),
                            "error_mean": float(np.mean(error_arr)) if error_arr.size else float("nan"),
                            "error_ci_low": err_low,
                            "error_ci_high": err_high,
                            "control_mean": float(np.mean(control_arr)) if control_arr.size else float("nan"),
                            "control_ci_low": ctl_low,
                            "control_ci_high": ctl_high,
                            "paired_difference": (
                                float(np.mean(difference)) if difference.size else float("nan")
                            ),
                            "difference_ci_low": diff_low,
                            "difference_ci_high": diff_high,
                            "paired_effect_dz": dz,
                            "matched_event_auroc": _auc(
                                np.concatenate(
                                    [
                                        np.ones(error_arr.size, dtype=np.int64),
                                        np.zeros(control_arr.size, dtype=np.int64),
                                    ]
                                ),
                                np.concatenate([error_arr, control_arr]),
                            ),
                            "sign_flip_p": p_value,
                            "bh_q": float("nan"),
                        }
                    )
    primary = [i for i, row in enumerate(rows) if row["offset"] == 0]
    q_values = _bh_qvalues([rows[i]["sign_flip_p"] for i in primary])
    for i, q in zip(primary, q_values):
        rows[i]["bh_q"] = float(q)
    return rows


def _expected_rank(scores: np.ndarray, gold_position: int) -> tuple[float, float]:
    gold = float(scores[gold_position])
    if not np.isfinite(gold):
        return float("nan"), float("nan")
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return float("nan"), float("nan")
    greater = float(np.sum(finite > gold))
    equal = float(np.sum(finite == gold))
    rank = 1.0 + greater + 0.5 * max(equal - 1.0, 0.0)
    top1 = 1.0 / equal if greater == 0.0 and equal > 0.0 else 0.0
    return rank, top1


def first_error_discrimination(
    fields: Sequence[np.ndarray],
    residual_fields: Sequence[np.ndarray],
    axis: AxisGeometryDataset,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, values in (("raw", fields), ("nuisance_residual", residual_fields)):
        for metric_idx, metric in enumerate(GEOMETRY_NAMES):
            for layer_pos, layer in enumerate(axis.layer_ids):
                labels: list[int] = []
                scores: list[float] = []
                ranks: list[float] = []
                top1: list[float] = []
                eligible = 0
                for i, event in enumerate(axis.event_indices):
                    event = int(event)
                    if event < 0 or event >= values[i].shape[0]:
                        continue
                    candidate = values[i][: event + 1, layer_pos, metric_idx]
                    for t, value in enumerate(candidate):
                        if np.isfinite(value):
                            labels.append(int(t == event))
                            scores.append(float(value))
                    rank, hit = _expected_rank(candidate, event)
                    if np.isfinite(rank):
                        ranks.append(rank)
                        top1.append(hit)
                        eligible += 1
                rows.append(
                    {
                        "axis": axis.axis_kind,
                        "variant": variant,
                        "metric": metric,
                        "layer": int(layer),
                        "n_rows": int(len(scores)),
                        "n_positive": int(np.sum(labels)),
                        "auroc_high_is_error": _auc(np.asarray(labels), np.asarray(scores)),
                        "eligible_chains": int(eligible),
                        "eligible_fraction": float(
                            eligible / max(int(np.sum(axis.event_indices >= 0)), 1)
                        ),
                        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
                        "expected_top1": float(np.mean(top1)) if top1 else float("nan"),
                    }
                )
    return rows


def run_first_error_geometry_audit(
    axis: AxisGeometryDataset,
    matches: Sequence[EventMatch],
    cfg: FirstErrorGeometryConfig,
) -> GeometryAuditResult:
    geometry = compute_geometry_fields(
        axis.trajectories,
        device=cfg.device,
        batch_size=cfg.batch_size,
    )
    residual = crossfit_correct_nuisance_residuals(
        geometry,
        axis,
        matches=matches,
        device=cfg.device,
        folds=cfg.nuisance_folds,
        ridge=cfg.nuisance_ridge,
        seed=cfg.random_seed,
    )
    offsets = cfg.step_offsets if axis.axis_kind == "step" else cfg.token_offsets
    event_rows = summarize_matched_events(
        geometry,
        residual,
        axis,
        matches,
        offsets=offsets,
        bootstrap=cfg.bootstrap,
        permutations=cfg.permutations,
        seed=cfg.random_seed,
    )
    discrimination = first_error_discrimination(geometry, residual, axis)
    return GeometryAuditResult(
        axis=axis,
        geometry=geometry,
        residual_geometry=residual,
        matches=list(matches),
        event_rows=event_rows,
        discrimination_rows=discrimination,
        metadata={
            "geometry_names": list(GEOMETRY_NAMES),
            "offsets": list(offsets),
            "requested_device": cfg.device,
            "effective_device": str(_device(cfg.device)),
            "batch_size": int(cfg.batch_size),
            "bootstrap": int(cfg.bootstrap),
            "permutations": int(cfg.permutations),
            "nuisance_folds": int(cfg.nuisance_folds),
            "nuisance_split": "same-problem and matched error/control pairs held out together",
            "n_matches": int(len(matches)),
            "online_availability": {
                "delta_norm": "available after the event token/step state is observed",
                "relative_delta_norm": "available after the event token/step state is observed",
                "turn_angle_rad": "requires one future token/step",
                "menger_curvature": "requires one future token/step",
                "scale_free_curvature": "requires one future token/step",
            },
        },
    )
