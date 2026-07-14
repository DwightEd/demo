from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .schema import TraceArtifact

if TYPE_CHECKING:
    from .geometry_features import GeometryCollection


@dataclass
class ReasoningSequence:
    chain_idx: int
    problem_id: int
    gold_error_step: int
    is_error: int
    output: np.ndarray
    geometry: np.ndarray
    step_token_counts: np.ndarray

    @property
    def n_steps(self) -> int:
        return int(self.output.shape[0])


@dataclass
class SequenceCollection:
    sequences: list[ReasoningSequence]
    output_names: tuple[str, ...]
    geometry_names: tuple[str, ...]
    geometry_groups: tuple[str, ...]
    preflight: dict[str, object]


@dataclass
class BinaryTask:
    name: str
    checkpoint: float
    x_output: np.ndarray
    x_geometry: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    chain_idx: np.ndarray
    output_names: tuple[str, ...]
    geometry_names: tuple[str, ...]
    geometry_groups: tuple[str, ...]
    nuisance_values: np.ndarray
    step_idx: np.ndarray | None = None


@dataclass
class ForecastTask:
    name: str
    x_output: np.ndarray
    x_geometry: np.ndarray
    target: np.ndarray
    groups: np.ndarray
    chain_idx: np.ndarray
    step_idx: np.ndarray
    output_names: tuple[str, ...]
    geometry_names: tuple[str, ...]
    geometry_groups: tuple[str, ...]
    target_names: tuple[str, ...]
    nuisance_values: np.ndarray


def join_trace_and_geometry(
    trace: TraceArtifact,
    geometry: GeometryCollection,
    *,
    label_policy: str = "process_error",
) -> SequenceCollection:
    """Join two independently extracted artifacts without positional guessing."""

    trace.validate()
    geometry.validate()
    if label_policy not in {"process_error", "final_answer"}:
        raise ValueError("label_policy must be `process_error` or `final_answer`")
    geometry_row = {int(chain): i for i, chain in enumerate(geometry.chain_idx)}
    output_keep = np.asarray(
        [
            i
            for i, name in enumerate(trace.step_feature_names)
            if str(name).startswith("out.")
        ],
        dtype=np.int64,
    )
    if len(output_keep) == 0:
        raise ValueError("compact trace contains no output features")
    output_names = tuple(trace.step_feature_names[i] for i in output_keep)
    sequences: list[ReasoningSequence] = []
    missing_geometry: list[int] = []
    mismatched_steps: list[int] = []
    mismatched_problem: list[int] = []
    mismatched_response: list[int] = []
    mismatched_label: list[int] = []
    missing_requested_label: list[int] = []
    for trace_row, chain in enumerate(trace.chain_idx):
        chain = int(chain)
        geometry_index = geometry_row.get(chain)
        if geometry_index is None:
            missing_geometry.append(chain)
            continue
        trace_steps = trace.step_matrix(trace_row)
        ranges = trace.step_ranges(trace_row)
        geom_steps = np.asarray(geometry.matrices[geometry_index], dtype=np.float32)
        if len(trace_steps) != len(geom_steps):
            mismatched_steps.append(chain)
            continue
        if int(trace.problem_id[trace_row]) != int(geometry.problem_id[geometry_index]):
            mismatched_problem.append(chain)
            continue
        trace_hash = str(trace.response_hash[trace_row])
        geometry_hash = str(geometry.response_hash[geometry_index])
        if trace_hash and geometry_hash and trace_hash != geometry_hash:
            mismatched_response.append(chain)
            continue
        if bool(geometry.preflight.get("gold_error_step_available", False)) and int(
            trace.gold_error_step[trace_row]
        ) != int(geometry.gold_error_step[geometry_index]):
            mismatched_label.append(chain)
            continue
        if label_policy == "final_answer" and int(trace.is_correct[trace_row]) < 0:
            missing_requested_label.append(chain)
            continue
        if (
            label_policy == "final_answer"
            and bool(geometry.preflight.get("final_answer_label_available", False))
            and int(trace.is_correct[trace_row])
            != int(geometry.is_correct[geometry_index])
        ):
            mismatched_label.append(chain)
            continue
        token_counts = ranges[:, 1] - ranges[:, 0] + 1
        sequences.append(
            ReasoningSequence(
                chain_idx=chain,
                problem_id=int(trace.problem_id[trace_row]),
                gold_error_step=int(trace.gold_error_step[trace_row]),
                is_error=(
                    int(trace.gold_error_step[trace_row] >= 0)
                    if label_policy == "process_error"
                    else int(1 - int(trace.is_correct[trace_row]))
                ),
                output=np.asarray(trace_steps[:, output_keep], dtype=np.float32),
                geometry=geom_steps,
                step_token_counts=token_counts.astype(np.float32),
            )
        )
    if (
        missing_geometry
        or mismatched_steps
        or mismatched_problem
        or mismatched_response
        or mismatched_label
        or missing_requested_label
    ):
        raise ValueError(
            "trace/geometry join failed: "
            f"missing_geometry={missing_geometry[:8]}, "
            f"step_mismatch={mismatched_steps[:8]}, "
            f"problem_mismatch={mismatched_problem[:8]}, "
            f"response_mismatch={mismatched_response[:8]}, "
            f"label_mismatch={mismatched_label[:8]}, "
            f"missing_requested_label={missing_requested_label[:8]}"
        )
    if not sequences:
        raise ValueError("trace and geometry artifacts have no aligned chains")
    return SequenceCollection(
        sequences=sequences,
        output_names=output_names,
        geometry_names=geometry.feature_names,
        geometry_groups=geometry.feature_groups,
        preflight={
            "num_trace_chains": int(trace.n_chains),
            "num_geometry_chains": int(len(geometry.chain_idx)),
            "num_joined_chains": int(len(sequences)),
            "num_errors": int(sum(sequence.is_error for sequence in sequences)),
            "output_features": int(len(output_names)),
            "geometry_features": int(len(geometry.feature_names)),
            "geometry": geometry.preflight,
            "trace_metadata": trace.metadata,
            "label_policy": label_policy,
        },
    )


def _column_summary(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError("prefix summary expects [time, feature]")
    finite = np.isfinite(array)
    count = finite.sum(axis=0)
    safe = np.where(finite, array, 0.0)
    mean = safe.sum(axis=0) / np.maximum(count, 1)
    variance = np.where(finite, (array - mean[None, :]) ** 2, 0.0).sum(axis=0)
    std = np.sqrt(variance / np.maximum(count, 1))
    last = array[-1].copy()
    if len(array) > 1:
        x = np.linspace(-1.0, 1.0, len(array))
        centered = np.where(finite, array - mean[None, :], 0.0)
        slope = (x[:, None] * centered).sum(axis=0) / max(float(np.dot(x, x)), 1e-12)
        jump = np.abs(np.diff(array, axis=0))
        jump = np.where(np.isfinite(jump), jump, -np.inf)
        max_jump = jump.max(axis=0)
        max_jump[~np.isfinite(max_jump)] = np.nan
    else:
        slope = np.zeros(array.shape[1], dtype=np.float64)
        max_jump = np.zeros(array.shape[1], dtype=np.float64)
    no_data = count == 0
    for part in (last, mean, std, slope, max_jump):
        part[no_data] = np.nan
    return np.concatenate([last, mean, std, slope, max_jump]).astype(np.float32)


def _summary_names(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        f"{stat}.{name}"
        for stat in ("last", "mean", "std", "slope", "max_jump")
        for name in names
    )


def build_response_tasks(
    collection: SequenceCollection,
    *,
    checkpoints: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
) -> dict[float, BinaryTask]:
    """Build online response-error tasks from strictly causal prefixes."""

    tasks: dict[float, BinaryTask] = {}
    for checkpoint in checkpoints:
        fraction = float(checkpoint)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("response checkpoints must lie in (0, 1]")
        x_output: list[np.ndarray] = []
        x_geometry: list[np.ndarray] = []
        labels: list[int] = []
        groups: list[int] = []
        chains: list[int] = []
        steps: list[int] = []
        nuisance: list[np.ndarray] = []
        for sequence in collection.sequences:
            stop = max(1, int(np.ceil(sequence.n_steps * fraction)))
            stop = min(stop, sequence.n_steps)
            controls = np.asarray(
                [
                    np.log1p(stop),
                    np.log1p(float(sequence.step_token_counts[:stop].sum())),
                    np.log1p(float(sequence.step_token_counts[:stop].mean())),
                ],
                dtype=np.float32,
            )
            x_output.append(
                np.concatenate([controls, _column_summary(sequence.output[:stop])])
            )
            x_geometry.append(_column_summary(sequence.geometry[:stop]))
            labels.append(int(sequence.is_error))
            groups.append(int(sequence.problem_id))
            chains.append(int(sequence.chain_idx))
            steps.append(int(stop - 1))
            nuisance.append(controls)
        tasks[fraction] = BinaryTask(
            name="response_error",
            checkpoint=fraction,
            x_output=np.stack(x_output).astype(np.float32),
            x_geometry=np.stack(x_geometry).astype(np.float32),
            y=np.asarray(labels, dtype=np.int8),
            groups=np.asarray(groups, dtype=np.int64),
            chain_idx=np.asarray(chains, dtype=np.int64),
            output_names=(
                "control.log1p_prefix_steps",
                "control.log1p_prefix_tokens",
                "control.log1p_mean_step_tokens",
            )
            + _summary_names(collection.output_names),
            geometry_names=_summary_names(collection.geometry_names),
            geometry_groups=tuple(
                group for _ in range(5) for group in collection.geometry_groups
            ),
            nuisance_values=np.stack(nuisance).astype(np.float32),
            step_idx=np.asarray(steps, dtype=np.int32),
        )
    return tasks


def build_online_response_task(
    collection: SequenceCollection,
    *,
    min_prefix_steps: int = 1,
) -> BinaryTask:
    """Build a deployable response-risk task over every observed prefix.

    Unlike relative checkpoints, this task never uses the response's eventual
    number of steps to choose a prefix. Repeated prefixes from one problem stay
    in the same outer fold and receive problem-balanced training/evaluation
    weights downstream.
    """

    if min_prefix_steps < 1:
        raise ValueError("min_prefix_steps must be positive")
    x_output: list[np.ndarray] = []
    x_geometry: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    chains: list[int] = []
    steps: list[int] = []
    nuisance: list[np.ndarray] = []
    for sequence in collection.sequences:
        for stop in range(int(min_prefix_steps), sequence.n_steps + 1):
            controls = np.asarray(
                [
                    np.log1p(stop),
                    np.log1p(float(sequence.step_token_counts[:stop].sum())),
                    np.log1p(float(sequence.step_token_counts[:stop].mean())),
                ],
                dtype=np.float32,
            )
            x_output.append(
                np.concatenate([controls, _column_summary(sequence.output[:stop])])
            )
            x_geometry.append(_column_summary(sequence.geometry[:stop]))
            labels.append(int(sequence.is_error))
            groups.append(int(sequence.problem_id))
            chains.append(int(sequence.chain_idx))
            steps.append(int(stop - 1))
            nuisance.append(controls)
    if not x_output:
        raise ValueError("no response has enough steps for the online prefix task")
    return BinaryTask(
        name="online_response_error",
        checkpoint=-1.0,
        x_output=np.stack(x_output).astype(np.float32),
        x_geometry=np.stack(x_geometry).astype(np.float32),
        y=np.asarray(labels, dtype=np.int8),
        groups=np.asarray(groups, dtype=np.int64),
        chain_idx=np.asarray(chains, dtype=np.int64),
        output_names=(
            "control.log1p_prefix_steps",
            "control.log1p_prefix_tokens",
            "control.log1p_mean_step_tokens",
        )
        + _summary_names(collection.output_names),
        geometry_names=_summary_names(collection.geometry_names),
        geometry_groups=tuple(
            group for _ in range(5) for group in collection.geometry_groups
        ),
        nuisance_values=np.stack(nuisance).astype(np.float32),
        step_idx=np.asarray(steps, dtype=np.int32),
    )


def _causal_history(values: np.ndarray, t: int, history: int) -> np.ndarray:
    blocks = [values[t - lag] for lag in range(history)]
    deltas = [values[t - lag] - values[t - lag - 1] for lag in range(history - 1)]
    return np.concatenate(blocks + deltas).astype(np.float32)


def _history_names(names: Sequence[str], history: int) -> tuple[str, ...]:
    levels = [f"lag{lag}.{name}" for lag in range(history) for name in names]
    deltas = [f"delta_lag{lag}.{name}" for lag in range(history - 1) for name in names]
    return tuple(levels + deltas)


def build_forecast_task(
    collection: SequenceCollection,
    *,
    history: int = 2,
    horizon: int = 1,
) -> ForecastTask:
    """Predict future changes in compact output state from current histories."""

    if history < 1 or horizon < 1:
        raise ValueError("history and horizon must be positive")
    target_indices = np.asarray(
        [i for i, name in enumerate(collection.output_names) if name.endswith(".last")],
        dtype=np.int64,
    )
    if len(target_indices) == 0:
        raise ValueError("output schema has no `.last` state features for forecasting")
    x_output: list[np.ndarray] = []
    x_geometry: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[int] = []
    chains: list[int] = []
    steps: list[int] = []
    nuisance: list[np.ndarray] = []
    for sequence in collection.sequences:
        first = max(history - 1, 1)
        last = sequence.n_steps - horizon
        for t in range(first, last):
            controls = np.asarray(
                [
                    np.log1p(t + 1),
                    np.log1p(float(sequence.step_token_counts[: t + 1].sum())),
                    np.log1p(float(sequence.step_token_counts[t])),
                ],
                dtype=np.float32,
            )
            x_output.append(
                np.concatenate([controls, _causal_history(sequence.output, t, history)])
            )
            x_geometry.append(_causal_history(sequence.geometry, t, history))
            targets.append(
                (
                    sequence.output[t + horizon, target_indices]
                    - sequence.output[t, target_indices]
                ).astype(np.float32)
            )
            groups.append(int(sequence.problem_id))
            chains.append(int(sequence.chain_idx))
            steps.append(int(t))
            nuisance.append(controls)
    if not x_output:
        raise ValueError(
            "no chain is long enough for the requested forecast history/horizon"
        )
    geometry_groups = tuple(
        group for _ in range(2 * history - 1) for group in collection.geometry_groups
    )
    return ForecastTask(
        name="future_output_delta",
        x_output=np.stack(x_output).astype(np.float32),
        x_geometry=np.stack(x_geometry).astype(np.float32),
        target=np.stack(targets).astype(np.float32),
        groups=np.asarray(groups, dtype=np.int64),
        chain_idx=np.asarray(chains, dtype=np.int64),
        step_idx=np.asarray(steps, dtype=np.int32),
        output_names=(
            "control.log1p_prefix_steps",
            "control.log1p_prefix_tokens",
            "control.log1p_current_step_tokens",
        )
        + _history_names(collection.output_names, history),
        geometry_names=_history_names(collection.geometry_names, history),
        geometry_groups=geometry_groups,
        target_names=tuple(collection.output_names[i] for i in target_indices),
        nuisance_values=np.stack(nuisance).astype(np.float32),
    )


def select_geometry_groups(task: BinaryTask | ForecastTask, groups: Sequence[str]):
    wanted = set(str(x) for x in groups)
    keep = np.asarray(
        [i for i, group in enumerate(task.geometry_groups) if group in wanted],
        dtype=np.int64,
    )
    if len(keep) == 0:
        raise ValueError(
            f"none of the requested geometry groups are present: {sorted(wanted)}"
        )
    return replace(
        task,
        x_geometry=task.x_geometry[:, keep],
        geometry_names=tuple(task.geometry_names[i] for i in keep),
        geometry_groups=tuple(task.geometry_groups[i] for i in keep),
    )


def select_output_tier(task: BinaryTask | ForecastTask, tier: str):
    """Select a frozen output-baseline saturation tier.

    ``scalar`` keeps controls and scalar uncertainty/confidence histories;
    ``distribution`` additionally keeps the complete-distribution count-sketch;
    ``full_compact`` keeps both distribution and top-k identity sketches.
    """

    tier = str(tier)
    if tier not in {"scalar", "distribution", "full_compact"}:
        raise ValueError(f"unknown output tier {tier!r}")
    keep: list[int] = []
    for i, name in enumerate(task.output_names):
        if tier == "full_compact":
            keep.append(i)
        elif "topk_count_sketch" in name:
            continue
        elif tier == "scalar" and "probability_count_sketch" in name:
            continue
        else:
            keep.append(i)
    if not keep:
        raise ValueError(f"output tier {tier!r} selected no features")
    index = np.asarray(keep, dtype=np.int64)
    return replace(
        task,
        x_output=task.x_output[:, index],
        output_names=tuple(task.output_names[i] for i in index),
    )
