from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .schema import (
    BASELINE_STEP_FEATURE_NAMES,
    DIRECTION_NAMES,
    CausalPullbackArtifact,
    CausalPullbackItem,
)


EPS = 1e-10


@dataclass(frozen=True)
class PullbackFeatureCollection:
    chain_idx: np.ndarray
    problem_ids: np.ndarray
    y_error: np.ndarray
    x_output: np.ndarray
    output_names: tuple[str, ...]
    x_field: np.ndarray
    field_names: tuple[str, ...]
    x_pullback: np.ndarray
    pullback_names: tuple[str, ...]
    pullback_groups: tuple[str, ...]
    nuisance: np.ndarray
    nuisance_names: tuple[str, ...]
    direct_scores: dict[str, np.ndarray]
    valid: np.ndarray


def _phase_bin(index: int, count: int, grid: int) -> int:
    if count <= 1:
        return 0
    phase = float(index) / float(count - 1)
    return min(int(grid) - 1, int(np.floor(phase * int(grid))))


def _bin_sequence(values: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    total = np.zeros(int(grid), dtype=np.float64)
    count = np.zeros(int(grid), dtype=np.float64)
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        slot = _phase_bin(index, len(values), grid)
        total[slot] += float(value)
        count[slot] += 1.0
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0.0)
    return mean, (count > 0.0).astype(np.float64)


def _bin_operator(
    values: np.ndarray,
    grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool a causal source-transition by future-step operator on a phase grid."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("operator must have shape [transition, output_step]")
    total = np.zeros((int(grid), int(grid)), dtype=np.float64)
    count = np.zeros_like(total)
    n_steps = int(matrix.shape[1])
    for transition in range(matrix.shape[0]):
        source_step = transition + 1
        source_bin = _phase_bin(source_step, n_steps, grid)
        for target_step in range(n_steps):
            value = matrix[transition, target_step]
            if not np.isfinite(value):
                continue
            target_bin = _phase_bin(target_step, n_steps, grid)
            total[source_bin, target_bin] += float(value)
            count[source_bin, target_bin] += 1.0
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0.0)
    return mean, (count > 0.0).astype(np.float64)


def _operator_summaries(values: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(matrix)
    local = matrix[finite]
    if local.size == 0:
        return {
            "mean": float("nan"),
            "max": float("nan"),
            "immediate": float("nan"),
            "long_range": float("nan"),
            "horizon": float("nan"),
            "spectral_concentration": float("nan"),
        }
    immediate = []
    long_range = []
    weighted_horizon = 0.0
    total_mass = 0.0
    for transition in range(matrix.shape[0]):
        source_step = transition + 1
        for target_step in range(matrix.shape[1]):
            value = matrix[transition, target_step]
            if not np.isfinite(value):
                continue
            lag = target_step - source_step
            if lag == 1:
                immediate.append(value)
            if lag >= 2:
                long_range.append(value)
            positive = max(float(value), 0.0)
            weighted_horizon += positive * max(lag, 0)
            total_mass += positive
    filled = np.where(finite, matrix, 0.0)
    total_energy = float(np.square(filled).sum())
    if total_energy <= EPS:
        concentration = 0.0
    else:
        # Deterministic power iteration avoids a LAPACK dependency for these
        # tiny, variable-size response operators.
        vector = np.ones(filled.shape[1], dtype=np.float64)
        vector /= max(float(np.linalg.norm(vector)), EPS)
        for _ in range(24):
            left = filled @ vector
            candidate = filled.T @ left
            norm = float(np.linalg.norm(candidate))
            if norm <= EPS:
                break
            vector = candidate / norm
        top_energy = float(np.square(filled @ vector).sum())
        concentration = top_energy / total_energy
    return {
        "mean": float(np.mean(local)),
        "max": float(np.max(local)),
        "immediate": float(np.mean(immediate)) if immediate else float("nan"),
        "long_range": float(np.mean(long_range)) if long_range else float("nan"),
        "horizon": float(weighted_horizon / max(total_mass, EPS)),
        "spectral_concentration": concentration,
    }


def _flatten_named(
    prefix: str,
    values: np.ndarray,
    names: list[str],
    output: list[float],
) -> None:
    array = np.asarray(values, dtype=np.float64)
    for index in np.ndindex(array.shape):
        suffix = ".".join(str(value) for value in index)
        names.append(f"{prefix}.{suffix}")
        output.append(float(array[index]))


def _item_features(
    item: CausalPullbackItem,
    *,
    phase_grid: int,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, float],
    bool,
]:
    output_values: list[float] = []
    output_names: list[str] = []
    baseline = np.asarray(item.baseline_step_features, dtype=np.float64)
    for column, feature in enumerate(BASELINE_STEP_FEATURE_NAMES):
        grid_value, coverage = _bin_sequence(baseline[:, column], phase_grid)
        _flatten_named(f"output.{feature}.phase", grid_value, output_names, output_values)
        output_names.extend(
            [
                f"output.{feature}.mean",
                f"output.{feature}.late",
                f"output.{feature}.max",
            ]
        )
        output_values.extend(
            [
                float(np.mean(baseline[:, column])),
                float(np.mean(baseline[len(baseline) // 2 :, column])),
                float(np.max(baseline[:, column])),
            ]
        )
        del coverage

    field_values: list[float] = []
    field_names: list[str] = []
    for name, values in (
        ("energy", item.field_energy),
        ("calibrated_energy", item.field_calibrated_energy),
        ("witness_norm", item.witness_norms[0]),
    ):
        grid_value, coverage = _bin_sequence(values, phase_grid)
        _flatten_named(f"field.{name}.phase", grid_value, field_names, field_values)
        _flatten_named(
            f"field.{name}.coverage", coverage, field_names, field_values
        )
        local = np.asarray(values, dtype=np.float64)
        field_names.extend((f"field.{name}.mean", f"field.{name}.max"))
        field_values.extend((float(np.nanmean(local)), float(np.nanmax(local))))

    fisher = np.asarray(item.fisher_transfer, dtype=np.float64)
    chosen = np.asarray(item.chosen_logprob_transfer, dtype=np.float64)
    entropy = np.asarray(item.entropy_transfer, dtype=np.float64)
    witness = np.asarray(item.witness_norms, dtype=np.float64)
    consequential = np.maximum(fisher, 0.0) * np.square(witness[:, :, None])
    chosen_effect = chosen * witness[:, :, None]
    entropy_effect = entropy * witness[:, :, None]

    pullback_values: list[float] = []
    pullback_names: list[str] = []
    pullback_groups: list[str] = []
    operator_grids: dict[str, np.ndarray] = {}
    for direction, direction_name in enumerate(DIRECTION_NAMES):
        for quantity_name, values, transform in (
            ("fisher", consequential[direction], lambda x: np.log1p(np.maximum(x, 0.0))),
            ("chosen", chosen_effect[direction], np.arcsinh),
            ("entropy", entropy_effect[direction], np.arcsinh),
        ):
            grid_value, grid_coverage = _bin_operator(transform(values), phase_grid)
            operator_grids[f"{direction_name}.{quantity_name}"] = grid_value
            before = len(pullback_names)
            _flatten_named(
                f"pullback.{direction_name}.{quantity_name}",
                grid_value,
                pullback_names,
                pullback_values,
            )
            _flatten_named(
                f"pullback.{direction_name}.{quantity_name}.coverage",
                grid_coverage,
                pullback_names,
                pullback_values,
            )
            pullback_groups.extend(
                [f"{direction_name}_{quantity_name}"]
                * (len(pullback_names) - before)
            )
        summaries = _operator_summaries(consequential[direction])
        for summary_name, value in summaries.items():
            pullback_names.append(
                f"pullback.{direction_name}.fisher.{summary_name}"
            )
            pullback_values.append(float(value))
            pullback_groups.append(f"{direction_name}_fisher")

    for null_name in ("shuffle", "random"):
        for quantity in ("fisher", "chosen", "entropy"):
            primary = operator_grids[f"field.{quantity}"]
            control = operator_grids[f"{null_name}.{quantity}"]
            contrast = primary - control
            before = len(pullback_names)
            _flatten_named(
                f"pullback.field_minus_{null_name}.{quantity}",
                contrast,
                pullback_names,
                pullback_values,
            )
            pullback_groups.extend(
                [f"field_minus_{null_name}_{quantity}"]
                * (len(pullback_names) - before)
            )

    half = np.asarray(item.primary_half_fisher_transfer, dtype=np.float64)
    primary = np.asarray(fisher[0], dtype=np.float64)
    finite_linearity = np.isfinite(primary) & np.isfinite(half)
    if np.any(finite_linearity):
        numerator = np.abs(primary[finite_linearity] - half[finite_linearity])
        denominator = np.maximum(
            np.maximum(np.abs(primary[finite_linearity]), np.abs(half[finite_linearity])),
            EPS,
        )
        linearity_error = float(np.median(numerator / denominator))
    else:
        linearity_error = float("nan")
    primary_summary = _operator_summaries(consequential[0])
    shuffle_summary = _operator_summaries(consequential[1])
    random_summary = _operator_summaries(consequential[2])
    direct = {
        "field_consequential_mean": primary_summary["mean"],
        "field_consequential_max": primary_summary["max"],
        "field_consequential_immediate": primary_summary["immediate"],
        "field_consequential_long_range": primary_summary["long_range"],
        "shuffle_consequential_mean": shuffle_summary["mean"],
        "random_consequential_mean": random_summary["mean"],
        "field_excess_over_shuffle": primary_summary["mean"]
        - shuffle_summary["mean"],
        "field_excess_over_random": primary_summary["mean"]
        - random_summary["mean"],
        "finite_difference_relative_error": linearity_error,
    }
    valid = bool(
        np.isfinite(primary).any()
        and np.nanmedian(item.replay_cosine) >= 0.0
        and np.isfinite(linearity_error)
    )
    return (
        np.asarray(output_values, dtype=np.float32),
        tuple(output_names),
        np.asarray(field_values, dtype=np.float32),
        tuple(field_names),
        np.asarray(pullback_values, dtype=np.float32),
        tuple(pullback_names),
        tuple(pullback_groups),
        direct,
        valid,
    )


def build_pullback_features(
    artifact: CausalPullbackArtifact,
    *,
    phase_grid: int = 4,
    replay_cosine_threshold: float | None = None,
) -> PullbackFeatureCollection:
    if int(phase_grid) < 2:
        raise ValueError("phase_grid must be at least two")
    if not artifact.items:
        raise ValueError("causal pullback artifact is empty")
    threshold = (
        float(replay_cosine_threshold)
        if replay_cosine_threshold is not None
        else float(artifact.metadata.get("config", {}).get("replay_cosine_threshold", 0.98))
    )
    rows = [_item_features(item, phase_grid=int(phase_grid)) for item in artifact.items]
    output_names = rows[0][1]
    field_names = rows[0][3]
    pullback_names = rows[0][5]
    pullback_groups = rows[0][6]
    if any(row[1] != output_names or row[3] != field_names or row[5] != pullback_names for row in rows):
        raise RuntimeError("feature schema changed across responses")

    direct_names = tuple(rows[0][7])
    direct_scores = {
        name: np.asarray([row[7][name] for row in rows], dtype=np.float64)
        for name in direct_names
    }
    median_replay = np.asarray(
        [np.nanmedian(item.replay_cosine) for item in artifact.items], dtype=np.float64
    )
    finite_difference = direct_scores["finite_difference_relative_error"]
    valid = np.asarray([row[8] for row in rows], dtype=bool)
    valid &= median_replay >= threshold
    valid &= np.isfinite(finite_difference)

    n_steps = np.asarray([item.n_steps for item in artifact.items], dtype=np.float64)
    response_chars = np.asarray(
        [item.response_chars for item in artifact.items], dtype=np.float64
    )
    donor_count = np.asarray(
        [item.donor_count for item in artifact.items], dtype=np.float64
    )
    nuisance = np.column_stack(
        [
            np.log1p(n_steps),
            np.log1p(response_chars),
            donor_count,
            median_replay,
            finite_difference,
        ]
    ).astype(np.float32)
    return PullbackFeatureCollection(
        chain_idx=np.asarray([item.chain_idx for item in artifact.items], dtype=np.int64),
        problem_ids=np.asarray(
            [item.problem_id for item in artifact.items], dtype=np.int64
        ),
        y_error=1
        - np.asarray([item.is_correct for item in artifact.items], dtype=np.int8),
        x_output=np.stack([row[0] for row in rows]),
        output_names=output_names,
        x_field=np.stack([row[2] for row in rows]),
        field_names=field_names,
        x_pullback=np.stack([row[4] for row in rows]),
        pullback_names=pullback_names,
        pullback_groups=pullback_groups,
        nuisance=nuisance,
        nuisance_names=(
            "log1p_n_steps",
            "log1p_response_chars",
            "donor_count",
            "median_replay_cosine",
            "finite_difference_relative_error",
        ),
        direct_scores=direct_scores,
        valid=valid,
    )
