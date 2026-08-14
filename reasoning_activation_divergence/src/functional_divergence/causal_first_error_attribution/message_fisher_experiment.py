from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FISHER_METRICS = (
    "largest_eigenvalue",
    "fisher_trace",
    "effective_rank",
    "anisotropy",
    "top_prompt_loading",
    "top_ffn_loading",
)


@dataclass(frozen=True)
class FirstErrorBoundaryPair:
    record_index: int
    chain_id: int
    first_error_step: int
    control_step: int
    control_decision_position: int
    event_decision_position: int


def first_error_boundary_pairs(
    trace_path: str | Path, *, max_cases: int, seed: int
) -> tuple[FirstErrorBoundaryPair, ...]:
    """Select adjacent pre-step boundaries without observing error-step tokens."""
    path = Path(trace_path)
    with np.load(path, allow_pickle=True) as archive:
        required = {
            "full_input_ids",
            "full_attention_mask",
            "step_token_ranges",
            "n_steps",
            "gold_error_step",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path}: trace is missing {missing}")
        inputs = np.asarray(archive["full_input_ids"])
        masks = np.asarray(archive["full_attention_mask"])
        ranges = np.asarray(archive["step_token_ranges"])
        step_counts = np.asarray(archive["n_steps"], dtype=np.int64).reshape(-1)
        errors = np.asarray(archive["gold_error_step"], dtype=np.int64).reshape(-1)
        chain_ids = (
            np.asarray(archive["chain_idx"], dtype=np.int64).reshape(-1)
            if "chain_idx" in archive.files
            else np.arange(len(errors), dtype=np.int64)
        )
    count = len(errors)
    if (
        inputs.ndim != 2
        or masks.shape != inputs.shape
        or ranges.ndim != 3
        or ranges.shape[0] != count
        or ranges.shape[2] != 2
        or step_counts.shape != (count,)
        or chain_ids.shape != (count,)
    ):
        raise ValueError("trace arrays are not record aligned")
    if isinstance(max_cases, (bool, np.bool_)) or int(max_cases) < 0:
        raise ValueError("max_cases must be a nonnegative integer")

    pairs = []
    for row, first_error in enumerate(errors):
        first_error = int(first_error)
        step_count = int(step_counts[row])
        if first_error < 1:
            continue
        if first_error >= step_count or step_count > ranges.shape[1]:
            raise ValueError(f"record {row}: first error lies outside step ranges")
        valid_count = int(np.sum(masks[row] == 1))
        if valid_count < 1 or not np.all(masks[row, :valid_count] == 1):
            raise ValueError(f"record {row}: invalid attention mask")
        control_step = first_error - 1
        control_position = int(ranges[row, control_step, 0]) - 1
        event_position = int(ranges[row, first_error, 0]) - 1
        if not (0 <= control_position < event_position < valid_count):
            raise ValueError(f"record {row}: invalid consecutive decision boundaries")
        pairs.append(
            FirstErrorBoundaryPair(
                record_index=row,
                chain_id=int(chain_ids[row]),
                first_error_step=first_error,
                control_step=control_step,
                control_decision_position=control_position,
                event_decision_position=event_position,
            )
        )
    generator = np.random.default_rng(int(seed))
    order = generator.permutation(len(pairs))
    selected = [pairs[int(index)] for index in order]
    if int(max_cases) > 0:
        selected = selected[: int(max_cases)]
    return tuple(selected)


def _paired_differences(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, np.ndarray]:
    grouped: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        role = str(row["boundary_role"])
        if role not in {"previous_correct", "first_error"}:
            raise ValueError(f"unsupported boundary role {role!r}")
        key = (str(row["domain"]), str(row["case_id"]), int(row["layer"]))
        if role in grouped[key]:
            raise ValueError(f"duplicate {role} row for {key}")
        value = float(row[metric])
        if not np.isfinite(value):
            raise ValueError(f"non-finite {metric} for {key}")
        grouped[key][role] = value
    by_domain_layer: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (domain, _case, layer), values in grouped.items():
        if set(values) != {"previous_correct", "first_error"}:
            raise ValueError("every case/layer requires both boundary roles")
        by_domain_layer[(domain, layer)].append(
            values["first_error"] - values["previous_correct"]
        )
    return {
        f"{domain}\t{layer}": np.asarray(values, dtype=np.float64)
        for (domain, layer), values in by_domain_layer.items()
    }


def _difference_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n_pairs": int(len(values)),
        "mean_event_minus_control": float(np.mean(values)),
        "median_event_minus_control": float(np.median(values)),
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def paired_fisher_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = DEFAULT_FISHER_METRICS,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize paired first-error changes with domain-stratified bootstrap."""
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    metrics = tuple(str(metric) for metric in metrics)
    if not metrics:
        raise ValueError("at least one Fisher metric is required")
    generator = np.random.default_rng(int(seed))
    result: dict[str, Any] = {"pooled": {}, "domains": {}}
    for metric in metrics:
        differences = _paired_differences(rows, metric)
        parsed = {
            tuple(key.split("\t", 1)): values for key, values in differences.items()
        }
        layers = sorted({int(layer) for _domain, layer in parsed})
        for layer in layers:
            domains = sorted(
                domain for domain, candidate in parsed if int(candidate) == layer
            )
            domain_values = [parsed[(domain, str(layer))] for domain in domains]
            pooled = np.concatenate(domain_values)
            estimates = np.empty(bootstrap_repeats, dtype=np.float64)
            for repeat in range(bootstrap_repeats):
                means = []
                for values in domain_values:
                    indices = generator.integers(0, len(values), size=len(values))
                    means.append(float(np.mean(values[indices])))
                estimates[repeat] = float(np.mean(means))
            entry = _difference_summary(pooled)
            entry["domain_macro_mean_event_minus_control"] = float(
                np.mean([np.mean(values) for values in domain_values])
            )
            entry["ci_low"] = float(np.quantile(estimates, 0.025))
            entry["ci_high"] = float(np.quantile(estimates, 0.975))
            result["pooled"].setdefault(str(layer), {})[metric] = entry
            for domain, values in zip(domains, domain_values):
                result["domains"].setdefault(domain, {}).setdefault(
                    str(layer), {}
                )[metric] = _difference_summary(values)
    return result


__all__ = [
    "DEFAULT_FISHER_METRICS",
    "FirstErrorBoundaryPair",
    "first_error_boundary_pairs",
    "paired_fisher_summary",
]
