from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .message_fisher import SourceMessageFisherResult, SourceMessageFisherRunner
from .trace_data import load_decision_prefix

DEFAULT_FISHER_METRICS = (
    "largest_eigenvalue",
    "fisher_trace",
    "effective_rank",
    "anisotropy",
    "top_prompt_loading",
    "top_ffn_loading",
    "mean_fisher_energy",
    "mean_euclidean_energy",
    "fisher_to_euclidean_trace_ratio",
)


@dataclass(frozen=True)
class FirstErrorBoundaryPair:
    record_index: int
    chain_id: int
    first_error_step: int
    control_step: int
    control_decision_position: int
    event_decision_position: int


@dataclass(frozen=True)
class FirstErrorReplayJob:
    domain: str
    trace_path: Path
    pair: FirstErrorBoundaryPair


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
        ranges = np.asarray(archive["step_token_ranges"], dtype=object)
        step_counts = np.asarray(archive["n_steps"], dtype=np.int64).reshape(-1)
        errors = np.asarray(archive["gold_error_step"], dtype=np.int64).reshape(-1)
        chain_ids = (
            np.asarray(archive["chain_idx"], dtype=np.int64).reshape(-1)
            if "chain_idx" in archive.files
            else np.arange(len(errors), dtype=np.int64)
        )
    count = len(errors)
    if (
        inputs.ndim < 1
        or masks.ndim < 1
        or ranges.ndim < 1
        or inputs.shape[0] != count
        or masks.shape[0] != count
        or ranges.shape[0] != count
        or step_counts.shape != (count,)
        or chain_ids.shape != (count,)
    ):
        shapes = ", ".join(
            (
                f"full_input_ids={inputs.shape}",
                f"full_attention_mask={masks.shape}",
                f"step_token_ranges={ranges.shape}",
                f"n_steps={step_counts.shape}",
                f"gold_error_step={errors.shape}",
                f"chain_idx={chain_ids.shape}",
            )
        )
        raise ValueError(f"{path}: trace arrays are not record aligned; {shapes}")
    if isinstance(max_cases, (bool, np.bool_)) or int(max_cases) < 0:
        raise ValueError("max_cases must be a nonnegative integer")

    pairs = []
    for row, first_error in enumerate(errors):
        first_error = int(first_error)
        step_count = int(step_counts[row])
        if first_error < 1:
            continue
        input_row = np.asarray(inputs[row], dtype=np.int64).reshape(-1)
        mask_row = np.asarray(masks[row], dtype=np.int64).reshape(-1)
        range_row = np.asarray(ranges[row], dtype=np.int64)
        if input_row.shape != mask_row.shape:
            raise ValueError(f"record {row}: token and mask shapes disagree")
        if range_row.ndim != 2 or range_row.shape[1] != 2:
            raise ValueError(f"record {row}: step ranges must have shape [step,2]")
        if first_error >= step_count or step_count > len(range_row):
            raise ValueError(f"record {row}: first error lies outside step ranges")
        valid_count = int(mask_row.sum())
        if (
            valid_count < 1
            or not np.isin(mask_row, (0, 1)).all()
            or not np.all(mask_row[:valid_count] == 1)
            or not np.all(mask_row[valid_count:] == 0)
        ):
            raise ValueError(f"record {row}: invalid attention mask")
        control_step = first_error - 1
        control_position = int(range_row[control_step, 0]) - 1
        event_position = int(range_row[first_error, 0]) - 1
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
        "n_pairs": len(values),
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
                result["domains"].setdefault(domain, {}).setdefault(str(layer), {})[
                    metric
                ] = _difference_summary(values)
    return result


@dataclass(frozen=True)
class MessageFisherExperimentConfig:
    data_root: Path
    domains: tuple[str, ...]
    layers: tuple[int, ...]
    output_dir: Path
    epsilon: float = 0.05
    perturbation_batch_size: int = 4
    max_cases_per_domain: int = 0
    bootstrap_repeats: int = 1000
    seed: int = 17
    reconstruction_tolerance: float = 3e-2

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.domains or any(not str(domain).strip() for domain in self.domains):
            raise ValueError("domains must be non-empty")
        if (
            not self.layers
            or min(self.layers) < 1
            or len(set(self.layers)) != len(self.layers)
        ):
            raise ValueError("layers must be unique positive one-based indices")
        if self.max_cases_per_domain < 0:
            raise ValueError("max_cases_per_domain must be nonnegative")
        if self.bootstrap_repeats < 1:
            raise ValueError("bootstrap_repeats must be positive")
        if not np.isfinite(self.reconstruction_tolerance) or not (
            0.0 < self.reconstruction_tolerance < 1.0
        ):
            raise ValueError("reconstruction_tolerance must lie in (0,1)")


def _save_fisher_artifact(
    path: Path,
    *,
    result: SourceMessageFisherResult,
    input_ids: np.ndarray,
    source_step_ids: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    result.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_ids": np.asarray(input_ids, dtype=np.int32),
        "source_step_ids": np.asarray(source_step_ids, dtype=np.int16),
        "layers": result.layers,
        "source_ids": result.source_ids,
        "direction_source_ids": result.direction_source_ids,
        "source_messages": result.source_messages.astype(np.float16),
        "attention_mass": result.attention_mass.astype(np.float32),
        "attention_output": result.attention_output.astype(np.float16),
        "mlp_output": result.mlp_output.astype(np.float16),
        "residual_pre": result.residual_pre.astype(np.float16),
        "residual_post": result.residual_post.astype(np.float16),
        "fisher_gram": result.fisher_gram.astype(np.float32),
        "euclidean_gram": result.euclidean_gram.astype(np.float32),
        "observed_symmetric_kl": result.observed_symmetric_kl.astype(np.float32),
        "predicted_quadratic_kl": result.predicted_quadratic_kl.astype(np.float32),
        "quadratic_relative_error": result.quadratic_relative_error.astype(np.float32),
        "attention_reconstruction_error": result.attention_reconstruction_error.astype(
            np.float32
        ),
        "block_reconstruction_error": result.block_reconstruction_error.astype(
            np.float32
        ),
        "baseline_entropy": np.asarray(result.baseline_entropy, dtype=np.float32),
        "epsilon": np.asarray(result.epsilon, dtype=np.float32),
        "metadata_json": np.asarray(json.dumps(dict(metadata), sort_keys=True)),
    }
    partial = path.with_suffix(".partial.npz")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    partial.replace(path)


class MessageFisherExperiment:
    """Run paired pre-error source-message Fisher measurements on ProcessBench."""

    def __init__(self, config: MessageFisherExperimentConfig) -> None:
        self.config = config

    def prepare(self) -> tuple[FirstErrorReplayJob, ...]:
        jobs = []
        for domain_index, domain in enumerate(self.config.domains):
            trace = (
                self.config.data_root
                / domain
                / "selected"
                / "trace.raw_residual_stream.npz"
            )
            if not trace.is_file():
                raise FileNotFoundError(trace)
            pairs = first_error_boundary_pairs(
                trace,
                max_cases=self.config.max_cases_per_domain,
                seed=self.config.seed + domain_index,
            )
            jobs.extend(
                FirstErrorReplayJob(domain=domain, trace_path=trace, pair=pair)
                for pair in pairs
            )
        if not jobs:
            raise ValueError("no first-error cases with a preceding correct step")
        return tuple(jobs)

    def run(
        self,
        model: object,
        jobs: Sequence[FirstErrorReplayJob] | None = None,
    ) -> dict[str, Any]:
        runner = SourceMessageFisherRunner(
            layers=self.config.layers,
            epsilon=self.config.epsilon,
            perturbation_batch_size=self.config.perturbation_batch_size,
        )
        jobs = self.prepare() if jobs is None else tuple(jobs)

        rows: list[dict[str, Any]] = []
        artifact_paths = []
        for job in tqdm(
            jobs, desc="source-message Fisher pairs", unit="pair"
        ):
            domain, trace, pair = job.domain, job.trace_path, job.pair
            case_id = f"{domain}_chain_{pair.chain_id}_row_{pair.record_index}"
            boundaries = (
                (
                    "previous_correct",
                    pair.control_step,
                    pair.control_decision_position,
                ),
                ("first_error", pair.first_error_step, pair.event_decision_position),
            )
            for role, target_step, decision_position in boundaries:
                prefix = load_decision_prefix(
                    trace,
                    record_index=pair.record_index,
                    decision_position=decision_position,
                )
                if prefix.first_error_step != pair.first_error_step:
                    raise ValueError(f"{case_id}: first-error metadata changed")
                result = runner.run(
                    model=model,
                    input_ids=prefix.input_ids,
                    source_step_ids=prefix.source_step_ids,
                )
                self._check_reconstruction(result, case_id=case_id, role=role)
                artifact = (
                    self.config.output_dir
                    / "artifacts"
                    / domain
                    / f"{case_id}.{role}.source_message_fisher_v1.npz"
                )
                metadata = {
                    "schema": "source_message_fisher_v1",
                    "case_id": case_id,
                    "domain": domain,
                    "record_index": pair.record_index,
                    "chain_id": pair.chain_id,
                    "replay_trace": str(trace),
                    "boundary_role": role,
                    "target_step": target_step,
                    "decision_position": decision_position,
                    "first_error_step": pair.first_error_step,
                    "future_tokens_used": False,
                    "coordinate_definition": (
                        "attention source-message coefficients plus local FFN coefficient"
                    ),
                }
                _save_fisher_artifact(
                    artifact,
                    result=result,
                    input_ids=prefix.input_ids,
                    source_step_ids=prefix.source_step_ids,
                    metadata=metadata,
                )
                artifact_paths.append(str(artifact))
                for diagnostic in result.summary_rows():
                    rows.append({**metadata, **diagnostic, "artifact": str(artifact)})

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        events_path = self.config.output_dir / "events.jsonl"
        events_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        paired = paired_fisher_summary(
            rows,
            metrics=DEFAULT_FISHER_METRICS,
            bootstrap_repeats=self.config.bootstrap_repeats,
            seed=self.config.seed,
        )
        report = {
            "schema": "source_message_fisher_experiment_v1",
            "paired_cases": len(jobs),
            "boundary_runs": 2 * len(jobs),
            "layers": list(self.config.layers),
            "epsilon": self.config.epsilon,
            "perturbation_batch_size": self.config.perturbation_batch_size,
            "max_cases_per_domain": self.config.max_cases_per_domain,
            "bootstrap_repeats": self.config.bootstrap_repeats,
            "artifacts": artifact_paths,
            "paired_summary": paired,
            "interpretation": (
                "descriptive functional geometry with intervention-locality audit; "
                "not evidence of a thermodynamic phase transition"
            ),
        }
        (self.config.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return report

    def _check_reconstruction(
        self, result: SourceMessageFisherResult, *, case_id: str, role: str
    ) -> None:
        attention_error = float(np.max(result.attention_reconstruction_error))
        block_error = float(np.max(result.block_reconstruction_error))
        if max(attention_error, block_error) > self.config.reconstruction_tolerance:
            raise ValueError(
                f"{case_id}/{role}: component reconstruction exceeded tolerance: "
                f"attention={attention_error:.6g}, block={block_error:.6g}"
            )


__all__ = [
    "DEFAULT_FISHER_METRICS",
    "FirstErrorBoundaryPair",
    "FirstErrorReplayJob",
    "MessageFisherExperiment",
    "MessageFisherExperimentConfig",
    "first_error_boundary_pairs",
    "paired_fisher_summary",
]
