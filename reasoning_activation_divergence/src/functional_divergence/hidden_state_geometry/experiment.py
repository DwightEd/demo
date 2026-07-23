from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np

from ..output import versioned_paths
from ..progress import NullProgress, ProgressReporter
from .contracts import HiddenGeometryDataset, TraceSource
from .data import load_hidden_geometry_dataset
from .evaluation import TaskEvaluation, evaluate_task
from .methods import load_builtin_methods
from .registry import method_spec, resolve_method_config
from .tasks import TaskDataset, build_strict_prefix_task, build_whole_chain_task


_TASK_BUILDERS = {
    "whole_chain": build_whole_chain_task,
    "strict_prefix": build_strict_prefix_task,
}


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False)
    for target in versioned_paths(path):
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(target)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty audit table: {path.name}")
    fields = list(rows[0])
    for target in versioned_paths(path):
        temporary = target.with_name(f".{target.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(target)


def _limit_dataset(
    data: HiddenGeometryDataset, maximum: int, seed: int
) -> HiddenGeometryDataset:
    if maximum <= 0:
        return data
    rng = np.random.default_rng(seed)
    retained: list[int] = []
    for domain in sorted(np.unique(data.domains)):
        indices = np.where(data.domains == domain)[0]
        if len(indices) > maximum:
            classes = {
                label: indices[data.labels[indices] == label] for label in (0, 1)
            }
            if maximum < 2 or any(len(values) == 0 for values in classes.values()):
                raise ValueError(
                    f"{domain}: smoke cap cannot retain both classes; "
                    f"maximum={maximum}, counts={[len(classes[0]), len(classes[1])]}"
                )
            event_count = int(round(maximum * len(classes[1]) / len(indices)))
            event_count = min(max(event_count, 1), maximum - 1, len(classes[1]))
            correct_count = maximum - event_count
            if correct_count > len(classes[0]):
                correct_count = len(classes[0])
                event_count = maximum - correct_count
            indices = np.sort(
                np.concatenate(
                    [
                        rng.choice(classes[0], size=correct_count, replace=False),
                        rng.choice(classes[1], size=event_count, replace=False),
                    ]
                )
            )
        retained.extend(int(index) for index in indices)
    retained = sorted(retained)
    return HiddenGeometryDataset(
        samples=tuple(data.samples[index] for index in retained),
        labels=data.labels[retained],
        evidence=replace(data.evidence, selected_records=len(retained)),
    )


def _manifest_count(source: TraceSource) -> int:
    with np.load(source.manifest, allow_pickle=True) as archive:
        for key in ("step_token_ranges", "gold_error_step", "chain_idx"):
            if key in archive.files:
                return int(len(np.asarray(archive[key])))
    raise ValueError(f"{source.manifest}: no record-aligned manifest array")


def _aligned_output_artifact(source: TraceSource) -> Path:
    with np.load(source.manifest, allow_pickle=True) as archive:
        if "step_scores" in archive.files:
            return source.manifest.resolve()
    return (source.exact_trace or source.manifest.parent / "trace.npz").resolve()


def _eligible_prefix_rows(samples: tuple) -> tuple[int, int]:
    rows = 0
    truncated = 0
    for sample in samples:
        if sample.first_error_step == 0:
            truncated += 1
        elif sample.first_error_step > 0:
            rows += sample.first_error_step
        else:
            rows += max(sample.n_steps - 1, 0)
    return rows, truncated


def _preflight(
    sources: tuple[TraceSource, ...],
    data: HiddenGeometryDataset,
    progress: ProgressReporter | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    reporter = progress or NullProgress()
    hash_owners: dict[str, str] = {}
    overlaps = []
    for sample in data.samples:
        if sample.problem_hash is None:
            continue
        owner = hash_owners.setdefault(sample.problem_hash, sample.dataset)
        if owner != sample.dataset:
            overlaps.append((sample.problem_hash, owner, sample.dataset))
    if overlaps:
        raise ValueError(
            "cross-domain problem overlap detected; deduplicate normalized problem text "
            f"before LODO, examples={overlaps[:5]}"
        )
    hashed_records = sum(sample.problem_hash is not None for sample in data.samples)
    signatures: dict[str, tuple[tuple[int, ...], int, tuple[int, ...]]] = {}
    global_schema: tuple[tuple[int, ...], int] | None = None
    tracked = reporter.track(
        data.samples, total=len(data.samples), description="preflight shards"
    )
    for sample in tracked:
        if not sample.state_path.is_file():
            raise FileNotFoundError(sample.state_path)
        shard = np.load(sample.state_path, mmap_mode="r", allow_pickle=False)
        if shard.ndim != 3 or shard.shape[1] != len(sample.layer_ids):
            raise ValueError(
                f"{sample.state_path}: expected [token,layer,hidden], got {shard.shape}"
            )
        if sample.state_count >= 0 and shard.shape[0] != sample.state_count:
            raise ValueError(f"{sample.state_path}: token count disagrees with manifest")
        schema = (tuple(sample.layer_ids.tolist()), int(shard.shape[-1]))
        if global_schema is None:
            global_schema = schema
        elif schema != global_schema:
            raise ValueError(
                "global hidden-state schema differs across domains or records: "
                f"expected={global_schema}, got={schema} at {sample.state_path}"
            )
        signatures.setdefault(
            sample.dataset,
            (schema[0], schema[1], tuple(int(value) for value in shard.shape)),
        )
    domains = []
    for source in sources:
        selected = tuple(sample for sample in data.samples if sample.dataset == source.dataset)
        if not selected:
            raise ValueError(f"{source.dataset}: no eligible records remain after filtering")
        layers = {tuple(sample.layer_ids.tolist()) for sample in selected}
        if len(layers) != 1:
            raise ValueError(f"{source.dataset}: stored layer grids differ across records")
        first = selected[0]
        _, hidden_dimension, first_shape = signatures[source.dataset]
        prefix_rows, truncated = _eligible_prefix_rows(selected)
        layer_ids = list(next(iter(layers)))
        output_artifact = _aligned_output_artifact(source)
        domains.append(
            {
                "dataset": source.dataset,
                "manifest": str(source.manifest.resolve()),
                "manifest_sha256": _sha256(source.manifest.resolve()),
                "aligned_output_artifact": str(output_artifact),
                "aligned_output_sha256": _sha256(output_artifact),
                "manifest_records": _manifest_count(source),
                "selected_records": len(selected),
                "error_records": sum(sample.first_error_step >= 0 for sample in selected),
                "correct_records": sum(sample.first_error_step == -1 for sample in selected),
                "problem_groups": len({sample.problem_group for sample in selected}),
                "problem_hash_records": sum(
                    sample.problem_hash is not None for sample in selected
                ),
                "layers": layer_ids,
                "depth_semantics": (
                    "adjacent_block"
                    if len(layer_ids) < 2 or np.all(np.diff(layer_ids) == 1)
                    else "sparse_depth_interval"
                ),
                "first_shard": str(first.state_path),
                "first_shard_shape": list(first_shape),
                "hidden_dimension": hidden_dimension,
                "response_generators": sorted({sample.generator for sample in selected}),
                "observer_models": sorted({sample.observer_model for sample in selected}),
                "strict_prefix_rows": prefix_rows,
                "left_truncated_step0_errors": truncated,
            }
        )
    return {
        "schema_version": "hidden_state_geometry_preflight_v1",
        "run_id": run_id or _new_run_id(),
        "shards_validated": len(data.samples),
        "problem_group_scheme": "dataset_qualified_explicit_local_group",
        "cross_domain_problem_hash_audit": {
            "status": (
                "complete"
                if hashed_records == len(data.samples)
                else "partial"
                if hashed_records
                else "unavailable"
            ),
            "hashed_records": hashed_records,
            "total_records": len(data.samples),
            "overlaps": 0,
        },
        "evidence": _jsonable(data.evidence),
        "domains": domains,
    }


def inspect_hidden_geometry_sources(
    *,
    sources: Iterable[TraceSource],
    response_generator: str,
    observer_model: str,
    output_features: Iterable[str] = ("token_entropy", "token_nll"),
    max_records_per_domain: int = 0,
    seed: int = 17,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Validate provenance and mmap-open every eligible shard without fitting."""
    source_list = tuple(sources)
    if not source_list:
        raise ValueError("at least one trace source is required")
    data = load_hidden_geometry_dataset(
        source_list,
        response_generator=response_generator,
        observer_model=observer_model,
        output_features=output_features,
    )
    return _preflight(
        source_list,
        _limit_dataset(data, int(max_records_per_domain), seed),
        progress,
        _new_run_id(),
    )


def _oof_rows(task: TaskDataset, result: TaskEvaluation) -> list[dict[str, Any]]:
    rows = []
    arms = sorted(result.probabilities)
    for index, example in enumerate(task.examples):
        sample = example.sample
        row: dict[str, Any] = {
            "task": task.name,
            "dataset": sample.dataset,
            "problem_group": sample.problem_group,
            "problem_hash": sample.problem_hash or "",
            "chain_id": sample.chain_id,
            "manifest_row": sample.manifest_row,
            "boundary_step": "" if example.boundary_step is None else example.boundary_step,
            "visible_steps": example.visible_steps,
            "label": int(task.labels[index]),
            "fold": int(result.fold_ids[index]),
        }
        row.update({name: float(result.probabilities[name][index]) for name in arms})
        rows.append(row)
    return rows


def _factor_payload(evaluations: dict[str, TaskEvaluation]) -> dict[str, np.ndarray]:
    return {
        f"{task}.{name}": np.asarray(values)
        for task, result in evaluations.items()
        for name, values in result.factors.items()
    }


def run_hidden_geometry_experiment(
    *,
    sources: Iterable[TraceSource],
    output_dir: str | Path,
    response_generator: str,
    observer_model: str,
    output_features: Iterable[str] = ("token_entropy", "token_nll"),
    tasks: Iterable[str] = ("whole_chain", "strict_prefix"),
    method_name: str = "raw_functional_probe",
    method_config: object = None,
    n_boot: int = 2000,
    seed: int = 17,
    progress: ProgressReporter | None = None,
    max_records_per_domain: int = 0,
) -> dict[str, Any]:
    """Run auditable held-domain tests without coupling to a concrete method."""
    source_list = tuple(sources)
    task_names = tuple(str(name) for name in tasks)
    if not source_list:
        raise ValueError("at least one trace source is required")
    if not task_names or len(set(task_names)) != len(task_names):
        raise ValueError("tasks must be a non-empty sequence without duplicates")
    unknown = sorted(set(task_names).difference(_TASK_BUILDERS))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}; available={sorted(_TASK_BUILDERS)}")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if max_records_per_domain < 0:
        raise ValueError("max_records_per_domain cannot be negative")

    reporter = progress or NullProgress()
    run_id = _new_run_id()
    load_builtin_methods()
    method_config = resolve_method_config(method_name, method_config)
    specification = method_spec(method_name)
    reporter.stage("load", f"{len(source_list)} domains")
    data = load_hidden_geometry_dataset(
        source_list,
        response_generator=response_generator,
        observer_model=observer_model,
        output_features=output_features,
    )
    data = _limit_dataset(data, int(max_records_per_domain), seed)
    preflight = _preflight(source_list, data, reporter, run_id)

    reporter.stage("build", ", ".join(task_names))
    built = {name: _TASK_BUILDERS[name](data.samples) for name in task_names}

    reporter.stage("evaluate", method_name)
    evaluations = {
        name: evaluate_task(
            task,
            method_name=method_name,
            method_config=method_config,
            n_boot=n_boot,
            seed=seed,
            progress=reporter,
        )
        for name, task in reporter.track(
            built.items(), total=len(built), description="analysis tasks"
        )
    }

    result = {
        "schema_version": "hidden_state_discriminative_geometry_v1",
        "run_id": run_id,
        "execution": {
            "seed": int(seed),
            "bootstrap_replicates": int(n_boot),
            "max_records_per_domain": int(max_records_per_domain),
        },
        "method": {
            "name": method_name,
            "config": _jsonable(method_config),
            "arm_definitions": dict(specification.arm_definitions),
            "contrasts": _jsonable(specification.contrasts),
            "randomizations": _jsonable(specification.randomizations),
        },
        "data": {
            "records": len(data.samples),
            "domains": sorted(np.unique(data.domains).tolist()),
            "response_generator": response_generator,
            "observer_model": observer_model,
            **_jsonable(data.evidence),
            "analysis_scope": (
                "supervised step-end trajectory discrimination; not a direct divergence, "
                "transport-spectrum, Jacobian, or Fisher estimator"
            ),
            "output_comparator_scope": (
                "stored entropy/NLL step summaries, not full-vocabulary logits"
            ),
            "known_limitation": (
                "raw hidden may encode reasoning content, domain, or problem difficulty; "
                "axis sensitivity alone does not identify geometric divergence"
            ),
            "claim_boundary": (
                "held-domain discriminative association only; functional or causal effect "
                "requires intervention, Jacobian/Fisher, or activation patching"
            ),
        },
        "tasks": {
            name: {
                "claim_scope": task.claim_scope,
                "rows": len(task.examples),
                "events": int(task.labels.sum()),
                "problem_groups": int(len(np.unique(task.groups))),
                "left_truncated_step0_errors": task.left_truncated_step0_errors,
                "estimand_population": (
                    "all eligible completed chains"
                    if name == "whole_chain"
                    else "step boundaries conditional on surviving step 0, no prior first "
                    "error, and an observed next reasoning step"
                ),
                "summary": evaluations[name].summary,
                "fold_diagnostics": evaluations[name].diagnostics,
            }
            for name, task in built.items()
        },
    }

    reporter.stage("write", str(Path(output_dir)))
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "preflight.json", preflight)
    _write_json(destination / "results.json", result)
    oof = [
        row
        for name, task in built.items()
        for row in _oof_rows(task, evaluations[name])
    ]
    audit = [
        {"task": name, **row}
        for name, evaluation in evaluations.items()
        for row in evaluation.fold_audit
    ]
    _write_csv(destination / "oof_predictions.csv", oof)
    _write_csv(destination / "fold_audit.csv", audit)
    factors = _factor_payload(evaluations)
    factors["__run_id__"] = np.asarray(run_id)
    if not factors:
        factors = {"empty": np.empty(0, dtype=np.float32)}
    for target in versioned_paths(destination / "model_factors.npz"):
        temporary = target.with_name(f".{target.name}.tmp.npz")
        np.savez_compressed(temporary, **factors)
        temporary.replace(target)
    artifact_names = (
        "preflight.json",
        "results.json",
        "oof_predictions.csv",
        "fold_audit.csv",
        "model_factors.npz",
    )
    _write_json(
        destination / "artifact_manifest.json",
        {
            "schema_version": "hidden_state_geometry_artifacts_v1",
            "run_id": run_id,
            "files": {
                name: _sha256(destination / name) for name in artifact_names
            },
        },
    )
    return _jsonable(result)
