from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ..progress import NullProgress, ProgressReporter
from .method import FoldInput
from .methods import load_builtin_methods
from .preprocessing import group_balanced_weights
from .registry import (
    ContrastSpec,
    RandomizationSpec,
    create_method,
    method_spec,
)
from .tasks import TaskDataset


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    held_domain: str
    train: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class TaskEvaluation:
    task_name: str
    probabilities: dict[str, np.ndarray]
    fold_ids: np.ndarray
    summary: dict[str, Any]
    fold_audit: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]
    factors: dict[str, np.ndarray]
    arm_definitions: dict[str, str]
    contrasts: tuple[ContrastSpec, ...]
    randomizations: tuple[RandomizationSpec, ...]


def _validate_declared_arms(
    names: set[str],
    contrasts: tuple[ContrastSpec, ...],
    randomizations: tuple[RandomizationSpec, ...],
) -> None:
    missing_contrasts = {
        specification.name: sorted(
            {specification.baseline, specification.candidate}.difference(names)
        )
        for specification in contrasts
        if not {specification.baseline, specification.candidate}.issubset(names)
    }
    if missing_contrasts:
        raise ValueError(
            f"method omitted declared contrast arms: {missing_contrasts}"
        )
    invalid_randomizations = {}
    for specification in randomizations:
        missing = []
        if specification.candidate not in names:
            missing.append(f"candidate={specification.candidate}")
        if not any(name.startswith(specification.baseline_prefix) for name in names):
            missing.append(f"null_prefix={specification.baseline_prefix}")
        if missing:
            invalid_randomizations[specification.name] = missing
    if invalid_randomizations:
        raise ValueError(
            "method omitted declared randomization arms: "
            f"{invalid_randomizations}"
        )


def lodo_splits(task: TaskDataset) -> tuple[FoldSplit, ...]:
    domains = np.unique(task.domains)
    if len(domains) < 2:
        raise ValueError("LODO requires at least two datasets")
    splits = []
    for fold, domain in enumerate(sorted(str(value) for value in domains)):
        test = np.where(task.domains == domain)[0]
        train = np.where(task.domains != domain)[0]
        if len(np.unique(task.labels[train])) != 2:
            raise ValueError(f"held domain {domain}: training rows do not contain both classes")
        if len(np.unique(task.labels[test])) != 2:
            raise ValueError(f"held domain {domain}: test rows do not contain both classes")
        train_hashes = {value for value in task.problem_hashes[train] if value}
        test_hashes = {value for value in task.problem_hashes[test] if value}
        if train_hashes.intersection(test_hashes):
            raise RuntimeError(f"held domain {domain}: problem-hash leakage")
        splits.append(FoldSplit(fold, domain, train, test))
    return tuple(splits)


def _loss(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    y = np.asarray(labels, dtype=np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def _metrics(labels: np.ndarray, probability: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    weights = group_balanced_weights(groups)
    two_classes = len(np.unique(labels)) == 2
    return {
        "auroc": float(roc_auc_score(labels, probability, sample_weight=weights))
        if two_classes
        else float("nan"),
        "auprc": float(average_precision_score(labels, probability, sample_weight=weights))
        if two_classes
        else float("nan"),
        "nll_nats": float(np.average(_loss(labels, probability), weights=weights)),
        "brier": float(brier_score_loss(labels, probability, sample_weight=weights)),
    }


def _macro_increment_bootstrap(
    labels: np.ndarray,
    domains: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    progress: ProgressReporter | None = None,
) -> dict[str, float | int]:
    difference = _loss(labels, baseline) - _loss(labels, candidate)
    domain_groups: dict[str, list[float]] = {}
    for domain in np.unique(domains):
        values = []
        mask = domains == domain
        for group in np.unique(groups[mask]):
            values.append(float(np.mean(difference[mask & (groups == group)])))
        domain_groups[str(domain)] = values
    point = float(np.mean([np.mean(values) for values in domain_groups.values()]))
    rng = np.random.default_rng(seed)
    bootstrap = []
    reporter = progress or NullProgress()
    iterations = reporter.track(
        range(int(n_boot)), total=int(n_boot), description="problem-group bootstrap"
    )
    for _ in iterations:
        domain_values = []
        for values in domain_groups.values():
            array = np.asarray(values)
            domain_values.append(float(np.mean(rng.choice(array, size=len(array), replace=True))))
        bootstrap.append(float(np.mean(domain_values)))
    return {
        "point": point,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "n_boot": int(n_boot),
        "inference_scope": "conditional_test_problem_group_cluster_bootstrap",
    }


def _macro_increment_point(
    labels: np.ndarray,
    domains: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> float:
    difference = _loss(labels, baseline) - _loss(labels, candidate)
    domain_points = []
    for domain in np.unique(domains):
        mask = domains == domain
        group_points = [
            float(np.mean(difference[mask & (groups == group)]))
            for group in np.unique(groups[mask])
        ]
        domain_points.append(float(np.mean(group_points)))
    return float(np.mean(domain_points))


def summarize_predictions(
    labels: np.ndarray,
    domains: np.ndarray,
    groups: np.ndarray,
    probabilities: dict[str, np.ndarray],
    *,
    n_boot: int,
    seed: int,
    progress: ProgressReporter | None = None,
    contrasts: tuple[ContrastSpec, ...] = (),
    randomizations: tuple[RandomizationSpec, ...] = (),
    visible_steps: np.ndarray | None = None,
) -> dict[str, Any]:
    _validate_declared_arms(set(probabilities), contrasts, randomizations)
    arms = {}
    for name, values in probabilities.items():
        per_domain = {
            str(domain): _metrics(labels[domains == domain], values[domains == domain], groups[domains == domain])
            for domain in np.unique(domains)
        }
        keys = next(iter(per_domain.values())).keys()
        arms[name] = {
            "overall": _metrics(labels, values, groups),
            "macro": {
                key: float(np.nanmean([metric[key] for metric in per_domain.values()]))
                for key in keys
            },
            "per_domain": per_domain,
        }
    increments = {
        contrast.name: _macro_increment_bootstrap(
            labels,
            domains,
            groups,
            probabilities[contrast.baseline],
            probabilities[contrast.candidate],
            n_boot=n_boot,
            seed=seed + index,
            progress=progress,
        )
        for index, contrast in enumerate(contrasts)
        if contrast.baseline in probabilities and contrast.candidate in probabilities
    }
    randomization_checks = {}
    for specification in randomizations:
        null_arms = sorted(
            name for name in probabilities if name.startswith(specification.baseline_prefix)
        )
        mask = np.ones(len(labels), dtype=bool)
        if specification.minimum_visible_steps > 1:
            if visible_steps is None:
                raise ValueError("visible_steps are required for this randomization stratum")
            mask = np.asarray(visible_steps) >= specification.minimum_visible_steps
        if not np.any(mask):
            randomization_checks[specification.name] = {
                "status": "not_identifiable",
                "candidate": specification.candidate,
                "null_arms": null_arms,
                "identifiable_rows": 0,
                "minimum_visible_steps": specification.minimum_visible_steps,
                "estimand_scope": "exploratory_axis_randomization_diagnostic",
                "training_scope": "all_outer_training_rows_not_stratum_refit",
                "claim_status": "sensitivity_only_not_axis_order_evidence",
            }
            continue
        seed_points = [
            _macro_increment_point(
                labels[mask],
                domains[mask],
                groups[mask],
                probabilities[name][mask],
                probabilities[specification.candidate][mask],
            )
            for name in null_arms
        ]
        randomization_checks[specification.name] = {
            "status": "available",
            "candidate": specification.candidate,
            "null_arms": null_arms,
            "identifiable_rows": int(mask.sum()),
            "minimum_visible_steps": specification.minimum_visible_steps,
            "n_randomizations": len(seed_points),
            "seed_points": seed_points,
            "mean": float(np.mean(seed_points)),
            "minimum": float(np.min(seed_points)),
            "maximum": float(np.max(seed_points)),
            "uncertainty_kind": "randomization_seed_sensitivity_not_ci",
            "estimand_scope": "exploratory_axis_randomization_diagnostic",
            "training_scope": "all_outer_training_rows_not_stratum_refit",
            "claim_status": "sensitivity_only_not_axis_order_evidence",
        }
    return {
        "arms": arms,
        "increments": increments,
        "randomization_checks": randomization_checks,
    }


def evaluate_task(
    task: TaskDataset,
    *,
    method_name: str,
    method_config: object,
    n_boot: int,
    seed: int,
    progress: ProgressReporter | None = None,
) -> TaskEvaluation:
    load_builtin_methods()
    specification = method_spec(method_name)
    reporter = progress or NullProgress()
    reporter.stage("evaluate", task.name)
    splits = lodo_splits(task)
    probabilities: dict[str, np.ndarray] = {}
    fold_ids = np.full(len(task.examples), -1, dtype=np.int16)
    audit = []
    diagnostics = []
    factors: dict[str, np.ndarray] = {}
    for split in reporter.track(splits, total=len(splits), description="LODO domains"):
        method = create_method(method_name, method_config)
        result = method.fit_predict(
            FoldInput(
                task_name=task.name,
                train_examples=tuple(task.examples[index] for index in split.train),
                train_labels=task.labels[split.train],
                train_groups=task.groups[split.train],
                test_examples=tuple(task.examples[index] for index in split.test),
                seed=seed + split.fold,
                progress=reporter,
            )
        )
        _validate_declared_arms(
            set(result.probabilities),
            specification.contrasts,
            specification.randomizations,
        )
        if not probabilities:
            probabilities = {
                name: np.full(len(task.examples), np.nan, dtype=np.float64)
                for name in result.probabilities
            }
        if set(result.probabilities) != set(probabilities):
            raise RuntimeError("method returned inconsistent prediction arms across folds")
        for name, values in result.probabilities.items():
            if values.shape != (len(split.test),):
                raise ValueError(f"{name}: prediction count disagrees with held fold")
            if (
                not np.isfinite(values).all()
                or np.any(values < 0.0)
                or np.any(values > 1.0)
            ):
                raise ValueError(
                    f"{name}: probabilities must be finite and lie in [0, 1]"
                )
            probabilities[name][split.test] = values
        fold_ids[split.test] = split.fold
        audit.append(
            {
                "fold": split.fold,
                "held_domain": split.held_domain,
                "train_rows": int(len(split.train)),
                "test_rows": int(len(split.test)),
                "train_groups": int(len(np.unique(task.groups[split.train]))),
                "test_groups": int(len(np.unique(task.groups[split.test]))),
                "train_events": int(task.labels[split.train].sum()),
                "test_events": int(task.labels[split.test].sum()),
            }
        )
        diagnostics.append({"fold": split.fold, **result.diagnostics})
        for name, values in result.factors.items():
            factors[f"fold_{split.fold}.{name}"] = np.asarray(values)
    if np.any(fold_ids < 0) or any(not np.isfinite(values).all() for values in probabilities.values()):
        raise RuntimeError("LODO did not produce one finite prediction per eligible row")
    reporter.stage("bootstrap", task.name)
    summary = summarize_predictions(
        task.labels,
        task.domains,
        task.groups,
        probabilities,
        n_boot=n_boot,
        seed=seed,
        progress=reporter,
        contrasts=specification.contrasts,
        randomizations=specification.randomizations,
        visible_steps=np.asarray(
            [example.visible_steps for example in task.examples], dtype=np.int64
        ),
    )
    return TaskEvaluation(
        task_name=task.name,
        probabilities=probabilities,
        fold_ids=fold_ids,
        summary=summary,
        fold_audit=tuple(audit),
        diagnostics=tuple(diagnostics),
        factors=factors,
        arm_definitions=dict(specification.arm_definitions),
        contrasts=specification.contrasts,
        randomizations=specification.randomizations,
    )
