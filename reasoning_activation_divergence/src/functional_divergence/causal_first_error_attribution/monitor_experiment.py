from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from .evaluation import MonitorRow, localization_metrics
from .monitor_data import (
    MonitorBoundaryRow,
    ProcessBenchMonitorData,
    load_processbench_monitor_data,
)
from .monitor_training import (
    MONITOR_ARMS,
    MonitorTrainingConfig,
    build_inner_group_split,
    fit_state_normalizer,
    train_monitor_arm,
)


@dataclass(frozen=True)
class MonitorExperimentConfig:
    data_root: Path
    domains: tuple[str, ...]
    output_dir: Path
    arms: tuple[str, ...] = MONITOR_ARMS
    output_features: tuple[str, ...] = ("token_entropy", "token_nll")
    max_chains_per_domain: int = 0
    validation_fraction: float = 0.15
    target_correct_chain_false_alarm: float = 0.1
    bootstrap_repeats: int = 1000
    seed: int = 17
    training: MonitorTrainingConfig = MonitorTrainingConfig()

    def __post_init__(self) -> None:
        unknown = sorted(set(self.arms).difference(MONITOR_ARMS))
        if unknown:
            raise ValueError(f"unknown monitor arms: {unknown}")
        if len(self.domains) < 2:
            raise ValueError("LODO monitoring requires at least two domains")
        if not 0.0 <= self.target_correct_chain_false_alarm < 1.0:
            raise ValueError("target false-alarm rate must lie in [0,1)")
        if self.bootstrap_repeats < 1 or self.max_chains_per_domain < 0:
            raise ValueError("bootstrap count must be positive and chain limit nonnegative")


def correct_chain_threshold(
    rows: Sequence[MonitorBoundaryRow],
    indices: np.ndarray,
    scores: np.ndarray,
    *,
    target_false_alarm_rate: float,
) -> float:
    """Freeze a threshold from validation correct-chain maximum scores."""

    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    probability = np.asarray(scores, dtype=np.float64).reshape(-1)
    if selected.shape != probability.shape:
        raise ValueError("indices and scores must be aligned")
    maxima: dict[str, float] = {}
    for index, score in zip(selected, probability):
        row = rows[int(index)]
        if row.first_error_step == -1:
            maxima[row.chain_id] = max(maxima.get(row.chain_id, -np.inf), float(score))
    if not maxima:
        return 0.5
    values = np.sort(np.asarray(list(maxima.values()), dtype=np.float64))[::-1]
    allowed = int(np.floor(float(target_false_alarm_rate) * len(values)))
    if allowed == 0:
        return float(np.nextafter(values[0], np.inf))
    return float(values[allowed - 1])


def _problem_balanced_weights(
    rows: Sequence[MonitorBoundaryRow], indices: np.ndarray
) -> np.ndarray:
    selected = [int(value) for value in np.asarray(indices, dtype=np.int64)]
    groups: dict[str, list[int]] = defaultdict(list)
    for position, index in enumerate(selected):
        groups[rows[index].problem_hash].append(position)
    result = np.zeros(len(selected), dtype=np.float64)
    for positions in groups.values():
        result[positions] = 1.0 / (len(groups) * len(positions))
    return result


def evaluate_boundary_scores(
    rows: Sequence[MonitorBoundaryRow],
    indices: np.ndarray,
    scores: np.ndarray,
    *,
    false_alarm_threshold: float,
) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    probability = np.asarray(scores, dtype=np.float64).reshape(-1)
    if selected.shape != probability.shape or not np.isfinite(probability).all():
        raise ValueError("indices and finite scores must be aligned")
    labels = np.asarray([rows[int(index)].label for index in selected], dtype=np.int8)
    weights = _problem_balanced_weights(rows, selected)
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    nll = -np.sum(
        weights * (labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
    )
    monitor_rows = [
        MonitorRow(
            chain_id=rows[int(index)].chain_id,
            problem_hash=rows[int(index)].problem_hash,
            sibling_group=rows[int(index)].sibling_group,
            domain=rows[int(index)].domain,
            candidate_step=rows[int(index)].candidate_step,
            first_error_step=rows[int(index)].first_error_step,
            score=float(score),
        )
        for index, score in zip(selected, probability)
    ]
    report = localization_metrics(
        monitor_rows, false_alarm_threshold=float(false_alarm_threshold)
    )
    report.update(
        {
            "rows": len(selected),
            "events": int(labels.sum()),
            "problem_groups": len({rows[int(index)].problem_hash for index in selected}),
            "row_nll": float(nll),
            "row_auroc": (
                float(roc_auc_score(labels, probability, sample_weight=weights))
                if len(np.unique(labels)) == 2
                else float("nan")
            ),
            "row_auprc": (
                float(
                    average_precision_score(labels, probability, sample_weight=weights)
                )
                if len(np.unique(labels)) == 2
                else float("nan")
            ),
            "false_alarm_threshold": float(false_alarm_threshold),
        }
    )
    return report


def _chain_rank(
    rows: Sequence[MonitorBoundaryRow], positions: list[int], scores: np.ndarray
) -> tuple[float, float] | None:
    gold = rows[positions[0]].first_error_step
    if gold == -1:
        return None
    ordered = sorted(
        positions,
        key=lambda position: (-float(scores[position]), rows[position].candidate_step),
    )
    rank = next(
        rank
        for rank, position in enumerate(ordered, start=1)
        if rows[position].candidate_step == gold
    )
    return float(rank == 1), 1.0 / rank


def _problem_contrasts(
    rows: Sequence[MonitorBoundaryRow],
    indices: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    candidate_thresholds: dict[str, float],
    baseline_thresholds: dict[str, float],
) -> dict[str, dict[str, dict[str, float]]]:
    selected_rows = [rows[int(index)] for index in indices]
    labels = np.asarray([row.label for row in selected_rows], dtype=np.int8)
    candidate = np.clip(np.asarray(candidate, dtype=np.float64), 1e-7, 1 - 1e-7)
    baseline = np.clip(np.asarray(baseline, dtype=np.float64), 1e-7, 1 - 1e-7)
    candidate_loss = -(labels * np.log(candidate) + (1 - labels) * np.log1p(-candidate))
    baseline_loss = -(labels * np.log(baseline) + (1 - labels) * np.log1p(-baseline))
    by_problem: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(selected_rows):
        by_problem[row.problem_hash].append(position)
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for problem, positions in by_problem.items():
        domain = selected_rows[positions[0]].domain
        values: dict[str, float] = {
            "nll_improvement": float(
                np.mean(baseline_loss[positions] - candidate_loss[positions])
            )
        }
        by_chain: dict[str, list[int]] = defaultdict(list)
        for position in positions:
            by_chain[selected_rows[position].chain_id].append(position)
        top1_deltas = []
        mrr_deltas = []
        false_alarm_deltas = []
        for chain_positions in by_chain.values():
            candidate_rank = _chain_rank(selected_rows, chain_positions, candidate)
            baseline_rank = _chain_rank(selected_rows, chain_positions, baseline)
            if candidate_rank is not None and baseline_rank is not None:
                top1_deltas.append(candidate_rank[0] - baseline_rank[0])
                mrr_deltas.append(candidate_rank[1] - baseline_rank[1])
            else:
                candidate_alarm = any(
                    candidate[position] >= candidate_thresholds[domain]
                    for position in chain_positions
                )
                baseline_alarm = any(
                    baseline[position] >= baseline_thresholds[domain]
                    for position in chain_positions
                )
                false_alarm_deltas.append(float(baseline_alarm) - float(candidate_alarm))
        values["top1_gain"] = (
            float(np.mean(top1_deltas)) if top1_deltas else float("nan")
        )
        values["mrr_gain"] = (
            float(np.mean(mrr_deltas)) if mrr_deltas else float("nan")
        )
        values["correct_chain_fpr_reduction"] = (
            float(np.mean(false_alarm_deltas))
            if false_alarm_deltas
            else float("nan")
        )
        result[domain][problem] = values
    return dict(result)


def _domain_problem_mean(
    values: dict[str, dict[str, dict[str, float]]], metric: str
) -> float:
    domain_means = []
    for problems in values.values():
        observed = [item[metric] for item in problems.values() if np.isfinite(item[metric])]
        if observed:
            domain_means.append(float(np.mean(observed)))
    return float(np.mean(domain_means)) if domain_means else float("nan")


def paired_problem_bootstrap(
    rows: Sequence[MonitorBoundaryRow],
    indices: np.ndarray,
    *,
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_thresholds: dict[str, float],
    baseline_thresholds: dict[str, float],
    repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Paired domain→problem bootstrap for graph-vs-control increments."""

    values = _problem_contrasts(
        rows,
        np.asarray(indices, dtype=np.int64),
        candidate_scores,
        baseline_scores,
        candidate_thresholds,
        baseline_thresholds,
    )
    domains = sorted(values)
    rng = np.random.default_rng(int(seed))
    metrics = (
        "nll_improvement",
        "top1_gain",
        "mrr_gain",
        "correct_chain_fpr_reduction",
    )
    bootstrap: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(int(repeats)):
        sampled_domains = rng.choice(domains, size=len(domains), replace=True)
        replicate: dict[str, list[float]] = {metric: [] for metric in metrics}
        for sampled_position, domain in enumerate(sampled_domains):
            problem_values = values[str(domain)]
            names = list(problem_values)
            sampled_names = rng.choice(names, size=len(names), replace=True)
            for metric in metrics:
                observed = [
                    problem_values[str(name)][metric]
                    for name in sampled_names
                    if np.isfinite(problem_values[str(name)][metric])
                ]
                if observed:
                    replicate[metric].append(float(np.mean(observed)))
        for metric in metrics:
            if replicate[metric]:
                bootstrap[metric].append(float(np.mean(replicate[metric])))
    report = {}
    for metric in metrics:
        samples = np.asarray(bootstrap[metric], dtype=np.float64)
        point = _domain_problem_mean(values, metric)
        report[metric] = {
            "point": point,
            "ci_low": (
                float(np.quantile(samples, 0.025)) if samples.size else float("nan")
            ),
            "ci_high": (
                float(np.quantile(samples, 0.975)) if samples.size else float("nan")
            ),
            "bootstrap_repeats": int(samples.size),
        }
    return report


def _json_config(config: MonitorExperimentConfig) -> dict[str, Any]:
    result = asdict(config)
    result["data_root"] = str(config.data_root)
    result["output_dir"] = str(config.output_dir)
    return result


class ProcessBenchMonitorExperiment:
    """Train and evaluate the residual-depth monitor with outer LODO splits."""

    def __init__(self, config: MonitorExperimentConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        config = self.config
        data = load_processbench_monitor_data(
            config.data_root,
            config.domains,
            output_features=config.output_features,
            max_chains_per_domain=config.max_chains_per_domain,
        )
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(_json_config(config), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        row_domains = np.asarray([row.domain for row in data.rows], dtype=object)
        global_scores = {
            arm: np.full(len(data.rows), np.nan, dtype=np.float64)
            for arm in config.arms
        }
        thresholds: dict[str, dict[str, float]] = {arm: {} for arm in config.arms}
        folds: dict[str, dict[str, Any]] = {}
        prediction_lines: list[str] = []
        for fold_number, held_domain in enumerate(
            tqdm(config.domains, desc="LODO folds", unit="domain")
        ):
            outer_train = np.flatnonzero(row_domains != held_domain)
            test = np.flatnonzero(row_domains == held_domain)
            train, validation = build_inner_group_split(
                data.rows,
                outer_train,
                validation_fraction=config.validation_fraction,
                seed=config.seed + fold_number,
            )
            print(
                f"[{held_domain}] train={len(train)} validation={len(validation)} "
                f"test={len(test)}"
            )
            state_normalizer = fit_state_normalizer(data, train)
            fold_report: dict[str, Any] = {
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "arms": {},
            }
            for arm in config.arms:
                trained = train_monitor_arm(
                    data,
                    train,
                    validation,
                    arm=arm,
                    config=config.training,
                    seed=config.seed + fold_number,
                    state_normalizer=(
                        state_normalizer
                        if arm not in ("nuisance", "output_history")
                        else None
                    ),
                )
                validation_scores = trained.predict(
                    data,
                    validation,
                    batch_size=config.training.batch_size,
                    device=config.training.device,
                )
                threshold = correct_chain_threshold(
                    data.rows,
                    validation,
                    validation_scores,
                    target_false_alarm_rate=config.target_correct_chain_false_alarm,
                )
                test_scores = trained.predict(
                    data,
                    test,
                    batch_size=config.training.batch_size,
                    device=config.training.device,
                )
                global_scores[arm][test] = test_scores
                thresholds[arm][held_domain] = threshold
                metrics = evaluate_boundary_scores(
                    data.rows,
                    test,
                    test_scores,
                    false_alarm_threshold=threshold,
                )
                metrics.update(
                    {
                        "validation_nll": trained.validation_nll,
                        "epochs_trained": trained.epochs_trained,
                        "parameters": int(
                            sum(value.numel() for value in trained.model.parameters())
                        ),
                    }
                )
                fold_report["arms"][arm] = metrics
                for index, score in zip(test, test_scores):
                    row = data.rows[int(index)]
                    prediction_lines.append(
                        json.dumps(
                            {
                                "arm": arm,
                                "held_domain": held_domain,
                                "chain_id": row.chain_id,
                                "problem_hash": row.problem_hash,
                                "candidate_step": row.candidate_step,
                                "first_error_step": row.first_error_step,
                                "label": row.label,
                                "score": float(score),
                                "threshold": float(threshold),
                            },
                            ensure_ascii=False,
                        )
                    )
            folds[held_domain] = fold_report

        for arm, scores in global_scores.items():
            if not np.isfinite(scores).all():
                raise RuntimeError(f"arm {arm} did not score every outer-test row")
        all_indices = np.arange(len(data.rows), dtype=np.int64)
        aggregate = {
            arm: {
                metric: float(
                    np.nanmean(
                        [folds[domain]["arms"][arm][metric] for domain in config.domains]
                    )
                )
                for metric in (
                    "top1_localization",
                    "mrr",
                    "correct_chain_false_alarm_rate",
                    "correct_step_false_alarm_rate",
                    "row_nll",
                    "row_auroc",
                    "row_auprc",
                )
            }
            for arm in config.arms
        }
        contrasts = {}
        if "depth_graph" in config.arms:
            for baseline in (
                "output_history",
                "layer_set",
                "depth_graph_shuffled",
            ):
                if baseline not in config.arms:
                    continue
                contrasts[f"depth_graph_vs_{baseline}"] = paired_problem_bootstrap(
                    data.rows,
                    all_indices,
                    candidate_scores=global_scores["depth_graph"],
                    baseline_scores=global_scores[baseline],
                    candidate_thresholds=thresholds["depth_graph"],
                    baseline_thresholds=thresholds[baseline],
                    repeats=config.bootstrap_repeats,
                    seed=config.seed,
                )
        summary = {
            "task": "future_free_first_error_boundary_monitoring",
            "claim_scope": "predictive_association_not_attention_or_ffn_root_cause",
            "state_view": "step_pre_state_at_token_start_minus_one",
            "risk_set": "error_chains_steps_0_through_first_error;correct_chains_all_steps",
            "rows": len(data.rows),
            "events": int(data.labels.sum()),
            "chains": len({row.chain_id for row in data.rows}),
            "domains": list(config.domains),
            "layers": data.layer_ids.tolist(),
            "hidden_size": data.hidden_size,
            "folds": folds,
            "domain_macro": aggregate,
            "paired_contrasts": contrasts,
        }
        (output_dir / "predictions.jsonl").write_text(
            "\n".join(prediction_lines) + "\n", encoding="utf-8"
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary


__all__ = [
    "MonitorExperimentConfig",
    "ProcessBenchMonitorExperiment",
    "correct_chain_threshold",
    "evaluate_boundary_scores",
    "paired_problem_bootstrap",
]
