from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from prompt_control_flow.evaluate import finite_json
from prompt_control_flow.ocgpi.models import (
    CrossFitConfig,
    fit_binary_predictor,
    group_balanced_weights,
    grouped_splits,
)

from .data import StepFeatureDataset


@dataclass(frozen=True)
class DetectorConfig:
    folds: int = 5
    logistic_c: float = 0.25
    seed: int = 17

    def validate(self) -> None:
        if self.folds < 2:
            raise ValueError("folds must be at least 2")
        if self.logistic_c <= 0.0:
            raise ValueError("logistic_c must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


def _probability(logit: np.ndarray) -> np.ndarray:
    value = np.asarray(logit, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _binary_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | int]:
    target = np.asarray(y, dtype=np.int8)
    score = np.asarray(probability, dtype=np.float64)
    weight = group_balanced_weights(np.asarray(groups))
    if len(np.unique(target)) < 2:
        return {
            "n": int(len(target)),
            "positives": int(target.sum()),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "nll_nats": float("nan"),
        }
    clipped = np.clip(score, 1e-8, 1.0 - 1e-8)
    row_nll = -(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))
    return {
        "n": int(len(target)),
        "positives": int(target.sum()),
        "prevalence": float(np.average(target, weights=weight)),
        "auroc": float(roc_auc_score(target, score, sample_weight=weight)),
        "auprc": float(
            average_precision_score(target, score, sample_weight=weight)
        ),
        "nll_nats": float(np.average(row_nll, weights=weight)),
    }


def _localization_metrics(
    data: StepFeatureDataset,
    probability: np.ndarray,
) -> dict[str, float | int | str]:
    ranks: list[float] = []
    top1: list[float] = []
    candidate_counts: list[int] = []
    for chain in np.unique(data.chain_idx):
        chain_rows = np.where(data.chain_idx == chain)[0]
        gold = int(data.gold_error_step[chain_rows[0]])
        if gold < 0:
            continue
        candidates = chain_rows[data.step_idx[chain_rows] <= gold]
        gold_rows = candidates[data.step_idx[candidates] == gold]
        if len(gold_rows) != 1:
            raise ValueError(f"chain {int(chain)} has no unique first-error row")
        values = probability[candidates]
        gold_score = float(probability[int(gold_rows[0])])
        greater = int(np.sum(values > gold_score))
        equal = int(np.sum(values == gold_score))
        ranks.append(1.0 + greater + 0.5 * max(equal - 1, 0))
        top1.append(1.0 / equal if greater == 0 and equal > 0 else 0.0)
        candidate_counts.append(len(candidates))
    return {
        "n_error_chains": int(len(ranks)),
        "top1": float(np.mean(top1)) if top1 else float("nan"),
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "mean_candidates": (
            float(np.mean(candidate_counts)) if candidate_counts else float("nan")
        ),
        "candidate_policy": "steps_up_to_and_including_gold_first_error",
        "tie_policy": "expected_uniform_random_tie_break",
    }


def _chain_detection_metrics(
    data: StepFeatureDataset,
    probability: np.ndarray,
) -> dict[str, float | int]:
    labels: list[int] = []
    scores: list[float] = []
    groups: list[int] = []
    for chain in np.unique(data.chain_idx):
        rows = np.where(data.chain_idx == chain)[0]
        labels.append(int(data.chain_error[rows[0]]))
        scores.append(float(np.max(probability[rows])))
        groups.append(int(data.problem_groups[rows[0]]))
    return _binary_metrics(
        np.asarray(labels, dtype=np.int8),
        np.asarray(scores, dtype=np.float64),
        np.asarray(groups, dtype=np.int64),
    )


class ProcessBenchErrorDetector:
    """Cross-fitted first-error detector over ProcessBench observer features."""

    def __init__(self, config: DetectorConfig) -> None:
        config.validate()
        self.config = config

    def run(
        self,
        data: StepFeatureDataset,
        output_dir: str | Path,
    ) -> dict[str, object]:
        data.validate()
        feature_sets, feature_names = self._feature_sets(data)
        eligible = np.flatnonzero(data.onset_eligible)
        target = data.onset_label[eligible]
        groups = data.problem_groups[eligible]
        if len(np.unique(target)) < 2:
            raise ValueError("first-error detector needs both onset and valid-step rows")

        splits = list(
            grouped_splits(
                target,
                groups,
                n_splits=self.config.folds,
                seed=self.config.seed,
                stratified=True,
            )
        )
        maximum_overlap = 0
        for train_local, test_local in splits:
            maximum_overlap = max(
                maximum_overlap,
                int(
                    np.intersect1d(
                        groups[train_local], groups[test_local]
                    ).size
                ),
            )

        probabilities: dict[str, np.ndarray] = {}
        reports: dict[str, object] = {}
        for name, matrix in feature_sets.items():
            score = self._crossfit(matrix, data, eligible, splits)
            probabilities[name] = score
            reports[name] = {
                "features": list(feature_names[name]),
                "onset": _binary_metrics(
                    data.onset_label[eligible],
                    score[eligible],
                    data.problem_groups[eligible],
                ),
                "localization": _localization_metrics(data, score),
                "chain_detection": _chain_detection_metrics(data, score),
            }

        report: dict[str, object] = {
            "method": "processbench_residual_routing_error_detector_v1",
            "data": {
                "source": data.source_path,
                "rows": data.n_rows,
                "eligible_onset_rows": int(len(eligible)),
                "first_error_rows": int(np.sum(data.onset_label == 1)),
                "chains": int(len(np.unique(data.chain_idx))),
                "problem_groups": int(len(np.unique(data.problem_groups))),
                "selected_layers": list(data.selected_layers),
            },
            "target": {
                "positive": "gold first-error step",
                "negative": "valid steps before first error plus all steps in process-correct chains",
                "excluded": "steps after the annotated first error",
                "detection_timing": "retrospective step-end detection",
            },
            "split": {
                "kind": "stratified grouped cross-fitting by problem",
                "folds": int(len(splits)),
                "problem_group_overlap": int(maximum_overlap),
            },
            "feature_sets": reports,
        }
        self._write_outputs(data, probabilities, report, Path(output_dir))
        return report

    def _crossfit(
        self,
        matrix: np.ndarray,
        data: StepFeatureDataset,
        eligible: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        score = np.full(data.n_rows, np.nan, dtype=np.float64)
        model_config = CrossFitConfig(
            outer_folds=self.config.folds,
            inner_folds=2,
            logistic_c=self.config.logistic_c,
            seed=self.config.seed,
        )
        eligible_groups = data.problem_groups[eligible]
        for train_local, test_local in splits:
            train = eligible[train_local]
            test_groups = np.unique(eligible_groups[test_local])
            test = np.flatnonzero(np.isin(data.problem_groups, test_groups))
            predictor = fit_binary_predictor(
                matrix[train],
                data.onset_label[train],
                data.problem_groups[train],
                model_config,
            )
            score[test] = _probability(predictor.decision_function(matrix[test]))
        if not np.isfinite(score).all():
            missing = np.unique(data.problem_groups[~np.isfinite(score)])
            raise RuntimeError(f"cross-fitting left unscored problem groups: {missing[:8]}")
        return score

    @staticmethod
    def _feature_sets(
        data: StepFeatureDataset,
    ) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
        matrices: dict[str, np.ndarray] = {
            "controls": data.control_features,
            "residual": np.concatenate(
                [data.control_features, data.residual_features], axis=1
            ),
        }
        names: dict[str, tuple[str, ...]] = {
            "controls": data.control_names,
            "residual": data.control_names + data.residual_names,
        }
        if data.output_features.shape[1]:
            matrices["output"] = np.concatenate(
                [data.control_features, data.output_features], axis=1
            )
            names["output"] = data.control_names + data.output_names
            matrices["output_residual"] = np.concatenate(
                [data.control_features, data.output_features, data.residual_features],
                axis=1,
            )
            names["output_residual"] = (
                data.control_names + data.output_names + data.residual_names
            )
        if data.routing_features.shape[1]:
            matrices["routing"] = np.concatenate(
                [data.control_features, data.routing_features], axis=1
            )
            names["routing"] = data.control_names + data.routing_names
        modality_parts = [data.output_features, data.routing_features, data.residual_features]
        matrices["joint"] = np.concatenate(
            [data.control_features, *[part for part in modality_parts if part.shape[1]]],
            axis=1,
        )
        names["joint"] = (
            data.control_names
            + data.output_names
            + data.routing_names
            + data.residual_names
        )
        return matrices, names

    @staticmethod
    def _write_outputs(
        data: StepFeatureDataset,
        probabilities: Mapping[str, np.ndarray],
        report: Mapping[str, object],
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(finite_json(report), handle, indent=2, ensure_ascii=False)
        names = tuple(probabilities)
        matrix = np.column_stack([probabilities[name] for name in names]).astype(
            np.float32
        )
        np.savez_compressed(
            output_dir / "oof_predictions.npz",
            chain_idx=data.chain_idx,
            problem_groups=data.problem_groups,
            step_idx=data.step_idx,
            gold_error_step=data.gold_error_step,
            onset_label=data.onset_label,
            onset_eligible=data.onset_eligible,
            chain_error=data.chain_error,
            score_names=np.asarray(names, dtype=str),
            probabilities=matrix,
        )
