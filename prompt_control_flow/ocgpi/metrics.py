from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .models import BinaryCrossFitResult, ForecastCrossFitResult, group_balanced_weights


def _binary_log_loss_per_row(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    target = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return -(target * np.log(p) + (1.0 - target) * np.log1p(-p))


def _binary_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, probability, sample_weight=weights)),
        "auprc": float(average_precision_score(y, probability, sample_weight=weights)),
        "brier": float(brier_score_loss(y, probability, sample_weight=weights)),
        "nll_nats": float(
            np.average(_binary_log_loss_per_row(y, probability), weights=weights)
        ),
    }


def _cluster_bootstrap_difference(
    groups: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    row_by_group = {group: np.where(groups == group)[0] for group in unique}
    point = float(statistic(np.arange(len(groups), dtype=np.int64)))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([row_by_group[group] for group in sampled])
        value = float(statistic(rows))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {
            "point": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boot": 0,
        }
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "point": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_boot": int(len(values)),
    }


def summarize_binary_increment(
    result: BinaryCrossFitResult,
    *,
    n_boot: int = 1000,
    seed: int = 17,
) -> dict[str, object]:
    y = result.y
    base = result.base_probability
    geometry = result.geometry_probability
    full = result.full_probability
    null = result.null_probability
    base_loss = _binary_log_loss_per_row(y, base)
    full_loss = _binary_log_loss_per_row(y, full)
    null_loss = _binary_log_loss_per_row(y, null)
    weights = group_balanced_weights(result.groups)

    def auc_delta(left: np.ndarray, right: np.ndarray) -> Callable[[np.ndarray], float]:
        def statistic(index: np.ndarray) -> float:
            yy = y[index]
            if len(np.unique(yy)) < 2:
                return float("nan")
            return float(
                roc_auc_score(yy, left[index], sample_weight=weights[index])
                - roc_auc_score(yy, right[index], sample_weight=weights[index])
            )

        return statistic

    usable_nats = _cluster_bootstrap_difference(
        result.groups,
        lambda index: float(
            np.average(base_loss[index] - full_loss[index], weights=weights[index])
        ),
        n_boot=n_boot,
        seed=seed,
    )
    usable_nats["point_bits"] = float(usable_nats["point"]) / np.log(2.0)
    usable_nats["ci_low_bits"] = float(usable_nats["ci_low"]) / np.log(2.0)
    usable_nats["ci_high_bits"] = float(usable_nats["ci_high"]) / np.log(2.0)
    return {
        "n_rows": int(len(y)),
        "positives": int(y.sum()),
        "problem_groups": int(len(np.unique(result.groups))),
        "output_only": _binary_metrics(y, base, weights),
        "controls_plus_geometry": _binary_metrics(y, geometry, weights),
        "output_plus_geometry": _binary_metrics(y, full, weights),
        "length_matched_null": _binary_metrics(y, null, weights),
        "increment": {
            "conditional_usable_information": usable_nats,
            "delta_auroc": _cluster_bootstrap_difference(
                result.groups,
                auc_delta(full, base),
                n_boot=n_boot,
                seed=seed + 1,
            ),
            "delta_auroc_vs_null": _cluster_bootstrap_difference(
                result.groups,
                auc_delta(full, null),
                n_boot=n_boot,
                seed=seed + 2,
            ),
            "brier_reduction": _cluster_bootstrap_difference(
                result.groups,
                lambda index: float(
                    np.average(
                        np.square(base[index] - y[index]), weights=weights[index]
                    )
                    - np.average(
                        np.square(full[index] - y[index]), weights=weights[index]
                    )
                ),
                n_boot=n_boot,
                seed=seed + 3,
            ),
            "usable_nats_vs_null": _cluster_bootstrap_difference(
                result.groups,
                lambda index: float(
                    np.average(
                        null_loss[index] - full_loss[index], weights=weights[index]
                    )
                ),
                n_boot=n_boot,
                seed=seed + 4,
            ),
        },
        "geometry_explained_by_output_r2": float(np.nanmean(result.geometry_r2)),
        "conditional_chart_dim": float(np.nanmedian(result.chart_dim)),
        "conditional_chart_retained_variance": float(
            np.nanmean(result.retained_variance)
        ),
    }


def _squared_error(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.square(
        np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    )


def summarize_forecast_increment(
    result: ForecastCrossFitResult,
    *,
    n_boot: int = 1000,
    seed: int = 17,
) -> dict[str, object]:
    target = result.standardized_target
    base_error = _squared_error(target, result.standardized_base_prediction).mean(
        axis=1
    )
    geometry_error = _squared_error(
        target, result.standardized_geometry_prediction
    ).mean(axis=1)
    full_error = _squared_error(target, result.standardized_full_prediction).mean(
        axis=1
    )
    null_error = _squared_error(target, result.standardized_null_prediction).mean(
        axis=1
    )
    weights = group_balanced_weights(result.groups)

    def partial_r2(index: np.ndarray, candidate: np.ndarray) -> float:
        denominator = float(np.dot(weights[index], base_error[index]))
        numerator = float(np.dot(weights[index], candidate[index]))
        return float(1.0 - numerator / max(denominator, 1e-12))

    def gaussian_bits(index: np.ndarray, candidate: np.ndarray) -> float:
        base_mse = float(np.average(base_error[index], weights=weights[index]))
        candidate_mse = float(np.average(candidate[index], weights=weights[index]))
        return float(
            0.5 * np.log(max(base_mse, 1e-12) / max(candidate_mse, 1e-12)) / np.log(2.0)
        )

    return {
        "n_rows": int(len(target)),
        "target_dim": int(target.shape[1]),
        "mse_space": "outer_train_fold_standardized_target",
        "problem_groups": int(len(np.unique(result.groups))),
        "output_only_mse": float(np.average(base_error, weights=weights)),
        "controls_plus_geometry_mse": float(
            np.average(geometry_error, weights=weights)
        ),
        "output_plus_geometry_mse": float(np.average(full_error, weights=weights)),
        "length_matched_null_mse": float(np.average(null_error, weights=weights)),
        "increment": {
            "partial_r2": _cluster_bootstrap_difference(
                result.groups,
                lambda index: partial_r2(index, full_error),
                n_boot=n_boot,
                seed=seed,
            ),
            "partial_r2_vs_null": _cluster_bootstrap_difference(
                result.groups,
                lambda index: float(
                    1.0
                    - np.dot(weights[index], full_error[index])
                    / max(float(np.dot(weights[index], null_error[index])), 1e-12)
                ),
                n_boot=n_boot,
                seed=seed + 1,
            ),
            "gaussian_conditional_information_bits": _cluster_bootstrap_difference(
                result.groups,
                lambda index: gaussian_bits(index, full_error),
                n_boot=n_boot,
                seed=seed + 2,
            ),
        },
        "geometry_explained_by_output_r2": float(np.nanmean(result.geometry_r2)),
        "conditional_chart_dim": float(np.nanmedian(result.chart_dim)),
        "conditional_chart_retained_variance": float(
            np.nanmean(result.retained_variance)
        ),
    }


def ranked_feature_importance(
    names: tuple[str, ...],
    groups: tuple[str, ...],
    values: np.ndarray,
    *,
    limit: int = 30,
) -> list[Mapping[str, object]]:
    importance = np.asarray(values, dtype=np.float64)
    if len(importance) != len(names):
        raise ValueError("feature importance length does not match names")
    order = np.argsort(-np.nan_to_num(importance, nan=-np.inf), kind="stable")[
        : int(limit)
    ]
    return [
        {
            "rank": rank + 1,
            "feature": str(names[index]),
            "group": str(groups[index]),
            "importance": float(importance[index]),
        }
        for rank, index in enumerate(order)
    ]
