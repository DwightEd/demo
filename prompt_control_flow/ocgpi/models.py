from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


@dataclass(frozen=True)
class CrossFitConfig:
    outer_folds: int = 5
    inner_folds: int = 4
    logistic_c: float = 0.25
    ridge_alpha: float = 10.0
    geometry_ridge_alpha: float = 10.0
    adapter_l2: float = 1.0
    chart_variance: float = 0.95
    chart_max_dim: int = 32
    seed: int = 17

    def validate(self) -> None:
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("outer_folds and inner_folds must be at least 2")
        if (
            min(
                self.logistic_c,
                self.ridge_alpha,
                self.geometry_ridge_alpha,
                self.adapter_l2,
            )
            <= 0.0
        ):
            raise ValueError("regularization strengths must be positive")
        if not 0.0 < self.chart_variance <= 1.0:
            raise ValueError("chart_variance must lie in (0, 1]")
        if self.chart_max_dim < 1:
            raise ValueError("chart_max_dim must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


def binary_task_bootstrap_seed(base_seed: int, checkpoint: float) -> int:
    """Return a deterministic non-negative seed for one binary task.

    Relative checkpoints use their percentage as a small offset. The online
    task uses ``checkpoint=-1`` as a semantic sentinel, so it receives a
    separate seed namespace instead of turning that sentinel into a negative
    NumPy seed.
    """

    seed = int(base_seed)
    if seed < 0:
        raise ValueError("base_seed must be non-negative")
    if float(checkpoint) < 0.0:
        return seed + 10_000
    return seed + int(round(float(checkpoint) * 100.0))


class FiniteStandardizer:
    """Median imputation and scaling with deterministic all-missing handling."""

    def __init__(self) -> None:
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> "FiniteStandardizer":
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("standardizer expects a two-dimensional matrix")
        if sample_weight is None:
            weight = np.ones(len(values), dtype=np.float64)
        else:
            weight = np.asarray(sample_weight, dtype=np.float64)
            if weight.shape != (len(values),):
                raise ValueError("sample_weight must have one value per row")
            if (
                not np.isfinite(weight).all()
                or np.any(weight < 0.0)
                or weight.sum() <= 0.0
            ):
                raise ValueError(
                    "sample_weight must be finite, nonnegative, and nonzero"
                )
        center = np.zeros(values.shape[1], dtype=np.float64)
        for j in range(values.shape[1]):
            mask = np.isfinite(values[:, j])
            if not np.any(mask):
                center[j] = 0.0
                continue
            column = values[mask, j]
            column_weight = weight[mask]
            order = np.argsort(column, kind="stable")
            sorted_weight = column_weight[order]
            threshold = 0.5 * float(sorted_weight.sum())
            index = int(
                np.searchsorted(np.cumsum(sorted_weight), threshold, side="left")
            )
            center[j] = float(column[order[min(index, len(order) - 1)]])
        filled = np.where(np.isfinite(values), values, center[None, :])
        scale = np.sqrt(
            np.average(np.square(filled - center[None, :]), axis=0, weights=weight)
        )
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        self.center = center
        self.scale = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None:
            raise RuntimeError("standardizer is not fitted")
        values = np.asarray(x, dtype=np.float64)
        filled = np.where(np.isfinite(values), values, self.center[None, :])
        return ((filled - self.center[None, :]) / self.scale[None, :]).astype(
            np.float64
        )

    def fit_transform(
        self,
        x: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.fit(x, sample_weight=sample_weight).transform(x)


@dataclass
class BinaryPredictor:
    transform: FiniteStandardizer
    model: LogisticRegression

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.model.decision_function(self.transform.transform(x)), dtype=np.float64
        )


@dataclass
class ContinuousPredictor:
    x_transform: FiniteStandardizer
    y_center: np.ndarray
    y_scale: np.ndarray
    model: Ridge

    def predict(self, x: np.ndarray) -> np.ndarray:
        standardized = self.model.predict(self.x_transform.transform(x))
        return standardized * self.y_scale[None, :] + self.y_center[None, :]


def group_balanced_weights(groups: np.ndarray) -> np.ndarray:
    values = np.asarray(groups)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights / max(weights.mean(), 1e-12)


def grouped_splits(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    seed: int,
    stratified: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    splits = min(int(n_splits), len(unique_groups))
    if splits < 2:
        raise ValueError("at least two unique problem groups are required")
    if stratified:
        splitter = StratifiedGroupKFold(
            n_splits=splits, shuffle=True, random_state=int(seed)
        )
        generated = list(splitter.split(np.zeros((len(y), 1)), y, groups))
    else:
        splitter = GroupKFold(n_splits=splits)
        generated = list(splitter.split(np.zeros((len(y), 1)), y, groups))
    for train, test in generated:
        if np.intersect1d(groups[train], groups[test]).size:
            raise RuntimeError("problem-group leakage in cross-validation split")
        if stratified and (len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2):
            raise ValueError("a grouped binary fold contains only one class")
        yield np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def fit_binary_predictor(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: CrossFitConfig,
) -> BinaryPredictor:
    weights = group_balanced_weights(groups)
    transform = FiniteStandardizer()
    standardized = transform.fit_transform(x, sample_weight=weights)
    model = LogisticRegression(
        C=float(cfg.logistic_c),
        solver="lbfgs",
        max_iter=3000,
        random_state=int(cfg.seed),
    )
    model.fit(
        standardized,
        np.asarray(y, dtype=np.int8),
        sample_weight=weights,
    )
    return BinaryPredictor(transform=transform, model=model)


def fit_continuous_predictor(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: CrossFitConfig,
) -> ContinuousPredictor:
    weights = group_balanced_weights(groups)
    x_transform = FiniteStandardizer()
    x_standardized = x_transform.fit_transform(x, sample_weight=weights)
    target = np.asarray(y, dtype=np.float64)
    y_center = np.average(target, axis=0, weights=weights)
    y_scale = np.sqrt(
        np.average(np.square(target - y_center[None, :]), axis=0, weights=weights)
    )
    y_scale[~np.isfinite(y_scale) | (y_scale < 1e-8)] = 1.0
    y_standardized = (target - y_center[None, :]) / y_scale[None, :]
    model = Ridge(alpha=float(cfg.ridge_alpha))
    model.fit(x_standardized, y_standardized, sample_weight=weights)
    return ContinuousPredictor(
        x_transform=x_transform,
        y_center=y_center,
        y_scale=y_scale,
        model=model,
    )


def inner_oof_binary_logits(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: CrossFitConfig,
    *,
    seed_offset: int,
) -> np.ndarray:
    result = np.full(len(y), np.nan, dtype=np.float64)
    for train, test in grouped_splits(
        y,
        groups,
        n_splits=cfg.inner_folds,
        seed=cfg.seed + seed_offset,
        stratified=True,
    ):
        predictor = fit_binary_predictor(x[train], y[train], groups[train], cfg)
        result[test] = predictor.decision_function(x[test])
    if not np.isfinite(result).all():
        raise RuntimeError(
            "inner binary cross-fitting did not score every training row"
        )
    return result


def inner_oof_continuous_predictions(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: CrossFitConfig,
) -> np.ndarray:
    result = np.full_like(np.asarray(y, dtype=np.float64), np.nan)
    dummy = np.zeros(len(y), dtype=np.int8)
    for train, test in grouped_splits(
        dummy,
        groups,
        n_splits=cfg.inner_folds,
        seed=cfg.seed,
        stratified=False,
    ):
        predictor = fit_continuous_predictor(x[train], y[train], groups[train], cfg)
        result[test] = predictor.predict(x[test])
    if not np.isfinite(result).all():
        raise RuntimeError(
            "inner continuous cross-fitting did not score every training row"
        )
    return result


@dataclass
class ConditionalGeometryChart:
    output_transform: FiniteStandardizer
    geometry_transform: FiniteStandardizer
    conditional_model: Ridge
    residual_center: np.ndarray
    basis: np.ndarray
    retained_variance: float

    @property
    def dim(self) -> int:
        return int(self.basis.shape[0])

    def standardized_geometry(self, geometry: np.ndarray) -> np.ndarray:
        return self.geometry_transform.transform(geometry)

    def predict_geometry(self, output: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.conditional_model.predict(self.output_transform.transform(output)),
            dtype=np.float64,
        )

    def residual(self, output: np.ndarray, geometry: np.ndarray) -> np.ndarray:
        return self.standardized_geometry(geometry) - self.predict_geometry(output)

    def transform(self, output: np.ndarray, geometry: np.ndarray) -> np.ndarray:
        residual = self.residual(output, geometry) - self.residual_center[None, :]
        return residual @ self.basis.T

    def explained_r2(self, output: np.ndarray, geometry: np.ndarray) -> float:
        observed = self.standardized_geometry(geometry)
        predicted = self.predict_geometry(output)
        numerator = float(np.square(observed - predicted).sum())
        denominator = float(np.square(observed).sum())
        return float(1.0 - numerator / max(denominator, 1e-12))

    def backproject(self, chart_coefficients: np.ndarray) -> np.ndarray:
        return self.basis.T @ np.asarray(chart_coefficients, dtype=np.float64)


def fit_conditional_geometry_chart(
    output: np.ndarray,
    geometry: np.ndarray,
    groups: np.ndarray,
    cfg: CrossFitConfig,
) -> ConditionalGeometryChart:
    weights = group_balanced_weights(groups)
    output_transform = FiniteStandardizer()
    geometry_transform = FiniteStandardizer()
    z = output_transform.fit_transform(output, sample_weight=weights)
    g = geometry_transform.fit_transform(geometry, sample_weight=weights)
    conditional = Ridge(alpha=float(cfg.geometry_ridge_alpha))
    conditional.fit(z, g, sample_weight=weights)
    residual = g - conditional.predict(z)
    residual_center = np.average(residual, axis=0, weights=weights)
    centered = residual - residual_center[None, :]
    weighted_centered = centered * np.sqrt(weights[:, None])
    _, singular, right = np.linalg.svd(weighted_centered, full_matrices=False)
    energy = np.square(singular)
    if float(energy.sum()) <= 1e-12:
        dimension = 1
    else:
        cumulative = np.cumsum(energy) / energy.sum()
        dimension = int(np.searchsorted(cumulative, float(cfg.chart_variance)) + 1)
    dimension = max(1, min(dimension, int(cfg.chart_max_dim), right.shape[0]))
    retained = float(energy[:dimension].sum() / max(float(energy.sum()), 1e-12))
    return ConditionalGeometryChart(
        output_transform=output_transform,
        geometry_transform=geometry_transform,
        conditional_model=conditional,
        residual_center=residual_center,
        basis=right[:dimension],
        retained_variance=retained,
    )


@dataclass
class OffsetLogisticAdapter:
    intercept: float
    slope_delta: float
    coefficients: np.ndarray

    def logits(self, offset: np.ndarray, features: np.ndarray) -> np.ndarray:
        base = np.asarray(offset, dtype=np.float64)
        x = np.asarray(features, dtype=np.float64)
        return base + self.intercept + self.slope_delta * base + x @ self.coefficients


def fit_offset_logistic_adapter(
    offset: np.ndarray,
    features: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    l2: float,
) -> OffsetLogisticAdapter:
    offset = np.asarray(offset, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(offset):
        raise ValueError("offset adapter feature shape mismatch")

    def objective(parameter: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameter[0]
        slope_delta = parameter[1]
        beta = parameter[2:]
        eta = offset + intercept + slope_delta * offset + x @ beta
        loss = np.logaddexp(0.0, eta) - target * eta
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -40.0, 40.0)))
        residual = (probability - target) * weight
        value = float((loss * weight).sum() / weight.sum())
        value += 0.5 * float(l2) * float(np.dot(beta, beta))
        value += 0.05 * float(l2) * float(slope_delta * slope_delta)
        gradient = np.empty_like(parameter)
        gradient[0] = residual.sum() / weight.sum()
        gradient[1] = (
            np.dot(residual, offset) / weight.sum() + 0.1 * float(l2) * slope_delta
        )
        gradient[2:] = x.T @ residual / weight.sum() + float(l2) * beta
        return value, gradient

    initial = np.zeros(2 + x.shape[1], dtype=np.float64)
    fit = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"offset logistic optimization failed: {fit.message}")
    return OffsetLogisticAdapter(
        intercept=float(fit.x[0]),
        slope_delta=float(fit.x[1]),
        coefficients=np.asarray(fit.x[2:], dtype=np.float64),
    )


def length_matched_permutation(
    values: np.ndarray,
    nuisance: np.ndarray,
    groups: np.ndarray,
    *,
    rng: np.random.Generator,
    bins: int = 5,
) -> np.ndarray:
    """Break geometry pairing while preserving prefix length and group isolation."""

    values = np.asarray(values)
    nuisance = np.asarray(nuisance, dtype=np.float64)
    groups = np.asarray(groups)
    if nuisance.ndim != 2 or len(nuisance) != len(values):
        raise ValueError("nuisance must have one row per geometry value")
    if groups.shape != (len(values),):
        raise ValueError("groups must have one value per geometry row")
    # Every frozen task stores cumulative prefix-token length in column one.
    # This is causal and more relevant than the current step's token count.
    length_column = 1 if nuisance.shape[1] > 1 else nuisance.shape[1] - 1
    length = nuisance[:, length_column]
    finite = np.isfinite(length)
    if finite.sum() < 4:
        return values[rng.permutation(len(values))]
    quantiles = np.unique(np.quantile(length[finite], np.linspace(0.0, 1.0, bins + 1)))
    if len(quantiles) <= 2:
        return values[rng.permutation(len(values))]
    labels = np.digitize(length, quantiles[1:-1], right=True)
    result = values.copy()
    for label in np.unique(labels):
        index = np.where(labels == label)[0]
        if len(index) >= 2:
            best_donor = index
            best_conflicts = len(index) + 1
            for _ in range(64):
                donor = rng.permutation(index)
                conflicts = int(np.sum(groups[donor] == groups[index]))
                if conflicts < best_conflicts:
                    best_donor = donor
                    best_conflicts = conflicts
                if conflicts == 0:
                    break
            result[index] = values[best_donor]
    return result


@dataclass
class BinaryCrossFitResult:
    y: np.ndarray
    groups: np.ndarray
    base_probability: np.ndarray
    geometry_probability: np.ndarray
    full_probability: np.ndarray
    null_probability: np.ndarray
    geometry_r2: np.ndarray
    chart_dim: np.ndarray
    retained_variance: np.ndarray
    feature_importance: np.ndarray


def crossfit_binary_increment(
    x_output: np.ndarray,
    x_geometry: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    nuisance: np.ndarray,
    cfg: CrossFitConfig,
) -> BinaryCrossFitResult:
    cfg.validate()
    y = np.asarray(y, dtype=np.int8)
    groups = np.asarray(groups, dtype=np.int64)
    n = len(y)
    base_probability = np.full(n, np.nan)
    geometry_probability = np.full(n, np.nan)
    full_probability = np.full(n, np.nan)
    null_probability = np.full(n, np.nan)
    geometry_r2 = np.full(n, np.nan)
    chart_dim = np.full(n, np.nan)
    retained = np.full(n, np.nan)
    importance_rows: list[np.ndarray] = []
    for fold, (train, test) in enumerate(
        grouped_splits(
            y,
            groups,
            n_splits=cfg.outer_folds,
            seed=cfg.seed,
            stratified=True,
        )
    ):
        baseline = fit_binary_predictor(x_output[train], y[train], groups[train], cfg)
        geometry_predictor = fit_binary_predictor(
            np.concatenate([nuisance[train], x_geometry[train]], axis=1),
            y[train],
            groups[train],
            cfg,
        )
        train_offset = inner_oof_binary_logits(
            x_output[train], y[train], groups[train], cfg, seed_offset=fold + 1
        )
        test_offset = baseline.decision_function(x_output[test])
        chart = fit_conditional_geometry_chart(
            x_output[train], x_geometry[train], groups[train], cfg
        )
        train_chart = chart.transform(x_output[train], x_geometry[train])
        test_chart = chart.transform(x_output[test], x_geometry[test])
        weights = group_balanced_weights(groups[train])
        base_adapter = fit_offset_logistic_adapter(
            train_offset,
            np.empty((len(train), 0), dtype=np.float64),
            y[train],
            weights,
            l2=cfg.adapter_l2,
        )
        full_adapter = fit_offset_logistic_adapter(
            train_offset,
            train_chart,
            y[train],
            weights,
            l2=cfg.adapter_l2,
        )
        rng = np.random.default_rng(cfg.seed + 10_003 * (fold + 1))
        null_train = length_matched_permutation(
            train_chart, nuisance[train], groups[train], rng=rng
        )
        null_test = length_matched_permutation(
            test_chart, nuisance[test], groups[test], rng=rng
        )
        null_adapter = fit_offset_logistic_adapter(
            train_offset,
            null_train,
            y[train],
            weights,
            l2=cfg.adapter_l2,
        )
        logits_base = base_adapter.logits(test_offset, np.empty((len(test), 0)))
        logits_full = full_adapter.logits(test_offset, test_chart)
        logits_null = null_adapter.logits(test_offset, null_test)
        base_probability[test] = 1.0 / (
            1.0 + np.exp(-np.clip(logits_base, -40.0, 40.0))
        )
        geometry_logits = geometry_predictor.decision_function(
            np.concatenate([nuisance[test], x_geometry[test]], axis=1)
        )
        geometry_probability[test] = 1.0 / (
            1.0 + np.exp(-np.clip(geometry_logits, -40.0, 40.0))
        )
        full_probability[test] = 1.0 / (
            1.0 + np.exp(-np.clip(logits_full, -40.0, 40.0))
        )
        null_probability[test] = 1.0 / (
            1.0 + np.exp(-np.clip(logits_null, -40.0, 40.0))
        )
        geometry_r2[test] = chart.explained_r2(x_output[test], x_geometry[test])
        chart_dim[test] = chart.dim
        retained[test] = chart.retained_variance
        importance_rows.append(np.abs(chart.backproject(full_adapter.coefficients)))
    for name, values in (
        ("base_probability", base_probability),
        ("geometry_probability", geometry_probability),
        ("full_probability", full_probability),
        ("null_probability", null_probability),
    ):
        if not np.isfinite(values).all():
            raise RuntimeError(f"binary cross-fitting left non-finite {name}")
    return BinaryCrossFitResult(
        y=y,
        groups=groups,
        base_probability=base_probability,
        geometry_probability=geometry_probability,
        full_probability=full_probability,
        null_probability=null_probability,
        geometry_r2=geometry_r2,
        chart_dim=chart_dim,
        retained_variance=retained,
        feature_importance=np.mean(np.stack(importance_rows), axis=0),
    )


@dataclass
class ForecastCrossFitResult:
    target: np.ndarray
    groups: np.ndarray
    base_prediction: np.ndarray
    geometry_prediction: np.ndarray
    full_prediction: np.ndarray
    null_prediction: np.ndarray
    standardized_target: np.ndarray
    standardized_base_prediction: np.ndarray
    standardized_geometry_prediction: np.ndarray
    standardized_full_prediction: np.ndarray
    standardized_null_prediction: np.ndarray
    geometry_r2: np.ndarray
    chart_dim: np.ndarray
    retained_variance: np.ndarray
    feature_importance: np.ndarray


@dataclass
class GeometryExplainabilityResult:
    overall_r2: float
    feature_r2: np.ndarray
    squared_error: np.ndarray
    baseline_square: np.ndarray


def crossfit_geometry_explainability(
    x_output: np.ndarray,
    x_geometry: np.ndarray,
    groups: np.ndarray,
    cfg: CrossFitConfig,
    *,
    stratify_y: np.ndarray | None = None,
) -> GeometryExplainabilityResult:
    """Measure which geometry coordinates are recoverable from output history."""

    groups = np.asarray(groups, dtype=np.int64)
    split_y = (
        np.asarray(stratify_y, dtype=np.int8)
        if stratify_y is not None
        else np.zeros(len(groups), dtype=np.int8)
    )
    squared_error = np.zeros(x_geometry.shape[1], dtype=np.float64)
    baseline_square = np.zeros(x_geometry.shape[1], dtype=np.float64)
    weights = group_balanced_weights(groups)
    for train, test in grouped_splits(
        split_y,
        groups,
        n_splits=cfg.outer_folds,
        seed=cfg.seed,
        stratified=stratify_y is not None,
    ):
        chart = fit_conditional_geometry_chart(
            x_output[train], x_geometry[train], groups[train], cfg
        )
        observed = chart.standardized_geometry(x_geometry[test])
        predicted = chart.predict_geometry(x_output[test])
        squared_error += (np.square(observed - predicted) * weights[test, None]).sum(
            axis=0
        )
        baseline_square += (np.square(observed) * weights[test, None]).sum(axis=0)
    feature_r2 = 1.0 - squared_error / np.maximum(baseline_square, 1e-12)
    overall = 1.0 - float(squared_error.sum()) / max(
        float(baseline_square.sum()), 1e-12
    )
    return GeometryExplainabilityResult(
        overall_r2=float(overall),
        feature_r2=feature_r2,
        squared_error=squared_error,
        baseline_square=baseline_square,
    )


def crossfit_forecast_increment(
    x_output: np.ndarray,
    x_geometry: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    nuisance: np.ndarray,
    cfg: CrossFitConfig,
) -> ForecastCrossFitResult:
    cfg.validate()
    target = np.asarray(target, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    base_prediction = np.full_like(target, np.nan)
    geometry_prediction = np.full_like(target, np.nan)
    full_prediction = np.full_like(target, np.nan)
    null_prediction = np.full_like(target, np.nan)
    standardized_target = np.full_like(target, np.nan)
    standardized_base = np.full_like(target, np.nan)
    standardized_geometry = np.full_like(target, np.nan)
    standardized_full = np.full_like(target, np.nan)
    standardized_null = np.full_like(target, np.nan)
    geometry_r2 = np.full(len(target), np.nan)
    chart_dim = np.full(len(target), np.nan)
    retained = np.full(len(target), np.nan)
    importance_rows: list[np.ndarray] = []
    dummy = np.zeros(len(target), dtype=np.int8)
    for fold, (train, test) in enumerate(
        grouped_splits(
            dummy,
            groups,
            n_splits=cfg.outer_folds,
            seed=cfg.seed,
            stratified=False,
        )
    ):
        baseline = fit_continuous_predictor(
            x_output[train], target[train], groups[train], cfg
        )
        geometry_predictor = fit_continuous_predictor(
            np.concatenate([nuisance[train], x_geometry[train]], axis=1),
            target[train],
            groups[train],
            cfg,
        )
        train_base = inner_oof_continuous_predictions(
            x_output[train], target[train], groups[train], cfg
        )
        test_base = baseline.predict(x_output[test])
        chart = fit_conditional_geometry_chart(
            x_output[train], x_geometry[train], groups[train], cfg
        )
        train_chart = chart.transform(x_output[train], x_geometry[train])
        test_chart = chart.transform(x_output[test], x_geometry[test])
        residual_target = target[train] - train_base
        residual_model = fit_continuous_predictor(
            train_chart, residual_target, groups[train], cfg
        )
        rng = np.random.default_rng(cfg.seed + 20_003 * (fold + 1))
        null_train = length_matched_permutation(
            train_chart, nuisance[train], groups[train], rng=rng
        )
        null_test = length_matched_permutation(
            test_chart, nuisance[test], groups[test], rng=rng
        )
        null_model = fit_continuous_predictor(
            null_train, residual_target, groups[train], cfg
        )
        geometry_test = geometry_predictor.predict(
            np.concatenate([nuisance[test], x_geometry[test]], axis=1)
        )
        full_test = test_base + residual_model.predict(test_chart)
        null_test_prediction = test_base + null_model.predict(null_test)
        base_prediction[test] = test_base
        geometry_prediction[test] = geometry_test
        full_prediction[test] = full_test
        null_prediction[test] = null_test_prediction
        scale = baseline.y_scale[None, :]
        center = baseline.y_center[None, :]
        standardized_target[test] = (target[test] - center) / scale
        standardized_base[test] = (test_base - center) / scale
        standardized_geometry[test] = (geometry_test - center) / scale
        standardized_full[test] = (full_test - center) / scale
        standardized_null[test] = (null_test_prediction - center) / scale
        geometry_r2[test] = chart.explained_r2(x_output[test], x_geometry[test])
        chart_dim[test] = chart.dim
        retained[test] = chart.retained_variance
        # Ridge is fitted in chart coordinates. Backprojection is a stable
        # group-discovery heuristic, not a causal attribution claim.
        chart_weight = np.sqrt(
            np.square(np.asarray(residual_model.model.coef_)).sum(axis=0)
        )
        importance_rows.append(np.abs(chart.backproject(chart_weight)))
    if not (
        np.isfinite(base_prediction).all()
        and np.isfinite(geometry_prediction).all()
        and np.isfinite(full_prediction).all()
        and np.isfinite(null_prediction).all()
        and np.isfinite(standardized_target).all()
        and np.isfinite(standardized_base).all()
        and np.isfinite(standardized_geometry).all()
        and np.isfinite(standardized_full).all()
        and np.isfinite(standardized_null).all()
    ):
        raise RuntimeError("forecast cross-fitting left non-finite predictions")
    return ForecastCrossFitResult(
        target=target,
        groups=groups,
        base_prediction=base_prediction,
        geometry_prediction=geometry_prediction,
        full_prediction=full_prediction,
        null_prediction=null_prediction,
        standardized_target=standardized_target,
        standardized_base_prediction=standardized_base,
        standardized_geometry_prediction=standardized_geometry,
        standardized_full_prediction=standardized_full,
        standardized_null_prediction=standardized_null,
        geometry_r2=geometry_r2,
        chart_dim=chart_dim,
        retained_variance=retained,
        feature_importance=np.mean(np.stack(importance_rows), axis=0),
    )
