from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def build_group_fold_ids(
    groups: Sequence[int] | np.ndarray,
    *,
    num_folds: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(groups, dtype=np.int64)
    unique = np.unique(values)
    if int(num_folds) < 2:
        raise ValueError("num_folds must be at least two")
    if len(unique) < int(num_folds):
        raise ValueError("fewer groups than requested folds")
    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    mapping = {int(group): index % int(num_folds) for index, group in enumerate(shuffled)}
    return np.asarray([mapping[int(group)] for group in values], dtype=np.int16)


@dataclass(frozen=True)
class RandomProjection:
    input_dim: int
    output_dim: int
    seed: int

    def validate(self) -> None:
        if min(int(self.input_dim), int(self.output_dim)) < 1:
            raise ValueError("projection dimensions must be positive")
        if int(self.seed) < 0:
            raise ValueError("projection seed must be non-negative")

    def matrix(self, *, dtype=np.float32) -> np.ndarray:
        self.validate()
        rng = np.random.default_rng(int(self.seed))
        values = rng.standard_normal(
            (int(self.input_dim), int(self.output_dim)), dtype=np.float32
        )
        values /= np.sqrt(float(self.output_dim))
        return values.astype(dtype, copy=False)

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.shape[-1] != int(self.input_dim):
            raise ValueError("feature width does not match random projection")
        return values @ self.matrix()


@dataclass
class RidgeChart:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    weight: np.ndarray
    alpha: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        alpha: float,
    ) -> "RidgeChart":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("features and targets must be aligned matrices")
        if len(x) < 2:
            raise ValueError("ridge chart requires at least two rows")
        if float(alpha) < 0.0:
            raise ValueError("ridge alpha must be non-negative")
        feature_mean = x.mean(axis=0)
        feature_scale = x.std(axis=0)
        feature_scale[feature_scale < 1e-6] = 1.0
        target_mean = y.mean(axis=0)
        standardized = (x - feature_mean) / feature_scale
        centered_target = y - target_mean
        try:
            if standardized.shape[1] <= standardized.shape[0]:
                gram = standardized.T @ standardized
                gram.flat[:: gram.shape[0] + 1] += float(alpha)
                rhs = standardized.T @ centered_target
                weight = np.linalg.solve(gram, rhs)
            else:
                gram = standardized @ standardized.T
                gram.flat[:: gram.shape[0] + 1] += float(alpha)
                dual = np.linalg.solve(gram, centered_target)
                weight = standardized.T @ dual
        except np.linalg.LinAlgError:
            augmented = np.vstack(
                [standardized, np.sqrt(float(alpha)) * np.eye(standardized.shape[1])]
            )
            augmented_target = np.vstack(
                [centered_target, np.zeros((standardized.shape[1], y.shape[1]))]
            )
            weight = np.linalg.lstsq(augmented, augmented_target, rcond=1e-8)[0]
        return cls(
            feature_mean=feature_mean.astype(np.float32),
            feature_scale=feature_scale.astype(np.float32),
            target_mean=target_mean.astype(np.float32),
            weight=weight.astype(np.float32),
            alpha=float(alpha),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        return (
            (values - self.feature_mean) / self.feature_scale
        ) @ self.weight + self.target_mean

    def project_direction(self, directions: np.ndarray) -> np.ndarray:
        values = np.asarray(directions, dtype=np.float32)
        return (values / self.feature_scale) @ self.weight


def cross_fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    fold_ids: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, list[RidgeChart]]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    folds = np.asarray(fold_ids, dtype=np.int16)
    if len(x) != len(y) or len(x) != len(folds):
        raise ValueError("cross-fit arrays are misaligned")
    predictions = np.empty_like(y, dtype=np.float32)
    models: list[RidgeChart] = []
    for fold in sorted(np.unique(folds).tolist()):
        test = folds == int(fold)
        train = ~test
        model = RidgeChart.fit(x[train], y[train], alpha=alpha)
        predictions[test] = model.predict(x[test])
        models.append(model)
    return predictions, models


def project_features(
    features: np.ndarray,
    projection: RandomProjection,
    *,
    compute_device: str = "cpu",
) -> np.ndarray:
    if str(compute_device).lower() == "cpu":
        return projection.transform(features)
    import torch

    device = torch.device(str(compute_device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA projection requested but CUDA is unavailable")
    with torch.inference_mode():
        values = torch.as_tensor(features, dtype=torch.float32, device=device)
        matrix = torch.as_tensor(
            projection.matrix(), dtype=torch.float32, device=device
        )
        return (values @ matrix).cpu().numpy()


def _fit_ridge_torch(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    compute_device: str,
) -> RidgeChart:
    import torch

    device = torch.device(str(compute_device))
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    feature_mean = x.mean(dim=0)
    feature_scale = x.std(dim=0, correction=0)
    feature_scale = torch.where(
        feature_scale < 1e-6, torch.ones_like(feature_scale), feature_scale
    )
    target_mean = y.mean(dim=0)
    standardized = (x - feature_mean) / feature_scale
    centered_target = y - target_mean
    if standardized.shape[1] <= standardized.shape[0]:
        gram = standardized.T @ standardized
        gram.diagonal().add_(float(alpha))
        rhs = standardized.T @ centered_target
        weight = torch.linalg.solve(gram, rhs)
    else:
        gram = standardized @ standardized.T
        gram.diagonal().add_(float(alpha))
        dual = torch.linalg.solve(gram, centered_target)
        weight = standardized.T @ dual
    return RidgeChart(
        feature_mean=feature_mean.cpu().numpy(),
        feature_scale=feature_scale.cpu().numpy(),
        target_mean=target_mean.cpu().numpy(),
        weight=weight.cpu().numpy(),
        alpha=float(alpha),
    )


def fit_ridge_accelerated(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    compute_device: str = "cpu",
) -> RidgeChart:
    if str(compute_device).lower() == "cpu":
        return RidgeChart.fit(features, targets, alpha=alpha)
    return _fit_ridge_torch(
        features,
        targets,
        alpha=alpha,
        compute_device=compute_device,
    )


def cross_fit_ridge_accelerated(
    features: np.ndarray,
    targets: np.ndarray,
    fold_ids: np.ndarray,
    *,
    alpha: float,
    compute_device: str = "cpu",
) -> tuple[np.ndarray, list[RidgeChart]]:
    if str(compute_device).lower() == "cpu":
        return cross_fit_ridge(features, targets, fold_ids, alpha=alpha)
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    folds = np.asarray(fold_ids, dtype=np.int16)
    predictions = np.empty_like(y, dtype=np.float32)
    models: list[RidgeChart] = []
    for fold in sorted(np.unique(folds).tolist()):
        test = folds == int(fold)
        train = ~test
        model = fit_ridge_accelerated(
            x[train],
            y[train],
            alpha=alpha,
            compute_device=compute_device,
        )
        predictions[test] = model.predict(x[test])
        models.append(model)
    return predictions, models


@dataclass
class LayerChartBundle:
    layers: np.ndarray
    pair_ids: np.ndarray
    pair_fold_ids: np.ndarray
    projection_input_dim: int
    projection_output_dim: int
    projection_seeds: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray
    alphas: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        folds, layers, projected, targets = self.weights.shape
        if layers != len(self.layers):
            raise ValueError("chart layer axis is misaligned")
        if projected != int(self.projection_output_dim):
            raise ValueError("chart projection width is misaligned")
        if self.feature_mean.shape != (folds, layers, projected):
            raise ValueError("feature_mean shape is invalid")
        if self.feature_scale.shape != (folds, layers, projected):
            raise ValueError("feature_scale shape is invalid")
        if self.target_mean.shape != (folds, layers, targets):
            raise ValueError("target_mean shape is invalid")
        if self.projection_seeds.shape != (layers,):
            raise ValueError("one projection seed is required per layer")
        if self.alphas.shape != (folds, layers):
            raise ValueError("alpha grid is misaligned")
        if self.pair_ids.shape != self.pair_fold_ids.shape:
            raise ValueError("pair-to-fold mapping is invalid")

    def fold_for_pair(self, pair_id: int) -> int:
        matches = np.flatnonzero(self.pair_ids == int(pair_id))
        if len(matches) != 1:
            raise KeyError(f"pair_id {pair_id} is absent or duplicated")
        return int(self.pair_fold_ids[int(matches[0])])

    def projection(self, layer_position: int) -> RandomProjection:
        return RandomProjection(
            input_dim=int(self.projection_input_dim),
            output_dim=int(self.projection_output_dim),
            seed=int(self.projection_seeds[int(layer_position)]),
        )

    def chart(self, fold: int, layer_position: int) -> RidgeChart:
        return RidgeChart(
            feature_mean=self.feature_mean[int(fold), int(layer_position)],
            feature_scale=self.feature_scale[int(fold), int(layer_position)],
            target_mean=self.target_mean[int(fold), int(layer_position)],
            weight=self.weights[int(fold), int(layer_position)],
            alpha=float(self.alphas[int(fold), int(layer_position)]),
        )

    def save(self, path: str | Path, *, compressed: bool = False) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "layers": self.layers,
            "pair_ids": self.pair_ids,
            "pair_fold_ids": self.pair_fold_ids,
            "projection_input_dim": np.asarray(self.projection_input_dim),
            "projection_output_dim": np.asarray(self.projection_output_dim),
            "projection_seeds": self.projection_seeds,
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "target_mean": self.target_mean,
            "weights": self.weights,
            "alphas": self.alphas,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        saver = np.savez_compressed if compressed else np.savez
        with output.open("wb") as handle:
            saver(handle, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "LayerChartBundle":
        with np.load(Path(path), allow_pickle=False) as data:
            bundle = cls(
                layers=data["layers"],
                pair_ids=data["pair_ids"],
                pair_fold_ids=data["pair_fold_ids"],
                projection_input_dim=int(data["projection_input_dim"].item()),
                projection_output_dim=int(data["projection_output_dim"].item()),
                projection_seeds=data["projection_seeds"],
                feature_mean=data["feature_mean"],
                feature_scale=data["feature_scale"],
                target_mean=data["target_mean"],
                weights=data["weights"],
                alphas=data["alphas"],
                metadata=json.loads(str(data["metadata_json"].item())),
            )
        bundle.validate()
        return bundle


def fit_layer_chart_bundle(
    states: np.ndarray,
    targets: np.ndarray,
    pair_ids: np.ndarray,
    layers: np.ndarray,
    *,
    num_folds: int,
    projection_dim: int,
    projection_seed: int,
    alpha: float,
    split_seed: int,
    compute_device: str = "cpu",
) -> tuple[LayerChartBundle, np.ndarray, np.ndarray]:
    values = np.asarray(states, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    groups = np.asarray(pair_ids, dtype=np.int64)
    if values.ndim != 3 or len(values) != len(y) or len(values) != len(groups):
        raise ValueError("states, targets, and pair IDs are misaligned")
    fold_ids = build_group_fold_ids(groups, num_folds=num_folds, seed=split_seed)
    unique_pairs = np.unique(groups)
    pair_fold_ids = np.asarray(
        [np.unique(fold_ids[groups == pair]).item() for pair in unique_pairs],
        dtype=np.int16,
    )
    num_layers = values.shape[1]
    seeds = np.asarray(
        [int(projection_seed) + 1009 * int(layer) for layer in np.asarray(layers)],
        dtype=np.int64,
    )
    projections = [
        RandomProjection(values.shape[2], int(projection_dim), int(seed))
        for seed in seeds
    ]
    projected = [
        project_features(
            values[:, index], projection, compute_device=compute_device
        )
        for index, projection in enumerate(projections)
    ]
    predictions = np.empty((len(values), num_layers, y.shape[1]), dtype=np.float32)
    num_actual_folds = len(np.unique(fold_ids))
    means = np.empty((num_actual_folds, num_layers, projection_dim), dtype=np.float32)
    scales = np.empty_like(means)
    target_means = np.empty((num_actual_folds, num_layers, y.shape[1]), dtype=np.float32)
    weights = np.empty(
        (num_actual_folds, num_layers, projection_dim, y.shape[1]), dtype=np.float32
    )
    alphas = np.full((num_actual_folds, num_layers), float(alpha), dtype=np.float32)
    for layer_position, x in enumerate(projected):
        layer_predictions, models = cross_fit_ridge_accelerated(
            x,
            y,
            fold_ids,
            alpha=alpha,
            compute_device=compute_device,
        )
        predictions[:, layer_position] = layer_predictions
        for fold, model in enumerate(models):
            means[fold, layer_position] = model.feature_mean
            scales[fold, layer_position] = model.feature_scale
            target_means[fold, layer_position] = model.target_mean
            weights[fold, layer_position] = model.weight
    bundle = LayerChartBundle(
        layers=np.asarray(layers, dtype=np.int64),
        pair_ids=unique_pairs,
        pair_fold_ids=pair_fold_ids,
        projection_input_dim=int(values.shape[2]),
        projection_output_dim=int(projection_dim),
        projection_seeds=seeds,
        feature_mean=means,
        feature_scale=scales,
        target_mean=target_means,
        weights=weights,
        alphas=alphas,
        metadata={
            "num_folds": int(num_folds),
            "projection_seed": int(projection_seed),
            "split_seed": int(split_seed),
            "alpha": float(alpha),
            "compute_device": str(compute_device),
        },
    )
    bundle.validate()
    return bundle, predictions, fold_ids
