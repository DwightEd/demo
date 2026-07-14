from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from .predictive_state_data import TransitionBundle


@dataclass(frozen=True)
class PredictiveModelConfig:
    latent_dim: int = 16
    ridge: float = 1e-2
    covariance_shrinkage: float = 0.1
    tangent_variance: float = 0.9

    def validate(self) -> None:
        if self.latent_dim < 2:
            raise ValueError("latent_dim must be at least 2")
        if self.ridge <= 0.0:
            raise ValueError("ridge must be positive")
        if not 0.0 <= self.covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must lie in [0, 1]")
        if not 0.0 < self.tangent_variance < 1.0:
            raise ValueError("tangent_variance must lie in (0, 1)")


@dataclass
class TokenNuisanceTransform:
    token_ids: torch.Tensor
    token_means: torch.Tensor
    global_token_mean: torch.Tensor
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    token_residual: bool
    min_token_count: int
    observed_token_count: int


@dataclass
class ReducedRankGaussianModel:
    x_mean: torch.Tensor
    y_mean: torch.Tensor
    coefficients: torch.Tensor
    chart: torch.Tensor
    residual_precision: torch.Tensor
    residual_eigenvectors: torch.Tensor
    residual_eigenvalues: torch.Tensor
    tangent_rank: int
    static_precision: torch.Tensor
    euclidean_scale: torch.Tensor
    offchart_scale: torch.Tensor

    @property
    def latent_dim(self) -> int:
        return int(self.chart.shape[1])

    @torch.inference_mode()
    def score(self, x: np.ndarray, y: np.ndarray, *, device: str) -> dict[str, np.ndarray]:
        compute = torch.device(device)
        x_t = torch.as_tensor(x, device=compute, dtype=torch.float32)
        y_t = torch.as_tensor(y, device=compute, dtype=torch.float32)
        x_centered = x_t - self.x_mean
        y_centered = y_t - self.y_mean
        true_latent = y_centered @ self.chart
        predicted_latent = (x_centered @ self.coefficients) @ self.chart
        residual = true_latent - predicted_latent
        mahalanobis = torch.einsum(
            "ni,ij,nj->n",
            residual,
            self.residual_precision,
            residual,
        ) / float(self.latent_dim)
        euclidean = residual.square().mean(dim=-1) / self.euclidean_scale.clamp_min(1e-8)
        static = torch.einsum(
            "ni,ij,nj->n",
            true_latent,
            self.static_precision,
            true_latent,
        ) / float(self.latent_dim)
        coordinates = residual @ self.residual_eigenvectors
        normalized = coordinates.square() / self.residual_eigenvalues.clamp_min(1e-8)
        tangent_rank = int(self.tangent_rank)
        if tangent_rank > 0:
            parallel = normalized[:, -tangent_rank:].mean(dim=-1)
        else:
            parallel = torch.full_like(mahalanobis, float("nan"))
        transverse_dim = self.latent_dim - tangent_rank
        if transverse_dim > 0:
            transverse = normalized[:, :transverse_dim].mean(dim=-1)
        else:
            transverse = torch.full_like(mahalanobis, float("nan"))
        projected_target = true_latent @ self.chart.T
        offchart = (y_centered - projected_target).square().mean(dim=-1)
        offchart = offchart / self.offchart_scale.clamp_min(1e-8)
        return {
            "mahalanobis": mahalanobis.detach().cpu().numpy(),
            "euclidean": euclidean.detach().cpu().numpy(),
            "parallel": parallel.detach().cpu().numpy(),
            "transverse": transverse.detach().cpu().numpy(),
            "static_mahalanobis": static.detach().cpu().numpy(),
            "offchart": offchart.detach().cpu().numpy(),
        }


def _response_equal_token_weights(
    sequences: Sequence[np.ndarray],
    indices: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    values = []
    for index in indices:
        length = len(sequences[int(index)])
        if length <= 0:
            raise ValueError("empty projected token sequence")
        values.append(torch.full((length,), 1.0 / length, device=device, dtype=torch.float32))
    return torch.cat(values)


@torch.inference_mode()
def fit_token_nuisance_transform(
    sequences: Sequence[np.ndarray],
    token_ids: Sequence[np.ndarray] | None,
    train_correct_indices: Sequence[int],
    *,
    token_residual: bool,
    min_token_count: int,
    compute_device: str,
) -> TokenNuisanceTransform:
    """Fit lexical nuisance and feature scaling on correct training responses only."""

    indices = np.asarray(train_correct_indices, dtype=np.int64)
    if indices.size < 2:
        raise ValueError("at least two correct training responses are required")
    if min_token_count < 1:
        raise ValueError("min_token_count must be positive")
    if token_residual and token_ids is None:
        raise ValueError("token-ID residualization requires exact stored token IDs")
    if token_ids is not None and len(sequences) != len(token_ids):
        raise ValueError("sequence/token-id lengths differ")
    device = torch.device(compute_device)
    joined = torch.as_tensor(
        np.concatenate([np.asarray(sequences[int(i)], dtype=np.float32) for i in indices], axis=0),
        device=device,
        dtype=torch.float32,
    )
    ids = (
        torch.as_tensor(
            np.concatenate(
                [np.asarray(token_ids[int(i)], dtype=np.int64) for i in indices]
            ),
            device=device,
            dtype=torch.long,
        )
        if token_ids is not None
        else None
    )
    weights = _response_equal_token_weights(sequences, indices, device=device)
    total_weight = weights.sum().clamp_min(1e-8)
    global_mean = (joined * weights[:, None]).sum(dim=0) / total_weight

    if token_residual:
        assert ids is not None
        unique_ids, inverse, counts = torch.unique(
            ids,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        weighted_sums = torch.zeros(
            (unique_ids.numel(), joined.shape[1]),
            device=device,
            dtype=torch.float32,
        )
        weighted_counts = torch.zeros(unique_ids.numel(), device=device, dtype=torch.float32)
        weighted_sums.index_add_(0, inverse, joined * weights[:, None])
        weighted_counts.index_add_(0, inverse, weights)
        means = weighted_sums / weighted_counts[:, None].clamp_min(1e-8)
        reliable = counts >= int(min_token_count)
        means = torch.where(reliable[:, None], means, global_mean[None, :])
        residual = joined - means[inverse]
    else:
        unique_ids = torch.empty(0, device=device, dtype=torch.long)
        means = torch.empty((0, joined.shape[1]), device=device, dtype=torch.float32)
        residual = joined - global_mean

    feature_mean = (residual * weights[:, None]).sum(dim=0) / total_weight
    centered = residual - feature_mean
    variance = (centered.square() * weights[:, None]).sum(dim=0) / total_weight
    feature_scale = torch.sqrt(variance.clamp_min(1e-6))
    return TokenNuisanceTransform(
        token_ids=unique_ids,
        token_means=means,
        global_token_mean=global_mean,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        token_residual=bool(token_residual),
        min_token_count=int(min_token_count),
        observed_token_count=int(unique_ids.numel()),
    )


@torch.inference_mode()
def transform_projected_sequences(
    sequences: Sequence[np.ndarray],
    token_ids: Sequence[np.ndarray] | None,
    transform: TokenNuisanceTransform,
    *,
    compute_device: str,
    max_batch_tokens: int = 32768,
) -> list[np.ndarray]:
    if transform.token_residual and token_ids is None:
        raise ValueError("token-ID residualization requires exact stored token IDs")
    if token_ids is not None and len(sequences) != len(token_ids):
        raise ValueError("sequence/token-id lengths differ")
    device = torch.device(compute_device)
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("empty projected token sequence")
    outputs: list[np.ndarray | None] = [None] * len(sequences)
    order = np.argsort(lengths, kind="stable")
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for index in order.tolist():
        length = int(lengths[index])
        if current and current_tokens + length > max_batch_tokens:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(index)
        current_tokens += length
    if current:
        batches.append(current)

    for batch in batches:
        joined = torch.as_tensor(
            np.concatenate([np.asarray(sequences[index], dtype=np.float32) for index in batch]),
            device=device,
            dtype=torch.float32,
        )
        joined_ids = (
            torch.as_tensor(
                np.concatenate(
                    [np.asarray(token_ids[index], dtype=np.int64) for index in batch]
                ),
                device=device,
                dtype=torch.long,
            )
            if token_ids is not None
            else None
        )
        if joined_ids is not None and joined.shape[0] != joined_ids.numel():
            raise ValueError("projected sequence/token-id mismatch")
        if transform.token_residual and transform.token_ids.numel():
            assert joined_ids is not None
            position = torch.searchsorted(transform.token_ids, joined_ids)
            clipped = position.clamp(max=transform.token_ids.numel() - 1)
            seen = (position < transform.token_ids.numel()) & (
                transform.token_ids[clipped] == joined_ids
            )
            nuisance = transform.global_token_mean.expand_as(joined).clone()
            if bool(torch.any(seen)):
                nuisance[seen] = transform.token_means[clipped[seen]]
        else:
            nuisance = transform.global_token_mean
        normalized = (joined - nuisance - transform.feature_mean) / transform.feature_scale
        offset = 0
        for index in batch:
            stop = offset + int(lengths[index])
            outputs[index] = np.ascontiguousarray(
                normalized[offset:stop].detach().cpu().numpy(), dtype=np.float32
            )
            offset = stop
    if any(value is None for value in outputs):
        raise RuntimeError("nuisance transform did not produce every response")
    return [value for value in outputs if value is not None]


def _weighted_mean(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x * weight[:, None]).sum(dim=0) / weight.sum().clamp_min(1e-8)


def _weighted_covariance(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    centered = x - _weighted_mean(x, weight)
    return (centered * weight[:, None]).T @ centered / weight.sum().clamp_min(1e-8)


def _shrink_covariance(
    covariance: torch.Tensor,
    shrinkage: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dim = int(covariance.shape[0])
    scale = torch.trace(covariance) / max(dim, 1)
    stable = (1.0 - shrinkage) * covariance + shrinkage * scale * torch.eye(
        dim, device=covariance.device, dtype=covariance.dtype
    )
    stable = stable + torch.eye(dim, device=covariance.device, dtype=covariance.dtype) * 1e-6
    eigenvalues, eigenvectors = torch.linalg.eigh(stable)
    eigenvalues = eigenvalues.clamp_min(1e-7)
    precision = (eigenvectors / eigenvalues[None, :]) @ eigenvectors.T
    return precision, eigenvectors, eigenvalues


def _ridge_coefficients(
    x_centered: torch.Tensor,
    y_centered: torch.Tensor,
    weight: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    sqrt_weight = torch.sqrt(weight.clamp_min(0.0))[:, None]
    x_weighted = x_centered * sqrt_weight
    y_weighted = y_centered * sqrt_weight
    gram = x_weighted.T @ x_weighted
    scale = torch.trace(gram) / max(int(gram.shape[0]), 1)
    regularized = gram + float(ridge) * scale.clamp_min(1e-6) * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    return torch.linalg.solve(regularized, x_weighted.T @ y_weighted)


@torch.inference_mode()
def fit_reduced_rank_gaussian(
    bundle: TransitionBundle,
    cfg: PredictiveModelConfig,
    *,
    compute_device: str,
    fit_targets: np.ndarray | None = None,
    fixed_chart: torch.Tensor | None = None,
    fixed_x_mean: torch.Tensor | None = None,
    fixed_y_mean: torch.Tensor | None = None,
) -> ReducedRankGaussianModel:
    """Fit a correct-only reduced-rank predictor and Gaussian innovation law.

    ``fit_targets`` changes only the regression pairing. Residual covariance is
    always calibrated against the true chronological targets in ``bundle.y``.
    This makes shuffled and same-problem-mismatch models comparable to the
    ordered model rather than letting each null redefine its test target.
    """

    cfg.validate()
    if bundle.n_rows < max(8, cfg.latent_dim + 1):
        raise ValueError(
            f"not enough correct transitions ({bundle.n_rows}) for latent_dim={cfg.latent_dim}"
        )
    device = torch.device(compute_device)
    x = torch.as_tensor(bundle.x, device=device, dtype=torch.float32)
    y_true = torch.as_tensor(bundle.y, device=device, dtype=torch.float32)
    y_fit = torch.as_tensor(
        bundle.y if fit_targets is None else fit_targets,
        device=device,
        dtype=torch.float32,
    )
    weight = torch.as_tensor(bundle.weights, device=device, dtype=torch.float32)
    x_mean = _weighted_mean(x, weight) if fixed_x_mean is None else fixed_x_mean
    y_mean = _weighted_mean(y_true, weight) if fixed_y_mean is None else fixed_y_mean
    x_centered = x - x_mean
    fit_centered = y_fit - y_mean
    coefficients = _ridge_coefficients(x_centered, fit_centered, weight, cfg.ridge)
    predicted_full = x_centered @ coefficients

    if fixed_chart is None:
        predicted_covariance = _weighted_covariance(predicted_full, weight)
        eigenvalues, eigenvectors = torch.linalg.eigh(predicted_covariance)
        rank = min(int(cfg.latent_dim), int(eigenvectors.shape[1]))
        chart = eigenvectors[:, -rank:]
    else:
        chart = fixed_chart
        rank = int(chart.shape[1])
    true_centered = y_true - y_mean
    true_latent = true_centered @ chart
    predicted_latent = predicted_full @ chart
    residual = true_latent - predicted_latent
    residual_cov = _weighted_covariance(residual, weight)
    residual_precision, residual_vectors, residual_values = _shrink_covariance(
        residual_cov, cfg.covariance_shrinkage
    )
    descending = torch.flip(residual_values, dims=[0])
    cumulative = torch.cumsum(descending, dim=0) / descending.sum().clamp_min(1e-8)
    threshold = torch.tensor(cfg.tangent_variance, device=device)
    tangent_rank = int(torch.searchsorted(cumulative, threshold).item()) + 1
    tangent_rank = min(max(tangent_rank, 1), rank)

    static_cov = _weighted_covariance(true_latent, weight)
    static_precision, _, _ = _shrink_covariance(static_cov, cfg.covariance_shrinkage)
    euclidean_scale = (residual.square().mean(dim=-1) * weight).sum() / weight.sum().clamp_min(1e-8)
    target_projection = true_latent @ chart.T
    offchart = (true_centered - target_projection).square().mean(dim=-1)
    offchart_scale = (offchart * weight).sum() / weight.sum().clamp_min(1e-8)
    return ReducedRankGaussianModel(
        x_mean=x_mean,
        y_mean=y_mean,
        coefficients=coefficients,
        chart=chart,
        residual_precision=residual_precision,
        residual_eigenvectors=residual_vectors,
        residual_eigenvalues=residual_values,
        tangent_rank=tangent_rank,
        static_precision=static_precision,
        euclidean_scale=euclidean_scale,
        offchart_scale=offchart_scale,
    )


def permute_transition_targets(
    bundle: TransitionBundle,
    *,
    mode: str,
    seed: int,
) -> np.ndarray:
    """Construct chronology-destroying targets while preserving marginals."""

    rng = np.random.default_rng(int(seed))
    output = np.asarray(bundle.y, dtype=np.float32).copy()
    if mode == "within_response":
        for sample in np.unique(bundle.sample_indices):
            indices = np.where(bundle.sample_indices == sample)[0]
            if indices.size < 2:
                continue
            shift = int(rng.integers(1, indices.size))
            output[indices] = bundle.y[np.roll(indices, shift)]
        return output
    if mode != "same_problem_mismatch":
        raise ValueError(f"unknown target permutation mode {mode!r}")
    for problem in np.unique(bundle.problem_ids):
        problem_rows = np.where(bundle.problem_ids == problem)[0]
        samples = np.unique(bundle.sample_indices[problem_rows])
        if samples.size < 2:
            continue
        order = samples.copy()
        rng.shuffle(order)
        target_sample = {
            source: order[(position + 1) % order.size]
            for position, source in enumerate(order.tolist())
        }
        for source in samples.tolist():
            source_rows = problem_rows[bundle.sample_indices[problem_rows] == source]
            candidate_rows = problem_rows[
                bundle.sample_indices[problem_rows] == target_sample[source]
            ]
            candidate_rows = candidate_rows[np.argsort(bundle.transition_positions[candidate_rows])]
            source_rows = source_rows[np.argsort(bundle.transition_positions[source_rows])]
            mapped = candidate_rows[np.arange(source_rows.size) % candidate_rows.size]
            output[source_rows] = bundle.y[mapped]
    return output


@torch.inference_mode()
def aggregate_transition_scores(
    values: np.ndarray,
    bundle: TransitionBundle,
    *,
    n_samples: int,
    compute_device: str,
) -> np.ndarray:
    device = torch.device(compute_device)
    score = torch.as_tensor(values, device=device, dtype=torch.float64)
    indices = torch.as_tensor(bundle.sample_indices, device=device, dtype=torch.long)
    sums = torch.zeros(n_samples, device=device, dtype=torch.float64)
    counts = torch.zeros(n_samples, device=device, dtype=torch.float64)
    sums.index_add_(0, indices, score)
    counts.index_add_(0, indices, torch.ones_like(score))
    result = sums / counts.clamp_min(1.0)
    result = torch.where(counts > 0, result, torch.full_like(result, float("nan")))
    return result.detach().cpu().numpy()


def average_horizon_scores(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("no horizon scores to aggregate")
    matrix = np.stack([np.asarray(value, dtype=np.float64) for value in values])
    finite = np.isfinite(matrix)
    count = finite.sum(axis=0)
    total = np.where(finite, matrix, 0.0).sum(axis=0)
    return np.divide(
        total,
        count,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )


@dataclass
class TokenBigramModel:
    pair_counts: Mapping[tuple[int, int], int]
    prefix_counts: Mapping[int, int]
    vocabulary_size: int
    alpha: float = 0.1

    def score(self, ids: np.ndarray, positions: np.ndarray) -> float:
        ids = np.asarray(ids, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.int64)
        adjacent = np.where(np.diff(positions) == 1)[0]
        if not adjacent.size:
            return float("nan")
        nll = []
        for index in adjacent.tolist():
            first, second = int(ids[index]), int(ids[index + 1])
            numerator = self.pair_counts.get((first, second), 0) + self.alpha
            denominator = self.prefix_counts.get(first, 0) + self.alpha * self.vocabulary_size
            nll.append(-math.log(numerator / max(denominator, 1e-12)))
        return float(np.mean(nll))


def fit_token_bigram(
    token_ids: Sequence[np.ndarray],
    token_positions: Sequence[np.ndarray],
    train_correct_indices: Sequence[int],
    *,
    alpha: float = 0.1,
) -> TokenBigramModel:
    pair_counts: dict[tuple[int, int], int] = {}
    prefix_counts: dict[int, int] = {}
    targets: set[int] = set()
    for sample in train_correct_indices:
        ids = np.asarray(token_ids[int(sample)], dtype=np.int64)
        positions = np.asarray(token_positions[int(sample)], dtype=np.int64)
        for index in np.where(np.diff(positions) == 1)[0].tolist():
            first, second = int(ids[index]), int(ids[index + 1])
            pair_counts[(first, second)] = pair_counts.get((first, second), 0) + 1
            prefix_counts[first] = prefix_counts.get(first, 0) + 1
            targets.add(second)
    return TokenBigramModel(
        pair_counts=pair_counts,
        prefix_counts=prefix_counts,
        vocabulary_size=max(len(targets) + 1, 2),
        alpha=float(alpha),
    )
