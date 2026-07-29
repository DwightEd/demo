from __future__ import annotations

import numpy as np
import torch

from .contracts import _validated_finite_real, _validated_int


class ShrunkMahalanobis:
    """Small, deterministic normal-reference scorer with covariance shrinkage."""

    def __init__(
        self,
        *,
        shrinkage: float = 0.1,
        regularization: float = 1e-8,
        max_features: int = 256,
        projection_seed: int = 0,
    ):
        self.shrinkage = _validated_finite_real(
            shrinkage,
            name="shrinkage",
            minimum=0.0,
        )
        if self.shrinkage > 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        self.regularization = _validated_finite_real(
            regularization,
            name="regularization",
            strictly_positive=True,
        )
        self.max_features = _validated_int(
            max_features,
            name="max_features",
            minimum=1,
        )
        self.projection_seed = _validated_int(
            projection_seed,
            name="projection_seed",
            minimum=0,
        )
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.projection_: np.ndarray | None = None
        self.input_dimension_: int | None = None

    def _transform(self, values: np.ndarray, *, fitting: bool) -> torch.Tensor:
        if fitting:
            self.input_dimension_ = int(values.shape[1])
            self.projection_ = None
            if values.shape[1] > self.max_features:
                rng = np.random.default_rng(self.projection_seed)
                self.projection_ = rng.normal(
                    0.0,
                    1.0 / np.sqrt(self.max_features),
                    size=(values.shape[1], self.max_features),
                )
        elif self.input_dimension_ != values.shape[1]:
            raise ValueError("values must align with the fitted feature dimension")
        tensor = torch.as_tensor(values, dtype=torch.float64)
        if self.projection_ is not None:
            tensor = tensor @ torch.as_tensor(
                self.projection_, dtype=torch.float64
            )
        return tensor

    def fit(self, reference: np.ndarray) -> "ShrunkMahalanobis":
        values = np.asarray(reference, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("reference must have shape [at least 2 rows, features]")
        if not np.isfinite(values).all():
            raise ValueError("reference must contain only finite values")
        tensor = self._transform(values, fitting=True)
        mean = tensor.mean(dim=0)
        centered = tensor - mean
        covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
        dimension = tensor.shape[1]
        isotropic_scale = float(torch.trace(covariance).item() / dimension)
        target_scale = max(isotropic_scale, self.regularization)
        covariance = (
            (1.0 - self.shrinkage) * covariance
            + self.shrinkage
            * target_scale
            * torch.eye(dimension, dtype=torch.float64)
            + self.regularization * torch.eye(dimension, dtype=torch.float64)
        )
        self.mean_ = mean.cpu().numpy()
        # The project already depends on PyTorch for its hypergraph encoder.
        # Using one LAPACK/OpenMP owner also avoids mixed NumPy-MKL/PyTorch
        # runtime collisions in common Windows Anaconda installations.
        precision = torch.linalg.pinv(covariance, hermitian=True)
        self.precision_ = precision.cpu().numpy()
        return self

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("fit must be called before score_samples")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("values must have shape [rows, features]")
        if not np.isfinite(array).all():
            raise ValueError("values must contain only finite values")
        tensor = self._transform(array, fitting=False)
        centered = tensor - torch.as_tensor(self.mean_, dtype=torch.float64)
        scores = torch.einsum(
            "ni,ij,nj->n",
            centered,
            torch.as_tensor(self.precision_, dtype=torch.float64),
            centered,
        )
        return torch.clamp_min(scores, 0.0).cpu().numpy()
