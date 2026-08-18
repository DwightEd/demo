from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans


def transition_coordinates(
    projected: np.ndarray, geometry: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return state and one-layer displacement for each token/layer edge."""
    if projected.ndim != 3:
        raise ValueError(f"projected states must be [token,layer,dim], got {projected.shape}")
    if geometry == "euclidean":
        state = projected[:, :-1]
        target = projected[:, 1:] - projected[:, :-1]
        return state, target
    if geometry != "spherical":
        raise ValueError(f"unknown geometry: {geometry}")

    norm = np.linalg.norm(projected, axis=-1, keepdims=True)
    unit = projected / np.maximum(norm, 1e-8)
    start = unit[:, :-1]
    end = unit[:, 1:]
    cosine = np.clip(np.sum(start * end, axis=-1, keepdims=True), -1.0, 1.0)
    angle = np.arccos(cosine)
    orthogonal = end - cosine * start
    scale = np.where(angle < 1e-6, 1.0, angle / np.maximum(np.sin(angle), 1e-8))
    return start, scale * orthogonal


@dataclass(frozen=True)
class _Component:
    coefficient: np.ndarray
    variance: np.ndarray


class MixtureLinearDynamics:
    """Small soft mixture of affine vector fields with diagonal residual noise."""

    def __init__(
        self,
        n_components: int,
        *,
        ridge: float = 1e-3,
        variance_floor: float = 1e-5,
        max_iter: int = 60,
        seed: int = 17,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self.ridge = ridge
        self.variance_floor = variance_floor
        self.max_iter = max_iter
        self.seed = seed
        self.components_: tuple[_Component, ...] | None = None
        self.weights_: np.ndarray | None = None

    @staticmethod
    def _design(state: np.ndarray, position: np.ndarray) -> np.ndarray:
        if state.ndim != 2 or position.shape != (state.shape[0],):
            raise ValueError("state/position arrays are not row aligned")
        return np.column_stack([state, position, np.ones(state.shape[0])])

    def _weighted_fit(
        self, design: np.ndarray, target: np.ndarray, weight: np.ndarray
    ) -> _Component:
        effective = max(float(np.sum(weight)), 1e-8)
        gram = design.T @ (weight[:, None] * design)
        penalty = np.eye(design.shape[1]) * self.ridge
        penalty[-1, -1] = 0.0
        coefficient = np.linalg.solve(gram + penalty, design.T @ (weight[:, None] * target))
        residual = target - design @ coefficient
        variance = np.sum(weight[:, None] * residual**2, axis=0) / effective
        return _Component(coefficient, np.maximum(variance, self.variance_floor))

    @staticmethod
    def _component_log_density(
        design: np.ndarray, target: np.ndarray, component: _Component
    ) -> np.ndarray:
        residual = target - design @ component.coefficient
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * component.variance)
            + residual**2 / component.variance,
            axis=1,
        )

    def fit(
        self, state: np.ndarray, target: np.ndarray, position: np.ndarray
    ) -> "MixtureLinearDynamics":
        state = np.asarray(state, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        position = np.asarray(position, dtype=np.float64)
        if state.shape[0] != target.shape[0] or target.ndim != 2:
            raise ValueError("state and target arrays are not row aligned")
        if state.shape[0] < max(4, self.n_components * 2):
            raise ValueError("too few transition rows for requested mixture")

        design = self._design(state, position)
        global_component = self._weighted_fit(design, target, np.ones(state.shape[0]))
        if self.n_components == 1:
            self.components_ = (global_component,)
            self.weights_ = np.ones(1)
            return self

        residual = target - design @ global_component.coefficient
        labels = KMeans(
            n_clusters=self.n_components,
            n_init=10,
            random_state=self.seed,
        ).fit_predict(residual)
        responsibility = np.eye(self.n_components)[labels]
        previous = -np.inf
        for _ in range(self.max_iter):
            masses = np.sum(responsibility, axis=0) + 1e-6
            weights = masses / np.sum(masses)
            components = tuple(
                self._weighted_fit(design, target, responsibility[:, component])
                for component in range(self.n_components)
            )
            joint = np.column_stack(
                [
                    np.log(weights[index])
                    + self._component_log_density(design, target, component)
                    for index, component in enumerate(components)
                ]
            )
            normalizer = logsumexp(joint, axis=1)
            responsibility = np.exp(joint - normalizer[:, None])
            objective = float(np.sum(normalizer))
            if objective - previous < 1e-6 * max(1.0, abs(previous)):
                break
            previous = objective

        self.components_ = components
        self.weights_ = weights
        return self

    def nll(self, state: np.ndarray, target: np.ndarray, position: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.weights_ is None:
            raise RuntimeError("fit must be called before nll")
        design = self._design(
            np.asarray(state, dtype=np.float64), np.asarray(position, dtype=np.float64)
        )
        target = np.asarray(target, dtype=np.float64)
        joint = np.column_stack(
            [
                np.log(self.weights_[index])
                + self._component_log_density(design, target, component)
                for index, component in enumerate(self.components_)
            ]
        )
        return -logsumexp(joint, axis=1)
