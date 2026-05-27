"""Constraint score S_j = E_j * C_j * N_j.

Each factor is a sigmoid gate mapping a feature to [0, 1]:
  E_j (Expressiveness):   effective rank should not be too LOW
  C_j (Compression):      displacement + spectral gap should not be too HIGH
  N_j (Non-degeneracy):   effective rank should not be too HIGH, angle should not be too LOW

Thresholds are auto-calibrated from correct-trajectory statistics.
"""

import numpy as np
from typing import Optional, Tuple


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ConstraintScore:
    """Compute S_j = E_j * C_j * N_j with auto-calibrated thresholds."""

    def __init__(self, sharpness: float = 5.0):
        self.sharpness = sharpness
        self.thresholds = None  # set by calibrate()

    def calibrate(self, correct_features: np.ndarray):
        """Calibrate thresholds from correct trajectories.

        Args:
            correct_features: [N, 5] array of (ER, SG, IV, d, alpha) from correct steps
        """
        mu = np.median(correct_features, axis=0)
        iqr = np.percentile(correct_features, 75, axis=0) - np.percentile(correct_features, 25, axis=0)
        iqr[iqr < 1e-8] = 1.0

        # Thresholds: (center, scale) per feature
        # ER=0, SG=1, IV=2, d=3, alpha=4
        self.thresholds = {
            "er_low": (np.percentile(correct_features[:, 0], 10), iqr[0]),
            "er_high": (np.percentile(correct_features[:, 0], 90), iqr[0]),
            "sg_high": (np.percentile(correct_features[:, 1], 90), iqr[1]),
            "d_high": (np.percentile(correct_features[:, 3], 90), iqr[3]),
            "alpha_low": (np.percentile(correct_features[:, 4], 10), iqr[4]),
        }

    def __call__(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute constraint scores.

        Args:
            features: [N, 5] or [T, 5] array

        Returns:
            S_j, E_j, C_j, N_j — each [N,] arrays
        """
        assert self.thresholds is not None, "Call calibrate() first"
        k = self.sharpness

        er = features[:, 0]
        sg = features[:, 1]
        d = features[:, 3]
        alpha = features[:, 4]

        # E_j: ER should not be too low → sigmoid(ER - threshold_low)
        t, s = self.thresholds["er_low"]
        E = _sigmoid(k * (er - t) / s)

        # C_j: displacement and spectral gap should not be too high
        # → sigmoid(-(d - threshold)) * sigmoid(-(sg - threshold))
        t_d, s_d = self.thresholds["d_high"]
        t_sg, s_sg = self.thresholds["sg_high"]
        C = _sigmoid(k * -(d - t_d) / s_d) * _sigmoid(k * -(sg - t_sg) / s_sg)

        # N_j: ER should not be too high AND angle should not be too low
        t_er_h, s_er_h = self.thresholds["er_high"]
        t_alpha, s_alpha = self.thresholds["alpha_low"]
        N = _sigmoid(k * -(er - t_er_h) / s_er_h) * _sigmoid(k * (alpha - t_alpha) / s_alpha)

        S = E * C * N
        return S, E, C, N
