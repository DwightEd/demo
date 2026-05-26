"""Discrete trajectory curvature.

Implements the discrete curvature signal from:
Manson. "Curved Inference: Geometric Analysis of LLM Residual Stream
Trajectories." arXiv:2507.21107 (2025).

Curved Inference paper shows that trajectory curvature on the residual
stream correlates with reasoning correctness. The formula is simply the
discrete second difference (acceleration vector) of the trajectory.

We provide raw curvature and length-normalized curvature variants.
"""

import numpy as np


def discrete_curvature(trajectory):
    """Discrete trajectory curvature at each interior point.

    κ_j = || r_{j+1} - 2 r_j + r_{j-1} ||_2

    This is the magnitude of the discrete acceleration vector (second
    difference). Defined for j = 1, ..., T-2 (interior points only).

    Args:
        trajectory: (T, d) array of hidden states

    Returns:
        kappa: (T,) array; kappa[0] = kappa[T-1] = nan (boundary)
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    T, d = trajectory.shape
    kappa = np.full(T, np.nan)

    if T < 3:
        return kappa

    # Second difference: r_{j+1} - 2 r_j + r_{j-1}
    second_diff = trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]
    # Norm at each interior point
    kappa[1:-1] = np.linalg.norm(second_diff, axis=1)
    return kappa


def normalized_curvature(trajectory):
    """Length-normalized discrete curvature.

    κ_norm_j = || r_{j+1} - 2 r_j + r_{j-1} || / (|| r_{j+1} - r_j || + || r_j - r_{j-1} ||)

    Normalizes by local step lengths to make curvature invariant to step size.
    Bounded in [0, 1].

    Args:
        trajectory: (T, d) array of hidden states

    Returns:
        kappa_norm: (T,) array; kappa_norm[0] = kappa_norm[T-1] = nan
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    T, d = trajectory.shape
    kappa_norm = np.full(T, np.nan)

    if T < 3:
        return kappa_norm

    kappa = discrete_curvature(trajectory)
    # Local step lengths
    forward = np.linalg.norm(trajectory[2:] - trajectory[1:-1], axis=1)
    backward = np.linalg.norm(trajectory[1:-1] - trajectory[:-2], axis=1)
    denom = forward + backward
    denom = np.where(denom > 1e-15, denom, np.nan)
    kappa_norm[1:-1] = kappa[1:-1] / denom
    return kappa_norm


def turning_angle(trajectory):
    """Local turning angle (radians) at each interior point.

    angle_j = arccos( <Δr_j, Δr_{j-1}> / (||Δr_j|| ||Δr_{j-1}||) )

    where Δr_j = r_{j+1} - r_j. Range [0, π]; 0 means trajectory continues
    straight, π means trajectory reverses direction.

    Args:
        trajectory: (T, d) array

    Returns:
        angles: (T,) array; angles[0] = angles[T-1] = nan
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    T, d = trajectory.shape
    angles = np.full(T, np.nan)

    if T < 3:
        return angles

    delta_forward = trajectory[2:] - trajectory[1:-1]  # Δr_j  for j = 1..T-2
    delta_backward = trajectory[1:-1] - trajectory[:-2]  # Δr_{j-1}

    norm_f = np.linalg.norm(delta_forward, axis=1)
    norm_b = np.linalg.norm(delta_backward, axis=1)
    denom = norm_f * norm_b
    cos_angle = np.where(
        denom > 1e-15,
        np.einsum("ij,ij->i", delta_forward, delta_backward) / np.where(denom > 1e-15, denom, 1.0),
        np.nan,
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles[1:-1] = np.arccos(cos_angle)
    return angles
