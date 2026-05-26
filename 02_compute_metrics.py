"""Step 2: Compute trajectory-level and step-level geometric/dynamical metrics.

Two analysis modes:
  (A) Whole-trajectory summary: one scalar per trajectory per metric.
      Used for correct vs error distribution comparison (t-test, violin plots).
  (B) Step-level evolution: per-step metric arrays for τ-aligned analysis.
      Uses prefix-incremental computation (cumulative [h_1,...,h_j]).

Metrics (trajectory-level):
    D_PR        : Bias-corrected PR on full trajectory point cloud
    V           : CIM-2 information volume on full trajectory
    spec_entropy: Spectral entropy (eigenvalue distribution uniformity)
    path_length : Total Euclidean path length sum_j ||r_j - r_{j-1}||
    mean_kappa  : Mean discrete curvature (interior points)
    mean_rho    : Mean manifold self-consistency (requires window for tangent est.)
    linearity   : End-to-end distance / path length (1=straight, 0=loops back)
    n_steps     : Number of reasoning steps (trajectory length T)

Metrics (step-level, prefix-incremental):
    D_prefix    : Bias-corrected PR on prefix [h_1,...,h_j]
    V_prefix    : CIM-2 information volume on prefix
    spec_entropy: Spectral entropy on prefix
    u           : Step displacement ||r_j - r_{j-1}||
    kappa       : Discrete curvature ||r_{j+1} - 2r_j + r_{j-1}||
    rho         : Manifold self-consistency of step onto local tangent
    theta       : Subspace angle between consecutive local tangent spaces

Usage:
    python 02_compute_metrics.py \
        --input data/hidden_states.npz \
        --window 5 \
        --output data/metrics.npz
"""

import argparse
import os
import numpy as np
from tqdm import tqdm

from utils import (
    bias_corrected_pr,
    two_nn_id,
    info_volume_cim,
    local_pca,
    subspace_angle_principal,
    manifold_self_consistency,
    discrete_curvature,
)


# ---------------------------------------------------------------------------
# (A) Whole-trajectory summary — one scalar per trajectory
# ---------------------------------------------------------------------------

def compute_trajectory_summary(traj, window=5, n_tangent_dims=None):
    """Compute global summary features for one trajectory.

    Args:
        traj: (T, d) array of step-representative hidden states
        window: sliding window size for tangent space estimation (rho)
        n_tangent_dims: PCA components for tangent basis; default min(window-1, 8)

    Returns:
        dict of scalar summary features
    """
    traj = np.asarray(traj, dtype=np.float64)
    traj = traj - traj.mean(axis=0, keepdims=True)
    T, d = traj.shape

    summary = {
        "n_steps": T,
        "D_PR": np.nan,
        "V": np.nan,
        "spec_entropy": np.nan,
        "path_length": np.nan,
        "mean_kappa": np.nan,
        "mean_rho": np.nan,
        "linearity": np.nan,
    }

    if T < 3:
        return summary

    # --- Global ID and V on full point cloud ---
    try:
        summary["D_PR"] = bias_corrected_pr(traj)
    except Exception:
        pass

    try:
        summary["V"] = info_volume_cim(traj)
    except Exception:
        pass

    # --- Spectral entropy on full trajectory ---
    try:
        s = np.linalg.svd(traj, full_matrices=False, compute_uv=False)
        eigvals = (s ** 2) / max(T - 1, 1)
        eigvals = eigvals[eigvals > 1e-15]
        if len(eigvals) > 0:
            p = eigvals / eigvals.sum()
            summary["spec_entropy"] = -float(np.sum(p * np.log(p + 1e-30)))
    except Exception:
        pass

    # --- Path geometry ---
    displacements = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
    summary["path_length"] = float(displacements.sum())

    end_to_end = float(np.linalg.norm(traj[-1] - traj[0]))
    if summary["path_length"] > 1e-12:
        summary["linearity"] = end_to_end / summary["path_length"]

    # --- Mean curvature ---
    kappa = discrete_curvature(traj)
    interior_kappa = kappa[1:-1]
    valid_kappa = interior_kappa[~np.isnan(interior_kappa)]
    if len(valid_kappa) > 0:
        summary["mean_kappa"] = float(np.mean(valid_kappa))

    # --- Mean manifold self-consistency (sliding window tangent) ---
    if n_tangent_dims is None:
        n_tangent_dims = min(window - 1, 8)

    rho_vals = []
    for j in range(window, T):
        P_j = traj[max(0, j - window + 1): j + 1]
        step_vec = traj[j] - traj[j - 1]
        try:
            T_basis, _ = local_pca(P_j, k=n_tangent_dims)
            rho_vals.append(manifold_self_consistency(step_vec, T_basis))
        except Exception:
            pass
    valid_rho = [r for r in rho_vals if not np.isnan(r)]
    if valid_rho:
        summary["mean_rho"] = float(np.mean(valid_rho))

    return summary


# ---------------------------------------------------------------------------
# (B) Step-level metrics — per-step arrays for τ-aligned analysis
# ---------------------------------------------------------------------------

def compute_step_metrics(traj, window=5, n_tangent_dims=None):
    """Compute per-step metric arrays on a single trajectory.

    Uses prefix-incremental computation for D_prefix, V_prefix, spec_entropy:
    at step j, the point cloud is [h_0, ..., h_j] (growing prefix).
    Cost per step: one SVD on (j+1, d) matrix — O(min(j,d) * j * d),
    which is trivial for T<=30, d=4096.

    Uses sliding window for tangent-space-dependent metrics (theta, rho).

    Args:
        traj: (T, d) array of step-representative hidden states
        window: sliding window size for tangent space estimation
        n_tangent_dims: top-k PCA components; default min(window-1, 8)

    Returns:
        dict of (T,) arrays; undefined positions are NaN
    """
    traj = np.asarray(traj, dtype=np.float64)
    traj = traj - traj.mean(axis=0, keepdims=True)
    T, d = traj.shape

    metrics = {
        "D_prefix": np.full(T, np.nan),
        "V_prefix": np.full(T, np.nan),
        "spec_entropy": np.full(T, np.nan),
        "u": np.full(T, np.nan),
        "kappa": np.full(T, np.nan),
        "theta": np.full(T, np.nan),
        "rho": np.full(T, np.nan),
    }

    if T < 3:
        return metrics

    # --- Step displacement ---
    metrics["u"][1:] = np.linalg.norm(traj[1:] - traj[:-1], axis=1)

    # --- Discrete curvature ---
    metrics["kappa"] = discrete_curvature(traj)

    # --- Prefix-incremental: D_prefix, V_prefix, spec_entropy ---
    min_prefix = max(3, window)
    for j in range(min_prefix - 1, T):
        prefix = traj[:j + 1]

        try:
            metrics["D_prefix"][j] = bias_corrected_pr(prefix)
        except Exception:
            pass

        try:
            metrics["V_prefix"][j] = info_volume_cim(prefix)
        except Exception:
            pass

        try:
            s = np.linalg.svd(prefix - prefix.mean(axis=0, keepdims=True),
                              full_matrices=False, compute_uv=False)
            eigvals = (s ** 2) / max(len(prefix) - 1, 1)
            eigvals = eigvals[eigvals > 1e-15]
            if len(eigvals) > 0:
                p = eigvals / eigvals.sum()
                metrics["spec_entropy"][j] = -float(np.sum(p * np.log(p + 1e-30)))
        except Exception:
            pass

    # --- Sliding-window tangent space: theta, rho ---
    if n_tangent_dims is None:
        n_tangent_dims = min(window - 1, 8)

    prev_T_basis = None
    for j in range(window - 1, T):
        P_j = traj[max(0, j - window + 1): j + 1]

        try:
            T_basis, _ = local_pca(P_j, k=n_tangent_dims)
        except Exception:
            T_basis = None

        if prev_T_basis is not None and T_basis is not None:
            try:
                metrics["theta"][j] = subspace_angle_principal(prev_T_basis, T_basis)
            except Exception:
                pass

        if T_basis is not None and j >= 1:
            step_vec = traj[j] - traj[j - 1]
            try:
                metrics["rho"][j] = manifold_self_consistency(step_vec, T_basis)
            except Exception:
                pass

        prev_T_basis = T_basis

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/hidden_states.npz")
    parser.add_argument("--window", type=int, default=5,
                        help="Sliding window size for tangent space estimation")
    parser.add_argument("--n_tangent_dims", type=int, default=None,
                        help="Top-k PCA components as tangent basis (default min(window-1, 8))")
    parser.add_argument("--output", default="data/metrics.npz")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    data = np.load(args.input, allow_pickle=True)
    ids = data["ids"]
    labels = data["labels"]
    trajs = data["trajs"]
    n_total = len(trajs)
    n_correct = int(np.sum(labels == -1))
    n_error = n_total - n_correct
    print(f"  -> {n_total} trajectories ({n_correct} correct, {n_error} error)")

    # (A) Whole-trajectory summaries
    print(f"Computing trajectory summaries ...")
    summaries = []
    for traj in tqdm(trajs, desc="summaries"):
        summaries.append(compute_trajectory_summary(
            traj, window=args.window, n_tangent_dims=args.n_tangent_dims
        ))

    # (B) Step-level metrics
    print(f"Computing step-level metrics (window={args.window}) ...")
    all_step_metrics = []
    for traj in tqdm(trajs, desc="step-level"):
        all_step_metrics.append(compute_step_metrics(
            traj, window=args.window, n_tangent_dims=args.n_tangent_dims
        ))

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    summary_names = list(summaries[0].keys())
    step_metric_names = list(all_step_metrics[0].keys())

    save_dict = {
        "ids": ids,
        "labels": labels,
        "window": np.array(args.window),
    }

    # Trajectory-level summaries: (N,) arrays per metric
    for name in summary_names:
        save_dict[f"summary_{name}"] = np.array(
            [s[name] for s in summaries], dtype=np.float64
        )

    # Step-level metrics: object arrays of variable-length (T,) arrays
    for name in step_metric_names:
        save_dict[name] = np.array(
            [m[name] for m in all_step_metrics], dtype=object
        )

    np.savez(args.output, **save_dict)
    print(f"Saved -> {args.output}")

    # Print summary statistics
    print(f"\n=== Trajectory-level summary: correct vs error ===")
    correct_mask = labels == -1
    error_mask = labels >= 0
    for name in summary_names:
        vals = np.array([s[name] for s in summaries], dtype=np.float64)
        c_vals = vals[correct_mask]
        e_vals = vals[error_mask]
        c_valid = c_vals[~np.isnan(c_vals)]
        e_valid = e_vals[~np.isnan(e_vals)]
        if len(c_valid) > 0 and len(e_valid) > 0:
            print(f"  {name:15s}  correct: {np.mean(c_valid):+.4f} ± {np.std(c_valid):.4f}"
                  f"  error: {np.mean(e_valid):+.4f} ± {np.std(e_valid):.4f}")


if __name__ == "__main__":
    main()
