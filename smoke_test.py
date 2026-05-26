"""Smoke test — runs without GPU or model loading.

Generates synthetic trajectories with known geometric structure and verifies
that all utility functions and the metric pipeline produce sensible output.

Expected behaviors:
  - Correct trajectory: low ID, high rho, low kappa, high linearity
  - Error trajectory (jump at step 7): kappa spike, rho drop near jump
  - Trajectory summary: correct has higher linearity, lower mean_kappa
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    two_nn_id,
    participation_ratio,
    bias_corrected_pr,
    info_volume_cim,
    local_pca,
    subspace_angle_principal,
    manifold_self_consistency,
    discrete_curvature,
    normalized_curvature,
    turning_angle,
)


def generate_synthetic_trajectory(T=15, d=200, intrinsic_dim=3, jump_at=None, seed=0):
    """Synthetic trajectory on a low-dim manifold in d-dim ambient space."""
    rng = np.random.default_rng(seed)
    embed = rng.standard_normal((intrinsic_dim, d))
    embed /= np.linalg.norm(embed, axis=1, keepdims=True)

    t_param = np.linspace(0, 2 * np.pi, T)
    low_dim = np.stack([
        np.sin(t_param + i * np.pi / intrinsic_dim) for i in range(intrinsic_dim)
    ], axis=1)

    traj = low_dim @ embed
    traj += 0.01 * rng.standard_normal(traj.shape)

    if jump_at is not None and 0 < jump_at < T:
        jump_direction = rng.standard_normal(d)
        jump_direction /= np.linalg.norm(jump_direction)
        traj[jump_at:] += 5.0 * jump_direction

    return traj


def test_estimators():
    print("=" * 60)
    print("Test 1: Intrinsic dimension estimators on 5D-in-200D")
    print("=" * 60)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 200))
    X[:, 5:] *= 0.001
    print(f"  two_nn_id      = {two_nn_id(X):.3f}     (expect ~5)")
    eigvals = np.var(X, axis=0)
    print(f"  PR (raw)       = {participation_ratio(eigvals):.3f}  (expect ~5)")
    print(f"  bias_corr_PR   = {bias_corrected_pr(X):.3f}  (expect ~5)")
    print(f"  info_volume    = {info_volume_cim(X):.3f}")


def test_tangent_space():
    print("=" * 60)
    print("Test 2: Tangent space + manifold self-consistency")
    print("=" * 60)
    traj = generate_synthetic_trajectory(T=15, d=200, intrinsic_dim=3, seed=1)
    T_basis, eigvals = local_pca(traj[3:8], k=3)
    print(f"  tangent basis shape: {T_basis.shape}, eigvals: {eigvals}")
    step = traj[5] - traj[4]
    rho = manifold_self_consistency(step, T_basis)
    print(f"  rho (on-manifold)  = {rho:.3f}    (expect ~1.0)")
    rng = np.random.default_rng(99)
    off_step = rng.standard_normal(200) * np.linalg.norm(step)
    rho_off = manifold_self_consistency(off_step, T_basis)
    print(f"  rho (off-manifold) = {rho_off:.3f}    (expect ~0.015)")
    angle_self = subspace_angle_principal(T_basis, T_basis)
    print(f"  subspace_angle(T,T) = {angle_self:.6f} rad (expect 0)")


def test_curvature():
    print("=" * 60)
    print("Test 3: Curvature with/without jump")
    print("=" * 60)
    correct = generate_synthetic_trajectory(T=15, d=200, jump_at=None, seed=2)
    error = generate_synthetic_trajectory(T=15, d=200, jump_at=7, seed=2)

    kappa_c = discrete_curvature(correct)
    kappa_e = discrete_curvature(error)
    print(f"  Correct mean kappa: {np.nanmean(kappa_c[1:-1]):.3f}")
    print(f"  Error   mean kappa: {np.nanmean(kappa_e[1:-1]):.3f}")
    print(f"  Error   kappa[7]:   {kappa_e[7]:.3f}  (expect SPIKE)")

    angles = turning_angle(error)
    print(f"  Error turning_angle[7]: {angles[7]:.3f} rad")


def test_trajectory_summary():
    """Test the whole-trajectory summary function."""
    print("=" * 60)
    print("Test 4: Trajectory-level summary (correct vs error)")
    print("=" * 60)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "metrics_module",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_compute_metrics.py"),
    )
    metrics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics_module)

    correct = generate_synthetic_trajectory(T=15, d=200, jump_at=None, seed=3)
    error = generate_synthetic_trajectory(T=15, d=200, jump_at=7, seed=3)

    s_correct = metrics_module.compute_trajectory_summary(correct, window=5)
    s_error = metrics_module.compute_trajectory_summary(error, window=5)

    print(f"  {'metric':>15s}  {'correct':>10s}  {'error':>10s}  expectation")
    print(f"  {'D_PR':>15s}  {s_correct['D_PR']:>10.3f}  {s_error['D_PR']:>10.3f}  error > correct (jump adds dim)")
    print(f"  {'V':>15s}  {s_correct['V']:>10.3f}  {s_error['V']:>10.3f}")
    print(f"  {'spec_entropy':>15s}  {s_correct['spec_entropy']:>10.3f}  {s_error['spec_entropy']:>10.3f}")
    print(f"  {'path_length':>15s}  {s_correct['path_length']:>10.3f}  {s_error['path_length']:>10.3f}  error > correct (jump)")
    print(f"  {'mean_kappa':>15s}  {s_correct['mean_kappa']:>10.3f}  {s_error['mean_kappa']:>10.3f}  error > correct")
    print(f"  {'mean_rho':>15s}  {s_correct['mean_rho']:>10.3f}  {s_error['mean_rho']:>10.3f}  error < correct")
    print(f"  {'linearity':>15s}  {s_correct['linearity']:>10.3f}  {s_error['linearity']:>10.3f}")


def test_step_metrics():
    """Test step-level metric pipeline."""
    print("=" * 60)
    print("Test 5: Step-level metrics on jumped trajectory")
    print("=" * 60)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "metrics_module",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_compute_metrics.py"),
    )
    metrics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics_module)

    error_traj = generate_synthetic_trajectory(T=15, d=200, jump_at=7, seed=3)
    m = metrics_module.compute_step_metrics(error_traj, window=5)

    print(f"  T=15, jump at step 7")
    header = f"  {'step':>4s}  {'D_pre':>7s}  {'V_pre':>7s}  {'ent':>7s}  {'u':>7s}  {'kappa':>7s}  {'rho':>7s}  {'theta':>7s}"
    print(header)
    for j in range(15):
        print(f"  {j:>4d}  {m['D_prefix'][j]:>7.3f}  {m['V_prefix'][j]:>7.3f}  "
              f"{m['spec_entropy'][j]:>7.3f}  {m['u'][j]:>7.3f}  "
              f"{m['kappa'][j]:>7.3f}  {m['rho'][j]:>7.3f}  {m['theta'][j]:>7.3f}")
    print("  Expect: rho drops, kappa spikes, theta increases near step 7.")


if __name__ == "__main__":
    test_estimators()
    print()
    test_tangent_space()
    print()
    test_curvature()
    print()
    test_trajectory_summary()
    print()
    test_step_metrics()
    print()
    print("All smoke tests completed.")
