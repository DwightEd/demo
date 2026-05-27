"""Smoke test for the (step × layer) low-rank pipeline.

Runs without GPU or HuggingFace. Three checks:

  Test 1 — Spectral primitives on a controlled token cloud.
      Verifies effective rank, spectral energy, top concentration on a known
      rank-k matrix in d-dim ambient space.

  Test 2 — Low-rank decomposition on a synthetic rank-1 + sparse anomaly.
      Generates M = u v^T + ε ⋅ δ_{j*, l*} with small Gaussian noise.
      Verifies that
        - lowrank_k=1 ≈ 1 for the rank-1 part,
        - the residual matrix peaks at (j*, l*),
        - step_residual_norms identifies the right row,
        - layer_residual_norms identifies the right column.

  Test 3 — Three signals on correct vs error synthetic trajectories.
      Builds N correct chains (pure rank-1) and N error chains (rank-1 + sparse
      anomaly at a random step). Reports AUROC for chain-level low-rankness
      and step-level residual norm. Expect both >> 0.5.

Run:
    python smoke_test.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    token_cloud_singular_values,
    effective_rank,
    spectral_energy,
    top_concentration,
    lowrank_decompose,
    chain_lowrankness,
    step_residual_norms,
    layer_residual_norms,
    layer_profile_corr_with_prefix,
)


# ---------------------------------------------------------------------------
# Test 1: Spectral primitives on a controlled token cloud
# ---------------------------------------------------------------------------

def test_spectral_primitives():
    print("=" * 60)
    print("Test 1: spectral primitives on synthetic rank-k cloud")
    print("=" * 60)
    rng = np.random.default_rng(0)
    n, d = 32, 200
    # rank-3 cloud: each row is a random combination of 3 basis vectors
    basis = rng.standard_normal((3, d))
    coeffs = rng.standard_normal((n, 3))
    H = coeffs @ basis + 0.01 * rng.standard_normal((n, d))

    sigmas = token_cloud_singular_values(H)
    print(f"  num σ > 1e-6:           {(sigmas > 1e-6).sum():3d}    (expect ~3)")
    print(f"  effective_rank:         {effective_rank(sigmas):8.3f}    (expect ≈ 3)")
    print(f"  spectral_energy:        {spectral_energy(sigmas):8.3e}")
    print(f"  top_concentration:      {top_concentration(sigmas):8.3f}    (≈ 1/3 if uniform)")


# ---------------------------------------------------------------------------
# Test 2: Single (T × L) low-rank decomposition with a sparse anomaly
# ---------------------------------------------------------------------------

def make_spectral_field(T=10, L=32, anomaly=None, noise=0.05, seed=0):
    """Synthetic (T, L) spectral field. anomaly = (j*, l*, magnitude) or None."""
    rng = np.random.default_rng(seed)
    # Rank-1 base: a layer profile v_l (high at shallow, low at deep) modulated
    # by a slowly-varying step coefficient u_j.
    v = np.linspace(20, 7, L) + 0.5 * rng.standard_normal(L)
    u = 1.0 + 0.05 * rng.standard_normal(T)
    M = np.outer(u, v) + noise * rng.standard_normal((T, L))
    if anomaly is not None:
        j, l, mag = anomaly
        M[j, l] += mag
    return M


def test_lowrank_decomposition():
    print("=" * 60)
    print("Test 2: low-rank + sparse decomposition on synthetic M")
    print("=" * 60)
    j_star, l_star, mag = 5, 20, 8.0
    M = make_spectral_field(T=10, L=32,
                            anomaly=(j_star, l_star, mag), seed=1)
    L1, R, sigmas, Vt = lowrank_decompose(M, k=1, center=True)

    print(f"  M.shape:                {M.shape}")
    print(f"  σ[:5]:                  {sigmas[:5].round(3).tolist()}")
    print(f"  chain lowrank_k=1:      {chain_lowrankness(sigmas, k=1):.4f}    (expect > 0.7)")
    print(f"  chain lowrank_k=2:      {chain_lowrankness(sigmas, k=2):.4f}    (expect > 0.9)")

    step_res = step_residual_norms(R)
    layer_res = layer_residual_norms(R)
    j_hat = int(np.argmax(step_res))
    l_hat = int(np.argmax(layer_res))
    print(f"  argmax row residual:    j_hat = {j_hat}  (expect {j_star})")
    print(f"  argmax col residual:    l_hat = {l_hat}  (expect {l_star})")
    print(f"  step_resid[j*]:         {step_res[j_star]:.3f}")
    print(f"  step_resid mean (others): {np.mean(np.delete(step_res, j_star)):.3f}")

    rho = layer_profile_corr_with_prefix(M)
    print(f"  layer-profile corr ρ:   {np.array2string(rho, precision=3, suppress_small=True)}")
    print(f"  ρ[j*] vs mean of ρ:     {rho[j_star]:.3f} vs {np.nanmean(np.delete(rho, j_star)):.3f}")


# ---------------------------------------------------------------------------
# Test 3: AUROC sanity check on synthetic correct vs error chains
# ---------------------------------------------------------------------------

def test_auroc_sanity(n_each=30, T=10, L=32):
    print("=" * 60)
    print(f"Test 3: chain & step AUROC on synthetic ({n_each} correct + {n_each} error)")
    print("=" * 60)
    rng = np.random.default_rng(7)
    chain_labels = []
    chain_lowrank = []
    step_labels_flat = []
    step_resid_flat = []

    for i in range(n_each):
        M = make_spectral_field(T=T, L=L, anomaly=None, seed=10 + i)
        _, R, s, _ = lowrank_decompose(M, k=1, center=True)
        chain_labels.append(0)
        chain_lowrank.append(chain_lowrankness(s, k=1))
        e = step_residual_norms(R)
        for j in range(T):
            step_labels_flat.append(0)
            step_resid_flat.append(e[j])

    for i in range(n_each):
        j_star = int(rng.integers(2, T - 1))
        l_star = int(rng.integers(0, L))
        mag = float(rng.uniform(5.0, 10.0))
        M = make_spectral_field(T=T, L=L, anomaly=(j_star, l_star, mag),
                                seed=100 + i)
        _, R, s, _ = lowrank_decompose(M, k=1, center=True)
        chain_labels.append(1)
        chain_lowrank.append(chain_lowrankness(s, k=1))
        e = step_residual_norms(R)
        for j in range(T):
            step_labels_flat.append(1 if j == j_star else 0)
            step_resid_flat.append(e[j])

    chain_labels = np.asarray(chain_labels)
    chain_lowrank = np.asarray(chain_lowrank)
    step_labels_flat = np.asarray(step_labels_flat)
    step_resid_flat = np.asarray(step_resid_flat)

    auroc_chain = roc_auc_score(chain_labels, -chain_lowrank)
    auroc_step = roc_auc_score(step_labels_flat, step_resid_flat)
    print(f"  AUROC(chain_lowrank_k=1, sign-flipped): {auroc_chain:.4f}    (expect > 0.85)")
    print(f"  AUROC(step_residual_norm):              {auroc_step:.4f}    (expect > 0.85)")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_spectral_primitives()
    print()
    test_lowrank_decomposition()
    print()
    test_auroc_sanity()
    print()
    print("Smoke tests completed.")
