#!/usr/bin/env python3
"""
Analyze the CIM convergence hypothesis at sample level.

CIM (Ma et al., arXiv:2605.08142) identifies three conditions for effective reasoning:
    C1: Representational expressivity (model-level, constant for us)
    C2: Spontaneous manifold compression — D_stim decreases over steps
    C3: Non-degenerate information volume — V stays high even as D drops

At sample level, our hypothesis:
    Correct reasoning -> C2+C3 both satisfied (compressing but not collapsing)
    Error reasoning -> C2 stalls/reverses OR C3 breaks (compression stalls or info collapses)

This script computes both proxies from multi-layer spectral data and analyzes
the WHOLE trajectory as a (T, L) object — no averaging, no layer-by-layer.

Usage:
    python scripts/analyze_convergence.py --data_path pilot/results/gsm8k_multilayer.pt
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────
# Spectral primitives
# ──────────────────────────────────────────────

def effective_rank(sigma):
    """C2 proxy: exp(spectral entropy). Lower = more compressed."""
    s2 = sigma ** 2 + 1e-10
    p = s2 / s2.sum(dim=-1, keepdim=True)
    H = -(p * torch.log(p + 1e-12)).sum(dim=-1)
    return torch.exp(H)


def spectral_energy(sigma):
    """C3 proxy: total spectral energy Σσ². Should stay high (non-degenerate)."""
    return (sigma ** 2).sum(dim=-1)


def top_concentration(sigma):
    """C3 proxy: σ₁²/Σσ² — fraction of energy in top direction.
    High = info concentrated (compressed but not degenerate)."""
    s2 = sigma ** 2 + 1e-10
    return s2[..., 0] / s2.sum(dim=-1)


def cim_diagnostic(eff_rank, energy, eps=0.1):
    """Sample-level CIM-inspired diagnostic:
    H_j = V_proxy / exp(eps * D_proxy)

    Higher = better (high info volume with low dimension).
    Correct reasoning: H_j should increase or stay stable over steps.
    """
    return energy / torch.exp(eps * eff_rank)


# ──────────────────────────────────────────────
# Data loading and trajectory extraction
# ──────────────────────────────────────────────

def load_data(path):
    raw = torch.load(path, weights_only=False)
    data, meta = raw["examples"], raw["meta"]
    L = len(meta["layer_indices"])
    k = meta["k"]
    print(f"Loaded {len(data)} examples, {L} layers x {k} SVs")
    return data, meta, L, k


def extract_trajectories(data, L, k):
    """Extract per-example trajectory grids.

    Returns:
        correct: list of dicts with keys 'D', 'V', 'H', 'conc', each (T, L)
        error: list of (dict, error_step)
    """
    correct = []
    error = []

    for ex in data:
        steps = ex["steps"]
        T = len(steps)
        if T < 3:
            continue

        D = np.zeros((T, L))  # C2: effective rank
        V = np.zeros((T, L))  # C3: spectral energy
        H = np.zeros((T, L))  # CIM diagnostic
        conc = np.zeros((T, L))  # top concentration

        for j, s in enumerate(steps):
            sigma_ml = s["sigma_ml"]  # (L, k) tensor
            for l in range(L):
                sig = sigma_ml[l]
                D[j, l] = effective_rank(sig).item()
                V[j, l] = spectral_energy(sig).item()
                conc[j, l] = top_concentration(sig).item()
                H[j, l] = cim_diagnostic(
                    torch.tensor(D[j, l]), torch.tensor(V[j, l])
                ).item()

        traj = {"D": D, "V": V, "H": H, "conc": conc}

        if ex["label"] == -1:
            correct.append(traj)
        else:
            error.append((traj, ex["label"]))

    return correct, error


# ──────────────────────────────────────────────
# Whole-trajectory analysis (no averaging!)
# ──────────────────────────────────────────────

def trajectory_svd_features(grid):
    """SVD of the (T, L) grid as a whole object.

    Returns:
        singular_values: the singular values of the grid
        left_sv: first left singular vector (the dominant temporal pattern)
        eff_rank: effective rank of the grid (how many modes the trajectory uses)
    """
    U, S, Vt = np.linalg.svd(grid, full_matrices=False)
    s2 = S ** 2 + 1e-10
    p = s2 / s2.sum()
    H = -(p * np.log(p + 1e-12)).sum()
    return S, U[:, 0], np.exp(H)


def compute_trajectory_slope_matrix(grid):
    """Fit linear slope along T axis for the whole (T, L) grid.

    Returns (L,) slopes — but we also return the residual norm
    as a measure of non-linearity (jumps/reversals).
    """
    T, L = grid.shape
    t = np.arange(T, dtype=np.float64)
    slopes = np.zeros(L)
    residuals = np.zeros(L)
    for l in range(L):
        coeffs = np.polyfit(t, grid[:, l], 1)
        slopes[l] = coeffs[0]
        fitted = np.polyval(coeffs, t)
        residuals[l] = np.sqrt(((grid[:, l] - fitted) ** 2).mean())
    return slopes, residuals


# ──────────────────────────────────────────────
# Statistical analysis
# ──────────────────────────────────────────────

def analyze_c2_compression(correct, error):
    """C2: Is effective rank decreasing over steps?"""
    print(f"\n{'='*60}")
    print("C2 Analysis: Manifold Compression (D should decrease)")
    print(f"{'='*60}")

    c_slopes = np.array([compute_trajectory_slope_matrix(t["D"])[0] for t in correct])
    e_slopes = np.array([compute_trajectory_slope_matrix(t["D"])[0] for t, _ in error])

    print(f"\n  Correct ({len(c_slopes)}):")
    print(f"    Mean D slope (all layers): {c_slopes.mean():.4f}")
    print(f"    % layers with negative slope: {(c_slopes < 0).mean()*100:.1f}%")

    print(f"\n  Error ({len(e_slopes)}):")
    print(f"    Mean D slope (all layers): {e_slopes.mean():.4f}")
    print(f"    % layers with negative slope: {(e_slopes < 0).mean()*100:.1f}%")

    # Non-linearity (residuals) — jumps/reversals
    c_res = np.array([compute_trajectory_slope_matrix(t["D"])[1] for t in correct])
    e_res = np.array([compute_trajectory_slope_matrix(t["D"])[1] for t, _ in error])
    print(f"\n  Residual (non-linearity, higher = more jumps):")
    print(f"    Correct: {c_res.mean():.4f}")
    print(f"    Error:   {e_res.mean():.4f}")


def analyze_c3_nondegeneracy(correct, error):
    """C3: Is information volume preserved?"""
    print(f"\n{'='*60}")
    print("C3 Analysis: Non-Degenerate Info Volume (V should not collapse)")
    print(f"{'='*60}")

    c_slopes = np.array([compute_trajectory_slope_matrix(t["V"])[0] for t in correct])
    e_slopes = np.array([compute_trajectory_slope_matrix(t["V"])[0] for t, _ in error])

    print(f"\n  Correct ({len(c_slopes)}):")
    print(f"    Mean V slope: {c_slopes.mean():.4f}")

    print(f"\n  Error ({len(e_slopes)}):")
    print(f"    Mean V slope: {e_slopes.mean():.4f}")

    # Top concentration
    c_conc = np.array([compute_trajectory_slope_matrix(t["conc"])[0] for t in correct])
    e_conc = np.array([compute_trajectory_slope_matrix(t["conc"])[0] for t, _ in error])
    print(f"\n  Top-1 concentration slope (should increase if compressing healthily):")
    print(f"    Correct: {c_conc.mean():.4f}")
    print(f"    Error:   {e_conc.mean():.4f}")


def analyze_cim_diagnostic(correct, error):
    """CIM-inspired H = V / exp(eps * D). Should increase for correct."""
    print(f"\n{'='*60}")
    print("CIM Diagnostic: H = V / exp(0.1 * D)")
    print(f"{'='*60}")

    c_slopes = np.array([compute_trajectory_slope_matrix(t["H"])[0] for t in correct])
    e_slopes = np.array([compute_trajectory_slope_matrix(t["H"])[0] for t, _ in error])

    print(f"\n  Correct: mean H slope = {c_slopes.mean():.4f}")
    print(f"  Error:   mean H slope = {e_slopes.mean():.4f}")


def analyze_whole_trajectory_svd(correct, error):
    """SVD of the whole (T, L) grids — trajectory as a single object."""
    print(f"\n{'='*60}")
    print("Whole-trajectory SVD analysis")
    print(f"{'='*60}")

    for name, grid_key in [("D (eff rank)", "D"), ("V (energy)", "V"),
                            ("H (CIM diag)", "H")]:
        c_eranks = [trajectory_svd_features(t[grid_key])[2] for t in correct]
        e_eranks = [trajectory_svd_features(t[grid_key])[2] for t, _ in error]

        print(f"\n  {name} grid effective rank:")
        print(f"    Correct: {np.mean(c_eranks):.3f} +/- {np.std(c_eranks):.3f}")
        print(f"    Error:   {np.mean(e_eranks):.3f} +/- {np.std(e_eranks):.3f}")

        # First left SV: does it show monotonic decrease?
        c_mono = []
        for t in correct:
            _, lsv, _ = trajectory_svd_features(t[grid_key])
            # Monotonicity: correlation with descending index
            T = len(lsv)
            corr = np.corrcoef(np.arange(T), lsv)[0, 1]
            c_mono.append(corr)

        e_mono = []
        for t, _ in error:
            _, lsv, _ = trajectory_svd_features(t[grid_key])
            T = len(lsv)
            corr = np.corrcoef(np.arange(T), lsv)[0, 1]
            e_mono.append(corr)

        print(f"    1st-SV monotonicity (corr with step): "
              f"correct={np.mean(c_mono):.3f}, error={np.mean(e_mono):.3f}")


# ──────────────────────────────────────────────
# AUROC evaluation
# ──────────────────────────────────────────────

def compute_aurocs(correct, error, L):
    """Try multiple scoring methods and report AUROC."""
    print(f"\n{'='*60}")
    print("Sequence-level AUROC (correct=0 vs error=1)")
    print(f"{'='*60}")

    methods = {}

    # 1. Mean D slope (should be more positive for error)
    c = [compute_trajectory_slope_matrix(t["D"])[0].mean() for t in correct]
    e = [compute_trajectory_slope_matrix(t["D"])[0].mean() for t, _ in error]
    methods["D slope (mean)"] = (c, e)

    # 2. D residual (non-linearity, higher for error)
    c = [compute_trajectory_slope_matrix(t["D"])[1].mean() for t in correct]
    e = [compute_trajectory_slope_matrix(t["D"])[1].mean() for t, _ in error]
    methods["D residual"] = (c, e)

    # 3. V slope
    c = [compute_trajectory_slope_matrix(t["V"])[0].mean() for t in correct]
    e = [compute_trajectory_slope_matrix(t["V"])[0].mean() for t, _ in error]
    methods["V slope (mean)"] = (c, e)

    # 4. H slope
    c = [compute_trajectory_slope_matrix(t["H"])[0].mean() for t in correct]
    e = [compute_trajectory_slope_matrix(t["H"])[0].mean() for t, _ in error]
    methods["H slope (mean)"] = (c, e)

    # 5. D grid effective rank (higher = more modes = less constrained)
    c = [trajectory_svd_features(t["D"])[2] for t in correct]
    e = [trajectory_svd_features(t["D"])[2] for t, _ in error]
    methods["D grid eff_rank"] = (c, e)

    # 6. Concentration slope
    c = [compute_trajectory_slope_matrix(t["conc"])[0].mean() for t in correct]
    e = [compute_trajectory_slope_matrix(t["conc"])[0].mean() for t, _ in error]
    methods["Conc slope"] = (c, e)

    # 7. Combined: D_slope - V_slope (compression without info loss)
    c = [compute_trajectory_slope_matrix(t["D"])[0].mean()
         - compute_trajectory_slope_matrix(t["V"])[0].mean() for t in correct]
    e = [compute_trajectory_slope_matrix(t["D"])[0].mean()
         - compute_trajectory_slope_matrix(t["V"])[0].mean() for t, _ in error]
    methods["D_slope - V_slope"] = (c, e)

    for name, (c_scores, e_scores) in methods.items():
        y = [0]*len(c_scores) + [1]*len(e_scores)
        s = c_scores + e_scores
        try:
            auroc = roc_auc_score(y, s)
            sep = (np.mean(e_scores) - np.mean(c_scores)) / (np.std(c_scores) + 1e-8)
            print(f"  {name:25s}: AUROC={auroc:.4f}  sep={sep:+.2f}σ")
        except Exception:
            print(f"  {name:25s}: failed")

    # Step-level AUROC (first-error detection)
    print(f"\n{'='*60}")
    print("Step-level first-error AUROC")
    print(f"{'='*60}")

    for name, grid_key in [("D slope→j", "D"), ("H slope→j", "H")]:
        correct_scores = []
        error_scores = []

        for traj in correct:
            T = traj[grid_key].shape[0]
            for j in range(2, T):
                slope = compute_trajectory_slope_matrix(traj[grid_key][:j+1])[0].mean()
                correct_scores.append(slope)

        for traj, err_step in error:
            T = traj[grid_key].shape[0]
            for j in range(2, T):
                slope = compute_trajectory_slope_matrix(traj[grid_key][:j+1])[0].mean()
                if j < err_step:
                    correct_scores.append(slope)
                elif j == err_step:
                    error_scores.append(slope)

        if correct_scores and error_scores:
            y = [0]*len(correct_scores) + [1]*len(error_scores)
            s = correct_scores + error_scores
            auroc = roc_auc_score(y, s)
            print(f"  {name:25s}: AUROC={auroc:.4f}  "
                  f"(correct={len(correct_scores)}, error={len(error_scores)})")


# ──────────────────────────────────────────────
# Visualizations
# ──────────────────────────────────────────────

def plot_dual_constraint(correct, error, output_dir, meta):
    """Plot C2 (compression) and C3 (non-degeneracy) together."""
    layer_indices = meta["layer_indices"]
    L = len(layer_indices)

    # Pick representative layers
    if L > 6:
        show_layers = [0, L//4, L//2, 3*L//4, L-2, L-1]
    else:
        show_layers = list(range(L))

    # --- Figure 1: D trajectories per layer ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, l in enumerate(show_layers[:6]):
        ax = axes[idx]
        for traj in correct[:30]:
            T = traj["D"].shape[0]
            ax.plot(range(T), traj["D"][:, l], color="blue", alpha=0.15, linewidth=0.8)
        for traj, err_step in error[:30]:
            T = traj["D"].shape[0]
            ax.plot(range(min(err_step, T)), traj["D"][:min(err_step, T), l],
                    color="green", alpha=0.2, linewidth=0.8)
            if err_step < T:
                ax.plot(range(err_step, T), traj["D"][err_step:, l],
                        color="red", alpha=0.3, linewidth=1.2)
        ax.set_title(f"Layer {layer_indices[l]}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Effective Rank (D)")
    plt.suptitle("C2 Compression: Effective Rank per Layer\n"
                 "Blue=correct, Green=error(before), Red=error(after)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "c2_compression_per_layer.png"), dpi=150)
    plt.close()
    print(f"  Saved c2_compression_per_layer.png")

    # --- Figure 2: V trajectories per layer ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, l in enumerate(show_layers[:6]):
        ax = axes[idx]
        for traj in correct[:30]:
            T = traj["V"].shape[0]
            ax.plot(range(T), traj["V"][:, l], color="blue", alpha=0.15, linewidth=0.8)
        for traj, err_step in error[:30]:
            T = traj["V"].shape[0]
            ax.plot(range(min(err_step, T)), traj["V"][:min(err_step, T), l],
                    color="green", alpha=0.2, linewidth=0.8)
            if err_step < T:
                ax.plot(range(err_step, T), traj["V"][err_step:, l],
                        color="red", alpha=0.3, linewidth=1.2)
        ax.set_title(f"Layer {layer_indices[l]}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Spectral Energy (V)")
    plt.suptitle("C3 Non-Degeneracy: Spectral Energy per Layer\n"
                 "Blue=correct, Green=error(before), Red=error(after)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "c3_nondegen_per_layer.png"), dpi=150)
    plt.close()
    print(f"  Saved c3_nondegen_per_layer.png")

    # --- Figure 3: Normalized convergence (whole trajectory as unit) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # C2: normalized D (D/D[0]) over fractional position
    ax = axes[0, 0]
    n_bins = 20
    for label, trajs, color in [("Correct", correct, "blue"),
                                  ("Error", [(t, e) for t, e in error], "red")]:
        binned = np.zeros(n_bins)
        count = np.zeros(n_bins)
        src = trajs if label == "Correct" else [t for t, _ in trajs]
        for traj in src:
            T = traj["D"].shape[0]
            if T < 3:
                continue
            # Whole-trajectory: mean across layers at each step
            d_traj = traj["D"].mean(axis=1)  # (T,)
            normed = d_traj / (d_traj[0] + 1e-8)
            for j in range(T):
                b = min(int(j / (T - 1) * n_bins), n_bins - 1)
                binned[b] += normed[j]
                count[b] += 1
        mask = count > 0
        x = np.linspace(0, 1, n_bins)
        ax.plot(x[mask], binned[mask] / count[mask], color=color, linewidth=2, label=label)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("C2: D / D[0] over trajectory")
    ax.set_ylabel("Normalized eff rank")
    ax.legend()

    # C3: normalized V
    ax = axes[0, 1]
    for label, trajs, color in [("Correct", correct, "blue"),
                                  ("Error", [(t, e) for t, e in error], "red")]:
        binned = np.zeros(n_bins)
        count = np.zeros(n_bins)
        src = trajs if label == "Correct" else [t for t, _ in trajs]
        for traj in src:
            T = traj["V"].shape[0]
            if T < 3:
                continue
            v_traj = traj["V"].mean(axis=1)
            normed = v_traj / (v_traj[0] + 1e-8)
            for j in range(T):
                b = min(int(j / (T - 1) * n_bins), n_bins - 1)
                binned[b] += normed[j]
                count[b] += 1
        mask = count > 0
        x = np.linspace(0, 1, n_bins)
        ax.plot(x[mask], binned[mask] / count[mask], color=color, linewidth=2, label=label)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("C3: V / V[0] over trajectory")
    ax.set_ylabel("Normalized spectral energy")
    ax.legend()

    # CIM diagnostic H
    ax = axes[1, 0]
    for label, trajs, color in [("Correct", correct, "blue"),
                                  ("Error", [(t, e) for t, e in error], "red")]:
        binned = np.zeros(n_bins)
        count = np.zeros(n_bins)
        src = trajs if label == "Correct" else [t for t, _ in trajs]
        for traj in src:
            T = traj["H"].shape[0]
            if T < 3:
                continue
            h_traj = traj["H"].mean(axis=1)
            normed = h_traj / (h_traj[0] + 1e-8)
            for j in range(T):
                b = min(int(j / (T - 1) * n_bins), n_bins - 1)
                binned[b] += normed[j]
                count[b] += 1
        mask = count > 0
        x = np.linspace(0, 1, n_bins)
        ax.plot(x[mask], binned[mask] / count[mask], color=color, linewidth=2, label=label)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("CIM: H / H[0] over trajectory")
    ax.set_xlabel("Fractional position")
    ax.set_ylabel("Normalized H")
    ax.legend()

    # Slope distributions
    ax = axes[1, 1]
    c_d = [compute_trajectory_slope_matrix(t["D"])[0].mean() for t in correct]
    e_d = [compute_trajectory_slope_matrix(t["D"])[0].mean() for t, _ in error]
    ax.hist(c_d, bins=25, alpha=0.5, color="blue", label="Correct", density=True)
    ax.hist(e_d, bins=25, alpha=0.5, color="red", label="Error", density=True)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_title("D slope distribution")
    ax.set_xlabel("Mean slope (negative = converging)")
    ax.legend()

    plt.suptitle("CIM C2+C3 Convergence Analysis", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cim_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved cim_convergence.png")

    # --- Figure 4: Heatmap of a few example trajectories ---
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    # Top row: correct examples
    for i in range(min(4, len(correct))):
        ax = axes[0, i]
        im = ax.imshow(correct[i]["D"].T, aspect="auto", cmap="viridis")
        ax.set_title(f"Correct #{i}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Layer")
        plt.colorbar(im, ax=ax, fraction=0.046)
    # Bottom row: error examples
    for i in range(min(4, len(error))):
        ax = axes[1, i]
        traj, err_step = error[i]
        im = ax.imshow(traj["D"].T, aspect="auto", cmap="viridis")
        ax.axvline(err_step - 0.5, color="red", linewidth=2, linestyle="--")
        ax.set_title(f"Error #{i} (err@{err_step})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Layer")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("(T, L) Effective Rank Heatmaps — whole trajectory view\n"
                 "Red line = error step", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_heatmaps.png"), dpi=150)
    plt.close()
    print(f"  Saved trajectory_heatmaps.png")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="convergence_analysis")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data, meta, L, k = load_data(args.data_path)
    correct, error = extract_trajectories(data, L, k)
    print(f"  Correct: {len(correct)}, Error: {len(error)}")

    # Analysis
    analyze_c2_compression(correct, error)
    analyze_c3_nondegeneracy(correct, error)
    analyze_cim_diagnostic(correct, error)
    analyze_whole_trajectory_svd(correct, error)
    compute_aurocs(correct, error, L)

    # Visualizations
    print(f"\n{'='*60}")
    print("Generating visualizations")
    print(f"{'='*60}")
    plot_dual_constraint(correct, error, args.output_dir, meta)

    print(f"\n{'='*60}")
    print(f"Done. All outputs in {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
