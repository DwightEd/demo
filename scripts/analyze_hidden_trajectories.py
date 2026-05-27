#!/usr/bin/env python3
"""
Analyze hidden state trajectories for CIM convergence hypothesis.

Now using ACTUAL hidden states h_{j,l} instead of sigma proxies.
Computes trajectory-level features that sigma can't capture:
    - Trajectory intrinsic dimension (SVD of step-stacked matrix)
    - Step-to-step displacement and direction change
    - Cross-layer convergence patterns
    - CIM information volume V = 0.5 * log det(I + (d/T) Z Z^T)

Usage:
    python scripts/analyze_hidden_trajectories.py \
        --data_path pilot/results/gsm8k_hidden_states.pt
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────
# Trajectory-level geometry
# ──────────────────────────────────────────────

def trajectory_intrinsic_dim(Z):
    """Effective rank of trajectory matrix Z (T, d). = CIM's D_stim proxy."""
    if Z.shape[0] < 2:
        return 1.0
    Z_c = Z - Z.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(Z_c, full_matrices=False)
    s2 = S ** 2 + 1e-10
    p = s2 / s2.sum()
    H = -(p * np.log(p + 1e-12)).sum()
    return np.exp(H)


def info_volume(Z):
    """CIM information volume: V = 0.5 * log det(I + (d/T) Z_c Z_c^T)."""
    T, d = Z.shape
    if T < 2:
        return 0.0
    Z_c = Z - Z.mean(axis=0, keepdims=True)
    M = np.eye(T) + (d / T) * (Z_c @ Z_c.T)
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        return 0.0
    return 0.5 * logdet


def step_displacement(Z):
    """Step-to-step L2 displacement: ||h_j - h_{j-1}||."""
    if Z.shape[0] < 2:
        return np.array([0.0])
    diffs = np.diff(Z, axis=0)  # (T-1, d)
    return np.linalg.norm(diffs, axis=1)


def step_cosine_change(Z):
    """Cosine of direction change between consecutive steps (curvature proxy).
    cos(angle between diff_j and diff_{j-1}). Low = sharp turn."""
    if Z.shape[0] < 3:
        return np.array([1.0])
    diffs = np.diff(Z, axis=0)  # (T-1, d)
    cosines = []
    for j in range(1, len(diffs)):
        d1 = diffs[j - 1]
        d2 = diffs[j]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-10 or n2 < 1e-10:
            cosines.append(1.0)
        else:
            cosines.append(np.clip(np.dot(d1, d2) / (n1 * n2), -1, 1))
    return np.array(cosines)


def cumulative_intrinsic_dim(Z):
    """Intrinsic dim using steps [0..j] for each j. Shows convergence trend."""
    T = Z.shape[0]
    dims = np.zeros(T)
    for j in range(T):
        if j < 1:
            dims[j] = 1.0
        else:
            dims[j] = trajectory_intrinsic_dim(Z[:j + 1])
    return dims


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_data(path):
    import torch
    raw = torch.load(path, weights_only=False)
    data, meta = raw["examples"], raw["meta"]
    L = len(meta["layer_indices"])
    d = meta["hidden_dim"]
    print(f"Loaded {len(data)} examples, {L} layers, hidden_dim={d}")
    print(f"Layers: {meta['layer_indices']}")
    return data, meta, L, d


def extract_h_trajectories(data, L, d):
    """Extract per-layer hidden state trajectories."""
    correct = []  # list of dicts with 'h': (T, L, d)
    error = []    # list of (dict, error_step)

    for ex in data:
        steps = ex["steps"]
        T = len(steps)
        if T < 3:
            continue

        # Stack h_ml across steps: (T, L, d)
        h_all = np.stack([s["h_ml"].numpy() for s in steps])  # (T, L, d)

        traj = {"h": h_all}
        if ex["label"] == -1:
            correct.append(traj)
        else:
            error.append((traj, ex["label"]))

    return correct, error


# ──────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────

def analyze_per_layer(correct, error, L, meta):
    """Per-layer trajectory analysis."""
    print(f"\n{'='*60}")
    print("Per-layer trajectory intrinsic dimension")
    print(f"{'='*60}")

    layer_indices = meta["layer_indices"]

    for l in range(L):
        c_dims = [trajectory_intrinsic_dim(t["h"][:, l, :]) for t in correct]
        e_dims = [trajectory_intrinsic_dim(t["h"][:, l, :]) for t, _ in error]
        c_vols = [info_volume(t["h"][:, l, :]) for t in correct]
        e_vols = [info_volume(t["h"][:, l, :]) for t, _ in error]

        y = [0]*len(c_dims) + [1]*len(e_dims)
        auroc_dim = roc_auc_score(y, c_dims + e_dims) if len(set(y)) >= 2 else 0.5
        auroc_vol = roc_auc_score(y, c_vols + e_vols) if len(set(y)) >= 2 else 0.5

        if abs(auroc_dim - 0.5) > 0.05 or abs(auroc_vol - 0.5) > 0.05:
            print(f"  Layer {layer_indices[l]:2d}: "
                  f"D_traj correct={np.mean(c_dims):.2f} error={np.mean(e_dims):.2f} "
                  f"AUROC={auroc_dim:.4f} | "
                  f"V correct={np.mean(c_vols):.1f} error={np.mean(e_vols):.1f} "
                  f"AUROC={auroc_vol:.4f}")


def analyze_displacement(correct, error, L, meta):
    """Step-to-step displacement analysis (TRACED-style)."""
    print(f"\n{'='*60}")
    print("Step-to-step displacement analysis")
    print(f"{'='*60}")

    layer_indices = meta["layer_indices"]

    for l in range(L):
        # Mean displacement per trajectory
        c_disp = [step_displacement(t["h"][:, l, :]).mean() for t in correct]
        e_disp = [step_displacement(t["h"][:, l, :]).mean() for t, _ in error]

        # Mean cosine (curvature)
        c_cos = [step_cosine_change(t["h"][:, l, :]).mean() for t in correct]
        e_cos = [step_cosine_change(t["h"][:, l, :]).mean() for t, _ in error]

        y = [0]*len(c_disp) + [1]*len(e_disp)
        auroc_disp = roc_auc_score(y, c_disp + e_disp) if len(set(y)) >= 2 else 0.5
        auroc_cos = roc_auc_score(y, c_cos + e_cos) if len(set(y)) >= 2 else 0.5

        if abs(auroc_disp - 0.5) > 0.05 or abs(auroc_cos - 0.5) > 0.05:
            print(f"  Layer {layer_indices[l]:2d}: "
                  f"displacement AUROC={auroc_disp:.4f} | "
                  f"cosine(curvature) AUROC={auroc_cos:.4f}")


def analyze_convergence_trend(correct, error, L, meta):
    """Cumulative intrinsic dim trend — does D decrease over steps?"""
    print(f"\n{'='*60}")
    print("Cumulative intrinsic dimension convergence")
    print(f"{'='*60}")

    layer_indices = meta["layer_indices"]

    for l in range(L):
        # Slope of cumulative ID
        c_slopes = []
        for t in correct:
            dims = cumulative_intrinsic_dim(t["h"][:, l, :])
            T = len(dims)
            if T >= 3:
                slope = np.polyfit(np.arange(T), dims, 1)[0]
                c_slopes.append(slope)

        e_slopes = []
        for t, _ in error:
            dims = cumulative_intrinsic_dim(t["h"][:, l, :])
            T = len(dims)
            if T >= 3:
                slope = np.polyfit(np.arange(T), dims, 1)[0]
                e_slopes.append(slope)

        if c_slopes and e_slopes:
            y = [0]*len(c_slopes) + [1]*len(e_slopes)
            auroc = roc_auc_score(y, c_slopes + e_slopes)
            if abs(auroc - 0.5) > 0.03:
                print(f"  Layer {layer_indices[l]:2d}: "
                      f"slope correct={np.mean(c_slopes):.4f} error={np.mean(e_slopes):.4f} "
                      f"AUROC={auroc:.4f}")


def analyze_cross_layer(correct, error, L, meta):
    """Cross-layer analysis: stack all layers at each step."""
    print(f"\n{'='*60}")
    print("Cross-layer trajectory analysis")
    print(f"{'='*60}")

    # Stack all layers: trajectory = (T, L*d)
    c_dims = [trajectory_intrinsic_dim(t["h"].reshape(t["h"].shape[0], -1)) for t in correct]
    e_dims = [trajectory_intrinsic_dim(t["h"].reshape(t["h"].shape[0], -1)) for t, _ in error]

    y = [0]*len(c_dims) + [1]*len(e_dims)
    auroc = roc_auc_score(y, c_dims + e_dims) if len(set(y)) >= 2 else 0.5
    print(f"  All-layer concat D_traj: correct={np.mean(c_dims):.2f} error={np.mean(e_dims):.2f} "
          f"AUROC={auroc:.4f}")

    # Cross-layer convergence: ||h_{j,L} - h_{j,L-1}|| / ||h_{j,1} - h_{j,0}||
    c_conv = []
    e_conv = []
    for t in correct:
        h = t["h"]  # (T, L, d)
        ratios = []
        for j in range(h.shape[0]):
            deep = np.linalg.norm(h[j, -1] - h[j, -2])
            shallow = np.linalg.norm(h[j, 1] - h[j, 0]) + 1e-10
            ratios.append(deep / shallow)
        c_conv.append(np.mean(ratios))
    for t, _ in error:
        h = t["h"]
        ratios = []
        for j in range(h.shape[0]):
            deep = np.linalg.norm(h[j, -1] - h[j, -2])
            shallow = np.linalg.norm(h[j, 1] - h[j, 0]) + 1e-10
            ratios.append(deep / shallow)
        e_conv.append(np.mean(ratios))

    auroc_conv = roc_auc_score([0]*len(c_conv) + [1]*len(e_conv), c_conv + e_conv)
    print(f"  Deep/shallow layer ratio: correct={np.mean(c_conv):.4f} error={np.mean(e_conv):.4f} "
          f"AUROC={auroc_conv:.4f}")


def step_level_auroc(correct, error, L, meta):
    """Step-level first-error detection."""
    print(f"\n{'='*60}")
    print("Step-level first-error AUROC")
    print(f"{'='*60}")

    layer_indices = meta["layer_indices"]

    # For each layer, try displacement and curvature as step-level scores
    for l in range(L):
        correct_scores = []
        error_scores = []

        # Displacement at each step
        for t in correct:
            disps = step_displacement(t["h"][:, l, :])
            correct_scores.extend(disps.tolist())

        for t, err_step in error:
            disps = step_displacement(t["h"][:, l, :])
            for j in range(len(disps)):
                actual_step = j + 1
                if actual_step < err_step:
                    correct_scores.append(disps[j])
                elif actual_step == err_step:
                    error_scores.append(disps[j])

        if correct_scores and error_scores:
            y = [0]*len(correct_scores) + [1]*len(error_scores)
            s = correct_scores + error_scores
            auroc = roc_auc_score(y, s)
            if abs(auroc - 0.5) > 0.05:
                print(f"  Layer {layer_indices[l]:2d} displacement: AUROC={auroc:.4f} "
                      f"(correct={len(correct_scores)}, error={len(error_scores)})")

    # Cumulative ID at each step as score
    print(f"\n  Cumulative ID at step:")
    for l in [0, L//2, L-1]:  # just a few layers
        correct_scores = []
        error_scores = []

        for t in correct:
            dims = cumulative_intrinsic_dim(t["h"][:, l, :])
            for j in range(2, len(dims)):
                correct_scores.append(dims[j])

        for t, err_step in error:
            dims = cumulative_intrinsic_dim(t["h"][:, l, :])
            for j in range(2, len(dims)):
                if j < err_step:
                    correct_scores.append(dims[j])
                elif j == err_step:
                    error_scores.append(dims[j])

        if correct_scores and error_scores:
            y = [0]*len(correct_scores) + [1]*len(error_scores)
            auroc = roc_auc_score(y, correct_scores + error_scores)
            print(f"    Layer {layer_indices[l]:2d}: AUROC={auroc:.4f}")


def plot_results(correct, error, L, meta, output_dir):
    """Visualizations."""
    layer_indices = meta["layer_indices"]

    # Figure 1: Cumulative ID over normalized steps, per layer
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    n_bins = 20

    for idx, l in enumerate(range(min(L, 9))):
        ax = axes[idx]
        for label, trajs, color in [("Correct", correct, "blue"),
                                     ("Error", [(t, e) for t, e in error], "red")]:
            binned = np.zeros(n_bins)
            count = np.zeros(n_bins)
            src = trajs if label == "Correct" else [t for t, _ in trajs]
            for traj in src:
                dims = cumulative_intrinsic_dim(traj["h"][:, l, :])
                T = len(dims)
                if T < 3:
                    continue
                for j in range(T):
                    b = min(int(j / (T - 1) * n_bins), n_bins - 1)
                    binned[b] += dims[j]
                    count[b] += 1
            mask = count > 0
            x = np.linspace(0, 1, n_bins)
            ax.plot(x[mask], binned[mask] / count[mask], color=color, linewidth=2, label=label)
        ax.set_title(f"Layer {layer_indices[l]}")
        ax.set_xlabel("Fractional position")
        ax.set_ylabel("Cumulative ID")
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle("Cumulative Intrinsic Dimension over Trajectory\n(Correct=blue, Error=red)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_id.png"), dpi=150)
    plt.close()
    print(f"  Saved cumulative_id.png")

    # Figure 2: Displacement per step
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, l in enumerate(range(min(L, 9))):
        ax = axes[idx]
        for label, trajs, color in [("Correct", correct, "blue"),
                                     ("Error", [(t, e) for t, e in error], "red")]:
            binned = np.zeros(n_bins)
            count = np.zeros(n_bins)
            src = trajs if label == "Correct" else [t for t, _ in trajs]
            for traj in src:
                disps = step_displacement(traj["h"][:, l, :])
                T = len(disps)
                if T < 2:
                    continue
                for j in range(T):
                    b = min(int(j / (T - 1) * n_bins), n_bins - 1)
                    binned[b] += disps[j]
                    count[b] += 1
            mask = count > 0
            x = np.linspace(0, 1, n_bins)
            ax.plot(x[mask], binned[mask] / count[mask], color=color, linewidth=2, label=label)
        ax.set_title(f"Layer {layer_indices[l]}")
        ax.set_xlabel("Fractional position")
        ax.set_ylabel("Displacement ||h_j - h_{j-1}||")
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle("Step-to-Step Displacement over Trajectory", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "displacement.png"), dpi=150)
    plt.close()
    print(f"  Saved displacement.png")

    # Figure 3: Info volume per layer comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    c_vols = np.array([[info_volume(t["h"][:, l, :]) for l in range(L)] for t in correct])
    e_vols = np.array([[info_volume(t["h"][:, l, :]) for l in range(L)] for t, _ in error])

    ax.errorbar(range(L), c_vols.mean(axis=0), yerr=c_vols.std(axis=0),
                label="Correct", color="blue", capsize=3)
    ax.errorbar(range(L), e_vols.mean(axis=0), yerr=e_vols.std(axis=0),
                label="Error", color="red", capsize=3)
    ax.set_xticks(range(L))
    ax.set_xticklabels([str(i) for i in layer_indices])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Information Volume V")
    ax.set_title("CIM Information Volume per Layer")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "info_volume_per_layer.png"), dpi=150)
    plt.close()
    print(f"  Saved info_volume_per_layer.png")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="hidden_trajectory_analysis")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data, meta, L, d = load_data(args.data_path)
    correct, error = extract_h_trajectories(data, L, d)
    print(f"  Correct: {len(correct)}, Error: {len(error)}")

    analyze_per_layer(correct, error, L, meta)
    analyze_displacement(correct, error, L, meta)
    analyze_convergence_trend(correct, error, L, meta)
    analyze_cross_layer(correct, error, L, meta)
    step_level_auroc(correct, error, L, meta)

    print(f"\n{'='*60}")
    print("Generating visualizations")
    print(f"{'='*60}")
    plot_results(correct, error, L, meta, args.output_dir)

    print(f"\n{'='*60}")
    print(f"Done. Outputs in {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
