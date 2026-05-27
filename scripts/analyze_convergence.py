#!/usr/bin/env python3
"""
Analyze the convergence hypothesis at sample level.

Core hypothesis (from CIM):
    Correct reasoning -> effective rank DECREASES over steps (converges toward low dim)
    Error reasoning -> this convergence trend breaks at or before the error step

This script visualizes and quantifies this effect per-layer and across layers.
No model training — pure data analysis.

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


def spectrum_to_distribution(sigma, eps=1e-10):
    s2 = sigma ** 2 + eps
    return s2 / s2.sum(dim=-1, keepdim=True)


def effective_rank(sigma):
    p = spectrum_to_distribution(sigma)
    H = -(p * torch.log(p + 1e-12)).sum(dim=-1)
    return torch.exp(H)


def load_data(path):
    raw = torch.load(path, weights_only=False)
    data, meta = raw["examples"], raw["meta"]
    L = len(meta["layer_indices"])
    k = meta["k"]
    print(f"Loaded {len(data)} examples, {L} layers x {k} SVs")
    return data, meta, L, k


def compute_eff_rank_trajectories(data, L, k):
    """Compute per-layer effective rank trajectory for each example."""
    correct_trajs = []  # list of (T, L) arrays
    error_trajs = []    # list of (T, L, error_step) tuples

    for ex in data:
        steps = ex["steps"]
        T = len(steps)
        if T < 3:
            continue

        # (T, L) effective rank at each step and layer
        ranks = np.zeros((T, L))
        for j, s in enumerate(steps):
            sigma_ml = s["sigma_ml"]  # (L, k)
            for l in range(L):
                ranks[j, l] = effective_rank(sigma_ml[l]).item()

        if ex["label"] == -1:
            correct_trajs.append(ranks)
        else:
            error_trajs.append((ranks, ex["label"]))

    return correct_trajs, error_trajs


def compute_convergence_slope(ranks_traj, window=None):
    """Compute the slope of effective rank over steps (per layer).

    Negative slope = converging toward low dim.
    Returns (L,) array of slopes.
    """
    T, L = ranks_traj.shape
    if window is not None:
        ranks_traj = ranks_traj[-window:]
        T = ranks_traj.shape[0]
    if T < 2:
        return np.zeros(L)

    t = np.arange(T, dtype=np.float64)
    slopes = np.zeros(L)
    for l in range(L):
        y = ranks_traj[:, l]
        # Linear regression slope
        slopes[l] = np.polyfit(t, y, 1)[0]
    return slopes


def analyze_convergence_trend(correct_trajs, error_trajs, L):
    """Compare convergence slopes between correct and error trajectories."""
    print(f"\n{'='*60}")
    print("Convergence slope analysis (negative = converging)")
    print(f"{'='*60}")

    # For correct: slope over entire trajectory
    correct_slopes = np.array([compute_convergence_slope(t) for t in correct_trajs])
    # For error: slope up to error step vs slope at/after error step
    error_slopes_before = []
    error_slopes_at = []

    for ranks, err_step in error_trajs:
        T = ranks.shape[0]
        if err_step >= 2:
            error_slopes_before.append(compute_convergence_slope(ranks[:err_step]))
        if err_step < T - 1:
            error_slopes_at.append(compute_convergence_slope(ranks[max(0, err_step-1):]))

    error_slopes_before = np.array(error_slopes_before) if error_slopes_before else np.zeros((0, L))
    error_slopes_at = np.array(error_slopes_at) if error_slopes_at else np.zeros((0, L))

    print(f"\n  Correct trajectories ({len(correct_slopes)}):")
    print(f"    Mean slope per layer: {correct_slopes.mean(axis=0)}")
    print(f"    Overall mean slope: {correct_slopes.mean():.4f}")
    print(f"    % with negative slope (converging): {(correct_slopes < 0).mean()*100:.1f}%")

    if len(error_slopes_before) > 0:
        print(f"\n  Error trajectories BEFORE error ({len(error_slopes_before)}):")
        print(f"    Mean slope per layer: {error_slopes_before.mean(axis=0)}")
        print(f"    Overall mean slope: {error_slopes_before.mean():.4f}")
        print(f"    % with negative slope: {(error_slopes_before < 0).mean()*100:.1f}%")

    if len(error_slopes_at) > 0:
        print(f"\n  Error trajectories AT/AFTER error ({len(error_slopes_at)}):")
        print(f"    Mean slope per layer: {error_slopes_at.mean(axis=0)}")
        print(f"    Overall mean slope: {error_slopes_at.mean():.4f}")
        print(f"    % with negative slope: {(error_slopes_at < 0).mean()*100:.1f}%")

    return correct_slopes, error_slopes_before, error_slopes_at


def compute_step_delta_rank(ranks_traj):
    """Compute step-to-step change in effective rank. Negative = converging."""
    T, L = ranks_traj.shape
    deltas = np.diff(ranks_traj, axis=0)  # (T-1, L)
    return deltas


def auroc_from_slopes(correct_trajs, error_trajs, L):
    """Use convergence slope as a simple anomaly score for sequence-level detection."""
    print(f"\n{'='*60}")
    print("Slope-based sequence AUROC")
    print(f"{'='*60}")

    scores = []
    labels = []

    for traj in correct_trajs:
        slope = compute_convergence_slope(traj).mean()
        scores.append(slope)  # more positive slope = less convergence = more anomalous
        labels.append(0)

    for ranks, err_step in error_trajs:
        slope = compute_convergence_slope(ranks).mean()
        scores.append(slope)
        labels.append(1)

    if len(set(labels)) < 2:
        print("  Insufficient data for AUROC")
        return

    auroc = roc_auc_score(labels, scores)
    print(f"  Sequence-level AUROC (mean slope): {auroc:.4f}")

    # Per-layer AUROC
    for l in range(L):
        scores_l = []
        labels_l = []
        for traj in correct_trajs:
            scores_l.append(compute_convergence_slope(traj)[ l])
            labels_l.append(0)
        for ranks, _ in error_trajs:
            scores_l.append(compute_convergence_slope(ranks)[l])
            labels_l.append(1)
        auroc_l = roc_auc_score(labels_l, scores_l)
        if auroc_l > 0.6 or auroc_l < 0.4:
            print(f"  Layer {l}: AUROC={auroc_l:.4f}")


def step_level_auroc(correct_trajs, error_trajs, L):
    """Step-level first-error detection using delta_rank at the step."""
    print(f"\n{'='*60}")
    print("Step-level AUROC (delta_rank at step)")
    print(f"{'='*60}")

    correct_deltas = []
    error_deltas = []

    for traj in correct_trajs:
        deltas = compute_step_delta_rank(traj)  # (T-1, L)
        # Each row is a "correct step" delta
        for j in range(deltas.shape[0]):
            correct_deltas.append(deltas[j])

    for ranks, err_step in error_trajs:
        deltas = compute_step_delta_rank(ranks)
        for j in range(deltas.shape[0]):
            actual_step = j + 1  # delta[j] corresponds to step j+1
            if actual_step < err_step:
                correct_deltas.append(deltas[j])
            elif actual_step == err_step:
                error_deltas.append(deltas[j])

    if not error_deltas or not correct_deltas:
        print("  Insufficient data")
        return

    correct_deltas = np.array(correct_deltas)
    error_deltas = np.array(error_deltas)

    print(f"  Correct steps: {len(correct_deltas)}, Error steps: {len(error_deltas)}")

    # Mean delta_rank across layers as score
    c_scores = correct_deltas.mean(axis=1)
    e_scores = error_deltas.mean(axis=1)
    y_true = [0]*len(c_scores) + [1]*len(e_scores)
    y_score = list(c_scores) + list(e_scores)
    auroc = roc_auc_score(y_true, y_score)
    print(f"  Mean delta_rank AUROC: {auroc:.4f}")
    print(f"    Correct mean: {c_scores.mean():.4f}, Error mean: {e_scores.mean():.4f}")

    # Per-layer
    best_l, best_auroc = -1, 0.5
    for l in range(L):
        c = correct_deltas[:, l]
        e = error_deltas[:, l]
        y = [0]*len(c) + [1]*len(e)
        s = list(c) + list(e)
        a = roc_auc_score(y, s)
        if abs(a - 0.5) > abs(best_auroc - 0.5):
            best_l, best_auroc = l, a
    print(f"  Best single layer: {best_l} with AUROC={best_auroc:.4f}")

    # Cumulative slope up to step j as score
    print(f"\n  Cumulative slope as score:")
    correct_cum = []
    error_cum = []

    for traj in correct_trajs:
        T = traj.shape[0]
        for j in range(2, T):
            slope = compute_convergence_slope(traj[:j+1]).mean()
            correct_cum.append(slope)

    for ranks, err_step in error_trajs:
        T = ranks.shape[0]
        for j in range(2, T):
            slope = compute_convergence_slope(ranks[:j+1]).mean()
            if j < err_step:
                correct_cum.append(slope)
            elif j == err_step:
                error_cum.append(slope)

    if error_cum and correct_cum:
        y = [0]*len(correct_cum) + [1]*len(error_cum)
        s = correct_cum + error_cum
        auroc_cum = roc_auc_score(y, s)
        print(f"    Cumulative slope AUROC: {auroc_cum:.4f}")
        print(f"    Correct mean: {np.mean(correct_cum):.4f}, Error mean: {np.mean(error_cum):.4f}")


def plot_trajectories(correct_trajs, error_trajs, L, output_dir, meta):
    """Plot effective rank trajectories: correct vs error, per layer."""
    layer_indices = meta["layer_indices"]

    # Pick a subset of layers to show (early, middle, late)
    if L > 6:
        show_layers = [0, L//4, L//2, 3*L//4, L-2, L-1]
    else:
        show_layers = list(range(L))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, l in enumerate(show_layers[:6]):
        ax = axes[idx]

        # Correct trajectories (blue, semi-transparent)
        for traj in correct_trajs[:30]:  # limit for readability
            T = traj.shape[0]
            ax.plot(range(T), traj[:, l], color="blue", alpha=0.15, linewidth=0.8)

        # Error trajectories (red before error, orange at/after error)
        for ranks, err_step in error_trajs[:30]:
            T = ranks.shape[0]
            ax.plot(range(err_step), ranks[:err_step, l],
                    color="green", alpha=0.2, linewidth=0.8)
            if err_step < T:
                ax.plot(range(err_step, T), ranks[err_step:, l],
                        color="red", alpha=0.3, linewidth=1.2)
                ax.axvline(err_step, color="red", alpha=0.1, linewidth=0.5)

        # Mean correct trajectory
        max_T_correct = max(t.shape[0] for t in correct_trajs) if correct_trajs else 0
        if max_T_correct > 0:
            mean_correct = np.zeros(max_T_correct)
            count = np.zeros(max_T_correct)
            for traj in correct_trajs:
                T = traj.shape[0]
                mean_correct[:T] += traj[:, l]
                count[:T] += 1
            mask = count > 0
            mean_correct[mask] /= count[mask]
            ax.plot(np.where(mask)[0], mean_correct[mask],
                    color="blue", linewidth=2.5, label="correct (mean)")

        ax.set_title(f"Layer {layer_indices[l]}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Effective Rank")
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle("Effective Rank Trajectories: Correct (blue) vs Error (red after error step)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eff_rank_trajectories.png"), dpi=150)
    plt.close()
    print(f"  Saved eff_rank_trajectories.png")


def plot_convergence_comparison(correct_trajs, error_trajs, L, output_dir):
    """Plot: normalized effective rank (fraction of initial) over normalized step position."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mean normalized trajectory (eff_rank / eff_rank[0]) vs fractional step
    ax = axes[0]
    n_bins = 20
    correct_binned = np.zeros((n_bins, L))
    correct_count = np.zeros(n_bins)

    for traj in correct_trajs:
        T = traj.shape[0]
        if T < 3:
            continue
        # Normalize: fraction of initial rank
        init_rank = traj[0] + 1e-8
        normed = traj / init_rank
        for j in range(T):
            frac = j / (T - 1)
            bin_idx = min(int(frac * n_bins), n_bins - 1)
            correct_binned[bin_idx] += normed[j].mean()  # mean across layers
            correct_count[bin_idx] += 1

    error_binned = np.zeros((n_bins, L))
    error_count = np.zeros(n_bins)
    for ranks, _ in error_trajs:
        T = ranks.shape[0]
        if T < 3:
            continue
        init_rank = ranks[0] + 1e-8
        normed = ranks / init_rank
        for j in range(T):
            frac = j / (T - 1)
            bin_idx = min(int(frac * n_bins), n_bins - 1)
            error_binned[bin_idx] += normed[j].mean()
            error_count[bin_idx] += 1

    x = np.linspace(0, 1, n_bins)
    mask_c = correct_count > 0
    mask_e = error_count > 0
    ax.plot(x[mask_c], (correct_binned[mask_c, :].mean(axis=1) / correct_count[mask_c]),
            'b-', linewidth=2, label="Correct")
    ax.plot(x[mask_e], (error_binned[mask_e, :].mean(axis=1) / error_count[mask_e]),
            'r-', linewidth=2, label="Error")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Fractional step position")
    ax.set_ylabel("Normalized effective rank (fraction of initial)")
    ax.set_title("Convergence: rank / rank[0] over trajectory")
    ax.legend()

    # Right: distribution of overall slopes
    ax = axes[1]
    c_slopes = [compute_convergence_slope(t).mean() for t in correct_trajs]
    e_slopes = [compute_convergence_slope(r).mean() for r, _ in error_trajs]

    ax.hist(c_slopes, bins=30, alpha=0.6, color="blue", label=f"Correct (n={len(c_slopes)})", density=True)
    ax.hist(e_slopes, bins=30, alpha=0.6, color="red", label=f"Error (n={len(e_slopes)})", density=True)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("Mean convergence slope (negative = converging)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of convergence slopes")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved convergence_comparison.png")


def plot_per_layer_signal(correct_trajs, error_trajs, L, output_dir, meta):
    """For each layer, compute mean eff_rank for correct vs error steps and show difference."""
    layer_indices = meta["layer_indices"]

    correct_means = np.zeros(L)
    correct_n = 0
    for traj in correct_trajs:
        correct_means += traj.mean(axis=0)
        correct_n += 1
    if correct_n > 0:
        correct_means /= correct_n

    error_means = np.zeros(L)
    error_n = 0
    for ranks, err_step in error_trajs:
        if err_step < ranks.shape[0]:
            error_means += ranks[err_step]
            error_n += 1
    if error_n > 0:
        error_means /= error_n

    # Per-layer slope AUROC
    slope_aurocs = np.zeros(L)
    for l in range(L):
        scores = []
        labels = []
        for traj in correct_trajs:
            scores.append(compute_convergence_slope(traj)[l])
            labels.append(0)
        for ranks, _ in error_trajs:
            scores.append(compute_convergence_slope(ranks)[l])
            labels.append(1)
        if len(set(labels)) >= 2:
            slope_aurocs[l] = roc_auc_score(labels, scores)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.bar(range(L), correct_means, alpha=0.6, label="Correct (mean)", color="blue")
    ax.bar(range(L), error_means, alpha=0.6, label="Error (at error step)", color="red")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Effective rank")
    ax.set_title("Mean effective rank per layer")
    ax.legend(fontsize=8)

    ax = axes[1]
    diff = error_means - correct_means
    colors = ["red" if d > 0 else "blue" for d in diff]
    ax.bar(range(L), diff, color=colors, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Error - Correct")
    ax.set_title("Effective rank difference (error - correct)")

    ax = axes[2]
    ax.bar(range(L), slope_aurocs, color="purple", alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-layer convergence slope AUROC")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_layer_signal.png"), dpi=150)
    plt.close()
    print(f"  Saved per_layer_signal.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="convergence_analysis")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data, meta, L, k = load_data(args.data_path)
    correct_trajs, error_trajs = compute_eff_rank_trajectories(data, L, k)
    print(f"  Correct: {len(correct_trajs)}, Error: {len(error_trajs)}")

    # 1. Analyze convergence slopes
    analyze_convergence_trend(correct_trajs, error_trajs, L)

    # 2. Sequence-level AUROC from slopes
    auroc_from_slopes(correct_trajs, error_trajs, L)

    # 3. Step-level AUROC from delta_rank and cumulative slope
    step_level_auroc(correct_trajs, error_trajs, L)

    # 4. Visualizations
    print(f"\n{'='*60}")
    print("Generating visualizations")
    print(f"{'='*60}")
    plot_trajectories(correct_trajs, error_trajs, L, args.output_dir, meta)
    plot_convergence_comparison(correct_trajs, error_trajs, L, args.output_dir)
    plot_per_layer_signal(correct_trajs, error_trajs, L, args.output_dir, meta)

    print(f"\n{'='*60}")
    print(f"Done. Outputs in {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
