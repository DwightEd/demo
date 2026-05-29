"""Step 4: Comprehensive visualisation of (step x layer) spectral field.

Goes beyond the 4 standard figures in 03_plot_results.py to provide a
multi-angle diagnostic of *where* the signal is and *why* it might be weak.

Produces 6 groups of figures into --outdir:

  Group 1  fig_g1_sample_heatmaps.png
           Individual sample heatmaps (D, V, C side-by-side) for 3 correct + 3 error,
           with D/D[0] normalised view. Shows per-sample full picture.

  Group 2  fig_g2_stepwise_boxplots.png
           Per-step box/violin comparing correct vs error at actual step indices
           (first N steps), for both raw D and D/D[0]. Shows distributional overlap.

  Group 3  fig_g3_joint_scatter.png
           ER vs Energy scatter at (step, layer) level, coloured by correct/error.
           Plus ER vs TopConcentration. Shows if correct/error occupy different regions.

  Group 4  fig_g4_layer_discriminability.png
           Per-layer AUROC and Cohen's d for multiple signals (raw D, D/D[0],
           step-residual, delta-D). Identifies which layers carry the most signal.

  Group 5  fig_g5_step_deltas.png
           Step-wise differences (Delta_D = D_j - D_{j-1}) distribution for
           correct vs error. Tests spectral stability hypothesis directly.

  Group 6  fig_g6_crosslayer_consistency.png
           Per-step cross-layer std / CV of D, comparing correct vs error.
           Tests whether correct reasoning has more coordinated layer behaviour.

  Bonus    fig_g7_auroc_summary.png
           Summary bar chart of AUROC across all signals x all layers + aggregated.

Usage:
    python 04_comprehensive_viz.py \
        --spectral data/gsm8k_spectral.npz \
        --outdir output/gsm8k_comprehensive/
"""

from __future__ import annotations

import argparse
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    mask = ~np.isnan(y_score)
    if mask.sum() < 4:
        return float("nan")
    yt, ys = y_true[mask], y_score[mask]
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, ys))


def cohens_d(a, b):
    """Effect size between two groups."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1))
                         / (len(a) + len(b) - 2))
    if pooled_std < 1e-15:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def load_spectral(path):
    """Load spectral field data, return structured dict."""
    data = np.load(path, allow_pickle=True)
    labels = data["labels"]  # -1 = correct, >=0 = first-error step
    M_D = data["M_D"]  # object array of (T_i, L) matrices
    M_V = data["M_V"]
    M_C = data["M_C"]
    layers_used = data["layers_used"]

    n = len(labels)
    correct_idx = [i for i in range(n) if labels[i] == -1]
    error_idx = [i for i in range(n) if labels[i] >= 0]

    return {
        "labels": labels,
        "M_D": M_D, "M_V": M_V, "M_C": M_C,
        "layers_used": layers_used,
        "correct_idx": correct_idx,
        "error_idx": error_idx,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Group 1: Individual sample heatmaps (raw + normalised)
# ---------------------------------------------------------------------------

def plot_g1_sample_heatmaps(spec, outdir, n_each=3):
    """Per-sample (T x L) heatmaps: raw D, D/D[0], V, C for a few examples."""
    labels = spec["labels"]
    M_D, M_V, M_C = spec["M_D"], spec["M_V"], spec["M_C"]
    ci = spec["correct_idx"][:n_each]
    ei = spec["error_idx"][:n_each]

    fig, axes = plt.subplots(2 * n_each, 4, figsize=(20, 3.5 * 2 * n_each))

    for row_offset, indices, tag in [(0, ci, "Correct"), (n_each, ei, "Error")]:
        for k, idx in enumerate(indices):
            r = row_offset + k
            D = np.asarray(M_D[idx], dtype=np.float64)
            V = np.asarray(M_V[idx], dtype=np.float64)
            C = np.asarray(M_C[idx], dtype=np.float64)

            # D/D[0] normalised
            D0 = D[0:1, :]
            D0[D0 < 1e-10] = 1e-10
            D_norm = D / D0

            tau = int(labels[idx]) if labels[idx] >= 0 else None
            label_str = f"{tag} #{k}" + (f" (τ={tau})" if tau is not None else "")

            for col, (mat, name, cmap) in enumerate([
                (D, "Eff. Rank (D)", "viridis"),
                (D_norm, "D / D[0]", "RdBu_r"),
                (V, "Spectral Energy (V)", "magma"),
                (C, "Top Conc. (C)", "plasma"),
            ]):
                ax = axes[r, col]
                if name == "D / D[0]":
                    vmin, vmax = 0.5, 1.5
                    im = ax.imshow(mat.T, aspect="auto", origin="lower",
                                   cmap=cmap, vmin=vmin, vmax=vmax)
                else:
                    im = ax.imshow(mat.T, aspect="auto", origin="lower", cmap=cmap)
                if tau is not None:
                    ax.axvline(tau - 0.5, color="red", linestyle="--", linewidth=1.5)
                ax.set_title(f"{label_str}\n{name}", fontsize=8)
                ax.set_xlabel("Step", fontsize=7)
                ax.set_ylabel("Layer", fontsize=7)
                fig.colorbar(im, ax=ax, fraction=0.04)

    fig.suptitle("Group 1: Per-sample (T × L) spectral field — D, D/D[0], V, C",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_g1_sample_heatmaps.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_g1_sample_heatmaps.png")


# ---------------------------------------------------------------------------
# Group 2: Per-step boxplots (correct vs error at actual step indices)
# ---------------------------------------------------------------------------

def plot_g2_stepwise_boxplots(spec, outdir, max_steps=10):
    """Box plots of D and D/D[0] at each step index, correct vs error."""
    labels = spec["labels"]
    M_D = spec["M_D"]
    ci, ei = spec["correct_idx"], spec["error_idx"]

    # Collect layer-averaged D per step
    def collect(indices, normalise=False):
        by_step = {j: [] for j in range(max_steps)}
        for idx in indices:
            D = np.asarray(M_D[idx], dtype=np.float64)
            d_mean = np.nanmean(D, axis=1)  # (T,)
            if normalise and d_mean[0] > 1e-10:
                d_mean = d_mean / d_mean[0]
            for j in range(min(len(d_mean), max_steps)):
                by_step[j].append(d_mean[j])
        return by_step

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for ax_idx, (normalise, title) in enumerate([
        (False, "Raw D (layer-averaged) per step"),
        (True, "D / D[0] (layer-averaged) per step"),
    ]):
        ax = axes[ax_idx]
        c_data = collect(ci, normalise=normalise)
        e_data = collect(ei, normalise=normalise)

        positions_c = []
        positions_e = []
        data_c = []
        data_e = []

        for j in range(max_steps):
            if c_data[j] and e_data[j]:
                positions_c.append(j * 3)
                positions_e.append(j * 3 + 1)
                data_c.append(c_data[j])
                data_e.append(e_data[j])

        if data_c:
            bp_c = ax.boxplot(data_c, positions=positions_c, widths=0.8,
                              patch_artist=True, showfliers=True,
                              flierprops=dict(markersize=3))
            for patch in bp_c["boxes"]:
                patch.set_facecolor("C0"); patch.set_alpha(0.4)

        if data_e:
            bp_e = ax.boxplot(data_e, positions=positions_e, widths=0.8,
                              patch_artist=True, showfliers=True,
                              flierprops=dict(markersize=3))
            for patch in bp_e["boxes"]:
                patch.set_facecolor("C3"); patch.set_alpha(0.4)

        tick_pos = [j * 3 + 0.5 for j in range(len(data_c))]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([f"Step {j}" for j in range(len(data_c))], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("D" if not normalise else "D / D[0]", fontsize=10)
        if normalise:
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.legend([plt.Rectangle((0,0),1,1, fc="C0", alpha=0.4),
                   plt.Rectangle((0,0),1,1, fc="C3", alpha=0.4)],
                  ["Correct", "Error"], fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Group 2: Per-step distribution — correct vs error", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outdir, "fig_g2_stepwise_boxplots.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig_g2_stepwise_boxplots.png")


# ---------------------------------------------------------------------------
# Group 3: Joint ER-Energy scatter
# ---------------------------------------------------------------------------

def plot_g3_joint_scatter(spec, outdir, max_points=2000):
    """Scatter plots of (D, V) and (D, C) per (step, layer) cell."""
    labels = spec["labels"]
    M_D, M_V, M_C = spec["M_D"], spec["M_V"], spec["M_C"]

    c_D, c_V, c_C = [], [], []
    e_D, e_V, e_C = [], [], []

    for i in range(spec["n"]):
        D = np.asarray(M_D[i], dtype=np.float64).ravel()
        V = np.asarray(M_V[i], dtype=np.float64).ravel()
        C = np.asarray(M_C[i], dtype=np.float64).ravel()
        if labels[i] == -1:
            c_D.extend(D); c_V.extend(V); c_C.extend(C)
        else:
            e_D.extend(D); e_V.extend(V); e_C.extend(C)

    c_D, c_V, c_C = np.array(c_D), np.array(c_V), np.array(c_C)
    e_D, e_V, e_C = np.array(e_D), np.array(e_V), np.array(e_C)

    # Subsample if too many points
    rng = np.random.default_rng(42)
    if len(c_D) > max_points:
        idx = rng.choice(len(c_D), max_points, replace=False)
        c_D, c_V, c_C = c_D[idx], c_V[idx], c_C[idx]
    if len(e_D) > max_points:
        idx = rng.choice(len(e_D), max_points, replace=False)
        e_D, e_V, e_C = e_D[idx], e_V[idx], e_C[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # D vs log(V)
    ax = axes[0]
    ax.scatter(c_D, np.log10(c_V + 1), c="C0", alpha=0.15, s=8, label="Correct")
    ax.scatter(e_D, np.log10(e_V + 1), c="C3", alpha=0.15, s=8, label="Error")
    ax.set_xlabel("Effective Rank (D)", fontsize=10)
    ax.set_ylabel("log₁₀(Spectral Energy + 1)", fontsize=10)
    ax.set_title("D vs V (all step×layer cells)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # D vs C
    ax = axes[1]
    ax.scatter(c_D, c_C, c="C0", alpha=0.15, s=8, label="Correct")
    ax.scatter(e_D, e_C, c="C3", alpha=0.15, s=8, label="Error")
    ax.set_xlabel("Effective Rank (D)", fontsize=10)
    ax.set_ylabel("Top Concentration (C)", fontsize=10)
    ax.set_title("D vs C (all step×layer cells)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Group 3: Joint spectral indicator scatter — correct vs error",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, "fig_g3_joint_scatter.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig_g3_joint_scatter.png")


# ---------------------------------------------------------------------------
# Group 4: Per-layer discriminability (AUROC + Cohen's d)
# ---------------------------------------------------------------------------

def plot_g4_layer_discriminability(spec, outdir):
    """For each layer, compute chain-level AUROC using multiple signals."""
    labels = spec["labels"]
    M_D = spec["M_D"]
    layers = spec["layers_used"]
    L = len(layers)
    ci, ei = spec["correct_idx"], spec["error_idx"]
    chain_labels = (labels >= 0).astype(np.int32)

    signals = {
        "mean_D": lambda D: np.nanmean(D, axis=0),           # (L,) mean across steps
        "std_D": lambda D: np.nanstd(D, axis=0),             # (L,) std across steps
        "D_last/D_first": lambda D: D[-1] / (D[0] + 1e-10),  # (L,) ratio
        "mean_deltaD": lambda D: np.nanmean(np.diff(D, axis=0), axis=0) if D.shape[0] > 1 else np.zeros(D.shape[1]),
        "std_deltaD": lambda D: np.nanstd(np.diff(D, axis=0), axis=0) if D.shape[0] > 1 else np.zeros(D.shape[1]),
    }

    results = {name: {"auroc": np.full(L, np.nan), "d": np.full(L, np.nan)}
               for name in signals}

    for sig_name, sig_fn in signals.items():
        # Compute per-sample per-layer feature
        features = np.full((spec["n"], L), np.nan)
        for i in range(spec["n"]):
            D = np.asarray(M_D[i], dtype=np.float64)
            feat = sig_fn(D)
            features[i, :len(feat)] = feat[:L]

        for l in range(L):
            col = features[:, l]
            auroc = safe_auroc(chain_labels, col)
            # Flip if below 0.5 (we want max discriminability regardless of sign)
            if not np.isnan(auroc) and auroc < 0.5:
                auroc_flipped = safe_auroc(chain_labels, -col)
                results[sig_name]["auroc"][l] = auroc_flipped
            else:
                results[sig_name]["auroc"][l] = auroc

            c_vals = col[labels == -1]
            e_vals = col[labels >= 0]
            results[sig_name]["d"][l] = abs(cohens_d(c_vals, e_vals))

    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 2, figsize=(16, 3.5 * n_signals))

    for row, (sig_name, res) in enumerate(results.items()):
        # AUROC
        ax = axes[row, 0]
        ax.bar(range(L), res["auroc"], color="steelblue", alpha=0.7)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("AUROC", fontsize=9)
        ax.set_title(f"{sig_name} — AUROC per layer", fontsize=10)
        ax.set_xticks(range(0, L, max(1, L//10)))
        ax.set_xticklabels([str(layers[i]) for i in range(0, L, max(1, L//10))],
                           fontsize=7)
        ax.set_xlabel("Layer index", fontsize=8)
        ax.set_ylim(0.3, 0.8)
        ax.grid(True, alpha=0.2, axis="y")

        # Cohen's d
        ax = axes[row, 1]
        ax.bar(range(L), res["d"], color="coral", alpha=0.7)
        ax.set_ylabel("|Cohen's d|", fontsize=9)
        ax.set_title(f"{sig_name} — |Cohen's d| per layer", fontsize=10)
        ax.set_xticks(range(0, L, max(1, L//10)))
        ax.set_xticklabels([str(layers[i]) for i in range(0, L, max(1, L//10))],
                           fontsize=7)
        ax.set_xlabel("Layer index", fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Group 4: Per-layer discriminability — chain-level signals",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_g4_layer_discriminability.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_g4_layer_discriminability.png")


# ---------------------------------------------------------------------------
# Group 5: Step deltas (ΔD = D_j - D_{j-1})
# ---------------------------------------------------------------------------

def plot_g5_step_deltas(spec, outdir):
    """Distribution of step-wise spectral changes for correct vs error."""
    labels = spec["labels"]
    M_D = spec["M_D"]

    # Collect ΔD (layer-averaged) for all consecutive step pairs
    c_deltas = []
    e_deltas_before = []  # error chain, steps before error
    e_deltas_at = []      # error chain, the error step transition
    e_deltas_after = []   # error chain, steps after error

    for i in range(spec["n"]):
        D = np.asarray(M_D[i], dtype=np.float64)
        d_mean = np.nanmean(D, axis=1)  # (T,)
        deltas = np.diff(d_mean)  # (T-1,)

        if labels[i] == -1:
            c_deltas.extend(deltas)
        else:
            tau = int(labels[i])
            for j, delta in enumerate(deltas):
                step_to = j + 1  # delta is D[j+1] - D[j]
                if step_to < tau:
                    e_deltas_before.append(delta)
                elif step_to == tau:
                    e_deltas_at.append(delta)
                else:
                    e_deltas_after.append(delta)

    c_deltas = np.array(c_deltas)
    e_deltas_before = np.array(e_deltas_before)
    e_deltas_at = np.array(e_deltas_at)
    e_deltas_after = np.array(e_deltas_after)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram: correct vs all error
    ax = axes[0]
    all_error = np.concatenate([e_deltas_before, e_deltas_at, e_deltas_after])
    if len(c_deltas) > 0:
        ax.hist(c_deltas, bins=40, alpha=0.5, color="C0", label=f"Correct (n={len(c_deltas)})",
                density=True)
    if len(all_error) > 0:
        ax.hist(all_error, bins=40, alpha=0.5, color="C3", label=f"Error (n={len(all_error)})",
                density=True)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_title("ΔD distribution: correct vs error (all steps)", fontsize=10)
    ax.set_xlabel("ΔD = D[j+1] - D[j] (layer-averaged)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Histogram: error subdivided
    ax = axes[1]
    for arr, label, color in [
        (e_deltas_before, f"Before error (n={len(e_deltas_before)})", "green"),
        (e_deltas_at, f"At error step (n={len(e_deltas_at)})", "red"),
        (e_deltas_after, f"After error (n={len(e_deltas_after)})", "orange"),
    ]:
        if len(arr) > 0:
            ax.hist(arr, bins=30, alpha=0.4, color=color, label=label, density=True)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_title("ΔD in error chains: before / at / after error step", fontsize=10)
    ax.set_xlabel("ΔD")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # |ΔD| comparison
    ax = axes[2]
    groups = []
    group_labels = []
    colors = []
    for arr, label, color in [
        (c_deltas, "Correct", "C0"),
        (e_deltas_before, "Err:before", "green"),
        (e_deltas_at, "Err:at", "red"),
        (e_deltas_after, "Err:after", "orange"),
    ]:
        if len(arr) > 0:
            groups.append(np.abs(arr))
            group_labels.append(f"{label}\n(n={len(arr)})")
            colors.append(color)
    if groups:
        parts = ax.violinplot(groups, showmedians=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.4)
        ax.set_xticks(range(1, len(groups)+1))
        ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_title("|ΔD| magnitude: correct vs error segments", fontsize=10)
    ax.set_ylabel("|ΔD|")
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Group 5: Step-wise spectral change (stability analysis)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, "fig_g5_step_deltas.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig_g5_step_deltas.png")


# ---------------------------------------------------------------------------
# Group 6: Cross-layer consistency
# ---------------------------------------------------------------------------

def plot_g6_crosslayer_consistency(spec, outdir, max_steps=12):
    """Per-step cross-layer std and CV of D, correct vs error."""
    labels = spec["labels"]
    M_D = spec["M_D"]
    ci, ei = spec["correct_idx"], spec["error_idx"]

    def collect_crosslayer_stats(indices):
        std_by_step = {j: [] for j in range(max_steps)}
        cv_by_step = {j: [] for j in range(max_steps)}
        for idx in indices:
            D = np.asarray(M_D[idx], dtype=np.float64)
            for j in range(min(D.shape[0], max_steps)):
                row = D[j, :]
                row_valid = row[~np.isnan(row)]
                if len(row_valid) < 2:
                    continue
                std_by_step[j].append(np.std(row_valid))
                mean_val = np.mean(row_valid)
                if abs(mean_val) > 1e-10:
                    cv_by_step[j].append(np.std(row_valid) / abs(mean_val))
        return std_by_step, cv_by_step

    c_std, c_cv = collect_crosslayer_stats(ci)
    e_std, e_cv = collect_crosslayer_stats(ei)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for ax_idx, (c_data, e_data, ylabel, title) in enumerate([
        (c_std, e_std, "Cross-layer Std(D)", "Cross-layer standard deviation of D per step"),
        (c_cv, e_cv, "Cross-layer CV(D)", "Cross-layer coefficient of variation of D per step"),
    ]):
        ax = axes[ax_idx]
        steps_with_data = [j for j in range(max_steps) if c_data[j] and e_data[j]]

        c_means = [np.mean(c_data[j]) for j in steps_with_data]
        c_stds = [np.std(c_data[j]) / np.sqrt(len(c_data[j])) for j in steps_with_data]
        e_means = [np.mean(e_data[j]) for j in steps_with_data]
        e_stds = [np.std(e_data[j]) / np.sqrt(len(e_data[j])) for j in steps_with_data]

        x = np.array(steps_with_data)
        ax.errorbar(x - 0.15, c_means, yerr=c_stds, fmt="o-", color="C0",
                    label="Correct", capsize=3, markersize=5)
        ax.errorbar(x + 0.15, e_means, yerr=e_stds, fmt="s-", color="C3",
                    label="Error", capsize=3, markersize=5)
        ax.set_xlabel("Step index", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xticks(steps_with_data)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Group 6: Cross-layer consistency — correct vs error", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outdir, "fig_g6_crosslayer_consistency.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig_g6_crosslayer_consistency.png")


# ---------------------------------------------------------------------------
# Bonus: AUROC summary bar chart
# ---------------------------------------------------------------------------

def plot_g7_auroc_summary(spec, outdir):
    """Summary of chain-level AUROC across multiple aggregated signals."""
    labels = spec["labels"]
    M_D, M_V, M_C = spec["M_D"], spec["M_V"], spec["M_C"]
    chain_labels = (labels >= 0).astype(np.int32)
    n = spec["n"]

    scored = {}

    # For each trajectory, compute multiple scalar features
    for signal_name, fn in [
        ("mean_D (all layers)", lambda i: np.nanmean(M_D[i])),
        ("std_D (all layers)", lambda i: np.nanstd(M_D[i])),
        ("mean_V (all layers)", lambda i: np.nanmean(M_V[i])),
        ("mean_C (all layers)", lambda i: np.nanmean(M_C[i])),
        ("D_range (max-min)", lambda i: np.nanmax(M_D[i]) - np.nanmin(M_D[i])),
        ("D last/first (mean layer)",
         lambda i: np.nanmean(M_D[i][-1]) / (np.nanmean(M_D[i][0]) + 1e-10)),
        ("mean |ΔD|",
         lambda i: np.nanmean(np.abs(np.diff(np.nanmean(np.asarray(M_D[i], dtype=np.float64), axis=1)))) if M_D[i].shape[0] > 1 else np.nan),
        ("std ΔD",
         lambda i: np.nanstd(np.diff(np.nanmean(np.asarray(M_D[i], dtype=np.float64), axis=1))) if M_D[i].shape[0] > 1 else np.nan),
        ("cross-layer std D (mean step)",
         lambda i: np.nanmean([np.nanstd(M_D[i][j]) for j in range(M_D[i].shape[0])])),
        ("cross-layer CV D (mean step)",
         lambda i: np.nanmean([np.nanstd(M_D[i][j]) / (np.nanmean(M_D[i][j]) + 1e-10) for j in range(M_D[i].shape[0])])),
    ]:
        scores = np.array([fn(i) for i in range(n)], dtype=np.float64)
        auroc = safe_auroc(chain_labels, scores)
        auroc_neg = safe_auroc(chain_labels, -scores)
        best = max(auroc, auroc_neg) if not (np.isnan(auroc) or np.isnan(auroc_neg)) else auroc
        scored[signal_name] = best

    # Sort by AUROC
    sorted_items = sorted(scored.items(), key=lambda x: x[1] if not np.isnan(x[1]) else 0,
                          reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    colors = ["forestgreen" if v >= 0.6 else "steelblue" if v >= 0.55 else "gray"
              for v in values]
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("AUROC (best orientation)", fontsize=10)
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="Random")
    ax.axvline(0.6, color="orange", linestyle="--", alpha=0.5, label="Weak signal")
    ax.set_xlim(0.35, 0.75)
    ax.set_title("Chain-level AUROC across aggregated spectral signals", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="x")
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_g7_auroc_summary.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig_g7_auroc_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectral", default="data/gsm8k_spectral.npz")
    parser.add_argument("--outdir", default="output/gsm8k_comprehensive/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Loading {args.spectral} ...")
    spec = load_spectral(args.spectral)
    print(f"  {spec['n']} trajectories "
          f"({len(spec['correct_idx'])} correct, {len(spec['error_idx'])} error), "
          f"L = {len(spec['layers_used'])} layers")

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("\nGenerating Group 1: Per-sample heatmaps ...")
    plot_g1_sample_heatmaps(spec, args.outdir)

    print("Generating Group 2: Per-step boxplots ...")
    plot_g2_stepwise_boxplots(spec, args.outdir)

    print("Generating Group 3: Joint scatter ...")
    plot_g3_joint_scatter(spec, args.outdir)

    print("Generating Group 4: Per-layer discriminability ...")
    plot_g4_layer_discriminability(spec, args.outdir)

    print("Generating Group 5: Step deltas ...")
    plot_g5_step_deltas(spec, args.outdir)

    print("Generating Group 6: Cross-layer consistency ...")
    plot_g6_crosslayer_consistency(spec, args.outdir)

    print("Generating Group 7: AUROC summary ...")
    plot_g7_auroc_summary(spec, args.outdir)

    print(f"\nDone. All figures in {args.outdir}")


if __name__ == "__main__":
    main()
