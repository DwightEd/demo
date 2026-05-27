"""Step 3: Visualisations for the (step × layer) low-rank analysis.

Produces four figures into --outdir:

    fig1_chain_lowrankness.png
        Violin + strip plot of chain-level low-rankness (k=1 and k=2) for
        correct vs error chains. AUROC overlaid in the title.

    fig2_step_signals.png
        Distribution comparison of step-level signals (residual norm,
        layer-profile correlation) between correct steps and first-error steps,
        with AUROC overlaid.

    fig3_spectral_field_heatmaps.png
        Side-by-side (T, L) heatmaps: 4 correct + 4 error trajectories.
        Visual sanity check for the "smooth gradient pattern vs disrupted"
        contrast that motivates the rank-1 hypothesis.

    fig4_lowrank_decomposition.png
        For two trajectories (one correct, one error with clear residual mass)
        show M, the rank-1 reconstruction L_1, and the residual R as heatmaps.

Usage:
    python 03_plot_results.py \
        --spectral data/spectral_field.npz \
        --analysis data/analysis.npz \
        --outdir output/
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from utils import lowrank_decompose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _violin(ax, c_vals, e_vals, title, ylabel):
    """Small helper to draw violin + jittered strip for two groups."""
    c_valid = c_vals[~np.isnan(c_vals)]
    e_valid = e_vals[~np.isnan(e_vals)]

    plot_data, positions, colors, tick_labels = [], [], [], []
    if c_valid.size:
        plot_data.append(c_valid); positions.append(0); colors.append("C0")
        tick_labels.append(f"Correct\n(n={c_valid.size})")
    if e_valid.size:
        plot_data.append(e_valid); positions.append(1); colors.append("C3")
        tick_labels.append(f"Error\n(n={e_valid.size})")
    if not plot_data:
        ax.set_title(f"{title}\n(no data)"); ax.axis("off"); return

    parts = ax.violinplot(plot_data, positions=positions,
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i]); pc.set_alpha(0.3)
    rng = np.random.default_rng(0)
    for d, pos, c in zip(plot_data, positions, colors):
        jitter = rng.uniform(-0.15, 0.15, len(d))
        ax.scatter(pos + jitter, d, c=c, alpha=0.4, s=12, zorder=3)
    ax.set_xticks(positions); ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")


def safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    mask = ~np.isnan(y_score)
    if mask.sum() < 4: return float("nan")
    yt, ys = y_true[mask], y_score[mask]
    if len(np.unique(yt)) < 2: return float("nan")
    return float(roc_auc_score(yt, ys))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_chain_lowrankness(analysis, output_path):
    chain_labels = analysis["chain_labels"]
    lk1 = analysis["lowrank_k1"]
    lk2 = analysis["lowrank_k2"]
    auroc_k1 = float(analysis["auroc_chain_k1"])
    auroc_k2 = float(analysis["auroc_chain_k2"])
    mask_c, mask_e = chain_labels == 0, chain_labels == 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _violin(axes[0], lk1[mask_c], lk1[mask_e],
            f"Chain low-rankness (k=1)\nAUROC = {auroc_k1:.3f}",
            r"$\sigma_1^2 / \Sigma \sigma_k^2$")
    _violin(axes[1], lk2[mask_c], lk2[mask_e],
            f"Chain low-rankness (k=2)\nAUROC = {auroc_k2:.3f}",
            r"$(\sigma_1^2 + \sigma_2^2) / \Sigma \sigma_k^2$")
    fig.suptitle("Trajectory-level: rank-1/2 dominance of (step × layer) spectral field",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"Saved -> {output_path}")


def fig_step_signals(analysis, output_path):
    step_resid = analysis["step_resid"]   # object array of (T,) arrays
    step_corr = analysis["step_corr"]
    labels = analysis["labels"]

    # Collect step-level residual + neg-corr per step, separating
    # "step is first-error" vs "step is a non-error step in any chain".
    resid_err, resid_ok = [], []
    corr_err, corr_ok = [], []
    for i, lab in enumerate(labels):
        rj = step_resid[i]; cj = step_corr[i]
        T = len(rj)
        for j in range(T):
            target_err = (lab >= 0 and j == lab)
            (resid_err if target_err else resid_ok).append(rj[j])
            (corr_err if target_err else corr_ok).append(cj[j])

    resid_err = np.asarray(resid_err, dtype=np.float64)
    resid_ok = np.asarray(resid_ok, dtype=np.float64)
    corr_err = np.asarray(corr_err, dtype=np.float64)
    corr_ok = np.asarray(corr_ok, dtype=np.float64)

    auroc_resid = float(analysis["auroc_step_residual"])
    auroc_negcorr = float(analysis["auroc_step_neg_corr"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _violin(axes[0], resid_ok, resid_err,
            f"Step residual norm\nAUROC = {auroc_resid:.3f}",
            r"$\|R_{j,:}\|_2$")
    axes[0].set_xticklabels([f"Other steps\n(n={(~np.isnan(resid_ok)).sum()})",
                              f"First-error\n(n={(~np.isnan(resid_err)).sum()})"],
                             fontsize=9)
    _violin(axes[1], corr_ok, corr_err,
            f"Layer-profile corr  (AUROC of -corr = {auroc_negcorr:.3f})",
            r"$\rho_j$")
    axes[1].set_xticklabels([f"Other steps\n(n={(~np.isnan(corr_ok)).sum()})",
                              f"First-error\n(n={(~np.isnan(corr_err)).sum()})"],
                             fontsize=9)
    fig.suptitle("Step-level signals: first-error step vs other steps", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"Saved -> {output_path}")


def fig_heatmaps(spectral, output_path, channel="D", n_each=4):
    """Side-by-side (T, L) heatmaps for sanity inspection."""
    key = {"D": "M_D", "V": "M_V", "C": "M_C"}[channel]
    M_field = spectral[key]
    labels = spectral["labels"]

    correct_idx = [i for i, l in enumerate(labels) if l == -1][:n_each]
    error_idx = [i for i, l in enumerate(labels) if l >= 0][:n_each]

    n_show = max(len(correct_idx), len(error_idx))
    if n_show == 0:
        print("No trajectories to draw heatmaps for; skipping fig3."); return

    fig, axes = plt.subplots(2, n_show, figsize=(2.6 * n_show, 6))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for col in range(n_show):
        # Correct row
        if col < len(correct_idx):
            ax = axes[0, col]
            M = np.asarray(M_field[correct_idx[col]], dtype=np.float64)
            im = ax.imshow(M.T, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(f"Correct #{col}", fontsize=9)
            ax.set_xlabel("Step"); ax.set_ylabel("Layer")
            fig.colorbar(im, ax=ax, fraction=0.04)
        else:
            axes[0, col].axis("off")
        # Error row
        if col < len(error_idx):
            ax = axes[1, col]
            i = error_idx[col]
            M = np.asarray(M_field[i], dtype=np.float64)
            im = ax.imshow(M.T, aspect="auto", origin="lower", cmap="viridis")
            tau = int(labels[i])
            ax.axvline(tau - 0.5, color="r", linestyle="--", linewidth=1)
            ax.set_title(f"Error #{col} (τ={tau})", fontsize=9)
            ax.set_xlabel("Step"); ax.set_ylabel("Layer")
            fig.colorbar(im, ax=ax, fraction=0.04)
        else:
            axes[1, col].axis("off")

    fig.suptitle(f"(step × layer) spectral field — channel M_{channel}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"Saved -> {output_path}")


def fig_decomposition(spectral, analysis, output_path, channel="D"):
    """Show M, L_1 (rank-1 reconstruction), R for one correct and one error sample.

    The error sample is the one whose first-error step has the largest residual
    norm among labelled trajectories — i.e. the most visually informative case.
    """
    key = {"D": "M_D", "V": "M_V", "C": "M_C"}[channel]
    M_field = spectral[key]
    labels = spectral["labels"]
    step_resid = analysis["step_resid"]

    correct_idx = [i for i, l in enumerate(labels) if l == -1]
    if not correct_idx:
        print("No correct trajectories; skipping fig4."); return
    pick_c = correct_idx[0]

    error_candidates = [
        (i, step_resid[i][int(labels[i])])
        for i, l in enumerate(labels)
        if l >= 0 and int(labels[i]) < len(step_resid[i])
    ]
    if not error_candidates:
        print("No error trajectories with valid residuals; skipping fig4.")
        return
    pick_e = max(error_candidates, key=lambda x: (x[1] if not np.isnan(x[1]) else -1))[0]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for row, idx, tag in [(0, pick_c, "Correct"), (1, pick_e, f"Error (τ={int(labels[pick_e])})")]:
        M = np.asarray(M_field[idx], dtype=np.float64)
        L1, R, sigmas, _ = lowrank_decompose(M, k=1, center=True)
        # Re-center M for display so M and L1, R are on the same convention
        Mc = M - M.mean(axis=0, keepdims=True)
        for col, (mat, name) in enumerate([(Mc, "M (centered)"),
                                            (L1, "$L_1$ (rank-1)"),
                                            (R,  "$R = M - L_1$")]):
            ax = axes[row, col]
            vmax = float(np.nanmax(np.abs(mat))) if mat.size else 1.0
            im = ax.imshow(mat.T, aspect="auto", origin="lower",
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{tag} — {name}", fontsize=10)
            ax.set_xlabel("Step"); ax.set_ylabel("Layer")
            if labels[idx] >= 0 and col >= 0:
                tau = int(labels[idx])
                ax.axvline(tau - 0.5, color="k", linestyle="--", linewidth=0.8)
            fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(f"Rank-1 low-rank decomposition (channel M_{channel})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"Saved -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectral", default="data/spectral_field.npz")
    parser.add_argument("--analysis", default="data/analysis.npz")
    parser.add_argument("--outdir", default="output/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Loading {args.spectral} ...")
    spectral = np.load(args.spectral, allow_pickle=True)
    print(f"Loading {args.analysis} ...")
    analysis = np.load(args.analysis, allow_pickle=True)
    channel = str(analysis.get("channel", "D"))

    fig_chain_lowrankness(
        analysis, os.path.join(args.outdir, "fig1_chain_lowrankness.png")
    )
    fig_step_signals(
        analysis, os.path.join(args.outdir, "fig2_step_signals.png")
    )
    fig_heatmaps(
        spectral, os.path.join(args.outdir, "fig3_spectral_field_heatmaps.png"),
        channel=channel,
    )
    fig_decomposition(
        spectral, analysis,
        os.path.join(args.outdir, "fig4_lowrank_decomposition.png"),
        channel=channel,
    )


if __name__ == "__main__":
    main()
