"""Step 3: Generate figures — correct vs error trajectory comparison.

Produces two figures:
  Figure 1 (output/figure_1_summary.png):
      Violin + strip plots comparing trajectory-level summary metrics between
      correct and error trajectories. Quick way to see if distributions differ.

  Figure 2 (output/figure_2_step_level.png):
      For each step-level metric, two overlaid curves:
        (a) Correct trajectories: aligned at j=0, mean ± std
        (b) Error trajectories: aligned at first-error step τ (set as 0 on x-axis)

Usage:
    python 03_plot_figure_1.py \
        --input data/metrics.npz \
        --outdir output/ \
        --window_around_tau 6
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------------------------------
# Figure 1: Trajectory-level summary comparison
# ---------------------------------------------------------------------------

SUMMARY_METRICS = [
    ("summary_D_PR",        "$D$ (PR)",             "Intrinsic dimension"),
    ("summary_V",           "$V$ (info volume)",    "Information volume"),
    ("summary_spec_entropy","Spectral entropy",     "Spectral entropy"),
    ("summary_path_length", "Path length",          "Total path length"),
    ("summary_mean_kappa",  r"Mean $\kappa$",       "Mean curvature"),
    ("summary_mean_rho",    r"Mean $\rho$",         "Mean self-consistency"),
    ("summary_linearity",   "Linearity",            "End-to-end / path length"),
    ("summary_n_steps",     "$T$ (steps)",          "Number of steps"),
]


def plot_summary_comparison(data, output_path):
    """Violin + strip plots for correct vs error trajectory-level metrics."""
    labels = data["labels"]
    correct_mask = labels == -1
    error_mask = labels >= 0
    n_correct = int(correct_mask.sum())
    n_error = int(error_mask.sum())

    # Filter to metrics that exist in the data
    available = [(key, title, ylabel) for key, title, ylabel in SUMMARY_METRICS
                 if key in data]
    if not available:
        print("Warning: no summary metrics found in data. Skipping Figure 1.")
        return

    n = len(available)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (key, title, ylabel) in zip(axes, available):
        vals = data[key].astype(np.float64)
        c_vals = vals[correct_mask]
        e_vals = vals[error_mask]

        # Remove NaN for plotting
        c_valid = c_vals[~np.isnan(c_vals)]
        e_valid = e_vals[~np.isnan(e_vals)]

        plot_data = []
        positions = []
        colors = []
        tick_labels = []
        if len(c_valid) > 0:
            plot_data.append(c_valid)
            positions.append(0)
            colors.append("C0")
            tick_labels.append(f"Correct\n(n={len(c_valid)})")
        if len(e_valid) > 0:
            plot_data.append(e_valid)
            positions.append(1)
            colors.append("C3")
            tick_labels.append(f"Error\n(n={len(e_valid)})")

        if len(plot_data) == 0:
            ax.set_title(f"{title}\n(no data)")
            ax.axis("off")
            continue

        parts = ax.violinplot(plot_data, positions=positions, showmedians=True,
                              showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.3)

        # Strip plot (jittered scatter)
        for i, (d, pos) in enumerate(zip(plot_data, positions)):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(d))
            ax.scatter(pos + jitter, d, c=colors[i], alpha=0.4, s=12, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")

        # Welch's t-test if both groups available
        if len(c_valid) >= 2 and len(e_valid) >= 2:
            t_stat, p_val = stats.ttest_ind(c_valid, e_valid, equal_var=False)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax.set_title(f"{title}\np={p_val:.3g} ({sig})", fontsize=10)

    for ax in axes[len(available):]:
        ax.axis("off")

    fig.suptitle(
        f"Trajectory-level: correct (n={n_correct}) vs error (n={n_error})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    print(f"Saved -> {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Step-level τ-aligned evolution
# ---------------------------------------------------------------------------

STEP_METRICS = [
    ("D_prefix",     "$D^{\\rm prefix}_j$",     "Prefix ID"),
    ("V_prefix",     "$V^{\\rm prefix}_j$",     "Prefix info volume"),
    ("spec_entropy", "Spectral entropy$_j$",    "Spectral entropy"),
    ("rho",          r"$\rho_j$",               "Self-consistency"),
    ("kappa",        r"$\kappa_j$",             "Curvature"),
    ("theta",        r"$\theta_j$",             "Tangent rotation (rad)"),
    ("u",            r"$u_j$",                  "Step displacement"),
]


def collect_correct_curves(metric_arrays, labels):
    return [m for m, lab in zip(metric_arrays, labels) if lab == -1]


def collect_error_curves_aligned(metric_arrays, labels, window):
    """Align error trajectories at τ (first-error step)."""
    L = 2 * window + 1
    out = []
    for m, lab in zip(metric_arrays, labels):
        if lab < 0:
            continue
        tau = int(lab)
        T = len(m)
        aligned = np.full(L, np.nan)
        for offset in range(-window, window + 1):
            j = tau + offset
            if 0 <= j < T:
                aligned[offset + window] = m[j]
        out.append(aligned)
    return out


def plot_step_level(data, output_path, window_around_tau=6):
    """Plot step-level metric evolution: correct vs τ-aligned error."""
    labels = data["labels"]
    n_correct = int(np.sum(labels == -1))
    n_error = int(np.sum(labels >= 0))

    available = [(key, title, ylabel) for key, title, ylabel in STEP_METRICS
                 if key in data]
    if not available:
        print("Warning: no step-level metrics found. Skipping Figure 2.")
        return

    n = len(available)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (key, title, ylabel) in zip(axes, available):
        arrays = data[key]

        # --- Correct curves (raw step index) ---
        correct_curves = collect_correct_curves(arrays, labels)
        if correct_curves:
            max_T = max(len(c) for c in correct_curves)
            mat = np.full((len(correct_curves), max_T), np.nan)
            for i, c in enumerate(correct_curves):
                mat[i, :len(c)] = c
            x = np.arange(max_T)
            mean_c = np.nanmean(mat, axis=0)
            std_c = np.nanstd(mat, axis=0)
            ax.plot(x, mean_c, color="C0", label="Correct")
            ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                            color="C0", alpha=0.15)

        # --- Error curves (τ-aligned) on secondary x-axis ---
        error_aligned = collect_error_curves_aligned(
            arrays, labels, window_around_tau
        )
        if error_aligned:
            mat = np.stack(error_aligned, axis=0)
            x_err = np.arange(-window_around_tau, window_around_tau + 1)
            mean_e = np.nanmean(mat, axis=0)
            std_e = np.nanstd(mat, axis=0)
            ax2 = ax.twiny()
            ax2.plot(x_err, mean_e, color="C3", linestyle="--", label="Error")
            ax2.fill_between(x_err, mean_e - std_e, mean_e + std_e,
                             color="C3", alpha=0.15)
            ax2.axvline(0, color="C3", linestyle=":", linewidth=0.8, alpha=0.7)
            ax2.set_xlabel(r"Step relative to $\tau$", color="C3", fontsize=8)
            ax2.tick_params(axis="x", colors="C3", labelsize=7)

        ax.set_xlabel("Step index", color="C0", fontsize=8)
        ax.tick_params(axis="x", colors="C0", labelsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.2)

    for ax in axes[len(available):]:
        ax.axis("off")

    fig.suptitle(
        f"Step-level metrics (correct={n_correct}, error={n_error}, "
        f"window={int(data.get('window', 5))})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    print(f"Saved -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/metrics.npz")
    parser.add_argument("--outdir", default="output/")
    parser.add_argument("--window_around_tau", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.input} ...")
    data = np.load(args.input, allow_pickle=True)

    labels = data["labels"]
    n_total = len(labels)
    n_correct = int(np.sum(labels == -1))
    n_error = n_total - n_correct
    print(f"  -> {n_total} trajectories ({n_correct} correct, {n_error} error)")

    # Figure 1: trajectory-level summary
    plot_summary_comparison(
        data, os.path.join(args.outdir, "figure_1_summary.png")
    )

    # Figure 2: step-level τ-aligned
    plot_step_level(
        data, os.path.join(args.outdir, "figure_2_step_level.png"),
        window_around_tau=args.window_around_tau,
    )


if __name__ == "__main__":
    main()
