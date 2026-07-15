from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from .evaluate import auroc, evaluate_all, finite_json, save_json


DEFAULT_STEP_METRICS = (
    "prompt_frac",
    "off_prompt",
    "prompt_control_ratio",
    "prefix_frac",
    "prefix_lock_ratio",
    "random_frac",
    "token_entropy",
    "token_nll",
    "icr_mean",
    "icr_max",
    "icr_top20_mean",
    "geom_boundary_proj",
    "geom_healthy_residual",
    "geom_lid",
    "geom_knn_error_frac",
    "geom_knn_label_entropy",
    "geom_local_spec_entropy",
    "geom_layer_nbr_instability",
    "geom_compartment_score",
    "sd_tube_dist",
    "sd_spectral_leak",
    "sd_tangent_off",
    "sd_committor",
    "sd_step_speed",
    "step_len",
    "rel_pos",
)


def response_error_labels(metrics: Mapping[str, Any]) -> np.ndarray:
    if "is_correct" in metrics:
        correct = np.asarray(metrics["is_correct"], dtype=np.float64)
        valid = np.isfinite(correct) & np.isin(correct, [0, 1])
        if np.any(valid):
            labels = np.full(correct.shape, -1, dtype=np.int32)
            labels[valid] = (correct[valid] == 0).astype(np.int32)
            return labels
    gold = np.asarray(metrics["gold_error_step"], dtype=np.int64)
    labels = np.full(gold.shape, -1, dtype=np.int32)
    valid = gold >= -1
    labels[valid] = (gold[valid] >= 0).astype(np.int32)
    return labels


def metric_index(names: Sequence[str], metric_name: str) -> int | None:
    try:
        return [str(x) for x in names].index(str(metric_name))
    except ValueError:
        return None


def write_separability_csv(metrics: Mapping[str, Any], out_path: str | Path) -> list[dict[str, Any]]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_chain = response_error_labels(metrics)
    chain_scores = np.asarray(metrics["chain_scores"], dtype=np.float64)
    chain_names = [str(x) for x in metrics["chain_score_names"].tolist()]
    rows: list[dict[str, Any]] = []
    for k, name in enumerate(chain_names):
        s = chain_scores[:, k]
        valid = y_chain >= 0
        rows.append(
            {
                "level": "response",
                "metric": name,
                "auroc_error_high": auroc(y_chain[valid], s[valid]),
                "mean_correct": _nanmean(s[y_chain == 0]),
                "mean_error": _nanmean(s[y_chain == 1]),
                "diff_error_minus_correct": _nanmean(s[y_chain == 1]) - _nanmean(s[y_chain == 0]),
                "n_correct": int(np.sum(y_chain == 0)),
                "n_error": int(np.sum(y_chain == 1)),
            }
        )

    if "gold_error_step" in metrics:
        step_scores = np.asarray(metrics["step_scores"], dtype=np.float64)
        step_names = [str(x) for x in metrics["step_score_names"].tolist()]
        gold = np.asarray(metrics["gold_error_step"], dtype=np.int64)
        n_steps = np.asarray(metrics["n_steps"], dtype=np.int64)
        step_y = []
        by_name = {name: [] for name in step_names}
        for i in range(step_scores.shape[0]):
            if gold[i] < 0:
                continue
            for j in range(int(n_steps[i])):
                step_y.append(1 if j == int(gold[i]) else 0)
                for k, name in enumerate(step_names):
                    by_name[name].append(float(step_scores[i, j, k]))
        step_y_arr = np.asarray(step_y, dtype=np.int32)
        for name, vals in by_name.items():
            s = np.asarray(vals, dtype=np.float64)
            rows.append(
                {
                    "level": "first_error_step",
                    "metric": name,
                    "auroc_error_high": auroc(step_y_arr, s),
                    "mean_correct": _nanmean(s[step_y_arr == 0]),
                    "mean_error": _nanmean(s[step_y_arr == 1]),
                    "diff_error_minus_correct": _nanmean(s[step_y_arr == 1]) - _nanmean(s[step_y_arr == 0]),
                    "n_correct": int(np.sum(step_y_arr == 0)),
                    "n_error": int(np.sum(step_y_arr == 1)),
                }
            )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["level", "metric", "auroc_error_high", "mean_correct", "mean_error", "diff_error_minus_correct", "n_correct", "n_error"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return rows


def write_trajectory_csv(
    metrics: Mapping[str, Any],
    out_path: str | Path,
    *,
    metric_names: Sequence[str] = DEFAULT_STEP_METRICS,
    grid_size: int = 21,
) -> list[dict[str, Any]]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    step_scores = np.asarray(metrics["step_scores"], dtype=np.float64)
    step_names = [str(x) for x in metrics["step_score_names"].tolist()]
    n_steps = np.asarray(metrics["n_steps"], dtype=np.int64)
    y = response_error_labels(metrics)
    grid = np.linspace(0.0, 1.0, int(grid_size))
    rows = []
    for metric in metric_names:
        k = metric_index(step_names, metric)
        if k is None:
            continue
        for label_value, label_name in [(0, "correct_response"), (1, "error_response")]:
            curves = []
            for i in np.where(y == label_value)[0]:
                T = int(n_steps[i])
                if T <= 0:
                    continue
                vals = step_scores[i, :T, k]
                xs = np.linspace(0.0, 1.0, T)
                ok = np.isfinite(vals)
                if np.sum(ok) == 0:
                    continue
                if np.sum(ok) == 1:
                    curves.append(np.full_like(grid, vals[ok][0], dtype=np.float64))
                else:
                    curves.append(np.interp(grid, xs[ok], vals[ok]))
            arr = np.vstack(curves) if curves else np.empty((0, grid.size))
            for g_idx, g in enumerate(grid):
                col = arr[:, g_idx] if arr.size else np.asarray([], dtype=np.float64)
                rows.append(
                    {
                        "metric": metric,
                        "group": label_name,
                        "rel_step": float(g),
                        "mean": _nanmean(col),
                        "stderr": _nanstderr(col),
                        "n": int(np.sum(np.isfinite(col))),
                    }
                )
    _write_rows(out_path, rows, ["metric", "group", "rel_step", "mean", "stderr", "n"])
    return rows


def write_first_error_aligned_csv(
    metrics: Mapping[str, Any],
    out_path: str | Path,
    *,
    metric_names: Sequence[str] = DEFAULT_STEP_METRICS,
    radius: int = 4,
) -> list[dict[str, Any]]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    step_scores = np.asarray(metrics["step_scores"], dtype=np.float64)
    step_names = [str(x) for x in metrics["step_score_names"].tolist()]
    gold = np.asarray(metrics["gold_error_step"], dtype=np.int64)
    n_steps = np.asarray(metrics["n_steps"], dtype=np.int64)
    rows = []
    for metric in metric_names:
        k = metric_index(step_names, metric)
        if k is None:
            continue
        for off in range(-int(radius), int(radius) + 1):
            vals = []
            for i in range(step_scores.shape[0]):
                g = int(gold[i])
                if g < 0:
                    continue
                j = g + off
                if 0 <= j < int(n_steps[i]):
                    vals.append(float(step_scores[i, j, k]))
            arr = np.asarray(vals, dtype=np.float64)
            rows.append(
                {
                    "metric": metric,
                    "offset_from_first_error": int(off),
                    "mean": _nanmean(arr),
                    "stderr": _nanstderr(arr),
                    "n": int(np.sum(np.isfinite(arr))),
                }
            )
    _write_rows(out_path, rows, ["metric", "offset_from_first_error", "mean", "stderr", "n"])
    return rows


def make_plots(
    metrics: Mapping[str, Any],
    output_dir: str | Path,
    *,
    metric_names: Sequence[str] = DEFAULT_STEP_METRICS,
) -> None:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_rows = write_trajectory_csv(metrics, output_dir / "trajectory_curves.csv", metric_names=metric_names)
    aligned_rows = write_first_error_aligned_csv(metrics, output_dir / "first_error_aligned_curves.csv", metric_names=metric_names)
    write_separability_csv(metrics, output_dir / "separability_summary.csv")
    save_json(evaluate_all(metrics), output_dir / "metric_auroc_summary.json")

    for metric in metric_names:
        sub = [r for r in trajectory_rows if r["metric"] == metric]
        if sub:
            fig, ax = plt.subplots(figsize=(7, 4))
            for group, color in [("correct_response", "#2563eb"), ("error_response", "#dc2626")]:
                rows = [r for r in sub if r["group"] == group and np.isfinite(r["mean"])]
                if not rows:
                    continue
                x = np.asarray([r["rel_step"] for r in rows], dtype=float)
                y = np.asarray([r["mean"] for r in rows], dtype=float)
                se = np.asarray([r["stderr"] for r in rows], dtype=float)
                ax.plot(x, y, label=group, color=color)
                ax.fill_between(x, y - se, y + se, color=color, alpha=0.18)
            ax.set_title(f"{metric}: normalized trajectory")
            ax.set_xlabel("relative step position")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"trajectory_{metric}.png", dpi=180)
            plt.close(fig)

        sub_aligned = [r for r in aligned_rows if r["metric"] == metric and np.isfinite(r["mean"])]
        if sub_aligned:
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.asarray([r["offset_from_first_error"] for r in sub_aligned], dtype=float)
            y = np.asarray([r["mean"] for r in sub_aligned], dtype=float)
            se = np.asarray([r["stderr"] for r in sub_aligned], dtype=float)
            ax.plot(x, y, marker="o", color="#7c3aed")
            ax.fill_between(x, y - se, y + se, color="#7c3aed", alpha=0.18)
            ax.axvline(0, color="#dc2626", linestyle="--", linewidth=1.2)
            ax.set_title(f"{metric}: aligned to first error")
            ax.set_xlabel("step offset from first error")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_dir / f"first_error_aligned_{metric}.png", dpi=180)
            plt.close(fig)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow(finite_json(dict(row)))


def _nanmean(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=np.float64)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def _nanstderr(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))
