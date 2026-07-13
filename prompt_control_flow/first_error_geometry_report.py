from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .first_error_geometry import GEOMETRY_NAMES, GeometryAuditResult


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_or_none(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def _top_offset_zero_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_coverage: float,
    limit: int = 20,
) -> list[Mapping[str, Any]]:
    eligible = [
        row
        for row in rows
        if row.get("variant") == "nuisance_residual"
        and int(row.get("offset", 1)) == 0
        and float(row.get("pair_coverage", 0.0)) >= float(min_coverage)
        and np.isfinite(float(row.get("matched_event_auroc", np.nan)))
    ]
    return sorted(
        eligible,
        key=lambda row: abs(float(row.get("matched_event_auroc", 0.5)) - 0.5),
        reverse=True,
    )[:limit]


def render_markdown(
    result: GeometryAuditResult,
    *,
    min_coverage: float = 0.80,
) -> str:
    axis = result.axis
    error_count = int(np.sum(axis.event_indices >= 0))
    correct_count = int(np.sum(axis.event_indices < 0))
    lines = [
        f"# First-Error Geometry Event Audit: {axis.axis_kind}",
        "",
        "This report treats the geometry measures as hypotheses to be tested, not as validated error detectors.",
        "",
        f"- Trajectories: `{axis.n_samples}` (`{error_count}` error, `{correct_count}` correct)",
        f"- Matched error/control events: `{len(result.matches)}`",
        f"- Layers: `{axis.layer_ids.tolist()}`",
        f"- Event definition: {axis.metadata.get('event_definition', 'NA')}",
        f"- Effective compute device: `{result.metadata.get('effective_device', 'NA')}`",
        f"- Primary-table minimum pair coverage: `{min_coverage:.0%}`",
        "",
        "## Geometry Definitions",
        "",
        r"For state $z_t^{(\ell)}$, the incoming update is $\Delta z_t^{(\ell)}=z_t^{(\ell)}-z_{t-1}^{(\ell)}$.",
        r"The turning angle is $\theta_t^{(\ell)}=\arccos\!\left(\langle \Delta z_t,\Delta z_{t+1}\rangle/(\|\Delta z_t\|\|\Delta z_{t+1}\|)\right)$.",
        r"Menger curvature is $\kappa_t^{(\ell)}=2\sin\theta_t^{(\ell)}/\|z_{t+1}^{(\ell)}-z_{t-1}^{(\ell)}\|$.",
        "",
        "Offset `0` is the first-error step/token boundary. Turning angle and curvature use one future state and are therefore diagnostic rather than strictly pre-error signals.",
        "",
        "## Matched Event Effects at Offset 0",
        "",
        "Rows below use cross-fitted nuisance residuals learned from correct chains only. Matching uses chain length, relative event position, and event-step length, never geometry.",
        "",
        "| metric | layer | pairs | coverage | error-control | 95% CI | paired dz | AUROC | q |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    top = _top_offset_zero_rows(result.event_rows, min_coverage=min_coverage)
    if top:
        for row in top:
            interval = f"[{_fmt(row['difference_ci_low'])}, {_fmt(row['difference_ci_high'])}]"
            lines.append(
                f"| `{row['metric']}` | {row['layer']} | {row['n_pairs']} | "
                f"{_fmt(row['pair_coverage'])} | {_fmt(row['paired_difference'])} | {interval} | "
                f"{_fmt(row['paired_effect_dz'])} | {_fmt(row['matched_event_auroc'])} | "
                f"{_fmt(row['bh_q'])} |"
            )
    else:
        lines.append("| NA | NA | 0 | NA | NA | NA | NA | NA | NA |")

    lines.extend(
        [
            "",
            "## First-Error Localization",
            "",
            "This is a secondary diagnostic. It compares the gold first-error point against earlier points in the same erroneous trajectory and remains vulnerable to causal-position structure.",
            "",
            "| variant | metric | layer | rows | positives | eligible chains | AUROC | expected top1 | mean rank |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ranked = sorted(
        result.discrimination_rows,
        key=lambda row: (
            row.get("variant") != "nuisance_residual",
            -float(row.get("auroc_high_is_error", -np.inf))
            if np.isfinite(float(row.get("auroc_high_is_error", np.nan)))
            else np.inf,
        ),
    )
    for row in ranked[:20]:
        lines.append(
            f"| `{row['variant']}` | `{row['metric']}` | {row['layer']} | {row['n_rows']} | "
            f"{row['n_positive']} | {row['eligible_chains']} | {_fmt(row['auroc_high_is_error'])} | "
            f"{_fmt(row['expected_top1'])} | {_fmt(row['mean_rank'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- A high raw score is insufficient; the nuisance-residualized matched effect is the primary result.",
            "- Metrics with pair coverage below the stated threshold are excluded from the headline table.",
            "- Menger curvature here is an inverse-distance geometric curvature, not directional concentration with the same symbol.",
            "- A positive event association does not establish causal awareness or online detectability.",
        ]
    )
    return "\n".join(lines) + "\n"


def _event_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    variant: str,
    layers: Sequence[int],
    offsets: Sequence[int],
    field: str,
) -> np.ndarray:
    lookup = {
        (int(row["layer"]), int(row["offset"])): float(row.get(field, np.nan))
        for row in rows
        if row.get("metric") == metric and row.get("variant") == variant
    }
    return np.asarray(
        [[lookup.get((int(layer), int(offset)), np.nan) for offset in offsets] for layer in layers],
        dtype=np.float64,
    )


def _plot_heatmaps(result: GeometryAuditResult, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [int(value) for value in result.axis.layer_ids]
    offsets = [int(value) for value in result.metadata["offsets"]]
    fig, axes = plt.subplots(len(GEOMETRY_NAMES), 1, figsize=(max(9.0, 0.42 * len(offsets)), 3.0 * len(GEOMETRY_NAMES)), squeeze=False)
    for row_index, metric in enumerate(GEOMETRY_NAMES):
        ax = axes[row_index, 0]
        matrix = _event_matrix(
            result.event_rows,
            metric=metric,
            variant="nuisance_residual",
            layers=layers,
            offsets=offsets,
            field="paired_effect_dz",
        )
        finite = np.abs(matrix[np.isfinite(matrix)])
        bound = max(float(np.quantile(finite, 0.95)) if finite.size else 1.0, 0.25)
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
        ax.set_title(metric)
        ax.set_ylabel("layer")
        ax.set_yticks(np.arange(len(layers)), labels=layers)
        tick_stride = max(1, len(offsets) // 16)
        tick_positions = np.arange(0, len(offsets), tick_stride)
        ax.set_xticks(tick_positions, labels=[offsets[i] for i in tick_positions])
        ax.axvline(offsets.index(0), color="black", linewidth=1.0, linestyle="--")
        fig.colorbar(image, ax=ax, label="paired effect dz")
    axes[-1, 0].set_xlabel("offset from first-error event")
    fig.tight_layout()
    fig.savefig(output_dir / "event_effect_heatmaps.png", dpi=180)
    plt.close(fig)


def _plot_metric_curves(result: GeometryAuditResult, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    offsets = np.asarray(result.metadata["offsets"], dtype=np.int64)
    layers = [int(value) for value in result.axis.layer_ids]
    for metric in GEOMETRY_NAMES:
        n_cols = min(4, len(layers))
        n_rows = int(np.ceil(len(layers) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), sharex=True, squeeze=False)
        for layer_index, layer in enumerate(layers):
            ax = axes[layer_index // n_cols, layer_index % n_cols]
            selected = {
                int(row["offset"]): row
                for row in result.event_rows
                if row.get("variant") == "nuisance_residual"
                and row.get("metric") == metric
                and int(row.get("layer", -1)) == layer
            }
            error_mean = np.asarray([selected.get(int(offset), {}).get("error_mean", np.nan) for offset in offsets], dtype=np.float64)
            error_low = np.asarray([selected.get(int(offset), {}).get("error_ci_low", np.nan) for offset in offsets], dtype=np.float64)
            error_high = np.asarray([selected.get(int(offset), {}).get("error_ci_high", np.nan) for offset in offsets], dtype=np.float64)
            control_mean = np.asarray([selected.get(int(offset), {}).get("control_mean", np.nan) for offset in offsets], dtype=np.float64)
            control_low = np.asarray([selected.get(int(offset), {}).get("control_ci_low", np.nan) for offset in offsets], dtype=np.float64)
            control_high = np.asarray([selected.get(int(offset), {}).get("control_ci_high", np.nan) for offset in offsets], dtype=np.float64)
            ax.plot(offsets, error_mean, color="#c23b33", label="error event")
            ax.fill_between(offsets, error_low, error_high, color="#c23b33", alpha=0.18)
            ax.plot(offsets, control_mean, color="#2369a1", label="matched correct")
            ax.fill_between(offsets, control_low, control_high, color="#2369a1", alpha=0.18)
            ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
            ax.axhline(0, color="0.65", linewidth=0.8)
            ax.set_title(f"layer {layer}")
            ax.grid(alpha=0.2)
        for empty in range(len(layers), n_rows * n_cols):
            axes[empty // n_cols, empty % n_cols].axis("off")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.supxlabel("offset from first-error event")
        fig.supylabel("cross-fitted nuisance residual")
        fig.suptitle(f"{result.axis.axis_kind}: {metric}")
        fig.tight_layout()
        fig.savefig(output_dir / f"event_curve_{metric}.png", dpi=180)
        plt.close(fig)


def save_result_npz(result: GeometryAuditResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        axis_kind=np.asarray(result.axis.axis_kind),
        original_indices=result.axis.original_indices,
        problem_ids=result.axis.problem_ids,
        is_correct=result.axis.is_correct,
        event_indices=result.axis.event_indices,
        layer_ids=result.axis.layer_ids,
        geometry_names=np.asarray(GEOMETRY_NAMES, dtype=object),
        geometry=np.asarray(result.geometry, dtype=object),
        residual_geometry=np.asarray(result.residual_geometry, dtype=object),
        match_error_rows=np.asarray([item.error_row for item in result.matches], dtype=np.int64),
        match_control_rows=np.asarray([item.control_row for item in result.matches], dtype=np.int64),
        match_error_events=np.asarray([item.error_step for item in result.matches], dtype=np.int64),
        match_control_events=np.asarray([item.control_step for item in result.matches], dtype=np.int64),
        match_cost=np.asarray([item.cost for item in result.matches], dtype=np.float64),
        match_same_problem=np.asarray([item.same_problem for item in result.matches], dtype=np.int8),
        match_reused_control=np.asarray([item.reused_control for item in result.matches], dtype=np.int8),
        metadata_json=np.asarray(json.dumps(_finite_or_none(result.metadata), ensure_ascii=True)),
    )


def write_geometry_audit_report(
    result: GeometryAuditResult,
    output_dir: str | Path,
    *,
    min_coverage: float = 0.80,
    render_plots: bool = True,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    event_csv = output_dir / "event_curves.csv"
    discrimination_csv = output_dir / "first_error_discrimination.csv"
    match_csv = output_dir / "matched_pairs.csv"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    result_npz = output_dir / "geometry_audit.npz"

    _write_rows(event_csv, result.event_rows)
    _write_rows(discrimination_csv, result.discrimination_rows)
    _write_rows(
        match_csv,
        [
            {
                "error_row": item.error_row,
                "control_row": item.control_row,
                "error_event": item.error_step,
                "control_event": item.control_step,
                "cost": item.cost,
                "same_problem": int(item.same_problem),
                "reused_control": int(item.reused_control),
            }
            for item in result.matches
        ],
    )
    summary = {
        "axis_kind": result.axis.axis_kind,
        "n_trajectories": result.axis.n_samples,
        "n_error": int(np.sum(result.axis.event_indices >= 0)),
        "n_correct": int(np.sum(result.axis.event_indices < 0)),
        "n_matches": len(result.matches),
        "layers": result.axis.layer_ids.tolist(),
        "axis_metadata": result.axis.metadata,
        "audit_metadata": result.metadata,
        "headline_offset_zero": _top_offset_zero_rows(
            result.event_rows, min_coverage=min_coverage
        ),
    }
    summary_json.write_text(
        json.dumps(_finite_or_none(summary), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    summary_md.write_text(
        render_markdown(result, min_coverage=min_coverage), encoding="utf-8"
    )
    save_result_npz(result, result_npz)
    if render_plots:
        _plot_heatmaps(result, output_dir)
        _plot_metric_curves(result, output_dir)
    return {
        "event_csv": str(event_csv),
        "discrimination_csv": str(discrimination_csv),
        "match_csv": str(match_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "result_npz": str(result_npz),
    }
