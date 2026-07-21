from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import paired_auc_difference, paired_summary
from .layer_time import crossfit_layer_time_scores, load_matched_layer_time_geometry
from .output import versioned_paths


METRIC_ORDER = (
    "radial_edge_change",
    "depth_operator_residual",
    "time_operator_residual",
    "plaquette_observed_disagreement",
)


def _pair_rows(source: Path, data, scores: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in np.unique(data.pair_ids):
        indices = np.where(data.pair_ids == pair)[0]
        error = int(indices[data.labels[indices] == 1][0])
        control = int(indices[data.labels[indices] == 0][0])
        row: dict[str, Any] = {
            "source": str(source.resolve()),
            "axis_kind": data.metadata["axis_kind"],
            "pair_id": int(pair),
            "component_id": int(data.component_ids[error]),
            "error_row_id": int(data.row_ids[error]),
            "control_row_id": int(data.row_ids[control]),
        }
        for name, values in scores.items():
            row[f"{name}.error"] = float(values[error])
            row[f"{name}.control"] = float(values[control])
            row[f"{name}.difference"] = float(values[error] - values[control])
        rows.append(row)
    return rows


def _write_plot(datasets: list[dict[str, Any]], paths: tuple[Path, ...]) -> None:
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        values = [dataset["metrics"][name]["paired_auroc"] for name in METRIC_ORDER]
        low = [dataset["metrics"][name]["paired_auroc_ci_low"] for name in METRIC_ORDER]
        high = [dataset["metrics"][name]["paired_auroc_ci_high"] for name in METRIC_ORDER]
        positions = np.arange(len(METRIC_ORDER))
        axis.barh(positions, values, color=["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3"])
        axis.errorbar(
            values,
            positions,
            xerr=[np.asarray(values) - low, np.asarray(high) - values],
            fmt="none",
            color="black",
            capsize=3,
        )
        axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_yticks(positions, labels=METRIC_ORDER)
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Matched-pair AUROC")
        axis.set_title(f"{dataset['axis_kind']} (n={dataset['n_pairs']})")
    fig.tight_layout()
    for path in paths:
        fig.savefig(path, dpi=180)
    plt.close(fig)


def run_layer_time_experiment(
    *,
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    offsets: Iterable[int] = (-1, 0),
    metrics: Iterable[str] = ("delta_norm", "relative_delta_norm"),
    rank: int = 2,
    n_splits: int = 5,
    n_boot: int = 2000,
    seed: int = 17,
    ridge_alpha: float = 1.0,
    variant: str = "raw",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for source_value in inputs:
        source = Path(source_value)
        data = load_matched_layer_time_geometry(
            source, offsets=offsets, metrics=metrics, variant=variant
        )
        scores, diagnostics = crossfit_layer_time_scores(
            data,
            rank=rank,
            n_splits=n_splits,
            seed=seed,
            ridge_alpha=ridge_alpha,
        )
        summaries = {
            name: paired_summary(values, data.labels, data.pair_ids, n_boot=n_boot, seed=seed + i)
            for i, (name, values) in enumerate(scores.items())
        }
        comparisons = {
            "plaquette_minus_radial": paired_auc_difference(
                scores["plaquette_observed_disagreement"],
                scores["radial_edge_change"],
                data.labels,
                data.pair_ids,
                n_boot=n_boot,
                seed=seed + 101,
            ),
            "plaquette_minus_depth_residual": paired_auc_difference(
                scores["plaquette_observed_disagreement"],
                scores["depth_operator_residual"],
                data.labels,
                data.pair_ids,
                n_boot=n_boot,
                seed=seed + 102,
            ),
        }
        datasets.append(
            {
                **data.metadata,
                "source_path": str(source.resolve()),
                "n_pairs": int(np.unique(data.pair_ids).size),
                "time_offsets": data.time_offsets.tolist(),
                "layer_ids": data.layer_ids.tolist(),
                "feature_names": list(data.feature_names),
                "diagnostics": diagnostics,
                "metrics": summaries,
                "comparisons": comparisons,
            }
        )
        rows.extend(_pair_rows(source, data, scores))
    result = {
        "schema_version": "layer_time_operator_v1",
        "method": "component-grouped cross-fitted affine operator field in one shared fold gauge",
        "evidence_boundary": "derived geometry proxy; not hidden-state Jacobian evidence",
        "seed": int(seed),
        "n_boot": int(n_boot),
        "requested_rank": int(rank),
        "ridge_alpha": float(ridge_alpha),
        "datasets": datasets,
    }
    text = json.dumps(result, indent=2)
    for path in versioned_paths(output_dir / "results.json"):
        path.write_text(text, encoding="utf-8")
    if rows:
        for path in versioned_paths(output_dir / "pair_scores.csv"):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    _write_plot(datasets, versioned_paths(output_dir / "metric_comparison.png"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offsets", default="-1,0")
    parser.add_argument("--metrics", default="delta_norm,relative_delta_norm")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--variant", choices=("raw", "nuisance_residual"), default="raw")
    args = parser.parse_args()
    result = run_layer_time_experiment(
        inputs=args.inputs,
        output_dir=args.output_dir,
        offsets=tuple(int(value) for value in args.offsets.split(",")),
        metrics=tuple(value.strip() for value in args.metrics.split(",") if value.strip()),
        rank=args.rank,
        n_splits=args.folds,
        n_boot=args.bootstrap,
        seed=args.seed,
        ridge_alpha=args.ridge_alpha,
        variant=args.variant,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
