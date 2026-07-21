from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .domain import ExperimentResult, LayerTimeDataset
from .output import versioned_paths


METRIC_ORDER = (
    "radial_edge_change",
    "depth_operator_residual",
    "time_operator_residual",
    "plaquette_observed_disagreement",
)


class ArtifactWriter:
    """Serialize one typed experiment result and its pair-level audit table."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def write(
        self,
        result: ExperimentResult,
        data: LayerTimeDataset,
        scores: dict[str, np.ndarray],
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(result.to_dict(), indent=2)
        for path in versioned_paths(self.output_dir / "results.json"):
            path.write_text(encoded, encoding="utf-8")
        rows = self._pair_rows(data, scores)
        for path in versioned_paths(self.output_dir / "pair_scores.csv"):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        self._plot(result, versioned_paths(self.output_dir / "metric_comparison.png"))

    @staticmethod
    def _pair_rows(
        data: LayerTimeDataset, scores: dict[str, np.ndarray]
    ) -> list[dict[str, Any]]:
        source = data.metadata.provenance
        rows: list[dict[str, Any]] = []
        for pair in np.unique(data.pair_ids):
            indices = np.where(data.pair_ids == pair)[0]
            error = int(indices[data.labels[indices] == 1][0])
            control = int(indices[data.labels[indices] == 0][0])
            row: dict[str, Any] = {
                "source": source.manifest_path,
                "axis_kind": source.axis_kind,
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

    @staticmethod
    def _plot(result: ExperimentResult, paths: tuple[Path, ...]) -> None:
        dataset = result.dataset
        values = [dataset.metrics[name]["paired_auroc"] for name in METRIC_ORDER]
        low = [dataset.metrics[name]["paired_auroc_ci_low"] for name in METRIC_ORDER]
        high = [dataset.metrics[name]["paired_auroc_ci_high"] for name in METRIC_ORDER]
        positions = np.arange(len(METRIC_ORDER))
        figure, axis = plt.subplots(figsize=(6, 4))
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
        axis.set_title(f"token (n={dataset.pairs})")
        figure.tight_layout()
        for path in paths:
            figure.savefig(path, dpi=180)
        plt.close(figure)
