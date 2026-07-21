from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .core import paired_auc_difference, paired_summary
from .layer_time import crossfit_layer_time_scores
from .layer_time_experiment import _pair_rows, _write_plot
from .output import versioned_paths
from .raw_residual import inspect_raw_residual_source, load_matched_raw_residual


def run_raw_residual_experiment(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    hidden_dir: str | Path | None = None,
    offsets: Iterable[int] = (-2, -1, 0, 1),
    layers: str | Iterable[int] = "all",
    max_pairs: int = 0,
    rank: int = 16,
    n_splits: int = 5,
    n_boot: int = 2000,
    seed: int = 17,
    ridge_alpha: float = 1.0,
    response_generator: str | None = None,
) -> dict[str, Any]:
    """Run the joint operator-field analysis on raw response-token residual shards."""
    source = Path(input_path)
    data = load_matched_raw_residual(
        source,
        hidden_dir=hidden_dir,
        offsets=offsets,
        layers=layers,
        max_pairs=max_pairs,
        response_generator=response_generator,
    )
    scores, diagnostics = crossfit_layer_time_scores(
        data,
        rank=rank,
        n_splits=n_splits,
        seed=seed,
        ridge_alpha=ridge_alpha,
    )
    metrics = {
        name: paired_summary(values, data.labels, data.pair_ids, n_boot=n_boot, seed=seed + index)
        for index, (name, values) in enumerate(scores.items())
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
        "plaquette_minus_time_residual": paired_auc_difference(
            scores["plaquette_observed_disagreement"],
            scores["time_operator_residual"],
            data.labels,
            data.pair_ids,
            n_boot=n_boot,
            seed=seed + 102,
        ),
    }
    dataset = {
        **data.metadata,
        "n_pairs": int(np.unique(data.pair_ids).size),
        "time_offsets": data.time_offsets.tolist(),
        "layer_ids": data.layer_ids.tolist(),
        "hidden_dim": int(data.states.shape[-1]),
        "diagnostics": diagnostics,
        "metrics": metrics,
        "comparisons": comparisons,
    }
    result = {
        "schema_version": "raw_residual_layer_time_operator_v1",
        "method": "component-grouped cross-fitted projected operator field on raw residual-stream shards",
        "evidence_boundary": (
            "empirical local operators on stored residual states; not autograd model Jacobians"
        ),
        "seed": int(seed),
        "n_boot": int(n_boot),
        "requested_rank": int(rank),
        "ridge_alpha": float(ridge_alpha),
        "dataset": dataset,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2)
    for path in versioned_paths(output / "results.json"):
        path.write_text(encoded, encoding="utf-8")
    rows = _pair_rows(source, data, scores)
    for path in versioned_paths(output / "pair_scores.csv"):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _write_plot([dataset], versioned_paths(output / "metric_comparison.png"))
    return result


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint time × layer operator-field analysis of raw response residual-stream shards."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--hidden-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--offsets", type=_parse_ints, default=(-2, -1, 0, 1))
    parser.add_argument("--layers", default="all")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--response-generator",
        default=None,
        help="keep only manifest rows whose response generator matches this normalized substring",
    )
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    layer_selection: str | tuple[int, ...] = (
        "all" if args.layers.strip().lower() == "all" else _parse_ints(args.layers)
    )
    if args.preflight:
        print(json.dumps(
            inspect_raw_residual_source(
                args.input,
                hidden_dir=args.hidden_dir,
                response_generator=args.response_generator,
            ),
            indent=2,
        ))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --preflight is used")
    result = run_raw_residual_experiment(
        input_path=args.input,
        hidden_dir=args.hidden_dir,
        output_dir=args.output_dir,
        offsets=args.offsets,
        layers=layer_selection,
        max_pairs=args.max_pairs,
        rank=args.rank,
        n_splits=args.folds,
        n_boot=args.bootstrap,
        seed=args.seed,
        ridge_alpha=args.ridge_alpha,
        response_generator=args.response_generator,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
