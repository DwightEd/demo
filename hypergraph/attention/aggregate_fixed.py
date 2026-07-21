"""Aggregate exactly one fixed held-out response test run."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


METRIC_NAMES = ("auroc", "aupr", "accuracy_0.5")


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed-holdout result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"fixed-holdout result must be a JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def aggregate_fixed_run(
    root: str | Path,
    run: str | Path,
    *,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Validate and summarize one validation-selected, fixed test run."""

    root_path = Path(root).expanduser().resolve()
    run_path = Path(run).expanduser().resolve()
    if run_path.parent != root_path:
        raise ValueError(f"fixed run must be a direct child of {root_path}: {run_path}")
    result_path = run_path / "results.json"
    prediction_path = run_path / "predictions_test.csv"
    payload = _read_json(result_path)
    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"missing fixed held-out prediction file: {prediction_path}"
        )

    split = payload.get("resolved", {}).get("split", {})
    if not isinstance(split, Mapping) or split.get("mode") != "fixed_holdout":
        raise ValueError(f"expected fixed_holdout split metadata, got {split!r}")
    test = payload.get("metrics", {}).get("test", {})
    if not isinstance(test, Mapping):
        raise ValueError("results.json lacks metrics.test")
    for name in METRIC_NAMES:
        value = test.get(name)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"fixed test metric {name} is undefined or non-finite")
    for name in ("n", "positives"):
        value = test.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"fixed test metric {name} must be numeric")

    summary = {
        "schema": "fixed_holdout_response_test_v1",
        "protocol": "single_problem_disjoint_fixed_holdout",
        "run": run_path.name,
        "best_epoch": payload.get("best_epoch"),
        "validation_monitor": payload.get("validation_monitor"),
        "partition_sizes": payload.get("partition_sizes"),
        "split": dict(split),
        "final_test": dict(test),
        "generator_final_test": payload.get("trace_detection_by_generator", {}).get(
            "test", {}
        ),
        "predictions_test": str(root_path / "predictions_test.csv"),
    }
    if write_outputs:
        root_path.mkdir(parents=True, exist_ok=True)
        _atomic_json(root_path / "aggregate_results.json", summary)
        shutil.copyfile(prediction_path, root_path / "predictions_test.csv")
        _atomic_json(root_path / "split_manifest.json", dict(split))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="dataset experiment root")
    parser.add_argument("--run", required=True, help="single fixed_seedN run directory")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = aggregate_fixed_run(args.root, args.run, write_outputs=True)
    test = summary["final_test"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("===== Final fixed held-out test =====")
    print(
        f"traces={int(test['n'])} positives={int(test['positives'])} "
        f"AUROC={float(test['auroc']):.6f} AUPRC={float(test['aupr']):.6f} "
        f"accuracy@0.5={float(test['accuracy_0.5']):.6f}"
    )
    print("aggregate results:", Path(args.root) / "aggregate_results.json")
    print("held-out predictions:", Path(args.root) / "predictions_test.csv")
    print("split manifest:", Path(args.root) / "split_manifest.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["aggregate_fixed_run", "build_parser", "main"]
