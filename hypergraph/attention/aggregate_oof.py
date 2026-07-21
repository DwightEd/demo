"""Aggregate mutually held-out fold predictions into final OOF test metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METRIC_NAMES = ("auroc", "aupr", "accuracy_0.5")
REQUIRED_COLUMNS = {
    "trace_id",
    "group_id",
    "generator_model",
    "label",
    "score",
}


def _binary_metrics(labels: Sequence[float], scores: Sequence[float]) -> dict[str, Any]:
    """Compute the same tie-aware binary metrics used by ``train.py``."""

    if len(labels) != len(scores):
        raise ValueError("binary labels and scores must be aligned")
    pairs = []
    for label, score in zip(labels, scores):
        label_value = float(label)
        score_value = float(score)
        if label_value not in (0.0, 1.0) or not math.isfinite(label_value):
            raise ValueError("binary metric labels must all be finite 0/1 values")
        if not math.isfinite(score_value):
            raise ValueError("binary metric scores must all be finite")
        pairs.append((label_value, score_value))

    positives = sum(int(label == 1.0) for label, _ in pairs)
    result: dict[str, Any] = {
        "n": len(pairs),
        "positives": positives,
        "prevalence": None if not pairs else positives / len(pairs),
    }
    negatives = len(pairs) - positives
    if not pairs or positives == 0 or negatives == 0:
        result.update({name: None for name in METRIC_NAMES})
        return result

    # Mann-Whitney interpretation of AUROC in O(n log n), with average ranks
    # for ties. This matches train.py without quadratic positive/negative loops.
    ascending = sorted(pairs, key=lambda item: item[1])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ascending):
        end = start + 1
        while end < len(ascending) and ascending[end][1] == ascending[start][1]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        positive_rank_sum += average_rank * sum(
            int(label == 1.0) for label, _ in ascending[start:end]
        )
        start = end
    auroc = (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)

    # Average precision at score-group boundaries, matching train.py.
    ordered = sorted(pairs, key=lambda item: item[1], reverse=True)
    true_positive = false_positive = 0
    previous_recall = average_precision = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        true_positive += sum(int(label == 1.0) for label, _ in ordered[start:end])
        false_positive += sum(int(label == 0.0) for label, _ in ordered[start:end])
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end

    result.update(
        {
            "auroc": auroc,
            "aupr": average_precision,
            "accuracy_0.5": sum(
                int((score >= 0.5) == bool(label)) for label, score in pairs
            )
            / len(pairs),
        }
    )
    return result


def _read_prediction_file(path: Path, *, fold: int, seed: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing held-out prediction file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"{path} lacks required prediction columns: {missing}")
        rows = []
        for row_index, raw in enumerate(reader, start=2):
            trace_id = str(raw.get("trace_id", "")).strip()
            if not trace_id:
                raise ValueError(f"{path}:{row_index} has an empty trace_id")
            try:
                label = float(str(raw["label"]).strip())
                score = float(str(raw["score"]).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{row_index} has a non-numeric label or score"
                ) from exc
            if label not in (0.0, 1.0) or not math.isfinite(label):
                raise ValueError(f"{path}:{row_index} label is not finite binary data")
            if not math.isfinite(score):
                raise ValueError(f"{path}:{row_index} score is not finite")
            rows.append(
                {
                    **raw,
                    "run": path.parent.name,
                    "fold": int(fold),
                    "seed": int(seed),
                    "trace_id": trace_id,
                    "group_id": str(raw.get("group_id", "")),
                    "generator_model": str(raw.get("generator_model", "")),
                    "label": label,
                    "score": score,
                }
            )
    if not rows:
        raise ValueError(f"held-out prediction file is empty: {path}")
    return rows


def _metric_summary(metrics_by_seed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    summary = {}
    for metric in METRIC_NAMES:
        values = [
            float(values[metric])
            for values in metrics_by_seed.values()
            if values.get(metric) is not None and math.isfinite(float(values[metric]))
        ]
        summary[metric] = {
            "n_defined_seeds": len(values),
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summary


def _metrics_by_generator(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        generator = str(row.get("generator_model") or "<missing>")
        grouped[generator].append(row)
    return {
        generator: _binary_metrics(
            [float(row["label"]) for row in generator_rows],
            [float(row["score"]) for row in generator_rows],
        )
        for generator, generator_rows in sorted(grouped.items())
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def aggregate_oof_run(
    run_root: str | Path,
    *,
    folds: int,
    seeds: Sequence[int],
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Pool held-out folds, with exactly one test prediction per trace and seed."""

    root = Path(run_root)
    seed_values = [int(seed) for seed in seeds]
    if folds < 1:
        raise ValueError("folds must be positive")
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be a non-empty sequence without duplicates")

    all_rows: list[dict[str, Any]] = []
    rows_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    observed_runs = []
    for seed in seed_values:
        for fold in range(int(folds)):
            run_name = f"fold{fold}_seed{seed}"
            rows = _read_prediction_file(
                root / run_name / "predictions_test.csv", fold=fold, seed=seed
            )
            observed_runs.append(run_name)
            all_rows.extend(rows)
            rows_by_seed[seed].extend(rows)

    metadata: dict[str, tuple[float, str, str]] = {}
    identity_sets: dict[int, set[str]] = {}
    for seed, rows in rows_by_seed.items():
        seen: dict[str, int] = {}
        for row in rows:
            trace_id = str(row["trace_id"])
            if trace_id in seen:
                raise ValueError(
                    f"trace {trace_id!r} appears in more than one held-out fold for "
                    f"seed {seed}: folds {seen[trace_id]} and {row['fold']}"
                )
            seen[trace_id] = int(row["fold"])
            stable = (
                float(row["label"]),
                str(row.get("group_id", "")),
                str(row.get("generator_model", "")),
            )
            previous = metadata.setdefault(trace_id, stable)
            if previous != stable:
                raise ValueError(
                    f"trace {trace_id!r} has inconsistent label/group/generator metadata"
                )
        identity_sets[seed] = set(seen)

    reference_seed = seed_values[0]
    reference_ids = identity_sets[reference_seed]
    for seed in seed_values[1:]:
        if identity_sets[seed] != reference_ids:
            missing = sorted(reference_ids - identity_sets[seed])[:10]
            extra = sorted(identity_sets[seed] - reference_ids)[:10]
            raise ValueError(
                f"trace identity set differs for seed {seed}; missing={missing}, extra={extra}"
            )

    by_seed = {
        str(seed): _binary_metrics(
            [float(row["label"]) for row in rows_by_seed[seed]],
            [float(row["score"]) for row in rows_by_seed[seed]],
        )
        for seed in seed_values
    }
    generator_by_seed = {
        str(seed): _metrics_by_generator(rows_by_seed[seed]) for seed in seed_values
    }

    rows_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        rows_by_trace[str(row["trace_id"])].append(row)
    ensemble_rows = []
    for trace_id in sorted(rows_by_trace):
        rows = sorted(rows_by_trace[trace_id], key=lambda row: int(row["seed"]))
        if len(rows) != len(seed_values):
            raise ValueError(
                f"trace {trace_id!r} has {len(rows)} seed predictions; "
                f"expected {len(seed_values)}"
            )
        ensemble_rows.append(
            {
                "trace_id": trace_id,
                "group_id": rows[0].get("group_id", ""),
                "generator_model": rows[0].get("generator_model", ""),
                "label": float(rows[0]["label"]),
                "score": statistics.fmean(float(row["score"]) for row in rows),
                "n_seed_predictions": len(rows),
                "seed_scores_json": json.dumps(
                    {str(row["seed"]): float(row["score"]) for row in rows},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    seed_ensemble = _binary_metrics(
        [float(row["label"]) for row in ensemble_rows],
        [float(row["score"]) for row in ensemble_rows],
    )

    result = {
        "schema": "pooled_oof_response_v1",
        "protocol": "pooled_out_of_fold_test",
        "description": (
            "For each seed, every trace contributes exactly one prediction from the "
            "fold where its problem group was held out. The seed ensemble averages "
            "those independently held-out probabilities per trace before computing "
            "one final test metric."
        ),
        "folds": int(folds),
        "seeds": seed_values,
        "integrity": {
            "expected_num_runs": int(folds) * len(seed_values),
            "observed_runs": observed_runs,
            "unique_traces_per_seed": len(reference_ids),
            "total_oof_rows": len(all_rows),
            "each_trace_tested_once_per_seed": True,
            "trace_sets_identical_across_seeds": True,
        },
        "by_seed": by_seed,
        "seed_metric_summary": _metric_summary(by_seed),
        "seed_ensemble": seed_ensemble,
        "generator_by_seed": generator_by_seed,
        "generator_seed_ensemble": _metrics_by_generator(ensemble_rows),
    }

    if write_outputs:
        root.mkdir(parents=True, exist_ok=True)
        _write_csv(root / "predictions_pooled_oof.csv", all_rows)
        _write_csv(root / "predictions_pooled_oof_seed_ensemble.csv", ensemble_rows)
        destination = root / "pooled_oof_results.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        aggregate_path = root / "aggregate_results.json"
        if aggregate_path.is_file():
            existing = json.loads(aggregate_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                raise ValueError(f"existing aggregate is not a JSON object: {aggregate_path}")
            existing["pooled_oof_test"] = result
            aggregate_temporary = aggregate_path.with_suffix(".json.tmp")
            aggregate_temporary.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(aggregate_temporary, aggregate_path)
    return result


def _parse_seeds(raw: str) -> list[int]:
    values = [int(value) for value in raw.replace(",", " ").split()]
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a final pooled out-of-fold test AUROC from completed folds."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=int)
    parser.add_argument("--seeds", required=True, type=_parse_seeds)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = aggregate_oof_run(
        args.run_root, folds=args.folds, seeds=args.seeds, write_outputs=True
    )
    final = result["seed_ensemble"]
    print("===== Final pooled OOF test =====")
    print(
        f"traces={final['n']} positives={final['positives']} "
        f"AUROC={final['auroc']:.6f} AUPRC={final['aupr']:.6f} "
        f"accuracy@0.5={final['accuracy_0.5']:.6f}"
    )
    for seed, metrics in result["by_seed"].items():
        print(
            f"seed {seed}: AUROC={metrics['auroc']:.6f} "
            f"AUPRC={metrics['aupr']:.6f} n={metrics['n']}"
        )
    print("report:", args.run_root / "pooled_oof_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
