from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def classify_result_kind(
    edge_kind_counts: dict[str, int],
    *,
    hypergraph_gate_passed: bool,
) -> str:
    if hypergraph_gate_passed:
        if edge_kind_counts.get("hyper", 0) <= 0:
            raise ValueError(
                "a passing hypergraph gate requires observed hyperedges"
            )
        return "real_processbench_cct_hg_baseline"
    if edge_kind_counts.get("hyper", 0) > 0:
        return "real_processbench_cct_sparse_hyperedge_witness"
    if sum(edge_kind_counts.values()) > 0:
        return "real_processbench_cct_pair_graph_witness"
    return "real_processbench_cct_no_edge_witness"


def _hypergraph_gate_failure_message(gate: dict[str, Any]) -> str:
    return (
        "hypergraph gate failed: "
        f"observed_total_hyperedges={gate['observed_total_hyperedges']}, "
        f"required_total_hyperedges={gate['min_total_hyperedges']}, "
        "observed_train_hyperedge_trace_coverage="
        f"{gate['observed_train_hyperedge_trace_coverage']:.6f}, "
        "required_train_hyperedge_trace_coverage="
        f"{gate['min_train_hyperedge_trace_coverage']:.6f}"
    )


def audit_trace_cohort(
    traces_dir: str | Path,
    *,
    split_seed: int,
    validation_ratio: float,
    test_ratio: float,
    require_hyperedges: bool,
    min_train_hyperedge_trace_coverage: float,
    min_total_hyperedges: int,
    min_effect: float,
    min_synergy: float,
) -> dict[str, Any]:
    from hypergraph.attention.cct.data import TraceRepository
    from hypergraph.attention.splitting import (
        FixedHoldoutConfig,
        FixedHoldoutSplitter,
    )
    from hypergraph.attention.cct.training import (
        trace_cohort_sha256,
        trace_sidecar_sha256,
    )

    if (
        not math.isfinite(min_train_hyperedge_trace_coverage)
        or not 0.0 < min_train_hyperedge_trace_coverage <= 1.0
    ):
        raise ValueError(
            "min_train_hyperedge_trace_coverage must be finite and in (0, 1]"
        )
    if (
        isinstance(min_total_hyperedges, bool)
        or not isinstance(min_total_hyperedges, int)
        or min_total_hyperedges <= 0
    ):
        raise ValueError("min_total_hyperedges must be a positive integer")
    if not math.isfinite(min_effect) or min_effect < 0.0:
        raise ValueError("min_effect must be finite and non-negative")
    if not math.isfinite(min_synergy) or min_synergy <= 0.0:
        raise ValueError("min_synergy must be finite and positive")

    traces = list(TraceRepository(traces_dir).traces())
    trace_ids = [trace.trace_id for trace in traces]
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("trace repository contains duplicate logical trace IDs")
    config_paths = sorted(Path(traces_dir).glob("extraction_config_*.json"))
    if not config_paths:
        raise ValueError("trace repository lacks extraction configuration")
    extraction_configs = [_read_json(path) for path in config_paths]
    for path, config in zip(config_paths, extraction_configs):
        try:
            configured_effect = float(config["min_effect"])
            configured_synergy = float(config["min_synergy"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{path} lacks valid intervention thresholds"
            ) from error
        if (
            not math.isclose(
                configured_effect,
                min_effect,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                configured_synergy,
                min_synergy,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"{path} intervention thresholds disagree with the audit"
            )
    written = sum(int(config.get("written", -1)) for config in extraction_configs)
    failed = sum(int(config.get("failed", -1)) for config in extraction_configs)
    if written != len(traces) or failed != 0:
        raise ValueError(
            "extraction configuration does not describe the complete "
            "failure-free trace cohort"
        )

    for trace in traces:
        features = trace.graph.edge_features
        if not len(features):
            continue
        signed_effect = features[:, 0]
        absolute_effect = features[:, 1]
        synergy = features[:, 2]
        prompt_fraction = features[:, 3]
        if not np.allclose(
            absolute_effect,
            np.abs(signed_effect),
            rtol=1e-8,
            atol=1e-12,
        ):
            raise ValueError("edge absolute_effect does not match signed_effect")
        if np.any((prompt_fraction < 0.0) | (prompt_fraction > 1.0)):
            raise ValueError("edge prompt_fraction must lie in [0, 1]")
        if np.any(np.abs(signed_effect) < min_effect):
            raise ValueError("an edge does not satisfy the extraction effect threshold")
        pair = trace.graph.edge_kind == "pair"
        hyper = trace.graph.edge_kind == "hyper"
        if np.any(np.abs(synergy[pair]) > 1e-12):
            raise ValueError("pair edges must have zero synergy")
        if np.any(np.abs(synergy[hyper]) < min_synergy):
            raise ValueError(
                "a hyperedge does not satisfy the extraction synergy threshold"
            )
    edge_kind_counts = Counter(
        str(kind)
        for trace in traces
        for kind in trace.graph.edge_kind.tolist()
    )
    graph = {
        "traces": len(traces),
        "nodes": sum(trace.graph.num_nodes for trace in traces),
        "edges": sum(trace.graph.num_edges for trace in traces),
        "memberships": sum(
            trace.graph.incidence.shape[1] for trace in traces
        ),
        "edge_kind_counts": dict(sorted(edge_kind_counts.items())),
        "traces_with_hyperedges": sum(
            bool((trace.graph.edge_kind == "hyper").any())
            for trace in traces
        ),
        "extraction_contract": {
            "config_files": [path.name for path in config_paths],
            "min_effect": min_effect,
            "min_synergy": min_synergy,
            "written": written,
            "failed": failed,
        },
    }

    splitter = FixedHoldoutSplitter(
        FixedHoldoutConfig(
            seed=split_seed,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
    )
    split = splitter.split([trace.split_record() for trace in traces])
    partitions: dict[str, Any] = {}
    for partition in (split.train, split.validation, split.test):
        cohort = [traces[index] for index in partition.indices]
        labels = [trace.response_label for trace in cohort]
        kinds = Counter(
            str(kind)
            for trace in cohort
            for kind in trace.graph.edge_kind.tolist()
        )
        response_class_counts = {
            str(label): sum(value == label for value in labels)
            for label in (0, 1)
        }
        partitions[partition.name] = {
            "traces": len(cohort),
            "response_class_counts": response_class_counts,
            "edge_kind_counts": dict(sorted(kinds.items())),
            "traces_with_hyperedges": sum(
                bool((trace.graph.edge_kind == "hyper").any())
                for trace in cohort
            ),
            "problem_ids": list(partition.group_ids),
            "trace_ids": list(partition.trace_ids),
        }
        if set(labels) != {0, 1}:
            raise ValueError(
                f"{partition.name} must contain both response classes; "
                f"response_class_counts={response_class_counts}"
            )
    train_traces = int(partitions["train"]["traces"])
    train_hyperedge_traces = int(
        partitions["train"]["traces_with_hyperedges"]
    )
    train_hyperedge_trace_coverage = train_hyperedge_traces / train_traces
    total_hyperedges = int(edge_kind_counts.get("hyper", 0))
    hypergraph_gate_passed = (
        total_hyperedges >= min_total_hyperedges
        and train_hyperedge_trace_coverage
        >= min_train_hyperedge_trace_coverage
    )
    hypergraph_gate = {
        "passed": hypergraph_gate_passed,
        "min_total_hyperedges": min_total_hyperedges,
        "observed_total_hyperedges": total_hyperedges,
        "min_train_hyperedge_trace_coverage": (
            min_train_hyperedge_trace_coverage
        ),
        "observed_train_hyperedge_traces": train_hyperedge_traces,
        "observed_train_traces": train_traces,
        "observed_train_hyperedge_trace_coverage": (
            train_hyperedge_trace_coverage
        ),
    }
    if require_hyperedges and not hypergraph_gate_passed:
        raise ValueError(_hypergraph_gate_failure_message(hypergraph_gate))

    return {
        "result_kind": classify_result_kind(
            graph["edge_kind_counts"],
            hypergraph_gate_passed=hypergraph_gate_passed,
        ),
        "graph": graph,
        "hypergraph_gate": hypergraph_gate,
        "split": {
            "split_seed": split_seed,
            "validation_ratio": validation_ratio,
            "test_ratio": test_ratio,
            "source_provenance": {
                "trace_cohort_sha256": trace_cohort_sha256(traces_dir),
                "trace_sidecar_sha256": trace_sidecar_sha256(traces_dir),
            },
            "partitions": partitions,
        },
    }


def _required_artifacts(root: Path) -> None:
    for name in (
        "metrics.json",
        "model.pt",
        "model.safetensors",
        "checkpoint.json",
        "normalizer.npz",
        "predictions_validation.csv",
        "predictions_test.csv",
        "split.json",
        "history.csv",
        "config.json",
    ):
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty real-training artifact: {path}")


def _string_list(value: object, *, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return value


def _prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = {
            "trace_id",
            "problem_id",
            "label",
            "probability",
            "first_error",
            "predicted_step",
            "step_probabilities",
        }
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(f"{path} has an unexpected CSV schema")
        return list(reader)


def _assert_nested_close(
    actual: object,
    expected: object,
    *,
    name: str,
    relative_tolerance: float = 1e-5,
    absolute_tolerance: float = 1e-5,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{name} has an unexpected object schema")
        for key, value in expected.items():
            _assert_nested_close(
                actual[key],
                value,
                name=f"{name}.{key}",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"{name} has an unexpected sequence schema")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            _assert_nested_close(
                actual_item,
                expected_item,
                name=f"{name}[{index}]",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        return
    if isinstance(expected, float):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(
                float(actual),
                expected,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            )
        ):
            raise ValueError(f"{name} does not match checkpoint evaluation")
        return
    if actual != expected:
        raise ValueError(f"{name} does not match checkpoint evaluation")


def audit_training_artifacts(
    training_dir: str | Path,
    traces_dir: str | Path,
    preflight_path: str | Path,
    *,
    source_sha256_path: str | Path | None = None,
) -> dict[str, Any]:
    from hypergraph.attention.cct.data import TraceRepository
    from hypergraph.attention.cct.training import (
        CausalTransportTrainer,
        load_training_checkpoint,
        trace_cohort_sha256,
        trace_sidecar_sha256,
    )

    root = Path(training_dir)
    _required_artifacts(root)
    metrics = _read_json(root / "metrics.json")
    split = _read_json(root / "split.json")
    preflight = _read_json(Path(preflight_path))
    traces = list(TraceRepository(traces_dir).traces())
    by_id = {trace.trace_id: trace for trace in traces}
    if len(by_id) != len(traces):
        raise ValueError("trace repository contains duplicate logical trace IDs")

    checkpoint = _read_json(root / "checkpoint.json")
    cohort_sha256 = trace_cohort_sha256(traces_dir)
    sidecar_sha256 = trace_sidecar_sha256(traces_dir)
    if checkpoint.get("trace_cohort_sha256") != cohort_sha256:
        raise ValueError("checkpoint trace cohort hash mismatch")
    if checkpoint.get("trace_sidecar_sha256") != sidecar_sha256:
        raise ValueError("checkpoint trace sidecar hash mismatch")
    preflight_provenance = preflight.get("source_provenance")
    if preflight_provenance != {
        "trace_cohort_sha256": cohort_sha256,
        "trace_sidecar_sha256": sidecar_sha256,
    }:
        raise ValueError("preflight trace provenance mismatch")
    if metrics.get("best_epoch") != checkpoint.get("best_epoch"):
        raise ValueError("metrics and checkpoint best epochs disagree")
    if _read_json(root / "config.json") != checkpoint.get("training_config"):
        raise ValueError("config and checkpoint training settings disagree")
    source_sha256: str | None = None
    if source_sha256_path is not None:
        source_sha256 = Path(source_sha256_path).read_text(
            encoding="utf-8"
        ).strip()
        if (
            len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)
        ):
            raise ValueError("source-tree SHA256 witness is not canonical")
        if checkpoint.get("source_tree_sha256") != source_sha256:
            raise ValueError("checkpoint source-tree hash mismatch")
    model, normalizer = load_training_checkpoint(root, device="cpu")
    model.eval()

    split_ids = split.get("partition_trace_ids")
    split_groups = split.get("partition_group_ids")
    preflight_partitions = preflight.get("partitions")
    if (
        not isinstance(split_ids, dict)
        or not isinstance(split_groups, dict)
        or not isinstance(preflight_partitions, dict)
    ):
        raise ValueError("split/preflight partition metadata are incomplete")

    names = ("train", "validation", "test")
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    for partition in names:
        ids = _string_list(
            split_ids.get(partition),
            name=f"split {partition} trace IDs",
        )
        groups = _string_list(
            split_groups.get(partition),
            name=f"split {partition} problem IDs",
        )
        if len(ids) != len(set(ids)) or seen_ids.intersection(ids):
            raise ValueError("split trace IDs are duplicated across partitions")
        if len(groups) != len(set(groups)) or seen_groups.intersection(groups):
            raise ValueError("split problem IDs are duplicated across partitions")
        seen_ids.update(ids)
        seen_groups.update(groups)
        if any(trace_id not in by_id for trace_id in ids):
            raise ValueError(f"{partition} contains an unknown trace ID")
        actual_groups = sorted({by_id[trace_id].problem_id for trace_id in ids})
        if groups != actual_groups:
            raise ValueError(f"{partition} problem IDs do not match its traces")
        preflight_partition = preflight_partitions.get(partition)
        if not isinstance(preflight_partition, dict):
            raise ValueError(f"preflight lacks the {partition} partition")
        if ids != preflight_partition.get("trace_ids"):
            raise ValueError(f"{partition} split disagrees with preflight")
        if groups != preflight_partition.get("problem_ids"):
            raise ValueError(f"{partition} groups disagree with preflight")
    if seen_ids != set(by_id):
        raise ValueError("split does not cover the complete trace cohort")

    summary: dict[str, Any] = {
        "best_epoch": metrics.get("best_epoch"),
        "checkpoint_schema_version": checkpoint.get("schema_version"),
        "source_tree_sha256": source_sha256,
        "partitions": {},
    }
    for partition in ("validation", "test"):
        ids = split_ids[partition]
        rows = _prediction_rows(root / f"predictions_{partition}.csv")
        row_ids = [row["trace_id"] for row in rows]
        if row_ids != ids or len(row_ids) != len(set(row_ids)):
            raise ValueError(
                f"{partition} predictions do not exactly match the split"
            )
        normalized = [normalizer.transform(by_id[trace_id]) for trace_id in ids]
        training_config = checkpoint["training_config"]
        reproduced = CausalTransportTrainer.evaluate(
            model,
            normalized,
            bootstrap_replicates=int(training_config["bootstrap_replicates"]),
            bootstrap_confidence=float(training_config["bootstrap_confidence"]),
            bootstrap_seed=int(training_config["seed"])
            + int(partition == "test"),
            batch_size=int(training_config["batch_size"]),
        )
        reproduced_payload = {
            "mean_loss": reproduced.mean_loss,
            "response": asdict(reproduced.response),
            "localization": asdict(reproduced.localization),
            "problem_bootstrap": (
                asdict(reproduced.uncertainty)
                if reproduced.uncertainty is not None
                else None
            ),
        }
        _assert_nested_close(
            metrics.get(partition),
            reproduced_payload,
            name=f"metrics.{partition}",
        )
        response = reproduced_payload["response"]
        n = int(response["n"])
        positives = int(response["positives"])
        if n != len(rows) or not 0 < positives < n:
            raise ValueError(f"{partition} is not a binary holdout")

        for row, trace_id, prediction in zip(
            rows,
            ids,
            reproduced.predictions,
        ):
            trace = by_id[trace_id]
            if row["problem_id"] != trace.problem_id:
                raise ValueError(f"{partition} prediction problem ID mismatch")
            if int(row["label"]) != trace.response_label:
                raise ValueError(f"{partition} prediction label mismatch")
            if int(row["first_error"]) != trace.labels.first_error:
                raise ValueError(
                    f"{partition} prediction first-error label mismatch"
                )
            if int(row["predicted_step"]) != prediction.predicted_step:
                raise ValueError(
                    f"{partition} prediction step argmax mismatch"
                )
            stored_steps = np.asarray(
                json.loads(row["step_probabilities"]),
                dtype=np.float64,
            )
            if (
                stored_steps.shape != (len(prediction.step_probabilities),)
                or not np.allclose(
                    prediction.step_probabilities,
                    stored_steps,
                    rtol=1e-5,
                    atol=1e-5,
                )
            ):
                raise ValueError(
                    f"{partition} checkpoint does not reproduce step scores"
                )
            if not math.isclose(
                prediction.probability,
                float(row["probability"]),
                rel_tol=1e-5,
                abs_tol=1e-5,
            ):
                raise ValueError(
                    f"{partition} checkpoint does not reproduce probability"
                )
        if positives != sum(by_id[trace_id].response_label for trace_id in ids):
            raise ValueError(f"{partition} positive count mismatch")
        summary["partitions"][partition] = {
            "response": response,
            "localization": reproduced_payload["localization"],
            "prediction_rows": len(rows),
            "checkpoint_reproduced_predictions": len(reproduced.predictions),
        }
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m crwh.real_cct_audit")
    commands = parser.add_subparsers(dest="command", required=True)
    traces = commands.add_parser("traces")
    traces.add_argument("--traces", required=True)
    traces.add_argument("--output-dir", required=True)
    traces.add_argument("--split-seed", type=int, required=True)
    traces.add_argument("--validation-ratio", type=float, required=True)
    traces.add_argument("--test-ratio", type=float, required=True)
    traces.add_argument(
        "--require-hyperedges",
        type=int,
        choices=(0, 1),
        required=True,
    )
    traces.add_argument(
        "--min-train-hyperedge-trace-coverage",
        type=float,
        required=True,
    )
    traces.add_argument("--min-total-hyperedges", type=int, required=True)
    traces.add_argument("--min-effect", type=float, required=True)
    traces.add_argument("--min-synergy", type=float, required=True)
    training = commands.add_parser("training")
    training.add_argument("--training-dir", required=True)
    training.add_argument("--traces", required=True)
    training.add_argument("--preflight", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--source-tree-sha256-file", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "traces":
        audit = audit_trace_cohort(
            args.traces,
            split_seed=args.split_seed,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            require_hyperedges=False,
            min_train_hyperedge_trace_coverage=(
                args.min_train_hyperedge_trace_coverage
            ),
            min_total_hyperedges=args.min_total_hyperedges,
            min_effect=args.min_effect,
            min_synergy=args.min_synergy,
        )
        root = Path(args.output_dir)
        _write_json(root / "cct-graph-audit.json", audit["graph"])
        _write_json(
            root / "cct-hypergraph-gate.json",
            audit["hypergraph_gate"],
        )
        _write_json(root / "cct-split-preflight.json", audit["split"])
        _write_json(
            root / "cct-result-kind.json",
            {"result_kind": audit["result_kind"]},
        )
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        if bool(args.require_hyperedges) and not audit["hypergraph_gate"][
            "passed"
        ]:
            raise ValueError(
                _hypergraph_gate_failure_message(audit["hypergraph_gate"])
            )
        return 0
    if args.command == "training":
        summary = audit_training_artifacts(
            args.training_dir,
            args.traces,
            args.preflight,
            source_sha256_path=args.source_tree_sha256_file,
        )
        _write_json(Path(args.output), summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
