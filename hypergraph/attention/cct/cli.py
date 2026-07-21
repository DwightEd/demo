from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..evaluation import PredictionRow
from ..splitting import FixedHoldoutConfig, FixedHoldoutSplitter
from .data import TraceRepository
from .training import (
    CausalTransportTrainer,
    EvaluationResult,
    TrainingConfig,
    TrainingResult,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _evaluation_payload(result: EvaluationResult) -> dict[str, Any]:
    return {
        "mean_loss": result.mean_loss,
        "response": asdict(result.response),
        "localization": asdict(result.localization),
        "problem_bootstrap": (
            asdict(result.uncertainty) if result.uncertainty is not None else None
        ),
    }


def _write_predictions(path: Path, rows: Sequence[PredictionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "trace_id",
                "problem_id",
                "label",
                "probability",
                "first_error",
                "predicted_step",
                "step_probabilities",
            ),
        )
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            values["step_probabilities"] = json.dumps(row.step_probabilities)
            writer.writerow(values)


def _response_summary(report, uncertainty) -> dict[str, Any]:
    return {
        "n": report.n,
        "positives": report.positives,
        "auroc": report.auroc,
        "aupr": report.aupr,
        "balanced_accuracy": report.balanced_accuracy,
        "mcc": report.mcc,
        "problem_bootstrap": asdict(uncertainty) if uncertainty is not None else None,
    }


def inspect_command(args: argparse.Namespace) -> int:
    traces = list(TraceRepository(args.traces).traces())
    summary = {
        "traces": len(traces),
        "problems": len({trace.problem_id for trace in traces}),
        "errors": sum(trace.response_label for trace in traces),
        "layers": sorted({trace.layer_id for trace in traces}),
        "node_dimensions": sorted(
            {trace.graph.node_features.shape[1] for trace in traces}
        ),
        "edge_dimensions": sorted(
            {trace.graph.edge_features.shape[1] for trace in traces}
        ),
        "generators": sorted({trace.generator_model for trace in traces}),
        "observers": sorted({trace.observer_model for trace in traces}),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def extract_command(args: argparse.Namespace) -> int:
    from .hf_backend import HuggingFaceExtractionConfig, HuggingFaceTransportBackend
    from .pipeline import CausalTraceAssembler
    from .processbench import ProcessBenchReader, ShardSpec

    config = HuggingFaceExtractionConfig(
        layer_id=args.layer,
        top_sources=args.top_sources,
        node_dim=args.node_dim,
        projection_seed=args.projection_seed,
        device=args.device,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
    )
    backend = HuggingFaceTransportBackend.from_pretrained(args.model, config)
    assembler = CausalTraceAssembler(
        bundle_energy=args.bundle_energy,
        min_effect=args.min_effect,
        min_synergy=args.min_synergy,
    )
    repository = TraceRepository(args.output)
    records = list(ProcessBenchReader(args.input).records())
    if args.limit is not None:
        records = records[: args.limit]
    shard = ShardSpec(args.num_shards, args.shard_index)
    records = [record for index, record in enumerate(records) if shard.includes(index)]
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(records, desc="CCT extraction", unit="trace")
    except ImportError:  # pragma: no cover
        iterator = records
    failures: list[dict[str, str]] = []
    written = 0
    for record in iterator:
        destination = Path(args.output) / f"{record.trace_id}.npz"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"{destination} already exists; pass --overwrite explicitly"
            )
        try:
            repository.save(assembler.assemble(backend.extract(record)))
            written += 1
        except Exception as error:
            if not args.skip_invalid:
                raise
            failures.append(
                {
                    "trace_id": record.trace_id,
                    "problem_id": record.problem_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
    output = Path(args.output)
    _write_json(output / f"extraction_failures_{shard.tag}.json", failures)
    _write_json(
        output / f"extraction_config_{shard.tag}.json",
        {
            "input": str(Path(args.input).resolve()),
            "model": args.model,
            "layer": args.layer,
            "top_sources": args.top_sources,
            "node_dim": args.node_dim,
            "projection_seed": args.projection_seed,
            "attention_implementation": args.attention_implementation,
            "bundle_energy": args.bundle_energy,
            "min_effect": args.min_effect,
            "min_synergy": args.min_synergy,
            "written": written,
            "failed": len(failures),
            "num_shards": shard.num_shards,
            "shard_index": shard.shard_index,
        },
    )
    print(f"extracted={written} failed={len(failures)} output={output.resolve()}")
    return 0


def _training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.model_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        gradient_clip=args.gradient_clip,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_confidence=args.bootstrap_confidence,
    )


def _splitter(args: argparse.Namespace) -> FixedHoldoutSplitter:
    return FixedHoldoutSplitter(
        FixedHoldoutConfig(
            seed=args.split_seed,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
        )
    )


def _save_result(output: Path, result: TrainingResult, config: TrainingConfig) -> None:
    import torch

    output.mkdir(parents=True, exist_ok=True)
    torch.save(result.model_state, output / "model.pt")
    np.savez(
        output / "normalizer.npz",
        mean=result.normalizer.mean,
        scale=result.normalizer.scale,
        content_dim=np.asarray(result.normalizer.content_dim, dtype=np.int64),
    )
    _write_json(output / "config.json", asdict(config))
    _write_json(
        output / "metrics.json",
        {
            "best_epoch": result.best_epoch,
            "validation": _evaluation_payload(result.validation),
            "test": _evaluation_payload(result.test),
        },
    )
    _write_predictions(
        output / "predictions_validation.csv", result.validation.predictions
    )
    _write_predictions(output / "predictions_test.csv", result.test.predictions)
    _write_json(output / "split.json", result.split.manifest())
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "epoch",
                "train_loss",
                "validation_loss",
                "validation_auroc",
            ),
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in result.history)


def train_command(args: argparse.Namespace) -> int:
    traces = list(TraceRepository(args.traces).traces())
    config = _training_config(args)
    result = CausalTransportTrainer(config).fit(traces, _splitter(args))
    output = Path(args.output)
    _save_result(output, result, config)

    test = result.test.response
    print(
        "test: "
        f"n={test.n} positives={test.positives} "
        f"AUROC={test.auroc} AUPRC={test.aupr} "
        f"balanced_accuracy={test.balanced_accuracy:.6f} MCC={test.mcc:.6f}"
    )
    print(f"results: {output.resolve()}")
    return 0


def benchmark_command(args: argparse.Namespace) -> int:
    from .baselines import NuisanceLogisticBaseline
    from .controls import (
        CausalCardinalityRewire,
        HiddenOnlyControl,
        NoEdgeControl,
        NoGeometryControl,
        PairwiseControl,
    )

    traces = list(TraceRepository(args.traces).traces())
    config = _training_config(args)
    controls = {
        "full": None,
        "hidden_only": HiddenOnlyControl(),
        "no_edge": NoEdgeControl(),
        "pairwise": PairwiseControl(),
        "causal_cardinality_rewire": CausalCardinalityRewire(seed=args.seed),
        "no_geometry": NoGeometryControl(),
    }
    summary: dict[str, dict[str, Any]] = {}
    root = Path(args.output)
    split = _splitter(args).split([trace.split_record() for trace in traces])
    nuisance = NuisanceLogisticBaseline().fit(
        traces,
        split,
        bootstrap_replicates=config.bootstrap_replicates,
        bootstrap_confidence=config.bootstrap_confidence,
        bootstrap_seed=config.seed,
    )
    summary["nuisance_only"] = _response_summary(
        nuisance.test, nuisance.test_uncertainty
    )
    nuisance_root = root / "nuisance_only"
    nuisance_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        nuisance_root / "metrics.json",
        {
            "validation": {
                "response": asdict(nuisance.validation),
                "problem_bootstrap": asdict(nuisance.validation_uncertainty),
            },
            "test": {
                "response": asdict(nuisance.test),
                "problem_bootstrap": asdict(nuisance.test_uncertainty),
            },
        },
    )
    _write_predictions(
        nuisance_root / "predictions_validation.csv",
        nuisance.validation_predictions,
    )
    _write_predictions(
        nuisance_root / "predictions_test.csv", nuisance.test_predictions
    )
    np.savez(
        nuisance_root / "model.npz",
        weights=nuisance.weights,
        mean=nuisance.mean,
        scale=nuisance.scale,
    )
    print(f"nuisance_only: AUROC={nuisance.test.auroc} AUPRC={nuisance.test.aupr}")
    for name, control in controls.items():
        cohort = (
            traces if control is None else [control.apply(trace) for trace in traces]
        )
        result = CausalTransportTrainer(config).fit(cohort, _splitter(args))
        _save_result(root / name, result, config)
        report = result.test.response
        summary[name] = _response_summary(report, result.test.uncertainty)
        print(f"{name}: AUROC={report.auroc} AUPRC={report.aupr}")
    _write_json(root / "benchmark.json", summary)
    print(f"benchmark: {root.resolve()}")
    return 0


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--model-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cct-hg")
    commands = parser.add_subparsers(dest="command", required=True)
    extract_parser = commands.add_parser(
        "extract", help="extract output-effective causal transport traces"
    )
    extract_parser.add_argument("--input", required=True)
    extract_parser.add_argument("--model", required=True)
    extract_parser.add_argument("--output", required=True)
    extract_parser.add_argument("--layer", type=int, required=True)
    extract_parser.add_argument("--top-sources", type=int, default=3)
    extract_parser.add_argument("--node-dim", type=int, default=64)
    extract_parser.add_argument("--projection-seed", type=int, default=17)
    extract_parser.add_argument("--bundle-energy", type=float, default=0.9)
    extract_parser.add_argument("--min-effect", type=float, default=0.01)
    extract_parser.add_argument("--min-synergy", type=float, default=0.01)
    extract_parser.add_argument("--dtype", default="bfloat16")
    extract_parser.add_argument(
        "--attention-implementation", choices=("sdpa", "eager"), default="sdpa"
    )
    extract_parser.add_argument("--device", default="cuda")
    extract_parser.add_argument("--limit", type=int)
    extract_parser.add_argument("--num-shards", type=int, default=1)
    extract_parser.add_argument("--shard-index", type=int, default=0)
    extract_parser.add_argument("--overwrite", action="store_true")
    extract_parser.add_argument("--skip-invalid", action="store_true")
    extract_parser.set_defaults(handler=extract_command)

    inspect_parser = commands.add_parser("inspect", help="validate a trace cohort")
    inspect_parser.add_argument("--traces", required=True)
    inspect_parser.set_defaults(handler=inspect_command)

    train_parser = commands.add_parser("train", help="train the CCT-HG detector")
    _add_training_arguments(train_parser)
    train_parser.set_defaults(handler=train_command)

    benchmark_parser = commands.add_parser(
        "benchmark", help="train CCT-HG and all mandatory graph controls"
    )
    _add_training_arguments(benchmark_parser)
    benchmark_parser.set_defaults(handler=benchmark_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
