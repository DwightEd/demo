from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import PairAuditor
from .evaluation import MonitorRow, localization_metrics
from .experiment import (
    InterventionExperiment,
    InterventionExperimentConfig,
    summarize_saved_interventions,
)
from .extraction import OnsetTraceExtraction, OnsetTraceExtractionConfig
from .monitor_experiment import (
    MonitorExperimentConfig,
    ProcessBenchMonitorExperiment,
)
from .monitor_training import MONITOR_ARMS, MonitorTrainingConfig


def _domains(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one domain")
    return result


def _layers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("expected at least one layer")
    if min(result) < 1:
        raise argparse.ArgumentTypeError("layers must be positive one-based indices")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("layers must be unique")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _fraction(value: str) -> float:
    result = float(value)
    if not 0.0 < result < 1.0:
        raise argparse.ArgumentTypeError("value must lie in (0,1)")
    return result


def _names(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated non-empty list")
    return result


def _arms(value: str) -> tuple[str, ...]:
    result = _names(value)
    unknown = sorted(set(result).difference(MONITOR_ARMS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown monitor arms {unknown}; available={list(MONITOR_ARMS)}"
        )
    return result


def _add_processbench_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--domains",
        type=_domains,
        default=("gsm8k", "math", "olympiadbench", "omnimath"),
    )


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    _add_processbench_arguments(parser)
    parser.add_argument("--pair-directory", default="causal_first_error_v1")


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    _add_data_arguments(parser)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--tokenizer-revision", default="unknown")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layers", type=_layers, required=True)
    parser.add_argument("--max-cases-per-domain", type=_nonnegative_int, default=0)
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled first-error attribution and repair"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="audit causal pair identifiability")
    _add_data_arguments(audit)
    audit.add_argument("--output", type=Path, default=None)

    extract = commands.add_parser("extract", help="extract decision-token graphs")
    _add_model_arguments(extract)
    extract.add_argument("--topk", type=_positive_int, default=20)

    intervene = commands.add_parser(
        "intervene", help="run controlled attention x FFN interventions"
    )
    _add_model_arguments(intervene)

    summarize = commands.add_parser("summarize", help="summarize saved interventions")
    _add_data_arguments(summarize)
    summarize.add_argument("--output", type=Path, default=None)

    evaluate = commands.add_parser(
        "evaluate-monitor", help="evaluate saved per-boundary monitor scores"
    )
    evaluate.add_argument("--scores", required=True, type=Path)
    evaluate.add_argument("--false-alarm-threshold", type=float, default=0.5)
    evaluate.add_argument("--output", type=Path, default=None)

    monitor = commands.add_parser(
        "train-monitor",
        help="train the future-free ProcessBench residual-depth monitor",
    )
    _add_processbench_arguments(monitor)
    monitor.add_argument("--output-dir", required=True, type=Path)
    monitor.add_argument("--arms", type=_arms, default=MONITOR_ARMS)
    monitor.add_argument(
        "--output-features", type=_names, default=("token_entropy", "token_nll")
    )
    monitor.add_argument("--max-chains-per-domain", type=_nonnegative_int, default=0)
    monitor.add_argument("--validation-fraction", type=_fraction, default=0.15)
    monitor.add_argument("--target-correct-chain-false-alarm", type=float, default=0.1)
    monitor.add_argument("--bootstrap", type=_positive_int, default=1000)
    monitor.add_argument("--shuffle-repeats", type=_positive_int, default=3)
    monitor.add_argument("--seed", type=int, default=17)
    monitor.add_argument("--width", type=_positive_int, default=64)
    monitor.add_argument("--message-passing-steps", type=_positive_int, default=2)
    monitor.add_argument("--dropout", type=float, default=0.1)
    monitor.add_argument("--epochs", type=_positive_int, default=20)
    monitor.add_argument("--patience", type=_positive_int, default=4)
    monitor.add_argument("--batch-size", type=_positive_int, default=32)
    monitor.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    monitor.add_argument("--weight-decay", type=float, default=1e-4)
    monitor.add_argument("--device", default="cuda")
    return parser


def _load_model(args: argparse.Namespace):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "causal extraction requires torch and transformers in the active Python"
        ) from exc
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {args.model_dir}")
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if args.dtype not in dtypes:
        raise ValueError(f"unsupported dtype {args.dtype!r}")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        torch_dtype=dtypes[args.dtype],
        attn_implementation="eager",
    )
    model.to(torch.device(args.device))
    model.eval()
    return model


def _print_audit(report: dict[str, object]) -> None:
    print(
        "causal audit: "
        f"chains={report['natural_chains']} | "
        f"errors={report['error_chains']} | correct={report['correct_chains']}"
    )
    print(
        "verified pairs: "
        f"target_correction={report['valid_target_correction_pairs']} | "
        f"controlled_root={report['valid_controlled_root_pairs']}"
    )
    if report["root_cause_claim"] == "not_identifiable_from_available_data":
        print("root-cause analysis: not identifiable from available data")
        print(f"next required artifact: {report['next_required_artifact']}")
    else:
        print("root-cause analysis: controlled pairs available")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train-monitor":
        report = ProcessBenchMonitorExperiment(
            MonitorExperimentConfig(
                data_root=args.data_root,
                domains=args.domains,
                output_dir=args.output_dir,
                arms=args.arms,
                output_features=args.output_features,
                max_chains_per_domain=args.max_chains_per_domain,
                validation_fraction=args.validation_fraction,
                target_correct_chain_false_alarm=args.target_correct_chain_false_alarm,
                bootstrap_repeats=args.bootstrap,
                shuffle_repeats=args.shuffle_repeats,
                seed=args.seed,
                training=MonitorTrainingConfig(
                    width=args.width,
                    message_passing_steps=args.message_passing_steps,
                    dropout=args.dropout,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    device=args.device,
                    show_progress=True,
                ),
            )
        ).run()
    elif args.command == "evaluate-monitor":
        rows = [
            MonitorRow(**json.loads(line))
            for line in args.scores.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report = localization_metrics(
            rows, false_alarm_threshold=args.false_alarm_threshold
        )
    elif args.command == "summarize":
        report = summarize_saved_interventions(
            args.data_root, args.domains, pair_directory=args.pair_directory
        )
    else:
        audit = PairAuditor(
            args.data_root,
            args.domains,
            pair_directory=args.pair_directory,
        ).run()
        if args.command == "audit":
            report = audit
            _print_audit(report)
        elif args.command == "extract":
            if not (
                audit["valid_target_correction_pairs"]
                or audit["valid_controlled_root_pairs"]
            ):
                print("run causal-audit first and create verified onset pairs")
                raise SystemExit("no verified onset pairs")
            model = _load_model(args)
            report = OnsetTraceExtraction(
                OnsetTraceExtractionConfig(
                    data_root=args.data_root,
                    domains=args.domains,
                    layers=args.layers,
                    model_name=args.model_name,
                    model_revision=args.model_revision,
                    tokenizer_name=args.tokenizer_name or args.model_name,
                    tokenizer_revision=args.tokenizer_revision,
                    pair_directory=args.pair_directory,
                    topk=args.topk,
                    max_cases_per_domain=args.max_cases_per_domain,
                    overwrite=args.overwrite,
                )
            ).run(model)
        else:
            if not audit["valid_controlled_root_pairs"]:
                print("controlled intervention requires verified controlled_root pairs")
                raise SystemExit("no verified controlled_root pairs")
            model = _load_model(args)
            report = InterventionExperiment(
                InterventionExperimentConfig(
                    data_root=args.data_root,
                    domains=args.domains,
                    layers=args.layers,
                    pair_directory=args.pair_directory,
                    max_cases_per_domain=args.max_cases_per_domain,
                    overwrite=args.overwrite,
                )
            ).run(model)
    output = getattr(args, "output", None)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.command != "audit":
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
