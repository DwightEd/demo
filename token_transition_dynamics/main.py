from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import ExperimentConfig, TokenTransitionExperiment


def _comma_strings(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("provide at least one comma-separated value")
    return items


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Correct-only raw-space token-window geometry on LLM hidden states."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--data-root", type=Path, required=True)
        subparser.add_argument("--domains", type=_comma_strings, required=True)
        subparser.add_argument(
            "--manifest-name", default="trace.raw_residual_stream.npz"
        )
        subparser.add_argument("--max-records-per-domain", type=int)
        subparser.add_argument("--output-dir", type=Path, default=Path("outputs/token_transition_dynamics"))
        subparser.add_argument("--seed", type=int, default=17)
    run = subparsers.choices["run"]
    run.add_argument("--window-size", type=int, default=24)
    run.add_argument("--neighbors", type=int, default=20)
    run.add_argument("--tle-centers", type=int, default=6)
    run.add_argument("--train-windows-per-chain", type=int, default=4)
    run.add_argument("--calibration-windows-per-chain", type=int, default=4)
    run.add_argument("--max-test-windows-per-chain", type=int, default=12)
    run.add_argument("--position-bins", type=int, default=4)
    run.add_argument("--min-baseline-samples", type=int, default=8)
    run.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser


def _config(arguments: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        data_root=arguments.data_root,
        domains=arguments.domains,
        output_dir=arguments.output_dir,
        manifest_name=arguments.manifest_name,
        window_size=getattr(arguments, "window_size", 24),
        neighbors=getattr(arguments, "neighbors", 20),
        tle_centers=getattr(arguments, "tle_centers", 6),
        train_windows_per_chain=getattr(arguments, "train_windows_per_chain", 4),
        calibration_windows_per_chain=getattr(
            arguments, "calibration_windows_per_chain", 4
        ),
        max_test_windows_per_chain=getattr(
            arguments, "max_test_windows_per_chain", 12
        ),
        position_bins=getattr(arguments, "position_bins", 4),
        min_baseline_samples=getattr(arguments, "min_baseline_samples", 8),
        bootstrap_samples=getattr(arguments, "bootstrap_samples", 1000),
        seed=arguments.seed,
        max_records_per_domain=arguments.max_records_per_domain,
    )


def main() -> None:
    arguments = _parser().parse_args()
    experiment = TokenTransitionExperiment(_config(arguments))
    result = experiment.inspect() if arguments.command == "preflight" else experiment.run()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
