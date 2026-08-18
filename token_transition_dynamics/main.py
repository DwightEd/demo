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


def _comma_ints(value: str) -> tuple[int, ...]:
    try:
        items = tuple(int(item) for item in _comma_strings(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("clusters must be comma-separated integers") from error
    if any(item < 1 for item in items):
        raise argparse.ArgumentTypeError("cluster counts must be positive")
    return items


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Correct-only token transition dynamics on raw LLM hidden states."
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
    run.add_argument("--pca-dim", type=int, default=16)
    run.add_argument("--clusters", type=_comma_ints, default=(1, 2))
    run.add_argument("--tokens-per-chain", type=int, default=8)
    run.add_argument("--max-test-tokens-per-chain", type=int, default=64)
    run.add_argument("--max-pca-rows", type=int, default=4096)
    run.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def _config(arguments: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        data_root=arguments.data_root,
        domains=arguments.domains,
        output_dir=arguments.output_dir,
        manifest_name=arguments.manifest_name,
        pca_dim=getattr(arguments, "pca_dim", 16),
        clusters=getattr(arguments, "clusters", (1, 2)),
        tokens_per_chain=getattr(arguments, "tokens_per_chain", 8),
        max_test_tokens_per_chain=getattr(arguments, "max_test_tokens_per_chain", 64),
        max_pca_rows=getattr(arguments, "max_pca_rows", 4096),
        bootstrap_samples=getattr(arguments, "bootstrap_samples", 2000),
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
