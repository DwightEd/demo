from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from ..progress import TqdmProgress
from .console import format_preflight_summary, format_run_summary
from .config import RawFunctionalConfig
from .contracts import TraceSource
from .experiment import inspect_hidden_geometry_sources, run_hidden_geometry_experiment


DEFAULT_DOMAINS = ("gsm8k", "math", "olympiadbench", "omnimath")


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def trace_sources(
    data_root: str | Path,
    domains: Iterable[str],
    manifest_name: str = "trace.raw_residual_stream.npz",
    acquisition_mode: str = "observer_teacher_forcing_replay",
) -> tuple[TraceSource, ...]:
    root = Path(data_root).expanduser()
    return tuple(
        TraceSource(
            dataset=str(domain),
            manifest=root / str(domain) / "selected" / manifest_name,
            acquisition_mode=acquisition_mode,
            exact_trace=root / str(domain) / "selected" / "trace.npz",
        )
        for domain in domains
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--domains", type=_csv, default=DEFAULT_DOMAINS)
    parser.add_argument(
        "--manifest-name", default="trace.raw_residual_stream.npz"
    )
    parser.add_argument("--response-generator", default="llama3.1-8b")
    parser.add_argument("--observer-model", default="llama3.1-8b")
    parser.add_argument(
        "--acquisition-mode", default="observer_teacher_forcing_replay"
    )
    parser.add_argument(
        "--output-features",
        type=_csv,
        default=("token_entropy", "token_nll"),
        help="stored logits-derived step summaries; these are not full-vocabulary logits",
    )
    parser.add_argument(
        "--max-records-per-domain",
        type=int,
        default=0,
        help="explicit smoke-test cap; 0 uses every eligible record",
    )
    parser.add_argument("--seed", type=int, default=17)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Held-domain discriminative analysis of real Llama observer residual streams."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="validate all real artifacts")
    _common(preflight)

    run = commands.add_parser("run", help="run the foreground LODO experiment")
    _common(run)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument(
        "--tasks", type=_csv, default=("whole_chain", "strict_prefix")
    )
    run.add_argument("--method", default="raw_functional_probe")
    run.add_argument("--pca-dim", type=int, default=16)
    run.add_argument("--time-basis", type=int, default=3)
    run.add_argument("--layer-basis", type=int, default=3)
    run.add_argument("--positions-per-chain", type=int, default=32)
    run.add_argument("--l2", type=float, default=1.0)
    run.add_argument("--restarts", type=int, default=3)
    run.add_argument("--max-iter", type=int, default=500)
    run.add_argument("--null-repeats", type=int, default=3)
    run.add_argument("--bootstrap", type=int, default=2000)
    run.add_argument(
        "--method-config-json",
        default=None,
        help="optional JSON object overriding a non-default method plugin's config",
    )
    return parser


def _method_config(args: argparse.Namespace) -> object:
    if args.method == "raw_functional_probe":
        return RawFunctionalConfig(
            pca_dim=args.pca_dim,
            time_basis=args.time_basis,
            layer_basis=args.layer_basis,
            positions_per_chain=args.positions_per_chain,
            l2=args.l2,
            restarts=args.restarts,
            max_iter=args.max_iter,
            null_repeats=args.null_repeats,
        )
    if args.method_config_json is None:
        return None
    try:
        config = json.loads(args.method_config_json)
    except json.JSONDecodeError as exc:
        raise ValueError("--method-config-json must contain valid JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("--method-config-json must decode to an object")
    return config


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sources = trace_sources(
        args.data_root, args.domains, args.manifest_name, args.acquisition_mode
    )
    common = dict(
        sources=sources,
        response_generator=args.response_generator,
        observer_model=args.observer_model,
        output_features=args.output_features,
        max_records_per_domain=args.max_records_per_domain,
        seed=args.seed,
    )
    if args.command == "preflight":
        result = inspect_hidden_geometry_sources(**common, progress=TqdmProgress())
        print(format_preflight_summary(result))
    else:
        result = run_hidden_geometry_experiment(
            **common,
            output_dir=args.output_dir,
            tasks=args.tasks,
            method_name=args.method,
            method_config=_method_config(args),
            n_boot=args.bootstrap,
            progress=TqdmProgress(),
        )
        print(format_run_summary(result, args.output_dir.expanduser().resolve()))


if __name__ == "__main__":
    main()
