from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .config import RunConfig, SourceConfig
from .raw_residual import inspect_raw_residual_source
from .progress import NullProgress, ProgressReporter, TqdmProgress
from .runner import ExperimentRunner


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
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Compatibility facade around the typed application service."""
    source = SourceConfig(
        manifest=Path(input_path),
        hidden_dir=None if hidden_dir is None else Path(hidden_dir),
        response_generator=response_generator,
    )
    config = RunConfig(
        offsets=tuple(offsets),
        layers=layers if isinstance(layers, str) else tuple(layers),
        max_pairs=max_pairs,
        rank=rank,
        folds=n_splits,
        bootstrap=n_boot,
        seed=seed,
        ridge_alpha=ridge_alpha,
    )
    runner = ExperimentRunner(
        source,
        config,
        output_dir,
        progress=progress or NullProgress(),
    )
    return runner.run().to_dict()


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
        progress=TqdmProgress(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
