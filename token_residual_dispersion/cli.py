"""Command line interface for a step-free residual dispersion audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .audit import analyze_trace, save_audit
from .data import iter_token_state_traces
from .metrics import DispersionConfig
from .selftest import run_selftest


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help=".npy state tensor or .npz extraction manifest")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/token_residual_dispersion"))
    parser.add_argument("--windows", type=_csv_ints, default=(4, 8, 16, 32))
    parser.add_argument("--min-tokens", type=int, default=3)
    parser.add_argument("--decay", type=float, default=0.0)
    parser.add_argument(
        "--rank-stride",
        type=int,
        default=1,
        help="compute effective rank every N tokens; pair dispersion remains dense",
    )
    parser.add_argument(
        "--layers", type=_csv_ints, help="required layer ids for direct state input lacking metadata"
    )
    parser.add_argument("--max-traces", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--snapshot-kind",
        choices=("raw_residual_stream", "unverified"),
        default="unverified",
        help="provenance assertion for direct state input",
    )
    parser.add_argument(
        "--allow-unverified-snapshots",
        action="store_true",
        help="exploratory pilot only; outputs cannot support block-write claims",
    )
    parser.add_argument(
        "--allow-sparse-depth-deltas",
        action="store_true",
        help="analyze h[target]-h[source] as multi-block interval deltas, never block writes",
    )
    parser.add_argument(
        "--legacy-sparse-pilot",
        action="store_true",
        help="convenience opt-in for old selected manifests: sparse depths + unverified provenance",
    )
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="validate input without writing outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        result = run_selftest()
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 2
    if args.input is None:
        raise SystemExit("--input is required unless --selftest is used")
    config = DispersionConfig(
        windows=args.windows,
        min_tokens=args.min_tokens,
        decay=args.decay,
        rank_stride=args.rank_stride,
    )
    layer_ids = None if args.layers is None else np.asarray(args.layers, dtype=np.int64)
    traces = iter_token_state_traces(
        args.input,
        layers=layer_ids,
        snapshot_kind=args.snapshot_kind,
        max_traces=args.max_traces,
    )
    raw_analyses = (
        analyze_trace(
            trace,
            config,
            allow_unverified_snapshots=(
                args.allow_unverified_snapshots or args.legacy_sparse_pilot
            ),
            allow_sparse_depth_deltas=(
                args.allow_sparse_depth_deltas or args.legacy_sparse_pilot
            ),
        )
        for trace in traces
    )

    def with_progress():
        for index, analysis in enumerate(raw_analyses, start=1):
            if args.progress_every > 0 and index % args.progress_every == 0:
                print(f"processed_traces={index}", file=sys.stderr, flush=True)
            yield analysis

    analyses = with_progress()
    if args.preflight:
        trace_count = sum(1 for _ in analyses)
        if trace_count == 0:
            raise ValueError("input contains no token-state traces")
        print(json.dumps({"status": "ok", "trace_count": trace_count}, indent=2))
        return 0
    summary_path = save_audit(analyses, args.output_dir, config)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
