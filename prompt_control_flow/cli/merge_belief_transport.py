from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from prompt_control_flow.belief_transport.artifact import merge_belief_trace_artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge disjoint belief-trace shards.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    artifact = merge_belief_trace_artifacts(args.inputs)
    output = Path(args.output)
    artifact.save(output, compressed=bool(args.compress))
    print(
        f"merged {len(args.inputs)} shards into {artifact.n_rows} rows at {output}"
    )


if __name__ == "__main__":
    main()
