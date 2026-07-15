from __future__ import annotations

import argparse
import json
from typing import Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory remote research artifacts without loading tensor payloads."
    )
    parser.add_argument("--roots", nargs="+", default=("data", "outputs"))
    parser.add_argument("--output_dir", default="outputs/artifact_inventory")
    parser.add_argument("--data_only", action="store_true")
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Hash every file. This can be slow for multi-GiB artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    from prompt_control_flow.artifact_inventory import inventory_artifacts, write_inventory

    rows = inventory_artifacts(
        args.roots,
        include_reports=not args.data_only,
        compute_hash=bool(args.sha256),
    )
    paths = write_inventory(rows, args.output_dir)
    print(
        json.dumps(
            {
                "files": len(rows),
                "total_gib": sum(row["bytes"] for row in rows) / (1024**3),
                "outputs": paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

