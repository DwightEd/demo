from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from prompt_control_flow.data_contract import inspect_reasoning_artifact


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit ProcessBench sources and reasoning NPZ artifacts for label, "
            "prompt, token-axis, hidden-state, and same-problem capabilities."
        )
    )
    parser.add_argument(
        "inputs", nargs="+", help="One or more ProcessBench .json/.jsonl or trace .npz paths."
    )
    parser.add_argument("--output", default="", help="Optional JSON report path.")
    parser.add_argument(
        "--strict_source",
        action="store_true",
        help="Exit non-zero when a ProcessBench source has malformed or incomplete rows.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    reports = [inspect_reasoning_artifact(path) for path in args.inputs]
    payload = {
        "artifacts": reports,
        "all_inputs_found": all(report.get("kind") != "missing" for report in reports),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if not payload["all_inputs_found"]:
        raise SystemExit(2)
    if args.strict_source:
        invalid = [
            report
            for report in reports
            if report.get("kind") == "processbench_source" and not report.get("ready")
        ]
        if invalid:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
