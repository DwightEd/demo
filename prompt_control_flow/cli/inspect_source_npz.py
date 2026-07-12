from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from prompt_control_flow.evaluate import finite_json
from prompt_control_flow.schema import inspect_npz_schema


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect whether an existing npz supports prompt-control-flow analysis.")
    p.add_argument("input", help="Source or metric npz.")
    p.add_argument("--output", default="", help="Optional JSON output path.")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    info = inspect_npz_schema(args.input)
    text = json.dumps(finite_json(info), ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
