from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prompt_control_flow.evaluate import load_metric_npz, save_json
from prompt_control_flow.layer_time_evaluate import (
    LayerTimeValidationConfig,
    evaluate_layer_time_geometry,
    render_layer_time_validation,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Run claim-driven event, nuisance-adjusted, same-problem, and "
            "transport-reliability validation for a layer-time geometry field."
        )
    )
    ap.add_argument("--input", required=True, help="NPZ produced by audit_layer_time_geometry.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--offsets", default="-2,-1,0,1,2")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--layer_reduction", choices=("median", "mean"), default="median")
    return ap


def parse_offsets(value: str) -> tuple[int, ...]:
    offsets = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if 0 not in offsets or -1 not in offsets or 1 not in offsets:
        raise ValueError("offsets must include -1, 0, and 1 for the local peak contrast")
    return offsets


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    metrics = load_metric_npz(args.input)
    cfg = LayerTimeValidationConfig(
        event_offsets=parse_offsets(args.offsets),
        bootstrap=max(0, int(args.bootstrap)),
        random_seed=int(args.seed),
        layer_reduction=str(args.layer_reduction),
    )
    summary = evaluate_layer_time_geometry(metrics, cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_dir / "layer_time_validation.json")
    markdown = render_layer_time_validation(summary)
    (output_dir / "layer_time_validation.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved claim-driven validation to {output_dir}")


if __name__ == "__main__":
    main()
