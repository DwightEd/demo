from __future__ import annotations

import argparse
from typing import Sequence

from prompt_control_flow.evaluate import load_metric_npz
from prompt_control_flow.visualize import DEFAULT_STEP_METRICS, make_plots


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize residual-flow mechanism separability and trajectories.")
    p.add_argument("--input", required=True, help="Metric npz from extract_mechanisms.py")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--metrics", default=",".join(DEFAULT_STEP_METRICS), help="Comma-separated step metrics to visualize.")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    metrics = load_metric_npz(args.input)
    names = tuple(x.strip() for x in args.metrics.split(",") if x.strip())
    make_plots(metrics, args.output_dir, metric_names=names)
    print(f"Wrote mechanism visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
