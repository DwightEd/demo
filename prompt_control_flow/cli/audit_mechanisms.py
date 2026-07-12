from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from prompt_control_flow.evaluate import evaluate_all, load_metric_npz, save_json
from prompt_control_flow.reports import render_markdown, write_step_csv


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Audit residual-flow mechanism metrics.")
    ap.add_argument("--input", required=True, help="Metric npz from extract_mechanisms.py")
    ap.add_argument("--output_dir", required=True)
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = load_metric_npz(args.input)
    summary = evaluate_all(metrics)
    save_json(summary, out / "summary.json")
    (out / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    write_step_csv(metrics, out / "step_scores.csv")
    print(render_markdown(summary))
    print(f"\nSaved mechanism audit to {out}")


if __name__ == "__main__":
    main()
