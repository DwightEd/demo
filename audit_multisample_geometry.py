#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from prompt_control_flow.flow_signature_data import inspect_flow_source, load_flow_trajectory_dataset
from prompt_control_flow.multisample_geometry import (
    MultisampleGeometryConfig,
    run_multisample_geometry_audit,
    write_multisample_geometry_outputs,
)


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate local hidden-state geometry as a response-level same-problem "
            "trajectory diagnostic. This command does not claim first-error localization."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/multisample_geometry/geometry_scores.npz")
    parser.add_argument("--output_dir", default="outputs/multisample_geometry/audit")
    parser.add_argument("--vector_key", default="auto")
    parser.add_argument("--layers", default="all", help="all, mid, or comma-separated actual layer ids")
    parser.add_argument(
        "--label_policy",
        default="answer_format_ok",
        choices=["answer", "strict", "answer_format_ok"],
    )
    parser.add_argument("--phase_points", type=int, default=16)
    parser.add_argument("--late_fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--compute_device", default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--min_correct_support", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--keep_profiles", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    return parser


def _preflight(args: argparse.Namespace) -> None:
    schema = inspect_flow_source(args.input, args.vector_key)
    dataset = load_flow_trajectory_dataset(
        args.input,
        vector_key=args.vector_key,
        layers=args.layers,
        label_policy=args.label_policy,
        max_samples=args.max_samples,
    )
    contrastive = 0
    for problem in np.unique(dataset.problem_ids):
        labels = dataset.y_error[dataset.problem_ids == problem]
        contrastive += int(np.any(labels == 0) and np.any(labels == 1))
    schema.update(
        {
            "kept_samples": dataset.n_samples,
            "kept_errors": int(np.sum(dataset.y_error == 1)),
            "kept_correct": int(np.sum(dataset.y_error == 0)),
            "kept_problems": int(np.unique(dataset.problem_ids).size),
            "contrastive_problems": int(contrastive),
            "label_policy": dataset.label_policy,
            "selected_layers": dataset.layer_ids.tolist(),
            "skipped": dataset.skipped,
            "ready": bool(contrastive > 0),
        }
    )
    print(json.dumps(schema, indent=2, ensure_ascii=False))
    if contrastive == 0:
        raise SystemExit("no same-problem correct/error contrast remains after filtering")


def main() -> None:
    args = build_parser().parse_args()
    if args.preflight:
        _preflight(args)
        return
    device = _device(args.compute_device)
    dataset = load_flow_trajectory_dataset(
        args.input,
        vector_key=args.vector_key,
        layers=args.layers,
        label_policy=args.label_policy,
        max_samples=args.max_samples,
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    report, packed = run_multisample_geometry_audit(
        dataset,
        MultisampleGeometryConfig(
            phase_points=args.phase_points,
            late_fraction=args.late_fraction,
            batch_size=args.batch_size,
            compute_device=device,
            folds=args.folds,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            min_correct_support=args.min_correct_support,
            seed=args.seed,
        ),
    )
    report["meta"]["gpu_peak_mb"] = (
        float(torch.cuda.max_memory_allocated(torch.device(device)) / (1024**2))
        if device.startswith("cuda")
        else 0.0
    )
    paths = write_multisample_geometry_outputs(
        report,
        packed,
        output=args.output,
        output_dir=args.output_dir,
        keep_profiles=args.keep_profiles,
        render_plots=not args.no_plots,
    )
    print("\n===== same-problem multisample geometry =====")
    meta = report["meta"]
    print(
        f"samples {meta['samples']} | errors {meta['errors']} | correct {meta['correct']} | "
        f"contrastive problems {meta['contrastive_problems']} | layers {meta['layers']}"
    )
    for row in report["headline_scores"]:
        print(
            f"  {row['name']:52s} within {row['same_problem_auroc']:.3f} "
            f"cross {row['cross_problem_auroc']:.3f} coverage {row['coverage']:.3f}"
        )
    delta = report["dynamic_minus_static"]
    residual_delta = report["dynamic_minus_static_length_residual"]
    print(f"dynamic-static within delta {delta['point']:+.3f} CI {delta['ci95']}")
    print(f"length-residual dynamic-static delta {residual_delta['point']:+.3f} CI {residual_delta['ci95']}")
    print("outputs:")
    for name, path in paths.items():
        print(f"  {name:16s} {path}")


if __name__ == "__main__":
    main()
