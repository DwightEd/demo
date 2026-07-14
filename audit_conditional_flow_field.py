#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import torch

from prompt_control_flow.conditional_flow_field import (
    ConditionalFlowFieldConfig,
    ConditionalFlowFieldValidationConfig,
    conditional_flow_field_preflight,
    run_conditional_flow_field,
    write_conditional_flow_field_report,
)
from prompt_control_flow.flow_signature_data import load_flow_trajectory_dataset


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a same-problem, causal-phase-conditioned distribution of "
            "hidden-state update directions using a proper spherical energy score."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output_dir", default="outputs/conditional_flow_field"
    )
    parser.add_argument("--vector_key", default="auto")
    parser.add_argument("--layers", default="16", help="all or actual layer ids")
    parser.add_argument(
        "--label_policy",
        choices=("answer", "strict", "answer_format_ok", "processbench"),
        default="answer_format_ok",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_donors", type=int, default=6)
    parser.add_argument("--max_donors", type=int, default=11)
    parser.add_argument("--state_window", type=int, default=2)
    parser.add_argument("--wrong_problem_draws", type=int, default=3)
    parser.add_argument("--late_fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--free_energy_beta", type=float, default=2.0)
    parser.add_argument("--cusum_drift", type=float, default=0.5)
    parser.add_argument("--calibration_floor", type=float, default=2e-2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--min_score_coverage", type=float, default=0.80)
    parser.add_argument("--min_problem_count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--preflight", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.min_donors < 3:
        raise ValueError("--min_donors must be at least three")
    if args.max_donors < args.min_donors:
        raise ValueError("--max_donors must be >= --min_donors")
    if args.state_window < 0:
        raise ValueError("--state_window must be non-negative")
    if not 0.0 < args.late_fraction <= 1.0:
        raise ValueError("--late_fraction must be in (0,1]")
    device = _device(args.device)
    scoring = ConditionalFlowFieldConfig(
        device=device,
        batch_size=args.batch_size,
        min_donors=args.min_donors,
        max_donors=args.max_donors,
        state_window=args.state_window,
        wrong_problem_draws=args.wrong_problem_draws,
        late_fraction=args.late_fraction,
        free_energy_beta=args.free_energy_beta,
        cusum_drift=args.cusum_drift,
        calibration_floor=args.calibration_floor,
        random_seed=args.seed,
    )
    dataset = load_flow_trajectory_dataset(
        args.input,
        vector_key=args.vector_key,
        layers=args.layers,
        label_policy=args.label_policy,
        max_samples=args.max_samples,
    )
    preflight = conditional_flow_field_preflight(dataset, scoring)
    preflight["effective_device"] = device
    print("===== conditional flow field preflight =====")
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.preflight:
        return

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.perf_counter()
    result = run_conditional_flow_field(dataset, scoring)
    result.metadata["wall_time_seconds"] = time.perf_counter() - started
    if device.startswith("cuda"):
        result.metadata["gpu_peak_memory_mb"] = float(
            torch.cuda.max_memory_allocated(torch.device(device)) / (1024**2)
        )
    summary, paths = write_conditional_flow_field_report(
        result,
        args.output_dir,
        ConditionalFlowFieldValidationConfig(
            folds=args.folds,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            min_score_coverage=args.min_score_coverage,
            min_problem_count=args.min_problem_count,
            random_seed=args.seed,
        ),
    )
    print("\n===== conditional spherical feasible-flow field =====")
    print(
        "geometry existence: "
        f"{summary['geometry_existence_gate']['pass']} | "
        "error excursion: "
        f"{summary['error_excursion_gate']['pass']}"
    )
    primary = summary["response_diagnostics"][
        summary["error_excursion_gate"]["primary_score"]
    ]
    print(
        "primary length-residual within-problem AUROC "
        f"{primary['within_problem_auroc_equal_weight']:.3f} "
        f"CI {primary['within_problem_ci95']}"
    )
    print(f"decision: {summary['decision']['status']}")
    for name, path in paths.items():
        print(f"  {name:14s} {path}")


if __name__ == "__main__":
    main()
