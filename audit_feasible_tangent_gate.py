#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from prompt_control_flow.feasible_tangent import (
    FeasibleTangentConfig,
    FeasibleTangentValidationConfig,
    run_feasible_tangent_gate,
    write_feasible_tangent_report,
)
from prompt_control_flow.feasible_tangent.data import feasible_tangent_preflight
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
            "Test whether leave-one-response-out correct samples from the same "
            "problem define a low-rank feasible transition tangent, then test "
            "persistent normal escape without logits or a label classifier."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output_dir",
        default="outputs/feasible_tangent",
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
    parser.add_argument("--layer_batch_size", type=int, default=2)
    parser.add_argument("--phase_sigma", type=float, default=0.20)
    parser.add_argument("--causal_time_scale", type=float, default=4.0)
    parser.add_argument("--rank_energy", type=float, default=0.90)
    parser.add_argument("--max_rank", type=int, default=4)
    parser.add_argument("--min_donors", type=int, default=6)
    parser.add_argument("--max_donors", type=int, default=12)
    parser.add_argument("--wrong_problem_draws", type=int, default=3)
    parser.add_argument("--late_fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--min_rank_coverage", type=float, default=0.80)
    parser.add_argument("--min_score_coverage", type=float, default=0.80)
    parser.add_argument("--min_problem_count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--preflight", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_rank < 1:
        raise ValueError("--max_rank must be positive")
    if args.min_donors < args.max_rank + 2:
        raise ValueError("--min_donors must be at least --max_rank + 2")
    if args.max_donors < args.min_donors:
        raise ValueError("--max_donors must be >= --min_donors")
    if args.layer_batch_size < 1:
        raise ValueError("--layer_batch_size must be positive")
    if not 0.0 < args.rank_energy < 1.0:
        raise ValueError("--rank_energy must be in (0,1)")
    if not 0.0 < args.late_fraction <= 1.0:
        raise ValueError("--late_fraction must be in (0,1]")
    device = _device(args.device)
    tangent_cfg = FeasibleTangentConfig(
        device=device,
        batch_size=args.batch_size,
        layer_batch_size=args.layer_batch_size,
        phase_sigma=args.phase_sigma,
        causal_time_scale=args.causal_time_scale,
        rank_energy=args.rank_energy,
        max_rank=args.max_rank,
        min_donors=args.min_donors,
        max_donors=args.max_donors,
        wrong_problem_draws=args.wrong_problem_draws,
        late_fraction=args.late_fraction,
        random_seed=args.seed,
    )
    dataset = load_flow_trajectory_dataset(
        args.input,
        vector_key=args.vector_key,
        layers=args.layers,
        label_policy=args.label_policy,
        max_samples=args.max_samples,
    )
    preflight = feasible_tangent_preflight(dataset, tangent_cfg)
    preflight["effective_device"] = device
    print("===== feasible tangent preflight =====")
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.preflight:
        return

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.perf_counter()
    result = run_feasible_tangent_gate(dataset, tangent_cfg)
    result.metadata["wall_time_seconds"] = time.perf_counter() - started
    if device.startswith("cuda"):
        result.metadata["gpu_peak_memory_mb"] = float(
            torch.cuda.max_memory_allocated(torch.device(device)) / (1024**2)
        )
    output_dir = Path(args.output_dir)
    summary, paths = write_feasible_tangent_report(
        result,
        output_dir,
        FeasibleTangentValidationConfig(
            folds=args.folds,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            min_rank_coverage=args.min_rank_coverage,
            min_score_coverage=args.min_score_coverage,
            min_problem_count=args.min_problem_count,
            random_seed=args.seed,
        ),
    )
    print("\n===== same-problem feasible tangent gate =====")
    print(
        "geometry existence: "
        f"{summary['geometry_existence_gate']['pass']} | "
        "error escape: "
        f"{summary['error_escape_gate']['pass']}"
    )
    primary = summary["response_diagnostics"][
        summary["error_escape_gate"]["primary_score"]
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
