#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from prompt_control_flow.flow_signature_audit import (
    FlowAuditConfig,
    make_order_sensitive_synthetic_dataset,
    run_flow_signature_audit,
    write_flow_signature_outputs,
)
from prompt_control_flow.flow_signature_data import (
    inspect_flow_source,
    load_flow_trajectory_dataset,
)
from prompt_control_flow.flow_signatures import FlowSignatureConfig, encode_reasoning_flows


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether same-problem correct hidden-state flows concentrate in an "
            "ordered order-two log-signature support set."
        )
    )
    parser.add_argument("--input", default=None, help="Canonical full_*.npz or multisample npz with raw step vectors.")
    parser.add_argument("--output", default="outputs/reasoning_flow_signatures/flow_signature_scores.npz")
    parser.add_argument("--output_dir", default="outputs/reasoning_flow_signatures/audit")
    parser.add_argument("--vector_key", default="auto")
    parser.add_argument("--layers", default="all", help="all, mid, or comma-separated actual layer ids")
    parser.add_argument(
        "--label_policy",
        default="answer_format_ok",
        choices=["answer", "strict", "answer_format_ok", "processbench"],
    )
    parser.add_argument("--projection_dim", type=int, default=8)
    parser.add_argument("--phase_points", type=int, default=16)
    parser.add_argument("--progress_weight", type=float, default=1.0)
    parser.add_argument("--state_normalization", default="none", choices=["none", "l2"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--compute_device", default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--min_correct_support", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--assert_gates", action="store_true")
    return parser


def print_report(report: dict) -> None:
    meta = report["meta"]
    headline = report["headline"]
    print("\n===== Conditional ordered reasoning-flow signatures =====")
    print(
        f"samples {meta['n_samples']} | errors {meta['n_error']} | "
        f"problems {meta['n_problems']} | layers {meta['layers']}"
    )
    print(
        f"primary {headline['primary_score']} | {headline['primary_auc_kind']} "
        f"{headline['primary_auc']:.4f}"
    )
    order = headline["order2_minus_order1"]
    chronology = headline["chronological_minus_shuffled"]
    print(f"order2 - order1: {order['point']:+.4f} CI {order['ci95']}")
    print(f"chronological - shuffled: {chronology['point']:+.4f} CI {chronology['ci95']}")
    print(f"length-residualized: {headline['length_residual_auc']:.4f}")
    print("gates:")
    for name, passed in report["hypothesis_gates"].items():
        print(f"  {name:36s} {passed}")
    print("top scores:")
    rows = sorted(
        report["scores"].items(),
        key=lambda item: np.nan_to_num(
            item[1]["same_problem_auroc"],
            nan=item[1]["cross_problem_auroc"],
        ),
        reverse=True,
    )
    for name, row in rows[:14]:
        print(
            f"  {name:45s} cross {row['cross_problem_auroc']:.3f} "
            f"within {row['same_problem_auroc']:.3f} pairs={row['same_problem_pairs']}"
        )


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.compute_device)
    if args.preflight:
        if not args.input:
            raise SystemExit("--preflight requires --input")
        print(json.dumps(inspect_flow_source(args.input, args.vector_key), indent=2, ensure_ascii=False))
        return

    if args.selftest:
        dataset = make_order_sensitive_synthetic_dataset(seed=args.seed)
    else:
        if not args.input:
            raise SystemExit("provide --input or --selftest")
        dataset = load_flow_trajectory_dataset(
            args.input,
            vector_key=args.vector_key,
            layers=args.layers,
            label_policy=args.label_policy,
            max_samples=args.max_samples,
        )

    signature_cfg = FlowSignatureConfig(
        projection_dim=args.projection_dim,
        phase_points=args.phase_points,
        progress_weight=args.progress_weight,
        state_normalization=args.state_normalization,
        seed=args.seed,
    )
    audit_cfg = FlowAuditConfig(
        folds=args.folds,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        min_correct_support=args.min_correct_support,
        seed=args.seed,
        compute_device=device,
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.perf_counter()
    chronological, shuffled = encode_reasoning_flows(
        dataset.trajectories,
        signature_cfg,
        device=device,
        batch_size=args.batch_size,
        include_shuffled=True,
    )
    report, packed = run_flow_signature_audit(dataset, chronological, shuffled, audit_cfg)
    report["runtime"] = {
        "seconds": time.perf_counter() - started,
        "device": device,
        "gpu_peak_mb": (
            float(torch.cuda.max_memory_allocated(torch.device(device)) / (1024**2))
            if device.startswith("cuda")
            else 0.0
        ),
    }
    output, json_path, markdown_path = write_flow_signature_outputs(
        report,
        packed,
        output=args.output,
        output_dir=args.output_dir,
    )
    print_report(report)
    print(f"\nsaved: {output}")
    print(f"report: {markdown_path}")
    print(f"json: {json_path}")
    if args.assert_gates:
        failed = [name for name, passed in report["hypothesis_gates"].items() if not passed]
        if failed:
            raise SystemExit(f"claim gates failed: {failed}")


if __name__ == "__main__":
    main()
