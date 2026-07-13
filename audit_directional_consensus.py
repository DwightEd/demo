#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from prompt_control_flow.directional_consensus import (
    DirectionalConsensusAuditConfig,
    DirectionalConsensusConfig,
    inspect_directional_cloud_source,
    load_directional_cloud_dataset,
    run_directional_consensus_audit,
    write_directional_consensus_outputs,
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
            "Test whether same-problem incorrect responses have lower token-direction "
            "consensus after removing the exact finite-token self-pair bias."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="outputs/directional_consensus/consensus_scores.npz",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/directional_consensus/audit",
    )
    parser.add_argument("--vector_key", default="auto")
    parser.add_argument(
        "--cloud_layers",
        default="all",
        help="all, mid, or comma-separated actual token-cloud layer ids",
    )
    parser.add_argument(
        "--label_policy",
        default="answer_format_ok",
        choices=["answer", "strict", "answer_format_ok"],
    )
    parser.add_argument("--late_fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--fixed_window_tokens", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=8192,
        help="maximum token-layer states per GPU batch",
    )
    parser.add_argument("--compute_device", default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--length_match_ratio", type=float, default=1.25)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    return parser


def _preflight(args: argparse.Namespace) -> None:
    schema = inspect_directional_cloud_source(
        args.input,
        vector_key=args.vector_key,
        cloud_layers=args.cloud_layers,
        label_policy=args.label_policy,
        max_samples=args.max_samples,
    )
    print(json.dumps(schema, indent=2, ensure_ascii=False))
    if not schema["ready"]:
        raise SystemExit("no same-problem correct/error contrast remains after filtering")


def _number(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "nan"


def main() -> None:
    args = build_parser().parse_args()
    if args.preflight:
        _preflight(args)
        return

    device = _device(args.compute_device)
    dataset = load_directional_cloud_dataset(
        args.input,
        vector_key=args.vector_key,
        cloud_layers=args.cloud_layers,
        label_policy=args.label_policy,
        max_samples=args.max_samples,
    )
    compute_cfg = DirectionalConsensusConfig(
        late_fraction=args.late_fraction,
        fixed_window_tokens=args.fixed_window_tokens,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        compute_device=device,
    )
    audit_cfg = DirectionalConsensusAuditConfig(
        folds=args.folds,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        length_match_ratio=args.length_match_ratio,
        seed=args.seed,
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    report, packed = run_directional_consensus_audit(dataset, compute_cfg, audit_cfg)
    report["meta"]["gpu_peak_mb"] = (
        float(torch.cuda.max_memory_allocated(torch.device(device)) / (1024**2))
        if device.startswith("cuda")
        else 0.0
    )
    paths = write_directional_consensus_outputs(
        report,
        packed,
        output=args.output,
        output_dir=args.output_dir,
        render_plots=not args.no_plots,
    )

    print("\n===== debiased directional consensus =====")
    meta = report["meta"]
    print(
        f"samples {meta['samples']} | errors {meta['errors']} | "
        f"correct {meta['correct']} | contrastive problems "
        f"{meta['contrastive_problems']} | cloud layers {meta['cloud_layers']}"
    )
    for row in report["scores"]:
        if not row["confirmatory"]:
            continue
        print(
            f"  {row['name']:62s} within {_number(row['same_problem_auroc'])} "
            f"CI {row['same_problem_ci95']} q {_number(row['same_problem_bh_q'])} "
            f"token-match {_number(row['token_length_matched_auroc'])}"
        )
    gate = report["decision_gate"]
    print(
        f"decision gate: {'PASS' if gate['passes'] else 'FAIL'} | "
        f"primary {gate['primary_score']}"
    )
    print("outputs:")
    for name, path in paths.items():
        print(f"  {name:20s} {path}")


if __name__ == "__main__":
    main()
