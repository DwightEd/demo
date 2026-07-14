#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from prompt_control_flow.predictive_state_audit import (
    PredictiveStateAuditConfig,
    run_predictive_state_audit,
    write_predictive_state_outputs,
)
from prompt_control_flow.predictive_state_data import (
    ProjectionConfig,
    WindowConfig,
    inspect_predictive_state_source,
    load_predictive_state_dataset,
)
from prompt_control_flow.predictive_state_model import PredictiveModelConfig


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return value


def _positive_int_tuple(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not values or any(item <= 0 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("horizons must be unique positive integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit correct-only reduced-rank predictive state charts and test whether "
            "OOF future-window innovation detects same-problem incorrect responses."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="outputs/predictive_state/predictive_state_scores.npz",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/predictive_state/audit",
    )
    parser.add_argument("--vector_key", default="sv_vec_step_exp")
    parser.add_argument("--cloud_layers", default="all")
    parser.add_argument(
        "--label_policy",
        default="answer_format_ok",
        choices=["answer", "strict", "answer_format_ok"],
    )
    parser.add_argument("--projection_dim", type=int, default=96)
    parser.add_argument("--window_tokens", type=int, default=16)
    parser.add_argument("--window_stride", type=int, default=16)
    parser.add_argument("--max_skipped_tokens", type=int, default=4)
    parser.add_argument("--horizons", type=_positive_int_tuple, default=(1, 2))
    parser.add_argument("--context_windows", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--covariance_shrinkage", type=float, default=0.1)
    parser.add_argument("--tangent_variance", type=float, default=0.9)
    parser.add_argument("--min_token_count", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_batch_tokens", type=int, default=8192)
    parser.add_argument("--window_batch_size", type=int, default=4096)
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


def _number(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "nan"


def main() -> None:
    args = build_parser().parse_args()
    if args.preflight:
        schema = inspect_predictive_state_source(
            args.input,
            vector_key=args.vector_key,
            cloud_layers=args.cloud_layers,
            label_policy=args.label_policy,
            max_samples=args.max_samples,
        )
        print(json.dumps(schema, indent=2, ensure_ascii=False))
        if not schema["ready"]:
            raise SystemExit("no same-problem correct/error contrast remains after filtering")
        return

    device = _device(args.compute_device)
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    dataset = load_predictive_state_dataset(
        args.input,
        vector_key=args.vector_key,
        cloud_layers=args.cloud_layers,
        label_policy=args.label_policy,
        max_samples=args.max_samples,
    )
    report, packed = run_predictive_state_audit(
        dataset,
        ProjectionConfig(
            projection_dim=args.projection_dim,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            seed=args.seed,
            compute_device=device,
        ),
        WindowConfig(
            window_tokens=args.window_tokens,
            window_stride=args.window_stride,
            max_skipped_tokens=args.max_skipped_tokens,
            window_batch_size=args.window_batch_size,
            compute_device=device,
        ),
        PredictiveModelConfig(
            latent_dim=args.latent_dim,
            ridge=args.ridge,
            covariance_shrinkage=args.covariance_shrinkage,
            tangent_variance=args.tangent_variance,
        ),
        PredictiveStateAuditConfig(
            folds=args.folds,
            horizons=args.horizons,
            context_windows=args.context_windows,
            min_token_count=args.min_token_count,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            length_match_ratio=args.length_match_ratio,
            seed=args.seed,
        ),
    )
    report["meta"]["gpu_peak_mb"] = (
        float(torch.cuda.max_memory_allocated(torch.device(device)) / (1024**2))
        if device.startswith("cuda")
        else 0.0
    )
    paths = write_predictive_state_outputs(
        report,
        packed,
        output=args.output,
        output_dir=args.output_dir,
        render_plots=not args.no_plots,
    )

    print("\n===== predictive state geometry pilot =====")
    meta = report["meta"]
    print(
        f"samples {meta['samples']} | errors {meta['errors']} | correct {meta['correct']} | "
        f"contrastive problems {meta['contrastive_problems']} | layers {meta['cloud_layers']}"
    )
    selected = {
        "predictive.token_residual.mahalanobis_mean",
        "predictive.token_residual.mahalanobis_mean.length_residual",
        "null.shuffle.mahalanobis_mean",
        "null.same_problem_mismatch.mahalanobis_mean",
        "static.token_residual.mahalanobis_mean",
        "control.token_bigram_nll",
        "baseline.fixed_window_consensus",
    }
    for row in report["scores"]:
        if row["name"] not in selected:
            continue
        print(
            f"  {row['name']:66s} within {_number(row['same_problem_auroc'])} "
            f"CI {row['same_problem_ci95']} coverage {_number(row['coverage'])}"
        )
    print("AUROC deltas:")
    for name, row in report["auc_deltas"].items():
        print(f"  {name:56s} {_number(row['point'])} CI {row['ci95']}")
    gate = report["decision_gate"]
    print(f"decision gate: {'PASS' if gate['passes'] else 'FAIL'}")
    for name, passed in gate["checks"].items():
        print(f"  {name:48s} {passed}")
    print("outputs:")
    for name, path in paths.items():
        print(f"  {name:20s} {path}")


if __name__ == "__main__":
    main()
