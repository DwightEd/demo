#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from prompt_control_flow.conditional_tangent import (
    ConditionalTangentConfig,
    load_conditional_tangent_dataset,
    run_conditional_tangent_audit,
)
from prompt_control_flow.conditional_tangent_report import (
    ConditionalTangentValidationConfig,
    write_validation_report,
)


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return value


def _offsets(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(
            sorted({int(item.strip()) for item in value.split(",") if item.strip()})
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "event offsets must be comma-separated integers"
        ) from exc
    if not parsed or 0 not in parsed:
        raise argparse.ArgumentTypeError("event offsets must include 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fit a question/phase-conditioned feasible transition space, "
            "measure persistent normal escape, and audit older direction/spectrum "
            "signals under explicit length controls."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Canonical data/features/full_*.npz with stepvec, qvec, and labels",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/conditional_tangent_escape",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Compact score NPZ; defaults to OUTPUT_DIR/conditional_tangent_scores.npz",
    )
    parser.add_argument("--vector_key", default="auto")
    parser.add_argument("--layers", default="all", help="all or actual layer ids")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--search_multiplier", type=int, default=4)
    parser.add_argument("--tangent_rank", type=int, default=6)
    parser.add_argument("--q_temperature", type=float, default=0.10)
    parser.add_argument("--phase_sigma", type=float, default=0.20)
    parser.add_argument(
        "--phase_mode",
        choices=("causal_step", "normalized_chain"),
        default="causal_step",
        help=(
            "causal_step uses only the current step index. normalized_chain "
            "uses final chain length and is an offline ablation only."
        ),
    )
    parser.add_argument("--causal_time_scale", type=float, default=4.0)
    parser.add_argument("--persistence_window", type=int, default=3)
    parser.add_argument(
        "--reference_policy",
        choices=("correct_only", "correct_plus_pre_error"),
        default="correct_only",
        help=(
            "Primary protocol is correct_only. The prefix-inclusive policy is an "
            "explicit contamination/precursor ablation."
        ),
    )
    parser.add_argument("--global_reference_cap", type=int, default=512)
    parser.add_argument("--nuisance_ridge", type=float, default=1e-3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--length_bins", type=int, default=5)
    parser.add_argument(
        "--event_offsets",
        type=_offsets,
        default=(-2, -1, 0, 1, 2),
        help="Pass as --event_offsets=-2,-1,0,1,2",
    )
    parser.add_argument("--min_coverage", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output_cotangent_key",
        default="auto",
        help=(
            "auto, none, or a record-aligned [step,layer,hidden] NPZ key. "
            "Missing cotangents leave the output-sensitivity gate untested."
        ),
    )
    parser.add_argument("--no_legacy_directional", action="store_true")
    parser.add_argument("--store_normal_vectors", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser


def _preflight(dataset, device: str) -> dict[str, object]:
    source = dataset.source
    return {
        "path": source.source_path,
        "vector_key": source.vector_key,
        "chains": source.n_samples,
        "errors": int(np.sum(source.gold_error_step >= 0)),
        "correct": int(np.sum(source.gold_error_step < 0)),
        "problem_groups": int(np.unique(source.problem_ids).size),
        "layers": source.layer_ids.tolist(),
        "hidden_dim": int(source.hidden_dim),
        "qvec_layers": dataset.metadata.get("qvec_layers"),
        "response_cloud_available": dataset.metadata.get(
            "response_cloud_available", False
        ),
        "stored_spread_available": dataset.metadata.get(
            "stored_spread_available", False
        ),
        "output_coupling_available": dataset.metadata.get(
            "output_coupling_available", False
        ),
        "output_cotangent_key": dataset.output_cotangent_key,
        "output_cotangent_kind": dataset.output_cotangent_kind,
        "effective_device": device,
        "skipped": source.skipped,
    }


def _print_headline(summary: dict[str, object], paths: dict[str, str]) -> None:
    gates = summary["hypothesis_gates"]
    print("\n===== conditional tangent escape =====")
    for name, gate in gates.items():
        print(
            f"{name:28s} pass={gate.get('pass')} "
            f"status={gate.get('status', 'tested')}"
        )
    print(f"report: {paths['summary_md']}")
    print(f"event curves: {paths['event_csv']}")
    print(f"response table: {paths['response_csv']}")


def main() -> None:
    args = build_parser().parse_args()
    if args.neighbors < args.tangent_rank + 2:
        raise ValueError("--neighbors must be at least --tangent_rank + 2")
    if args.persistence_window < 1:
        raise ValueError("--persistence_window must be positive")
    if args.causal_time_scale <= 0:
        raise ValueError("--causal_time_scale must be positive")
    if not 0.0 < args.min_coverage <= 1.0:
        raise ValueError("--min_coverage must be in (0,1]")
    device = _device(args.device)
    dataset = load_conditional_tangent_dataset(
        args.input,
        vector_key=args.vector_key,
        layers=args.layers,
        max_samples=args.max_samples,
        output_cotangent_key=args.output_cotangent_key,
    )
    preflight = _preflight(dataset, device)
    print("===== conditional tangent preflight =====")
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.preflight:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = (
        Path(args.output)
        if args.output
        else output_dir / "conditional_tangent_scores.npz"
    )
    started = time.perf_counter()
    result = run_conditional_tangent_audit(
        dataset,
        ConditionalTangentConfig(
            device=device,
            batch_size=args.batch_size,
            folds=args.folds,
            neighbors=args.neighbors,
            search_multiplier=args.search_multiplier,
            tangent_rank=args.tangent_rank,
            q_temperature=args.q_temperature,
            phase_sigma=args.phase_sigma,
            phase_mode=args.phase_mode,
            causal_time_scale=args.causal_time_scale,
            persistence_window=args.persistence_window,
            reference_policy=args.reference_policy,
            global_reference_cap=args.global_reference_cap,
            nuisance_ridge=args.nuisance_ridge,
            random_seed=args.seed,
        ),
        include_legacy_directional=not args.no_legacy_directional,
    )
    result.metadata["audit_wall_time_seconds"] = time.perf_counter() - started
    summary, paths = write_validation_report(
        result,
        output_dir,
        ConditionalTangentValidationConfig(
            folds=args.folds,
            bootstrap=args.bootstrap,
            length_bins=args.length_bins,
            event_offsets=args.event_offsets,
            min_coverage=args.min_coverage,
            random_seed=args.seed,
        ),
        render_plots=not args.no_plots,
        score_output=output,
        include_normal_vectors=args.store_normal_vectors,
    )
    _print_headline(summary, paths)


if __name__ == "__main__":
    main()
