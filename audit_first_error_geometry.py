#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from prompt_control_flow.first_error_geometry import (
    FirstErrorGeometryConfig,
    load_step_geometry_dataset,
    load_token_axis,
    make_step_axis,
    map_matches_to_axis,
    match_correct_pseudo_events,
    run_first_error_geometry_audit,
)
from prompt_control_flow.first_error_geometry_report import write_geometry_audit_report


def _resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return value


def _parse_offsets(value: str) -> tuple[int, ...]:
    try:
        offsets = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("offsets must be comma-separated integers") from exc
    if not offsets:
        raise argparse.ArgumentTypeError("at least one offset is required")
    if 0 not in offsets:
        raise argparse.ArgumentTypeError("offsets must include 0")
    return offsets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether hidden-state velocity, turning angle, and Menger curvature "
            "change around ProcessBench first-error events."
        )
    )
    parser.add_argument("--input", required=True, help="Canonical full_*.npz feature artifact")
    parser.add_argument(
        "--output_dir",
        default="outputs/first_error_geometry",
        help="Each requested axis is written to a step/ or token/ child directory",
    )
    parser.add_argument("--modes", default="step", help="step, token, or step,token")
    parser.add_argument("--vector_key", default="auto")
    parser.add_argument("--step_layers", default="all", help="all or actual layer ids")
    parser.add_argument("--token_layers", default="all", help="all or hidden-shard layer ids")
    parser.add_argument("--hidden_dir", default=None, help="Override the hidden shard directory")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--nuisance_folds", type=int, default=5)
    parser.add_argument("--nuisance_ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--step_offsets",
        type=_parse_offsets,
        default=(-2, -1, 0, 1, 2),
        help="Comma-separated offsets; pass as --step_offsets=-2,-1,0,1,2",
    )
    parser.add_argument("--token_radius", type=int, default=16)
    parser.add_argument("--same_problem_bonus", type=float, default=25.0)
    parser.add_argument("--min_pair_coverage", type=float, default=0.80)
    parser.add_argument("--no_plots", action="store_true", help="Skip PNG rendering")
    parser.add_argument("--preflight", action="store_true")
    return parser


def _parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip()))
    invalid = sorted(set(modes) - {"step", "token"})
    if not modes or invalid:
        raise ValueError(f"--modes must contain step and/or token; invalid={invalid}")
    return modes


def _headline(result, min_coverage: float) -> list[dict]:
    rows = [
        row
        for row in result.event_rows
        if row["variant"] == "nuisance_residual"
        and row["offset"] == 0
        and row["pair_coverage"] >= min_coverage
        and np.isfinite(row["matched_event_auroc"])
    ]
    return sorted(
        rows,
        key=lambda row: abs(float(row["matched_event_auroc"]) - 0.5),
        reverse=True,
    )[:8]


def _print_result(result, paths: dict[str, str], min_coverage: float) -> None:
    print(f"\n===== first-error geometry | {result.axis.axis_kind} =====")
    print(
        f"trajectories {result.axis.n_samples} | errors {np.sum(result.axis.event_indices >= 0)} | "
        f"matches {len(result.matches)} | layers {result.axis.layer_ids.tolist()}"
    )
    print("offset-0 nuisance-residualized matched effects:")
    for row in _headline(result, min_coverage):
        print(
            f"  {row['metric']:25s} layer {row['layer']:>3d} "
            f"AUC {row['matched_event_auroc']:.3f} dz {row['paired_effect_dz']:+.3f} "
            f"coverage {row['pair_coverage']:.3f} q {row['bh_q']:.3g}"
        )
    print(f"report: {paths['summary_md']}")
    print(f"event data: {paths['event_csv']}")


def main() -> None:
    args = build_parser().parse_args()
    modes = _parse_modes(args.modes)
    if args.token_radius < 1:
        raise ValueError("--token_radius must be positive")
    if not 0.0 < args.min_pair_coverage <= 1.0:
        raise ValueError("--min_pair_coverage must be in (0,1]")
    device = _resolve_device(args.device)
    source = load_step_geometry_dataset(
        args.input,
        vector_key=args.vector_key,
        layers=args.step_layers,
        max_samples=args.max_samples,
    )
    matches = match_correct_pseudo_events(
        source,
        same_problem_bonus=args.same_problem_bonus,
    )
    if not matches:
        raise RuntimeError("no error/correct event matches could be constructed")

    preflight = {
        "path": source.source_path,
        "vector_key": source.vector_key,
        "samples": source.n_samples,
        "errors": int(np.sum(source.gold_error_step >= 0)),
        "correct": int(np.sum(source.gold_error_step < 0)),
        "step_layers": source.layer_ids.tolist(),
        "hidden_layers": source.metadata.get("hidden_layers", []),
        "has_hidden_shards": source.metadata.get("has_hidden_shards", False),
        "matches": len(matches),
        "same_problem_matches": int(np.sum([item.same_problem for item in matches])),
        "reused_controls": int(np.sum([item.reused_control for item in matches])),
        "skipped": source.skipped,
        "requested_modes": list(modes),
        "effective_device": device,
    }
    print("===== first-error geometry preflight =====")
    print(json.dumps(preflight, indent=2, ensure_ascii=False))
    if args.preflight:
        return

    token_offsets = tuple(range(-int(args.token_radius), int(args.token_radius) + 1))
    cfg = FirstErrorGeometryConfig(
        device=device,
        batch_size=args.batch_size,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        nuisance_folds=args.nuisance_folds,
        nuisance_ridge=args.nuisance_ridge,
        random_seed=args.seed,
        step_offsets=tuple(args.step_offsets),
        token_offsets=token_offsets,
    )
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    for mode in modes:
        if mode == "step":
            axis = make_step_axis(source)
        else:
            axis = load_token_axis(
                source,
                hidden_dir=args.hidden_dir,
                layers=args.token_layers,
            )
        axis_matches = map_matches_to_axis(matches, source, axis)
        if not axis_matches:
            raise RuntimeError(f"no matched events remain on the {mode} axis")
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        started = time.perf_counter()
        result = run_first_error_geometry_audit(axis, axis_matches, cfg)
        result.metadata["runtime_seconds"] = float(time.perf_counter() - started)
        result.metadata["gpu_peak_mb"] = (
            float(torch.cuda.max_memory_allocated(torch.device(device)) / (1024**2))
            if device.startswith("cuda")
            else 0.0
        )
        paths = write_geometry_audit_report(
            result,
            output_root / mode,
            min_coverage=args.min_pair_coverage,
            render_plots=not args.no_plots,
        )
        _print_result(result, paths, args.min_pair_coverage)


if __name__ == "__main__":
    main()
