from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit logits-conditional geometry increments with grouped cross-fitting."
    )
    parser.add_argument(
        "--trace", required=True, help="Compact trace from extract_ocgpi_traces.py."
    )
    parser.add_argument(
        "--geometry", required=True, help="Existing hidden-state/mechanism NPZ."
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoints", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--history", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=4)
    parser.add_argument("--logistic_c", type=float, default=0.25)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--geometry_ridge_alpha", type=float, default=10.0)
    parser.add_argument("--adapter_l2", type=float, default=1.0)
    parser.add_argument("--chart_variance", type=float, default=0.95)
    parser.add_argument("--chart_max_dim", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--compute_device", default="cuda")
    parser.add_argument("--geometry_batch_size", type=int, default=32)
    parser.add_argument("--no_legacy_geometry", action="store_true")
    parser.add_argument("--allow_model_mismatch", action="store_true")
    parser.add_argument(
        "--label_policy",
        default="process_error",
        choices=("process_error", "final_answer"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    from prompt_control_flow.ocgpi.audit import OCGPIAuditConfig, run_ocgpi_audit
    from prompt_control_flow.ocgpi.models import CrossFitConfig

    crossfit = CrossFitConfig(
        outer_folds=int(args.outer_folds),
        inner_folds=int(args.inner_folds),
        logistic_c=float(args.logistic_c),
        ridge_alpha=float(args.ridge_alpha),
        geometry_ridge_alpha=float(args.geometry_ridge_alpha),
        adapter_l2=float(args.adapter_l2),
        chart_variance=float(args.chart_variance),
        chart_max_dim=int(args.chart_max_dim),
        seed=int(args.seed),
    )
    cfg = OCGPIAuditConfig(
        checkpoints=_floats(args.checkpoints),
        history=int(args.history),
        horizon=int(args.horizon),
        bootstrap=int(args.bootstrap),
        include_legacy_geometry=not bool(args.no_legacy_geometry),
        compute_device=str(args.compute_device),
        geometry_batch_size=int(args.geometry_batch_size),
        allow_model_mismatch=bool(args.allow_model_mismatch),
        label_policy=str(args.label_policy),
        crossfit=crossfit,
    )
    report = run_ocgpi_audit(
        trace_path=args.trace,
        geometry_path=args.geometry,
        output_dir=args.output_dir,
        cfg=cfg,
    )
    final_key = f"{max(cfg.checkpoints):.2f}"
    response = report["response_detection"][final_key]
    online = report["online_response_detection"]
    forecast = report["future_output_forecast"]
    print("===== OC-GPI audit =====")
    print(
        f"joined {report['preflight']['num_joined_chains']} | "
        f"errors {report['preflight']['num_errors']} | "
        f"geometry tier {report['preflight']['geometry']['tier']}"
    )
    print(
        "response: output AUROC "
        f"{response['output_only']['auroc']:.3f} | +geometry "
        f"{response['output_plus_geometry']['auroc']:.3f} | usable bits "
        f"{response['increment']['conditional_usable_information']['point_bits']:+.4f}"
    )
    print(
        "online prefixes: output AUROC "
        f"{online['output_only']['auroc']:.3f} | +geometry "
        f"{online['output_plus_geometry']['auroc']:.3f} | usable bits "
        f"{online['increment']['conditional_usable_information']['point_bits']:+.4f}"
    )
    print(
        "future logits: partial R2 "
        f"{forecast['increment']['partial_r2']['point']:+.4f} | vs null "
        f"{forecast['increment']['partial_r2_vs_null']['point']:+.4f}"
    )
    print(f"decision gate: {report['decision_gate']}")
    print(f"report: {Path(args.output_dir) / 'summary.md'}")


if __name__ == "__main__":
    main()
