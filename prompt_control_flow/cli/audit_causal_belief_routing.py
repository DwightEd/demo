from __future__ import annotations

import argparse

from prompt_control_flow.causal_belief_routing.audit import (
    RepresentationAuditConfig,
    run_representation_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether residual states preserve future-relevant finite-field beliefs "
            "that are absent from the matched current output distribution."
        )
    )
    parser.add_argument("--input", required=True, help="Causal belief trace NPZ")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--projection_dim", type=int, default=32)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max_current_alias_js", type=float, default=0.05)
    parser.add_argument("--min_future_accuracy_gain", type=float, default=0.10)
    parser.add_argument("--compute_device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_representation_audit(
        args.input,
        args.output_dir,
        RepresentationAuditConfig(
            folds=int(args.folds),
            projection_dim=int(args.projection_dim),
            ridge_alpha=float(args.ridge_alpha),
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
            max_current_alias_js=float(args.max_current_alias_js),
            min_future_accuracy_gain=float(args.min_future_accuracy_gain),
            compute_device=str(args.compute_device),
        ),
    )
    alias = report["alias_checks"]
    information = report["conditional_information"]
    gate = report["decision_gate"]
    print("===== causal belief routing: representation gate =====")
    print(
        f"pairs {report['data']['pairs']} | current rows {report['data']['current_rows']} "
        f"| layers {report['data']['layers']}"
    )
    print(
        f"current alias JS median {alias['model_current_js_median']:.6f} | "
        f"future accuracy {alias['model_future_accuracy']:.3f}"
    )
    for name, metric in information.items():
        print(
            f"  {name:<24} {metric['point']:+.4f} bits "
            f"CI [{metric['ci_low']:+.4f}, {metric['ci_high']:+.4f}]"
        )
    print(
        "representation supported: "
        f"{gate['representation_supported']} | "
        f"ready for routing analysis: {gate['ready_for_routing_analysis']}"
    )
    print(f"report: {args.output_dir}/summary.md")


if __name__ == "__main__":
    main()
