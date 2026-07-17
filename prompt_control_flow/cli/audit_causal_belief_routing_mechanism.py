from __future__ import annotations

import argparse

from prompt_control_flow.causal_belief_routing.routing_audit import (
    RoutingAuditConfig,
    run_routing_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fit and audit evidence-specific OV belief-update routing."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top_heads", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    report = run_routing_audit(
        args.input,
        args.output_dir,
        RoutingAuditConfig(
            folds=int(args.folds),
            top_heads=int(args.top_heads),
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
        ),
    )
    print("===== causal belief routing: mechanism gate =====")
    print(
        f"rows {report['data']['rows']} | pairs {report['data']['pairs']} | "
        f"layers {report['data']['layers']}"
    )
    for name, metric in report["tests"].items():
        print(
            f"  {name:<40} {metric['point']:+.5f} "
            f"CI [{metric['ci_low']:+.5f}, {metric['ci_high']:+.5f}]"
        )
    gate = report["decision_gate"]
    print(
        f"routing supported: {gate['routing_supported']} | "
        f"ready for causal patching: {gate['ready_for_causal_patching']}"
    )
    print(f"report: {args.output_dir}/summary.md")


if __name__ == "__main__":
    main()
