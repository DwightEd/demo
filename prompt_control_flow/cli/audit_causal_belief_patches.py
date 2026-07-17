from __future__ import annotations

import argparse

from prompt_control_flow.causal_belief_routing.patch_audit import (
    SourcePatchAuditConfig,
    run_source_patch_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit source-specific causal patches.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--min_coverage", type=float, default=0.80)
    args = parser.parse_args()
    report = run_source_patch_audit(
        args.input,
        args.output_dir,
        SourcePatchAuditConfig(
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
            min_coverage=float(args.min_coverage),
        ),
    )
    print("===== causal belief routing: source-patch gate =====")
    print(
        f"directions {report['data']['directions']} | pairs {report['data']['pairs']} | "
        f"coverage {report['data']['coverage']:.3f}"
    )
    for name, metric in report["tests"].items():
        print(
            f"  {name:<48} {metric['point']:+.5f} "
            f"CI [{metric['ci_low']:+.5f}, {metric['ci_high']:+.5f}]"
        )
    print(
        "causal routing supported: "
        f"{report['decision_gate']['causal_routing_supported']}"
    )
    print(f"report: {args.output_dir}/summary.md")


if __name__ == "__main__":
    main()
