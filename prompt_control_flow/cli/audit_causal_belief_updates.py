from __future__ import annotations

import argparse

from prompt_control_flow.causal_belief_update_decomposition.update_audit import (
    BeliefUpdateAuditConfig,
    run_belief_update_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit attention and MLP residual writes in cross-fitted analytic "
            "belief-update coordinates."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--primary_layer", type=int, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--max_reconstruction_p95", type=float, default=1e-4)
    parser.add_argument("--max_state_replay_p95", type=float, default=1e-3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_belief_update_audit(
        args.input,
        args.output_dir,
        BeliefUpdateAuditConfig(
            primary_layer=int(args.primary_layer),
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
            max_reconstruction_p95=float(args.max_reconstruction_p95),
            max_state_replay_p95=float(args.max_state_replay_p95),
        ),
    )
    gate = report["decision_gate"]
    print("===== causal belief update decomposition =====")
    print(
        f"pairs {report['data']['pairs']} | primary layer "
        f"{report['data']['primary_layer']} | reconstruction p95 "
        f"{report['data']['reconstruction_p95']:.8f} | state replay p95 "
        f"{report['data']['state_replay_p95']:.8f}"
    )
    for name, metric in report["tests"].items():
        print(
            f"  {name:<42} {metric['point']:+.5f} "
            f"CI [{metric['ci_low']:+.5f}, {metric['ci_high']:+.5f}]"
        )
    print(
        f"decomposition valid: {gate['decomposition_valid']} | "
        f"attention signature: {gate['attention_update_signature']} | "
        f"MLP signature: {gate['mlp_update_signature']} | "
        f"ready for factorial patching: {gate['ready_for_factorial_patching']}"
    )
    print(f"report: {args.output_dir}/summary.md")


if __name__ == "__main__":
    main()
