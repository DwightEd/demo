from __future__ import annotations

import argparse
from typing import Sequence

from prompt_control_flow.belief_transport.audit import (
    BeliefAuditConfig,
    run_belief_transport_audit,
)
from prompt_control_flow.belief_transport.decoder import DecoderConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-fit exact belief readout and directional transport tests."
    )
    parser.add_argument("--input", required=True, help="Merged belief trace NPZ.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--primary_layer", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--decoder_kind", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--max_null_information_gain_gap",
        type=float,
        default=1e-8,
        help="Maximum allowed p95 true/null information-gain mismatch in nats.",
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    decoder = DecoderConfig(
        num_folds=int(args.folds),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        validation_fraction=float(args.validation_fraction),
        patience=int(args.patience),
        decoder_kind=str(args.decoder_kind),
        hidden_dim=int(args.hidden_dim),
        device=str(args.device),
        seed=int(args.seed),
    )
    cfg = BeliefAuditConfig(
        decoder=decoder,
        primary_layer=int(args.primary_layer),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
        max_null_information_gain_gap=float(args.max_null_information_gain_gap),
    )
    report = run_belief_transport_audit(args.input, args.output_dir, cfg)
    print("===== constraint-supported belief transport =====")
    print(
        f"rows {report['rows']} | problems {report['problems']} | "
        f"primary layer {report['primary_layer']}"
    )
    primary = report["belief_metrics"]["primary_hidden"]
    increment = report["joint_over_output_usable_bits"]
    direction = report["directionality"]
    print(
        f"belief support AUROC {primary['support_auc']:.3f} | "
        f"entropy rho {primary['entropy_spearman']:.3f}"
    )
    print(
        f"joint over logits usable bits {increment['mean']:+.4f} "
        f"CI [{increment['ci_low']:+.4f}, {increment['ci_high']:+.4f}]"
    )
    print(
        f"operator AUROC {direction['operator_auc']:.3f} | "
        f"FR advantage {direction['operator_advantage']['mean']:+.4f} | "
        f"null gain gap p95 {direction['null_information_gain_gap_p95']:.4f}"
    )
    print(f"decision gate: {report['decision_gate']}")


if __name__ == "__main__":
    main()
