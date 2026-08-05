from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import PairAuditor


def _domains(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one domain")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled first-error attribution and repair"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="audit causal pair identifiability")
    audit.add_argument("--data-root", required=True, type=Path)
    audit.add_argument(
        "--domains",
        type=_domains,
        default=("gsm8k", "math", "olympiadbench", "omnimath"),
    )
    audit.add_argument(
        "--pair-directory", default="causal_first_error_v1"
    )
    audit.add_argument("--output", type=Path, default=None)
    return parser


def _print_audit(report: dict[str, object]) -> None:
    print(
        "causal audit: "
        f"chains={report['natural_chains']} | "
        f"errors={report['error_chains']} | correct={report['correct_chains']}"
    )
    print(
        "verified pairs: "
        f"target_correction={report['valid_target_correction_pairs']} | "
        f"controlled_root={report['valid_controlled_root_pairs']}"
    )
    if report["root_cause_claim"] == "not_identifiable_from_available_data":
        print("root-cause analysis: not identifiable from available data")
        print(f"next required artifact: {report['next_required_artifact']}")
    else:
        print("root-cause analysis: controlled pairs available")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = PairAuditor(
        args.data_root,
        args.domains,
        pair_directory=args.pair_directory,
    ).run()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    _print_audit(report)


if __name__ == "__main__":
    main()

