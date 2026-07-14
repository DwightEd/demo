from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the causal pullback operator under same-problem and output controls."
    )
    parser.add_argument("--input", required=True, help="Causal pullback trace NPZ.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--phase_grid", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--min_coverage", type=float, default=0.80)
    parser.add_argument("--min_contrastive_problems", type=int, default=100)
    parser.add_argument("--max_finite_difference_error", type=float, default=0.50)
    parser.add_argument("--max_acausal_fisher_leakage", type=float, default=1e-5)
    parser.add_argument("--replay_cosine_threshold", type=float, default=0.98)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    from prompt_control_flow.causal_pullback import (
        CausalPullbackAuditConfig,
        run_causal_pullback_audit,
    )
    from prompt_control_flow.ocgpi.models import CrossFitConfig

    cfg = CausalPullbackAuditConfig(
        phase_grid=args.phase_grid,
        bootstrap=args.bootstrap,
        min_coverage=args.min_coverage,
        min_contrastive_problems=args.min_contrastive_problems,
        max_finite_difference_error=args.max_finite_difference_error,
        max_acausal_fisher_leakage=args.max_acausal_fisher_leakage,
        replay_cosine_threshold=args.replay_cosine_threshold,
        random_seed=args.seed,
        crossfit=CrossFitConfig(
            outer_folds=args.folds,
            inner_folds=args.inner_folds,
            seed=args.seed,
        ),
    )
    report = run_causal_pullback_audit(args.input, args.output_dir, cfg)
    validation = report["validation"]
    direct = report["direct_same_problem_diagnosis"]
    primary = next(
        row for row in direct["scores"] if row["name"] == direct["primary_score"]
    )
    combined = report["conditional_increment"]["field_plus_causal_pullback"]
    print("===== causal pullback flow field =====")
    print(
        f"responses {report['preflight']['responses']} | errors {report['preflight']['errors']} "
        f"| valid coverage {report['preflight']['valid_coverage']:.3f}"
    )
    print(
        f"primary within {primary['within_problem_auroc']:.3f} "
        f"CI {primary['within_problem_ci95']}"
    )
    print(
        f"output AUROC {combined['output_only']['auroc']:.3f} | "
        f"+ pullback {combined['output_plus_geometry']['auroc']:.3f} | "
        f"usable bits "
        f"{combined['increment']['conditional_usable_information']['point_bits']:+.4f}"
    )
    print(
        "gates: "
        f"numerical={validation['numerical_validity']['pass']} "
        f"mechanism={validation['mechanism_supported']['pass']} "
        f"increment={validation['detector_increment_supported']['pass']} "
        f"confirmatory={validation['confirmatory_ready']}"
    )
    print(f"report: {Path(args.output_dir) / 'summary.md'}")


if __name__ == "__main__":
    main()
