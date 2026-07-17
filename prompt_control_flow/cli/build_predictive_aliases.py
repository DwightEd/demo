from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from prompt_control_flow.causal_belief_routing.world import (
    AliasWorldConfig,
    generate_alias_worlds,
    write_alias_worlds_jsonl,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build exact finite-field predictive-alias constraint worlds."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_pairs", type=int, default=2000)
    parser.add_argument("--modulus", type=int, default=3)
    parser.add_argument("--num_variables", type=int, default=4)
    parser.add_argument("--common_rank", type=int, default=2)
    parser.add_argument("--template_families", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = AliasWorldConfig(
        modulus=int(args.modulus),
        num_variables=int(args.num_variables),
        common_rank=int(args.common_rank),
        template_families=int(args.template_families),
        seed=int(args.seed),
    )
    worlds = generate_alias_worlds(int(args.num_pairs), cfg)
    output = Path(args.output)
    write_alias_worlds_jsonl(output, worlds, cfg)
    branch_support = cfg.modulus ** (cfg.num_variables - cfg.common_rank - 1)
    print(f"saved {len(worlds)} predictive-alias pairs to {output}")
    print(
        f"GF({cfg.modulus})^{cfg.num_variables} | hypotheses="
        f"{cfg.modulus ** cfg.num_variables} | branch_support={branch_support} | "
        "current=uniform | future=deterministic-and-opposed"
    )


if __name__ == "__main__":
    main()
