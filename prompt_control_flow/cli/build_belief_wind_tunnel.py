from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

from prompt_control_flow.belief_transport.world import (
    WindTunnelConfig,
    generate_worlds,
    write_worlds_jsonl,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an exact finite constraint-belief wind tunnel."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_problems", type=int, default=2000)
    parser.add_argument("--domain_size", type=int, default=8)
    parser.add_argument("--min_steps", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--template_families", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = WindTunnelConfig(
        domain_size=int(args.domain_size),
        min_steps=int(args.min_steps),
        max_steps=int(args.max_steps),
        template_families=int(args.template_families),
        seed=int(args.seed),
    )
    worlds = generate_worlds(int(args.num_problems), cfg)
    output = Path(args.output)
    write_worlds_jsonl(output, worlds, cfg)
    step_counts = [len(world.conditions) for world in worlds]
    histogram = Counter(step_counts)
    print(f"saved {len(worlds)} exact constraint worlds to {output}")
    print(
        f"hypotheses={cfg.domain_size ** 2} | steps="
        f"{min(step_counts)}/{sum(step_counts) / len(step_counts):.2f}/{max(step_counts)} "
        "(min/mean/max)"
    )
    print(
        "step histogram: "
        + ", ".join(f"{steps}:{histogram[steps]}" for steps in sorted(histogram))
    )


if __name__ == "__main__":
    main()
