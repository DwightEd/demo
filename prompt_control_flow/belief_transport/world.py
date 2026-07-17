from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCHEMA_VERSION = "constraint_belief_wind_tunnel_v1"


@dataclass(frozen=True)
class WindTunnelConfig:
    domain_size: int = 8
    min_steps: int = 3
    max_steps: int = 6
    template_families: int = 3
    seed: int = 17

    def validate(self) -> None:
        if self.domain_size < 4:
            raise ValueError("domain_size must be at least 4")
        if self.min_steps < 2:
            raise ValueError("min_steps must be at least 2")
        if self.max_steps < self.min_steps + 1:
            raise ValueError("max_steps must leave room for a non-trivial endpoint")
        if not 1 <= self.template_families <= 3:
            raise ValueError("template_families must lie in [1, 3]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class ConstraintSpec:
    kind: str
    value: int
    a: int = 0
    b: int = 0
    modulus: int = 0


@dataclass(frozen=True)
class PrefixState:
    prefix_index: int
    condition: ConstraintSpec | None
    feasible_mask: np.ndarray


@dataclass(frozen=True)
class ConstraintWorld:
    problem_id: int
    target: tuple[int, int]
    template_family: int
    conditions: tuple[ConstraintSpec, ...]
    domain_size: int

    def prefix_states(self, hypotheses: np.ndarray | None = None) -> list[PrefixState]:
        grid = (
            build_hypothesis_grid(self.domain_size)
            if hypotheses is None
            else np.asarray(hypotheses, dtype=np.int64)
        )
        feasible = np.ones(len(grid), dtype=bool)
        states = [PrefixState(0, None, feasible.copy())]
        target_index = self.target[0] * self.domain_size + self.target[1]
        for index, condition in enumerate(self.conditions, start=1):
            updated = feasible & constraint_mask(condition, grid)
            if not updated[target_index]:
                raise ValueError("a generated condition excludes its target")
            if int(updated.sum()) >= int(feasible.sum()):
                raise ValueError("every generated condition must reduce the feasible set")
            feasible = updated
            states.append(PrefixState(index, condition, feasible.copy()))
        return states


def build_hypothesis_grid(domain_size: int) -> np.ndarray:
    if int(domain_size) < 1:
        raise ValueError("domain_size must be positive")
    return np.asarray(
        [(x, y) for x in range(int(domain_size)) for y in range(int(domain_size))],
        dtype=np.int64,
    )


def constraint_mask(constraint: ConstraintSpec, hypotheses: np.ndarray) -> np.ndarray:
    grid = np.asarray(hypotheses, dtype=np.int64)
    if grid.ndim != 2 or grid.shape[1] != 2:
        raise ValueError("hypotheses must have shape [num_hypotheses, 2]")
    x = grid[:, 0]
    y = grid[:, 1]
    kind = constraint.kind
    if kind == "x_le":
        return x <= constraint.value
    if kind == "x_ge":
        return x >= constraint.value
    if kind == "y_le":
        return y <= constraint.value
    if kind == "y_ge":
        return y >= constraint.value
    if kind == "x_eq":
        return x == constraint.value
    if kind == "y_eq":
        return y == constraint.value
    if kind == "affine_eq":
        return constraint.a * x + constraint.b * y == constraint.value
    if kind == "affine_mod":
        if constraint.modulus < 2:
            raise ValueError("affine_mod requires modulus >= 2")
        return (constraint.a * x + constraint.b * y) % constraint.modulus == constraint.value
    raise ValueError(f"unknown constraint kind: {kind}")


def render_constraint(constraint: ConstraintSpec, template_family: int) -> str:
    family = int(template_family) % 3
    variable, relation = "", ""
    if constraint.kind[:1] in {"x", "y"} and constraint.kind in {
        "x_le", "x_ge", "y_le", "y_ge", "x_eq", "y_eq"
    }:
        variable = constraint.kind[0]
        relation = constraint.kind[2:]
        symbols = {"le": "<=", "ge": ">=", "eq": "="}
        words = {"le": "at most", "ge": "at least", "eq": "exactly"}
        alternatives = {
            "le": "cannot exceed",
            "ge": "cannot be smaller than",
            "eq": "is fixed at",
        }
        if family == 0:
            return f"{variable} {symbols[relation]} {constraint.value}."
        if family == 1:
            return f"The value of {variable} is {words[relation]} {constraint.value}."
        return f"{variable} {alternatives[relation]} {constraint.value}."
    expression = _affine_expression(constraint.a, constraint.b)
    if constraint.kind == "affine_eq":
        if family == 0:
            return f"{expression} = {constraint.value}."
        if family == 1:
            return f"The linear expression {expression} equals {constraint.value}."
        return f"Evaluating {expression} gives {constraint.value}."
    if constraint.kind == "affine_mod":
        if family == 0:
            return f"({expression}) mod {constraint.modulus} = {constraint.value}."
        if family == 1:
            return (
                f"Dividing {expression} by {constraint.modulus} leaves remainder "
                f"{constraint.value}."
            )
        return (
            f"The residue of {expression} modulo {constraint.modulus} is "
            f"{constraint.value}."
        )
    raise ValueError(f"cannot render constraint kind {constraint.kind}")


def render_prefix_prompt(world: ConstraintWorld, prefix_index: int) -> str:
    if not 0 <= int(prefix_index) <= len(world.conditions):
        raise ValueError("prefix_index is outside the condition sequence")
    domain = ", ".join(str(value) for value in range(world.domain_size))
    observed = world.conditions[: int(prefix_index)]
    if observed:
        lines = "\n".join(
            f"{index}. {render_constraint(condition, world.template_family)}"
            for index, condition in enumerate(observed, start=1)
        )
    else:
        lines = "No constraints have been observed yet."
    return (
        "Track the complete feasible set for two integer variables.\n"
        f"Both x and y belong to {{{domain}}}.\n"
        "Use every observed constraint. Do not guess a single pair when several remain.\n"
        "Observed constraints:\n"
        f"{lines}\n"
        "Which ordered pairs (x, y) are still feasible?"
    )


def _affine_expression(a: int, b: int) -> str:
    terms: list[str] = []
    for coefficient, variable in ((int(a), "x"), (int(b), "y")):
        if coefficient == 0:
            continue
        magnitude = "" if abs(coefficient) == 1 else str(abs(coefficient))
        term = f"{magnitude}{variable}"
        if not terms:
            terms.append(term if coefficient > 0 else f"-{term}")
        else:
            terms.append((" + " if coefficient > 0 else " - ") + term)
    return "".join(terms) or "0"


def _candidate_constraints(target: tuple[int, int], domain_size: int) -> list[ConstraintSpec]:
    x, y = target
    candidates: list[ConstraintSpec] = []
    for kind, value in (
        ("x_le", x),
        ("x_ge", x),
        ("y_le", y),
        ("y_ge", y),
        ("x_eq", x),
        ("y_eq", y),
    ):
        candidates.append(ConstraintSpec(kind=kind, value=int(value)))
    coefficients = ((1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (-1, 2))
    for a, b in coefficients:
        value = a * x + b * y
        candidates.append(
            ConstraintSpec(kind="affine_eq", value=int(value), a=a, b=b)
        )
        for modulus in range(2, min(domain_size, 5) + 1):
            candidates.append(
                ConstraintSpec(
                    kind="affine_mod",
                    value=int(value % modulus),
                    a=a,
                    b=b,
                    modulus=modulus,
                )
            )
    return candidates


def _deduplicate_by_mask(
    candidates: Sequence[ConstraintSpec],
    hypotheses: np.ndarray,
) -> list[ConstraintSpec]:
    unique: dict[bytes, ConstraintSpec] = {}
    for candidate in candidates:
        key = np.packbits(constraint_mask(candidate, hypotheses)).tobytes()
        unique.setdefault(key, candidate)
    return list(unique.values())


def _generate_one_world(
    problem_id: int,
    cfg: WindTunnelConfig,
    rng: np.random.Generator,
) -> ConstraintWorld | None:
    hypotheses = build_hypothesis_grid(cfg.domain_size)
    target = (
        int(rng.integers(0, cfg.domain_size)),
        int(rng.integers(0, cfg.domain_size)),
    )
    candidates = _deduplicate_by_mask(
        _candidate_constraints(target, cfg.domain_size), hypotheses
    )
    feasible = np.ones(len(hypotheses), dtype=bool)
    selected: list[ConstraintSpec] = []

    weak_target_steps = cfg.min_steps - 1
    while len(selected) < weak_target_steps:
        options: list[tuple[float, ConstraintSpec, np.ndarray]] = []
        desired = max(2, int(round(float(feasible.sum()) * 0.55)))
        for candidate in candidates:
            if candidate.kind in {"x_eq", "y_eq"}:
                continue
            updated = feasible & constraint_mask(candidate, hypotheses)
            count = int(updated.sum())
            if 2 <= count < int(feasible.sum()):
                score = abs(np.log(count) - np.log(desired)) + float(rng.random()) * 0.05
                options.append((score, candidate, updated))
        if not options:
            return None
        _, chosen, feasible = min(options, key=lambda item: item[0])
        selected.append(chosen)
        candidates.remove(chosen)

    # Exact unary constraints are reserved for the endpoint. They guarantee a
    # unique solution without allowing the weak prefix to collapse immediately.
    endpoint_order = [
        ConstraintSpec(kind="x_eq", value=target[0]),
        ConstraintSpec(kind="y_eq", value=target[1]),
    ]
    if bool(rng.integers(0, 2)):
        endpoint_order.reverse()
    for candidate in endpoint_order:
        if int(feasible.sum()) == 1:
            break
        updated = feasible & constraint_mask(candidate, hypotheses)
        if 0 < int(updated.sum()) < int(feasible.sum()):
            selected.append(candidate)
            feasible = updated
        if len(selected) >= cfg.max_steps:
            break

    if int(feasible.sum()) != 1:
        remaining = []
        for candidate in candidates:
            updated = feasible & constraint_mask(candidate, hypotheses)
            if 0 < int(updated.sum()) < int(feasible.sum()):
                remaining.append((int(updated.sum()), candidate, updated))
        while remaining and int(feasible.sum()) > 1 and len(selected) < cfg.max_steps:
            _, chosen, updated = min(remaining, key=lambda item: item[0])
            selected.append(chosen)
            feasible = updated
            remaining = [
                (int((feasible & constraint_mask(candidate, hypotheses)).sum()), candidate,
                 feasible & constraint_mask(candidate, hypotheses))
                for _, candidate, _ in remaining
                if 0 < int((feasible & constraint_mask(candidate, hypotheses)).sum())
                < int(feasible.sum())
            ]
    if int(feasible.sum()) != 1 or not cfg.min_steps <= len(selected) <= cfg.max_steps:
        return None
    return ConstraintWorld(
        problem_id=int(problem_id),
        target=target,
        template_family=int(rng.integers(0, cfg.template_families)),
        conditions=tuple(selected),
        domain_size=cfg.domain_size,
    )


def generate_worlds(num_problems: int, cfg: WindTunnelConfig) -> list[ConstraintWorld]:
    cfg.validate()
    if int(num_problems) < 1:
        raise ValueError("num_problems must be positive")
    rng = np.random.default_rng(cfg.seed)
    worlds: list[ConstraintWorld] = []
    for problem_id in range(int(num_problems)):
        world = None
        for _ in range(100):
            world = _generate_one_world(problem_id, cfg, rng)
            if world is not None:
                break
        if world is None:
            raise RuntimeError(f"could not construct problem {problem_id}")
        worlds.append(world)
    return worlds


def _world_to_payload(world: ConstraintWorld, cfg: WindTunnelConfig) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(cfg),
        "problem_id": world.problem_id,
        "target": list(world.target),
        "template_family": world.template_family,
        "domain_size": world.domain_size,
        "conditions": [asdict(condition) for condition in world.conditions],
    }


def write_worlds_jsonl(
    path: str | Path,
    worlds: Iterable[ConstraintWorld],
    cfg: WindTunnelConfig,
) -> None:
    cfg.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for world in worlds:
            handle.write(json.dumps(_world_to_payload(world, cfg), sort_keys=True) + "\n")


def load_worlds_jsonl(path: str | Path) -> tuple[list[ConstraintWorld], WindTunnelConfig]:
    source = Path(path)
    worlds: list[ConstraintWorld] = []
    cfg: WindTunnelConfig | None = None
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"unsupported schema on line {line_number}")
            row_cfg = WindTunnelConfig(**payload["config"])
            row_cfg.validate()
            if cfg is None:
                cfg = row_cfg
            elif row_cfg != cfg:
                raise ValueError("all wind-tunnel rows must use the same config")
            worlds.append(
                ConstraintWorld(
                    problem_id=int(payload["problem_id"]),
                    target=tuple(int(value) for value in payload["target"]),
                    template_family=int(payload["template_family"]),
                    conditions=tuple(
                        ConstraintSpec(**condition) for condition in payload["conditions"]
                    ),
                    domain_size=int(payload["domain_size"]),
                )
            )
    if cfg is None or not worlds:
        raise ValueError("wind-tunnel file is empty")
    return worlds, cfg
