from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .finite_field import (
    affine_support_mask,
    enumerate_vectors,
    in_row_span,
    linear_combination,
    matrix_rank_mod,
    validate_prime_modulus,
)


SCHEMA_VERSION = "predictive_alias_affine_world_v1"
VARIABLE_NAMES = ("u", "v", "w", "z", "r", "s")


@dataclass(frozen=True)
class AliasWorldConfig:
    modulus: int = 3
    num_variables: int = 4
    common_rank: int = 2
    template_families: int = 3
    seed: int = 17

    def validate(self) -> None:
        validate_prime_modulus(self.modulus)
        if not 3 <= int(self.num_variables) <= len(VARIABLE_NAMES):
            raise ValueError(f"num_variables must lie in [3, {len(VARIABLE_NAMES)}]")
        if not 1 <= int(self.common_rank) <= int(self.num_variables) - 2:
            raise ValueError("common_rank must leave one branch and one alias dimension")
        if not 1 <= int(self.template_families) <= 3:
            raise ValueError("template_families must lie in [1, 3]")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class LinearConstraint:
    coefficients: tuple[int, ...]
    rhs: int


@dataclass(frozen=True)
class PredictiveAliasWorld:
    pair_id: int
    modulus: int
    num_variables: int
    template_family: int
    common_constraints: tuple[LinearConstraint, ...]
    branch_constraints: tuple[LinearConstraint, LinearConstraint]
    current_query: tuple[int, ...]
    future_query: tuple[int, ...]

    @property
    def assignments(self) -> np.ndarray:
        return enumerate_vectors(self.modulus, self.num_variables)

    def _system(self, branch: int | None) -> tuple[np.ndarray, np.ndarray]:
        constraints = list(self.common_constraints)
        if branch is not None:
            if int(branch) not in (0, 1):
                raise ValueError("branch must be 0 or 1")
            constraints.append(self.branch_constraints[int(branch)])
        coefficients = np.asarray(
            [constraint.coefficients for constraint in constraints], dtype=np.int64
        )
        rhs = np.asarray([constraint.rhs for constraint in constraints], dtype=np.int64)
        return coefficients, rhs

    def base_support_mask(self) -> np.ndarray:
        coefficients, rhs = self._system(None)
        return affine_support_mask(self.assignments, coefficients, rhs, self.modulus)

    def support_mask(self, branch: int) -> np.ndarray:
        coefficients, rhs = self._system(int(branch))
        return affine_support_mask(self.assignments, coefficients, rhs, self.modulus)


def _sample_independent_vector(
    rng: np.random.Generator,
    rows: np.ndarray,
    modulus: int,
) -> np.ndarray:
    dimension = int(rows.shape[1])
    for _ in range(1000):
        candidate = rng.integers(0, modulus, size=dimension, dtype=np.int64)
        if np.any(candidate) and not in_row_span(candidate, rows, modulus):
            return candidate
    raise RuntimeError("could not sample an independent finite-field vector")


def _generate_world(
    pair_id: int,
    cfg: AliasWorldConfig,
    rng: np.random.Generator,
) -> PredictiveAliasWorld:
    p = int(cfg.modulus)
    seed_row = np.zeros((0, cfg.num_variables), dtype=np.int64)
    common_rows: list[np.ndarray] = []
    current_matrix = seed_row
    while len(common_rows) < int(cfg.common_rank):
        row = _sample_independent_vector(rng, current_matrix, p)
        common_rows.append(row)
        current_matrix = np.vstack(common_rows)
    common_matrix = np.asarray(common_rows, dtype=np.int64)
    common_rhs = rng.integers(0, p, size=cfg.common_rank, dtype=np.int64)

    branch_vector = _sample_independent_vector(rng, common_matrix, p)
    branch_values = rng.choice(p, size=2, replace=False).astype(np.int64)
    extended = np.vstack([common_matrix, branch_vector])
    current_query = _sample_independent_vector(rng, extended, p)

    branch_weight = int(rng.integers(1, p))
    common_weights = rng.integers(0, p, size=cfg.common_rank, dtype=np.int64)
    if not np.any(common_weights):
        common_weights[int(rng.integers(0, cfg.common_rank))] = 1
    future_query = (
        branch_weight * branch_vector
        + linear_combination(common_matrix, common_weights, p)
    ) % p

    common_constraints = tuple(
        LinearConstraint(tuple(int(value) for value in row), int(rhs))
        for row, rhs in zip(common_matrix, common_rhs)
    )
    branches = tuple(
        LinearConstraint(
            tuple(int(value) for value in branch_vector), int(branch_value)
        )
        for branch_value in branch_values
    )
    world = PredictiveAliasWorld(
        pair_id=int(pair_id),
        modulus=p,
        num_variables=int(cfg.num_variables),
        template_family=int(rng.integers(0, cfg.template_families)),
        common_constraints=common_constraints,
        branch_constraints=(branches[0], branches[1]),
        current_query=tuple(int(value) for value in current_query),
        future_query=tuple(int(value) for value in future_query),
    )
    _validate_world(world, cfg)
    return world


def _validate_world(world: PredictiveAliasWorld, cfg: AliasWorldConfig) -> None:
    p = int(cfg.modulus)
    common = np.asarray(
        [constraint.coefficients for constraint in world.common_constraints],
        dtype=np.int64,
    )
    branch = np.asarray(world.branch_constraints[0].coefficients, dtype=np.int64)
    current = np.asarray(world.current_query, dtype=np.int64)
    future = np.asarray(world.future_query, dtype=np.int64)
    if matrix_rank_mod(common, p) != int(cfg.common_rank):
        raise ValueError("common constraints are rank deficient")
    if matrix_rank_mod(np.vstack([common, branch]), p) != int(cfg.common_rank) + 1:
        raise ValueError("branch evidence is redundant")
    if matrix_rank_mod(np.vstack([common, branch, current]), p) != int(cfg.common_rank) + 2:
        raise ValueError("current query does not create a predictive alias")
    if not in_row_span(future, np.vstack([common, branch]), p):
        raise ValueError("future query is not determined by the observed constraints")
    expected_size = p ** (int(cfg.num_variables) - int(cfg.common_rank) - 1)
    supports = [world.support_mask(branch_id) for branch_id in (0, 1)]
    if any(int(mask.sum()) != expected_size for mask in supports):
        raise ValueError("branch supports have an unexpected cardinality")
    if np.any(supports[0] & supports[1]):
        raise ValueError("alias branches must be distinct affine cosets")


def generate_alias_worlds(
    num_pairs: int,
    cfg: AliasWorldConfig,
) -> list[PredictiveAliasWorld]:
    cfg.validate()
    if int(num_pairs) < 1:
        raise ValueError("num_pairs must be positive")
    rng = np.random.default_rng(cfg.seed)
    return [_generate_world(pair_id, cfg, rng) for pair_id in range(int(num_pairs))]


def _linear_expression(coefficients: Sequence[int], modulus: int) -> str:
    terms: list[str] = []
    for coefficient, variable in zip(coefficients, VARIABLE_NAMES):
        value = int(coefficient) % int(modulus)
        if value == 0:
            continue
        term = variable if value == 1 else f"{value}{variable}"
        terms.append(term)
    return " + ".join(terms) if terms else "0"


def render_constraint(
    constraint: LinearConstraint,
    *,
    modulus: int,
    template_family: int,
) -> str:
    expression = _linear_expression(constraint.coefficients, modulus)
    family = int(template_family) % 3
    if family == 0:
        return f"({expression}) mod {modulus} = {constraint.rhs}."
    if family == 1:
        return (
            f"The remainder of {expression} after division by {modulus} is "
            f"{constraint.rhs}."
        )
    return f"The linear residue {expression} is congruent to {constraint.rhs} modulo {modulus}."


def render_query(query: Sequence[int], *, modulus: int) -> str:
    expression = _linear_expression(query, modulus)
    return f"What is the residue of {expression} modulo {modulus}?"


@dataclass(frozen=True)
class RenderedAliasPrompt:
    user_text: str
    branch_evidence_text: str
    query_role: str


def render_alias_prompt(
    world: PredictiveAliasWorld,
    branch: int,
    query_role: str,
) -> RenderedAliasPrompt:
    if int(branch) not in (0, 1):
        raise ValueError("branch must be 0 or 1")
    role = str(query_role).strip().lower()
    if role not in {"current", "future"}:
        raise ValueError("query_role must be current or future")
    names = ", ".join(VARIABLE_NAMES[: world.num_variables])
    common_lines = "\n".join(
        f"{index}. {render_constraint(constraint, modulus=world.modulus, template_family=world.template_family)}"
        for index, constraint in enumerate(world.common_constraints, start=1)
    )
    branch_text = render_constraint(
        world.branch_constraints[int(branch)],
        modulus=world.modulus,
        template_family=world.template_family,
    )
    query = world.current_query if role == "current" else world.future_query
    user_text = (
        "Track all assignments consistent with the modular constraints.\n"
        f"Variables {names} each belong to {{0, ..., {world.modulus - 1}}}.\n"
        "Common evidence:\n"
        f"{common_lines}\n"
        "New evidence:\n"
        f"{branch_text}\n"
        f"{render_query(query, modulus=world.modulus)}\n"
        f"Answer with one residue from 0 to {world.modulus - 1} and no explanation."
    )
    return RenderedAliasPrompt(
        user_text=user_text,
        branch_evidence_text=branch_text,
        query_role=role,
    )


def _world_payload(world: PredictiveAliasWorld, cfg: AliasWorldConfig) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(cfg),
        "pair_id": int(world.pair_id),
        "modulus": int(world.modulus),
        "num_variables": int(world.num_variables),
        "template_family": int(world.template_family),
        "common_constraints": [asdict(value) for value in world.common_constraints],
        "branch_constraints": [asdict(value) for value in world.branch_constraints],
        "current_query": list(world.current_query),
        "future_query": list(world.future_query),
    }


def write_alias_worlds_jsonl(
    path: str | Path,
    worlds: Iterable[PredictiveAliasWorld],
    cfg: AliasWorldConfig,
) -> None:
    cfg.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for world in worlds:
            handle.write(json.dumps(_world_payload(world, cfg), sort_keys=True) + "\n")


def load_alias_worlds_jsonl(
    path: str | Path,
) -> tuple[list[PredictiveAliasWorld], AliasWorldConfig]:
    worlds: list[PredictiveAliasWorld] = []
    cfg: AliasWorldConfig | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"unsupported alias-world schema on line {line_number}")
            row_cfg = AliasWorldConfig(**payload["config"])
            row_cfg.validate()
            if cfg is None:
                cfg = row_cfg
            elif cfg != row_cfg:
                raise ValueError("all alias worlds must share one config")
            worlds.append(
                PredictiveAliasWorld(
                    pair_id=int(payload["pair_id"]),
                    modulus=int(payload["modulus"]),
                    num_variables=int(payload["num_variables"]),
                    template_family=int(payload["template_family"]),
                    common_constraints=tuple(
                        LinearConstraint(
                            tuple(int(v) for v in value["coefficients"]),
                            int(value["rhs"]),
                        )
                        for value in payload["common_constraints"]
                    ),
                    branch_constraints=tuple(
                        LinearConstraint(
                            tuple(int(v) for v in value["coefficients"]),
                            int(value["rhs"]),
                        )
                        for value in payload["branch_constraints"]
                    ),
                    current_query=tuple(int(v) for v in payload["current_query"]),
                    future_query=tuple(int(v) for v in payload["future_query"]),
                )
            )
    if cfg is None or not worlds:
        raise ValueError("alias-world file is empty")
    return worlds, cfg
