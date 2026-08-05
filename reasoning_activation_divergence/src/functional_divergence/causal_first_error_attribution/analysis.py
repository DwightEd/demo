from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InterventionRecord:
    case_id: str
    domain: str
    problem_hash: str
    pair_kind: str
    attention_effect: float
    ffn_effect: float
    interaction_effect: float
    pre_state_effect: float
    random_donor_attention: float
    wrong_source_attention: float
    random_donor_ffn: float
    token_rescue: bool
    step_rescue: bool

    def __post_init__(self) -> None:
        if not self.case_id or not self.domain or not self.problem_hash:
            raise ValueError("case, domain, and problem identifiers cannot be empty")
        if self.pair_kind not in {"target_correction", "controlled_root"}:
            raise ValueError("unsupported pair_kind")
        values = np.asarray(
            [
                self.attention_effect,
                self.ffn_effect,
                self.interaction_effect,
                self.pre_state_effect,
                self.random_donor_attention,
                self.wrong_source_attention,
                self.random_donor_ffn,
            ],
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError("intervention effects must be finite")


def _domain_problem_bootstrap(
    records: list[InterventionRecord],
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    if not records or values.shape != (len(records),):
        raise ValueError("bootstrap records and values must be non-empty and aligned")
    grouped: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record, value in zip(records, values, strict=True):
        grouped[record.domain][record.problem_hash].append(float(value))
    domain_problem_values = {
        domain: np.asarray(
            [np.mean(items) for items in problems.values()], dtype=np.float64
        )
        for domain, problems in grouped.items()
    }
    point = float(
        np.mean([np.mean(domain_values) for domain_values in domain_problem_values.values()])
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot), dtype=np.float64)
    for draw in range(int(n_boot)):
        domain_draws = []
        for domain_values in domain_problem_values.values():
            sampled = rng.choice(
                domain_values, size=len(domain_values), replace=True
            )
            domain_draws.append(float(np.mean(sampled)))
        draws[draw] = float(np.mean(domain_draws))
    return {
        "point": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_boot": int(n_boot),
        "resampling_unit": "domain_then_problem_hash",
    }


def _gate(
    records: list[InterventionRecord],
    contrast: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[str, dict[str, float | int]]:
    inference = _domain_problem_bootstrap(
        records, contrast, n_boot=n_boot, seed=seed
    )
    rescued = any(record.token_rescue or record.step_rescue for record in records)
    status = "supported" if inference["ci_low"] > 0.0 and rescued else "unsupported"
    return status, inference


def summarize_interventions(
    records: list[InterventionRecord] | tuple[InterventionRecord, ...],
    *,
    n_boot: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Summarize continuous case effects without upgrading natural rescue to cause."""
    values = list(records)
    if not values:
        raise ValueError("at least one intervention record is required")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be positive")
    controlled = [record for record in values if record.pair_kind == "controlled_root"]
    natural = [record for record in values if record.pair_kind == "target_correction"]
    claims: dict[str, str] = {}
    inference: dict[str, dict[str, float | int]] = {}

    if controlled:
        routing = np.asarray(
            [
                record.attention_effect
                - max(record.random_donor_attention, record.wrong_source_attention)
                for record in controlled
            ],
            dtype=np.float64,
        )
        ffn = np.asarray(
            [
                record.ffn_effect - record.random_donor_ffn
                for record in controlled
            ],
            dtype=np.float64,
        )
        claims["routing_root_cause"], inference["routing_vs_controls"] = _gate(
            controlled, routing, n_boot=n_boot, seed=seed
        )
        claims["ffn_root_cause"], inference["ffn_vs_controls"] = _gate(
            controlled, ffn, n_boot=n_boot, seed=seed + 1
        )
    else:
        claims["routing_root_cause"] = "not_identifiable"
        claims["ffn_root_cause"] = "not_identifiable"

    if natural:
        rescue = np.asarray(
            [
                max(
                    record.attention_effect
                    - max(
                        record.random_donor_attention,
                        record.wrong_source_attention,
                    ),
                    record.ffn_effect - record.random_donor_ffn,
                    record.pre_state_effect,
                )
                for record in natural
            ],
            dtype=np.float64,
        )
        claims["natural_rescue"], inference["natural_rescue_vs_controls"] = _gate(
            natural, rescue, n_boot=n_boot, seed=seed + 2
        )
    else:
        claims["natural_rescue"] = "not_evaluated"

    return {
        "cases": len(values),
        "controlled_root_cases": len(controlled),
        "target_correction_cases": len(natural),
        "claims": claims,
        "inference": inference,
        "individual_effects": [asdict(record) for record in values],
    }


__all__ = ["InterventionRecord", "summarize_interventions"]

