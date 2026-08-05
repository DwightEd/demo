from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PAIR_SCHEMA_VERSION = "onset_pair_v1"
PAIR_KINDS = frozenset({"target_correction", "controlled_root"})


def _text(values: Mapping[str, Any], name: str) -> str:
    value = str(values.get(name, "")).strip()
    if not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(values: Mapping[str, Any], name: str) -> int:
    value = values.get(name)
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class OnsetPair:
    """One verified natural correction or controlled pre-decision contrast."""

    case_id: str
    dataset: str
    problem_hash: str
    error_chain_id: str
    error_trace_record: int
    first_error_step: int
    error_step_token_start: int
    error_step_token_end: int
    pair_kind: str
    correctness_verifier: str
    verification_evidence: str
    decision_position: int
    corrected_step_text: str | None = None
    first_divergent_token_index: int | None = None
    wrong_token_id: int | None = None
    correct_token_id: int | None = None
    same_prefix_before_divergence: bool | None = None
    generation_config_status: str | None = None
    template_id: str | None = None
    condition_id: str | None = None
    counterfactual_condition_id: str | None = None
    counterfactual_trace_record: int | None = None
    counterfactual_decision_position: int | None = None
    intervention_variable: str | None = None
    donor_alignment: Any | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> OnsetPair:
        if _text(values, "schema_version") != PAIR_SCHEMA_VERSION:
            raise ValueError("unsupported onset pair schema_version")
        kind = _text(values, "pair_kind")
        if kind not in PAIR_KINDS:
            raise ValueError(f"pair_kind must be one of {sorted(PAIR_KINDS)}")

        start = _integer(values, "error_step_token_start")
        end = _integer(values, "error_step_token_end")
        if start < 1 or end <= start:
            raise ValueError("error step must be a non-empty token interval")
        first_error_step = _integer(values, "first_error_step")
        if first_error_step < 0:
            raise ValueError("an onset pair requires a natural first-error step")

        target: dict[str, Any] = {}
        controlled: dict[str, Any] = {}
        if kind == "target_correction":
            divergent = _integer(values, "first_divergent_token_index")
            if divergent < start or divergent >= end:
                raise ValueError(
                    "first_divergent_token_index must lie inside the error step"
                )
            if values.get("same_prefix_before_divergence") is not True:
                raise ValueError(
                    "target_correction requires same_prefix_before_divergence=true"
                )
            decision_position = divergent - 1
            target = {
                "corrected_step_text": _text(values, "corrected_step_text"),
                "first_divergent_token_index": divergent,
                "wrong_token_id": _integer(values, "wrong_token_id"),
                "correct_token_id": _integer(values, "correct_token_id"),
                "same_prefix_before_divergence": True,
                "generation_config_status": _text(
                    values, "generation_config_status"
                ),
            }
        else:
            decision_position = _integer(values, "decision_position")
            counterfactual_position = _integer(
                values, "counterfactual_decision_position"
            )
            if decision_position < 0 or counterfactual_position < 0:
                raise ValueError("controlled decision positions must be nonnegative")
            controlled = {
                "template_id": _text(values, "template_id"),
                "condition_id": _text(values, "condition_id"),
                "counterfactual_condition_id": _text(
                    values, "counterfactual_condition_id"
                ),
                "counterfactual_trace_record": _integer(
                    values, "counterfactual_trace_record"
                ),
                "counterfactual_decision_position": counterfactual_position,
                "intervention_variable": _text(values, "intervention_variable"),
                "donor_alignment": values.get("donor_alignment"),
            }
            if controlled["donor_alignment"] is None:
                raise ValueError("controlled_root requires donor_alignment")

        return cls(
            case_id=_text(values, "case_id"),
            dataset=_text(values, "dataset"),
            problem_hash=_text(values, "problem_hash"),
            error_chain_id=str(values.get("error_chain_id", "")).strip(),
            error_trace_record=_integer(values, "error_trace_record"),
            first_error_step=first_error_step,
            error_step_token_start=start,
            error_step_token_end=end,
            pair_kind=kind,
            correctness_verifier=_text(values, "correctness_verifier"),
            verification_evidence=_text(values, "verification_evidence"),
            decision_position=decision_position,
            **target,
            **controlled,
        )

    def __post_init__(self) -> None:
        if not self.error_chain_id:
            raise ValueError("error_chain_id cannot be empty")
        if self.error_trace_record < 0:
            raise ValueError("error_trace_record must be nonnegative")

    @property
    def root_cause_eligible(self) -> bool:
        return self.pair_kind == "controlled_root"

    @property
    def allowed_claim(self) -> str:
        if self.root_cause_eligible:
            return "controlled_path_attribution_candidate"
        return "target_reference_only"

    def require_root_cause_eligibility(self) -> None:
        if not self.root_cause_eligible:
            raise ValueError(
                "root-cause intervention requires pair_kind='controlled_root'; "
                "a target_correction can support only target reference or rescue"
            )
