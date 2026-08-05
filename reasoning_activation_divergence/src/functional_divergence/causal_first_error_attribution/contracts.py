from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PAIR_SCHEMA_VERSION = "onset_pair_v1"
PAIR_KINDS = frozenset({"target_correction", "controlled_root"})
ONSET_TRACE_SCHEMA = "onset_trace_v1"


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


@dataclass(frozen=True)
class OnsetTraceArtifact:
    """One future-free decision trace with a head/source-token message graph."""

    input_ids: np.ndarray
    decision_position: int
    source_token_positions: np.ndarray
    source_step_ids: np.ndarray
    selected_layers: np.ndarray
    wrong_token_id: int
    correct_token_id: int
    logits_topk_ids: np.ndarray
    logits_topk_values: np.ndarray
    resid_pre_block: np.ndarray
    attn_head_output: np.ndarray
    attn_branch_output: np.ndarray
    mlp_output: np.ndarray
    attn_edge_margin_proxy: np.ndarray
    attn_edge_mass: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        input_ids = _integer_array(self.input_ids, "input_ids", rank=1)
        source_positions = _integer_array(
            self.source_token_positions, "source_token_positions", rank=1
        )
        source_steps = _integer_array(
            self.source_step_ids, "source_step_ids", rank=1
        )
        layers = _integer_array(self.selected_layers, "selected_layers", rank=1)
        topk_ids = _integer_array(self.logits_topk_ids, "logits_topk_ids", rank=1)
        topk_values = _float_array(
            self.logits_topk_values, "logits_topk_values", rank=1
        )
        resid = _float_array(self.resid_pre_block, "resid_pre_block", rank=2)
        head_output = _float_array(
            self.attn_head_output, "attn_head_output", rank=3
        )
        attn_output = _float_array(
            self.attn_branch_output, "attn_branch_output", rank=2
        )
        mlp_output = _float_array(self.mlp_output, "mlp_output", rank=2)
        edge_proxy = _float_array(
            self.attn_edge_margin_proxy, "attn_edge_margin_proxy", rank=3
        )
        edge_mass = _float_array(
            self.attn_edge_mass, "attn_edge_mass", rank=3
        )

        token_count = len(input_ids)
        if token_count < 1 or self.decision_position != token_count - 1:
            raise ValueError("decision_position must be the final observable token")
        if not np.array_equal(source_positions, np.arange(token_count)):
            raise ValueError("source_token_positions must cover the observable prefix")
        if source_steps.shape != (token_count,):
            raise ValueError("source_step_ids must align with source_token_positions")
        if np.any(source_steps < -1):
            raise ValueError("source_step_ids must use -1 for prompt or a step index")
        if layers.size < 1 or len(np.unique(layers)) != len(layers):
            raise ValueError("selected_layers must contain unique layer ids")
        if self.correct_token_id < 0 or self.wrong_token_id < 0:
            raise ValueError("correct and wrong token ids must be nonnegative")
        if self.correct_token_id == self.wrong_token_id:
            raise ValueError("correct_token_id and wrong_token_id must differ")
        if topk_ids.shape != topk_values.shape or topk_ids.size < 1:
            raise ValueError("top-k ids and values must be non-empty and aligned")

        layer_count = len(layers)
        if resid.shape[0] != layer_count:
            raise ValueError("resid_pre_block must align with selected_layers")
        hidden = int(resid.shape[1])
        if hidden < 1:
            raise ValueError("hidden dimension must be positive")
        if head_output.shape[0] != layer_count:
            raise ValueError("attn_head_output must align with selected_layers")
        heads, head_dim = int(head_output.shape[1]), int(head_output.shape[2])
        if heads < 1 or head_dim < 1 or heads * head_dim != hidden:
            raise ValueError("attention head topology must reconstruct hidden size")
        expected_branch = (layer_count, hidden)
        if attn_output.shape != expected_branch or mlp_output.shape != expected_branch:
            raise ValueError("attention and MLP outputs must have shape [L,D]")
        graph_shape = (layer_count, heads, token_count)
        if edge_proxy.shape != graph_shape or edge_mass.shape != graph_shape:
            raise ValueError("attention graph arrays must have shape [L,H,T]")
        if np.any(edge_mass < 0.0) or np.any(edge_mass > 1.0):
            raise ValueError("attention edge mass must lie in [0,1]")
        if not np.allclose(edge_mass.sum(axis=2), 1.0, atol=1e-4, rtol=1e-4):
            raise ValueError("attention edge mass must sum to one per layer/head")
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        required = (
            "schema",
            "case_id",
            "model_name",
            "model_revision_or_unknown",
            "tokenizer_name",
            "tokenizer_revision_or_unknown",
            "source_trace_fingerprint",
            "pair_file_fingerprint",
            "extraction_config",
            "attention_reconstruction_max_relative_error",
        )
        if self.metadata.get("schema") != ONSET_TRACE_SCHEMA:
            raise ValueError("unsupported onset trace schema")
        missing = [name for name in required if name not in self.metadata]
        if missing:
            raise ValueError(f"onset trace metadata is missing {missing}")
        for name in required[:-2]:
            if not str(self.metadata[name]).strip():
                raise ValueError(f"onset trace metadata {name} cannot be empty")
        if not isinstance(self.metadata["extraction_config"], dict):
            raise TypeError("extraction_config must be a JSON object")
        error = float(self.metadata["attention_reconstruction_max_relative_error"])
        if not np.isfinite(error) or error < 0.0:
            raise ValueError("attention reconstruction error must be nonnegative")

    def save(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            np.savez_compressed(
                handle,
                input_ids=self.input_ids,
                decision_position=np.asarray(self.decision_position, dtype=np.int32),
                source_token_positions=self.source_token_positions,
                source_step_ids=self.source_step_ids,
                selected_layers=self.selected_layers,
                wrong_token_id=np.asarray(self.wrong_token_id, dtype=np.int32),
                correct_token_id=np.asarray(self.correct_token_id, dtype=np.int32),
                logits_topk_ids=self.logits_topk_ids,
                logits_topk_values=self.logits_topk_values,
                resid_pre_block=self.resid_pre_block,
                attn_head_output=self.attn_head_output,
                attn_branch_output=self.attn_branch_output,
                mlp_output=self.mlp_output,
                attn_edge_margin_proxy=self.attn_edge_margin_proxy,
                attn_edge_mass=self.attn_edge_mass,
                metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
            )

    @classmethod
    def load(cls, path: str | Path) -> OnsetTraceArtifact:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            artifact = cls(
                input_ids=archive["input_ids"],
                decision_position=int(np.asarray(archive["decision_position"]).item()),
                source_token_positions=archive["source_token_positions"],
                source_step_ids=archive["source_step_ids"],
                selected_layers=archive["selected_layers"],
                wrong_token_id=int(np.asarray(archive["wrong_token_id"]).item()),
                correct_token_id=int(np.asarray(archive["correct_token_id"]).item()),
                logits_topk_ids=archive["logits_topk_ids"],
                logits_topk_values=archive["logits_topk_values"],
                resid_pre_block=archive["resid_pre_block"],
                attn_head_output=archive["attn_head_output"],
                attn_branch_output=archive["attn_branch_output"],
                mlp_output=archive["mlp_output"],
                attn_edge_margin_proxy=archive["attn_edge_margin_proxy"],
                attn_edge_mass=archive["attn_edge_mass"],
                metadata=metadata,
            )
        artifact.validate()
        return artifact


def _integer_array(values: np.ndarray, name: str, *, rank: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != rank or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a rank-{rank} integer array")
    return array


def _float_array(values: np.ndarray, name: str, *, rank: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != rank or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be a rank-{rank} floating array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array
