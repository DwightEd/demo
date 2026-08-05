from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .interventions import CausalInterventionRunner
from .pairs import load_pair_file
from .trace_data import load_decision_prefix


@dataclass(frozen=True)
class InterventionExperimentConfig:
    data_root: Path
    domains: tuple[str, ...]
    layers: tuple[int, ...]
    pair_directory: str = "causal_first_error_v1"
    max_cases_per_domain: int = 0
    overwrite: bool = False


class InterventionExperiment:
    """Run controlled branch composition; formal gates wait for null controls."""

    def __init__(self, config: InterventionExperimentConfig) -> None:
        self.config = config

    def run(self, model: object) -> dict[str, Any]:
        jobs = self._jobs()
        written = []
        reused = []
        for pair, trace_path, layer, output_path in tqdm(
            jobs, desc="factorial interventions", unit="case-layer"
        ):
            if output_path.is_file() and not self.config.overwrite:
                reused.append(output_path.as_posix())
                continue
            recipient = load_decision_prefix(
                trace_path,
                record_index=pair.error_trace_record,
                decision_position=pair.decision_position,
            )
            donor = load_decision_prefix(
                trace_path,
                record_index=int(pair.counterfactual_trace_record),
                decision_position=int(pair.counterfactual_decision_position),
            )
            desired, baseline = pair.outcome_token_ids()
            result = CausalInterventionRunner(layer=layer).run(
                model=model,
                recipient_input_ids=recipient.input_ids,
                donor_input_ids=donor.input_ids,
                correct_token_id=desired,
                wrong_token_id=baseline,
                pair_kind=pair.pair_kind,
            )
            payload = {
                "schema": "controlled_root_v1",
                "case_id": pair.case_id,
                "dataset": pair.dataset,
                "problem_hash": pair.problem_hash,
                "pair_kind": pair.pair_kind,
                "layer": layer,
                "factorial": asdict(result.factorial),
                "effects": asdict(result.effects),
                "pre_state_margin": result.pre_state_margin,
                "pre_state_effect": result.pre_state_margin - result.factorial.m00,
                "claim_scope": result.claim_scope,
                "formal_gate_status": "controls_pending",
                "missing_controls": [
                    "random_donor",
                    "wrong_source",
                    "wrong_head_or_layer",
                    "norm_matched_noise",
                ],
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            written.append(output_path.as_posix())
        return {
            "case_layers": len(jobs),
            "written": written,
            "reused": reused,
            "formal_claim": "not_evaluated_until_controls_exist",
        }

    def _jobs(self):
        jobs = []
        for domain in self.config.domains:
            selected = self.config.data_root / domain / "selected"
            pair_path = (
                selected
                / self.config.pair_directory
                / "onset_pairs_v1.jsonl"
            )
            pairs, errors = load_pair_file(pair_path)
            if errors:
                raise ValueError(f"{pair_path}: invalid pair records: {errors}")
            controlled = [pair for pair in pairs if pair.root_cause_eligible]
            if self.config.max_cases_per_domain:
                controlled = controlled[: self.config.max_cases_per_domain]
            output_dir = selected / self.config.pair_directory / "interventions"
            for pair in controlled:
                for layer in self.config.layers:
                    jobs.append(
                        (
                            pair,
                            selected / "trace.npz",
                            layer,
                            output_dir
                            / f"case_{pair.case_id}.layer_{layer}.controlled_root_v1.json",
                        )
                    )
        return jobs


def summarize_saved_interventions(
    data_root: str | Path,
    domains: tuple[str, ...],
    *,
    pair_directory: str = "causal_first_error_v1",
) -> dict[str, Any]:
    payloads = []
    for domain in domains:
        folder = Path(data_root) / domain / "selected" / pair_directory / "interventions"
        for path in sorted(folder.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return {
        "intervention_files": len(payloads),
        "formal_claim": "not_evaluated_until_controls_exist",
        "claim_scopes": sorted(
            {str(payload.get("claim_scope", "unknown")) for payload in payloads}
        ),
        "missing_controls": sorted(
            {
                control
                for payload in payloads
                for control in payload.get("missing_controls", [])
            }
        ),
    }


__all__ = [
    "InterventionExperiment",
    "InterventionExperimentConfig",
    "summarize_saved_interventions",
]
