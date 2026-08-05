from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .contracts import OnsetPair, OnsetTraceArtifact
from .pairs import load_pair_file
from .replay import DecisionTraceExtractor
from .trace_data import load_decision_prefix


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class OnsetTraceExtractionConfig:
    data_root: Path
    domains: tuple[str, ...]
    layers: tuple[int, ...]
    model_name: str
    model_revision: str
    tokenizer_name: str
    tokenizer_revision: str
    pair_directory: str = "causal_first_error_v1"
    topk: int = 20
    max_cases_per_domain: int = 0
    overwrite: bool = False


@dataclass(frozen=True)
class _TraceJob:
    pair: OnsetPair
    trace_path: Path
    pair_path: Path
    record_index: int
    decision_position: int
    trajectory_role: str
    output_path: Path


class OnsetTraceExtraction:
    """Extract future-free token/head/source graphs for verified onset pairs."""

    def __init__(self, config: OnsetTraceExtractionConfig) -> None:
        self.config = config

    def run(self, model: object) -> dict[str, Any]:
        extractor = DecisionTraceExtractor(
            layers=self.config.layers, topk=self.config.topk
        )
        jobs = self._jobs()
        written: list[str] = []
        reused: list[str] = []
        fingerprints: dict[Path, str] = {}

        def fingerprint(path: Path) -> str:
            if path not in fingerprints:
                fingerprints[path] = file_sha256(path)
            return fingerprints[path]

        for job in tqdm(
            jobs, desc="onset decision traces", unit="trace"
        ):
            if job.output_path.is_file() and not self.config.overwrite:
                OnsetTraceArtifact.load(job.output_path)
                reused.append(job.output_path.as_posix())
                continue
            prefix = load_decision_prefix(
                job.trace_path,
                record_index=job.record_index,
                decision_position=job.decision_position,
            )
            desired_token, baseline_token = job.pair.outcome_token_ids()
            artifact = extractor.extract(
                model=model,
                input_ids=prefix.input_ids,
                source_step_ids=prefix.source_step_ids,
                correct_token_id=desired_token,
                wrong_token_id=baseline_token,
                metadata={
                    "case_id": job.pair.case_id,
                    "pair_kind": job.pair.pair_kind,
                    "allowed_claim": job.pair.allowed_claim,
                    "trajectory_role": job.trajectory_role,
                    "trace_record": job.record_index,
                    "model_name": self.config.model_name,
                    "model_revision_or_unknown": self.config.model_revision,
                    "tokenizer_name": self.config.tokenizer_name,
                    "tokenizer_revision_or_unknown": self.config.tokenizer_revision,
                    "source_trace_fingerprint": fingerprint(job.trace_path),
                    "pair_file_fingerprint": fingerprint(job.pair_path),
                },
            )
            artifact.save(job.output_path)
            written.append(job.output_path.as_posix())
        return {
            "selected_pairs": len({job.pair.case_id for job in jobs}),
            "selected_traces": len(jobs),
            "written": written,
            "reused": reused,
            "artifact_schema": "onset_trace_v1",
        }

    def _jobs(self):
        jobs = []
        for domain in self.config.domains:
            selected = self.config.data_root / domain / "selected"
            trace_path = selected / "trace.npz"
            pair_path = (
                selected
                / self.config.pair_directory
                / "onset_pairs_v1.jsonl"
            )
            pairs, errors = load_pair_file(pair_path)
            if errors:
                raise ValueError(f"{pair_path}: invalid pair records: {errors}")
            if self.config.max_cases_per_domain:
                pairs = pairs[: self.config.max_cases_per_domain]
            output_dir = selected / self.config.pair_directory / "onset_traces"
            for pair in pairs:
                if pair.dataset != domain:
                    raise ValueError(
                        f"{pair.case_id}: pair dataset disagrees with directory"
                    )
                records = [
                    ("recipient", pair.error_trace_record, pair.decision_position)
                ]
                if pair.root_cause_eligible:
                    records.append(
                        (
                            "counterfactual",
                            int(pair.counterfactual_trace_record),
                            int(pair.counterfactual_decision_position),
                        )
                    )
                for role, record_index, decision_position in records:
                    jobs.append(
                        _TraceJob(
                            pair=pair,
                            trace_path=trace_path,
                            pair_path=pair_path,
                            record_index=record_index,
                            decision_position=decision_position,
                            trajectory_role=role,
                            output_path=output_dir
                            / (
                                f"case_{pair.case_id}.{role}."
                                "onset_trace_v1.npz"
                            ),
                        )
                    )
        return jobs


__all__ = [
    "OnsetTraceExtraction",
    "OnsetTraceExtractionConfig",
    "file_sha256",
]
