from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..profiler import MechanismProfiler
from .data import PullbackSource, prepare_record_trace
from .field import ConditionalFieldBank
from .replay import compute_causal_pullback
from .schema import (
    CausalPullbackArtifact,
    CausalPullbackConfig,
    CausalPullbackItem,
)


class CausalPullbackAccumulator:
    """Checkpointable collection of successful and failed replay jobs."""

    def __init__(self, artifact: CausalPullbackArtifact | None = None) -> None:
        self.items = list(artifact.items) if artifact is not None else []
        self.skipped = list(artifact.skipped) if artifact is not None else []

    @property
    def completed_original_indices(self) -> set[int]:
        """Return only durable successes.

        Failed rows remain in the checkpoint as diagnostics, but they must be
        retried on ``--resume`` because the user may have changed the batch
        size, sequence limit, or available GPU memory.
        """

        return {int(item.original_index) for item in self.items}

    def _clear_skip(self, original_index: int) -> None:
        original_index = int(original_index)
        self.skipped = [
            row
            for row in self.skipped
            if int(row.get("original_index", -1)) != original_index
        ]

    def add(self, item: CausalPullbackItem) -> None:
        item.validate()
        self._clear_skip(item.original_index)
        self.items.append(item)

    def add_skip(
        self,
        *,
        original_index: int,
        chain_idx: int,
        problem_id: Any,
        reason: str,
        detail: str,
    ) -> None:
        self._clear_skip(original_index)
        self.skipped.append(
            {
                "original_index": int(original_index),
                "chain_idx": int(chain_idx),
                "problem_id": str(problem_id),
                "reason": str(reason),
                "detail": str(detail)[:1000],
            }
        )

    def artifact(self, metadata: dict[str, Any]) -> CausalPullbackArtifact:
        return CausalPullbackArtifact(
            items=sorted(self.items, key=lambda item: item.original_index),
            metadata=dict(metadata),
            skipped=list(self.skipped),
        )


def extract_causal_pullback_item(
    model,
    tokenizer,
    source: PullbackSource,
    field_bank: ConditionalFieldBank,
    dataset_index: int,
    cfg: CausalPullbackConfig,
    *,
    ordered_questions: list[str] | None,
    max_seq_len: int,
) -> CausalPullbackItem:
    """Replay and causally probe one same-problem response."""

    dataset_index = int(dataset_index)
    dataset = source.dataset
    original_index = int(dataset.original_indices[dataset_index])
    record = source.records_by_original_index[original_index]
    witnesses = field_bank.witnesses(dataset_index)
    if witnesses is None:
        raise ValueError("insufficient_correct_same_problem_donors")
    trace, _ = prepare_record_trace(
        record,
        tokenizer,
        prompt_style=source.prompt_style,
        ordered_questions=ordered_questions,
        max_seq_len=int(max_seq_len),
    )
    states = np.asarray(dataset.trajectories[dataset_index][:, 0, :], dtype=np.float32)
    replay = compute_causal_pullback(model, trace, states, witnesses, cfg)
    return CausalPullbackItem(
        chain_idx=int(record.chain_idx),
        original_index=original_index,
        problem_id=int(dataset.problem_ids[dataset_index]),
        sample_idx=int(dataset.sample_idx[dataset_index]),
        is_correct=int(dataset.is_correct[dataset_index]),
        n_steps=int(dataset.n_steps[dataset_index]),
        response_chars=int(dataset.response_chars[dataset_index]),
        layer=int(cfg.layer),
        donor_count=int(witnesses.donor_count),
        replay_kind=str(trace["replay_kind"]),
        replay_cosine=replay.replay_cosine,
        baseline_step_features=replay.baseline_step_features,
        field_energy=witnesses.field_energy,
        field_calibrated_energy=witnesses.field_calibrated_energy,
        witness_norms=replay.witness_norms,
        fisher_transfer=replay.fisher_transfer,
        chosen_logprob_transfer=replay.chosen_logprob_transfer,
        entropy_transfer=replay.entropy_transfer,
        primary_half_fisher_transfer=replay.primary_half_fisher_transfer,
        perturbation_scale=replay.perturbation_scale,
        metadata={
            **replay.metadata,
            "sequence_tokens": int(len(trace["input_ids"])),
            "gold_error_step": int(record.gold_error_step),
            "dataset": str(record.dataset or ""),
            "generator": str(record.generator or ""),
        },
    )


def extraction_metadata(
    source: PullbackSource,
    cfg: CausalPullbackConfig,
    *,
    observer_model: str,
    tokenizer_name: str,
    device: str,
    dtype: str,
    complete: bool,
) -> dict[str, Any]:
    exact = all(
        source.records_by_original_index[int(index)].exact_input_ids is not None
        for index in source.dataset.original_indices
    )
    return {
        "schema": "causal_pullback_flow_v1",
        "method": "Causal Pullback Flow Field",
        "observer_model": str(observer_model),
        "source_model": str(source.model_name),
        "tokenizer": str(tokenizer_name),
        "device": str(device),
        "dtype": str(dtype),
        "source_path": str(source.dataset.source_path),
        "vector_key": str(source.dataset.vector_key),
        "label_policy": str(source.dataset.label_policy),
        "layer": int(cfg.layer),
        "config": asdict(cfg),
        "full_logits_persisted": False,
        "complete": bool(complete),
        "evidence_tier": (
            "exact_trace_candidate" if exact else "legacy_replay_validated_exploration"
        ),
        "teacher_regime": (
            "same_problem_correct_ensemble_reference; not a deployable single-chain detector"
        ),
        "causal_operator_axis": (
            "observed transition destination step to strictly later output steps"
        ),
        "tangent_identification": (
            "empirical ambient-coordinate identification from hidden displacement "
            "direction to residual-state intervention direction"
        ),
    }


def run_causal_pullback_extraction(
    model,
    tokenizer,
    source: PullbackSource,
    cfg: CausalPullbackConfig,
    *,
    output_path: str | Path,
    observer_model: str,
    ordered_questions: list[str] | None,
    max_seq_len: int,
    checkpoint_every: int = 10,
    resume: bool = False,
    profiler: MechanismProfiler | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> CausalPullbackArtifact:
    """Run extraction with durable checkpoints and explicit skip records."""

    cfg.validate()
    output_path = Path(output_path)
    previous = (
        CausalPullbackArtifact.load(output_path)
        if bool(resume) and output_path.exists()
        else None
    )
    accumulator = CausalPullbackAccumulator(previous)
    completed = accumulator.completed_original_indices
    field_bank = ConditionalFieldBank.build(source.dataset, cfg)
    profiler = profiler or MechanismProfiler()
    total = source.dataset.n_samples

    def metadata(complete: bool) -> dict[str, Any]:
        return extraction_metadata(
            source,
            cfg,
            observer_model=observer_model,
            tokenizer_name=str(getattr(tokenizer, "name_or_path", "")),
            device=str(next(model.parameters()).device),
            dtype=str(next(model.parameters()).dtype),
            complete=complete,
        )

    attempted_since_checkpoint = 0
    for dataset_index in range(total):
        original_index = int(source.dataset.original_indices[dataset_index])
        if original_index in completed:
            if progress is not None:
                progress(dataset_index + 1, total)
            continue
        record = source.records_by_original_index[original_index]
        profiler.record_chain()
        try:
            with profiler.phase("field_and_causal_replay"):
                item = extract_causal_pullback_item(
                    model,
                    tokenizer,
                    source,
                    field_bank,
                    dataset_index,
                    cfg,
                    ordered_questions=ordered_questions,
                    max_seq_len=max_seq_len,
                )
            accumulator.add(item)
            profiler.record_success()
            profiler.record_seq_len(item.metadata.get("sequence_tokens"))
        except Exception as exc:
            accumulator.add_skip(
                original_index=original_index,
                chain_idx=record.chain_idx,
                problem_id=source.dataset.problem_ids[dataset_index],
                reason=type(exc).__name__,
                detail=str(exc),
            )
            profiler.record_skip(type(exc).__name__)
        attempted_since_checkpoint += 1
        if checkpoint_every > 0 and attempted_since_checkpoint >= checkpoint_every:
            accumulator.artifact(metadata(False)).save(output_path)
            attempted_since_checkpoint = 0
        if progress is not None:
            progress(dataset_index + 1, total)

    artifact = accumulator.artifact(metadata(True))
    artifact.save(output_path)
    return artifact
