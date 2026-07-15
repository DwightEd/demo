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

    def retain_original_indices(self, original_indices: set[int]) -> None:
        """Keep a resumed checkpoint aligned with the current target cohort."""

        allowed = {int(value) for value in original_indices}
        self.items = [
            item for item in self.items if int(item.original_index) in allowed
        ]
        self.skipped = [
            row
            for row in self.skipped
            if int(row.get("original_index", -1)) in allowed
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


def select_replay_targets(
    problem_ids: np.ndarray,
    y_error: np.ndarray,
    sample_idx: np.ndarray,
    eligible_indices: np.ndarray,
    *,
    max_targets: int,
    seed: int,
) -> np.ndarray:
    """Select an optional pilot without truncating the donor reference bank.

    Contrastive problems contribute one error/correct pair first. Remaining
    slots are filled round-robin across problems. This makes a small pilot
    useful for both replay diagnostics and preliminary within-problem checks.
    """

    eligible = np.asarray(eligible_indices, dtype=np.int64)
    if eligible.size == 0:
        return eligible
    if int(max_targets) <= 0 or int(max_targets) >= eligible.size:
        return np.sort(eligible)

    problem_ids = np.asarray(problem_ids)
    y_error = np.asarray(y_error, dtype=np.int8)
    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    groups: dict[Any, list[int]] = {}
    for index in eligible:
        raw = problem_ids[int(index)]
        problem = raw.item() if isinstance(raw, np.generic) else raw
        groups.setdefault(problem, []).append(int(index))
    for rows in groups.values():
        rows.sort(key=lambda index: (int(sample_idx[index]), index))

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    problems = list(groups)
    rng.shuffle(problems)
    selected: list[int] = []
    selected_set: set[int] = set()

    # Preserve the same-problem comparison in a small pilot.
    for problem in problems:
        if len(selected) + 2 > int(max_targets):
            break
        rows = groups[problem]
        errors = [index for index in rows if y_error[index] == 1]
        correct = [index for index in rows if y_error[index] == 0]
        if not errors or not correct:
            continue
        pair = (errors[0], correct[0])
        selected.extend(pair)
        selected_set.update(pair)

    remaining = {
        problem: [index for index in groups[problem] if index not in selected_set]
        for problem in problems
    }
    while len(selected) < int(max_targets):
        advanced = False
        for problem in problems:
            rows = remaining[problem]
            if not rows:
                continue
            selected.append(rows.pop(0))
            advanced = True
            if len(selected) >= int(max_targets):
                break
        if not advanced:
            break
    return np.asarray(selected, dtype=np.int64)


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
    source_sample_count: int,
    donor_eligible_count: int,
    target_count: int,
    max_targets: int,
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
        "source_sample_count": int(source_sample_count),
        "donor_eligible_count": int(donor_eligible_count),
        "target_count": int(target_count),
        "max_targets": int(max_targets),
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
    max_targets: int = 0,
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
    field_bank = ConditionalFieldBank.build(source.dataset, cfg)
    eligible = field_bank.eligible_target_indices()
    target_indices = select_replay_targets(
        source.dataset.problem_ids,
        source.dataset.y_error,
        source.dataset.sample_idx,
        eligible,
        max_targets=int(max_targets),
        seed=cfg.random_seed,
    )
    if target_indices.size == 0:
        raise ValueError(
            "no target has enough same-problem correct donors; lower --min_donors "
            "only as a declared ablation or inspect the label policy"
        )
    target_original_indices = {
        int(source.dataset.original_indices[int(index)]) for index in target_indices
    }
    accumulator.retain_original_indices(target_original_indices)
    completed = accumulator.completed_original_indices
    profiler = profiler or MechanismProfiler()
    total = int(target_indices.size)

    def metadata(complete: bool) -> dict[str, Any]:
        return extraction_metadata(
            source,
            cfg,
            observer_model=observer_model,
            tokenizer_name=str(getattr(tokenizer, "name_or_path", "")),
            device=str(next(model.parameters()).device),
            dtype=str(next(model.parameters()).dtype),
            complete=complete,
            source_sample_count=source.dataset.n_samples,
            donor_eligible_count=int(eligible.size),
            target_count=total,
            max_targets=int(max_targets),
        )

    attempted_since_checkpoint = 0
    for target_position, dataset_index in enumerate(target_indices):
        dataset_index = int(dataset_index)
        original_index = int(source.dataset.original_indices[dataset_index])
        if original_index in completed:
            if progress is not None:
                progress(target_position + 1, total)
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
            progress(target_position + 1, total)

    artifact = accumulator.artifact(metadata(True))
    artifact.save(output_path)
    return artifact
