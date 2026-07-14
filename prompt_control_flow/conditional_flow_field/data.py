from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np

from ..flow_signature_data import FlowTrajectoryDataset
from ..feasible_tangent.data import build_problem_supports, problem_key
from .schema import ConditionalFlowFieldConfig


@dataclass(frozen=True)
class FieldProblemSupport:
    problem_id: Hashable
    sample_indices: tuple[int, ...]
    correct_indices: tuple[int, ...]
    donor_count: int


def build_field_supports(
    dataset: FlowTrajectoryDataset,
    cfg: ConditionalFlowFieldConfig,
) -> dict[Hashable, FieldProblemSupport]:
    """Build supports with one label-independent donor count per problem.

    Correct targets need leave-one-out references, so every target from the
    same problem receives at most ``n_correct - 1`` donors. Incorrect targets
    are deliberately downsampled to that same count.
    """

    supports = build_problem_supports(dataset)
    output: dict[Hashable, FieldProblemSupport] = {}
    for raw_problem, support in supports.items():
        donor_count = min(int(cfg.max_donors), len(support.correct_indices) - 1)
        output[raw_problem] = FieldProblemSupport(
            problem_id=raw_problem,
            sample_indices=support.sample_indices,
            correct_indices=support.correct_indices,
            donor_count=max(0, donor_count),
        )
    return output


def wrong_problem_order(
    dataset: FlowTrajectoryDataset,
    supports: dict[Hashable, FieldProblemSupport],
) -> dict[Hashable, tuple[Hashable, ...]]:
    """Order wrong-problem controls by response-size difficulty only."""

    eligible = [support for support in supports.values() if support.donor_count > 0]
    if not eligible:
        return {key: tuple() for key in supports}
    controls = []
    for support in eligible:
        index = np.asarray(support.sample_indices, dtype=np.int64)
        controls.append(
            [
                np.median(np.log1p(dataset.n_steps[index])),
                np.median(np.log1p(dataset.response_chars[index])),
            ]
        )
    controls = np.asarray(controls, dtype=np.float64)
    center = np.median(controls, axis=0)
    scale = 1.4826 * np.median(np.abs(controls - center), axis=0)
    fallback = np.std(controls, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
    standardized = (controls - center) / scale
    output: dict[Hashable, tuple[Hashable, ...]] = {}
    for target in supports.values():
        index = np.asarray(target.sample_indices, dtype=np.int64)
        target_control = np.asarray(
            [
                np.median(np.log1p(dataset.n_steps[index])),
                np.median(np.log1p(dataset.response_chars[index])),
            ],
            dtype=np.float64,
        )
        distance = np.linalg.norm(
            standardized - ((target_control - center) / scale)[None, :], axis=1
        )
        order = np.argsort(distance, kind="stable")
        output[target.problem_id] = tuple(
            eligible[int(position)].problem_id
            for position in order
            if eligible[int(position)].problem_id != target.problem_id
        )
    return output


def conditional_flow_field_preflight(
    dataset: FlowTrajectoryDataset,
    cfg: ConditionalFlowFieldConfig,
) -> dict[str, Any]:
    supports = build_field_supports(dataset, cfg)
    eligible = [x for x in supports.values() if x.donor_count >= cfg.min_donors]
    eligible_samples = sum(len(x.sample_indices) for x in eligible)
    contrastive = 0
    eligible_contrastive = 0
    for support in supports.values():
        index = np.asarray(support.sample_indices, dtype=np.int64)
        labels = dataset.y_error[index]
        is_contrastive = bool(np.any(labels == 0) and np.any(labels == 1))
        contrastive += int(is_contrastive)
        eligible_contrastive += int(
            is_contrastive and support.donor_count >= cfg.min_donors
        )
    return {
        "path": dataset.source_path,
        "vector_key": dataset.vector_key,
        "samples": dataset.n_samples,
        "errors": int(dataset.y_error.sum()),
        "correct": int((dataset.y_error == 0).sum()),
        "problems": len(supports),
        "contrastive_problems": contrastive,
        "eligible_problems": len(eligible),
        "eligible_contrastive_problems": eligible_contrastive,
        "eligible_sample_coverage": eligible_samples / max(dataset.n_samples, 1),
        "layers": dataset.layer_ids.tolist(),
        "hidden_dim": dataset.hidden_dim,
        "min_donors": cfg.min_donors,
        "max_donors": cfg.max_donors,
        "state_window": cfg.state_window,
        "donor_count_is_target_label_independent": True,
        "skipped": dataset.skipped,
    }
