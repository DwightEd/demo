from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .contracts import CounterfactualExample
from .features import paired_token_signatures
from .one_class import ShrunkMahalanobis


@dataclass(frozen=True)
class ScoreBundle:
    state: np.ndarray
    relation: np.ndarray
    context_state: np.ndarray
    context_relation: np.ndarray
    context: np.ndarray
    fused: np.ndarray
    fused_percentile: np.ndarray


class _CalibratedScorer:
    def __init__(
        self,
        *,
        shrinkage: float,
        max_features: int,
        projection_seed: int,
    ) -> None:
        self.model = ShrunkMahalanobis(
            shrinkage=shrinkage,
            max_features=max_features,
            projection_seed=projection_seed,
        )
        self.location_: float | None = None
        self.scale_: float | None = None

    def fit(self, values: np.ndarray) -> None:
        self.model.fit(values)
        reference_scores = self.model.score_samples(values)
        location = float(np.median(reference_scores))
        mad = float(np.median(np.abs(reference_scores - location)))
        scale = max(1.4826 * mad, float(reference_scores.std()), 1e-8)
        self.location_ = location
        self.scale_ = scale

    def score(self, values: np.ndarray) -> np.ndarray:
        if self.location_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before score")
        raw = self.model.score_samples(values)
        return (raw - self.location_) / self.scale_


class MultiViewOneClassDetector:
    """Normal-reference scoring for state and context-influence components."""

    def __init__(
        self,
        *,
        shrinkage: float = 0.1,
        max_features: int = 256,
        projection_seed: int = 0,
    ) -> None:
        kwargs = {
            "shrinkage": shrinkage,
            "max_features": max_features,
            "projection_seed": projection_seed,
        }
        self.state = _CalibratedScorer(**kwargs)
        self.relation = _CalibratedScorer(**kwargs)
        self.context_state = _CalibratedScorer(**kwargs)
        self.context_relation = _CalibratedScorer(**kwargs)
        self.fitted_ = False
        self.reference_fused_: np.ndarray | None = None

    def fit(
        self,
        reference_examples: Iterable[CounterfactualExample],
    ) -> "MultiViewOneClassDetector":
        examples = tuple(reference_examples)
        if not examples:
            raise ValueError("at least one reference example is required")
        if any(not example.normal_reference for example in examples):
            raise ValueError(
                "every fit example must be explicitly marked normal_reference"
            )
        signatures = [paired_token_signatures(example) for example in examples]
        state_values = np.concatenate(
            [item.factual_state for item in signatures]
        )
        relation_values = np.concatenate(
            [item.factual_relation for item in signatures]
        )
        context_state_values = np.concatenate(
            [item.context_state_delta for item in signatures]
        )
        context_relation_values = np.concatenate(
            [item.context_relation_delta for item in signatures]
        )
        self.state.fit(state_values)
        self.relation.fit(relation_values)
        self.context_state.fit(context_state_values)
        self.context_relation.fit(context_relation_values)
        reference_components = np.column_stack(
            (
                self.state.score(state_values),
                self.relation.score(relation_values),
                self.context_state.score(context_state_values),
                self.context_relation.score(context_relation_values),
            )
        )
        self.reference_fused_ = np.sort(reference_components.mean(axis=1))
        self.fitted_ = True
        return self

    def score(self, example: CounterfactualExample) -> ScoreBundle:
        if not self.fitted_ or self.reference_fused_ is None:
            raise RuntimeError("fit must be called before score")
        signatures = paired_token_signatures(example)
        state = self.state.score(signatures.factual_state)
        relation = self.relation.score(signatures.factual_relation)
        context_state = self.context_state.score(
            signatures.context_state_delta
        )
        context_relation = self.context_relation.score(
            signatures.context_relation_delta
        )
        context = np.mean(
            np.column_stack((context_state, context_relation)),
            axis=1,
        )
        fused = np.mean(
            np.column_stack(
                (state, relation, context_state, context_relation)
            ),
            axis=1,
        )
        rank = np.searchsorted(self.reference_fused_, fused, side="right")
        fused_percentile = (rank + 1.0) / (len(self.reference_fused_) + 2.0)
        return ScoreBundle(
            state=state,
            relation=relation,
            context_state=context_state,
            context_relation=context_relation,
            context=context,
            fused=fused,
            fused_percentile=fused_percentile,
        )
