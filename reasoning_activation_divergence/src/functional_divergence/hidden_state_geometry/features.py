from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..progress import NullProgress, ProgressReporter
from .method import FoldInput
from .representation import ChainBalancedPCA, FunctionalEncoder
from .tasks import TaskExample, load_visible_states, nuisance_features


ChainKey = tuple[str, int]


def _chain_key(example: TaskExample) -> ChainKey:
    return example.sample.dataset, example.sample.chain_id


@dataclass(frozen=True)
class EncodedRows:
    nuisance: np.ndarray
    output: np.ndarray
    hidden: np.ndarray


@dataclass(frozen=True)
class EncodedPartition:
    examples: tuple[TaskExample, ...]
    rows: EncodedRows
    projection_cache: dict[ChainKey, np.ndarray]

    def null_hidden(
        self,
        encoder: FunctionalEncoder,
        *,
        reporter: ProgressReporter | None = None,
        description: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        progress = reporter or NullProgress()
        time, layer = [], []
        tracked = progress.track(
            self.examples, total=len(self.examples), description=description
        )
        for example in tracked:
            projected = self.projection_cache[_chain_key(example)]
            time.append(
                encoder.hidden_tensor(example, null="time", projected=projected)
            )
            layer.append(
                encoder.hidden_tensor(example, null="layer", projected=projected)
            )
        return np.stack(time, axis=0), np.stack(layer, axis=0)


@dataclass(frozen=True)
class EncodedFold:
    projector: ChainBalancedPCA
    encoder: FunctionalEncoder
    train: EncodedPartition
    test: EncodedPartition


class FunctionalFeatureBuilder:
    """Fit fold-local projection and encode raw time-layer residual streams."""

    def __init__(
        self,
        *,
        pca_dim: int,
        time_basis: int,
        layer_basis: int,
        positions_per_chain: int,
        seed: int,
    ) -> None:
        self.pca_dim = int(pca_dim)
        self.time_basis = int(time_basis)
        self.layer_basis = int(layer_basis)
        self.positions_per_chain = int(positions_per_chain)
        self.seed = int(seed)
        self.projector: ChainBalancedPCA | None = None
        self.encoder: FunctionalEncoder | None = None

    def fit(
        self,
        examples: tuple[TaskExample, ...],
        *,
        reporter: ProgressReporter | None = None,
    ) -> "FunctionalFeatureBuilder":
        progress = reporter or NullProgress()
        self.projector = ChainBalancedPCA(
            dim=self.pca_dim,
            positions_per_chain=self.positions_per_chain,
            seed=self.seed,
        ).fit(examples, progress=progress)
        self.encoder = FunctionalEncoder(
            self.projector,
            time_basis=self.time_basis,
            layer_basis=self.layer_basis,
            null_seed=self.seed,
        )
        return self

    def transform(
        self,
        examples: Iterable[TaskExample],
        *,
        reporter: ProgressReporter | None = None,
        projection_description: str,
        encoding_description: str,
    ) -> EncodedPartition:
        if self.projector is None or self.encoder is None:
            raise RuntimeError("feature builder is not fitted")
        rows = tuple(examples)
        progress = reporter or NullProgress()
        cache = self._project(rows, progress, projection_description)
        encoded = []
        tracked = progress.track(
            rows, total=len(rows), description=encoding_description
        )
        for example in tracked:
            projected = cache[_chain_key(example)]
            encoded.append(
                (
                    nuisance_features(example)[1],
                    self.encoder.output_features(example),
                    self.encoder.hidden_tensor(example, projected=projected),
                )
            )
        if not encoded:
            raise ValueError("cannot encode an empty partition")
        columns = tuple(np.stack(values, axis=0) for values in zip(*encoded))
        return EncodedPartition(rows, EncodedRows(*columns), cache)

    def build(self, fold: FoldInput) -> EncodedFold:
        progress = fold.progress or NullProgress()
        progress.stage("projection", fold.task_name)
        self.fit(fold.train_examples, reporter=progress)
        progress.stage("encode", fold.task_name)
        train = self.transform(
            fold.train_examples,
            reporter=progress,
            projection_description="train projected chains",
            encoding_description="train examples",
        )
        test = self.transform(
            fold.test_examples,
            reporter=progress,
            projection_description="test projected chains",
            encoding_description="test examples",
        )
        if self.projector is None or self.encoder is None:
            raise RuntimeError("feature builder unexpectedly lost fitted state")
        return EncodedFold(self.projector, self.encoder, train, test)

    def _project(
        self,
        examples: tuple[TaskExample, ...],
        reporter: ProgressReporter,
        description: str,
    ) -> dict[ChainKey, np.ndarray]:
        if self.projector is None:
            raise RuntimeError("feature builder is not fitted")
        latest: dict[ChainKey, TaskExample] = {}
        for example in examples:
            key = _chain_key(example)
            if key not in latest or example.visible_steps > latest[key].visible_steps:
                latest[key] = example
        cache = {}
        tracked = reporter.track(
            latest.items(), total=len(latest), description=description
        )
        for key, example in tracked:
            cache[key] = self.projector.transform(load_visible_states(example))
        return cache
