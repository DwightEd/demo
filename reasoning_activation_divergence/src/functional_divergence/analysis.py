from __future__ import annotations

from typing import Any

import numpy as np

from .config import RunConfig
from .domain import LayerTimeDataset
from .layer_time import crossfit_layer_time_scores
from .progress import NullProgress, ProgressReporter


class OperatorFieldAnalyzer:
    """Fit and score the control-only joint token-times-layer operator field."""

    def __init__(
        self, config: RunConfig, progress: ProgressReporter | None = None
    ) -> None:
        self.config = config
        self.progress = progress or NullProgress()

    def analyze(
        self, data: LayerTimeDataset
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        return crossfit_layer_time_scores(
            data,
            rank=self.config.rank,
            n_splits=self.config.folds,
            seed=self.config.seed,
            ridge_alpha=self.config.ridge_alpha,
            progress=self.progress,
        )
