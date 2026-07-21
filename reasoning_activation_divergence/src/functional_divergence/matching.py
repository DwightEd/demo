from __future__ import annotations

from .config import RunConfig
from .domain import LayerTimeDataset
from .progress import NullProgress, ProgressReporter
from .raw_residual import _load_matched_from_source
from .source import RawResidualRepository


class MatchedWindowBuilder:
    """Build matched event windows without exposing manifest metadata dictionaries."""

    def __init__(
        self,
        repository: RawResidualRepository,
        config: RunConfig,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.repository = repository
        self.config = config
        self.progress = progress or NullProgress()

    def build(self) -> LayerTimeDataset:
        config = self.config
        return _load_matched_from_source(
            self.repository.source,
            offsets=config.offsets,
            layers=config.layers,
            max_pairs=config.max_pairs,
            progress=self.progress,
        )
