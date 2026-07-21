from __future__ import annotations

from typing import Any

import numpy as np

from .config import SourceConfig
from .raw_residual import _RawSource, _load_shard, _resolve_source


class RawResidualRepository:
    """Audited manifest and memory-mapped shard access behind one typed boundary."""

    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        self._source = _resolve_source(
            config.manifest, config.hidden_dir, config.response_generator
        )

    @property
    def source(self) -> _RawSource:
        return self._source

    def shard(self, row: int) -> np.ndarray:
        return _load_shard(self._source, row)

    def inspect(self) -> dict[str, Any]:
        source = self._source
        first = self.shard(0)
        return {
            "manifest_path": str(source.manifest_path),
            "source_format": source.source_format,
            "snapshot_kind": source.snapshot_kind,
            "n_manifest_records": source.n_manifest_records,
            "n_records": int(source.gold_error_step.size),
            "n_error_records": int(np.sum(source.gold_error_step >= 0)),
            "n_correct_records": int(np.sum(source.gold_error_step < 0)),
            "layers": source.layers.tolist(),
            "depth_semantics": (
                "adjacent_block"
                if np.all(np.diff(source.layers) == 1)
                else "sparse_depth_interval"
            ),
            "first_shard": str(source.files[0]),
            "first_shard_shape": list(first.shape),
            "response_generator_filter": source.generator_filter,
            "generator_field": source.generator_field,
            "response_generators": (
                []
                if source.response_generators is None
                else sorted({str(value) for value in source.response_generators})
            ),
        }
