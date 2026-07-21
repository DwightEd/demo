from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import OperatorFieldAnalyzer
from .config import RunConfig, SourceConfig
from .domain import DatasetResult, ExperimentResult
from .matching import MatchedWindowBuilder
from .progress import NullProgress, ProgressReporter
from .reporting import ArtifactWriter
from .source import RawResidualRepository
from .statistics import paired_auc_difference, paired_summary


class ExperimentRunner:
    """Application service for one audited raw-residual experiment."""

    def __init__(
        self,
        source: SourceConfig,
        config: RunConfig,
        output_dir: str | Path,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.source = source
        self.config = config
        self.writer = ArtifactWriter(output_dir)
        self.progress = progress or NullProgress()

    def run(self) -> ExperimentResult:
        config, source, progress = self.config, self.source, self.progress
        progress.stage("load", source.manifest.stem)
        repository = RawResidualRepository(source)
        data = MatchedWindowBuilder(repository, config, progress).build()
        progress.stage("analyze", f"{data.metadata.cohort.retained_pairs} matched pairs")
        scores, diagnostics = OperatorFieldAnalyzer(config, progress).analyze(data)
        progress.stage("statistics", f"{config.bootstrap} bootstrap replicates")
        metrics = {
            name: paired_summary(
                values,
                data.labels,
                data.pair_ids,
                n_boot=config.bootstrap,
                seed=config.seed + index,
            )
            for index, (name, values) in enumerate(scores.items())
        }
        comparisons = {
            "plaquette_minus_radial": paired_auc_difference(
                scores["plaquette_observed_disagreement"],
                scores["radial_edge_change"],
                data.labels,
                data.pair_ids,
                n_boot=config.bootstrap,
                seed=config.seed + 101,
            ),
            "plaquette_minus_time_residual": paired_auc_difference(
                scores["plaquette_observed_disagreement"],
                scores["time_operator_residual"],
                data.labels,
                data.pair_ids,
                n_boot=config.bootstrap,
                seed=config.seed + 102,
            ),
        }
        result = ExperimentResult(
            dataset=DatasetResult(
                metadata=data.metadata,
                pairs=int(np.unique(data.pair_ids).size),
                time_offsets=tuple(int(value) for value in data.time_offsets),
                layer_ids=tuple(int(value) for value in data.layer_ids),
                hidden_dim=int(data.states.shape[-1]),
                diagnostics=diagnostics,
                metrics=metrics,
                comparisons=comparisons,
            ),
            seed=config.seed,
            bootstrap=config.bootstrap,
            rank=config.rank,
            ridge_alpha=config.ridge_alpha,
        )
        progress.stage("write", str(self.writer.output_dir))
        self.writer.write(result, data, scores)
        return result
