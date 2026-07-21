from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..splitting import TraceMeta
from .contracts import CausalHypergraph, FirstErrorLabels


@dataclass(frozen=True)
class CausalTrace:
    trace_id: str
    problem_id: str
    generator_model: str
    observer_model: str
    layer_id: int
    prompt_tokens: int
    response_tokens: int
    graph: CausalHypergraph
    labels: FirstErrorLabels

    def __post_init__(self) -> None:
        if not self.trace_id or not self.problem_id:
            raise ValueError("trace_id and problem_id are required")
        if not self.generator_model or not self.observer_model:
            raise ValueError("generator_model and observer_model are required")
        if self.layer_id < 0 or self.prompt_tokens <= 0 or self.response_tokens <= 0:
            raise ValueError("layer and token counts are invalid")
        if len(self.graph.response_nodes) != self.labels.num_steps:
            raise ValueError("response_nodes must contain one node per reasoning step")

    @property
    def response_label(self) -> int:
        return int(self.labels.first_error >= 0)

    def split_record(self) -> TraceMeta:
        return TraceMeta(
            trace_id=self.trace_id,
            group_id=self.problem_id,
            group_is_fallback=False,
            split=None,
            response_label=self.response_label,
            gold_step=self.labels.first_error,
            num_steps=self.labels.num_steps,
            num_response_tokens=self.response_tokens,
            generator_model=self.generator_model,
        )


class TraceRepository:
    """Versioned, pickle-free storage for causal traces."""

    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, trace: CausalTrace) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{trace.trace_id}.npz"
        temporary = path.with_suffix(".npz.tmp")
        graph = trace.graph
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                schema_version=np.asarray(self.SCHEMA_VERSION, dtype=np.int64),
                trace_id=np.asarray(trace.trace_id),
                problem_id=np.asarray(trace.problem_id),
                generator_model=np.asarray(trace.generator_model),
                observer_model=np.asarray(trace.observer_model),
                layer_id=np.asarray(trace.layer_id, dtype=np.int64),
                prompt_tokens=np.asarray(trace.prompt_tokens, dtype=np.int64),
                response_tokens=np.asarray(trace.response_tokens, dtype=np.int64),
                num_steps=np.asarray(trace.labels.num_steps, dtype=np.int64),
                first_error=np.asarray(trace.labels.first_error, dtype=np.int64),
                node_features=graph.node_features,
                incidence=graph.incidence,
                receivers=graph.receivers,
                edge_features=graph.edge_features,
                edge_kind=graph.edge_kind,
                response_nodes=graph.response_nodes,
            )
        os.replace(temporary, path)
        return path

    def load(self, path: str | Path) -> CausalTrace:
        with np.load(Path(path), allow_pickle=False) as archive:
            version = int(archive["schema_version"])
            if version != self.SCHEMA_VERSION:
                raise ValueError(f"unsupported causal trace schema {version}")
            graph = CausalHypergraph(
                node_features=archive["node_features"],
                incidence=archive["incidence"],
                receivers=archive["receivers"],
                edge_features=archive["edge_features"],
                edge_kind=archive["edge_kind"],
                response_nodes=archive["response_nodes"],
            )
            labels = FirstErrorLabels(
                num_steps=int(archive["num_steps"]),
                first_error=int(archive["first_error"]),
            )
            return CausalTrace(
                trace_id=str(archive["trace_id"]),
                problem_id=str(archive["problem_id"]),
                generator_model=str(archive["generator_model"]),
                observer_model=str(archive["observer_model"]),
                layer_id=int(archive["layer_id"]),
                prompt_tokens=int(archive["prompt_tokens"]),
                response_tokens=int(archive["response_tokens"]),
                graph=graph,
                labels=labels,
            )

    def traces(self) -> Iterator[CausalTrace]:
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        files = sorted(self.root.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no causal traces in {self.root}")
        for path in files:
            yield self.load(path)
