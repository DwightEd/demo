from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT.parent
for import_root in (PROJECT_ROOT / "src", DEMO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from crwh.real_cct_audit import audit_trace_cohort, classify_result_kind
from hypergraph.attention.cct.contracts import CausalHypergraph, FirstErrorLabels
from hypergraph.attention.cct.data import CausalTrace, TraceRepository
from hypergraph.attention.splitting import FixedHoldoutConfig, FixedHoldoutSplitter


MIN_TRAIN_HYPEREDGE_TRACE_COVERAGE = 0.25
MIN_TOTAL_HYPEREDGES = 4
MIN_EFFECT = 0.01
MIN_SYNERGY = 0.01


def _trace(index: int) -> CausalTrace:
    has_error = index % 2
    signal = 1.0 if has_error else -1.0
    node_features = np.zeros((3, 9), dtype=np.float32)
    node_features[:, 0] = signal
    node_features[2, -1] = 1.0
    return CausalTrace(
        trace_id=f"gate-trace-{index:03d}",
        problem_id=f"gate-problem-{index:03d}",
        generator_model="gate-generator",
        observer_model="gate-observer",
        layer_id=1,
        prompt_tokens=2,
        response_tokens=1,
        graph=CausalHypergraph(
            node_features=node_features,
            incidence=np.asarray([[0, 2], [0, 0]], dtype=np.int64),
            receivers=np.asarray([2], dtype=np.int64),
            edge_features=np.asarray(
                [[signal, 1.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            edge_kind=np.asarray(["pair"]),
            response_nodes=np.asarray([2], dtype=np.int64),
        ),
        labels=FirstErrorLabels(
            num_steps=1,
            first_error=0 if has_error else -1,
        ),
    )


def _cohort_with_train_hyperedges(
    root: Path,
    *,
    train_hyperedge_traces: int,
) -> tuple[Path, int]:
    traces = [_trace(index) for index in range(24)]
    repository = TraceRepository(root)
    for trace in traces:
        repository.save(trace)
    traces = list(repository.traces())
    split = FixedHoldoutSplitter(
        FixedHoldoutConfig(
            seed=17,
            validation_ratio=0.2,
            test_ratio=0.2,
        )
    ).split([trace.split_record() for trace in traces])
    hyperedge_indices = set(
        split.train.indices[:train_hyperedge_traces]
    )
    for index, trace in enumerate(traces):
        if index in hyperedge_indices:
            edge_features = trace.graph.edge_features.copy()
            edge_features[0, 2] = 0.02
            trace = replace(
                trace,
                graph=replace(
                    trace.graph,
                    incidence=np.asarray(
                        [[0, 1, 2], [0, 0, 0]],
                        dtype=np.int64,
                    ),
                    edge_features=edge_features,
                    edge_kind=np.asarray(["hyper"]),
                ),
            )
            repository.save(trace)
    (root / "extraction_config_00000-of-00001.json").write_text(
        json.dumps(
            {
                "min_effect": MIN_EFFECT,
                "min_synergy": MIN_SYNERGY,
                "written": len(traces),
                "failed": 0,
                "num_shards": 1,
                "shard_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root, len(split.train.indices)


def test_result_kind_requires_an_explicit_passing_hypergraph_gate() -> None:
    counts = {"hyper": 1, "pair": 23}

    assert (
        classify_result_kind(counts, hypergraph_gate_passed=False)
        == "real_processbench_cct_sparse_hyperedge_witness"
    )
    assert (
        classify_result_kind(counts, hypergraph_gate_passed=True)
        == "real_processbench_cct_hg_baseline"
    )


@pytest.mark.parametrize(
    ("members", "kind"),
    (
        ((0, 2), "hyper"),
        ((0, 1, 2), "pair"),
        ((0, 1, 2), "unknown"),
        ((0, 0, 2), "hyper"),
    ),
)
def test_causal_hypergraph_rejects_mislabeled_or_malformed_edges(
    members: tuple[int, ...],
    kind: str,
) -> None:
    base = _trace(0).graph
    incidence = np.asarray(
        [
            list(members),
            [0] * len(members),
        ],
        dtype=np.int64,
    )

    with pytest.raises(ValueError):
        CausalHypergraph(
            node_features=base.node_features,
            incidence=incidence,
            receivers=base.receivers,
            edge_features=base.edge_features,
            edge_kind=np.asarray([kind]),
            response_nodes=base.response_nodes,
        )


def test_hypergraph_gate_rejects_one_hyperedge_in_the_training_cohort(
    tmp_path: Path,
) -> None:
    traces_dir, _ = _cohort_with_train_hyperedges(
        tmp_path / "traces",
        train_hyperedge_traces=1,
    )

    with pytest.raises(ValueError, match="hypergraph gate"):
        audit_trace_cohort(
            traces_dir,
            split_seed=17,
            validation_ratio=0.2,
            test_ratio=0.2,
            require_hyperedges=True,
            min_train_hyperedge_trace_coverage=(
                MIN_TRAIN_HYPEREDGE_TRACE_COVERAGE
            ),
            min_total_hyperedges=MIN_TOTAL_HYPEREDGES,
            min_effect=MIN_EFFECT,
            min_synergy=MIN_SYNERGY,
        )


def test_hypergraph_gate_records_and_accepts_the_minimum_coverage(
    tmp_path: Path,
) -> None:
    traces_dir, train_traces = _cohort_with_train_hyperedges(
        tmp_path / "traces",
        train_hyperedge_traces=MIN_TOTAL_HYPEREDGES,
    )

    audit = audit_trace_cohort(
        traces_dir,
        split_seed=17,
        validation_ratio=0.2,
        test_ratio=0.2,
        require_hyperedges=True,
        min_train_hyperedge_trace_coverage=(
            MIN_TRAIN_HYPEREDGE_TRACE_COVERAGE
        ),
        min_total_hyperedges=MIN_TOTAL_HYPEREDGES,
        min_effect=MIN_EFFECT,
        min_synergy=MIN_SYNERGY,
    )

    gate = audit["hypergraph_gate"]
    assert gate["passed"] is True
    assert gate["min_total_hyperedges"] == MIN_TOTAL_HYPEREDGES
    assert gate["observed_total_hyperedges"] == MIN_TOTAL_HYPEREDGES
    assert (
        gate["min_train_hyperedge_trace_coverage"]
        == MIN_TRAIN_HYPEREDGE_TRACE_COVERAGE
    )
    assert gate["observed_train_hyperedge_traces"] == MIN_TOTAL_HYPEREDGES
    assert gate["observed_train_hyperedge_trace_coverage"] == pytest.approx(
        MIN_TOTAL_HYPEREDGES / train_traces
    )
    assert audit["result_kind"] == "real_processbench_cct_hg_baseline"


def test_hypergraph_gate_rejects_a_mislabeled_zero_synergy_hyperedge(
    tmp_path: Path,
) -> None:
    traces_dir, _ = _cohort_with_train_hyperedges(
        tmp_path / "traces",
        train_hyperedge_traces=MIN_TOTAL_HYPEREDGES,
    )
    repository = TraceRepository(traces_dir)
    hyper_trace = next(
        trace
        for trace in repository.traces()
        if trace.graph.edge_kind.tolist() == ["hyper"]
    )
    edge_features = hyper_trace.graph.edge_features.copy()
    edge_features[0, 2] = 0.0
    repository.save(
        replace(
            hyper_trace,
            graph=replace(
                hyper_trace.graph,
                edge_features=edge_features,
            ),
        )
    )

    with pytest.raises(ValueError, match="synergy"):
        audit_trace_cohort(
            traces_dir,
            split_seed=17,
            validation_ratio=0.2,
            test_ratio=0.2,
            require_hyperedges=True,
            min_train_hyperedge_trace_coverage=(
                MIN_TRAIN_HYPEREDGE_TRACE_COVERAGE
            ),
            min_total_hyperedges=MIN_TOTAL_HYPEREDGES,
            min_effect=MIN_EFFECT,
            min_synergy=MIN_SYNERGY,
        )
