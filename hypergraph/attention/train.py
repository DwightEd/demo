#!/usr/bin/env python3
"""Inspect, build, and train the faithful attention-row hypergraph.

Examples (run from ``demo``)::

    python -m hypergraph.attention.train inspect traces/
    python -m hypergraph.attention.train build traces/ --output graphs/
    python -m hypergraph.attention.train train traces/ --objective step_bce \
        --output runs/faithful_step

The default graph/model settings follow the local original HyperCHARM method.
Every departure (top-k pruning, attention-weighted incidences, receiver-only
propagation, hidden activation features, or receiver/source interaction) is an
explicit flag and is recorded in the output configuration.  No command reports
experimental scores unless a model was actually trained and evaluated.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import hmac
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .construction import build_attention_hypergraph, validate_attention_hypergraph
from .data import (
    MODEL_COMMIT_SOURCES,
    TRACE_CONTRACT,
    VERIFIED_MODEL_COMMIT_SOURCES,
    AttentionTrace,
    TraceFormatError,
    TraceLoadConfig,
    commit_hashes_match,
    discover_trace_files,
    is_immutable_commit_hash,
    iter_traces,
    model_identity_matches,
    safe_trace_stem,
    trace_method_provenance,
    trace_representation_fingerprint,
    trace_representation_provenance,
    trace_source_provenance,
    trace_summary,
)
from .schema import AttentionHypergraph, AttentionHypergraphConfig
from .shards import audit_scope_records


OBJECTIVES = ("token_bce", "step_bce", "response_bce")


@dataclass(frozen=True)
class _TraceMeta:
    trace_id: str
    group_id: str
    group_is_fallback: bool
    split: Optional[str]
    response_label: Optional[int]
    gold_step: Optional[int]
    num_steps: int
    num_response_tokens: int
    generator_model: Optional[str]


@dataclass(frozen=True)
class _ReplayProvenanceAudit:
    complete: bool
    observer: bool
    unverified_weights: bool


@dataclass
class _TraceCohortAuditState:
    representation_fingerprints: List[str] = field(default_factory=list)
    missing_provenance: List[str] = field(default_factory=list)
    representation_provenance_records: List[Dict[str, Any]] = field(
        default_factory=list
    )
    extraction_scope_fingerprints: set[str] = field(default_factory=set)
    extraction_scope_records: List[Dict[str, Any]] = field(default_factory=list)
    observer_traces: List[str] = field(default_factory=list)
    unverified_weight_traces: List[str] = field(default_factory=list)
    unverified_chunk_topology_traces: List[str] = field(default_factory=list)
    axis_contracts: set[Tuple[Any, ...]] = field(default_factory=set)
    generator_model_counts: Counter[str] = field(default_factory=Counter)
    generator_commit_counts: Counter[str] = field(default_factory=Counter)


@dataclass
class _LoadedGraphCohort:
    trace_metadata: List[_TraceMeta]
    graphs: List[AttentionHypergraph]
    trace_rows: List[Dict[str, Any]]
    graph_rows: List[Dict[str, Any]]
    missing_supervision: int
    provenance_info: Dict[str, Any]
    selection_info: Dict[str, Any]


def _finite_json(value: Any) -> Any:
    if is_dataclass(value):
        return _finite_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_finite_json(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _training_runtime_provenance(torch) -> Dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]

    def git_output(*arguments: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    code_hashes = {}
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        code_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    status = git_output("status", "--porcelain", "--untracked-files=all")
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_device_names": [
            str(torch.cuda.get_device_name(index))
            for index in range(torch.cuda.device_count())
        ],
        "cuda_compute_capabilities": [
            list(torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        ],
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(
            hasattr(torch.backends, "cudnn") and torch.backends.cudnn.deterministic
        ),
        "cudnn_benchmark": bool(
            hasattr(torch.backends, "cudnn") and torch.backends.cudnn.benchmark
        ),
        "repository_commit": git_output("rev-parse", "HEAD"),
        "repository_dirty": None if status is None else bool(status),
        "code_sha256": code_hashes,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_scope(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve and hash every candidate trace artifact used by a command."""

    files = discover_trace_files(
        args.inputs, recursive=not bool(args.no_recursive)
    )
    if not files:
        raise FileNotFoundError("no .npz/.pt/.pth attention traces found")
    return {
        "input_specs": [str(value) for value in args.inputs],
        "recursive": not bool(args.no_recursive),
        "trace_limit": None if args.limit is None else int(args.limit),
        "generator_models": list(_parse_generator_models(args.generator_model)),
        "candidate_files": [
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
            for path in files
        ],
    }


def _parse_indices(raw: Any, *, allow_all: bool = True) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        values = np.asarray(raw)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("index lists must contain exact integers")
        return tuple(int(value) for value in values.tolist())
    text = str(raw).strip().lower()
    if allow_all and text in {"", "all", "none"}:
        return None
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _parse_activation_layers(raw: Any) -> Optional[Tuple[int, ...]]:
    if raw is None or (isinstance(raw, str) and raw.strip().lower() == "all"):
        return None
    if isinstance(raw, str) and raw.strip().lower() == "last":
        return (-1,)
    return _parse_indices(raw, allow_all=False)


def _parse_generator_models(raw: Any) -> Tuple[str, ...]:
    """Parse exact dataset generator tags; no model-name aliasing is inferred."""

    if raw is None:
        return ()
    if not isinstance(raw, str):
        raise ValueError("generator model filter must be a comma-separated string")
    text = raw.strip()
    if not text:
        return ()
    values = tuple(part.strip() for part in text.split(","))
    if any(not value for value in values):
        raise ValueError("generator model filter contains an empty tag")
    normalized = [value.casefold() for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError("generator model filter contains duplicate tags")
    return values


def _trace_matches_generator_filter(
    trace: AttentionTrace, args: argparse.Namespace
) -> bool:
    wanted = _parse_generator_models(args.generator_model)
    if not wanted:
        return True
    observed = trace.metadata.get("generator_model")
    if observed in (None, ""):
        return False
    observed_key = str(observed).strip().casefold()
    return observed_key in {value.casefold() for value in wanted}


def _command_actions(
    parser: argparse.ArgumentParser, command: str
) -> List[argparse.Action]:
    actions = list(parser._actions)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            selected = action.choices.get(command)
            if selected is not None:
                actions.extend(selected._actions)
    return actions


def _validated_config_value(action: argparse.Action, value: Any, *, key: str) -> Any:
    boolean_actions = (argparse._StoreTrueAction, argparse._StoreFalseAction)
    if isinstance(action, boolean_actions) or isinstance(
        action, argparse.BooleanOptionalAction
    ):
        if not isinstance(value, bool):
            raise SystemExit(f"config key {key!r} must be a JSON boolean")
        return value

    if action.dest in {"selected_layers", "selected_heads"}:
        try:
            _parse_indices(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"invalid config key {key!r}: {exc}") from exc
        return value
    if action.dest == "activation_layers":
        try:
            _parse_activation_layers(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"invalid config key {key!r}: {exc}") from exc
        return value

    if value is None and action.default is None:
        return None
    if action.type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"config key {key!r} must be an exact JSON integer")
        validated = value
    elif action.type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(f"config key {key!r} must be a JSON number")
        validated = float(value)
    elif action.type is str:
        if not isinstance(value, str):
            raise SystemExit(f"config key {key!r} must be a JSON string")
        validated = value
    elif action.type is not None:
        try:
            validated = action.type(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"invalid config key {key!r}: {exc}") from exc
    else:
        validated = value
        if action.dest == "pos_weight" and (
            isinstance(value, bool) or not isinstance(value, (str, int, float))
        ):
            raise SystemExit(
                f"config key {key!r} must be auto/none or a positive JSON number"
            )
        if isinstance(action.default, str) and action.dest != "pos_weight":
            if not isinstance(value, str):
                raise SystemExit(f"config key {key!r} must be a JSON string")

    if action.choices is not None and validated not in action.choices:
        raise SystemExit(
            f"config key {key!r} must be one of {list(action.choices)!r}, "
            f"got {validated!r}"
        )
    return validated


def _strict_json_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_config(
    args: argparse.Namespace,
    argv: Sequence[str],
    parser: argparse.ArgumentParser,
) -> None:
    """Apply JSON defaults without overriding explicit command-line flags."""

    if not getattr(args, "config", None):
        return
    path = Path(args.config)
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_json_object
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"invalid --config JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise SystemExit("--config must contain a JSON object")
    flattened: Dict[str, Tuple[str, Any]] = {}
    for key, value in document.items():
        if key in {"loader", "graph", "model", "training", "data"}:
            if not isinstance(value, Mapping):
                raise SystemExit(f"config section {key!r} must be an object")
            items = value.items()
        else:
            items = ((key, value),)
        for nested_key, nested_value in items:
            normalized = str(nested_key).replace("-", "_")
            if normalized in flattened:
                previous = flattened[normalized][0]
                raise SystemExit(
                    f"config key {nested_key!r} duplicates {previous!r} after normalization"
                )
            flattened[normalized] = (str(nested_key), nested_value)

    valid = vars(args)
    actions = _command_actions(parser, str(args.command))
    actions_by_dest = {action.dest: action for action in actions}
    option_to_dest = {
        option: action.dest for action in actions for option in action.option_strings
    }
    explicit = {
        option_to_dest[token.split("=", 1)[0]]
        for token in argv
        if token.startswith("--") and token.split("=", 1)[0] in option_to_dest
    }
    protected = {"command", "inputs", "config", "output", "objective", "handler"}
    for normalized, (raw_key, value) in flattened.items():
        if normalized not in valid:
            raise SystemExit(f"unknown config key: {raw_key}")
        if normalized in protected:
            raise SystemExit(
                f"config key {raw_key!r} is protected; provide it explicitly on the command line"
            )
        action = actions_by_dest.get(normalized)
        if action is None:
            raise SystemExit(f"config key {raw_key!r} is not configurable")
        validated = _validated_config_value(action, value, key=raw_key)
        if normalized not in explicit:
            setattr(args, normalized, validated)


def _load_config_from_args(args: argparse.Namespace) -> TraceLoadConfig:
    return TraceLoadConfig(
        step_end=str(args.step_end),
        step_axis=str(args.step_axis),
        activation_layout=str(args.activation_layout),
        activation_layers=_parse_activation_layers(args.activation_layers),
        require_attention=True,
        require_causal=bool(args.require_causal),
        causal_tolerance=float(args.causal_tolerance),
    )


def _graph_config_from_args(args: argparse.Namespace) -> AttentionHypergraphConfig:
    node_feature_mode = str(args.node_feature_mode)
    if bool(args.use_activation):
        if node_feature_mode not in {"attention_diagonal", "diagonal_plus_activation"}:
            raise ValueError(
                "--use-activation is the legacy alias for diagonal_plus_activation and "
                "cannot be combined with --node-feature-mode activation_only"
            )
        node_feature_mode = "diagonal_plus_activation"
    return AttentionHypergraphConfig(
        threshold=float(args.threshold),
        top_k=args.top_k,
        min_sources=args.min_sources,
        include_center=bool(args.include_center),
        source_scope=str(args.source_scope),
        incidence_weight_mode=str(args.incidence_weight_mode),
        source_selection=str(args.source_selection),
        cumulative_mass=float(args.cumulative_mass),
        propagation_mode=str(args.propagation_mode),
        edge_attr_mode=str(args.edge_attr_mode),
        node_feature_mode=node_feature_mode,
        selected_layers=_parse_indices(args.selected_layers),
        selected_heads=_parse_indices(args.selected_heads),
    )


def _validate_limit_input_order(
    args: argparse.Namespace, trace: Optional[AttentionTrace] = None
) -> None:
    """Reject limits whose selected rows could depend on physical shard traversal."""

    if args.limit is None:
        return
    if len(args.inputs) > 1:
        raise SystemExit(
            "--limit across multiple trace inputs is storage-order dependent; "
            "materialize the limited cohort before sharding, or omit --limit"
        )
    if trace is None:
        return
    scope_json = trace.metadata.get("extraction_scope_json")
    if scope_json in (None, ""):
        return
    try:
        scope = json.loads(str(scope_json))
    except json.JSONDecodeError:
        return  # The strict extraction-scope audit reports the malformed record.
    if isinstance(scope, Mapping) and int(scope.get("num_shards", 1)) > 1:
        raise SystemExit(
            "--limit over a sharded extraction scope is storage-order dependent even "
            "through one parent directory/glob; materialize the limited cohort first"
        )


def _iter_input_traces(
    args: argparse.Namespace, *, apply_selection: bool = True
) -> Iterable[AttentionTrace]:
    """Stream traces so quadratic dense attention is freed after construction."""

    if apply_selection:
        _validate_limit_input_order(args)
    config = _load_config_from_args(args)
    seen: set[str] = set()
    count = 0
    for trace in iter_traces(args.inputs, config=config, recursive=not args.no_recursive):
        if apply_selection:
            _validate_limit_input_order(args, trace)
        if trace.trace_id in seen:
            raise SystemExit(f"trace_id values must be unique; duplicate {trace.trace_id!r}")
        seen.add(trace.trace_id)
        if apply_selection and not _trace_matches_generator_filter(trace, args):
            continue
        if apply_selection and args.limit is not None and count >= int(args.limit):
            break
        yield trace
        count += 1


def _build_graph(
    trace: AttentionTrace,
    graph_config: AttentionHypergraphConfig,
) -> AttentionHypergraph:
    use_activation = graph_config.node_feature_mode != "attention_diagonal"
    graph = build_attention_hypergraph(
        **trace.builder_kwargs(use_activation=use_activation), config=graph_config
    )
    validate_attention_hypergraph(graph, graph_config)
    return graph


def _per_graph_zscore(value: np.ndarray, low: float, high: float) -> np.ndarray:
    """Reproduce the original per-graph clamp/column standardization safely."""

    array = np.clip(np.asarray(value, np.float32), low, high)
    if not array.size:
        return np.ascontiguousarray(array)
    mean = np.mean(array, axis=0, keepdims=True)
    # torch.std used by the original preprocessing is sample std.  A one-row
    # edge table is a degenerate corner case; use zero variance rather than NaN.
    std = (
        np.std(array, axis=0, ddof=1, keepdims=True)
        if array.shape[0] > 1
        else np.zeros((1, array.shape[1]), dtype=np.float32)
    )
    return np.ascontiguousarray((array - mean) / (std + 1e-6), dtype=np.float32)


def _preprocess_training_graph(
    graph: AttentionHypergraph, mode: str
) -> AttentionHypergraph:
    if mode == "none":
        return graph
    if mode != "per_graph_zscore":
        raise ValueError(f"unknown preprocessing mode {mode!r}")
    return replace(
        graph,
        x=_per_graph_zscore(graph.x, -5.0, 5.0),
        he_attr=_per_graph_zscore(graph.he_attr, 0.0, 1.0),
    )


def _aggregate_inspection(
    trace_rows: Sequence[Mapping[str, Any]], graph_rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    def values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows if row.get(key) is not None], float)

    lengths = values(trace_rows, "num_tokens")
    response_lengths = values(trace_rows, "num_response_tokens")
    edges = values(graph_rows, "num_hyperedges")
    incidences = values(graph_rows, "num_incidences")
    coverage = values(graph_rows, "receiver_edge_coverage")
    edges_per_receiver = values(graph_rows, "mean_edges_per_response_token")
    member_counts = values(graph_rows, "mean_members_per_edge")
    response_labels = values(trace_rows, "response_y")
    generator_counts = Counter(
        str(row.get("source_provenance", {}).get("generator_model", "<missing>"))
        for row in trace_rows
    )
    position_bins = []
    for bin_index in range(5):
        bin_values = np.asarray(
            [
                row["position_bin_edges_per_token"][bin_index]
                for row in graph_rows
                if row.get("position_bin_edges_per_token") is not None
                and row["position_bin_edges_per_token"][bin_index] is not None
            ],
            dtype=float,
        )
        position_bins.append(None if not len(bin_values) else float(np.mean(bin_values)))
    return {
        "num_traces": len(trace_rows),
        "num_groups": len({row["group_id"] for row in trace_rows}),
        "num_fallback_groups": sum(bool(row["group_is_fallback"]) for row in trace_rows),
        "num_with_exact_token_labels": sum(
            bool(row["has_exact_token_labels"]) for row in trace_rows
        ),
        "num_with_step_labels": sum(row["gold_step"] is not None for row in trace_rows),
        "num_with_response_labels": int(len(response_labels)),
        "response_positive_rate": (
            None if not len(response_labels) else float(response_labels.mean())
        ),
        "generator_model_counts": dict(sorted(generator_counts.items())),
        "tokens": _distribution(lengths),
        "response_tokens": _distribution(response_lengths),
        "hyperedges": _distribution(edges),
        "incidences": _distribution(incidences),
        "receiver_edge_coverage": _distribution(coverage),
        "edges_per_response_token": _distribution(edges_per_receiver),
        "members_per_edge": _distribution(member_counts),
        "position_bin_edges_per_token": position_bins,
        "response_length_edge_count_correlation": _safe_pearson(
            response_lengths, edges
        ),
        "response_length_coverage_correlation": _safe_pearson(
            response_lengths, coverage
        ),
    }


def _distribution(values: np.ndarray) -> Dict[str, Any]:
    if not len(values):
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    left, right = np.asarray(left, float), np.asarray(right, float)
    if len(left) != len(right) or len(left) < 2:
        return None
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _graph_summary(graph: AttentionHypergraph, trace: AttentionTrace) -> Dict[str, Any]:
    n_response = graph.num_nodes - int(graph.response_idx)
    edge_counts = np.bincount(
        graph.he_receiver - int(graph.response_idx), minlength=n_response
    ) if graph.num_hyperedges else np.zeros(n_response, dtype=np.int64)
    position_bins: List[Optional[float]] = []
    bin_ids = np.minimum(4, np.floor(np.arange(n_response) * 5 / n_response).astype(int))
    for bin_index in range(5):
        selected = edge_counts[bin_ids == bin_index]
        position_bins.append(None if not len(selected) else float(np.mean(selected)))
    return {
        "trace_id": trace.trace_id,
        "group_id": trace.group_id,
        "split": trace.split,
        "num_nodes": graph.num_nodes,
        "node_dim": int(graph.x.shape[1]),
        "num_hyperedges": graph.num_hyperedges,
        "edge_dim": int(graph.he_attr.shape[1]),
        "num_incidences": graph.num_incidences,
        "prompt_cross_edges": int(np.sum(graph.he_mark[:, 0])) if graph.num_hyperedges else 0,
        "response_only_edges": int(np.sum(graph.he_mark[:, 1])) if graph.num_hyperedges else 0,
        "receiver_edge_coverage": float(np.mean(edge_counts > 0)),
        "mean_edges_per_response_token": float(np.mean(edge_counts)),
        "mean_members_per_edge": (
            None if not graph.num_hyperedges else float(np.mean(graph.he_count))
        ),
        "position_bin_edges_per_token": position_bins,
        "propagation_mode": graph.propagation_mode,
        "incidence_weight_mode": graph.incidence_weight_mode,
        "edge_attr_names": list(graph.edge_attr_names),
    }


def command_inspect(args: argparse.Namespace) -> Dict[str, Any]:
    graph_config = _graph_config_from_args(args)
    input_scope = _input_scope(args)
    cohort_audit = None
    selection = None
    if args.objective is not None:
        cohort = _load_audited_graph_cohort(
            args,
            graph_config,
            objective=str(args.objective),
            preprocessing=None,
        )
        trace_rows = cohort.trace_rows
        graph_rows = cohort.graph_rows
        cohort_audit = cohort.provenance_info
        selection = cohort.selection_info
    else:
        trace_rows, graph_rows = [], []
        for trace in _iter_input_traces(args):
            graph = _build_graph(trace, graph_config)
            trace_rows.append(trace_summary(trace))
            graph_rows.append(_graph_summary(graph, trace))
    if not trace_rows:
        raise SystemExit("no usable attention traces were loaded")
    result = {
        "command": "inspect",
        "inspection_mode": (
            "supervised_cohort_gate" if args.objective is not None else "structural_only"
        ),
        "cohort_gate_passed": bool(args.objective is not None),
        "objective": args.objective,
        "loader_config": asdict(_load_config_from_args(args)),
        "input_scope": input_scope,
        "graph_config": asdict(graph_config),
        "selection": selection,
        "cohort_audit": cohort_audit,
        "use_activation": graph_config.node_feature_mode != "attention_diagonal",
        "summary": _aggregate_inspection(trace_rows, graph_rows),
        "traces": trace_rows if args.verbose_records else trace_rows[: min(10, len(trace_rows))],
        "graphs": graph_rows if args.verbose_records else graph_rows[: min(10, len(graph_rows))],
        "records_truncated": bool(not args.verbose_records and len(trace_rows) > 10),
    }
    if args.output:
        _write_json(Path(args.output), result)
    print(json.dumps(_finite_json(result), ensure_ascii=False, indent=2, sort_keys=True))
    return result


def _graph_arrays(graph: AttentionHypergraph, metadata: Mapping[str, Any]) -> Dict[str, Any]:
    arrays: Dict[str, Any] = {}
    for item in fields(graph):
        value = getattr(graph, item.name)
        if value is None:
            continue
        if is_dataclass(value):
            arrays[f"{item.name}_json"] = np.asarray(
                json.dumps(_finite_json(value), ensure_ascii=False, sort_keys=True)
            )
        elif isinstance(value, str):
            arrays[item.name] = np.asarray(value)
        else:
            arrays[item.name] = np.asarray(value)
    arrays["metadata_json"] = np.asarray(
        json.dumps(_finite_json(metadata), ensure_ascii=False, sort_keys=True)
    )
    return arrays


def command_build(args: argparse.Namespace) -> Dict[str, Any]:
    graph_config = _graph_config_from_args(args)
    input_scope = _input_scope(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output / 'manifest.json'}; pass --overwrite"
        )
    rows = []
    seen_names: set[str] = set()
    for trace in _iter_input_traces(args):
        graph = _build_graph(trace, graph_config)
        digest = hashlib.sha256(
            f"{trace.source_path}\0{trace.trace_id}".encode("utf-8")
        ).hexdigest()[:10]
        name = f"{safe_trace_stem(trace)}-{digest}.npz"
        if name in seen_names:
            raise RuntimeError(f"duplicate graph output name: {name}")
        seen_names.add(name)
        target = output / name
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {target}; pass --overwrite")
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                **_graph_arrays(
                    graph,
                    {
                        "trace": trace_summary(trace),
                        "graph_config": asdict(graph_config),
                        "use_activation": graph_config.node_feature_mode != "attention_diagonal",
                    },
                ),
            )
        os.replace(temporary, target)
        row = _graph_summary(graph, trace)
        row["path"] = str(target)
        rows.append(row)
    if not rows:
        raise SystemExit("no usable attention traces were loaded")
    manifest = {
        "schema": "faithful_attention_hypergraph_v1",
        "command": "build",
        "loader_config": asdict(_load_config_from_args(args)),
        "input_scope": input_scope,
        "graph_config": asdict(graph_config),
        "use_activation": graph_config.node_feature_mode != "attention_diagonal",
        "num_graphs": len(rows),
        "graphs": rows,
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(_finite_json(manifest), ensure_ascii=False, indent=2, sort_keys=True))
    return manifest


def _objective_available(trace: AttentionTrace, objective: str) -> bool:
    if objective == "token_bce":
        return (
            trace.token_y is not None
            and trace.token_label_mask is not None
            and bool(np.any(trace.token_label_mask[trace.response_idx :]))
        )
    if objective == "step_bce":
        return (
            trace.step_ranges is not None
            and trace.gold_step is not None
            and len(trace.step_ranges) > 0
            and (
                trace.step_loss_mask is None
                or bool(np.any(trace.step_loss_mask))
            )
        )
    if objective == "response_bce":
        return trace.response_y is not None
    raise ValueError(f"unknown objective {objective!r}")


def _trace_meta(trace: AttentionTrace) -> _TraceMeta:
    return _TraceMeta(
        trace_id=trace.trace_id,
        group_id=trace.group_id,
        group_is_fallback=trace.group_is_fallback,
        split=trace.split,
        response_label=(
            None if trace.response_y is None else int(float(trace.response_y) >= 0.5)
        ),
        gold_step=trace.gold_step,
        num_steps=0 if trace.step_ranges is None else int(len(trace.step_ranges)),
        num_response_tokens=int(trace.num_response_tokens),
        generator_model=(
            None
            if trace.metadata.get("generator_model") in (None, "")
            else str(trace.metadata["generator_model"])
        ),
    )


_REPLAY_FIDELITIES = {
    "weight_and_token_verified_replay",
    "token_axis_verified_weights_unverified",
    "observer_counterfactual",
}


def _is_explicit_true(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


_METHOD_PROVENANCE_BINDINGS = {
    "trace_contract": "trace_contract",
    "model_name": "model_name",
    "model_commit_hash": "model_commit_hash",
    "model_commit_source": "model_commit_source",
    "tokenizer_name": "tokenizer_name",
    "prompt_style": "prompt_style",
    "replay_mode": "replay_mode",
    "extraction_dtype": "dtype",
    "attention_storage_dtype": "attention_storage_dtype",
    "activation_layer": "activation_layer",
}


def _normalize_bound_provenance(key: str, value: Any) -> Any:
    if key == "activation_layer":
        if value in (None, "", -1):
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError("activation_layer provenance must be an integer or null")
        return int(value)
    return None if value in (None, "") else str(value)


def _bind_method_provenance(
    method_provenance: Mapping[str, Any], parsed_method: Mapping[str, Any]
) -> None:
    """Bind authenticated extraction config to separately loaded trace fields."""

    for provenance_key, config_key in _METHOD_PROVENANCE_BINDINGS.items():
        if config_key not in parsed_method:
            raise ValueError(
                f"extraction_method_json lacks bound field {config_key!r}"
            )
        observed = _normalize_bound_provenance(
            provenance_key, method_provenance.get(provenance_key)
        )
        declared = _normalize_bound_provenance(
            provenance_key, parsed_method.get(config_key)
        )
        if observed != declared:
            raise ValueError(
                f"trace provenance {provenance_key}={observed!r} disagrees with "
                f"extraction method {config_key}={declared!r}"
            )
    allow_unverified = parsed_method.get("allow_unverified_generator_weights")
    if not isinstance(allow_unverified, bool):
        raise ValueError(
            "extraction_method_json allow_unverified_generator_weights must be boolean"
        )
    trace_unverified = method_provenance.get(
        "unverified_generator_weights_explicitly_allowed"
    )
    if not isinstance(trace_unverified, (bool, np.bool_)):
        raise ValueError(
            "trace unverified_generator_weights_explicitly_allowed must be boolean"
        )
    if bool(trace_unverified) and not allow_unverified:
        raise ValueError(
            "trace claims unverified weights were allowed but extraction method forbids it"
        )


def _validate_trace_axes_against_method(trace: AttentionTrace) -> None:
    method_json = trace.metadata.get("extraction_method_json")
    if method_json in (None, ""):
        return
    try:
        method = json.loads(str(method_json))
    except json.JSONDecodeError as exc:
        raise ValueError("extraction_method_json is not valid JSON") from exc
    if not isinstance(method, dict):
        raise ValueError("extraction_method_json must encode an object")
    comparisons = (
        ("attention_layers", trace.attention_layer_ids.tolist()),
        ("attention_heads", trace.attention_head_ids.tolist()),
        ("num_model_layers", int(trace.num_model_layers)),
        ("num_model_heads", int(trace.num_model_heads)),
    )
    for key, observed in comparisons:
        if key not in method or method[key] != observed:
            raise ValueError(
                f"trace attention axis {key}={observed!r} disagrees with "
                f"extraction method {method.get(key)!r}"
            )
    if trace.metadata.get("extraction_forward_mode") == "cached_query_chunks":
        chunk_result = json.loads(str(trace.metadata.get("chunk_equivalence_json")))
        chunk_size = method.get("query_chunk_size")
        verify_tokens = method.get("chunk_verify_tokens")
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size < 1
            or isinstance(verify_tokens, bool)
            or not isinstance(verify_tokens, int)
            or verify_tokens < 2
        ):
            raise ValueError("chunk method sizes must be positive integers")
        expected_tokens = min(
            trace.num_tokens, max(verify_tokens, chunk_size + 1)
        )
        if chunk_result.get("tokens") != expected_tokens:
            raise ValueError(
                "chunk equivalence token count disagrees with method and trace length"
            )


def _validate_chunk_graph_contract(
    trace: AttentionTrace,
    graph_config: AttentionHypergraphConfig,
    *,
    allow_unverified: bool,
) -> bool:
    if trace.metadata.get("extraction_forward_mode") != "cached_query_chunks":
        return False
    try:
        chunk_result = json.loads(str(trace.metadata.get("chunk_equivalence_json")))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("chunk_equivalence_json is not valid JSON") from exc
    if not isinstance(chunk_result, dict):
        raise ValueError("chunk_equivalence_json must encode an object")
    topology_threshold = chunk_result.get("topology_threshold")
    threshold_is_numeric = (
        not isinstance(topology_threshold, bool)
        and isinstance(topology_threshold, (int, float))
        and math.isfinite(float(topology_threshold))
    )
    threshold_matches = (
        graph_config.source_selection == "threshold"
        and graph_config.top_k is None
        and threshold_is_numeric
        and float(topology_threshold) == float(graph_config.threshold)
    )
    if not threshold_matches and not allow_unverified:
        raise ValueError(
            "cached-query trace was not topology-gated for this graph selector/threshold; "
            "use full-forward traces or --allow-unverified-chunk-topology only for a "
            "separately named diagnostic"
        )
    return not threshold_matches


def _audit_replay_provenance(
    method_provenance: Mapping[str, Any], metadata: Mapping[str, Any]
) -> _ReplayProvenanceAudit:
    """Validate replay provenance as a state machine, not trusted free text."""

    required = (
        "trace_contract",
        "model_name",
        "model_commit_source",
        "tokenizer_name",
        "prompt_style",
        "replay_mode",
        "replay_fidelity",
        "prompt_provenance",
        "prompt_add_special_tokens",
        "extraction_dtype",
        "attention_storage_dtype",
        "extraction_fingerprint",
    )
    complete = all(method_provenance.get(key) not in (None, "") for key in required)
    method_fingerprint = method_provenance.get("extraction_fingerprint")
    method_json = metadata.get("extraction_method_json")
    parsed_method: Optional[Dict[str, Any]] = None
    complete = complete and method_json not in (None, "")
    if method_json not in (None, ""):
        try:
            parsed_method = json.loads(str(method_json))
        except json.JSONDecodeError as exc:
            raise ValueError("extraction_method_json is not valid JSON") from exc
        if not isinstance(parsed_method, dict):
            raise ValueError("extraction_method_json must encode an object")
        canonical_method = json.dumps(
            parsed_method, sort_keys=True, separators=(",", ":")
        )
        computed_method_fingerprint = hashlib.sha256(
            canonical_method.encode("utf-8")
        ).hexdigest()
        if method_fingerprint not in (None, "") and not hmac.compare_digest(
            str(method_fingerprint), computed_method_fingerprint
        ):
            raise ValueError(
                "extraction_fingerprint does not match its method JSON"
            )
        _bind_method_provenance(method_provenance, parsed_method)
    scope_fingerprint = metadata.get("extraction_scope_fingerprint")
    scope_json = metadata.get("extraction_scope_json")
    source_input_sha256 = metadata.get("source_input_sha256")
    source_row_index = metadata.get("source_row_index")
    complete = complete and all(
        value not in (None, "")
        for value in (
            scope_fingerprint,
            scope_json,
            source_input_sha256,
            source_row_index,
        )
    )
    if scope_json not in (None, ""):
        try:
            parsed_scope = json.loads(str(scope_json))
        except json.JSONDecodeError as exc:
            raise ValueError("extraction_scope_json is not valid JSON") from exc
        if not isinstance(parsed_scope, dict):
            raise ValueError("extraction_scope_json must encode an object")
        canonical_scope = json.dumps(
            parsed_scope, sort_keys=True, separators=(",", ":")
        )
        computed_scope_fingerprint = hashlib.sha256(
            canonical_scope.encode("utf-8")
        ).hexdigest()
        if scope_fingerprint not in (None, "") and not hmac.compare_digest(
            str(scope_fingerprint), computed_scope_fingerprint
        ):
            raise ValueError("extraction_scope_fingerprint does not match its scope JSON")
        declared_input_sha256 = parsed_scope.get("input_sha256")
        if (
            source_input_sha256 not in (None, "")
            and str(declared_input_sha256) != str(source_input_sha256)
        ):
            raise ValueError("source_input_sha256 does not match extraction scope")
    if source_row_index not in (None, "") and (
        isinstance(source_row_index, (bool, np.bool_))
        or not isinstance(source_row_index, (int, np.integer))
        or int(source_row_index) < 0
    ):
        raise ValueError("source_row_index must be a non-negative integer")
    forward_mode = metadata.get("extraction_forward_mode")
    chunk_status = metadata.get("chunk_equivalence_status")
    chunk_json = metadata.get("chunk_equivalence_json")
    if forward_mode not in (None, "", "full", "cached_query_chunks"):
        raise ValueError(f"unknown extraction_forward_mode {forward_mode!r}")
    if chunk_status not in (
        None,
        "",
        "not_applicable",
        "prefix_pass",
        "disabled",
    ):
        raise ValueError(f"unknown chunk_equivalence_status {chunk_status!r}")
    if forward_mode == "cached_query_chunks":
        if chunk_status != "prefix_pass":
            raise ValueError(
                "cached-query extraction requires a passing per-trace prefix gate"
            )
        complete = complete and chunk_json not in (None, "")
        if chunk_json not in (None, ""):
            try:
                chunk_result = json.loads(str(chunk_json))
            except json.JSONDecodeError as exc:
                raise ValueError("chunk_equivalence_json is not valid JSON") from exc
            if not isinstance(chunk_result, dict):
                raise ValueError("chunk_equivalence_json must encode an object")
            if parsed_method is None:
                complete = False
            else:
                expected_pairs = (
                    (
                        "query_chunk_size",
                        parsed_method.get("query_chunk_size"),
                    ),
                    (
                        "atol",
                        parsed_method.get("chunk_equivalence_atol"),
                    ),
                    (
                        "topology_threshold",
                        parsed_method.get("chunk_equivalence_threshold"),
                    ),
                )
                for result_key, expected in expected_pairs:
                    if chunk_result.get(result_key) != expected:
                        raise ValueError(
                            f"chunk equivalence {result_key} disagrees with extraction method"
                        )
                if chunk_result.get("status") != "prefix_pass":
                    raise ValueError("cached extraction lacks a passing prefix gate")
                tokens = chunk_result.get("tokens")
                if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 2:
                    raise ValueError("chunk equivalence tokens must be an integer >= 2")
                max_abs = chunk_result.get("max_abs_error")
                atol = chunk_result.get("atol")
                if (
                    isinstance(max_abs, bool)
                    or not isinstance(max_abs, (int, float))
                    or not math.isfinite(float(max_abs))
                    or isinstance(atol, bool)
                    or not isinstance(atol, (int, float))
                    or not math.isfinite(float(atol))
                    or float(max_abs) > float(atol)
                ):
                    raise ValueError("chunk equivalence numerical tolerance did not pass")
                disagreements = chunk_result.get("topology_disagreements")
                if (
                    isinstance(disagreements, bool)
                    or not isinstance(disagreements, int)
                    or disagreements != 0
                ):
                    raise ValueError("chunk equivalence has topology disagreements")
    elif forward_mode == "full":
        if chunk_status not in (None, "", "not_applicable"):
            raise ValueError(
                "full extraction must use chunk_equivalence_status='not_applicable'"
            )
        complete = complete and chunk_status == "not_applicable"
        if chunk_json not in (None, ""):
            try:
                full_chunk_record = json.loads(str(chunk_json))
            except json.JSONDecodeError as exc:
                raise ValueError("chunk_equivalence_json is not valid JSON") from exc
            if full_chunk_record != {"status": "not_applicable"}:
                raise ValueError(
                    "full extraction chunk record must be exactly not_applicable"
                )
        else:
            complete = False
    else:
        complete = False

    contract = method_provenance.get("trace_contract")
    if contract not in (None, "", TRACE_CONTRACT):
        raise ValueError(
            f"unsupported trace_contract {contract!r}; expected {TRACE_CONTRACT!r}"
        )

    commit_source = method_provenance.get("model_commit_source")
    if commit_source not in (None, "") and commit_source not in MODEL_COMMIT_SOURCES:
        raise ValueError(f"unknown model_commit_source {commit_source!r}")
    replay_commit = method_provenance.get("model_commit_hash")
    if replay_commit not in (None, "") and not is_immutable_commit_hash(replay_commit):
        raise ValueError(
            f"model_commit_hash {replay_commit!r} is not an immutable hexadecimal commit hash"
        )
    if commit_source == "unavailable" and replay_commit not in (None, ""):
        raise ValueError("model_commit_source='unavailable' cannot carry model_commit_hash")
    if commit_source not in (None, "", "unavailable") and replay_commit in (None, ""):
        raise ValueError(
            f"model_commit_source={commit_source!r} requires model_commit_hash"
        )

    mode = method_provenance.get("replay_mode")
    fidelity = method_provenance.get("replay_fidelity")
    if mode not in (None, "", "same_generator", "observer"):
        raise ValueError(f"unknown replay_mode {mode!r}")
    if fidelity not in (None, "") and fidelity not in _REPLAY_FIDELITIES:
        raise ValueError(f"unknown replay_fidelity {fidelity!r}")

    observer = mode == "observer" or fidelity == "observer_counterfactual"
    unverified = fidelity == "token_axis_verified_weights_unverified"
    if mode in (None, "") or fidelity in (None, ""):
        return _ReplayProvenanceAudit(
            complete=False, observer=observer, unverified_weights=unverified
        )

    expected_fidelities = {
        "same_generator": {
            "weight_and_token_verified_replay",
            "token_axis_verified_weights_unverified",
        },
        "observer": {"observer_counterfactual"},
    }
    if fidelity not in expected_fidelities[str(mode)]:
        raise ValueError(
            f"replay_mode={mode!r} is inconsistent with replay_fidelity={fidelity!r}"
        )

    prompt_provenance = method_provenance.get("prompt_provenance")
    prompt_style = method_provenance.get("prompt_style")
    if mode == "same_generator":
        if prompt_provenance not in (None, "", "stored_rendered_prompt"):
            raise ValueError(
                "same_generator replay requires prompt_provenance='stored_rendered_prompt'"
            )
        generator_model = method_provenance.get("generator_model")
        replay_model = method_provenance.get("model_name")
        identities_present = generator_model not in (None, "") and replay_model not in (
            None,
            "",
        )
        if identities_present and not model_identity_matches(
            str(generator_model), str(replay_model)
        ):
            raise ValueError(
                f"generator_model {generator_model!r} does not match model_name "
                f"{replay_model!r}"
            )

        generator_commit = method_provenance.get("generator_model_commit")
        for field_name, value in (("generator_model_commit", generator_commit),):
            if value not in (None, "") and not is_immutable_commit_hash(value):
                raise ValueError(
                    f"{field_name} {value!r} is not an immutable hexadecimal commit hash"
                )
        commits_present = generator_commit not in (None, "") and replay_commit not in (
            None,
            "",
        )
        if commits_present and not commit_hashes_match(generator_commit, replay_commit):
            raise ValueError(
                f"generator_model_commit {generator_commit!r} does not match "
                f"model_commit_hash {replay_commit!r}"
            )

        complete = complete and identities_present
        complete = complete and prompt_provenance == "stored_rendered_prompt"
        complete = complete and metadata.get("rendered_prompt_sha256") not in (None, "")
        complete = complete and metadata.get("response_text_sha256") not in (None, "")
        if fidelity == "weight_and_token_verified_replay":
            if commit_source not in VERIFIED_MODEL_COMMIT_SOURCES:
                raise ValueError(
                    f"verified replay requires a resolved/pinned model revision, got "
                    f"model_commit_source={commit_source!r}"
                )
            complete = complete and commits_present
            if _is_explicit_true(
                method_provenance.get(
                    "unverified_generator_weights_explicitly_allowed", False
                )
            ):
                raise ValueError(
                    "verified replay cannot claim that unverified generator weights were allowed"
                )
        else:
            complete = complete and _is_explicit_true(
                method_provenance.get(
                    "unverified_generator_weights_explicitly_allowed", False
                )
            )
    else:
        allowed_observer_prompts = {"stored_rendered_prompt"}
        if prompt_style not in (None, ""):
            allowed_observer_prompts.add(f"frozen_{prompt_style}_observer")
        if prompt_provenance not in (None, "") and prompt_provenance not in allowed_observer_prompts:
            raise ValueError(
                f"observer replay has unsupported prompt_provenance {prompt_provenance!r}"
            )
        complete = complete and prompt_provenance in allowed_observer_prompts

    return _ReplayProvenanceAudit(
        complete=bool(complete), observer=observer, unverified_weights=unverified
    )


def _record_extraction_scope(
    state: _TraceCohortAuditState, trace: AttentionTrace
) -> None:
    scope_json = trace.metadata.get("extraction_scope_json")
    parsed_scope = None
    if scope_json not in (None, ""):
        try:
            parsed_scope = json.loads(str(scope_json))
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"trace {trace.trace_id!r} has invalid extraction_scope_json"
            ) from exc
    scope_fingerprint = trace.metadata.get("extraction_scope_fingerprint")
    if scope_fingerprint not in (None, ""):
        state.extraction_scope_fingerprints.add(str(scope_fingerprint))
    state.extraction_scope_records.append(
        {
            "trace_id": trace.trace_id,
            "scope": parsed_scope,
            "source_input_sha256": trace.metadata.get("source_input_sha256"),
            "source_row_index": trace.metadata.get("source_row_index"),
            "extraction_scope_fingerprint": scope_fingerprint,
            "status": "ok",
        }
    )


def _record_representation_compatibility(
    state: _TraceCohortAuditState,
    trace: AttentionTrace,
    graph_config: AttentionHypergraphConfig,
    args: argparse.Namespace,
) -> None:
    fingerprint = trace_representation_fingerprint(trace)
    representation_provenance = trace_representation_provenance(trace)
    method_provenance = trace_method_provenance(trace)
    source_provenance = trace_source_provenance(trace)
    generator = str(source_provenance.get("generator_model", "<missing>"))
    generator_commit = str(
        source_provenance.get("generator_model_commit", "<missing>")
    )
    state.generator_model_counts[generator] += 1
    state.generator_commit_counts[generator_commit] += 1
    try:
        replay_audit = _audit_replay_provenance(method_provenance, trace.metadata)
        _validate_trace_axes_against_method(trace)
        if _validate_chunk_graph_contract(
            trace,
            graph_config,
            allow_unverified=bool(args.allow_unverified_chunk_topology),
        ):
            state.unverified_chunk_topology_traces.append(trace.trace_id)
    except ValueError as exc:
        raise SystemExit(
            f"trace {trace.trace_id!r} has invalid replay provenance: {exc}"
        ) from exc
    if replay_audit.observer:
        state.observer_traces.append(trace.trace_id)
    if replay_audit.unverified_weights:
        state.unverified_weight_traces.append(trace.trace_id)
    state.axis_contracts.add(
        (
            tuple(np.asarray(trace.attention_layer_ids, dtype=int).tolist()),
            tuple(np.asarray(trace.attention_head_ids, dtype=int).tolist()),
            int(trace.num_model_layers),
            int(trace.num_model_heads),
        )
    )
    if fingerprint is None or not replay_audit.complete:
        state.missing_provenance.append(trace.trace_id)
    else:
        state.representation_fingerprints.append(fingerprint)
        state.representation_provenance_records.append(representation_provenance)


def _finalize_trace_cohort_audit(
    state: _TraceCohortAuditState, args: argparse.Namespace
) -> Dict[str, Any]:
    try:
        extraction_scope_audit = audit_scope_records(
            state.extraction_scope_records,
            allow_incomplete=bool(args.allow_incomplete_extraction_scope),
            allow_multiple_inputs=bool(args.allow_multiple_input_datasets),
        )
    except ValueError as exc:
        raise SystemExit(f"extraction shard audit failed: {exc}") from exc

    unique_fingerprints = sorted(set(state.representation_fingerprints))
    if len(state.axis_contracts) > 1:
        raise SystemExit(
            "attention traces use different layer/head axes or model layer/head totals; "
            "refusing same-dimensional but semantically incompatible node features"
        )
    if len(unique_fingerprints) > 1:
        raise SystemExit(
            "attention traces have different observer/template/layer representation "
            "fingerprints; refusing to silently mix incompatible representations in one run"
        )
    if state.missing_provenance and not args.allow_missing_provenance:
        raise SystemExit(
            f"{len(state.missing_provenance)} supervised traces lack model/template "
            "extraction provenance. Re-extract with attention.extract, or explicitly pass "
            "--allow-missing-provenance for a legacy diagnostic run."
        )
    if state.observer_traces and not args.allow_observer_traces:
        raise SystemExit(
            f"{len(state.observer_traces)} supervised traces are counterfactual observer "
            "replays. They are not original-generation mechanism traces; pass "
            "--allow-observer-traces only for a separately named observer experiment."
        )
    if (
        state.unverified_weight_traces
        and not args.allow_unverified_generator_weights
    ):
        raise SystemExit(
            f"{len(state.unverified_weight_traces)} traces have an exact token axis but "
            "unverified generator weights. Pass --allow-unverified-generator-weights "
            "only for a separately named diagnostic run."
        )

    axis_contract = None
    if state.axis_contracts:
        layer_ids, head_ids, num_layers, num_heads = next(iter(state.axis_contracts))
        axis_contract = {
            "layer_ids": list(layer_ids),
            "head_ids": list(head_ids),
            "num_model_layers": int(num_layers),
            "num_model_heads": int(num_heads),
        }
    representation_provenance = (
        state.representation_provenance_records[0]
        if state.representation_provenance_records
        else None
    )
    return {
        "fingerprint": unique_fingerprints[0] if len(unique_fingerprints) == 1 else None,
        "representation_fingerprint": (
            unique_fingerprints[0] if len(unique_fingerprints) == 1 else None
        ),
        "representation_provenance": representation_provenance,
        # Backward-compatible key.  Unlike the old value, this no longer
        # misrepresents the first sample's generator as cohort-wide method state.
        "method_provenance": representation_provenance,
        "source_provenance": {
            "generator_model_counts": dict(
                sorted(state.generator_model_counts.items())
            ),
            "generator_model_commit_counts": dict(
                sorted(state.generator_commit_counts.items())
            ),
        },
        "num_missing": len(state.missing_provenance),
        "missing_explicitly_allowed": bool(
            state.missing_provenance and args.allow_missing_provenance
        ),
        "extraction_scope_fingerprints": sorted(
            state.extraction_scope_fingerprints
        ),
        "extraction_scope_audit": extraction_scope_audit,
        "num_observer_traces": len(state.observer_traces),
        "observer_explicitly_allowed": bool(
            state.observer_traces and args.allow_observer_traces
        ),
        "num_unverified_weight_traces": len(state.unverified_weight_traces),
        "unverified_weights_explicitly_allowed": bool(
            state.unverified_weight_traces
            and args.allow_unverified_generator_weights
        ),
        "num_unverified_chunk_topology_traces": len(
            state.unverified_chunk_topology_traces
        ),
        "unverified_chunk_topology_explicitly_allowed": bool(
            state.unverified_chunk_topology_traces
            and args.allow_unverified_chunk_topology
        ),
        "attention_axis_contract": axis_contract,
    }


def _load_audited_graph_cohort(
    args: argparse.Namespace,
    graph_config: AttentionHypergraphConfig,
    *,
    objective: str,
    preprocessing: Optional[str],
) -> _LoadedGraphCohort:
    """Audit the full extraction scope, then construct only the selected cohort."""

    _validate_limit_input_order(args)
    trace_metadata: List[_TraceMeta] = []
    graphs: List[AttentionHypergraph] = []
    trace_rows: List[Dict[str, Any]] = []
    graph_rows: List[Dict[str, Any]] = []
    state = _TraceCohortAuditState()
    wanted_generators = _parse_generator_models(args.generator_model)
    input_generator_counts: Counter[str] = Counter()
    selected_generator_counts: Counter[str] = Counter()
    num_input = 0
    num_matched_before_limit = 0
    num_selected = 0
    num_excluded_generator = 0
    num_excluded_limit = 0
    missing_generator_count = 0
    label_counts: Counter[str] = Counter()
    generator_label_counts: Dict[str, Counter[str]] = {}
    missing_supervision = 0

    for trace in _iter_input_traces(args, apply_selection=False):
        num_input += 1
        _record_extraction_scope(state, trace)
        _validate_limit_input_order(args, trace)
        generator_value = trace.metadata.get("generator_model")
        generator = (
            "<missing>"
            if generator_value in (None, "")
            else str(generator_value)
        )
        input_generator_counts[generator] += 1
        if generator == "<missing>":
            missing_generator_count += 1
        if not _trace_matches_generator_filter(trace, args):
            num_excluded_generator += 1
            continue
        num_matched_before_limit += 1
        if args.limit is not None and num_selected >= int(args.limit):
            num_excluded_limit += 1
            continue
        num_selected += 1
        selected_generator_counts[generator] += 1
        if trace.response_y is None:
            label_key = "unlabeled"
        elif float(trace.response_y) >= 0.5:
            label_key = "positive"
        else:
            label_key = "negative"
        label_counts[label_key] += 1
        generator_label_counts.setdefault(generator, Counter())[label_key] += 1
        if not _objective_available(trace, objective):
            missing_supervision += 1
            if not args.skip_unlabeled:
                raise SystemExit(
                    f"trace {trace.trace_id!r} lacks {objective} supervision; "
                    "use the matching objective or pass --skip-unlabeled explicitly"
                )
            continue

        raw_graph = _build_graph(trace, graph_config)
        _record_representation_compatibility(state, trace, graph_config, args)
        graph = (
            raw_graph
            if preprocessing is None
            else _preprocess_training_graph(raw_graph, preprocessing)
        )
        graphs.append(graph)
        trace_metadata.append(_trace_meta(trace))
        trace_rows.append(trace_summary(trace))
        graph_rows.append(_graph_summary(raw_graph, trace))

    if not graphs:
        selector = ",".join(wanted_generators) if wanted_generators else "<all>"
        raise SystemExit(
            f"no traces carry supervision for {objective} after generator selector {selector!r}"
        )
    provenance_info = _finalize_trace_cohort_audit(state, args)
    selection_info = {
        "generator_models": list(wanted_generators),
        "match_policy": "exact_case_insensitive_dataset_tag",
        "num_input_traces": num_input,
        "num_matched_before_limit": num_matched_before_limit,
        "num_selected": num_selected,
        "num_usable": len(graphs),
        "num_excluded_generator": num_excluded_generator,
        "num_excluded_limit": num_excluded_limit,
        "missing_generator_count": missing_generator_count,
        "generator_distribution_input": dict(sorted(input_generator_counts.items())),
        "generator_distribution_selected": dict(
            sorted(selected_generator_counts.items())
        ),
        "response_label_counts_selected": {
            key: int(label_counts.get(key, 0))
            for key in ("negative", "positive", "unlabeled")
        },
        "generator_response_label_counts_selected": {
            generator: {
                key: int(counts.get(key, 0))
                for key in ("negative", "positive", "unlabeled")
            }
            for generator, counts in sorted(generator_label_counts.items())
        },
    }
    return _LoadedGraphCohort(
        trace_metadata=trace_metadata,
        graphs=graphs,
        trace_rows=trace_rows,
        graph_rows=graph_rows,
        missing_supervision=missing_supervision,
        provenance_info=provenance_info,
        selection_info=selection_info,
    )


def _load_training_graphs(
    args: argparse.Namespace, graph_config: AttentionHypergraphConfig
) -> Tuple[List[_TraceMeta], List[AttentionHypergraph], int, Dict[str, Any]]:
    """Construct while streaming and retain only compact graphs/metadata."""

    cohort = _load_audited_graph_cohort(
        args,
        graph_config,
        objective=str(args.objective),
        preprocessing=str(args.preprocessing),
    )
    cohort.provenance_info["selection"] = cohort.selection_info
    return (
        cohort.trace_metadata,
        cohort.graphs,
        cohort.missing_supervision,
        cohort.provenance_info,
    )


def _explicit_split(
    traces: Sequence[_TraceMeta], args: argparse.Namespace
) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    missing = [trace.trace_id for trace in traces if trace.split is None]
    if missing:
        raise SystemExit(
            f"explicit split requested but {len(missing)} traces have no split field"
        )
    wanted = (str(args.train_split).lower(), str(args.val_split).lower(), str(args.test_split).lower())
    if len(set(wanted)) != 3:
        raise SystemExit("train/validation/test split names must be three distinct values")
    observed = {str(trace.split).lower() for trace in traces if trace.split is not None}
    unknown = observed - set(wanted)
    if unknown:
        raise SystemExit(
            f"explicit split metadata contains unrequested partitions {sorted(unknown)}; "
            "refusing to silently discard traces"
        )
    partitions = [
        [index for index, trace in enumerate(traces) if trace.split == split]
        for split in wanted
    ]
    if any(not indices for indices in partitions):
        counts = {split: sum(trace.split == split for trace in traces) for split in wanted}
        raise SystemExit(f"explicit train/val/test split is empty: {counts}")
    covered = sorted(index for partition in partitions for index in partition)
    if covered != list(range(len(traces))):
        raise SystemExit("explicit train/validation/test partitions do not cover every trace")
    _assert_group_disjoint(traces, partitions)
    return partitions[0], partitions[1], partitions[2], {
        "mode": "explicit",
        "split_names": {"train": wanted[0], "val": wanted[1], "test": wanted[2]},
    }


def _group_cv_split(
    traces: Sequence[_TraceMeta], args: argparse.Namespace
) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    official = [trace.trace_id for trace in traces if trace.split is not None]
    if official and not args.allow_resplit_official_data:
        raise SystemExit(
            f"group_cv found official split metadata on {len(official)} traces; refusing "
            "to move validation/test examples into generated folds. Remove all split fields "
            "from a deliberate development copy or pass --allow-resplit-official-data for "
            "a diagnostic-only repartition."
        )
    fallback = [trace.trace_id for trace in traces if trace.group_is_fallback]
    if fallback and not args.allow_trace_as_group:
        raise SystemExit(
            f"group CV requires a real problem/question id; {len(fallback)} traces only "
            "have trace ids. Add problem_id or pass --allow-trace-as-group for a diagnostic run."
        )
    groups: Dict[str, List[int]] = {}
    for index, trace in enumerate(traces):
        groups.setdefault(trace.group_id, []).append(index)
    folds = int(args.folds)
    if folds < 3:
        raise SystemExit("--folds must be at least 3 (separate train, validation, and test groups)")
    if len(groups) < folds:
        raise SystemExit(f"need at least {folds} groups, found {len(groups)}")

    balance_names = (
        "groups",
        "traces",
        "negative_traces",
        "positive_traces",
        "early_errors",
        "middle_errors",
        "late_errors",
        "response_tokens",
    )

    def group_vector(indices: Sequence[int]) -> np.ndarray:
        vector = np.zeros(len(balance_names), dtype=np.float64)
        vector[0] = 1.0
        vector[1] = float(len(indices))
        for index in indices:
            trace = traces[index]
            if trace.response_label == 0:
                vector[2] += 1.0
            elif trace.response_label == 1:
                vector[3] += 1.0
            if trace.gold_step is not None and trace.gold_step >= 0 and trace.num_steps > 0:
                relative = (float(trace.gold_step) + 0.5) / float(trace.num_steps)
                position = 4 if relative <= 1.0 / 3.0 else 5 if relative <= 2.0 / 3.0 else 6
                vector[position] += 1.0
            vector[7] += float(trace.num_response_tokens)
        return vector

    rng = np.random.default_rng(int(args.seed))
    enriched = [(group_id, indices, group_vector(indices)) for group_id, indices in groups.items()]
    rng.shuffle(enriched)
    total_vector = np.sum([item[2] for item in enriched], axis=0)
    target = total_vector / float(folds)
    scale = np.maximum(target, 1.0)
    # Place groups that contribute most to a rare balance dimension first.
    enriched.sort(
        key=lambda item: (float(np.max(item[2] / scale)), len(item[1])), reverse=True
    )
    buckets: List[List[int]] = [[] for _ in range(folds)]
    bucket_vectors = np.zeros((folds, len(balance_names)), dtype=np.float64)
    for _, indices, vector in enriched:
        def assignment_cost(fold: int) -> Tuple[float, int, int]:
            proposed = bucket_vectors.copy()
            proposed[fold] += vector
            imbalance = float(np.sum(((proposed - target) / scale) ** 2))
            return imbalance, len(buckets[fold]), fold

        destination = min(range(folds), key=assignment_cost)
        buckets[destination].extend(indices)
        bucket_vectors[destination] += vector
    test_fold = int(args.fold_index)
    if not 0 <= test_fold < folds:
        raise SystemExit(f"--fold-index must lie in [0,{folds})")
    val_fold = (test_fold + 1) % folds
    test = sorted(buckets[test_fold])
    val = sorted(buckets[val_fold])
    train = sorted(
        index
        for fold, bucket in enumerate(buckets)
        if fold not in {test_fold, val_fold}
        for index in bucket
    )
    partitions = (train, val, test)
    if any(not indices for indices in partitions):
        raise SystemExit("group-fold assignment produced an empty partition")
    _assert_group_disjoint(traces, partitions)
    return train, val, test, {
        "mode": "group_cv",
        "folds": folds,
        "test_fold": test_fold,
        "val_fold": val_fold,
        "seed": int(args.seed),
        "assignment": "greedy_group_stratified_response_error_position_length",
        "fold_balance": [
            {
                name: int(round(bucket_vectors[fold, position]))
                for position, name in enumerate(balance_names)
            }
            for fold in range(folds)
        ],
    }


def _fixed_holdout_split(
    traces: Sequence[_TraceMeta], args: argparse.Namespace
) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    """Create one deterministic, problem-disjoint train/validation/test split.

    This is the ProcessBench analogue of the original project's fixed test
    protocol.  Because ProcessBench does not carry that project's external
    RAGTruth train/test directories, the holdout is created once from problem
    groups and is controlled by a split seed that is independent of model
    initialization seeds.
    """

    official = [trace.trace_id for trace in traces if trace.split is not None]
    if official and not args.allow_resplit_official_data:
        raise SystemExit(
            f"fixed_holdout found official split metadata on {len(official)} traces; "
            "use --split-mode explicit instead of replacing the official test set, or "
            "pass --allow-resplit-official-data for a diagnostic-only repartition"
        )
    fallback = [trace.trace_id for trace in traces if trace.group_is_fallback]
    if fallback and not args.allow_trace_as_group:
        raise SystemExit(
            f"fixed holdout requires a real problem/question id; {len(fallback)} "
            "traces only have trace ids. Add problem_id or pass --allow-trace-as-group "
            "for a diagnostic run."
        )

    val_ratio = float(args.val_ratio)
    test_ratio = float(args.test_ratio)
    train_ratio = 1.0 - val_ratio - test_ratio
    if not 0.0 < val_ratio < 1.0 or not 0.0 < test_ratio < 1.0 or train_ratio <= 0.0:
        raise SystemExit(
            "fixed holdout requires 0 < val_ratio, test_ratio < 1 and "
            "val_ratio + test_ratio < 1"
        )

    groups: Dict[str, List[int]] = {}
    for index, trace in enumerate(traces):
        groups.setdefault(trace.group_id, []).append(index)
    if len(groups) < 3:
        raise SystemExit("fixed holdout requires at least three problem groups")

    balance_names = (
        "groups",
        "traces",
        "negative_traces",
        "positive_traces",
        "early_errors",
        "middle_errors",
        "late_errors",
        "response_tokens",
    )

    def group_vector(indices: Sequence[int]) -> np.ndarray:
        vector = np.zeros(len(balance_names), dtype=np.float64)
        vector[0] = 1.0
        vector[1] = float(len(indices))
        for index in indices:
            trace = traces[index]
            if trace.response_label == 0:
                vector[2] += 1.0
            elif trace.response_label == 1:
                vector[3] += 1.0
            if trace.gold_step is not None and trace.gold_step >= 0 and trace.num_steps > 0:
                relative = (float(trace.gold_step) + 0.5) / float(trace.num_steps)
                position = 4 if relative <= 1.0 / 3.0 else 5 if relative <= 2.0 / 3.0 else 6
                vector[position] += 1.0
            vector[7] += float(trace.num_response_tokens)
        return vector

    rng = np.random.default_rng(int(args.split_seed))
    enriched = [(group_id, indices, group_vector(indices)) for group_id, indices in groups.items()]
    rng.shuffle(enriched)
    total_vector = np.sum([item[2] for item in enriched], axis=0)
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    targets = ratios[:, None] * total_vector[None, :]
    scales = np.maximum(targets, 1.0)
    global_scale = np.maximum(total_vector, 1.0)
    # Difficult and class-informative groups are placed first. The shuffled
    # order remains the deterministic tie breaker for otherwise equal groups.
    enriched.sort(
        key=lambda item: (
            float(np.max(item[2] / global_scale)),
            float(item[2][1]),
        ),
        reverse=True,
    )

    partitions: List[List[int]] = [[], [], []]
    partition_vectors = np.zeros_like(targets)
    for _, indices, vector in enriched:

        def assignment_cost(partition: int) -> Tuple[float, float, int]:
            proposed = partition_vectors.copy()
            proposed[partition] += vector
            fit = float(np.sum(((proposed - targets) / scales) ** 2))
            overflow = float(
                np.sum((np.maximum(proposed - targets, 0.0) / scales) ** 2)
            )
            fullness = float(
                proposed[partition, 0] / max(targets[partition, 0], 1.0)
            )
            return fit + 2.0 * overflow, fullness, partition

        destination = min(range(3), key=assignment_cost)
        partitions[destination].extend(indices)
        partition_vectors[destination] += vector

    train, val, test = (sorted(indices) for indices in partitions)
    if any(not indices for indices in (train, val, test)):
        raise SystemExit(
            "fixed holdout assignment produced an empty partition; increase the "
            "cohort or adjust validation/test ratios"
        )
    _assert_group_disjoint(traces, (train, val, test))
    partition_names = ("train", "validation", "test")
    partition_indices = (train, val, test)
    return train, val, test, {
        "mode": "fixed_holdout",
        "split_seed": int(args.split_seed),
        "ratios": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": test_ratio,
        },
        "assignment": "greedy_group_stratified_response_error_position_length",
        "partition_balance": {
            name: {
                field: int(round(partition_vectors[position, field_index]))
                for field_index, field in enumerate(balance_names)
            }
            for position, name in enumerate(partition_names)
        },
        "partition_trace_ids": {
            name: [traces[index].trace_id for index in indices]
            for name, indices in zip(partition_names, partition_indices)
        },
        "partition_group_ids": {
            name: sorted({traces[index].group_id for index in indices})
            for name, indices in zip(partition_names, partition_indices)
        },
    }


def _assert_group_disjoint(
    traces: Sequence[_TraceMeta], partitions: Sequence[Sequence[int]]
) -> None:
    group_sets = [{traces[index].group_id for index in part} for part in partitions]
    if (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[1] & group_sets[2]
    ):
        raise SystemExit("problem/group leakage detected across train/validation/test")


def _make_split(
    traces: Sequence[_TraceMeta], args: argparse.Namespace
) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    mode = str(args.split_mode)
    if mode == "auto":
        present = [trace.split for trace in traces if trace.split is not None]
        explicit_names = {
            str(args.train_split).lower(),
            str(args.val_split).lower(),
            str(args.test_split).lower(),
        }
        if not present:
            mode = "group_cv"
        elif len(present) != len(traces) or set(present) != explicit_names:
            raise SystemExit(
                "split_mode=auto found partial or non-matching split metadata; refusing "
                "to mix an official test partition into generated group folds. Supply all "
                "train/validation/test splits (or matching --*-split names), or remove split "
                "metadata from every trace and request group_cv explicitly."
            )
        else:
            mode = "explicit"
    if mode == "explicit":
        return _explicit_split(traces, args)
    if mode == "fixed_holdout":
        return _fixed_holdout_split(traces, args)
    return _group_cv_split(traces, args)


def _assert_partition_class_coverage(
    graphs: Sequence[AttentionHypergraph],
    partitions: Mapping[str, Sequence[int]],
    objective: str,
    *,
    allow_single_class: bool,
) -> Dict[str, Dict[str, int]]:
    def trace_label(graph: AttentionHypergraph) -> int:
        if objective == "step_bce":
            return int(graph.gold_step is not None and graph.gold_step >= 0)
        if objective == "response_bce":
            return int(float(graph.response_y) >= 0.5)
        assert graph.token_y is not None and graph.token_label_mask is not None
        return int(np.any(graph.token_y[graph.token_label_mask] == 1.0))

    report: Dict[str, Dict[str, int]] = {}
    failures = []
    for name, indices in partitions.items():
        labels = np.asarray([trace_label(graphs[index]) for index in indices], dtype=int)
        report[name] = {
            "negative_traces": int(np.sum(labels == 0)),
            "positive_traces": int(np.sum(labels == 1)),
        }
        if len(np.unique(labels)) < 2:
            failures.append(name)
    if failures and not allow_single_class:
        raise SystemExit(
            "single-class trace partitions make held-out detection metrics undefined: "
            f"{failures}. Change the group fold/seed or explicitly pass "
            "--allow-single-class-partition for a diagnostic-only run."
        )
    return report


def _partition_distribution_report(
    traces: Sequence[_TraceMeta], partitions: Mapping[str, Sequence[int]]
) -> Dict[str, Dict[str, Any]]:
    """Expose length and first-error-position balance for every held-out split."""

    report: Dict[str, Dict[str, Any]] = {}
    for name, indices in partitions.items():
        response_lengths = np.asarray(
            [traces[index].num_response_tokens for index in indices], dtype=float
        )
        step_counts = np.asarray([traces[index].num_steps for index in indices], dtype=float)
        relative_errors = np.asarray(
            [
                (float(traces[index].gold_step) + 0.5) / float(traces[index].num_steps)
                for index in indices
                if traces[index].gold_step is not None
                and traces[index].gold_step >= 0
                and traces[index].num_steps > 0
            ],
            dtype=float,
        )

        def distribution(values: np.ndarray) -> Dict[str, Any]:
            if not len(values):
                return {"n": 0, "mean": None, "q25": None, "median": None, "q75": None}
            return {
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.5)),
                "q75": float(np.quantile(values, 0.75)),
            }

        report[name] = {
            "response_tokens": distribution(response_lengths),
            "num_steps": distribution(step_counts),
            "relative_first_error_position": distribution(relative_errors),
        }
    return report


def _binary_metrics(labels: Sequence[float], scores: Sequence[float]) -> Dict[str, Any]:
    raw_y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(scores, dtype=np.float64).reshape(-1)
    if raw_y.shape != p.shape:
        raise ValueError("binary labels and scores must be aligned vectors")
    if not np.isfinite(raw_y).all() or not np.isin(raw_y, (0.0, 1.0)).all():
        raise ValueError("binary metric labels must all be finite 0/1 values")
    if not np.isfinite(p).all():
        raise RuntimeError("binary metric scores contain NaN or infinity")
    y = raw_y.astype(np.int64)
    result: Dict[str, Any] = {
        "n": int(len(y)),
        "positives": int(np.sum(y == 1)),
        "prevalence": None if not len(y) else float(np.mean(y)),
    }
    if not len(y) or len(np.unique(y)) < 2:
        result.update({"auroc": None, "aupr": None, "accuracy_0.5": None})
        return result

    order = np.argsort(p, kind="stable")
    sorted_scores = p[order]
    ranks = np.empty(len(y), dtype=np.float64)
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    n_pos = int(np.sum(y == 1))
    n_neg = len(y) - n_pos
    auroc = (float(np.sum(ranks[y == 1])) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    descending = np.argsort(-p, kind="stable")
    y_desc, p_desc = y[descending], p[descending]
    tp = fp = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < len(y_desc):
        end = start + 1
        while end < len(y_desc) and p_desc[end] == p_desc[start]:
            end += 1
        tp += int(np.sum(y_desc[start:end] == 1))
        fp += int(np.sum(y_desc[start:end] == 0))
        recall = tp / n_pos
        precision = tp / (tp + fp)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    result.update(
        {
            "auroc": float(auroc),
            "aupr": float(average_precision),
            "accuracy_0.5": float(np.mean((p >= 0.5) == y)),
        }
    )
    return result


def _trace_metrics_by_generator(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute trace-level detection metrics without pooling generators together."""

    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        generator_value = row.get("generator_model")
        generator = (
            "<missing>"
            if generator_value in (None, "")
            else str(generator_value)
        )
        if row.get("label") is None or row.get("score") is None:
            raise ValueError("trace prediction rows require label and score")
        bucket = grouped.setdefault(generator, {"labels": [], "scores": []})
        bucket["labels"].append(float(row["label"]))
        bucket["scores"].append(float(row["score"]))
    return {
        generator: _binary_metrics(values["labels"], values["scores"])
        for generator, values in sorted(grouped.items())
    }


def _tie_aware_localization_rank(
    scores: Sequence[float], gold_step: int, valid_mask: Sequence[bool]
) -> Tuple[float, float]:
    """Rank the gold step among all inference-visible steps, never a gold-trimmed risk set."""

    probability = np.asarray(scores, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    gold_step = int(gold_step)
    if probability.ndim != 1 or valid.shape != probability.shape:
        raise ValueError("scores and valid_mask must be aligned vectors")
    if not 0 <= gold_step < len(probability) or not valid[gold_step]:
        raise ValueError("gold_step must select a valid localization candidate")
    candidates = np.flatnonzero(valid)
    if not np.isfinite(probability[candidates]).all():
        raise RuntimeError("localization candidate scores contain NaN or infinity")
    gold_score = probability[gold_step]
    better = int(np.sum(probability[candidates] > gold_score))
    tied = int(np.sum(probability[candidates] == gold_score))
    rank = 1.0 + better + 0.5 * (tied - 1)
    return rank, float(better == 0 and tied == 1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _loss_for_graph(
    model,
    graph: AttentionHypergraph,
    objective: str,
    *,
    pooling: str,
    temperature: float,
    pos_weight: Optional[float],
):
    import torch

    from .objectives import (
        make_first_error_targets,
        pool_token_logits_to_response,
        pool_token_logits_to_steps,
        response_bce,
        step_bce,
        token_bce,
    )

    logits = model(graph)
    device = logits.device
    if objective == "token_bce":
        if graph.token_y is None or graph.token_label_mask is None:
            raise ValueError("token_bce requires exact token_y and token_label_mask")
        target = torch.as_tensor(graph.token_y, dtype=logits.dtype, device=device)
        position = torch.arange(len(target), device=device)
        mask = torch.as_tensor(graph.token_label_mask, dtype=torch.bool, device=device)
        mask &= position >= int(graph.response_idx)
        return token_bce(
            logits,
            target.clamp(0, 1),
            exact_label_mask=mask,
            pos_weight=pos_weight,
        )
    if objective == "step_bce":
        if graph.step_ranges is None or graph.gold_step is None:
            raise ValueError("step_bce requires step_ranges and gold_step")
        scores = pool_token_logits_to_steps(
            logits, graph.step_ranges, pooling=pooling, temperature=temperature
        )
        target, risk = make_first_error_targets(
            len(graph.step_ranges), int(graph.gold_step), device=device
        )
        if graph.step_loss_mask is not None:
            risk &= torch.as_tensor(graph.step_loss_mask, dtype=torch.bool, device=device)
        return step_bce(scores, target, risk_mask=risk, pos_weight=pos_weight)
    if objective == "response_bce":
        if graph.response_y is None:
            raise ValueError("response_bce requires response_y")
        score = pool_token_logits_to_response(
            logits,
            graph.response_idx,
            pooling=pooling,
            temperature=temperature,
        ).reshape(1)
        target = torch.as_tensor([graph.response_y], dtype=logits.dtype, device=device)
        return response_bce(score, target, pos_weight=pos_weight)
    raise ValueError(f"unknown objective {objective!r}")


def _targets_for_weight(graph: AttentionHypergraph, objective: str) -> np.ndarray:
    if objective == "token_bce":
        assert graph.token_y is not None and graph.token_label_mask is not None
        target = np.asarray(graph.token_y[graph.response_idx :], float)
        mask = np.asarray(graph.token_label_mask[graph.response_idx :], bool)
        return target[mask]
    if objective == "step_bce":
        assert graph.step_ranges is not None and graph.gold_step is not None
        n = len(graph.step_ranges)
        target = np.zeros(n, dtype=float)
        risk = np.ones(n, dtype=bool)
        if graph.gold_step >= 0:
            target[graph.gold_step] = 1.0
            risk[graph.gold_step + 1 :] = False
        if graph.step_loss_mask is not None:
            risk &= np.asarray(graph.step_loss_mask, bool)
        return target[risk]
    assert graph.response_y is not None
    return np.asarray([graph.response_y], float)


def _resolve_pos_weight(
    raw: Any,
    graphs: Sequence[AttentionHypergraph],
    train_indices: Sequence[int],
    objective: str,
    *,
    maximum: Optional[float],
) -> Optional[float]:
    text = str(raw).strip().lower()
    if text in {"none", "off", "false"}:
        return None
    if text != "auto":
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise SystemExit("--pos-weight must be auto, none, or a positive number")
        return value
    target = np.concatenate([_targets_for_weight(graphs[index], objective) for index in train_indices])
    positives = int(np.sum(target == 1))
    negatives = int(np.sum(target == 0))
    if positives == 0 or negatives == 0:
        raise SystemExit(
            f"training partition needs both classes for {objective}; got {positives} positive, {negatives} negative"
        )
    value = float(negatives / positives)
    return value if maximum is None else min(value, maximum)


def _evaluate(
    model,
    graphs: Sequence[AttentionHypergraph],
    traces: Sequence[_TraceMeta],
    indices: Sequence[int],
    objective: str,
    *,
    pooling: str,
    temperature: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import torch

    from .objectives import pool_token_logits_to_response, pool_token_logits_to_steps

    labels: List[float] = []
    scores: List[float] = []
    rows: List[Dict[str, Any]] = []
    localization_top1: List[float] = []
    localization_rr: List[float] = []
    trace_labels: List[float] = []
    trace_scores: List[float] = []
    model.eval()
    with torch.no_grad():
        for index in indices:
            graph, trace = graphs[index], traces[index]
            logits = model(graph)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(
                    f"non-finite evaluation logits for trace {trace.trace_id!r}"
                )
            row_context = {
                "trace_id": trace.trace_id,
                "group_id": trace.group_id,
                "generator_model": trace.generator_model,
                "response_tokens": int(trace.num_response_tokens),
                "num_steps": int(trace.num_steps),
                "gold_relative_position": (
                    None
                    if trace.gold_step is None or trace.gold_step < 0 or trace.num_steps < 1
                    else (float(trace.gold_step) + 0.5) / float(trace.num_steps)
                ),
            }
            if objective == "token_bce":
                target = np.asarray(graph.token_y, float)
                probability = torch.sigmoid(logits).detach().cpu().numpy()
                mask = np.arange(len(target)) >= graph.response_idx
                mask &= np.asarray(graph.token_label_mask, bool)
                labels.extend(target[mask].tolist())
                scores.extend(probability[mask].tolist())
                rows.append(
                    {
                        **row_context,
                        "n_scored": int(mask.sum()),
                        "score": float(np.mean(probability[mask])),
                        "label": float(np.any(target[mask] == 1)),
                    }
                )
            elif objective == "step_bce":
                step_logits = pool_token_logits_to_steps(
                    logits, graph.step_ranges, pooling=pooling, temperature=temperature
                )
                probability = torch.sigmoid(step_logits).detach().cpu().numpy()
                target = np.zeros(len(probability), dtype=float)
                # Inference/localization must expose every step.  A
                # step_loss_mask may describe missing supervision, but it must
                # never prune the deployable score surface or threshold search.
                inference_steps = np.ones(len(probability), dtype=bool)
                supervision_steps = np.ones(len(probability), dtype=bool)
                if graph.step_loss_mask is not None:
                    supervision_steps &= np.asarray(graph.step_loss_mask, bool)
                risk = supervision_steps.copy()
                if graph.gold_step >= 0:
                    target[graph.gold_step] = 1.0
                    risk[graph.gold_step + 1 :] = False
                labels.extend(target[risk].tolist())
                scores.extend(probability[risk].tolist())
                row = {
                    **row_context,
                    "n_scored": int(risk.sum()),
                    "n_localization_candidates": int(inference_steps.sum()),
                    # Deployable trace score: never use gold_step to remove a
                    # high-scoring later step from the prediction surface.
                    "score": float(np.max(probability[inference_steps])),
                    "risk_set_score": float(np.max(probability[risk])),
                    "label": float(graph.gold_step >= 0),
                    "gold_step": int(graph.gold_step),
                    "step_token_lengths_json": json.dumps(
                        (
                            np.asarray(graph.step_ranges, dtype=np.int64)[:, 1]
                            - np.asarray(graph.step_ranges, dtype=np.int64)[:, 0]
                        ).tolist(),
                        separators=(",", ":"),
                    ),
                    "step_probabilities_json": json.dumps(
                        probability.tolist(), separators=(",", ":")
                    ),
                    "valid_steps_json": json.dumps(
                        inference_steps.astype(int).tolist(), separators=(",", ":")
                    ),
                    "supervision_steps_json": json.dumps(
                        supervision_steps.astype(int).tolist(), separators=(",", ":")
                    ),
                }
                trace_labels.append(float(graph.gold_step >= 0))
                trace_scores.append(row["score"])
                if graph.gold_step >= 0 and inference_steps[graph.gold_step]:
                    # Localization is evaluated against every valid step.
                    # The training risk set may exclude post-error consequences,
                    # but gold labels cannot prune inference candidates.
                    rank, top1 = _tie_aware_localization_rank(
                        probability, int(graph.gold_step), inference_steps
                    )
                    localization_top1.append(top1)
                    localization_rr.append(1.0 / rank)
                    row.update({"gold_rank": rank, "unique_top1": top1})
                rows.append(row)
            else:
                response_logit = pool_token_logits_to_response(
                    logits,
                    graph.response_idx,
                    pooling=pooling,
                    temperature=temperature,
                )
                probability = float(torch.sigmoid(response_logit).item())
                labels.append(float(graph.response_y))
                scores.append(probability)
                rows.append(
                    {
                        **row_context,
                        "n_scored": 1,
                        "score": probability,
                        "label": float(graph.response_y),
                    }
                )
    metrics = _binary_metrics(labels, scores)
    if objective == "step_bce":
        metrics["binary_metric_scope"] = "first_error_risk_set"
        metrics["risk_set"] = {
            key: value for key, value in metrics.items() if key != "risk_set"
        }
        metrics["trace_detection"] = _binary_metrics(trace_labels, trace_scores)
        metrics["error_trace_unique_top1"] = (
            None if not localization_top1 else float(np.mean(localization_top1))
        )
        metrics["error_trace_mrr"] = (
            None if not localization_rr else float(np.mean(localization_rr))
        )
        metrics["num_localized_error_traces"] = len(localization_rr)
    return metrics, rows


def _first_crossing_metrics(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> Dict[str, Any]:
    """Evaluate a validation-chosen first-threshold-crossing rule on all traces."""

    gold_values: List[int] = []
    predicted_values: List[int] = []
    for row in rows:
        probabilities = np.asarray(json.loads(str(row["step_probabilities_json"])), float)
        valid = np.asarray(json.loads(str(row["valid_steps_json"])), bool)
        hits = np.flatnonzero(valid & (probabilities >= float(threshold)))
        predicted_values.append(-1 if not len(hits) else int(hits[0]))
        gold_values.append(int(row["gold_step"]))
    gold = np.asarray(gold_values, dtype=int)
    predicted = np.asarray(predicted_values, dtype=int)
    gold_error = gold >= 0
    predicted_error = predicted >= 0
    tp = int(np.sum(gold_error & predicted_error))
    fp = int(np.sum(~gold_error & predicted_error))
    fn = int(np.sum(gold_error & ~predicted_error))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both = gold_error & predicted_error
    return {
        "threshold": float(threshold),
        "num_traces": int(len(gold)),
        "first_error_exact_accuracy": float(np.mean(predicted == gold)),
        "response_precision": float(precision),
        "response_recall": float(recall),
        "response_f1": float(f1),
        "correct_trace_false_alarm_rate": (
            None if not np.any(~gold_error) else float(np.mean(predicted[~gold_error] >= 0))
        ),
        "error_trace_exact_localization": (
            None if not np.any(gold_error) else float(np.mean(predicted[gold_error] == gold[gold_error]))
        ),
        "detected_error_step_mae": (
            None if not np.any(both) else float(np.mean(np.abs(predicted[both] - gold[both])))
        ),
    }


def _select_first_crossing_threshold(
    validation_rows: Sequence[Mapping[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    probabilities = []
    for row in validation_rows:
        score = np.asarray(json.loads(str(row["step_probabilities_json"])), float)
        valid = np.asarray(json.loads(str(row["valid_steps_json"])), bool)
        probabilities.extend(score[valid].tolist())
    if not probabilities:
        raise ValueError("validation rows contain no valid step probabilities")
    maximum = float(np.max(probabilities))
    candidates = sorted(
        {float(value) for value in probabilities}
        | {float(np.nextafter(maximum, math.inf))}
    )
    ranked = []
    for threshold in candidates:
        metrics = _first_crossing_metrics(validation_rows, threshold)
        ranked.append(
            (
                float(metrics["first_error_exact_accuracy"]),
                float(metrics["response_f1"]),
                float(threshold),
                metrics,
            )
        )
    _, _, threshold, metrics = max(ranked, key=lambda item: item[:3])
    return threshold, metrics


def _write_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def command_train(args: argparse.Namespace) -> Dict[str, Any]:
    graph_config = _graph_config_from_args(args)
    localization_objective = args.objective in {"token_bce", "step_bce"}
    offline_issues = []
    if localization_objective and graph_config.propagation_mode == "symmetric":
        offline_issues.append("symmetric propagation sends later receiver messages to past nodes")
    if localization_objective and str(args.preprocessing) != "none":
        offline_issues.append(
            "per-graph z-score uses mean/variance from future and post-error tokens"
        )
    if args.message_operator == "pairwise" and graph_config.propagation_mode != "receiver":
        raise SystemExit(
            "the pairwise query-key baseline is directed; use --propagation-mode receiver"
        )
    if args.message_operator == "pairwise" and graph_config.edge_attr_mode != "faithful":
        raise SystemExit(
            "the clean pairwise baseline requires faithful 3-D relation attributes; "
            "extended attributes contain set-size statistics"
        )
    if args.message_operator == "pairwise" and str(args.preprocessing) != "none":
        raise SystemExit(
            "the clean pairwise baseline requires --preprocessing none so its raw "
            "query-key attention and the hypergraph relation attributes share one scale"
        )
    if offline_issues and not args.allow_offline_full_context:
        raise SystemExit(
            f"{args.objective} is configured with future context: "
            + "; ".join(offline_issues)
            + ". Use --propagation-mode receiver --preprocessing none for causal "
            "localization, or explicitly pass --allow-offline-full-context for a "
            "legacy offline comparison."
        )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependent
        raise SystemExit("training requires PyTorch; inspect/build remain available without it") from exc

    from .model import build_model

    input_scope = _input_scope(args)

    output = Path(args.output).resolve()
    existing_run_files = [
        output / name for name in ("checkpoint.pt", "config.json", "results.json")
        if (output / name).exists()
    ]
    if existing_run_files and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing run {output}; pass --overwrite"
        )
    _set_seed(int(args.seed))
    traces, graphs, missing_supervision, provenance_info = _load_training_graphs(
        args, graph_config
    )
    node_dims = {int(graph.x.shape[1]) for graph in graphs}
    edge_dims = {int(graph.he_attr.shape[1]) for graph in graphs}
    if len(node_dims) != 1 or len(edge_dims) != 1:
        raise SystemExit(
            f"all graphs must share feature dimensions; node={sorted(node_dims)}, edge={sorted(edge_dims)}"
        )
    train_idx, val_idx, test_idx, split_info = _make_split(traces, args)
    split_info["class_coverage"] = _assert_partition_class_coverage(
        graphs,
        {"train": train_idx, "val": val_idx, "test": test_idx},
        args.objective,
        allow_single_class=bool(args.allow_single_class_partition),
    )
    split_info["distributions"] = _partition_distribution_report(
        traces, {"train": train_idx, "val": val_idx, "test": test_idx}
    )
    maximum_pos_weight = (
        None if float(args.max_pos_weight) <= 0 else float(args.max_pos_weight)
    )
    pos_weight = _resolve_pos_weight(
        args.pos_weight,
        graphs,
        train_idx,
        args.objective,
        maximum=maximum_pos_weight,
    )
    device = str(args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is false")
    incidence_weighting = graph_config.incidence_weight_mode
    directed = graph_config.propagation_mode == "receiver"
    model = build_model(
        node_dim=next(iter(node_dims)),
        hedge_dim=next(iter(edge_dims)),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.model_layers),
        dropout=float(args.dropout),
        residual=bool(args.residual),
        directed_receiver_only=directed,
        incidence_weighting=incidence_weighting,
        receiver_source_interaction=bool(args.receiver_source_interaction),
        mlp_norm=bool(args.mlp_norm),
        classifier_norm=bool(args.classifier_norm),
        init_weights=bool(args.init_weights),
        message_operator=str(args.message_operator),
    ).to(device)
    total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_parameters = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )

    accumulation = max(1, int(args.grad_accumulation))
    steps_per_epoch = max(1, math.ceil(len(train_idx) / accumulation))
    total_steps = max(1, int(args.epochs) * steps_per_epoch)
    warmup_steps = int(float(args.warmup_ratio) * total_steps)

    def learning_rate_factor(step: int) -> float:
        if args.scheduler == "constant":
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)

    rng = np.random.default_rng(int(args.seed))
    best_state = None
    best_epoch = -1
    best_value = -math.inf
    stale = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = np.asarray(train_idx, dtype=int).copy()
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for position, index in enumerate(order, start=1):
            accumulation_group_start = ((position - 1) // accumulation) * accumulation
            accumulation_group_size = min(
                accumulation, len(order) - accumulation_group_start
            )
            loss = _loss_for_graph(
                model,
                graphs[int(index)],
                args.objective,
                pooling=args.pooling,
                temperature=float(args.pooling_temperature),
                pos_weight=pos_weight,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}, trace {traces[int(index)].trace_id}")
            (loss / accumulation_group_size).backward()
            losses.append(float(loss.detach().cpu()))
            if position % accumulation == 0 or position == len(order):
                if float(args.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        val_metrics, _ = _evaluate(
            model,
            graphs,
            traces,
            val_idx,
            args.objective,
            pooling=args.pooling,
            temperature=float(args.pooling_temperature),
        )
        monitor = val_metrics.get(str(args.monitor))
        monitor_value = -math.inf if monitor is None else float(monitor)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val_metrics}
        history.append(row)
        print(json.dumps(_finite_json(row), ensure_ascii=False, sort_keys=True), flush=True)
        if monitor_value > best_value + float(args.min_delta):
            best_value = monitor_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError(
            f"validation metric {args.monitor!r} was undefined in every epoch; check class coverage"
        )
    model.load_state_dict(best_state)
    train_metrics, train_rows = _evaluate(
        model, graphs, traces, train_idx, args.objective,
        pooling=args.pooling, temperature=float(args.pooling_temperature)
    )
    val_metrics, val_rows = _evaluate(
        model, graphs, traces, val_idx, args.objective,
        pooling=args.pooling, temperature=float(args.pooling_temperature)
    )
    test_metrics, test_rows = _evaluate(
        model, graphs, traces, test_idx, args.objective,
        pooling=args.pooling, temperature=float(args.pooling_temperature)
    )
    trace_detection_by_generator = {
        "train": _trace_metrics_by_generator(train_rows),
        "val": _trace_metrics_by_generator(val_rows),
        "test": _trace_metrics_by_generator(test_rows),
    }
    first_crossing_threshold = None
    if args.objective == "step_bce":
        first_crossing_threshold, val_operating_metrics = _select_first_crossing_threshold(
            val_rows
        )
        train_metrics["first_crossing"] = _first_crossing_metrics(
            train_rows, first_crossing_threshold
        )
        val_metrics["first_crossing"] = val_operating_metrics
        test_metrics["first_crossing"] = _first_crossing_metrics(
            test_rows, first_crossing_threshold
        )

    output.mkdir(parents=True, exist_ok=True)
    loader_configuration = asdict(_load_config_from_args(args))
    loader_configuration.pop("require_attention", None)
    configuration = {
        "loader": loader_configuration,
        "graph": asdict(graph_config),
        "data": {
            "input_scope": input_scope,
            "use_activation": graph_config.node_feature_mode != "attention_diagonal",
            "skip_unlabeled": bool(args.skip_unlabeled),
            "allow_missing_provenance": bool(args.allow_missing_provenance),
            "allow_incomplete_extraction_scope": bool(
                args.allow_incomplete_extraction_scope
            ),
            "allow_multiple_input_datasets": bool(
                args.allow_multiple_input_datasets
            ),
            "allow_observer_traces": bool(args.allow_observer_traces),
            "allow_unverified_generator_weights": bool(
                args.allow_unverified_generator_weights
            ),
            "allow_unverified_chunk_topology": bool(
                args.allow_unverified_chunk_topology
            ),
            "trace_provenance": provenance_info,
            "preprocessing": str(args.preprocessing),
            "allow_offline_full_context": bool(args.allow_offline_full_context),
        },
        "model": {
            "hidden_dim": int(args.hidden_dim),
            "model_layers": int(args.model_layers),
            "dropout": float(args.dropout),
            "residual": bool(args.residual),
            "receiver_source_interaction": bool(args.receiver_source_interaction),
            "mlp_norm": bool(args.mlp_norm),
            "classifier_norm": bool(args.classifier_norm),
            "init_weights": bool(args.init_weights),
            "message_operator": str(args.message_operator),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        },
        "training": {
            "objective": args.objective,
            "pooling": args.pooling,
            "pooling_temperature": float(args.pooling_temperature),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "pos_weight": pos_weight,
            "max_pos_weight": float(args.max_pos_weight),
            "seed": int(args.seed),
            "split_mode": str(args.split_mode),
            "split_seed": int(args.split_seed),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
            "allow_single_class_partition": bool(args.allow_single_class_partition),
            "allow_resplit_official_data": bool(args.allow_resplit_official_data),
            "allow_trace_as_group": bool(args.allow_trace_as_group),
            "folds": int(args.folds),
            "fold_index": int(args.fold_index),
            "train_split": str(args.train_split),
            "val_split": str(args.val_split),
            "test_split": str(args.test_split),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "monitor": str(args.monitor),
            "grad_accumulation": int(args.grad_accumulation),
            "grad_clip": float(args.grad_clip),
            "scheduler": str(args.scheduler),
            "warmup_ratio": float(args.warmup_ratio),
            "device": str(args.device),
        },
        "runtime": _training_runtime_provenance(torch),
    }
    resolved = {
        "node_dim": next(iter(node_dims)),
        "edge_dim": next(iter(edge_dims)),
        "directed_receiver_only": directed,
        "model_incidence_weighting": incidence_weighting,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "pos_weight": pos_weight,
        "split": split_info,
        "first_crossing_threshold_from_validation": first_crossing_threshold,
    }
    checkpoint = {
        "model_state": best_state,
        "configuration": configuration,
        "resolved": resolved,
        "best_epoch": best_epoch,
        "validation_monitor": {"name": args.monitor, "value": best_value},
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    _write_json(output / "config.json", configuration)
    _write_json(output / "history.json", history)
    _write_predictions(output / "predictions_train.csv", train_rows)
    _write_predictions(output / "predictions_val.csv", val_rows)
    _write_predictions(output / "predictions_test.csv", test_rows)
    result = {
        "command": "train",
        "best_epoch": best_epoch,
        "validation_monitor": {"name": args.monitor, "value": best_value},
        "partition_sizes": {
            "train": len(train_idx), "val": len(val_idx), "test": len(test_idx)
        },
        "num_skipped_unlabeled": int(missing_supervision),
        "metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
        "trace_detection_by_generator": trace_detection_by_generator,
        "output": str(output),
        "configuration": configuration,
        "resolved": resolved,
    }
    _write_json(output / "results.json", result)
    print(json.dumps(_finite_json(result), ensure_ascii=False, indent=2, sort_keys=True))
    return result


def _add_common_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+", help="attention .npz/.pt files, directories, or globs")
    parser.add_argument("--config", help="optional JSON defaults; explicit CLI flags take precedence")
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "load at most this many traces after generator filtering; with multiple "
            "trace roots, materialize the limited cohort before sharding"
        ),
    )
    parser.add_argument(
        "--generator-model",
        help=(
            "exact dataset generator tag, or comma-separated tags; filtering occurs "
            "before --limit and never guesses aliases from the observer model path"
        ),
    )
    parser.add_argument("--no-recursive", action="store_true", help="do not recurse into input directories")
    parser.add_argument("--step-end", choices=("exclusive", "inclusive"), default="exclusive")
    parser.add_argument("--step-axis", choices=("auto", "full", "response"), default="auto")
    parser.add_argument(
        "--activation-layout", choices=("auto", "node_first", "layer_first"), default="auto"
    )
    parser.add_argument(
        "--activation-layers", default="last", help="last (default), all, or comma-separated layer indices"
    )
    parser.add_argument("--require-causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--causal-tolerance", type=float, default=1e-5)


def _add_cohort_audit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-missing-provenance",
        action="store_true",
        help="unsafe legacy diagnostic: permit traces without model/template fingerprint",
    )
    parser.add_argument(
        "--allow-incomplete-extraction-scope",
        action="store_true",
        help="diagnostic only: permit missing shards/rows or skipped extraction records",
    )
    parser.add_argument(
        "--allow-multiple-input-datasets",
        action="store_true",
        help="explicitly permit more than one source dataset SHA256 in a combined run",
    )
    parser.add_argument(
        "--allow-observer-traces",
        action="store_true",
        help="permit counterfactual observer traces in a separately reported diagnostic run",
    )
    parser.add_argument(
        "--allow-unverified-generator-weights",
        action="store_true",
        help="unsafe diagnostic: permit token-exact traces whose weight revision is unverified",
    )
    parser.add_argument(
        "--allow-unverified-chunk-topology",
        action="store_true",
        help=(
            "diagnostic only: use cached traces with a selector/threshold not covered "
            "by their prefix equivalence gate"
        ),
    )


def _add_graph_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="generic legacy default; the original-aligned response wrapper passes tau=0.05",
    )
    parser.add_argument("--top-k", type=int, default=None, help="innovation; default keeps threshold-only edges")
    parser.add_argument(
        "--source-selection",
        choices=(
            "threshold",
            "threshold_fallback_topk",
            "top_k_only",
            "cumulative_mass",
        ),
        default="threshold",
        help=(
            "threshold_fallback_topk reproduces the local original's fallback when no "
            "source crosses tau; other modes are explicit sparsifier ablations"
        ),
    )
    parser.add_argument(
        "--cumulative-mass",
        type=float,
        default=0.8,
        help="eligible-history attention mass retained by cumulative_mass selection",
    )
    parser.add_argument("--min-sources", type=int, default=1)
    parser.add_argument("--include-center", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--source-scope", choices=("all_past", "prompt_only", "response_only"), default="all_past"
    )
    parser.add_argument(
        "--incidence-weight-mode",
        choices=("uniform", "attention", "normalized_attention"),
        default="uniform",
    )
    parser.add_argument("--propagation-mode", choices=("symmetric", "receiver"), default="symmetric")
    parser.add_argument(
        "--edge-attr-mode",
        choices=("faithful", "extended"),
        default="faithful",
        help="faithful keeps original 3-D attrs; extended is an explicit innovation",
    )
    parser.add_argument("--selected-layers", default="all", help="all or comma-separated attention layers")
    parser.add_argument("--selected-heads", default="all", help="all or comma-separated attention heads")
    parser.add_argument(
        "--node-feature-mode",
        choices=("attention_diagonal", "activation_only", "diagonal_plus_activation"),
        default="attention_diagonal",
        help="separate topology from node content; attention_diagonal is faithful",
    )
    parser.add_argument(
        "--use-activation",
        action="store_true",
        help="legacy alias for --node-feature-mode diagonal_plus_activation",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="validate raw traces and construct graph summaries")
    _add_common_data_arguments(inspect_parser)
    _add_graph_arguments(inspect_parser)
    _add_cohort_audit_arguments(inspect_parser)
    inspect_parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        help="enable the same strict supervised cohort gate used by train",
    )
    inspect_parser.add_argument("--skip-unlabeled", action="store_true")
    inspect_parser.add_argument("--output", help="optional JSON report path")
    inspect_parser.add_argument("--verbose-records", action="store_true")
    inspect_parser.set_defaults(handler=command_inspect)

    build = subparsers.add_parser("build", help="build one framework-neutral graph NPZ per trace")
    _add_common_data_arguments(build)
    _add_graph_arguments(build)
    build.add_argument("--output", required=True, help="graph output directory")
    build.add_argument("--overwrite", action="store_true")
    build.set_defaults(handler=command_build)

    train = subparsers.add_parser("train", help="train/evaluate with explicit or group-aware splits")
    _add_common_data_arguments(train)
    _add_graph_arguments(train)
    _add_cohort_audit_arguments(train)
    train.add_argument("--output", required=True, help="run output directory")
    train.add_argument("--overwrite", action="store_true")
    train.add_argument("--objective", choices=OBJECTIVES, required=True)
    train.add_argument("--skip-unlabeled", action="store_true")
    train.add_argument(
        "--allow-offline-full-context",
        "--allow-offline-symmetric-step",
        dest="allow_offline_full_context",
        action="store_true",
        help=(
            "explicitly permit future-dependent symmetric propagation and/or per-graph "
            "normalization for legacy offline token/step comparisons"
        ),
    )
    train.add_argument(
        "--preprocessing",
        choices=("per_graph_zscore", "none"),
        default="per_graph_zscore",
        help="per_graph_zscore reproduces original per-graph feature standardization",
    )
    train.add_argument(
        "--split-mode",
        choices=("auto", "explicit", "fixed_holdout", "group_cv"),
        default="auto",
    )
    train.add_argument("--train-split", default="train")
    train.add_argument("--val-split", default="validation")
    train.add_argument("--test-split", default="test")
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--fold-index", type=int, default=0)
    train.add_argument(
        "--split-seed",
        type=int,
        default=17,
        help="data-partition seed for fixed_holdout; independent of model initialization",
    )
    train.add_argument("--val-ratio", type=float, default=0.1)
    train.add_argument("--test-ratio", type=float, default=0.2)
    train.add_argument("--allow-trace-as-group", action="store_true")
    train.add_argument(
        "--allow-resplit-official-data",
        action="store_true",
        help="diagnostic only: permit group_cv to ignore existing official split metadata",
    )
    train.add_argument("--allow-single-class-partition", action="store_true")
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--model-layers", type=int, default=2, help="set 0 for the feature-only control")
    train.add_argument("--dropout", type=float, default=0.25)
    train.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--receiver-source-interaction", action="store_true")
    train.add_argument(
        "--message-operator",
        choices=("hypergraph", "pairwise"),
        default="hypergraph",
        help="pairwise is a parameter-matched directed attention-edge control",
    )
    train.add_argument("--mlp-norm", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument(
        "--classifier-norm", action=argparse.BooleanOptionalAction, default=True
    )
    train.add_argument(
        "--init-weights", action=argparse.BooleanOptionalAction, default=True
    )
    train.add_argument("--pooling", choices=("mean", "logmeanexp"), default="mean")
    train.add_argument("--pooling-temperature", type=float, default=1.0)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-3)
    train.add_argument("--scheduler", choices=("cosine", "constant"), default="cosine")
    train.add_argument("--warmup-ratio", type=float, default=0.05)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--min-delta", type=float, default=1e-4)
    train.add_argument("--monitor", choices=("aupr", "auroc"), default="aupr")
    train.add_argument("--pos-weight", default="auto")
    train.add_argument("--max-pos-weight", type=float, default=10.0)
    train.add_argument("--grad-accumulation", type=int, default=1)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=17)
    train.set_defaults(handler=command_train)
    return parser


def _validate_runtime_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.limit is not None and int(args.limit) < 1:
        parser.error("--limit must be positive")
    if args.command != "train":
        return
    if int(args.epochs) < 1 or int(args.patience) < 1:
        parser.error("--epochs and --patience must be positive")
    if int(args.grad_accumulation) < 1:
        parser.error("--grad-accumulation must be positive")
    if not math.isfinite(float(args.learning_rate)) or float(args.learning_rate) <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(float(args.weight_decay)) or float(args.weight_decay) < 0:
        parser.error("--weight-decay must be finite and non-negative")
    if not 0.0 <= float(args.warmup_ratio) < 1.0:
        parser.error("--warmup-ratio must lie in [0,1)")
    if not math.isfinite(float(args.max_pos_weight)):
        parser.error("--max-pos-weight must be finite; use <=0 to disable clipping")
    if not math.isfinite(float(args.pooling_temperature)) or float(args.pooling_temperature) <= 0:
        parser.error("--pooling-temperature must be finite and positive")
    if not math.isfinite(float(args.grad_clip)) or float(args.grad_clip) < 0:
        parser.error("--grad-clip must be finite and non-negative")


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    _load_config(args, argv, parser)
    _validate_runtime_arguments(args, parser)
    try:
        args.handler(args)
    except (TraceFormatError, FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
