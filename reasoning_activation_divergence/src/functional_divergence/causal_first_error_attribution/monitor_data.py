from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MonitorBoundaryRow:
    chain_id: str
    problem_hash: str
    sibling_group: str
    domain: str
    candidate_step: int
    first_error_step: int
    decision_position: int
    label: int
    nuisance: np.ndarray
    output_context: np.ndarray
    store_index: int
    state_index: int


@dataclass(frozen=True)
class _BoundaryStateStore:
    path: Path
    values: np.ndarray


@dataclass(frozen=True)
class ProcessBenchMonitorData:
    rows: tuple[MonitorBoundaryRow, ...]
    stores: tuple[_BoundaryStateStore, ...]
    layer_ids: np.ndarray
    hidden_size: int
    output_feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("monitor dataset contains no eligible boundary rows")
        if not self.stores:
            raise ValueError("monitor dataset contains no boundary-state stores")

    def state(self, row_index: int) -> np.ndarray:
        row = self.rows[int(row_index)]
        value = np.asarray(
            self.stores[row.store_index].values[row.state_index], dtype=np.float32
        )
        expected = (len(self.layer_ids), self.hidden_size)
        if value.shape != expected:
            raise ValueError(
                f"{self.stores[row.store_index].path}: expected state {expected}, "
                f"got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(
                f"chain {row.chain_id} step {row.candidate_step}: "
                "pre-step state contains non-finite values"
            )
        return value

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([row.label for row in self.rows], dtype=np.float32)

    @property
    def nuisance(self) -> np.ndarray:
        return np.stack([row.nuisance for row in self.rows]).astype(np.float32)

    @property
    def output_context(self) -> np.ndarray:
        return np.stack([row.output_context for row in self.rows]).astype(np.float32)


def _record_vector(
    archive: np.lib.npyio.NpzFile,
    names: Iterable[str],
    count: int,
    *,
    default: object | None = None,
) -> np.ndarray:
    for name in names:
        if name not in archive.files:
            continue
        values = np.asarray(archive[name], dtype=object)
        if values.ndim == 0:
            return np.full(count, values.item(), dtype=object)
        values = values.reshape(-1)
        if values.shape != (count,):
            raise ValueError(f"{name} is not record-aligned")
        return values
    if default is None:
        raise ValueError(f"trace is missing one of {tuple(names)}")
    return np.full(count, default, dtype=object)


def _risk_set(first_error: int, step_count: int) -> range:
    if step_count < 1:
        raise ValueError("n_steps must be positive")
    if first_error == -1:
        return range(step_count)
    if not 0 <= first_error < step_count:
        raise ValueError(
            f"gold_error_step={first_error} lies outside n_steps={step_count}"
        )
    return range(first_error + 1)


def _past_output_context(step_scores: np.ndarray, step: int) -> np.ndarray:
    feature_count = int(step_scores.shape[1])
    if step == 0:
        return np.zeros(2 * feature_count + 1, dtype=np.float32)
    past = np.asarray(step_scores[:step], dtype=np.float32)
    finite = np.isfinite(past)
    sums = np.where(finite, past, 0.0).sum(axis=0)
    counts = finite.sum(axis=0)
    means = np.divide(
        sums,
        counts,
        out=np.zeros(feature_count, dtype=np.float32),
        where=counts > 0,
    )
    last = np.where(np.isfinite(past[-1]), past[-1], 0.0)
    return np.concatenate(
        [last.astype(np.float32), means.astype(np.float32), np.asarray([1.0])]
    )


def _nuisance_features(ranges: np.ndarray, step: int) -> np.ndarray:
    decision_position = int(ranges[step, 0]) - 1
    if step == 0:
        previous_length = 0.0
        mean_past_length = 0.0
    else:
        completed = ranges[:step, 1] - ranges[:step, 0] + 1
        previous_length = float(completed[-1])
        mean_past_length = float(np.mean(completed))
    return np.asarray(
        [
            np.log1p(float(step)),
            np.log1p(float(decision_position + 1)),
            np.log1p(previous_length),
            np.log1p(mean_past_length),
            float(step == 0),
        ],
        dtype=np.float32,
    )


def _problem_hash(domain: str, value: object) -> str:
    text = str(value)
    return text if text.startswith("problem_sha256:") else f"{domain}::{text}"


def _load_domain(
    data_root: Path,
    domain: str,
    *,
    output_features: tuple[str, ...],
    max_chains: int,
    store_index: int,
) -> tuple[list[MonitorBoundaryRow], _BoundaryStateStore, np.ndarray, int]:
    trace_path = data_root / domain / "geometry" / "trace.npz"
    if not trace_path.is_file():
        raise FileNotFoundError(
            f"missing whole-layer geometry manifest: {trace_path}; "
            "the monitor requires geometry/trace.npz, not post-step component traces"
        )
    with np.load(trace_path, allow_pickle=True) as archive:
        required = {
            "chain_idx",
            "gold_error_step",
            "n_steps",
            "step_token_ranges",
            "step_scores",
            "step_score_names",
            "step_pre_state_memmap_path",
            "step_pre_state_memmap_count",
            "step_pre_state_vector_chain_idx",
            "step_pre_state_vector_step_idx",
            "step_layer_state_vector_layers",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{trace_path}: trace is missing required arrays {missing}")

        chain_ids = np.asarray(archive["chain_idx"], dtype=np.int64).reshape(-1)
        count = len(chain_ids)
        if len(np.unique(chain_ids)) != count:
            raise ValueError(f"{trace_path}: chain_idx must be unique")
        selected_count = count if max_chains == 0 else min(count, int(max_chains))
        selected_chain_ids = set(int(value) for value in chain_ids[:selected_count])
        first_errors = np.asarray(archive["gold_error_step"], dtype=np.int64).reshape(-1)
        n_steps = np.asarray(archive["n_steps"], dtype=np.int64).reshape(-1)
        if first_errors.shape != (count,) or n_steps.shape != (count,):
            raise ValueError(f"{trace_path}: labels and n_steps must be record-aligned")
        ranges_all = np.asarray(archive["step_token_ranges"])
        if ranges_all.shape[0] != count:
            raise ValueError(f"{trace_path}: step_token_ranges is not record-aligned")

        score_names = tuple(str(value) for value in archive["step_score_names"])
        unknown = sorted(set(output_features).difference(score_names))
        if unknown:
            raise ValueError(
                f"{trace_path}: requested output features are absent: {unknown}; "
                f"available={list(score_names)}"
            )
        score_indices = np.asarray(
            [score_names.index(name) for name in output_features], dtype=np.int64
        )
        all_scores = np.asarray(archive["step_scores"], dtype=np.float32)
        if all_scores.ndim != 3 or all_scores.shape[0] != count:
            raise ValueError(f"{trace_path}: step_scores must have shape [chain,step,feature]")

        groups = _record_vector(
            archive, ("problem_group_id", "problem_group_ids", "problem_ids"), count
        )
        hashes = _record_vector(
            archive, ("problem_ids", "problem_group_id", "problem_group_ids"), count
        )
        datasets = _record_vector(archive, ("dataset",), count, default=domain)
        if any(str(value) and str(value) != domain for value in datasets[:selected_count]):
            raise ValueError(f"{trace_path}: dataset field disagrees with directory {domain}")

        point_chain_ids = np.asarray(
            archive["step_pre_state_vector_chain_idx"], dtype=np.int64
        ).reshape(-1)
        point_steps = np.asarray(
            archive["step_pre_state_vector_step_idx"], dtype=np.int64
        ).reshape(-1)
        state_count = int(np.asarray(archive["step_pre_state_memmap_count"]).item())
        if point_chain_ids.shape != (state_count,) or point_steps.shape != (state_count,):
            raise ValueError(f"{trace_path}: pre-step state indices disagree with memmap count")
        state_lookup: dict[tuple[int, int], int] = {}
        for state_row, (chain_id, step) in enumerate(zip(point_chain_ids, point_steps)):
            key = (int(chain_id), int(step))
            if key in state_lookup:
                raise ValueError(f"{trace_path}: duplicate pre-step state for {key}")
            state_lookup[key] = state_row

        state_path = Path(str(np.asarray(archive["step_pre_state_memmap_path"]).item()))
        if not state_path.is_absolute():
            state_path = trace_path.parent / state_path
        layers = np.asarray(
            archive["step_layer_state_vector_layers"], dtype=np.int64
        ).reshape(-1)

        rows: list[MonitorBoundaryRow] = []
        for chain_row in range(selected_count):
            chain_id = int(chain_ids[chain_row])
            step_count = int(n_steps[chain_row])
            first_error = int(first_errors[chain_row])
            ranges = np.asarray(ranges_all[chain_row], dtype=np.int64)[:step_count]
            if ranges.shape != (step_count, 2) or np.any(ranges < 0):
                raise ValueError(f"chain {chain_id}: invalid unpadded step_token_ranges")
            if np.any(ranges[:, 1] < ranges[:, 0]) or np.any(np.diff(ranges[:, 0]) <= 0):
                raise ValueError(f"chain {chain_id}: invalid or unordered step_token_ranges")
            scores = all_scores[chain_row, :step_count][:, score_indices]
            group = f"{domain}::{groups[chain_row]}"
            problem_hash = _problem_hash(domain, hashes[chain_row])
            for step in _risk_set(first_error, step_count):
                key = (chain_id, int(step))
                if key not in state_lookup:
                    raise ValueError(f"{trace_path}: missing pre-step state for {key}")
                rows.append(
                    MonitorBoundaryRow(
                        chain_id=f"{domain}::{chain_id}",
                        problem_hash=problem_hash,
                        sibling_group=group,
                        domain=domain,
                        candidate_step=int(step),
                        first_error_step=first_error,
                        decision_position=int(ranges[step, 0]) - 1,
                        label=int(first_error == step),
                        nuisance=_nuisance_features(ranges, int(step)),
                        output_context=_past_output_context(scores, int(step)),
                        store_index=store_index,
                        state_index=state_lookup[key],
                    )
                )

    if not state_path.is_file():
        raise FileNotFoundError(f"pre-step state memmap does not exist: {state_path}")
    values = np.load(state_path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 3 or not 0 < state_count <= values.shape[0]:
        raise ValueError(
            f"{state_path}: expected [point,layer,hidden] with count={state_count}, "
            f"got {values.shape}"
        )
    if values.shape[1] != len(layers):
        raise ValueError(f"{state_path}: layer axis disagrees with declared layer IDs")
    if selected_chain_ids.difference(point_chain_ids.tolist()):
        raise ValueError(f"{trace_path}: selected chains are absent from the state index")
    return (
        rows,
        _BoundaryStateStore(path=state_path, values=values[:state_count]),
        layers,
        int(values.shape[2]),
    )


def load_processbench_monitor_data(
    data_root: str | Path,
    domains: Iterable[str],
    *,
    output_features: Iterable[str] = ("token_entropy", "token_nll"),
    max_chains_per_domain: int = 0,
) -> ProcessBenchMonitorData:
    """Load exact pre-step states and construct the first-error risk set.

    Error chains contribute steps ``0..t*`` and correct chains contribute all
    steps.  No row after the first error is exposed to the monitor.
    """

    root = Path(data_root).expanduser()
    domain_names = tuple(str(value).strip() for value in domains if str(value).strip())
    features = tuple(str(value).strip() for value in output_features if str(value).strip())
    if not domain_names:
        raise ValueError("at least one domain is required")
    if not features:
        raise ValueError("at least one output feature is required")
    if int(max_chains_per_domain) < 0:
        raise ValueError("max_chains_per_domain must be nonnegative")

    rows: list[MonitorBoundaryRow] = []
    stores: list[_BoundaryStateStore] = []
    common_layers: np.ndarray | None = None
    common_hidden_size: int | None = None
    for domain in domain_names:
        domain_rows, store, layers, hidden_size = _load_domain(
            root,
            domain,
            output_features=features,
            max_chains=int(max_chains_per_domain),
            store_index=len(stores),
        )
        if common_layers is None:
            common_layers = layers
            common_hidden_size = hidden_size
        elif not np.array_equal(common_layers, layers) or common_hidden_size != hidden_size:
            raise ValueError("all domains must share the same layer and hidden axes")
        rows.extend(domain_rows)
        stores.append(store)
    assert common_layers is not None and common_hidden_size is not None
    hash_domains: dict[str, set[str]] = {}
    for row in rows:
        hash_domains.setdefault(row.problem_hash, set()).add(row.domain)
    cross_domain = sorted(
        problem_hash
        for problem_hash, owners in hash_domains.items()
        if len(owners) > 1
    )
    if cross_domain:
        raise ValueError(
            "problem hash spans LODO domains and would leak across train/test: "
            f"{cross_domain[:3]}"
        )
    return ProcessBenchMonitorData(
        rows=tuple(rows),
        stores=tuple(stores),
        layer_ids=common_layers,
        hidden_size=common_hidden_size,
        output_feature_names=features,
    )


__all__ = [
    "MonitorBoundaryRow",
    "ProcessBenchMonitorData",
    "load_processbench_monitor_data",
]
