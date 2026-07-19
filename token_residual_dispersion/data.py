"""Load token-state traces from direct arrays or extraction manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class TokenStateTrace:
    trace_id: str
    states: np.ndarray
    layers: np.ndarray
    source: str
    snapshot_kind: str = "unverified"


def _scalar_or_default(archive: np.lib.npyio.NpzFile, key: str, default: object) -> object:
    if key not in archive.files:
        return default
    value = np.asarray(archive[key])
    return value.item() if value.ndim == 0 else value


def _validate_trace(trace: TokenStateTrace) -> TokenStateTrace:
    states = np.asarray(trace.states)
    if states.dtype != np.dtype(np.float32) and states.dtype != np.dtype(np.float64):
        states = states.astype(np.float32)
    layers = np.asarray(trace.layers, dtype=np.int64)
    if states.ndim != 3:
        raise ValueError(f"{trace.trace_id}: states must be [token, depth, hidden]")
    if states.shape[0] == 0 or states.shape[2] == 0:
        raise ValueError(f"{trace.trace_id}: token and hidden axes must be non-empty")
    if layers.shape != (states.shape[1],):
        raise ValueError(f"{trace.trace_id}: layer count does not match state depth")
    if not np.all(np.isfinite(states)):
        raise ValueError(f"{trace.trace_id}: states contain non-finite values")
    return TokenStateTrace(trace.trace_id, states, layers, trace.source, trace.snapshot_kind)


def iter_token_state_traces(
    path: str | Path,
    *,
    layers: np.ndarray | None = None,
    snapshot_kind: str = "unverified",
    max_traces: int | None = None,
) -> Iterator[TokenStateTrace]:
    """Yield traces without retaining every response-state shard in RAM."""

    source_path = Path(path).expanduser().resolve()
    if max_traces is not None and max_traces < 0:
        raise ValueError("max_traces must be non-negative")
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.suffix == ".npy":
        states = np.load(source_path, allow_pickle=False)
        if layers is None:
            raise ValueError("direct .npy input requires explicit layer ids")
        layer_ids = np.asarray(layers)
        if max_traces == 0:
            return
        yield _validate_trace(TokenStateTrace(
            source_path.stem, states, layer_ids, str(source_path), snapshot_kind
        ))
        return
    if source_path.suffix != ".npz":
        raise ValueError("input must be a .npy state tensor or .npz manifest")

    with np.load(source_path, allow_pickle=True) as archive:
        if "states" in archive.files or "response_token_layer_states" in archive.files:
            state_key = "states" if "states" in archive.files else "response_token_layer_states"
            states = np.asarray(archive[state_key])
            if "layers" in archive.files:
                layer_ids = np.asarray(archive["layers"])
            elif layers is not None:
                layer_ids = np.asarray(layers)
            else:
                raise ValueError("direct npz state input requires explicit layer ids")
            stored_kind = str(_scalar_or_default(archive, "snapshot_kind", snapshot_kind))
            if max_traces == 0:
                return
            yield _validate_trace(TokenStateTrace(
                source_path.stem, states, layer_ids, str(source_path), stored_kind
            ))
        elif "response_token_state_files" in archive.files:
            files = np.asarray(archive["response_token_state_files"], dtype=object).reshape(-1)
            if "response_token_state_storage_kind" in archive.files:
                storage_kind = str(np.asarray(archive["response_token_state_storage_kind"]).item())
                if storage_kind != "per_chain_npy_shards_v1":
                    raise ValueError(f"unsupported response token state storage: {storage_kind}")
            if "response_token_state_layers" not in archive.files:
                raise ValueError("extraction manifest is missing response_token_state_layers")
            stored_layers = np.asarray(
                archive["response_token_state_layers"], dtype=np.int64
            ).reshape(-1)
            trace_ids = np.asarray(
                _scalar_or_default(archive, "chain_idx", np.arange(len(files))),
                dtype=object,
            ).reshape(-1)
            counts = np.asarray(
                _scalar_or_default(archive, "response_token_state_counts", []),
                dtype=np.int64,
            ).reshape(-1)
            if counts.size and counts.size != len(files):
                raise ValueError("response_token_state_counts/files length mismatch")
            stored_kind = str(_scalar_or_default(
                archive, "response_token_state_snapshot_kind", "unverified"
            ))
            limit = len(files) if max_traces is None else min(len(files), max_traces)
            for index in range(limit):
                shard_path = Path(str(files[index]))
                if not shard_path.is_absolute():
                    shard_path = source_path.parent / shard_path
                states = np.load(shard_path.resolve(), allow_pickle=False)
                if counts.size and int(counts[index]) != states.shape[0]:
                    raise ValueError(f"{shard_path}: token count disagrees with manifest")
                layer_ids = stored_layers
                trace_id = str(trace_ids[index]) if index < trace_ids.size else f"trace_{index:05d}"
                yield _validate_trace(TokenStateTrace(
                    trace_id, states, layer_ids, str(shard_path), stored_kind
                ))
        else:
            raise ValueError(
                "npz needs states/response_token_layer_states or response_token_state_files"
            )



def load_token_state_traces(
    path: str | Path,
    *,
    layers: np.ndarray | None = None,
    snapshot_kind: str = "unverified",
    max_traces: int | None = None,
) -> list[TokenStateTrace]:
    """Materialize traces for small callers; CLI audits use the streaming iterator."""

    traces = list(iter_token_state_traces(
        path,
        layers=layers,
        snapshot_kind=snapshot_kind,
        max_traces=max_traces,
    ))
    if not traces:
        raise ValueError("input contains no token-state traces")
    return traces
