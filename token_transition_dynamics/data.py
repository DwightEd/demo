from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChainRecord:
    domain: str
    row: int
    chain_id: str
    problem_group: str
    error_step: int
    step_ranges: np.ndarray
    response_start: int
    state_path: Path
    expected_tokens: int | None


@dataclass(frozen=True)
class RawDomain:
    name: str
    manifest_path: Path
    layers: np.ndarray
    records: tuple[ChainRecord, ...]
    source_format: str

    def load_states(self, record: ChainRecord) -> np.ndarray:
        values = np.load(record.state_path, mmap_mode="r", allow_pickle=False)
        expected_shape = f"[token,{self.layers.size},hidden]"
        if values.ndim != 3 or values.shape[1] != self.layers.size:
            raise ValueError(
                f"{record.state_path}: expected {expected_shape}, got {values.shape}"
            )
        if record.expected_tokens is not None and values.shape[0] != record.expected_tokens:
            raise ValueError(
                f"{record.state_path}: shard tokens={values.shape[0]} disagree with "
                f"manifest count={record.expected_tokens}"
            )
        return values


def _scalar(archive: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in archive.files:
        return default
    value = np.asarray(archive[key])
    return value.item() if value.ndim == 0 else value


def _step_ranges(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    array = np.asarray(raw.tolist() if raw.dtype == object else value, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"step_token_ranges must have shape [step,2], got {array.shape}")
    array = array[array[:, 1] >= array[:, 0]]
    if array.shape[0] == 0:
        raise ValueError("step_token_ranges contains no valid intervals")
    return array


class RawHiddenRepository:
    """Reads only token-level raw residual manifests and their mmap shards."""

    def __init__(
        self,
        data_root: Path,
        manifest_name: str = "trace.raw_residual_stream.npz",
        max_records_per_domain: int | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.manifest_name = manifest_name
        self.max_records_per_domain = max_records_per_domain

    def read_domain(self, domain: str) -> RawDomain:
        manifest = self.data_root / domain / "selected" / self.manifest_name
        if not manifest.is_file():
            raise FileNotFoundError(f"missing raw hidden-state manifest: {manifest}")

        with np.load(manifest, allow_pickle=True) as archive:
            if "gold_error_step_kept" in archive.files:
                gold = np.asarray(archive["gold_error_step_kept"], dtype=np.int64).reshape(-1)
            elif "gold_error_step" in archive.files:
                gold = np.asarray(archive["gold_error_step"], dtype=np.int64).reshape(-1)
            else:
                raise ValueError(f"{manifest}: missing gold_error_step")
            if "step_token_ranges" not in archive.files:
                raise ValueError(f"{manifest}: missing step_token_ranges")
            ranges = tuple(_step_ranges(value) for value in archive["step_token_ranges"])
            n_records = len(ranges)

            if "response_token_state_files" in archive.files:
                files = np.asarray(archive["response_token_state_files"], dtype=object).reshape(-1)
                if "response_token_state_layers" not in archive.files:
                    raise ValueError(f"{manifest}: missing response_token_state_layers")
                layers = np.asarray(
                    archive["response_token_state_layers"], dtype=np.int64
                ).reshape(-1)
                snapshot = str(
                    _scalar(archive, "response_token_state_snapshot_kind", "unverified")
                )
                if snapshot != "raw_residual_stream":
                    raise ValueError(
                        f"{manifest}: expected raw_residual_stream snapshots, got {snapshot!r}"
                    )
                counts_value = _scalar(archive, "response_token_state_counts", None)
                counts = (
                    None
                    if counts_value is None
                    else np.asarray(counts_value, dtype=np.int64).reshape(-1)
                )
                base = manifest.parent
                source_format = "exact_response_state_manifest_v1"
            elif "hidden_files" in archive.files:
                files = np.asarray(archive["hidden_files"], dtype=object).reshape(-1)
                if "hidden_layers" not in archive.files:
                    raise ValueError(f"{manifest}: missing hidden_layers")
                layers = np.asarray(archive["hidden_layers"], dtype=np.int64).reshape(-1)
                stored_dir = _scalar(archive, "hidden_dir", None)
                if stored_dir is None:
                    raise ValueError(f"{manifest}: canonical manifest is missing hidden_dir")
                base = Path(str(np.asarray(stored_dir).reshape(-1)[0])).expanduser()
                counts = None
                source_format = "canonical_full_hidden_shards_v1"
            else:
                raise ValueError(f"{manifest}: missing raw hidden-state file list")

            if "problem_group_id" in archive.files:
                groups = np.asarray(archive["problem_group_id"], dtype=object).reshape(-1)
            elif "problem_ids" in archive.files:
                groups = np.asarray(archive["problem_ids"], dtype=object).reshape(-1)
            else:
                groups = np.arange(n_records, dtype=object)
            if "chain_idx" in archive.files:
                chain_ids = np.asarray(archive["chain_idx"], dtype=object).reshape(-1)
            else:
                chain_ids = np.arange(n_records, dtype=object)
            if "response_token_ranges" in archive.files:
                response_starts = np.asarray(
                    [int(np.asarray(value).reshape(-1)[0]) for value in archive["response_token_ranges"]],
                    dtype=np.int64,
                )
            elif "prompt_token_counts" in archive.files:
                response_starts = np.asarray(archive["prompt_token_counts"], dtype=np.int64).reshape(-1)
            else:
                response_starts = np.asarray([value[0, 0] for value in ranges], dtype=np.int64)

            aligned = {
                "gold_error_step": gold.shape[0],
                "files": files.shape[0],
                "problem groups": groups.shape[0],
                "chain ids": chain_ids.shape[0],
                "response starts": response_starts.shape[0],
            }
            if counts is not None:
                aligned["state counts"] = counts.shape[0]
            bad = {key: value for key, value in aligned.items() if value != n_records}
            if bad:
                raise ValueError(f"{manifest}: arrays are not record aligned: {bad}, expected={n_records}")
            if layers.size < 2 or np.any(np.diff(layers) != 1):
                raise ValueError(
                    f"{manifest}: token transition dynamics requires consecutive hidden layers; "
                    f"found {layers.tolist()}"
                )

            if self.max_records_per_domain is None or self.max_records_per_domain >= n_records:
                selected_rows = np.arange(n_records, dtype=np.int64)
            else:
                selected_rows = np.unique(
                    np.linspace(
                        0,
                        n_records - 1,
                        self.max_records_per_domain,
                        dtype=np.int64,
                    )
                )
            records = []
            for row_value in selected_rows:
                row = int(row_value)
                file_value = Path(str(files[row])).expanduser()
                state_path = file_value if file_value.is_absolute() else base / file_value
                error_step = int(gold[row])
                if error_step >= len(ranges[row]):
                    raise ValueError(
                        f"{manifest}: error step {error_step} is outside chain {row} ranges"
                    )
                records.append(
                    ChainRecord(
                        domain=domain,
                        row=row,
                        chain_id=str(chain_ids[row]),
                        problem_group=str(groups[row]),
                        error_step=error_step,
                        step_ranges=ranges[row],
                        response_start=int(response_starts[row]),
                        state_path=state_path.resolve(),
                        expected_tokens=None if counts is None else int(counts[row]),
                    )
                )

        raw_domain = RawDomain(
            name=domain,
            manifest_path=manifest.resolve(),
            layers=layers,
            records=tuple(records),
            source_format=source_format,
        )
        if not raw_domain.records:
            raise ValueError(f"{manifest}: no records selected")
        raw_domain.load_states(raw_domain.records[0])
        return raw_domain
