from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


class StateStorageError(RuntimeError):
    """Raised when a state artifact cannot be stored without corruption."""


def storage_dtype(name: str) -> np.dtype:
    aliases = {
        "float16": np.dtype(np.float16),
        "fp16": np.dtype(np.float16),
        "float32": np.dtype(np.float32),
        "fp32": np.dtype(np.float32),
    }
    try:
        return aliases[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported state storage dtype {name!r}") from exc


def cast_state_array(array: np.ndarray, dtype: str | np.dtype) -> np.ndarray:
    """Validate and cast hidden states without silent overflow."""

    source = np.asarray(array, dtype=np.float32)
    if not np.isfinite(source).all():
        bad = int(source.size - np.isfinite(source).sum())
        raise StateStorageError(f"state tensor contains {bad} non-finite value(s)")
    target = storage_dtype(str(dtype)) if isinstance(dtype, str) else np.dtype(dtype)
    if target == np.dtype(np.float16):
        limit = float(np.finfo(np.float16).max)
        peak = float(np.max(np.abs(source))) if source.size else 0.0
        if peak > limit:
            raise StateStorageError(
                f"state magnitude {peak:.6g} exceeds float16 limit {limit:.0f}; "
                "re-run with --state_storage_dtype float32"
            )
    return source.astype(target, copy=False)


@dataclass
class FixedStateMemmap:
    """Append fixed-tail tensors to an atomically finalized NPY memmap."""

    partial_path: Path
    final_path: Path
    capacity: int
    tail_shape: tuple[int, ...]
    dtype: str = "float16"
    _array: np.memmap = field(init=False, repr=False)
    count: int = field(default=0, init=False)
    chain_idx: list[int] = field(default_factory=list, init=False)
    item_idx: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.partial_path = Path(self.partial_path)
        self.final_path = Path(self.final_path)
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path.unlink(missing_ok=True)
        if int(self.capacity) <= 0:
            raise ValueError("state memmap capacity must be positive")
        self._array = np.lib.format.open_memmap(
            self.partial_path,
            mode="w+",
            dtype=storage_dtype(self.dtype),
            shape=(int(self.capacity), *tuple(int(x) for x in self.tail_shape)),
        )

    def append(self, array: np.ndarray, *, chain_idx: int) -> tuple[int, int]:
        values = cast_state_array(array, self.dtype)
        if values.ndim != len(self.tail_shape) + 1:
            raise StateStorageError(
                f"expected state rank {len(self.tail_shape) + 1}, got {values.shape}"
            )
        if tuple(values.shape[1:]) != tuple(self.tail_shape):
            raise StateStorageError(
                f"state tail {values.shape[1:]} does not match {self.tail_shape}"
            )
        start = int(self.count)
        stop = start + int(values.shape[0])
        if stop > int(self.capacity):
            raise StateStorageError(
                f"state memmap capacity {self.capacity} is smaller than required {stop}"
            )
        self._array[start:stop] = values
        self.chain_idx.extend([int(chain_idx)] * int(values.shape[0]))
        self.item_idx.extend(range(int(values.shape[0])))
        self.count = stop
        return start, stop

    def finalize(self) -> Path:
        self._close()
        self.final_path.unlink(missing_ok=True)
        self.partial_path.replace(self.final_path)
        return self.final_path

    def abort(self) -> None:
        self._close()
        self.partial_path.unlink(missing_ok=True)

    def rollback(self) -> None:
        """Remove both partial and finalized files owned by this run."""

        self.abort()
        self.final_path.unlink(missing_ok=True)

    def _close(self) -> None:
        array = getattr(self, "_array", None)
        if array is None:
            return
        array.flush()
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
        self._array = None  # type: ignore[assignment]


@dataclass
class ResponseStateShardWriter:
    """Write per-response token states without accumulating them in RAM."""

    partial_dir: Path
    final_dir: Path
    dtype: str = "float16"
    files: list[str] = field(default_factory=list, init=False)
    counts: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.partial_dir = Path(self.partial_dir)
        self.final_dir = Path(self.final_dir)
        if self.partial_dir.exists():
            shutil.rmtree(self.partial_dir)
        self.partial_dir.mkdir(parents=True, exist_ok=False)

    def write(self, array: np.ndarray, *, chain_idx: int) -> str:
        values = cast_state_array(array, self.dtype)
        if values.ndim != 3:
            raise StateStorageError(
                f"response token states must be [token, layer, hidden], got {values.shape}"
            )
        # chain_idx is metadata, not a guaranteed primary key. Same-problem
        # artifacts may reuse it, so include the writer row to avoid overwrite.
        name = f"row_{len(self.files):08d}_chain_{int(chain_idx):08d}.npy"
        target = self.partial_dir / name
        temp = self.partial_dir / f".{name}.tmp"
        with temp.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        temp.replace(target)
        self.files.append(f"{self.final_dir.name}/{name}")
        self.counts.append(int(values.shape[0]))
        return self.files[-1]

    def finalize(self) -> Path:
        if self.final_dir.exists():
            shutil.rmtree(self.final_dir)
        self.partial_dir.replace(self.final_dir)
        return self.final_dir

    def abort(self) -> None:
        if self.partial_dir.exists():
            shutil.rmtree(self.partial_dir)

    def rollback(self) -> None:
        """Remove both partial and finalized directories owned by this run."""

        self.abort()
        if self.final_dir.exists():
            shutil.rmtree(self.final_dir)
