from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class MechanismProfiler:
    """Small phase profiler for extraction runs.

    It records wall time, coarse CUDA peak memory, sequence lengths, and skip
    reasons.  The profiler is intentionally dependency-light so it can run in
    local CPU smoke tests and remote GPU jobs.
    """

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.phase_time: dict[str, float] = defaultdict(float)
        self.num_chains = 0
        self.num_extracted = 0
        self.skip_reasons: Counter[str] = Counter()
        self.seq_lens: list[int] = []
        self.gpu_peak_mb = 0.0

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self._reset_cuda_peak()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.phase_time[name] += time.perf_counter() - t0
            self.gpu_peak_mb = max(self.gpu_peak_mb, self._cuda_peak_mb())

    def record_chain(self, seq_len: int | None = None) -> None:
        self.num_chains += 1
        if seq_len is not None:
            self.seq_lens.append(int(seq_len))

    def record_seq_len(self, seq_len: int | None) -> None:
        if seq_len is not None:
            self.seq_lens.append(int(seq_len))

    def record_success(self) -> None:
        self.num_extracted += 1

    def record_skip(self, reason: str) -> None:
        self.skip_reasons[str(reason)] += 1

    def summary(self) -> dict[str, Any]:
        total = time.perf_counter() - self.started
        return {
            "num_chains": int(self.num_chains),
            "num_extracted": int(self.num_extracted),
            "num_skipped": int(sum(self.skip_reasons.values())),
            "skip_reasons": dict(self.skip_reasons),
            "total_time": float(total),
            "phase_time": {k: float(v) for k, v in self.phase_time.items()},
            "gpu_peak_mb": float(self.gpu_peak_mb),
            "avg_seq_len": float(np.mean(self.seq_lens)) if self.seq_lens else None,
            "max_seq_len": int(max(self.seq_lens)) if self.seq_lens else None,
            "samples_per_second": float(self.num_extracted / total) if total > 0 else None,
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def _reset_cuda_peak() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            return

    @staticmethod
    def _cuda_peak_mb() -> float:
        try:
            import torch

            if torch.cuda.is_available():
                return float(torch.cuda.max_memory_allocated() / (1024.0**2))
        except Exception:
            return 0.0
        return 0.0
