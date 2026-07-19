"""Deterministic synthetic transition used by tests and CLI preflight."""

from __future__ import annotations

import numpy as np

from .data import TokenStateTrace
from .metrics import DispersionConfig, block_writes_from_states, compute_dispersion_field


def synthetic_trace(seed: int = 7) -> TokenStateTrace:
    rng = np.random.default_rng(seed)
    token_count, block_count, hidden_size = 96, 4, 32
    transition = token_count // 2
    bases = rng.normal(size=(block_count, hidden_size))
    bases /= np.linalg.norm(bases, axis=-1, keepdims=True)
    drift = rng.normal(size=(block_count, hidden_size))
    drift -= np.sum(drift * bases, axis=-1, keepdims=True) * bases
    drift /= np.linalg.norm(drift, axis=-1, keepdims=True)
    writes = np.empty((token_count, block_count, hidden_size), dtype=np.float64)
    for token in range(token_count):
        if token < transition:
            # Coherent regime: small variation along one tangent direction only.
            writes[token] = bases + 0.04 * rng.normal(size=(block_count, 1)) * drift
        else:
            writes[token] = rng.normal(size=bases.shape)
        writes[token] /= np.linalg.norm(writes[token], axis=-1, keepdims=True)

    states = np.zeros((token_count, block_count + 1, hidden_size), dtype=np.float64)
    states[:, 0, :] = 0.1 * rng.normal(size=(token_count, hidden_size))
    for block in range(block_count):
        states[:, block + 1, :] = states[:, block, :] + writes[:, block, :]
    return TokenStateTrace("synthetic_transition", states, np.arange(block_count + 1), "synthetic")


def run_selftest() -> dict[str, float | bool]:
    trace = synthetic_trace()
    writes, _ = block_writes_from_states(trace.states, trace.layers)
    field = compute_dispersion_field(
        writes,
        DispersionConfig(windows=(8, 16), min_tokens=4),
    )
    dispersion = field["pair_dispersion"]
    rank = field["effective_rank"]
    early_dispersion = float(np.nanmean(dispersion[20:40]))
    late_dispersion = float(np.nanmean(dispersion[72:92]))
    early_rank = float(np.nanmean(rank[20:40]))
    late_rank = float(np.nanmean(rank[72:92]))
    passed = late_dispersion > early_dispersion + 0.35 and late_rank > early_rank + 1.0
    return {
        "passed": passed,
        "early_pair_dispersion": early_dispersion,
        "late_pair_dispersion": late_dispersion,
        "early_effective_rank": early_rank,
        "late_effective_rank": late_rank,
    }
