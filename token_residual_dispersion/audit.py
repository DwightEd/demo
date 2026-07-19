"""End-to-end audit helpers for residual directional dispersion."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data import TokenStateTrace
from .metrics import (
    DispersionConfig,
    compute_dispersion_field,
    depth_deltas_from_states,
    residual_arc_length,
)


def analyze_trace(
    trace: TokenStateTrace,
    config: DispersionConfig,
    *,
    allow_unverified_snapshots: bool = False,
    allow_sparse_depth_deltas: bool = False,
) -> dict[str, Any]:
    if trace.snapshot_kind != "raw_residual_stream" and not allow_unverified_snapshots:
        raise ValueError(
            f"{trace.trace_id}: snapshot provenance is {trace.snapshot_kind!r}; "
            "raw_residual_stream is required for block-write claims"
        )
    deltas, source_depths, target_depths = depth_deltas_from_states(
        trace.states,
        trace.layers,
        allow_sparse=allow_sparse_depth_deltas,
    )
    depth_spans = target_depths - source_depths
    delta_kind = (
        "single_block_residual_write"
        if np.all(depth_spans == 1)
        else "sparse_multi_block_depth_interval_delta_pilot"
    )
    field = compute_dispersion_field(deltas, config)
    arc = residual_arc_length(deltas, eps=config.eps)
    return {
        "trace_id": trace.trace_id,
        "source": trace.source,
        "snapshot_kind": trace.snapshot_kind,
        "delta_kind": delta_kind,
        "source_depths": source_depths,
        "target_depths": target_depths,
        "depth_spans": depth_spans,
        **field,
        **arc,
    }


def summarize_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    dispersion = np.asarray(analysis["pair_dispersion"])
    rank = np.asarray(analysis["effective_rank"])
    token_count = dispersion.shape[0]
    split = max(1, token_count // 2)

    def safe_mean(value: np.ndarray) -> float | None:
        finite = value[np.isfinite(value)]
        return float(finite.mean()) if finite.size else None

    def safe_max(value: np.ndarray) -> float | None:
        finite = value[np.isfinite(value)]
        return float(finite.max()) if finite.size else None

    return {
        "trace_id": analysis["trace_id"],
        "source": analysis["source"],
        "snapshot_kind": analysis["snapshot_kind"],
        "delta_kind": analysis["delta_kind"],
        "depth_intervals": [
            [int(source), int(target)]
            for source, target in zip(
                np.asarray(analysis["source_depths"]),
                np.asarray(analysis["target_depths"]),
            )
        ],
        "token_count": token_count,
        "block_count": dispersion.shape[1],
        "windows": np.asarray(analysis["windows"]).astype(int).tolist(),
        "early_pair_dispersion": safe_mean(dispersion[:split]),
        "late_pair_dispersion": safe_mean(dispersion[split:]),
        "early_effective_rank": safe_mean(rank[:split]),
        "late_effective_rank": safe_mean(rank[split:]),
        "total_arc_length": float(np.asarray(analysis["cumulative_arc_length"])[-1]),
        "max_identity_error": safe_max(np.asarray(analysis["identity_error"])),
    }


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "trace"


def save_audit(
    analyses: Iterable[dict[str, Any]],
    output_dir: str | Path,
    config: DispersionConfig,
) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    trace_dir = output_path / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    delta_kinds: set[str] = set()
    for index, analysis in enumerate(analyses):
        summary = summarize_analysis(analysis)
        summaries.append(summary)
        delta_kinds.add(str(analysis["delta_kind"]))
        filename = f"{index:05d}_{_safe_name(str(analysis['trace_id']))}.npz"
        summary["output_file"] = f"traces/{filename}"
        arrays = {
            key: value
            for key, value in analysis.items()
            if isinstance(value, np.ndarray)
        }
        np.savez_compressed(trace_dir / filename, **arrays)

    if not summaries:
        raise ValueError("input contains no token-state traces")

    payload = {
        "schema_version": "token_residual_dispersion_v1",
        "field_semantics": {
            "dispersion_fields": "causal_post_token_diagnostic_predicts_next_token_when_shifted",
            "cumulative_arc_length": "causal_activity_coordinate",
            "retrospective_arc_phase": "acausal_full_trace_visualization_only",
        },
        "delta_kinds": sorted(delta_kinds),
        "segmentation": "none",
        "config": {
            "windows": list(config.windows),
            "min_tokens": config.min_tokens,
            "decay": config.decay,
            "eps": config.eps,
            "rank_stride": config.rank_stride,
        },
        "traces": summaries,
    }
    summary_path = output_path / "audit_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path
