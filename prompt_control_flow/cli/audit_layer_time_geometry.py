from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np

# Support both
#   python prompt_control_flow/cli/audit_layer_time_geometry.py
# and
#   python -m prompt_control_flow.cli.audit_layer_time_geometry
# from the repository root.  Direct script execution otherwise places only
# ``prompt_control_flow/cli`` on sys.path and cannot resolve the package.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prompt_control_flow.evaluate import evaluate_all, load_metric_npz, save_json
from prompt_control_flow.layer_time_geometry import (
    LAYER_TIME_FIELD_NAMES,
    LayerTimeGeometryConfig,
    append_layer_time_geometry,
)
from prompt_control_flow.layer_time_evaluate import (
    LayerTimeValidationConfig,
    evaluate_layer_time_geometry,
    render_layer_time_validation,
)
from prompt_control_flow.reports import render_markdown, write_step_csv
from prompt_control_flow.schema import inspect_npz_schema


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Measure cross-fitted whole-layer representation geometry over the "
            "network-depth x reasoning-time grid."
        )
    )
    ap.add_argument(
        "--input",
        required=True,
        help=(
            "Mainline: --geometry_only extraction or exact sv_vec_mean with all layers. "
            "Legacy full_*.npz is sparse/step_exp and requires both pilot opt-ins."
        ),
    )
    ap.add_argument(
        "--output",
        default="",
        help="Output NPZ with the layer-time field and reductions; required unless --inspect_only.",
    )
    ap.add_argument("--output_dir", default="", help="Directory for evaluation and field summaries.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--knn_k", type=int, default=20)
    ap.add_argument("--tangent_k", type=int, default=24)
    ap.add_argument("--tangent_rank", type=int, default=6)
    ap.add_argument(
        "--fiber_rank_mode",
        choices=("fixed", "lid"),
        default="fixed",
        help="Primary bundle rank. Fixed separates curvature from LID/rank fronts; lid is a legacy ablation.",
    )
    ap.add_argument(
        "--max_transport_residual",
        type=float,
        default=0.35,
        help="Maximum worst-edge Procrustes residual for reliability-gated Wilson curvature.",
    )
    ap.add_argument(
        "--null_mode",
        choices=("none", "phase_shuffle", "reference_id_shuffle"),
        default="none",
        help="Structural null. Run into a separate output artifact; never mix with the primary field.",
    )
    ap.add_argument(
        "--phase_mode",
        choices=("linear", "nearest"),
        default="linear",
        help="Reference-chain phase alignment; compare both before accepting a curvature claim.",
    )
    ap.add_argument(
        "--compute_backend",
        choices=("auto", "numpy", "torch"),
        default="auto",
        help="Bulk projection/kNN backend. Auto uses Torch on an available CUDA device.",
    )
    ap.add_argument("--compute_device", default="cuda")
    ap.add_argument("--projection_dim", type=int, default=64, help="Shared JL dimension; <=0 uses exact ambient vectors.")
    ap.add_argument("--max_reference", type=int, default=256, help="Problem-balanced train-chain cap; <=0 equalizes every problem to the minimum sample count.")
    ap.add_argument("--chunk_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--phase_grid_size", type=int, default=11, help="Train-only phase grid used to fit normalization without length weighting.")
    ap.add_argument("--validation_bootstrap", type=int, default=1000)
    ap.add_argument("--event_offsets", default="-2,-1,0,1,2")
    ap.add_argument("--allow_sparse_layers", action="store_true", help="Pilot only: permit non-contiguous layer depths.")
    ap.add_argument("--allow_legacy_pooling", action="store_true", help="Ablation only: permit non-mean legacy stepvec pooling.")
    ap.add_argument("--keep_state_vectors", action="store_true", help="Also duplicate the large raw state tensor into the field output (off by default).")
    ap.add_argument(
        "--inspect_only",
        action="store_true",
        help="Inspect layer coverage, pooling, and storage without computing the field.",
    )
    return ap


def render_input_report(status: dict[str, object]) -> str:
    layers = status.get("layer_time_layers") or []
    lines = [
        "===== Layer-time input preflight =====",
        f"path: {status['path']}",
        f"input kind: {status.get('layer_time_input_kind', 'unsupported')}",
        f"layers: {layers}",
        f"contiguous layers: {status.get('layer_time_contiguous_layers', False)}",
        f"pooling: {status.get('layer_time_pooling_kind', '') or 'unknown'}",
        f"representation: {status.get('layer_time_representation_kind', '') or 'unknown'}",
        f"mainline ready: {status.get('layer_time_mainline_ready', False)}",
        f"recommendation: {status.get('layer_time_recommendation', '')}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    status = inspect_npz_schema(args.input)
    print(render_input_report(status))
    if args.inspect_only:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return
    if not args.output:
        parser.error("--output is required unless --inspect_only is used")
    if not bool(status.get("layer_time_mainline_ready", False)) and not (
        args.allow_sparse_layers and args.allow_legacy_pooling
    ):
        parser.error(
            "input is not mainline-ready whole-layer mean hidden state data; "
            "use a --geometry_only/exact sv_vec_mean artifact, or explicitly add both "
            "--allow_sparse_layers and --allow_legacy_pooling for a legacy ablation"
        )
    cfg = LayerTimeGeometryConfig(
        n_folds=int(args.folds),
        knn_k=int(args.knn_k),
        tangent_k=int(args.tangent_k),
        tangent_rank=int(args.tangent_rank),
        fiber_rank_mode=str(args.fiber_rank_mode),
        max_transport_residual=float(args.max_transport_residual),
        null_mode=str(args.null_mode),
        phase_mode=str(args.phase_mode),
        compute_backend=str(args.compute_backend),
        compute_device=str(args.compute_device),
        projection_dim=int(args.projection_dim),
        max_reference=int(args.max_reference),
        random_seed=int(args.seed),
        chunk_size=int(args.chunk_size),
        phase_grid_size=int(args.phase_grid_size),
        require_contiguous_layers=not bool(args.allow_sparse_layers),
        require_mean_pooling=not bool(args.allow_legacy_pooling),
    )
    metrics = load_metric_npz(args.input)
    input_path = Path(args.input).resolve()
    output_parent = Path(args.output).resolve().parent
    cache_id = uuid.uuid4().hex[:12]
    metrics["layer_time_input_path"] = np.asarray(str(input_path), dtype=object)
    metrics["layer_time_cache_dir"] = np.asarray(str(output_parent), dtype=object)
    metrics["layer_time_cache_id"] = np.asarray(cache_id, dtype=object)
    conversion_cache = output_parent / f".{input_path.stem}.ltg-mean-states.{cache_id}.npy"
    conversion_partial = output_parent / f".{input_path.stem}.ltg-mean-states.{cache_id}.partial.npy"
    try:
        enriched = append_layer_time_geometry(metrics, cfg)
    except Exception:
        conversion_cache.unlink(missing_ok=True)
        conversion_partial.unlink(missing_ok=True)
        raise

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.input).resolve()
    enriched["layer_time_geometry_source_path"] = np.asarray(str(source_path), dtype=object)
    enriched["layer_time_geometry_source_size_bytes"] = np.asarray(source_path.stat().st_size, dtype=np.int64)
    state_memmap_key = "step_layer_state_memmap_path" if "step_layer_state_memmap_path" in enriched else None
    if state_memmap_key is not None:
        state_path = Path(str(np.asarray(enriched[state_memmap_key]).item()))
        if not state_path.is_absolute():
            state_path = source_path.parent / state_path
        state_store = np.load(state_path, mmap_mode="r")
        enriched["layer_time_geometry_state_source_path"] = np.asarray(str(state_path.resolve()), dtype=object)
        enriched["layer_time_geometry_state_source_size_bytes"] = np.asarray(state_path.stat().st_size, dtype=np.int64)
        enriched["layer_time_geometry_state_source_shape"] = np.asarray(state_store.shape, dtype=np.int64)
        enriched["layer_time_geometry_state_source_dtype"] = np.asarray(str(state_store.dtype), dtype=object)
    elif "step_layer_state_vectors" in enriched:
        state_store = np.asarray(enriched["step_layer_state_vectors"])
        enriched["layer_time_geometry_state_source_path"] = np.asarray(str(source_path), dtype=object)
        enriched["layer_time_geometry_state_source_size_bytes"] = np.asarray(source_path.stat().st_size, dtype=np.int64)
        enriched["layer_time_geometry_state_source_shape"] = np.asarray(state_store.shape, dtype=np.int64)
        enriched["layer_time_geometry_state_source_dtype"] = np.asarray(str(state_store.dtype), dtype=object)
    saved = dict(enriched)
    if not args.keep_state_vectors:
        for key in (
            "step_layer_state_vectors",
            "step_state_vectors",
            "step_vectors",
            "step_layer_state_memmap_path",
            "step_layer_state_memmap_count",
            "step_layer_state_temporary_memmap_path",
        ):
            saved.pop(key, None)
        saved["layer_time_geometry_state_vectors_embedded"] = np.asarray(False)
    else:
        saved["layer_time_geometry_state_vectors_embedded"] = np.asarray(True)
    np.savez_compressed(output, **saved)
    temporary_state_path = enriched.get("step_layer_state_temporary_memmap_path")
    if temporary_state_path is not None:
        raw_state = enriched.get("step_layer_state_vectors")
        mmap_obj = getattr(raw_state, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()
        Path(str(np.asarray(temporary_state_path).item())).unlink(missing_ok=True)
    output_dir = Path(args.output_dir) if args.output_dir else output.with_suffix("").parent / f"{output.stem}_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluate_all(enriched)
    field_summary = {
        "shape": list(np.asarray(enriched["layer_time_geometry_field"]).shape),
        "observables": list(LAYER_TIME_FIELD_NAMES),
        "layers": np.asarray(enriched["layer_time_geometry_layers"], dtype=np.int64).tolist(),
        "fold_by_chain_row": np.asarray(enriched["layer_time_geometry_fold"], dtype=np.int64).tolist(),
        "reference_sizes": np.asarray(enriched["layer_time_geometry_reference_sizes"], dtype=np.int64).tolist(),
        "lid_coverage": float(np.asarray(enriched["layer_time_geometry_lid_coverage"]).item()),
        "connection_coverage": float(np.asarray(enriched["layer_time_geometry_connection_coverage"]).item()),
        "holonomy_coverage": float(np.asarray(enriched["layer_time_geometry_holonomy_coverage"]).item()),
        "reliable_holonomy_coverage": float(
            np.asarray(enriched["layer_time_geometry_reliable_holonomy_coverage"]).item()
        ),
        "fiber_rank_mode": str(np.asarray(enriched["layer_time_geometry_fiber_rank_mode"]).item()),
        "max_transport_residual": float(
            np.asarray(enriched["layer_time_geometry_max_transport_residual"]).item()
        ),
        "null_mode": str(np.asarray(enriched["layer_time_geometry_null_mode"]).item()),
        "phase_mode": str(np.asarray(enriched["layer_time_geometry_phase_mode"]).item()),
        "compute_backend": str(
            np.asarray(enriched["layer_time_geometry_compute_backend"]).item()
        ),
        "compute_device": str(
            np.asarray(enriched["layer_time_geometry_compute_device"]).item()
        ),
        "reference_policy": str(np.asarray(enriched["layer_time_geometry_reference_policy"]).item()),
        "pooling_kind": str(np.asarray(enriched["layer_time_geometry_pooling_kind"]).item()),
    }
    save_json(summary, output_dir / "summary.json")
    save_json(field_summary, output_dir / "field_summary.json")
    compatibility_markdown = render_markdown(
        summary,
        title="Layer-Time Geometry Compatibility Metrics",
    )
    (output_dir / "summary.md").write_text(compatibility_markdown, encoding="utf-8")
    event_offsets = tuple(int(x.strip()) for x in str(args.event_offsets).split(",") if x.strip())
    if not {-1, 0, 1}.issubset(set(event_offsets)):
        raise ValueError("event_offsets must include -1, 0, and 1")
    validation = evaluate_layer_time_geometry(
        enriched,
        LayerTimeValidationConfig(
            event_offsets=event_offsets,
            bootstrap=max(0, int(args.validation_bootstrap)),
            random_seed=int(args.seed),
        ),
    )
    save_json(validation, output_dir / "layer_time_validation.json")
    (output_dir / "layer_time_validation.md").write_text(
        render_layer_time_validation(validation),
        encoding="utf-8",
    )
    write_step_csv(enriched, output_dir / "step_scores.csv")
    print(compatibility_markdown)
    print("\nGuardrail: the field is label-free and grouped by problem_id; scalar reductions are validation readouts, not the method object.")
    print(f"Saved layer-time geometry to {output}")
    print(f"Saved audit files to {output_dir}")


if __name__ == "__main__":
    main()
