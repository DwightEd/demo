from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from prompt_control_flow.evaluate import evaluate_all, load_metric_npz, save_json
from prompt_control_flow.reports import render_markdown, write_step_csv
from prompt_control_flow.representation_geometry import GeometryAuditConfig, append_geometry_audit


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Append cross-fitted representation-geometry scores to mechanism metrics.")
    ap.add_argument("--input", required=True, help="Metric npz from extract_mechanisms.py")
    ap.add_argument("--output", required=True, help="Output npz with appended geom_* scores")
    ap.add_argument("--output_dir", default="", help="Optional directory for summary.json/summary.md/step_scores.csv")
    ap.add_argument("--vector_key", default="step_state_vectors", choices=["step_state_vectors", "step_vectors"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--knn_k", type=int, default=20)
    ap.add_argument("--pca_var", type=float, default=0.90)
    ap.add_argument("--max_pca_rank", type=int, default=64)
    ap.add_argument("--random_projection_dim", type=int, default=128)
    ap.add_argument("--layer_projection_dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--chunk_size", type=int, default=256)
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = GeometryAuditConfig(
        vector_key=args.vector_key,
        n_folds=int(args.folds),
        knn_k=int(args.knn_k),
        pca_var=float(args.pca_var),
        max_pca_rank=int(args.max_pca_rank),
        random_projection_dim=int(args.random_projection_dim),
        layer_projection_dim=int(args.layer_projection_dim),
        random_seed=int(args.seed),
        chunk_size=int(args.chunk_size),
    )
    metrics = load_metric_npz(args.input)
    enriched = append_geometry_audit(metrics, cfg)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **enriched)

    out_dir = Path(args.output_dir) if args.output_dir else out_path.with_suffix("").parent / f"{out_path.stem}_geometry_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate_all(enriched)
    save_json(summary, out_dir / "summary.json")
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    write_step_csv(enriched, out_dir / "step_scores.csv")
    print(render_markdown(summary))
    print(f"\nSaved geometry-enriched metrics to {out_path}")
    print(f"Saved geometry audit to {out_dir}")


if __name__ == "__main__":
    main()
