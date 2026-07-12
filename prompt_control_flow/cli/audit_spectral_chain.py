from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from prompt_control_flow.evaluate import evaluate_all, load_metric_npz, save_json
from prompt_control_flow.reports import render_markdown, write_step_csv
from prompt_control_flow.spectral_chain_dynamics import SpectralChainConfig, append_spectral_chain_dynamics


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Append whole-chain spectral-manifold dynamics scores. "
            "Accepts canonical ProcessBench full_*.npz files with stepvec, "
            "or mechanism npz files with step_state_vectors/step_vectors."
        )
    )
    ap.add_argument("--input", required=True, help="Canonical full_*.npz or mechanism metric npz")
    ap.add_argument("--output", required=True, help="Output npz with appended sd_* scores")
    ap.add_argument("--output_dir", default="", help="Optional directory for summary.json/summary.md/step_scores.csv")
    ap.add_argument("--vector_key", default="step_state_vectors", choices=["step_state_vectors", "step_vectors"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--low_modes", type=int, default=4)
    ap.add_argument("--max_landmarks", type=int, default=1800)
    ap.add_argument("--kernel_k", type=int, default=20)
    ap.add_argument("--committor_k", type=int, default=30)
    ap.add_argument("--tube_k", type=int, default=20)
    ap.add_argument("--phase_bandwidth", type=float, default=0.15)
    ap.add_argument("--tangent_k", type=int, default=32)
    ap.add_argument("--tangent_rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = SpectralChainConfig(
        vector_key=str(args.vector_key),
        n_folds=int(args.folds),
        n_modes=int(args.modes),
        low_modes=int(args.low_modes),
        max_landmarks=int(args.max_landmarks),
        kernel_k=int(args.kernel_k),
        committor_k=int(args.committor_k),
        tube_k=int(args.tube_k),
        phase_bandwidth=float(args.phase_bandwidth),
        tangent_k=int(args.tangent_k),
        tangent_rank=int(args.tangent_rank),
        random_seed=int(args.seed),
    )
    metrics = load_metric_npz(args.input)
    enriched = append_spectral_chain_dynamics(metrics, cfg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **enriched)

    out_dir = Path(args.output_dir) if args.output_dir else out_path.with_suffix("").parent / f"{out_path.stem}_spectral_chain_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate_all(enriched)
    save_json(summary, out_dir / "summary.json")
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    write_step_csv(enriched, out_dir / "step_scores.csv")
    print(render_markdown(summary))
    print("\nGuardrail: canonical full_*.npz supports ProcessBench first-error and cross-problem response diagnosis, not same-problem paired AUROC.")
    print(f"Saved spectral-chain metrics to {out_path}")
    print(f"Saved spectral-chain audit to {out_dir}")


if __name__ == "__main__":
    main()
