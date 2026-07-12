#!/usr/bin/env python3
"""Single entry point for ECGH environment checks and deterministic gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

from anchorflow_validation_suite import audit_npz, synthetic_validation
import second_moment_dynamics_audit as smd


ROOT = Path(__file__).resolve().parent


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def doctor_report() -> dict:
    packages = {
        name: _available(module)
        for name, module in {
            "numpy": "numpy",
            "scipy": "scipy",
            "scikit-learn": "sklearn",
            "pytest": "pytest",
            "hydra-core": "hydra",
            "omegaconf": "omegaconf",
            "torch": "torch",
            "transformers": "transformers",
            "datasets": "datasets",
            "torch-geometric": "torch_geometric",
            "python-pptx": "pptx",
        }.items()
    }
    required_files = [
        "10_sample_and_extract.py",
        "01_extract_spectral_field.py",
        "anchorflow_validation_suite.py",
        "second_moment_dynamics_audit.py",
        "multisample_temporal_rupture_audit.py",
        "hypergraph_token_hgn.py",
    ]
    files = {name: (ROOT / name).exists() for name in required_files}
    skill_root = ROOT.parent / ".agents" / "skills"
    skill_count = sum(
        1 for path in skill_root.iterdir()
        if path.is_dir() and (path / "SKILL.md").exists()
    ) if skill_root.exists() else 0

    cpu_ready = all(packages[name] for name in (
        "numpy", "scipy", "scikit-learn", "pytest", "hydra-core", "omegaconf"
    )) and all(files.values())
    gpu_ready = cpu_ready and all(packages[name] for name in (
        "torch", "transformers", "datasets"
    ))
    hypergraph_ready = gpu_ready and packages["torch-geometric"]
    return {
        "status": "passed" if cpu_ready else "failed",
        "python": sys.version.split()[0],
        "project_root": str(ROOT),
        "packages": packages,
        "files": files,
        "installed_project_skills": skill_count,
        "workflows": {
            "cpu_unit_and_synthetic": "ready" if cpu_ready else "blocked_dependency",
            "gpu_exact_trace": "environment_ready_data_and_model_required" if gpu_ready else "blocked_dependency",
            "hypergraph_upper_bound": "environment_ready_data_required" if hypergraph_ready else "blocked_dependency",
            "real_second_moment_audit": "input_npz_required",
            "streaming_intervention": "prototype_only_not_claimed_runnable",
        },
    }


def selftest_report(seed: int) -> dict:
    anchorflow = synthetic_validation(seed)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "second_moment_selftest.npz")
        smd.make_selftest(path, seed=seed, n_problems=12, samples_per_problem=6)
        args = smd.build_arg_parser().parse_args([])
        args.seed = int(seed)
        args.folds = 3
        args.bootstrap = 30
        args.spectral_backend = "cpu"
        args.no_progress = True
        second = smd.run(path, args)
        smd.assert_selftest(second)
    return {
        "status": "passed",
        "anchorflow": anchorflow,
        "second_moment": {
            "headline": second["headline"],
            "coverage": second["coverage"],
        },
    }


def _write_or_print(result: dict, output: str | None) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_doctor = sub.add_parser("doctor", help="check dependencies and workflow readiness")
    p_doctor.add_argument("--output")
    p_self = sub.add_parser("selftest", help="run CPU-only end-to-end synthetic gates")
    p_self.add_argument("--seed", type=int, default=7)
    p_self.add_argument("--output")
    p_trace = sub.add_parser("trace-audit", help="audit one exact-trace NPZ")
    p_trace.add_argument("--input", required=True)
    p_trace.add_argument("--layer", type=int, default=16)
    p_trace.add_argument("--max_chains", type=int, default=0)
    p_trace.add_argument("--output")
    args = parser.parse_args()

    if args.command == "doctor":
        result = doctor_report()
    elif args.command == "selftest":
        result = selftest_report(args.seed)
    else:
        result = audit_npz(args.input, args.layer, args.max_chains)
    _write_or_print(result, args.output)
    if result.get("status") != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
