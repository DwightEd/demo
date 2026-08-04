from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from ..progress import TqdmProgress
from .cli import DEFAULT_DOMAINS, _csv, trace_sources
from .component_extraction import (
    ComponentExtractionConfig,
    ComponentTraceExtractor,
)
from .data import load_hidden_geometry_dataset
from .experiment import select_deterministic_domain_subset


def _layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated one-based decoder depths"
        ) from exc
    if not layers:
        raise argparse.ArgumentTypeError("at least one component layer is required")
    return layers


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    value = result.stdout.strip()
    return value or "unknown"


def _torch_dtype(torch_module, name: str):
    value = str(name).lower()
    if value == "auto":
        return "auto"
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if value not in mapping:
        raise ValueError(f"unsupported dtype {name!r}")
    return mapping[value]


def _load_model_and_tokenizer(args: argparse.Namespace):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "component extraction requires torch and transformers in the active "
            "Python environment"
        ) from exc

    model_dir = Path(args.model_dir).expanduser()
    tokenizer_dir = (
        Path(args.tokenizer_dir).expanduser() if args.tokenizer_dir else model_dir
    )
    if not model_dir.exists():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"tokenizer directory not found: {tokenizer_dir}")
    dtype = _torch_dtype(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    implementation = getattr(model.config, "_attn_implementation", None)
    if implementation is not None and str(implementation) != "eager":
        raise RuntimeError(
            "transformers did not load eager attention; "
            f"attn_implementation={implementation!r}"
        )
    return model, tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay stored ProcessBench token IDs and extract component_step_v1 "
            "attention/MLP/residual artifacts."
        )
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--domains", type=_csv, default=DEFAULT_DOMAINS)
    parser.add_argument("--manifest-name", default="trace.raw_residual_stream.npz")
    parser.add_argument("--component-dir-name", default="component_step_v1")
    parser.add_argument("--response-generator", default="llama3.1-8b")
    parser.add_argument("--observer-model", default="llama3.1-8b")
    parser.add_argument("--acquisition-mode", default="observer_teacher_forcing_replay")
    parser.add_argument(
        "--output-features",
        type=_csv,
        default=("token_entropy", "token_nll"),
    )
    parser.add_argument("--max-records-per-domain", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-revision", default="auto")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--tokenizer-revision", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--component-layers", type=_layers, required=True)
    parser.add_argument(
        "--attention-reconstruction-atol",
        type=float,
        default=3e-2,
    )
    parser.add_argument(
        "--attention-reconstruction-rtol",
        type=float,
        default=3e-2,
    )
    parser.add_argument("--replay-fidelity-rtol", type=float, default=3e-2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_records_per_domain < 0:
        raise ValueError("--max-records-per-domain cannot be negative")
    sources = trace_sources(
        args.data_root,
        args.domains,
        args.manifest_name,
        args.acquisition_mode,
        args.component_dir_name,
    )
    data = load_hidden_geometry_dataset(
        sources,
        response_generator=args.response_generator,
        observer_model=args.observer_model,
        output_features=args.output_features,
    )
    data = select_deterministic_domain_subset(
        data,
        int(args.max_records_per_domain),
        int(args.seed),
    )
    model, tokenizer = _load_model_and_tokenizer(args)
    model_name = args.model_name or str(
        getattr(model.config, "_name_or_path", args.model_dir)
    )
    tokenizer_name = args.tokenizer_name or str(
        getattr(tokenizer, "name_or_path", args.tokenizer_dir or args.model_dir)
    )
    config = ComponentExtractionConfig(
        layers=args.component_layers,
        model_name=model_name,
        model_revision=args.model_revision,
        tokenizer_name=tokenizer_name,
        tokenizer_revision=args.tokenizer_revision,
        extractor_commit=_git_commit(),
        attention_reconstruction_atol=args.attention_reconstruction_atol,
        attention_reconstruction_rtol=args.attention_reconstruction_rtol,
        replay_fidelity_rtol=args.replay_fidelity_rtol,
        overwrite=bool(args.overwrite),
    )
    reporter = TqdmProgress()
    reporter.stage(
        "extract",
        (
            f"{len(data.samples)} chains | layers={list(config.layers)} | "
            f"replay_config={config.replay_config_sha256()}"
        ),
    )
    result = ComponentTraceExtractor(config).extract(
        model=model,
        samples=data.samples,
        sources=sources,
        progress=reporter,
    )
    domains = ", ".join(str(value) for value in np.unique(data.domains))
    print(f"component_step_v1 domains: {domains}")
    print(f"selected_chains: {len(data.samples)}")
    print(f"written: {len(result.written)}")
    print(f"skipped: {len(result.skipped)}")
    for source in sources:
        print(f"{source.dataset}: component_dir={source.component_dir}")


if __name__ == "__main__":
    main()
