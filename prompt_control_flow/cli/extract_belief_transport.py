from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time
from typing import Sequence

from prompt_control_flow.belief_transport.extraction import (
    BoundaryExtractionConfig,
    extract_boundary_belief_trace,
)
from prompt_control_flow.belief_transport.world import load_worlds_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract compact residual belief-boundary traces on one device."
    )
    parser.add_argument("--input", required=True, help="Wind-tunnel JSONL.")
    parser.add_argument("--model", required=True, help="HF model name or local path.")
    parser.add_argument("--output", required=True, help="Output trace NPZ.")
    parser.add_argument("--layers", default="8,12,16,20,24,28,32")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batch_tokens", type=int, default=3072)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--output_top_k", type=int, default=64)
    parser.add_argument("--output_sketch_dim", type=int, default=64)
    parser.add_argument("--output_sketch_seed", type=int, default=991)
    parser.add_argument("--state_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--max_problems", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    return parser


def _torch_dtype(torch, requested: str, device) -> object | None:
    if requested == "auto":
        if device.type != "cuda":
            return None
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[requested]


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if int(args.num_shards) < 1:
        raise SystemExit("--num_shards must be positive")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise SystemExit("--shard_index must lie in [0, num_shards)")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    dtype = _torch_dtype(torch, str(args.dtype), device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    worlds, world_cfg = load_worlds_jsonl(args.input)
    if int(args.max_problems) > 0:
        worlds = worlds[: int(args.max_problems)]
    worlds = [
        world
        for world in worlds
        if int(world.problem_id) % int(args.num_shards) == int(args.shard_index)
    ]
    if not worlds:
        raise SystemExit("selected wind-tunnel shard is empty")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise SystemExit("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_kwargs = {
        "trust_remote_code": bool(args.trust_remote_code),
        "low_cpu_mem_usage": True,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()

    extraction_cfg = BoundaryExtractionConfig(
        layers=str(args.layers),
        batch_size=int(args.batch_size),
        max_batch_tokens=int(args.max_batch_tokens),
        max_seq_len=int(args.max_seq_len),
        output_top_k=int(args.output_top_k),
        output_sketch_dim=int(args.output_sketch_dim),
        output_sketch_seed=int(args.output_sketch_seed),
        state_dtype=str(args.state_dtype),
        show_progress=not bool(args.no_progress),
    )
    started = time.perf_counter()
    artifact = extract_boundary_belief_trace(
        model,
        tokenizer,
        worlds,
        extraction_cfg,
        metadata={
            "model": str(args.model),
            "tokenizer": str(getattr(tokenizer, "name_or_path", args.model)),
            "device": str(device),
            "model_dtype": str(next(model.parameters()).dtype),
            "world_config": asdict(world_cfg),
            "source": str(Path(args.input)),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
        },
    )
    elapsed = time.perf_counter() - started
    artifact.metadata["elapsed_seconds"] = float(elapsed)
    artifact.metadata["rows_per_second"] = float(artifact.n_rows / max(elapsed, 1e-9))
    if device.type == "cuda":
        artifact.metadata["gpu_peak_mb"] = float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        )
    output = Path(args.output)
    artifact.save(output, compressed=bool(args.compress))
    print(
        f"saved {artifact.n_rows} prefix rows from "
        f"{len(set(artifact.problem_ids.tolist()))} problems to {output}"
    )
    print(
        f"layers={artifact.layers.tolist()} | elapsed={elapsed:.1f}s | "
        f"rows/s={artifact.n_rows / max(elapsed, 1e-9):.2f}"
    )


if __name__ == "__main__":
    main()
