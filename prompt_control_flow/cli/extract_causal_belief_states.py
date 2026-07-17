from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time
from typing import Sequence

from prompt_control_flow.causal_belief_routing.extraction import (
    StateExtractionConfig,
    extract_causal_belief_states,
)
from prompt_control_flow.causal_belief_routing.world import load_alias_worlds_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract compact all-layer states for predictive-alias worlds."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,28,32")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batch_tokens", type=int, default=4096)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--logit_sketch_dim", type=int, default=256)
    parser.add_argument("--logit_sketch_seed", type=int, default=1701)
    parser.add_argument("--state_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    return parser


def _torch_dtype(torch, requested: str, device):
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

    worlds, world_cfg = load_alias_worlds_jsonl(args.input)
    if int(args.max_pairs) > 0:
        worlds = worlds[: int(args.max_pairs)]
    worlds = [
        world
        for world in worlds
        if int(world.pair_id) % int(args.num_shards) == int(args.shard_index)
    ]
    if not worlds:
        raise SystemExit("selected predictive-alias shard is empty")

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

    cfg = StateExtractionConfig(
        layers=str(args.layers),
        batch_size=int(args.batch_size),
        max_batch_tokens=int(args.max_batch_tokens),
        max_seq_len=int(args.max_seq_len),
        logit_sketch_dim=int(args.logit_sketch_dim),
        logit_sketch_seed=int(args.logit_sketch_seed),
        state_dtype=str(args.state_dtype),
        show_progress=not bool(args.no_progress),
    )
    started = time.perf_counter()
    trace = extract_causal_belief_states(
        model,
        tokenizer,
        worlds,
        cfg,
        metadata={
            "model": str(args.model),
            "tokenizer": str(getattr(tokenizer, "name_or_path", args.model)),
            "model_revision": str(getattr(model.config, "_commit_hash", "") or ""),
            "transformers_version": str(getattr(model.config, "transformers_version", "")),
            "device": str(device),
            "model_dtype": str(next(model.parameters()).dtype),
            "world_config": asdict(world_cfg),
            "source": str(Path(args.input)),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
        },
    )
    elapsed = time.perf_counter() - started
    trace.metadata["elapsed_seconds"] = float(elapsed)
    trace.metadata["rows_per_second"] = float(trace.n_rows / max(elapsed, 1e-9))
    if device.type == "cuda":
        trace.metadata["gpu_peak_mb"] = float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        )
    output = Path(args.output)
    trace.save(output, compressed=bool(args.compress))
    print(
        f"saved {trace.n_rows} rows from {len(set(trace.pair_ids.tolist()))} "
        f"predictive-alias pairs to {output}"
    )
    print(
        f"layers={trace.layers.tolist()} | elapsed={elapsed:.1f}s | "
        f"rows/s={trace.n_rows / max(elapsed, 1e-9):.2f}"
    )


if __name__ == "__main__":
    main()
