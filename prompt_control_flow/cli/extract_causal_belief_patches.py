from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from prompt_control_flow.causal_belief_routing.patching import (
    SourcePatchConfig,
    extract_source_patches,
)
from prompt_control_flow.causal_belief_routing.schema import CausalBeliefTrace


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run donor-to-recipient source-specific attention-head path patches."
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--routing_summary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_replay_js", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--allow_model_mismatch", action="store_true")
    parser.add_argument("--allow_failed_routing_gate", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--no_compress", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trace = CausalBeliefTrace.load(args.trace)
    recorded_model = str(trace.metadata.get("model", ""))
    if (
        recorded_model
        and Path(recorded_model).name != Path(args.model).name
        and not args.allow_model_mismatch
    ):
        raise SystemExit(
            f"observer mismatch: trace={recorded_model!r}, replay={args.model!r}"
        )
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
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_kwargs = {
        "trust_remote_code": bool(args.trust_remote_code),
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()
    started = time.perf_counter()
    result = extract_source_patches(
        model,
        tokenizer,
        trace,
        args.routing_summary,
        SourcePatchConfig(
            max_pairs=int(args.max_pairs),
            max_seq_len=int(args.max_seq_len),
            max_replay_js=float(args.max_replay_js),
            show_progress=not bool(args.no_progress),
            allow_failed_routing_gate=bool(args.allow_failed_routing_gate),
        ),
        metadata={
            "model": str(args.model),
            "trace": str(args.trace),
            "device": str(device),
            "model_dtype": str(next(model.parameters()).dtype),
        },
    )
    elapsed = time.perf_counter() - started
    result.metadata["elapsed_seconds"] = float(elapsed)
    if device.type == "cuda":
        result.metadata["gpu_peak_mb"] = float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        )
    result.save(args.output, compressed=not bool(args.no_compress))
    print(
        f"saved {len(result.pair_ids)} patch directions from "
        f"{len(np.unique(result.pair_ids))} pairs to {args.output}"
    )
    print(
        f"coverage={result.metadata['coverage']:.3f} | "
        f"max replay JS={np.max(result.replay_js):.6f} | elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
