from __future__ import annotations

import argparse
from pathlib import Path
import time

from prompt_control_flow.causal_belief_update_decomposition.charts import LayerChartBundle
from prompt_control_flow.causal_belief_update_decomposition.schema import CausalBeliefTrace
from prompt_control_flow.causal_belief_update_decomposition.update_extraction import (
    BeliefUpdateExtractionConfig,
    extract_belief_update_decomposition,
    representation_gate_failure_message,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract target-token attention, MLP, and actual block writes in "
            "held-out finite-field belief coordinates."
        )
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--charts", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batch_tokens", type=int, default=4096)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--allow_model_mismatch", action="store_true")
    parser.add_argument("--allow_failed_representation_gate", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--no_compress", action="store_true")
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


def main() -> None:
    args = build_parser().parse_args()
    charts = LayerChartBundle.load(args.charts)
    gate_failure = representation_gate_failure_message(charts)
    if gate_failure is not None:
        gate_report = Path(args.charts).parent / "summary.json"
        if not args.allow_failed_representation_gate:
            raise SystemExit(
                f"{gate_failure}\n"
                "model loading skipped because the resulting component scores would "
                "not support a formal mechanism claim\n"
                f"representation report: {gate_report}"
            )
        print(f"WARNING: exploratory failed-gate override enabled: {gate_failure}")

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
            f"observer mismatch: trace={recorded_model!r}, replay={args.model!r}; "
            "use the exact observer or explicitly mark the run exploratory"
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
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()
    started = time.perf_counter()
    updates = extract_belief_update_decomposition(
        model,
        tokenizer,
        trace,
        charts,
        BeliefUpdateExtractionConfig(
            batch_size=int(args.batch_size),
            max_batch_tokens=int(args.max_batch_tokens),
            max_seq_len=int(args.max_seq_len),
            show_progress=not bool(args.no_progress),
            allow_failed_representation_gate=bool(
                args.allow_failed_representation_gate
            ),
        ),
        metadata={
            "model": str(args.model),
            "source_trace": str(args.trace),
            "source_charts": str(args.charts),
            "model_dtype": str(next(model.parameters()).dtype),
            "device": str(device),
        },
    )
    elapsed = time.perf_counter() - started
    updates.metadata["elapsed_seconds"] = float(elapsed)
    updates.metadata["rows_per_second"] = float(
        len(updates.row_indices) / max(elapsed, 1e-9)
    )
    if device.type == "cuda":
        updates.metadata["gpu_peak_mb"] = float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        )
    updates.save(args.output, compressed=not bool(args.no_compress))
    print(
        f"saved belief-update decomposition for {len(updates.row_indices)} rows, "
        f"layers={updates.layers.tolist()}"
    )
    print(f"output: {args.output} | elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
