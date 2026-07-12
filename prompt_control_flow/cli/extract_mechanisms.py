from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from prompt_control_flow.config import ExtractionConfig
from prompt_control_flow.data import load_chain_records
from prompt_control_flow.extraction import extract_chain_mechanisms, save_extractions
from prompt_control_flow.extractors import build_extractors
from prompt_control_flow.profiler import MechanismProfiler


def parse_layers(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract residual-flow mechanism metrics.")
    ap.add_argument("--input", required=True, help="ProcessBench jsonl, full npz, or multisample npz.")
    ap.add_argument("--input_format", default="auto", choices=["auto", "npz", "processbench_jsonl", "jsonl"])
    ap.add_argument("--model", required=True, help="HF model path/name.")
    ap.add_argument("--output", required=True, help="Output metric npz.")
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--layers", default="8,10,12,14,16,18,20,22")
    ap.add_argument("--subspace_k", type=int, default=16)
    ap.add_argument("--prefix_k", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--full_attention_token_threshold", type=int, default=1200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--enable_prompt_flow", action="store_true", help="Kept for explicit CLI symmetry; prompt-flow is enabled unless --no_prompt_flow is set.")
    ap.add_argument("--no_prompt_flow", action="store_true")
    ap.add_argument("--enable_uncertainty", action="store_true", help="Kept for explicit CLI symmetry; uncertainty is enabled unless --no_uncertainty is set.")
    ap.add_argument("--no_uncertainty", action="store_true")
    ap.add_argument("--enable_icr", action="store_true", help="Enable attention-residual ICR-style mismatch scores.")
    ap.add_argument("--icr_top_k", type=int, default=20)
    ap.add_argument("--icr_top_p", type=float, default=None)
    ap.add_argument("--store_step_vectors", action="store_true")
    ap.add_argument("--store_step_state_vectors", action="store_true", help="Store pooled per-step hidden states for representation-geometry audits.")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = ExtractionConfig(
        layers=parse_layers(args.layers),
        subspace_k=int(args.subspace_k),
        prefix_k=int(args.prefix_k),
        max_seq_len=int(args.max_seq_len),
        device=args.device,
        dtype=args.dtype,
        include_entropy=not args.no_uncertainty,
        store_step_vectors=bool(args.store_step_vectors),
        store_step_state_vectors=bool(args.store_step_state_vectors),
        full_attention_token_threshold=int(args.full_attention_token_threshold),
        icr_top_k=int(args.icr_top_k),
        icr_top_p=args.icr_top_p,
    )
    prompt_flow = not args.no_prompt_flow
    uncertainty = not args.no_uncertainty
    extractors = build_extractors(prompt_flow=prompt_flow, uncertainty=uncertainty, icr=bool(args.enable_icr))
    if not extractors:
        raise SystemExit("No extractors enabled.")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    dtype = None
    if args.dtype.lower() in {"float16", "fp16"}:
        dtype = torch.float16
    elif args.dtype.lower() in {"bfloat16", "bf16"}:
        dtype = torch.bfloat16
    elif args.dtype.lower() in {"float32", "fp32"}:
        dtype = torch.float32

    records = load_chain_records(args.input, max_chains=args.max_chains, input_format=args.input_format)
    print(f"records: {len(records)}")
    print(f"extractors: {[e.name for e in extractors]}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if not getattr(tok, "is_fast", False):
        raise SystemExit("This extractor requires a fast tokenizer for offset_mapping.")

    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if args.enable_icr:
        model_kwargs["attn_implementation"] = "eager"
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()

    profiler = MechanismProfiler()
    extractions = []
    skip_rows = []
    for rec in tqdm(records, desc="chains"):
        profiler.record_chain()
        try:
            with profiler.phase("teacher_forcing_and_mechanism_extraction"):
                item = extract_chain_mechanisms(model, tok, rec, cfg, extractors)
        except Exception as exc:
            profiler.record_skip(type(exc).__name__)
            skip_rows.append({"chain_idx": rec.chain_idx, "problem_id": rec.problem_id, "reason": repr(exc)})
            continue
        if item is None:
            profiler.record_skip("empty_or_unaligned")
            skip_rows.append({"chain_idx": rec.chain_idx, "problem_id": rec.problem_id, "reason": "empty_or_unaligned"})
            continue
        profiler.record_success()
        profiler.record_seq_len(item.metadata.get("seq_len"))
        extractions.append(item)

    if not extractions:
        raise SystemExit("No chains were successfully extracted.")
    save_extractions(extractions, args.output)
    out_path = Path(args.output)
    profiler.save_json(out_path.parent / "profile_summary.json")
    with (out_path.parent / "skip_report.json").open("w", encoding="utf-8") as f:
        json.dump(skip_rows, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(extractions)} chain metrics to {args.output}")
    print(f"Saved profile to {out_path.parent / 'profile_summary.json'}")


if __name__ == "__main__":
    main()
