from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from prompt_control_flow.config import ExtractionConfig
from prompt_control_flow.data import load_chain_records
from prompt_control_flow.extraction import extract_chain_prompt_flow, save_extractions


def parse_layers(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract prompt-controlled residual-flow metrics.")
    ap.add_argument("--input", required=True, help="ProcessBench full or multisample npz with problems/steps_text.")
    ap.add_argument("--model", required=True, help="HF model path/name.")
    ap.add_argument("--output", required=True, help="Output metric npz.")
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--layers", default="8,10,12,14,16,18,20,22")
    ap.add_argument("--subspace_k", type=int, default=16)
    ap.add_argument("--prefix_k", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--no_entropy", action="store_true")
    ap.add_argument(
        "--store_step_vectors",
        action="store_true",
        help="Save per-step residual-flow vectors for PCA/VAE/spectral chart comparison.",
    )
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
        include_entropy=not args.no_entropy,
        store_step_vectors=bool(args.store_step_vectors),
    )
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    dtype = None
    if args.dtype.lower() in {"float16", "fp16"}:
        dtype = torch.float16
    elif args.dtype.lower() in {"bfloat16", "bf16"}:
        dtype = torch.bfloat16
    elif args.dtype.lower() in {"float32", "fp32"}:
        dtype = torch.float32

    print(f"Loading records from {args.input}")
    records = load_chain_records(args.input, max_chains=args.max_chains)
    print(f"  records: {len(records)}")
    print(f"Loading model {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if not getattr(tok, "is_fast", False):
        raise SystemExit("This extractor requires a fast tokenizer for offset_mapping.")
    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()

    extractions = []
    for rec in tqdm(records, desc="chains"):
        item = extract_chain_prompt_flow(model, tok, rec, cfg)
        if item is not None:
            extractions.append(item)
    if not extractions:
        raise SystemExit("No chains were successfully extracted.")
    save_extractions(extractions, args.output)
    print(f"Saved {len(extractions)} chain metrics to {args.output}")


if __name__ == "__main__":
    main()
