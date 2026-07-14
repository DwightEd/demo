from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prompt_control_flow.data import (
    load_chain_records,
    validate_records_against_reference,
)
from prompt_control_flow.profiler import MechanismProfiler


def _identity_matches(source: str, requested: str) -> bool:
    source = str(source).strip().replace("\\", "/").rstrip("/").lower()
    requested = str(requested).strip().replace("\\", "/").rstrip("/").lower()
    if not source or not requested or source == requested:
        return True
    return source.rsplit("/", 1)[-1] == requested.rsplit("/", 1)[-1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract compact causal output traces for the OC-GPI audit."
    )
    parser.add_argument(
        "--input", required=True, help="ProcessBench JSONL or exact-trace NPZ."
    )
    parser.add_argument(
        "--input_format",
        default="auto",
        choices=(
            "auto",
            "npz",
            "processbench_jsonl",
            "processbench_source",
            "jsonl",
        ),
    )
    parser.add_argument(
        "--subset",
        default="",
        help="ProcessBench subset for --input_format processbench_source.",
    )
    parser.add_argument(
        "--model", required=True, help="Observer causal-LM path or HF name."
    )
    parser.add_argument(
        "--geometry_reference",
        default="",
        help="Optional canonical full_*.npz checked before loading the model.",
    )
    parser.add_argument("--output", required=True, help="Output compact trace NPZ.")
    parser.add_argument("--max_chains", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--sketch_dim", type=int, default=64)
    parser.add_argument("--token_chunk_size", type=int, default=32)
    parser.add_argument("--min_success_fraction", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"),
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--allow_model_mismatch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not 0.0 <= float(args.min_success_fraction) <= 1.0:
        raise SystemExit("--min_success_fraction must lie in [0, 1]")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from prompt_control_flow.ocgpi.extraction import (
        CompactTraceAccumulator,
        extract_compact_trace_item,
    )
    from prompt_control_flow.ocgpi.logit_trace import LogitTraceConfig

    torch.set_float32_matmul_precision("high")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(str(args.dtype).lower())
    cfg = LogitTraceConfig(
        top_k=int(args.top_k),
        sketch_dim=int(args.sketch_dim),
        token_chunk_size=int(args.token_chunk_size),
    )
    cfg.validate()
    records = load_chain_records(
        args.input,
        max_chains=int(args.max_chains),
        input_format=args.input_format,
        subset=str(args.subset) or None,
    )
    if not records:
        raise SystemExit("No records were loaded.")
    source_preflight = (
        validate_records_against_reference(records, args.geometry_reference)
        if args.geometry_reference
        else {"reference_path": "", "checked_records": 0}
    )
    exact = [record for record in records if record.exact_input_ids is not None]
    mismatched = sorted(
        {
            str(record.generator)
            for record in exact
            if record.generator
            and not _identity_matches(str(record.generator), args.model)
        }
    )
    if mismatched and not args.allow_model_mismatch:
        raise SystemExit(
            f"Exact trace source model {mismatched} does not match --model {args.model!r}."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit("OC-GPI fallback alignment requires a fast tokenizer.")
    tokenizer_mismatch = sorted(
        {
            str(record.source_tokenizer)
            for record in exact
            if record.source_tokenizer
            and not _identity_matches(
                str(record.source_tokenizer), str(tokenizer.name_or_path)
            )
        }
    )
    if tokenizer_mismatch and not args.allow_model_mismatch:
        raise SystemExit(
            f"Exact trace source tokenizer {tokenizer_mismatch} does not match the loaded tokenizer."
        )
    model_kwargs = {
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()

    accumulator = CompactTraceAccumulator(cfg=cfg)
    profiler = MechanismProfiler()
    for record in tqdm(records, desc="OC-GPI output traces"):
        profiler.record_chain()
        try:
            with profiler.phase("compact_teacher_forcing"):
                item = extract_compact_trace_item(
                    model,
                    tokenizer,
                    record,
                    cfg=cfg,
                    max_seq_len=int(args.max_seq_len),
                )
            accumulator.add(item)
            profiler.record_success()
            profiler.record_seq_len(item.metadata.get("sequence_tokens"))
        except Exception as exc:
            accumulator.add_skip(record, exc)
            profiler.record_skip(type(exc).__name__)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    profiler.save_json(output.parent / f"{output.stem}_extraction_profile.json")
    with (output.parent / f"{output.stem}_skip_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(accumulator.skipped, handle, indent=2, ensure_ascii=False)
    coverage = len(accumulator.items) / max(len(records), 1)
    covered_problems = {item.problem_id for item in accumulator.items}
    problem_coverage = len(covered_problems) / max(
        len({record.problem_id for record in records}), 1
    )
    if min(coverage, problem_coverage) < float(args.min_success_fraction):
        raise SystemExit(
            f"Extraction coverage failed: chains={coverage:.3f}, problems={problem_coverage:.3f}, "
            f"required={float(args.min_success_fraction):.3f}. No trace artifact was written."
        )
    artifact = accumulator.pack(
        {
            "model": str(args.model),
            "tokenizer": str(tokenizer.name_or_path),
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "chain_coverage": float(coverage),
            "problem_coverage": float(problem_coverage),
            "full_logits_persisted": False,
            "causal_alignment": "logits[position-1] predicts response token[position]",
            "source_preflight": source_preflight,
        }
    )
    artifact.save(output)
    print(f"saved {artifact.n_chains} compact traces to {output}")
    print(f"chain coverage {coverage:.3f} | problem coverage {problem_coverage:.3f}")


if __name__ == "__main__":
    main()
