from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prompt_control_flow.config import ExtractionConfig
from prompt_control_flow.data import load_chain_records
from prompt_control_flow.extraction import extract_chain_mechanisms, pack_extractions, save_extractions
from prompt_control_flow.extractors import build_extractors
from prompt_control_flow.profiler import MechanismProfiler
from utils.step_boundaries import TokenAlignmentError


def parse_layers(s: str) -> tuple[int, ...]:
    if str(s).strip().lower() == "all":
        return ()
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def model_identity_matches(source: str, requested: str) -> bool:
    source = str(source).strip().replace("\\", "/").rstrip("/").lower()
    requested = str(requested).strip().replace("\\", "/").rstrip("/").lower()
    if not source or not requested:
        return True
    if source == requested:
        return True
    source_local = source.startswith("/") or (len(source) >= 3 and source[1:3] == ":/")
    requested_local = requested.startswith("/") or (len(requested) >= 3 and requested[1:3] == ":/")
    return bool(
        (source_local or requested_local)
        and source.rsplit("/", 1)[-1] == requested.rsplit("/", 1)[-1]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract residual-flow mechanism metrics.")
    ap.add_argument("--input", required=True, help="ProcessBench jsonl, full npz, or multisample npz.")
    ap.add_argument("--input_format", default="auto", choices=["auto", "npz", "processbench_jsonl", "jsonl"])
    ap.add_argument("--model", required=True, help="HF model path/name.")
    ap.add_argument("--output", required=True, help="Output metric npz.")
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--min_success_fraction", type=float, default=0.95, help="Fail without writing a state artifact if chain or problem extraction coverage falls below this value.")
    ap.add_argument("--layers", default="8,10,12,14,16,18,20,22", help="Comma-separated hidden-state depths, or 'all' for every post-block layer.")
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
    ap.add_argument("--geometry_only", action="store_true", help="Extract only the whole-layer [step, layer, hidden] state tensor; implies --layers all.")
    ap.add_argument("--allow_model_mismatch", action="store_true", help="Explicitly allow exact token IDs from a differently named source model/tokenizer (unsafe ablation).")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    required_coverage = float(args.min_success_fraction)
    if not 0.0 <= required_coverage <= 1.0:
        raise SystemExit("--min_success_fraction must be in [0, 1]")
    if args.geometry_only and (args.enable_icr or args.store_step_vectors):
        raise SystemExit("--geometry_only cannot be combined with --enable_icr or --store_step_vectors")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = ExtractionConfig(
        layers=() if args.geometry_only else parse_layers(args.layers),
        subspace_k=int(args.subspace_k),
        prefix_k=int(args.prefix_k),
        max_seq_len=int(args.max_seq_len),
        device=args.device,
        dtype=args.dtype,
        include_entropy=False if args.geometry_only else not args.no_uncertainty,
        store_step_vectors=False if args.geometry_only else bool(args.store_step_vectors),
        store_step_state_vectors=bool(args.store_step_state_vectors or args.geometry_only),
        store_flat_step_state_vectors=not bool(args.geometry_only),
        full_attention_token_threshold=int(args.full_attention_token_threshold),
        icr_top_k=int(args.icr_top_k),
        icr_top_p=args.icr_top_p,
    )
    prompt_flow = False if args.geometry_only else not args.no_prompt_flow
    uncertainty = False if args.geometry_only else not args.no_uncertainty
    extractors = build_extractors(
        prompt_flow=prompt_flow,
        uncertainty=uncertainty,
        icr=False if args.geometry_only else bool(args.enable_icr),
    )
    if not extractors and not cfg.store_step_state_vectors:
        raise SystemExit("No extractors enabled and no state tensor requested.")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    dtype = None
    if args.dtype.lower() in {"float16", "fp16"}:
        dtype = torch.float16
    elif args.dtype.lower() in {"bfloat16", "bf16"}:
        dtype = torch.bfloat16
    elif args.dtype.lower() in {"float32", "fp32"}:
        dtype = torch.float32

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_chain_records(args.input, max_chains=args.max_chains, input_format=args.input_format)
    exact_records = [record for record in records if record.exact_input_ids is not None]
    source_models = sorted({record.generator for record in exact_records if record.generator})
    mismatched_models = [name for name in source_models if not model_identity_matches(name, args.model)]
    if mismatched_models and not args.allow_model_mismatch:
        raise SystemExit(
            f"Exact artifact model {mismatched_models} does not match --model {args.model!r}. "
            "Use the original model, or pass --allow_model_mismatch only for an explicit unsafe ablation."
        )
    print(f"records: {len(records)}")
    print(f"extractors: {[e.name for e in extractors]}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if not getattr(tok, "is_fast", False):
        raise SystemExit("This extractor requires a fast tokenizer for offset_mapping.")
    tokenizer_name = str(getattr(tok, "name_or_path", args.model))
    source_tokenizers = sorted({record.source_tokenizer for record in exact_records if record.source_tokenizer})
    mismatched_tokenizers = [
        name for name in source_tokenizers if not model_identity_matches(name, tokenizer_name)
    ]
    if mismatched_tokenizers and not args.allow_model_mismatch:
        raise SystemExit(
            f"Exact artifact tokenizer {mismatched_tokenizers} does not match loaded tokenizer "
            f"{tokenizer_name!r}."
        )
    loaded_tokenizer_revision = str(getattr(tok, "init_kwargs", {}).get("revision", "") or "")
    source_tokenizer_revisions = sorted(
        {record.source_tokenizer_revision for record in exact_records if record.source_tokenizer_revision}
    )
    if (
        loaded_tokenizer_revision
        and source_tokenizer_revisions
        and any(revision != loaded_tokenizer_revision for revision in source_tokenizer_revisions)
        and not args.allow_model_mismatch
    ):
        raise SystemExit(
            f"Exact artifact tokenizer revision {source_tokenizer_revisions} does not match "
            f"loaded revision {loaded_tokenizer_revision!r}."
        )

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
    loaded_model_revision = str(getattr(model.config, "_commit_hash", "") or "")
    source_model_revisions = sorted(
        {record.source_model_revision for record in exact_records if record.source_model_revision}
    )
    if (
        loaded_model_revision
        and source_model_revisions
        and any(revision != loaded_model_revision for revision in source_model_revisions)
        and not args.allow_model_mismatch
    ):
        raise SystemExit(
            f"Exact artifact model revision {source_model_revisions} does not match "
            f"loaded revision {loaded_model_revision!r}."
        )

    profiler = MechanismProfiler()
    extractions = []
    skip_rows = []
    state_memmap = None
    state_run_id = uuid.uuid4().hex[:12]
    state_partial_path = out_path.with_suffix(f".states.{state_run_id}.partial.npy")
    state_final_path = out_path.with_suffix(f".states.{state_run_id}.npy")
    manifest_partial_path = out_path.with_suffix(".partial.npz")
    if args.geometry_only:
        state_partial_path.unlink(missing_ok=True)
        manifest_partial_path.unlink(missing_ok=True)
    state_cursor = 0
    state_chain_idx: list[int] = []
    state_step_idx: list[int] = []
    expected_state_capacity = sum(
        len(record.exact_step_token_ranges)
        if record.exact_step_token_ranges is not None
        else len(record.steps)
        for record in records
    )
    for rec in tqdm(records, desc="chains"):
        profiler.record_chain()
        try:
            with profiler.phase("teacher_forcing_and_mechanism_extraction"):
                item = extract_chain_mechanisms(model, tok, rec, cfg, extractors)
        except TokenAlignmentError:
            # Exact-token corruption invalidates the run, not just one row.
            if state_memmap is not None:
                state_memmap.flush()
                state_memmap._mmap.close()
                state_partial_path.unlink(missing_ok=True)
            raise
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
        if args.geometry_only:
            try:
                tensor = np.asarray(item.step_layer_state_vectors, dtype=np.float16)
                if tensor.ndim != 3 or tensor.shape[0] != item.n_steps:
                    raise RuntimeError("geometry-only extraction produced an invalid layer-state tensor")
                if state_memmap is None:
                    if expected_state_capacity <= 0:
                        raise RuntimeError("cannot allocate a whole-layer state store with zero capacity")
                    state_memmap = np.lib.format.open_memmap(
                        state_partial_path,
                        mode="w+",
                        dtype=np.float16,
                        shape=(expected_state_capacity, tensor.shape[1], tensor.shape[2]),
                    )
                end = state_cursor + tensor.shape[0]
                if end > state_memmap.shape[0]:
                    raise RuntimeError("whole-layer state store capacity was underestimated")
                state_memmap[state_cursor:end] = tensor
                state_chain_idx.extend([int(item.record.chain_idx)] * tensor.shape[0])
                state_step_idx.extend(range(tensor.shape[0]))
                state_cursor = end
                item.step_layer_state_vectors = None
            except Exception:
                if state_memmap is not None:
                    state_memmap.flush()
                    state_memmap._mmap.close()
                    state_memmap = None
                state_partial_path.unlink(missing_ok=True)
                raise
        extractions.append(item)

    profiler.save_json(out_path.parent / "profile_summary.json")
    with (out_path.parent / "skip_report.json").open("w", encoding="utf-8") as f:
        json.dump(skip_rows, f, indent=2, ensure_ascii=False)
    if not extractions:
        if state_memmap is not None:
            state_memmap.flush()
            state_memmap._mmap.close()
            state_partial_path.unlink(missing_ok=True)
        raise SystemExit("No chains were successfully extracted.")
    chain_coverage = len(extractions) / max(len(records), 1)
    expected_problems = {int(record.problem_id) for record in records}
    covered_problems = {int(item.record.problem_id) for item in extractions}
    problem_coverage = len(covered_problems) / max(len(expected_problems), 1)
    if min(chain_coverage, problem_coverage) < required_coverage:
        if state_memmap is not None:
            state_memmap.flush()
            state_memmap._mmap.close()
            state_partial_path.unlink(missing_ok=True)
        raise SystemExit(
            f"Extraction coverage failed: chains={chain_coverage:.3f}, "
            f"problems={problem_coverage:.3f}, required={required_coverage:.3f}. "
            f"Diagnostics were saved under {out_path.parent}; no state artifact was written."
        )
    for item in extractions:
        item.metadata["extraction_chain_coverage"] = float(chain_coverage)
        item.metadata["extraction_problem_coverage"] = float(problem_coverage)
    if args.geometry_only:
        if state_memmap is None:
            raise RuntimeError("geometry-only extraction did not initialize its state store")
        state_memmap.flush()
        state_memmap._mmap.close()
        state_memmap = None
        packed = pack_extractions(extractions)
        packed["step_layer_state_memmap_path"] = np.asarray(state_final_path.name, dtype=object)
        packed["step_layer_state_memmap_count"] = np.asarray(state_cursor, dtype=np.int64)
        packed["step_layer_state_vector_chain_idx"] = np.asarray(state_chain_idx, dtype=np.int64)
        packed["step_layer_state_vector_step_idx"] = np.asarray(state_step_idx, dtype=np.int64)
        packed["step_layer_state_vector_layers"] = np.asarray(extractions[0].layers, dtype=np.int64)
        packed["state_storage_kind"] = np.asarray("npy_memmap_v1", dtype=object)
        np.savez_compressed(manifest_partial_path, **packed)
        state_partial_path.replace(state_final_path)
        manifest_partial_path.replace(out_path)
    else:
        save_extractions(extractions, args.output)
    print(f"Saved {len(extractions)} chain metrics to {args.output}")
    print(f"Saved profile to {out_path.parent / 'profile_summary.json'}")


if __name__ == "__main__":
    main()
