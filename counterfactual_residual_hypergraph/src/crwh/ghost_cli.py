from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .ghost import select_mid_depths
from .ghost_eval import EvaluationConfig, evaluate_ghost
from .ghost_hf import (
    extract_processbench_embeddings,
    load_embedding_artifact,
    save_embedding_artifact,
    tokenize_chat_record,
)


@dataclass(frozen=True)
class _Candidate:
    trace_id: str
    problem_id: str
    response_label: int
    token_count: int
    record: object


def _stable_rank(candidate: _Candidate, *, seed: int) -> bytes:
    return hashlib.sha256(
        f"{seed}\0{candidate.problem_id}\0{candidate.trace_id}".encode("utf-8")
    ).digest()


def _json_record(record: object) -> dict[str, object]:
    return {
        "id": record.trace_id,
        "problem_id": record.problem_id,
        "problem": record.question,
        "steps": list(record.steps),
        "label": int(record.labels.first_error),
        "generator": record.generator_model,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def select_processbench_cohort(
    *,
    source: str | Path,
    output: str | Path,
    manifest_path: str | Path,
    tokenizer,
    mode: str,
    limit: int,
    max_tokens: int,
    seed: int,
) -> dict[str, object]:
    from hypergraph.attention.cct.processbench import ProcessBenchReader

    source_path = Path(source)
    output_path = Path(output)
    manifest_destination = Path(manifest_path)
    if mode not in {"balanced_unique", "all_eligible"}:
        raise ValueError("selection mode must be balanced_unique or all_eligible")
    if max_tokens < 1 or seed < 0 or limit < 0:
        raise ValueError("selection limit, max_tokens, and seed are invalid")
    candidates = []
    excluded = []
    source_records = 0
    for record in ProcessBenchReader(source_path).records():
        source_records += 1
        tokenized = tokenize_chat_record(tokenizer, record)
        token_count = len(tokenized.input_ids)
        if token_count > max_tokens:
            excluded.append(
                {
                    "trace_id": record.trace_id,
                    "problem_id": record.problem_id,
                    "token_count": token_count,
                    "reason": "exceeds_max_tokens_without_truncation",
                }
            )
            continue
        candidates.append(
            _Candidate(
                trace_id=record.trace_id,
                problem_id=record.problem_id,
                response_label=int(record.labels.first_error >= 0),
                token_count=token_count,
                record=record,
            )
        )
    if mode == "balanced_unique":
        from crwh.cohort import select_balanced_unique

        if limit < 2:
            raise ValueError("balanced selection requires a positive even limit")
        selected = list(select_balanced_unique(candidates, limit=limit, seed=seed))
        selection_policy = "balanced_response_label_problem_unique_sha256_rank"
        label_use = "labels used to balance the evaluation cohort"
    else:
        selected = sorted(
            candidates,
            key=lambda candidate: _stable_rank(candidate, seed=seed),
        )
        if limit:
            if limit > len(selected):
                raise ValueError(
                    f"requested {limit} eligible traces but only {len(selected)} exist"
                )
            selected = selected[:limit]
        selection_policy = "all_eligible_sha256_order"
        label_use = "labels not used for cohort inclusion"
    if not selected:
        raise ValueError("selection produced no eligible ProcessBench traces")
    if len({item.trace_id for item in selected}) != len(selected):
        raise ValueError("selected ProcessBench trace IDs are not unique")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or manifest_destination.exists():
        raise FileExistsError("refusing to overwrite a cohort artifact")
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in selected:
            stream.write(
                json.dumps(_json_record(item.record), ensure_ascii=False) + "\n"
            )
    class_counts = Counter(item.response_label for item in selected)
    generators = Counter(item.record.generator_model for item in selected)
    manifest = {
        "schema_version": "ghost_cohort_v1",
        "source": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_records": source_records,
        "eligible_records": len(candidates),
        "excluded_too_long": len(excluded),
        "excluded": excluded,
        "selection_mode": mode,
        "selection_policy": selection_policy,
        "selection_seed": seed,
        "selection_label_use": label_use,
        "max_tokens": max_tokens,
        "truncation_policy": "forbidden",
        "prompt_format": "observer_chat_template",
        "step_rendering": "strip_outer_whitespace_then_join_with_two_newlines",
        "selected_records": len(selected),
        "selected_unique_problem_ids": len(
            {item.problem_id for item in selected}
        ),
        "response_class_counts": {
            str(label): int(class_counts.get(label, 0)) for label in (0, 1)
        },
        "generator_counts": dict(sorted(generators.items())),
        "trace_ids": [item.trace_id for item in selected],
        "problem_ids": [item.problem_id for item in selected],
        "output": str(output_path.resolve()),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    _write_json(manifest_destination, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m crwh.ghost_cli",
        description="GHOST-style normal-reference hidden-state geometry",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select")
    select.add_argument("--input", required=True)
    select.add_argument("--model", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--manifest", required=True)
    select.add_argument(
        "--mode",
        choices=("balanced_unique", "all_eligible"),
        default="balanced_unique",
    )
    select.add_argument("--limit", type=int, default=48)
    select.add_argument("--max-tokens", type=int, default=768)
    select.add_argument("--seed", type=int, default=17)

    extract = commands.add_parser("extract")
    extract.add_argument("--input", required=True)
    extract.add_argument("--model", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--cohort-manifest")
    extract.add_argument("--mid-depths", default="auto")
    extract.add_argument("--mid-start", type=float, default=0.25)
    extract.add_argument("--mid-end", type=float, default=0.50)
    extract.add_argument("--probe-count", type=int, default=6)
    extract.add_argument("--max-tokens", type=int, default=768)
    extract.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--attention-implementation", default="sdpa")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--embeddings", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--split-seed", type=int, default=17)
    evaluate.add_argument("--validation-ratio", type=float, default=0.2)
    evaluate.add_argument("--test-ratio", type=float, default=0.2)
    evaluate.add_argument("--threshold-quantile", type=float, default=0.95)
    evaluate.add_argument("--shrinkage", type=float, default=0.1)
    evaluate.add_argument("--regularization", type=float, default=1e-8)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=1000)
    evaluate.add_argument("--bootstrap-confidence", type=float, default=0.95)
    return parser


def _run_select(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        local_files_only=True,
    )
    manifest = select_processbench_cohort(
        source=args.input,
        output=args.output,
        manifest_path=args.manifest,
        tokenizer=tokenizer,
        mode=args.mode,
        limit=args.limit,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _parse_mid_depths(
    raw: str,
    *,
    num_hidden_layers: int,
    start: float,
    stop: float,
    count: int,
) -> tuple[int, ...]:
    if raw == "auto":
        return select_mid_depths(
            num_hidden_layers=num_hidden_layers,
            start_fraction=start,
            end_fraction=stop,
            count=count,
        )
    try:
        depths = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise ValueError("--mid-depths must be auto or comma-separated integers") from error
    if (
        len(depths) != count
        or tuple(sorted(set(depths))) != depths
        or depths[0] < 1
        or depths[-1] >= num_hidden_layers
    ):
        raise ValueError("explicit mid depths violate the fixed probe contract")
    return depths


def _run_extract(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from hypergraph.attention.cct.processbench import ProcessBenchReader

    output = Path(args.output)
    manifest_path = Path(args.manifest)
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite extraction artifacts")
    source = Path(args.input)
    records = tuple(ProcessBenchReader(source).records())
    cohort_contract = {}
    if args.cohort_manifest:
        cohort_path = Path(args.cohort_manifest)
        cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
        if cohort.get("output_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError("selected input does not match the cohort manifest hash")
        if cohort.get("trace_ids") != [record.trace_id for record in records]:
            raise ValueError("selected input trace order disagrees with cohort manifest")
        cohort_contract = {
            "manifest": str(cohort_path.resolve()),
            "manifest_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
            "selection_mode": cohort.get("selection_mode"),
            "selection_policy": cohort.get("selection_policy"),
            "selection_label_use": cohort.get("selection_label_use"),
            "response_class_counts": cohort.get("response_class_counts"),
        }

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        local_files_only=True,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = AutoModel.from_pretrained(
        args.model,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        attn_implementation=args.attention_implementation,
    )
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    num_hidden_layers = int(model.config.num_hidden_layers)
    mid_depths = _parse_mid_depths(
        args.mid_depths,
        num_hidden_layers=num_hidden_layers,
        start=args.mid_start,
        stop=args.mid_end,
        count=args.probe_count,
    )
    traces = extract_processbench_embeddings(
        records,
        model=model,
        tokenizer=tokenizer,
        mid_depths=mid_depths,
        final_depth=num_hidden_layers,
        max_tokens=args.max_tokens,
    )
    generators = Counter(trace.generator_model for trace in traces)
    metadata = {
        "schema_version": "ghost_extraction_metadata_v1",
        "method": "GHOST-inspired abstract-level mid-layer Mahalanobis",
        "observer_model": str(Path(args.model).resolve()),
        "observer_model_type": str(model.config.model_type),
        "observer_num_hidden_layers": num_hidden_layers,
        "observer_hidden_size": int(model.config.hidden_size),
        "observer_dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "device": str(device),
        "inference_passes_per_trace": 1,
        "generation_performed": False,
        "prompt_format": "observer_chat_template",
        "step_rendering": "strip_outer_whitespace_then_join_with_two_newlines",
        "pooling_primary": "response_mean",
        "pooling_sensitivity": "response_last",
        "pooling_accumulation_dtype": "float32",
        "mid_depth_rule": {
            "start_fraction": args.mid_start,
            "end_fraction": args.mid_end,
            "count": args.probe_count,
        },
        "mid_depths": list(mid_depths),
        "mid_block_indices": [depth - 1 for depth in mid_depths],
        "hf_hidden_state_indices": list(mid_depths),
        "final_depth": num_hidden_layers,
        "final_representation": "normalized_AutoModel_last_hidden_state",
        "max_tokens": args.max_tokens,
        "truncation_policy": "forbidden",
        "input": str(source.resolve()),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "traces": len(traces),
        "generator_counts": dict(sorted(generators.items())),
        "cohort_selection": cohort_contract,
        "scientific_boundary": (
            "Responses were generated by the ProcessBench generator models and "
            "observed under teacher forcing by the declared observer model."
        ),
    }
    save_embedding_artifact(output, traces, metadata=metadata)
    extraction_manifest = {
        **metadata,
        "embedding_artifact": str(output.resolve()),
        "embedding_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    _write_json(manifest_path, extraction_manifest)
    print(json.dumps(extraction_manifest, indent=2, ensure_ascii=False))
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    traces, metadata = load_embedding_artifact(args.embeddings)
    summary = evaluate_ghost(
        traces,
        output_dir=args.output_dir,
        config=EvaluationConfig(
            split_seed=args.split_seed,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            threshold_quantile=args.threshold_quantile,
            shrinkage=args.shrinkage,
            regularization=args.regularization,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_confidence=args.bootstrap_confidence,
        ),
        embedding_metadata=metadata,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select":
        return _run_select(args)
    if args.command == "extract":
        return _run_extract(args)
    if args.command == "evaluate":
        return _run_evaluate(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
