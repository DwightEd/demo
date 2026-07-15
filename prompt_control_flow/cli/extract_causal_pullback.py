from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Sequence


DEFAULT_MODEL = "/gz-data/models/Meta-Llama-3.1-8B-Instruct"


def _identity_matches(source: str, requested: str) -> bool:
    source = str(source).strip().replace("\\", "/").rstrip("/").lower()
    requested = str(requested).strip().replace("\\", "/").rstrip("/").lower()
    if not source:
        return True
    return source == requested or source.rsplit("/", 1)[-1] == requested.rsplit("/", 1)[-1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay same-problem responses and extract a geometry-to-output causal "
            "pullback operator without saving full logits."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--vector_key",
        default="sv_vec_step_exp",
        help=(
            "Legacy step-vector key used only to align rows and semantic steps. "
            "Residual interventions always pool raw sv_clouds; projected sv_vec_* "
            "values are never injected into the model."
        ),
    )
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument(
        "--label_policy",
        default="answer_format_ok",
        choices=("answer", "strict", "answer_format_ok", "processbench"),
    )
    parser.add_argument(
        "--problem_format",
        default="auto",
        choices=("auto", "gsm8k", "processbench"),
        help="Legacy prompt source. auto reads the NPZ `dataset` provenance.",
    )
    parser.add_argument(
        "--problem_source",
        default="",
        help="Optional explicit HF/local source overriding NPZ provenance.",
    )
    parser.add_argument("--problem_subset", default="")
    parser.add_argument("--problem_split", default="test")
    parser.add_argument("--prompt_style", default="")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help=(
            "Maximum expensive replay targets. Same-problem donor supports are "
            "always built from the complete input artifact."
        ),
    )
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--min_donors", type=int, default=6)
    parser.add_argument("--max_donors", type=int, default=11)
    parser.add_argument("--epsilon_fraction", type=float, default=0.02)
    parser.add_argument("--variant_batch_size", type=int, default=8)
    parser.add_argument("--logit_token_chunk", type=int, default=16)
    parser.add_argument("--replay_cosine_threshold", type=float, default=0.98)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--min_success_fraction", type=float, default=0.80)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"),
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--allow_model_mismatch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not 0.0 <= args.min_success_fraction <= 1.0:
        raise SystemExit("--min_success_fraction must lie in [0, 1]")

    from prompt_control_flow.causal_pullback import (
        CausalPullbackConfig,
        load_ordered_problem_questions,
        load_pullback_source,
        resolve_problem_source_spec,
        run_causal_pullback_extraction,
        source_preflight,
        validate_problem_question_map,
    )
    from prompt_control_flow.causal_pullback.field import ConditionalFieldBank

    source = load_pullback_source(
        args.input,
        vector_key=args.vector_key,
        layer=args.layer,
        label_policy=args.label_policy,
        # The complete source is required to build same-problem donor supports.
        # ``--max_samples`` limits replay targets only.
        max_samples=0,
    )
    if args.prompt_style:
        source.prompt_style = str(args.prompt_style)
    cfg = CausalPullbackConfig(
        layer=args.layer,
        min_donors=args.min_donors,
        max_donors=args.max_donors,
        epsilon_fraction=args.epsilon_fraction,
        variant_batch_size=args.variant_batch_size,
        logit_token_chunk=args.logit_token_chunk,
        replay_cosine_threshold=args.replay_cosine_threshold,
    )
    cfg.validate()
    donor_bank = ConditionalFieldBank.build(source.dataset, cfg)
    eligible = donor_bank.eligible_target_indices()
    preflight = source_preflight(source)
    preflight.update(
        {
            "donor_eligible_samples": int(eligible.size),
            "donor_eligible_fraction": float(
                eligible.size / max(source.dataset.n_samples, 1)
            ),
            "requested_replay_targets": int(args.max_samples),
        }
    )
    print(json.dumps(preflight, indent=2, ensure_ascii=False))
    if not preflight["residual_intervention_ready"]:
        raise SystemExit(
            "Causal residual intervention requires raw sv_clouds pooled in the "
            "model hidden dimension; projected sv_vec_* states are not valid."
        )
    if eligible.size == 0:
        raise SystemExit(
            "No sample has enough same-problem correct donors under the selected "
            "label policy and --min_donors."
        )
    if source.model_name and not _identity_matches(source.model_name, args.model):
        if not args.allow_model_mismatch:
            raise SystemExit(
                f"Stored state model {source.model_name!r} does not match observer "
                f"{args.model!r}; pass --allow_model_mismatch only for a declared ablation."
            )

    ordered_questions = None
    problem_spec = None
    if preflight["legacy_problem_reconstruction_required"]:
        problem_spec = resolve_problem_source_spec(
            source.dataset_provenance,
            dataset_format=args.problem_format,
            path=args.problem_source,
            subset=args.problem_subset,
            split=args.problem_split,
        )
        ordered_questions = load_ordered_problem_questions(problem_spec)
        map_status = validate_problem_question_map(
            source, ordered_questions, problem_spec
        )
        print(
            json.dumps(
                {"problem_source": problem_spec.as_dict(), **map_status},
                indent=2,
                ensure_ascii=False,
            )
        )

    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model_hidden_dim = int(
        getattr(model_config, "hidden_size", getattr(model_config, "d_model", -1))
    )
    if model_hidden_dim <= 0:
        raise SystemExit("Observer model config does not expose hidden_size or d_model.")
    if int(source.dataset.hidden_dim) != model_hidden_dim:
        raise SystemExit(
            "Raw stored hidden dimension does not match observer residual width: "
            f"stored={source.dataset.hidden_dim}, model={model_hidden_dim}. "
            "Check --model, --layer, cloud_layers, and extraction provenance."
        )
    print(
        json.dumps(
            {
                "observer_model": args.model,
                "observer_hidden_dim": model_hidden_dim,
                "stored_hidden_dim": int(source.dataset.hidden_dim),
                "hidden_width_match": True,
            },
            indent=2,
        )
    )
    if args.preflight:
        return

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from prompt_control_flow.profiler import MechanismProfiler

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
    if dtype is None and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit("Causal replay requires a fast tokenizer for exact step alignment.")
    model_kwargs = {"trust_remote_code": bool(args.trust_remote_code)}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()

    profiler = MechanismProfiler()
    total = args.max_samples if args.max_samples > 0 else source.dataset.n_samples
    progress = tqdm(total=total, desc="causal pullback replay")

    def update(current: int, _total: int) -> None:
        if int(progress.total or 0) != int(_total):
            progress.total = int(_total)
        progress.n = int(current)
        progress.refresh()

    try:
        artifact = run_causal_pullback_extraction(
            model,
            tokenizer,
            source,
            cfg,
            output_path=args.output,
            observer_model=args.model,
            ordered_questions=ordered_questions,
            max_seq_len=args.max_seq_len,
            checkpoint_every=args.checkpoint_every,
            max_targets=args.max_samples,
            resume=args.resume,
            profiler=profiler,
            progress=update,
        )
    finally:
        progress.close()

    output = Path(args.output)
    profiler.save_json(output.parent / f"{output.stem}_profile.json")
    (output.parent / f"{output.stem}_skips.json").write_text(
        json.dumps(artifact.skipped, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    target_count = int(artifact.metadata.get("target_count", total))
    success = artifact.n_items / max(target_count, 1)
    print(
        f"saved {artifact.n_items}/{target_count} selected responses to {output} "
        f"(coverage={success:.3f})"
    )
    if artifact.skipped:
        counts = Counter(str(row.get("reason", "unknown")) for row in artifact.skipped)
        print(f"skip reasons: {dict(counts.most_common())}")
        for row in artifact.skipped[:3]:
            print(
                "  skip "
                f"original={row.get('original_index')} reason={row.get('reason')}: "
                f"{row.get('detail')}"
            )
    if success < args.min_success_fraction:
        raise SystemExit(
            f"Extraction coverage {success:.3f} is below required "
            f"{args.min_success_fraction:.3f}; inspect the skip report."
        )


if __name__ == "__main__":
    main()
