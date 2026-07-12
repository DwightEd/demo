"""Step 10 (Phase 2): same-problem multi-sampling + step-vector extraction.

Why this exists
---------------
On ProcessBench every problem has ONE solution, and correct vs error chains come
from DIFFERENT problems -- so "error" is collinear with "problem difficulty".
That is why the prefix-vs-full result (08 section f) cannot tell apart:
    (a) error chains are just harder problems (difficulty signal), vs
    (b) the model enters a diffuse/high-participation regime BEFORE it errs
        (a genuine early-failure-prediction signal).

This script removes the confound at the source: for the SAME problem we sample
K solutions, label each by FINAL-ANSWER correctness against the gold answer
(deterministic, no LLM judge), and extract per-(step,layer) activation
participation for every sampled solution. The within-problem comparison
(11_within_problem_analysis.py) then asks: holding the problem fixed, do the
FAILING samples have higher participation than the SUCCEEDING ones? That is
difficulty-controlled by construction.

Pipeline
--------
  for each problem:
    1) generate K step-by-step solutions (chat template, temperature sampling)
    2) parse the final answer, compare to gold  -> is_correct in {0,1}
    3) split each solution into steps (sentence/line)
    4) teacher-forcing forward pass -> per-(step,layer) PR/AE of the step vector
       (REUSES 01_extract_spectral_field.extract_spectral_field verbatim, same
        reasoning subspace + step_exp weighting, so numbers are comparable to
        the ProcessBench results)

Output: data/<tag>_multisample_sv.npz with, per kept sample:
    problem_id, sample_idx, n_steps, pred/gold answer,
    sv_pr_<mode> / sv_ae_<mode> : (T, L) matrices, sv_out_entropy : (T,)
    layers_used, sv_modes, sv_stored=True
  Labels (BOTH stored so downstream can pick either policy):
    is_correct        : lenient -- last-number fallback counts (v1 behaviour)
    is_correct_strict : strict  -- requires '####' marker AND numeric match
    format_ok         : 1 iff response contains a '####' line
    pred_source       : 'marker' | 'last_number' | 'none'
  Text fields (NEW, needed for any text-side / event-level analysis):
    responses         : raw model generation per sample
    steps_text        : list of step strings used for token-range matching
    step_split        : 'line' (new default) | 'sentence' (v1 legacy)
  Exact trace fields (schema exact_generation_trace_v1):
    prompts / prompt_token_ids / prompt_attention_mask
    input_ids / attention_mask / token_offsets (+ untruncated full_* variants)
    question_*_span, response_*_range, kept_steps, step_token_ranges
    time_axis_original_step_indices / time_axis_token_ranges
    generated_token_ids (including separately recorded terminal EOS/pad IDs)

Judging:
  GSM8K answers are integers/decimals -> exact numeric match, no judge.
  The lenient path (current `is_correct`) treats the last number in the
  response as the answer when the model forgets the '####' line; this can
  silently misscore long chains that hit token budget mid-derivation. The
  strict path (`is_correct_strict`) only counts samples that emitted the
  marker, with `last-number` cases pushed into a format-fail bucket that
  the audit summary at the end of run prints separately. For analysis
  that wants to isolate the geometric failure signal from generation /
  format failure, use `is_correct_strict`.
  For MATH-style symbolic answers, plug a math_verify check into `answers_match`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from utils.step_boundaries import (
    TokenAlignmentError,
    build_exact_trace_alignment,
    trace_records_to_npz,
    trim_trailing_generation_tokens,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_local_module(filename, name):
    """Import a sibling script whose name starts with a digit (not importable)."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(SCRIPT_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the exact extraction (reasoning subspace + step_exp + PR/AE) from 01.
_ex = _load_local_module("01_extract_spectral_field.py", "extract01")


# ---------------------------------------------------------------------------
# Answer parsing / judging (GSM8K-style numeric gold; deterministic)
# ---------------------------------------------------------------------------

def _to_number(s: str):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    s = s.rstrip(".")
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def gold_answer(answer_field: str):
    """GSM8K gold answer is the number after the final '#### '."""
    m = re.search(r"####\s*(.+)", answer_field)
    return _to_number(m.group(1)) if m else _to_number(answer_field)


def predicted_answer(text: str):
    """Prefer the number after the last '####'; else the last number in text."""
    hits = list(re.finditer(r"####\s*([^\n]+)", text))
    if hits:
        v = _to_number(hits[-1].group(1))
        if v is not None:
            return v
    nums = re.findall(r"[-+]?\$?\d[\d,]*\.?\d*", text)
    return _to_number(nums[-1]) if nums else None


def predicted_answer_with_source(text: str):
    """Same as predicted_answer but also returns which parser path succeeded.

    Returns (value, source) where source in {"marker", "last_number", "none"}.
    - "marker"      : the number came from the '#### N' line the prompt asked for;
                      this is the FORMAT-COMPLIANT path and the answer is trustworthy.
    - "last_number" : the '####' line is missing or unparsable; we fell back to the
                      last number in the response. **This is a format failure** and
                      the prediction may be a mid-chain intermediate result, not the
                      final answer. Should be flagged for audit, not silently
                      treated as the model's answer.
    - "none"        : no number anywhere -> can't score.
    """
    hits = list(re.finditer(r"####\s*([^\n]+)", text))
    if hits:
        v = _to_number(hits[-1].group(1))
        if v is not None:
            return v, "marker"
    nums = re.findall(r"[-+]?\$?\d[\d,]*\.?\d*", text)
    if nums:
        v = _to_number(nums[-1])
        if v is not None:
            return v, "last_number"
    return None, "none"


def answers_match(pred, gold, tol: float = 1e-4) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= tol * max(1.0, abs(gold))


# ---------------------------------------------------------------------------
# Step segmentation for self-generated solutions (Streaming-HD: sentence = step)
# ---------------------------------------------------------------------------

def split_into_steps(text: str, granularity: str = "line"):
    """Split a generated solution into steps. Each step must be a verbatim
    substring of `text` so find_step_token_ranges can locate it.

    granularity:
        "line"     : one step per newline. Matches the prompt's instruction
                     ("one short step per line") and the Streaming-HD step
                     convention. **This is the new default** -- preferred because
                     sentence-based splitting over-segments within-step inferences
                     (e.g. "5*3=15. So we have 15." becomes two fake steps for
                     what is one reasoning action). For analyses that aggregate
                     per chain, this affects step-level statistics but the
                     within-problem chain-level numbers (probe/SPE/ensemble) are
                     mostly insensitive.
        "sentence" : v1 behaviour (line + sentence split). Kept for backward
                     compatibility -- pass --step_split sentence to reproduce
                     old npz step boundaries exactly.
    """
    text = text.strip()
    if granularity == "line":
        return [ln.strip() for ln in re.split(r"\n+", text)
                if len(ln.strip()) >= 2]
    # sentence (legacy)
    steps = []
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue
        # split into sentences, but do NOT break decimals like '3.5'
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z(\d#])", line):
            sent = sent.strip()
            if len(sent) >= 2:
                steps.append(sent)
    return steps


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

CUSTOM_ZEROSHOT_TEMPLATE = (
    "Solve the following grade-school math problem. Reason step by step, with "
    "one short step per line. Then end with a final line of exactly the form "
    "'#### <answer>' where <answer> is just the number.\n\n"
    "Problem: {q}"
)

# Wei et al. 2022 (Chain-of-Thought Prompting) GSM8K few-shot exemplars,
# rewritten so the final answer line is '#### N' (the GSM8K canonical format the
# rest of the pipeline parses). Question text verbatim from the original paper;
# CoT text condensed but logically identical. lm-evaluation-harness uses the
# same set (subset of 5 for the gsm8k task, all 8 for gsm8k-cot). With this
# fixed prompt set, repeat runs are reproducible and cross-paper comparable.
GSM8K_COT_EXEMPLARS = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove "
        "today. After they are done, there will be 21 trees. How many trees did the "
        "grove workers plant today?",
        "There are 15 trees originally. There were 21 trees after planting. So the "
        "workers planted 21 - 15 = 6 trees.\n#### 6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many "
        "cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
        "pieces do they have left in total?",
        "Originally, Leah had 32 chocolates and her sister had 42. Together they had "
        "32 + 42 = 74. After eating 35, they have 74 - 35 = 39 left.\n#### 39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
        "lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. After giving some to Denny he had 12. He "
        "gave Denny 20 - 12 = 8 lollipops.\n#### 8",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and "
        "dad. How many toys does he have now?",
        "Shawn started with 5 toys. He received 2 from mom and 2 from dad, that is "
        "2 + 2 = 4 more toys. Now he has 5 + 4 = 9 toys.\n#### 9",
    ),
    (
        "There were nine computers in the server room. Five more computers were "
        "installed each day, from monday to thursday. How many computers are now in "
        "the server room?",
        "There were originally 9 computers. Five computers were added each day for "
        "4 days, that is 5 * 4 = 20. Total computers now is 9 + 20 = 29.\n#### 29",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, "
        "he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday he had "
        "58 - 23 = 35. After losing 2 more on wednesday he had 35 - 2 = 33.\n#### 33",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does "
        "she have left?",
        "Olivia started with 23 dollars. Five bagels at 3 dollars each cost "
        "5 * 3 = 15 dollars. She has 23 - 15 = 8 dollars left.\n#### 8",
    ),
]


def build_chat_messages(style: str, question: str):
    """Return chat messages for apply_chat_template, depending on prompt style.

    custom_zeroshot : v1 single-user message with the bespoke 'one short step per
                      line + ####' instruction. Format failures are common because
                      the model has no examples to anchor the layout.
    lm_eval_5shot   : 5-shot CoT (Wei et al. 2022 / lm-evaluation-harness `gsm8k`
                      style) as a chat dialogue -- 5 (user question, assistant
                      CoT-then-'#### N') turns, then the target question as a
                      final user turn. Format-fail rate ~ 0-3 %.
    lm_eval_8shot   : same as 5shot but with all 8 Wei et al. exemplars
                      (matches lm-evaluation-harness `gsm8k-cot`).
    """
    if style == "custom_zeroshot":
        return [{"role": "user",
                 "content": CUSTOM_ZEROSHOT_TEMPLATE.format(q=question)}]
    if style in ("lm_eval_5shot", "lm_eval_8shot"):
        k = 5 if style == "lm_eval_5shot" else 8
        msgs = []
        for ex_q, ex_a in GSM8K_COT_EXEMPLARS[:k]:
            msgs.append({"role": "user",   "content": f"Question: {ex_q}"})
            msgs.append({"role": "assistant", "content": ex_a})
        msgs.append({"role": "user", "content": f"Question: {question}"})
        return msgs
    raise ValueError(f"unknown prompt_style: {style!r}")


def load_problems(args):
    """Return a list of (question, gold_number) pairs.

    gsm8k       : openai/gsm8k, gold = number after '#### ' in the answer field.
    processbench: LOCAL ProcessBench dir, split=subset. The `problem` is the
                  GSM8K question; gold is taken from an explicit gold field if
                  present, else derived from a CORRECT solution (label==-1 /
                  final_answer_correct) whose final answer is the gold by
                  definition. No external dataset / no LLM judge needed.
    """
    if args.dataset_format == "gsm8k":
        ds = load_dataset(args.dataset, args.subset, split=args.split)
        out = []
        for ex in ds:
            g = gold_answer(ex["answer"])
            if g is not None:
                out.append((ex["question"], g))
        return out

    ds = load_dataset(args.dataset, split=args.subset)   # ProcessBench local dir
    gold_fields = ["answer", "final_answer", "gt_answer", "ground_truth",
                   "gold_answer"]
    probs = {}
    for ex in ds:
        prob = ex.get("problem")
        if not prob:
            continue
        gold = None
        for f in gold_fields:
            if ex.get(f) is not None:
                gold = _to_number(str(ex[f]))
                if gold is not None:
                    break
        if gold is None:
            lab = int(ex.get("label", -1))
            fac = ex.get("final_answer_correct", None)
            if lab == -1 or fac is True:        # a correct solution -> gold
                gold = predicted_answer("\n".join(ex.get("steps", []) or []))
        if gold is not None and prob not in probs:
            probs[prob] = gold
    return list(probs.items())


def build_gen_inputs(tokenizer, question, device, prompt_style="custom_zeroshot"):
    """Tokenize the chat-formatted prompt according to `prompt_style`.

    Render the chat string first (tokenize=False) so apply_chat_template's
    version-dependent return type does not matter, then tokenize explicitly
    with the standard tokenizer call (which always returns a BatchEncoding
    whose ["input_ids"] is a (1, L) tensor). The template already adds
    special tokens, so add_special_tokens=False here.
    Return the exact rendered prompt together with input_ids and attention_mask
    (pad==eos otherwise warns and can give unreliable generation). The rendered
    string and these IDs are retained as the teacher-forcing source of truth.
    """
    messages = build_chat_messages(prompt_style, question)
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return text, {k: v.to(device) for k, v in enc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/gz-data/models/Meta-Llama-3.1-8B-Instruct",
                    help="Local model path on the GPU server; defaults to the canonical "
                         "Llama-3.1-8B-Instruct mirror on /gz-data. Override for remote "
                         "models (e.g. 'meta-llama/Llama-3.1-8B-Instruct' or Qwen / "
                         "DeepSeek-R1-Distill for cross-model robustness.")
    ap.add_argument("--dataset_format", default="processbench",
                    choices=["gsm8k", "processbench"],
                    help="gsm8k=openai/gsm8k (needs net); processbench=local "
                         "ProcessBench dir (problem + gold derived from correct "
                         "solutions).")
    ap.add_argument("--dataset", default="data/hf_datasets/ProcessBench",
                    help="HF name (gsm8k) or local path (processbench).")
    ap.add_argument("--subset", default="gsm8k",
                    help="processbench split / gsm8k config.")
    ap.add_argument("--split", default="test", help="only used for gsm8k format.")
    ap.add_argument("--n_problems", type=int, default=300)
    ap.add_argument("--k_samples", type=int, default=8,
                    help="solutions sampled per problem")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--min_steps", type=int, default=3)
    ap.add_argument("--step_split", default="line", choices=["line", "sentence"],
                    help="line=one step per newline (default; matches prompt 'one "
                         "short step per line'). sentence=v1 legacy split (newline "
                         "AND sentence-end). Switch to sentence to reproduce v1 npz.")
    ap.add_argument("--prompt_style", default="custom_zeroshot",
                    choices=["custom_zeroshot", "lm_eval_5shot", "lm_eval_8shot"],
                    help="custom_zeroshot=v1 bespoke prompt ('one short step per line "
                         "+ ####', no examples). lm_eval_5shot=5-shot CoT chat (Wei "
                         "et al. 2022 / lm-evaluation-harness 'gsm8k' style; format-"
                         "fail rate drops to ~0-3%%). lm_eval_8shot=8-shot CoT (lm-"
                         "evaluation-harness 'gsm8k-cot'). Use the latter two as a "
                         "robustness check on whether the v1 within-AUROC ~0.71 is an "
                         "artefact of the bespoke prompt.")
    # reasoning subspace (identical defaults to 01 so participation is comparable)
    ap.add_argument("--no_reasoning_subspace", action="store_true")
    ap.add_argument("--reasoning_mode", default="energy",
                    choices=["energy", "dim_ratio"])
    ap.add_argument("--reasoning_threshold", type=float, default=0.95)
    ap.add_argument("--unembedding_cache", default="data/unembedding_svd.npz")
    ap.add_argument("--sv_modes", default="last,mean,linear,step_exp")
    ap.add_argument("--whiten_baseline", default=None,
                    help="healthy_baseline.npz from build_healthy_baseline.py: "
                         "standardize each step vector per-dim vs correct reasoning "
                         "BEFORE participation (the anchor-faithful 'abnormal dims').")
    ap.add_argument("--store_vectors", action="store_true",
                    help="store raw step vectors (fp16) so participation can be "
                         "re-normalized (raw / healthy-standardized) in analysis "
                         "(11) WITHOUT re-sampling. Use --sv_modes step_exp to keep "
                         "storage small.")
    ap.add_argument("--store_prompt_hidden", action="store_true",
                    help="Store the exact rendered-prompt hidden span (fp16) "
                         "at --prompt_hidden_layers for semantic anchors.")
    ap.add_argument("--prompt_hidden_layers", default="16",
                    help="Comma-separated hidden-state indices stored for the "
                         "prompt span when --store_prompt_hidden is enabled.")
    ap.add_argument("--store_clouds", action="store_true",
                    help="ALSO store the raw per-step token clouds (n_j x d, fp16, "
                         "BEFORE reasoning-subspace projection) for the layers in "
                         "--cloud_layers. This preserves the within-step token "
                         "structure that step-vector pooling destroys (21 analyses "
                         "it). Storage ~ n_samples x response_tokens x d x 2 bytes per "
                         "cloud layer; keep --cloud_layers to 1-2 layers.")
    ap.add_argument("--cloud_layers", default="16",
                    help="comma list of layer indices to store token clouds for "
                         "(only with --store_clouds). Default a single mid layer.")
    ap.add_argument("--store_token_uncertainty", action="store_true",
                    help="capture per-generated-token uncertainty (entropy + committal "
                         "p(1-p)) from generation logits, for the uncertainty-trace-profile "
                         "analysis (33/34). Stores sv_tok_entropy / sv_tok_committal "
                         "(fp32, one variable-length array per kept chain).")
    ap.add_argument("--output", default="data/gsm8k_multisample_sv.npz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        device_map="auto" if device == "cuda" else device)
    model.eval()

    V_R = None
    if not args.no_reasoning_subspace:
        cache_path = args.unembedding_cache
        if cache_path:
            tag = os.path.basename(args.model.rstrip("/")).replace("/", "_")
            root, ext = os.path.splitext(cache_path)
            cache_path = f"{root}.{tag}{ext}"
        V_R, _ = _ex.prepare_reasoning_subspace(
            model, mode=args.reasoning_mode,
            threshold=args.reasoning_threshold, cache_path=cache_path)

    layer_indices = None if args.layers == "all" else \
        [int(x) for x in args.layers.split(",") if x.strip()]
    prompt_hidden_layers = tuple(
        int(x) for x in args.prompt_hidden_layers.split(",") if x.strip()
    ) if args.store_prompt_hidden else None
    sv_modes = tuple(args.sv_modes.split(","))
    cloud_layers = tuple(int(x) for x in args.cloud_layers.split(",") if x.strip()) \
        if args.store_clouds else None
    if args.store_clouds:
        print(f"  storing raw token clouds for layers {cloud_layers} (fp16, pre-projection)")

    # Optional healthy baseline -> per-layer (mu, sigma) for vector-level
    # standardization of step vectors before participation (the anchor).
    whiten = None
    if args.whiten_baseline:
        wb = np.load(args.whiten_baseline, allow_pickle=True)
        if bool(wb["reasoning_subspace_used"]) != (V_R is not None):
            raise SystemExit(
                "whiten_baseline subspace setting != extraction setting "
                f"(baseline projected={bool(wb['reasoning_subspace_used'])}, "
                f"eval projected={V_R is not None}). Rebuild with matching setting.")
        wlayers = wb["whiten_layers"].astype(int)
        wmu, wsg = wb["whiten_mu"].astype(np.float64), wb["whiten_sigma"].astype(np.float64)
        whiten = {int(l): (wmu[i], wsg[i]) for i, l in enumerate(wlayers)
                  if np.isfinite(wmu[i]).all()}
        print(f"  whitening ON: healthy baseline over {len(whiten)} layers "
              f"(d={wmu.shape[1]}) from {args.whiten_baseline}")

    print(f"Loading problems ({args.dataset_format}: {args.dataset} / {args.subset}) ...")
    problems = load_problems(args)
    if not problems:
        raise SystemExit("No problems with a usable gold answer were found.")
    n_prob = min(args.n_problems, len(problems))
    print(f"  {len(problems)} problems with gold; using {n_prob} "
          f"x K={args.k_samples} samples (T={args.temperature}, top_p={args.top_p})")

    rows = []
    n_gen = n_correct = 0
    # break down drop reasons so the audit can attribute lost samples
    n_dropped_text = n_dropped_extract = n_dropped_steps = 0
    # format / correctness audit counters (over ALL generated samples, before drop)
    n_marker = n_last_number = n_no_number = 0
    n_correct_strict_all = 0
    generation_eos_token_id = getattr(
        getattr(model, "generation_config", None),
        "eos_token_id",
        tokenizer.eos_token_id,
    )
    gen_kwargs = dict(
        do_sample=True, temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens, num_return_sequences=args.k_samples,
        pad_token_id=tokenizer.pad_token_id)

    for pi in tqdm(range(n_prob), desc="problems"):
        question, gold = problems[pi]
        if gold is None:
            continue

        rendered_prompt, gen_in = build_gen_inputs(
            tokenizer, question, device, prompt_style=args.prompt_style
        )
        plen = gen_in["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**gen_in, **gen_kwargs)
        sequences = out.sequences if hasattr(out, "sequences") else out
        expected_prefix = gen_in["input_ids"][0]
        if sequences.shape[1] < plen or not torch.equal(
            sequences[:, :plen], expected_prefix.unsqueeze(0).expand(sequences.shape[0], -1)
        ):
            raise TokenAlignmentError(
                f"generate() output for problem {pi} does not preserve its exact prompt prefix"
            )

        for si in range(sequences.shape[0]):
            n_gen += 1
            generated_tail = sequences[si, plen:].detach().cpu().tolist()
            response_ids, terminal_ids = trim_trailing_generation_tokens(
                generated_tail,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=generation_eos_token_id,
            )
            response = tokenizer.decode(
                response_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            # parse + record which path produced the prediction. "marker" = the
            # '#### N' line the prompt requested (trustworthy); "last_number" =
            # fallback to the last number in the response (format failure --
            # could be a mid-chain intermediate, NOT the final answer).
            pred, pred_src = predicted_answer_with_source(response)
            has_marker = (pred_src == "marker")
            correct = answers_match(pred, gold)              # lenient (= v1 behaviour)
            correct_strict = bool(correct and has_marker)    # strict: marker AND match
            n_correct += int(correct)
            n_correct_strict_all += int(correct_strict)
            if pred_src == "marker":
                n_marker += 1
            elif pred_src == "last_number":
                n_last_number += 1
            else:
                n_no_number += 1

            steps = split_into_steps(response, granularity=args.step_split)
            if len(steps) < args.min_steps:
                n_dropped_text += 1
                continue
            try:
                trace_alignment = build_exact_trace_alignment(
                    tokenizer,
                    rendered_prompt,
                    response,
                    steps,
                    prompt_token_ids=gen_in["input_ids"][0],
                    prompt_attention_mask=gen_in["attention_mask"][0],
                    response_token_ids=response_ids,
                    response_attention_mask=[1] * len(response_ids),
                    question_text=question,
                    fail_on_unmatched=True,
                )
                trace_alignment["generated_token_ids"] = list(generated_tail)
                trace_alignment["generation_terminal_token_ids"] = list(terminal_ids)
                M_D, _, _, kept_steps, layers_used, _, _, SV, TRACE = _ex.extract_spectral_field(
                    model, tokenizer, rendered_prompt, response, steps, device,
                    layer_indices=layer_indices, max_seq_len=args.max_seq_len,
                    V_R=V_R, step_vectors=True, sv_modes=sv_modes, whiten=whiten,
                    store_vectors=args.store_vectors,
                    store_clouds=args.store_clouds, cloud_layer_indices=cloud_layers,
                    token_uncertainty=args.store_token_uncertainty,
                    trace_alignment=trace_alignment,
                    question_text=question,
                    return_trace=True,
                    store_prompt_hidden=args.store_prompt_hidden,
                    prompt_hidden_layer_indices=prompt_hidden_layers)
            except TokenAlignmentError:
                # A prompt/response mismatch would invalidate every downstream
                # hidden trace. Do not silently count it as a dropped sample.
                raise
            except Exception as e:
                print(f"  warn: extraction failed (p{pi} s{si}): {e}")
                n_dropped_extract += 1
                continue
            if M_D is None or M_D.shape[0] < args.min_steps or SV is None:
                n_dropped_steps += 1
                continue

            rows.append({
                "id": f"p{pi}_s{si}",
                "problem_id": int(pi),
                "sample_idx": int(si),
                "is_correct": int(correct),                    # lenient (v1 compat)
                "is_correct_strict": int(correct_strict),      # marker + match
                "format_ok": int(has_marker),
                "pred_source": pred_src,                       # marker/last_number/none
                "n_steps": int(M_D.shape[0]),
                "pred": pred if pred is not None else float("nan"),
                "gold": gold,
                "layers_used": np.asarray(layers_used, dtype=np.int32),
                "response": response,                          # exact decoded generation
                "steps_text": list(steps),                     # NEW: parsed steps
                "kept_steps": np.asarray(kept_steps, dtype=np.int32),
                "tok_entropy": (SV.get("tok_entropy") if SV else None),
                "tok_committal": (SV.get("tok_committal") if SV else None),
                "SV": SV,
                "TRACE": TRACE,
            })

    if not rows:
        print("ERROR: no usable samples. Check generation / step splitting.")
        return

    modes = rows[0]["SV"]["modes"]
    save = dict(
        ids=np.array([r["id"] for r in rows], dtype=object),
        problem_ids=np.array([r["problem_id"] for r in rows], dtype=np.int32),
        sample_idx=np.array([r["sample_idx"] for r in rows], dtype=np.int32),
        is_correct=np.array([r["is_correct"] for r in rows], dtype=np.int32),
        # NEW: strict label (also requires '####' format) + format-fail flag +
        # parser path. Lets downstream scripts rerun with either label policy.
        is_correct_strict=np.array([r["is_correct_strict"] for r in rows], dtype=np.int32),
        format_ok=np.array([r["format_ok"] for r in rows], dtype=np.int32),
        pred_source=np.array([r["pred_source"] for r in rows], dtype=object),
        n_steps=np.array([r["n_steps"] for r in rows], dtype=np.int32),
        pred_answers=np.array([r["pred"] for r in rows], dtype=np.float64),
        gold_answers=np.array([r["gold"] for r in rows], dtype=np.float64),
        layers_used=rows[0]["layers_used"],
        sv_stored=np.array(True),
        sv_modes=np.array(modes, dtype=object),
        # NEW: raw generated text + parsed step strings. Required for any
        # text-side analysis (event-level semantics, audit, score-vs-text
        # backtracking). Costs ~3-5 MB / 2000 samples, negligible vs SV blobs.
        responses=np.array([r["response"] for r in rows], dtype=object),
        steps_text=np.array([r["steps_text"] for r in rows], dtype=object),
        step_split=np.array(args.step_split),
        prompt_style=np.array(args.prompt_style),
        model_name=np.array(args.model),
        dataset=np.array(f"{args.dataset_format}:{args.dataset}/{args.subset}"),
        whitened=np.array(whiten is not None),
        whiten_baseline=np.array(args.whiten_baseline or ""),
    )
    save.update(trace_records_to_npz([r["TRACE"] for r in rows]))
    model_sampling_meta = {
        "model_name": args.model,
        "model_type": str(getattr(model.config, "model_type", "")),
        "model_revision": str(getattr(model.config, "_commit_hash", "")),
        "tokenizer_name": str(getattr(tokenizer, "name_or_path", args.model)),
        "tokenizer_revision": str(getattr(tokenizer, "init_kwargs", {}).get("revision", "")),
        "torch_dtype": str(dtype),
        "device": str(device),
        "source_mode": "sampled_generation_then_exact_teacher_forcing",
        "do_sample": True,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_new_tokens": int(args.max_new_tokens),
        "num_return_sequences": int(args.k_samples),
        "seed": int(args.seed),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": generation_eos_token_id,
        "max_seq_len": int(args.max_seq_len),
        "prompt_style": args.prompt_style,
        "add_special_tokens": False,
    }
    save["model_sampling_metadata_json"] = np.array(
        json.dumps(model_sampling_meta, sort_keys=True, ensure_ascii=False)
    )
    save["sampling_seed"] = np.array(args.seed, dtype=np.int64)
    save["sampling_temperature"] = np.array(args.temperature, dtype=np.float64)
    save["sampling_top_p"] = np.array(args.top_p, dtype=np.float64)
    save["sampling_max_new_tokens"] = np.array(args.max_new_tokens, dtype=np.int32)
    for m in modes:
        save[f"sv_pr_{m}"] = np.array(
            [r["SV"]["pr"][m] for r in rows], dtype=object)
        save[f"sv_ae_{m}"] = np.array(
            [r["SV"]["ae"][m] for r in rows], dtype=object)
    save["sv_out_entropy"] = np.array(
        [r["SV"]["out_entropy"] for r in rows], dtype=object)
    save["sv_out_committal"] = np.array(
        [r["SV"]["out_committal"] for r in rows], dtype=object)
    if rows[0]["SV"].get("tok_entropy") is not None:
        save["sv_tok_entropy"] = np.array(
            [r["SV"].get("tok_entropy") for r in rows], dtype=object)
        save["sv_tok_committal"] = np.array(
            [r["SV"].get("tok_committal") for r in rows], dtype=object)
    if args.store_vectors and rows[0]["SV"].get("vec") is not None:
        save["sv_vectors_stored"] = np.array(True)
        for m in modes:
            save[f"sv_vec_{m}"] = np.array(
                [r["SV"]["vec"][m] for r in rows], dtype=object)
    else:
        save["sv_vectors_stored"] = np.array(False)

    # raw token clouds (fp16): per solution (n_tok, L_cloud, d) + per-step token sizes
    if args.store_clouds and rows[0]["SV"].get("clouds") is not None:
        save["clouds_stored"] = np.array(True)
        save["cloud_layers"] = rows[0]["SV"]["clouds"]["layers"]
        save["sv_clouds"] = np.array(
            [r["SV"]["clouds"]["clouds"] if r["SV"].get("clouds") is not None else None
             for r in rows], dtype=object)
        save["cloud_sizes"] = np.array(
            [r["SV"]["clouds"]["sizes"] if r["SV"].get("clouds") is not None else None
             for r in rows], dtype=object)
    else:
        save["clouds_stored"] = np.array(False)

    # per-token uncertainty traces (entropy + committal), for 33/34 profile analysis
    if args.store_token_uncertainty and rows[0].get("tok_entropy") is not None:
        save["tok_uncertainty_stored"] = np.array(True)
        save["sv_tok_entropy"] = np.array([r["tok_entropy"] for r in rows], dtype=object)
        save["sv_tok_committal"] = np.array([r["tok_committal"] for r in rows], dtype=object)
    else:
        save["tok_uncertainty_stored"] = np.array(False)

    np.savez(args.output, **save)

    n_prob_kept = len(set(r["problem_id"] for r in rows))
    # problems usable for the within-problem test under each labeling policy
    by_prob_len = {}; by_prob_str = {}
    for r in rows:
        by_prob_len.setdefault(r["problem_id"], []).append(r["is_correct"])
        by_prob_str.setdefault(r["problem_id"], []).append(r["is_correct_strict"])
    n_contrastive_lenient = sum(1 for v in by_prob_len.values()
                                if any(v) and not all(v))
    n_contrastive_strict = sum(1 for v in by_prob_str.values()
                               if any(v) and not all(v))
    # format-fail mass = lenient - strict on the SAME sample set: how many
    # samples were marked correct only because the last-number fallback hit gold
    n_lenient_kept = sum(r["is_correct"] for r in rows)
    n_strict_kept = sum(r["is_correct_strict"] for r in rows)
    n_dropped = n_dropped_text + n_dropped_extract + n_dropped_steps

    print(f"\n=== generation / labeling audit ===")
    print(f"Prompt style: {args.prompt_style}; step split: {args.step_split}")
    print(f"Generated {n_gen} samples.")
    print(f"  Format ok ('####' present):  {n_marker}/{n_gen} = "
          f"{n_marker / max(1, n_gen):.3f}")
    print(f"  Format fail (last-number):   {n_last_number}/{n_gen} = "
          f"{n_last_number / max(1, n_gen):.3f}  <- silently scored against gold")
    print(f"  No number at all:            {n_no_number}/{n_gen}")
    print(f"  Lenient correctness (= v1):  {n_correct}/{n_gen} = "
          f"{n_correct / max(1, n_gen):.3f}")
    print(f"  Strict  correctness (=marker+match): {n_correct_strict_all}/{n_gen} = "
          f"{n_correct_strict_all / max(1, n_gen):.3f}")
    print(f"  Gap (lenient - strict): "
          f"{(n_correct - n_correct_strict_all) / max(1, n_gen):.3f}  "
          f"<- mass that was 'correct' only via last-number fallback")
    print(f"\n=== drop reasons ===")
    print(f"  text split <{args.min_steps} steps : {n_dropped_text}")
    print(f"  extraction exception              : {n_dropped_extract}")
    print(f"  extracted <{args.min_steps} steps after extraction : {n_dropped_steps}")
    print(f"  total dropped                     : {n_dropped}")
    print(f"\nKept {len(rows)} samples across {n_prob_kept} problems.")
    print(f"  Kept lenient correct: {n_lenient_kept}; strict correct: {n_strict_kept}")
    print(f"  Contrastive problems (BOTH classes), lenient labels: "
          f"{n_contrastive_lenient}")
    print(f"  Contrastive problems (BOTH classes), strict labels:  "
          f"{n_contrastive_strict}  <- USE THIS for the audit-grade within-problem test")
    print(f"Saved -> {args.output}")
    if min(n_contrastive_lenient, n_contrastive_strict) < 20:
        print("  WARNING: few contrastive problems. Raise --k_samples or "
              "--n_problems, or --temperature for more diversity.")


if __name__ == "__main__":
    main()
