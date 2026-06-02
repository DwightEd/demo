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
    problem_id, sample_idx, is_correct, n_steps, pred/gold answer,
    sv_pr_<mode> / sv_ae_<mode> : (T, L) matrices, sv_out_entropy : (T,)
    layers_used, sv_modes, sv_stored=True

Judging: GSM8K answers are integers/decimals -> exact numeric match, no judge.
For MATH-style symbolic answers, plug a math_verify check into `answers_match`.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


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


def answers_match(pred, gold, tol: float = 1e-4) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= tol * max(1.0, abs(gold))


# ---------------------------------------------------------------------------
# Step segmentation for self-generated solutions (Streaming-HD: sentence = step)
# ---------------------------------------------------------------------------

def split_into_steps(text: str):
    """Split a generated solution into steps. Each step must be a verbatim
    substring of `text` so find_step_token_ranges can locate it."""
    text = text.strip()
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

PROMPT_TEMPLATE = (
    "Solve the following grade-school math problem. Reason step by step, with "
    "one short step per line. Then end with a final line of exactly the form "
    "'#### <answer>' where <answer> is just the number.\n\n"
    "Problem: {q}"
)


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


def build_gen_inputs(tokenizer, question, device):
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(q=question)}]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt")
    if isinstance(enc, dict):
        enc = enc["input_ids"]
    return enc.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
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
    # reasoning subspace (identical defaults to 01 so participation is comparable)
    ap.add_argument("--no_reasoning_subspace", action="store_true")
    ap.add_argument("--reasoning_mode", default="energy",
                    choices=["energy", "dim_ratio"])
    ap.add_argument("--reasoning_threshold", type=float, default=0.95)
    ap.add_argument("--unembedding_cache", default="data/unembedding_svd.npz")
    ap.add_argument("--sv_modes", default="last,mean,linear,step_exp")
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
    sv_modes = tuple(args.sv_modes.split(","))

    print(f"Loading problems ({args.dataset_format}: {args.dataset} / {args.subset}) ...")
    problems = load_problems(args)
    if not problems:
        raise SystemExit("No problems with a usable gold answer were found.")
    n_prob = min(args.n_problems, len(problems))
    print(f"  {len(problems)} problems with gold; using {n_prob} "
          f"x K={args.k_samples} samples (T={args.temperature}, top_p={args.top_p})")

    rows = []
    n_gen = n_correct = n_dropped = 0
    gen_kwargs = dict(
        do_sample=True, temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens, num_return_sequences=args.k_samples,
        pad_token_id=tokenizer.pad_token_id)

    for pi in tqdm(range(n_prob), desc="problems"):
        question, gold = problems[pi]
        if gold is None:
            continue

        gen_in = build_gen_inputs(tokenizer, question, device)
        with torch.no_grad():
            out = model.generate(gen_in, **gen_kwargs)
        gen_only = out[:, gen_in.shape[1]:]
        texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        # plain extraction prompt, same style as 01 (teacher-forcing context)
        extract_prompt = f"Problem: {question}\n\nSolution:\n\n"

        for si, response in enumerate(texts):
            n_gen += 1
            response = response.strip()
            pred = predicted_answer(response)
            correct = answers_match(pred, gold)
            n_correct += int(correct)

            steps = split_into_steps(response)
            if len(steps) < args.min_steps:
                n_dropped += 1
                continue
            try:
                M_D, _, _, kept_steps, layers_used, _, _, SV = _ex.extract_spectral_field(
                    model, tokenizer, extract_prompt, response, steps, device,
                    layer_indices=layer_indices, max_seq_len=args.max_seq_len,
                    V_R=V_R, step_vectors=True, sv_modes=sv_modes)
            except Exception as e:
                print(f"  warn: extraction failed (p{pi} s{si}): {e}")
                n_dropped += 1
                continue
            if M_D is None or M_D.shape[0] < args.min_steps or SV is None:
                n_dropped += 1
                continue

            rows.append({
                "id": f"p{pi}_s{si}",
                "problem_id": int(pi),
                "sample_idx": int(si),
                "is_correct": int(correct),
                "n_steps": int(M_D.shape[0]),
                "pred": pred if pred is not None else float("nan"),
                "gold": gold,
                "layers_used": np.asarray(layers_used, dtype=np.int32),
                "SV": SV,
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
        n_steps=np.array([r["n_steps"] for r in rows], dtype=np.int32),
        pred_answers=np.array([r["pred"] for r in rows], dtype=np.float64),
        gold_answers=np.array([r["gold"] for r in rows], dtype=np.float64),
        layers_used=rows[0]["layers_used"],
        sv_stored=np.array(True),
        sv_modes=np.array(modes, dtype=object),
        model_name=np.array(args.model),
        dataset=np.array(f"{args.dataset_format}:{args.dataset}/{args.subset}"),
    )
    for m in modes:
        save[f"sv_pr_{m}"] = np.array(
            [r["SV"]["pr"][m] for r in rows], dtype=object)
        save[f"sv_ae_{m}"] = np.array(
            [r["SV"]["ae"][m] for r in rows], dtype=object)
    save["sv_out_entropy"] = np.array(
        [r["SV"]["out_entropy"] for r in rows], dtype=object)

    np.savez(args.output, **save)

    n_prob_kept = len(set(r["problem_id"] for r in rows))
    # problems usable for the within-problem test need both classes present
    by_prob = {}
    for r in rows:
        by_prob.setdefault(r["problem_id"], []).append(r["is_correct"])
    n_contrastive = sum(1 for v in by_prob.values()
                        if any(v) and not all(v))
    print(f"\nGenerated {n_gen} samples; final-answer accuracy = "
          f"{n_correct}/{n_gen} = {n_correct / max(1, n_gen):.3f}")
    print(f"Kept {len(rows)} samples ({n_dropped} dropped: <{args.min_steps} steps "
          f"or extraction fail) across {n_prob_kept} problems.")
    print(f"Problems with BOTH a correct and an incorrect sample "
          f"(usable for within-problem test): {n_contrastive}")
    print(f"Saved -> {args.output}")
    if n_contrastive < 20:
        print("  WARNING: few contrastive problems. Raise --k_samples or "
              "--n_problems, or --temperature for more diversity.")


if __name__ == "__main__":
    main()
