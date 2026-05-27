"""Step 1: Extract the per-trajectory (step × layer) spectral field on ProcessBench.

For each reasoning chain we:
  1. Tokenize prompt + response and align each pre-parsed step with its token range.
  2. Run one forward pass with output_hidden_states=True to get hidden states at
     all L+1 layers (embedding + L transformer blocks).
  3. For every (step j, layer l), collect the token cloud H_j^(l) ∈ R^{n_j × d}
     and reduce it to three scalars: effective rank D, spectral energy V,
     and top concentration C. Stack across steps and layers to get three
     (T, L+1) matrices M_D, M_V, M_C — this is what downstream analysis consumes.

The single behavioural difference from prior demo versions:
  *all* tokens in a step are kept (the cloud H_j^(l) is what carries the
  spectral signal); prior versions kept only the last token of each step.

Usage:
    python 01_extract_spectral_field.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset Qwen/ProcessBench \
        --subset gsm8k \
        --n_correct 50 \
        --n_error 50 \
        --output data/spectral_field.npz
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from utils import find_step_token_ranges, step_layer_spectral_summary


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_processbench_subset(dataset_name, subset, n_correct, n_error, seed=42):
    """Stratified sample of ProcessBench by label sign (-1 = all correct)."""
    print(f"Loading {dataset_name} split={subset} ...")
    ds = load_dataset(dataset_name, split=subset)

    correct = [ex for ex in ds if ex.get("label", -1) == -1]
    error = [ex for ex in ds if ex.get("label", -1) >= 0]
    print(f"  dataset: {len(correct)} correct, {len(error)} error")

    rng = np.random.default_rng(seed)
    n_c = min(n_correct, len(correct))
    n_e = min(n_error, len(error))
    correct_idx = rng.choice(len(correct), size=n_c, replace=False)
    error_idx = rng.choice(len(error), size=n_e, replace=False)
    print(f"  sampled: {n_c} correct, {n_e} error")
    return [correct[i] for i in correct_idx], [error[i] for i in error_idx]


def build_prompt_and_response(example):
    """ProcessBench → (prompt, response, steps)."""
    problem = example["problem"]
    steps = example.get("steps", [])
    if not steps:
        return None, None, None
    response = "\n\n".join(steps)
    prompt = f"Problem: {problem}\n\nSolution:\n\n"
    return prompt, response, steps


# ---------------------------------------------------------------------------
# Spectral field for one trajectory
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_spectral_field(
    model, tokenizer, prompt, response, steps,
    device, layer_indices=None, max_seq_len=4096,
):
    """Run one forward pass and reduce each (step, layer) token cloud to (D, V, C).

    Returns:
        M_D, M_V, M_C: (T, L_sub) float arrays where L_sub = len(layer_indices).
                       Rows in original step order; NaN rows are dropped.
        kept_steps:    indices (in the original 0..T-1) of steps actually kept.
        layers_used:   list of int layer indices that were sampled.
    """
    ranges = find_step_token_ranges(tokenizer, prompt, response, steps)
    if len(ranges) < 3:
        return None, None, None, None, None

    encoding = tokenizer(
        prompt + response,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    ).to(device)
    seq_len = encoding["input_ids"].shape[1]

    # Keep only steps whose token range fits inside the truncated sequence.
    safe = [(j, a, b) for j, (a, b) in enumerate(ranges) if b < seq_len and b - a + 1 >= 2]
    if len(safe) < 3:
        return None, None, None, None, None

    outputs = model(**encoding, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, d) tensors
    n_layers_total = len(hidden_states)

    if layer_indices is None:
        layer_indices = list(range(n_layers_total))
    layer_indices = [l for l in layer_indices if 0 <= l < n_layers_total]
    L_sub = len(layer_indices)

    T_eff = len(safe)
    M_D = np.full((T_eff, L_sub), np.nan, dtype=np.float64)
    M_V = np.full((T_eff, L_sub), np.nan, dtype=np.float64)
    M_C = np.full((T_eff, L_sub), np.nan, dtype=np.float64)

    for li, l in enumerate(layer_indices):
        H_l = hidden_states[l][0].float().cpu().numpy()  # (seq_len, d)
        for row, (_, a, b) in enumerate(safe):
            H_jl = H_l[a : b + 1]  # (n_j, d)
            D, V, C = step_layer_spectral_summary(H_jl)
            M_D[row, li] = D
            M_V[row, li] = V
            M_C[row, li] = C

    kept_steps = np.array([j for j, _, _ in safe], dtype=np.int32)
    return M_D, M_V, M_C, kept_steps, layer_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", default="Qwen/ProcessBench")
    parser.add_argument("--subset", default="gsm8k",
                        choices=["gsm8k", "math", "olympiadbench", "omnimath"])
    parser.add_argument("--n_correct", type=int, default=50)
    parser.add_argument("--n_error", type=int, default=50)
    parser.add_argument("--layers", default="all",
                        help='"all" or a comma-separated list of layer indices, '
                             'e.g. "0,8,16,24,30,31". Index 0 = embedding output, '
                             '1..L = transformer block outputs.')
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--output", default="data/spectral_field.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else device,
    )
    model.eval()

    layer_indices = None if args.layers == "all" else \
        [int(x) for x in args.layers.split(",") if x.strip()]

    correct_examples, error_examples = load_processbench_subset(
        args.dataset, args.subset, args.n_correct, args.n_error, seed=args.seed
    )

    print(f"Extracting spectral fields (layers={args.layers}) ...")
    rows = []
    skipped = 0
    for ex_set, _tag in [(correct_examples, "correct"), (error_examples, "error")]:
        for ex in tqdm(ex_set, desc=_tag):
            prompt, response, steps = build_prompt_and_response(ex)
            if prompt is None:
                skipped += 1
                continue
            try:
                M_D, M_V, M_C, kept_steps, layers_used = extract_spectral_field(
                    model, tokenizer, prompt, response, steps,
                    device, layer_indices=layer_indices,
                    max_seq_len=args.max_seq_len,
                )
            except Exception as e:
                print(f"  warn: extraction failed: {e}")
                skipped += 1
                continue

            if M_D is None or M_D.shape[0] < 3:
                skipped += 1
                continue

            # Map original label (first-error step in ORIGINAL step indexing) to
            # the kept-step indexing so step-level evaluation is consistent.
            orig_label = int(ex.get("label", -1))
            if orig_label < 0:
                mapped_label = -1
            else:
                kept = kept_steps.tolist()
                mapped_label = kept.index(orig_label) if orig_label in kept else -2
                # -2 means "had an error but it was dropped by truncation"; skip
                if mapped_label == -2:
                    skipped += 1
                    continue

            rows.append({
                "id": str(ex.get("id", len(rows))),
                "label": mapped_label,
                "n_steps": int(M_D.shape[0]),
                "M_D": M_D.astype(np.float32),
                "M_V": M_V.astype(np.float32),
                "M_C": M_C.astype(np.float32),
                "kept_steps": kept_steps,
                "layers_used": np.asarray(layers_used, dtype=np.int32),
            })

    if not rows:
        print("ERROR: no valid trajectories.")
        return

    n_layers_sub = rows[0]["M_D"].shape[1]
    print(f"\nKept {len(rows)} trajectories ({skipped} skipped), "
          f"L_sub = {n_layers_sub} sampled layers.")

    np.savez(
        args.output,
        ids=np.array([r["id"] for r in rows], dtype=object),
        labels=np.array([r["label"] for r in rows], dtype=np.int32),
        n_steps=np.array([r["n_steps"] for r in rows], dtype=np.int32),
        M_D=np.array([r["M_D"] for r in rows], dtype=object),
        M_V=np.array([r["M_V"] for r in rows], dtype=object),
        M_C=np.array([r["M_C"] for r in rows], dtype=object),
        kept_steps=np.array([r["kept_steps"] for r in rows], dtype=object),
        layers_used=rows[0]["layers_used"],
        model_name=np.array(args.model),
        subset=np.array(args.subset),
    )
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
