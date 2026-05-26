"""Step 1: Extract step-level hidden states from a model on ProcessBench traces.

ProcessBench provides pre-generated reasoning traces with step-level error labels.
We tokenize each trace, run a single forward pass, and extract hidden states at
the last token of each reasoning step.

Usage:
    python 01_extract_hidden_states.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset Qwen/ProcessBench \
        --subset gsm8k \
        --n_correct 50 \
        --n_error 50 \
        --layer -1 \
        --output data/hidden_states.npz

Note: requires GPU for reasonable speed. Each trace is a single forward pass
(no generation needed since ProcessBench traces are pre-generated).
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from utils import find_step_token_indices


def load_processbench_subset(dataset_name, subset, n_correct, n_error, seed=42):
    """Load a stratified sample from ProcessBench.

    ProcessBench schema (per Qwen/ProcessBench):
        - problem: str
        - steps: list[str]  (parsed reasoning steps)
        - label: int  (first error step index 0-based; -1 if all correct)

    Returns:
        correct_examples, error_examples: lists of dicts
    """
    print(f"Loading {dataset_name}/{subset} ...")
    ds = load_dataset(dataset_name, subset, split="test")

    correct = [ex for ex in ds if ex.get("label", -1) == -1]
    error = [ex for ex in ds if ex.get("label", -1) >= 0]
    print(f"  -> Dataset: {len(correct)} correct, {len(error)} error total")

    rng = np.random.default_rng(seed)
    n_c = min(n_correct, len(correct))
    n_e = min(n_error, len(error))
    correct_idx = rng.choice(len(correct), size=n_c, replace=False)
    error_idx = rng.choice(len(error), size=n_e, replace=False)

    correct_sample = [correct[i] for i in correct_idx]
    error_sample = [error[i] for i in error_idx]
    print(f"  -> Sampled: {len(correct_sample)} correct, {len(error_sample)} error")
    return correct_sample, error_sample


def build_prompt_and_response(example):
    """Construct prompt + response text from a ProcessBench example."""
    problem = example["problem"]
    steps = example.get("steps", [])
    if not steps:
        return None, None, None
    response = "\n\n".join(steps)
    prompt = f"Problem: {problem}\n\nSolution:\n\n"
    return prompt, response, steps


@torch.no_grad()
def extract_one_trajectory(model, tokenizer, prompt, response, steps, layer, device,
                           max_seq_len=4096):
    """Run forward pass and extract step-level hidden states.

    Args:
        layer: -1 for last layer, 0 for embedding, 1..L for transformer blocks
        max_seq_len: truncate if sequence exceeds this length

    Returns:
        traj: (T, d) numpy array, or None if extraction fails
    """
    full_text = prompt + response
    step_token_indices = find_step_token_indices(tokenizer, prompt, response, steps)

    if len(step_token_indices) == 0:
        return None

    encoding = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    ).to(device)

    seq_len = encoding["input_ids"].shape[1]

    # Filter step indices that fall within truncated sequence
    safe_indices = [i for i in step_token_indices if i < seq_len]
    if len(safe_indices) < 3:
        return None

    outputs = model(**encoding, output_hidden_states=True)
    h = outputs.hidden_states[layer][0]  # (seq_len, d)

    traj = h[safe_indices].float().cpu().numpy()  # (T, d)
    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", default="Qwen/ProcessBench")
    parser.add_argument("--subset", default="gsm8k",
                        choices=["gsm8k", "math", "olympiadbench", "omnimath"])
    parser.add_argument("--n_correct", type=int, default=50)
    parser.add_argument("--n_error", type=int, default=50)
    parser.add_argument("--layer", type=int, default=-1,
                        help="Which layer's hidden state (-1=last, 0=embed, 1..L=blocks)")
    parser.add_argument("--max_seq_len", type=int, default=4096,
                        help="Max sequence length (truncate longer traces)")
    parser.add_argument("--output", default="data/hidden_states.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load model
    print(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto" if device == "cuda" else device
    )
    model.eval()

    # Load data
    correct_examples, error_examples = load_processbench_subset(
        args.dataset, args.subset, args.n_correct, args.n_error, seed=args.seed
    )

    # Extract trajectories
    trajectories = []
    skipped = 0
    print(f"Extracting hidden states at layer {args.layer} ...")

    for example_set, set_label in [(correct_examples, "correct"),
                                    (error_examples, "error")]:
        for ex in tqdm(example_set, desc=set_label):
            prompt, response, steps = build_prompt_and_response(ex)
            if prompt is None:
                skipped += 1
                continue
            try:
                traj = extract_one_trajectory(
                    model, tokenizer, prompt, response, steps,
                    args.layer, device, max_seq_len=args.max_seq_len,
                )
            except Exception as e:
                print(f"  Warning: extraction failed for example: {e}")
                skipped += 1
                continue

            if traj is None or traj.shape[0] < 3:
                skipped += 1
                continue

            trajectories.append({
                "id": ex.get("id", str(len(trajectories))),
                "label": ex.get("label", -1),
                "n_steps": traj.shape[0],
                "traj": traj.astype(np.float32),
            })

    if len(trajectories) == 0:
        print("ERROR: No valid trajectories extracted. Check model/data compatibility.")
        return

    # Save
    print(f"Saving {len(trajectories)} trajectories ({skipped} skipped) -> {args.output}")
    ids = np.array([t["id"] for t in trajectories], dtype=object)
    labels = np.array([t["label"] for t in trajectories], dtype=np.int32)
    n_steps_arr = np.array([t["n_steps"] for t in trajectories], dtype=np.int32)
    trajs = np.array([t["traj"] for t in trajectories], dtype=object)
    np.savez(
        args.output,
        ids=ids,
        labels=labels,
        n_steps=n_steps_arr,
        trajs=trajs,
        model_name=np.array(args.model),
        layer=np.array(args.layer),
        subset=np.array(args.subset),
    )
    print("Done.")


if __name__ == "__main__":
    main()
