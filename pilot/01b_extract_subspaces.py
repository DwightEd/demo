"""
Extract step-level subspaces from LLM hidden states.

For each reasoning step j, computes SVD of token matrix H_j in R^{n_j x d},
saves top-k right singular vectors V_j in R^{d x k} and singular values sigma_j in R^k.

V_j defines a point on the Grassmannian Gr(k, d).
sigma_j encodes the spectral shape (constraint structure).

Output: results/{split}_subspaces.pt

Usage (on GPU server):
    python pilot/01b_extract_subspaces.py \
        --model_path /gz-data/models/LLM-Research/Meta-Llama-3___1-8B-Instruct \
        --data_dir data/processbench \
        --splits gsm8k \
        --k 8
"""

import argparse
import json
import os
import time
import gc

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_prompt_and_ranges(problem, steps, tokenizer):
    """Build prompt and find token ranges per step (same as 01_extract_geometry.py)."""
    prefix = f"Problem: {problem}\nSolution:"
    step_ranges = []
    current_text = prefix
    prev_len = len(tokenizer.encode(current_text, add_special_tokens=True))

    for i, step in enumerate(steps):
        step_text = f"\nStep {i+1}: {step}"
        current_text += step_text
        new_len = len(tokenizer.encode(current_text, add_special_tokens=True))
        step_ranges.append((prev_len, new_len - 1))
        prev_len = new_len

    full_ids = tokenizer.encode(current_text, add_special_tokens=True)
    return full_ids, step_ranges


def process_example(example, model, tokenizer, target_layer, k=8, max_seq_len=2048):
    """Extract top-k subspace per step.

    Returns list of dicts with V (d, k), sigma (k,), metadata.
    Returns None if example is unusable.
    """
    problem = example["problem"]
    steps = example["steps"]
    label = example["label"]

    input_ids, step_ranges = build_prompt_and_ranges(problem, steps, tokenizer)

    # Truncate if needed
    if len(input_ids) > max_seq_len:
        valid_steps = [i for i, (s, e) in enumerate(step_ranges) if e < max_seq_len]
        if len(valid_steps) < 2:
            return None
        steps = steps[:len(valid_steps)]
        step_ranges = step_ranges[:len(valid_steps)]
        input_ids = input_ids[:max_seq_len]

    input_tensor = torch.tensor([input_ids], device=model.device)

    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    layer_idx = target_layer + 1  # index 0 = embedding layer
    if layer_idx >= len(hidden_states):
        layer_idx = len(hidden_states) - 1

    h = hidden_states[layer_idx][0]  # (seq_len, d)

    results = []
    for j, (tok_start, tok_end) in enumerate(step_ranges):
        if tok_end >= len(input_ids):
            tok_end = len(input_ids) - 1
        if tok_start > tok_end:
            break

        H_j = h[tok_start:tok_end + 1].float()  # (n_j, d)
        n_j = H_j.shape[0]

        if n_j < k:
            continue  # not enough tokens for k-dim subspace

        # SVD: H_j = U @ diag(S) @ Vh
        # Right singular vectors: V = Vh[:k].T -> (d, k)
        _, S, Vh = torch.linalg.svd(H_j, full_matrices=False)
        V_j = Vh[:k].T.contiguous()  # (d, k)
        sigma_j = S[:k].contiguous()  # (k,)

        is_error = 0
        is_first_error = 0
        if label != -1:
            if j >= label:
                is_error = 1
            if j == label:
                is_first_error = 1

        results.append({
            "V": V_j.cpu().half(),        # float16 to save space
            "sigma": sigma_j.cpu().float(),
            "is_error": is_error,
            "is_first_error": is_first_error,
            "step_idx": j,
            "n_tokens": n_j,
        })

    if len(results) < 2:
        return None
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract step-level subspaces")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processbench")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--splits", type=str, default="gsm8k")
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--k", type=int, default=8, help="Subspace dimension")
    parser.add_argument("--target_layer", type=int, default=-1,
                        help="Layer index (-1 = last layer)")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    print(f"Loading model from {args.model_path} ...")
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} not found")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    target_layer = args.target_layer if args.target_layer >= 0 else num_layers - 1

    print(f"Model: {num_layers} layers, hidden_dim={hidden_dim}")
    print(f"Target layer: {target_layer}, k={args.k}")

    for split in args.splits.split(","):
        split = split.strip()
        data_path = os.path.join(args.data_dir, f"{split}.jsonl")
        if not os.path.exists(data_path):
            print(f"[WARN] {data_path} not found, skipping")
            continue

        examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))

        if args.max_examples > 0:
            examples = examples[:args.max_examples]

        print(f"\n{'='*60}")
        print(f"Processing {split}: {len(examples)} examples")
        print(f"{'='*60}")

        all_data = []
        n_skipped = 0
        t0 = time.time()

        for idx, ex in enumerate(examples):
            try:
                result = process_example(
                    ex, model, tokenizer, target_layer,
                    k=args.k, max_seq_len=args.max_seq_len
                )
            except Exception as e:
                print(f"  [{idx}] ERROR: {e}")
                n_skipped += 1
                continue

            if result is None:
                n_skipped += 1
                continue

            all_data.append({
                "id": ex["id"],
                "label": ex["label"],
                "steps": result,
            })

            if (idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  [{idx+1}/{len(examples)}] {(idx+1)/elapsed:.1f} ex/s, "
                      f"{len(all_data)} ok, {n_skipped} skipped")

            if (idx + 1) % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        out_path = os.path.join(args.output_dir, f"{split}_subspaces.pt")
        torch.save(all_data, out_path)

        elapsed = time.time() - t0
        n_steps = sum(len(d["steps"]) for d in all_data)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"\nDone {split}: {len(all_data)} examples ({n_skipped} skipped), "
              f"{n_steps} steps in {elapsed:.1f}s")
        print(f"  Saved to {out_path} ({size_mb:.1f} MB)")

    print("\nAll done!")


if __name__ == "__main__":
    main()
