"""
Extract multi-layer step-level spectral features from LLM hidden states.

For each reasoning step j, at EACH layer l, computes SVD of token matrix H_{j,l},
saves top-k singular values sigma_{j,l}. This gives a (L, k) spectral profile per step.

No subspace V saved (Grassmannian direction confirmed noise in ablation).

Output: results/{split}_multilayer.pt

Usage (on GPU server):
    python pilot/01c_extract_multilayer.py \
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
    """Build prompt and find token ranges per step."""
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


def process_example(example, model, tokenizer, num_layers, k=8, max_seq_len=2048,
                    layer_indices=None):
    """Extract multi-layer spectral profile per step.

    Returns list of dicts with sigma_ml (L, k) per step.
    Returns None if example is unusable.
    """
    problem = example["problem"]
    steps = example["steps"]
    label = example["label"]

    input_ids, step_ranges = build_prompt_and_ranges(problem, steps, tokenizer)

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

    hidden_states = outputs.hidden_states  # (num_layers+1,) each (1, seq_len, d)

    if layer_indices is None:
        layer_indices = list(range(num_layers))

    results = []
    for j, (tok_start, tok_end) in enumerate(step_ranges):
        if tok_end >= len(input_ids):
            tok_end = len(input_ids) - 1
        if tok_start > tok_end:
            break

        n_j = tok_end - tok_start + 1
        if n_j < k:
            continue

        sigma_layers = []
        for l in layer_indices:
            h_l = hidden_states[l + 1][0]  # +1 to skip embedding layer
            H_j_l = h_l[tok_start:tok_end + 1].float()  # (n_j, d)
            _, S, _ = torch.linalg.svd(H_j_l, full_matrices=False)
            sigma_layers.append(S[:k].cpu())

        sigma_ml = torch.stack(sigma_layers)  # (L_selected, k)

        is_error = 0
        is_first_error = 0
        if label != -1:
            if j >= label:
                is_error = 1
            if j == label:
                is_first_error = 1

        results.append({
            "sigma_ml": sigma_ml.float(),  # (L_selected, k)
            "is_error": is_error,
            "is_first_error": is_first_error,
            "step_idx": j,
            "n_tokens": n_j,
        })

    if len(results) < 2:
        return None
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract multi-layer spectral features")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processbench")
    parser.add_argument("--output_dir", type=str, default="pilot/results")
    parser.add_argument("--splits", type=str, default="gsm8k")
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--k", type=int, default=8, help="Number of singular values per layer")
    parser.add_argument("--layers", type=str, default="all",
                        help="Layer indices: 'all' or comma-separated (e.g., '8,16,24,31')")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    if args.layers == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]

    print(f"Model: {num_layers} layers, hidden_dim={hidden_dim}")
    print(f"Extracting layers: {layer_indices} ({len(layer_indices)} layers), k={args.k}")
    print(f"Per-step feature dim: {len(layer_indices)} x {args.k} = {len(layer_indices) * args.k}")

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
                    ex, model, tokenizer, num_layers,
                    k=args.k, max_seq_len=args.max_seq_len,
                    layer_indices=layer_indices,
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

        # Save with metadata
        out_path = os.path.join(args.output_dir, f"{split}_multilayer.pt")
        save_data = {
            "examples": all_data,
            "meta": {
                "k": args.k,
                "layer_indices": layer_indices,
                "num_layers": num_layers,
                "hidden_dim": hidden_dim,
            }
        }
        torch.save(save_data, out_path)

        elapsed = time.time() - t0
        n_steps = sum(len(d["steps"]) for d in all_data)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"\nDone {split}: {len(all_data)} examples ({n_skipped} skipped), "
              f"{n_steps} steps in {elapsed:.1f}s")
        print(f"  Saved to {out_path} ({size_mb:.1f} MB)")

    print("\nAll done!")


if __name__ == "__main__":
    main()
