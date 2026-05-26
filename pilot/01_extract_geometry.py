"""
Q1 Pilot: Extract step-level hidden states and geometric metrics from ProcessBench.

Core hypothesis: correct reasoning trajectories evolve in a "constrained but
non-degenerate" subspace — neither diverging to high dimensions (losing structure)
nor collapsing to overly low dimensions (losing computational freedom).

For each ProcessBench example:
1. Build prompt from problem + steps
2. Forward pass, collect hidden states at each step's last token (multiple layers)
3. Compute per-step geometric metrics:
   - displacement:     ||h_j - h_{j-1}||        (step-to-step movement)
   - cosine_sim:       cos(h_j, h_{j-1})         (direction continuity)
   - norm:             ||h_j||                    (scale stability)
   - curvature:        angle between consecutive displacement vectors
   - cross-layer effective rank: SVD of [h_j^(l1), h_j^(l2), ...] (dimensionality)

Output: results/{split}_geometry.jsonl
"""

import argparse
import json
import os
import time
import gc
import math

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


# ──────────────────────────────────────────────
# 1. Build prompt and find step boundary positions
# ──────────────────────────────────────────────

def build_prompt_and_boundaries(problem: str, steps: list, tokenizer):
    """
    Build prompt and find the token position of each step's last token.
    Returns: input_ids (list[int]), step_end_positions (list[int])
    """
    prefix = f"Problem: {problem}\nSolution:"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)

    step_end_positions = []
    current_text = prefix

    for i, step in enumerate(steps):
        step_text = f"\nStep {i+1}: {step}"
        current_text += step_text
        new_ids = tokenizer.encode(current_text, add_special_tokens=True)
        step_end_positions.append(len(new_ids) - 1)

    full_ids = tokenizer.encode(current_text, add_special_tokens=True)
    return full_ids, step_end_positions


# ──────────────────────────────────────────────
# 2. Compute geometric metrics from hidden states
# ──────────────────────────────────────────────

def effective_rank(matrix):
    """
    Effective rank via Shannon entropy of normalized singular values.
    ER = exp(H), where H = -sum(p_i * ln(p_i)), p_i = sigma_i / sum(sigma)
    """
    S = torch.linalg.svdvals(matrix.float())
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    H = -(p * p.log()).sum()
    return H.exp().item()


def compute_step_geometry(hidden_states_per_layer, prev_hidden=None, prev_displacement=None):
    """
    Given hidden states at one step across multiple layers,
    compute geometric metrics.

    Args:
        hidden_states_per_layer: dict {layer_idx: tensor [hidden_dim]}
        prev_hidden: dict {layer_idx: tensor [hidden_dim]} from previous step (or None)
        prev_displacement: tensor [hidden_dim] displacement vector of previous step (or None)

    Returns: dict of metrics, current displacement vector
    """
    # Use the last available layer as the "primary" hidden state
    layer_indices = sorted(hidden_states_per_layer.keys())
    last_layer = layer_indices[-1]
    h_j = hidden_states_per_layer[last_layer].float()

    metrics = {}

    # --- Norm (scale stability) ---
    metrics["norm"] = torch.norm(h_j).item()

    # --- Displacement & cosine similarity (vs previous step) ---
    displacement = None
    if prev_hidden is not None and last_layer in prev_hidden:
        h_prev = prev_hidden[last_layer].float()
        diff = h_j - h_prev
        displacement = diff

        metrics["displacement"] = torch.norm(diff).item()
        cos_sim = torch.nn.functional.cosine_similarity(
            h_j.unsqueeze(0), h_prev.unsqueeze(0)
        ).item()
        metrics["cosine_sim"] = cos_sim

        # Normalized displacement (displacement / norm, scale-invariant)
        metrics["displacement_normed"] = metrics["displacement"] / (metrics["norm"] + 1e-8)
    else:
        metrics["displacement"] = None
        metrics["cosine_sim"] = None
        metrics["displacement_normed"] = None

    # --- Curvature (angle between consecutive displacement vectors) ---
    if displacement is not None and prev_displacement is not None:
        cos_angle = torch.nn.functional.cosine_similarity(
            displacement.unsqueeze(0), prev_displacement.unsqueeze(0)
        ).item()
        # Clamp for numerical stability
        cos_angle = max(-1.0, min(1.0, cos_angle))
        metrics["curvature"] = math.acos(cos_angle)  # radians, 0=straight, pi=reversal
    else:
        metrics["curvature"] = None

    # --- Cross-layer effective rank ---
    # Stack hidden states from sampled layers into a matrix [n_layers, hidden_dim]
    if len(layer_indices) >= 3:
        layer_matrix = torch.stack([
            hidden_states_per_layer[l].float() for l in layer_indices
        ])  # [n_layers, hidden_dim]
        metrics["effective_rank"] = effective_rank(layer_matrix)
    else:
        metrics["effective_rank"] = None

    return metrics, displacement


# ──────────────────────────────────────────────
# 3. Main extraction loop
# ──────────────────────────────────────────────

def process_example(example, model, tokenizer, target_layers, max_seq_len=2048):
    """Process one ProcessBench example, return per-step geometric metrics."""
    problem = example["problem"]
    steps = example["steps"]
    label = example["label"]  # -1 = all correct, else 0-indexed first error step

    # Build prompt
    input_ids, step_end_positions = build_prompt_and_boundaries(problem, steps, tokenizer)

    # Truncate if too long
    if len(input_ids) > max_seq_len:
        valid_steps = [i for i, pos in enumerate(step_end_positions) if pos < max_seq_len]
        if len(valid_steps) < 2:
            return None
        steps = steps[:len(valid_steps)]
        step_end_positions = step_end_positions[:len(valid_steps)]
        input_ids = input_ids[:max_seq_len]

    input_tensor = torch.tensor([input_ids], device=model.device)

    # Forward pass with hidden states output
    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=True, use_cache=False)

    # outputs.hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, hidden_dim]
    # Index 0 = embedding, index i = after layer i-1
    all_hidden = outputs.hidden_states

    # Collect per-step metrics
    step_metrics = []
    prev_hidden = None
    prev_displacement = None

    for j, pos in enumerate(step_end_positions):
        if pos >= len(input_ids):
            break

        # Collect hidden states at this position for target layers
        h_at_layers = {}
        for l in target_layers:
            layer_hs_idx = l + 1  # hidden_states[0] is embedding, [1] is after layer 0
            if layer_hs_idx < len(all_hidden):
                h_at_layers[l] = all_hidden[layer_hs_idx][0, pos, :].cpu()

        # Compute geometric metrics
        metrics, displacement = compute_step_geometry(h_at_layers, prev_hidden, prev_displacement)

        # Step metadata
        metrics["step_idx"] = j
        metrics["token_pos"] = pos

        # Step label
        if label == -1:
            metrics["is_error"] = 0
            metrics["is_first_error"] = 0
        else:
            metrics["is_error"] = 1 if j >= label else 0
            metrics["is_first_error"] = 1 if j == label else 0

        step_metrics.append(metrics)
        prev_hidden = h_at_layers
        prev_displacement = displacement

    return step_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--splits", type=str, default="gsm8k")
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--n_sample_layers", type=int, default=8,
                        help="Number of evenly-spaced layers to sample for cross-layer metrics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    # Load model
    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"Model loaded. Layers: {num_layers}, Hidden: {hidden_dim}")

    # Select target layers: evenly spaced + last layer
    n = args.n_sample_layers
    target_layers = sorted(set(
        [int(i * (num_layers - 1) / (n - 1)) for i in range(n)]
        + [num_layers - 1]
    ))
    print(f"Target layers ({len(target_layers)}): {target_layers}")

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

        out_path = os.path.join(args.output_dir, f"{split}_geometry.jsonl")
        t0 = time.time()
        n_steps = 0
        n_skipped = 0

        with open(out_path, "w", encoding="utf-8") as fout:
            for idx, ex in enumerate(examples):
                try:
                    step_metrics = process_example(
                        ex, model, tokenizer, target_layers,
                        max_seq_len=args.max_seq_len
                    )
                except Exception as e:
                    print(f"  [{idx}] ERROR: {e}")
                    n_skipped += 1
                    continue

                if step_metrics is None:
                    n_skipped += 1
                    continue

                record = {
                    "id": ex["id"],
                    "label": ex["label"],
                    "num_steps": len(ex["steps"]),
                    "step_metrics": step_metrics,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_steps += len(step_metrics)

                if (idx + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{idx+1}/{len(examples)}] {(idx+1)/elapsed:.1f} ex/s, "
                          f"{n_steps} steps, {n_skipped} skipped")

                if (idx + 1) % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"\nDone {split}: {len(examples)-n_skipped} ok, "
              f"{n_steps} steps in {elapsed:.1f}s → {out_path}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
