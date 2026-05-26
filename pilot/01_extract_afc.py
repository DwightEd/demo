"""
Q1 Pilot: Extract AFC (Attention-FFN Coherence) metrics from ProcessBench.

For each example in ProcessBench:
1. Build prompt from problem + steps
2. Forward pass through LLM, hook Attention & FFN outputs at each layer
3. At each step's last token, collect Attention contribution A_j and FFN contribution F_j
4. Compute 3 AFC candidate metrics: cosine, vocab-JSD, projection-ratio
5. Also collect raw hidden states h_j for CTC baseline (conditional surprise)

Output: results/{split}_afc.jsonl  — one line per example with per-step metrics.
"""

import argparse
import json
import os
import time
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


# ──────────────────────────────────────────────
# 1. Hook machinery: capture Attn and FFN outputs
# ──────────────────────────────────────────────

class ActivationCapture:
    """Register hooks on every decoder layer to capture attn / ffn outputs."""

    def __init__(self, model):
        self.attn_outputs = {}   # layer_idx -> tensor [seq_len, hidden_dim]
        self.ffn_outputs = {}
        self.hooks = []
        self._register(model)

    def _register(self, model):
        # Llama-style: model.model.layers[i].self_attn / mlp
        layers = model.model.layers
        for i, layer in enumerate(layers):
            # Hook after self_attn (captures attn output BEFORE residual add)
            h = layer.self_attn.register_forward_hook(self._make_hook(self.attn_outputs, i))
            self.hooks.append(h)
            # Hook after mlp (captures ffn output BEFORE residual add)
            h = layer.mlp.register_forward_hook(self._make_hook(self.ffn_outputs, i))
            self.hooks.append(h)

    @staticmethod
    def _make_hook(storage, layer_idx):
        def hook_fn(module, input, output):
            # self_attn returns (attn_output, attn_weights, past_kv) or just attn_output
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            # Store detached on CPU to save GPU memory
            storage[layer_idx] = out.detach().cpu()
        return hook_fn

    def clear(self):
        self.attn_outputs.clear()
        self.ffn_outputs.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ──────────────────────────────────────────────
# 2. Build prompt and find step boundary positions
# ──────────────────────────────────────────────

def build_prompt_and_boundaries(problem: str, steps: list[str], tokenizer):
    """
    Build a prompt like:
        Problem: {problem}
        Solution:
        Step 1: {step1}
        Step 2: {step2}
        ...
    Returns: input_ids (1D tensor), step_end_positions (list of token indices)
    """
    parts = [f"Problem: {problem}\nSolution:"]
    for i, step in enumerate(steps):
        parts.append(f"\nStep {i+1}: {step}")

    # Tokenize incrementally to find step boundaries
    prefix = f"Problem: {problem}\nSolution:"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)

    step_end_positions = []
    current_text = prefix
    current_len = len(prefix_ids)

    for i, step in enumerate(steps):
        step_text = f"\nStep {i+1}: {step}"
        current_text += step_text
        new_ids = tokenizer.encode(current_text, add_special_tokens=True)
        step_end_pos = len(new_ids) - 1  # last token of this step
        step_end_positions.append(step_end_pos)
        current_len = len(new_ids)

    full_ids = tokenizer.encode(current_text, add_special_tokens=True)
    return full_ids, step_end_positions


# ──────────────────────────────────────────────
# 3. Compute AFC metrics
# ──────────────────────────────────────────────

def compute_afc_metrics(A_j: torch.Tensor, F_j: torch.Tensor,
                        unembed_weight: torch.Tensor = None,
                        temperature: float = 1.0):
    """
    Given step-level attention contribution A_j and FFN contribution F_j (both [hidden_dim]),
    compute three AFC candidate metrics.

    Returns dict with keys: afc_cos, afc_vocab_jsd, afc_proj
    """
    A = A_j.float()
    F_vec = F_j.float()

    # --- Candidate 1: Cosine similarity ---
    cos_sim = F.cosine_similarity(A.unsqueeze(0), F_vec.unsqueeze(0)).item()

    # --- Candidate 2: Vocab-space JSD ---
    afc_vocab_jsd = None
    if unembed_weight is not None:
        W = unembed_weight.float()  # [vocab, hidden]
        logits_A = (W @ A) / temperature
        logits_F = (W @ F_vec) / temperature
        p_A = torch.softmax(logits_A, dim=0)
        p_F = torch.softmax(logits_F, dim=0)
        # JSD = 0.5 * KL(p||m) + 0.5 * KL(q||m), m = 0.5*(p+q)
        m = 0.5 * (p_A + p_F)
        jsd = 0.5 * (F.kl_div(m.log(), p_A, reduction='sum', log_target=False)
                      + F.kl_div(m.log(), p_F, reduction='sum', log_target=False))
        afc_vocab_jsd = 1.0 - 0.5 * jsd.item()  # AFC = 1 - 0.5*JSD

    # --- Candidate 3: Projection ratio ---
    A_norm = torch.norm(A)
    if A_norm > 1e-8:
        F_parallel = (torch.dot(F_vec, A) / (A_norm ** 2)) * A
        proj_ratio = torch.norm(F_parallel).item() / (torch.norm(F_vec).item() + 1e-8)
    else:
        proj_ratio = 0.0

    return {
        "afc_cos": cos_sim,
        "afc_vocab_jsd": afc_vocab_jsd,
        "afc_proj": proj_ratio,
    }


# ──────────────────────────────────────────────
# 4. Main extraction loop
# ──────────────────────────────────────────────

def process_example(example, model, tokenizer, capture, unembed_weight,
                    target_layer_frac=(0.7, 1.0), max_seq_len=2048):
    """Process one ProcessBench example, return per-step metrics."""
    problem = example["problem"]
    steps = example["steps"]
    label = example["label"]  # -1 = all correct, else 0-indexed first error step
    num_layers = len(model.model.layers)
    layer_lo = int(target_layer_frac[0] * num_layers)
    layer_hi = num_layers

    # Build prompt
    input_ids, step_end_positions = build_prompt_and_boundaries(problem, steps, tokenizer)

    # Truncate if too long
    if len(input_ids) > max_seq_len:
        # Find which steps fit
        valid_steps = [i for i, pos in enumerate(step_end_positions) if pos < max_seq_len]
        if len(valid_steps) < 2:
            return None  # too short to be useful
        steps = steps[:len(valid_steps)]
        step_end_positions = step_end_positions[:len(valid_steps)]
        input_ids = input_ids[:max_seq_len]

    input_tensor = torch.tensor([input_ids], device=model.device)

    # Forward pass
    capture.clear()
    with torch.no_grad():
        model(input_tensor, use_cache=False)

    # Collect per-step metrics
    step_metrics = []
    for j, pos in enumerate(step_end_positions):
        if pos >= len(input_ids):
            break

        # Aggregate Attn and FFN contributions across target layers
        A_j = torch.zeros(model.config.hidden_size)
        F_j = torch.zeros(model.config.hidden_size)

        for l in range(layer_lo, layer_hi):
            if l in capture.attn_outputs and l in capture.ffn_outputs:
                A_j += capture.attn_outputs[l][0, pos, :]  # [hidden_dim]
                F_j += capture.ffn_outputs[l][0, pos, :]

        # Hidden state at this position (last layer)
        h_j = None
        last_layer = num_layers - 1
        if last_layer in capture.attn_outputs and last_layer in capture.ffn_outputs:
            # h_j = residual, but we approximate with attn+ffn of last layer
            h_j_vec = (capture.attn_outputs[last_layer][0, pos, :]
                       + capture.ffn_outputs[last_layer][0, pos, :])

        # Compute AFC metrics
        metrics = compute_afc_metrics(A_j, F_j, unembed_weight)

        # Also store norms for sanity checks
        metrics["attn_norm"] = torch.norm(A_j).item()
        metrics["ffn_norm"] = torch.norm(F_j).item()
        metrics["step_idx"] = j
        metrics["token_pos"] = pos

        # Step label: 0 = correct, 1 = first error
        if label == -1:
            metrics["is_error"] = 0
            metrics["is_first_error"] = 0
        else:
            metrics["is_error"] = 1 if j >= label else 0
            metrics["is_first_error"] = 1 if j == label else 0

        step_metrics.append(metrics)

    capture.clear()
    return step_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to local LLM (e.g. gz-data/models/.../Llama-3.1-8B-Instruct)")
    parser.add_argument("--data_dir", type=str, default="data/processbench",
                        help="Directory containing ProcessBench jsonl files")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--splits", type=str, default="gsm8k",
                        help="Comma-separated splits to process (gsm8k,math,olympiadbench,omnimath)")
    parser.add_argument("--max_examples", type=int, default=-1,
                        help="Max examples per split (-1 = all)")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    # Load model
    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Layers: {len(model.model.layers)}, Hidden: {model.config.hidden_size}")

    # Get unembed weight for vocab-space AFC
    unembed_weight = model.lm_head.weight.detach().cpu()  # [vocab, hidden]

    # Register hooks
    capture = ActivationCapture(model)

    for split in args.splits.split(","):
        split = split.strip()
        data_path = os.path.join(args.data_dir, f"{split}.jsonl")
        if not os.path.exists(data_path):
            print(f"[WARN] {data_path} not found, skipping")
            continue

        # Load data
        examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))

        if args.max_examples > 0:
            examples = examples[:args.max_examples]

        print(f"\n{'='*60}")
        print(f"Processing {split}: {len(examples)} examples")
        print(f"{'='*60}")

        out_path = os.path.join(args.output_dir, f"{split}_afc.jsonl")
        t0 = time.time()
        n_steps_total = 0
        n_skipped = 0

        with open(out_path, "w", encoding="utf-8") as fout:
            for idx, ex in enumerate(examples):
                try:
                    step_metrics = process_example(
                        ex, model, tokenizer, capture, unembed_weight,
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
                    "final_answer_correct": ex["final_answer_correct"],
                    "num_steps": len(ex["steps"]),
                    "step_metrics": step_metrics,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_steps_total += len(step_metrics)

                if (idx + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    rate = (idx + 1) / elapsed
                    print(f"  [{idx+1}/{len(examples)}] {rate:.1f} ex/s, "
                          f"{n_steps_total} steps extracted, {n_skipped} skipped")

                # Periodic GPU memory cleanup
                if (idx + 1) % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"\nDone {split}: {len(examples)-n_skipped} examples, "
              f"{n_steps_total} steps in {elapsed:.1f}s")
        print(f"Saved to {out_path}")

    capture.remove()
    print("\nAll done!")


if __name__ == "__main__":
    main()
