"""
Q1 Pilot v3: Step-level manifold metrics using NATURAL token windows.

Core hypothesis: correct reasoning trajectories evolve in a "constrained but
non-degenerate" subspace.

Key insight (v3): each reasoning step spans a range of tokens — this IS the
natural window. We extract hidden states for ALL tokens within each step,
forming a matrix [n_tokens_in_step, hidden_dim], and compute manifold geometry
directly on it. No artificial sliding windows, no warm-up problem.

Per-step metrics (available for EVERY step, no warm-up):
  A. Within-step manifold shape (last layer):
     - step_effective_rank:   effective rank of [n_tokens, hidden_dim]
     - step_rank_normed:      step_effective_rank / n_tokens
     - step_spectral_gap:     sigma_1 / sigma_2
     - norm_mean / norm_std:  token norm statistics

  B. Cross-layer coherence (per step):
     - crosslayer_rank:       effective rank of [n_layers, hidden_dim] (layer means)
     - crosslayer_rank_normed

Cross-step metrics (available from step 1):
  C. Inter-step dynamics:
     - displacement:          ||mean(step_j) - mean(step_{j-1})||
     - cosine_sim:            cosine between step means
     - displacement_normed
     - curvature:             angle between consecutive displacement vectors (step 2+)

  D. Inter-step manifold comparison:
     - inter_step_angle:      largest principal angle between adjacent step subspaces
     - inter_step_deviation:  how well step j tokens are explained by step j-1 subspace
     - crosslayer_disp_align: do layers agree on displacement direction?

  E. Trajectory-level (accumulating step means):
     - traj_effective_rank:   effective rank of [step_means_0..j, hidden_dim]
     - traj_rank_normed
     - traj_spectral_gap

  F. Running z-scores

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
# 1. Build prompt and find step TOKEN RANGES
# ──────────────────────────────────────────────

def build_prompt_and_ranges(problem: str, steps: list, tokenizer):
    """Return (full_ids, step_ranges) where step_ranges is list of (start, end) inclusive."""
    prefix = f"Problem: {problem}\nSolution:"
    step_ranges = []
    current_text = prefix
    prev_len = len(tokenizer.encode(current_text, add_special_tokens=True))

    for i, step in enumerate(steps):
        step_text = f"\nStep {i+1}: {step}"
        current_text += step_text
        new_len = len(tokenizer.encode(current_text, add_special_tokens=True))
        step_ranges.append((prev_len, new_len - 1))  # (start, end) inclusive
        prev_len = new_len

    full_ids = tokenizer.encode(current_text, add_special_tokens=True)
    return full_ids, step_ranges


# ──────────────────────────────────────────────
# 2. Geometric primitives
# ──────────────────────────────────────────────

def effective_rank(matrix):
    """Effective rank via Shannon entropy of normalized singular values."""
    S = torch.linalg.svdvals(matrix.float())
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    H = -(p * p.log()).sum()
    return H.exp().item()


def spectral_gap(matrix):
    """Ratio of top singular value to second. Large = concentrated."""
    S = torch.linalg.svdvals(matrix.float())
    S = S[S > 1e-10]
    if len(S) < 2:
        return float('inf')
    return (S[0] / S[1]).item()


def subspace_deviation(basis_matrix, target_matrix, variance_threshold=0.95):
    """
    How well target_matrix's rows are explained by basis_matrix's PCA subspace.
    Returns mean normalized reconstruction error.

    basis_matrix:  [n1, hidden_dim] — defines the subspace
    target_matrix: [n2, hidden_dim] — projected onto that subspace
    """
    if basis_matrix.shape[0] < 2:
        return None

    mean = basis_matrix.mean(dim=0)
    centered_basis = (basis_matrix - mean).float()
    centered_target = (target_matrix - mean).float()

    U, S, Vh = torch.linalg.svd(centered_basis, full_matrices=False)

    cumvar = (S ** 2).cumsum(0) / (S ** 2).sum()
    n_comp = max(2, int((cumvar < variance_threshold).sum().item()) + 1)
    n_comp = min(n_comp, len(S))

    basis = Vh[:n_comp]  # [n_comp, hidden_dim]
    projected = centered_target @ basis.T @ basis  # [n2, hidden_dim]
    residuals = centered_target - projected

    per_row_error = torch.norm(residuals, dim=1)
    per_row_norm = torch.norm(centered_target, dim=1)

    valid = per_row_norm > 1e-8
    if valid.sum() == 0:
        return 0.0
    return (per_row_error[valid] / per_row_norm[valid]).mean().item()


def principal_angle(matrix_a, matrix_b, n_components=3):
    """
    Largest principal angle (radians) between PCA subspaces of two matrices.
    Measures how much the manifold "rotates" between steps.
    """
    # Adapt n_components to available data
    max_comp = min(matrix_a.shape[0], matrix_b.shape[0]) - 1
    n_components = min(n_components, max_comp)
    if n_components < 1:
        return None

    centered_a = (matrix_a - matrix_a.mean(dim=0)).float()
    centered_b = (matrix_b - matrix_b.mean(dim=0)).float()

    _, _, Vh_a = torch.linalg.svd(centered_a, full_matrices=False)
    _, _, Vh_b = torch.linalg.svd(centered_b, full_matrices=False)

    k = min(n_components, Vh_a.shape[0], Vh_b.shape[0])
    Q_a = Vh_a[:k]
    Q_b = Vh_b[:k]

    cos_angles = torch.linalg.svdvals(Q_a @ Q_b.T)
    cos_angles = cos_angles.clamp(-1, 1)
    angles = torch.acos(cos_angles)
    return angles.max().item()


def crosslayer_displacement_alignment(h_step_curr, h_step_prev, layer_indices):
    """
    Compute per-layer displacement (mean of step j - mean of step j-1),
    then measure pairwise cosine similarity. High = layers agree on direction.
    """
    displacements = []
    for l in layer_indices:
        if l in h_step_curr and l in h_step_prev:
            d = h_step_curr[l].float().mean(dim=0) - h_step_prev[l].float().mean(dim=0)
            if torch.norm(d) > 1e-10:
                displacements.append(d)

    if len(displacements) < 2:
        return None

    cos_sims = []
    for i in range(len(displacements)):
        for j in range(i + 1, len(displacements)):
            cs = torch.nn.functional.cosine_similarity(
                displacements[i].unsqueeze(0), displacements[j].unsqueeze(0)
            ).item()
            cos_sims.append(cs)

    return float(np.mean(cos_sims))


# ──────────────────────────────────────────────
# 3. Running statistics tracker
# ──────────────────────────────────────────────

class RunningStats:
    def __init__(self):
        self.values = {}

    def update(self, name, val):
        if val is None:
            return
        if name not in self.values:
            self.values[name] = []
        self.values[name].append(val)

    def zscore(self, name, val):
        if val is None or name not in self.values or len(self.values[name]) < 2:
            return None
        arr = np.array(self.values[name])
        mu, sigma = arr.mean(), arr.std()
        if sigma < 1e-10:
            return 0.0
        return float((val - mu) / sigma)


# ──────────────────────────────────────────────
# 4. Per-step metric computation (v3)
# ──────────────────────────────────────────────

def compute_step_metrics(
    h_step,              # {layer_idx: [n_tokens, hidden_dim]} — current step
    prev_h_step,         # same, previous step (or None)
    prev_displacement,   # displacement vector (or None)
    step_means_history,  # list of mean vectors (last-layer) from previous steps
    layer_indices,
    running_stats,
):
    last_layer = layer_indices[-1]
    tokens_matrix = h_step[last_layer].float()  # [n_tokens, hidden_dim]
    step_mean = tokens_matrix.mean(dim=0)        # [hidden_dim]
    n_tokens = tokens_matrix.shape[0]
    metrics = {}

    # ── A. Within-step manifold shape (every step, no warm-up) ──
    metrics["n_tokens"] = n_tokens
    norms = torch.norm(tokens_matrix, dim=1)
    metrics["norm_mean"] = norms.mean().item()
    metrics["norm_std"] = norms.std().item()

    if n_tokens >= 2:
        metrics["step_effective_rank"] = effective_rank(tokens_matrix)
        metrics["step_rank_normed"] = metrics["step_effective_rank"] / n_tokens
        metrics["step_spectral_gap"] = spectral_gap(tokens_matrix)
    else:
        metrics["step_effective_rank"] = None
        metrics["step_rank_normed"] = None
        metrics["step_spectral_gap"] = None

    # ── B. Cross-layer coherence (every step) ──
    if len(layer_indices) >= 3:
        layer_means = torch.stack([h_step[l].float().mean(dim=0) for l in layer_indices])
        metrics["crosslayer_rank"] = effective_rank(layer_means)
        metrics["crosslayer_rank_normed"] = metrics["crosslayer_rank"] / len(layer_indices)
    else:
        metrics["crosslayer_rank"] = None
        metrics["crosslayer_rank_normed"] = None

    # ── C. Inter-step dynamics (from step 1) ──
    displacement = None
    if prev_h_step is not None:
        prev_mean = prev_h_step[last_layer].float().mean(dim=0)
        diff = step_mean - prev_mean
        displacement = diff
        metrics["displacement"] = torch.norm(diff).item()
        metrics["cosine_sim"] = torch.nn.functional.cosine_similarity(
            step_mean.unsqueeze(0), prev_mean.unsqueeze(0)
        ).item()
        metrics["displacement_normed"] = metrics["displacement"] / (metrics["norm_mean"] + 1e-8)
    else:
        metrics["displacement"] = None
        metrics["cosine_sim"] = None
        metrics["displacement_normed"] = None

    if displacement is not None and prev_displacement is not None:
        cos_angle = torch.nn.functional.cosine_similarity(
            displacement.unsqueeze(0), prev_displacement.unsqueeze(0)
        ).item()
        cos_angle = max(-1.0, min(1.0, cos_angle))
        metrics["curvature"] = math.acos(cos_angle)
    else:
        metrics["curvature"] = None

    # ── D. Inter-step manifold comparison (from step 1) ──
    if prev_h_step is not None:
        prev_tokens = prev_h_step[last_layer].float()
        metrics["inter_step_deviation"] = subspace_deviation(prev_tokens, tokens_matrix)
        metrics["inter_step_angle"] = principal_angle(prev_tokens, tokens_matrix)
        metrics["crosslayer_disp_align"] = crosslayer_displacement_alignment(
            h_step, prev_h_step, layer_indices
        )
    else:
        metrics["inter_step_deviation"] = None
        metrics["inter_step_angle"] = None
        metrics["crosslayer_disp_align"] = None

    # ── E. Trajectory-level (accumulating step means) ──
    all_means = step_means_history + [step_mean]
    if len(all_means) >= 2:
        traj_matrix = torch.stack(all_means)
        metrics["traj_effective_rank"] = effective_rank(traj_matrix)
        metrics["traj_rank_normed"] = metrics["traj_effective_rank"] / len(all_means)
        metrics["traj_spectral_gap"] = spectral_gap(traj_matrix)
    else:
        metrics["traj_effective_rank"] = None
        metrics["traj_rank_normed"] = None
        metrics["traj_spectral_gap"] = None

    # ── F. Running z-scores ──
    zscore_metrics = [
        "step_effective_rank", "step_spectral_gap", "norm_mean",
        "displacement", "cosine_sim", "curvature",
        "inter_step_deviation", "inter_step_angle",
        "crosslayer_disp_align", "displacement_normed",
    ]
    for m in zscore_metrics:
        val = metrics.get(m)
        z = running_stats.zscore(m, val)
        metrics[f"{m}_zscore"] = z
        running_stats.update(m, val)

    return metrics, displacement


# ──────────────────────────────────────────────
# 5. Main extraction loop
# ──────────────────────────────────────────────

def process_example(example, model, tokenizer, target_layers, max_seq_len=2048):
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

    all_hidden = outputs.hidden_states  # tuple of [1, seq_len, hidden_dim]
    layer_indices = sorted(target_layers)
    last_layer = layer_indices[-1]

    step_metrics_list = []
    prev_h_step = None
    prev_displacement = None
    step_means_history = []
    running_stats = RunningStats()

    for j, (tok_start, tok_end) in enumerate(step_ranges):
        if tok_end >= len(input_ids):
            tok_end = len(input_ids) - 1
        if tok_start > tok_end:
            break

        # Extract hidden states for ALL tokens in this step, at each sampled layer
        h_step = {}
        for l in layer_indices:
            layer_hs_idx = l + 1
            if layer_hs_idx < len(all_hidden):
                h_step[l] = all_hidden[layer_hs_idx][0, tok_start:tok_end+1, :].cpu()

        metrics, displacement = compute_step_metrics(
            h_step, prev_h_step, prev_displacement,
            step_means_history, layer_indices, running_stats,
        )

        metrics["step_idx"] = j
        metrics["token_range"] = [tok_start, tok_end]

        if label == -1:
            metrics["is_error"] = 0
            metrics["is_first_error"] = 0
        else:
            metrics["is_error"] = 1 if j >= label else 0
            metrics["is_first_error"] = 1 if j == label else 0

        step_metrics_list.append(metrics)

        # Update state for next step
        step_means_history.append(h_step[last_layer].float().mean(dim=0))
        prev_h_step = h_step
        prev_displacement = displacement

    return step_metrics_list


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
    parser.add_argument("--n_sample_layers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    print(f"Loading model from {args.model_path} ...")
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(
            f"Model path {args.model_path} is not a local directory. "
            f"Please provide an absolute path to the model files."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"Model loaded. Layers: {num_layers}, Hidden: {hidden_dim}")

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
              f"{n_steps} steps in {elapsed:.1f}s -> {out_path}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
