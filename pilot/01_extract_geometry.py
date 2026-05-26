"""
Q1 Pilot v2: Extract step-level MANIFOLD metrics from ProcessBench.

Core hypothesis: correct reasoning trajectories evolve in a "constrained but
non-degenerate" subspace. v1 used naive per-step statistics (displacement, cosine)
which showed ~0.5 AUROC — essentially random. The problem: those metrics are
point-wise, not trajectory-level.

v2 adds trajectory-aware metrics that directly test the manifold hypothesis:

A. Trajectory-level (sliding window over steps):
   - traj_effective_rank:    effective rank of [window_steps x hidden_dim] matrix
                             → does the trajectory live in a constrained subspace?
   - traj_spectral_gap:      sigma_1 / sigma_2 ratio
                             → how concentrated is the trajectory's energy?
   - subspace_deviation:      reconstruction error when projecting step j onto
                             PCA subspace of steps [0..j-1]
                             → does step j stay on the established manifold?
   - subspace_angle_change:   principal angle change when step j is added
                             → does the subspace rotate suddenly?

B. Cross-layer coherence:
   - crosslayer_disp_align:   mean pairwise cosine of displacement vectors
                             across sampled layers
                             → do all layers "agree" on the direction of change?
   - crosslayer_rank_ratio:   effective_rank / n_layers → normalized dimensionality

C. Running anomaly scores:
   - *_zscore:                z-score of each basic metric relative to running
                             mean/std of previous steps in the same trajectory
                             → is this step anomalous for THIS trajectory?

D. Basic metrics (kept for reference):
   - displacement, cosine_sim, norm, curvature, effective_rank, displacement_normed

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
    prefix = f"Problem: {problem}\nSolution:"
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
    """Ratio of top singular value to second singular value. Large = concentrated."""
    S = torch.linalg.svdvals(matrix.float())
    S = S[S > 1e-10]
    if len(S) < 2:
        return float('inf')
    return (S[0] / S[1]).item()


def subspace_deviation(history_matrix, new_vec, n_components=None):
    """
    Project new_vec onto PCA subspace of history_matrix.
    Returns reconstruction error (normalized).

    history_matrix: [n_steps, hidden_dim]
    new_vec: [hidden_dim]
    """
    if history_matrix.shape[0] < 2:
        return None

    # Center
    mean = history_matrix.mean(dim=0)
    centered = history_matrix - mean
    new_centered = new_vec - mean

    # SVD for PCA
    U, S, Vh = torch.linalg.svd(centered.float(), full_matrices=False)

    # Keep components explaining 95% variance, or at least 2
    if n_components is None:
        cumvar = (S ** 2).cumsum(0) / (S ** 2).sum()
        n_components = max(2, int((cumvar < 0.95).sum().item()) + 1)
        n_components = min(n_components, len(S))

    # Project and reconstruct
    basis = Vh[:n_components]  # [n_components, hidden_dim]
    proj_coeffs = new_centered.float() @ basis.T  # [n_components]
    reconstructed = proj_coeffs @ basis  # [hidden_dim]

    residual = new_centered.float() - reconstructed
    error = torch.norm(residual).item()
    norm = torch.norm(new_centered.float()).item()

    return error / (norm + 1e-8)


def principal_angle_change(history_matrix, new_vec, n_components=3):
    """
    Measure how much the principal subspace rotates when new_vec is added.
    Returns the largest principal angle (radians) between old and new subspaces.
    """
    if history_matrix.shape[0] < n_components + 1:
        return None

    old_centered = history_matrix - history_matrix.mean(dim=0)
    new_matrix = torch.cat([history_matrix, new_vec.unsqueeze(0)], dim=0)
    new_centered = new_matrix - new_matrix.mean(dim=0)

    _, _, Vh_old = torch.linalg.svd(old_centered.float(), full_matrices=False)
    _, _, Vh_new = torch.linalg.svd(new_centered.float(), full_matrices=False)

    k = min(n_components, Vh_old.shape[0], Vh_new.shape[0])
    Q_old = Vh_old[:k]  # [k, hidden_dim]
    Q_new = Vh_new[:k]

    # Principal angles via SVD of Q_old @ Q_new.T
    cos_angles = torch.linalg.svdvals(Q_old @ Q_new.T)
    cos_angles = cos_angles.clamp(-1, 1)
    angles = torch.acos(cos_angles)

    return angles.max().item()


def crosslayer_displacement_alignment(h_at_layers_curr, h_at_layers_prev, layer_indices):
    """
    Compute displacement vectors at each sampled layer, then measure
    their pairwise cosine similarity. High = layers agree on direction.
    """
    displacements = []
    for l in layer_indices:
        if l in h_at_layers_curr and l in h_at_layers_prev:
            d = (h_at_layers_curr[l] - h_at_layers_prev[l]).float()
            if torch.norm(d) > 1e-10:
                displacements.append(d)

    if len(displacements) < 2:
        return None

    # Pairwise cosine similarities
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
    """Track running mean and std for z-score computation."""

    def __init__(self):
        self.values = {}  # metric_name -> list of values

    def update(self, name, val):
        if val is None:
            return
        if name not in self.values:
            self.values[name] = []
        self.values[name].append(val)

    def zscore(self, name, val):
        if val is None or name not in self.values or len(self.values[name]) < 3:
            return None
        arr = np.array(self.values[name])
        mu, sigma = arr.mean(), arr.std()
        if sigma < 1e-10:
            return 0.0
        return float((val - mu) / sigma)


# ──────────────────────────────────────────────
# 4. Per-step metric computation (v2)
# ──────────────────────────────────────────────

def compute_step_geometry_v2(
    h_at_layers,           # dict {layer_idx: tensor [hidden_dim]}
    prev_h_at_layers,      # same, previous step (or None)
    prev_displacement,     # tensor [hidden_dim] (or None)
    trajectory_history,    # list of tensors [hidden_dim] (last-layer, all previous steps)
    layer_indices,         # sorted list of layer indices
    running_stats,         # RunningStats instance
):
    last_layer = layer_indices[-1]
    h_j = h_at_layers[last_layer].float()
    metrics = {}

    # ── A. Basic metrics (same as v1) ──
    metrics["norm"] = torch.norm(h_j).item()

    displacement = None
    if prev_h_at_layers is not None and last_layer in prev_h_at_layers:
        h_prev = prev_h_at_layers[last_layer].float()
        diff = h_j - h_prev
        displacement = diff
        metrics["displacement"] = torch.norm(diff).item()
        metrics["cosine_sim"] = torch.nn.functional.cosine_similarity(
            h_j.unsqueeze(0), h_prev.unsqueeze(0)
        ).item()
        metrics["displacement_normed"] = metrics["displacement"] / (metrics["norm"] + 1e-8)
    else:
        metrics["displacement"] = None
        metrics["cosine_sim"] = None
        metrics["displacement_normed"] = None

    # Curvature
    if displacement is not None and prev_displacement is not None:
        cos_angle = torch.nn.functional.cosine_similarity(
            displacement.unsqueeze(0), prev_displacement.unsqueeze(0)
        ).item()
        cos_angle = max(-1.0, min(1.0, cos_angle))
        metrics["curvature"] = math.acos(cos_angle)
    else:
        metrics["curvature"] = None

    # Cross-layer effective rank (single step, across layers)
    if len(layer_indices) >= 3:
        layer_matrix = torch.stack([h_at_layers[l].float() for l in layer_indices])
        metrics["crosslayer_rank"] = effective_rank(layer_matrix)
        metrics["crosslayer_rank_normed"] = metrics["crosslayer_rank"] / len(layer_indices)
    else:
        metrics["crosslayer_rank"] = None
        metrics["crosslayer_rank_normed"] = None

    # ── B. Cross-layer displacement alignment ──
    if prev_h_at_layers is not None:
        metrics["crosslayer_disp_align"] = crosslayer_displacement_alignment(
            h_at_layers, prev_h_at_layers, layer_indices
        )
    else:
        metrics["crosslayer_disp_align"] = None

    # ── C. Trajectory-level metrics (need history) ──
    if len(trajectory_history) >= 2:
        history_matrix = torch.stack(trajectory_history)  # [n_prev_steps, hidden_dim]

        # Trajectory effective rank (all steps so far including current)
        full_traj = torch.cat([history_matrix, h_j.unsqueeze(0)], dim=0)
        metrics["traj_effective_rank"] = effective_rank(full_traj)
        metrics["traj_rank_normed"] = metrics["traj_effective_rank"] / full_traj.shape[0]
        metrics["traj_spectral_gap"] = spectral_gap(full_traj)

        # Subspace deviation: how well does step j fit in the subspace of previous steps?
        metrics["subspace_deviation"] = subspace_deviation(history_matrix, h_j)

        # Principal angle change: does the subspace rotate when step j is added?
        metrics["subspace_angle_change"] = principal_angle_change(history_matrix, h_j)

    elif len(trajectory_history) == 1:
        # Only 1 previous step, can compute limited metrics
        full_traj = torch.stack(trajectory_history + [h_j])
        metrics["traj_effective_rank"] = effective_rank(full_traj)
        metrics["traj_rank_normed"] = metrics["traj_effective_rank"] / full_traj.shape[0]
        metrics["traj_spectral_gap"] = spectral_gap(full_traj)
        metrics["subspace_deviation"] = None
        metrics["subspace_angle_change"] = None
    else:
        metrics["traj_effective_rank"] = None
        metrics["traj_rank_normed"] = None
        metrics["traj_spectral_gap"] = None
        metrics["subspace_deviation"] = None
        metrics["subspace_angle_change"] = None

    # ── D. Sliding window trajectory rank (last 5 steps) ──
    window_size = 5
    if len(trajectory_history) >= window_size - 1:
        window = trajectory_history[-(window_size - 1):] + [h_j]
        window_matrix = torch.stack(window)
        metrics["window_rank"] = effective_rank(window_matrix)
        metrics["window_rank_normed"] = metrics["window_rank"] / window_size
        metrics["window_spectral_gap"] = spectral_gap(window_matrix)
    else:
        metrics["window_rank"] = None
        metrics["window_rank_normed"] = None
        metrics["window_spectral_gap"] = None

    # ── E. Running z-scores (anomaly relative to this trajectory) ──
    zscore_metrics = [
        "displacement", "cosine_sim", "norm", "curvature",
        "crosslayer_disp_align", "subspace_deviation", "traj_effective_rank",
        "displacement_normed"
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

    input_ids, step_end_positions = build_prompt_and_boundaries(problem, steps, tokenizer)

    if len(input_ids) > max_seq_len:
        valid_steps = [i for i, pos in enumerate(step_end_positions) if pos < max_seq_len]
        if len(valid_steps) < 2:
            return None
        steps = steps[:len(valid_steps)]
        step_end_positions = step_end_positions[:len(valid_steps)]
        input_ids = input_ids[:max_seq_len]

    input_tensor = torch.tensor([input_ids], device=model.device)

    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=True, use_cache=False)

    all_hidden = outputs.hidden_states
    layer_indices = sorted(target_layers)
    last_layer = layer_indices[-1]

    step_metrics = []
    prev_h_at_layers = None
    prev_displacement = None
    trajectory_history = []  # list of last-layer hidden states
    running_stats = RunningStats()

    for j, pos in enumerate(step_end_positions):
        if pos >= len(input_ids):
            break

        h_at_layers = {}
        for l in layer_indices:
            layer_hs_idx = l + 1
            if layer_hs_idx < len(all_hidden):
                h_at_layers[l] = all_hidden[layer_hs_idx][0, pos, :].cpu()

        metrics, displacement = compute_step_geometry_v2(
            h_at_layers, prev_h_at_layers, prev_displacement,
            trajectory_history, layer_indices, running_stats,
        )

        metrics["step_idx"] = j
        metrics["token_pos"] = pos

        if label == -1:
            metrics["is_error"] = 0
            metrics["is_first_error"] = 0
        else:
            metrics["is_error"] = 1 if j >= label else 0
            metrics["is_first_error"] = 1 if j == label else 0

        step_metrics.append(metrics)
        prev_h_at_layers = h_at_layers
        prev_displacement = displacement
        trajectory_history.append(h_at_layers[last_layer].float())

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
    parser.add_argument("--n_sample_layers", type=int, default=8)
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
