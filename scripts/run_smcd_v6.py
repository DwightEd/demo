#!/usr/bin/env python3
"""
SMCD v6: Discriminative Trajectory Probe with E×C×N Inductive Bias.

Key shift: from generative (density estimation) to discriminative (probe).
We have labels — use them. Learn what features distinguish correct from error.

Pipeline:
    1. Multi-layer spectra → per-(step,layer) info-geometric features
    2. Layer attention learns which layers matter and in what direction
    3. GRU captures sequential evolution patterns
    4. BCE + E×C×N regularization trains the probe
    5. Output: per-step error probability → CUSUM for sequence-level

Usage:
    python scripts/run_smcd_v6.py --data_path pilot/results/gsm8k_multilayer.pt
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smcd.info_geometry import spectrum_to_distribution, effective_rank, hellinger_distance
from smcd.probe import TrajectoryProbe
from smcd.detector import CUSUMDetector, evaluate_detection


def parse_args():
    p = argparse.ArgumentParser(description="SMCD v6: discriminative probe")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="smcd_v6_output")
    p.add_argument("--seed", type=int, default=42)
    # Model
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--gru_hidden", type=int, default=128)
    p.add_argument("--gru_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.15)
    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--ecn_weight", type=float, default=0.1,
                   help="E×C×N regularization weight")
    # Detection
    p.add_argument("--cusum_threshold", type=float, default=3.0)
    # Split
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_grid_features(data, meta):
    """Build per-(step, layer) feature grid.

    For each step j, layer l, compute:
        - effective_rank(p_{j,l})
        - spectral_entropy(p_{j,l})
        - hellinger_distance(p_{j-1,l}, p_{j,l})  (0 for j=0)
        - delta_effective_rank(j, l)               (0 for j=0)
        - spectral_gap σ_1/σ_2
        - energy_ratio (σ_1^2 / Σσ^2)

    Output: (T, L, F) feature grid per example, F=6 features per node.
    """
    L = len(meta["layer_indices"])
    k = meta["k"]
    F = 6  # features per (step, layer) node

    examples = []
    for ex in data:
        steps = ex["steps"]
        T = len(steps)
        if T < 2:
            continue

        sigma_all = [s["sigma_ml"] for s in steps]  # list of (L, k)
        p_all = [spectrum_to_distribution(s) for s in sigma_all]

        grid = torch.zeros(T, L, F)
        ecn_summary = torch.zeros(T, 3)  # [E_mean, C_mean, N_abs_mean]

        for j in range(T):
            p_j = p_all[j]      # (L, k)
            sigma_j = sigma_all[j]  # (L, k)

            # Feature 0: effective rank
            eff_r = effective_rank(p_j)  # (L,)
            grid[j, :, 0] = eff_r

            # Feature 1: spectral entropy (log of effective rank)
            grid[j, :, 1] = torch.log(eff_r + 1e-6)

            # Feature 2: Hellinger distance to previous step
            if j > 0:
                h_dist = hellinger_distance(p_all[j - 1], p_j)  # (L,)
                grid[j, :, 2] = h_dist
            # else: 0 (default)

            # Feature 3: delta effective rank
            if j > 0:
                eff_r_prev = effective_rank(p_all[j - 1])
                grid[j, :, 3] = eff_r - eff_r_prev

            # Feature 4: spectral gap σ_1/σ_2
            grid[j, :, 4] = sigma_j[:, 0] / (sigma_j[:, 1] + 1e-8)

            # Feature 5: energy concentration (σ_1^2 / Σσ^2)
            s2 = sigma_j ** 2
            grid[j, :, 5] = s2[:, 0] / (s2.sum(dim=-1) + 1e-8)

            # ECN summary for regularization
            ecn_summary[j, 0] = eff_r.mean()          # E
            ecn_summary[j, 1] = grid[j, :, 2].mean()  # C
            ecn_summary[j, 2] = grid[j, :, 3].abs().mean()  # |N|

        labels = torch.tensor([s["is_error"] for s in steps], dtype=torch.float32)

        examples.append({
            "grid": grid,            # (T, L, F)
            "ecn": ecn_summary,      # (T, 3)
            "labels": labels,        # (T,)
            "example_label": ex["label"],
        })

    return examples


def normalize_grids(examples, train_idx):
    """Per-feature normalization across all training (step, layer) nodes."""
    all_grids = []
    for i in train_idx:
        all_grids.append(examples[i]["grid"])  # (T, L, F)

    cat = torch.cat([g.reshape(-1, g.shape[-1]) for g in all_grids], dim=0)
    mu = cat.mean(dim=0)
    std = cat.std(dim=0)
    std[std < 1e-8] = 1.0
    return mu, std


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_probe(examples, train_idx, val_idx, mu, std, meta, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = len(meta["layer_indices"])
    F = examples[0]["grid"].shape[-1]

    print(f"\n{'='*60}")
    print("Training TrajectoryProbe")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Grid: {L} layers × {F} features per node")
    print(f"  Model: d_model={args.d_model}, heads={args.n_heads}, "
          f"GRU={args.gru_hidden}×{args.gru_layers}")

    model = TrajectoryProbe(
        n_layers=L, feat_per_layer=F,
        d_model=args.d_model, n_heads=args.n_heads,
        gru_hidden=args.gru_hidden, gru_layers=args.gru_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Count step-level labels in training set
    n_correct_steps = sum(
        (examples[i]["labels"] == 0).sum().item() for i in train_idx)
    n_error_steps = sum(
        (examples[i]["labels"] == 1).sum().item() for i in train_idx)
    print(f"  Train steps: {n_correct_steps} correct, {n_error_steps} error")

    # Class weight for imbalanced labels
    if n_error_steps > 0:
        pos_weight = n_correct_steps / n_error_steps
        print(f"  Pos weight: {pos_weight:.2f}")
    else:
        pos_weight = 1.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auroc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        train_order = np.random.permutation(len(train_idx))
        epoch_loss = 0.0
        n_batches = 0

        for bs in range(0, len(train_order), args.batch_size):
            batch_idx = [train_idx[train_order[i]]
                         for i in range(bs, min(bs + args.batch_size, len(train_order)))]

            grids = [examples[i]["grid"] for i in batch_idx]
            labels = [examples[i]["labels"] for i in batch_idx]
            ecns = [examples[i]["ecn"] for i in batch_idx]
            lengths = torch.tensor([g.shape[0] for g in grids])
            max_len = lengths.max().item()
            B = len(batch_idx)
            Ld = grids[0].shape[1]
            Fd = grids[0].shape[2]

            padded_grid = torch.zeros(B, max_len, Ld, Fd)
            padded_labels = torch.zeros(B, max_len)
            padded_ecn = torch.zeros(B, max_len, 3)
            mask = torch.zeros(B, max_len)

            for i in range(B):
                T = grids[i].shape[0]
                padded_grid[i, :T] = (grids[i] - mu) / std
                padded_labels[i, :T] = labels[i]
                padded_ecn[i, :T] = ecns[i]
                mask[i, :T] = 1.0

            padded_grid = padded_grid.to(device)
            padded_labels = padded_labels.to(device)
            padded_ecn = padded_ecn.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)

            loss, bce_val, logits = model.compute_loss(
                padded_grid, padded_labels, lengths, mask,
                ecn_features=padded_ecn, ecn_weight=args.ecn_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += bce_val
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation AUROC
        model.eval()
        val_scores = []
        val_labels_flat = []
        with torch.no_grad():
            for i in val_idx:
                g = ((examples[i]["grid"] - mu) / std).unsqueeze(0).to(device)
                l = torch.tensor([g.shape[1]]).to(device)
                probs, _ = model.predict_scores(g, l)
                probs = probs[0].cpu().numpy()
                labs = examples[i]["labels"].numpy()
                for j in range(1, len(probs)):
                    val_scores.append(probs[j])
                    val_labels_flat.append(labs[j])

        if len(set(val_labels_flat)) > 1:
            val_auroc = roc_auc_score(val_labels_flat, val_scores)
        else:
            val_auroc = 0.5

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: bce={avg_loss:.4f}, val_auroc={val_auroc:.4f}, "
                  f"best={best_val_auroc:.4f}")

        if no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  Best val AUROC: {best_val_auroc:.4f}")
    return model


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate(model, examples, test_idx, mu, std, device):
    print(f"\n{'='*60}")
    print("Step-level AUROC (test set)")
    print(f"{'='*60}")

    model.eval()
    correct_scores = []
    error_scores = []
    first_error_scores = []

    all_correct = []
    all_error = []

    with torch.no_grad():
        for i in test_idx:
            g = ((examples[i]["grid"] - mu) / std).unsqueeze(0).to(device)
            l = torch.tensor([g.shape[1]]).to(device)
            probs, attn_w = model.predict_scores(g, l)
            probs = probs[0].cpu().numpy()

            label = examples[i]["example_label"]
            labs = examples[i]["labels"].numpy()

            for j in range(1, len(probs)):
                if labs[j] == 0:
                    all_correct.append(probs[j])
                else:
                    all_error.append(probs[j])

            if label == -1:
                correct_scores.extend(probs[1:].tolist())
            else:
                for j in range(1, len(probs)):
                    if j < label:
                        correct_scores.append(probs[j])
                    elif j == label:
                        first_error_scores.append(probs[j])

    # First-error AUROC
    if first_error_scores and correct_scores:
        y_true = [0] * len(correct_scores) + [1] * len(first_error_scores)
        y_score = correct_scores + first_error_scores
        auroc_first = roc_auc_score(y_true, y_score)
        c, e = np.array(correct_scores), np.array(first_error_scores)
        sep = (e.mean() - c.mean()) / (c.std() + 1e-8)
        print(f"  First-error AUROC = {auroc_first:.4f} | "
              f"correct={len(c)}, error={len(e)} | sep={sep:.2f}σ")
    else:
        auroc_first = None
        print("  [WARN] Insufficient first-error data")

    # All-step AUROC
    auroc_all = None
    if all_error and all_correct:
        y2 = [0] * len(all_correct) + [1] * len(all_error)
        s2 = all_correct + all_error
        auroc_all = roc_auc_score(y2, s2)
        print(f"  All-step AUROC   = {auroc_all:.4f}")

    print(f"\n  Baselines: handcrafted=0.694, Sun_et_al=0.87")

    # Score distribution
    if correct_scores and first_error_scores:
        c = np.array(correct_scores)
        e = np.array(first_error_scores)
        print(f"  Correct: mean={c.mean():.4f}, std={c.std():.4f}")
        print(f"  Error:   mean={e.mean():.4f}, std={e.std():.4f}")

    return auroc_first, auroc_all


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # Load
    print(f"{'='*60}")
    print("Loading data")
    print(f"{'='*60}")
    raw = torch.load(args.data_path, weights_only=False)
    data, meta = raw["examples"], raw["meta"]
    L = len(meta["layer_indices"])
    k = meta["k"]
    n_correct = sum(1 for d in data if d["label"] == -1)
    print(f"  {len(data)} examples (correct={n_correct}, error={len(data)-n_correct})")
    print(f"  {L} layers × {k} SVs")

    # Build grid features
    print(f"\n{'='*60}")
    print("Building (step × layer) feature grid")
    print(f"{'='*60}")
    examples = build_grid_features(data, meta)
    F = examples[0]["grid"].shape[-1]
    print(f"  {len(examples)} usable examples")
    print(f"  Grid: T × {L} layers × {F} features")
    print(f"  Features: eff_rank, log_eff_rank, hellinger, delta_rank, spectral_gap, energy_ratio")

    # Split
    indices = np.arange(len(examples))
    np.random.shuffle(indices)
    n_train = int(len(indices) * args.train_ratio)
    n_val = int(len(indices) * args.val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Normalize
    mu, std = normalize_grids(examples, train_idx)

    # Train
    model = train_probe(examples, train_idx, val_idx, mu, std, meta, args)
    if model is None:
        return

    # Evaluate
    auroc_first, auroc_all = evaluate(model, examples, test_idx, mu, std, device)

    # CUSUM on probe scores
    print(f"\n{'='*60}")
    print("CUSUM")
    print(f"{'='*60}")
    model.eval()
    score_list = []
    with torch.no_grad():
        for ex in examples:
            g = ((ex["grid"] - mu) / std).unsqueeze(0).to(device)
            l = torch.tensor([g.shape[1]]).to(device)
            probs, _ = model.predict_scores(g, l)
            score_list.append(probs[0].cpu().numpy())

    cal = [score_list[i] for i in train_idx if examples[i]["example_label"] == -1]
    cusum_results = None
    if cal:
        detector = CUSUMDetector(threshold=args.cusum_threshold)
        detector.calibrate(cal)
        print(f"  k = {detector.k:.4f}")
        test_scores = [score_list[i] for i in test_idx]
        test_labels = [examples[i]["example_label"] for i in test_idx]
        cusum_results = evaluate_detection(detector, test_scores, test_labels)
        print(f"  TP={cusum_results['TP']}, FP={cusum_results['FP']}, "
              f"TN={cusum_results['TN']}, FN={cusum_results['FN']}")
        print(f"  P={cusum_results['precision']:.4f}, R={cusum_results['recall']:.4f}, "
              f"F1={cusum_results['f1']:.4f}")

    # Save
    elapsed = time.time() - t0
    results = {
        "n_examples": len(examples),
        "n_layers": L, "k": k, "feat_per_layer": F,
        "n_params": sum(p.numel() for p in model.parameters()),
        "auroc_first_error": auroc_first,
        "auroc_all_step": auroc_all,
        "cusum": cusum_results,
        "elapsed_seconds": elapsed,
        "args": vars(args),
    }
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "probe.pt"))

    print(f"\n{'='*60}")
    print(f"Saved to {out_path}")
    print(f"Total: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
