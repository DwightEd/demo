#!/usr/bin/env python3
"""
SMCD v4: Learned Spectral Encoder + Trajectory Evolution.

Key idea: don't manually reduce dimensions. Feed rich multi-layer spectra (L*k dims)
into a learned encoder that discovers what matters, then model evolution in latent space.

Usage:
    python scripts/run_smcd_v4.py --data_path pilot/results/gsm8k_multilayer.pt
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smcd.model import TrajectoryModel
from smcd.detector import CUSUMDetector, evaluate_detection


def parse_args():
    p = argparse.ArgumentParser(description="SMCD v4 pipeline")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="smcd_v4_output")
    p.add_argument("--seed", type=int, default=42)
    # Encoder
    p.add_argument("--d_model", type=int, default=64,
                   help="Spectral encoder hidden dim")
    p.add_argument("--enc_heads", type=int, default=4)
    p.add_argument("--enc_layers", type=int, default=2)
    p.add_argument("--latent_dim", type=int, default=128,
                   help="Latent representation dim (learned)")
    # Evolution model
    p.add_argument("--evo_heads", type=int, default=4)
    p.add_argument("--evo_layers", type=int, default=4)
    # Training
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=40)
    # Detection
    p.add_argument("--cusum_threshold", type=float, default=5.0)
    # Split
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path):
    raw = torch.load(path, weights_only=False)
    data = raw["examples"]
    meta = raw["meta"]
    return data, meta


def build_sequences(data):
    """Convert data to lists of (sigma_ml_sequence, label_sequence, example_label)."""
    sequences = []
    for ex in data:
        steps = ex["steps"]
        if len(steps) < 2:
            continue
        sigma_seq = torch.stack([s["sigma_ml"] for s in steps])  # (T, L, k)
        labels = torch.tensor([s["is_error"] for s in steps], dtype=torch.float32)
        sequences.append({
            "sigma_seq": sigma_seq,
            "labels": labels,
            "example_label": ex["label"],
        })
    return sequences


def normalize_spectra(sequences, train_idx):
    """Compute per-layer, per-sv-index mean/std from training correct trajectories."""
    all_sigma = []
    for i in train_idx:
        if sequences[i]["example_label"] == -1:
            all_sigma.append(sequences[i]["sigma_seq"])  # (T, L, k)

    if not all_sigma:
        return None, None

    cat = torch.cat(all_sigma, dim=0)  # (N_steps, L, k)
    mu = cat.mean(dim=0)    # (L, k)
    std = cat.std(dim=0)    # (L, k)
    std[std < 1e-8] = 1.0
    return mu, std


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train(sequences, train_idx, val_idx, mu, std, meta, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = len(meta["layer_indices"])
    k = meta["k"]

    print(f"\n{'='*60}")
    print("Training TrajectoryModel")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Input: {L} layers x {k} SVs = {L*k} dims/step")
    print(f"  Encoder: d_model={args.d_model}, heads={args.enc_heads}, layers={args.enc_layers}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Evolution: heads={args.evo_heads}, layers={args.evo_layers}")

    model = TrajectoryModel(
        n_layers=L, k=k,
        d_model=args.d_model, enc_heads=args.enc_heads, enc_layers=args.enc_layers,
        latent_dim=args.latent_dim,
        evo_heads=args.evo_heads, evo_layers=args.evo_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Correct trajectories for training
    def get_correct_seqs(indices):
        seqs = []
        for i in indices:
            if sequences[i]["example_label"] == -1:
                s = (sequences[i]["sigma_seq"] - mu) / std
                seqs.append(s)
        return seqs

    train_seqs = get_correct_seqs(train_idx)
    val_seqs = get_correct_seqs(val_idx)
    print(f"  Train: {len(train_seqs)} correct trajectories")
    print(f"  Val: {len(val_seqs)} correct trajectories")

    if not train_seqs:
        print("  [ERROR] No correct trajectories!")
        return None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(train_seqs)
        epoch_loss = 0.0
        n_batches = 0

        for bs in range(0, len(train_seqs), args.batch_size):
            batch = train_seqs[bs:bs + args.batch_size]
            lengths = [s.shape[0] for s in batch]
            max_len = max(lengths)
            B = len(batch)
            L_dim = batch[0].shape[1]
            k_dim = batch[0].shape[2]

            padded = torch.zeros(B, max_len, L_dim, k_dim)
            mask = torch.zeros(B, max_len)
            for i, seq in enumerate(batch):
                T = seq.shape[0]
                padded[i, :T] = seq
                mask[i, :T] = 1.0

            padded = padded.to(device)
            mask = mask.to(device)

            loss = model.nll_loss(padded, mask=mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for seq in val_seqs:
                s = seq.unsqueeze(0).to(device)
                m = torch.ones(1, seq.shape[0], device=device)
                nll = model.nll_loss(s, mask=m)
                val_loss += nll.item()
                val_n += 1

        avg_val = val_loss / max(val_n, 1)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train={avg_train:.4f}, val={avg_val:.4f}, "
                  f"best={best_val:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        if no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  Best val NLL: {best_val:.4f}")
    return model


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def compute_scores(model, sequences, mu, std, device):
    print(f"\n{'='*60}")
    print("Computing anomaly scores")
    print(f"{'='*60}")

    model.eval()
    delta_list = []

    with torch.no_grad():
        for seq_data in sequences:
            s = (seq_data["sigma_seq"] - mu) / std
            s = s.unsqueeze(0).to(device)
            delta = model.compute_anomaly_scores(s)
            delta_list.append(delta[0].cpu().numpy())

    all_d = np.concatenate(delta_list)
    print(f"  Scores: mean={all_d.mean():.4f}, std={all_d.std():.4f}, "
          f"median={np.median(all_d):.4f}, max={all_d.max():.4f}")
    return delta_list


def evaluate_auroc(delta_list, sequences, test_idx):
    print(f"\n{'='*60}")
    print("Step-level AUROC")
    print(f"{'='*60}")

    correct_scores = []
    error_scores = []

    for i in test_idx:
        label = sequences[i]["example_label"]
        deltas = delta_list[i]

        if label == -1:
            correct_scores.extend(deltas[1:].tolist())
        else:
            for j in range(1, len(deltas)):
                if j < label:
                    correct_scores.append(deltas[j])
                elif j == label:
                    error_scores.append(deltas[j])

    if not error_scores or not correct_scores:
        print("  [WARN] Insufficient data")
        return None

    y_true = [0] * len(correct_scores) + [1] * len(error_scores)
    y_score = correct_scores + error_scores
    auroc = roc_auc_score(y_true, y_score)

    print(f"  Correct steps: {len(correct_scores)}, First-error steps: {len(error_scores)}")
    print(f"  >>> First-error AUROC = {auroc:.4f} <<<")
    print(f"  (Baselines: handcrafted=0.694, v2_sigma_only=0.61, Sun_et_al=0.87)")

    # All-step AUROC
    all_correct, all_error = [], []
    for i in test_idx:
        deltas = delta_list[i]
        labs = sequences[i]["labels"].numpy()
        for j in range(1, len(deltas)):
            if labs[j] == 0:
                all_correct.append(deltas[j])
            else:
                all_error.append(deltas[j])

    if all_error and all_correct:
        y2 = [0] * len(all_correct) + [1] * len(all_error)
        s2 = all_correct + all_error
        print(f"  All-step AUROC: {roc_auc_score(y2, s2):.4f}")

    # Score stats
    c, e = np.array(correct_scores), np.array(error_scores)
    print(f"  Correct: mean={c.mean():.4f}, std={c.std():.4f}")
    print(f"  Error:   mean={e.mean():.4f}, std={e.std():.4f}")
    sep = (e.mean() - c.mean()) / (c.std() + 1e-8)
    print(f"  Separation: {sep:.4f} sigma")

    return auroc


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
    data, meta = load_data(args.data_path)
    L = len(meta["layer_indices"])
    k = meta["k"]
    n_correct = sum(1 for d in data if d["label"] == -1)
    print(f"  {len(data)} examples (correct={n_correct}, error={len(data)-n_correct})")
    print(f"  {L} layers x {k} SVs = {L*k} dims per step")

    sequences = build_sequences(data)
    print(f"  Usable: {len(sequences)} examples")

    # Split
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    n_train = int(len(indices) * args.train_ratio)
    n_val = int(len(indices) * args.val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Normalize
    mu, std = normalize_spectra(sequences, train_idx)
    if mu is None:
        print("No correct training data!")
        return

    # Train
    model = train(sequences, train_idx, val_idx, mu, std, meta, args)
    if model is None:
        return

    # Score
    delta_list = compute_scores(model, sequences, mu, std, device)

    # AUROC
    auroc = evaluate_auroc(delta_list, sequences, test_idx)

    # CUSUM
    print(f"\n{'='*60}")
    print("CUSUM")
    print(f"{'='*60}")
    cal = [delta_list[i] for i in train_idx
           if sequences[i]["example_label"] == -1]
    cusum_results = None
    if cal:
        detector = CUSUMDetector(threshold=args.cusum_threshold)
        detector.calibrate(cal)
        print(f"  k = {detector.k:.4f}")
        test_scores = [delta_list[i] for i in test_idx]
        test_labels = [sequences[i]["example_label"] for i in test_idx]
        cusum_results = evaluate_detection(detector, test_scores, test_labels)
        print(f"  TP={cusum_results['TP']}, FP={cusum_results['FP']}, "
              f"TN={cusum_results['TN']}, FN={cusum_results['FN']}")
        print(f"  P={cusum_results['precision']:.4f}, R={cusum_results['recall']:.4f}, "
              f"F1={cusum_results['f1']:.4f}")

    # Save
    elapsed = time.time() - t0
    results = {
        "data_path": args.data_path,
        "n_examples": len(sequences),
        "n_layers": L, "k": k, "input_dim": L * k,
        "latent_dim": args.latent_dim,
        "n_params": sum(p.numel() for p in model.parameters()),
        "delta_auroc": auroc,
        "cusum": cusum_results,
        "elapsed_seconds": elapsed,
        "args": vars(args),
    }
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    torch.save({"mu": mu, "std": std, "meta": meta},
               os.path.join(args.output_dir, "normalization.pt"))

    print(f"\n{'='*60}")
    print(f"Saved to {out_path}")
    print(f"Total: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
