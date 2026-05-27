#!/usr/bin/env python3
"""
SMCD v3: Multi-layer Spectral Trajectory pipeline.

Key difference from v2: uses per-layer singular values from ALL transformer layers,
giving much richer per-step representations (L*k dims instead of k dims).

Usage:
    python scripts/run_smcd_v3.py --data_path pilot/results/gsm8k_multilayer.pt

Pipeline:
    1. Load multi-layer spectral data (sigma_{j,l} per step per layer)
    2. Build representations: [sigma_flat, delta_sigma_flat] per step
    3. Train conditional density model p(t_j | t_{<j}) on correct trajectories
    4. Compute anomaly scores delta_j = -log p(t_j | t_{<j})
    5. Evaluate: step-level AUROC
    6. CUSUM sequence-level detection
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

from smcd.density import ConditionalDensity
from smcd.detector import CUSUMDetector, evaluate_detection


def parse_args():
    p = argparse.ArgumentParser(description="SMCD v3 multi-layer pipeline")
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to multilayer.pt (from 01c_extract_multilayer.py)")
    p.add_argument("--output_dir", type=str, default="smcd_v3_output")
    p.add_argument("--seed", type=int, default=42)
    # Representation
    p.add_argument("--use_delta", action="store_true", default=True,
                   help="Include delta_sigma in representation")
    p.add_argument("--no_delta", action="store_true",
                   help="Disable delta_sigma")
    # Density model
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    # Detection
    p.add_argument("--cusum_threshold", type=float, default=5.0)
    # Split
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    args = p.parse_args()
    if args.no_delta:
        args.use_delta = False
    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_multilayer_data(path):
    """Load multi-layer spectral data."""
    raw = torch.load(path, weights_only=False)
    data = raw["examples"]
    meta = raw["meta"]
    return data, meta


def build_representations(data, meta, use_delta=True):
    """Build per-step feature vectors from multi-layer sigma.

    Each step: sigma_ml is (L, k).
    Representation: [sigma_flat] or [sigma_flat, delta_sigma_flat]
    """
    L = len(meta["layer_indices"])
    k = meta["k"]

    features_list = []
    labels_list = []
    example_labels = []

    for ex in data:
        steps = ex["steps"]
        T = len(steps)
        if T < 2:
            continue

        t_list = []
        for j in range(T):
            sigma_flat = steps[j]["sigma_ml"].flatten()  # (L*k,)
            parts = [sigma_flat]

            if use_delta:
                if j == 0:
                    parts.append(torch.zeros(L * k))
                else:
                    prev_flat = steps[j - 1]["sigma_ml"].flatten()
                    parts.append(sigma_flat - prev_flat)

            t_list.append(torch.cat(parts))

        features = torch.stack(t_list).numpy()
        labels = np.array([s["is_error"] for s in steps], dtype=np.float32)

        features_list.append(features)
        labels_list.append(labels)
        example_labels.append(ex["label"])

    return features_list, labels_list, example_labels


def compute_normalization(features_list):
    all_feats = np.concatenate(features_list, axis=0)
    mu = all_feats.mean(axis=0)
    sigma = all_feats.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_density_model(features_list, labels_list, example_labels,
                        train_idx, val_idx, mu, sigma, feat_dim, args):
    print(f"\n{'='*60}")
    print("Training conditional density model")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Input dim: {feat_dim}, Hidden dim: {args.hidden_dim}, "
          f"Layers: {args.n_layers}, Dropout: {args.dropout}")

    def get_correct_sequences(indices):
        seqs = []
        for i in indices:
            if example_labels[i] == -1:
                f = (features_list[i] - mu) / sigma
                seqs.append(torch.tensor(f, dtype=torch.float32))
        return seqs

    train_seqs = get_correct_sequences(train_idx)
    val_seqs = get_correct_sequences(val_idx)
    print(f"  Training: {len(train_seqs)} correct trajectories")
    print(f"  Validation: {len(val_seqs)} correct trajectories")

    if len(train_seqs) == 0:
        print("  [ERROR] No correct trajectories!")
        return None

    model = ConditionalDensity(
        input_dim=feat_dim, hidden_dim=args.hidden_dim,
        n_layers=args.n_layers, dropout=args.dropout,
        cov_type="diag",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_state = None
    patience = 30
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(train_seqs)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(train_seqs), args.batch_size):
            batch = train_seqs[batch_start:batch_start + args.batch_size]
            lengths = torch.tensor([s.shape[0] for s in batch])
            max_len = lengths.max().item()

            B = len(batch)
            padded = torch.zeros(B, max_len, feat_dim)
            mask = torch.zeros(B, max_len)
            for i, seq in enumerate(batch):
                T = seq.shape[0]
                padded[i, :T] = seq
                mask[i, :T] = 1.0

            padded = padded.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)

            loss = model.nll_loss(padded, lengths, mask)

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
        val_nll = 0.0
        val_n = 0
        with torch.no_grad():
            for seq in val_seqs:
                s = seq.unsqueeze(0).to(device)
                l = torch.tensor([seq.shape[0]]).to(device)
                m = torch.ones(1, seq.shape[0]).to(device)
                nll = model.nll_loss(s, l, m)
                val_nll += nll.item()
                val_n += 1

        avg_val = val_nll / max(val_n, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train={avg_train:.4f}, val={avg_val:.4f}, "
                  f"best_val={best_val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  Best val NLL: {best_val_loss:.4f}")

    return model


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def compute_scores(model, features_list, mu, sigma, device):
    print(f"\n{'='*60}")
    print("Computing anomaly scores")
    print(f"{'='*60}")

    model.eval()
    delta_list = []

    with torch.no_grad():
        for feats in features_list:
            f_norm = (feats - mu) / sigma
            seq = torch.tensor(f_norm, dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([feats.shape[0]]).to(device)
            delta = model.compute_anomaly_scores(seq, lengths)
            delta_list.append(delta[0].cpu().numpy())

    all_d = np.concatenate(delta_list)
    print(f"  Scores: mean={all_d.mean():.4f}, std={all_d.std():.4f}, "
          f"median={np.median(all_d):.4f}, max={all_d.max():.4f}")
    return delta_list


def evaluate_auroc(delta_list, labels_list, example_labels, test_idx):
    print(f"\n{'='*60}")
    print("Step-level AUROC")
    print(f"{'='*60}")

    correct_scores = []
    error_scores = []

    for i in test_idx:
        label = example_labels[i]
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
    print(f"  (Baseline: handcrafted effective_rank AUROC ~ 0.694)")

    # All-step AUROC
    all_correct = []
    all_error = []
    for i in test_idx:
        label = example_labels[i]
        deltas = delta_list[i]
        labs = labels_list[i]
        for j in range(1, len(deltas)):
            if labs[j] == 0:
                all_correct.append(deltas[j])
            else:
                all_error.append(deltas[j])

    if all_error and all_correct:
        y_true2 = [0] * len(all_correct) + [1] * len(all_error)
        y_score2 = all_correct + all_error
        auroc_all = roc_auc_score(y_true2, y_score2)
        print(f"  All-step AUROC: {auroc_all:.4f}")

    # Score statistics
    c_arr = np.array(correct_scores)
    e_arr = np.array(error_scores)
    print(f"  Correct score: mean={c_arr.mean():.4f}, std={c_arr.std():.4f}")
    print(f"  Error score:   mean={e_arr.mean():.4f}, std={e_arr.std():.4f}")
    print(f"  Separation:    {(e_arr.mean() - c_arr.mean()) / (c_arr.std() + 1e-8):.4f} sigma")

    return auroc


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # 1. Load data
    print(f"{'='*60}")
    print("Loading multi-layer spectral data")
    print(f"{'='*60}")
    data, meta = load_multilayer_data(args.data_path)
    n_correct = sum(1 for d in data if d["label"] == -1)
    n_error = len(data) - n_correct
    L = len(meta["layer_indices"])
    k = meta["k"]
    print(f"  {len(data)} examples (correct={n_correct}, error={n_error})")
    print(f"  {L} layers x {k} singular values = {L*k} base dims")
    print(f"  use_delta={args.use_delta}")

    # 2. Build representations
    features_list, labels_list, example_labels = build_representations(
        data, meta, use_delta=args.use_delta)
    feat_dim = features_list[0].shape[1]
    print(f"  Feature dim: {feat_dim}")
    print(f"  Usable examples: {len(features_list)}")

    # Split
    indices = np.arange(len(features_list))
    np.random.shuffle(indices)
    n_train = int(len(indices) * args.train_ratio)
    n_val = int(len(indices) * args.val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Normalize
    train_feats = [features_list[i] for i in train_idx]
    mu, sigma = compute_normalization(train_feats)

    # 3. Train
    model = train_density_model(
        features_list, labels_list, example_labels,
        train_idx, val_idx, mu, sigma, feat_dim, args)

    if model is None:
        print("Training failed.")
        return

    # 4. Score
    delta_list = compute_scores(model, features_list, mu, sigma, device)

    # 5. AUROC
    auroc = evaluate_auroc(delta_list, labels_list, example_labels, test_idx)

    # 6. CUSUM
    print(f"\n{'='*60}")
    print("CUSUM sequence-level detection")
    print(f"{'='*60}")
    cal_scores = [delta_list[i] for i in train_idx if example_labels[i] == -1]
    if cal_scores:
        detector = CUSUMDetector(threshold=args.cusum_threshold)
        detector.calibrate(cal_scores)
        print(f"  CUSUM k = {detector.k:.4f}")
        test_scores = [delta_list[i] for i in test_idx]
        test_labels = [example_labels[i] for i in test_idx]
        cusum_results = evaluate_detection(detector, test_scores, test_labels)
        print(f"  TP={cusum_results['TP']}, FP={cusum_results['FP']}, "
              f"TN={cusum_results['TN']}, FN={cusum_results['FN']}")
        print(f"  Precision={cusum_results['precision']:.4f}, "
              f"Recall={cusum_results['recall']:.4f}, F1={cusum_results['f1']:.4f}")
    else:
        cusum_results = None

    # Save
    elapsed = time.time() - t0
    results = {
        "data_path": args.data_path,
        "n_examples": len(features_list),
        "feat_dim": feat_dim,
        "n_layers": L,
        "k": k,
        "use_delta": args.use_delta,
        "hidden_dim": args.hidden_dim,
        "model_layers": args.n_layers,
        "delta_auroc": auroc,
        "cusum": cusum_results,
        "elapsed_seconds": elapsed,
        "args": vars(args),
    }

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "density_model.pt"))
    torch.save({"mu": mu, "sigma": sigma, "meta": meta},
               os.path.join(args.output_dir, "normalization.pt"))

    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
