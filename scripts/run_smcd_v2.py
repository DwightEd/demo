#!/usr/bin/env python3
"""
SMCD v2: Grassmannian Spectral Trajectory pipeline.

Usage:
    python scripts/run_smcd_v2.py --data_path results/gsm8k_subspaces.pt

Pipeline:
    1. Load subspace data (V_j, sigma_j per step)
    2. Learn tangent space PCA from correct trajectories
    3. Compute transition representations t_j = [v_j, sigma_j, delta_sigma_j]
    4. Train conditional density model p(t_j | t_{<j}) on correct trajectories
    5. Compute anomaly scores delta_j = -log p(t_j | t_{<j})
    6. Evaluate: step-level AUROC (the make-or-break test)
    7. Optional: CUSUM sequence-level detection
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

from smcd.representation import load_subspaces, learn_tangent_pca, compute_representations, compute_normalization
from smcd.density import ConditionalDensity
from smcd.detector import CUSUMDetector, evaluate_detection


def parse_args():
    p = argparse.ArgumentParser(description="SMCD v2 pipeline")
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to subspaces.pt (from 01b_extract_subspaces.py)")
    p.add_argument("--output_dir", type=str, default="smcd_v2_output")
    p.add_argument("--seed", type=int, default=42)
    # Representation
    p.add_argument("--pca_dim", type=int, default=16,
                   help="Tangent vector PCA dimension (r)")
    # Density model
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
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


# ──────────────────────────────────────────────
# Stage 1: Load data & learn representations
# ──────────────────────────────────────────────

def stage_load_and_represent(args):
    print("=" * 60)
    print("Stage 1: Load subspaces & build representations")
    print("=" * 60)

    data = load_subspaces(args.data_path)
    n_correct = sum(1 for d in data if d["label"] == -1)
    n_error = len(data) - n_correct
    k = data[0]["steps"][0]["V"].shape[1]
    d = data[0]["steps"][0]["V"].shape[0]
    print(f"  Loaded {len(data)} examples (correct={n_correct}, error={n_error})")
    print(f"  Subspace: Gr({k}, {d})")

    # Learn tangent PCA
    pca = learn_tangent_pca(data, n_components=args.pca_dim)

    # Compute representations
    features_list, labels_list, example_labels = compute_representations(data, pca)
    feat_dim = features_list[0].shape[1]
    print(f"  Representation dim: {feat_dim} (r={args.pca_dim}, k={k})")
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

    # Normalization from training set
    train_feats = [features_list[i] for i in train_idx]
    mu, sigma = compute_normalization(train_feats)

    return (features_list, labels_list, example_labels,
            train_idx, val_idx, test_idx, mu, sigma, pca, feat_dim)


# ──────────────────────────────────────────────
# Stage 2: Train conditional density model
# ──────────────────────────────────────────────

def stage_train_density(features_list, labels_list, example_labels,
                        train_idx, val_idx, mu, sigma, feat_dim, args):
    print("\n" + "=" * 60)
    print("Stage 2: Train conditional density model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Correct trajectories for training
    def get_correct_sequences(indices):
        seqs = []
        for i in indices:
            if example_labels[i] == -1:
                f = (features_list[i] - mu) / sigma
                seqs.append(torch.tensor(f, dtype=torch.float32))
        return seqs

    train_seqs = get_correct_sequences(train_idx)
    val_seqs = get_correct_sequences(val_idx)
    print(f"  Training on {len(train_seqs)} correct trajectories")
    print(f"  Validation: {len(val_seqs)} correct trajectories")

    if len(train_seqs) == 0:
        print("  [ERROR] No correct trajectories!")
        return None

    model = ConditionalDensity(
        input_dim=feat_dim, hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(train_seqs)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(train_seqs), args.batch_size):
            batch = train_seqs[batch_start:batch_start + args.batch_size]
            lengths = torch.tensor([s.shape[0] for s in batch])
            max_len = lengths.max().item()

            # Pad
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

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_nll={avg_train:.4f}, val_nll={avg_val:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  Best val NLL: {best_val_loss:.4f}")

    return model


# ──────────────────────────────────────────────
# Stage 3: Compute anomaly scores
# ──────────────────────────────────────────────

def stage_compute_scores(model, features_list, mu, sigma, device):
    print("\n" + "=" * 60)
    print("Stage 3: Compute anomaly scores")
    print("=" * 60)

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
    print(f"  Anomaly scores: mean={all_d.mean():.4f}, std={all_d.std():.4f}, "
          f"max={all_d.max():.4f}")
    return delta_list


# ──────────────────────────────────────────────
# Stage 4: Step-level AUROC (make-or-break)
# ──────────────────────────────────────────────

def stage_evaluate_auroc(delta_list, labels_list, example_labels, test_idx):
    print("\n" + "=" * 60)
    print("Stage 4: Step-level AUROC (make-or-break test)")
    print("=" * 60)

    # Correct-vs-first-error evaluation (same as pilot/02_evaluate.py)
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
        print("  [WARN] Insufficient data for AUROC")
        return None

    y_true = [0] * len(correct_scores) + [1] * len(error_scores)
    y_score = correct_scores + error_scores
    auroc = roc_auc_score(y_true, y_score)

    print(f"  Correct steps: {len(correct_scores)}, First-error steps: {len(error_scores)}")
    print(f"  >>> delta_j AUROC = {auroc:.4f} <<<")
    print(f"  (Baseline: handcrafted step_effective_rank AUROC ~ 0.694)")

    # Also compute all-step AUROC (correct vs ALL error steps)
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
        print(f"  All-step AUROC (correct vs all errors): {auroc_all:.4f}")

    return auroc


# ──────────────────────────────────────────────
# Stage 5: CUSUM sequence-level detection
# ──────────────────────────────────────────────

def stage_cusum(delta_list, example_labels, train_idx, test_idx, args):
    print("\n" + "=" * 60)
    print("Stage 5: CUSUM sequence-level detection")
    print("=" * 60)

    # Calibrate on correct training trajectories
    cal_scores = []
    for i in train_idx:
        if example_labels[i] == -1:
            cal_scores.append(delta_list[i])

    if not cal_scores:
        print("  [WARN] No correct trajectories for calibration")
        return None

    detector = CUSUMDetector(threshold=args.cusum_threshold)
    detector.calibrate(cal_scores)
    print(f"  CUSUM k = {detector.k:.4f}")

    # Evaluate
    test_scores = [delta_list[i] for i in test_idx]
    test_labels = [example_labels[i] for i in test_idx]
    results = evaluate_detection(detector, test_scores, test_labels)

    print(f"  Test: {results['n']} examples")
    print(f"    TP={results['TP']}, FP={results['FP']}, "
          f"TN={results['TN']}, FN={results['FN']}")
    print(f"    Precision={results['precision']:.4f}, "
          f"Recall={results['recall']:.4f}, F1={results['f1']:.4f}")
    if results['avg_detection_delay'] is not None:
        print(f"    Avg detection delay: {results['avg_detection_delay']:.2f} steps")

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # 1. Load & represent
    (features_list, labels_list, example_labels,
     train_idx, val_idx, test_idx, mu, sigma, pca, feat_dim) = \
        stage_load_and_represent(args)

    # 2. Train density model
    model = stage_train_density(
        features_list, labels_list, example_labels,
        train_idx, val_idx, mu, sigma, feat_dim, args)

    if model is None:
        print("Training failed.")
        return

    # 3. Compute anomaly scores
    delta_list = stage_compute_scores(model, features_list, mu, sigma, device)

    # 4. Step-level AUROC
    auroc = stage_evaluate_auroc(delta_list, labels_list, example_labels, test_idx)

    # 5. CUSUM
    cusum_results = stage_cusum(delta_list, example_labels, train_idx, test_idx, args)

    # Save results
    elapsed = time.time() - t0
    results = {
        "data_path": args.data_path,
        "n_examples": len(features_list),
        "feat_dim": feat_dim,
        "pca_dim": args.pca_dim,
        "delta_auroc": auroc,
        "cusum": cusum_results,
        "elapsed_seconds": elapsed,
        "args": vars(args),
    }

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "density_model.pt"))
    torch.save({"pca": pca, "mu": mu, "sigma": sigma},
               os.path.join(args.output_dir, "representation.pt"))

    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
