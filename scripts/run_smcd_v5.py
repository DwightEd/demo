#!/usr/bin/env python3
"""
SMCD v5: Information-Geometric E×C×N Features + GRU Conditional Density.

Theory-driven pipeline:
    1. Multi-layer spectra → probability simplex (normalization)
    2. Information-geometric features: E(effective rank), C(Hellinger distance), N(delta rank)
    3. GRU conditional density on E/C/N profiles: p(f_j | f_{<j})
    4. Anomaly score δ_j = -log p(f_j | f_{<j})
    5. Also compute direct S_j = E_j × C_j × N_j as interpretable baseline
    6. CUSUM aggregation for sequence-level detection

Every feature has clear mathematical meaning grounded in information geometry.
No arbitrary architecture choices.

Usage:
    python scripts/run_smcd_v5.py --data_path pilot/results/gsm8k_multilayer.pt
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

from smcd.info_geometry import compute_ecn_features, compute_constraint_scores
from smcd.density import ConditionalDensity
from smcd.detector import CUSUMDetector, evaluate_detection


def parse_args():
    p = argparse.ArgumentParser(description="SMCD v5: E×C×N + GRU density")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="smcd_v5_output")
    p.add_argument("--seed", type=int, default=42)
    # GRU density model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--weight_decay", type=float, default=1e-4)
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


def compute_normalization(features_list, train_idx):
    feats = [features_list[i] for i in train_idx]
    all_f = np.concatenate(feats, axis=0)
    mu = all_f.mean(axis=0)
    std = all_f.std(axis=0)
    std[std < 1e-8] = 1.0
    return mu, std


def train_gru_density(features_list, labels_list, example_labels,
                      train_idx, val_idx, mu, std, feat_dim, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Training GRU conditional density on E/C/N features")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Feature dim: {feat_dim}, Hidden: {args.hidden_dim}, Layers: {args.n_layers}")

    def get_correct_seqs(indices):
        seqs = []
        for i in indices:
            if example_labels[i] == -1:
                f = (features_list[i] - mu) / std
                seqs.append(torch.tensor(f, dtype=torch.float32))
        return seqs

    train_seqs = get_correct_seqs(train_idx)
    val_seqs = get_correct_seqs(val_idx)
    print(f"  Train: {len(train_seqs)} correct trajectories")
    print(f"  Val: {len(val_seqs)} correct trajectories")

    if not train_seqs:
        return None

    model = ConditionalDensity(
        input_dim=feat_dim, hidden_dim=args.hidden_dim,
        n_layers=args.n_layers, dropout=args.dropout,
        cov_type="diag",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

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

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for seq in val_seqs:
                s = seq.unsqueeze(0).to(device)
                l = torch.tensor([seq.shape[0]]).to(device)
                m = torch.ones(1, seq.shape[0]).to(device)
                nll = model.nll_loss(s, l, m)
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
                  f"best={best_val:.4f}")

        if no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  Best val NLL: {best_val:.4f}")
    return model


def compute_delta_scores(model, features_list, mu, std, device):
    model.eval()
    delta_list = []
    with torch.no_grad():
        for feats in features_list:
            f_norm = (feats - mu) / std
            seq = torch.tensor(f_norm, dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([feats.shape[0]]).to(device)
            delta = model.compute_anomaly_scores(seq, lengths)
            delta_list.append(delta[0].cpu().numpy())
    return delta_list


def evaluate_step_auroc(scores_list, labels_list, example_labels, test_idx, name=""):
    correct_scores = []
    error_scores = []

    for i in test_idx:
        label = example_labels[i]
        scores = scores_list[i]

        if label == -1:
            correct_scores.extend(scores[1:].tolist())
        else:
            for j in range(1, len(scores)):
                if j < label:
                    correct_scores.append(scores[j])
                elif j == label:
                    error_scores.append(scores[j])

    if not error_scores or not correct_scores:
        print(f"  [{name}] Insufficient data")
        return None

    y_true = [0] * len(correct_scores) + [1] * len(error_scores)
    y_score = correct_scores + error_scores
    auroc = roc_auc_score(y_true, y_score)

    c, e = np.array(correct_scores), np.array(error_scores)
    sep = (e.mean() - c.mean()) / (c.std() + 1e-8)
    print(f"  [{name}] AUROC={auroc:.4f} | correct={len(c)}, error={len(e)} | "
          f"separation={sep:.2f}σ")
    return auroc


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # ── Load data ──
    print(f"{'='*60}")
    print("Loading multi-layer spectral data")
    print(f"{'='*60}")
    raw = torch.load(args.data_path, weights_only=False)
    data, meta = raw["examples"], raw["meta"]
    L = len(meta["layer_indices"])
    k = meta["k"]
    n_correct = sum(1 for d in data if d["label"] == -1)
    print(f"  {len(data)} examples (correct={n_correct}, error={len(data)-n_correct})")
    print(f"  {L} layers x {k} SVs")

    # ── Compute E×C×N features ──
    print(f"\n{'='*60}")
    print("Computing information-geometric E×C×N features")
    print(f"{'='*60}")
    features_list, labels_list, example_labels = compute_ecn_features(data, meta)
    feat_dim = features_list[0].shape[1]
    print(f"  Feature dim: {feat_dim} (= 3 × {L} layers)")
    print(f"    E: effective rank per layer (dims 0..{L-1})")
    print(f"    C: Hellinger distance per layer (dims {L}..{2*L-1})")
    print(f"    N: delta effective rank per layer (dims {2*L}..{3*L-1})")
    print(f"  Usable examples: {len(features_list)}")

    # ── Split ──
    indices = np.arange(len(features_list))
    np.random.shuffle(indices)
    n_train = int(len(indices) * args.train_ratio)
    n_val = int(len(indices) * args.val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ── Direct S_j = E×C×N baseline (no learning) ──
    print(f"\n{'='*60}")
    print("Direct E×C×N constraint scores (no learning)")
    print(f"{'='*60}")
    s_list = []
    for feats in features_list:
        S_j = compute_constraint_scores(feats, L)
        # Anomaly = 1 - S_j (low constraint score = anomalous)
        s_list.append(1.0 - S_j)

    auroc_direct = evaluate_step_auroc(
        s_list, labels_list, example_labels, test_idx, name="S_j=E×C×N")

    # ── Evaluate individual components ──
    print(f"\n  Component-wise AUROC:")
    for comp_name, start, end in [("E (eff_rank)", 0, L), ("C (Hellinger)", L, 2*L), ("N (delta_rank)", 2*L, 3*L)]:
        comp_list = []
        for feats in features_list:
            # Use mean across layers as score
            comp_score = feats[:, start:end].mean(axis=1)
            comp_list.append(comp_score)
        evaluate_step_auroc(comp_list, labels_list, example_labels, test_idx, name=comp_name)

    # ── Normalize for GRU ──
    mu, std = compute_normalization(features_list, train_idx)

    # ── Train GRU conditional density ──
    model = train_gru_density(
        features_list, labels_list, example_labels,
        train_idx, val_idx, mu, std, feat_dim, args)

    if model is None:
        print("Training failed.")
        return

    # ── Compute anomaly scores ──
    print(f"\n{'='*60}")
    print("Computing GRU conditional anomaly scores")
    print(f"{'='*60}")
    delta_list = compute_delta_scores(model, features_list, mu, std, device)
    all_d = np.concatenate(delta_list)
    print(f"  Scores: mean={all_d.mean():.4f}, std={all_d.std():.4f}, "
          f"median={np.median(all_d):.4f}")

    # ── Step-level AUROC ──
    print(f"\n{'='*60}")
    print("Step-level AUROC comparison")
    print(f"{'='*60}")
    auroc_gru = evaluate_step_auroc(
        delta_list, labels_list, example_labels, test_idx, name="GRU δ_j")

    print(f"\n  Summary:")
    print(f"    Direct S_j=E×C×N:    {auroc_direct:.4f}" if auroc_direct else "    Direct: N/A")
    print(f"    GRU conditional:     {auroc_gru:.4f}" if auroc_gru else "    GRU: N/A")
    print(f"    Baseline (eff_rank): 0.694")

    # ── CUSUM on GRU scores ──
    print(f"\n{'='*60}")
    print("CUSUM sequence-level detection")
    print(f"{'='*60}")
    cal = [delta_list[i] for i in train_idx if example_labels[i] == -1]
    cusum_results = None
    if cal:
        detector = CUSUMDetector(threshold=args.cusum_threshold)
        detector.calibrate(cal)
        print(f"  k = {detector.k:.4f}")
        test_scores = [delta_list[i] for i in test_idx]
        test_labels = [example_labels[i] for i in test_idx]
        cusum_results = evaluate_detection(detector, test_scores, test_labels)
        print(f"  TP={cusum_results['TP']}, FP={cusum_results['FP']}, "
              f"TN={cusum_results['TN']}, FN={cusum_results['FN']}")
        print(f"  P={cusum_results['precision']:.4f}, R={cusum_results['recall']:.4f}, "
              f"F1={cusum_results['f1']:.4f}")

    # ── Save ──
    elapsed = time.time() - t0
    results = {
        "n_examples": len(features_list),
        "feat_dim": feat_dim,
        "n_layers": L, "k": k,
        "auroc_direct_ecn": auroc_direct,
        "auroc_gru": auroc_gru,
        "cusum": cusum_results,
        "elapsed_seconds": elapsed,
        "args": vars(args),
    }
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "density_model.pt"))
    torch.save({"mu": mu, "std": std, "meta": meta},
               os.path.join(args.output_dir, "normalization.pt"))

    print(f"\n{'='*60}")
    print(f"Saved to {out_path}")
    print(f"Total: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
