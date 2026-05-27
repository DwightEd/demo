#!/usr/bin/env python3
"""
SMCD: One-click train + evaluate pipeline.

Usage:
    python scripts/run_smcd.py --data_path pilot/results/gsm8k_geometry.jsonl

Pipeline:
    1. Load geometry JSONL → 5-dim feature vectors
    2. Split: correct trajectories for kernel training, all for probe
    3. Auto-calibrate constraint score thresholds
    4. Train transition kernel on correct trajectories (NLL)
    5. Compute deviation scores delta_j for all examples
    6. Train constraint-aware probe (BCE + lambda * L_cst)
    7. Evaluate: step-level AUROC + CUSUM sequence detection
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smcd.features import load_geometry, compute_global_stats, FeatureConfig
from smcd.constraint_score import ConstraintScore
from smcd.transition_kernel import TransitionKernel
from smcd.probe import ConstraintProbe, constraint_loss
from smcd.detector import CUSUMDetector, evaluate_detection
from smcd.dataset import SMCDDataset, collate_sequences


def parse_args():
    p = argparse.ArgumentParser(description="SMCD pipeline")
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to geometry JSONL (e.g. pilot/results/gsm8k_geometry.jsonl)")
    p.add_argument("--output_dir", type=str, default="smcd_output")
    p.add_argument("--seed", type=int, default=42)
    # Kernel
    p.add_argument("--kernel_hidden", type=int, default=64)
    p.add_argument("--kernel_layers", type=int, default=2)
    p.add_argument("--kernel_epochs", type=int, default=50)
    p.add_argument("--kernel_lr", type=float, default=1e-3)
    p.add_argument("--kernel_batch", type=int, default=32)
    # Probe
    p.add_argument("--probe_hidden", type=int, default=32)
    p.add_argument("--probe_epochs", type=int, default=30)
    p.add_argument("--probe_lr", type=float, default=1e-3)
    p.add_argument("--probe_batch", type=int, default=64)
    p.add_argument("--probe_lambda", type=float, default=0.1)
    # Detector
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
# Stage 1: Load & split data
# ──────────────────────────────────────────────

def load_and_split(args):
    print("=" * 60)
    print("Stage 1: Loading data")
    print("=" * 60)

    cfg = FeatureConfig()
    features_list, labels_list, example_labels = load_geometry(args.data_path, cfg)
    print(f"  Loaded {len(features_list)} examples, feature dim = {cfg.feature_keys}")

    n_correct = sum(1 for l in example_labels if l == -1)
    n_error = len(example_labels) - n_correct
    print(f"  Correct: {n_correct}, With errors: {n_error}")

    # Shuffle and split
    indices = np.arange(len(features_list))
    np.random.shuffle(indices)

    n_train = int(len(indices) * args.train_ratio)
    n_val = int(len(indices) * args.val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return features_list, labels_list, example_labels, train_idx, val_idx, test_idx, cfg


# ──────────────────────────────────────────────
# Stage 2: Constraint score calibration
# ──────────────────────────────────────────────

def calibrate_constraints(features_list, labels_list, example_labels, train_idx):
    print("\n" + "=" * 60)
    print("Stage 2: Calibrating constraint scores")
    print("=" * 60)

    # Collect correct steps from training set
    correct_steps = []
    for i in train_idx:
        if example_labels[i] == -1:
            correct_steps.append(features_list[i])  # all steps correct
        else:
            # Only steps before error
            err = example_labels[i]
            if err > 0:
                correct_steps.append(features_list[i][:err])

    all_correct = np.concatenate(correct_steps, axis=0)
    print(f"  Correct steps for calibration: {len(all_correct)}")

    scorer = ConstraintScore(sharpness=5.0)
    scorer.calibrate(all_correct)

    # Show threshold values
    for name, (center, scale) in scorer.thresholds.items():
        print(f"    {name}: center={center:.4f}, scale={scale:.4f}")

    return scorer, all_correct


# ──────────────────────────────────────────────
# Stage 3: Train transition kernel
# ──────────────────────────────────────────────

def train_kernel(features_list, labels_list, example_labels, train_idx, val_idx,
                 mu, sigma, args):
    print("\n" + "=" * 60)
    print("Stage 3: Training transition kernel")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Correct trajectories only for kernel training
    train_correct = []
    for i in train_idx:
        if example_labels[i] == -1:
            f = (features_list[i] - mu) / sigma
            train_correct.append(torch.tensor(f, dtype=torch.float32))

    val_correct = []
    for i in val_idx:
        if example_labels[i] == -1:
            f = (features_list[i] - mu) / sigma
            val_correct.append(torch.tensor(f, dtype=torch.float32))

    print(f"  Kernel training: {len(train_correct)} correct trajectories")
    print(f"  Kernel validation: {len(val_correct)} correct trajectories")

    if len(train_correct) == 0:
        print("  [WARN] No correct trajectories for kernel training!")
        return None

    feat_dim = train_correct[0].shape[-1]
    kernel = TransitionKernel(
        feat_dim=feat_dim, hidden_dim=args.kernel_hidden,
        n_layers=args.kernel_layers
    ).to(device)

    optimizer = torch.optim.Adam(kernel.parameters(), lr=args.kernel_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.kernel_epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(args.kernel_epochs):
        kernel.train()
        np.random.shuffle(train_correct)

        # Simple batching: group by similar length
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(train_correct), args.kernel_batch):
            batch_seqs = train_correct[batch_start:batch_start + args.kernel_batch]
            lengths = torch.tensor([s.shape[0] for s in batch_seqs])
            max_len = lengths.max().item()

            # Pad
            B = len(batch_seqs)
            padded = torch.zeros(B, max_len, feat_dim)
            mask = torch.zeros(B, max_len)
            for i, seq in enumerate(batch_seqs):
                T = seq.shape[0]
                padded[i, :T] = seq
                mask[i, :T] = 1.0

            padded = padded.to(device)
            mask = mask.to(device)
            lengths_dev = lengths.to(device)

            mu_pred, L_pred = kernel(padded, lengths_dev)

            # NLL loss (shifted: predict step j from <j)
            target = padded[:, 1:]
            pred_mu = mu_pred[:, :-1]
            pred_L = L_pred[:, :-1]
            m = mask[:, 1:]

            diff = (target - pred_mu).unsqueeze(-1)
            v = torch.linalg.solve_triangular(pred_L, diff, upper=False)
            mahal = (v.squeeze(-1) ** 2).sum(dim=-1)
            log_det = 2.0 * pred_L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            import math
            nll = 0.5 * (mahal + log_det + feat_dim * math.log(2 * math.pi))
            loss = (nll * m).sum() / m.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(kernel.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        kernel.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for seq in val_correct:
                seq_t = seq.unsqueeze(0).to(device)
                length_t = torch.tensor([seq.shape[0]]).to(device)
                mu_pred, L_pred = kernel(seq_t, length_t)

                target = seq_t[:, 1:]
                pred_mu = mu_pred[:, :-1]
                pred_L = L_pred[:, :-1]
                diff = (target - pred_mu).unsqueeze(-1)
                v = torch.linalg.solve_triangular(pred_L, diff, upper=False)
                mahal = (v.squeeze(-1) ** 2).sum(dim=-1)
                log_det = 2.0 * pred_L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
                nll = 0.5 * (mahal + log_det + feat_dim * math.log(2 * math.pi))
                val_loss += nll.sum().item()
                val_n += nll.numel()

        avg_val = val_loss / max(val_n, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in kernel.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_nll={avg_loss:.4f}, val_nll={avg_val:.4f}")

    if best_state is not None:
        kernel.load_state_dict(best_state)
    kernel.eval()
    print(f"  Best val NLL: {best_val_loss:.4f}")

    return kernel


# ──────────────────────────────────────────────
# Stage 4: Compute deviation scores
# ──────────────────────────────────────────────

def compute_deviations(kernel, features_list, mu, sigma, device):
    print("\n" + "=" * 60)
    print("Stage 4: Computing deviation scores")
    print("=" * 60)

    delta_list = []
    kernel.eval()

    with torch.no_grad():
        for i, feats in enumerate(features_list):
            f_norm = (feats - mu) / sigma
            seq = torch.tensor(f_norm, dtype=torch.float32).unsqueeze(0).to(device)
            length = torch.tensor([feats.shape[0]]).to(device)

            delta = kernel.compute_deviation(seq, length)
            delta_list.append(delta[0].cpu().numpy())

    # Stats
    all_deltas = np.concatenate(delta_list)
    print(f"  Deviation scores: mean={all_deltas.mean():.4f}, "
          f"std={all_deltas.std():.4f}, max={all_deltas.max():.4f}")

    return delta_list


# ──────────────────────────────────────────────
# Stage 5: Evaluate delta_j AUROC (make-or-break test)
# ──────────────────────────────────────────────

def evaluate_delta_auroc(delta_list, labels_list, example_labels, test_idx):
    print("\n" + "=" * 60)
    print("Stage 5: delta_j standalone AUROC (make-or-break test)")
    print("=" * 60)

    # Step-level: correct vs first-error step (same as pilot/02_evaluate.py)
    correct_deltas = []
    error_deltas = []

    for i in test_idx:
        label = example_labels[i]
        deltas = delta_list[i]

        if label == -1:
            # All correct — use all steps (skip step 0, no prediction)
            correct_deltas.extend(deltas[1:].tolist())
        else:
            # Correct steps before error
            for j in range(1, len(deltas)):
                if j < label:
                    correct_deltas.append(deltas[j])
                elif j == label:
                    error_deltas.append(deltas[j])

    if len(error_deltas) == 0 or len(correct_deltas) == 0:
        print("  [WARN] Not enough data for AUROC")
        return None

    y_true = [0] * len(correct_deltas) + [1] * len(error_deltas)
    y_score = correct_deltas + error_deltas
    auroc = roc_auc_score(y_true, y_score)

    print(f"  Correct steps: {len(correct_deltas)}, Error steps: {len(error_deltas)}")
    print(f"  >>> delta_j AUROC = {auroc:.4f} <<<")
    print(f"  (Baseline: handcrafted step_effective_rank AUROC ~ 0.694)")

    return auroc


# ──────────────────────────────────────────────
# Stage 6: Train probe
# ──────────────────────────────────────────────

def train_probe(features_list, labels_list, delta_list, constraint_scores_list,
                train_idx, val_idx, mu, sigma, args):
    print("\n" + "=" * 60)
    print("Stage 6: Training constraint-aware probe")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim = features_list[0].shape[-1]

    # Build datasets
    train_feats = [features_list[i] for i in train_idx]
    train_labs = [labels_list[i] for i in train_idx]
    train_cst = [constraint_scores_list[i] for i in train_idx]
    train_delta = [delta_list[i] for i in train_idx]

    val_feats = [features_list[i] for i in val_idx]
    val_labs = [labels_list[i] for i in val_idx]
    val_cst = [constraint_scores_list[i] for i in val_idx]
    val_delta = [delta_list[i] for i in val_idx]

    probe = ConstraintProbe(feat_dim=feat_dim, hidden_dim=args.probe_hidden).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.probe_lr, weight_decay=1e-4)

    best_val_auroc = 0.0
    best_state = None

    for epoch in range(args.probe_epochs):
        probe.train()
        perm = np.random.permutation(len(train_feats))
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(perm), args.probe_batch):
            batch_idx = perm[batch_start:batch_start + args.probe_batch]

            # Flatten all steps into a single batch (simpler than sequence padding)
            all_f, all_d, all_l, all_c = [], [], [], []
            for bi in batch_idx:
                f_norm = (train_feats[bi] - mu) / sigma
                T = len(f_norm)
                all_f.append(f_norm)
                all_d.append(train_delta[bi][:T])
                all_l.append(train_labs[bi][:T])
                all_c.append(train_cst[bi][:T])

            f_t = torch.tensor(np.concatenate(all_f), dtype=torch.float32).to(device)
            d_t = torch.tensor(np.concatenate(all_d), dtype=torch.float32).to(device)
            l_t = torch.tensor(np.concatenate(all_l), dtype=torch.float32).to(device)
            c_t = torch.tensor(np.concatenate(all_c), dtype=torch.float32).to(device)

            logits = probe(f_t, d_t)
            loss, bce, cst = constraint_loss(logits, l_t, c_t,
                                              lam=args.probe_lambda)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation AUROC
        probe.eval()
        with torch.no_grad():
            all_logits, all_labels = [], []
            for bi in range(len(val_feats)):
                f_norm = (val_feats[bi] - mu) / sigma
                T = len(f_norm)
                f_t = torch.tensor(f_norm, dtype=torch.float32).to(device)
                d_t = torch.tensor(val_delta[bi][:T], dtype=torch.float32).to(device)
                logits = probe(f_t, d_t)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(val_labs[bi][:T])

            y_score = np.concatenate(all_logits)
            y_true = np.concatenate(all_labels)

            if len(np.unique(y_true)) > 1:
                val_auroc = roc_auc_score(y_true, y_score)
            else:
                val_auroc = 0.5

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, val_AUROC={val_auroc:.4f}")

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    print(f"  Best val AUROC: {best_val_auroc:.4f}")

    return probe


# ──────────────────────────────────────────────
# Stage 7: CUSUM sequence-level detection
# ──────────────────────────────────────────────

def evaluate_cusum(probe, features_list, labels_list, delta_list,
                   example_labels, train_idx, test_idx, mu, sigma, args):
    print("\n" + "=" * 60)
    print("Stage 7: CUSUM sequence-level detection")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe.eval()

    def get_probe_scores(indices):
        scores = []
        with torch.no_grad():
            for i in indices:
                f_norm = (features_list[i] - mu) / sigma
                T = len(f_norm)
                f_t = torch.tensor(f_norm, dtype=torch.float32).to(device)
                d_t = torch.tensor(delta_list[i][:T], dtype=torch.float32).to(device)
                logits = probe(f_t, d_t)
                scores.append(torch.sigmoid(logits).cpu().numpy())
        return scores

    # Calibrate CUSUM on correct training trajectories
    correct_train_idx = [i for i in train_idx if example_labels[i] == -1]
    cal_scores = get_probe_scores(correct_train_idx)

    detector = CUSUMDetector(threshold=args.cusum_threshold)
    detector.calibrate(cal_scores)
    print(f"  CUSUM k (reference value): {detector.k:.4f}")

    # Evaluate on test set
    test_scores = get_probe_scores(test_idx)
    test_labels = [example_labels[i] for i in test_idx]

    results = evaluate_detection(detector, test_scores, test_labels)

    print(f"  Test set: {results['n']} examples")
    print(f"    TP={results['TP']}, FP={results['FP']}, "
          f"TN={results['TN']}, FN={results['FN']}")
    print(f"    Precision: {results['precision']:.4f}")
    print(f"    Recall:    {results['recall']:.4f}")
    print(f"    F1:        {results['f1']:.4f}")
    if results['avg_detection_delay'] is not None:
        print(f"    Avg detection delay: {results['avg_detection_delay']:.2f} steps")

    return results


# ──────────────────────────────────────────────
# Stage 8: Full step-level AUROC (probe output)
# ──────────────────────────────────────────────

def evaluate_probe_auroc(probe, features_list, labels_list, delta_list,
                          test_idx, mu, sigma):
    print("\n" + "=" * 60)
    print("Stage 8: Probe step-level AUROC")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for i in test_idx:
            f_norm = (features_list[i] - mu) / sigma
            T = len(f_norm)
            f_t = torch.tensor(f_norm, dtype=torch.float32).to(device)
            d_t = torch.tensor(delta_list[i][:T], dtype=torch.float32).to(device)
            logits = probe(f_t, d_t)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels_list[i][:T])

    y_score = np.concatenate(all_logits)
    y_true = np.concatenate(all_labels)

    if len(np.unique(y_true)) < 2:
        print("  [WARN] Only one class in test set")
        return None

    auroc = roc_auc_score(y_true, y_score)
    print(f"  Step-level AUROC (probe): {auroc:.4f}")
    return auroc


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()

    # 1. Load & split
    features_list, labels_list, example_labels, train_idx, val_idx, test_idx, cfg = \
        load_and_split(args)

    # 2. Constraint score calibration
    scorer, correct_steps = calibrate_constraints(
        features_list, labels_list, example_labels, train_idx
    )

    # Compute constraint scores for all examples
    constraint_scores_list = []
    for feats in features_list:
        S, E, C, N = scorer(feats)
        constraint_scores_list.append(S)

    # Global normalization stats (from training correct steps)
    mu, sigma = compute_global_stats(
        [features_list[i] for i in train_idx]
    )

    # 3. Train transition kernel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel = train_kernel(
        features_list, labels_list, example_labels,
        train_idx, val_idx, mu, sigma, args
    )

    # 4. Compute deviation scores
    if kernel is not None:
        delta_list = compute_deviations(kernel, features_list, mu, sigma, device)
    else:
        print("\n[WARN] No kernel trained, using zero deviations")
        delta_list = [np.zeros(len(f)) for f in features_list]

    # 5. delta_j standalone AUROC (make-or-break)
    delta_auroc = evaluate_delta_auroc(delta_list, labels_list, example_labels, test_idx)

    # 6. Train probe
    probe = train_probe(
        features_list, labels_list, delta_list, constraint_scores_list,
        train_idx, val_idx, mu, sigma, args
    )

    # 7. CUSUM detection
    cusum_results = evaluate_cusum(
        probe, features_list, labels_list, delta_list,
        example_labels, train_idx, test_idx, mu, sigma, args
    )

    # 8. Probe step-level AUROC
    probe_auroc = evaluate_probe_auroc(
        probe, features_list, labels_list, delta_list,
        test_idx, mu, sigma
    )

    # ── Save results ──
    elapsed = time.time() - t0
    results = {
        "data_path": args.data_path,
        "n_examples": len(features_list),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "delta_auroc": delta_auroc,
        "probe_auroc": probe_auroc,
        "cusum": cusum_results,
        "elapsed_seconds": elapsed,
        "args": vars(args),
    }

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'=' * 60}")
    print(f"Results saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    # Save models
    if kernel is not None:
        torch.save(kernel.state_dict(), os.path.join(args.output_dir, "kernel.pt"))
    torch.save(probe.state_dict(), os.path.join(args.output_dir, "probe.pt"))
    np.savez(os.path.join(args.output_dir, "norm_stats.npz"), mu=mu, sigma=sigma)


if __name__ == "__main__":
    main()
