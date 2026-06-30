#!/usr/bin/env python3
"""计算轨迹几何指标的AUROC

检查：
1. smoothness的AUROC
2. coherence的AUROC
3. stability的AUROC
4. 组合指标的AUROC
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from trajectory_phase_transition import load_full_npz, compute_trajectory_metrics

npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"

print("=" * 80)
print("AUROC Analysis")
print("=" * 80)

trajectories, metadata = load_full_npz(npz_path)

print(f"Loaded {metadata['n_chains']} chains ({metadata['n_correct']} correct, {metadata['n_error']} error)")

layer = 14

# 收集所有指标
labels = []
smoothness_list = []
coherence_list = []
stability_list = []
kappa_list = []

for traj in trajectories:
    if not traj.has_layer(layer):
        continue

    labels.append(0 if traj.is_correct else 1)

    metrics = compute_trajectory_metrics(traj, layer)
    smoothness_list.append(metrics['smoothness'])
    coherence_list.append(metrics['coherence'])
    stability_list.append(metrics['stability'])

    # 平均kappa
    geom_seq = traj.get_geometry_sequence(layer)
    kappas = [g.kappa for g in geom_seq if np.isfinite(g.kappa)]
    kappa_list.append(np.mean(kappas) if kappas else np.nan)

labels = np.array(labels)
smoothness = np.array(smoothness_list)
coherence = np.array(coherence_list)
stability = np.array(stability_list)
kappa = np.array(kappa_list)

print(f"\nValid chains: {len(labels)}")
print(f"Correct: {np.sum(labels == 0)}, Error: {np.sum(labels == 1)}")

# 计算AUROC
metrics = {
    'smoothness': smoothness,
    'coherence': coherence,
    'stability': stability,
    'kappa': kappa,
}

print("\n" + "=" * 80)
print("AUROC Results (higher = better at separating error from correct)")
print("=" * 80)
print(f"{'Metric':<15} {'AUROC':<10} {'Direction':<15} {'Interpretation'}")
print("-" * 80)

for name, values in metrics.items():
    # 移除NaN
    valid_mask = ~np.isnan(values)
    valid_labels = labels[valid_mask]
    valid_values = values[valid_mask]

    if len(np.unique(valid_labels)) < 2:
        print(f"{name:<15} {'N/A':<10} {'Only one class':<15}")
        continue

    # AUROC (higher score = correct, so we use 1 - labels or flip values)
    # 对于error检测：我们希望error有更低分数，所以用1-labels
    try:
        auroc = roc_auc_score(1 - valid_labels, valid_values)  # error=1, correct=0
        direction = "lower→error" if auroc > 0.5 else "higher→error"
        interp = "Good" if auroc > 0.6 else "Random" if auroc < 0.55 else "Moderate"
        print(f"{name:<15} {auroc:.4f}    {direction:<15} {interp}")
    except Exception as e:
        print(f"{name:<15} {'Error':<10} {str(e)}")

# 计算组合指标的AUROC
print("\n" + "=" * 80)
print("Combined Metrics")
print("=" * 80)

# 组合1: 平均
valid_mask = (~np.isnan(smoothness)) & (~np.isnan(coherence)) & (~np.isnan(stability))
valid_labels = labels[valid_mask]

combined_avg = (smoothness[valid_mask] + coherence[valid_mask] + stability[valid_mask]) / 3
auroc_avg = roc_auc_score(1 - valid_labels, combined_avg)
print(f"Average (smooth+coh+stab): {auroc_avg:.4f}")

# 组合2: 只用smoothness + stability
valid_mask2 = (~np.isnan(smoothness)) & (~np.isnan(stability))
valid_labels2 = labels[valid_mask2]
combined_ss = (smoothness[valid_mask2] + stability[valid_mask2]) / 2
auroc_ss = roc_auc_score(1 - valid_labels2, combined_ss)
print(f"Average (smooth+stab): {auroc_ss:.4f}")

# 与kappa比较
valid_mask_k = (~np.isnan(kappa))
valid_labels_k = labels[valid_mask_k]
auroc_k = roc_auc_score(1 - valid_labels_k, kappa[valid_mask_k])
print(f"Kappa alone: {auroc_k:.4f}")

print("\n" + "=" * 80)
print("Improvement over kappa:")
print(f"  Combined avg: {auroc_avg - auroc_k:+.4f}")
print(f"  Smooth+Stab: {auroc_ss - auroc_k:+.4f}")
print("=" * 80)
