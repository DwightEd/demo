#!/usr/bin/env python3
"""检查stepcloud中原始特征的AUROC

验证哪些特征真的能区分error vs correct
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"

print("=" * 80)
print("Raw Feature AUROC Check")
print("=" * 80)

data = np.load(npz_path, allow_pickle=True)

is_correct = data['is_correct_strict']  # 0=correct, 1=error
stepcloud = data['stepcloud']  # (N, T, 33, 9)

if 'sv_layers' in data:
    sv_layers = [int(l) for l in data['sv_layers']]
else:
    sv_layers = list(range(33))

# 特征名称
FEATURE_NAMES = ["cloud_D", "cloud_V", "cloud_C", "coherence",
                 "mean_tok_norm", "resultant", "resultant_bulk",
                 "resultant_unif", "norm_bulk"]

print(f"\nLayers: {sv_layers}")
print(f"Feature names: {FEATURE_NAMES}")
print(f"Total chains: {len(is_correct)}")

# 选择L14
layer_id = 14
if layer_id in sv_layers:
    layer_idx = sv_layers.index(layer_id)
else:
    layer_idx = 0
    layer_id = sv_layers[0]

print(f"\nAnalyzing layer {layer_id} (idx {layer_idx})")

# 为每个特征计算AUROC
results = {}

for feat_idx, feat_name in enumerate(FEATURE_NAMES):
    labels = []
    values = []

    for i in range(len(is_correct)):
        chain_sc = stepcloud[i]  # (T, 33, 9)
        T = chain_sc.shape[0]

        if T == 0:
            continue

        # 计算该链在该层该特征的平均值
        feat_values = chain_sc[:, layer_idx, feat_idx]

        # 去NaN
        feat_values = feat_values[~np.isnan(feat_values)]

        if len(feat_values) == 0:
            continue

        # 聚合方式：mean, min, max, std
        agg_mean = np.mean(feat_values)
        agg_min = np.min(feat_values)
        agg_max = np.max(feat_values)
        agg_std = np.std(feat_values)

        # 收集所有聚合方式
        labels.append(1 if is_correct[i] == 1 else 0)
        values.append((agg_mean, agg_min, agg_max, agg_std))

    if len(labels) < 10:
        continue

    labels = np.array(labels)
    values = np.array(values)

    # 计算不同聚合方式的AUROC
    agg_names = ['mean', 'min', 'max', 'std']
    for agg_idx, agg_name in enumerate(agg_names):
        agg_values = values[:, agg_idx]
        valid_mask = ~np.isnan(agg_values)

        if np.sum(valid_mask) < 10:
            continue

        valid_labels = labels[valid_mask]
        valid_values = agg_values[valid_mask]

        if len(np.unique(valid_labels)) < 2:
            continue

        try:
            auroc = roc_auc_score(valid_labels, valid_values)
            results[f"{feat_name}_{agg_name}"] = auroc
        except:
            pass

# 打印结果
print("\n" + "=" * 80)
print("AUROC Results (error detection: higher → more error-like)")
print("=" * 80)
print(f"{'Feature':<25} {'AUROC':<10} {'Interpretation'}")
print("-" * 80)

# 按AUROC排序
sorted_results = sorted(results.items(), key=lambda x: -x[1])

for feat_name, auroc in sorted_results:
    if auroc > 0.7:
        interp = "EXCELLENT"
    elif auroc > 0.6:
        interp = "GOOD"
    elif auroc > 0.55:
        interp = "MODERATE"
    elif auroc > 0.45:
        interp = "WEAK"
    elif auroc > 0.4:
        interp = "POOR"
    else:
        interp = "REVERSE"

    print(f"{feat_name:<25} {auroc:.4f}    {interp}")

print("\n" + "=" * 80)
print("Top features (AUROC > 0.65):")
print("=" * 80)
for feat_name, auroc in sorted_results:
    if auroc > 0.65:
        print(f"  {feat_name}: {auroc:.4f}")

print("\n" + "=" * 80)
print("Features with AUROC < 0.35 (good for correctness detection):")
print("=" * 80)
for feat_name, auroc in sorted_results:
    if auroc < 0.35:
        print(f"  {feat_name}: {auroc:.4f}")

print("=" * 80)
