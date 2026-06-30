#!/usr/bin/env python3
"""简化正确的AUROC分析

直接使用stepcloud中的特征，不做复杂变换
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"

print("=" * 80)
print("Simple AUROC Analysis")
print("=" * 80)

# 加载数据
data = np.load(npz_path, allow_pickle=True)

problem_ids = data['problem_ids']
is_correct = data['is_correct_strict']  # 0=correct, 1=error
stepcloud = data['stepcloud']  # (N, T, 33, 9)

if 'sv_layers' in data:
    sv_layers = [int(l) for l in data['sv_layers']]
else:
    sv_layers = list(range(33))

print(f"CLOUD_NAMES: ('cloud_D', 'cloud_V', 'cloud_C', 'coherence', 'mean_tok_norm', 'resultant', 'resultant_bulk', 'resultant_unif', 'norm_bulk')")
print(f"Layers: {sv_layers}")
print(f"Chains: {len(problem_ids)}")

# 特征索引
IDX_CLOUD_D = 0
IDX_CLOUD_V = 1
IDX_CLOUD_C = 2
IDX_COHERENCE = 3
IDX_MEAN_TOK_NORM = 4
IDX_RESULTANT = 5  # kappa
IDX_RESULTANT_BULK = 6
IDX_RESULTANT_UNIF = 7
IDX_NORM_BULK = 8

# 选择分析的层（例如L14，在sv_layers中的索引）
layer_id = 14
if layer_id in sv_layers:
    layer_idx = sv_layers.index(layer_id)
else:
    print(f"Layer {layer_id} not found, using first layer")
    layer_idx = 0
    layer_id = sv_layers[0]

print(f"\nAnalyzing layer {layer_id} (index {layer_idx})")

# 为每条链计算聚合特征
labels = []  # 0=correct, 1=error
chain_features = {}

feature_names = [
    'mean_cloud_D', 'mean_cloud_V', 'mean_cloud_C', 'mean_coherence',
    'mean_norm', 'mean_kappa', 'min_kappa', 'kappa_range',
    'mean_resultant_bulk', 'mean_norm_bulk'
]

for name in feature_names:
    chain_features[name] = []

for i in tqdm(range(len(problem_ids)), desc="Processing chains"):
    # 标签
    labels.append(1 if is_correct[i] == 1 else 0)  # 1=error, 0=correct

    # 该链的stepcloud: (T, 33, 9)
    chain_sc = stepcloud[i]
    T, L, F = chain_sc.shape

    if T == 0:
        for name in feature_names:
            chain_features[name].append(np.nan)
        continue

    # 提取该层的所有步骤
    layer_data = chain_sc[:, layer_idx, :]  # (T, 9)

    # 计算各种聚合特征
    # cloud_D (eff_rank)
    chain_features['mean_cloud_D'].append(np.nanmean(layer_data[:, IDX_CLOUD_D]))

    # cloud_V
    chain_features['mean_cloud_V'].append(np.nanmean(layer_data[:, IDX_CLOUD_V]))

    # cloud_C
    chain_features['mean_cloud_C'].append(np.nanmean(layer_data[:, IDX_CLOUD_C]))

    # coherence
    chain_features['mean_coherence'].append(np.nanmean(layer_data[:, IDX_COHERENCE]))

    # mean_tok_norm
    chain_features['mean_norm'].append(np.nanmean(layer_data[:, IDX_MEAN_TOK_NORM]))

    # kappa (resultant)
    kappas = layer_data[:, IDX_RESULTANT]
    chain_features['mean_kappa'].append(np.nanmean(kappas))
    chain_features['min_kappa'].append(np.nanmin(kappas))
    chain_features['kappa_range'].append(np.nanmax(kappas) - np.nanmin(kappas))

    # resultant_bulk
    chain_features['mean_resultant_bulk'].append(np.nanmean(layer_data[:, IDX_RESULTANT_BULK]))

    # norm_bulk
    chain_features['mean_norm_bulk'].append(np.nanmean(layer_data[:, IDX_NORM_BULK]))

labels = np.array(labels)

print(f"\nValid labels: error={np.sum(labels==1)}, correct={np.sum(labels==0)}")

# 计算AUROC
print("\n" + "=" * 80)
print("AUROC Results (error detection: higher score → more error-like)")
print("=" * 80)
print(f"{'Feature':<20} {'AUROC':<10} {'>0.5?':<8} {'Interpretation'}")
print("-" * 80)

results = {}
for name in feature_names:
    values = np.array(chain_features[name])

    # 移除NaN
    valid_mask = ~np.isnan(values)
    valid_labels = labels[valid_mask]
    valid_values = values[valid_mask]

    if len(np.unique(valid_labels)) < 2:
        print(f"{name:<20} {'N/A':<10} {'Only one class':<8}")
        continue

    try:
        # AUROC: 检测error的能力
        # valid_labels: 1=error, 0=correct
        # 如果feature在error中更高，AUROC>0.5
        auroc = roc_auc_score(valid_labels, valid_values)

        is_good = "YES" if auroc > 0.55 else ("NO" if auroc < 0.45 else "MARGINAL")
        interp = "Error↑" if auroc > 0.5 else "Correct↑"

        print(f"{name:<20} {auroc:.4f}    {is_good:<8} {interp}")
        results[name] = auroc

    except Exception as e:
        print(f"{name:<20} {'Error':<10} {str(e)}")

# 找最好的特征
print("\n" + "=" * 80)
print("Top features for error detection (AUROC > 0.6):")
print("=" * 80)
sorted_features = sorted(results.items(), key=lambda x: -x[1])
for name, auroc in sorted_features:
    if auroc > 0.6:
        print(f"  {name}: {auroc:.4f}")

if not any(auroc > 0.6 for auroc in results.values()):
    print("  None found (best below 0.6)")

print("\n" + "=" * 80)
print("Top features for correctness detection (AUROC < 0.4):")
print("=" * 80)
for name, auroc in sorted_features:
    if auroc < 0.4:
        print(f"  {name}: {auroc:.4f}")

if not any(auroc < 0.4 for auroc in results.values()):
    print("  None found (worst above 0.4)")

print("=" * 80)
