#!/usr/bin/env python3
"""检查omnimath数据格式，诊断检测失败原因"""

import numpy as np
from pathlib import Path

npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"

print("=" * 60)
print("Inspecting omnimath data")
print("=" * 60)

data = np.load(npz_path, allow_pickle=True)

print("\n=== Available keys ===")
for key in data.keys():
    print(f"  {key}")

print("\n=== Metadata ===")
print(f"  problem_ids shape: {data['problem_ids'].shape}")
print(f"  is_correct_strict shape: {data['is_correct_strict'].shape}")
print(f"  n_correct: {np.sum(data['is_correct_strict'] == 0)}")
print(f"  n_error: {np.sum(data['is_correct_strict'] == 1)}")

if 'sv_layers' in data:
    print(f"  sv_layers: {data['sv_layers']}")
else:
    print("  sv_layers: NOT FOUND (using default 0-32)")

print("\n=== stepcloud ===")
stepcloud = data['stepcloud']
print(f"  shape: {stepcloud.shape}")
print(f"  expected: (N, T, 33, 9)")

N, T, L, F = stepcloud.shape
print(f"  N (chains): {N}")
print(f"  T (steps): {T}")
print(f"  L (layers): {L}")
print(f"  F (features): {F}")

if 'cloud_feature_names' in data:
    feature_names = data['cloud_feature_names']
    print(f"\n  cloud_feature_names: {feature_names}")
else:
    print(f"\n  cloud_feature_names: NOT FOUND")

# 检查一个样本的stepcloud数据
print("\n=== Sample stepcloud data (chain 0, step 0) ===")
for layer_idx in range(min(5, L)):
    features = stepcloud[0, 0, layer_idx, :]
    print(f"  Layer {layer_idx}: {features}")

# 检查特征值的分布
print("\n=== Feature statistics (all chains, all steps) ===")
for feat_idx in range(F):
    feat_vals = stepcloud[:, :, :, feat_idx].flatten()
    valid_vals = feat_vals[~np.isnan(feat_vals)]
    if len(valid_vals) > 0:
        print(f"  Feature {feat_idx}: min={valid_vals.min():.3f}, max={valid_vals.max():.3f}, mean={valid_vals.mean():.3f}, nan_count={np.isnan(feat_vals).sum()}")

# 检查kappa和eff_rank的值
print("\n=== Checking kappa and eff_rank values ===")
# 假设kappa在索引1，eff_rank在索引2
kappa_vals = stepcloud[:, :, :, 1].flatten()
kappa_valid = kappa_vals[~np.isnan(kappa_vals)]
print(f"  Kappa (idx 1): {len(kappa_valid)} valid values, range [{kappa_valid.min():.3f}, {kappa_valid.max():.3f}]")

eff_rank_vals = stepcloud[:, :, :, 2].flatten()
eff_rank_valid = eff_rank_vals[~np.isnan(eff_rank_vals)]
print(f"  Eff_rank (idx 2): {len(eff_rank_valid)} valid values, range [{eff_rank_valid.min():.3f}, {eff_rank_valid.max():.3f}]")

# 检查spectral相关特征
print("\n=== Checking spectral features (idx 3-7) ===")
for feat_idx in range(3, min(8, F)):
    feat_vals = stepcloud[:, :, :, feat_idx].flatten()
    valid_vals = feat_vals[~np.isnan(feat_vals)]
    if len(valid_vals) > 0:
        print(f"  Feature {feat_idx}: min={valid_vals.min():.3f}, max={valid_vals.max():.3f}, mean={valid_vals.mean():.3f}")

print("\n" + "=" * 60)
