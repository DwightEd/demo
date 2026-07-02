#!/usr/bin/env python3
"""检查NPZ文件中的特征名称"""

import numpy as np

npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"

data = np.load(npz_path, allow_pickle=True)

print("NPZ文件中的所有字段:")
for key in data.files:
    print(f"  {key}")

# 检查特征名称
if "stepgeom" in data.files:
    stepgeom = data["stepgeom"]
    print(f"\nstepgeom shape: {stepgeom.shape}")

if "geom_feature_names" in data.files:
    gnames = [str(x) for x in data["geom_feature_names"]]
    print(f"\ngeom_feature_names: {gnames}")

if "cloud_feature_names" in data.files:
    cnames = [str(x) for x in data["cloud_feature_names"]]
    print(f"\ncloud_feature_names: {cnames}")

if "stepcloud" in data.files:
    stepcloud = data["stepcloud"]
    print(f"\nstepcloud shape: {stepcloud.shape}")

# 检查第一个chain的stepgeom数据
if "stepgeom" in data.files:
    sample = data["stepgeom"][0]
    print(f"\n第一个chain的stepgeom:")
    print(f"  type: {type(sample)}")
    if hasattr(sample, 'shape'):
        print(f"  shape: {sample.shape}")
    elif hasattr(sample, '__len__'):
        print(f"  length: {len(sample)}")
        if len(sample) > 0 and hasattr(sample[0], '__dict__'):
            print(f"  第一个step的属性: {list(sample[0].__dict__.keys())}")
