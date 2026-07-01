#!/usr/bin/env python3
"""诊断结果反转问题 - 检查is_correct判断和数据"""

import pickle
import numpy as np
from pathlib import Path
import importlib.util

# 导入data_loading_gpu中的类定义
spec = importlib.util.spec_from_file_location("data_loading_gpu", "data_loading_gpu.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ReasoningTrajectory = module.ReasoningTrajectory


def diagnose():
    """诊断is_correct判断"""
    cache_dir = Path("/gz-data/research/demo/data/hidden/cache/omnimath")
    npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"

    # 加载NPZ获取原始标签
    data = np.load(npz_path, allow_pickle=True)
    is_correct_strict = data['is_correct_strict']

    print("="*60)
    print("诊断is_correct判断")
    print("="*60)

    # 检查前5个chain的标签
    print("\n原始NPZ中的is_correct_strict:")
    for i in range(5):
        label = is_correct_strict[i]
        print(f"  Chain {i}: is_correct_strict={label} ({'正确' if label == 0 else '错误' if label == 1 else '未知'})")

    # 检查缓存中的is_correct
    cache_files = sorted(cache_dir.glob("chain_*.pkl"),
                         key=lambda p: int(p.stem.split('_')[1]))[:5]

    print("\n缓存中的is_correct:")
    for cf in cache_files:
        chain_id = int(cf.stem.split('_')[1])
        with open(cf, 'rb') as f:
            traj = pickle.load(f)

        original_label = is_correct_strict[chain_id]
        cached_is_correct = traj.is_correct

        print(f"  Chain {chain_id}:")
        print(f"    NPZ标签: is_correct_strict={original_label}")
        print(f"    缓存is_correct: {cached_is_correct}")
        print(f"    匹配? {('✓' if (original_label == 0) == cached_is_correct else '✗')}")

        # 检查第一个step的kappa
        if traj.has_layer(14):
            geoms = traj.get_geometry_sequence(14)
            if geoms:
                g = geoms[0]
                print(f"    Step 0 kappa: {g.kappa:.4f}")

    # 统计
    print("\n" + "="*60)
    print("统计所有chain的is_correct分布")
    print("="*60)

    n_correct_strict_0 = int(np.sum(is_correct_strict == 0))
    n_correct_strict_1 = int(np.sum(is_correct_strict == 1))

    print(f"\nNPZ中:")
    print(f"  is_correct_strict == 0 (正确): {n_correct_strict_0}")
    print(f"  is_correct_strict == 1 (错误): {n_correct_strict_1}")

    # 统计缓存
    n_cached_correct = 0
    n_cached_error = 0

    for cf in cache_dir.glob("chain_*.pkl"):
        with open(cf, 'rb') as f:
            traj = pickle.load(f)
        if traj.is_correct:
            n_cached_correct += 1
        else:
            n_cached_error += 1

    print(f"\n缓存中:")
    print(f"  is_correct == True: {n_cached_correct}")
    print(f"  is_correct == False: {n_cached_error}")

    # 检查是否匹配
    print(f"\n匹配检查:")
    print(f"  NPZ正确 vs 缓存正确: {'✓' if n_correct_strict_0 == n_cached_correct else '✗'}")


def check_kappa_values():
    """检查kappa值是否合理"""
    cache_dir = Path("/gz-data/research/demo/data/hidden/cache/omnimath")

    print("\n" + "="*60)
    print("检查Kappa值分布")
    print("="*60)

    correct_kappas = []
    error_kappas = []

    for cf in list(cache_dir.glob("chain_*.pkl"))[:50]:  # 抽样50个
        with open(cf, 'rb') as f:
            traj = pickle.load(f)

        if traj.has_layer(14):
            geoms = traj.get_geometry_sequence(14)
            for g in geoms:
                if traj.is_correct:
                    correct_kappas.append(g.kappa)
                else:
                    error_kappas.append(g.kappa)

    correct_kappas = np.array(correct_kappas)
    error_kappas = np.array(error_kappas)

    print(f"\n正确step的kappa (n={len(correct_kappas)}):")
    print(f"  均值: {np.mean(correct_kappas):.4f}")
    print(f"  标准差: {np.std(correct_kappas):.4f}")
    print(f"  范围: [{np.min(correct_kappas):.4f}, {np.max(correct_kappas):.4f}]")

    print(f"\n错误step的kappa (n={len(error_kappas)}):")
    print(f"  均值: {np.mean(error_kappas):.4f}")
    print(f"  标准差: {np.std(error_kappas):.4f}")
    print(f"  范围: [{np.min(error_kappas):.4f}, {np.max(error_kappas):.4f}]")

    print(f"\n比较:")
    print(f"  正确均值 - 错误均值 = {np.mean(correct_kappas) - np.mean(error_kappas):.4f}")
    print(f"  结论: {'错误step的kappa更高' if np.mean(error_kappas) > np.mean(correct_kappas) else '正确step的kappa更高'}")


if __name__ == "__main__":
    diagnose()
    check_kappa_values()
