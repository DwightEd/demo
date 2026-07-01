#!/usr/bin/env python3
"""检查缓存目录状态和采样数据

可以在本地或远程服务器上运行
"""

import pickle
from pathlib import Path
import numpy as np
import sys


def load_pickle_raw(file_path):
    """不依赖类定义，直接读取pickle内容"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def check_cache_dir(cache_dir_str: str):
    """检查缓存目录"""
    cache_dir = Path(cache_dir_str)

    print(f"检查缓存目录: {cache_dir}")
    print(f"  完整路径: {cache_dir.absolute()}")
    print("-" * 60)

    if not cache_dir.exists():
        print(f"❌ 目录不存在: {cache_dir}")
        print(f"\n请在远程服务器上运行:")
        print(f"  ssh 到服务器")
        print(f"  cd /gz-data/research/demo/")
        print(f"  python check_cache.py")
        return None

    cache_files = list(cache_dir.glob("chain_*.pkl"))

    if not cache_files:
        print(f"❌ 目录为空，没有缓存文件")
        print(f"\n需要先运行计算:")
        print(f"  python data_loading_gpu.py")
        return None

    print(f"✓ 找到 {len(cache_files)} 个缓存文件")

    # 抽样检查第一个文件
    sample_file = sorted(cache_files, key=lambda p: int(p.stem.split('_')[1]))[0]

    print(f"\n抽样检查: {sample_file.name}")
    print("-" * 60)

    try:
        traj = load_pickle_raw(sample_file)

        # 直接访问属性，不依赖类定义
        print(f"✓ chain_id: {traj.chain_id}")
        print(f"✓ is_correct: {traj.is_correct}")
        print(f"✓ n_steps: {traj.n_steps}")
        print(f"✓ 可用层: {list(traj.steps.keys())}")

        # 显示第一个step的特征
        for layer in [10, 14, 18, 22]:
            if layer in traj.steps and len(traj.steps[layer]) > 0:
                step_dict = traj.steps[layer]
                first_step_id = sorted(step_dict.keys())[0]
                g = step_dict[first_step_id]

                print(f"\n  Layer {layer}, Step {first_step_id}:")
                print(f"    kappa: {g.kappa:.4f}")
                print(f"    eff_rank: {g.eff_rank:.2f}")
                print(f"    spectral_entropy: {g.spectral_entropy:.4f}")
                break

        return cache_dir

    except Exception as e:
        print(f"❌ 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # 默认路径（远程服务器）
    default_path = "/gz-data/research/demo/data/hidden/cache/omnimath"

    if len(sys.argv) > 1:
        cache_dir_str = sys.argv[1]
    else:
        cache_dir_str = default_path

    print(f"\n{'='*70}")
    print(f"缓存检查工具")
    print(f"{'='*70}")

    result = check_cache_dir(cache_dir_str)

    if result is not None:
        print(f"\n✓✓✓ 找到有效缓存目录! ✓✓✓")
        print(f"\n运行分析:")
        print(f"  python analyze_results.py")
        print(f"  python analyze_dynamics.py")
    else:
        print(f"\n❌❌❌ 未找到有效缓存 ❌❌❌")
        print(f"\n请确认:")
        print(f"  1. 是否在远程服务器上运行?")
        print(f"  2. 是否已运行计算?")


if __name__ == "__main__":
    main()
