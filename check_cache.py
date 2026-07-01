#!/usr/bin/env python3
"""检查缓存目录状态和采样数据"""

import pickle
from pathlib import Path
import numpy as np
import sys

def check_cache_dir(cache_dir_str: str):
    """检查缓存目录"""
    cache_dir = Path(cache_dir_str)

    print(f"检查缓存目录: {cache_dir}")
    print("-" * 60)

    if not cache_dir.exists():
        print(f"❌ 目录不存在: {cache_dir}")
        print(f"\n请确认:")
        print(f"1. 数据是否已计算完成？")
        print(f"2. 路径是否正确？")
        return None

    cache_files = list(cache_dir.glob("chain_*.pkl"))

    if not cache_files:
        print(f"❌ 目录为空，没有缓存文件")
        return None

    print(f"✓ 找到 {len(cache_files)} 个缓存文件")

    # 抽样检查第一个文件
    sample_file = sorted(cache_files, key=lambda p: int(p.stem.split('_')[1]))[0]

    print(f"\n抽样检查: {sample_file.name}")
    print("-" * 60)

    try:
        with open(sample_file, 'rb') as f:
            traj = pickle.load(f)

        print(f"✓ chain_id: {traj.chain_id}")
        print(f"✓ is_correct: {traj.is_correct}")
        print(f"✓ n_steps: {traj.n_steps}")
        print(f"✓ 可用层: {list(traj.steps.keys())}")

        # 显示第一个step的特征
        for layer in [10, 14, 18, 22]:
            if traj.has_layer(layer):
                geoms = traj.get_geometry_sequence(layer)
                if geoms:
                    g = geoms[0]
                    print(f"\n  Layer {layer}, Step 0:")
                    print(f"    kappa: {g.kappa:.4f}")
                    print(f"    eff_rank: {g.eff_rank:.2f}")
                    print(f"    spectral_entropy: {g.spectral_entropy:.4f}")
                    break

        return cache_dir

    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return None


def main():
    # 尝试多个可能的路径
    possible_paths = [
        "/gz-data/research/demo/data/cache/omnimath",  # Linux服务器
        "F:/projects/python_projects/research/constrained_manifolds/data/cache/omnimath",  # Windows本地
        "F:/projects/python_projects/research/constrained_manifolds/demo/data/cache/omnimath",
        "../data/cache/omnimath",  # 相对路径
    ]

    if len(sys.argv) > 1:
        possible_paths = [sys.argv[1]]

    for path_str in possible_paths:
        print(f"\n尝试路径: {path_str}")
        result = check_cache_dir(path_str)
        if result is not None:
            print(f"\n✓✓✓ 找到有效缓存目录! ✓✓✓")
            print(f"运行分析命令:")
            print(f'  python analyze_results.py --cache-dir "{result}"')
            return

    print(f"\n❌❌❌ 未找到有效缓存目录 ❌❌❌")
    print(f"\n请指定缓存目录:")
    print(f"  python check_cache.py /path/to/cache/omnimath")


if __name__ == "__main__":
    main()
