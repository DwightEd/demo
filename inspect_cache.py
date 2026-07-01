#!/usr/bin/env python3
"""检查缓存目录和单个缓存文件的详细内容"""

import pickle
from pathlib import Path
import numpy as np
import importlib.util

# 导入data_loading_gpu中的类定义，以便pickle加载
try:
    from data_loading_gpu import ReasoningTrajectory
except ImportError:
    spec = importlib.util.spec_from_file_location("data_loading_gpu", "data_loading_gpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ReasoningTrajectory = module.ReasoningTrajectory


def inspect_cache_directory(cache_dir_str: str):
    """检查缓存目录结构"""
    cache_dir = Path(cache_dir_str)

    print("=" * 70)
    print(f"检查缓存目录: {cache_dir}")
    print("=" * 70)

    if not cache_dir.exists():
        print(f"❌ 目录不存在: {cache_dir}")
        return None

    cache_files = sorted(cache_dir.glob("chain_*.pkl"),
                         key=lambda p: int(p.stem.split('_')[1]))

    if not cache_files:
        print(f"❌ 目录为空，没有缓存文件")
        return None

    print(f"✓ 找到 {len(cache_files)} 个缓存文件")
    print(f"  大小范围: {cache_files[0].stat().st_size / 1024:.1f} KB - "
          f"{cache_files[-1].stat().st_size / 1024:.1f} KB")

    return cache_files


def inspect_single_cache_file(cache_file_path: Path, verbose: bool = True):
    """详细检查单个缓存文件"""
    print(f"\n检查文件: {cache_file_path.name}")
    print("-" * 70)

    try:
        with open(cache_file_path, 'rb') as f:
            traj = pickle.load(f)

        # 基本信息
        print(f"✓ chain_id: {traj.chain_id}")
        print(f"✓ problem_id: {traj.problem_id}")
        print(f"✓ is_correct: {traj.is_correct}")
        print(f"✓ n_steps: {traj.n_steps}")

        # Step ranges
        if traj.step_ranges:
            print(f"✓ step_ranges: {len(traj.step_ranges)} steps")
            print(f"  范围示例: {traj.step_ranges[:2]}..." if len(traj.step_ranges) > 2 else f"  范围: {traj.step_ranges}")

        # 可用层
        available_layers = list(traj.steps.keys())
        print(f"✓ 可用层: {available_layers}")

        if verbose:
            # 每层的详细特征
            for layer in [10, 14, 18, 22]:
                if traj.has_layer(layer):
                    geoms = traj.get_geometry_sequence(layer)
                    print(f"\n  Layer {layer}: {len(geoms)} steps")
                    if geoms:
                        g = geoms[0]
                        print(f"    Step 0:")
                        print(f"      kappa: {g.kappa:.4f}")
                        print(f"      eff_rank: {g.eff_rank:.2f}")
                        print(f"      spectral_entropy: {g.spectral_entropy:.4f}")
                        print(f"      eigenvalues (前5): {g.eigenvalues[:5] if len(g.eigenvalues) >= 5 else g.eigenvalues}")

                        if len(geoms) > 1:
                            g_last = geoms[-1]
                            print(f"    Step {len(geoms)-1} (最后):")
                            print(f"      kappa: {g_last.kappa:.4f}")
                            print(f"      eff_rank: {g_last.eff_rank:.2f}")

        return traj

    except Exception as e:
        print(f"❌ 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import sys

    # 默认路径（正确路径：hidden/cache/omnimath）
    default_cache = "/gz-data/research/demo/data/hidden/cache/omnimath"

    if len(sys.argv) > 1:
        cache_dir_str = sys.argv[1]
    else:
        cache_dir_str = default_cache

    # 检查目录
    cache_files = inspect_cache_directory(cache_dir_str)

    if cache_files is None:
        print(f"\n请检查路径或重新运行计算")
        return

    # 抽样检查前3个文件
    print("\n" + "=" * 70)
    print("抽样检查前3个缓存文件")
    print("=" * 70)

    sample_files = cache_files[:min(3, len(cache_files))]

    for cf in sample_files:
        inspect_single_cache_file(cf, verbose=True)

    # 统计正确/错误数量
    print("\n" + "=" * 70)
    print("统计所有缓存文件")
    print("=" * 70)

    n_correct = 0
    n_error = 0
    n_valid = 0

    for cf in tqdm(cache_files) if len(cache_files) > 50 else cache_files:
        try:
            with open(cf, 'rb') as f:
                traj = pickle.load(f)
                n_valid += 1
                if traj.is_correct:
                    n_correct += 1
                else:
                    n_error += 1
        except:
            pass

    print(f"\n有效缓存: {n_valid}/{len(cache_files)}")
    print(f"  正确: {n_correct}")
    print(f"  错误: {n_error}")

    # 计算总特征数
    print(f"\n缓存数据规模:")
    print(f"  假设平均每条链4个step，4层")
    print(f"  总step特征数 ≈ {n_valid * 4 * 4}")


if __name__ == "__main__":
    from tqdm import tqdm
    main()
