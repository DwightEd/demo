#!/usr/bin/env python3
"""检查缓存数据版本，判断是否需要重新计算"""

import pickle
from pathlib import Path
import numpy as np


def check_cache_version(cache_dir: str = "/gz-data/research/demo/data/hidden/cache/omnimath"):
    """检查缓存是否使用了正确的特征值分解方法"""

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"缓存目录不存在: {cache_dir}")
        return

    cache_files = list(cache_path.glob("chain_*.pkl"))
    if not cache_files:
        print(f"缓存目录为空")
        return

    print(f"检查 {len(cache_files)} 个缓存文件...")

    # 抽样检查前5个文件
    sample_files = sorted(cache_files, key=lambda p: int(p.stem.split('_')[1]))[:5]

    issues_found = []

    for cf in sample_files:
        try:
            # 直接读取pickle内容（不依赖类）
            with open(cf, 'rb') as f:
                traj = pickle.load(f)

            # 检查特征值
            has_eigenvalues = False
            eigenvalue_count = 0

            for layer in traj.steps:
                for step_id, geom in traj.steps[layer].items():
                    if hasattr(geom, 'eigenvalues') and geom.eigenvalues is not None:
                        eigenvalues = geom.eigenvalues
                        eigenvalue_count = len(eigenvalues)

                        # 检查特征值是否归一化（和应该接近1）
                        if eigenvalue_count > 0:
                            eig_sum = np.sum(eigenvalues)
                            # 特征值和应该接近1（归一化）
                            if abs(eig_sum - 1.0) > 0.1:
                                issues_found.append(f"{cf.name}: 特征值和={eig_sum:.3f} (应该≈1.0)")

                            has_eigenvalues = True
                        break

                if has_eigenvalues:
                    break

            print(f"✓ {cf.name}: eigenvalues={eigenvalue_count}")

        except Exception as e:
            print(f"✗ {cf.name}: 读取失败 - {e}")

    # 结论
    print("\n" + "="*60)
    print("诊断结论")
    print("="*60)

    if issues_found:
        print("❌ 缓存数据有问题:")
        for issue in issues_found:
            print(f"  - {issue}")
        print("\n建议: 清除缓存并重新计算")
        print("  python data_loading_gpu.py --clear")
        print("  python data_loading_gpu.py --force")
    else:
        print("✓ 缓存数据看起来正常")
        print("\n但结果反转可能意味着:")
        print("  1. 缓存是用旧代码计算的（对角近似）")
        print("  2. 需要清除缓存重新计算")


def clear_and_recompute():
    """清除缓存并重新计算的指令"""
    print("\n" + "="*60)
    print("清除缓存并重新计算")
    print("="*60)
    print("""
cd /gz-data/research/demo/

# 1. 清除旧缓存
python data_loading_gpu.py --clear

# 2. 重新计算（使用完整特征值分解）
python data_loading_gpu.py --force

# 3. 或使用CPU版本（更稳定）
python data_loading_fast.py

注意：这需要几个小时！
""")


if __name__ == "__main__":
    import sys
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "/gz-data/research/demo/data/hidden/cache/omnimath"
    check_cache_version(cache_dir)
    clear_and_recompute()
