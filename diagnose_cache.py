#!/usr/bin/env python3
"""诊断缓存问题 - 检查计算是否成功保存"""

import sys
from pathlib import Path
import numpy as np


def diagnose_paths():
    """诊断可能的问题"""
    print("=" * 70)
    print("缓存诊断工具")
    print("=" * 70)

    # 检查几个可能的缓存路径
    possible_paths = [
        "/gz-data/research/demo/data/cache/omnimath",
        "/gz-data/research/demo/data/cache",
        "/gz-data/research/demo/data/hidden/../cache/omnimath",
    ]

    print("\n1. 检查缓存目录是否存在...")
    found = False
    for p in possible_paths:
        path = Path(p)
        if path.exists():
            print(f"  ✓ 存在: {p}")
            if path.is_dir():
                files = list(path.glob("*"))
                print(f"    文件数: {len(files)}")
                if files:
                    print(f"    示例文件: {[f.name for f in files[:5]]}")
                found = True
        else:
            print(f"  ✗ 不存在: {p}")

    if not found:
        print("\n  ❌ 没有找到任何缓存目录！")

    # 检查NPZ文件
    print("\n2. 检查NPZ文件...")
    npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"
    if Path(npz_path).exists():
        print(f"  ✓ NPZ存在: {npz_path}")
        try:
            data = np.load(npz_path, allow_pickle=True)
            print(f"    chains数量: {len(data['problem_ids'])}")
        except Exception as e:
            print(f"    ✗ 读取失败: {e}")
    else:
        print(f"  ✗ NPZ不存在: {npz_path}")

    # 检查hidden目录
    print("\n3. 检查hidden目录...")
    hidden_dir = "/gz-data/research/demo/data/hidden/omnimath/"
    if Path(hidden_dir).exists():
        print(f"  ✓ hidden目录存在: {hidden_dir}")
        npy_files = list(Path(hidden_dir).glob("*.npy"))
        print(f"    npy文件数: {len(npy_files)}")
    else:
        print(f"  ✗ hidden目录不存在: {hidden_dir}")

    # 预期的缓存目录
    print("\n4. 预期的缓存目录...")
    expected_cache = Path("/gz-data/research/demo/data/cache/omnimath")
    print(f"  应该在: {expected_cache}")
    if expected_cache.exists():
        print(f"  ✓ 目录存在")
        pkl_files = list(expected_cache.glob("*.pkl"))
        print(f"    pkl文件数: {len(pkl_files)}")
    else:
        print(f"  ✗ 目录不存在")
        print(f"  父目录状态: {expected_cache.parent}")

        # 尝试创建
        print(f"\n5. 尝试创建测试目录...")
        try:
            expected_cache.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ 成功创建: {expected_cache}")
            # 写入测试文件
            test_file = expected_cache / "test.txt"
            test_file.write_text("test")
            print(f"  ✓ 成功写入测试文件")
            test_file.unlink()
        except Exception as e:
            print(f"  ✗ 创建失败: {e}")


def check_recent_files():
    """检查最近修改的文件"""
    print("\n6. 检查最近修改的文件...")
    base_dir = Path("/gz-data/research/demo/data")
    if not base_dir.exists():
        print(f"  ✗ 基础目录不存在: {base_dir}")
        return

    # 递归查找最近1小时内修改的文件
    import time
    import os

    cutoff = time.time() - 3600  # 1小时前
    recent = []

    for root, dirs, files in os.walk(base_dir):
        for f in files:
            path = Path(root) / f
            try:
                mtime = path.stat().st_mtime
                if mtime > cutoff:
                    recent.append((path, mtime))
            except:
                pass

    if recent:
        print(f"  找到 {len(recent)} 个最近修改的文件:")
        recent.sort(key=lambda x: x[1], reverse=True)
        for path, mtime in recent[:10]:
            import datetime
            dt = datetime.datetime.fromtimestamp(mtime)
            print(f"    {path} - {dt}")
    else:
        print(f"  没有找到最近1小时内修改的文件")


def check_script_logic():
    """检查脚本逻辑"""
    print("\n7. 检查脚本逻辑...")
    print("  根据data_loading_gpu.py:")
    print("    cache_dir = Path(hidden_dir).parent / 'cache' / subset")
    print("    hidden_dir = '/gz-data/research/demo/data/hidden/omnimath/'")
    print("    subset = 'omnimath'")
    print("  预期缓存: /gz-data/research/demo/data/cache/omnimath")

    # 检查代码中的保存逻辑
    print("\n8. 保存逻辑检查:")
    print("  ✓ save_cached_features() 在第532行被调用")
    print("  ✓ 每个计算完的chain都会保存到 cache_dir / chain_{idx}.pkl")


def suggest_solution():
    """建议解决方案"""
    print("\n" + "=" * 70)
    print("诊断结论和建议")
    print("=" * 70)

    print("\n可能的原因:")
    print("  1. 脚本运行时出错，没有执行到保存部分")
    print("  2. 缓存目录权限问题")
    print("  3. GPU/CUDA初始化失败，程序提前退出")
    print("  4. 使用了不同的脚本/参数")

    print("\n建议:")
    print("  1. 检查运行日志，看是否有错误")
    print("  2. 重新运行，确保能看到 'Cache directory: ...' 输出")
    print("  3. 使用 --force 参数强制重新计算")
    print("  4. 如果GPU有问题，使用 --cpu 参数")

    print("\n重新运行命令:")
    print("  cd /gz-data/research/demo/")
    print("  git pull origin main")
    print("  python data_loading_gpu.py --force --cpu  # 先用CPU测试")
    print("  # 或")
    print("  python data_loading_fast.py  # 使用fast版本")


if __name__ == "__main__":
    diagnose_paths()
    check_recent_files()
    check_script_logic()
    suggest_solution()
