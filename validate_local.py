#!/usr/bin/env python3
"""本地测试脚本：验证trajectory_geometry_validation.py代码正确性

使用mock数据测试三层架构的逻辑完整性
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def create_mock_npz(output_path: str, n_chains: int = 100, n_steps: int = 5):
    """创建mock NPZ文件用于本地测试

    格式与full_*.npz一致：
    - stepcloud: (N, T, 33, 9) - T steps, 33 layers, 9 features
    - problem_ids: (N,)
    - is_correct_strict: (N,) - 1=correct, 0=error（与 extract_features._pb_record 一致；
      旧版 mock 按 0=correct 生成，与真实数据相反，会使反转的消费端在本地测试中
      互相自洽地"通过"，从而掩盖真实数据上的方向错误）
    - gold_error_step: (N,) - -1=全对, >=0=首错步（ProcessBench 约定）
    - sv_layers: (33,) - 层索引
    """
    np.random.seed(42)

    # 创建mock数据
    N = n_chains
    T = n_steps
    L = 33
    F = 9  # features per step

    # stepcloud: 每个步骤每层的几何特征
    # Feature order (approximate): [n_tokens, kappa, eff_rank, lam1, gap, ...]
    stepcloud = np.random.rand(N, T, L, F).astype(np.float32)

    # 模拟error和correct的差异：
    # Error chains: 更低的kappa (index 1), 更高的eff_rank (index 2, more scattered)
    error_indices = np.random.choice(N, size=N//2, replace=False)
    for i in error_indices:
        # 降低kappa（index 1）
        stepcloud[i, :, :, 1] *= 0.5
        # 提高eff_rank（index 2）
        stepcloud[i, :, :, 2] *= 1.5

    # 元数据（写入端约定: is_correct_strict 1=correct; gold_error_step -1=全对）
    problem_ids = np.arange(N)
    is_correct_strict = np.ones(N, dtype=int)
    is_correct_strict[error_indices] = 0
    gold_error_step = np.full(N, -1, dtype=int)
    gold_error_step[error_indices] = np.random.randint(0, T, size=len(error_indices))
    sv_layers = np.arange(33)

    # 保存
    np.savez_compressed(
        output_path,
        stepcloud=stepcloud,
        problem_ids=problem_ids,
        is_correct_strict=is_correct_strict,
        gold_error_step=gold_error_step,
        sv_layers=sv_layers,
    )

    print(f"✓ Created mock NPZ: {output_path}")
    print(f"  Chains: {N} ({N - len(error_indices)} correct, {len(error_indices)} error)")
    print(f"  Steps: {T}, Layers: {L}, Features: {F}")


def run_local_validation():
    """运行本地验证测试"""
    print("=" * 60)
    print("Local Validation Test")
    print("=" * 60)

    # 1. 创建mock数据
    mock_path = Path("data/mock_validation.npz")
    mock_path.parent.mkdir(parents=True, exist_ok=True)
    create_mock_npz(str(mock_path), n_chains=100, n_steps=5)

    # 2. 导入验证模块
    print("\nImporting trajectory_geometry_validation...")
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from trajectory_geometry_validation import (
            run_all_validations,
        )
        print("✓ Module imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    # 3. 运行验证
    print("\nRunning validation on mock data...")
    try:
        results = run_all_validations(
            str(mock_path),
            layers=[14],  # 只测试L14
            output_dir="data/mock_results",
        )
        print("\n✓ Validation completed successfully")
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 检查结果
    results_path = Path("data/mock_results/trajectory_validation_results.json")
    if not results_path.exists():
        print(f"✗ Results file not created: {results_path}")
        return False

    with open(results_path) as f:
        data = json.load(f)

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    for key, val in data['results'].items():
        print(f"{key}: {val['interpretation']}")

    # 验证基本完整性
    required_keys = ['L14_H1', 'L14_H2', 'L14_H3']
    for key in required_keys:
        if key not in data['results']:
            print(f"✗ Missing result: {key}")
            return False

    print("\n✓ All required results present")
    return True


if __name__ == "__main__":
    success = run_local_validation()

    print("\n" + "=" * 60)
    if success:
        print("✓ LOCAL TEST PASSED")
        print("Code is ready for remote server execution.")
    else:
        print("✗ LOCAL TEST FAILED")
        print("Please fix issues before running on remote server.")
    print("=" * 60)

    sys.exit(0 if success else 1)
