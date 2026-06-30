"""远程服务器运行脚本：几何验证实验

使用方法：
    python run_geometry_validation.py --npz_path /path/to/data.npz --output_dir ./results

依赖：
    numpy, scipy, tqdm
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def load_npz_data(npz_path: str) -> tuple[list, list, list]:
    """从NPZ文件加载数据

    期望格式（兼容extract_features.py输出）：
    - stepgeom: (N, T, L, F) per-step几何特征
    - layers_used: 使用的层
    - problem_ids: 问题ID
    - is_correct_strict: 正确性标签

    Returns:
        chains_data: list of dict, 每个dict包含一条链的数据
        layers: list of int, 层索引
        feature_names: list of str, 特征名称
    """
    data = np.load(npz_path, allow_pickle=True)

    # 基本元数据
    layers = [int(l) for l in data.get('layers_used', [14])]
    problem_ids = data['problem_ids'].astype(int)
    labels = data['is_correct_strict'].astype(int)

    N = len(problem_ids)

    chains_data = []
    for i in range(N):
        chain = {
            'problem_id': int(problem_ids[i]),
            'is_correct': bool(labels[i] == 0),  # 0=correct, 1=error
        }

        # 尝试加载stepgeom
        if 'stepgeom' in data:
            stepgeom = data['stepgeom'][i]  # (T, L, F)
            chain['stepgeom'] = stepgeom
            chain['n_steps'] = stepgeom.shape[0]
        else:
            chain['stepgeom'] = None
            chain['n_steps'] = data.get('n_steps', [0])[i] if 'n_steps' in data else 0

        chains_data.append(chain)

    # 获取特征名称
    if 'geom_feature_names' in data:
        feature_names = [str(n) for n in data['geom_feature_names']]
    else:
        feature_names = ['norm', 'pr', 'ae', 'ed_half', 'e50', 'e90']

    return chains_data, layers, feature_names


def compute_geometry_from_stepgeom(stepgeom: np.ndarray,
                                   layers: list[int],
                                   feature_names: list[str]) -> dict:
    """从stepgeom计算几何特征

    stepgeom: (T, L, F) array
    """
    if stepgeom is None:
        return {}

    T, L, F = stepgeom.shape

    # 尝试找到对应特征的索引
    feature_map = {name: idx for idx, name in enumerate(feature_names)}

    geom_data = {}

    # 为每个步骤计算特征
    for layer_idx, layer in enumerate(layers):
        for step in range(T):
            key = f'step_{step}_layer_{layer}'

            # 获取该步骤该层的所有特征
            features = stepgeom[step, layer_idx, :]

            geom_data[key] = {
                'norm': float(features[feature_map.get('norm', 0)]) if 'norm' in feature_map else np.nan,
                'pr': float(features[feature_map.get('pr', 1)]) if 'pr' in feature_map else np.nan,
                'ae': float(features[feature_map.get('ae', 2)]) if 'ae' in feature_map else np.nan,
            }

    return geom_data


def run_simple_validation(npz_path: str, output_dir: str):
    """运行简化版验证实验

    只验证：在现有数据上，哪些特征能区分error vs correct
    """
    print(f"Loading data from {npz_path}...")
    chains, layers, feature_names = load_npz_data(npz_path)

    correct = [c for c in chains if c['is_correct']]
    error = [c for c in chains if not c['is_correct']]

    print(f"Loaded {len(correct)} correct, {len(error)} error chains")
    print(f"Layers: {layers}")
    print(f"Features: {feature_names}")

    results = {}

    # 分析每个特征
    if 'stepgeom' in chains[0]:
        stepgeom_data = [c['stepgeom'] for c in chains if c['stepgeom'] is not None]
        labels = np.array([c['is_correct'] for c in chains if c['stepgeom'] is not None])

        if len(stepgeom_data) == 0:
            print("No stepgeom data found. Exiting.")
            return

        # (N, T, L, F)
        stepgeom_array = np.array([s for s in stepgeom_data])

        N, T, L, F = stepgeom_array.shape

        print(f"\nStepgeom shape: ({N}, {T}, {L}, {F})")
        print("Running feature-wise validation...")

        for feat_idx, feat_name in enumerate(feature_names):
            # 取该特征在所有步骤、所有层的平均值
            feat_vals = stepgeom_array[:, :, :, feat_idx]  # (N, T, L)

            # 按链平均
            chain_means = feat_vals.mean(axis=(1, 2))  # (N,)

            # 分组
            correct_vals = chain_means[labels == 1]
            error_vals = chain_means[labels == 0]

            if len(correct_vals) == 0 or len(error_vals) == 0:
                continue

            # 简单统计
            from scipy import stats
            stat, pval = stats.mannwhitneyu(error_vals, correct_vals, alternative='two-sided')

            pooled_std = np.sqrt(((len(correct_vals)-1)*correct_vals.var(ddof=1) +
                                 (len(error_vals)-1)*error_vals.var(ddof=1)) /
                                (len(correct_vals)+len(error_vals)-2))
            cohens_d = (error_vals.mean() - correct_vals.mean()) / pooled_std if pooled_std > 0 else 0

            results[feat_name] = {
                'error_mean': float(error_vals.mean()),
                'correct_mean': float(correct_vals.mean()),
                'cohens_d': float(cohens_d),
                'p_value': float(pval),
                'n_error': len(error_vals),
                'n_correct': len(correct_vals),
            }

            direction = ">" if error_vals.mean() > correct_vals.mean() else "<"
            print(f"{feat_name:15s}: error{direction}correct "
                  f"| error={error_vals.mean():.3f}, correct={correct_vals.mean():.3f} "
                  f"| d={cohens_d:.3f}, p={pval:.4f}")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'validation_results.json')

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # 打印总结
    print("\n=== Summary ===")
    print("Significant features (p < 0.05):")
    for name, res in results.items():
        if res['p_value'] < 0.05:
            print(f"  {name}: d={res['cohens_d']:.3f}, p={res['p_value']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Run geometry validation experiments on remote server'
    )
    parser.add_argument('--npz_path', required=True,
                       help='Path to NPZ data file')
    parser.add_argument('--output_dir', default='./results',
                       help='Output directory for results')

    args = parser.parse_args()

    print("=" * 60)
    print("Geometry Validation Experiment")
    print("=" * 60)
    print(f"NPZ path: {args.npz_path}")
    print(f"Output dir: {args.output_dir}")
    print()

    run_simple_validation(args.npz_path, args.output_dir)


if __name__ == '__main__':
    main()
