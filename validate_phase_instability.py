#!/usr/bin/env python3
"""相变不稳定性指标验证

验证 compute_phase_instability_metrics 函数在区分正确/错误推理中的有效性。
使用 full_*.npz 数据中的 stepvec 和 stepcloud 字段。

数据来源 (DATA.md):
    data/features/full_gsm8k.npz, full_math.npz, full_omnimath.npz
    - stepvec: (T, 8, 4096) pooled step vectors at 8 sv-layers
    - stepcloud: (T, 33, 9) cloud features including 'resultant' (kappa)
    - is_correct_strict: correctness labels
    - problem_ids: problem identifiers

使用方法:
    python validate_phase_instability.py --dataset gsm8k --output_dir ./results
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm


def compute_phase_instability_metrics(unit_vectors):
    """统一相变检测（重命名版，逻辑与原版完全一致，含未修复的已知问题）"""

    n_vectors = len(unit_vectors)

    # 1. 方向集中度：平均合向量长度（mean resultant length）
    #    注：这是 vMF concentration 的近似替代，不是严格反解出的 κ
    mean_vector = unit_vectors.mean(axis=0)
    mean_resultant_length = np.linalg.norm(mean_vector)

    # 方向张量（orientation matrix，非协方差矩阵——未去均值）
    orientation_matrix = (unit_vectors.T @ unit_vectors) / n_vectors
    eigenvalues = np.linalg.eigvalsh(orientation_matrix)
    eigenvalues = eigenvalues[eigenvalues > 0]
    eigenvalues = eigenvalues / eigenvalues.sum()
    effective_rank = np.exp(-(eigenvalues * np.log(eigenvalues + 1e-12)).sum())

    # 2. 三种候选的联合不稳定性指标（尚未收敛，供对比）

    # 方案A：几何偏差
    expected_effective_rank = 1 / (1 + mean_resultant_length**2)
    effective_rank_deviation = abs(effective_rank - expected_effective_rank)
    geometric_deviation_score = effective_rank_deviation

    # 方案B：布尔异常判定
    low_concentration_flag = mean_resultant_length < 0.5
    rank_explosion_flag = effective_rank > 3
    combined_anomaly_flag = low_concentration_flag and rank_explosion_flag

    # 方案C：连续联合分数（EDIS风格）
    # ⚠️ effective_rank / max(effective_rank, 1) 在 >=1 时恒为1，
    #    导致 effective_rank 信息丢失——此前指出的 bug，这里逻辑未改动
    combined_instability_score = (1 - mean_resultant_length) * (
        effective_rank / max(effective_rank, 1)
    )

    return {
        'concentration': mean_resultant_length,
        'effective_rank': effective_rank,
        'geometric_deviation_score': geometric_deviation_score,   # 方案A
        'combined_anomaly_flag': combined_anomaly_flag,           # 方案B
        'combined_instability_score': combined_instability_score, # 方案C
    }


def load_full_npz(npz_path: str) -> dict:
    """加载 full_*.npz 数据文件

    返回包含以下字段的字典:
        - stepvec: (N, T, 8, 4096) step vectors
        - stepcloud: (N, T, 33, 9) cloud features
        - cloud_feature_names: list of 9 feature names
        - is_correct_strict: (N,) correctness labels (0=correct, 1=error)
        - problem_ids: (N,) problem identifiers
    """
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    # 基本信息
    problem_ids = data['problem_ids']
    labels = data['is_correct_strict']
    N = len(problem_ids)

    # stepvec: (N, T, 8, 4096) 或 (T, 8, 4096) 对于单个链
    if 'stepvec' in data:
        stepvec = data['stepvec']
        if stepvec.ndim == 3:  # 单条链
            stepvec = stepvec[np.newaxis, ...]
    else:
        stepvec = None

    # stepcloud: (N, T, 33, 9) 或类似
    if 'stepcloud' in data:
        stepcloud = data['stepcloud']
        if stepcloud.ndim == 3:  # 单条链
            stepcloud = stepcloud[np.newaxis, ...]
    else:
        stepcloud = None

    # cloud_feature_names
    if 'cloud_feature_names' in data:
        cloud_feature_names = [str(n) for n in data['cloud_feature_names']]
    else:
        cloud_feature_names = None

    print(f"  Loaded {N} chains")
    if stepvec is not None:
        print(f"  stepvec shape: {stepvec.shape}")
    if stepcloud is not None:
        print(f"  stepcloud shape: {stepcloud.shape}")
        print(f"  cloud features: {cloud_feature_names}")

    return {
        'stepvec': stepvec,
        'stepcloud': stepcloud,
        'cloud_feature_names': cloud_feature_names,
        'labels': labels,
        'problem_ids': problem_ids,
        'N': N,
    }


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """将向量归一化为单位向量

    Args:
        vectors: (..., d) array of vectors

    Returns:
        (..., d) array of unit vectors
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)  # 避免除零
    return vectors / norms


def compute_chain_phase_metrics(
    stepvec: np.ndarray,
    layer_idx: int = 0,
) -> dict:
    """计算单条链的相变不稳定性指标

    Args:
        stepvec: (T, n_layers, d) step vectors for one chain
        layer_idx: 要分析的层索引 (0-7 对应 sv_layers)

    Returns:
        包含 phase_instability_metrics 的字典
    """
    if stepvec is None or stepvec.size == 0:
        return None

    # 提取指定层的所有步骤向量
    # stepvec shape: (T, n_layers, d)
    T, n_layers, d = stepvec.shape

    if layer_idx >= n_layers:
        layer_idx = 0

    vectors = stepvec[:, layer_idx, :]  # (T, d)

    # 归一化为单位向量
    unit_vectors = normalize_vectors(vectors)

    # 计算相变指标
    metrics = compute_phase_instability_metrics(unit_vectors)

    return metrics


def compute_per_chain_metrics(data: dict, layer_idx: int = 0) -> list:
    """为所有链计算相变指标

    Args:
        data: load_full_npz 返回的数据字典
        layer_idx: 要分析的层索引

    Returns:
        list of dict, 每个包含一条链的指标和标签
    """
    results = []

    if data['stepvec'] is None:
        print("No stepvec data available.")
        return results

    stepvec = data['stepvec']
    labels = data['labels']
    N = data['N']

    for i in tqdm(range(N), desc="Computing phase metrics"):
        # 提取该链的 stepvec: (T, 8, 4096)
        if stepvec.ndim == 4:  # (N, T, 8, 4096)
            chain_stepvec = stepvec[i]
        else:  # (T, 8, 4096) - 单条链
            chain_stepvec = stepvec

        # 计算指标
        metrics = compute_chain_phase_metrics(chain_stepvec, layer_idx=layer_idx)

        if metrics is not None:
            results.append({
                'chain_idx': i,
                'problem_id': int(data['problem_ids'][i]),
                'is_correct': bool(labels[i] == 0),  # 0=correct
                'metrics': metrics,
            })

    return results


def compute_auroc_scores(results: list) -> dict:
    """计算各指标的 AUROC (预测 error 的能力)

    Args:
        results: compute_per_chain_metrics 返回的结果列表

    Returns:
        各指标的 AUROC 分数
    """
    # 准备数据: y=1 表示 error, y=0 表示 correct
    y_true = np.array([0 if r['is_correct'] else 1 for r in results])

    auroc_results = {}

    # 测试每个连续指标
    metric_keys = ['concentration', 'effective_rank', 'geometric_deviation_score',
                   'combined_instability_score']

    for key in metric_keys:
        scores = np.array([r['metrics'][key] for r in results])

        # 移除 NaN
        valid_mask = np.isfinite(scores)
        y_valid = y_true[valid_mask]
        scores_valid = scores[valid_mask]

        if len(np.unique(y_valid)) < 2:
            auroc_results[key] = {
                'auroc': np.nan,
                'n_valid': len(scores_valid),
            }
            continue

        try:
            auroc = roc_auc_score(y_valid, scores_valid)
            auroc_results[key] = {
                'auroc': float(auroc),
                'n_valid': len(scores_valid),
            }
        except Exception as e:
            auroc_results[key] = {
                'auroc': np.nan,
                'n_valid': len(scores_valid),
                'error': str(e),
            }

    # 对于布尔指标，直接使用 flag 作为预测
    anomaly_pred = np.array([r['metrics']['combined_anomaly_flag'] for r in results], dtype=float)
    valid_mask = np.isfinite(anomaly_pred)
    y_valid = y_true[valid_mask]
    pred_valid = anomaly_pred[valid_mask]

    if len(np.unique(y_valid)) >= 2:
        try:
            auroc = roc_auc_score(y_valid, pred_valid)
            auroc_results['combined_anomaly_flag'] = {
                'auroc': float(auroc),
                'n_valid': len(pred_valid),
            }
        except:
            auroc_results['combined_anomaly_flag'] = {
                'auroc': np.nan,
                'n_valid': len(pred_valid),
            }

    return auroc_results


def run_statistical_tests(results: list) -> dict:
    """对相变指标进行统计检验，区分 error vs correct

    Args:
        results: compute_per_chain_metrics 返回的结果列表

    Returns:
        统计检验结果字典，包含 AUROC
    """
    # 分离 correct 和 error
    correct_results = [r for r in results if r['is_correct']]
    error_results = [r for r in results if not r['is_correct']]

    n_correct = len(correct_results)
    n_error = len(error_results)

    print(f"\nStatistical testing:")
    print(f"  Correct: {n_correct}, Error: {n_error}")

    if n_correct == 0 or n_error == 0:
        print("  Insufficient data for statistical testing.")
        return {}

    # 计算 AUROC
    auroc_results = compute_auroc_scores(results)

    test_results = {}

    # 测试每个指标
    metric_keys = ['concentration', 'effective_rank', 'geometric_deviation_score',
                   'combined_instability_score']

    for key in metric_keys:
        correct_vals = np.array([r['metrics'][key] for r in correct_results])
        error_vals = np.array([r['metrics'][key] for r in error_results])

        # 移除 NaN
        correct_vals = correct_vals[np.isfinite(correct_vals)]
        error_vals = error_vals[np.isfinite(error_vals)]

        if len(correct_vals) == 0 or len(error_vals) == 0:
            test_results[key] = {
                'n_correct': 0,
                'n_error': 0,
                'correct_mean': np.nan,
                'error_mean': np.nan,
                'mean_diff': np.nan,
                'statistic': np.nan,
                'p_value': np.nan,
                'cohens_d': np.nan,
                'auroc': np.nan,
                'significant': False,
            }
            continue

        # Mann-Whitney U test
        try:
            stat, pval = stats.mannwhitneyu(error_vals, correct_vals, alternative='two-sided')
        except:
            stat, pval = np.nan, np.nan

        # Cohen's d
        pooled_std = np.sqrt(((len(correct_vals)-1)*correct_vals.var(ddof=1) +
                             (len(error_vals)-1)*error_vals.var(ddof=1)) /
                            (len(correct_vals)+len(error_vals)-2))
        cohens_d = (error_vals.mean() - correct_vals.mean()) / pooled_std if pooled_std > 0 else 0

        # 获取 AUROC
        auroc = auroc_results.get(key, {}).get('auroc', np.nan)

        test_results[key] = {
            'n_correct': len(correct_vals),
            'n_error': len(error_vals),
            'correct_mean': float(correct_vals.mean()),
            'error_mean': float(error_vals.mean()),
            'mean_diff': float(error_vals.mean() - correct_vals.mean()),
            'statistic': float(stat) if np.isfinite(stat) else np.nan,
            'p_value': float(pval) if np.isfinite(pval) else np.nan,
            'cohens_d': float(cohens_d) if np.isfinite(cohens_d) else np.nan,
            'auroc': float(auroc) if np.isfinite(auroc) else np.nan,
            'significant': bool(pval < 0.05) if np.isfinite(pval) else False,
        }

        direction = ">" if error_vals.mean() > correct_vals.mean() else "<"
        sig_str = "*" if test_results[key]['significant'] else ""
        print(f"  {key:30s}: error{direction}correct | "
              f"error={error_vals.mean():.4f}, correct={correct_vals.mean():.4f} | "
              f"d={cohens_d:.3f}, p={pval:.4f}, auroc={auroc:.3f}{sig_str}")

    # 测试布尔指标
    correct_anomaly = sum(1 for r in correct_results if r['metrics']['combined_anomaly_flag'])
    error_anomaly = sum(1 for r in error_results if r['metrics']['combined_anomaly_flag'])

    correct_no_anomaly = n_correct - correct_anomaly
    error_no_anomaly = n_error - error_anomaly

    # Fisher's exact test
    try:
        oddsratio, pval = stats.fisher_exact(
            [[error_anomaly, error_no_anomaly],
             [correct_anomaly, correct_no_anomaly]],
            alternative='greater'
        )
    except:
        oddsratio, pval = np.nan, np.nan

    # 获取 AUROC
    auroc = auroc_results.get('combined_anomaly_flag', {}).get('auroc', np.nan)

    test_results['combined_anomaly_flag'] = {
        'n_correct': n_correct,
        'n_error': n_error,
        'correct_anomaly_rate': float(correct_anomaly / n_correct) if n_correct > 0 else np.nan,
        'error_anomaly_rate': float(error_anomaly / n_error) if n_error > 0 else np.nan,
        'correct_anomaly': correct_anomaly,
        'error_anomaly': error_anomaly,
        'odds_ratio': float(oddsratio) if np.isfinite(oddsratio) else np.nan,
        'p_value': float(pval) if np.isfinite(pval) else np.nan,
        'auroc': float(auroc) if np.isfinite(auroc) else np.nan,
        'significant': bool(pval < 0.05) if np.isfinite(pval) else False,
    }

    sig_str = "*" if test_results['combined_anomaly_flag']['significant'] else ""
    print(f"  {'combined_anomaly_flag':30s}: error={error_anomaly}/{n_error} ({error_anomaly/n_error*100:.1f}%), "
          f"correct={correct_anomaly}/{n_correct} ({correct_anomaly/n_correct*100:.1f}%) | "
          f"OR={oddsratio:.2f}, p={pval:.4f}, auroc={auroc:.3f}{sig_str}")

    return test_results


def main():
    parser = argparse.ArgumentParser(
        description='验证相变不稳定性指标'
    )
    parser.add_argument('--dataset',
                       choices=['gsm8k', 'math', 'omnimath'],
                       default='gsm8k',
                       help='数据集名称')
    parser.add_argument('--data_dir',
                       default='F:/projects/python_projects/research/constrained_manifolds/demo/data/features',
                       help='数据目录路径')
    parser.add_argument('--output_dir',
                       default='./results/phase_instability',
                       help='输出目录')
    parser.add_argument('--layer_idx',
                       type=int,
                       default=0,
                       help='要分析的 stepvec 层索引 (0-7 对应 sv_layers)')

    args = parser.parse_args()

    # 构建 NPZ 文件路径
    npz_path = os.path.join(args.data_dir, f'full_{args.dataset}.npz')

    print("=" * 70)
    print("相变不稳定性指标验证")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"NPZ path: {npz_path}")
    print(f"Layer index: {args.layer_idx}")
    print()

    # 检查文件是否存在
    if not os.path.exists(npz_path):
        print(f"错误: 文件不存在: {npz_path}")
        print(f"\n提示: 根据配置，数据应在远程服务器上:")
        print(f"  /gz-data/research/demo/data/features/full_{args.dataset}.npz")
        return

    # 加载数据
    data = load_full_npz(npz_path)

    # 计算指标
    print("\nComputing phase instability metrics...")
    results = compute_per_chain_metrics(data, layer_idx=args.layer_idx)

    if len(results) == 0:
        print("No results computed. Exiting.")
        return

    print(f"Computed metrics for {len(results)} chains")

    # 统计检验
    test_results = run_statistical_tests(results)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(args.output_dir, f'{args.dataset}_layer{args.layer_idx}_results.json')

    output = {
        'dataset': args.dataset,
        'layer_idx': args.layer_idx,
        'npz_path': npz_path,
        'n_chains': len(results),
        'n_correct': len([r for r in results if r['is_correct']]),
        'n_error': len([r for r in results if not r['is_correct']]),
        'test_results': test_results,
        'per_chain_results': results,
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n结果保存至: {output_file}")

    # 打印总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)

    # AUROC 排序
    auroc_sorted = sorted(
        [(k, v.get('auroc', np.nan)) for k, v in test_results.items() if 'auroc' in v],
        key=lambda x: x[1] if np.isfinite(x[1]) else -1,
        reverse=True
    )

    print("\nAUROC 排序 (预测 error 的能力):")
    for key, auroc in auroc_sorted:
        if np.isfinite(auroc):
            sig_str = "*" if test_results[key].get('significant', False) else ""
            print(f"  - {key:30s}: {auroc:.4f}{sig_str}")

    significant_metrics = [k for k, v in test_results.items()
                          if v.get('significant', False)]

    if significant_metrics:
        print(f"\n显著区分 error vs correct 的指标 (p < 0.05):")
        for key in significant_metrics:
            val = test_results[key]
            if 'cohens_d' in val:
                auroc_str = f", AUROC = {val.get('auroc', np.nan):.3f}" if np.isfinite(val.get('auroc', np.nan)) else ""
                print(f"  - {key}: Cohen's d = {val['cohens_d']:.3f}, p = {val['p_value']:.4f}{auroc_str}")
            else:
                auroc_str = f", AUROC = {val.get('auroc', np.nan):.3f}" if np.isfinite(val.get('auroc', np.nan)) else ""
                print(f"  - {key}: OR = {val['odds_ratio']:.2f}, p = {val['p_value']:.4f}{auroc_str}")
    else:
        print("\n没有指标达到显著性水平 (p < 0.05)")

    print("\n说明:")
    print("  - concentration: 方向集中度 (mean resultant length)")
    print("  - effective_rank: 有效秩 (基于方向张量的熵)")
    print("  - geometric_deviation_score: 几何偏差 (方案A)")
    print("  - combined_anomaly_flag: 布尔异常判定 (方案B)")
    print("  - combined_instability_score: 连续联合分数 (方案C, 含已知bug)")
    print("  - AUROC: Area Under ROC Curve, 0.5=随机, 1.0=完美区分 error vs correct")
    print("  - * 表示 p < 0.05 显著")


if __name__ == '__main__':
    main()
