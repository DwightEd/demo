#!/usr/bin/env python3
"""OMniMath结果分析和汇报脚本

计算：
1. 特征分布统计（均值、标准差）
2. 正确vs错误的差异检验（Cohen's d）
3. 每层的AUC/分类能力
4. Step-level trends
5. 生成LaTeX表格和图表
"""

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
import sys

HIDDEN_LAYERS = [10, 14, 18, 22]


# ========== 必须在__main__中定义相同的类，以便pickle加载 ==========
# 这些类定义必须与data_loading_gpu.py中完全一致

@dataclass
class StepGeometry:
    step_id: int
    layer: int
    n_tokens: int
    kappa: float = np.nan
    eff_rank: float = np.nan
    spectral_entropy: float = np.nan
    norm: float = np.nan
    eigenvalues: np.ndarray = None
    principal_directions: np.ndarray = None

    def __post_init__(self):
        if self.eigenvalues is None:
            self.eigenvalues = np.array([])
        if self.principal_directions is None:
            self.principal_directions = np.array([])


class ReasoningTrajectory:
    def __init__(self, chain_id, problem_id, is_correct, n_steps, step_ranges=None, steps=None):
        self.chain_id = chain_id
        self.problem_id = problem_id
        self.is_correct = bool(is_correct)
        self.n_steps = n_steps
        if step_ranges is None:
            self.step_ranges = []
        elif isinstance(step_ranges, np.ndarray):
            self.step_ranges = step_ranges.tolist()
        elif hasattr(step_ranges, '__iter__'):
            self.step_ranges = list(step_ranges)
        else:
            self.step_ranges = []
        self.steps = steps or {}

    def has_layer(self, layer):
        return layer in self.steps and len(self.steps[layer]) > 0

    def get_geometry_sequence(self, layer):
        if not self.has_layer(layer):
            return []
        return [self.steps[layer][i] for i in sorted(self.steps[layer].keys())]

# ========== 类定义结束 ==========


@dataclass
class LayerMetrics:
    layer: int
    correct_kappa: List[float] = field(default_factory=list)
    error_kappa: List[float] = field(default_factory=list)
    correct_eff_rank: List[float] = field(default_factory=list)
    error_eff_rank: List[float] = field(default_factory=list)
    correct_entropy: List[float] = field(default_factory=list)
    error_entropy: List[float] = field(default_factory=list)


def cohen_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """计算Cohen's d效应量"""
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (np.mean(x1) - np.mean(x2)) / pooled_std


def compute_auc(x1: np.ndarray, x2: np.ndarray) -> float:
    """计算AUC（x1为正类，x2为负类）"""
    from scipy.stats import mannwhitneyu
    try:
        result = mannwhitneyu(x1, x2, alternative='greater')
        u1 = result.statistic
        n1, n2 = len(x1), len(x2)
        auc = u1 / (n1 * n2)
        return auc
    except:
        return 0.5


def load_trajectories_simple(cache_dir: Path) -> List:
    """加载轨迹：使用__main__中定义的类来加载pickle"""
    trajectories = []
    cache_files = sorted(cache_dir.glob("chain_*.pkl"),
                         key=lambda p: int(p.stem.split('_')[1]))

    print(f"Loading {len(cache_files)} cached trajectories...")

    for cf in tqdm(cache_files):
        try:
            with open(cf, 'rb') as f:
                traj = pickle.load(f)
                trajectories.append(traj)
        except Exception as e:
            print(f"Failed to load {cf}: {e}")

    return trajectories


def extract_step_features(trajectories: List, layer: int) -> LayerMetrics:
    """提取指定层的所有step特征"""
    metrics = LayerMetrics(layer=layer)

    for traj in trajectories:
        # 检查是否有该层的数据
        if not hasattr(traj, 'steps'):
            continue

        steps = traj.steps
        if layer not in steps or len(steps[layer]) == 0:
            continue

        # 遍历该层的所有step
        for step_id in sorted(steps[layer].keys()):
            geom = steps[layer][step_id]

            if not hasattr(geom, 'kappa'):
                continue

            is_correct = traj.is_correct if hasattr(traj, 'is_correct') else None

            if is_correct:
                metrics.correct_kappa.append(float(geom.kappa) if np.isfinite(geom.kappa) else np.nan)
                metrics.correct_eff_rank.append(float(geom.eff_rank) if np.isfinite(geom.eff_rank) else np.nan)
                metrics.correct_entropy.append(float(geom.spectral_entropy) if np.isfinite(geom.spectral_entropy) else np.nan)
            elif is_correct is False:
                metrics.error_kappa.append(float(geom.kappa) if np.isfinite(geom.kappa) else np.nan)
                metrics.error_eff_rank.append(float(geom.eff_rank) if np.isfinite(geom.eff_rank) else np.nan)
                metrics.error_entropy.append(float(geom.spectral_entropy) if np.isfinite(geom.spectral_entropy) else np.nan)

    return metrics


def compute_statistics(metrics: LayerMetrics) -> Dict:
    """计算统计指标"""
    result = {
        'layer': metrics.layer,
        'n_correct_steps': len(metrics.correct_kappa),
        'n_error_steps': len(metrics.error_kappa),
        'kappa': {},
        'eff_rank': {},
        'entropy': {},
    }

    # Kappa统计
    if len(metrics.correct_kappa) > 0 and len(metrics.error_kappa) > 0:
        corr_k = np.array(metrics.correct_kappa)
        err_k = np.array(metrics.error_kappa)

        # 移除NaN
        corr_k = corr_k[np.isfinite(corr_k)]
        err_k = err_k[np.isfinite(err_k)]

        if len(corr_k) > 0 and len(err_k) > 0:
            result['kappa'] = {
                'correct_mean': float(np.mean(corr_k)),
                'correct_std': float(np.std(corr_k)),
                'error_mean': float(np.mean(err_k)),
                'error_std': float(np.std(err_k)),
                'cohens_d': float(cohen_d(corr_k, err_k)),
                'auc': float(compute_auc(err_k, corr_k)),
                'p_value': float(stats.ttest_ind(corr_k, err_k).pvalue),
                'diff_significant': stats.ttest_ind(corr_k, err_k).pvalue < 0.05,
            }

    # Effective Rank统计
    if len(metrics.correct_eff_rank) > 0 and len(metrics.error_eff_rank) > 0:
        corr_e = np.array(metrics.correct_eff_rank)
        err_e = np.array(metrics.error_eff_rank)

        corr_e = corr_e[np.isfinite(corr_e)]
        err_e = err_e[np.isfinite(err_e)]

        if len(corr_e) > 0 and len(err_e) > 0:
            result['eff_rank'] = {
                'correct_mean': float(np.mean(corr_e)),
                'correct_std': float(np.std(corr_e)),
                'error_mean': float(np.mean(err_e)),
                'error_std': float(np.std(err_e)),
                'cohens_d': float(cohen_d(corr_e, err_e)),
                'auc': float(compute_auc(err_e, corr_e)),
                'p_value': float(stats.ttest_ind(corr_e, err_e).pvalue),
                'diff_significant': stats.ttest_ind(corr_e, err_e).pvalue < 0.05,
            }

    # Spectral Entropy统计
    if len(metrics.correct_entropy) > 0 and len(metrics.error_entropy) > 0:
        corr_s = np.array(metrics.correct_entropy)
        err_s = np.array(metrics.error_entropy)

        corr_s = corr_s[np.isfinite(corr_s)]
        err_s = err_s[np.isfinite(err_s)]

        if len(corr_s) > 0 and len(err_s) > 0:
            result['entropy'] = {
                'correct_mean': float(np.mean(corr_s)),
                'correct_std': float(np.std(corr_s)),
                'error_mean': float(np.mean(err_s)),
                'error_std': float(np.std(err_s)),
                'cohens_d': float(cohen_d(corr_s, err_s)),
                'auc': float(compute_auc(err_s, corr_s)),
                'p_value': float(stats.ttest_ind(corr_s, err_s).pvalue),
                'diff_significant': stats.ttest_ind(corr_s, err_s).pvalue < 0.05,
            }

    return result


def print_latex_table(all_stats: List[Dict]):
    """打印LaTeX表格"""
    print("\n" + "="*80)
    print("LaTeX表格 - 复制到论文")
    print("="*80 + "\n")

    # Kappa表格
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Step-level Kappa分布（正确 vs 错误）}")
    print("\\label{tab:kappa_distribution}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("Layer & Correct & Error & Cohen's $d$ & $p$-value \\\\")
    print("\\hline")

    for stats in all_stats:
        k = stats['kappa']
        if k:
            sig = "**" if k['diff_significant'] else ""
            print(f"{stats['layer']} & {k['correct_mean']:.3f}$\\pm${k['correct_std']:.3f} & "
                  f"{k['error_mean']:.3f}$\\pm${k['error_std']:.3f} & "
                  f"{k['cohens_d']:.3f}{sig} & {k['p_value']:.4f} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    # Eff Rank表格
    print("\n\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Step-level Effective Rank分布（正确 vs 错误）}")
    print("\\label{tab:eff_rank_distribution}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("Layer & Correct & Error & Cohen's $d$ & $p$-value \\\\")
    print("\\hline")

    for stats in all_stats:
        e = stats['eff_rank']
        if e:
            sig = "**" if e['diff_significant'] else ""
            print(f"{stats['layer']} & {e['correct_mean']:.2f}$\\pm${e['correct_std']:.2f} & "
                  f"{e['error_mean']:.2f}$\\pm${e['error_std']:.2f} & "
                  f"{e['cohens_d']:.3f}{sig} & {e['p_value']:.4f} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def print_summary_report(all_stats: List[Dict], metadata: Dict):
    """打印总结报告"""
    print("\n" + "="*80)
    print("OMniMath 几何特征分析报告")
    print("="*80 + "\n")

    print(f"数据集: {metadata.get('subset', 'omnimath')}")
    print(f"总轨迹数: {metadata.get('n_chains', 'N/A')}")
    print(f"正确: {metadata.get('n_correct', 'N/A')}, 错误: {metadata.get('n_error', 'N/A')}")

    print("\n" + "-"*80)
    print("关键发现汇总")
    print("-"*80 + "\n")

    # Kappa发现
    print("【Kappa - 向量集中度】")
    print("假设: 错误step的kappa更低（向量更发散）")
    for stats in all_stats:
        k = stats['kappa']
        if k:
            direction = "✓ 确认" if k['error_mean'] < k['correct_mean'] else "✗ 反转"
            print(f"  Layer {stats['layer']}: {direction}")
            print(f"    正确={k['correct_mean']:.4f}, 错误={k['error_mean']:.4f}, "
                  f"d={k['cohens_d']:.3f}, p={k['p_value']:.4f}")

    # Eff Rank发现
    print("\n【Effective Rank - 有效秩】")
    print("假设: 错误step的eff_rank更高（更多维度活跃）")
    for stats in all_stats:
        e = stats['eff_rank']
        if e:
            direction = "✓ 确认" if e['error_mean'] > e['correct_mean'] else "✗ 反转"
            print(f"  Layer {stats['layer']}: {direction}")
            print(f"    正确={e['correct_mean']:.2f}, 错误={e['error_mean']:.2f}, "
                  f"d={e['cohens_d']:.3f}, p={e['p_value']:.4f}")

    # Entropy发现
    print("\n【Spectral Entropy - 谱熵】")
    print("假设: 错误step的熵更高（分布更均匀）")
    for stats in all_stats:
        s = stats['entropy']
        if s:
            direction = "✓ 确认" if s['error_mean'] > s['correct_mean'] else "✗ 反转"
            print(f"  Layer {stats['layer']}: {direction}")
            print(f"    正确={s['correct_mean']:.4f}, 错误={s['error_mean']:.4f}, "
                  f"d={s['cohens_d']:.3f}, p={s['p_value']:.4f}")

    # 总体结论
    print("\n" + "-"*80)
    print("总体结论")
    print("-"*80)

    # 统计显著性数量
    sig_kappa = sum(1 for s in all_stats if s['kappa'] and s['kappa']['diff_significant'])
    sig_eff = sum(1 for s in all_stats if s['eff_rank'] and s['eff_rank']['diff_significant'])
    sig_ent = sum(1 for s in all_stats if s['entropy'] and s['entropy']['diff_significant'])

    print(f"\n显著性结果 (p<0.05):")
    print(f"  Kappa: {sig_kappa}/{len(all_stats)} 层显著")
    print(f"  Eff_Rank: {sig_eff}/{len(all_stats)} 层显著")
    print(f"  Entropy: {sig_ent}/{len(all_stats)} 层显著")

    # 大效应量数量 (|d| > 0.8)
    large_kappa = sum(1 for s in all_stats if s['kappa'] and abs(s['kappa']['cohens_d']) > 0.8)
    large_eff = sum(1 for s in all_stats if s['eff_rank'] and abs(s['eff_rank']['cohens_d']) > 0.8)
    large_ent = sum(1 for s in all_stats if s['entropy'] and abs(s['entropy']['cohens_d']) > 0.8)

    print(f"\n大效应量 (|d| > 0.8):")
    print(f"  Kappa: {large_kappa}/{len(all_stats)} 层")
    print(f"  Eff_Rank: {large_eff}/{len(all_stats)} 层")
    print(f"  Entropy: {large_ent}/{len(all_stats)} 层")


def save_json_results(all_stats: List[Dict], metadata: Dict, output_path: Path):
    """保存JSON结果"""
    # 转换numpy类型为Python原生类型
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    output = convert({
        'metadata': metadata,
        'layer_results': all_stats,
    })

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n结果已保存到: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='分析OMniMath几何特征结果')
    parser.add_argument('--cache-dir', type=str, default='/gz-data/research/demo/data/hidden/cache/omnimath')
    parser.add_argument('--npz-path', type=str, default='/gz-data/research/demo/data/features/full_omnimath.npz')
    parser.add_argument('--output', type=str, default='/gz-data/research/demo/data/results/omnimath_analysis.json')

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载轨迹
    trajectories = load_trajectories_simple(cache_dir)
    print(f"成功加载 {len(trajectories)} 条轨迹")

    if len(trajectories) == 0:
        print("错误: 没有加载到任何轨迹！")
        print(f"缓存目录: {cache_dir}")
        print(f"请确认缓存是否存在")
        return

    # 加载元数据
    data = np.load(args.npz_path, allow_pickle=True)
    metadata = {
        'subset': 'omnimath',
        'n_chains': len(data['problem_ids']),
        'n_correct': int(np.sum(data['is_correct_strict'] == 1)),  # npz约定: 1=correct
        'n_error': int(np.sum(data['is_correct_strict'] == 0)),
    }

    # 分析每一层
    all_stats = []
    for layer in HIDDEN_LAYERS:
        print(f"\n分析 Layer {layer}...")
        metrics = extract_step_features(trajectories, layer)
        stats = compute_statistics(metrics)
        all_stats.append(stats)

        print(f"  正确steps: {stats['n_correct_steps']}, 错误steps: {stats['n_error_steps']}")
        if stats['kappa']:
            print(f"  Kappa d={stats['kappa']['cohens_d']:.3f}, p={stats['kappa']['p_value']:.4f}")

    # 打印报告
    print_summary_report(all_stats, metadata)
    print_latex_table(all_stats)

    # 保存JSON
    try:
        save_json_results(all_stats, metadata, output_path)
    except Exception as e:
        print(f"\nJSON保存失败: {e}")
        print("但统计结果已在上方显示")


if __name__ == "__main__":
    main()
