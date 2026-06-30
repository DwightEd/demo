#!/usr/bin/env python3
"""验证实验：H1-H4 假设检验

包含：
- H1: 轨迹平滑度区分error vs correct (Mann-Whitney U)
- H2: Shallow Lock-in模式在error中更频繁 (Fisher's exact)
- H3: Deep Decay模式在error中更频繁 (Fisher's exact)
- H4: 基于轨迹的检测器优于基于单步几何的检测器 (Bootstrap AUROC比较)
"""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from data_loading import ReasoningTrajectory, get_trajectory_with_min_steps
from trajectory_geometry import (
    TrajectoryMetrics, compute_trajectory_metrics,
    compute_all_metrics
)
from phase_transition import (
    LockinResult, DecayResult,
    batch_detect_lockin, batch_detect_decay,
    compute_lockin_statistics, compute_decay_statistics
)


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class ValidationResult:
    """单个验证实验的结果"""
    hypothesis: str  # 'H1', 'H2', 'H3', 'H4'
    metric: str
    layer: int

    # 统计量
    error_mean: float
    correct_mean: float
    mean_diff: float

    # 假设检验
    test_type: str  # 'Mann-Whitney U', 'Fisher exact', 'Bootstrap'
    statistic: float
    p_value: float
    cohens_d: Optional[float] = None
    odds_ratio: Optional[float] = None

    # 置信区间
    ci_lower: float = np.nan
    ci_upper: float = np.nan

    # 样本量
    n_error: int = 0
    n_correct: int = 0

    # 解释
    interpretation: str = ""
    significant: bool = False


# =============================================================================
# Bootstrap方法
# =============================================================================

def bootstrap_mean_diff(arr1: np.ndarray,
                       arr2: np.ndarray,
                       n_bootstrap: int = 5000,
                       confidence: float = 0.95,
                       seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap计算均值差异的置信区间

    Args:
        arr1: 第一组数据（通常是correct）
        arr2: 第二组数据（通常是error）
        n_bootstrap: Bootstrap次数
        confidence: 置信水平
        seed: 随机种子

    Returns:
        (mean_diff, ci_lower, ci_upper)
    """
    np.random.seed(seed)

    # 移除NaN
    arr1_clean = arr1[~np.isnan(arr1)]
    arr2_clean = arr2[~np.isnan(arr2)]

    if len(arr1_clean) == 0 or len(arr2_clean) == 0:
        return np.nan, np.nan, np.nan

    # 观测到的均值差异
    observed_diff = np.mean(arr1_clean) - np.mean(arr2_clean)

    # Bootstrap
    boot_diffs = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(arr1_clean, size=len(arr1_clean), replace=True)
        sample2 = np.random.choice(arr2_clean, size=len(arr2_clean), replace=True)
        boot_diff = np.mean(sample1) - np.mean(sample2)
        boot_diffs.append(boot_diff)

    boot_diffs = np.array(boot_diffs)

    # 计算置信区间
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    return float(observed_diff), float(ci_lower), float(ci_upper)


def bootstrap_auroc_diff(y_true: np.ndarray,
                        y_score1: np.ndarray,
                        y_score2: np.ndarray,
                        n_bootstrap: int = 5000,
                        seed: int = 42) -> Tuple[float, float, float, float]:
    """Bootstrap比较两个AUROC的差异

    Args:
        y_true: 真实标签 (1=error, 0=correct)
        y_score1: 第一个检测器的分数
        y_score2: 第二个检测器的分数
        n_bootstrap: Bootstrap次数
        seed: 随机种子

    Returns:
        (auroc1, auroc2, diff, p_value)
    """
    np.random.seed(seed)

    # 移除NaN
    valid_mask = ~(np.isnan(y_score1) | np.isnan(y_score2))
    y_true_v = y_true[valid_mask]
    y_score1_v = y_score1[valid_mask]
    y_score2_v = y_score2[valid_mask]

    if len(np.unique(y_true_v)) < 2:
        return np.nan, np.nan, np.nan, np.nan

    # 观测到的AUROC
    try:
        auroc1 = roc_auc_score(y_true_v, y_score1_v)
        auroc2 = roc_auc_score(y_true_v, y_score2_v)
        observed_diff = auroc1 - auro2
    except:
        return np.nan, np.nan, np.nan, np.nan

    # Bootstrap
    boot_diffs = []
    n1_better = 0

    for _ in range(n_bootstrap):
        # Bootstrap采样
        indices = np.random.choice(len(y_true_v), size=len(y_true_v), replace=True)
        y_true_boot = y_true_v[indices]
        y_score1_boot = y_score1_v[indices]
        y_score2_boot = y_score2_v[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            auroc1_boot = roc_auc_score(y_true_boot, y_score1_boot)
            auroc2_boot = roc_auc_score(y_true_boot, y_score2_boot)
            boot_diff = auroc1_boot - auroc2_boot
            boot_diffs.append(boot_diff)

            if boot_diff > 0:
                n1_better += 1
        except:
            continue

    if len(boot_diffs) == 0:
        return auroc1, auroc2, observed_diff, np.nan

    # p-value: 方法1比方法2好的比例
    p_value = 1.0 - n1_better / len(boot_diffs)

    return float(auroc1), float(auroc2), float(observed_diff), float(p_value)


def cohens_d(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """计算Cohen's d效应量

    Args:
        arr1: 第一组数据
        arr2: 第二组数据

    Returns:
        Cohen's d
    """
    # 移除NaN
    arr1_clean = arr1[~np.isnan(arr1)]
    arr2_clean = arr2[~np.isnan(arr2)]

    if len(arr1_clean) == 0 or len(arr2_clean) == 0:
        return np.nan

    mean1, mean2 = np.mean(arr1_clean), np.mean(arr2_clean)
    var1, var2 = np.var(arr1_clean, ddof=1), np.var(arr2_clean, ddof=1)
    n1, n2 = len(arr1_clean), len(arr2_clean)

    # 合并标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    d = (mean1 - mean2) / pooled_std
    return float(d)


def fishers_exact(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
    """Fisher's exact test

    Args:
        a, b: 第一组的检测结果/未检测数
        c, d: 第二组的检测结果/未检测数

    Returns:
        (odds_ratio, p_value)
    """
    if (a + c) == 0 or (b + d) == 0:
        return np.nan, np.nan

    try:
        oddsratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
        return float(oddsratio), float(p_value)
    except:
        return np.nan, np.nan


# =============================================================================
# H1: 轨迹平滑度区分
# =============================================================================

def run_validation_h1(trajectories: List[ReasoningTrajectory],
                    layer: int = 14,
                    n_bootstrap: int = 5000) -> ValidationResult:
    """H1: 轨迹平滑度区分error vs correct

    假设: 错误推理的轨迹smoothness低于正确推理
    检验: Mann-Whitney U test (one-tailed: error < correct)
    """
    # 计算指标
    metrics_list = compute_all_metrics(trajectories, layer=layer, verbose=False)

    # 分离correct/error
    correct_smoothness = []
    error_smoothness = []

    for traj, metrics in zip(trajectories, metrics_list):
        if np.isfinite(metrics.smoothness):
            if traj.is_correct:
                correct_smoothness.append(metrics.smoothness)
            else:
                error_smoothness.append(metrics.smoothness)

    correct_smoothness = np.array(correct_smoothness)
    error_smoothness = np.array(error_smoothness)

    n_correct = len(correct_smoothness)
    n_error = len(error_smoothness)

    if n_correct == 0 or n_error == 0:
        return ValidationResult(
            hypothesis='H1',
            metric='smoothness',
            layer=layer,
            error_mean=np.nan,
            correct_mean=np.nan,
            mean_diff=np.nan,
            test_type='N/A',
            statistic=np.nan,
            p_value=np.nan,
            n_error=n_error,
            n_correct=n_correct,
            interpretation='Insufficient data',
        )

    # Mann-Whitney U test
    stat, pval = stats.mannwhitneyu(error_smoothness, correct_smoothness, alternative='less')

    # Cohen's d
    d = cohens_d(correct_smoothness, error_smoothness)

    # Bootstrap CI
    mean_diff, ci_low, ci_high = bootstrap_mean_diff(
        correct_smoothness, error_smoothness, n_bootstrap=n_bootstrap
    )

    # 解释
    significant = pval < 0.05
    if significant:
        if d < 0.2:
            interp = "Significant but small effect"
        elif d < 0.5:
            interp = "Significant with medium effect"
        else:
            interp = "Significant with large effect"
    else:
        interp = "Not significant"

    return ValidationResult(
        hypothesis='H1',
        metric='smoothness',
        layer=layer,
        error_mean=float(np.mean(error_smoothness)),
        correct_mean=float(np.mean(correct_smoothness)),
        mean_diff=mean_diff,
        test_type='Mann-Whitney U',
        statistic=float(stat),
        p_value=float(pval),
        cohens_d=float(d),
        ci_lower=float(ci_low),
        ci_upper=float(ci_high),
        n_error=n_error,
        n_correct=n_correct,
        interpretation=interp,
        significant=significant,
    )


# =============================================================================
# H2: Shallow Lock-in频率
# =============================================================================

def run_validation_h2(trajectories: List[ReasoningTrajectory],
                    layer: int = 14) -> ValidationResult:
    """H2: Shallow Lock-in模式在error中更频繁

    假设: 错误推理中检测到Shallow Lock-in的比例更高
    检验: Fisher's exact test (alternative='greater')
    """
    # 检测
    lockin_results = batch_detect_lockin(trajectories, layer=layer, verbose=False)

    # 统计
    is_correct = np.array([t.is_correct for t in trajectories])

    error_detected = sum(1 for r, ic in zip(lockin_results, is_correct) if r.detected and not ic)
    error_total = sum(~is_correct)
    correct_detected = sum(1 for r, ic in zip(lockin_results, is_correct) if r.detected and ic)
    correct_total = sum(is_correct)

    error_not_detected = error_total - error_detected
    correct_not_detected = correct_total - correct_detected

    # Fisher's exact
    # [[error_detected, error_not_detected],
    #  [correct_detected, correct_not_detected]]
    or_val, pval = fishers_exact(error_detected, error_not_detected,
                                correct_detected, correct_not_detected)

    # 解释
    significant = pval < 0.05
    if significant:
        interp = f"Lock-in more frequent in error (OR={or_val:.2f})"
    else:
        interp = "No significant difference"

    return ValidationResult(
        hypothesis='H2',
        metric='shallow_lockin',
        layer=layer,
        error_mean=float(error_detected) / error_total if error_total > 0 else np.nan,
        correct_mean=float(correct_detected) / correct_total if correct_total > 0 else np.nan,
        mean_diff=float(error_detected / error_total - correct_detected / correct_total),
        test_type='Fisher exact',
        statistic=np.nan,
        p_value=float(pval),
        odds_ratio=float(or_val) if np.isfinite(or_val) else np.nan,
        n_error=error_total,
        n_correct=correct_total,
        interpretation=interp,
        significant=significant,
    )


# =============================================================================
# H3: Deep Decay频率
# =============================================================================

def run_validation_h3(trajectories: List[ReasoningTrajectory],
                    layer: int = 14) -> ValidationResult:
    """H3: Deep Decay模式在error中更频繁

    假设: 错误推理中检测到Deep Decay的比例更高
    检验: Fisher's exact test (alternative='greater')
    """
    # 检测
    decay_results = batch_detect_decay(trajectories, layer=layer, verbose=False)

    # 统计
    is_correct = np.array([t.is_correct for t in trajectories])

    error_detected = sum(1 for r, ic in zip(decay_results, is_correct) if r.detected and not ic)
    error_total = sum(~is_correct)
    correct_detected = sum(1 for r, ic in zip(decay_results, is_correct) if r.detected and ic)
    correct_total = sum(is_correct)

    error_not_detected = error_total - error_detected
    correct_not_detected = correct_total - correct_detected

    # Fisher's exact
    or_val, pval = fishers_exact(error_detected, error_not_detected,
                                correct_detected, correct_not_detected)

    # 解释
    significant = pval < 0.05
    if significant:
        interp = f"Decay more frequent in error (OR={or_val:.2f})"
    else:
        interp = "No significant difference"

    return ValidationResult(
        hypothesis='H3',
        metric='deep_decay',
        layer=layer,
        error_mean=float(error_detected) / error_total if error_total > 0 else np.nan,
        correct_mean=float(correct_detected) / correct_total if correct_total > 0 else np.nan,
        mean_diff=float(error_detected / error_total - correct_detected / correct_total),
        test_type='Fisher exact',
        statistic=np.nan,
        p_value=float(pval),
        odds_ratio=float(or_val) if np.isfinite(or_val) else np.nan,
        n_error=error_total,
        n_correct=correct_total,
        interpretation=interp,
        significant=significant,
    )


# =============================================================================
# H4: 轨迹 vs 单步
# =============================================================================

def run_validation_h4(trajectories: List[ReasoningTrajectory],
                    layer: int = 14,
                    n_bootstrap: int = 5000) -> ValidationResult:
    """H4: 基于轨迹的检测器优于基于单步几何的检测器

    假设: 轨迹级指标（smoothness + coherence + stability）优于单步指标（mean kappa）
    检验: Bootstrap比较AUROC
    """
    # 计算轨迹级指标
    metrics_list = compute_all_metrics(trajectories, layer=layer, verbose=False)

    # 计算单步指标（mean kappa）
    stepwise_scores = []
    trajectory_scores = []
    labels = []

    for traj, metrics in zip(trajectories, metrics_list):
        if not traj.is_correct is None:
            labels.append(1 if not traj.is_correct else 0)

            # 单步指标：1 - mean(kappa)（error倾向于低kappa）
            geom_seq = traj.get_geometry_sequence(layer)
            if geom_seq:
                kappas = [g.kappa for g in geom_seq if np.isfinite(g.kappa)]
                if kappas:
                    stepwise_score = 1.0 - np.mean(kappas)  # 转换：低kappa→高分
                    stepwise_scores.append(stepwise_score)
                else:
                    stepwise_scores.append(np.nan)
            else:
                stepwise_scores.append(np.nan)

            # 轨迹指标：1 - smoothness（error倾向于低smoothness）
            if np.isfinite(metrics.smoothness):
                trajectory_score = 1.0 - metrics.smoothness
                trajectory_scores.append(trajectory_score)
            else:
                trajectory_scores.append(np.nan)

    labels = np.array(labels)
    stepwise_scores = np.array(stepwise_scores)
    trajectory_scores = np.array(trajectory_scores)

    # 移除NaN
    valid_mask = ~(np.isnan(stepwise_scores) | np.isnan(trajectory_scores))
    labels_v = labels[valid_mask]
    stepwise_scores_v = stepwise_scores[valid_mask]
    trajectory_scores_v = trajectory_scores[valid_mask]

    n_error = np.sum(labels_v == 1)
    n_correct = np.sum(labels_v == 0)

    if len(np.unique(labels_v)) < 2:
        return ValidationResult(
            hypothesis='H4',
            metric='trajectory_vs_stepwise',
            layer=layer,
            error_mean=np.nan,
            correct_mean=np.nan,
            mean_diff=np.nan,
            test_type='Bootstrap AUROC',
            statistic=np.nan,
            p_value=np.nan,
            n_error=n_error,
            n_correct=n_correct,
            interpretation='Insufficient data for AUROC comparison',
        )

    # Bootstrap比较
    auroc_traj, auroc_step, diff, pval = bootstrap_auroc_diff(
        labels_v, trajectory_scores_v, stepwise_scores_v, n_bootstrap=n_bootstrap
    )

    # 解释
    significant = pval < 0.05
    if significant:
        if diff > 0:
            interp = f"Trajectory significantly better (Δ={diff:.3f})"
        else:
            interp = f"Stepwise significantly better (Δ={diff:.3f})"
    else:
        interp = "No significant difference"

    return ValidationResult(
        hypothesis='H4',
        metric='trajectory_vs_stepwise',
        layer=layer,
        error_mean=np.nan,
        correct_mean=np.nan,
        mean_diff=diff,
        test_type='Bootstrap AUROC',
        statistic=np.nan,
        p_value=float(pval),
        ci_lower=float(diff),  # 简化：使用diff作为点估计
        ci_upper=float(diff),
        n_error=n_error,
        n_correct=n_correct,
        interpretation=interp,
        significant=significant,
    )


# =============================================================================
# 运行所有验证
# =============================================================================

def run_all_validations(trajectories: List[ReasoningTrajectory],
                       layers: List[int] = [14],
                       n_bootstrap: int = 5000,
                       verbose: bool = True) -> Dict[str, ValidationResult]:
    """运行所有验证实验

    Args:
        trajectories: 轨迹列表
        layers: 要分析的层列表
        n_bootstrap: Bootstrap次数
        verbose: 显示进度

    Returns:
        {f"L{layer}_H{h}": ValidationResult}
    """
    results = {}

    for layer in layers:
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Running validations for Layer {layer}")
            print(f"{'=' * 80}")

        # H1
        if verbose:
            print("Running H1: smoothness...")
        h1_result = run_validation_h1(trajectories, layer=layer, n_bootstrap=n_bootstrap)
        results[f"L{layer}_H1"] = h1_result

        # H2
        if verbose:
            print("Running H2: shallow lock-in...")
        h2_result = run_validation_h2(trajectories, layer=layer)
        results[f"L{layer}_H2"] = h2_result

        # H3
        if verbose:
            print("Running H3: deep decay...")
        h3_result = run_validation_h3(trajectories, layer=layer)
        results[f"L{layer}_H3"] = h3_result

        # H4
        if verbose:
            print("Running H4: trajectory vs stepwise...")
        h4_result = run_validation_h4(trajectories, layer=layer, n_bootstrap=n_bootstrap)
        results[f"L{layer}_H4"] = h4_result

    return results


def print_validation_summary(results: Dict[str, ValidationResult]):
    """打印验证结果摘要"""
    print("\n" + "=" * 100)
    print("VALIDATION SUMMARY")
    print("=" * 100)
    print(f"{'Test':<12} {'Metric':<25} {'Error':<10} {'Correct':<10} {'Diff':<10} "
          f"{'Test':<15} {'p-value':<10} {'Sig?'}")
    print("-" * 100)

    for key, result in sorted(results.items()):
        # 格式化统计量
        if result.test_type == 'Mann-Whitney U':
            stat_str = f"d={result.cohens_d:.2f}"
        elif result.test_type == 'Fisher exact':
            stat_str = f"OR={result.odds_ratio:.2f}"
        elif result.test_type == 'Bootstrap AUROC':
            stat_str = f"Δ={result.mean_diff:.3f}"
        else:
            stat_str = "N/A"

        # 格式化值
        if result.metric in ['shallow_lockin', 'deep_decay']:
            # 比例
            error_str = f"{result.error_mean:.1%}"
            correct_str = f"{result.correct_mean:.1%}"
            diff_str = f"{result.mean_diff:.1%}"
        elif result.metric == 'trajectory_vs_stepwise':
            error_str = "N/A"
            correct_str = "N/A"
            diff_str = f"{result.mean_diff:.3f}"
        else:
            # 数值
            error_str = f"{result.error_mean:.3f}"
            correct_str = f"{result.correct_mean:.3f}"
            diff_str = f"{result.mean_diff:.3f}"

        sig_str = "*" if result.significant else ""

        print(f"{key:<12} {result.metric:<25} {error_str:<10} {correct_str:<10} {diff_str:<10} "
              f"{stat_str:<15} {result.p_value:<10.4f} {sig_str}")

    print("-" * 100)
    print("* p < 0.05")
    print("=" * 100)


def save_validation_results(results: Dict[str, ValidationResult],
                            metadata: Dict,
                            output_path: str):
    """保存验证结果为JSON"""
    output = {
        'metadata': metadata,
        'results': {}
    }

    for key, result in results.items():
        output['results'][key] = {
            'hypothesis': result.hypothesis,
            'metric': result.metric,
            'layer': result.layer,
            'error_mean': result.error_mean,
            'correct_mean': result.correct_mean,
            'mean_diff': result.mean_diff,
            'test_type': result.test_type,
            'statistic': result.statistic,
            'p_value': result.p_value,
            'cohens_d': result.cohens_d,
            'odds_ratio': result.odds_ratio,
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper,
            'n_error': result.n_error,
            'n_correct': result.n_correct,
            'interpretation': result.interpretation,
            'significant': result.significant,
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    # 示例用法
    from data_loading import load_all_trajectories, get_trajectory_with_min_steps

    npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"
    hidden_dir = "/gz-data/research/demo/data/hidden/omnimath/"

    print("Loading trajectories...")
    trajectories, metadata = load_all_trajectories(
        npz_path=npz_path,
        hidden_dir=hidden_dir,
        verbose=True,
    )

    # 过滤出至少有3个步骤的轨迹
    filtered = get_trajectory_with_min_steps(trajectories, min_steps=3, layer=14)
    print(f"\nFiltered to {len(filtered)} trajectories with ≥3 steps")

    # 运行验证
    results = run_all_validations(filtered, layers=[14], n_bootstrap=1000, verbose=True)

    # 打印摘要
    print_validation_summary(results)

    # 保存结果
    save_validation_results(
        results,
        metadata={**metadata, 'n_filtered': len(filtered)},
        output_path="/gz-data/research/demo/trajectory_validation_results.json"
    )
