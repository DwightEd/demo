#!/usr/bin/env python3
"""相变检测：Layer 3 - Phase Transition Detection

包含两种检测模式：
1. Shallow Lock-in（浅层信息流锁定）：浅层平滑度突降
2. Deep Decay（深层信息衰减）：深层谱演化失稳 + 晚期谱熵低
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from data_loading import ReasoningTrajectory
from trajectory_geometry import (
    geometric_sim, geometric_sim_scalar_only,
    spectral_evolution_stability, extract_scalar_sequence
)


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class LockinResult:
    """Shallow Lock-in检测结果"""
    detected: bool
    layer: Optional[int]
    lockin_step: int
    drop_magnitude: float
    coherence_before: float
    coherence_after: float
    coherence_profile: np.ndarray


@dataclass
class DecayResult:
    """Deep Decay检测结果"""
    detected: bool
    layer: Optional[int]
    stability: float
    late_entropy: float
    entropy_threshold: float
    stability_threshold: float


@dataclass
class PhaseTransitionResult:
    """综合相变检测结果"""
    has_transition: bool
    transition_type: str  # 'none', 'lockin', 'decay', 'both'
    lockin: LockinResult
    decay: DecayResult


# =============================================================================
# Shallow Lock-in 检测
# =============================================================================

def compute_coherence_profile(geometry_sequence: List) -> np.ndarray:
    """计算coherence profile：每一步与之前所有步骤的平均相似度

    Args:
        geometry_sequence: 步骤几何序列

    Returns:
        coherence_profile数组，长度为len(geometry_sequence)-1
    """
    if len(geometry_sequence) < 2:
        return np.array([])

    coherence_profile = []
    for i in range(1, len(geometry_sequence)):
        sim_to_past = []
        for j in range(i):
            g1 = geometry_sequence[i]
            g2 = geometry_sequence[j]

            if g1.principal_directions.size > 0 and g2.principal_directions.size > 0:
                sim = geometric_sim(g1, g2)
            else:
                sim = geometric_sim_scalar_only(g1, g2)

            if np.isfinite(sim):
                sim_to_past.append(sim)

        if sim_to_past:
            coherence_profile.append(np.mean(sim_to_past))
        else:
            coherence_profile.append(0.5)

    return np.array(coherence_profile)


def detect_shallow_lockin(trajectory: ReasoningTrajectory,
                         shallow_layers: List[int] = [10, 14, 18],
                         drop_threshold: float = 0.15,
                         min_window: int = 2) -> LockinResult:
    """检测Shallow Lock-in：浅层平滑度突降

    检测逻辑：
    1. 计算coherence profile：每一步与之前所有步骤的平均相似度
    2. 检测是否有突然下降（> threshold）

    Args:
        trajectory: 推理轨迹
        shallow_layers: 浅层列表
        drop_threshold: 下降阈值
        min_window: 窗口大小

    Returns:
        LockinResult
    """
    for layer in shallow_layers:
        geom_sequence = trajectory.get_geometry_sequence(layer)
        if len(geom_sequence) < 3:
            continue

        # 计算coherence profile
        coherence_profile = compute_coherence_profile(geom_sequence)

        if len(coherence_profile) < min_window * 2:
            continue

        # 检测突然下降
        for i in range(min_window, len(coherence_profile) - min_window):
            before = np.mean(coherence_profile[i-min_window:i])
            after = np.mean(coherence_profile[i:i+min_window])

            if before - after > drop_threshold:
                return LockinResult(
                    detected=True,
                    layer=layer,
                    lockin_step=i + 1,  # 转换为step_id
                    drop_magnitude=float(before - after),
                    coherence_before=float(before),
                    coherence_after=float(after),
                    coherence_profile=coherence_profile,
                )

    return LockinResult(
        detected=False,
        layer=None,
        lockin_step=-1,
        drop_magnitude=0.0,
        coherence_before=np.nan,
        coherence_after=np.nan,
        coherence_profile=np.array([]),
    )


def detect_shallow_lockin_adaptive(trajectory: ReasoningTrajectory,
                                  shallow_layers: List[int] = [10, 14, 18],
                                  drop_sigma: float = 2.0,
                                  min_window: int = 2) -> LockinResult:
    """自适应Shallow Lock-in检测：使用统计阈值

    检测逻辑：
    1. 计算coherence profile
    2. 使用统计阈值：mean - drop_sigma * std

    Args:
        trajectory: 推理轨迹
        shallow_layers: 浅层列表
        drop_sigma: 下降的标准差倍数
        min_window: 窗口大小

    Returns:
        LockinResult
    """
    for layer in shallow_layers:
        geom_sequence = trajectory.get_geometry_sequence(layer)
        if len(geom_sequence) < 5:
            continue

        coherence_profile = compute_coherence_profile(geom_sequence)

        if len(coherence_profile) < min_window * 2:
            continue

        # 计算统计阈值
        mean_coh = np.mean(coherence_profile)
        std_coh = np.std(coherence_profile)
        threshold = mean_coh - drop_sigma * std_coh

        # 检测是否有连续低于阈值的区域
        for i in range(len(coherence_profile) - min_window):
            window = coherence_profile[i:i+min_window]
            if np.mean(window) < threshold:
                return LockinResult(
                    detected=True,
                    layer=layer,
                    lockin_step=i,
                    drop_magnitude=float(mean_coh - np.mean(window)),
                    coherence_before=float(mean_coh),
                    coherence_after=float(np.mean(window)),
                    coherence_profile=coherence_profile,
                )

    return LockinResult(
        detected=False,
        layer=None,
        lockin_step=-1,
        drop_magnitude=0.0,
        coherence_before=np.nan,
        coherence_after=np.nan,
        coherence_profile=np.array([]),
    )


# =============================================================================
# Deep Decay 检测
# =============================================================================

def spectral_entropy_from_geom(spectrum: np.ndarray) -> float:
    """从谱分布计算谱熵"""
    eps = 1e-12
    s = spectrum / (spectrum.sum() + eps)
    s = s[s > 0]
    return float(-np.sum(s * np.log(s + eps)))


def detect_deep_decay(trajectory: ReasoningTrajectory,
                     deep_layers: List[int] = [18, 22],
                     stability_threshold: float = 0.5,
                     entropy_threshold: float = 0.7,
                     n_late: int = 3) -> DecayResult:
    """检测Deep Decay：深层谱演化失稳

    检测逻辑：
    1. 计算谱演化稳定性
    2. 计算后期步骤的平均谱熵
    3. 检测：stability < threshold AND late_entropy < threshold

    Args:
        trajectory: 推理轨迹
        deep_layers: 深层列表
        stability_threshold: 稳定性阈值
        entropy_threshold: 谱熵阈值
        n_late: 后期步骤数

    Returns:
        DecayResult
    """
    for layer in deep_layers:
        geom_sequence = trajectory.get_geometry_sequence(layer)
        if len(geom_sequence) < 4:
            continue

        # 计算谱演化稳定性
        stability = spectral_evolution_stability(geom_sequence)
        if not np.isfinite(stability):
            continue

        # 计算后期步骤的平均谱熵
        late_spectra = [g.eigenvalues[:10] for g in geom_sequence[-n_late:]
                        if g.eigenvalues.size >= 2]
        if not late_spectra:
            continue

        late_entropies = [spectral_entropy_from_geom(s) for s in late_spectra]
        late_entropy = np.mean(late_entropies)

        # Decay信号：稳定性低 AND 晚期谱熵低（信息丢失）
        if stability < stability_threshold and late_entropy < entropy_threshold:
            return DecayResult(
                detected=True,
                layer=layer,
                stability=float(stability),
                late_entropy=float(late_entropy),
                entropy_threshold=entropy_threshold,
                stability_threshold=stability_threshold,
            )

    return DecayResult(
        detected=False,
        layer=None,
        stability=np.nan,
        late_entropy=np.nan,
        entropy_threshold=entropy_threshold,
        stability_threshold=stability_threshold,
    )


def detect_deep_decay_alternative(trajectory: ReasoningTrajectory,
                                  deep_layers: List[int] = [18, 22],
                                  stability_quantile: float = 0.25,
                                  entropy_quantile: float = 0.25,
                                  n_late: int = 3) -> DecayResult:
    """自适应Deep Decay检测：使用分位数阈值

    检测逻辑：稳定性低于Q1且谱熵低于Q1
    """
    for layer in deep_layers:
        geom_sequence = trajectory.get_geometry_sequence(layer)
        if len(geom_sequence) < 4:
            continue

        stability = spectral_evolution_stability(geom_sequence)
        if not np.isfinite(stability):
            continue

        late_spectra = [g.eigenvalues[:10] for g in geom_sequence[-n_late:]
                        if g.eigenvalues.size >= 2]
        if not late_spectra:
            continue

        late_entropies = [spectral_entropy_from_geom(s) for s in late_spectra]
        late_entropy = np.mean(late_entropies)

        # 使用固定阈值（基于经验）
        if stability < 0.3 and late_entropy < 0.5:
            return DecayResult(
                detected=True,
                layer=layer,
                stability=float(stability),
                late_entropy=float(late_entropy),
                entropy_threshold=entropy_quantile,
                stability_threshold=stability_quantile,
            )

    return DecayResult(
        detected=False,
        layer=None,
        stability=np.nan,
        late_entropy=np.nan,
        entropy_threshold=entropy_quantile,
        stability_threshold=stability_quantile,
    )


# =============================================================================
# 综合检测
# =============================================================================

def detect_phase_transition(trajectory: ReasoningTrajectory,
                            layer: int = 14,
                            method: str = 'standard') -> PhaseTransitionResult:
    """组合检测信号

    Args:
        trajectory: 推理轨迹
        layer: 主要分析的层
        method: 'standard' 或 'adaptive'

    Returns:
        PhaseTransitionResult
    """
    if method == 'adaptive':
        lockin = detect_shallow_lockin_adaptive(
            trajectory,
            shallow_layers=[layer],
        )
    else:
        lockin = detect_shallow_lockin(
            trajectory,
            shallow_layers=[layer],
        )

    decay = detect_deep_decay(
        trajectory,
        deep_layers=[layer],
    )

    # 确定transition类型
    if lockin.detected and decay.detected:
        transition_type = 'both'
    elif lockin.detected:
        transition_type = 'lockin'
    elif decay.detected:
        transition_type = 'decay'
    else:
        transition_type = 'none'

    return PhaseTransitionResult(
        has_transition=lockin.detected or decay.detected,
        transition_type=transition_type,
        lockin=lockin,
        decay=decay,
    )


# =============================================================================
# 批量检测
# =============================================================================

def batch_detect_lockin(trajectories: List[ReasoningTrajectory],
                       layer: int = 14,
                       method: str = 'standard',
                       verbose: bool = False) -> List[LockinResult]:
    """批量检测Shallow Lock-in

    Args:
        trajectories: 轨迹列表
        layer: 分析的层
        method: 'standard' 或 'adaptive'
        verbose: 显示进度

    Returns:
        检测结果列表
    """
    results = []

    iterator = trajectories
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(trajectories, desc=f"Detecting lock-in (L{layer})")

    for traj in iterator:
        if method == 'adaptive':
            result = detect_shallow_lockin_adaptive(traj, shallow_layers=[layer])
        else:
            result = detect_shallow_lockin(traj, shallow_layers=[layer])
        results.append(result)

    return results


def batch_detect_decay(trajectories: List[ReasoningTrajectory],
                       layer: int = 14,
                       verbose: bool = False) -> List[DecayResult]:
    """批量检测Deep Decay

    Args:
        trajectories: 轨迹列表
        layer: 分析的层
        verbose: 显示进度

    Returns:
        检测结果列表
    """
    results = []

    iterator = trajectories
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(trajectories, desc=f"Detecting decay (L{layer})")

    for traj in iterator:
        result = detect_deep_decay(traj, deep_layers=[layer])
        results.append(result)

    return results


def batch_detect_phase_transition(trajectories: List[ReasoningTrajectory],
                                 layer: int = 14,
                                 method: str = 'standard',
                                 verbose: bool = False) -> List[PhaseTransitionResult]:
    """批量检测相变

    Args:
        trajectories: 轨迹列表
        layer: 分析的层
        method: 'standard' 或 'adaptive'
        verbose: 显示进度

    Returns:
        检测结果列表
    """
    results = []

    iterator = trajectories
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(trajectories, desc=f"Detecting transitions (L{layer})")

    for traj in iterator:
        result = detect_phase_transition(traj, layer=layer, method=method)
        results.append(result)

    return results


# =============================================================================
# 统计函数
# =============================================================================

def compute_lockin_statistics(lockin_results: List[LockinResult],
                             is_correct_list: List[bool]) -> Dict:
    """计算Lock-in检测统计

    Returns:
        {
            'total': int,
            'correct_detected': int,
            'error_detected': int,
            'correct_total': int,
            'error_total': int,
            'correct_rate': float,
            'error_rate': float,
        }
    """
    is_correct = np.array(is_correct_list)

    correct_detected = sum(1 for r, ic in zip(lockin_results, is_correct)
                          if r.detected and ic)
    error_detected = sum(1 for r, ic in zip(lockin_results, is_correct)
                        if r.detected and not ic)
    correct_total = sum(is_correct)
    error_total = sum(~is_correct)

    return {
        'total': len(lockin_results),
        'correct_detected': correct_detected,
        'error_detected': error_detected,
        'correct_total': correct_total,
        'error_total': error_total,
        'correct_rate': correct_detected / correct_total if correct_total > 0 else 0,
        'error_rate': error_detected / error_total if error_total > 0 else 0,
    }


def compute_decay_statistics(decay_results: List[DecayResult],
                            is_correct_list: List[bool]) -> Dict:
    """计算Decay检测统计"""
    return compute_lockin_statistics(decay_results, is_correct_list)  # 同样结构


def print_detection_summary(lockin_stats: Dict, decay_stats: Dict):
    """打印检测统计摘要"""
    print("\n" + "=" * 80)
    print("Phase Transition Detection Summary")
    print("=" * 80)

    print("\nShallow Lock-in:")
    print(f"  Correct: {lockin_stats['correct_detected']}/{lockin_stats['correct_total']} "
          f"({lockin_stats['correct_rate']:.1%})")
    print(f"  Error:   {lockin_stats['error_detected']}/{lockin_stats['error_total']} "
          f"({lockin_stats['error_rate']:.1%})")

    print("\nDeep Decay:")
    print(f"  Correct: {decay_stats['correct_detected']}/{decay_stats['correct_total']} "
          f"({decay_stats['correct_rate']:.1%})")
    print(f"  Error:   {decay_stats['error_detected']}/{decay_stats['error_total']} "
          f"({decay_stats['error_rate']:.1%})")

    # Odds ratio for lock-in
    a = lockin_stats['error_detected']
    b = lockin_stats['error_total'] - lockin_stats['error_detected']
    c = lockin_stats['correct_detected']
    d = lockin_stats['correct_total'] - lockin_stats['correct_detected']

    if (a + c) > 0 and (b + d) > 0:
        or_lockin = (a / (a + c)) / (c / (c + d)) if c > 0 else np.inf
        print(f"\nOdds Ratio (Lock-in): {or_lockin:.2f}")

    print("=" * 80)


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

    # 检测相变
    lockin_results = batch_detect_lockin(filtered, layer=14, verbose=True)
    decay_results = batch_detect_decay(filtered, layer=14, verbose=True)

    # 计算统计
    is_correct_list = [t.is_correct for t in filtered]
    lockin_stats = compute_lockin_statistics(lockin_results, is_correct_list)
    decay_stats = compute_decay_statistics(decay_results, is_correct_list)

    # 打印摘要
    print_detection_summary(lockin_stats, decay_stats)
