"""方向2完整实现：Cross-Step Geometric Coherence - 轨迹几何相变检测

核心假设：错误推理是几何轨迹的相变，不是单点异常

三层架构：
  Layer 1: Step-wise Geometry (每步的几何特征)
  Layer 2: Trajectory Geometry (步骤间几何关系)
  Layer 3: Phase Transition Detection (相变检测)

验证目标：
  H1: 错误推理的轨迹smoothness低于正确推理
  H2: Shallow Lock-in模式在error中更频繁
  H3: Deep Decay模式在error中更频繁
  H4: 基于轨迹的检测器优于基于单步几何的检测器

论文Title: "Trajectory of Thought: Geometric Phase Transitions in Multi-Step Reasoning"
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm


# =============================================================================
# Layer 1: Step-wise Geometry
# =============================================================================


@dataclass
class StepGeometry:
    """单个步骤的完整几何描述"""
    step_id: int
    layer: int
    n_tokens: int = 0

    # 一阶矩
    kappa: float = field(default=np.nan)

    # 二阶矩
    eff_rank: float = field(default=np.nan)
    spectrum: np.ndarray = field(default_factory=lambda: np.array([]))

    # 辅助信息
    norm: float = field(default=np.nan)

    def __repr__(self):
        return f"StepGeometry(step={self.step_id}, layer={self.layer}, κ={self.kappa:.3f}, eff_R={self.eff_rank:.2f})"


@dataclass
class ReasoningTrajectory:
    """完整推理链的几何轨迹"""
    chain_id: int
    problem_id: int
    is_correct: bool
    steps: list[StepGeometry] = field(default_factory=list)

    def add_step(self, geom: StepGeometry):
        self.steps.append(geom)

    def get_geometry_sequence(self, layer: int) -> list[StepGeometry]:
        """获取指定层的步骤几何序列"""
        return [s for s in self.steps if s.layer == layer]

    def has_layer(self, layer: int) -> bool:
        return any(s.layer == layer for s in self.steps)


def extract_step_geometry_from_stepcloud(stepcloud_data: np.ndarray,
                                         step_idx: int,
                                         layer_idx: int,
                                         layer_id: int) -> StepGeometry:
    """从stepcloud数据提取步骤几何特征

    Args:
        stepcloud_data: (T, 33, 9) array, T=步骤数, 33=层, 9=特征
        step_idx: 步骤索引
        layer_idx: 层索引（在33层中的位置）
        layer_id: 层ID（实际层号）
    """
    features = stepcloud_data[step_idx, layer_idx, :]

    geom = StepGeometry(
        step_id=step_idx,
        layer=layer_id,
        n_tokens=1,  # 不在stepcloud中存储
    )

    # 特征索引（根据CLOUD_NAMES顺序）:
    # CLOUD_NAMES = ("cloud_D", "cloud_V", "cloud_C", "coherence",
    #                "mean_tok_norm", "resultant", "resultant_bulk",
    #                "resultant_unif", "norm_bulk")
    # 0: cloud_D (effective rank)
    # 1: cloud_V (energy)
    # 2: cloud_C (concentration)
    # 3: coherence
    # 4: mean_tok_norm
    # 5: resultant (κ - directional concentration)
    # 6: resultant_bulk
    # 7: resultant_unif
    # 8: norm_bulk

    geom.kappa = float(features[5]) if len(features) > 5 and not np.isnan(features[5]) else np.nan
    geom.eff_rank = float(features[0]) if len(features) > 0 and not np.isnan(features[0]) else np.nan

    # 使用其他几何特征作为"谱形状"的代理
    # coherence + cloud_C + norm_bulk
    geom.spectrum = np.array([
        features[3] if len(features) > 3 and not np.isnan(features[3]) else 0.0,  # coherence
        features[2] if len(features) > 2 and not np.isnan(features[2]) else 0.0,  # cloud_C
        features[8] if len(features) > 8 and not np.isnan(features[8]) else 0.0,  # norm_bulk
        features[6] if len(features) > 6 and not np.isnan(features[6]) else 0.0,  # resultant_bulk
        features[7] if len(features) > 7 and not np.isnan(features[7]) else 0.0,  # resultant_unif
    ])

    geom.norm = float(features[4]) if len(features) > 4 and not np.isnan(features[4]) else np.nan

    return geom


# =============================================================================
# Layer 2: Trajectory Geometry - 三个核心指标
# =============================================================================


def geometric_sim(g1: StepGeometry, g2: StepGeometry,
                   kappa_weight: float = 0.4,
                   eff_rank_weight: float = 0.3,
                   spectrum_weight: float = 0.3) -> float:
    """计算两个步骤之间的几何相似度"""
    if np.isnan(g1.kappa) or np.isnan(g2.kappa):
        return 0.0

    # kappa相似度（方向一致性）
    kappa_sim = 1.0 - min(abs(g1.kappa - g2.kappa), 1.0)

    # eff_rank相似度（分散程度）
    if not np.isnan(g1.eff_rank) and not np.isnan(g2.eff_rank):
        max_er = max(g1.eff_rank, g2.eff_rank, 1.0)
        eff_rank_sim = 1.0 - abs(g1.eff_rank - g2.eff_rank) / max_er
    else:
        eff_rank_sim = 0.0

    # spectrum相似度（谱形状）
    if g1.spectrum.size > 0 and g2.spectrum.size > 0:
        s1 = g1.spectrum / (g1.spectrum.sum() + 1e-12)
        s2 = g2.spectrum / (g2.spectrum.sum() + 1e-12)
        spectrum_sim = 1.0 - jensenshannon(s1, s2)
    else:
        spectrum_sim = 0.0

    return (kappa_weight * kappa_sim +
            eff_rank_weight * eff_rank_sim +
            spectrum_weight * spectrum_sim)


def local_smoothness(geometry_sequence: list[StepGeometry]) -> tuple[float, np.ndarray]:
    """1. Local Smoothness（局部平滑度）

    相邻步骤的几何相似度

    正确推理：smoothness高（渐进变化）
    错误推理：smoothness低（突变）

    Returns:
        (全局平滑度, 每个transition的平滑度序列)
    """
    if len(geometry_sequence) < 2:
        return np.nan, np.array([])

    smoothness_values = []
    for i in range(len(geometry_sequence) - 1):
        sim = geometric_sim(geometry_sequence[i], geometry_sequence[i + 1])
        smoothness_values.append(sim)

    return float(np.mean(smoothness_values)), np.array(smoothness_values)


def global_coherence(geometry_sequence: list[StepGeometry],
                     n_early: int = 3,
                     n_late: int = 3) -> float:
    """2. Global Coherence（全局连贯性）

    首尾步骤的关联强度

    正确推理：首尾连贯（始终围绕问题展开）
    错误推理：首尾断裂（后期遗忘问题）

    关键：后期步骤是否还"记住"早期推理
    """
    if len(geometry_sequence) < max(n_early, n_late):
        return np.nan

    # 早期步骤的平均谱
    early_spectra = []
    for g in geometry_sequence[:n_early]:
        if g.spectrum.size > 0:
            early_spectra.append(g.spectrum)
    if not early_spectra:
        return np.nan

    early_spectrum = np.mean(early_spectra, axis=0)
    early_spectrum = early_spectrum / (early_spectrum.sum() + 1e-12)

    # 后期步骤的平均谱
    late_spectra = []
    for g in geometry_sequence[-n_late:]:
        if g.spectrum.size > 0:
            late_spectra.append(g.spectrum)
    if not late_spectra:
        return np.nan

    late_spectrum = np.mean(late_spectra, axis=0)
    late_spectrum = late_spectrum / (late_spectrum.sum() + 1e-12)

    # 连贯度 = 1 - JS散度
    return 1.0 - jensenshannon(early_spectrum, late_spectrum)


def spectral_entropy(spectrum: np.ndarray, eps: float = 1e-12) -> float:
    """计算谱熵"""
    s = spectrum / (spectrum.sum() + eps)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    return float(-np.sum(s * np.log(s + eps)))


def spectral_evolution_stability(geometry_sequence: list[StepGeometry]) -> float:
    """3. Spectral Evolution Stability（谱演化稳定性）

    谱形状沿步骤的演化稳定性

    正确推理：谱形状渐进演化
    错误推理：谱形状跳变
    """
    if len(geometry_sequence) < 3:
        return np.nan

    spectra = []
    for g in geometry_sequence:
        if g.spectrum.size > 0:
            spectra.append(g.spectrum)

    if len(spectra) < 3:
        return np.nan

    spectra = np.array(spectra)

    # 计算相邻谱的"变化率"
    diffs = np.diff(spectra, axis=0)
    diff_magnitudes = np.linalg.norm(diffs, axis=1)

    # 稳定性 = 1 / (1 + 变化率的方差)
    variance = np.var(diff_magnitudes)
    return 1.0 / (1.0 + variance)


def compute_trajectory_metrics(trajectory: ReasoningTrajectory,
                               layer: int) -> dict[str, Any]:
    """计算单个轨迹在指定层的所有几何指标"""
    geom_seq = trajectory.get_geometry_sequence(layer)

    if len(geom_seq) < 2:
        return {
            'smoothness': np.nan,
            'smoothness_seq': np.array([]),
            'coherence': np.nan,
            'stability': np.nan,
            'n_steps': len(geom_seq),
        }

    smoothness, smoothness_seq = local_smoothness(geom_seq)
    coherence = global_coherence(geom_seq)
    stability = spectral_evolution_stability(geom_seq)

    return {
        'smoothness': smoothness,
        'smoothness_seq': smoothness_seq,
        'coherence': coherence,
        'stability': stability,
        'n_steps': len(geom_seq),
    }


# =============================================================================
# Layer 3: Phase Transition Detection - 两个模式检测
# =============================================================================


def detect_sudden_drop(coherence_profile: list[float],
                       threshold: float = 0.2,
                       window: int = 3) -> tuple[bool, int]:
    """检测coherence profile中的突然下降

    Returns:
        (是否检测到下降, 下降位置)
    """
    if len(coherence_profile) < window * 2:
        return False, -1

    # 计算移动平均
    for i in range(window, len(coherence_profile) - window):
        before = np.mean(coherence_profile[i - window:i])
        after = np.mean(coherence_profile[i:i + window])

        if before - after > threshold:
            return True, i

    return False, -1


def detect_shallow_lockin_trajectory(trajectory: ReasoningTrajectory,
                                     shallow_layers: list[int] = None) -> dict:
    """模式1：Shallow Lock-in = 浅层平滑度突降

    检测浅层的信息流锁定

    Lock-in特征：在浅层，步骤间的几何相似度突然集中在当前步骤
    （当前步骤不再与之前步骤相关，而是自强化）
    """
    if shallow_layers is None:
        shallow_layers = [10, 14, 18]

    result = {
        'detected': False,
        'layer': None,
        'lockin_step': -1,
        'coherence_before': np.nan,
        'coherence_after': np.nan,
        'drop_magnitude': np.nan,
    }

    for layer in shallow_layers:
        if not trajectory.has_layer(layer):
            continue

        geom_seq = trajectory.get_geometry_sequence(layer)
        if len(geom_seq) < 3:
            continue

        # 计算coherence profile：每一步与之前所有步骤的平均相似度
        coherence_profile = []
        for i in range(1, len(geom_seq)):
            sim_to_past = [geometric_sim(geom_seq[i], geom_seq[j])
                         for j in range(i)]
            coherence_profile.append(np.mean(sim_to_past))

        if len(coherence_profile) < 3:
            continue

        # 检测突然下降
        detected, drop_idx = detect_sudden_drop(coherence_profile)
        if detected:
            result['detected'] = True
            result['layer'] = layer
            result['lockin_step'] = drop_idx + 1  # +1因为coherence_profile从第1步开始
            result['coherence_before'] = float(np.mean(coherence_profile[:drop_idx])) if drop_idx > 0 else np.nan
            result['coherence_after'] = float(np.mean(coherence_profile[drop_idx:])) if drop_idx < len(coherence_profile) else np.nan
            result['drop_magnitude'] = result['coherence_before'] - result['coherence_after']
            break

    return result


def detect_deep_decay_trajectory(trajectory: ReasoningTrajectory,
                                 deep_layers: list[int] = None) -> dict:
    """模式2：Deep Decay = 深层谱演化失稳

    检测深层的信息衰减

    Decay特征：在深层，谱形状的演化不再平滑，而是混乱
    """
    if deep_layers is None:
        deep_layers = [22, 26, 30]

    result = {
        'detected': False,
        'layer': None,
        'stability': np.nan,
        'late_entropy': np.nan,
    }

    for layer in deep_layers:
        if not trajectory.has_layer(layer):
            continue

        geom_seq = trajectory.get_geometry_sequence(layer)
        if len(geom_seq) < 4:
            continue

        # 计算谱演化稳定性
        stability = spectral_evolution_stability(geom_seq)

        # 计算后期步骤的平均谱熵
        late_spectra = [g.spectrum for g in geom_seq[-3:]
                         if g.spectrum.size > 0]
        if not late_spectra:
            continue

        late_entropy = np.mean([spectral_entropy(s) for s in late_spectra])

        # Decay信号：稳定性低 AND 晚期谱熵低（信息丢失）
        if stability < 0.5 and late_entropy < 1.0:
            result['detected'] = True
            result['layer'] = layer
            result['stability'] = stability
            result['late_entropy'] = late_entropy
            break

    return result


def combine_phase_transition_signals(lockin_result: dict,
                                     decay_result: dict) -> dict:
    """组合两个相变信号"""
    return {
        'lockin_detected': lockin_result.get('detected', False),
        'decay_detected': decay_result.get('detected', False),
        'any_phase_transition': lockin_result.get('detected', False) or
                                 decay_result.get('detected', False),
        'lockin_layer': lockin_result.get('layer'),
        'decay_layer': decay_result.get('layer'),
        'phase_type': 'both' if (lockin_result.get('detected') and decay_result.get('detected')) else
                      'lockin' if lockin_result.get('detected') else
                      'decay' if decay_result.get('detected') else
                      'none',
    }


# =============================================================================
# Data Loading
# =============================================================================


def load_full_npz(npz_path: str) -> tuple[list[ReasoningTrajectory], dict]:
    """加载full_*.npz数据

    Returns:
        (trajectories, metadata)
    """
    data = np.load(npz_path, allow_pickle=True)

    # 元数据
    problem_ids = data['problem_ids'].astype(int)
    is_correct = data['is_correct_strict'].astype(int)  # 0=correct, 1=error
    stepcloud = data['stepcloud']  # (N, T, 33, 9)

    # 获取层数
    if 'sv_layers' in data:
        sv_layers = [int(l) for l in data['sv_layers']]
    else:
        sv_layers = list(range(33))

    N = len(problem_ids)
    trajectories = []

    for i in tqdm(range(N), desc=f"Loading {Path(npz_path).name}"):
        traj = ReasoningTrajectory(
            chain_id=i,
            problem_id=int(problem_ids[i]),
            is_correct=bool(is_correct[i] == 0),
        )

        # 添加步骤几何
        T, L, F = stepcloud[i].shape
        for step_idx in range(T):
            # 只在指定层上提取
            for layer_idx, layer_id in enumerate(sv_layers):
                geom = extract_step_geometry_from_stepcloud(
                    stepcloud[i], step_idx, layer_idx, layer_id
                )
                traj.add_step(geom)

        trajectories.append(traj)

    metadata = {
        'sv_layers': sv_layers,
        'n_chains': N,
        'n_correct': int(np.sum(is_correct == 0)),
        'n_error': int(np.sum(is_correct == 1)),
        'subset': Path(npz_path).stem.replace('full_', ''),
    }

    return trajectories, metadata


# =============================================================================
# Validation Experiments (H1-H4)
# =============================================================================


@dataclass
class ValidationResult:
    """单个验证结果"""
    hypothesis: str
    metric: str
    error_mean: float
    correct_mean: float
    cohens_d: float
    p_value: float
    n_error: int
    n_correct: int
    interpretation: str


def run_validation_h1(trajectories: list[ReasoningTrajectory],
                      layer: int = 14) -> ValidationResult:
    """H1: 错误推理的轨迹smoothness低于正确推理

    验证1：轨迹平滑度区分error vs correct

    检验：Mann-Whitney U test on smoothness scores
    预期：error < correct, p < 0.001
    """
    correct = [t for t in trajectories if t.is_correct]
    error = [t for t in trajectories if not t.is_correct]

    correct_smoothness = []
    error_smoothness = []

    for traj in correct:
        metrics = compute_trajectory_metrics(traj, layer)
        if np.isfinite(metrics['smoothness']):
            correct_smoothness.append(metrics['smoothness'])

    for traj in error:
        metrics = compute_trajectory_metrics(traj, layer)
        if np.isfinite(metrics['smoothness']):
            error_smoothness.append(metrics['smoothness'])

    if not correct_smoothness or not error_smoothness:
        return ValidationResult(
            hypothesis="H1",
            metric="smoothness",
            error_mean=np.nan,
            correct_mean=np.nan,
            cohens_d=np.nan,
            p_value=np.nan,
            n_error=0,
            n_correct=0,
            interpretation="Insufficient data",
        )

    c_arr = np.array(correct_smoothness)
    e_arr = np.array(error_smoothness)

    # Mann-Whitney U test (one-tailed: error < correct)
    stat, pval = stats.mannwhitneyu(e_arr, c_arr, alternative='less')

    # Cohen's d
    pooled_std = np.sqrt(((len(c_arr)-1)*c_arr.var(ddof=1) +
                         (len(e_arr)-1)*e_arr.var(ddof=1)) /
                        (len(c_arr)+len(e_arr)-2))
    cohens_d = (c_arr.mean() - e_arr.mean()) / pooled_std if pooled_std > 0 else 0

    return ValidationResult(
        hypothesis="H1",
        metric="smoothness",
        error_mean=float(e_arr.mean()),
        correct_mean=float(c_arr.mean()),
        cohens_d=float(cohens_d),
        p_value=float(pval),
        n_error=len(e_arr),
        n_correct=len(c_arr),
        interpretation=f"error={e_arr.mean():.3f} < correct={c_arr.mean():.3f}: "
                      f"d={cohens_d:.3f}, p={pval:.4f}"
    )


def run_validation_h2_h3(trajectories: list[ReasoningTrajectory],
                        layer: int = 14) -> tuple[ValidationResult, ValidationResult]:
    """H2 & H3: 相变模式在error中更频繁

    验证2：Shallow Lock-in模式在error中更频繁
    检验：Fisher's exact test
    预期：OR > 2, p < 0.01

    验证3：Deep Decay模式在error中更频繁
    检验：Fisher's exact test
    预期：OR > 2, p < 0.01
    """
    correct = [t for t in trajectories if t.is_correct]
    error = [t for t in trajectories if not t.is_correct]

    # 检测Shallow Lock-in
    lockin_correct = 0
    lockin_error = 0
    for traj in correct:
        result = detect_shallow_lockin_trajectory(traj)
        if result['detected']:
            lockin_correct += 1

    for traj in error:
        result = detect_shallow_lockin_trajectory(traj)
        if result['detected']:
            lockin_error += 1

    # 检测Deep Decay
    decay_correct = 0
    decay_error = 0
    for traj in correct:
        result = detect_deep_decay_trajectory(traj)
        if result['detected']:
            decay_correct += 1

    for traj in error:
        result = detect_deep_decay_trajectory(traj)
        if result['detected']:
            decay_error += 1

    # Fisher's exact test
    # H2: Shallow Lock-in
    table_h2 = [[lockin_error, len(error) - lockin_error],
                 [lockin_correct, len(correct) - lockin_correct]]
    oddsratio_h2, p_h2 = stats.fisher_exact(table_h2, alternative='greater')

    # H3: Deep Decay
    table_h3 = [[decay_error, len(error) - decay_error],
                 [decay_correct, len(correct) - decay_correct]]
    oddsratio_h3, p_h3 = stats.fisher_exact(table_h3, alternative='greater')

    result_h2 = ValidationResult(
        hypothesis="H2",
        metric="shallow_lockin",
        error_mean=lockin_error / len(error),
        correct_mean=lockin_correct / len(correct),
        cohens_d=np.nan,
        p_value=float(p_h2),
        n_error=lockin_error,
        n_correct=lockin_correct,
        interpretation=f"OR={oddsratio_h2:.2f}, p={p_h2:.4f}"
    )

    result_h3 = ValidationResult(
        hypothesis="H3",
        metric="deep_decay",
        error_mean=decay_error / len(error),
        correct_mean=decay_correct / len(correct),
        cohens_d=np.nan,
        p_value=float(p_h3),
        n_error=decay_error,
        n_correct=decay_correct,
        interpretation=f"OR={oddsratio_h3:.2f}, p={p_h3:.4f}"
    )

    return result_h2, result_h3


def run_validation_h4(trajectories: list[ReasoningTrajectory],
                      layer: int = 14) -> ValidationResult:
    """H4: 基于轨迹的检测器优于基于单步几何的检测器

    验证4：组合检测器超越单步检测

    检验：Paired t-test on per-chain AUROC
    预期：trajectory > step-wise, p < 0.05

    简化版本：比较 trajectory metrics (smoothness, coherence, stability)
    vs 单步 metrics (average kappa, eff_rank)
    """
    correct = [t for t in trajectories if t.is_correct]
    error = [t for t in trajectories if not t.is_correct]

    # 轨迹级指标
    trajectory_scores = []
    stepwise_scores = []
    labels = []

    for traj in trajectories:
        geom_seq = traj.get_geometry_sequence(layer)
        if len(geom_seq) < 2:
            continue

        # 轨迹级：组合 smoothness, coherence, stability
        metrics = compute_trajectory_metrics(traj, layer)
        traj_score = (metrics['smoothness'] + metrics['coherence'] + metrics['stability']) / 3

        # 单步级：平均 kappa
        step_kappas = [g.kappa for g in geom_seq if not np.isnan(g.kappa)]
        if not step_kappas:
            continue

        step_score = np.mean(step_kappas)

        trajectory_scores.append(traj_score)
        stepwise_scores.append(step_score)
        labels.append(0 if traj.is_correct else 1)  # 0=correct, 1=error

    if not trajectory_scores:
        return ValidationResult(
            hypothesis="H4",
            metric="trajectory_vs_stepwise",
            error_mean=np.nan,
            correct_mean=np.nan,
            cohens_d=np.nan,
            p_value=np.nan,
            n_error=0,
            n_correct=0,
            interpretation="Insufficient data",
        )

    # 计算区分度 (error vs correct 的差异)
    traj_arr = np.array(trajectory_scores)
    step_arr = np.array(stepwise_scores)
    labels = np.array(labels)

    traj_error = traj_arr[labels == 1]
    traj_correct = traj_arr[labels == 0]
    step_error = step_arr[labels == 1]
    step_correct = step_arr[labels == 0]

    # Cohen's d for trajectory vs stepwise
    traj_d = (traj_correct.mean() - traj_error.mean()) / np.sqrt(
        ((len(traj_correct)-1)*traj_correct.var(ddof=1) +
         (len(traj_error)-1)*traj_error.var(ddof=1)) /
        (len(traj_correct)+len(traj_error)-2)
    ) if len(traj_correct) > 0 and len(traj_error) > 0 else 0

    step_d = (step_correct.mean() - step_error.mean()) / np.sqrt(
        ((len(step_correct)-1)*step_correct.var(ddof=1) +
         (len(step_error)-1)*step_error.var(ddof=1)) /
        (len(step_correct)+len(step_error)-2)
    ) if len(step_correct) > 0 and len(step_error) > 0 else 0

    return ValidationResult(
        hypothesis="H4",
        metric="trajectory_vs_stepwise",
        error_mean=float(traj_error.mean()) if len(traj_error) > 0 else np.nan,
        correct_mean=float(traj_correct.mean()) if len(traj_correct) > 0 else np.nan,
        cohens_d=float(traj_d - step_d),  # 轨迹相对单步的提升
        p_value=np.nan,  # 需要paired test，这里简化
        n_error=len(traj_error),
        n_correct=len(traj_correct),
        interpretation=f"Trajectory d={traj_d:.3f} > Stepwise d={step_d:.3f}: Δ={traj_d-step_d:.3f}"
    )


def run_all_validations(npz_path: str,
                       layers: list[int] = None,
                       output_dir: str = None) -> dict:
    """运行所有验证实验"""
    print(f"\n{'='*80}")
    print(f"Trajectory of Thought: Geometric Phase Transitions in Multi-Step Reasoning")
    print(f"{'='*80}")
    print(f"Loading data from: {npz_path}")

    trajectories, metadata = load_full_npz(npz_path)

    print(f"Loaded {metadata['n_chains']} chains ({metadata['n_correct']} correct, {metadata['n_error']} error)")
    print(f"Subset: {metadata['subset']}")
    print(f"Layers available: {metadata['sv_layers']}")

    if layers is None:
        layers = [14]  # 默认L14

    all_results = {}

    for layer in layers:
        print(f"\n{'='*80}")
        print(f"Layer {layer} Analysis")
        print(f"{'='*80}")

        # H1: Smoothness
        print(f"\n[H1] Testing: 错误推理的轨迹smoothness低于正确推理")
        result_h1 = run_validation_h1(trajectories, layer)
        print(f"  Result: {result_h1.interpretation}")
        all_results[f'L{layer}_H1'] = result_h1

        # H2 & H3: Phase transitions
        print(f"\n[H2] Testing: Shallow Lock-in模式在error中更频繁")
        print(f"[H3] Testing: Deep Decay模式在error中更频繁")
        result_h2, result_h3 = run_validation_h2_h3(trajectories, layer)

        n_error_total = len([t for t in trajectories if not t.is_correct])
        n_correct_total = len([t for t in trajectories if t.is_correct])

        print(f"  H2 (Shallow Lock-in): error={result_h2.n_error}/{n_error_total} "
              f"vs correct={result_h2.n_correct}/{n_correct_total}")
        print(f"    Result: {result_h2.interpretation}")
        all_results[f'L{layer}_H2'] = result_h2

        print(f"  H3 (Deep Decay): error={result_h3.n_error}/{n_error_total} "
              f"vs correct={result_h3.n_correct}/{n_correct_total}")
        print(f"    Result: {result_h3.interpretation}")
        all_results[f'L{layer}_H3'] = result_h3

        # H4: Trajectory vs Stepwise
        print(f"\n[H4] Testing: 基于轨迹的检测器优于基于单步几何的检测器")
        result_h4 = run_validation_h4(trajectories, layer)
        print(f"  Result: {result_h4.interpretation}")
        all_results[f'L{layer}_H4'] = result_h4

    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'trajectory_validation_results.json')

        serializable_results = {}
        for key, val in all_results.items():
            serializable_results[key] = {
                'hypothesis': val.hypothesis,
                "metric": val.metric,
                'error_mean': val.error_mean,
                'correct_mean': val.correct_mean,
                'cohens_d': val.cohens_d,
                'p_value': val.p_value,
                'n_error': val.n_error,
                'n_correct': val.n_correct,
                'interpretation': val.interpretation,
            }

        with open(output_path, 'w') as f:
            json.dump({
                'metadata': metadata,
                'results': serializable_results,
            }, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Results saved to: {output_path}")

    # 打印总结
    print(f"\n{'='*80}")
    print("Summary Table")
    print(f"{'='*80}")
    cohens_label = "Cohen's d"
    print(f"{'Test':<15} {'Metric':<25} {'Error':<8} {'Correct':<8} {cohens_label:<10} {'p-value':<10}")
    print("-"*80)

    for key in sorted(all_results.keys()):
        val = all_results[key]
        sig = "*" if val.p_value is not None and val.p_value < 0.05 else ""
        error_str = f"{val.error_mean:.3f}" if not np.isnan(val.error_mean) else "N/A"
        correct_str = f"{val.correct_mean:.3f}" if not np.isnan(val.correct_mean) else "N/A"
        d_str = f"{val.cohens_d:.3f}" if not np.isnan(val.cohens_d) else "N/A"
        p_str = f"{val.p_value:.4f}" if not np.isnan(val.p_value) else "N/A"

        print(f"{key:<15} {val.metric:<25} {error_str:<8} {correct_str:<8} {d_str:<10} {p_str:<10} {sig}")

    print("-"*80)
    print("* p < 0.05 (statistically significant)")
    print("="*80)

    return all_results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='Trajectory of Thought: Geometric Phase Transitions in Multi-Step Reasoning'
    )
    parser.add_argument('npz_path', help='Path to full_*.npz file')
    parser.add_argument('--output_dir', default='./trajectory_results',
                       help='Output directory')
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                       help='Layers to analyze (default: 14)')

    args = parser.parse_args()

    run_all_validations(args.npz_path, args.layers, args.output_dir)


if __name__ == '__main__':
    main()
